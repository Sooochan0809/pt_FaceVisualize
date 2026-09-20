import Delaunator from "https://cdn.jsdelivr.net/npm/delaunator@5/+esm";

const MAX_MEDIA = 7;
const MORPH_WIDTH = 600;
const MORPH_HEIGHT = 800;
const MORPH_RATIO = MORPH_WIDTH / MORPH_HEIGHT;
const VIDEO_RENDER_FPS = 30;
const VIDEO_DETECTION_INTERVAL = 250;
const VIDEO_SYNC_TOLERANCE = 0.08;
const TRIANGLE_OVERDRAW_PX = 0.75;
const MEDIAPIPE_TASKS_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";
const MEDIAPIPE_WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
const FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";

const fileInput = document.getElementById("fileInput");
const fileButton = document.getElementById("fileButton");
const addLayerSlot = document.getElementById("addLayerSlot");
const clearButton = document.getElementById("clearButton");
const monoToggleButton = document.getElementById("monoToggleButton");
const morphCanvas = document.getElementById("morphCanvas");
const stageMessage = document.getElementById("stageMessage");
const layerList = document.getElementById("layerList");
const layersContainer = document.getElementById("layers");
const status = document.getElementById("status");

let layers = [];
let faceLandmarkerPromise = null;
let monochromeEnabled = true;
let animationFrameId = null;
let lastRenderedAt = 0;
let loadingCount = 0;
let videoTimelinePosition = 0;
let videoTimelineStartedAt = null;

function createId() {
    return crypto.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function getMediaSize(media) {
    return {
        width: media.videoWidth || media.naturalWidth || media.width || 1,
        height: media.videoHeight || media.naturalHeight || media.height || 1
    };
}

function getCenteredCropRect(sourceWidth, sourceHeight) {
    const sourceRatio = sourceWidth / sourceHeight;
    if (sourceRatio > MORPH_RATIO) {
        const height = sourceHeight;
        const width = height * MORPH_RATIO;
        return { x: (sourceWidth - width) / 2, y: 0, width, height };
    }

    const width = sourceWidth;
    const height = width / MORPH_RATIO;
    return { x: 0, y: (sourceHeight - height) / 2, width, height };
}

function createSizedCanvas() {
    const canvas = document.createElement("canvas");
    canvas.width = MORPH_WIDTH;
    canvas.height = MORPH_HEIGHT;
    return canvas;
}

function drawMediaToCanvas(media, canvas) {
    const { width, height } = getMediaSize(media);
    const crop = getCenteredCropRect(width, height);
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, MORPH_WIDTH, MORPH_HEIGHT);
    ctx.drawImage(
        media,
        crop.x,
        crop.y,
        crop.width,
        crop.height,
        0,
        0,
        MORPH_WIDTH,
        MORPH_HEIGHT
    );
}

async function getFaceLandmarker() {
    if (!faceLandmarkerPromise) {
        faceLandmarkerPromise = import(MEDIAPIPE_TASKS_URL)
            .then(async ({ FaceLandmarker, FilesetResolver }) => {
                const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);
                return FaceLandmarker.createFromOptions(vision, {
                    baseOptions: { modelAssetPath: FACE_MODEL_URL },
                    runningMode: "IMAGE",
                    numFaces: 1
                });
            })
            .catch((error) => {
                faceLandmarkerPromise = null;
                throw error;
            });
    }
    return faceLandmarkerPromise;
}

function detectPoints(faceLandmarker, canvas) {
    const result = faceLandmarker.detect(canvas);
    const landmarks = result.faceLandmarks?.[0];
    if (!landmarks) return null;

    const points = landmarks.map((point) => [
        point.x * MORPH_WIDTH,
        point.y * MORPH_HEIGHT
    ]);
    points.push([0, 0]);
    points.push([MORPH_WIDTH - 1, 0]);
    points.push([MORPH_WIDTH - 1, MORPH_HEIGHT - 1]);
    points.push([0, MORPH_HEIGHT - 1]);
    return points;
}

function loadImage(src) {
    return new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = reject;
        image.src = src;
    });
}

function loadVideo(src) {
    return new Promise((resolve, reject) => {
        const video = document.createElement("video");
        video.muted = true;
        video.loop = true;
        video.playsInline = true;
        video.preload = "auto";
        video.onloadeddata = () => resolve(video);
        video.onerror = reject;
        video.src = src;
        video.load();
    });
}

function readFileAsDataUrl(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

async function makeLayer(file) {
    const isVideo = file.type.startsWith("video/");
    const objectUrl = isVideo ? URL.createObjectURL(file) : null;
    const src = objectUrl || await readFileAsDataUrl(file);

    try {
        const media = isVideo ? await loadVideo(src) : await loadImage(src);
        const sourceCanvas = createSizedCanvas();
        drawMediaToCanvas(media, sourceCanvas);
        const faceLandmarker = await getFaceLandmarker();
        const points = detectPoints(faceLandmarker, sourceCanvas);

        return {
            id: createId(),
            name: file.name,
            src,
            objectUrl,
            media,
            mediaType: isVideo ? "video" : "image",
            sourceCanvas,
            warpCanvas: createSizedCanvas(),
            points,
            weight: 100,
            lastDetectionAt: 0,
            hasEverDetectedFace: Boolean(points)
        };
    } catch (error) {
        if (objectUrl) URL.revokeObjectURL(objectUrl);
        throw error;
    }
}

function getMorphLayers() {
    return layers.filter((layer) => layer.points);
}

function getNormalizedWeights(morphLayers = getMorphLayers()) {
    const total = morphLayers.reduce((sum, layer) => sum + Math.max(0, layer.weight), 0);
    return new Map(morphLayers.map((layer) => [
        layer.id,
        total > 0 ? Math.max(0, layer.weight) / total : 0
    ]));
}

function blendPoints(weightedLayers) {
    const pointCount = weightedLayers[0].layer.points.length;
    return Array.from({ length: pointCount }, (_, pointIndex) => (
        weightedLayers.reduce((sum, item) => {
            const point = item.layer.points[pointIndex];
            sum[0] += point[0] * item.weight;
            sum[1] += point[1] * item.weight;
            return sum;
        }, [0, 0])
    ));
}

function getExpandedTriangle(triangle) {
    const center = triangle.reduce((sum, point) => [sum[0] + point[0] / 3, sum[1] + point[1] / 3], [0, 0]);
    return triangle.map((point) => {
        const dx = point[0] - center[0];
        const dy = point[1] - center[1];
        const length = Math.hypot(dx, dy) || 1;
        return [
            point[0] + dx / length * TRIANGLE_OVERDRAW_PX,
            point[1] + dy / length * TRIANGLE_OVERDRAW_PX
        ];
    });
}

function drawTriangleImage(ctx, image, sourceTriangle, destinationTriangle) {
    const [sx0, sy0] = sourceTriangle[0];
    const [sx1, sy1] = sourceTriangle[1];
    const [sx2, sy2] = sourceTriangle[2];
    const [dx0, dy0] = destinationTriangle[0];
    const [dx1, dy1] = destinationTriangle[1];
    const [dx2, dy2] = destinationTriangle[2];
    const denominator = sx0 * (sy1 - sy2) + sx1 * (sy2 - sy0) + sx2 * (sy0 - sy1);
    if (Math.abs(denominator) < 1e-6) return;

    const a = (dx0 * (sy1 - sy2) + dx1 * (sy2 - sy0) + dx2 * (sy0 - sy1)) / denominator;
    const b = (dy0 * (sy1 - sy2) + dy1 * (sy2 - sy0) + dy2 * (sy0 - sy1)) / denominator;
    const c = (dx0 * (sx2 - sx1) + dx1 * (sx0 - sx2) + dx2 * (sx1 - sx0)) / denominator;
    const d = (dy0 * (sx2 - sx1) + dy1 * (sx0 - sx2) + dy2 * (sx1 - sx0)) / denominator;
    const e = (dx0 * (sx1 * sy2 - sx2 * sy1) + dx1 * (sx2 * sy0 - sx0 * sy2) + dx2 * (sx0 * sy1 - sx1 * sy0)) / denominator;
    const f = (dy0 * (sx1 * sy2 - sx2 * sy1) + dy1 * (sx2 * sy0 - sx0 * sy2) + dy2 * (sx0 * sy1 - sx1 * sy0)) / denominator;

    const expanded = getExpandedTriangle(destinationTriangle);
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(expanded[0][0], expanded[0][1]);
    ctx.lineTo(expanded[1][0], expanded[1][1]);
    ctx.lineTo(expanded[2][0], expanded[2][1]);
    ctx.closePath();
    ctx.clip();
    ctx.setTransform(a, b, c, d, e, f);
    ctx.drawImage(image, 0, 0);
    ctx.restore();
}

function renderWarpedLayer(layer, destinationPoints, triangles) {
    const ctx = layer.warpCanvas.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, MORPH_WIDTH, MORPH_HEIGHT);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";

    for (let index = 0; index < triangles.length; index += 3) {
        const i0 = triangles[index];
        const i1 = triangles[index + 1];
        const i2 = triangles[index + 2];
        drawTriangleImage(
            ctx,
            layer.sourceCanvas,
            [layer.points[i0], layer.points[i1], layer.points[i2]],
            [destinationPoints[i0], destinationPoints[i1], destinationPoints[i2]]
        );
    }
}

function renderMorph() {
    const ctx = morphCanvas.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, MORPH_WIDTH, MORPH_HEIGHT);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";

    const morphLayers = getMorphLayers();
    const normalizedWeights = getNormalizedWeights(morphLayers);
    const activeLayers = morphLayers
        .map((layer) => ({ layer, weight: normalizedWeights.get(layer.id) || 0 }))
        .filter((item) => item.weight > 0);

    if (stageMessage) stageMessage.hidden = activeLayers.length > 0;
    if (activeLayers.length === 0) {
        if (stageMessage) {
            stageMessage.textContent = layers.length
                ? "顔を検出できる素材を追加してください"
                : "顔が写った画像または動画を2つ以上追加してください";
        }
        return;
    }

    if (activeLayers.length === 1) {
        ctx.drawImage(activeLayers[0].layer.sourceCanvas, 0, 0);
        return;
    }

    const destinationPoints = blendPoints(activeLayers);
    const triangles = Delaunator.from(destinationPoints).triangles;
    let cumulativeWeight = 0;

    activeLayers.forEach((item) => {
        renderWarpedLayer(item.layer, destinationPoints, triangles);
        cumulativeWeight += item.weight;
        ctx.globalAlpha = item.weight / cumulativeWeight;
        ctx.drawImage(item.layer.warpCanvas, 0, 0);
    });
    ctx.globalAlpha = 1;
}

function updateLayerState(layer) {
    const node = layersContainer.querySelector(`[data-state-id="${layer.id}"]`);
    if (!node) return;
    const detected = Boolean(layer.points);
    node.textContent = detected
        ? (layer.mediaType === "video" ? "顔を追跡中" : "顔を検出済み")
        : (layer.mediaType === "video" ? "顔を検索中…" : "顔を検出できません");
    node.classList.toggle("is-error", !detected && layer.mediaType !== "video");
    const slider = layersContainer.querySelector(`[data-weight-id="${layer.id}"]`);
    if (slider) slider.disabled = !detected;
}

function getVideoTimelinePosition(now = performance.now()) {
    if (videoTimelineStartedAt === null) return videoTimelinePosition;
    return videoTimelinePosition + (now - videoTimelineStartedAt) / 1000;
}

function syncVideoElement(video, timelinePosition, force = false) {
    if (video.readyState < 1) return;

    if (Number.isFinite(video.duration) && video.duration > 0) {
        const targetTime = timelinePosition % video.duration;
        const directDrift = Math.abs(video.currentTime - targetTime);
        const loopDrift = Math.min(directDrift, Math.abs(video.duration - directDrift));
        if (force || loopDrift > VIDEO_SYNC_TOLERANCE) video.currentTime = targetTime;
    }
    if (!document.hidden && video.paused) video.play().catch(() => {});
}

function syncVideoPlayback(now = performance.now(), force = false) {
    const videoLayers = layers.filter((layer) => layer.mediaType === "video");
    if (videoLayers.length === 0) {
        videoTimelinePosition = 0;
        videoTimelineStartedAt = null;
        return;
    }

    if (!document.hidden && videoTimelineStartedAt === null) videoTimelineStartedAt = now;
    const timelinePosition = getVideoTimelinePosition(now);
    videoLayers.forEach((layer) => {
        syncVideoElement(layer.media, timelinePosition, force);
    });
}

function pauseVideoPlayback(now = performance.now()) {
    videoTimelinePosition = getVideoTimelinePosition(now);
    videoTimelineStartedAt = null;
    layers.forEach((layer) => {
        if (layer.mediaType !== "video") return;
        layer.media.pause();
    });
}

function updateVideoLayers(now) {
    syncVideoPlayback(now);
    const faceLandmarkerReady = faceLandmarkerPromise;
    layers.forEach((layer) => {
        if (layer.mediaType !== "video" || layer.media.readyState < 2) return;
        drawMediaToCanvas(layer.media, layer.sourceCanvas);
        if (!faceLandmarkerReady || now - layer.lastDetectionAt < VIDEO_DETECTION_INTERVAL) return;

        layer.lastDetectionAt = now;
        faceLandmarkerReady.then((faceLandmarker) => {
            const nextPoints = detectPoints(faceLandmarker, layer.sourceCanvas);
            if (nextPoints) {
                layer.points = nextPoints;
                layer.hasEverDetectedFace = true;
            }
            updateLayerState(layer);
        }).catch(() => {});
    });
}

function animationLoop(now) {
    animationFrameId = null;
    const hasVideo = layers.some((layer) => layer.mediaType === "video");
    if (!hasVideo) return;

    if (!document.hidden && now - lastRenderedAt >= 1000 / VIDEO_RENDER_FPS) {
        lastRenderedAt = now;
        updateVideoLayers(now);
        renderMorph();
    }
    animationFrameId = requestAnimationFrame(animationLoop);
}

function ensureAnimationLoop() {
    const hasVideo = layers.some((layer) => layer.mediaType === "video");
    if (hasVideo && animationFrameId === null) {
        animationFrameId = requestAnimationFrame(animationLoop);
    } else if (!hasVideo) {
        if (animationFrameId !== null) cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
        videoTimelinePosition = 0;
        videoTimelineStartedAt = null;
    }
}

function disposeLayer(layer) {
    if (layer.mediaType === "video") {
        layer.media.pause();
        layer.media.removeAttribute("src");
        layer.media.load();
    }
    if (layer.objectUrl) URL.revokeObjectURL(layer.objectUrl);
}

function removeLayer(id) {
    const layer = layers.find((item) => item.id === id);
    if (!layer) return;
    disposeLayer(layer);
    layers = layers.filter((item) => item.id !== id);
    renderAll();
}

function setLayerWeight(id, value) {
    const layer = layers.find((item) => item.id === id);
    if (!layer) return;
    layer.weight = Number(value);
    renderMorph();
    renderOutputs();
}

function createVideoThumbnail(layer) {
    const canvas = document.createElement("canvas");
    const size = 148;
    canvas.width = size;
    canvas.height = size;
    canvas.setAttribute("aria-hidden", "true");

    const source = layer.sourceCanvas;
    const scale = Math.min(size / source.width, size / source.height);
    const width = source.width * scale;
    const height = source.height * scale;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(source, (size - width) / 2, (size - height) / 2, width, height);
    return canvas;
}

function createLayerItem(layer) {
    const item = document.createElement("article");
    item.className = "layer";

    const removeButton = document.createElement("button");
    removeButton.className = "layerRemoveButton";
    removeButton.type = "button";
    removeButton.setAttribute("aria-label", `${layer.name}を削除`);
    const removeIcon = document.createElement("img");
    removeIcon.src = "../overlap/icon-clause.png";
    removeIcon.alt = "";
    removeButton.appendChild(removeIcon);
    removeButton.addEventListener("click", () => removeLayer(layer.id));

    const thumbnail = document.createElement("div");
    thumbnail.className = "layerThumb";
    const preview = layer.mediaType === "video"
        ? createVideoThumbnail(layer)
        : layer.media.cloneNode(true);
    if (layer.mediaType === "image") preview.alt = "";
    thumbnail.appendChild(preview);
    if (layer.mediaType === "video") {
        const badge = document.createElement("span");
        badge.className = "mediaBadge";
        badge.textContent = "VIDEO";
        thumbnail.appendChild(badge);
    }

    const body = document.createElement("div");
    body.className = "layerBody";
    const name = document.createElement("div");
    name.className = "fileName";
    name.textContent = layer.name;

    const control = document.createElement("div");
    control.className = "morphControl";
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "0";
    slider.max = "100";
    slider.value = String(layer.weight);
    slider.dataset.weightId = layer.id;
    slider.setAttribute("aria-label", `${layer.name}のモーフィング割合`);
    slider.addEventListener("input", () => setLayerWeight(layer.id, slider.value));
    const output = document.createElement("output");
    output.dataset.outputId = layer.id;
    control.append(slider, output);

    const state = document.createElement("div");
    state.className = "layerState";
    state.dataset.stateId = layer.id;
    body.append(name, control, state);
    item.append(removeButton, thumbnail, body);
    return item;
}

function renderLayerList() {
    layersContainer.replaceChildren(...layers.map(createLayerItem));
    layerList.classList.toggle("has-layers", layers.length > 0);
    layers.forEach(updateLayerState);
}

function renderOutputs() {
    const normalizedWeights = getNormalizedWeights();
    layers.forEach((layer) => {
        const output = layersContainer.querySelector(`[data-output-id="${layer.id}"]`);
        if (!output) return;
        const percentage = Math.round((normalizedWeights.get(layer.id) || 0) * 100);
        output.value = `${percentage}%`;
        output.textContent = output.value;
    });
}

function renderStatus() {
    const total = layers.length + loadingCount;
    status.textContent = loadingCount ? `${layers.length} / ${MAX_MEDIA}（読み込み中）` : `${layers.length} / ${MAX_MEDIA}`;
    clearButton.disabled = total === 0;
    fileInput.disabled = total >= MAX_MEDIA;
    fileButton.classList.toggle("is-disabled", total >= MAX_MEDIA);
    addLayerSlot.hidden = total >= MAX_MEDIA;
    monoToggleButton.classList.toggle("is-off", !monochromeEnabled);
    monoToggleButton.setAttribute("aria-pressed", String(monochromeEnabled));
}

function renderAll() {
    renderLayerList();
    renderOutputs();
    renderStatus();
    renderMorph();
    ensureAnimationLoop();
}

async function addFiles(fileList) {
    const available = Math.max(0, MAX_MEDIA - layers.length - loadingCount);
    const files = Array.from(fileList)
        .filter((file) => file.type.startsWith("image/") || file.type.startsWith("video/"))
        .slice(0, available);
    if (files.length === 0) return;

    loadingCount += files.length;
    renderStatus();
    const results = await Promise.allSettled(files.map(makeLayer));
    results.forEach((result, index) => {
        if (result.status === "fulfilled") layers.push(result.value);
        else console.error(`Failed to load ${files[index].name}.`, result.reason);
    });
    loadingCount -= files.length;
    renderAll();
}

fileInput.addEventListener("change", async (event) => {
    await addFiles(event.target.files);
    fileInput.value = "";
});

clearButton.addEventListener("click", () => {
    layers.forEach(disposeLayer);
    layers = [];
    renderAll();
});

monoToggleButton.addEventListener("click", () => {
    monochromeEnabled = !monochromeEnabled;
    morphCanvas.classList.toggle("is-monochrome", monochromeEnabled);
    renderStatus();
});

document.addEventListener("visibilitychange", () => {
    if (document.hidden) pauseVideoPlayback();
    else syncVideoPlayback(performance.now(), true);
});

window.addEventListener("beforeunload", () => layers.forEach(disposeLayer));

renderAll();

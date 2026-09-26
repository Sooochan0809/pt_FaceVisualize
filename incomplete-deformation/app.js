(() => {
  "use strict";

  const MODEL_URL = "../modules/faceAPI/models";
  const MEDIAPIPE_TASKS_URL =
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34";
  const MEDIAPIPE_WASM_URL =
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34/wasm";
  const MEDIAPIPE_FACE_MODEL_URL =
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";
  const DELAUNATOR_URL = "https://cdn.jsdelivr.net/npm/delaunator@5/+esm";
  const BASE_EMOTION = "neutral";
  const EMOTIONS = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgusted",
    "surprised",
  ];
  const LABELS = {
    neutral: "無表情",
    happy: "喜び",
    sad: "悲しみ",
    angry: "怒り",
    fearful: "恐れ",
    disgusted: "嫌悪",
    surprised: "驚き",
  };
  const FILE_EMOTION_PREFIXES = [
    ["01_happy", "happy"],
    ["02_sad", "sad"],
    ["03_angry", "angry"],
    ["04_fearful", "fearful"],
    ["05_surprised", "surprised"],
    ["06_disgusted", "disgusted"],
  ];
  const COLORS = {
    neutral: "#7f8993",
    target: "#ff6b45",
    onset: "#42d392",
    peak: "#8b6fc0",
    settle: "#8b6fc0",
  };
  const LANDMARK_COLOR = "#3478d4";
  const PROGRESSION_COLOR = "#171a1d";
  const LANDMARK_WEIGHT = 0.75;
  const EXPRESSION_WEIGHT = 0.25;
  const EXPRESSION_LANDMARK_INDICES = [
    33, 46, 52, 53, 55, 61, 63, 65, 66, 70, 78, 80, 81, 82, 84, 87, 88, 91, 95,
    101, 105, 107, 133, 144, 146, 153, 158, 160, 178, 181, 191, 205, 263, 276,
    282, 283, 285, 291, 293, 295, 296, 300, 308, 310, 311, 312, 314, 317, 318,
    321, 324, 330, 334, 336, 362, 373, 375, 380, 385, 387, 402, 405, 415, 425,
  ];
  const FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
  ];
  const MORPH_LANDMARK_INDICES = [
    ...new Set([
      ...FACE_OVAL_INDICES,
      ...EXPRESSION_LANDMARK_INDICES,
      ...Array.from({ length: 78 }, (_, index) => index * 6).filter(
        (index) => index < 468,
      ),
    ]),
  ];
  const TRIANGLE_OVERDRAW = 0.75;
  const PREVIEW_MAX_EDGE = 720;
  const EXPORT_FPS = 30;
  const MEDIABUNNY_URL = "https://cdn.jsdelivr.net/npm/mediabunny@1.49.0/+esm";
  const el = Object.fromEntries(
    [
      "sourceVideo",
      "transitionVideo",
      "clipPreviewVideo",
      "clipPreviewMeta",
      "outputCanvas",
      "stageTitle",
      "stagePhase",
      "playButton",
      "playhead",
      "timeOutput",
      "sequenceTimeline",
      "fileInput",
      "jsonInput",
      "importJsonButton",
      "analyzeAllButton",
      "clipCount",
      "clipList",
      "risePlaybackDuration",
      "skipDurationSetting",
      "decayDuration",
      "holdDuration",
      "morphDuration",
      "sampleInterval",
      "loopToggle",
      "clipEditor",
      "analyzeClipButton",
      "emotionSelect",
      "skipOutRange",
      "skipInRange",
      "detectionBadge",
      "analysisChart",
      "emotionLegend",
      "skipOutInput",
      "skipInInput",
      "skipOutDuration",
      "skipDuration",
      "exportButton",
      "exportJsonButton",
      "status",
      "progress",
    ].map((id) => [id, document.getElementById(id)]),
  );
  const PLAYBACK_SETTINGS = [
    ["risePlaybackDuration", "riseDuration", 0.45],
    ["skipDurationSetting", "skipDuration", 0.2],
    ["decayDuration", "decayDuration", 0.6],
    ["holdDuration", "holdDuration", 0.1],
    ["morphDuration", "morphDuration", 0.2],
  ];

  const detectorOptions = new faceapi.TinyFaceDetectorOptions({
    inputSize: 320,
    scoreThreshold: 0.35,
  });
  const outputContext = el.outputCanvas.getContext("2d", { alpha: false });
  const analysisCanvas = document.createElement("canvas");
  const analysisContext = analysisCanvas.getContext("2d", { alpha: false });
  const alignedCanvasA = document.createElement("canvas");
  const alignedCanvasB = document.createElement("canvas");
  const warpedCanvasA = document.createElement("canvas");
  const warpedCanvasB = document.createElement("canvas");
  const morphCanvas = document.createElement("canvas");
  const maskCanvas = document.createElement("canvas");
  let clips = [];
  let selectedId = null;
  let modelsReady = false;
  let busy = false;
  let playing = false;
  let playPosition = 0;
  let playStartedAt = 0;
  let animationId = 0;
  let loadedClipId = null;
  let transitionLoadedClipId = null;
  let lastRenderedTime = -1;
  let previewPhaseKey = "";
  let previewRendering = false;
  let previewVideo = el.sourceVideo;
  let previewMorphVideos = [el.sourceVideo, el.transitionVideo];
  let nextId = 1;
  let importedConfig = null;
  let faceLandmarkerPromise = null;
  const delaunatorPromise = import(DELAUNATOR_URL).then(
    (module) => module.default,
  );
  const mediabunnyPromise = import(MEDIABUNNY_URL);

  function getFaceLandmarker() {
    if (!faceLandmarkerPromise) {
      faceLandmarkerPromise = import(MEDIAPIPE_TASKS_URL)
        .then(async ({ FaceLandmarker, FilesetResolver }) => {
          const vision =
            await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);
          return FaceLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: MEDIAPIPE_FACE_MODEL_URL },
            runningMode: "IMAGE",
            numFaces: 1,
            minFaceDetectionConfidence: 0.35,
            minFacePresenceConfidence: 0.35,
          });
        })
        .catch((error) => {
          faceLandmarkerPromise = null;
          throw error;
        });
    }
    return faceLandmarkerPromise;
  }

  const models = Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
    faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
    getFaceLandmarker(),
    delaunatorPromise,
    mediabunnyPromise,
  ]);

  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
  const number = (value, fallback = 0) =>
    Number.isFinite(Number(value)) ? Number(value) : fallback;
  const setting = (id, fallback) =>
    clamp(
      number(el[id].value, fallback),
      number(el[id].min, -Infinity),
      number(el[id].max, Infinity),
    );
  const seconds = (value) => `${Math.max(0, number(value)).toFixed(2)}s`;
  const selectedClip = () =>
    clips.find((clip) => clip.id === selectedId) || null;

  function emotionFromFileName(fileName) {
    const normalized = fileName.trim().toLowerCase();
    return (
      FILE_EMOTION_PREFIXES.find(([prefix]) =>
        normalized.startsWith(prefix),
      )?.[1] || "happy"
    );
  }

  function setStatus(text, current = 0, total = 0, error = false) {
    el.status.textContent = text;
    el.status.classList.toggle("error", error);
    el.progress.hidden = total <= 0;
    el.progress.max = Math.max(1, total);
    el.progress.value = current;
  }

  function setBusy(next) {
    busy = next;
    updateControls();
  }

  function updateControls() {
    const hasClips = clips.length > 0;
    el.fileInput.disabled = busy;
    const addClipButton = el.clipList.querySelector('[data-action="add"]');
    if (addClipButton) addClipButton.disabled = busy;
    el.jsonInput.disabled = busy;
    el.importJsonButton.disabled = busy;
    el.analyzeAllButton.disabled = busy || !hasClips || !modelsReady;
    el.analyzeClipButton.disabled = busy || !selectedClip() || !modelsReady;
    el.playButton.disabled = busy || !hasClips;
    el.playhead.disabled = busy || !hasClips;
    [el.skipOutRange, el.skipInRange].forEach((input) => {
      input.disabled = busy || !selectedClip();
    });
    el.exportJsonButton.disabled = busy || !hasClips;
    el.exportButton.disabled = busy || !hasClips || !("VideoEncoder" in window);
  }

  function readMetadata(clip) {
    return new Promise((resolve, reject) => {
      const video = document.createElement("video");
      video.preload = "metadata";
      video.muted = true;
      video.onloadedmetadata = () => {
        clip.duration = video.duration;
        clip.width = video.videoWidth;
        clip.height = video.videoHeight;
        clip.onset = Math.min(
          video.duration * 0.25,
          Math.max(0.1, video.duration - 0.2),
        );
        clip.peak = Math.min(
          video.duration * 0.65,
          Math.max(clip.onset + 0.05, video.duration - 0.1),
        );
        clip.settle = video.duration;
        clip.skipOut = clip.onset + (clip.peak - clip.onset) * 0.7;
        clip.plateauEnd = clip.peak;
        clip.skipIn = clip.settle;
        resolve();
      };
      video.onerror = () =>
        reject(new Error(`${clip.name} を読み込めませんでした`));
      video.src = clip.url;
    });
  }

  async function addFiles(files) {
    pause();
    const additions = [...files]
      .filter((file) => file.type.startsWith("video/"))
      .map((file) => ({
        id: nextId++,
        file,
        name: file.name,
        url: URL.createObjectURL(file),
        duration: 0,
        width: 0,
        height: 0,
        emotion: emotionFromFileName(file.name),
        onset: 0,
        peak: 0,
        skipOut: 0,
        plateauEnd: 0,
        skipIn: 0,
        settle: 0,
        skipInManual: false,
        samples: [],
        analysisSeries: [],
        alignmentPose: null,
        landmarkDetectionRate: null,
        analyzed: false,
      }));
    if (!additions.length) return;
    setBusy(true);
    try {
      for (const clip of additions) await readMetadata(clip);
      clips.push(...additions);
      selectedId = selectedId ?? additions[0].id;
      if (importedConfig) {
        const result = await applyImportedConfig(importedConfig, false);
        setStatus(
          `設定を適用しました（${result.matched}/${importedConfig.clips.length}本一致${result.missing ? `・${result.missing}本未選択` : ""}）`,
        );
      } else {
        rebuildTransitions();
        renderAll();
        await renderAt(playPosition, true);
        setStatus(`${additions.length}本の映像を追加しました`);
      }
    } catch (error) {
      additions.forEach((clip) => URL.revokeObjectURL(clip.url));
      setStatus(error.message, 0, 0, true);
    } finally {
      setBusy(false);
    }
  }

  function cutTime(clip) {
    return clamp(
      number(clip.skipOut, clip.onset),
      clip.onset,
      Math.max(clip.onset, clip.peak),
    );
  }

  function setBoundaries(clip, boundaries) {
    clip.onset = clamp(number(boundaries.onset, clip.onset), 0, clip.duration);
    clip.peak = clamp(
      number(boundaries.peak, clip.peak),
      clip.onset,
      clip.duration,
    );
    clip.skipOut = clamp(
      number(
        boundaries.skipOut,
        clip.skipOut || clip.onset + (clip.peak - clip.onset) * 0.7,
      ),
      clip.onset,
      clip.peak,
    );
    clip.plateauEnd = clamp(
      number(boundaries.plateauEnd, clip.plateauEnd || clip.peak),
      clip.peak,
      clip.duration,
    );
    clip.settle = clamp(
      number(boundaries.settle, clip.settle),
      clip.plateauEnd,
      clip.duration,
    );
    clip.skipIn = clamp(
      number(boundaries.skipIn, clip.skipIn || clip.plateauEnd),
      Math.min(clip.settle, clip.skipOut + 0.02),
      clip.settle,
    );
  }

  function clipPlan(clip, index = clips.indexOf(clip)) {
    const previousClip = clips[index - 1] || null;
    const cut = cutTime(clip);
    const skipIn = clamp(clip.skipIn || clip.settle, cut, clip.settle);
    const morphIn = previousClip ? setting("morphDuration", 0.2) : 0;
    const intro = previousClip ? 0 : setting("morphDuration", 0.2);
    const forward = setting("risePlaybackDuration", 0.45);
    const skip = setting("skipDurationSetting", 0.2);
    const decay = setting("decayDuration", 0.6);
    const hold = setting("holdDuration", 0.1);
    return {
      clip,
      index,
      previousClip,
      sourceStart: clip.onset,
      cut,
      skipIn,
      intro,
      morphIn,
      forward,
      skip,
      decay,
      hold,
      duration: intro + morphIn + forward + skip + decay + hold,
    };
  }

  function plans() {
    return clips.map(clipPlan);
  }

  function sequenceDuration() {
    return plans().reduce((sum, plan) => sum + plan.duration, 0);
  }

  function mapSequenceTime(time) {
    const all = plans();
    const total = all.reduce((sum, plan) => sum + plan.duration, 0);
    let cursor = clamp(time, 0, Math.max(0, total));
    const mapped = (plan, index, state) => ({ ...plan, index, ...state });
    for (let index = 0; index < all.length; index += 1) {
      const plan = all[index];
      if (cursor <= plan.duration || index === all.length - 1) {
        if (cursor <= plan.intro && plan.intro > 0) {
          return mapped(plan, index, {
            sourceTime: plan.clip.onset * clamp(cursor / plan.intro, 0, 1),
            phase: "INTRO",
          });
        }
        cursor -= plan.intro;
        if (cursor <= plan.morphIn && plan.morphIn > 0) {
          const transitionProgress = clamp(cursor / plan.morphIn, 0, 1);
          return mapped(plan, index, {
            clip: plan.previousClip,
            nextClip: plan.clip,
            sourceTime: Math.min(
              plan.previousClip.duration,
              plan.previousClip.settle + cursor,
            ),
            nextSourceTime: Math.max(
              0,
              plan.clip.onset - plan.morphIn + cursor,
            ),
            transitionProgress,
            phase: "MORPH_IN",
          });
        }
        cursor -= plan.morphIn;
        if (cursor <= plan.forward)
          return mapped(plan, index, {
            sourceTime:
              plan.sourceStart + (plan.cut - plan.sourceStart) * (cursor / plan.forward),
            phase: "FORWARD",
          });
        cursor -= plan.forward;
        if (cursor <= plan.skip) {
          const transitionProgress = clamp(cursor / plan.skip, 0, 1);
          return mapped(plan, index, {
            nextClip: plan.clip,
            sourceTime: plan.cut,
            nextSourceTime: plan.skipIn,
            transitionProgress,
            phase: "MORPH_SKIP",
          });
        }
        cursor -= plan.skip;
        if (cursor <= plan.decay) {
          return mapped(plan, index, {
            sourceTime:
              plan.skipIn +
              (plan.clip.settle - plan.skipIn) * (cursor / plan.decay),
            phase: "DECAY",
          });
        }
        cursor -= plan.decay;
        return mapped(plan, index, {
          sourceTime: plan.clip.settle,
          phase: "HOME",
        });
      }
      cursor -= plan.duration;
    }
    return null;
  }

  function loadVideoSlot(video, clip) {
    const secondary = video === el.transitionVideo;
    const loadedId = secondary ? transitionLoadedClipId : loadedClipId;
    if (loadedId === clip.id && video.src) return Promise.resolve();
    if (secondary) transitionLoadedClipId = clip.id;
    else {
      loadedClipId = clip.id;
      lastRenderedTime = -1;
    }
    video.src = clip.url;
    video.load();
    if (video.readyState >= 1) return Promise.resolve();
    return new Promise((resolve, reject) => {
      video.addEventListener("loadedmetadata", resolve, { once: true });
      video.addEventListener(
        "error",
        () => reject(new Error("映像を読み込めませんでした")),
        { once: true },
      );
    });
  }

  function seekVideoSlot(video, time) {
    const target = clamp(time, 0, Math.max(0, video.duration - 0.002));
    if (Math.abs(video.currentTime - target) < 0.008 && video.readyState >= 2)
      return Promise.resolve();
    return new Promise((resolve) => {
      video.addEventListener("seeked", () => requestAnimationFrame(resolve), {
        once: true,
      });
      video.currentTime = target;
    });
  }

  function sizeCanvas() {
    const reference = clips[0];
    const sourceWidth = Math.max(1, reference?.width || 1280);
    const sourceHeight = Math.max(1, reference?.height || 720);
    const scale = Math.min(1, PREVIEW_MAX_EDGE / Math.max(sourceWidth, sourceHeight));
    const width = Math.round(sourceWidth * scale);
    const height = Math.round(sourceHeight * scale);
    if (el.outputCanvas.width !== width || el.outputCanvas.height !== height) {
      el.outputCanvas.width = width;
      el.outputCanvas.height = height;
    }
  }

  function drawContained(
    ctx,
    media,
    sourceWidth,
    sourceHeight,
    targetWidth,
    targetHeight,
  ) {
    const scale = Math.min(
      targetWidth / sourceWidth,
      targetHeight / sourceHeight,
    );
    const width = sourceWidth * scale;
    const height = sourceHeight * scale;
    ctx.drawImage(
      media,
      (targetWidth - width) / 2,
      (targetHeight - height) / 2,
      width,
      height,
    );
  }

  function alignmentTransform(clip, targetWidth, targetHeight) {
    const reference = clips[0];
    const sourcePose = clip.alignmentPose;
    const targetPose = reference?.alignmentPose;
    if (
      !sourcePose ||
      !targetPose ||
      !clip.width ||
      !clip.height ||
      !reference.width ||
      !reference.height
    )
      return null;
    const sourceLeft = {
      x: sourcePose.left.x * clip.width,
      y: sourcePose.left.y * clip.height,
    };
    const sourceRight = {
      x: sourcePose.right.x * clip.width,
      y: sourcePose.right.y * clip.height,
    };
    const targetScaleX = targetWidth / reference.width;
    const targetScaleY = targetHeight / reference.height;
    const targetLeft = {
      x: targetPose.left.x * reference.width * targetScaleX,
      y: targetPose.left.y * reference.height * targetScaleY,
    };
    const targetRight = {
      x: targetPose.right.x * reference.width * targetScaleX,
      y: targetPose.right.y * reference.height * targetScaleY,
    };
    const sourceCenter = {
      x: (sourceLeft.x + sourceRight.x) / 2,
      y: (sourceLeft.y + sourceRight.y) / 2,
    };
    const targetCenter = {
      x: (targetLeft.x + targetRight.x) / 2,
      y: (targetLeft.y + targetRight.y) / 2,
    };
    const sourceAngle = Math.atan2(
      sourceRight.y - sourceLeft.y,
      sourceRight.x - sourceLeft.x,
    );
    const targetAngle = Math.atan2(
      targetRight.y - targetLeft.y,
      targetRight.x - targetLeft.x,
    );
    const sourceDistance = Math.hypot(
      sourceRight.x - sourceLeft.x,
      sourceRight.y - sourceLeft.y,
    );
    const targetDistance = Math.hypot(
      targetRight.x - targetLeft.x,
      targetRight.y - targetLeft.y,
    );
    if (sourceDistance < 1) return null;
    return {
      sourceCenter,
      targetCenter,
      rotation: targetAngle - sourceAngle,
      scale: targetDistance / sourceDistance,
    };
  }

  function transformPoint(point, clip, transform) {
    const x = point.x * clip.width - transform.sourceCenter.x;
    const y = point.y * clip.height - transform.sourceCenter.y;
    const cos = Math.cos(transform.rotation);
    const sin = Math.sin(transform.rotation);
    return [
      transform.targetCenter.x + (x * cos - y * sin) * transform.scale,
      transform.targetCenter.y + (x * sin + y * cos) * transform.scale,
    ];
  }

  function drawAlignedFrame(
    ctx,
    clip,
    targetWidth,
    targetHeight,
    media = el.sourceVideo,
  ) {
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, targetWidth, targetHeight);
    const transform = alignmentTransform(clip, targetWidth, targetHeight);
    if (!transform) {
      drawContained(
        ctx,
        media,
        clip.width,
        clip.height,
        targetWidth,
        targetHeight,
      );
      return;
    }
    ctx.save();
    ctx.translate(transform.targetCenter.x, transform.targetCenter.y);
    ctx.rotate(transform.rotation);
    ctx.scale(transform.scale, transform.scale);
    ctx.translate(-transform.sourceCenter.x, -transform.sourceCenter.y);
    ctx.drawImage(media, 0, 0, clip.width, clip.height);
    ctx.restore();
  }

  function resizeWorkCanvases(width, height) {
    [
      alignedCanvasA,
      alignedCanvasB,
      warpedCanvasA,
      warpedCanvasB,
      morphCanvas,
      maskCanvas,
    ].forEach((canvas) => {
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
    });
  }

  function nearestMeshSample(clip, time) {
    return clip.samples.reduce((best, sample) => {
      if (!sample.faceMesh) return best;
      return !best || Math.abs(sample.time - time) < Math.abs(best.time - time)
        ? sample
        : best;
    }, null);
  }

  function alignedMeshPoints(clip, faceMesh, width, height) {
    const transform = alignmentTransform(clip, width, height);
    if (!transform || !faceMesh) return null;
    return MORPH_LANDMARK_INDICES.map((index) =>
      transformPoint(faceMesh[index], clip, transform),
    );
  }

  function expandedTriangle(triangle) {
    const center = triangle.reduce(
      (sum, point) => [sum[0] + point[0] / 3, sum[1] + point[1] / 3],
      [0, 0],
    );
    return triangle.map((point) => {
      const dx = point[0] - center[0];
      const dy = point[1] - center[1];
      const length = Math.hypot(dx, dy) || 1;
      return [
        point[0] + (dx / length) * TRIANGLE_OVERDRAW,
        point[1] + (dy / length) * TRIANGLE_OVERDRAW,
      ];
    });
  }

  function drawTriangleImage(ctx, image, sourceTriangle, destinationTriangle) {
    const [[sx0, sy0], [sx1, sy1], [sx2, sy2]] = sourceTriangle;
    const [[dx0, dy0], [dx1, dy1], [dx2, dy2]] = destinationTriangle;
    const denominator =
      sx0 * (sy1 - sy2) + sx1 * (sy2 - sy0) + sx2 * (sy0 - sy1);
    if (Math.abs(denominator) < 1e-6) return;
    const a =
      (dx0 * (sy1 - sy2) + dx1 * (sy2 - sy0) + dx2 * (sy0 - sy1)) / denominator;
    const b =
      (dy0 * (sy1 - sy2) + dy1 * (sy2 - sy0) + dy2 * (sy0 - sy1)) / denominator;
    const c =
      (dx0 * (sx2 - sx1) + dx1 * (sx0 - sx2) + dx2 * (sx1 - sx0)) / denominator;
    const d =
      (dy0 * (sx2 - sx1) + dy1 * (sx0 - sx2) + dy2 * (sx1 - sx0)) / denominator;
    const e =
      (dx0 * (sx1 * sy2 - sx2 * sy1) +
        dx1 * (sx2 * sy0 - sx0 * sy2) +
        dx2 * (sx0 * sy1 - sx1 * sy0)) /
      denominator;
    const f =
      (dy0 * (sx1 * sy2 - sx2 * sy1) +
        dy1 * (sx2 * sy0 - sx0 * sy2) +
        dy2 * (sx0 * sy1 - sx1 * sy0)) /
      denominator;
    const expanded = expandedTriangle(destinationTriangle);
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

  function warpFace(
    sourceCanvas,
    sourcePoints,
    destinationPoints,
    triangles,
    targetCanvas,
  ) {
    const ctx = targetCanvas.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    for (let index = 0; index < triangles.length; index += 3) {
      const i0 = triangles[index],
        i1 = triangles[index + 1],
        i2 = triangles[index + 2];
      drawTriangleImage(
        ctx,
        sourceCanvas,
        [sourcePoints[i0], sourcePoints[i1], sourcePoints[i2]],
        [destinationPoints[i0], destinationPoints[i1], destinationPoints[i2]],
      );
    }
  }

  async function renderMorphFrame(
    ctx,
    mapped,
    width,
    height,
    seek = true,
    mediaA = el.sourceVideo,
    mediaB = el.transitionVideo,
  ) {
    const nextClip = mapped.nextClip;
    if (!nextClip) {
      drawAlignedFrame(ctx, mapped.clip, width, height);
      return;
    }
    if (seek) {
      await Promise.all([
        loadVideoSlot(mediaA, mapped.clip),
        loadVideoSlot(mediaB, nextClip),
      ]);
      await Promise.all([
        seekVideoSlot(mediaA, mapped.sourceTime),
        seekVideoSlot(mediaB, mapped.nextSourceTime),
      ]);
    }
    resizeWorkCanvases(width, height);
    drawAlignedFrame(
      alignedCanvasA.getContext("2d", { alpha: false }),
      mapped.clip,
      width,
      height,
      mediaA,
    );
    drawAlignedFrame(
      alignedCanvasB.getContext("2d", { alpha: false }),
      nextClip,
      width,
      height,
      mediaB,
    );
    const progress =
      mapped.transitionProgress *
      mapped.transitionProgress *
      (3 - 2 * mapped.transitionProgress);
    ctx.globalAlpha = 1;
    ctx.drawImage(alignedCanvasA, 0, 0);
    ctx.globalAlpha = progress;
    ctx.drawImage(alignedCanvasB, 0, 0);
    ctx.globalAlpha = 1;

    const sampleA = nearestMeshSample(
      mapped.clip,
      seek ? mapped.sourceTime : mediaA.currentTime,
    );
    const sampleB = nearestMeshSample(
      nextClip,
      seek ? mapped.nextSourceTime : mediaB.currentTime,
    );
    const pointsA = alignedMeshPoints(
      mapped.clip,
      sampleA?.faceMesh,
      width,
      height,
    );
    const pointsB = alignedMeshPoints(
      nextClip,
      sampleB?.faceMesh,
      width,
      height,
    );
    if (!pointsA || !pointsB || pointsA.length !== pointsB.length) return;
    const destinationPoints = pointsA.map((point, index) => [
      point[0] + (pointsB[index][0] - point[0]) * progress,
      point[1] + (pointsB[index][1] - point[1]) * progress,
    ]);
    const Delaunator = await delaunatorPromise;
    const triangles = Delaunator.from(destinationPoints).triangles;
    warpFace(
      alignedCanvasA,
      pointsA,
      destinationPoints,
      triangles,
      warpedCanvasA,
    );
    warpFace(
      alignedCanvasB,
      pointsB,
      destinationPoints,
      triangles,
      warpedCanvasB,
    );
    const morphCtx = morphCanvas.getContext("2d");
    morphCtx.setTransform(1, 0, 0, 1, 0, 0);
    morphCtx.clearRect(0, 0, width, height);
    morphCtx.globalAlpha = 1;
    morphCtx.drawImage(warpedCanvasA, 0, 0);
    morphCtx.globalAlpha = progress;
    morphCtx.drawImage(warpedCanvasB, 0, 0);
    morphCtx.globalAlpha = 1;

    const indexMap = new Map(
      MORPH_LANDMARK_INDICES.map((landmarkIndex, pointIndex) => [
        landmarkIndex,
        pointIndex,
      ]),
    );
    const oval = FACE_OVAL_INDICES.map(
      (index) => destinationPoints[indexMap.get(index)],
    ).filter(Boolean);
    const maskCtx = maskCanvas.getContext("2d");
    maskCtx.setTransform(1, 0, 0, 1, 0, 0);
    maskCtx.clearRect(0, 0, width, height);
    maskCtx.filter = `blur(${Math.max(8, Math.round(width * 0.012))}px)`;
    maskCtx.fillStyle = "#fff";
    maskCtx.beginPath();
    oval.forEach((point, index) =>
      index
        ? maskCtx.lineTo(point[0], point[1])
        : maskCtx.moveTo(point[0], point[1]),
    );
    maskCtx.closePath();
    maskCtx.fill();
    maskCtx.filter = "none";
    morphCtx.globalCompositeOperation = "destination-in";
    morphCtx.drawImage(maskCanvas, 0, 0);
    morphCtx.globalCompositeOperation = "source-over";
    ctx.drawImage(morphCanvas, 0, 0);
  }

  function updateStage(mapped) {
    el.stageTitle.textContent =
      mapped.phase === "MORPH_IN"
        ? `${mapped.index}. ${mapped.clip.name} → ${mapped.index + 1}. ${mapped.nextClip.name}`
        : `${mapped.index + 1}. ${mapped.clip.name}`;
    el.stagePhase.textContent = mapped.phase.startsWith("MORPH")
      ? `${mapped.phase === "MORPH_IN" ? "NEUTRAL → ONSET" : "PEAK SKIP"}  ${Math.round(mapped.transitionProgress * 100)}%`
      : mapped.phase === "HOME"
        ? `NEUTRAL  ${mapped.sourceTime.toFixed(2)}s`
        : `${mapped.phase}  ${mapped.sourceTime.toFixed(2)}s / CUT ${mapped.cut.toFixed(2)}s`;
  }

  function playbackRate(mapped) {
    if (mapped.phase === "INTRO") return mapped.clip.onset / mapped.intro;
    if (mapped.phase === "FORWARD")
      return (mapped.cut - mapped.sourceStart) / mapped.forward;
    if (mapped.phase === "DECAY")
      return (mapped.clip.settle - mapped.skipIn) / mapped.decay;
    return mapped.phase === "MORPH_IN" ? 1 : 0;
  }

  const loadedId = (video) =>
    video === el.transitionVideo ? transitionLoadedClipId : loadedClipId;

  function closestLoadedVideo(clip, time) {
    return [el.sourceVideo, el.transitionVideo]
      .filter((video) => loadedId(video) === clip.id)
      .sort(
        (a, b) =>
          Math.abs(a.currentTime - time) - Math.abs(b.currentTime - time),
      )[0];
  }

  async function seekIfNeeded(video, time) {
    if (Math.abs(video.currentTime - time) > 0.18) {
      video.pause();
      await seekVideoSlot(video, time);
    }
  }

  async function preparePreview(mapped, key) {
    if (mapped.phase.startsWith("MORPH")) {
      const mediaA =
        closestLoadedVideo(mapped.clip, mapped.sourceTime) || previewVideo;
      const mediaB =
        mediaA === el.sourceVideo ? el.transitionVideo : el.sourceVideo;
      await Promise.all([
        loadVideoSlot(mediaA, mapped.clip),
        loadVideoSlot(mediaB, mapped.nextClip),
      ]);
      await Promise.all([
        seekIfNeeded(mediaA, mapped.sourceTime),
        seekIfNeeded(mediaB, mapped.nextSourceTime),
      ]);
      previewMorphVideos = [mediaA, mediaB];
      previewVideo = mediaB;
    } else {
      previewVideo =
        closestLoadedVideo(mapped.clip, mapped.sourceTime) || previewVideo;
      await loadVideoSlot(previewVideo, mapped.clip);
      await seekIfNeeded(previewVideo, mapped.sourceTime);
      (previewVideo === el.sourceVideo
        ? el.transitionVideo
        : el.sourceVideo
      ).pause();
    }
    if (!playing || previewPhaseKey !== key) return;
    const rate = clamp(playbackRate(mapped), 0, 16);
    if (rate > 0) {
      previewVideo.playbackRate = Math.max(0.0625, rate);
      await previewVideo.play();
      if (mapped.phase === "MORPH_IN") {
        previewMorphVideos[0].playbackRate = 1;
        await previewMorphVideos[0].play();
      }
    } else {
      previewMorphVideos[0].pause();
      previewMorphVideos[1].pause();
    }
  }

  async function renderPreview(position) {
    const mapped = mapSequenceTime(position);
    if (!mapped) return;
    const key = `${mapped.index}:${mapped.phase}`;
    if (key !== previewPhaseKey) {
      previewPhaseKey = key;
      await preparePreview(mapped, key);
    }
    if (!playing || previewPhaseKey !== key) return;
    sizeCanvas();
    if (mapped.phase.startsWith("MORPH")) {
      await renderMorphFrame(
        outputContext,
        mapped,
        el.outputCanvas.width,
        el.outputCanvas.height,
        false,
        ...previewMorphVideos,
      );
    } else {
      drawAlignedFrame(
        outputContext,
        mapped.clip,
        el.outputCanvas.width,
        el.outputCanvas.height,
        previewVideo,
      );
    }
    updateStage(mapped);
  }

  async function renderAt(position, force = false) {
    const mapped = mapSequenceTime(position);
    if (!mapped) {
      outputContext.fillStyle = "#111416";
      outputContext.fillRect(
        0,
        0,
        el.outputCanvas.width,
        el.outputCanvas.height,
      );
      return;
    }
    sizeCanvas();
    if (mapped.phase.startsWith("MORPH")) {
      await renderMorphFrame(
        outputContext,
        mapped,
        el.outputCanvas.width,
        el.outputCanvas.height,
      );
      lastRenderedTime = -1;
    } else {
      await loadVideoSlot(el.sourceVideo, mapped.clip);
      if (force || Math.abs(lastRenderedTime - mapped.sourceTime) > 0.006) {
        await seekVideoSlot(el.sourceVideo, mapped.sourceTime);
        drawAlignedFrame(
          outputContext,
          mapped.clip,
          el.outputCanvas.width,
          el.outputCanvas.height,
        );
        lastRenderedTime = mapped.sourceTime;
      }
    }
    updateStage(mapped);
    updateTransport(mapped.index);
  }

  function updateTransport(
    activeIndex = mapSequenceTime(playPosition)?.index ?? -1,
  ) {
    const total = sequenceDuration();
    el.playhead.max = String(Math.max(0.001, total));
    el.playhead.value = String(clamp(playPosition, 0, total));
    el.sequenceTimeline.style.setProperty(
      "--timeline-progress",
      `${total ? clamp((playPosition / total) * 100, 0, 100) : 0}%`,
    );
    el.timeOutput.textContent = `${playPosition.toFixed(2)} / ${total.toFixed(2)}s`;
    el.playButton.textContent = playing ? "❚❚" : "▶";
    [...el.sequenceTimeline.children].forEach((node, index) =>
      node.classList.toggle("active", index === activeIndex),
    );
  }

  function tick(now) {
    if (!playing) return;
    const total = sequenceDuration();
    playPosition = (now - playStartedAt) / 1000;
    if (playPosition >= total) {
      if (el.loopToggle.checked && total > 0) {
        playPosition %= total;
        playStartedAt = now - playPosition * 1000;
      } else {
        playPosition = total;
        pause();
      }
    }
    const mapped = mapSequenceTime(playPosition);
    updateTransport(mapped?.index ?? -1);
    if (mapped) updateStage(mapped);
    if (!previewRendering) {
      previewRendering = true;
      renderPreview(playPosition)
        .catch((error) => {
          pause();
          setStatus(error.message, 0, 0, true);
        })
        .finally(() => {
          previewRendering = false;
        });
    }
    if (playing) animationId = requestAnimationFrame(tick);
  }

  function play() {
    const total = sequenceDuration();
    if (!total || busy) return;
    if (playPosition >= total - 0.001) playPosition = 0;
    playing = true;
    previewPhaseKey = "";
    playStartedAt = performance.now() - playPosition * 1000;
    updateTransport();
    animationId = requestAnimationFrame(tick);
  }

  function pause() {
    playing = false;
    cancelAnimationFrame(animationId);
    el.sourceVideo.pause();
    el.transitionVideo.pause();
    previewPhaseKey = "";
    updateTransport();
  }

  function averageLandmarkPoint(points, indices) {
    const valid = indices.map((index) => points[index]).filter(Boolean);
    if (!valid.length) return null;
    return valid.reduce(
      (sum, point) => ({
        x: sum.x + point.x / valid.length,
        y: sum.y + point.y / valid.length,
        z: sum.z + (point.z || 0) / valid.length,
      }),
      { x: 0, y: 0, z: 0 },
    );
  }

  function normalizeMediaPipeLandmarks(points) {
    if (!points || points.length < 426) return null;
    const leftEye = averageLandmarkPoint(points, [33, 133]);
    const rightEye = averageLandmarkPoint(points, [362, 263]);
    if (!leftEye || !rightEye) return null;
    const center = {
      x: (leftEye.x + rightEye.x) / 2,
      y: (leftEye.y + rightEye.y) / 2,
      z: (leftEye.z + rightEye.z) / 2,
    };
    const dx = rightEye.x - leftEye.x;
    const dy = rightEye.y - leftEye.y;
    const scale = Math.hypot(dx, dy);
    if (scale < 1e-5) return null;
    const cos = dx / scale;
    const sin = dy / scale;
    return EXPRESSION_LANDMARK_INDICES.flatMap((index) => {
      const point = points[index];
      const x = point.x - center.x;
      const y = point.y - center.y;
      return [
        (x * cos + y * sin) / scale,
        (-x * sin + y * cos) / scale,
        (((point.z || 0) - center.z) / scale) * 0.5,
      ];
    });
  }

  function mediaPipeEyePose(points) {
    if (!points || points.length < 363) return null;
    const left = averageLandmarkPoint(points, [33, 133]);
    const right = averageLandmarkPoint(points, [362, 263]);
    return left && right ? { left, right } : null;
  }

  function averageEyePoses(poses) {
    if (!poses.length) return null;
    const averageSide = (side) =>
      poses.reduce(
        (sum, pose) => ({
          x: sum.x + pose[side].x / poses.length,
          y: sum.y + pose[side].y / poses.length,
          z: sum.z + pose[side].z / poses.length,
        }),
        { x: 0, y: 0, z: 0 },
      );
    return { left: averageSide("left"), right: averageSide("right") };
  }

  function averageVector(vectors) {
    if (!vectors.length) return null;
    return vectors[0].map(
      (_, index) =>
        vectors.reduce((sum, vector) => sum + vector[index], 0) /
        vectors.length,
    );
  }

  function vectorDistance(a, b) {
    if (!a || !b || a.length !== b.length) return null;
    return Math.sqrt(
      a.reduce((sum, value, index) => sum + (value - b[index]) ** 2, 0) /
        a.length,
    );
  }

  function prepareLandmarkDistances(clip) {
    const shapes = clip.samples
      .filter((sample) => sample.landmarks)
      .map((sample) => sample.landmarks);
    if (shapes.length < 3) {
      clip.samples.forEach((sample) => {
        sample.landmarkDistance = null;
        sample.landmarkDelta = null;
      });
      return;
    }
    const baselineCount = Math.max(
      3,
      Math.min(shapes.length, Math.round(shapes.length * 0.15)),
    );
    const startShape = averageVector(shapes.slice(0, baselineCount));
    const endShape = averageVector(shapes.slice(-baselineCount));
    clip.samples.forEach((sample, index) => {
      const nearbyShapes = clip.samples
        .slice(Math.max(0, index - 2), index + 3)
        .map((item) => item.landmarks)
        .filter(Boolean);
      const shape = averageVector(nearbyShapes);
      sample.landmarkDistance = shape
        ? Math.min(vectorDistance(shape, startShape), vectorDistance(shape, endShape))
        : null;
      sample.landmarkDelta = shape;
    });
  }

  function updateAlignmentPose(clip) {
    const radius = Math.max(0.2, setting("sampleInterval", 0.08) * 3);
    let poses = clip.samples
      .filter(
        (sample) =>
          sample.eyePose && Math.abs(sample.time - clip.onset) <= radius,
      )
      .map((sample) => sample.eyePose);
    if (!poses.length)
      poses = clip.samples
        .filter((sample) => sample.eyePose)
        .slice(0, 5)
        .map((sample) => sample.eyePose);
    clip.alignmentPose = averageEyePoses(poses);
  }

  function updateSkipMatch(clip) {
    if (clip.skipInManual || !clip.samples.length) return;
    const target = clip.samples.reduce(
      (best, sample) =>
        sample.landmarkDelta &&
        (!best || Math.abs(sample.time - cutTime(clip)) < Math.abs(best.time - cutTime(clip)))
          ? sample
          : best,
      null,
    );
    const match = clip.samples
      .filter(
        (sample) =>
          sample.landmarkDelta &&
          sample.time >= clip.plateauEnd &&
          sample.time <= clip.settle,
      )
      .reduce((best, sample) => {
        const distance = vectorDistance(target?.landmarkDelta, sample.landmarkDelta);
        return distance !== null && (!best || distance < best.distance)
          ? { time: sample.time, distance }
          : best;
      }, null);
    if (match) {
      clip.skipIn = match.time;
      clip.skipMatchDistance = match.distance;
    }
  }

  function rebuildTransitions() {
    clips.forEach(updateSkipMatch);
  }

  function smoothed(clip) {
    return clip.samples.map((sample, index) => {
      const window = clip.samples.slice(Math.max(0, index - 2), index + 3);
      const expressionSamples = window.filter((item) => item.expressions);
      const landmarkSamples = window.filter((item) =>
        Number.isFinite(item.landmarkDistance),
      );
      const values = expressionSamples.length
        ? Object.fromEntries(
            EMOTIONS.map((emotion) => [
              emotion,
              expressionSamples.reduce(
                (sum, item) => sum + item.expressions[emotion],
                0,
              ) / expressionSamples.length,
            ]),
          )
        : null;
      const landmarkMotion = landmarkSamples.length
        ? landmarkSamples.reduce(
            (sum, item) => sum + item.landmarkDistance,
            0,
          ) / landmarkSamples.length
        : null;
      return { ...sample, smooth: values, landmarkMotion };
    });
  }

  function metricProfile(values, minimumAmplitude) {
    const valid = values.filter(Number.isFinite);
    if (valid.length < 3) return { baseline: 0, amplitude: 0, usable: false };
    const baselineCount = Math.max(
      1,
      Math.min(valid.length, Math.round(valid.length * 0.15)),
    );
    const edges = [
      ...valid.slice(0, baselineCount),
      ...valid.slice(-baselineCount),
    ];
    const baseline = edges.reduce((sum, value) => sum + value, 0) / edges.length;
    const amplitude = Math.max(...valid) - baseline;
    return { baseline, amplitude, usable: amplitude >= minimumAmplitude };
  }

  function crossing(items, level, direction, start, end, sustain = 1) {
    for (let i = Math.max(1, start); i <= end; i += 1) {
      const a = items[i - 1],
        b = items[i];
      const hit =
        direction === "up"
          ? a.value <= level && b.value >= level
          : a.value >= level && b.value <= level;
      const held = items
        .slice(i, Math.min(items.length, i + sustain))
        .every((item) =>
          direction === "up" ? item.value >= level : item.value <= level,
        );
      if (!hit || !held) continue;
      const ratio =
        Math.abs(b.value - a.value) < 1e-6
          ? 0
          : clamp((level - a.value) / (b.value - a.value), 0, 1);
      return a.time + (b.time - a.time) * ratio;
    }
    return null;
  }

  function detectBoundaries(clip) {
    prepareLandmarkDistances(clip);
    const data = smoothed(clip)
      .filter(
        (sample) => sample.smooth || Number.isFinite(sample.landmarkMotion),
      )
      .map((sample) => ({
        time: sample.time,
        emotionValue: sample.smooth
          ? sample.smooth[clip.emotion] /
            Math.max(
              0.0001,
              sample.smooth[BASE_EMOTION] + sample.smooth[clip.emotion],
            )
          : null,
        landmarkValue: sample.landmarkMotion,
      }));
    if (data.length < 3) {
      updateAlignmentPose(clip);
      return false;
    }
    const emotionProfile = metricProfile(
      data.map((item) => item.emotionValue),
      0.03,
    );
    const landmarkProfile = metricProfile(
      data.map((item) => item.landmarkValue),
      0.003,
    );
    if (!emotionProfile.usable && !landmarkProfile.usable) {
      clip.analysisSeries = [];
      updateAlignmentPose(clip);
      return false;
    }
    data.forEach((item) => {
      const emotionProgress =
        emotionProfile.usable && Number.isFinite(item.emotionValue)
          ? clamp(
              (item.emotionValue - emotionProfile.baseline) /
                emotionProfile.amplitude,
              0,
              1,
            )
          : null;
      const landmarkProgress =
        landmarkProfile.usable && Number.isFinite(item.landmarkValue)
          ? clamp(
              (item.landmarkValue - landmarkProfile.baseline) /
                landmarkProfile.amplitude,
              0,
              1,
            )
          : null;
      let weightedValue = 0;
      let weight = 0;
      if (landmarkProgress !== null) {
        weightedValue += landmarkProgress * LANDMARK_WEIGHT;
        weight += LANDMARK_WEIGHT;
      }
      if (emotionProgress !== null) {
        weightedValue += emotionProgress * EXPRESSION_WEIGHT;
        weight += EXPRESSION_WEIGHT;
      }
      item.emotionProgress = emotionProgress;
      item.landmarkProgress = landmarkProgress;
      item.value = weight ? weightedValue / weight : 0;
    });
    clip.analysisSeries = data.map((item) => ({
      time: item.time,
      landmark: item.landmarkProgress,
      progression: item.value,
    }));
    let peakIndex = 0;
    data.forEach((item, index) => {
      if (item.value > data[peakIndex].value) peakIndex = index;
    });
    const onset =
      crossing(data, 0.15, "up", 1, peakIndex, 3) ??
      data[Math.max(0, peakIndex - 1)].time;
    const peak =
      crossing(data, 0.9, "up", 1, peakIndex, 2) ?? data[peakIndex].time;
    let plateauEndIndex = peakIndex;
    data.forEach((item, index) => {
      if (index >= peakIndex && item.value >= 0.9) plateauEndIndex = index;
    });
    const plateauEnd =
      crossing(data, 0.9, "down", plateauEndIndex + 1, data.length - 1) ??
      data[plateauEndIndex].time;
    const settle =
      crossing(data, 0.15, "down", plateauEndIndex + 1, data.length - 1, 3) ??
      clip.duration;
    const boundedOnset = clamp(onset, 0, clip.duration);
    setBoundaries(clip, {
      onset: boundedOnset,
      peak: Math.max(boundedOnset + 0.01, peak),
      skipOut: boundedOnset + (Math.max(boundedOnset + 0.01, peak) - boundedOnset) * 0.7,
      plateauEnd: Math.max(peak, plateauEnd),
      settle: Math.max(peak, settle),
    });
    clip.skipInManual = false;
    updateSkipMatch(clip);
    updateAlignmentPose(clip);
    return true;
  }

  async function analyzeClip(clip, progressOffset = 0, progressTotal = 1) {
    await models;
    const faceLandmarker = await getFaceLandmarker();
    const { Input, ALL_FORMATS, BlobSource, CanvasSink } =
      await mediabunnyPromise;
    const input = new Input({
      formats: ALL_FORMATS,
      source: new BlobSource(clip.file),
    });
    const track = await input.getPrimaryVideoTrack();
    if (!track || !(await track.canDecode()))
      throw new Error(`${clip.name} をデコードできません`);
    const firstTimestamp = await track.getFirstTimestamp();
    clip.duration = await track.computeDuration();
    const sink = new CanvasSink(track, {
      width: 320,
      alpha: false,
      poolSize: 2,
    });
    const interval = setting("sampleInterval", 0.08);
    const count = Math.max(2, Math.ceil(clip.duration / interval));
    clip.samples = [];
    clip.analysisSeries = [];
    let nextSample = 0;
    for await (const wrapped of sink.canvases()) {
      const time = wrapped.timestamp - firstTimestamp;
      if (time + 0.001 < nextSample) continue;
      nextSample = time + interval;
      setStatus(
        `${clip.name} を連続解析中 ${clip.samples.length + 1}/${count}`,
        progressOffset + time / Math.max(clip.duration, 0.001),
        progressTotal,
      );
      const frame = wrapped.canvas;
      if (
        analysisCanvas.width !== frame.width ||
        analysisCanvas.height !== frame.height
      ) {
        analysisCanvas.width = frame.width;
        analysisCanvas.height = frame.height;
      }
      analysisContext.drawImage(frame, 0, 0);
      const landmarkResult = faceLandmarker.detect(analysisCanvas);
      const detectedMesh = landmarkResult.faceLandmarks?.[0];
      const faceMesh =
        detectedMesh?.map((point) => ({
          x: point.x,
          y: point.y,
        })) || null;
      const landmarks = normalizeMediaPipeLandmarks(detectedMesh);
      const eyePose = mediaPipeEyePose(detectedMesh);
      const detection = await faceapi
        .detectSingleFace(analysisCanvas, detectorOptions)
        .withFaceExpressions();
      clip.samples.push({
        time,
        landmarks,
        faceMesh,
        eyePose,
        expressions: detection
          ? Object.fromEntries(
              EMOTIONS.map((emotion) => [
                emotion,
                detection.expressions[emotion] || 0,
              ]),
            )
          : null,
      });
      if (clip.samples.length % 8 === 0)
        await new Promise(requestAnimationFrame);
    }
    if (!clip.samples.length) throw new Error(`${clip.name} に映像フレームがありません`);
    clip.landmarkDetectionRate =
      clip.samples.filter((sample) => sample.landmarks).length /
      clip.samples.length;
    clip.analyzed = detectBoundaries(clip);
    rebuildTransitions();
  }

  async function runAnalysis(targets) {
    if (busy || !modelsReady || !targets.length) return;
    pause();
    setBusy(true);
    try {
      for (let i = 0; i < targets.length; i += 1) {
        await analyzeClip(targets[i], i, targets.length);
        renderAll();
      }
      const failed = targets.filter((clip) => !clip.analyzed).length;
      setStatus(
        failed
          ? `${targets.length}本を解析しました（${failed}本はランドマークまたは表情変化を検出できませんでした）`
          : `${targets.length}本のMediaPipe解析が完了しました`,
      );
    } catch (error) {
      console.error(error);
      setStatus(`解析に失敗しました: ${error.message}`, 0, 0, true);
    } finally {
      setBusy(false);
      renderAll();
      await renderAt(playPosition, true).catch(() => {});
    }
  }

  function moveClip(id, delta) {
    const from = clips.findIndex((clip) => clip.id === id);
    const to = clamp(from + delta, 0, clips.length - 1);
    if (from === to) return;
    pause();
    clips.splice(to, 0, clips.splice(from, 1)[0]);
    rebuildTransitions();
    playPosition = 0;
    renderAll();
    renderAt(0, true);
  }

  function removeClip(id) {
    pause();
    const index = clips.findIndex((clip) => clip.id === id);
    if (index < 0) return;
    URL.revokeObjectURL(clips[index].url);
    clips.splice(index, 1);
    if (selectedId === id)
      selectedId = clips[Math.min(index, clips.length - 1)]?.id ?? null;
    if (loadedClipId === id) {
      loadedClipId = null;
      el.sourceVideo.removeAttribute("src");
      el.sourceVideo.load();
    }
    if (transitionLoadedClipId === id) {
      transitionLoadedClipId = null;
      el.transitionVideo.removeAttribute("src");
      el.transitionVideo.load();
    }
    rebuildTransitions();
    playPosition = 0;
    renderAll();
    if (clips.length) renderAt(0, true);
    else clearStage();
  }

  function phasePercent(value, duration) {
    return `${clamp((value / Math.max(0.001, duration)) * 100, 0, 100)}%`;
  }

  function renderClipList() {
    el.clipCount.textContent = `${clips.length} clips`;
    const addButton = document.createElement("button");
    addButton.type = "button";
    addButton.className = "clipEmpty";
    addButton.dataset.action = "add";
    addButton.disabled = busy;
    addButton.setAttribute("aria-label", "表情映像を追加");
    addButton.innerHTML = '<img src="icon-plus.png" alt="">';
    if (!clips.length) {
      el.clipList.replaceChildren(addButton);
      return;
    }
    const cards = clips.map((clip, index) => {
      const plan = clipPlan(clip, index);
      const card = document.createElement("article");
      card.className = `clipCard${clip.id === selectedId ? " selected" : ""}`;
      card.dataset.id = clip.id;
      card.innerHTML = `
        <div class="clipTop">
          <span class="clipIndex">${String(index + 1).padStart(2, "0")}</span>
          <div><div class="clipName" title="${escapeHtml(clip.name)}">${escapeHtml(clip.name)}</div><div class="clipMeta">${LABELS[clip.emotion]} · IN ${seconds(plan.sourceStart)} · SKIP ${seconds(plan.cut)}→${seconds(plan.skipIn)} · OUT ${seconds(clip.settle)}</div></div>
          <div class="clipActions"><button class="iconButton" data-action="up" title="上へ">↑</button><button class="iconButton" data-action="down" title="下へ">↓</button><button class="iconButton" data-action="remove" title="削除">×</button></div>
        </div>
        <div class="phaseBar"><i class="before" style="width:${phasePercent(clip.onset, clip.duration)}"></i><i class="rise" style="width:${phasePercent(clip.peak - clip.onset, clip.duration)}"></i><i class="settle" style="width:${phasePercent(clip.settle - clip.peak, clip.duration)}"></i></div>
        <div class="phaseLabels"><span>発生 ${seconds(clip.onset)}</span><span>再開 ${seconds(clip.skipIn)}</span><span>${clip.samples.length ? `MP ${Math.round(clip.landmarkDetectionRate * 100)}%${clip.analyzed ? "" : " / 境界未検出"}` : "未解析"}</span></div>`;
      return card;
    });
    el.clipList.replaceChildren(...cards, addButton);
    const selectedCard = cards.find(
      (card) => Number(card.dataset.id) === selectedId,
    );
    if (selectedCard) selectedCard.append(el.clipEditor);
  }

  function escapeHtml(value) {
    const node = document.createElement("span");
    node.textContent = value;
    return node.innerHTML;
  }

  function timelinePart(baseClass, part, total) {
    const [variant, duration, label, description] = part;
    const node = document.createElement("span");
    node.className = `${baseClass} ${variant}`;
    node.style.width = `${(duration / Math.max(0.001, total)) * 100}%`;
    node.setAttribute("aria-label", label);
    node.title = `${description} ${seconds(duration)}`;
    return node;
  }

  function timelineMarker(variant, offset, total, title) {
    const marker = document.createElement("i");
    marker.className = `timelineEvent ${variant}`;
    marker.style.left = `${(offset / Math.max(0.001, total)) * 100}%`;
    marker.title = title;
    return marker;
  }

  function renderTimeline() {
    const all = plans();
    const total = all.reduce((sum, plan) => sum + plan.duration, 0);
    el.sequenceTimeline.replaceChildren(
      ...all.map((plan, index) => {
        const segment = document.createElement("div");
        segment.className = "timelineClip";
        segment.style.width = `${(plan.duration / Math.max(0.001, total)) * 100}%`;
        const clipNumber = String(index + 1).padStart(2, "0");
        const header = document.createElement("div");
        header.className = "timelineClipLabel";
        header.textContent = `${clipNumber} ${LABELS[plan.clip.emotion]} · ${seconds(plan.duration)}`;

        const meaning = document.createElement("div");
        meaning.className = "timelineMeaning";
        const phases = [
          [
            "intro",
            plan.intro + plan.morphIn,
            "導入",
            index === 0 ? "冒頭→発生" : "前素材の無表情→発生",
          ],
          ["expression", plan.forward, "表情化", "発生→上昇終了"],
          ["skip", plan.skip, "スキップ", "ピーク維持区間をモーフで省略"],
          ["decay", plan.decay, "収束", "下降側→無表情"],
          ["hold", plan.hold, "保持", "無表情保持"],
        ];
        meaning.replaceChildren(
          ...phases
            .filter(([, duration]) => duration > 0)
            .map((part) => timelinePart("timelinePhase", part, plan.duration)),
        );

        const composition = document.createElement("div");
        composition.className = "timelineComposition";
        const sources = [
          [
            index === 0 ? "single" : "morph",
            plan.intro + plan.morphIn,
            "導入",
            index === 0 ? `${plan.clip.name}のみ` : "前素材と次素材のモーフ",
          ],
          ["single", plan.forward, `${clipNumber}のみ`, `${plan.clip.name}のみ`],
          ["morph", plan.skip, "スキップ", "同一素材内のピークスキップ・モーフ"],
          ["single", plan.decay, `${clipNumber}のみ`, `${plan.clip.name}のみ`],
          ["home", plan.hold, "無表情", "無表情のみ"],
        ];
        composition.replaceChildren(
          ...sources
            .filter(([, duration]) => duration > 0)
            .map((part) => timelinePart("timelineSource", part, plan.duration)),
        );

        const onsetOffset = plan.intro + plan.morphIn;
        const skipOutOffset = onsetOffset + plan.forward;
        const skipInOffset = skipOutOffset + plan.skip;
        const events = [
          ["onset", onsetOffset, `表情発生 ${seconds(plan.clip.onset)}`],
          ["skipPoint", skipOutOffset, `スキップ開始 ${seconds(plan.cut)}`],
          ["skipPoint", skipInOffset, `下降側を再開 ${seconds(plan.skipIn)}`],
        ].map(([variant, offset, title]) =>
          timelineMarker(variant, offset, plan.duration, title),
        );
        segment.replaceChildren(header, meaning, composition, ...events);
        segment.title = `${plan.clip.name}: 合計 ${seconds(plan.duration)}`;
        return segment;
      }),
    );
    updateTransport();
  }

  function renderEditor() {
    const clip = selectedClip();
    el.clipEditor.hidden = !clip;
    if (!clip) {
      el.clipPreviewVideo.removeAttribute("src");
      el.clipPreviewMeta.textContent = "素材を選択してください";
      return;
    }
    if (el.clipPreviewVideo.dataset.clipId !== String(clip.id)) {
      el.clipPreviewVideo.pause();
      el.clipPreviewVideo.src = clip.url;
      el.clipPreviewVideo.dataset.clipId = clip.id;
      el.clipPreviewVideo.addEventListener(
        "loadedmetadata",
        () => {
          el.clipPreviewVideo.currentTime = cutTime(clip);
        },
        { once: true },
      );
    }
    el.clipPreviewMeta.textContent = `${clip.name} · ${clip.width}×${clip.height} · ${seconds(clip.duration)}`;
    el.emotionSelect.value = clip.emotion;
    el.emotionLegend.textContent = LABELS[clip.emotion];
    el.skipOutInput.value = cutTime(clip).toFixed(2);
    el.skipInInput.value = clip.skipIn.toFixed(2);
    el.skipOutInput.max = clip.duration;
    el.skipInInput.max = clip.duration;
    [el.skipOutRange, el.skipInRange].forEach((input) => {
      input.max = clip.duration;
      input.disabled = false;
    });
    el.skipOutRange.value = cutTime(clip);
    el.skipInRange.value = clip.skipIn;
    el.skipOutDuration.textContent = `上昇側 ${seconds(cutTime(clip))}`;
    el.skipDuration.textContent = `下降側 ${seconds(clip.skipIn)}`;
    el.detectionBadge.textContent = clip.samples.length
      ? `MediaPipe ${Math.round(clip.landmarkDetectionRate * 100)}%${clip.analyzed ? ` / SKIP ${clip.skipIn.toFixed(2)}s` : " / 境界未検出"}`
      : "未解析";
    el.detectionBadge.classList.toggle("ready", clip.analyzed);
    drawChart(clip);
  }

  function drawChart(clip) {
    const canvas = el.analysisChart;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    const pad = { x: 7, y: 8, bottom: 16 };
    const x = (time) =>
      pad.x +
      (time / Math.max(0.001, clip.duration)) * (rect.width - pad.x * 2);
    const y = (value) =>
      pad.y + (1 - clamp(value, 0, 1)) * (rect.height - pad.y - pad.bottom);
    ctx.strokeStyle = "#eceeec";
    ctx.lineWidth = 1;
    [0.25, 0.5, 0.75].forEach((value) => {
      ctx.beginPath();
      ctx.moveTo(pad.x, y(value));
      ctx.lineTo(rect.width - pad.x, y(value));
      ctx.stroke();
    });
    ctx.fillStyle = "rgba(139, 111, 192, .16)";
    ctx.fillRect(
      x(cutTime(clip)),
      pad.y,
      Math.max(0, x(clip.skipIn) - x(cutTime(clip))),
      rect.height - pad.y - pad.bottom,
    );
    const drawLine = (samples, valueOf, color, width) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      let started = false;
      samples.forEach((sample) => {
        const value = valueOf(sample);
        if (!Number.isFinite(value)) {
          started = false;
          return;
        }
        const point = [x(sample.time), y(value)];
        started ? ctx.lineTo(...point) : ctx.moveTo(...point);
        started = true;
      });
      ctx.stroke();
    };
    const data = smoothed(clip);
    [
      [BASE_EMOTION, COLORS.neutral],
      [clip.emotion, COLORS.target],
    ].forEach(([emotion, color]) =>
      drawLine(data, (sample) => sample.smooth?.[emotion], color, 2),
    );
    [
      ["landmark", LANDMARK_COLOR, 2],
      ["progression", PROGRESSION_COLOR, 2.5],
    ].forEach(([key, color, width]) =>
      drawLine(clip.analysisSeries, (sample) => sample[key], color, width),
    );
    [
      [clip.onset, COLORS.onset],
      [clip.peak, COLORS.peak],
      [clip.plateauEnd, COLORS.peak],
      [clip.settle, COLORS.settle],
    ].forEach(([time, color]) => {
      ctx.strokeStyle = color;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(x(time), pad.y);
      ctx.lineTo(x(time), rect.height - pad.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
    });
    ctx.fillStyle = COLORS.target;
    [cutTime(clip), clip.skipIn].forEach((time) =>
      ctx.fillRect(x(time) - 2, pad.y, 4, rect.height - pad.y - pad.bottom),
    );
  }

  function renderAll() {
    renderClipList();
    renderTimeline();
    renderEditor();
    updateControls();
  }

  function clearStage() {
    outputContext.fillStyle = "#111416";
    outputContext.fillRect(0, 0, el.outputCanvas.width, el.outputCanvas.height);
    el.stageTitle.textContent = "";
    el.stagePhase.textContent = "";
    playPosition = 0;
    updateTransport();
  }

  function updateSelectedBoundaries(event) {
    const clip = selectedClip();
    if (!clip) return;
    updateSkipRange(
      el.skipOutInput.value,
      el.skipInInput.value,
      event.target.value,
    );
    renderAll();
  }

  function updateSkipRange(start, end, previewTime) {
    const clip = selectedClip();
    if (!clip) return;
    clip.skipOut = clamp(number(start, clip.skipOut), clip.onset, clip.peak);
    clip.skipIn = clamp(
      number(end, clip.skipIn),
      Math.min(clip.settle, clip.skipOut + 0.02),
      clip.settle,
    );
    clip.skipInManual = true;
    el.skipOutRange.value = clip.skipOut;
    el.skipInRange.value = clip.skipIn;
    el.skipOutInput.value = clip.skipOut.toFixed(2);
    el.skipInInput.value = clip.skipIn.toFixed(2);
    el.skipOutDuration.textContent = `上昇側 ${seconds(clip.skipOut)}`;
    el.skipDuration.textContent = `下降側 ${seconds(clip.skipIn)}`;
    el.clipPreviewVideo.currentTime = previewTime;
    pause();
    playPosition = clamp(playPosition, 0, sequenceDuration());
    renderTimeline();
    drawChart(clip);
  }

  function download(blob, fileName) {
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = fileName;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function applySetting(input, value) {
    if (!input || !Number.isFinite(Number(value))) return;
    input.value = String(
      clamp(
        Number(value),
        number(input.min, -Infinity),
        number(input.max, Infinity),
      ),
    );
  }

  async function applyImportedConfig(config, announce = true) {
    const settings = config.settings || {};
    PLAYBACK_SETTINGS.forEach(([id, key]) =>
      applySetting(
        el[id],
        settings[key] ??
          (id === "skipDurationSetting"
            ? settings.rewindDuration
            : id === "decayDuration"
              ? settings.returnDuration
              : id === "morphDuration"
                ? settings.introDuration
                : NaN),
      ),
    );
    applySetting(el.sampleInterval, settings.sampleInterval);

    const configClips = [...config.clips].sort(
      (a, b) => number(a.order, 0) - number(b.order, 0),
    );
    const remaining = [...clips];
    const ordered = [];
    let matched = 0;
    configClips.forEach((saved) => {
      const source = String(saved.source || "");
      let matchIndex = remaining.findIndex((clip) => clip.name === source);
      if (matchIndex < 0) {
        matchIndex = remaining.findIndex(
          (clip) => clip.name.toLowerCase() === source.toLowerCase(),
        );
      }
      if (matchIndex < 0) return;
      const clip = remaining.splice(matchIndex, 1)[0];
      if (EMOTIONS.includes(saved.emotion) && saved.emotion !== BASE_EMOTION) {
        clip.emotion = saved.emotion;
      }
      const boundaries = { ...(saved.boundaries || {}) };
      if (!Number.isFinite(Number(boundaries.skipOut))) {
        const ratio = clamp(number(saved.cutRatio, 0.7), 0.1, 0.99);
        const onset = number(boundaries.onset, clip.onset);
        const peak = number(boundaries.peak, clip.peak);
        boundaries.skipOut = onset + (peak - onset) * ratio;
      }
      setBoundaries(clip, boundaries);
      clip.skipInManual = Number.isFinite(Number(saved.boundaries?.skipIn));
      if (clip.samples.length) updateAlignmentPose(clip);
      ordered.push(clip);
      matched += 1;
    });
    clips = [...ordered, ...remaining];
    selectedId = clips.some((clip) => clip.id === selectedId)
      ? selectedId
      : (clips[0]?.id ?? null);
    rebuildTransitions();
    pause();
    playPosition = 0;
    renderAll();
    if (clips.length) await renderAt(0, true);
    else clearStage();
    const missing = Math.max(0, configClips.length - matched);
    if (announce) {
      setStatus(
        clips.length
          ? `設定JSONを読み込みました（${matched}/${configClips.length}本一致${missing ? `・${missing}本未選択` : ""}）`
          : `設定JSONを読み込みました。${configClips.length}本の元動画を追加してください`,
      );
    }
    return { matched, missing };
  }

  async function importJson(file) {
    if (!file || busy) return;
    setBusy(true);
    try {
      const config = JSON.parse(await file.text());
      if (
        !config ||
        typeof config !== "object" ||
        !Array.isArray(config.clips)
      ) {
        throw new Error("Incomplete Deformationの設定JSONではありません");
      }
      importedConfig = config;
      await applyImportedConfig(config);
    } catch (error) {
      console.error(error);
      setStatus(`設定JSONを読み込めませんでした: ${error.message}`, 0, 0, true);
    } finally {
      setBusy(false);
    }
  }

  function exportJson() {
    const data = {
      version: 5,
      mode: "incomplete-deformation-peak-skip",
      algorithm: "fixed-tempo_rise_peak-skip_natural-decay",
      settings: {
        ...Object.fromEntries(
          PLAYBACK_SETTINGS.map(([id, key, fallback]) => [
            key,
            number(el[id].value, fallback),
          ]),
        ),
        introDuration: number(el.morphDuration.value, 0.2),
        sampleInterval: number(el.sampleInterval.value, 0.08),
      },
      clips: clips.map((clip, index) => {
        const plan = clipPlan(clip, index);
        return {
          order: index + 1,
          source: clip.name,
          duration: clip.duration,
          emotion: clip.emotion,
          analysisMethod: "mediapipe-landmarks-75_expression-25",
          landmarkDetectionRate: clip.landmarkDetectionRate,
          eyeAlignedToFirstClip: Boolean(
            clip.alignmentPose && clips[0]?.alignmentPose,
          ),
          boundaries: {
            onset: clip.onset,
            peak: clip.peak,
            skipOut: clip.skipOut,
            plateauEnd: clip.plateauEnd,
            skipIn: clip.skipIn,
            settle: clip.settle,
          },
          sourceStart: plan.sourceStart,
          skipOut: plan.cut,
          skipIn: plan.skipIn,
          skipMatchDistance: clip.skipMatchDistance,
          outputPhases: {
            intro: plan.intro,
            morphIn: plan.morphIn,
            rise: plan.forward,
            peakSkipMorph: plan.skip,
            decay: plan.decay,
            hold: plan.hold,
          },
          outputDuration: plan.duration,
        };
      }),
    };
    download(
      new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
      "incomplete-deformation.json",
    );
  }

  async function exportWebM() {
    if (busy || !clips.length || !("VideoEncoder" in window)) return;
    pause();
    setBusy(true);
    const total = sequenceDuration();
    const frameCount = Math.max(2, Math.round(total * EXPORT_FPS));
    const width = clips[0].width;
    const height = clips[0].height;
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d", { alpha: false });
    try {
      const {
        BufferTarget,
        CanvasSource,
        Output,
        WebMOutputFormat,
        canEncodeVideo,
      } = await import(MEDIABUNNY_URL);
      const options = {
        width,
        height,
        bitrate: 8_000_000,
        latencyMode: "quality",
      };
      const codec = (await canEncodeVideo("vp9", options))
        ? "vp9"
        : (await canEncodeVideo("vp8", options))
          ? "vp8"
          : "";
      if (!codec) throw new Error("VP8/VP9エンコーダーを利用できません");
      const target = new BufferTarget();
      const output = new Output({ format: new WebMOutputFormat(), target });
      const source = new CanvasSource(canvas, {
        codec,
        bitrate: options.bitrate,
        latencyMode: "quality",
        keyFrameInterval: 1,
      });
      output.addVideoTrack(source, { frameRate: EXPORT_FPS });
      await output.start();
      for (let frame = 0; frame < frameCount; frame += 1) {
        const time = frame / EXPORT_FPS;
        const mapped = mapSequenceTime(Math.min(time, total));
        if (mapped.phase.startsWith("MORPH")) {
          await renderMorphFrame(ctx, mapped, width, height);
        } else {
          await loadVideoSlot(el.sourceVideo, mapped.clip);
          await seekVideoSlot(el.sourceVideo, mapped.sourceTime);
          drawAlignedFrame(ctx, mapped.clip, width, height);
        }
        await source.add(time, 1 / EXPORT_FPS, { keyFrame: frame === 0 });
        setStatus(
          `WebMを書き出し中 ${frame + 1}/${frameCount}`,
          frame + 1,
          frameCount,
        );
      }
      source.close();
      await output.finalize();
      if (!target.buffer) throw new Error("映像データを生成できませんでした");
      download(
        new Blob([target.buffer], { type: "video/webm" }),
        "incomplete-deformation.webm",
      );
      setStatus(
        `WebMを書き出しました（${(frameCount / EXPORT_FPS).toFixed(2)}秒）`,
      );
    } catch (error) {
      console.error(error);
      setStatus(`書き出しに失敗しました: ${error.message}`, 0, 0, true);
    } finally {
      setBusy(false);
      await renderAt(playPosition, true).catch(() => {});
    }
  }

  function populateEmotions() {
    el.emotionSelect.replaceChildren(
      ...EMOTIONS.filter((emotion) => emotion !== BASE_EMOTION).map(
        (emotion) => {
          const option = document.createElement("option");
          option.value = emotion;
          option.textContent = LABELS[emotion];
          return option;
        },
      ),
    );
  }

  function bindEvents() {
    el.importJsonButton.addEventListener("click", () => el.jsonInput.click());
    el.jsonInput.addEventListener("change", async () => {
      const file = el.jsonInput.files?.[0];
      el.jsonInput.value = "";
      if (file) await importJson(file);
    });
    el.fileInput.addEventListener("change", () => {
      addFiles(el.fileInput.files);
      el.fileInput.value = "";
    });
    el.playButton.addEventListener("click", () => (playing ? pause() : play()));
    el.playhead.addEventListener("input", () => {
      pause();
      playPosition = number(el.playhead.value);
      renderAt(playPosition, true);
    });
    el.analyzeAllButton.addEventListener("click", () => runAnalysis(clips));
    el.analyzeClipButton.addEventListener("click", () => {
      const clip = selectedClip();
      if (clip) runAnalysis([clip]);
    });
    el.clipList.addEventListener("click", (event) => {
      if (event.target.closest('[data-action="add"]')) {
        el.fileInput.click();
        return;
      }
      const card = event.target.closest(".clipCard");
      if (!card) return;
      const id = Number(card.dataset.id);
      const action = event.target.closest("button")?.dataset.action;
      if (action === "up") moveClip(id, -1);
      else if (action === "down") moveClip(id, 1);
      else if (action === "remove") removeClip(id);
      else if (!event.target.closest(".clipEditorDisclosure")) {
        if (selectedId !== id) el.clipEditor.open = false;
        selectedId = id;
        renderAll();
      }
    });
    el.clipEditor.addEventListener("toggle", () => {
      if (!el.clipEditor.open) return;
      const clip = selectedClip();
      if (clip) requestAnimationFrame(() => drawChart(clip));
    });
    [el.skipOutInput, el.skipInInput].forEach((input) =>
      input.addEventListener("change", updateSelectedBoundaries),
    );
    el.emotionSelect.addEventListener("change", () => {
      const clip = selectedClip();
      if (!clip) return;
      clip.emotion = el.emotionSelect.value;
      if (clip.samples.length) detectBoundaries(clip);
      rebuildTransitions();
      renderAll();
    });
    el.skipOutRange.addEventListener("input", () => {
      const clip = selectedClip();
      if (!clip) return;
      updateSkipRange(
        Math.min(number(el.skipOutRange.value), clip.skipIn - 0.02),
        el.skipInRange.value,
        number(el.skipOutRange.value),
      );
    });
    el.skipInRange.addEventListener("input", () => {
      const clip = selectedClip();
      if (!clip) return;
      updateSkipRange(
        el.skipOutRange.value,
        Math.max(number(el.skipInRange.value), cutTime(clip) + 0.02),
        number(el.skipInRange.value),
      );
    });
    [el.skipOutRange, el.skipInRange].forEach((input) =>
      input.addEventListener("change", () => {
        renderAll();
        renderAt(playPosition, true);
      }),
    );
    PLAYBACK_SETTINGS.map(([id]) => el[id]).forEach((input) =>
      input.addEventListener("change", () => {
        input.value = setting(input.id);
        rebuildTransitions();
        pause();
        playPosition = clamp(playPosition, 0, sequenceDuration());
        renderAll();
        renderAt(playPosition, true);
      }),
    );
    el.exportJsonButton.addEventListener("click", exportJson);
    el.exportButton.addEventListener("click", exportWebM);
    new ResizeObserver(() => {
      const clip = selectedClip();
      if (clip) drawChart(clip);
    }).observe(el.analysisChart);
    window.addEventListener("beforeunload", () =>
      clips.forEach((clip) => URL.revokeObjectURL(clip.url)),
    );
  }

  async function initialize() {
    populateEmotions();
    bindEvents();
    clearStage();
    renderAll();
    try {
      await models;
      modelsReady = true;
      setStatus("MediaPipe / Face API準備完了。映像を追加してください");
    } catch (error) {
      console.error(error);
      setStatus(
        "MediaPipeまたはFace APIモデルの読み込みに失敗しました",
        0,
        0,
        true,
      );
    }
    updateControls();
  }

  console.assert(clamp(2, 0, 1) === 1 && number("0.2") === 0.2);
  initialize();
})();

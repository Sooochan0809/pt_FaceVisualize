(() => {
    "use strict";

    const MM_PER_INCH = 25.4;
    const REQUIRED_IMAGE_COUNT = 6;
    const FULL_CIRCLE_RADIANS = Math.PI * 2;
    const MEDIAPIPE_VERSION = "0.10.34";
    const MEDIAPIPE_TASKS_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}`;
    const MEDIAPIPE_WASM_URL = `${MEDIAPIPE_TASKS_URL}/wasm`;
    const FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";
    const HEAD_ALIGNMENT_LANDMARKS = [33, 133, 362, 263, 1, 4, 61, 291, 199, 152];
    const REGISTRATION_MARK = Object.freeze({
        insetMm: 1.6,
        minimumRadiusMm: 1.05,
        maximumRadiusMm: 1.4,
        lensPitchRadiusScale: 1.5,
        insetRadiusScale: 0.75,
        innerCircleScale: 0.78,
        columnOffsetScale: 0.36,
        rowOffsetScale: 0.42,
        dotRadiusScale: 0.16
    });
    const SETTINGS = Object.freeze({
        widthMm: 36,
        heightMm: 48,
        ppi: 600,
        lpi: 50,
        phase: 0,
        brightness: 240,
        lensDirection: "horizontal",
        reverseOrder: false,
        registrationMarks: true,
        headAlignment: true
    });
    let faceLandmarkerPromise = null;

    const mod = (value, divisor) => ((value % divisor) + divisor) % divisor;
    const mmToPixels = (mm, ppi) => Math.max(1, Math.round(mm / MM_PER_INCH * ppi));

    function loadImage(imageUrl) {
        return new Promise((resolve, reject) => {
            const image = new Image();
            image.decoding = "async";
            image.onload = () => resolve(image);
            image.onerror = reject;
            image.src = imageUrl;
        });
    }

    function getFaceLandmarker() {
        if (!faceLandmarkerPromise) {
            faceLandmarkerPromise = import(MEDIAPIPE_TASKS_URL)
                .then(async ({ FaceLandmarker, FilesetResolver }) => {
                    const vision = await FilesetResolver.forVisionTasks(
                        MEDIAPIPE_WASM_URL
                    );
                    return FaceLandmarker.createFromOptions(vision, {
                        baseOptions: { modelAssetPath: FACE_MODEL_URL },
                        runningMode: "IMAGE",
                        numFaces: 1
                    });
                })
                .catch(error => {
                    faceLandmarkerPromise = null;
                    throw error;
                });
        }
        return faceLandmarkerPromise;
    }

    async function detectFaceLandmarks(image) {
        try {
            const faceLandmarker = await getFaceLandmarker();
            const landmarks = faceLandmarker.detect(image).faceLandmarks?.[0];
            if (!landmarks) return null;
            return landmarks.map(point => ({
                x: point.x * image.naturalWidth,
                y: point.y * image.naturalHeight
            }));
        } catch (error) {
            console.warn("頭部位置を検出できませんでした", error);
            return null;
        }
    }

    function createAxisImageMap(length, imageCount) {
        const map = new Uint8Array(length);
        for (let axis = 0; axis < length; axis++) {
            const lensPosition = (axis + 0.5) * SETTINGS.lpi / SETTINGS.ppi;
            const phase = mod(lensPosition + SETTINGS.phase, 1);
            let imageIndex = Math.min(imageCount - 1, Math.floor(phase * imageCount));
            if (SETTINGS.reverseOrder) imageIndex = imageCount - 1 - imageIndex;
            map[axis] = imageIndex;
        }
        return map;
    }

    function forEachMappedRun(axisMap, imageIndex, axisStart, axisEnd, callback) {
        let runStart = -1;

        for (let axis = axisStart; axis <= axisEnd; axis++) {
            const matches = axis < axisEnd && axisMap[axis] === imageIndex;
            if (matches && runStart < 0) {
                runStart = axis;
            } else if (!matches && runStart >= 0) {
                callback(runStart, axis - runStart);
                runStart = -1;
            }
        }
    }

    function addMappedRunsToPath(ctx, axisMap, imageIndex, width, height) {
        const horizontalLens = SETTINGS.lensDirection === "horizontal";
        const axisEnd = horizontalLens ? height : width;

        forEachMappedRun(axisMap, imageIndex, 0, axisEnd, (start, runLength) => {
            if (horizontalLens) ctx.rect(0, start, width, runLength);
            else ctx.rect(start, 0, runLength, height);
        });
    }

    function getCoverTransform(image, width, height) {
        const scale = Math.max(width / image.naturalWidth, height / image.naturalHeight);
        return {
            a: scale,
            b: 0,
            c: 0,
            d: scale,
            e: (width - image.naturalWidth * scale) / 2,
            f: (height - image.naturalHeight * scale) / 2
        };
    }

    function transformPoint(point, transform) {
        return {
            x: transform.a * point.x + transform.c * point.y + transform.e,
            y: transform.b * point.x + transform.d * point.y + transform.f
        };
    }

    function averagePoint(points) {
        const sum = points.reduce((total, point) => ({
            x: total.x + point.x,
            y: total.y + point.y
        }), { x: 0, y: 0 });
        return {
            x: sum.x / points.length,
            y: sum.y / points.length
        };
    }

    function getHeadAlignmentTransform(sourceLandmarks, targetLandmarks) {
        const pairs = HEAD_ALIGNMENT_LANDMARKS
            .filter(index => sourceLandmarks[index] && targetLandmarks[index])
            .map(index => ({
                source: sourceLandmarks[index],
                target: targetLandmarks[index]
            }));
        if (pairs.length < 2) return null;

        const sourceCenter = averagePoint(pairs.map(pair => pair.source));
        const targetCenter = averagePoint(pairs.map(pair => pair.target));
        const fit = pairs.reduce((sum, pair) => {
            const sourceX = pair.source.x - sourceCenter.x;
            const sourceY = pair.source.y - sourceCenter.y;
            const targetX = pair.target.x - targetCenter.x;
            const targetY = pair.target.y - targetCenter.y;
            return {
                dot: sum.dot + sourceX * targetX + sourceY * targetY,
                cross: sum.cross + sourceX * targetY - sourceY * targetX,
                size: sum.size + sourceX * sourceX + sourceY * sourceY
            };
        }, { dot: 0, cross: 0, size: 0 });
        if (!fit.size) return null;

        const a = fit.dot / fit.size;
        const b = fit.cross / fit.size;
        return {
            a,
            b,
            c: -b,
            d: a,
            e: targetCenter.x - a * sourceCenter.x + b * sourceCenter.y,
            f: targetCenter.y - b * sourceCenter.x - a * sourceCenter.y
        };
    }

    function getImageTransforms(items, width, height) {
        const fittedTransforms = items.map(({ image }) =>
            getCoverTransform(image, width, height)
        );
        if (!SETTINGS.headAlignment) return fittedTransforms;

        const referenceIndex = items.findIndex(item => item.landmarks);
        if (referenceIndex < 0) return fittedTransforms;
        const referenceLandmarks = items[referenceIndex].landmarks.map(point =>
            transformPoint(point, fittedTransforms[referenceIndex])
        );

        return items.map((item, index) => {
            if (index === referenceIndex || !item.landmarks) {
                return fittedTransforms[index];
            }
            return getHeadAlignmentTransform(
                item.landmarks,
                referenceLandmarks
            ) ?? fittedTransforms[index];
        });
    }

    function drawTransformedImage(ctx, image, transform) {
        ctx.save();
        ctx.transform(
            transform.a,
            transform.b,
            transform.c,
            transform.d,
            transform.e,
            transform.f
        );
        ctx.drawImage(image, 0, 0);
        ctx.restore();
    }

    function drawSixDotMark(ctx, centerX, centerY, radius) {
        const columnOffset = radius * REGISTRATION_MARK.columnOffsetScale;
        const rowOffset = radius * REGISTRATION_MARK.rowOffsetScale;
        const dotRadius = radius * REGISTRATION_MARK.dotRadiusScale;
        const offsets = [
            [-columnOffset, -rowOffset],
            [columnOffset, -rowOffset],
            [-columnOffset, 0],
            [columnOffset, 0],
            [-columnOffset, rowOffset],
            [columnOffset, rowOffset]
        ];

        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, FULL_CIRCLE_RADIANS);
        ctx.fillStyle = "black";
        ctx.fill();
        ctx.beginPath();
        ctx.arc(
            centerX,
            centerY,
            radius * REGISTRATION_MARK.innerCircleScale,
            0,
            FULL_CIRCLE_RADIANS
        );
        ctx.fillStyle = "white";
        ctx.fill();
        ctx.beginPath();
        offsets.forEach(([offsetX, offsetY]) => {
            ctx.moveTo(centerX + offsetX + dotRadius, centerY + offsetY);
            ctx.arc(
                centerX + offsetX,
                centerY + offsetY,
                dotRadius,
                0,
                FULL_CIRCLE_RADIANS
            );
        });
        ctx.fillStyle = "black";
        ctx.fill();
    }

    function drawRegistrationMarks(ctx, width, height, axisMap, imageCount) {
        const pixelsPerMm = SETTINGS.ppi / MM_PER_INCH;
        const inset = Math.min(
            REGISTRATION_MARK.insetMm * pixelsPerMm,
            width / 4,
            height / 4
        );
        const lensPitch = SETTINGS.ppi / SETTINGS.lpi;
        const radius = Math.min(
            Math.max(
                REGISTRATION_MARK.minimumRadiusMm * pixelsPerMm,
                lensPitch * REGISTRATION_MARK.lensPitchRadiusScale
            ),
            REGISTRATION_MARK.maximumRadiusMm * pixelsPerMm,
            inset * REGISTRATION_MARK.insetRadiusScale,
            width / 8,
            height / 8
        );
        const referenceImageIndex = Math.floor((imageCount - 1) / 2);
        const positions = [
            [inset, inset],
            [width - inset, inset],
            [inset, height - inset],
            [width - inset, height - inset]
        ];

        positions.forEach(([x, y]) => {
            const left = Math.max(0, Math.floor(x - radius));
            const right = Math.min(width, Math.ceil(x + radius));
            const top = Math.max(0, Math.floor(y - radius));
            const bottom = Math.min(height, Math.ceil(y + radius));
            const horizontalLens = SETTINGS.lensDirection === "horizontal";
            const axisStart = horizontalLens ? top : left;
            const axisEnd = horizontalLens ? bottom : right;

            ctx.save();
            ctx.beginPath();
            forEachMappedRun(
                axisMap,
                referenceImageIndex,
                axisStart,
                axisEnd,
                (start, runLength) => {
                    if (horizontalLens) ctx.rect(left, start, right - left, runLength);
                    else ctx.rect(start, top, runLength, bottom - top);
                }
            );
            ctx.clip();
            drawSixDotMark(ctx, x, y, radius);
            ctx.restore();
        });
    }

    async function create(imageUrls) {
        if (!Array.isArray(imageUrls) || imageUrls.length !== REQUIRED_IMAGE_COUNT) {
            throw new Error(`レンチキュラー画像には${REQUIRED_IMAGE_COUNT}枚の写真が必要です。`);
        }

        const images = await Promise.all(imageUrls.map(loadImage));
        const landmarks = await Promise.all(images.map(detectFaceLandmarks));
        const items = images.map((image, index) => ({
            image,
            landmarks: landmarks[index]
        }));
        const width = mmToPixels(SETTINGS.widthMm, SETTINGS.ppi);
        const height = mmToPixels(SETTINGS.heightMm, SETTINGS.ppi);
        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, width, height);

        const horizontalLens = SETTINGS.lensDirection === "horizontal";
        const axisMap = createAxisImageMap(horizontalLens ? height : width, images.length);
        const imageTransforms = getImageTransforms(items, width, height);
        images.forEach((image, imageIndex) => {
            ctx.save();
            ctx.beginPath();
            addMappedRunsToPath(ctx, axisMap, imageIndex, width, height);
            ctx.clip();
            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = "high";
            ctx.filter = `brightness(${SETTINGS.brightness / 100})`;
            drawTransformedImage(ctx, image, imageTransforms[imageIndex]);
            ctx.restore();
        });

        if (SETTINGS.registrationMarks) {
            drawRegistrationMarks(ctx, width, height, axisMap, images.length);
        }
        return {
            imageUrl: canvas.toDataURL("image/png"),
            width,
            height,
            settings: SETTINGS,
            alignment: {
                enabled: SETTINGS.headAlignment,
                detectedImages: landmarks.filter(Boolean).length
            }
        };
    }

    window.MotionLenticular = Object.freeze({
        create,
        prepare: getFaceLandmarker,
        settings: SETTINGS
    });
})();

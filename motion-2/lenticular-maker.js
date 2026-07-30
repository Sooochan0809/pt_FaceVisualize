(() => {
    "use strict";

    const MM_PER_INCH = 25.4;
    const REQUIRED_IMAGE_COUNT = 6;
    const FULL_CIRCLE_RADIANS = Math.PI * 2;
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
        brightness: 160,
        lensDirection: "horizontal",
        reverseOrder: false,
        registrationMarks: true
    });

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

    function drawCover(ctx, image, width, height) {
        const scale = Math.max(width / image.naturalWidth, height / image.naturalHeight);
        const drawWidth = image.naturalWidth * scale;
        const drawHeight = image.naturalHeight * scale;
        ctx.drawImage(
            image,
            (width - drawWidth) / 2,
            (height - drawHeight) / 2,
            drawWidth,
            drawHeight
        );
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
        images.forEach((image, imageIndex) => {
            ctx.save();
            ctx.beginPath();
            addMappedRunsToPath(ctx, axisMap, imageIndex, width, height);
            ctx.clip();
            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = "high";
            ctx.filter = `brightness(${SETTINGS.brightness / 100})`;
            drawCover(ctx, image, width, height);
            ctx.restore();
        });

        if (SETTINGS.registrationMarks) {
            drawRegistrationMarks(ctx, width, height, axisMap, images.length);
        }
        return {
            imageUrl: canvas.toDataURL("image/png"),
            width,
            height,
            settings: SETTINGS
        };
    }

    window.MotionLenticular = Object.freeze({ create, settings: SETTINGS });
})();

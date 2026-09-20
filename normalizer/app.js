(() => {
    "use strict";

    const MODEL_URL = "../modules/faceAPI/models";
    const BASE_EMOTION = "neutral";
    const EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"];
    const EMOTION_LABELS = {
        neutral: "無表情",
        happy: "喜び",
        sad: "悲しみ",
        angry: "怒り",
        fearful: "恐れ",
        disgusted: "嫌悪",
        surprised: "驚き"
    };
    const BOUNDARY_KEYS = ["start", "onset", "peak", "settle"];
    const CHART_PADDING = { left: 46, right: 18, top: 16, bottom: 30 };
    const $ = (id) => document.getElementById(id);

    const elements = {
        video: $("video"),
        emptyVideo: $("emptyVideo"),
        fileInput: $("fileInput"),
        analyzeButton: $("analyzeButton"),
        status: $("status"),
        analysisProgress: $("analysisProgress"),
        chart: $("chart"),
        targetEmotion: $("targetEmotion"),
        sampleInterval: $("sampleInterval"),
        smoothWindow: $("smoothWindow"),
        targetLegend: $("targetLegend"),
        boundaryLog: $("boundaryLog"),
        neutralDuration: $("neutralDuration"),
        riseDuration: $("riseDuration"),
        settleDuration: $("settleDuration"),
        neutralRaw: $("neutralRaw"),
        riseRaw: $("riseRaw"),
        settleRaw: $("settleRaw"),
        settleSegment: $("settleSegment"),
        previewButton: $("previewButton"),
        previewRange: $("previewRange"),
        previewTime: $("previewTime"),
        rawDurationMetric: $("rawDurationMetric"),
        targetDurationMetric: $("targetDurationMetric"),
        detectionMetric: $("detectionMetric"),
        exportVideoButton: $("exportVideoButton"),
        exportCsvButton: $("exportCsvButton"),
        exportJsonButton: $("exportJsonButton")
    };

    const detectorOptions = new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.35 });
    const analysisCanvas = document.createElement("canvas");
    let objectUrl = "";
    let sourceFileName = "";
    let videoReady = false;
    let modelsReady = false;
    let busy = false;
    let samples = [];
    let analysisToken = 0;
    let anchors = { start: 0, onset: 0, peak: 0, settle: 0 };
    let includeSettle = true;
    let previewPlaying = false;
    let previewPosition = 0;
    let previewStartedAt = 0;
    let previewFrameId = null;

    const models = Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
        faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL)
    ]);

    function clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
    }

    function finiteNumber(value, fallback = 0) {
        const number = Number(value);
        return Number.isFinite(number) ? number : fallback;
    }

    function formatSeconds(value) {
        return `${Math.max(0, finiteNumber(value)).toFixed(2)}秒`;
    }

    function formatClock(value) {
        const seconds = Math.max(0, finiteNumber(value));
        const minutes = Math.floor(seconds / 60);
        return `${minutes}:${(seconds - minutes * 60).toFixed(2).padStart(5, "0")}`;
    }

    function setStatus(message, current = 0, total = 0, isError = false) {
        elements.status.textContent = message;
        elements.status.classList.toggle("error", isError);
        elements.analysisProgress.hidden = total <= 0;
        elements.analysisProgress.max = Math.max(1, total);
        elements.analysisProgress.value = current;
    }

    function populateEmotionSelects() {
        const options = EMOTIONS.filter((emotion) => emotion !== BASE_EMOTION).map((emotion) => {
            const option = document.createElement("option");
            option.value = emotion;
            option.textContent = EMOTION_LABELS[emotion];
            return option;
        });
        elements.targetEmotion.replaceChildren(...options);
        elements.targetEmotion.value = "happy";
        updateLegend();
    }

    function updateLegend() {
        elements.targetLegend.textContent = EMOTION_LABELS[elements.targetEmotion.value];
    }

    function updateButtonStates() {
        const hasAnalysis = samples.length > 0;
        const canNormalize = hasAnalysis && videoReady && anchors.peak > anchors.onset && anchors.onset >= anchors.start;
        elements.analyzeButton.disabled = busy || !videoReady || !modelsReady;
        elements.previewButton.disabled = busy || !canNormalize;
        elements.previewRange.disabled = busy || !canNormalize;
        elements.exportVideoButton.disabled = busy || !canNormalize || !window.MediaRecorder;
        elements.exportCsvButton.disabled = busy || !hasAnalysis;
        elements.exportJsonButton.disabled = busy || !hasAnalysis;
        elements.fileInput.disabled = busy;
        elements.targetEmotion.disabled = busy;
        elements.sampleInterval.disabled = busy;
        elements.smoothWindow.disabled = busy;
        elements.neutralDuration.disabled = busy || !hasAnalysis;
        elements.riseDuration.disabled = busy || !hasAnalysis;
        elements.settleDuration.disabled = busy || !hasAnalysis || !includeSettle;
    }

    function setBusy(nextBusy) {
        busy = nextBusy;
        updateButtonStates();
    }

    function waitForVideoMetadata() {
        if (elements.video.readyState >= 1) return Promise.resolve();
        return new Promise((resolve, reject) => {
            const loaded = () => {
                elements.video.removeEventListener("error", failed);
                resolve();
            };
            const failed = () => {
                elements.video.removeEventListener("loadedmetadata", loaded);
                reject(new Error("動画を読み込めませんでした"));
            };
            elements.video.addEventListener("loadedmetadata", loaded, { once: true });
            elements.video.addEventListener("error", failed, { once: true });
        });
    }

    function seekVideo(time) {
        return new Promise((resolve) => {
            const duration = Number.isFinite(elements.video.duration) ? elements.video.duration : 0;
            const target = clamp(time, 0, Math.max(0, duration - 0.01));
            if (Math.abs(elements.video.currentTime - target) < 0.002 && elements.video.readyState >= 2) {
                requestAnimationFrame(resolve);
                return;
            }
            const onSeeked = () => requestAnimationFrame(resolve);
            elements.video.addEventListener("seeked", onSeeked, { once: true });
            elements.video.currentTime = target;
        });
    }

    function drawAnalysisFrame() {
        const sourceWidth = Math.max(1, elements.video.videoWidth);
        const sourceHeight = Math.max(1, elements.video.videoHeight);
        const scale = Math.min(1, 320 / Math.max(sourceWidth, sourceHeight));
        const width = Math.max(1, Math.round(sourceWidth * scale));
        const height = Math.max(1, Math.round(sourceHeight * scale));
        if (analysisCanvas.width !== width || analysisCanvas.height !== height) {
            analysisCanvas.width = width;
            analysisCanvas.height = height;
        }
        const ctx = analysisCanvas.getContext("2d", { alpha: false });
        ctx.drawImage(elements.video, 0, 0, width, height);
        return analysisCanvas;
    }

    function setDefaultAnchors() {
        const duration = Number.isFinite(elements.video.duration) ? elements.video.duration : 0;
        includeSettle = true;
        anchors = {
            start: 0,
            onset: duration * 0.25,
            peak: duration * 0.6,
            settle: duration * 0.9
        };
        updateNormalizationSummary();
        updateBoundaryLog();
    }

    function updateBoundaryLog() {
        if (!samples.length) {
            elements.boundaryLog.textContent = "区間境界: 未解析";
            return;
        }
        const settleText = includeSettle ? `${anchors.settle.toFixed(2)}s` : "未検出";
        elements.boundaryLog.textContent = [
            `区間境界  start ${anchors.start.toFixed(2)}s`,
            `発生 ${anchors.onset.toFixed(2)}s`,
            `ピーク ${anchors.peak.toFixed(2)}s`,
            `収束 ${settleText}`
        ].join("  /  ");
    }

    function rawDurations() {
        return {
            neutral: Math.max(0, anchors.onset - anchors.start),
            rise: Math.max(0, anchors.peak - anchors.onset),
            settle: Math.max(0, anchors.settle - anchors.peak)
        };
    }

    function targetDurations() {
        return {
            neutral: Math.max(0.01, finiteNumber(elements.neutralDuration.value, 0.5)),
            rise: Math.max(0.01, finiteNumber(elements.riseDuration.value, 0.5)),
            settle: Math.max(0.01, finiteNumber(elements.settleDuration.value, 0.5))
        };
    }

    function getSegments() {
        const target = targetDurations();
        const segments = [
            { key: "neutral", sourceStart: anchors.start, sourceEnd: anchors.onset, targetDuration: target.neutral },
            { key: "rise", sourceStart: anchors.onset, sourceEnd: anchors.peak, targetDuration: target.rise }
        ];
        if (includeSettle && anchors.settle > anchors.peak) {
            segments.push({ key: "settle", sourceStart: anchors.peak, sourceEnd: anchors.settle, targetDuration: target.settle });
        }
        return segments.filter((segment) => segment.sourceEnd > segment.sourceStart && segment.targetDuration > 0);
    }

    function normalizedDuration() {
        return getSegments().reduce((sum, segment) => sum + segment.targetDuration, 0);
    }

    function mapNormalizedTime(normalizedTime) {
        const segments = getSegments();
        let elapsed = 0;
        for (const segment of segments) {
            const end = elapsed + segment.targetDuration;
            if (normalizedTime <= end || segment === segments.at(-1)) {
                const progress = clamp((normalizedTime - elapsed) / segment.targetDuration, 0, 1);
                return segment.sourceStart + (segment.sourceEnd - segment.sourceStart) * progress;
            }
            elapsed = end;
        }
        return anchors.start;
    }

    function updateRawDuration(node, rawDuration) {
        node.textContent = `raw ${rawDuration.toFixed(2)}秒`;
    }

    function updateNormalizationSummary() {
        const raw = rawDurations();
        updateRawDuration(elements.neutralRaw, raw.neutral);
        updateRawDuration(elements.riseRaw, raw.rise);
        updateRawDuration(elements.settleRaw, raw.settle);
        elements.settleSegment.hidden = !includeSettle;
        elements.settleDuration.disabled = !includeSettle || busy || !samples.length;

        const sourceEnd = includeSettle ? anchors.settle : anchors.peak;
        elements.rawDurationMetric.textContent = formatSeconds(Math.max(0, sourceEnd - anchors.start));
        const duration = normalizedDuration();
        elements.targetDurationMetric.textContent = formatSeconds(duration);
        elements.previewRange.max = String(Math.max(0.001, duration));
        previewPosition = clamp(previewPosition, 0, duration);
        updatePreviewUI();
        updateButtonStates();
    }

    function getSmoothedSamples() {
        const windowSize = Math.max(1, Math.round(finiteNumber(elements.smoothWindow.value, 1)));
        const radius = Math.floor(windowSize / 2);
        return samples.map((sample, index) => {
            if (!sample.detected) return { ...sample, smooth: null };
            const neighbors = samples.slice(Math.max(0, index - radius), index + radius + 1).filter((item) => item.detected);
            const smooth = Object.fromEntries(EMOTIONS.map((emotion) => [emotion,
                neighbors.reduce((sum, item) => sum + item.expressions[emotion], 0) / Math.max(1, neighbors.length)
            ]));
            return { ...sample, smooth };
        });
    }

    function drawSeries(ctx, data, emotion, color, xFor, yFor) {
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        let drawing = false;
        data.forEach((sample) => {
            const source = sample.smooth;
            if (!sample.detected || !source) {
                drawing = false;
                return;
            }
            const x = xFor(sample.time);
            const y = yFor(source[emotion] || 0);
            if (drawing) ctx.lineTo(x, y);
            else ctx.moveTo(x, y);
            drawing = true;
        });
        ctx.stroke();
        ctx.restore();
    }

    function drawChart() {
        const canvas = elements.chart;
        const rect = canvas.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) return;
        const dpr = window.devicePixelRatio || 1;
        canvas.width = Math.round(rect.width * dpr);
        canvas.height = Math.round(rect.height * dpr);
        const ctx = canvas.getContext("2d");
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, rect.width, rect.height);

        const duration = Number.isFinite(elements.video.duration) ? elements.video.duration : 0;
        const plotWidth = Math.max(1, rect.width - CHART_PADDING.left - CHART_PADDING.right);
        const plotHeight = Math.max(1, rect.height - CHART_PADDING.top - CHART_PADDING.bottom);
        const xFor = (time) => CHART_PADDING.left + clamp(time / Math.max(duration, 0.001), 0, 1) * plotWidth;
        const yFor = (value) => CHART_PADDING.top + (1 - clamp(value, 0, 1)) * plotHeight;

        ctx.font = "11px sans-serif";
        ctx.fillStyle = "#6d737b";
        ctx.strokeStyle = "#e4e7eb";
        ctx.lineWidth = 1;
        for (let value = 0; value <= 1.001; value += 0.25) {
            const y = yFor(value);
            ctx.beginPath();
            ctx.moveTo(CHART_PADDING.left, y);
            ctx.lineTo(rect.width - CHART_PADDING.right, y);
            ctx.stroke();
            ctx.fillText(value.toFixed(2), 5, y + 4);
        }
        for (let index = 0; index <= 5; index += 1) {
            const time = duration * index / 5;
            const x = xFor(time);
            ctx.beginPath();
            ctx.moveTo(x, CHART_PADDING.top);
            ctx.lineTo(x, CHART_PADDING.top + plotHeight);
            ctx.stroke();
            ctx.fillText(`${time.toFixed(1)}s`, Math.min(x + 3, rect.width - 42), rect.height - 8);
        }

        if (samples.length) {
            const smoothed = getSmoothedSamples();
            const base = BASE_EMOTION;
            const target = elements.targetEmotion.value;
            drawSeries(ctx, smoothed, base, "#718096", xFor, yFor);
            drawSeries(ctx, smoothed, target, "#ef7f45", xFor, yFor);
        }

        const markerColors = { start: "#53606e", onset: "#42a77b", peak: "#ef7f45", settle: "#8b6fc0" };
        if (samples.length) BOUNDARY_KEYS.forEach((key, index) => {
            if (key === "settle" && !includeSettle) return;
            const x = xFor(anchors[key]);
            ctx.strokeStyle = markerColors[key];
            ctx.lineWidth = 1.5;
            ctx.setLineDash([5, 4]);
            ctx.beginPath();
            ctx.moveTo(x, CHART_PADDING.top);
            ctx.lineTo(x, CHART_PADDING.top + plotHeight);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = markerColors[key];
            ctx.fillText(String(index + 1), x + 3, CHART_PADDING.top + 12);
        });

        if (videoReady) {
            const x = xFor(elements.video.currentTime);
            ctx.strokeStyle = "#1b2026";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(x, CHART_PADDING.top);
            ctx.lineTo(x, CHART_PADDING.top + plotHeight);
            ctx.stroke();
        }
    }

    function autoDetectBoundaries() {
        if (!samples.length) return;
        const data = getSmoothedSamples().filter((sample) => sample.detected && sample.smooth);
        if (data.length < 3) {
            setStatus("顔を検出できた点が少ないため、自動検出できません", 0, 0, true);
            return;
        }
        const base = BASE_EMOTION;
        const target = elements.targetEmotion.value;
        const progression = data.map((sample) => ({
            sample,
            value: sample.smooth[target] / Math.max(0.0001, sample.smooth[base] + sample.smooth[target])
        }));
        let peakIndex = 0;
        progression.forEach((item, index) => {
            if (item.value > progression[peakIndex].value) peakIndex = index;
        });
        const beforePeak = progression.slice(0, peakIndex + 1);
        const baselineCount = Math.max(1, Math.min(beforePeak.length, Math.round(beforePeak.length * .2)));
        const baseline = beforePeak.slice(0, baselineCount).reduce((sum, item) => sum + item.value, 0) / baselineCount;
        const amplitude = Math.max(0.01, progression[peakIndex].value - baseline);
        const onsetLevel = baseline + amplitude * .05;
        const peakLevel = baseline + amplitude * .95;
        const settleLevel = baseline + amplitude * .2;

        const crossingTime = (items, level, direction, startIndex = 1, endIndex = items.length - 1) => {
            for (let index = startIndex; index <= endIndex; index += 1) {
                const previous = items[index - 1];
                const current = items[index];
                const crossed = direction === "rising"
                    ? previous.value <= level && current.value >= level
                    : previous.value >= level && current.value <= level;
                if (!crossed) continue;
                const valueDistance = current.value - previous.value;
                const ratio = Math.abs(valueDistance) < 1e-6 ? 0 : clamp((level - previous.value) / valueDistance, 0, 1);
                return previous.sample.time + (current.sample.time - previous.sample.time) * ratio;
            }
            return null;
        };

        const onsetTime = crossingTime(progression, onsetLevel, "rising", 1, peakIndex)
            ?? progression[Math.max(0, peakIndex - 1)].sample.time;
        const peakTime = crossingTime(progression, peakLevel, "rising", 1, peakIndex)
            ?? progression[peakIndex].sample.time;
        const settleTime = crossingTime(progression, settleLevel, "falling", peakIndex + 1);
        const neutralLead = Math.max(finiteNumber(elements.sampleInterval.value, .1), peakTime - onsetTime);
        const roundTime = (time) => Math.round(time * 100) / 100;
        anchors.start = roundTime(Math.max(0, onsetTime - neutralLead));
        anchors.onset = roundTime(Math.max(anchors.start, onsetTime));
        anchors.peak = roundTime(Math.max(anchors.onset, peakTime));
        if (settleTime !== null) {
            anchors.settle = roundTime(Math.max(anchors.peak, settleTime));
            includeSettle = true;
        } else {
            anchors.settle = Math.max(anchors.peak, elements.video.duration);
            includeSettle = false;
        }
        updateNormalizationSummary();
        updateBoundaryLog();
        drawChart();
        setStatus(settleTime !== null ? "動き始め・ピーク・収束を自動検出しました" : "動き始め・ピークを検出しました（収束なし）");
    }

    async function analyzeVideo() {
        if (!videoReady || busy) return;
        const token = ++analysisToken;
        const previousTime = elements.video.currentTime;
        pausePreview();
        elements.video.pause();
        samples = [];
        setBusy(true);

        try {
            await models;
            const interval = clamp(finiteNumber(elements.sampleInterval.value, .1), .05, 2);
            elements.sampleInterval.value = String(interval);
            const total = Math.max(1, Math.ceil(elements.video.duration / interval));
            for (let index = 0; index < total; index += 1) {
                if (token !== analysisToken) return;
                const time = Math.min(index * interval, Math.max(0, elements.video.duration - .01));
                setStatus(`感情を解析中 ${index + 1} / ${total}`, index + 1, total);
                await seekVideo(time);
                const detection = await faceapi.detectSingleFace(drawAnalysisFrame(), detectorOptions).withFaceExpressions();
                const expressions = Object.fromEntries(EMOTIONS.map((emotion) => [emotion, detection?.expressions[emotion] || 0]));
                samples.push({ time, detected: Boolean(detection), expressions });
                if (index % 5 === 0) drawChart();
            }
            if (token !== analysisToken) return;
            const detectedCount = samples.filter((sample) => sample.detected).length;
            elements.detectionMetric.textContent = `${Math.round(detectedCount / samples.length * 100)}%`;
            autoDetectBoundaries();
            setStatus(`${samples.length}点を解析しました（顔検出 ${detectedCount}点）`);
        } catch (error) {
            console.error(error);
            setStatus("解析に失敗しました。動画形式とFace APIモデルを確認してください", 0, 0, true);
        } finally {
            if (token === analysisToken) {
                await seekVideo(previousTime).catch(() => {});
                setBusy(false);
                updateNormalizationSummary();
                drawChart();
            }
        }
    }

    function updatePreviewUI() {
        const duration = normalizedDuration();
        elements.previewRange.value = String(clamp(previewPosition, 0, Math.max(duration, .001)));
        elements.previewTime.textContent = `${previewPosition.toFixed(2)} / ${duration.toFixed(2)}秒`;
        elements.previewButton.textContent = previewPlaying ? "❚❚" : "▶";
    }

    function previewTick(now) {
        if (!previewPlaying) return;
        const duration = normalizedDuration();
        previewPosition = (now - previewStartedAt) / 1000;
        if (previewPosition >= duration) {
            previewPosition = duration;
            pausePreview();
            elements.video.currentTime = mapNormalizedTime(duration);
            drawChart();
            return;
        }
        const sourceTime = mapNormalizedTime(previewPosition);
        if (!elements.video.seeking && Math.abs(elements.video.currentTime - sourceTime) > .015) {
            elements.video.currentTime = sourceTime;
        }
        updatePreviewUI();
        drawChart();
        previewFrameId = requestAnimationFrame(previewTick);
    }

    function playPreview() {
        const duration = normalizedDuration();
        if (duration <= 0) return;
        elements.video.pause();
        if (previewPosition >= duration - .01) previewPosition = 0;
        previewPlaying = true;
        previewStartedAt = performance.now() - previewPosition * 1000;
        updatePreviewUI();
        previewFrameId = requestAnimationFrame(previewTick);
    }

    function pausePreview() {
        previewPlaying = false;
        if (previewFrameId !== null) cancelAnimationFrame(previewFrameId);
        previewFrameId = null;
        updatePreviewUI();
    }

    function downloadBlob(blob, fileName) {
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = fileName;
        anchor.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    }

    function safeBaseName() {
        return (sourceFileName.replace(/\.[^.]+$/, "") || "expression").replace(/[\\/:*?"<>|]+/g, "-");
    }

    function exportCsv() {
        const header = ["time", "detected", ...EMOTIONS];
        const rows = samples.map((sample) => [
            sample.time.toFixed(4),
            sample.detected ? 1 : 0,
            ...EMOTIONS.map((emotion) => sample.expressions[emotion].toFixed(6))
        ]);
        const csv = [header, ...rows].map((row) => row.join(",")).join("\n");
        downloadBlob(new Blob([`\uFEFF${csv}`], { type: "text/csv;charset=utf-8" }), `${safeBaseName()}-analysis.csv`);
    }

    function exportJson() {
        const data = {
            version: 1,
            source: sourceFileName,
            sourceDuration: elements.video.duration,
            analysis: {
                sampleInterval: finiteNumber(elements.sampleInterval.value, .1),
                smoothingWindow: finiteNumber(elements.smoothWindow.value, 5),
                baseEmotion: BASE_EMOTION,
                targetEmotion: elements.targetEmotion.value,
                detectionRate: samples.length ? samples.filter((sample) => sample.detected).length / samples.length : 0
            },
            boundaries: { ...anchors },
            includeSettle,
            targetDurations: targetDurations(),
            segments: getSegments().map((segment) => ({
                ...segment,
                sourceDuration: segment.sourceEnd - segment.sourceStart,
                playbackRate: (segment.sourceEnd - segment.sourceStart) / segment.targetDuration
            }))
        };
        downloadBlob(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }), `${safeBaseName()}-normalization.json`);
    }

    function supportedWebmType() {
        if (!window.MediaRecorder) return "";
        return ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"]
            .find((type) => MediaRecorder.isTypeSupported(type)) || "";
    }

    function delay(milliseconds) {
        return new Promise((resolve) => setTimeout(resolve, Math.max(0, milliseconds)));
    }

    async function exportNormalizedVideo() {
        const mimeType = supportedWebmType();
        if (!mimeType || busy) {
            setStatus("このブラウザはWebM書き出しに対応していません", 0, 0, true);
            return;
        }
        pausePreview();
        elements.video.pause();
        const restoreTime = elements.video.currentTime;
        const duration = normalizedDuration();
        const fps = 30;
        const frameCount = Math.max(1, Math.ceil(duration * fps));
        const canvas = document.createElement("canvas");
        canvas.width = elements.video.videoWidth;
        canvas.height = elements.video.videoHeight;
        const ctx = canvas.getContext("2d", { alpha: false });
        const stream = canvas.captureStream(fps);
        const recorder = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 8_000_000 });
        const chunks = [];
        recorder.addEventListener("dataavailable", (event) => {
            if (event.data.size) chunks.push(event.data);
        });
        const stopped = new Promise((resolve) => recorder.addEventListener("stop", resolve, { once: true }));
        setBusy(true);

        try {
            recorder.start(250);
            const startedAt = performance.now();
            for (let frame = 0; frame < frameCount; frame += 1) {
                const outputTime = Math.min(frame / fps, duration);
                setStatus(`正規化WebMを書き出し中 ${frame + 1} / ${frameCount}`, frame + 1, frameCount);
                await seekVideo(mapNormalizedTime(outputTime));
                ctx.drawImage(elements.video, 0, 0, canvas.width, canvas.height);
                const nextFrameAt = startedAt + (frame + 1) * 1000 / fps;
                await delay(nextFrameAt - performance.now());
            }
            recorder.stop();
            await stopped;
            const blob = new Blob(chunks, { type: mimeType });
            downloadBlob(blob, `${safeBaseName()}-normalized.webm`);
            setStatus(`正規化WebMを保存しました（${duration.toFixed(2)}秒）`);
        } catch (error) {
            console.error(error);
            if (recorder.state !== "inactive") recorder.stop();
            setStatus("WebMの書き出しに失敗しました", 0, 0, true);
        } finally {
            stream.getTracks().forEach((track) => track.stop());
            await seekVideo(restoreTime).catch(() => {});
            setBusy(false);
            updateNormalizationSummary();
        }
    }

    async function loadVideoFile(file) {
        analysisToken += 1;
        pausePreview();
        elements.video.pause();
        if (objectUrl) URL.revokeObjectURL(objectUrl);
        objectUrl = URL.createObjectURL(file);
        sourceFileName = file.name;
        videoReady = false;
        samples = [];
        elements.detectionMetric.textContent = "—";
        elements.video.hidden = false;
        elements.emptyVideo.hidden = true;
        elements.video.src = objectUrl;
        elements.video.load();
        setStatus("動画を読み込んでいます");
        updateButtonStates();

        try {
            await waitForVideoMetadata();
            videoReady = true;
            elements.video.currentTime = 0;
            previewPosition = 0;
            setDefaultAnchors();
            setStatus(`${file.name} / ${formatClock(elements.video.duration)}`);
            updateButtonStates();
            drawChart();
        } catch (error) {
            console.error(error);
            elements.video.hidden = true;
            elements.emptyVideo.hidden = false;
            elements.emptyVideo.textContent = "動画を読み込めませんでした";
            setStatus("動画の読み込みに失敗しました", 0, 0, true);
        }
    }

    function bindEvents() {
        elements.fileInput.addEventListener("change", () => {
            const file = elements.fileInput.files[0];
            if (file) loadVideoFile(file);
            elements.fileInput.value = "";
        });
        elements.analyzeButton.addEventListener("click", analyzeVideo);

        elements.targetEmotion.addEventListener("change", () => {
            updateLegend();
            updateButtonStates();
            if (samples.length) autoDetectBoundaries();
            else drawChart();
        });
        elements.smoothWindow.addEventListener("change", () => {
            if (samples.length) autoDetectBoundaries();
            else drawChart();
        });

        [elements.neutralDuration, elements.riseDuration, elements.settleDuration].forEach((input) => {
            input.addEventListener("input", updateNormalizationSummary);
            input.addEventListener("change", () => {
                input.value = Math.max(.01, finiteNumber(input.value, .5)).toFixed(2);
                updateNormalizationSummary();
            });
        });
        elements.previewButton.addEventListener("click", () => previewPlaying ? pausePreview() : playPreview());
        elements.previewRange.addEventListener("input", () => {
            pausePreview();
            previewPosition = finiteNumber(elements.previewRange.value, 0);
            elements.video.currentTime = mapNormalizedTime(previewPosition);
            updatePreviewUI();
            drawChart();
        });

        elements.video.addEventListener("timeupdate", () => {
            if (!previewPlaying && !busy) drawChart();
        });
        elements.video.addEventListener("play", pausePreview);
        elements.exportCsvButton.addEventListener("click", exportCsv);
        elements.exportJsonButton.addEventListener("click", exportJson);
        elements.exportVideoButton.addEventListener("click", exportNormalizedVideo);
        window.addEventListener("beforeunload", () => {
            analysisToken += 1;
            if (objectUrl) URL.revokeObjectURL(objectUrl);
        });
        new ResizeObserver(drawChart).observe(elements.chart);
    }

    async function initialize() {
        populateEmotionSelects();
        bindEvents();
        updateBoundaryLog();
        updateNormalizationSummary();
        drawChart();
        try {
            await models;
            modelsReady = true;
            setStatus("動画を選択してください");
        } catch (error) {
            console.error(error);
            setStatus("Face APIモデルの読み込みに失敗しました", 0, 0, true);
        }
        updateButtonStates();
    }

    initialize();
})();

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
  const EXPORT_FPS = 30;
  const MEDIABUNNY_URL = "https://cdn.jsdelivr.net/npm/mediabunny@1.49.0/+esm";
  const $ = (id) => document.getElementById(id);
  const el = Object.fromEntries(
    [
      "sourceVideo",
      "transitionVideo",
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
      "rewindDuration",
      "returnDuration",
      "preOnsetReturn",
      "holdDuration",
      "morphDuration",
      "sampleInterval",
      "loopToggle",
      "clipEditor",
      "analyzeClipButton",
      "emotionSelect",
      "clipCutRatio",
      "detectionBadge",
      "analysisChart",
      "emotionLegend",
      "onsetInput",
      "peakInput",
      "settleInput",
      "onsetDuration",
      "riseDuration",
      "settleDuration",
      "exportButton",
      "exportJsonButton",
      "status",
      "progress",
    ].map((id) => [id, $(id)]),
  );

  const detectorOptions = new faceapi.TinyFaceDetectorOptions({
    inputSize: 320,
    scoreThreshold: 0.35,
  });
  const analysisCanvas = document.createElement("canvas");
  const outputContext = el.outputCanvas.getContext("2d", { alpha: false });
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
  let analysisGeneration = 0;
  let nextId = 1;
  let importedConfig = null;
  let faceLandmarkerPromise = null;
  const delaunatorPromise = import(DELAUNATOR_URL).then(
    (module) => module.default,
  );

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
  ]);

  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
  const easeIntoFold = (progress) => {
    const value = clamp(progress, 0, 1);
    return value + value ** 2 - value ** 3;
  };
  const easeOutOfFold = (progress) => {
    const value = clamp(progress, 0, 1);
    return 2 * value ** 2 - value ** 3;
  };
  const number = (value, fallback = 0) =>
    Number.isFinite(Number(value)) ? Number(value) : fallback;
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
        cutRatio: 85,
        onset: 0,
        peak: 0,
        settle: 0,
        samples: [],
        analysisSeries: [],
        baselineShape: null,
        alignmentPose: null,
        homeMatch: null,
        importedHomeMatch: null,
        detectionRate: null,
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
    const ratio = clamp(number(clip.cutRatio, 85) / 100, 0.1, 0.99);
    return clip.onset + Math.max(0.01, clip.peak - clip.onset) * ratio;
  }

  function preOnsetTarget(clip) {
    return Math.max(
      0,
      clip.onset - clamp(number(el.preOnsetReturn.value, 0.3), 0, 2),
    );
  }

  function clipPlan(clip, index = clips.indexOf(clip)) {
    const homeClip = clips[0] || clip;
    const homeTime = preOnsetTarget(homeClip);
    const cut = cutTime(clip);
    const returnTime =
      index === 0
        ? clamp(homeTime, 0, cut)
        : clamp(clip.homeMatch?.returnTime ?? preOnsetTarget(clip), 0, cut);
    const morphIn = index > 0
      ? clamp(number(el.morphDuration.value, 0.2), 0, 1)
      : 0;
    const morphInStart = index > 0
      ? clamp(clip.onset - morphIn, 0, clip.onset)
      : 0;
    const morphInEnd = index > 0 ? clip.onset : 0;
    const intro = index === 0
      ? clamp(number(el.morphDuration.value, 0.2), 0, 1)
      : 0;
    const sourceStart = clip.onset;
    const forward = clamp(number(el.risePlaybackDuration.value, 0.45), 0.05, 5);
    const reverse = clamp(number(el.rewindDuration.value, 0.45), 0.05, 5);
    const homeReturn = clamp(number(el.returnDuration.value, 0.3), 0.05, 5);
    const reverseMorphStart = clip.onset;
    const morphOut = index > 0 ? homeReturn : 0;
    const returnReverse = index === 0 ? homeReturn : 0;
    const hold = clamp(number(el.holdDuration.value, 0.1), 0, 5);
    return {
      clip,
      index,
      homeClip,
      homeTime,
      sourceStart,
      returnTime,
      cut,
      intro,
      morphInStart,
      morphInEnd,
      morphIn,
      forward,
      reverse,
      reverseMorphStart,
      morphOut,
      returnReverse,
      hold,
      duration: intro + morphIn + forward + reverse + morphOut + returnReverse + hold,
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
    for (let index = 0; index < all.length; index += 1) {
      const plan = all[index];
      if (cursor <= plan.duration || index === all.length - 1) {
        if (cursor <= plan.intro && plan.intro > 0) {
          return {
            ...plan,
            index,
            sourceTime: plan.clip.onset * clamp(cursor / plan.intro, 0, 1),
            phase: "INTRO",
          };
        }
        cursor -= plan.intro;
        if (cursor <= plan.morphIn && plan.morphIn > 0) {
          return {
            ...plan,
            index,
            clip: plan.homeClip,
            nextClip: plan.clip,
            sourceTime: plan.homeTime,
            nextSourceTime:
              plan.morphInStart +
              (plan.morphInEnd - plan.morphInStart) *
                clamp(cursor / plan.morphIn, 0, 1),
            transitionProgress: clamp(cursor / plan.morphIn, 0, 1),
            phase: "MORPH_IN",
          };
        }
        cursor -= plan.morphIn;
        if (cursor <= plan.forward)
          return {
            ...plan,
            index,
            sourceTime:
              plan.sourceStart +
              (plan.cut - plan.sourceStart) *
                easeIntoFold(cursor / plan.forward),
            phase: "FORWARD",
          };
        cursor -= plan.forward;
        if (cursor <= plan.reverse) {
          return {
            ...plan,
            index,
            sourceTime:
              plan.cut -
              (plan.cut - plan.reverseMorphStart) *
                easeOutOfFold(cursor / plan.reverse),
            phase: "REWIND",
          };
        }
        cursor -= plan.reverse;
        if (cursor <= plan.morphOut && plan.morphOut > 0) {
          return {
            ...plan,
            index,
            nextClip: plan.homeClip,
            sourceTime:
              plan.reverseMorphStart -
              (plan.reverseMorphStart - plan.returnTime) *
                clamp(cursor / plan.morphOut, 0, 1),
            nextSourceTime: plan.homeTime,
            transitionProgress: clamp(cursor / plan.morphOut, 0, 1),
            phase: "MORPH_OUT",
          };
        }
        cursor -= plan.morphOut;
        if (cursor <= plan.returnReverse && plan.returnReverse > 0) {
          return {
            ...plan,
            index,
            sourceTime:
              plan.reverseMorphStart -
              (plan.reverseMorphStart - plan.returnTime) *
                clamp(cursor / plan.returnReverse, 0, 1),
            phase: "REWIND_HOME",
          };
        }
        cursor -= plan.returnReverse;
        return {
          ...plan,
          index,
          clip: plan.homeClip,
          sourceTime: plan.homeTime,
          phase: "HOME",
        };
      }
      cursor -= plan.duration;
    }
    return null;
  }

  function loadVideoSlot(video, clip, secondary = false) {
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

  const loadSource = (clip) => loadVideoSlot(el.sourceVideo, clip, false);
  const seekSource = (time) => seekVideoSlot(el.sourceVideo, time);

  function sizeCanvas() {
    const reference = clips[0];
    const width = Math.max(1, reference?.width || 1280);
    const height = Math.max(1, reference?.height || 720);
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

  async function renderMorphFrame(ctx, mapped, width, height) {
    const nextClip = mapped.nextClip;
    if (!nextClip) {
      drawAlignedFrame(ctx, mapped.clip, width, height);
      return;
    }
    await Promise.all([
      loadVideoSlot(el.sourceVideo, mapped.clip, false),
      loadVideoSlot(el.transitionVideo, nextClip, true),
    ]);
    await Promise.all([
      seekVideoSlot(el.sourceVideo, mapped.sourceTime),
      seekVideoSlot(el.transitionVideo, mapped.nextSourceTime),
    ]);
    resizeWorkCanvases(width, height);
    drawAlignedFrame(
      alignedCanvasA.getContext("2d", { alpha: false }),
      mapped.clip,
      width,
      height,
      el.sourceVideo,
    );
    drawAlignedFrame(
      alignedCanvasB.getContext("2d", { alpha: false }),
      nextClip,
      width,
      height,
      el.transitionVideo,
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

    const sampleA = nearestMeshSample(mapped.clip, mapped.sourceTime);
    const sampleB = nearestMeshSample(nextClip, mapped.nextSourceTime);
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
      await loadSource(mapped.clip);
      if (force || Math.abs(lastRenderedTime - mapped.sourceTime) > 0.006) {
        await seekSource(mapped.sourceTime);
        drawAlignedFrame(
          outputContext,
          mapped.clip,
          el.outputCanvas.width,
          el.outputCanvas.height,
        );
        lastRenderedTime = mapped.sourceTime;
      }
    }
    el.stageTitle.textContent = mapped.phase.startsWith("MORPH")
      ? `${mapped.index + 1}. ${mapped.clip.name} → ${mapped.nextClip.name}`
      : `${mapped.index + 1}. ${mapped.clip.name}`;
    el.stagePhase.textContent = mapped.phase.startsWith("MORPH")
      ? `${mapped.phase === "MORPH_IN" ? "HOME → ONSET" : "REWIND + MORPH → HOME"}  ${Math.round(mapped.transitionProgress * 100)}%`
      : mapped.phase === "HOME"
        ? `HOME  ${mapped.sourceTime.toFixed(2)}s`
        : `${mapped.phase}  ${mapped.sourceTime.toFixed(2)}s / CUT ${mapped.cut.toFixed(2)}s`;
    updateTransport(mapped.index);
  }

  function updateTransport(
    activeIndex = mapSequenceTime(playPosition)?.index ?? -1,
  ) {
    const total = sequenceDuration();
    el.playhead.max = String(Math.max(0.001, total));
    el.playhead.value = String(clamp(playPosition, 0, total));
    el.timeOutput.textContent = `${playPosition.toFixed(2)} / ${total.toFixed(2)}s`;
    el.playButton.textContent = playing ? "❚❚" : "▶";
    [...el.sequenceTimeline.children].forEach((node, index) =>
      node.classList.toggle("active", index === activeIndex),
    );
  }

  async function tick(now) {
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
    try {
      await renderAt(playPosition);
    } catch (error) {
      pause();
      setStatus(error.message, 0, 0, true);
      return;
    }
    if (playing) animationId = requestAnimationFrame(tick);
  }

  function play() {
    const total = sequenceDuration();
    if (!total || busy) return;
    if (playPosition >= total - 0.001) playPosition = 0;
    playing = true;
    playStartedAt = performance.now() - playPosition * 1000;
    updateTransport();
    animationId = requestAnimationFrame(tick);
  }

  function pause() {
    playing = false;
    cancelAnimationFrame(animationId);
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
      clip.baselineShape = null;
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
    const baselineShape = averageVector(shapes.slice(0, baselineCount));
    clip.baselineShape = baselineShape;
    clip.samples.forEach((sample, index) => {
      const nearbyShapes = clip.samples
        .slice(Math.max(0, index - 2), index + 3)
        .map((item) => item.landmarks)
        .filter(Boolean);
      const shape = averageVector(nearbyShapes);
      sample.landmarkDistance = shape
        ? vectorDistance(shape, baselineShape)
        : null;
      sample.landmarkDelta = shape
        ? shape.map((value, shapeIndex) => value - baselineShape[shapeIndex])
        : null;
    });
  }

  function updateAlignmentPose(clip) {
    const radius = Math.max(
      0.2,
      clamp(number(el.sampleInterval.value, 0.08), 0.04, 1) * 3,
    );
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

  function findHomeMatch(homeClip, targetClip) {
    if (!homeClip.analyzed || !targetClip.analyzed) return null;
    const interval = clamp(number(el.sampleInterval.value, 0.08), 0.04, 1);
    const preOnset = clamp(number(el.preOnsetReturn.value, 0.3), 0, 2);
    const homeTime = preOnsetTarget(homeClip);
    const homeCandidates = homeClip.samples.filter(
      (sample) => sample.landmarkDelta,
    );
    const homeSample = homeCandidates.reduce(
      (best, sample) =>
        !best ||
        Math.abs(sample.time - homeTime) < Math.abs(best.time - homeTime)
          ? sample
          : best,
      null,
    );
    if (!homeSample) return null;
    const targetEarliest = Math.max(
      0,
      targetClip.onset - Math.max(interval, preOnset * 2),
    );
    const targetLatest = Math.max(
      0,
      targetClip.onset - Math.min(interval * 0.5, Math.max(0.01, preOnset)),
    );
    const targetCandidates = targetClip.samples.filter(
      (sample) =>
        sample.landmarkDelta &&
        sample.time >= targetEarliest &&
        sample.time <= targetLatest,
    );
    let best = null;
    targetCandidates.forEach((sample) => {
      const distance = vectorDistance(
        homeSample.landmarkDelta,
        sample.landmarkDelta,
      );
      if (distance === null || (best && distance >= best.distance)) return;
      best = {
        homeTime,
        entryTime: targetClip.onset,
        returnTime: sample.time,
        distance,
      };
    });
    return best;
  }

  function rebuildTransitions() {
    clips.forEach((clip) => {
      clip.homeMatch = null;
    });
    const homeClip = clips[0];
    for (let index = 1; index < clips.length; index += 1) {
      const clip = clips[index];
      const importedMatch = clip.importedHomeMatch;
      clip.homeMatch = findHomeMatch(homeClip, clip) || (
        importedMatch && Number.isFinite(Number(importedMatch.returnTime))
          ? {
              ...importedMatch,
              homeTime: preOnsetTarget(homeClip),
              entryTime: clip.onset,
              returnTime: clamp(Number(importedMatch.returnTime), 0, clip.onset),
            }
          : null
      );
    }
  }

  function smoothed(clip) {
    return clip.samples.map((sample, index) => {
      const window = clip.samples.slice(Math.max(0, index - 2), index + 3);
      const expressionSamples = window.filter((item) => item.faceApiDetected);
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
    const baseline =
      valid.slice(0, baselineCount).reduce((sum, value) => sum + value, 0) /
      baselineCount;
    const amplitude = Math.max(...valid) - baseline;
    return { baseline, amplitude, usable: amplitude >= minimumAmplitude };
  }

  function crossing(items, level, direction, start, end) {
    for (let i = Math.max(1, start); i <= end; i += 1) {
      const a = items[i - 1],
        b = items[i];
      const hit =
        direction === "up"
          ? a.value <= level && b.value >= level
          : a.value >= level && b.value <= level;
      if (!hit) continue;
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
      crossing(data, 0.08, "up", 1, peakIndex) ??
      data[Math.max(0, peakIndex - 1)].time;
    const peak =
      crossing(data, 0.92, "up", 1, peakIndex) ?? data[peakIndex].time;
    const settle =
      crossing(data, 0.2, "down", peakIndex + 1, data.length - 1) ??
      clip.duration;
    clip.onset = clamp(onset, 0, clip.duration);
    clip.peak = clamp(Math.max(clip.onset + 0.01, peak), 0, clip.duration);
    clip.settle = clamp(Math.max(clip.peak, settle), 0, clip.duration);
    updateAlignmentPose(clip);
    return true;
  }

  function analysisFrame() {
    const scale = Math.min(
      1,
      320 / Math.max(el.sourceVideo.videoWidth, el.sourceVideo.videoHeight),
    );
    analysisCanvas.width = Math.max(
      1,
      Math.round(el.sourceVideo.videoWidth * scale),
    );
    analysisCanvas.height = Math.max(
      1,
      Math.round(el.sourceVideo.videoHeight * scale),
    );
    analysisCanvas
      .getContext("2d", { alpha: false })
      .drawImage(
        el.sourceVideo,
        0,
        0,
        analysisCanvas.width,
        analysisCanvas.height,
      );
    return analysisCanvas;
  }

  async function analyzeClip(
    clip,
    generation,
    progressOffset = 0,
    progressTotal = 1,
  ) {
    await models;
    const faceLandmarker = await getFaceLandmarker();
    await loadSource(clip);
    const interval = clamp(number(el.sampleInterval.value, 0.08), 0.04, 1);
    const count = Math.max(2, Math.ceil(clip.duration / interval));
    clip.samples = [];
    clip.analysisSeries = [];
    for (let index = 0; index < count; index += 1) {
      if (generation !== analysisGeneration) return;
      const time = Math.min(
        index * interval,
        Math.max(0, clip.duration - 0.01),
      );
      setStatus(
        `${clip.name} を解析中 ${index + 1}/${count}`,
        progressOffset + (index + 1) / count,
        progressTotal,
      );
      await seekSource(time);
      const frame = analysisFrame();
      const landmarkResult = faceLandmarker.detect(frame);
      const detectedMesh = landmarkResult.faceLandmarks?.[0];
      const faceMesh =
        detectedMesh?.map((point) => ({
          x: point.x,
          y: point.y,
          z: point.z || 0,
        })) || null;
      const landmarks = normalizeMediaPipeLandmarks(detectedMesh);
      const eyePose = mediaPipeEyePose(detectedMesh);
      const detection = await faceapi
        .detectSingleFace(frame, detectorOptions)
        .withFaceExpressions();
      clip.samples.push({
        time,
        detected: Boolean(detection || landmarks),
        faceApiDetected: Boolean(detection),
        mediapipeDetected: Boolean(landmarks),
        landmarks,
        faceMesh,
        eyePose,
        expressions: Object.fromEntries(
          EMOTIONS.map((emotion) => [
            emotion,
            detection?.expressions[emotion] || 0,
          ]),
        ),
      });
    }
    clip.detectionRate =
      clip.samples.filter((sample) => sample.detected).length /
      clip.samples.length;
    clip.landmarkDetectionRate =
      clip.samples.filter((sample) => sample.mediapipeDetected).length /
      clip.samples.length;
    clip.analyzed = detectBoundaries(clip);
    clip.importedHomeMatch = null;
    rebuildTransitions();
  }

  async function runAnalysis(targets) {
    if (busy || !modelsReady || !targets.length) return;
    pause();
    const generation = ++analysisGeneration;
    setBusy(true);
    try {
      for (let i = 0; i < targets.length; i += 1) {
        await analyzeClip(targets[i], generation, i, targets.length);
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
          <div><div class="clipName" title="${escapeHtml(clip.name)}">${escapeHtml(clip.name)}</div><div class="clipMeta">${LABELS[clip.emotion]} · IN ${seconds(plan.sourceStart)} · CUT ${seconds(plan.cut)} · OUT ${seconds(plan.returnTime)}</div></div>
          <div class="clipActions"><button class="iconButton" data-action="up" title="上へ">↑</button><button class="iconButton" data-action="down" title="下へ">↓</button><button class="iconButton" data-action="remove" title="削除">×</button></div>
        </div>
        <div class="phaseBar"><i class="before" style="width:${phasePercent(clip.onset, clip.duration)}"></i><i class="rise" style="width:${phasePercent(clip.peak - clip.onset, clip.duration)}"></i><i class="settle" style="width:${phasePercent(clip.settle - clip.peak, clip.duration)}"></i></div>
        <div class="phaseLabels"><span>発生 ${seconds(clip.onset)}</span><span>ピーク ${seconds(clip.peak)}</span><span>${clip.samples.length ? `MP ${Math.round(clip.landmarkDetectionRate * 100)}%${clip.analyzed ? "" : " / 境界未検出"}` : "未解析"}</span></div>`;
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
        const entryDuration = plan.intro + plan.morphIn;
        const returnDuration = plan.morphOut + plan.returnReverse;
        const phases = [
          ["intro", entryDuration, "導入", index === 0 ? "冒頭→発生" : "HOME→発生"],
          ["expression", plan.forward, "表情化→", "発生→折返し"],
          ["rewind", plan.reverse, "←戻す", "折返し→発生"],
          ["returnHome", returnDuration, "HOMEへ", "発生→ホーム"],
          ["hold", plan.hold, "保持", "ホーム保持"],
        ];
        const phaseNodes = phases
          .filter(([, duration]) => duration > 0)
          .map(([phase, duration, label, description]) => {
            const node = document.createElement("span");
            node.className = `timelinePhase ${phase}`;
            node.style.width = `${(duration / Math.max(0.001, plan.duration)) * 100}%`;
            node.setAttribute("aria-label", label);
            node.title = `${description} ${seconds(duration)}`;
            return node;
          });
        meaning.replaceChildren(...phaseNodes);

        const composition = document.createElement("div");
        composition.className = "timelineComposition";
        const sources = index === 0
          ? [["single", plan.duration, `${clipNumber}のみ`, `${plan.clip.name}のみ`]]
          : [
              ["morph", plan.morphIn, `HOME→${clipNumber}`, `HOMEと${plan.clip.name}のモーフ`],
              ["single", plan.forward + plan.reverse, `${clipNumber}のみ`, `${plan.clip.name}のみ`],
              ["morph", plan.morphOut, `${clipNumber}→HOME`, `${plan.clip.name}とHOMEのモーフ`],
              ["home", plan.hold, "HOME", "HOMEのみ"],
            ];
        const sourceNodes = sources
          .filter(([, duration]) => duration > 0)
          .map(([sourceClass, duration, label, description]) => {
            const node = document.createElement("span");
            node.className = `timelineSource ${sourceClass}`;
            node.style.width = `${(duration / Math.max(0.001, plan.duration)) * 100}%`;
            node.setAttribute("aria-label", label);
            node.title = `${description} ${seconds(duration)}`;
            return node;
          });
        composition.replaceChildren(...sourceNodes);

        const events = [];
        const forwardOnset = document.createElement("i");
        forwardOnset.className = "timelineEvent onset";
        const forwardOnsetOffset = plan.intro + plan.morphIn;
        forwardOnset.style.left = `${(forwardOnsetOffset / Math.max(0.001, plan.duration)) * 100}%`;
        forwardOnset.title = `表情発生↑ ${seconds(plan.clip.onset)}`;
        events.push(forwardOnset);
        const fold = document.createElement("i");
        fold.className = "timelineEvent fold";
        const foldOffset = forwardOnsetOffset + plan.forward;
        fold.style.left = `${(foldOffset / Math.max(0.001, plan.duration)) * 100}%`;
        fold.title = `折返し ${seconds(plan.cut)}`;
        events.push(fold);
        const reverseOnset = document.createElement("i");
        reverseOnset.className = "timelineEvent onset";
        const reverseOnsetOffset = plan.intro + plan.morphIn + plan.forward + plan.reverse;
        reverseOnset.style.left = `${(reverseOnsetOffset / Math.max(0.001, plan.duration)) * 100}%`;
        reverseOnset.title = `表情発生↓ ${seconds(plan.clip.onset)}`;
        events.push(reverseOnset);
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
    if (!clip) return;
    el.emotionSelect.value = clip.emotion;
    el.clipCutRatio.value = String(clip.cutRatio);
    el.emotionLegend.textContent = LABELS[clip.emotion];
    el.onsetInput.value = clip.onset.toFixed(2);
    el.peakInput.value = clip.peak.toFixed(2);
    el.settleInput.value = clip.settle.toFixed(2);
    el.onsetInput.max = clip.duration;
    el.peakInput.max = clip.duration;
    el.settleInput.max = clip.duration;
    el.onsetDuration.textContent = `区間 ${seconds(clip.onset)}`;
    el.riseDuration.textContent = `区間 ${seconds(clip.peak - clip.onset)}`;
    el.settleDuration.textContent = `区間 ${seconds(clip.settle - clip.peak)}`;
    el.detectionBadge.textContent = clip.samples.length
      ? `MediaPipe ${Math.round(clip.landmarkDetectionRate * 100)}%${clip.homeMatch ? ` / HOME ${clip.homeMatch.returnTime.toFixed(2)}→${clip.homeMatch.homeTime.toFixed(2)}s` : ""}${clip.analyzed ? "" : " / 境界未検出"}`
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
    const data = smoothed(clip);
    [
      [BASE_EMOTION, COLORS.neutral],
      [clip.emotion, COLORS.target],
    ].forEach(([emotion, color]) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      let started = false;
      data.forEach((sample) => {
        if (!sample.smooth) {
          started = false;
          return;
        }
        const px = x(sample.time),
          py = y(sample.smooth[emotion]);
        started ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
        started = true;
      });
      ctx.stroke();
    });
    [
      ["landmark", LANDMARK_COLOR, 2],
      ["progression", PROGRESSION_COLOR, 2.5],
    ].forEach(([key, color, width]) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      let started = false;
      clip.analysisSeries.forEach((sample) => {
        if (!Number.isFinite(sample[key])) {
          started = false;
          return;
        }
        const px = x(sample.time),
          py = y(sample[key]);
        started ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
        started = true;
      });
      ctx.stroke();
    });
    [
      [clip.onset, COLORS.onset],
      [clip.peak, COLORS.peak],
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
    ctx.fillRect(
      x(cutTime(clip)) - 2,
      pad.y,
      4,
      rect.height - pad.y - pad.bottom,
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

  function updateSelectedBoundaries() {
    const clip = selectedClip();
    if (!clip) return;
    clip.onset = clamp(
      number(el.onsetInput.value, clip.onset),
      0,
      clip.duration,
    );
    clip.peak = clamp(
      number(el.peakInput.value, clip.peak),
      clip.onset,
      clip.duration,
    );
    clip.settle = clamp(
      number(el.settleInput.value, clip.settle),
      clip.peak,
      clip.duration,
    );
    if (clips[0]?.id === clip.id) {
      clips.forEach((item) => { item.importedHomeMatch = null; });
    } else {
      clip.importedHomeMatch = null;
    }
    updateAlignmentPose(clip);
    rebuildTransitions();
    renderAll();
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
      clamp(Number(value), number(input.min, -Infinity), number(input.max, Infinity)),
    );
  }

  async function applyImportedConfig(config, announce = true) {
    const settings = config.settings || {};
    applySetting(el.risePlaybackDuration, settings.riseDuration);
    applySetting(el.rewindDuration, settings.rewindDuration);
    applySetting(el.returnDuration, settings.returnDuration);
    applySetting(el.preOnsetReturn, settings.preOnsetReturn);
    applySetting(el.holdDuration, settings.holdDuration);
    applySetting(el.sampleInterval, settings.sampleInterval);
    applySetting(
      el.morphDuration,
      settings.morphDuration ?? settings.introDuration,
    );

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
      const ratio = number(saved.cutRatio, clip.cutRatio / 100);
      clip.cutRatio = clamp(ratio <= 1 ? ratio * 100 : ratio, 10, 99);
      const boundaries = saved.boundaries || {};
      clip.onset = clamp(number(boundaries.onset, clip.onset), 0, clip.duration);
      clip.peak = clamp(
        number(boundaries.peak, clip.peak),
        clip.onset,
        clip.duration,
      );
      clip.settle = clamp(
        number(boundaries.settle, clip.settle),
        clip.peak,
        clip.duration,
      );
      clip.importedHomeMatch = saved.homeMatch || (
        Number.isFinite(Number(saved.returnTime))
          ? { returnTime: Number(saved.returnTime) }
          : null
      );
      if (clip.samples.length) updateAlignmentPose(clip);
      ordered.push(clip);
      matched += 1;
    });
    clips = [...ordered, ...remaining];
    selectedId = clips.some((clip) => clip.id === selectedId)
      ? selectedId
      : clips[0]?.id ?? null;
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
      if (!config || typeof config !== "object" || !Array.isArray(config.clips)) {
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
      version: 3,
      mode: "incomplete-deformation-home-anchor",
      algorithm: "fixed-tempo_onset-overlap_forward-reverse_home-loop",
      settings: {
        riseDuration: number(el.risePlaybackDuration.value, 0.45),
        rewindDuration: number(el.rewindDuration.value, 0.45),
        returnDuration: number(el.returnDuration.value, 0.3),
        preOnsetReturn: number(el.preOnsetReturn.value, 0.3),
        holdDuration: number(el.holdDuration.value, 0.1),
        introDuration: number(el.morphDuration.value, 0.2),
        morphDuration: number(el.morphDuration.value, 0.2),
        sampleInterval: number(el.sampleInterval.value, 0.08),
      },
      clips: clips.map((clip, index) => {
        const plan = clipPlan(clip, index);
        return {
          order: index + 1,
          source: clip.name,
          duration: clip.duration,
          emotion: clip.emotion,
          cutRatio: clip.cutRatio / 100,
          analysisMethod: "mediapipe-landmarks-75_expression-25",
          landmarkDetectionRate: clip.landmarkDetectionRate,
          eyeAlignedToFirstClip: Boolean(
            clip.alignmentPose && clips[0]?.alignmentPose,
          ),
          boundaries: {
            onset: clip.onset,
            peak: clip.peak,
            settle: clip.settle,
          },
          sourceStart: plan.sourceStart,
          morphInRange: [plan.morphInStart, plan.morphInEnd],
          cutTime: plan.cut,
          returnTime: plan.returnTime,
          reverseMorphStart: plan.reverseMorphStart,
          homeTime: plan.homeTime,
          homeMatch: clip.homeMatch,
          outputPhases: {
            intro: plan.intro,
            morphIn: plan.morphIn,
            rise: plan.forward,
            rewind: plan.reverse,
            returnHome: plan.morphOut + plan.returnReverse,
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
          await loadSource(mapped.clip);
          await seekSource(mapped.sourceTime);
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
    [el.onsetInput, el.peakInput, el.settleInput].forEach((input) =>
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
    el.clipCutRatio.addEventListener("input", () => {
      const clip = selectedClip();
      if (!clip) return;
      clip.cutRatio = clamp(
        number(el.clipCutRatio.value, clip.cutRatio),
        10,
        99,
      );
    });
    el.clipCutRatio.addEventListener("change", () => {
      const clip = selectedClip();
      if (!clip) return;
      clip.cutRatio = clamp(
        number(el.clipCutRatio.value, clip.cutRatio),
        10,
        99,
      );
      el.clipCutRatio.value = String(clip.cutRatio);
      rebuildTransitions();
      pause();
      playPosition = clamp(playPosition, 0, sequenceDuration());
      requestAnimationFrame(() => {
        renderAll();
        renderAt(playPosition, true);
      });
    });
    [
      el.risePlaybackDuration,
      el.rewindDuration,
      el.returnDuration,
      el.preOnsetReturn,
      el.holdDuration,
      el.morphDuration,
    ].forEach((input) =>
      input.addEventListener("change", () => {
        input.value = String(
          clamp(number(input.value), number(input.min), number(input.max)),
        );
        if (input === el.preOnsetReturn) {
          clips.forEach((clip) => { clip.importedHomeMatch = null; });
        }
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

  initialize();
})();

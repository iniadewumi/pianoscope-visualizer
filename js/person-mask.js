/**
 * Person mask for Pianoscope Visualizer
 *
 * A person/background matte from MediaPipe's selfie segmenter, for the "Shader on
 * body" and "Shader on room" blends. Shaders read it as iPerson (R = person
 * confidence) in the live source's own frame, so it is sampled through videoUV()
 * like the source is; personAt(uv) in the converter preamble does that.
 *
 * Nothing loads until one of those blends is picked: the runtime is ~12 MB of
 * WASM. Inference runs in js/person-mask-worker.js (about 7 ms a frame, which the
 * projector cannot spare on the shared main thread); the main thread only
 * downscales the frame, posts it, and smooths the mask that comes back. If the
 * worker cannot be built or cannot load the runtime, the same segmenter runs in
 * the page instead, as it did before the worker existed.
 *
 * For a show without internet, copy the package's wasm/ folder and
 * vision_bundle.mjs, plus the .tflite model, next to the site and point
 * TASKS_BASE and MODEL_URL at them. Copy js/person-mask-worker.js too: it is
 * fetched as its own file, and it gets the two URLs from here.
 */
window.PianoscopePersonMask = (function () {
    'use strict';

    const TASKS_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1';
    const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/'
        + 'selfie_segmenter/float16/latest/selfie_segmenter.tflite';

    // The model resizes its input to 256x256 anyway. Segmenting a downscaled copy
    // keeps the mask it hands back, and the upload, at tens of KB rather than a
    // full frame of floats.
    const INPUT_LONG_EDGE = 256;
    const MAX_FPS = 20;
    // Share of the previous mask kept on each run. Hides the model's edge shimmer
    // from one frame to the next, at the cost of a little lag on fast movement.
    const KEEP = 0.45;

    let status = 'idle'; // idle | loading | ready | failed
    let segmenter = null;
    let personIndex = -1;

    let scratch = null;
    let scratchCtx = null;
    let mask = null;
    let maskW = 0;
    let maskH = 0;
    let version = 0;

    let lastRunMs = -Infinity;
    let lastFrameKey = null;
    let lastTimestamp = 0;
    let warnedRun = false;

    // Worker path. worker is null once we have fallen back to the in-page segmenter.
    let worker = null;
    let inFlight = false;
    // Bumped by reset(); a mask that comes back for an older generation is dropped.
    let generation = 0;

    function startWorker() {
        status = 'loading';
        try {
            worker = new Worker('js/person-mask-worker.js');
        } catch (err) {
            worker = null;
            fallBack(err);
            return;
        }
        worker.onmessage = (event) => {
            const msg = event.data;
            if (msg.type === 'ready') {
                status = 'ready';
            } else if (msg.type === 'failed') {
                fallBack(msg.message);
            } else if (msg.generation !== undefined) {
                inFlight = false;
                if (msg.type === 'mask' && msg.generation === generation) {
                    writeMask(msg.confidence, msg.width, msg.height);
                } else if (msg.type === 'error' && !warnedRun) {
                    warnedRun = true;
                    console.warn('Person mask: segmentation failed on a frame.', msg.message);
                }
            }
        };
        worker.onerror = (event) => {
            if (status !== 'ready') fallBack(event.message || 'worker error');
        };
        worker.postMessage({ type: 'init', tasksBase: TASKS_BASE, modelUrl: MODEL_URL });
    }

    function fallBack(reason) {
        if (worker) {
            worker.terminate();
            worker = null;
        }
        inFlight = false;
        console.warn('Person mask: the worker did not start, so segmentation runs on the '
            + 'main thread instead.', reason);
        load();
    }

    async function load() {
        status = 'loading';
        try {
            const vision = await import(TASKS_BASE + '/vision_bundle.mjs');
            const fileset = await vision.FilesetResolver.forVisionTasks(TASKS_BASE + '/wasm');
            segmenter = await vision.ImageSegmenter.createFromOptions(fileset, {
                baseOptions: { modelAssetPath: MODEL_URL, delegate: 'CPU' },
                runningMode: 'VIDEO',
                outputCategoryMask: false,
                outputConfidenceMasks: true
            });
            const labels = segmenter.getLabels ? segmenter.getLabels() : [];
            personIndex = labels.findIndex((label) => /person|human|selfie|foreground/i.test(label));
            status = 'ready';
        } catch (err) {
            status = 'failed';
            console.warn('Person mask: the segmenter did not load, so the body/room blends '
                + 'show the plain shader.', err);
        }
    }

    function writeMask(confidence, width, height) {
        const fresh = !mask || maskW !== width || maskH !== height;
        if (fresh) {
            mask = new Uint8Array(width * height);
            maskW = width;
            maskH = height;
        }
        for (let i = 0; i < mask.length; i++) {
            const v = confidence[i] * 255;
            mask[i] = fresh ? v : mask[i] * KEEP + v * (1 - KEEP);
        }
        version++;
    }

    /**
     * Segment the source if a new frame is due. Cheap to call every render tick:
     * it returns straight away until the runtime has loaded, MAX_FPS allows
     * another run, and frameKey says the picture has changed.
     */
    function update(source, width, height, frameKey, nowMs) {
        if (status === 'idle') {
            if (typeof Worker === 'function') startWorker();
            else fallBack('no Worker support');
            return;
        }
        if (status !== 'ready' || !source || !width || !height) return;
        if (worker && inFlight) return;
        if (nowMs - lastRunMs < 1000 / MAX_FPS || frameKey === lastFrameKey) return;
        lastRunMs = nowMs;
        lastFrameKey = frameKey;

        const scale = Math.min(1, INPUT_LONG_EDGE / Math.max(width, height));
        const w = Math.max(1, Math.round(width * scale));
        const h = Math.max(1, Math.round(height * scale));

        // VIDEO mode rejects a timestamp that does not move forward.
        const timestamp = Math.max(Math.round(nowMs), lastTimestamp + 1);
        lastTimestamp = timestamp;

        if (worker) {
            inFlight = true;
            const sent = generation;
            createImageBitmap(source, { resizeWidth: w, resizeHeight: h, resizeQuality: 'low' })
                .then((bitmap) => {
                    // Reset or fallback happened while the bitmap was being made.
                    if (!worker || sent !== generation) {
                        bitmap.close();
                        inFlight = false;
                        return;
                    }
                    worker.postMessage({ type: 'frame', bitmap, timestamp, generation: sent },
                        [bitmap]);
                })
                .catch((err) => {
                    inFlight = false;
                    if (!warnedRun) {
                        warnedRun = true;
                        console.warn('Person mask: could not copy a frame.', err);
                    }
                });
            return;
        }

        if (!scratch) {
            scratch = document.createElement('canvas');
            scratchCtx = scratch.getContext('2d');
        }
        if (scratch.width !== w || scratch.height !== h) {
            scratch.width = w;
            scratch.height = h;
        }
        scratchCtx.drawImage(source, 0, 0, w, h);

        let result = null;
        try {
            result = segmenter.segmentForVideo(scratch, timestamp);
            const masks = result.confidenceMasks || [];
            // Labelled models say which mask is the person. Otherwise a pair is
            // background then person, and a single mask is the person.
            const index = personIndex >= 0 ? personIndex : (masks.length > 1 ? 1 : 0);
            const m = masks[index];
            if (m) writeMask(m.getAsFloat32Array(), m.width, m.height);
        } catch (err) {
            if (!warnedRun) {
                warnedRun = true;
                console.warn('Person mask: segmentation failed on a frame.', err);
            }
        } finally {
            if (result && result.close) result.close();
        }
    }

    /** Drop the matte so a new source does not start blended with the old one. */
    function reset() {
        mask = null;
        maskW = 0;
        maskH = 0;
        lastFrameKey = null;
        generation++;
        inFlight = false;
        version++;
    }

    // One reused object: the visualizer and the kiosk both read this every frame.
    const state = { active: false, data: null, width: 0, height: 0, version: 0, status: 'idle' };
    function read() {
        state.active = status === 'ready' && !!mask;
        state.data = mask;
        state.width = maskW;
        state.height = maskH;
        state.version = version;
        state.status = status;
        return state;
    }

    return { update, reset, read };
})();

window.pianoscopeReadPersonMask = () => window.PianoscopePersonMask.read();

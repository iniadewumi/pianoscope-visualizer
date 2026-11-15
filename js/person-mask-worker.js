/**
 * Person mask worker for Pianoscope Visualizer
 *
 * Runs MediaPipe's selfie segmenter off the main thread so the projector window
 * keeps its frame budget. js/person-mask.js owns the protocol:
 *
 *   in:  { type: 'init', tasksBase, modelUrl }
 *        { type: 'frame', bitmap, timestamp, generation }
 *   out: { type: 'ready' } | { type: 'failed', message }
 *        { type: 'mask', confidence: Float32Array, width, height, generation }
 *
 * This is a classic worker on purpose: the runtime's loader pulls its WASM glue in
 * with importScripts, which module workers do not have. The ES bundle is still
 * reachable from a classic worker through dynamic import().
 */
'use strict';

let segmenter = null;
let personIndex = -1;

async function load(tasksBase, modelUrl) {
    try {
        const vision = await import(tasksBase + '/vision_bundle.mjs');
        const fileset = await vision.FilesetResolver.forVisionTasks(tasksBase + '/wasm');
        segmenter = await vision.ImageSegmenter.createFromOptions(fileset, {
            baseOptions: { modelAssetPath: modelUrl, delegate: 'CPU' },
            runningMode: 'VIDEO',
            outputCategoryMask: false,
            outputConfidenceMasks: true
        });
        const labels = segmenter.getLabels ? segmenter.getLabels() : [];
        personIndex = labels.findIndex((label) => /person|human|selfie|foreground/i.test(label));
        self.postMessage({ type: 'ready' });
    } catch (err) {
        self.postMessage({ type: 'failed', message: String((err && err.message) || err) });
    }
}

function segment(msg) {
    const bitmap = msg.bitmap;
    let result = null;
    let out = null;
    try {
        if (!segmenter) return;
        result = segmenter.segmentForVideo(bitmap, msg.timestamp);
        const masks = result.confidenceMasks || [];
        // Labelled models say which mask is the person. Otherwise a pair is
        // background then person, and a single mask is the person.
        const index = personIndex >= 0 ? personIndex : (masks.length > 1 ? 1 : 0);
        const m = masks[index];
        if (m) {
            // getAsFloat32Array may be a view onto WASM memory: copy before close().
            out = {
                type: 'mask',
                confidence: new Float32Array(m.getAsFloat32Array()),
                width: m.width,
                height: m.height,
                generation: msg.generation
            };
        }
    } catch (err) {
        out = { type: 'error', message: String((err && err.message) || err), generation: msg.generation };
    } finally {
        if (result && result.close) result.close();
        if (bitmap && bitmap.close) bitmap.close();
    }
    // Always answer, even with nothing, so the main thread clears its in-flight flag.
    if (!out) out = { type: 'empty', generation: msg.generation };
    if (out.confidence) self.postMessage(out, [out.confidence.buffer]);
    else self.postMessage(out);
}

self.onmessage = (event) => {
    const msg = event.data;
    if (msg.type === 'init') load(msg.tasksBase, msg.modelUrl);
    else if (msg.type === 'frame') segment(msg);
};

/**
 * Pianoscope Visualizer — the projector window (kiosk.html).
 *
 * Draws the same shader as the operator's window in its own WebGL context.
 * Compiling, the uniform table, feedback buffers and the draw are the shared
 * code in js/gl-shared.js; what lives here is where this window's textures come
 * from. Nothing is sent to it but shader source: audio, video, display text,
 * the person matte and the pointer are all pulled from the operator's window
 * every frame, because that window's own loop stops whenever it is hidden
 * (minimised, or covered by fullscreen DJ software) and anything it pushed
 * froze on the projector exactly then.
 */
(function () {
    'use strict';

    const GLX = window.PianoscopeGL;

    const canvas = document.getElementById('kiosk-canvas');
    const overlayCanvas = document.getElementById('kiosk-text-overlay');
    const overlayCtx = overlayCanvas ? overlayCanvas.getContext('2d') : null;
    let lastOverlayVersion = -1;

    const MAX_PIXEL_COUNT = 2560 * 1440;
    const DPR_CAP = 1.75;
    const BLACK_PIXEL = new Uint8Array([0, 0, 0, 255]);

    let gl = null;
    let parallelCompileExt = null;
    let currentProgram = null;
    let pendingBuild = null;
    let uniforms = {};

    let usesFeedbackPass = false;
    let feedbackWantsHDR = false;
    let feedbackState = null;
    let resolutionIsVec2 = false;

    let audioTexture = null;
    let audioData = null;
    let blackTexture = null;
    let noiseTexture = null;
    // Receives frames from the operator's video element.
    let videoTexture = null;
    // Size of the video texture's storage (texStorage2D); 0 until allocated.
    let videoTexWidth = 0;
    let videoTexHeight = 0;
    // The operator's person matte (js/person-mask.js), copied in when it changes.
    let personTexture = null;
    let personTextureVersion = -1;
    let displayTextTexture = null;
    let lastDisplayTextVersion = -1;
    let fittedTextTexture = null;
    let lastFittedVersion = -1;

    let animationFrameId = null;
    const startTime = performance.now() / 1000;
    let lastTime = startTime;
    let frameCount = 0;
    // Seconds added to this window's own clock to match the operator's iTime.
    let clockOffset = null;

    // This window's half of a frame's uniforms; one reused object.
    const frameState = {
        width: 1, height: 1, resolutionIsVec2: false,
        time: 0, delta: 0, frame: 0,
        pointer: null, clickFresh: false,
        videoNewFrame: false, personActive: false, sampleRate: 0,
        audioFeatures: null
    };

    // --- The operator's window -----------------------------------------

    /**
     * The window that drives this one: the page that opened it, or, when the
     * kiosk is embedded in a same-origin frame, the page around it.
     * Null once that window is gone.
     */
    function host() {
        try {
            if (window.opener && !window.opener.closed) return window.opener;
            if (window.parent !== window
                && window.parent.location.origin === window.location.origin) {
                return window.parent;
            }
        } catch (e) {}
        return null;
    }

    /** A named value off the operator's window; null while it is closed or navigating. */
    function fromHost(name) {
        try {
            const h = host();
            return h ? (h[name] || null) : null;
        } catch (e) {
            return null;
        }
    }

    function callHost(name) {
        try {
            const fn = fromHost(name);
            return typeof fn === 'function' ? fn() : null;
        } catch (e) {
            return null;
        }
    }

    // --- Sizing ---------------------------------------------------------

    function desiredKioskSize() {
        const displayWidth = canvas.clientWidth;
        const displayHeight = canvas.clientHeight;
        if (displayWidth === 0 || displayHeight === 0) {
            return { width: 1, height: 1 };
        }
        try {
            const perf = fromHost('PianoscopePerf');
            if (perf) {
                // The operator's scale, pixel cap and shader bias carry over;
                // the density has to be this screen's, not the laptop's.
                return perf.bufferSize(displayWidth, displayHeight, window.devicePixelRatio || 1);
            }
        } catch (e) {}
        const ratio = Math.min(window.devicePixelRatio || 1, DPR_CAP);
        let width = Math.max(1, Math.round(displayWidth * ratio));
        let height = Math.max(1, Math.round(displayHeight * ratio));
        if (width * height > MAX_PIXEL_COUNT) {
            const shrink = Math.sqrt(MAX_PIXEL_COUNT / (width * height));
            width = Math.max(1, Math.round(width * shrink));
            height = Math.max(1, Math.round(height * shrink));
        }
        return { width: width, height: height };
    }

    function resizeKioskCanvas() {
        const size = desiredKioskSize();
        if (canvas.width !== size.width || canvas.height !== size.height) {
            canvas.width = size.width;
            canvas.height = size.height;
            if (gl && usesFeedbackPass) {
                feedbackState = GLX.ensureFeedbackState(
                    gl, feedbackState, size.width, size.height, feedbackWantsHDR);
            }
        }
        if (gl) gl.viewport(0, 0, size.width, size.height);
    }

    // The operator's PianoscopePerf the driver came from, and the driver it
    // handed us. Tracking the owner lets us re-attach by ourselves after the
    // operator window reloads, when the old driver belongs to a dead page.
    let perfOwner = null;
    let perfDriver = null;

    function projectorPerf() {
        try {
            const perf = fromHost('PianoscopePerf');
            if (perf !== perfOwner) {
                perfOwner = perf;
                perfDriver = (perf && typeof perf.attachProjector === 'function' && gl)
                    ? perf.attachProjector(gl) : null;
            }
            return perfDriver;
        } catch (e) {
            perfOwner = null;
            perfDriver = null;
            return null;
        }
    }

    // A real window resize or fullscreen toggle changes the workload, so the
    // controller's samples are stale. Not done inside resizeKioskCanvas: that
    // runs every frame and when the controller itself changes the scale, and
    // invalidating there would reset the controller each time it acts.
    function onKioskWindowResize() {
        resizeKioskCanvas();
        try {
            if (perfDriver) perfDriver.invalidate();
        } catch (e) {
            perfOwner = null;
            perfDriver = null;
        }
    }

    window.addEventListener('resize', onKioskWindowResize);
    document.addEventListener('fullscreenchange', onKioskWindowResize);
    if (typeof ResizeObserver !== 'undefined') {
        new ResizeObserver(resizeKioskCanvas).observe(canvas);
    }
    resizeKioskCanvas();

    // Fullscreen the document, not the WebGL canvas. The text overlay is a
    // sibling canvas, and everything outside the fullscreen element sits
    // behind the backdrop, so canvas-only fullscreen hid the overlay on the
    // projector.
    function toggleFullscreen() {
        if (document.fullscreenElement || document.webkitFullscreenElement) {
            if (document.exitFullscreen) document.exitFullscreen();
            else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
            return;
        }
        const root = document.documentElement;
        if (root.requestFullscreen) root.requestFullscreen();
        else if (root.webkitRequestFullscreen) root.webkitRequestFullscreen();
    }

    canvas.addEventListener('dblclick', toggleFullscreen);

    // A GPU reset leaves this window black. preventDefault() asks the browser
    // to restore the context; on restore, reload. The page holds no state: on
    // load it announces itself to the operator window and is sent the current
    // shader again.
    let contextLost = false;
    canvas.addEventListener('webglcontextlost', function (event) {
        event.preventDefault();
        contextLost = true;
    });
    canvas.addEventListener('webglcontextrestored', function () {
        window.location.reload();
    });

    document.addEventListener('keydown', function (e) {
        if (e.key === 'f' || e.key === 'F') {
            toggleFullscreen();
        } else if (e.key === 'Escape') {
            window.close();
        }
    });

    // --- Messages from the operator's window ----------------------------

    window.addEventListener('message', function (event) {
        // Only the page that drives us may send shaders.
        const h = host();
        if (!h || event.source !== h) return;

        const message = event.data;
        if (!message || typeof message !== 'object') return;

        if (message.type === 'init') {
            // Initialize WebGL once; later inits are reconnects after the
            // operator's window reloaded and need a fresh shader, not a new
            // context.
            initGL();
            h.postMessage({ type: 'kiosk-ready' }, window.location.origin);
        } else if (message.type === 'shader-update') {
            updateShaderProgram(message.vertexShader, message.fragmentShader);
        }
        // 'display-text' is deliberately ignored. This window reads text, font,
        // overlay and style straight off the operator's PianoscopeDisplayText
        // every frame, so that single instance is already authoritative here.
        // Do NOT "mirror" the payload by calling a setter on it: that setter
        // runs in the operator's realm, where it ends in notify(), which posts
        // the same message back to this window.
    });

    // --- Setup ----------------------------------------------------------

    function createPlainTexture(filter) {
        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
        return texture;
    }

    function createBlackTexture(filter) {
        const texture = createPlainTexture(filter);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, BLACK_PIXEL);
        return texture;
    }

    function initGL() {
        if (gl) return;

        // WebGL 2 only, like the operator's window: every shader is GLSL ES 3.00.
        gl = canvas.getContext('webgl2', GLX.CONTEXT_ATTRIBUTES);
        if (!gl) {
            console.error('WebGL 2 not supported in kiosk mode');
            return;
        }

        parallelCompileExt = gl.getExtension('KHR_parallel_shader_compile');

        const quadBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, GLX.QUAD_VERTICES, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

        audioData = new Uint8Array(GLX.AUDIO_TEX_WIDTH);
        audioTexture = createPlainTexture(gl.LINEAR);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, audioData.length, 1, 0,
            gl.LUMINANCE, gl.UNSIGNED_BYTE, audioData);

        blackTexture = createBlackTexture(gl.NEAREST);
        videoTexture = createBlackTexture(gl.LINEAR);
        fittedTextTexture = createBlackTexture(gl.LINEAR);
        displayTextTexture = createBlackTexture(gl.LINEAR);

        // Starts as a 1x1 zero so an unbound read means "no person".
        personTexture = createPlainTexture(gl.LINEAR);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, 1, 1, 0, gl.RED, gl.UNSIGNED_BYTE, new Uint8Array(1));

        startAnimationLoop();
        resizeKioskCanvas();
    }

    // --- Shader switching -----------------------------------------------
    // A switch is compiled in the background while the current program keeps
    // drawing. Asking for link status before the driver is done blocks until it
    // is, which on the projector was a visible freeze on every switch.

    function updateShaderProgram(vertexSource, fragmentSource) {
        if (!gl || contextLost) return;

        // Holding an arrow key in the operator's window outruns compilation,
        // and only the latest request is of any interest.
        if (pendingBuild) {
            GLX.destroyBuild(gl, pendingBuild);
            pendingBuild = null;
        }
        pendingBuild = GLX.startBuild(gl, vertexSource, fragmentSource);
    }

    /** Called at the top of every frame. Never blocks. */
    function pollShaderProgram() {
        if (!pendingBuild) return;
        if (!GLX.buildFinished(gl, parallelCompileExt, pendingBuild)) return;

        const build = pendingBuild;
        pendingBuild = null;

        if (!GLX.buildLinked(gl, build)) {
            console.error('Kiosk shader link error:', GLX.buildLog(gl, build));
            GLX.destroyBuild(gl, build);
            return;
        }

        adoptProgram(build);
    }

    function adoptProgram(build) {
        if (currentProgram) gl.deleteProgram(currentProgram);
        // Shaders live as long as the program they are attached to, so
        // flagging them now frees them with it instead of leaking a pair
        // per switch.
        gl.deleteShader(build.vertexShader);
        gl.deleteShader(build.fragmentShader);

        currentProgram = build.program;
        // The operator window resets its frame counter on every shader switch.
        frameCount = 0;
        gl.useProgram(currentProgram);
        uniforms = GLX.locateUniforms(gl, currentProgram);

        // Read from the program being adopted, not the request: the previous
        // program keeps drawing until now and still needs its own settings.
        resolutionIsVec2 = /uniform\s+vec2\s+iResolution/.test(build.fragmentSource);
        usesFeedbackPass = build.fragmentSource.includes('iWriteFeedback');
        feedbackWantsHDR = build.fragmentSource.includes('FEEDBACK_HDR');
        if (usesFeedbackPass && !noiseTexture) {
            noiseTexture = GLX.createNoiseTexture(gl);
        }
        feedbackState = GLX.configureFeedback(
            gl, feedbackState, build.fragmentSource, canvas.width, canvas.height);
        gl.viewport(0, 0, canvas.width, canvas.height);

        GLX.warmUpDraw(gl);
    }

    // --- Copying the operator's sources into this context ---------------
    // Uploads go through unit 7, which no sampler uses, so they never disturb
    // what bindChannels set up.

    function uploadAudio() {
        const data = callHost('pianoscopeReadAudio');
        if (!data) return;

        const n = Math.min(audioData.length, data.length);
        if (data.subarray) {
            audioData.set(data.subarray(0, n));
        } else {
            for (let i = 0; i < n; i++) audioData[i] = data[i];
        }

        gl.activeTexture(gl.TEXTURE7);
        gl.bindTexture(gl.TEXTURE_2D, audioTexture);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, audioData.length, 1,
            gl.LUMINANCE, gl.UNSIGNED_BYTE, audioData);
    }

    // The operator's window runs the segmenter; this one only copies each new
    // matte. Returns whether a matte is live.
    function uploadPersonMask(wanted) {
        const mask = wanted ? callHost('pianoscopeReadPersonMask') : null;
        const active = !!(mask && mask.active && mask.data);
        if (!active || mask.version === personTextureVersion) return active;

        personTextureVersion = mask.version;
        gl.activeTexture(gl.TEXTURE7);
        gl.bindTexture(gl.TEXTURE_2D, personTexture);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, mask.width, mask.height, 0,
            gl.RED, gl.UNSIGNED_BYTE, mask.data);
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
        return true;
    }

    function uploadTextCanvas(texture, source) {
        gl.activeTexture(gl.TEXTURE7);
        gl.bindTexture(gl.TEXTURE_2D, texture);
        if (source) {
            gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 1);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
            gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 0);
        } else {
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, BLACK_PIXEL);
        }
    }

    function uploadDisplayText(text) {
        // The legacy 2048x512 sheet on iChannel3.
        const packed = text ? text.getPackedCanvas() : null;
        if (packed) {
            const version = text.getVersion();
            if (version !== lastDisplayTextVersion) {
                lastDisplayTextVersion = version;
                uploadTextCanvas(displayTextTexture, packed);
            }
        } else if (lastDisplayTextVersion !== -2) {
            lastDisplayTextVersion = -2;
            uploadTextCanvas(displayTextTexture, null);
        }

        // The fitted sheet. Gated on sheetVersion, not version: a scramble
        // redraws the sheet many times inside one string change, and version
        // only ticks when the string itself changes.
        if (uniforms.iTextSheet && text) {
            const fitted = text.getFittedCanvas();
            const sheetVersion = text.getSheetVersion();
            if (fitted && sheetVersion !== lastFittedVersion) {
                lastFittedVersion = sheetVersion;
                uploadTextCanvas(fittedTextTexture, fitted);
            }
        }
    }

    // This window uploads its own copy of each frame, so it dedupes them
    // independently of the operator's window rather than sharing a flag.
    let watchedVideoElement = null;
    let videoUsesFrameCallback = false;
    let videoHasNewFrame = true;

    function watchFrames(element, still) {
        if (watchedVideoElement === element) return;

        watchedVideoElement = element;
        videoHasNewFrame = true;
        videoUsesFrameCallback = !still && typeof element.requestVideoFrameCallback === 'function';
        if (!videoUsesFrameCallback) return;

        const onFrame = () => {
            videoHasNewFrame = true;
            if (watchedVideoElement === element) {
                element.requestVideoFrameCallback(onFrame);
            }
        };
        element.requestVideoFrameCallback(onFrame);
    }

    /**
     * Both windows share an origin, so the operator's <video> element can be
     * uploaded straight into this context instead of shipping pixels through
     * postMessage. One decoder feeds both canvases in sync. Returns whether a
     * new frame went up (iVideoNewFrame).
     */
    function uploadVideoFrame(controller) {
        const handler = controller.videoHandler;
        const el = handler.getTextureElement();
        if (!el) return false;

        const still = handler.isStill();
        watchFrames(el, still);

        const vw = still ? el.naturalWidth : el.videoWidth;
        const vh = still ? el.naturalHeight : el.videoHeight;
        const ready = still ? el.complete : el.readyState >= 2;
        // Stills upload once; clips skip repeat frames via requestVideoFrameCallback.
        const hasNew = still ? videoHasNewFrame : (!videoUsesFrameCallback || videoHasNewFrame);
        if (!ready || !hasNew || !vw || !vh) return false;

        videoHasNewFrame = false;

        gl.activeTexture(gl.TEXTURE7);
        // Avoid the browser's default colorspace conversion / Y-flip work on
        // every frame; the shaders already handle the flip.
        gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);

        if (videoTexWidth !== vw || videoTexHeight !== vh) {
            // Immutable storage can't be resized in place; recreate the texture.
            gl.deleteTexture(videoTexture);
            videoTexture = createPlainTexture(gl.LINEAR);
            gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, vw, vh);
            videoTexWidth = vw;
            videoTexHeight = vh;
        }

        gl.bindTexture(gl.TEXTURE_2D, videoTexture);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, el);
        gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.BROWSER_DEFAULT_WEBGL);
        return true;
    }

    // --- Binding --------------------------------------------------------
    // The same unit layout as bindTextureChannels in the operator's window.

    let frameHasSource = false;
    let frameEffectLook = false;

    function bindUnit(unit, texture, uniform) {
        gl.activeTexture(gl.TEXTURE0 + unit);
        gl.bindTexture(gl.TEXTURE_2D, texture);
        if (uniform) gl.uniform1i(uniform, unit);
    }

    function bindChannels(readTexture) {
        if (usesFeedbackPass && !frameEffectLook && noiseTexture) {
            // Shader-mode feedback: noise on 0, the previous frame on 1, audio on 2.
            bindUnit(0, noiseTexture, uniforms.iChannel0);
            bindUnit(1, readTexture || blackTexture, uniforms.iChannel1);
            bindUnit(2, audioTexture, uniforms.iChannel2);
        } else {
            // Audio keeps iChannel0 even in the video-effect look so effects
            // stay music-reactive; the frame goes to iChannel1.
            bindUnit(0, audioTexture, uniforms.iChannel0);
            const videoOnOne = frameEffectLook && frameHasSource;
            bindUnit(1, videoOnOne ? videoTexture : blackTexture, uniforms.iChannel1);
            bindUnit(2, blackTexture, uniforms.iChannel2);
        }

        bindUnit(3, displayTextTexture, uniforms.iChannel3);
        if (uniforms.iTextSheet) bindUnit(4, fittedTextTexture, uniforms.iTextSheet);
        if (uniforms.iFeedback) bindUnit(5, readTexture || blackTexture, uniforms.iFeedback);
        if (uniforms.iVideo) bindUnit(6, frameHasSource ? videoTexture : blackTexture, uniforms.iVideo);
        if (uniforms.iPerson) bindUnit(8, personTexture, uniforms.iPerson);
    }

    // --- Frame loop -----------------------------------------------------

    // Mirrors the operator's pointer, so a mouse-driven shader shows the same
    // view here as in the preview. A press is spotted by its count changing,
    // since this loop can run between two of the operator's frames. -1 until
    // the first read, so a press from before the kiosk opened is not replayed
    // as fresh.
    let lastPressCount = -1;

    function readPointer() {
        const p = callHost('pianoscopeReadPointer');
        frameState.pointer = p;
        frameState.clickFresh = false;
        if (!p) return;
        frameState.clickFresh = lastPressCount !== -1 && p.presses !== lastPressCount;
        lastPressCount = p.presses;
    }

    function drawOverlay(text) {
        if (!overlayCtx) return;
        if (!text || !text.isOverlayEnabled()) {
            overlayCanvas.style.display = 'none';
            return;
        }
        overlayCanvas.style.display = 'block';
        const src = text.getOverlayCanvas();
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const ow = Math.max(1, Math.round(overlayCanvas.clientWidth * dpr));
        const oh = Math.max(1, Math.round(overlayCanvas.clientHeight * dpr));
        if (overlayCanvas.width !== ow || overlayCanvas.height !== oh) {
            overlayCanvas.width = ow;
            overlayCanvas.height = oh;
            lastOverlayVersion = -1;
        }
        if (src && text.getVersion() !== lastOverlayVersion) {
            lastOverlayVersion = text.getVersion();
            overlayCtx.clearRect(0, 0, ow, oh);
            overlayCtx.drawImage(src, 0, 0, ow, oh);
        }
    }

    function drawOneFrame(now) {
        const perfDrv = projectorPerf();
        const nowSec = now / 1000;
        // iTime follows the operator's clock so both windows show the same moment
        // of a time-driven shader. Only re-sync past 50 ms of drift: the host's
        // clock is read at an arbitrary point inside its frame, so syncing every
        // frame would inject that jitter into otherwise smooth motion.
        const hostClock = callHost('pianoscopeReadClock');
        if (typeof hostClock === 'number'
            && (clockOffset === null || Math.abs((nowSec + clockOffset) - hostClock) > 0.05)) {
            clockOffset = hostClock - nowSec;
        }
        frameState.time = nowSec + (clockOffset === null ? -startTime : clockOffset);
        frameState.delta = nowSec - lastTime;
        lastTime = nowSec;
        frameCount++;
        frameState.frame = frameCount;
        frameState.width = canvas.width;
        frameState.height = canvas.height;
        frameState.resolutionIsVec2 = resolutionIsVec2;
        frameState.sampleRate = callHost('pianoscopeReadSampleRate') || 0;
        readPointer();

        // Everything below reaches into the operator's window, which can be
        // closed or navigating while this loop is still running. A frame that
        // loses it midway is drawn with what was set so far.
        let controller = fromHost('videoController');
        let text = fromHost('PianoscopeDisplayText');

        gl.useProgram(currentProgram);

        frameHasSource = false;
        frameEffectLook = false;
        frameState.videoNewFrame = false;
        frameState.personActive = false;
        try {
            if (controller) {
                frameHasSource = !!controller.hasSource();
                frameEffectLook = !!controller.isInVideoMode();
                // Upload only when the program on screen can see the frame: the
                // video-effect look always can, and so can any shader that
                // declares iVideo. Every other shader skips the cost.
                if (frameHasSource && (frameEffectLook || uniforms.iVideo)) {
                    try {
                        frameState.videoNewFrame = uploadVideoFrame(controller);
                    } catch (e) {
                        videoHasNewFrame = true;
                    }
                }
                frameState.personActive = uploadPersonMask(controller.wantsPersonMask());
            }
            uploadAudio();
            // After uploadAudio so the features belong to the same analyser read.
            frameState.audioFeatures = callHost('pianoscopeReadAudioFeatures') || null;
            uploadDisplayText(text);
        } catch (e) {
            controller = null;
            text = null;
        }

        try {
            GLX.applyFrameUniforms(gl, uniforms, frameState, controller, text);
        } catch (e) {}

        // The driver owns the GPU timer; the closing operator window must never
        // be able to break this frame, so every call is guarded.
        try { if (perfDrv) perfDrv.beginGpuWork(); } catch (e) { perfOwner = null; perfDriver = null; }
        GLX.drawFrame(gl, uniforms, usesFeedbackPass ? feedbackState : null,
            canvas.width, canvas.height, bindChannels);
        try { if (perfDrv) perfDrv.endGpuWork(); } catch (e) { perfOwner = null; perfDriver = null; }

        // This window's own frame interval is what the room sees, so it steers.
        const frameMs = frameState.delta * 1000;
        if (perfDrv && frameMs > 0 && isFinite(frameMs)) {
            try { perfDrv.endFrame(frameMs); } catch (e) { perfOwner = null; perfDriver = null; }
        }

        try {
            drawOverlay(text);
        } catch (e) {}
    }

    function startAnimationLoop() {
        if (animationFrameId) cancelAnimationFrame(animationFrameId);

        function render(now) {
            animationFrameId = requestAnimationFrame(render);

            if (document.hidden || !gl || contextLost) return;
            resizeKioskCanvas();

            // Promote a finished build before anything else, so the whole
            // frame draws with one program.
            pollShaderProgram();
            // Drive the operator window's housekeeping (shader promotion, video
            // look, person mask, text sequencer) every frame so it survives that
            // window being hidden. Exactly one call per frame, even with no program.
            callHost('pianoscopeTick');
            if (!currentProgram) return;

            drawOneFrame(now);
        }

        animationFrameId = requestAnimationFrame(render);
    }

    // --- Handshake ------------------------------------------------------

    function announceToHost() {
        const h = host();
        if (!h) return;
        try {
            h.postMessage({ type: 'kiosk-loaded' }, window.location.origin);
        } catch (e) {}
    }

    announceToHost();

    // The operator's message listener dies on reload. Keep offering a
    // handshake until the new page claims this window again.
    setInterval(function () {
        const h = host();
        if (!h) return;
        try {
            if (h.kioskWindow !== window) announceToHost();
        } catch (e) {}
    }, 500);

    // For debugging from the console and from test harnesses.
    window.pianoscopeKiosk = {
        state: function () {
            return {
                hasContext: !!gl,
                hasProgram: !!currentProgram,
                building: !!pendingBuild,
                frames: frameCount,
                time: frameState.time,
                width: canvas.width,
                height: canvas.height,
                feedback: feedbackState ? feedbackState.format : null
            };
        },
        // Runs one frame by hand, for environments where rAF is throttled.
        step: function () {
            if (!gl) return false;
            resizeKioskCanvas();
            pollShaderProgram();
            callHost('pianoscopeTick');
            if (!currentProgram) return false;
            drawOneFrame(performance.now());
            return true;
        }
    };
})();

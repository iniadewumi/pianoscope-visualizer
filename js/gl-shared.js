/**
 * Pianoscope Visualizer — the GL code both windows run.
 *
 * The operator's window (js/visualizer-with-shadertoy.js) and the projector
 * window (kiosk.html, js/kiosk-window.js) each own a WebGL context and draw the
 * same shader. Everything here is the part that must not differ between them:
 * how a shader is compiled, which uniforms exist, how a frame's uniforms are
 * filled in, how feedback buffers are allocated and how the frame is drawn.
 *
 * A new uniform is added HERE (UNIFORM_NAMES, and applyFrameUniforms if it is
 * set per frame) and both windows get it. What stays per window is where
 * textures come from: the operator's window owns the sources, the projector
 * copies them into its own context.
 *
 * Loaded as a classic script by index.html and kiosk.html, so each window gets
 * its own instance. Exposes window.PianoscopeGL.
 */
window.PianoscopeGL = (function () {
    'use strict';

    // The defaults are all wrong for a fullscreen-quad shader. MSAA only
    // supersamples primitive edges, and our only primitive edges are the four
    // screen borders, so it buys nothing and costs a multisampled surface plus a
    // resolve every frame. The depth and stencil buffers are never read. Chrome
    // 80+ and Safari pick the integrated GPU unless asked otherwise, which is
    // exactly wrong for 30-iteration raymarchers on a dual-GPU laptop.
    //
    // `alpha` deliberately stays true: asking for an opaque RGB surface makes
    // some drivers emulate it on top of RGBA, which costs more than just writing
    // 1.0 to the alpha channel from the shader.
    const CONTEXT_ATTRIBUTES = {
        alpha: true,
        depth: false,
        stencil: false,
        antialias: false,
        premultipliedAlpha: false,
        preserveDrawingBuffer: false,
        powerPreference: 'high-performance'
    };

    const QUAD_VERTICES = new Float32Array([
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,
         1.0,  1.0
    ]);

    const AUDIO_TEX_WIDTH = 256;

    // Every uniform the runtime can set. Arrays are looked up by their first
    // element and stored under the bare name.
    const UNIFORM_NAMES = [
        'iResolution', 'iTime', 'iTimeDelta', 'iFrame', 'iMouse', 'iDate', 'iSampleRate',
        'iChannel0', 'iChannel1', 'iChannel2', 'iChannel3', 'iChannelResolution[0]',
        'iWriteFeedback', 'iFeedback',
        'iVideo', 'iVideoActive', 'iVideoBlend', 'iVideoBlendAmount', 'iVideoResolution',
        'iFitMode', 'iVideoMirror', 'iVideoNewFrame',
        'iPerson', 'iPersonActive',
        'iTextMorph', 'iTextLineH', 'iTextExtent', 'iTextExtentB', 'iTextLegacyExtent',
        'iTextTransition', 'iTextSheet', 'iTextSheetRes', 'iTextWords[0]', 'iTextWordCount',
        'iKick', 'iBass', 'iLowMid', 'iMid', 'iHigh', 'iKickOnset'
    ];

    function locateUniforms(gl, program) {
        const uniforms = {};
        if (!program) return uniforms;
        for (let i = 0; i < UNIFORM_NAMES.length; i++) {
            const name = UNIFORM_NAMES[i];
            const key = name.endsWith('[0]') ? name.slice(0, -3) : name;
            uniforms[key] = gl.getUniformLocation(program, name);
        }
        return uniforms;
    }

    // --- Compiling ------------------------------------------------------

    function compileShader(gl, type, source) {
        const shader = gl.createShader(type);

        const prefix = '#version 300 es\nprecision highp float;\n' +
            (type === gl.FRAGMENT_SHADER ? 'out vec4 fragColor;\n' : '') +
            '#define texture2D texture\n' +
            (type === gl.VERTEX_SHADER ? '#define attribute in\n' : '');

        if (type === gl.FRAGMENT_SHADER) {
            source = source.replace(/gl_FragColor/g, 'fragColor');
        }

        gl.shaderSource(shader, prefix + source);
        gl.compileShader(shader);

        // Deliberately no COMPILE_STATUS query. It forces a synchronous
        // round-trip to the GPU process and defeats ANGLE's compile cache (which
        // does not store compile status, so asking makes it compile again just to
        // answer). Nearly every real error surfaces at link time instead, and the
        // info logs are still read there if linking fails.
        return shader;
    }

    /**
     * Start compiling and linking a program. Returns at once; nothing may query
     * the program until buildFinished() says so, because any query against a
     * program that is still linking (getUniformLocation included) stalls until
     * it is done.
     */
    function startBuild(gl, vertexSource, fragmentSource) {
        const vertexShader = compileShader(gl, gl.VERTEX_SHADER, vertexSource);
        const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource);

        const program = gl.createProgram();
        gl.attachShader(program, vertexShader);
        gl.attachShader(program, fragmentShader);
        // Location 0 for every program avoids the expensive desktop-GL emulation
        // path for a disabled attrib 0, and stops ANGLE re-specializing a program
        // the first time it meets a vertex layout it did not see at link time.
        gl.bindAttribLocation(program, 0, 'position');
        gl.linkProgram(program);

        return {
            program: program,
            vertexShader: vertexShader,
            fragmentShader: fragmentShader,
            vertexSource: vertexSource,
            fragmentSource: fragmentSource,
            startMs: performance.now()
        };
    }

    /** Whether LINK_STATUS can now be read without blocking. */
    function buildFinished(gl, parallelCompileExt, build) {
        if (parallelCompileExt) {
            return !!gl.getProgramParameter(build.program, parallelCompileExt.COMPLETION_STATUS_KHR);
        }
        // Without the extension, LINK_STATUS blocks. Giving the driver a frame of
        // wall clock to finish in the background shortens the stall even though
        // it cannot remove it.
        return performance.now() - build.startMs >= 24;
    }

    function buildLinked(gl, build) {
        return !!gl.getProgramParameter(build.program, gl.LINK_STATUS);
    }

    function buildLog(gl, build) {
        return gl.getShaderInfoLog(build.fragmentShader)
            || gl.getShaderInfoLog(build.vertexShader)
            || gl.getProgramInfoLog(build.program)
            || 'unknown link failure';
    }

    function destroyBuild(gl, build) {
        gl.deleteProgram(build.program);
        gl.deleteShader(build.vertexShader);
        gl.deleteShader(build.fragmentShader);
    }

    /**
     * Linking is not the last of the driver's work: the first draw with a
     * program can trigger ANGLE specialization and, on WebKit, a walk of every
     * active uniform to cache its location. A one-pixel scissored draw pays that
     * cost here rather than in the first full-resolution frame.
     */
    function warmUpDraw(gl) {
        gl.enable(gl.SCISSOR_TEST);
        gl.scissor(0, 0, 1, 1);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        gl.disable(gl.SCISSOR_TEST);
    }

    // --- Textures -------------------------------------------------------

    function createNoiseTexture(gl) {
        const size = 512;
        const data = new Uint8Array(size * size * 4);
        for (let i = 0; i < data.length; i++) {
            data[i] = (Math.random() * 256) | 0;
        }

        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, size, size, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
        return texture;
    }

    // --- Frame feedback -------------------------------------------------
    // A shader carrying iWriteFeedback is drawn into a ping-pong buffer first and
    // to the canvas second, reading last frame back through iFeedback.

    // Feedback buffers are RGBA8 by default, which is the right storage for a
    // shader that accumulates colour. A shader that accumulates *numbers* — a
    // running mean, a variance — has neither the range (8 bits clamp at 1.0) nor
    // the resolution (1/255 swamps a small variance) for that, and says so with
    // FEEDBACK_HDR in its source. Probing costs driver round-trips, so the answer
    // is cached per context.
    const hdrFormats = new WeakMap();

    function feedbackFormat(gl, wantsHDR) {
        const rgba8 = {
            label: 'rgba8',
            internalFormat: gl.RGBA8,
            format: gl.RGBA,
            type: gl.UNSIGNED_BYTE,
            filter: gl.LINEAR
        };
        if (!wantsHDR) return rgba8;
        if (hdrFormats.has(gl)) return hdrFormats.get(gl) || rgba8;

        let hdr = null;
        const renderFloat = gl.getExtension('EXT_color_buffer_float');
        if (renderFloat && gl.getExtension('OES_texture_float_linear')) {
            hdr = {
                label: 'rgba32f',
                internalFormat: gl.RGBA32F,
                format: gl.RGBA,
                type: gl.FLOAT,
                filter: gl.LINEAR
            };
        } else if (renderFloat || gl.getExtension('EXT_color_buffer_half_float')) {
            // Half float is filterable in core WebGL 2, so LINEAR needs no extension.
            hdr = {
                label: 'rgba16f',
                internalFormat: gl.RGBA16F,
                format: gl.RGBA,
                type: gl.HALF_FLOAT,
                filter: gl.LINEAR
            };
        }
        hdrFormats.set(gl, hdr);

        if (!hdr) {
            // Worth saying out loud: a statistic accumulated in 8 bits does not
            // just lose detail, it stops converging — the update per frame rounds
            // to zero and whatever the buffer holds freezes there.
            console.warn('Feedback buffer: no renderable float format here, so FEEDBACK_HDR '
                + 'shaders fall back to rgba8 and their accumulation will quantise.');
        }
        return hdr || rgba8;
    }

    function destroyFeedbackState(gl, state) {
        if (!state) return;
        state.fbos.forEach((fbo) => gl.deleteFramebuffer(fbo));
        state.textures.forEach((texture) => gl.deleteTexture(texture));
    }

    /** Leaves the viewport at the buffer's size, which is also the canvas's. */
    function clearFeedbackState(gl, state) {
        gl.viewport(0, 0, state.width, state.height);
        gl.clearColor(0, 0, 0, 0);
        state.fbos.forEach((fbo) => {
            gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
            gl.clear(gl.COLOR_BUFFER_BIT);
        });
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    // Returns null rather than a broken state when the driver accepts the texture
    // format but refuses to render to it, which is the usual way float support
    // fails.
    function createFeedbackState(gl, width, height, desc) {
        const textures = [gl.createTexture(), gl.createTexture()];
        textures.forEach((texture) => {
            gl.bindTexture(gl.TEXTURE_2D, texture);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, desc.filter);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, desc.filter);
            // Immutable storage lets the driver lay the texture out in GPU
            // memory immediately, and levels=1 avoids allocating a mip chain
            // we never sample.
            gl.texStorage2D(gl.TEXTURE_2D, 1, desc.internalFormat, width, height);
        });

        // One framebuffer per texture with a permanent attachment. Swapping an
        // attachment every frame instead would invalidate framebuffer
        // completeness and force the driver to revalidate on each ping-pong.
        const fbos = [gl.createFramebuffer(), gl.createFramebuffer()];
        let complete = true;
        fbos.forEach((fbo, index) => {
            gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, textures[index], 0);
            if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) complete = false;
        });
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);

        if (!complete) {
            destroyFeedbackState(gl, { textures, fbos });
            return null;
        }

        return { textures, fbos, readIndex: 0, width, height, format: desc.label };
    }

    /**
     * The feedback buffers for a canvas of this size, reusing `state` when it
     * already fits. May return null when the driver can render to nothing usable.
     */
    function ensureFeedbackState(gl, state, width, height, wantsHDR) {
        const wanted = wantsHDR ? 'hdr' : 'ldr';
        if (state && state.width === width && state.height === height && state.request === wanted) {
            return state;
        }

        destroyFeedbackState(gl, state);

        const desc = feedbackFormat(gl, wantsHDR);
        let next = createFeedbackState(gl, width, height, desc);
        if (!next && desc.label !== 'rgba8') {
            console.warn('Feedback buffer: ' + desc.label + ' is not renderable here, using rgba8');
            next = createFeedbackState(gl, width, height, feedbackFormat(gl, false));
        }
        if (!next) return null;

        next.request = wanted;
        // New storage always starts undefined, resize included, so this clear is
        // what keeps an accumulator from reading garbage on its first frame.
        clearFeedbackState(gl, next);
        return next;
    }

    /**
     * What a newly adopted program needs from the feedback machinery. Returns
     * the state to keep (null when the program has no feedback pass).
     */
    function configureFeedback(gl, state, fragmentSource, width, height) {
        if (!fragmentSource.includes('iWriteFeedback')) {
            destroyFeedbackState(gl, state);
            return null;
        }
        const next = ensureFeedbackState(gl, state, width, height,
            fragmentSource.includes('FEEDBACK_HDR'));
        // A buffer kept across a program swap still holds the previous program's
        // numbers, which the new one would read as its own history.
        if (next && next === state) clearFeedbackState(gl, next);
        return next;
    }

    // --- Per-frame uniforms ---------------------------------------------

    const channelResolutions = new Float32Array([
        AUDIO_TEX_WIDTH, 1, 1,
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
    ]);

    /**
     * Fill in every per-frame uniform except the samplers.
     *
     * `frame` is this window's own clock, buffer and pointer:
     *   width, height, resolutionIsVec2, time, delta, frame,
     *   pointer { u, v, clickU, clickV, down }, clickFresh,
     *   videoNewFrame, personActive, sampleRate,
     *   audioFeatures (Float32Array(6): kick, bass, lowMid, mid, high, kickOnset; or null)
     * `controller` is the operator window's VideoController and `text` its
     * PianoscopeDisplayText; the projector passes the opener's. Either may be
     * null.
     */
    function applyFrameUniforms(gl, u, frame, controller, text) {
        if (u.iResolution) {
            if (frame.resolutionIsVec2) {
                gl.uniform2f(u.iResolution, frame.width, frame.height);
            } else {
                gl.uniform3f(u.iResolution, frame.width, frame.height, 1.0);
            }
        }

        if (u.iTime) gl.uniform1f(u.iTime, frame.time);
        if (u.iTimeDelta) gl.uniform1f(u.iTimeDelta, frame.delta);
        if (u.iFrame) gl.uniform1f(u.iFrame, frame.frame);

        if (u.iMouse) {
            // xy follows the pointer even without a press (Shadertoy only
            // updates it while dragging; the library was tuned on hover). zw is
            // Shadertoy's: the press position, z positive while held and w only
            // on the frame of the press, both negated otherwise, so ported code
            // can tell "held" and "just clicked" from a remembered position.
            const p = frame.pointer;
            if (p) {
                const clickX = p.clickU * frame.width;
                const clickY = p.clickV * frame.height;
                gl.uniform4f(u.iMouse,
                    p.u * frame.width, p.v * frame.height,
                    p.down ? clickX : -clickX,
                    frame.clickFresh ? clickY : -clickY);
            } else {
                gl.uniform4f(u.iMouse, 0, 0, 0, 0);
            }
        }

        if (u.iDate) {
            const d = new Date();
            gl.uniform4f(u.iDate,
                d.getFullYear(),
                d.getMonth(),
                d.getDate(),
                d.getHours() * 3600 + d.getMinutes() * 60 + d.getSeconds() + d.getMilliseconds() / 1000
            );
        }
        if (u.iSampleRate && frame.sampleRate) gl.uniform1f(u.iSampleRate, frame.sampleRate);

        // Shaders need the real size of whatever feeds each channel so they can
        // sample it without distortion.
        const hasSource = !!(controller && controller.hasSource());
        const videoSize = controller ? controller.getVideoSize() : null;
        const videoW = videoSize ? videoSize.width : 1;
        const videoH = videoSize ? videoSize.height : 1;
        const effectLook = !!(controller && controller.isInVideoMode());

        if (u.iVideoResolution) gl.uniform2f(u.iVideoResolution, videoW, videoH);
        if (u.iFitMode && controller) gl.uniform1f(u.iFitMode, controller.getFitMode());
        if (u.iVideoMirror && controller) gl.uniform1f(u.iVideoMirror, controller.getMirror());
        if (u.iVideoNewFrame) gl.uniform1f(u.iVideoNewFrame, frame.videoNewFrame ? 1.0 : 0.0);
        const af = frame.audioFeatures;
        if (u.iKick) gl.uniform1f(u.iKick, af ? af[0] : 0);
        if (u.iBass) gl.uniform1f(u.iBass, af ? af[1] : 0);
        if (u.iLowMid) gl.uniform1f(u.iLowMid, af ? af[2] : 0);
        if (u.iMid) gl.uniform1f(u.iMid, af ? af[3] : 0);
        if (u.iHigh) gl.uniform1f(u.iHigh, af ? af[4] : 0);
        if (u.iKickOnset) gl.uniform1f(u.iKickOnset, af ? af[5] : 0);
        if (u.iPersonActive) gl.uniform1f(u.iPersonActive, frame.personActive ? 1.0 : 0.0);
        if (u.iVideoActive) {
            const live = hasSource
                && (controller.videoHandler.videoLoaded || controller.videoHandler.isLive());
            gl.uniform1f(u.iVideoActive, live ? 1.0 : 0.0);
        }
        if (u.iVideoBlend) gl.uniform1f(u.iVideoBlend, controller ? controller.getBlendMode() : 0);
        if (u.iVideoBlendAmount) {
            gl.uniform1f(u.iVideoBlendAmount, controller ? controller.getBlendAmount() : 1);
        }

        if (u.iChannelResolution) {
            const textSize = text ? text.getSize() : null;
            channelResolutions[3] = effectLook ? videoW : 1;
            channelResolutions[4] = effectLook ? videoH : 1;
            channelResolutions[9] = textSize ? textSize.width : 1;
            channelResolutions[10] = textSize ? textSize.height : 1;
            gl.uniform3fv(u.iChannelResolution, channelResolutions);
        }

        if (u.iTextMorph) gl.uniform1f(u.iTextMorph, text ? text.getMorph() : 0);
        if (u.iTextLineH) gl.uniform1f(u.iTextLineH, text ? text.getLineFrac() : 0.5);
        if (u.iTextExtent) {
            const ext = text ? text.getExtent() : null;
            gl.uniform2f(u.iTextExtent, ext ? ext.width : 0.5, ext ? ext.height : 0.5);
        }
        if (u.iTextExtentB) {
            const extB = text ? text.getExtentB() : null;
            gl.uniform2f(u.iTextExtentB, extB ? extB.width : 0.5, extB ? extB.height : 0.5);
        }
        if (u.iTextLegacyExtent) {
            const leg = text ? text.getLegacyExtent() : null;
            gl.uniform2f(u.iTextLegacyExtent, leg ? leg.width : 0.386, leg ? leg.height : 0.670);
        }
        if (u.iTextTransition) gl.uniform1f(u.iTextTransition, text ? text.getTransitionIndex() : 0);
        if (u.iTextSheetRes) {
            const sheet = text ? text.getFittedSize() : null;
            gl.uniform2f(u.iTextSheetRes, sheet ? sheet.width : 1, sheet ? sheet.height : 1);
        }
        if (u.iTextWords && text) gl.uniform4fv(u.iTextWords, text.getWordRects());
        if (u.iTextWordCount) gl.uniform1f(u.iTextWordCount, text ? text.getWordCount() : 0);
    }

    // --- Drawing --------------------------------------------------------

    /**
     * Draw one frame of the current program. `bindChannels(readTexture)` binds
     * this window's textures; readTexture is last frame's feedback buffer, or
     * null when the program has no feedback pass.
     */
    function drawFrame(gl, u, feedbackState, width, height, bindChannels) {
        if (feedbackState) {
            const readTexture = feedbackState.textures[feedbackState.readIndex];
            const writeIndex = 1 - feedbackState.readIndex;

            bindChannels(readTexture);

            gl.bindFramebuffer(gl.FRAMEBUFFER, feedbackState.fbos[writeIndex]);
            gl.viewport(0, 0, feedbackState.width, feedbackState.height);
            if (u.iWriteFeedback) gl.uniform1f(u.iWriteFeedback, 1.0);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            gl.viewport(0, 0, width, height);
            if (u.iWriteFeedback) gl.uniform1f(u.iWriteFeedback, 0.0);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

            feedbackState.readIndex = writeIndex;
            return;
        }

        bindChannels(null);
        if (u.iWriteFeedback) gl.uniform1f(u.iWriteFeedback, 0.0);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    return {
        CONTEXT_ATTRIBUTES,
        QUAD_VERTICES,
        AUDIO_TEX_WIDTH,
        UNIFORM_NAMES,
        locateUniforms,
        startBuild,
        buildFinished,
        buildLinked,
        buildLog,
        destroyBuild,
        warmUpDraw,
        createNoiseTexture,
        ensureFeedbackState,
        configureFeedback,
        destroyFeedbackState,
        applyFrameUniforms,
        drawFrame
    };
})();

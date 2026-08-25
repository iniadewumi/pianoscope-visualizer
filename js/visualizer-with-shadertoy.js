/**
 * Piansoscope Visualizer with Shadertoy Support
 * 
 * This script handles the WebGL and audio processing to visualize 
 * microphone input in real-time with shader effects.
 */

document.addEventListener('DOMContentLoaded', () => {
    "use strict";
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

function destroyFeedbackState(gl, state) {
    if (!state) return;
    state.fbos.forEach((fbo) => gl.deleteFramebuffer(fbo));
    state.textures.forEach((texture) => gl.deleteTexture(texture));
}

function clearFeedbackTextures(gl, state) {
    gl.viewport(0, 0, state.width, state.height);
    gl.clearColor(0, 0, 0, 0);
    state.fbos.forEach((fbo) => {
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.clear(gl.COLOR_BUFFER_BIT);
    });
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

// Feedback buffers are RGBA8 by default, which is the right storage for a shader
// that accumulates colour. A shader that accumulates *numbers* — a running mean,
// a variance — has neither the range (8 bits clamp at 1.0) nor the resolution
// (1/255 swamps a small variance) for that, and says so with FEEDBACK_HDR in its
// source. Probing costs driver round-trips, so the answer is cached per context.
let hdrFeedbackFormat;
function feedbackFormat(gl, isWebGL2, wantsHDR) {
    const rgba8 = {
        label: 'rgba8',
        internalFormat: isWebGL2 ? gl.RGBA8 : gl.RGBA,
        format: gl.RGBA,
        type: gl.UNSIGNED_BYTE,
        filter: gl.LINEAR
    };
    if (!wantsHDR) return rgba8;
    if (hdrFeedbackFormat !== undefined) return hdrFeedbackFormat || rgba8;

    hdrFeedbackFormat = null;
    if (isWebGL2) {
        const renderFloat = gl.getExtension('EXT_color_buffer_float');
        if (renderFloat && gl.getExtension('OES_texture_float_linear')) {
            hdrFeedbackFormat = {
                label: 'rgba32f',
                internalFormat: gl.RGBA32F,
                format: gl.RGBA,
                type: gl.FLOAT,
                filter: gl.LINEAR
            };
        } else if (renderFloat || gl.getExtension('EXT_color_buffer_half_float')) {
            // Half float is filterable in core WebGL 2, so LINEAR needs no extension.
            hdrFeedbackFormat = {
                label: 'rgba16f',
                internalFormat: gl.RGBA16F,
                format: gl.RGBA,
                type: gl.HALF_FLOAT,
                filter: gl.LINEAR
            };
        }
    } else {
        const halfTexture = gl.getExtension('OES_texture_half_float');
        if (halfTexture && gl.getExtension('EXT_color_buffer_half_float')) {
            hdrFeedbackFormat = {
                label: 'rgba16f',
                internalFormat: gl.RGBA,
                format: gl.RGBA,
                type: halfTexture.HALF_FLOAT_OES,
                filter: gl.getExtension('OES_texture_half_float_linear') ? gl.LINEAR : gl.NEAREST
            };
        }
    }

    if (!hdrFeedbackFormat) {
        // Worth saying out loud: a statistic accumulated in 8 bits does not just
        // lose detail, it stops converging — the update per frame rounds to zero
        // and whatever the buffer holds freezes there.
        console.warn('Feedback buffer: no renderable float format here, so FEEDBACK_HDR '
            + 'shaders fall back to rgba8 and their accumulation will quantise.');
    }
    return hdrFeedbackFormat || rgba8;
}

// Returns null rather than a broken state when the driver accepts the texture
// format but refuses to render to it, which is the usual way float support fails.
function createFeedbackState(gl, isWebGL2, width, height, desc) {
    const textures = [gl.createTexture(), gl.createTexture()];
    textures.forEach((texture) => {
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, desc.filter);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, desc.filter);
        if (isWebGL2) {
            // Immutable storage lets the driver lay the texture out in GPU
            // memory immediately, and levels=1 avoids allocating a mip chain
            // we never sample.
            gl.texStorage2D(gl.TEXTURE_2D, 1, desc.internalFormat, width, height);
        } else {
            gl.texImage2D(gl.TEXTURE_2D, 0, desc.internalFormat, width, height, 0,
                desc.format, desc.type, null);
        }
    });

    // One framebuffer per texture with a permanent attachment. Swapping an
    // attachment every frame instead would invalidate framebuffer completeness
    // and force the driver to revalidate on each ping-pong.
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
    // === CONFIGURATION ===
    // Easy to modify settings
    const audioSmoothingFactor = 0.8; // How smooth the audio response is (0-1)
    const audioSensitivity = 1.5;     // Amplify the audio signal
    
    // === ELEMENTS ===
    const canvas = document.getElementById('visualizer');
    const micToggle = document.getElementById('mic-toggle');
    const fullscreenBtn = document.getElementById('fullscreen');
    const statusEl = document.getElementById('status');
    const h = document.getElementsByTagName('h1')[0];
    const hSub = document.getElementsByTagName('h1')[1];

    window.currentVertexShader = null;
    window.currentFragmentShader = null;
    window.currentVertexShaderSource = null;
    window.currentFragmentShaderSource = null;
    window.audioData = null;
    
    // Check if elements exist to prevent errors
    if (!canvas) {
        console.error('Canvas element with id "visualizer" not found');
        return;
    }
    
    if (!micToggle) {
        console.error('Button element with id "mic-toggle" not found');
        return;
    }
    
    // === AUDIO SETUP ===
    let audioContext;
    let analyser;
    let audioData;
    let micStream;
    let isListening = false;
    let seconds = 0;
    const loud_volume_threshold = 30;
    let frequencyArray;
    let currentDb = 0;
    let lastRenderedDb = -1;
    let nextVolumeReadoutMs = 0;
    
    // === SHADER SOURCES === 
    // Default vertex shader
    const vertexShaderSource = `
        attribute vec2 position;
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
        }
    `;
    
    // Default fragment shader - pulled from your index.html
    const fragmentShaderSource = `
    precision highp float;

    uniform vec2 iResolution;
    uniform float iTime;
    uniform sampler2D iChannel0;

    void main() {
        // Sample audio with minimal processing
        float bass = texture2D(iChannel0, vec2(0.05, 0.0)).x;
        float lowMid = texture2D(iChannel0, vec2(0.15, 0.0)).x;
        float mid = texture2D(iChannel0, vec2(0.3, 0.0)).x;
        float high = texture2D(iChannel0, vec2(0.7, 0.0)).x;
        
        // Beat detection - compare current bass to a threshold
        // This creates more defined pulses instead of continuous response
        float beatThreshold = 0.15;
        float beatPulse = smoothstep(beatThreshold, beatThreshold + 0.1, bass) * 0.3;
        
        // Overall audio intensity (weighted toward bass)
        float audioIntensity = bass * 0.6 + lowMid * 0.25 + mid * 0.1 + high * 0.05;
        
        // Keep original time speed
        float timeFactor = 0.2; // Original speed from XorDev's shader
        
        // Small pulse effect for beats
        float beatPulseFactor = 1.0 + beatPulse;
        
        // Black hole size/density responds to audio intensity
        // Inverse relationship - higher sound = smaller hole
        float baseHoleSize = 0.7; // Base size
        float holeSize = baseHoleSize * (1.0 - audioIntensity * 0.35);
        
        // Density factor - increases with audio intensity
        float densityFactor = 1.0 + audioIntensity * 0.9;
        
        // Audio-reactive parameters that respect the original algorithm
        float distortion = 0.1;  // Keep original distortion
        float iterationBase = 9.0; // Original iteration count
        float iterationCount = iterationBase; // Keep consistent iterations
        
        // Begin with the same coordinate setup as the original
        float i;
        vec2 r = iResolution.xy,
            p = (gl_FragCoord.xy + gl_FragCoord.xy - r) / r.y / 0.7,
            d = vec2(-1, 1), 
            q = 5.0 * p - d,
            c = p * mat2(1, 1, d / (distortion + 5.0 / dot(q, q)));
            
        // Create the spiral effect using log-polar coordinates - maintaining original speed
        vec2 v = c * mat2(cos(log(length(c)) + iTime * timeFactor + vec4(0, 33, 11, 0))) * 5.0;
    
        // Initialize output color
        vec4 O = vec4(0.0);
        i = 0.0;
        
        // Create the iterative pattern
        for(int j = 0; j < 15; j++) {
            // Break based on audio-reactive iteration count
            if(float(j) >= iterationCount) break;
            
            // Add audio-reactive variation to the sine pattern
            float audioMod = 1.0 + bass * 0.2 * sin(i * 0.7); // Subtle per-iteration audio effect
            O += (1.0 + sin(v.xyyx)) * audioMod;
            
            // Update v with the original formula plus subtle audio influence
            v += 0.7 * sin(v.yx * i + iTime) / (i + 0.1) + 0.5; // Added 0.1 to prevent division by zero
            
            i += 1.0; // Increment our float counter
        }
        
        // Calculate i using the original formula
        i = length(sin(v / 0.3) * 0.2 + c * vec2(1, 2)) - 1.0;
        
        // Apply pulse effect to the visual elements
        // Ring size changes with the hole size to create expansion/contraction
        float ringSize = holeSize;
        
        // Ring width pulses with beats 
        float ringWidth = 0.03 * beatPulseFactor;
        float ringEffect = ringWidth + abs(length(p) - ringSize);
        
        // Glow intensity increases with audio intensity and pulses with beats
        float glowIntensity = 3.5 * beatPulseFactor * densityFactor;
        float glowEffect = 0.5 + glowIntensity * exp(0.3 * c.y - dot(c, c));
        
        // Apply subtle color shift with mid frequency
        vec4 colorModifier = vec4(0.6, -0.4, -1.0, 0.0);
        
        // Final color calculation - density increases with audio intensity
        O = 1.0 - exp(-exp(c.x * colorModifier) / O / (1.0 + i * i * densityFactor) / glowEffect / ringEffect * beatPulseFactor);
        
        // Output
        gl_FragColor = O;
    }
    `;
    
    // === WEBGL SETUP ===
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
    const GL_ATTRIBUTES = {
        alpha: true,
        depth: false,
        stencil: false,
        antialias: false,
        premultipliedAlpha: false,
        preserveDrawingBuffer: false,
        powerPreference: 'high-performance'
    };

    const gl = canvas.getContext('webgl2', GL_ATTRIBUTES)
        || canvas.getContext('webgl', GL_ATTRIBUTES)
        || canvas.getContext('experimental-webgl', GL_ATTRIBUTES);
    const isWebGL2 = gl instanceof WebGL2RenderingContext;
    const glslVersion = isWebGL2 ? '300 es' : '100';

    // Asking for the discrete GPU makes context loss meaningfully more likely,
    // because switching GPUs drops the context. Without a handler that is a black
    // screen until the operator reloads, which is unacceptable mid-performance.
    canvas.addEventListener('webglcontextlost', (event) => {
        event.preventDefault();
        animationRunning = false;
        if (statusEl) statusEl.textContent = 'GPU context lost — restoring…';
    });

    canvas.addEventListener('webglcontextrestored', () => {
        if (statusEl) statusEl.textContent = 'GPU context restored — reloading shader';
        // Every GL object belonged to the dead context. Reloading is far more
        // reliable than trying to rebuild the whole graph by hand.
        window.location.reload();
    });
    
    // Initialize video controller after GL context is created
    let videoController = null;
    if (gl && typeof VideoController !== 'undefined') {
        videoController = new VideoController(gl);
        window.videoController = videoController;
    }

    // Fragment source of the last non-video shader, used to rebuild it when
    // video mode is turned off.
    let savedFragmentSource = null;

    function leaveVideoMode() {
        if (videoController) videoController.setVideoMode(false);
        window.currentShaderIsVideoDisplay = false;
    }
    
    function addVersionHeader(shaderSource, isFragmentShader) {
        const versionHeader = glslVersion === '300 es' 
            ? `#version 300 es
               precision highp float;
               ${isFragmentShader ? 'out vec4 fragColor;' : ''}
               #define gl_FragColor fragColor`
            : `#version 100
               precision highp float;`;
        
        return versionHeader + '\n' + shaderSource;
    }

    let usesFeedbackPass = false;
    let feedbackWantsHDR = false;
    let iResolutionIsVec2 = false;
    let noiseTexture = null;
    let feedbackState = null;
    
    // Performance manager owns the render resolution. It is optional so the
    // visualizer still runs if perf.js fails to load.
    const perf = window.PianoscopePerf
        ? window.PianoscopePerf.attach(gl, canvas, () => resizeCanvas())
        : null;

    /**
     * Size the drawing buffer.
     *
     * The CSS box stays put and the compositor upscales for us, so this is the
     * cheapest possible quality dial: shader cost is linear in fragment count,
     * and iResolution follows the buffer so resolution-dependent shaders get
     * cheaper too. Rendering at native DPR on a 4K Retina panel is ~8.3M
     * fragments of raymarching per frame, which is what makes heavy shaders lag.
     */
    function resizeCanvas() {
        const displayWidth = canvas.clientWidth;
        const displayHeight = canvas.clientHeight;
        if (displayWidth === 0 || displayHeight === 0) return;

        let bufferWidth;
        let bufferHeight;
        if (perf) {
            const size = perf.bufferSize(displayWidth, displayHeight);
            bufferWidth = size.width;
            bufferHeight = size.height;
        } else {
            const ratio = Math.min(window.devicePixelRatio || 1, 1.75);
            bufferWidth = Math.max(1, Math.round(displayWidth * ratio));
            bufferHeight = Math.max(1, Math.round(displayHeight * ratio));
            const maxPx = 2560 * 1440;
            if (bufferWidth * bufferHeight > maxPx) {
                const shrink = Math.sqrt(maxPx / (bufferWidth * bufferHeight));
                bufferWidth = Math.max(1, Math.round(bufferWidth * shrink));
                bufferHeight = Math.max(1, Math.round(bufferHeight * shrink));
            }
        }

        if (canvas.width !== bufferWidth || canvas.height !== bufferHeight) {
            canvas.width = bufferWidth;
            canvas.height = bufferHeight;

            if (usesFeedbackPass) {
                setupFeedbackBuffers(canvas.width, canvas.height);
            }
        }

        gl.viewport(0, 0, canvas.width, canvas.height);
        if (window.PianoscopeDisplayText && window.PianoscopeDisplayText.syncOverlay) {
            window.PianoscopeDisplayText.syncOverlay();
        }
    }

    // Reallocating the drawing buffer is a heavyweight GPU operation, and a
    // window drag fires resize observations continuously. Coalesce them onto a
    // frame boundary so a drag costs one reallocation, not dozens.
    let resizePending = true;
    function requestResize() {
        resizePending = true;
        if (perf) perf.invalidate();
    }

    window.addEventListener('resize', requestResize);
    if (typeof ResizeObserver !== 'undefined') {
        new ResizeObserver(requestResize).observe(canvas);
    }
    resizeCanvas();
    
    // === SHADER MANAGEMENT ===
    let currentProgram = null;
    let currentVertexShader = null;
    let currentFragmentShader = null;
    let frameCount = 0;
    let lastTime = 0;
    let deltaTime = 0;
    let uniforms = {};
    
    // Lets us poll link completion instead of blocking on it. Broadly available
    // in Chrome, Edge and Safari; Firefox is the notable gap and takes the
    // one-frame deferral fallback in pollShaderProgram below.
    const parallelCompileExt = gl.getExtension('KHR_parallel_shader_compile');

    // The quad never changes, so its buffer and attribute state are configured
    // once here and then left alone. Every program binds `position` to location 0
    // before linking, which avoids the expensive desktop-GL emulation path for a
    // disabled attrib 0 and stops ANGLE from re-specializing a program the first
    // time it meets a vertex layout it did not see at link time.
    const quadVao = isWebGL2 ? gl.createVertexArray() : null;
    if (quadVao) gl.bindVertexArray(quadVao);

    const quadBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,
         1.0,  1.0
    ]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    function createShader(gl, type, source) {
        const shader = gl.createShader(type);
        const isWebGL2 = gl instanceof WebGL2RenderingContext;

        const prefix = isWebGL2
            ? "#version 300 es\nprecision highp float;\n" +
                (type === gl.FRAGMENT_SHADER ? "out vec4 fragColor;\n" : "") +
                "#define texture2D texture\n" +
                (type === gl.VERTEX_SHADER ? "#define attribute in\n" : "")
            : "#version 100\nprecision highp float;\n#define fragColor gl_FragColor\n";

        // For fragment shaders, remap the output
        if (type === gl.FRAGMENT_SHADER && isWebGL2) {
            source = source.replace(/gl_FragColor/g, 'fragColor');
        }

        const fullSource = prefix + source;

        gl.shaderSource(shader, fullSource);
        gl.compileShader(shader);

        // Deliberately no COMPILE_STATUS query. It forces a synchronous
        // round-trip to the GPU process and defeats ANGLE's compile cache (which
        // does not store compile status, so asking makes it compile again just to
        // answer). Nearly every real error surfaces at link time instead, and the
        // info logs are still read there if linking fails.
        return shader;
    }

    // A build in flight. The previous program keeps rendering until this one is
    // ready, so switching shaders never shows a black frame or blocks the loop.
    let pendingProgram = null;
    let currentShaderLabel = 'default';

    function destroyBuild(build) {
        gl.deleteProgram(build.program);
        gl.deleteShader(build.vertexShader);
        gl.deleteShader(build.fragmentShader);
    }

    /**
     * Kick off a shader switch. Returns immediately without touching the new
     * program again, because any query against a program that is still linking
     * (including getUniformLocation) stalls until it finishes.
     *
     * @param {Function} [onDone] called with (success, detail) once the result is
     *   known, which is at least one frame later.
     * @returns {boolean} whether the request was accepted, not whether it linked.
     */
    function queueFragmentShader(fragShaderSource, label) {
        if (label) currentShaderLabel = label;
        return requestShaderProgram(vertexShaderSource, fragShaderSource);
    }

    function requestShaderProgram(vertexSource, fragmentSource, onDone) {
        // Holding down the arrow key can easily outrun compilation, and only the
        // most recent request is of any interest.
        if (pendingProgram) {
            const superseded = pendingProgram.onDone;
            destroyBuild(pendingProgram);
            pendingProgram = null;
            if (superseded) superseded(false, 'superseded');
        }

        const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexSource);
        const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentSource);

        const program = gl.createProgram();
        gl.attachShader(program, vertexShader);
        gl.attachShader(program, fragmentShader);
        gl.bindAttribLocation(program, 0, 'position');
        gl.linkProgram(program);

        pendingProgram = {
            program,
            vertexShader,
            fragmentShader,
            vertexSource,
            fragmentSource,
            onDone,
            startMs: performance.now()
        };

        return true;
    }

    function adoptProgram(build, compileMs) {
        const previousProgram = currentProgram;
        const previousVertexShader = currentVertexShader;
        const previousFragmentShader = currentFragmentShader;

        currentProgram = build.program;
        currentVertexShader = build.vertexShader;
        currentFragmentShader = build.fragmentShader;

        window.currentVertexShader = currentVertexShader;
        window.currentFragmentShader = currentFragmentShader;
        window.currentVertexShaderSource = build.vertexSource;
        window.currentFragmentShaderSource = build.fragmentSource;

        gl.useProgram(currentProgram);
        uniforms = getShaderUniforms();
        configureFeedbackPass(build.fragmentSource);
        frameCount = 0;

        // Linking is not the last of the driver's work: the first draw with a
        // program can trigger ANGLE specialization and, on WebKit, a walk of every
        // active uniform to cache its location. A one-pixel scissored draw pays
        // that cost here rather than in the first full-resolution frame.
        gl.enable(gl.SCISSOR_TEST);
        gl.scissor(0, 0, 1, 1);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        gl.disable(gl.SCISSOR_TEST);

        if (previousProgram) gl.deleteProgram(previousProgram);
        if (previousVertexShader) gl.deleteShader(previousVertexShader);
        if (previousFragmentShader) gl.deleteShader(previousFragmentShader);

        if (perf) perf.noteShader(currentShaderLabel, compileMs, build.fragmentSource);

        if (window.kioskWindow && !window.kioskWindow.closed) {
            window.kioskWindow.postMessage({
                type: 'shader-update',
                vertexShader: build.vertexSource,
                fragmentShader: build.fragmentSource
            }, '*');
        }
    }

    /** Called once per frame, before drawing. Never blocks. */
    function pollShaderProgram() {
        if (!pendingProgram) return;

        if (parallelCompileExt) {
            const done = gl.getProgramParameter(pendingProgram.program, parallelCompileExt.COMPLETION_STATUS_KHR);
            if (!done) return;
        } else if (performance.now() - pendingProgram.startMs < 24) {
            // Without the extension, LINK_STATUS blocks. Giving the driver a frame
            // of wall clock to finish in the background shortens the stall even
            // though it cannot remove it.
            return;
        }

        const build = pendingProgram;
        pendingProgram = null;

        const compileMs = performance.now() - build.startMs;

        if (!gl.getProgramParameter(build.program, gl.LINK_STATUS)) {
            const detail = gl.getShaderInfoLog(build.fragmentShader)
                || gl.getShaderInfoLog(build.vertexShader)
                || gl.getProgramInfoLog(build.program)
                || 'unknown link failure';
            console.error('Shader link error:', detail);
            if (statusEl) statusEl.textContent = 'Shader Error: ' + detail.substring(0, 100) + '...';
            destroyBuild(build);
            if (build.onDone) build.onDone(false, detail);
            return;
        }

        adoptProgram(build, compileMs);
        if (build.onDone) build.onDone(true);
    }

    
    // Get all shader uniforms to be updated each frame
    function getShaderUniforms() {
        if (!currentProgram) return {};
        
        return {
            iResolution: gl.getUniformLocation(currentProgram, 'iResolution'),
            iTime: gl.getUniformLocation(currentProgram, 'iTime'),
            iTimeDelta: gl.getUniformLocation(currentProgram, 'iTimeDelta'),
            iFrame: gl.getUniformLocation(currentProgram, 'iFrame'),
            iMouse: gl.getUniformLocation(currentProgram, 'iMouse'),
            iChannel0: gl.getUniformLocation(currentProgram, 'iChannel0'),
            iChannel1: gl.getUniformLocation(currentProgram, 'iChannel1'),
            iChannel2: gl.getUniformLocation(currentProgram, 'iChannel2'),
            iChannel3: gl.getUniformLocation(currentProgram, 'iChannel3'),
            iChannelResolution: gl.getUniformLocation(currentProgram, 'iChannelResolution[0]'),
            iDate: gl.getUniformLocation(currentProgram, 'iDate'),
            iSampleRate: gl.getUniformLocation(currentProgram, 'iSampleRate'),
            iWriteFeedback: gl.getUniformLocation(currentProgram, 'iWriteFeedback'),
            iFeedback: gl.getUniformLocation(currentProgram, 'iFeedback'),
            iVideoResolution: gl.getUniformLocation(currentProgram, 'iVideoResolution'),
            iFitMode: gl.getUniformLocation(currentProgram, 'iFitMode'),
            iVideoMirror: gl.getUniformLocation(currentProgram, 'iVideoMirror'),
            iTextMorph: gl.getUniformLocation(currentProgram, 'iTextMorph'),
            iTextLineH: gl.getUniformLocation(currentProgram, 'iTextLineH'),
            iTextExtent: gl.getUniformLocation(currentProgram, 'iTextExtent'),
            iTextExtentB: gl.getUniformLocation(currentProgram, 'iTextExtentB'),
            iTextLegacyExtent: gl.getUniformLocation(currentProgram, 'iTextLegacyExtent'),
            iTextTransition: gl.getUniformLocation(currentProgram, 'iTextTransition'),
            iTextSheet: gl.getUniformLocation(currentProgram, 'iTextSheet'),
            iTextSheetRes: gl.getUniformLocation(currentProgram, 'iTextSheetRes'),
            iTextWords: gl.getUniformLocation(currentProgram, 'iTextWords[0]'),
            iTextWordCount: gl.getUniformLocation(currentProgram, 'iTextWordCount')
        };
    }
    
    // Mouse tracking
    let mouseX = 0;
    let mouseY = 0;
    let mouseDown = 0;
    let clickX = 0;
    let clickY = 0;
    
    canvas.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        mouseX = e.clientX - rect.left;
        mouseY = rect.height - (e.clientY - rect.top); // Flip Y for WebGL
    });
    
    canvas.addEventListener('mousedown', (e) => {
        mouseDown = 1;
        clickX = mouseX;
        clickY = mouseY;
    });
    
    canvas.addEventListener('mouseup', () => {
        mouseDown = 0;
    });
    
    canvas.addEventListener('mouseleave', () => {
        // Keep the last click position but indicate no mouse down
        mouseDown = 0;
    });
    
    // Touch support
    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        mouseDown = 1;
        if (e.touches.length > 0) {
            const rect = canvas.getBoundingClientRect();
            mouseX = e.touches[0].clientX - rect.left;
            mouseY = rect.height - (e.touches[0].clientY - rect.top);
            clickX = mouseX;
            clickY = mouseY;
        }
    });
    
    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        if (e.touches.length > 0) {
            const rect = canvas.getBoundingClientRect();
            mouseX = e.touches[0].clientX - rect.left;
            mouseY = rect.height - (e.touches[0].clientY - rect.top);
        }
    });
    
    canvas.addEventListener('touchend', (e) => {
        e.preventDefault();
        mouseDown = 0;
    });
    
    // Create audio texture
    const audioTexWidth = 256;

    // Shadertoy's iChannelResolution[4]; channel 0 is rewritten per frame, the
    // rest stay 1x1 because they hold placeholder textures.
    const channelResolutions = new Float32Array([
        audioTexWidth, 1, 1,
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
    ]);

    const audioTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, audioTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    
    // Create audio data array
    audioData = new Uint8Array(audioTexWidth);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, audioTexWidth, 1, 0, gl.LUMINANCE, gl.UNSIGNED_BYTE, audioData);
    
    // Initialize with default shader. Result arrives via pollShaderProgram on a
    // later frame; the loop simply draws nothing until then.
    requestShaderProgram(vertexShaderSource, fragmentShaderSource, (ok, detail) => {
        if (!ok && detail !== 'superseded') {
            if (statusEl) statusEl.textContent = 'Failed to initialize default shader';
            console.error('Failed to initialize default shader:', detail);
        }
    });
    
    // Create additional channels/textures (initialize with null data)
    const dummyTexture = new Uint8Array([0, 0, 0, 255]);
    const extraTextures = [];
    for (let i = 1; i < 4; i++) {
        const texture = gl.createTexture();
        gl.activeTexture(gl.TEXTURE0 + i);
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, dummyTexture);
        extraTextures.push(texture);
    }

    if (window.PianoscopeDisplayText) {
        window.PianoscopeDisplayText.init(gl);
    }

    function displayTextTexture() {
        if (window.PianoscopeDisplayText && window.PianoscopeDisplayText.getTexture()) {
            return window.PianoscopeDisplayText.getTexture();
        }
        return extraTextures[2];
    }

    /**
     * The fitted sheet lives on its own texture unit so iChannel3 can keep the
     * legacy layout the existing text shaders were tuned against.
     */
    function bindFittedTextSheet() {
        if (!uniforms.iTextSheet) return;
        const dt = window.PianoscopeDisplayText;
        const tex = dt && dt.getFittedTexture ? dt.getFittedTexture() : null;
        if (!tex) return;
        gl.activeTexture(gl.TEXTURE4);
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.uniform1i(uniforms.iTextSheet, 4);
    }

    function bindDisplayTextChannel() {
        gl.activeTexture(gl.TEXTURE3);
        gl.bindTexture(gl.TEXTURE_2D, displayTextTexture());
        if (uniforms.iChannel3) gl.uniform1i(uniforms.iChannel3, 3);
    }

    function setupFeedbackBuffers(width, height) {
        const wanted = feedbackWantsHDR ? 'hdr' : 'ldr';
        if (feedbackState && feedbackState.width === width
            && feedbackState.height === height && feedbackState.request === wanted) {
            return;
        }

        destroyFeedbackState(gl, feedbackState);

        const desc = feedbackFormat(gl, isWebGL2, feedbackWantsHDR);
        let state = createFeedbackState(gl, isWebGL2, width, height, desc);
        if (!state && desc.label !== 'rgba8') {
            console.warn('Feedback buffer: ' + desc.label + ' is not renderable here, using rgba8');
            state = createFeedbackState(gl, isWebGL2, width, height,
                feedbackFormat(gl, isWebGL2, false));
        }

        feedbackState = state;
        if (!feedbackState) return;

        feedbackState.request = wanted;
        // New storage always starts undefined, resize included, so this clear is
        // what keeps an accumulator from reading garbage on its first frame.
        clearFeedbackTextures(gl, feedbackState);
    }

    function configureFeedbackPass(fragmentSource) {
        usesFeedbackPass = fragmentSource.includes('iWriteFeedback');
        feedbackWantsHDR = fragmentSource.includes('FEEDBACK_HDR');
        iResolutionIsVec2 = /uniform\s+vec2\s+iResolution/.test(fragmentSource);
        if (usesFeedbackPass) {
            if (!noiseTexture) {
                noiseTexture = createNoiseTexture(gl);
            }
            setupFeedbackBuffers(canvas.width, canvas.height);
        } else {
            destroyFeedbackState(gl, feedbackState);
            feedbackState = null;
        }
    }

    // The previous frame also has a channel of its own, off the iChannel0..3 run.
    // Video mode needs iChannel1 for the frame, so a feedback effect there has no
    // spare channel to borrow; unit 5 is free in every mode.
    function bindFeedbackChannel(feedbackReadTexture) {
        if (!uniforms.iFeedback) return;
        gl.activeTexture(gl.TEXTURE5);
        gl.bindTexture(gl.TEXTURE_2D, feedbackReadTexture || extraTextures[0]);
        gl.uniform1i(uniforms.iFeedback, 5);
    }

    function bindTextureChannels(isVideoMode, feedbackReadTexture) {
        // Shader-mode feedback keeps its original layout: noise on 0, the previous
        // frame on 1, audio on 2.
        if (usesFeedbackPass && !isVideoMode && noiseTexture) {
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, noiseTexture);
            if (uniforms.iChannel0) gl.uniform1i(uniforms.iChannel0, 0);

            gl.activeTexture(gl.TEXTURE1);
            gl.bindTexture(gl.TEXTURE_2D, feedbackReadTexture || extraTextures[0]);
            if (uniforms.iChannel1) gl.uniform1i(uniforms.iChannel1, 1);

            gl.activeTexture(gl.TEXTURE2);
            gl.bindTexture(gl.TEXTURE_2D, audioTexture);
            if (uniforms.iChannel2) gl.uniform1i(uniforms.iChannel2, 2);

            bindFeedbackChannel(feedbackReadTexture);
            bindDisplayTextChannel();
        bindFittedTextSheet();
            return;
        }

        // Audio keeps iChannel0 even in video mode so effects stay music-reactive;
        // the frame goes to iChannel1, the channel left spare outside feedback.
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, audioTexture);
        if (uniforms.iChannel0) gl.uniform1i(uniforms.iChannel0, 0);

        if (isVideoMode && videoController && uniforms.iChannel1) {
            gl.activeTexture(gl.TEXTURE1);
            gl.bindTexture(gl.TEXTURE_2D, videoController.getVideoTexture());
            gl.uniform1i(uniforms.iChannel1, 1);
        }

        for (let i = isVideoMode ? 2 : 1; i < 3; i++) {
            const channelUniform = uniforms['iChannel' + i];
            if (channelUniform) {
                gl.activeTexture(gl.TEXTURE0 + i);
                gl.bindTexture(gl.TEXTURE_2D, extraTextures[i - 1]);
                gl.uniform1i(channelUniform, i);
            }
        }

        bindFeedbackChannel(feedbackReadTexture);
        bindDisplayTextChannel();
        bindFittedTextSheet();
    }

    function setShaderUniforms(currentTime, isVideoMode) {
        if (uniforms.iResolution) {
            if (iResolutionIsVec2) {
                gl.uniform2f(uniforms.iResolution, canvas.width, canvas.height);
            } else {
                gl.uniform3f(uniforms.iResolution, canvas.width, canvas.height, 1.0);
            }
        }

        // Shaders need the real size of whatever feeds each channel so they can
        // sample it without distortion.
        const videoSize = videoController ? videoController.getVideoSize() : { width: 1, height: 1 };

        if (uniforms.iVideoResolution) {
            gl.uniform2f(uniforms.iVideoResolution, videoSize.width, videoSize.height);
        }
        if (uniforms.iFitMode && videoController) {
            gl.uniform1f(uniforms.iFitMode, videoController.getFitMode());
        }
        if (uniforms.iVideoMirror && videoController) {
            gl.uniform1f(uniforms.iVideoMirror, videoController.getMirror());
        }
        if (uniforms.iChannelResolution) {
            channelResolutions[3] = isVideoMode ? videoSize.width : 1;
            channelResolutions[4] = isVideoMode ? videoSize.height : 1;
            const textSize = window.PianoscopeDisplayText
                ? window.PianoscopeDisplayText.getSize()
                : { width: 1, height: 1 };
            channelResolutions[9] = textSize.width;
            channelResolutions[10] = textSize.height;
            channelResolutions[11] = 1;
            gl.uniform3fv(uniforms.iChannelResolution, channelResolutions);
        }
        if (uniforms.iTextMorph || uniforms.iTextLineH || uniforms.iTextExtent || uniforms.iTextTransition) {
            const dt = window.PianoscopeDisplayText;
            if (uniforms.iTextMorph) {
                gl.uniform1f(uniforms.iTextMorph, dt ? dt.getMorph() : 0);
            }
            if (uniforms.iTextLineH) {
                gl.uniform1f(uniforms.iTextLineH, dt ? dt.getLineFrac() : 0.5);
            }
            if (uniforms.iTextExtent) {
                const ext = dt ? dt.getExtent() : { width: 0.5, height: 0.5 };
                gl.uniform2f(uniforms.iTextExtent, ext.width, ext.height);
            }
            if (uniforms.iTextExtentB) {
                const extB = dt && dt.getExtentB
                    ? dt.getExtentB()
                    : (dt ? dt.getExtent() : { width: 0.5, height: 0.5 });
                gl.uniform2f(uniforms.iTextExtentB, extB.width, extB.height);
            }
            if (uniforms.iTextLegacyExtent) {
                const leg = dt && dt.getLegacyExtent
                    ? dt.getLegacyExtent()
                    : { width: 0.386, height: 0.670 };
                gl.uniform2f(uniforms.iTextLegacyExtent, leg.width, leg.height);
            }
            if (uniforms.iTextTransition) {
                gl.uniform1f(uniforms.iTextTransition, dt ? dt.getTransitionIndex() : 0);
            }
            if (uniforms.iTextSheetRes) {
                const fs = dt ? dt.getFittedSize() : { width: 1, height: 1 };
                gl.uniform2f(uniforms.iTextSheetRes, fs.width, fs.height);
            }
            if (uniforms.iTextWords && dt) {
                gl.uniform4fv(uniforms.iTextWords, dt.getWordRects());
            }
            if (uniforms.iTextWordCount) {
                gl.uniform1f(uniforms.iTextWordCount, dt ? dt.getWordCount() : 0);
            }
        }
        if (uniforms.iTime) gl.uniform1f(uniforms.iTime, currentTime);
        if (uniforms.iTimeDelta) gl.uniform1f(uniforms.iTimeDelta, deltaTime);
        if (uniforms.iFrame) gl.uniform1f(uniforms.iFrame, frameCount);
        if (uniforms.iMouse) gl.uniform4f(uniforms.iMouse, mouseX, mouseY, mouseDown, clickX);

        if (uniforms.iDate) {
            const d = new Date();
            gl.uniform4f(uniforms.iDate,
                d.getFullYear(),
                d.getMonth(),
                d.getDate(),
                d.getHours() * 3600 + d.getMinutes() * 60 + d.getSeconds() + d.getMilliseconds() / 1000
            );
        }

        if (uniforms.iSampleRate && audioContext) {
            gl.uniform1f(uniforms.iSampleRate, audioContext.sampleRate);
        }
    }
    
    // === AUDIO FUNCTIONS ===
    async function startMic() {
        try {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            const source = audioContext.createMediaStreamSource(micStream);
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 1024;
            analyser.smoothingTimeConstant = audioSmoothingFactor;
            
            source.connect(analyser);
            
            const bufferLength = analyser.frequencyBinCount;
            frequencyArray = new Uint8Array(bufferLength);
            isListening = true;
            
            if (statusEl) statusEl.textContent = 'Listening to audio';
            if (h) h.textContent = 'Listening...';
            micToggle.innerHTML = '<i class="fas fa-microphone-slash"></i><span>Stop Listen</span>';
            micToggle.className = "red-button";

            // Start animation if not already running
            if (!animationRunning) {
                startTime = performance.now() / 1000;
                lastTime = startTime;
                animationRunning = true;
                requestAnimationFrame(render);
            }
            
        } catch (err) {
            if (statusEl) statusEl.textContent = 'Mic Error: ' + err.message;
            console.error('Error accessing microphone', err);
        }
    }
    
    function stopMic() {
        if (micStream) {
            micStream.getTracks().forEach(track => track.stop());
        }
        if (audioContext) {
            audioContext.close();
        }
        isListening = false;
        micToggle.innerHTML = '<i class="fas fa-microphone"></i><span>Start Listen</span>';
        micToggle.className = "green-button";
        if (statusEl) statusEl.textContent = 'Audio stopped';
        if (h) h.textContent = 'Please allow the use of your microphone.';
        if (hSub) hSub.textContent = '';
        
        // Clear audio data
        audioData.fill(0);
        currentDb = 0;
        uploadAudioTexture();
    }

    /**
     * Read the analyser once per frame.
     *
     * Previously the FFT was read from inside the texture-binding helper, which
     * runs twice per frame for feedback shaders, and again on a separate 500ms
     * timer for the dB readout. One read per frame feeds every consumer.
     */
    function sampleAudio() {
        if (!isListening || !analyser) return;

        analyser.getByteFrequencyData(frequencyArray);

        const count = Math.min(audioData.length, frequencyArray.length);
        let sumOfSquares = 0;

        for (let i = 0; i < count; i++) {
            const raw = frequencyArray[i];
            sumOfSquares += raw * raw;
            const boosted = raw * audioSensitivity;
            audioData[i] = boosted > 255 ? 255 : boosted;
        }
        for (let i = count; i < audioData.length; i++) {
            audioData[i] = 0;
        }

        const rms = Math.sqrt(sumOfSquares / count);
        currentDb = rms > 0 ? Math.max(20 * Math.log10(rms), 0) : 0;
    }

    function uploadAudioTexture() {
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, audioTexture);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, audioTexWidth, 1, gl.LUMINANCE, gl.UNSIGNED_BYTE, audioData);
        window.audioData = audioData;
    }

    // The readout markup is built once so each update is a single text-node
    // mutation rather than an innerHTML reparse. Other code writes plain text into
    // the same heading, so the structure is rebuilt whenever it goes missing.
    let dbTextNode = null;

    function ensureVolumeReadout() {
        if (!h || (dbTextNode && dbTextNode.parentNode === h)) return;

        const unit = document.createElement('span');
        unit.style.color = '#FFD700';
        unit.textContent = ' dB';

        dbTextNode = document.createTextNode('0');
        h.textContent = '';
        h.appendChild(dbTextNode);
        h.appendChild(unit);
        lastRenderedDb = -1;
    }

    /**
     * Writing to the DOM invalidates style and layout, so this is throttled hard
     * and driven from the render loop: the 500ms setTimeout it replaces landed in
     * the middle of arbitrary frames and showed up as a periodic hitch.
     */
    function updateVolumeReadout(nowMs) {
        if (!isListening || nowMs < nextVolumeReadoutMs) return;
        nextVolumeReadoutMs = nowMs + 250;

        const dbValue = Math.floor(currentDb);
        if (dbValue !== lastRenderedDb) {
            ensureVolumeReadout();
            if (dbTextNode) {
                lastRenderedDb = dbValue;
                dbTextNode.nodeValue = String(dbValue);
            }
        }

        if (currentDb >= loud_volume_threshold) {
            seconds += 0.25;
        } else if (seconds !== 0) {
            seconds = 0;
            if (hSub && hSub.textContent) hSub.textContent = '';
        }
    }
    
    // === RENDER LOOP ===
    let startTime = performance.now() / 1000;
    let animationRunning = false;
    
    // Modified render function with video support
    function render(now) {
        if (!animationRunning) return;
        
        requestAnimationFrame(render);

        if (perf && !perf.beginFrame(now)) return;

        // Calculate time values
        const nowSec = now / 1000;
        const currentTime = nowSec - startTime;
        deltaTime = nowSec - lastTime;
        lastTime = nowSec;
        frameCount++;

        // Promote a finished shader build before doing anything else, so the rest
        // of the frame draws with a single consistent program.
        pollShaderProgram();

        if (resizePending || (perf && perf.takeResolutionDirty())) {
            resizePending = false;
            resizeCanvas();
        }

        sampleAudio();
        updateVolumeReadout(now);

        // Check if we're in video mode
        const isVideoMode = videoController && videoController.isInVideoMode && videoController.isInVideoMode();

        // Video mode swaps in an effect program. Adopting a new program deletes
        // the one it replaces, so the previous shader is restored by recompiling
        // its source rather than by reusing a freed program handle. The flags flip
        // as soon as the request is queued, otherwise every frame until the
        // compile lands would queue another identical request.
        if (isVideoMode) {
            const needsVideoProgram = !window.currentShaderIsVideoDisplay || videoController.isEffectDirty();
            if (needsVideoProgram && !pendingProgram) {
                if (!window.currentShaderIsVideoDisplay) {
                    savedFragmentSource = window.currentFragmentShaderSource;
                }
                window.currentShaderIsVideoDisplay = true;
                videoController.clearEffectDirty();
                requestShaderProgram(vertexShaderSource, videoController.getVideoDisplayShader());
            }
        } else if (window.currentShaderIsVideoDisplay && !pendingProgram) {
            window.currentShaderIsVideoDisplay = false;
            requestShaderProgram(vertexShaderSource, savedFragmentSource || fragmentShaderSource);
        }
        
        // Set uniform values for shader if program exists
        if (!currentProgram) {
            if (perf) perf.endFrame();
            return;
        }

        // Texture uploads temporarily switch programs and framebuffers inside the
        // driver, so they belong before the draw setup rather than between state
        // changes where they would flush the pipeline.
        if (isVideoMode) videoController.updateVideoTexture();
        uploadAudioTexture();

        setShaderUniforms(currentTime, isVideoMode);

        if (perf) perf.beginGpuWork();

        if (usesFeedbackPass && feedbackState) {
            const readTexture = feedbackState.textures[feedbackState.readIndex];
            const writeIndex = 1 - feedbackState.readIndex;

            bindTextureChannels(isVideoMode, readTexture);

            gl.bindFramebuffer(gl.FRAMEBUFFER, feedbackState.fbos[writeIndex]);
            gl.viewport(0, 0, feedbackState.width, feedbackState.height);
            if (uniforms.iWriteFeedback) gl.uniform1f(uniforms.iWriteFeedback, 1.0);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            gl.viewport(0, 0, canvas.width, canvas.height);
            if (uniforms.iWriteFeedback) gl.uniform1f(uniforms.iWriteFeedback, 0.0);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

            feedbackState.readIndex = writeIndex;
        } else {
            bindTextureChannels(isVideoMode);
            if (uniforms.iWriteFeedback) gl.uniform1f(uniforms.iWriteFeedback, 0.0);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        }

        if (perf) {
            perf.endGpuWork();
            perf.endFrame();
        }
    }
    
    function ensureRenderLoop() {
        if (animationRunning) return;
        animationRunning = true;
        lastTime = performance.now() / 1000;
        requestAnimationFrame(render);
    }

    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            animationRunning = false;
        } else {
            ensureRenderLoop();
        }
    });

    startTime = performance.now() / 1000;
    lastTime = startTime;
    ensureRenderLoop();
    
    // === EVENT HANDLERS ===
    micToggle.addEventListener('click', () => {
        if (isListening) {
            stopMic();
        } else {
            startMic();
        }
    });
    
    if (fullscreenBtn) {
        fullscreenBtn.addEventListener('click', () => {
            if (canvas.requestFullscreen) {
                canvas.requestFullscreen();
            } else if (canvas.webkitRequestFullscreen) {
                canvas.webkitRequestFullscreen();
            } else if (canvas.mozRequestFullScreen) {
                canvas.mozRequestFullScreen();
            } else if (canvas.msRequestFullscreen) {
                canvas.msRequestFullscreen();
            }
        });
    }

// === KEYBOARD SHADER NAVIGATION ===
// -1 = default shader (Pianoscope Text), outside the numbered cycle
const DEFAULT_SHADER_NAME = 'Pianoscope Text';
// Pinned slots: shader 2, 3, 4 (0-based indices 1, 2, 3)
const PINNED_SHADER_SLOTS = [
    null,
    'PIANOSCOPE Adinkra Symbol Field Abstract',
    'Sunset on river -  Raph',
    'Ocean Waves',
];
let currentShaderIndex = -1;
let shaderKeys = [];

function orderShaderKeys(allKeys) {
    const ordered = [];
    const used = new Set();

    for (let i = 1; i < PINNED_SHADER_SLOTS.length; i++) {
        const name = PINNED_SHADER_SLOTS[i];
        if (name && allKeys.includes(name)) {
            ordered[i] = name;
            used.add(name);
        }
    }

    for (const key of allKeys) {
        if (!used.has(key)) {
            ordered[0] = key;
            used.add(key);
            break;
        }
    }

    for (const key of allKeys) {
        if (!used.has(key)) {
            ordered.push(key);
            used.add(key);
        }
    }

    return ordered.filter(Boolean);
}

// Initialize shader navigation when ShaderConverter is available
if (window.ShaderConverter && window.ShaderConverter.SAMPLE_SHADERS) {
    shaderKeys = orderShaderKeys(Object.keys(window.ShaderConverter.SAMPLE_SHADERS));
    console.log(`Loaded ${shaderKeys.length} shaders for keyboard navigation`);
}

function logAppliedShader(shaderName) {
    console.log(`Applied shader: ${shaderName}`);
}

function applyDefaultShader() {
    leaveVideoMode();
    if (window.ShaderConverter && window.visualizer) {
        const shaderSource = window.ShaderConverter.SAMPLE_SHADERS[DEFAULT_SHADER_NAME];
        if (shaderSource) {
            try {
                const convertedShader = window.ShaderConverter.convertShaderToyToWebGL(shaderSource);
                const success = window.visualizer.applyShader(convertedShader, DEFAULT_SHADER_NAME);
                if (success) {
                    currentShaderIndex = -1;
                    logAppliedShader(DEFAULT_SHADER_NAME);
                    if (statusEl) statusEl.textContent = `Shader: ${DEFAULT_SHADER_NAME} (default)`;
                    const shaderTextarea = document.querySelector('.shader-editor-textarea');
                    if (shaderTextarea) shaderTextarea.value = shaderSource;
                    return true;
                }
            } catch (error) {
                console.error('Error loading default shader:', error);
            }
        }
    }
    if (queueFragmentShader(fragmentShaderSource, 'default')) {
        currentShaderIndex = -1;
        return true;
    }
    return false;
}

function applyShaderAtIndex(newIndex) {
    if (newIndex === currentShaderIndex) return;

    if (newIndex === -1) {
        if (window.visualizer && window.visualizer.resetToDefault()) {
            currentShaderIndex = -1;
        }
        return;
    }

    const shaderName = shaderKeys[newIndex];
    const shaderSource = window.ShaderConverter.SAMPLE_SHADERS[shaderName];

    if (!shaderSource || !window.visualizer) return;

    currentShaderIndex = newIndex;

    try {
        const convertedShader = window.ShaderConverter.convertShaderToyToWebGL(shaderSource);
        const success = window.visualizer.applyShader(convertedShader, shaderName);

        if (success) {
            logAppliedShader(shaderName);
            if (statusEl) {
                statusEl.textContent = `Shader: ${shaderName} (${currentShaderIndex + 1}/${shaderKeys.length})`;
            }

            const shaderTextarea = document.querySelector('.shader-editor-textarea');
            if (shaderTextarea) {
                shaderTextarea.value = shaderSource;
            }

        } else {
            console.error(`Failed to apply converted shader: ${shaderName}`);
            if (statusEl) {
                statusEl.textContent = `Error applying shader: ${shaderName} (${currentShaderIndex + 1}/${shaderKeys.length}) — press → to skip`;
            }
        }
    } catch (error) {
        console.error(`Error converting shader ${shaderName}:`, error);
        if (statusEl) {
            statusEl.textContent = `Conversion error: ${shaderName} (${currentShaderIndex + 1}/${shaderKeys.length}) — press → to skip`;
        }
    }
}

function getNextShaderIndex(direction) {
    const last = shaderKeys.length - 1;

    if (direction === 'left') {
        if (currentShaderIndex === -1) return last;
        if (currentShaderIndex === 0) return -1;
        return currentShaderIndex - 1;
    }

    if (currentShaderIndex === -1) return 0;
    if (currentShaderIndex === last) return -1;
    return currentShaderIndex + 1;
}

// Keyboard event listener for shader cycling
document.addEventListener('keydown', (e) => {
    if (shaderKeys.length === 0) return;

    const active = document.activeElement;
    if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA' || active.isContentEditable)) {
        return;
    }

    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        e.preventDefault();
        const direction = e.key === 'ArrowLeft' ? 'left' : 'right';
        applyShaderAtIndex(getNextShaderIndex(direction));
    }
});

// Function to set a specific shader by name (useful for debugging)
window.setShaderByName = (shaderName) => {
    const index = shaderKeys.indexOf(shaderName);
    if (index !== -1) {
        const shaderSource = window.ShaderConverter.SAMPLE_SHADERS[shaderName];
        if (window.visualizer) {
            try {
                const convertedShader = window.ShaderConverter.convertShaderToyToWebGL(shaderSource);
                const success = window.visualizer.applyShader(convertedShader, shaderName);
                if (success) {
                    currentShaderIndex = index;
                    logAppliedShader(shaderName);
                    if (statusEl) {
                        statusEl.textContent = `Shader: ${shaderName} (${currentShaderIndex + 1}/${shaderKeys.length})`;
                    }
                }
                return success;
            } catch (error) {
                console.error(`Error converting shader ${shaderName}:`, error);
                return false;
            }
        }
    }
    console.error(`Shader not found: ${shaderName}`);
    return false;
};

// Function to get current shader info (useful for debugging)
window.getCurrentShaderInfo = () => {
    if (shaderKeys.length === 0) return null;
    if (currentShaderIndex === -1) {
        return { name: DEFAULT_SHADER_NAME, index: -1, total: shaderKeys.length };
    }
    return {
        name: shaderKeys[currentShaderIndex],
        index: currentShaderIndex,
        total: shaderKeys.length
    };
};
    // Initial setup
    gl.clearColor(0.1, 0.1, 0.1, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (statusEl) statusEl.textContent = 'Ready - click Start Listen to begin';
    
    // === SHADERTOY INTEGRATION ===
    // Create the shader test UI if the ShaderConverter is available
    if (window.ShaderConverter) {
        const shaderUI = window.ShaderConverter.createShaderTestUI(document.body, (newShaderSource) => {
            // When user applies a new shader, set it up
            leaveVideoMode();
            if (queueFragmentShader(newShaderSource, 'editor')) {
                if (statusEl) statusEl.textContent = 'New shader applied successfully';
                return true;
            }
            return false;
        });

        // Hand the video tab over to the controller now that the shell exists
        if (videoController) {
            videoController.mountUI(document.getElementById('video-tab'));
        }
        
        // Add helper functions to window for console testing
        window.visualizer = {
            applyShader: (fragShaderSource, label) => {
                leaveVideoMode();
                return queueFragmentShader(fragShaderSource, label || 'custom');
            },
            setDisplayText: (text) => {
                if (window.PianoscopeDisplayText) {
                    return window.PianoscopeDisplayText.applyText(text);
                }
                return '';
            },
            
            loadFromShadertoy: async (shaderCode) => {
                try {
                    leaveVideoMode();
                    const convertedCode = window.ShaderConverter.convertShaderToyToWebGL(shaderCode);
                    if (convertedCode && queueFragmentShader(convertedCode, 'shadertoy')) {
                        if (statusEl) statusEl.textContent = 'Shadertoy shader loaded successfully';
                        return true;
                    }
                } catch (error) {
                    if (statusEl) statusEl.textContent = 'Shader error: ' + error.message;
                    console.error('Shader error:', error);
                }
                return false;
            },
            
            resetToDefault: () => {
                if (applyDefaultShader()) {
                    if (statusEl) statusEl.textContent = 'Restored default shader';
                    return true;
                }
                return false;
            }
        };

        applyDefaultShader();
    } else {
        console.warn('ShaderConverter module not found. Shadertoy integration is disabled.');
    }
});
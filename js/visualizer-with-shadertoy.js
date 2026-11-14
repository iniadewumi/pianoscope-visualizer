/**
 * Piansoscope Visualizer with Shadertoy Support
 * 
 * This script handles the WebGL and audio processing to visualize 
 * microphone input in real-time with shader effects.
 */

document.addEventListener('DOMContentLoaded', () => {
    "use strict";
    // Compiling, the uniform table, feedback buffers and the draw itself are
    // shared with the projector window; see js/gl-shared.js.
    const GLX = window.PianoscopeGL;

    // === CONFIGURATION ===
    // Easy to modify settings
    const audioSmoothingFactor = 0.8; // How smooth the audio response is (0-1)
    const audioSensitivity = 1.5;     // Amplify the audio signal
    
    // === ELEMENTS ===
    const canvas = document.getElementById('visualizer');
    const micToggle = document.getElementById('mic-toggle');
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
    let lastAudioSampleMs = -Infinity;
    
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
    // WebGL 2 only. Every shader is compiled as GLSL ES 3.00; the WebGL 1
    // fallback that used to live here could not compile a single library
    // shader, so all it did was turn a clear message into a blank canvas.
    const gl = canvas.getContext('webgl2', GLX.CONTEXT_ATTRIBUTES);
    if (!gl) {
        const message = 'This visualizer needs WebGL 2, which this browser or GPU does not provide.';
        if (h) h.textContent = message;
        if (statusEl) statusEl.textContent = message;
        micToggle.disabled = true;
        console.error(message);
        return;
    }

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
            const size = perf.previewBufferSize(displayWidth, displayHeight);
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
                feedbackState = GLX.ensureFeedbackState(
                    gl, feedbackState, canvas.width, canvas.height, feedbackWantsHDR);
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
    // True on render frames where a new source frame was uploaded (iVideoNewFrame).
    let videoFrameFresh = false;
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
    const quadVao = gl.createVertexArray();
    gl.bindVertexArray(quadVao);

    const quadBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, GLX.QUAD_VERTICES, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    // A build in flight. The previous program keeps rendering until this one is
    // ready, so switching shaders never shows a black frame or blocks the loop.
    let pendingProgram = null;
    let currentShaderLabel = 'default';

    /**
     * Kick off a shader switch. Returns immediately without touching the new
     * program again, because any query against a program that is still linking
     * (including getUniformLocation) stalls until it finishes.
     *
     * @param {Function} [onDone] called with (success, detail) once the result is
     *   known, which is at least one frame later.
     * @returns {boolean} whether the request was accepted, not whether it linked.
     */
    function queueFragmentShader(fragShaderSource, label, onDone) {
        if (label) currentShaderLabel = label;
        return requestShaderProgram(vertexShaderSource, fragShaderSource, onDone);
    }

    function requestShaderProgram(vertexSource, fragmentSource, onDone) {
        // Holding down the arrow key can easily outrun compilation, and only the
        // most recent request is of any interest.
        if (pendingProgram) {
            const superseded = pendingProgram.onDone;
            GLX.destroyBuild(gl, pendingProgram);
            pendingProgram = null;
            if (superseded) superseded(false, 'superseded');
        }

        pendingProgram = GLX.startBuild(gl, vertexSource, fragmentSource);
        pendingProgram.onDone = onDone;

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
        uniforms = GLX.locateUniforms(gl, currentProgram);
        configureFeedbackPass(build.fragmentSource);
        frameCount = 0;

        GLX.warmUpDraw(gl);

        if (previousProgram) gl.deleteProgram(previousProgram);
        if (previousVertexShader) gl.deleteShader(previousVertexShader);
        if (previousFragmentShader) gl.deleteShader(previousFragmentShader);

        if (perf) perf.noteShader(currentShaderLabel, compileMs, build.fragmentSource);

        if (window.kioskWindow && !window.kioskWindow.closed) {
            window.kioskWindow.postMessage({
                type: 'shader-update',
                vertexShader: build.vertexSource,
                fragmentShader: build.fragmentSource
            }, window.location.origin);
        }
    }

    /** Called once per frame, before drawing. Never blocks. */
    function pollShaderProgram() {
        if (!pendingProgram) return;

        if (!GLX.buildFinished(gl, parallelCompileExt, pendingProgram)) return;

        const build = pendingProgram;
        pendingProgram = null;

        const compileMs = performance.now() - build.startMs;

        if (!GLX.buildLinked(gl, build)) {
            const detail = GLX.buildLog(gl, build);
            console.error('Shader link error:', detail);
            if (statusEl) {
                const readable = window.ShaderConverter && window.ShaderConverter.formatCompileLog
                    ? window.ShaderConverter.formatCompileLog(detail)[0]
                    : null;
                statusEl.textContent = 'Shader Error: ' + (readable || detail).substring(0, 100);
            }
            GLX.destroyBuild(gl, build);
            if (build.onDone) build.onDone(false, detail);
            return;
        }

        adoptProgram(build, compileMs);
        if (build.onDone) build.onDone(true);
    }


    // Mouse tracking. Positions are kept as fractions of the canvas box (y up)
    // and turned into pixels when iMouse is uploaded. iResolution is in
    // drawing-buffer pixels, which are rarely CSS pixels (DPR cap, adaptive
    // scale, shader bias) and change mid-show, so CSS pixels here made
    // iMouse.xy / iResolution.xy top out around 0.57 on a Retina screen.
    let mouseU = 0;
    let mouseV = 0;
    let clickU = 0;
    let clickV = 0;
    let mouseDown = false;
    // True for the first frame after a press, which is when Shadertoy makes
    // iMouse.w positive.
    let clickFresh = false;
    // The kiosk renders on its own clock and can miss a one-frame flag, so it
    // spots a new press by this count changing instead.
    let pressCount = 0;

    // The kiosk mirrors this pointer so mouse-driven shaders look the same on
    // the projector as in the preview. Fractions of the canvas, which the kiosk
    // scales to its own buffer. One reused object: this is read every frame.
    const pointerState = { u: 0, v: 0, clickU: 0, clickV: 0, down: false, presses: 0 };
    window.pianoscopeReadPointer = () => {
        pointerState.u = mouseU;
        pointerState.v = mouseV;
        pointerState.clickU = clickU;
        pointerState.clickV = clickV;
        pointerState.down = mouseDown;
        pointerState.presses = pressCount;
        return pointerState;
    };

    function trackPointer(clientX, clientY) {
        const rect = canvas.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return;
        mouseU = (clientX - rect.left) / rect.width;
        mouseV = 1 - (clientY - rect.top) / rect.height; // Flip Y for WebGL
    }

    function pressPointer() {
        mouseDown = true;
        clickFresh = true;
        pressCount++;
        clickU = mouseU;
        clickV = mouseV;
    }

    canvas.addEventListener('mousemove', (e) => {
        trackPointer(e.clientX, e.clientY);
    });

    canvas.addEventListener('mousedown', (e) => {
        trackPointer(e.clientX, e.clientY);
        pressPointer();
    });

    canvas.addEventListener('mouseup', () => {
        mouseDown = false;
    });

    canvas.addEventListener('mouseleave', () => {
        // Keep the last click position but indicate no mouse down
        mouseDown = false;
    });

    // Touch support
    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        if (e.touches.length > 0) {
            trackPointer(e.touches[0].clientX, e.touches[0].clientY);
        }
        pressPointer();
    });

    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        if (e.touches.length > 0) {
            trackPointer(e.touches[0].clientX, e.touches[0].clientY);
        }
    });

    canvas.addEventListener('touchend', (e) => {
        e.preventDefault();
        mouseDown = false;
    });
    
    // Create audio texture
    const audioTexWidth = GLX.AUDIO_TEX_WIDTH;

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

    function configureFeedbackPass(fragmentSource) {
        usesFeedbackPass = fragmentSource.includes('iWriteFeedback');
        feedbackWantsHDR = fragmentSource.includes('FEEDBACK_HDR');
        iResolutionIsVec2 = /uniform\s+vec2\s+iResolution/.test(fragmentSource);
        if (usesFeedbackPass && !noiseTexture) {
            noiseTexture = GLX.createNoiseTexture(gl);
        }
        feedbackState = GLX.configureFeedback(
            gl, feedbackState, fragmentSource, canvas.width, canvas.height);
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

    // Every program can sample the live source (camera/screen/clip) through
    // iVideo on unit 6, independent of whether the current look is the video
    // effect. Falls back to a spare black texture when nothing is attached.
    function bindVideoChannel() {
        if (!uniforms.iVideo) return;
        gl.activeTexture(gl.TEXTURE6);
        const hasSrc = videoController && videoController.hasSource();
        gl.bindTexture(gl.TEXTURE_2D, hasSrc ? videoController.getVideoTexture() : extraTextures[0]);
        gl.uniform1i(uniforms.iVideo, 6);
    }

    // The person matte (js/person-mask.js) on unit 8, re-uploaded only when the
    // segmenter has produced a new one. Starts as a 1x1 zero so an unbound read
    // means "no person" rather than whatever unit 0 holds.
    let personTexture = null;
    let personTextureVersion = -1;
    let personActive = false;

    function uploadPersonMask(wanted) {
        const pm = window.PianoscopePersonMask;
        const mask = wanted && pm ? pm.read() : null;
        personActive = !!(mask && mask.active);
        gl.activeTexture(gl.TEXTURE8);
        if (!personTexture) {
            personTexture = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, personTexture);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, 1, 1, 0, gl.RED, gl.UNSIGNED_BYTE, new Uint8Array(1));
        }
        if (!personActive || mask.version === personTextureVersion) return;
        personTextureVersion = mask.version;
        gl.bindTexture(gl.TEXTURE_2D, personTexture);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, mask.width, mask.height, 0,
            gl.RED, gl.UNSIGNED_BYTE, mask.data);
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    }

    function bindPersonChannel() {
        if (!uniforms.iPerson || !personTexture) return;
        gl.activeTexture(gl.TEXTURE8);
        gl.bindTexture(gl.TEXTURE_2D, personTexture);
        gl.uniform1i(uniforms.iPerson, 8);
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
            bindVideoChannel();
            bindPersonChannel();
            return;
        }

        // Audio keeps iChannel0 even in video mode so effects stay music-reactive;
        // the frame goes to iChannel1, the channel left spare outside feedback.
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, audioTexture);
        if (uniforms.iChannel0) gl.uniform1i(uniforms.iChannel0, 0);

        // Keyed on the program actually on screen rather than the requested look:
        // while a library shader is still compiling, the effect program keeps
        // drawing and would otherwise show black until the swap lands.
        const effectOnScreen = window.currentShaderIsVideoDisplay
            && videoController && videoController.hasSource();
        if (effectOnScreen && uniforms.iChannel1) {
            gl.activeTexture(gl.TEXTURE1);
            gl.bindTexture(gl.TEXTURE_2D, videoController.getVideoTexture());
            gl.uniform1i(uniforms.iChannel1, 1);
        }

        for (let i = effectOnScreen ? 2 : 1; i < 3; i++) {
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
        bindVideoChannel();
        bindPersonChannel();
    }

    // This window's half of a frame's uniforms: its clock, buffer and pointer.
    // One reused object, filled in every frame.
    const frameState = {
        width: 0, height: 0, resolutionIsVec2: false,
        time: 0, delta: 0, frame: 0,
        pointer: pointerState, clickFresh: false,
        videoNewFrame: false, personActive: false, sampleRate: 0
    };

    function setShaderUniforms(currentTime) {
        frameState.width = canvas.width;
        frameState.height = canvas.height;
        frameState.resolutionIsVec2 = iResolutionIsVec2;
        frameState.time = currentTime;
        frameState.delta = deltaTime;
        frameState.frame = frameCount;
        window.pianoscopeReadPointer();
        frameState.clickFresh = clickFresh;
        frameState.videoNewFrame = videoFrameFresh;
        frameState.personActive = personActive;
        frameState.sampleRate = audioContext ? audioContext.sampleRate : 0;

        GLX.applyFrameUniforms(gl, uniforms, frameState, videoController,
            window.PianoscopeDisplayText || null);
        clickFresh = false;
    }

    // The projector has no audio context of its own to ask.
    window.pianoscopeReadSampleRate = () => (audioContext ? audioContext.sampleRate : 0);
    
    // === AUDIO FUNCTIONS ===

    // True while getUserMedia is pending. A second click during the permission
    // prompt used to start a second capture: the first stream was orphaned, so
    // the mic light stayed on after Stop.
    let micStarting = false;

    /**
     * A start that is not a user gesture (a scripted click, or a browser that
     * applies its autoplay policy anyway) can bring the context up suspended,
     * and then the analyser reads silence. Try to resume now, and failing that
     * on the next real interaction.
     */
    function resumeAudioWhenAllowed() {
        const context = audioContext;
        if (!context || context.state !== 'suspended') return;
        context.resume().catch(() => {});
        const onGesture = () => {
            if (context.state === 'suspended') context.resume().catch(() => {});
        };
        document.addEventListener('pointerdown', onGesture, { once: true, capture: true });
        document.addEventListener('keydown', onGesture, { once: true, capture: true });
    }

    async function startMic() {
        if (micStarting || isListening) return;
        micStarting = true;
        // Created before the await so it is still inside the click's user
        // activation; a context made later starts suspended.
        const context = new (window.AudioContext || window.webkitAudioContext)();
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioContext = context;
            micStream = stream;

            const source = audioContext.createMediaStreamSource(micStream);
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 1024;
            analyser.smoothingTimeConstant = audioSmoothingFactor;
            
            source.connect(analyser);
            
            const bufferLength = analyser.frequencyBinCount;
            frequencyArray = new Uint8Array(bufferLength);
            isListening = true;
            resumeAudioWhenAllowed();

            if (statusEl) statusEl.textContent = 'Listening to audio';
            if (h) h.textContent = 'Listening...';
            micToggle.innerHTML = '<i class="fas fa-microphone-slash"></i><span>&nbsp;Stop Listen</span>';
            micToggle.className = "red-button";

            // Start animation if not already running
            if (!animationRunning) {
                startTime = performance.now() / 1000;
                lastTime = startTime;
                animationRunning = true;
                requestAnimationFrame(render);
            }
            
        } catch (err) {
            // Never became the live context, so nothing else will close it.
            context.close().catch(() => {});
            if (statusEl) statusEl.textContent = 'Mic Error: ' + err.message;
            console.error('Error accessing microphone', err);
        } finally {
            micStarting = false;
        }
    }

    function stopMic() {
        if (micStream) {
            micStream.getTracks().forEach(track => track.stop());
        }
        if (audioContext) {
            audioContext.close().catch(() => {});
        }
        micStream = null;
        audioContext = null;
        analyser = null;
        isListening = false;
        micToggle.innerHTML = '<i class="fas fa-microphone"></i><span>&nbsp;Start Listen</span>';
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
        lastAudioSampleMs = performance.now();

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

    /**
     * The kiosk pulls the spectrum from its own render loop rather than being
     * sent it. This window's loop stops whenever the window is hidden
     * (minimised, or covered by fullscreen DJ software), and forwarding its
     * samples froze the projector on the last spectrum exactly then. While the
     * loop is running it owns the analyser and the kiosk reuses its latest read:
     * smoothing is applied per read, so reading at both windows' combined rate
     * would change how every shader responds.
     */
    window.pianoscopeReadAudio = () => {
        if (!animationRunning || performance.now() - lastAudioSampleMs > 50) {
            sampleAudio();
        }
        return audioData;
    };

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
    
    // Hoisted so the loop does not allocate a closure per frame.
    let drawIsVideoMode = false;
    const bindChannelsForDraw = (readTexture) => bindTextureChannels(drawIsVideoMode, readTexture);

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

        // Cheap per-frame check; perf only resizes when it flips (see
        // previewScaleWithKiosk in perf.js).
        if (perf) perf.setKioskLive(window.kioskWindow && !window.kioskWindow.closed);

        if (resizePending || (perf && perf.takeResolutionDirty())) {
            resizePending = false;
            resizeCanvas();
        }

        sampleAudio();
        updateVolumeReadout(now);

        // Check if we're in video mode
        const isVideoMode = videoController && videoController.isInVideoMode && videoController.isInVideoMode();
        // Separate from look: is a source (clip/camera/screen) attached at all?
        // Every program can sample it via iVideo regardless of which look is up.
        const hasSource = videoController && videoController.hasSource();

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
        // Upload only when the current program can actually see the frame: the
        // video-effect look always needs it, and any library shader that declares
        // iVideo needs it too, whether or not iVideoActive is on. Every other
        // shader saves the upload cost entirely.
        videoFrameFresh = false;
        if (hasSource && (isVideoMode || window.currentShaderIsVideoDisplay || uniforms.iVideo)) {
            videoFrameFresh = videoController.updateVideoTexture() === true;
        }
        const wantsPerson = !!(videoController && videoController.wantsPersonMask
            && videoController.wantsPersonMask());
        if (wantsPerson) videoController.updatePersonMask(now);
        uploadPersonMask(wantsPerson);
        uploadAudioTexture();

        setShaderUniforms(currentTime);

        if (perf) perf.beginGpuWork();

        drawIsVideoMode = isVideoMode;
        GLX.drawFrame(gl, uniforms, usesFeedbackPass ? feedbackState : null,
            canvas.width, canvas.height, bindChannelsForDraw);

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

// === KEYBOARD SHADER NAVIGATION ===
// -1 = default shader (Pianoscope Text), outside the numbered cycle
const DEFAULT_SHADER_NAME = 'Pianoscope Text';
// Pinned slots: shader 2, 3, 4 (0-based indices 1, 2, 3)
const PINNED_SHADER_SLOTS = [
    null,
    'Procgen Cats'

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

        if (videoController && videoController.isInVideoMode()) {
            videoController.cycleEffect(direction);
            return;
        }

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
        const shaderUI = window.ShaderConverter.createShaderTestUI(document.body, (newShaderSource, onResult) => {
            // Queued now, linked on a later frame: success is only announced
            // once the program has actually linked. A failure has already put
            // the error in the status bar by the time onResult hears of it.
            leaveVideoMode();
            return queueFragmentShader(newShaderSource, 'editor', (ok, detail) => {
                if (ok && statusEl) statusEl.textContent = 'New shader applied successfully';
                if (onResult) onResult(ok, detail);
            });
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

        // Filter edits reload the page. Re-open the camera and last effect so
        // the operator is not dumped back on the default shader each save.
        // Must run *instead of* the default apply — that path persists
        // videoMode:false and would erase the session we are about to restore.
        const restoreVideo = videoController && videoController.hasRestorableSession();
        if (restoreVideo) {
            videoController.restoreSession().then((restored) => {
                // A restored source with look:'shader' still needs a shader on
                // screen — only skip the default apply when the effect look
                // actually took over the program.
                if (!restored || !videoController.isInVideoMode()) applyDefaultShader();
            });
        } else {
            applyDefaultShader();
        }
    } else {
        console.warn('ShaderConverter module not found. Shadertoy integration is disabled.');
    }
});
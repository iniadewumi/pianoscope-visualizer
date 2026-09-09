// WebGL shader prefix templates as separate JS object
// This avoids nesting complex strings in our HTML template
const ShaderPrefixes = {
    // WebGL2 vertex shader prefix
    webgl2VertexPrefix: 
        "#version 300 es\n" +
        "precision highp float;\n" +
        "#define texture2D texture\n" +
        "#define attribute in"+"\n",
    
    // WebGL2 fragment shader prefix
    webgl2FragmentPrefix: 
        "#version 300 es\n" +
        "precision highp float;\n" +
        "out vec4 fragColor;\n" +
        "#define texture2D texture\n" +
        "#define gl_FragColor fragColor\n",
    
    // WebGL1 shader prefix (common for both vertex and fragment)
    webgl1Prefix: 
        "#version 100\n" +
        "precision highp float;\n" +
        "#define fragColor gl_FragColor"+"\n"
};

// Create the kiosk HTML content with reference to our external prefix constants
const kioskHtmlContent = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>Pianoscope Kiosk Mode</title>
    <style>
        html, body {
            width: 100%;
            height: 100%;
            padding: 0;
            margin: 0;
            background-color: #121212;
            overflow: hidden;
            position: relative;
        }
        
        #kiosk-canvas {
            display: block;
            width: 100%;
            height: 100%;
            padding: 0;
            margin: 0;
            image-rendering: auto;
        }

        #kiosk-text-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 2;
        }
    </style>
</head>
<body>
    <canvas id="kiosk-canvas"></canvas>
    <canvas id="kiosk-text-overlay"></canvas>
    
    <script>
        // Will be filled by the parent window
        let gl = null;
        let isWebGL2 = false;
        let currentProgram = null;
        let audioTexture = null;
        let videoTexture = null;
        // Size of the currently allocated video texture storage (WebGL2 texStorage2D
        // path only). 0 means nothing has been allocated yet.
        let videoTexWidth = 0;
        let videoTexHeight = 0;
        let displayTextTexture = null;
        let fittedTextTexture = null;
        let fittedTextW = 1;
        let fittedTextH = 1;
        let lastFittedVersion = -1;
        let audioData = null;
        let animationFrameId = null;
        let startTime = performance.now() / 1000;
        let lastTime = startTime;
        let frameCount = 0;
        let iResolutionIsVec2 = false;
        let usesFeedbackPass = false;
        let feedbackWantsHDR = false;
        let feedbackState = null;
        let hdrFeedbackFormat;
        let displayTextW = 1;
        let displayTextH = 1;
        let lastDisplayTextVersion = -1;
        
        // Shader prefix constants passed from parent window
        let shaderPrefixes = null;
        let kioskUniforms = {};
        let quadBuffer = null;
        
        // Set up the canvas
        const canvas = document.getElementById('kiosk-canvas');
        const overlayCanvas = document.getElementById('kiosk-text-overlay');
        const overlayCtx = overlayCanvas ? overlayCanvas.getContext('2d') : null;
        let lastOverlayVersion = -1;
        const MAX_PIXEL_COUNT = 2560 * 1440;
        const DPR_CAP = 1.75;

        function desiredKioskSize() {
            const displayWidth = canvas.clientWidth;
            const displayHeight = canvas.clientHeight;
            if (displayWidth === 0 || displayHeight === 0) {
                return { width: 1, height: 1 };
            }
            try {
                if (window.opener && !window.opener.closed && window.opener.PianoscopePerf) {
                    return window.opener.PianoscopePerf.bufferSize(displayWidth, displayHeight);
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

        // --- Frame feedback ---------------------------------------------
        // Same contract as the opener: a shader carrying iWriteFeedback is drawn
        // into a ping-pong buffer first and to the canvas second, reading last
        // frame back through iFeedback. Without this the projector window shows
        // black for every accumulating shader while the operator's screen is fine.
        function feedbackFormat(wantsHDR) {
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
            return hdrFeedbackFormat || rgba8;
        }

        function destroyFeedbackState(state) {
            if (!state) return;
            state.fbos.forEach(function (fbo) { gl.deleteFramebuffer(fbo); });
            state.textures.forEach(function (texture) { gl.deleteTexture(texture); });
        }

        function createFeedbackState(width, height, desc) {
            const textures = [gl.createTexture(), gl.createTexture()];
            textures.forEach(function (texture) {
                gl.bindTexture(gl.TEXTURE_2D, texture);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, desc.filter);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, desc.filter);
                if (isWebGL2) {
                    gl.texStorage2D(gl.TEXTURE_2D, 1, desc.internalFormat, width, height);
                } else {
                    gl.texImage2D(gl.TEXTURE_2D, 0, desc.internalFormat, width, height, 0,
                        desc.format, desc.type, null);
                }
            });

            const fbos = [gl.createFramebuffer(), gl.createFramebuffer()];
            let complete = true;
            fbos.forEach(function (fbo, index) {
                gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
                gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, textures[index], 0);
                if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) complete = false;
            });
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);

            if (!complete) {
                destroyFeedbackState({ textures: textures, fbos: fbos });
                return null;
            }

            gl.viewport(0, 0, width, height);
            gl.clearColor(0, 0, 0, 0);
            fbos.forEach(function (fbo) {
                gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
                gl.clear(gl.COLOR_BUFFER_BIT);
            });
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            gl.viewport(0, 0, canvas.width, canvas.height);

            return { textures: textures, fbos: fbos, readIndex: 0, width: width, height: height };
        }

        function setupFeedbackBuffers(width, height) {
            const wanted = feedbackWantsHDR ? 'hdr' : 'ldr';
            if (feedbackState && feedbackState.width === width
                && feedbackState.height === height && feedbackState.request === wanted) {
                return;
            }

            destroyFeedbackState(feedbackState);

            const desc = feedbackFormat(feedbackWantsHDR);
            let state = createFeedbackState(width, height, desc);
            if (!state && desc.label !== 'rgba8') {
                state = createFeedbackState(width, height, feedbackFormat(false));
            }

            feedbackState = state;
            if (feedbackState) feedbackState.request = wanted;
        }

        function configureFeedbackPass(fragmentSource) {
            usesFeedbackPass = fragmentSource.indexOf('iWriteFeedback') !== -1;
            feedbackWantsHDR = fragmentSource.indexOf('FEEDBACK_HDR') !== -1;
            if (usesFeedbackPass) {
                setupFeedbackBuffers(canvas.width, canvas.height);
            } else {
                destroyFeedbackState(feedbackState);
                feedbackState = null;
            }
        }

        function resizeKioskCanvas() {
            const size = desiredKioskSize();
            if (canvas.width === size.width && canvas.height === size.height) {
                if (gl) gl.viewport(0, 0, size.width, size.height);
                return;
            }
            canvas.width = size.width;
            canvas.height = size.height;
            if (gl) {
                gl.viewport(0, 0, size.width, size.height);
                if (usesFeedbackPass) setupFeedbackBuffers(size.width, size.height);
            }
        }
        
        window.addEventListener('resize', resizeKioskCanvas);
        document.addEventListener('fullscreenchange', resizeKioskCanvas);
        if (typeof ResizeObserver !== 'undefined') {
            new ResizeObserver(resizeKioskCanvas).observe(canvas);
        }
        resizeKioskCanvas();
        
        // Double-click for fullscreen
        canvas.addEventListener('dblclick', function() {
            if (document.fullscreenElement) {
                document.exitFullscreen();
            } else {
                if (canvas.requestFullscreen) {
                    canvas.requestFullscreen();
                } else if (canvas.webkitRequestFullscreen) {
                    canvas.webkitRequestFullscreen();
                } else if (canvas.mozRequestFullScreen) {
                    canvas.mozRequestFullScreen();
                } else if (canvas.msRequestFullscreen) {
                    canvas.msRequestFullscreen();
                }
            }
        });
        
        // Handle keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            if (e.key === 'f' || e.key === 'F') {
                if (document.fullscreenElement) {
                    document.exitFullscreen();
                } else {
                    if (canvas.requestFullscreen) {
                        canvas.requestFullscreen();
                    } else if (canvas.webkitRequestFullscreen) {
                        canvas.webkitRequestFullscreen();
                    } else if (canvas.mozRequestFullScreen) {
                        canvas.mozRequestFullScreen();
                    } else if (canvas.msRequestFullscreen) {
                        canvas.msRequestFullscreen();
                    }
                }
            } else if (e.key === 'Escape') {
                window.close();
            }
        });
        
        // Listen for messages from parent window
        window.addEventListener('message', function(event) {
            const message = event.data;
            
            if (message.type === 'init') {
                // Store shader prefixes
                shaderPrefixes = message.shaderPrefixes;
                
                // Initialize WebGL once; later inits are reconnects after the
                // opener reloaded and need a fresh shader, not a new context.
                initGL();
                
                // Send ready message
                if (window.opener) window.opener.postMessage({ type: 'kiosk-ready' }, '*');
            } else if (message.type === 'shader-update') {
                iResolutionIsVec2 = /uniform\\s+vec2\\s+iResolution/.test(message.fragmentShader);
                updateShaderProgram(message.vertexShader, message.fragmentShader);
            } else if (message.type === 'audio-data') {
                // Update audio data
                updateAudioData(message.audioData);
            } else if (message.type === 'display-text') {
                // Deliberately empty. The kiosk shares the opener's realm and reads
                // text, font, overlay, style AND the band's meta line straight off
                // window.opener's PianoscopeDisplayText every frame (see
                // getOpenerDisplayText), and blits the opener's already-painted
                // overlay canvas - so that single instance is already
                // authoritative here.
                //
                // Do NOT "mirror" the payload by calling a setter on it: that setter
                // runs in the OPENER's realm, where it ends in notify(), which posts
                // this very message back to this window - which would call the setter
                // again. The only thing standing between that and an unbounded loop
                // would be a setter that short-circuits on an unchanged value, and
                // that is not a property worth depending on from over here.
            }
        });
        
        // Initialize WebGL
        function initGL() {
            if (gl) return;

            // Try to get WebGL2 context first, then fall back to WebGL
            const kioskGlAttrs = {
                alpha: true,
                depth: false,
                stencil: false,
                antialias: false,
                premultipliedAlpha: false,
                preserveDrawingBuffer: false,
                powerPreference: 'high-performance'
            };
            gl = canvas.getContext('webgl2', kioskGlAttrs)
                || canvas.getContext('webgl', kioskGlAttrs)
                || canvas.getContext('experimental-webgl', kioskGlAttrs);
            isWebGL2 = gl instanceof WebGL2RenderingContext;
            
            if (!gl) {
                console.error('WebGL not supported in kiosk mode');
                return;
            }

            quadBuffer = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
            gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
                -1.0, -1.0,
                 1.0, -1.0,
                -1.0,  1.0,
                 1.0,  1.0
            ]), gl.STATIC_DRAW);
            
            // Set up audio texture
            audioTexture = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, audioTexture);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            
            // Create empty audio data
            const audioTexWidth = 256;
            audioData = new Uint8Array(audioTexWidth);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, audioTexWidth, 1, 0, gl.LUMINANCE, gl.UNSIGNED_BYTE, audioData);
            
            // Receives frames from the opener's video element in video mode
            videoTexture = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, videoTexture);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));

            fittedTextTexture = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, fittedTextTexture);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
                new Uint8Array([0, 0, 0, 255]));

            displayTextTexture = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, displayTextTexture);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
            
            // Start animation loop
            startAnimationLoop();

            resizeKioskCanvas();
        }
        
        // Update shader program
        function updateShaderProgram(vertexSource, fragmentSource) {
            if (!gl) return;

            const vertexShader = createShader(gl.VERTEX_SHADER, vertexSource);
            const fragmentShader = createShader(gl.FRAGMENT_SHADER, fragmentSource);
            if (!vertexShader || !fragmentShader) {
                if (vertexShader) gl.deleteShader(vertexShader);
                if (fragmentShader) gl.deleteShader(fragmentShader);
                return;
            }

            const program = gl.createProgram();
            gl.attachShader(program, vertexShader);
            gl.attachShader(program, fragmentShader);
            gl.linkProgram(program);

            if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
                console.error('Could not link program', gl.getProgramInfoLog(program));
                gl.deleteProgram(program);
                gl.deleteShader(vertexShader);
                gl.deleteShader(fragmentShader);
                return;
            }

            if (currentProgram) gl.deleteProgram(currentProgram);
            currentProgram = program;
            gl.useProgram(currentProgram);
            cacheKioskUniforms();
            configureFeedbackPass(fragmentSource);

            gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
            const positionLocation = gl.getAttribLocation(currentProgram, 'position');
            gl.enableVertexAttribArray(positionLocation);
            gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);
        }

        function cacheKioskUniforms() {
            kioskUniforms = {
                iResolution: gl.getUniformLocation(currentProgram, 'iResolution'),
                iTime: gl.getUniformLocation(currentProgram, 'iTime'),
                iTimeDelta: gl.getUniformLocation(currentProgram, 'iTimeDelta'),
                iFrame: gl.getUniformLocation(currentProgram, 'iFrame'),
                iChannel0: gl.getUniformLocation(currentProgram, 'iChannel0'),
                iChannel1: gl.getUniformLocation(currentProgram, 'iChannel1'),
                iChannel3: gl.getUniformLocation(currentProgram, 'iChannel3'),
                iWriteFeedback: gl.getUniformLocation(currentProgram, 'iWriteFeedback'),
                iFeedback: gl.getUniformLocation(currentProgram, 'iFeedback'),
                iChannelResolution: gl.getUniformLocation(currentProgram, 'iChannelResolution[0]'),
                iVideo: gl.getUniformLocation(currentProgram, 'iVideo'),
                iVideoActive: gl.getUniformLocation(currentProgram, 'iVideoActive'),
                iVideoBlend: gl.getUniformLocation(currentProgram, 'iVideoBlend'),
                iVideoBlendAmount: gl.getUniformLocation(currentProgram, 'iVideoBlendAmount'),
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
        
        // Create shader using prefixes passed from parent window
        function createShader(type, source) {
            const shader = gl.createShader(type);
            
            // Get the appropriate prefix from our parent-defined prefixes
            let prefix;
            if (isWebGL2) {
                prefix = (type === gl.VERTEX_SHADER) ? 
                    shaderPrefixes.webgl2VertexPrefix : 
                    shaderPrefixes.webgl2FragmentPrefix;
                
                // Replace gl_FragColor with fragColor for WebGL2 fragment shaders
                if (type === gl.FRAGMENT_SHADER) {
                    source = source.replace(/gl_FragColor/g, 'fragColor');
                }
            } else {
                prefix = shaderPrefixes.webgl1Prefix;
            }
            
            // Combine prefix with source
            const fullSource = prefix + source;
            
            gl.shaderSource(shader, fullSource);
            gl.compileShader(shader);
            
            if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
                console.error('Shader compilation error', gl.getShaderInfoLog(shader));
                gl.deleteShader(shader);
                return null;
            }
            
            return shader;
        }
        
        // Update audio data
        function updateAudioData(newAudioData) {
            if (!gl || !audioTexture || !audioData) return;
            
            const n = Math.min(audioData.length, newAudioData.length);
            if (newAudioData.subarray) {
                audioData.set(newAudioData.subarray(0, n));
            } else {
                for (let i = 0; i < n; i++) audioData[i] = newAudioData[i];
            }
            
            // Update texture
            // Pin the unit: this runs from a postMessage handler, so the active unit
            // is whatever the render loop happened to leave behind.
            gl.activeTexture(gl.TEXTURE7);
            gl.bindTexture(gl.TEXTURE_2D, audioTexture);
            gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, audioData.length, 1, gl.LUMINANCE, gl.UNSIGNED_BYTE, audioData);
        }
        
        function getOpenerDisplayText() {
            try {
                const dt = window.opener && window.opener.PianoscopeDisplayText;
                if (!dt) return null;
                const size = dt.getSize();
                // morph advances every frame while a swap is in flight but does
                // not bump version - the baked texture is unchanged, only the
                // blend between its two slots moves - so it is read per frame and
                // the upload stays gated on version.
                return {
                    canvas: dt.getPackedCanvas(),
                    width: size.width,
                    height: size.height,
                    version: dt.getVersion(),
                    morph: dt.getMorph(),
                    lineFrac: dt.getLineFrac(),
                    sheetVersion: dt.getSheetVersion ? dt.getSheetVersion() : dt.getVersion(),
                    extent: dt.getExtent(),
                    extentB: dt.getExtentB ? dt.getExtentB() : dt.getExtent(),
                    legacyExtent: dt.getLegacyExtent ? dt.getLegacyExtent() : { width: 0.386, height: 0.670 },
                    transition: dt.getTransitionIndex(),
                    fittedCanvas: dt.getFittedCanvas ? dt.getFittedCanvas() : null,
                    fittedSize: dt.getFittedSize ? dt.getFittedSize() : { width: 1, height: 1 },
                    wordRects: dt.getWordRects ? dt.getWordRects() : null,
                    wordCount: dt.getWordCount ? dt.getWordCount() : 0
                };
            } catch (e) {
                return null;
            }
        }

        // Video state lives in the opener. Guarded because the opener can be
        // closed or navigating while this loop is still running.
        function getOpenerVideoState() {
            try {
                const controller = window.opener && window.opener.videoController;
                if (!controller || !controller.hasSource()) return null;

                const handler = controller.videoHandler;
                const element = handler.getTextureElement ? handler.getTextureElement() : handler.videoElement;
                if (!element) return null;

                const size = controller.getVideoSize();
                const still = handler.isStill ? handler.isStill() : false;
                return {
                    element: element,
                    still: still,
                    width: size.width,
                    height: size.height,
                    fitMode: controller.getFitMode(),
                    mirror: controller.getMirror(),
                    // Whether the opener's current *look* is the video-effect
                    // program (vs. a library shader that may still sample iVideo).
                    effectLook: controller.isInVideoMode(),
                    blend: controller.getBlendMode(),
                    blendAmount: controller.getBlendAmount(),
                    active: controller.videoHandler.hasSource() ? 1 : 0
                };
            } catch (e) {
                return null;
            }
        }

        // This window uploads its own copy of each frame, so it dedupes them
        // independently of the opener rather than sharing a flag.
        let watchedVideoElement = null;
        let videoUsesFrameCallback = false;
        let videoHasNewFrame = true;

        function watchOpenerFrames(element) {
            if (watchedVideoElement === element) return;

            watchedVideoElement = element;
            videoHasNewFrame = true;
            videoUsesFrameCallback = typeof element.requestVideoFrameCallback === 'function';
            if (!videoUsesFrameCallback) return;

            const onFrame = () => {
                videoHasNewFrame = true;
                if (watchedVideoElement === element) {
                    element.requestVideoFrameCallback(onFrame);
                }
            };
            element.requestVideoFrameCallback(onFrame);
        }
        
        // Animation loop
        function startAnimationLoop() {
            if (animationFrameId) {
                cancelAnimationFrame(animationFrameId);
            }
            
            function render(now) {
                animationFrameId = requestAnimationFrame(render);

                if (document.hidden) return;
                resizeKioskCanvas();
                
                if (!gl || !currentProgram) return;
                
                // Calculate time values
                const nowSec = now / 1000;
                const currentTime = nowSec - startTime;
                const deltaTime = nowSec - lastTime;
                lastTime = nowSec;
                frameCount++;
                
                gl.useProgram(currentProgram);

                if (kioskUniforms.iResolution) {
                    if (iResolutionIsVec2) {
                        gl.uniform2f(kioskUniforms.iResolution, canvas.width, canvas.height);
                    } else {
                        gl.uniform3f(kioskUniforms.iResolution, canvas.width, canvas.height, 1.0);
                    }
                }

                if (kioskUniforms.iTime) gl.uniform1f(kioskUniforms.iTime, currentTime);
                if (kioskUniforms.iTimeDelta) gl.uniform1f(kioskUniforms.iTimeDelta, deltaTime);
                if (kioskUniforms.iFrame) gl.uniform1f(kioskUniforms.iFrame, frameCount);
                
                // Both windows share an origin, so the opener's <video> element can
                // be uploaded straight into this context instead of shipping pixels
                // through postMessage. One decoder feeds both canvases in sync.
                const videoState = getOpenerVideoState();
                
                if (videoState) {
                    try {
                        if (videoState.still) {
                            if (watchedVideoElement !== videoState.element) {
                                watchedVideoElement = videoState.element;
                                videoHasNewFrame = true;
                                videoUsesFrameCallback = false;
                            }
                        } else {
                            watchOpenerFrames(videoState.element);
                        }

                        gl.activeTexture(gl.TEXTURE1);
                        gl.bindTexture(gl.TEXTURE_2D, videoTexture);

                        const el = videoState.element;
                        const vw = videoState.still ? el.naturalWidth : el.videoWidth;
                        const vh = videoState.still ? el.naturalHeight : el.videoHeight;
                        const ready = videoState.still
                            ? (el.complete && vw && vh)
                            : (el.readyState >= 2);

                        // Upload when the video-effect look needs iChannel1, OR any
                        // shader on screen declares iVideo — a library shader with
                        // no iVideo uniform gets nothing to skip the upload cost.
                        // Stills upload once; clips skip repeat frames via rVFC.
                        const wantsUpload = videoState.effectLook || kioskUniforms.iVideo;
                        const hasNew = videoState.still ? videoHasNewFrame : (!videoUsesFrameCallback || videoHasNewFrame);

                        if (wantsUpload && ready && hasNew && vw && vh) {
                            videoHasNewFrame = false;

                            // Avoid the browser's default colorspace conversion / Y-flip
                            // work on every frame; the shaders already handle the flip.
                            gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
                            gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);

                            if (isWebGL2) {
                                if (videoTexWidth !== vw || videoTexHeight !== vh) {
                                    // Immutable storage can't be resized in place; recreate
                                    // the texture object and re-apply its sampling params.
                                    gl.deleteTexture(videoTexture);
                                    videoTexture = gl.createTexture();
                                    gl.activeTexture(gl.TEXTURE1);
                                    gl.bindTexture(gl.TEXTURE_2D, videoTexture);
                                    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
                                    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
                                    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
                                    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
                                    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, vw, vh);
                                    videoTexWidth = vw;
                                    videoTexHeight = vh;
                                }

                                gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, el);
                            } else {
                                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, el);
                            }

                            gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.BROWSER_DEFAULT_WEBGL);
                        }
                    } catch (e) {
                        videoHasNewFrame = true;
                    }

                    // iChannel1 = video is the video-effect look's legacy contract;
                    // any other program reads the source through iVideo instead.
                    if (videoState.effectLook && kioskUniforms.iChannel1) {
                        gl.activeTexture(gl.TEXTURE1);
                        gl.bindTexture(gl.TEXTURE_2D, videoTexture);
                        gl.uniform1i(kioskUniforms.iChannel1, 1);
                    }

                    if (kioskUniforms.iVideo) {
                        gl.activeTexture(gl.TEXTURE6);
                        gl.bindTexture(gl.TEXTURE_2D, videoTexture);
                        gl.uniform1i(kioskUniforms.iVideo, 6);
                    }

                    if (kioskUniforms.iVideoResolution) {
                        gl.uniform2f(kioskUniforms.iVideoResolution, videoState.width, videoState.height);
                    }
                    if (kioskUniforms.iFitMode) gl.uniform1f(kioskUniforms.iFitMode, videoState.fitMode);
                    if (kioskUniforms.iVideoMirror) gl.uniform1f(kioskUniforms.iVideoMirror, videoState.mirror);
                    if (kioskUniforms.iVideoActive) {
                        gl.uniform1f(kioskUniforms.iVideoActive, videoState.active ? 1.0 : 0.0);
                    }
                    if (kioskUniforms.iVideoBlend) {
                        gl.uniform1f(kioskUniforms.iVideoBlend, videoState.blend || 0);
                    }
                    if (kioskUniforms.iVideoBlendAmount) {
                        gl.uniform1f(kioskUniforms.iVideoBlendAmount,
                            typeof videoState.blendAmount === 'number' ? videoState.blendAmount : 1);
                    }
                } else {
                    if (kioskUniforms.iVideoActive) gl.uniform1f(kioskUniforms.iVideoActive, 0.0);
                    if (kioskUniforms.iVideoBlend) gl.uniform1f(kioskUniforms.iVideoBlend, 0.0);
                    if (kioskUniforms.iVideoBlendAmount) gl.uniform1f(kioskUniforms.iVideoBlendAmount, 0.0);
                }
                
                // Audio holds iChannel0 in both modes so video effects stay reactive.
                if (kioskUniforms.iChannel0) {
                    gl.activeTexture(gl.TEXTURE0);
                    gl.bindTexture(gl.TEXTURE_2D, audioTexture);
                    gl.uniform1i(kioskUniforms.iChannel0, 0);
                }

                const textState = getOpenerDisplayText();
                if (displayTextTexture) {
                    gl.activeTexture(gl.TEXTURE3);
                    gl.bindTexture(gl.TEXTURE_2D, displayTextTexture);

                    if (textState && textState.canvas && textState.version !== lastDisplayTextVersion) {
                        lastDisplayTextVersion = textState.version;
                        displayTextW = textState.width;
                        displayTextH = textState.height;
                        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 1);
                        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, textState.canvas);
                        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 0);
                    } else if ((!textState || !textState.canvas) && lastDisplayTextVersion !== -2) {
                        lastDisplayTextVersion = -2;
                        displayTextW = 1;
                        displayTextH = 1;
                        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
                    }

                    if (kioskUniforms.iChannel3) gl.uniform1i(kioskUniforms.iChannel3, 3);
                }

                if (kioskUniforms.iTextMorph) {
                    gl.uniform1f(kioskUniforms.iTextMorph, textState ? textState.morph : 0);
                }
                if (kioskUniforms.iTextLineH) {
                    gl.uniform1f(kioskUniforms.iTextLineH, textState ? textState.lineFrac : 0.5);
                }
                if (kioskUniforms.iTextExtent) {
                    const ext = (textState && textState.extent) || { width: 0.5, height: 0.5 };
                    gl.uniform2f(kioskUniforms.iTextExtent, ext.width, ext.height);
                }
                if (kioskUniforms.iTextExtentB) {
                    const extB = (textState && textState.extentB)
                        || (textState && textState.extent) || { width: 0.5, height: 0.5 };
                    gl.uniform2f(kioskUniforms.iTextExtentB, extB.width, extB.height);
                }
                if (kioskUniforms.iTextLegacyExtent) {
                    const leg = (textState && textState.legacyExtent) || { width: 0.386, height: 0.670 };
                    gl.uniform2f(kioskUniforms.iTextLegacyExtent, leg.width, leg.height);
                }
                if (kioskUniforms.iTextTransition) {
                    gl.uniform1f(kioskUniforms.iTextTransition, textState ? textState.transition : 0);
                }

                if (kioskUniforms.iTextSheet && fittedTextTexture) {
                    gl.activeTexture(gl.TEXTURE4);
                    gl.bindTexture(gl.TEXTURE_2D, fittedTextTexture);
                    // Gated on sheetVersion, not version: a scramble redraws the sheet
                    // many times inside one string change, and version only ticks when
                    // the string itself changes.
                    if (textState && textState.fittedCanvas && textState.sheetVersion !== lastFittedVersion) {
                        lastFittedVersion = textState.sheetVersion;
                        fittedTextW = textState.fittedSize.width;
                        fittedTextH = textState.fittedSize.height;
                        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 1);
                        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE,
                            textState.fittedCanvas);
                        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 0);
                    }
                    gl.uniform1i(kioskUniforms.iTextSheet, 4);
                }
                if (kioskUniforms.iTextSheetRes) {
                    gl.uniform2f(kioskUniforms.iTextSheetRes, fittedTextW, fittedTextH);
                }
                if (kioskUniforms.iTextWords && textState && textState.wordRects) {
                    gl.uniform4fv(kioskUniforms.iTextWords, textState.wordRects);
                }
                if (kioskUniforms.iTextWordCount) {
                    gl.uniform1f(kioskUniforms.iTextWordCount, textState ? textState.wordCount : 0);
                }

                if (kioskUniforms.iChannelResolution) {
                    const videoEffectActive = videoState && videoState.effectLook;
                    const videoW = videoEffectActive ? videoState.width : 1;
                    const videoH = videoEffectActive ? videoState.height : 1;
                    gl.uniform3fv(kioskUniforms.iChannelResolution, new Float32Array([
                        256, 1, 1,
                        videoW, videoH, 1,
                        1, 1, 1,
                        displayTextW, displayTextH, 1
                    ]));
                }
                
                if (usesFeedbackPass && feedbackState) {
                    const readTexture = feedbackState.textures[feedbackState.readIndex];
                    const writeIndex = 1 - feedbackState.readIndex;

                    gl.activeTexture(gl.TEXTURE5);
                    gl.bindTexture(gl.TEXTURE_2D, readTexture);
                    if (kioskUniforms.iFeedback) gl.uniform1i(kioskUniforms.iFeedback, 5);
                    // Outside the video-effect look iChannel1 is spare, and that is
                    // where shader-mode feedback reads its previous frame from.
                    if (!(videoState && videoState.effectLook) && kioskUniforms.iChannel1) {
                        gl.activeTexture(gl.TEXTURE1);
                        gl.bindTexture(gl.TEXTURE_2D, readTexture);
                        gl.uniform1i(kioskUniforms.iChannel1, 1);
                    }

                    gl.bindFramebuffer(gl.FRAMEBUFFER, feedbackState.fbos[writeIndex]);
                    gl.viewport(0, 0, feedbackState.width, feedbackState.height);
                    if (kioskUniforms.iWriteFeedback) gl.uniform1f(kioskUniforms.iWriteFeedback, 1.0);
                    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

                    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
                    gl.viewport(0, 0, canvas.width, canvas.height);
                    if (kioskUniforms.iWriteFeedback) gl.uniform1f(kioskUniforms.iWriteFeedback, 0.0);
                    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

                    feedbackState.readIndex = writeIndex;
                } else {
                    // Draw fullscreen quad
                    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
                }

                if (overlayCtx && overlayCanvas) {
                    try {
                        const dt = window.opener && window.opener.PianoscopeDisplayText;
                        if (!dt || !dt.isOverlayEnabled()) {
                            overlayCanvas.style.display = 'none';
                        } else {
                            overlayCanvas.style.display = 'block';
                            const src = dt.getOverlayCanvas();
                            const dpr = Math.min(window.devicePixelRatio || 1, 2);
                            const ow = Math.max(1, Math.round(overlayCanvas.clientWidth * dpr));
                            const oh = Math.max(1, Math.round(overlayCanvas.clientHeight * dpr));
                            if (overlayCanvas.width !== ow || overlayCanvas.height !== oh) {
                                overlayCanvas.width = ow;
                                overlayCanvas.height = oh;
                                lastOverlayVersion = -1;
                            }
                            if (src && dt.getVersion() !== lastOverlayVersion) {
                                lastOverlayVersion = dt.getVersion();
                                overlayCtx.clearRect(0, 0, ow, oh);
                                overlayCtx.drawImage(src, 0, 0, ow, oh);
                            }
                        }
                    } catch (e) {}
                }
            }
            
            animationFrameId = requestAnimationFrame(render);
        }
        
        // Auto-enter fullscreen mode
        function enterFullscreen() {
            if (canvas.requestFullscreen) {
                canvas.requestFullscreen();
            } else if (canvas.webkitRequestFullscreen) {
                canvas.webkitRequestFullscreen();
            } else if (canvas.mozRequestFullScreen) {
                canvas.mozRequestFullScreen();
            } else if (canvas.msRequestFullscreen) {
                canvas.msRequestFullscreen();
            }
        }
        
        function announceToOpener() {
            if (!window.opener || window.opener.closed) return;
            try {
                window.opener.postMessage({ type: 'kiosk-loaded' }, '*');
            } catch (e) {}
        }

        announceToOpener();

        // The opener's message listener dies on reload. Keep offering a
        // handshake until the new page claims this window again.
        setInterval(function() {
            if (!window.opener || window.opener.closed) return;
            try {
                if (window.opener.kioskWindow !== window) announceToOpener();
            } catch (e) {}
        }, 500);
    </script>
</body>
</html>`;

// Function to create the kiosk.html file
function createKioskHtmlFile() {
    // Create a Blob with the HTML content
    const blob = new Blob([kioskHtmlContent], { type: 'text/html' });
    
    // Create a URL for the Blob
    const url = URL.createObjectURL(blob);
    
    return url;
}

function handleKioskMessage(event) {
    const message = event.data;
    if (!message || typeof message !== 'object') return;

    if (message.type === 'kiosk-loaded') {
        window.kioskWindow = event.source;
        event.source.postMessage({
            type: 'init',
            shaderPrefixes: ShaderPrefixes
        }, '*');
    } else if (message.type === 'kiosk-ready') {
        if (window.kioskWindow && event.source !== window.kioskWindow) return;
        sendCurrentShaderToKiosk();
    }
}

// Function to open the kiosk mode window
function openKioskMode() {
    if (window.kioskWindow && !window.kioskWindow.closed) {
        window.kioskWindow.focus();
        sendCurrentShaderToKiosk();
        return;
    }

    // Create and get a URL for the kiosk.html file
    const kioskUrl = createKioskHtmlFile();
    
    // Open the kiosk window
    const kioskWindow = window.open(kioskUrl, 'PianoscopeKiosk', 'width=800,height=600');
    
    if (!kioskWindow) {
        alert('Please allow popups to open the kiosk mode');
        return;
    }
    
    // Store a reference to the kiosk window
    window.kioskWindow = kioskWindow;
    
    // Try to position on second screen if available
    try {
        if (window.screen.isExtended) {
            kioskWindow.moveTo(window.screen.availLeft, 0);
            kioskWindow.resizeTo(window.screen.availWidth, window.screen.availHeight);
        }
    } catch (e) {
        console.log('Could not position on second screen', e);
    }
}

// Function to send the current shader to the kiosk window
function sendCurrentShaderToKiosk() {
    if (!window.kioskWindow || window.kioskWindow.closed) return;
    
    // Use raw shader sources (before WebGL prefix) to avoid double-prefix compile errors
    let vertexSource = window.currentVertexShaderSource || null;
    let fragmentSource = window.currentFragmentShaderSource || null;
    
    // If we couldn't get the shader sources, use default fallback shaders
    if (!vertexSource) {
        vertexSource = `
            attribute vec2 position;
            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
            }
        `;
    }
    
    if (!fragmentSource) {
        fragmentSource = `
            precision highp float;
            
            uniform vec3 iResolution;
            uniform float iTime;
            uniform sampler2D iChannel0;
            
            void main() {
                vec2 uv = gl_FragCoord.xy / iResolution.xy;
                
                // Simple color gradient
                vec3 color = 0.5 + 0.5 * cos(iTime + uv.xyx + vec3(0,2,4));
                
                // Add some audio reactivity if available
                float audio = texture2D(iChannel0, vec2(uv.x, 0.0)).x;
                color += audio * 0.2;
                
                gl_FragColor = vec4(color, 1.0);
            }
        `;
    }
    
    // Send the shaders to the kiosk window
    window.kioskWindow.postMessage({
        type: 'shader-update',
        vertexShader: vertexSource,
        fragmentShader: fragmentSource
    }, '*');
}

// Function to send audio data to the kiosk window
function sendAudioDataToKiosk() {
    if (!window.kioskWindow || window.kioskWindow.closed) return;
    
    // Get the audio data from your visualizer
    // This assumes you have a global variable for this
    let audioData = null;
    
    if (window.audioData) {
        audioData = window.audioData;
    } else if (window.visualizer && window.visualizer.audioData) {
        audioData = window.visualizer.audioData;
    }
    
    if (!audioData) return;
    
    // Send the audio data to the kiosk window
    window.kioskWindow.postMessage({
        type: 'audio-data',
        audioData: audioData
    }, '*');
}

// Hook into the shader update and audio update functions
// This will need to be customized based on your visualizer's architecture

// For shader updates - add this where you update your shaders
function onShaderUpdate() {
    // Call this after you update the shader in your main window
    if (window.kioskWindow && !window.kioskWindow.closed) {
        sendCurrentShaderToKiosk();
    }
}

// For audio updates - add this to your audio processing loop
function setupAudioSync() {
    // Set up a periodic sync of audio data
    setInterval(function() {
        if (window.kioskWindow && !window.kioskWindow.closed) {
            sendAudioDataToKiosk();
        }
    }, 50); // Update every 50ms (20fps)
}

// Add button to open kiosk mode
document.addEventListener('DOMContentLoaded', function() {
    const buttonContainer = document.querySelector('.button-container');
    if (buttonContainer) {
        const kioskButton = document.createElement('button');
        kioskButton.id = 'kiosk-mode-button';
        kioskButton.className = 'gold-button';
        kioskButton.innerHTML = '<i class="fas fa-desktop"></i><span>&nbsp;Kiosk Mode</span>';
        kioskButton.title = 'Open in kiosk mode (fullscreen on second display)';
        kioskButton.addEventListener('click', openKioskMode);
        buttonContainer.appendChild(kioskButton);
    }

    window.addEventListener('message', handleKioskMessage);
    
    // Set up audio sync when the page is loaded
    setupAudioSync();
});
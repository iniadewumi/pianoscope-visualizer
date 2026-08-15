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
        }
        
        #kiosk-canvas {
            display: block;
            width: 100%;
            height: 100%;
            padding: 0;
            margin: 0;
        }
    </style>
</head>
<body>
    <canvas id="kiosk-canvas"></canvas>
    
    <script>
        // Will be filled by the parent window
        let gl = null;
        let isWebGL2 = false;
        let currentProgram = null;
        let audioTexture = null;
        let videoTexture = null;
        let audioData = null;
        let animationFrameId = null;
        let startTime = performance.now() / 1000;
        let lastTime = startTime;
        let frameCount = 0;
        let iResolutionIsVec2 = false;
        
        // Shader prefix constants passed from parent window
        let shaderPrefixes = null;
        
        // Set up the canvas
        const canvas = document.getElementById('kiosk-canvas');

        function resizeKioskCanvas() {
            const dpr = window.devicePixelRatio || 1;
            const displayWidth = canvas.clientWidth;
            const displayHeight = canvas.clientHeight;
            if (displayWidth === 0 || displayHeight === 0) return;

            const bufferWidth = Math.round(displayWidth * dpr);
            const bufferHeight = Math.round(displayHeight * dpr);
            canvas.width = bufferWidth;
            canvas.height = bufferHeight;

            if (gl) {
                gl.viewport(0, 0, bufferWidth, bufferHeight);
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
            }
        });
        
        // Initialize WebGL
        function initGL() {
            if (gl) return;

            // Try to get WebGL2 context first, then fall back to WebGL
            gl = canvas.getContext('webgl2') || canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
            isWebGL2 = gl instanceof WebGL2RenderingContext;
            
            if (!gl) {
                console.error('WebGL not supported in kiosk mode');
                return;
            }
            
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
            
            // Set up geometry (full-screen quad)
            const positionBuffer = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
            gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
                -1.0, -1.0,
                 1.0, -1.0,
                -1.0,  1.0,
                 1.0,  1.0
            ]), gl.STATIC_DRAW);
            
            const positionLocation = gl.getAttribLocation(currentProgram, 'position');
            gl.enableVertexAttribArray(positionLocation);
            gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);
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
            
            // Copy new audio data
            for (let i = 0; i < audioData.length && i < newAudioData.length; i++) {
                audioData[i] = newAudioData[i];
            }
            
            // Update texture
            gl.bindTexture(gl.TEXTURE_2D, audioTexture);
            gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, audioData.length, 1, gl.LUMINANCE, gl.UNSIGNED_BYTE, audioData);
        }
        
        // Video state lives in the opener. Guarded because the opener can be
        // closed or navigating while this loop is still running.
        function getOpenerVideoState() {
            try {
                const controller = window.opener && window.opener.videoController;
                if (!controller || !controller.isInVideoMode()) return null;

                const element = controller.videoHandler.videoElement;
                if (!element) return null;

                const size = controller.getVideoSize();
                return {
                    element: element,
                    width: size.width,
                    height: size.height,
                    fitMode: controller.getFitMode(),
                    mirror: controller.getMirror()
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
                
                if (!gl || !currentProgram) return;
                
                // Calculate time values
                const nowSec = now / 1000;
                const currentTime = nowSec - startTime;
                const deltaTime = nowSec - lastTime;
                lastTime = nowSec;
                frameCount++;
                
                gl.useProgram(currentProgram);
                
                // Set common uniforms if they exist
                const iResolution = gl.getUniformLocation(currentProgram, 'iResolution');
                if (iResolution) {
                    if (iResolutionIsVec2) {
                        gl.uniform2f(iResolution, canvas.width, canvas.height);
                    } else {
                        gl.uniform3f(iResolution, canvas.width, canvas.height, 1.0);
                    }
                }
                
                const iTime = gl.getUniformLocation(currentProgram, 'iTime');
                if (iTime) gl.uniform1f(iTime, currentTime);
                
                const iTimeDelta = gl.getUniformLocation(currentProgram, 'iTimeDelta');
                if (iTimeDelta) gl.uniform1f(iTimeDelta, deltaTime);

                const iFrame = gl.getUniformLocation(currentProgram, 'iFrame');
                if (iFrame) gl.uniform1f(iFrame, frameCount);
                
                // Both windows share an origin, so the opener's <video> element can
                // be uploaded straight into this context instead of shipping pixels
                // through postMessage. One decoder feeds both canvases in sync.
                const videoState = getOpenerVideoState();
                
                if (videoState) {
                    try {
                        watchOpenerFrames(videoState.element);

                        gl.activeTexture(gl.TEXTURE1);
                        gl.bindTexture(gl.TEXTURE_2D, videoTexture);

                        if (videoState.element.readyState >= 2 &&
                            (!videoUsesFrameCallback || videoHasNewFrame)) {
                            videoHasNewFrame = false;
                            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, videoState.element);
                        }
                    } catch (e) {
                        videoHasNewFrame = true;
                    }
                    
                    const iChannel1 = gl.getUniformLocation(currentProgram, 'iChannel1');
                    if (iChannel1) gl.uniform1i(iChannel1, 1);
                    
                    // The fit is computed against this window's own resolution, so
                    // the kiosk display crops or letterboxes independently.
                    const iVideoResolution = gl.getUniformLocation(currentProgram, 'iVideoResolution');
                    if (iVideoResolution) gl.uniform2f(iVideoResolution, videoState.width, videoState.height);
                    
                    const iFitMode = gl.getUniformLocation(currentProgram, 'iFitMode');
                    if (iFitMode) gl.uniform1f(iFitMode, videoState.fitMode);
                    
                    const iVideoMirror = gl.getUniformLocation(currentProgram, 'iVideoMirror');
                    if (iVideoMirror) gl.uniform1f(iVideoMirror, videoState.mirror);
                }
                
                // Audio holds iChannel0 in both modes so video effects stay reactive.
                const iChannel0 = gl.getUniformLocation(currentProgram, 'iChannel0');
                if (iChannel0) {
                    gl.activeTexture(gl.TEXTURE0);
                    gl.bindTexture(gl.TEXTURE_2D, audioTexture);
                    gl.uniform1i(iChannel0, 0);
                }
                
                // Draw fullscreen quad
                gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
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
        audioData: Array.from(audioData) // Convert to regular array for transfer
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
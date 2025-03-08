// Create a dedicated kiosk mode HTML file: kiosk.html
// This is a minimal page that just contains a canvas element

// First, let's create the kiosk.html file content
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
        let currentProgram = null;
        let audioTexture = null;
        let audioData = null;
        let animationFrameId = null;
        let startTime = performance.now() / 1000;
        let lastTime = startTime;
        
        // Set up the canvas
        const canvas = document.getElementById('kiosk-canvas');
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        
        // Handle window resize
        window.addEventListener('resize', function() {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            if (gl) {
                gl.viewport(0, 0, canvas.width, canvas.height);
            }
        });
        
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
                // Initialize WebGL
                initGL();
                // Send ready message
                window.opener.postMessage({ type: 'kiosk-ready' }, '*');
            } else if (message.type === 'shader-update') {
                // Update shader program
                updateShaderProgram(message.vertexShader, message.fragmentShader);
            } else if (message.type === 'audio-data') {
                // Update audio data
                updateAudioData(message.audioData);
            }
        });
        
        // Initialize WebGL
        function initGL() {
            gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
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
            
            // Start animation loop
            startAnimationLoop();
        }
        
        // Update shader program
        function updateShaderProgram(vertexSource, fragmentSource) {
            if (!gl) return;
            
            // Clean up previous program
            if (currentProgram) {
                gl.deleteProgram(currentProgram);
            }
            
            // Create shaders
            const vertexShader = createShader(gl.VERTEX_SHADER, vertexSource);
            const fragmentShader = createShader(gl.FRAGMENT_SHADER, fragmentSource);
            
            if (!vertexShader || !fragmentShader) return;
            
            // Create program
            currentProgram = gl.createProgram();
            gl.attachShader(currentProgram, vertexShader);
            gl.attachShader(currentProgram, fragmentShader);
            gl.linkProgram(currentProgram);
            
            if (!gl.getProgramParameter(currentProgram, gl.LINK_STATUS)) {
                console.error('Could not link program', gl.getProgramInfoLog(currentProgram));
                return;
            }
            
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
        
        // Create shader
        function createShader(type, source) {
            const shader = gl.createShader(type);
            gl.shaderSource(shader, source);
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
                
                gl.useProgram(currentProgram);
                
                // Set common uniforms if they exist
                const iResolution = gl.getUniformLocation(currentProgram, 'iResolution');
                if (iResolution) gl.uniform2f(iResolution, canvas.width, canvas.height);
                
                const iTime = gl.getUniformLocation(currentProgram, 'iTime');
                if (iTime) gl.uniform1f(iTime, currentTime);
                
                const iTimeDelta = gl.getUniformLocation(currentProgram, 'iTimeDelta');
                if (iTimeDelta) gl.uniform1f(iTimeDelta, deltaTime);
                
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
        
        // Notify parent that we're ready to initialize
        window.opener.postMessage({ type: 'kiosk-loaded' }, '*');
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

// Function to open the kiosk mode window
function openKioskMode() {
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
    
    // Listen for messages from the kiosk window
    window.addEventListener('message', function(event) {
        const message = event.data;
        
        if (message.type === 'kiosk-loaded') {
            // Kiosk window is loaded, initialize WebGL
            kioskWindow.postMessage({ type: 'init' }, '*');
        } else if (message.type === 'kiosk-ready') {
            // Kiosk is ready, send current shader
            sendCurrentShaderToKiosk();
        }
    });
    
    // Clean up when kiosk window is closed
    kioskWindow.addEventListener('beforeunload', function() {
        window.kioskWindow = null;
    });
}

// Function to get the current shader source
function getShaderSource(shader) {
    if (!shader) return null;
    
    const gl = document.getElementById('visualizer').getContext('webgl') || 
              document.getElementById('visualizer').getContext('experimental-webgl');
    
    if (!gl) return null;
    
    return gl.getShaderSource(shader);
}

// Function to send the current shader to the kiosk window
function sendCurrentShaderToKiosk() {
    if (!window.kioskWindow || window.kioskWindow.closed) return;
    
    // Get the current vertex and fragment shaders
    // This assumes you have global variables for these in your visualizer
    let vertexSource = null;
    let fragmentSource = null;
    
    // Try to get shader sources from different possible global variables
    if (window.currentVertexShader) {
        vertexSource = getShaderSource(window.currentVertexShader);
    } else if (window.visualizer && window.visualizer.currentVertexShader) {
        vertexSource = getShaderSource(window.visualizer.currentVertexShader);
    }
    
    if (window.currentFragmentShader) {
        fragmentSource = getShaderSource(window.currentFragmentShader);
    } else if (window.visualizer && window.visualizer.currentFragmentShader) {
        fragmentSource = getShaderSource(window.visualizer.currentFragmentShader);
    }
    
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
            
            uniform vec2 iResolution;
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
    
    // Set up audio sync when the page is loaded
    setupAudioSync();
});
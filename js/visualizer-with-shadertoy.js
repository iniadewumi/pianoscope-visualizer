/**
 * HTML5 Microphone Shader Visualizer with Shadertoy Support
 * 
 * This script handles the WebGL and audio processing to visualize 
 * microphone input in real-time with shader effects.
 */

document.addEventListener('DOMContentLoaded', () => {
    "use strict";
    
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
        
        const float iter = 64.0;
        const float divAng = 24.0 * 6.2831853/360.0;
        const float circRad = 0.23;
        const float rat = 0.045/circRad;
        
        float nearestMult(float v, float of) {
            float m = mod(v, of);
            v -= m * sign(of/2.0 - m);
            return v - mod(v, of);
        }
        
        vec4 pal(float t) {
            return 0.5 + 0.5 * cos(6.283 * (t + vec4(0.0, 1.0, 2.0, 0.0)/3.0));
        }
        
        void main() {
            vec2 R = iResolution.xy;
            vec2 uv = gl_FragCoord.xy;
            vec2 center = vec2(0.0);
            float M = max(R.x, R.y);
            uv = (uv - 0.5 * R) / M / 0.7;
            
            float l = length(uv);
            float sl = texture2D(iChannel0, vec2(0.0, 0.0)).x;
            float sl2 = texture2D(iChannel0, vec2(0.25, 0.0)).x * 0.5;
            float sm = texture2D(iChannel0, vec2(0.5, 0.0)).x * 0.2;
            float sm2 = texture2D(iChannel0, vec2(0.75, 0.0)).x * 0.2;
            float sh = texture2D(iChannel0, vec2(1.0, 0.0)).x * 0.2;
            float st = (sl + sl2 + sm + sm2 + sh);
            
            float time = iTime;
            float sCircRad = circRad * rat;
            float ds = (2.0 + 1.4 * st) * rat;
            float ang, dist;
            vec2 p;
            
            vec4 o = vec4(0.1, 0.1, 0.1, 1.0);
            
            for(float i = 0.0; i < iter; i += 1.0) {
                p = uv - center;
                ang = atan(p.y, p.x);
                ang = nearestMult(ang, divAng);
                center += sCircRad / rat * vec2(cos(ang), sin(ang));
                dist = distance(center, uv);
                
                if (dist <= sCircRad) {
                    o += 30.0 * dist * pal(fract(dist / sCircRad + st + l));
                }
                
                sCircRad *= ds;
            }
            
            gl_FragColor = o;
        }
    `;
    
    // === WEBGL SETUP ===
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    if (!gl) {
        if (statusEl) statusEl.textContent = 'WebGL not supported in your browser';
        console.error('WebGL not supported');
        return;
    }
    
    // Resize canvas to window
    function resizeCanvas() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight * 0.8;
        gl.viewport(0, 0, canvas.width, canvas.height);
    }
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
    
    // === SHADER MANAGEMENT ===
    let currentProgram = null;
    let currentVertexShader = null;
    let currentFragmentShader = null;
    let frameCount = 0;
    let lastTime = 0;
    let deltaTime = 0;
    let uniforms = {};
    
    // Compile shader with error detection
    function createShader(gl, type, source) {
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            const error = gl.getShaderInfoLog(shader);
            if (statusEl) statusEl.textContent = 'Shader Error: ' + error.substring(0, 100) + '...';
            console.error('Shader compile error:', error);
            gl.deleteShader(shader);
            return null;
        }
        return shader;
    }
    
    // Create and link a program with given shaders
    function createProgram(gl, vertexShader, fragmentShader) {
        const program = gl.createProgram();
        gl.attachShader(program, vertexShader);
        gl.attachShader(program, fragmentShader);
        gl.linkProgram(program);
        
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            const error = gl.getProgramInfoLog(program);
            if (statusEl) statusEl.textContent = 'Program Error: ' + error.substring(0, 100) + '...';
            console.error('Program link error:', error);
            return null;
        }
        
        return program;
    }
    
    // Set up a new shader program
    function setupShaderProgram(vertexSource, fragmentSource) {
        // Clean up previous program if it exists
        if (currentProgram) {
            gl.deleteProgram(currentProgram);
        }
        if (currentVertexShader) {
            gl.deleteShader(currentVertexShader);
        }
        if (currentFragmentShader) {
            gl.deleteShader(currentFragmentShader);
        }
        
        // Create new shaders
        currentVertexShader = createShader(gl, gl.VERTEX_SHADER, vertexSource);
        if (!currentVertexShader) return false;
        
        currentFragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
        if (!currentFragmentShader) return false;
        
        // Create and use program
        currentProgram = createProgram(gl, currentVertexShader, currentFragmentShader);
        if (!currentProgram) return false;
        
        gl.useProgram(currentProgram);
        
        // Set up vertex attributes
        const positions = new Float32Array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0
        ]);
        
        const positionBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
        
        const positionLocation = gl.getAttribLocation(currentProgram, 'position');
        gl.enableVertexAttribArray(positionLocation);
        gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);
        
        // Reset frame counter
        frameCount = 0;
        
        // Get all uniforms after successful program setup
        uniforms = getShaderUniforms();
        
        return true;
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
            iDate: gl.getUniformLocation(currentProgram, 'iDate'),
            iSampleRate: gl.getUniformLocation(currentProgram, 'iSampleRate')
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
    const audioTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, audioTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    
    // Create audio data array
    audioData = new Uint8Array(audioTexWidth);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, audioTexWidth, 1, 0, gl.LUMINANCE, gl.UNSIGNED_BYTE, audioData);
    
    // Initialize with default shader
    const initSuccess = setupShaderProgram(vertexShaderSource, fragmentShaderSource);
    if (!initSuccess) {
        if (statusEl) statusEl.textContent = 'Failed to initialize default shader';
        console.error('Failed to initialize default shader');
    }
    
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
            micToggle.innerHTML = '<span class="fa fa-stop"></span>Stop Listen';
            micToggle.className = "red-button";
            
            // Start volume monitoring
            showVolume();
            
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
        micToggle.innerHTML = '<span class="fa fa-play"></span>Start Listen';
        micToggle.className = "green-button";
        if (statusEl) statusEl.textContent = 'Audio stopped';
        if (h) h.textContent = 'Please allow the use of your microphone.';
        if (hSub) hSub.textContent = '';
        
        // Clear audio data
        audioData.fill(0);
        updateAudioTexture();
    }
    
    function updateAudioTexture() {
        if (isListening && analyser) {
            analyser.getByteFrequencyData(frequencyArray);
            
            // Copy and apply sensitivity
            for (let i = 0; i < audioData.length; i++) {
                if (i < frequencyArray.length) {
                    audioData[i] = Math.min(255, frequencyArray[i] * audioSensitivity);
                } else {
                    audioData[i] = 0;
                }
            }
        }
        
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, audioTexture);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, audioTexWidth, 1, gl.LUMINANCE, gl.UNSIGNED_BYTE, audioData);
    }
    
    function showVolume() {
        if (!isListening) return;
        
        if (analyser) {
            analyser.getByteFrequencyData(frequencyArray);
            let total = 0;
            
            for(let i = 0; i < frequencyArray.length; i++) {
                const x = frequencyArray[i];
                total += x * x;
            }
            
            const rms = Math.sqrt(total / frequencyArray.length);
            const db = 20 * (Math.log(rms) / Math.log(10));
            const dbValue = Math.max(db, 0); // sanity check
            
            if (h) h.innerHTML = Math.floor(dbValue) + " dB";
            
            if (dbValue >= loud_volume_threshold) {
                seconds += 0.5;
                if (seconds >= 5) {
                    if (hSub) hSub.innerHTML = "You've been in loud environment for<span> " + Math.floor(seconds) + " </span>seconds.";
                }
            } else {
                seconds = 0;
                if (hSub) hSub.innerHTML = "";
            }
        }
        
        // Continue monitoring volume every 500ms
        setTimeout(showVolume, 500);
    }
    
    // === RENDER LOOP ===
    let startTime = performance.now() / 1000;
    let animationRunning = false;
    
    function render(now) {
        if (!animationRunning) return;
        
        requestAnimationFrame(render);
        
        // Calculate time values
        const nowSec = now / 1000;
        const currentTime = nowSec - startTime;
        deltaTime = nowSec - lastTime;
        lastTime = nowSec;
        frameCount++;
        
        // Set uniform values for shader if program exists
        if (!currentProgram) return;
        
        // Set uniform values for shader
        if (uniforms.iResolution) gl.uniform2f(uniforms.iResolution, canvas.width, canvas.height);
        if (uniforms.iTime) gl.uniform1f(uniforms.iTime, currentTime);
        if (uniforms.iTimeDelta) gl.uniform1f(uniforms.iTimeDelta, deltaTime);
        if (uniforms.iFrame) gl.uniform1i(uniforms.iFrame, frameCount);
        
        // Build the mouse uniform: [mouseX, mouseY, mouseDown, clickTime]
        if (uniforms.iMouse) gl.uniform4f(uniforms.iMouse, mouseX, mouseY, mouseDown, clickX);
        
        // Update date uniform if exists
        if (uniforms.iDate) {
            const d = new Date();
            gl.uniform4f(uniforms.iDate, 
                d.getFullYear(), // year
                d.getMonth(),    // month
                d.getDate(),     // day
                d.getHours() * 3600 + d.getMinutes() * 60 + d.getSeconds() + d.getMilliseconds() / 1000 // time in seconds
            );
        }
        
        // Set sample rate if available
        if (uniforms.iSampleRate && audioContext) {
            gl.uniform1f(uniforms.iSampleRate, audioContext.sampleRate);
        }
        
        // Update audio texture (Channel 0)
        updateAudioTexture();
        
        // Bind all textures to their proper targets
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, audioTexture);
        if (uniforms.iChannel0) gl.uniform1i(uniforms.iChannel0, 0);
        
        // Set up other channels (currently unused but available for future)
        for (let i = 1; i < 4; i++) {
            const channelUniform = uniforms['iChannel' + i];
            if (channelUniform) {
                gl.activeTexture(gl.TEXTURE0 + i);
                gl.bindTexture(gl.TEXTURE_2D, extraTextures[i-1]);
                gl.uniform1i(channelUniform, i);
            }
        }
        
        // Draw
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
    
    // Start render loop immediately
    animationRunning = true;
    startTime = performance.now() / 1000;
    lastTime = startTime;
    requestAnimationFrame(render);
    
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
    
    // Initial setup
    gl.clearColor(0.1, 0.1, 0.1, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (statusEl) statusEl.textContent = 'Ready - click Start Listen to begin';
    
    // === SHADERTOY INTEGRATION ===
    // Create the shader test UI if the ShaderConverter is available
    if (window.ShaderConverter) {
        const shaderUI = window.ShaderConverter.createShaderTestUI(document.body, (newShaderSource) => {
            // When user applies a new shader, set it up
            if (setupShaderProgram(vertexShaderSource, newShaderSource)) {
                uniforms = getShaderUniforms();
                if (statusEl) statusEl.textContent = 'New shader applied successfully';
                return true;
            }
            return false;
        });
        
        // Add helper functions to window for console testing
        window.visualizer = {
            applyShader: (fragShaderSource) => {
                if (setupShaderProgram(vertexShaderSource, fragShaderSource)) {
                    uniforms = getShaderUniforms();
                    return true;
                }
                return false;
            },
            
            loadFromShadertoy: async (shaderCode) => {
                try {
                    const convertedCode = window.ShaderConverter.convertShaderToyToWebGL(shaderCode);
                    if (convertedCode && setupShaderProgram(vertexShaderSource, convertedCode)) {
                        uniforms = getShaderUniforms();
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
                if (setupShaderProgram(vertexShaderSource, fragmentShaderSource)) {
                    uniforms = getShaderUniforms();
                    if (statusEl) statusEl.textContent = 'Restored default shader';
                    return true;
                }
                return false;
            }
        };
    } else {
        console.warn('ShaderConverter module not found. Shadertoy integration is disabled.');
    }
});
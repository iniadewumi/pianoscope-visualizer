// Main visualizer script
document.addEventListener('DOMContentLoaded', () => {
    // === CONFIGURATION ===
    // Easy to modify settings
    const audioSmoothingFactor = 0.8; // How smooth the audio response is (0-1)
    const audioSensitivity = 1.5;     // Amplify the audio signal
    
    // === ELEMENTS ===
    const canvas = document.getElementById('visualizer');
    const micToggle = document.getElementById('mic-toggle');
    const fullscreenBtn = document.getElementById('fullscreen');
    const statusEl = document.getElementById('status');
    
    // === AUDIO SETUP ===
    let audioContext;
    let analyser;
    let audioData;
    let micStream;
    let isListening = false;
    
    // === WEBGL SETUP ===
    const gl = canvas.getContext('webgl');
    if (!gl) {
        statusEl.textContent = 'WebGL not supported';
        return;
    }
    
    // Resize canvas to window
    function resizeCanvas() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        gl.viewport(0, 0, canvas.width, canvas.height);
    }
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
    
    // === SHADER CODE ===
    // This is the part you'll want to modify frequently
    // I've separated it for easy swapping
    const vertexShaderSource = `
        attribute vec2 position;
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
        }
    `;
    
    // Your fragment shader - easy to replace
    const fragmentShaderSource = `
        precision mediump float;
        
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
        
        // Color palette function
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
    
    // === WEBGL INITIALIZATION ===
    // Compile shader
    function createShader(gl, type, source) {
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            statusEl.textContent = 'Shader Error: ' + gl.getShaderInfoLog(shader);
            gl.deleteShader(shader);
            return null;
        }
        return shader;
    }
    
    // Initialize WebGL program
    const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);
    
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        statusEl.textContent = 'Program Error: ' + gl.getProgramInfoLog(program);
        return;
    }
    
    gl.useProgram(program);
    
    // Create fullscreen quad
    const positions = new Float32Array([
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,
         1.0,  1.0
    ]);
    
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    
    const positionLocation = gl.getAttribLocation(program, 'position');
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);
    
    // Setup uniforms
    const resolutionLocation = gl.getUniformLocation(program, 'iResolution');
    const timeLocation = gl.getUniformLocation(program, 'iTime');
    const channel0Location = gl.getUniformLocation(program, 'iChannel0');
    
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
            
            audioData = new Uint8Array(analyser.frequencyBinCount);
            isListening = true;
            
            statusEl.textContent = 'Listening to audio';
            micToggle.textContent = 'Stop Mic';
            
            // Start animation if not already running
            if (!animationRunning) {
                startTime = performance.now() / 1000;
                animationRunning = true;
                requestAnimationFrame(render);
            }
            
        } catch (err) {
            statusEl.textContent = 'Mic Error: ' + err.message;
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
        micToggle.textContent = 'Start Mic';
        statusEl.textContent = 'Audio stopped';
        
        // Clear audio data
        audioData.fill(0);
        updateAudioTexture();
    }
    
    function updateAudioTexture() {
        if (isListening && analyser) {
            analyser.getByteFrequencyData(audioData);
            
            // Apply sensitivity
            for (let i = 0; i < audioData.length; i++) {
                audioData[i] = Math.min(255, audioData[i] * audioSensitivity);
            }
        }
        
        gl.bindTexture(gl.TEXTURE_2D, audioTexture);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, audioTexWidth, 1, gl.LUMINANCE, gl.UNSIGNED_BYTE, audioData);
    }
    
    // === RENDER LOOP ===
    let startTime = performance.now() / 1000;
    let animationRunning = false;
    
    function render(now) {
        requestAnimationFrame(render);
        
        // Update time uniform
        const currentTime = now / 1000 - startTime;
        gl.uniform1f(timeLocation, currentTime);
        
        // Update resolution uniform
        gl.uniform2f(resolutionLocation, canvas.width, canvas.height);
        
        // Update audio texture
        updateAudioTexture();
        
        // Activate texture
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, audioTexture);
        gl.uniform1i(channel0Location, 0);
        
        // Draw
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
    
    // Start render loop
    animationRunning = true;
    startTime = performance.now() / 1000;
    requestAnimationFrame(render);
    
    // === EVENT HANDLERS ===
    micToggle.addEventListener('click', () => {
        if (isListening) {
            stopMic();
        } else {
            startMic();
        }
    });
    
    fullscreenBtn.addEventListener('click', () => {
        if (canvas.requestFullscreen) {
            canvas.requestFullscreen();
        } else if (canvas.webkitRequestFullscreen) {
            canvas.webkitRequestFullscreen();
        } else if (canvas.mozRequestFullScreen) {
            canvas.mozRequestFullScreen();
        }
    });
});
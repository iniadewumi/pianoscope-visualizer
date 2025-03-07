/**
 * Piansoscope Visualizer
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
    
    // === AUDIO SETUP ===
    let audioContext;
    let analyser;
    let audioData;
    let micStream;
    let isListening = false;
    let seconds = 0;
    const loud_volume_threshold = 30;
    let frequencyArray;
    
    // === WEBGL SETUP ===
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    if (!gl) {
        statusEl.textContent = 'WebGL not supported in your browser';
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
    
    // === WEBGL INITIALIZATION ===
    // Compile shader
    function createShader(gl, type, source) {
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            statusEl.textContent = 'Shader Error: ' + gl.getShaderInfoLog(shader);
            console.error('Shader compile error:', gl.getShaderInfoLog(shader));
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
        console.error('Program link error:', gl.getProgramInfoLog(program));
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
    const mouseLocation = gl.getUniformLocation(program, 'iMouse');
    
    // Mouse tracking
    let mouseX = 0;
    let mouseY = 0;
    let mouseDown = 0;
    
    canvas.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        mouseX = e.clientX - rect.left;
        mouseY = rect.height - (e.clientY - rect.top); // Flip Y for WebGL
    });
    
    canvas.addEventListener('mousedown', () => {
        mouseDown = 1;
    });
    
    canvas.addEventListener('mouseup', () => {
        mouseDown = 0;
    });
    
    canvas.addEventListener('mouseleave', () => {
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
            
            statusEl.textContent = 'Listening to audio';
            h.textContent = 'Listening...';
            micToggle.innerHTML = '<span class="fa fa-stop"></span>Stop Listen';
            micToggle.className = "red-button";
            
            // Start volume monitoring
            showVolume();
            
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
        micToggle.innerHTML = '<span class="fa fa-play"></span>Start Listen';
        micToggle.className = "green-button";
        statusEl.textContent = 'Audio stopped';
        h.textContent = 'Please allow the use of your microphone.';
        hSub.textContent = '';
        
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
            
            h.innerHTML = Math.floor(dbValue) + " dB";
            
            if (dbValue >= loud_volume_threshold) {
                seconds += 0.5;
                if (seconds >= 5) {
                    hSub.innerHTML = "You've been in loud environment for<span> " + Math.floor(seconds) + " </span>seconds.";
                }
            } else {
                seconds = 0;
                hSub.innerHTML = "";
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
        
        // Update time uniform
        const currentTime = now / 1000 - startTime;
        gl.uniform1f(timeLocation, currentTime);
        
        // Update resolution uniform
        gl.uniform2f(resolutionLocation, canvas.width, canvas.height);
        
        // Update mouse uniform
        gl.uniform4f(mouseLocation, mouseX, mouseY, mouseDown, 0.0);
        
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
        } else if (canvas.msRequestFullscreen) {
            canvas.msRequestFullscreen();
        }
    });
    
    // Initial setup
    gl.clearColor(0.1, 0.1, 0.1, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    statusEl.textContent = 'Ready - click Start Listen to begin';
});
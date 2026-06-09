/**
 * Piansoscope Visualizer with Shadertoy Support
 * 
 * This script handles the WebGL and audio processing to visualize 
 * microphone input in real-time with shader effects.
 */

document.addEventListener('DOMContentLoaded', () => {
    "use strict";
// Add this at the top where other global variables are declared
let globalTextures = [];

// Use a CORS-friendly URL for textures
const DEFAULT_TEXTURE_URL = "https://picsum.photos/800/600";

// Add this function to your visualizer-with-shadertoy.js
function setupTextures(gl, program) {
    // Create texture objects for iChannel0 and iChannel1
    // We'll use the same texture for both for simplicity
    const textures = [];
    const textureCount = 4; // Shadertoy supports up to 4 texture channels
    
    for (let i = 0; i < textureCount; i++) {
        textures[i] = gl.createTexture();
        gl.activeTexture(gl[`TEXTURE${i}`]);
        gl.bindTexture(gl.TEXTURE_2D, textures[i]);
        
        // Set up temporary pixel while we load
        const pixel = new Uint8Array([128, 128, 128, 255]); // mid-gray
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
        
        // Set texture parameters for proper sampling
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        
        // Set texture uniform
        const location = gl.getUniformLocation(program, `iChannel${i}`);
        if (location) {
            gl.uniform1i(location, i);
        }
    }
    
    // Load the texture into all channels
    loadTextureIntoChannels(gl, textures, DEFAULT_TEXTURE_URL);
    
    // Save textures globally for use in render loop
    globalTextures = textures;
    
    return textures;
}

function loadTextureIntoChannels(gl, textures, url) {
    const image = new Image();
    image.crossOrigin = "anonymous"; // Important for CORS
    
    image.onload = function() {
        console.log(`Image loaded successfully: ${url}`);
        console.log(`Dimensions: ${image.width}x${image.height}`);
        
        // Load the same image into all texture channels
        for (let i = 0; i < textures.length; i++) {
            gl.activeTexture(gl[`TEXTURE${i}`]);
            gl.bindTexture(gl.TEXTURE_2D, textures[i]);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
            
            // Only generate mipmap if power-of-two dimensions
            if (isPowerOf2(image.width) && isPowerOf2(image.height)) {
                gl.generateMipmap(gl.TEXTURE_2D);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
                console.log(`Generated mipmaps for texture ${i}`);
            } else {
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
                console.log(`Using CLAMP_TO_EDGE for non-power-of-two texture ${i}`);
            }
            console.log(`Texture ${i} set up completely`);
        }
    };
    
    image.onerror = function(e) {
        console.error(`Failed to load texture from ${url}`);
        console.error(e);
    };
    
    console.log(`Starting to load image from: ${url}`);
    image.src = url;
}

function isPowerOf2(value) {
    return (value & (value - 1)) === 0;
}

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
    gl.deleteFramebuffer(state.fbo);
    state.textures.forEach((texture) => gl.deleteTexture(texture));
}

function clearFeedbackTextures(gl, state) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, state.fbo);
    gl.viewport(0, 0, state.width, state.height);
    gl.clearColor(0, 0, 0, 0);
    state.textures.forEach((texture) => {
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
    });
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
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
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    const isWebGL2 = gl instanceof WebGL2RenderingContext;
    const glslVersion = isWebGL2 ? '300 es' : '100';
    
    // Initialize video controller after GL context is created
    let videoController = null;
    if (gl && typeof VideoController !== 'undefined') {
        videoController = new VideoController(gl);
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
    let noiseTexture = null;
    let feedbackState = null;
    
    // Resize canvas to window
    function resizeCanvas() {
        // Check if in pure view mode
        const isPureViewMode = document.body.classList.contains('pure-view-mode');
        
        // Set appropriate height
        const canvasHeight = isPureViewMode ? window.innerHeight : window.innerHeight * 0.8;
        
        const dpr = window.devicePixelRatio || 1;
        canvas.width = window.innerWidth * dpr;
        canvas.height = canvasHeight * dpr;
        gl.viewport(0, 0, canvas.width, canvas.height);

        if (usesFeedbackPass) {
            setupFeedbackBuffers(canvas.width, canvas.height, true);
        }
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
    // Modified setupShaderProgram function

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

    window.currentVertexShader = currentVertexShader;

    window.currentFragmentShader = currentFragmentShader; 

    

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

    

    // *** ADD TEXTURE SETUP HERE ***

    setupTextures(gl, currentProgram);

    

    // Reset frame counter

    frameCount = 0;

    

    // Get all uniforms after successful program setup

    uniforms = getShaderUniforms();
    configureFeedbackPass(fragmentSource);

    

    if (window.kioskWindow && !window.kioskWindow.closed) {

        window.kioskWindow.postMessage({

            type: 'shader-update',

            vertexShader: vertexSource,

            fragmentShader: fragmentSource

        }, '*');

    }

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
            iSampleRate: gl.getUniformLocation(currentProgram, 'iSampleRate'),
            iWriteFeedback: gl.getUniformLocation(currentProgram, 'iWriteFeedback')
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

    function setupFeedbackBuffers(width, height, preserveOnResize) {
        if (feedbackState && feedbackState.width === width && feedbackState.height === height) {
            return;
        }

        destroyFeedbackState(gl, feedbackState);

        const textures = [gl.createTexture(), gl.createTexture()];
        textures.forEach((texture) => {
            gl.bindTexture(gl.TEXTURE_2D, texture);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        });

        const fbo = gl.createFramebuffer();
        feedbackState = { textures, fbo, readIndex: 0, width, height };

        if (!preserveOnResize) {
            clearFeedbackTextures(gl, feedbackState);
        }
    }

    function configureFeedbackPass(fragmentSource) {
        usesFeedbackPass = fragmentSource.includes('iWriteFeedback');
        if (usesFeedbackPass) {
            if (!noiseTexture) {
                noiseTexture = createNoiseTexture(gl);
            }
            setupFeedbackBuffers(canvas.width, canvas.height, false);
        } else {
            destroyFeedbackState(gl, feedbackState);
            feedbackState = null;
        }
    }

    function bindTextureChannels(isVideoMode, feedbackReadTexture) {
        if (usesFeedbackPass && noiseTexture) {
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, noiseTexture);
            if (uniforms.iChannel0) gl.uniform1i(uniforms.iChannel0, 0);

            gl.activeTexture(gl.TEXTURE1);
            gl.bindTexture(gl.TEXTURE_2D, feedbackReadTexture || extraTextures[0]);
            if (uniforms.iChannel1) gl.uniform1i(uniforms.iChannel1, 1);

            updateAudioTexture();
            gl.activeTexture(gl.TEXTURE2);
            gl.bindTexture(gl.TEXTURE_2D, audioTexture);
            if (uniforms.iChannel2) gl.uniform1i(uniforms.iChannel2, 2);

            const channel3Uniform = uniforms.iChannel3;
            if (channel3Uniform) {
                gl.activeTexture(gl.TEXTURE3);
                gl.bindTexture(gl.TEXTURE_2D, extraTextures[2]);
                gl.uniform1i(channel3Uniform, 3);
            }
            return;
        }

        if (isVideoMode && videoController) {
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, videoController.getVideoTexture());
            if (uniforms.iChannel0) gl.uniform1i(uniforms.iChannel0, 0);
        } else {
            updateAudioTexture();
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, audioTexture);
            if (uniforms.iChannel0) gl.uniform1i(uniforms.iChannel0, 0);
        }

        for (let i = 1; i < 4; i++) {
            const channelUniform = uniforms['iChannel' + i];
            if (channelUniform) {
                gl.activeTexture(gl.TEXTURE0 + i);
                gl.bindTexture(gl.TEXTURE_2D, extraTextures[i - 1]);
                gl.uniform1i(channelUniform, i);
            }
        }
    }

    function setShaderUniforms(currentTime) {
        if (uniforms.iResolution) gl.uniform2f(uniforms.iResolution, canvas.width, canvas.height);
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
        micToggle.innerHTML = '<i class="fas fa-microphone"></i><span>Start Listen</span>';
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
        window.audioData = audioData;
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
            
            
            if (h) h.innerHTML = Math.floor(dbValue) + " <span style='color:#FFD700;'>dB</span>";

            
            if (dbValue >= loud_volume_threshold) {
                seconds += 0.5;
                if (seconds >= 5) {
                    // if (hSub) hSub.innerHTML = "You've been in loud environment for<span> " + Math.floor(seconds) + " </span>seconds.";
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
    
    // Modified render function with video support
    function render(now) {
        if (!animationRunning) return;
        
        requestAnimationFrame(render);
        
        // Calculate time values
        const nowSec = now / 1000;
        const currentTime = nowSec - startTime;
        deltaTime = nowSec - lastTime;
        lastTime = nowSec;
        frameCount++;
        
        // Check if we're in video mode
        const isVideoMode = videoController && videoController.isInVideoMode && videoController.isInVideoMode();
        
        // If in video mode, update the video texture and use the video display shader
        if (isVideoMode) {
            // Update video texture
            videoController.updateVideoTexture();
            
            // If we need to switch to the video display shader, do it
            if (currentProgram && !window.currentShaderIsVideoDisplay) {
                // Save the current shader for returning later
                window.savedShaderProgram = currentProgram;
                window.savedVertexShader = currentVertexShader;
                window.savedFragmentShader = currentFragmentShader;
                
                // Apply the video display shader
                setupShaderProgram(vertexShaderSource, videoController.getVideoDisplayShader());
                window.currentShaderIsVideoDisplay = true;
            }
        } else if (window.currentShaderIsVideoDisplay) {
            // If leaving video mode, restore the previous shader
            if (window.savedShaderProgram) {
                currentProgram = window.savedShaderProgram;
                currentVertexShader = window.savedVertexShader;
                currentFragmentShader = window.savedFragmentShader;
                gl.useProgram(currentProgram);
                window.currentShaderIsVideoDisplay = false;
                
                // Re-get uniforms
                uniforms = getShaderUniforms();
            }
        }
        
        // Set uniform values for shader if program exists
        if (!currentProgram) return;

        setShaderUniforms(currentTime);

        if (usesFeedbackPass && feedbackState) {
            const readTexture = feedbackState.textures[feedbackState.readIndex];
            const writeIndex = 1 - feedbackState.readIndex;
            const writeTexture = feedbackState.textures[writeIndex];

            gl.bindFramebuffer(gl.FRAMEBUFFER, feedbackState.fbo);
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, writeTexture, 0);
            gl.viewport(0, 0, feedbackState.width, feedbackState.height);
            bindTextureChannels(false, readTexture);
            if (uniforms.iWriteFeedback) gl.uniform1f(uniforms.iWriteFeedback, 1.0);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            gl.viewport(0, 0, canvas.width, canvas.height);
            bindTextureChannels(false, readTexture);
            if (uniforms.iWriteFeedback) gl.uniform1f(uniforms.iWriteFeedback, 0.0);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

            feedbackState.readIndex = writeIndex;
        } else {
            bindTextureChannels(isVideoMode);
            if (uniforms.iWriteFeedback) gl.uniform1f(uniforms.iWriteFeedback, 0.0);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        }
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
    // Add this code to your visualizer-with-shadertoy.js file
// Place it in the "=== EVENT HANDLERS ===" section, after the existing event listeners

// === KEYBOARD SHADER NAVIGATION ===
// -1 = default shader (Pianoscope Text), outside the numbered cycle
const DEFAULT_SHADER_NAME = 'Pianoscope Text';
// Pinned slots: shader 2, 3, 4 (0-based indices 1, 2, 3)
const PINNED_SHADER_SLOTS = [
    null,
    'Sunset on river -  Raph',
    'Ocean Waves',
    'Sunset Train',
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

function applyDefaultShader() {
    if (window.ShaderConverter && window.visualizer) {
        const shaderSource = window.ShaderConverter.SAMPLE_SHADERS[DEFAULT_SHADER_NAME];
        if (shaderSource) {
            try {
                const convertedShader = window.ShaderConverter.convertShaderToyToWebGL(shaderSource);
                const success = window.visualizer.applyShader(convertedShader);
                if (success) {
                    currentShaderIndex = -1;
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
    if (setupShaderProgram(vertexShaderSource, fragmentShaderSource)) {
        uniforms = getShaderUniforms();
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
        const success = window.visualizer.applyShader(convertedShader);

        if (success) {
            if (statusEl) {
                statusEl.textContent = `Shader: ${shaderName} (${currentShaderIndex + 1}/${shaderKeys.length})`;
            }

            const shaderTextarea = document.querySelector('.shader-editor-textarea');
            if (shaderTextarea) {
                shaderTextarea.value = shaderSource;
            }

            console.log(`Applied shader: ${shaderName}`);
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
                const success = window.visualizer.applyShader(convertedShader);
                if (success) {
                    currentShaderIndex = index;
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
/**
 * ShaderConverter - A utility for converting Shadertoy shaders to WebGL
 *
 * This module handles the conversion of Shadertoy-style fragment shaders to
 * standard WebGL shaders, and provides a modern UI for testing shaders.
 */

// Ensure SHADERS and SHADERS2 are correctly imported if they exist in separate files
// Assuming they are exported correctly from shaders.js and shaders2.js
import { SHADERS } from './shaders.js';
import { SHADERS2 } from './shaders2.js'; // Assuming shaders2.js exists and exports SHADERS2

// Combine the shader objects
const SAMPLE_SHADERS = { ...SHADERS, ...SHADERS2 };

window.ShaderConverter = (function () {
    "use strict";

    // Define WebGL preambles and main function structure
    const WEBGL_PREAMBLE = `
precision highp float;

// Version compatibility layer
#if __VERSION__ >= 300
    #define attribute in
    #define varying out
    #define texture2D texture
    #define textureCube texture
    #define texture2DLod textureLod
    #define textureCubeLod textureLod
    #define texture2DProj textureProj
    #define texture2DProjLod textureProjLod
#else
    #define in varying
    #define out varying
#endif

uniform vec2 iResolution;
uniform float iTime;
uniform float iTimeDelta;
uniform float iFrame;
uniform float iSampleRate;
uniform vec4 iMouse;
uniform vec4 iDate;
uniform sampler2D iChannel0;
uniform sampler2D iChannel1;
uniform sampler2D iChannel2;
uniform sampler2D iChannel3;
`;

    const WEBGL_MAIN = `
#if __VERSION__ >= 300
void main() {
    mainImage(fragColor, gl_FragCoord.xy);
}
#else
void main() {
    vec4 fragColor;
    mainImage(fragColor, gl_FragCoord.xy);
    gl_FragColor = fragColor;
}
#endif
`;

    /**
     * Converts a Shadertoy fragment shader to WebGL format
     * @param {string} shadertoyCode - The Shadertoy shader code
     * @return {string} - The converted WebGL shader code
     */
    function convertShaderToyToWebGL(shadertoyCode) {
        if (!shadertoyCode) {
            throw new Error("No shader code provided");
        }

        // Check if code already contains a main function
        if (shadertoyCode.includes("void main()")) {
            // Already has a main function, could be a regular WebGL shader
            return shadertoyCode;
        }

        // Check if it's a Shadertoy shader (should have mainImage function)
        if (!shadertoyCode.includes("void mainImage(")) {
            throw new Error("Not a valid Shadertoy shader (missing mainImage function)");
        }

        // Replace texture with texture2D which is needed for WebGL 1.0
        let convertedCode = shadertoyCode.replace(/\btexture\(/g, 'texture2D(');

        // Combine the shader parts
        return WEBGL_PREAMBLE + convertedCode + WEBGL_MAIN;
    }

    /**
     * Creates a shader test UI with sample shaders and editor
     * @param {HTMLElement} container - Container to append the UI to
     * @param {Function} onShaderApply - Callback when a shader is applied
     * @return {Object} - The UI elements
     */
    function createShaderTestUI(container, onShaderApply) {

        // Create UI elements
        const uiContainer = document.createElement('div');
        uiContainer.className = 'shader-test-ui';

        // Create header
        const header = document.createElement('div');
        header.className = 'shader-editor-header';
        const title = document.createElement('h3');
        title.textContent = 'Shader Editor';
        header.appendChild(title);
        uiContainer.appendChild(header); // Header still belongs to main container

        // Create tab system
        const tabContainer = document.createElement('div');
        tabContainer.className = 'tab-container';
        const shaderTabBtn = document.createElement('button');
        shaderTabBtn.className = 'tab-button active';
        shaderTabBtn.dataset.tab = 'shader';
        shaderTabBtn.textContent = 'Shader';
        const videoTabBtn = document.createElement('button');
        videoTabBtn.className = 'tab-button';
        videoTabBtn.dataset.tab = 'video';
        videoTabBtn.textContent = 'Video';
        tabContainer.appendChild(shaderTabBtn);
        tabContainer.appendChild(videoTabBtn);
        uiContainer.appendChild(tabContainer); // Tabs belong to main container

        // Create tab content containers
        const shaderTabContent = document.createElement('div');
        shaderTabContent.className = 'tab-content';
        shaderTabContent.id = 'shader-tab';

        const videoTabContent = document.createElement('div');
        videoTabContent.className = 'tab-content hidden';
        videoTabContent.id = 'video-tab';

        // --- Elements to be placed inside shaderTabContent ---
        const controls = document.createElement('div');
        controls.className = 'shader-editor-controls';
        const applyButton = document.createElement('button');
        applyButton.innerHTML = '<i class="fa fa-check"></i> Apply Shader';
        applyButton.className = 'shader-editor-button apply-button';
        const resetButton = document.createElement('button');
        resetButton.innerHTML = '<i class="fa fa-refresh"></i> Reset';
        resetButton.className = 'shader-editor-button reset-button';
        controls.appendChild(applyButton);
        controls.appendChild(resetButton);

        const sampleContainer = document.createElement('div');
        sampleContainer.className = 'sample-container';
        // Add sample shader buttons
        for (const [name, code] of Object.entries(SAMPLE_SHADERS)) {
            const sampleButton = document.createElement('button');
            sampleButton.textContent = name;
            sampleButton.className = 'sample-button';
            sampleButton.addEventListener('click', () => {
                sampleButton.style.transform = 'scale(0.95)';
                setTimeout(() => { sampleButton.style.transform = ''; }, 150);
                shaderTextarea.value = code;
                showMessage('Sample shader loaded: ' + name, 'info');
            });
            sampleContainer.appendChild(sampleButton);
        }

        const shaderTextarea = document.createElement('textarea');
        shaderTextarea.className = 'shader-editor-textarea';
        shaderTextarea.placeholder = 'Paste Shadertoy shader code here...';
        // Use a consistent default shader name, handle potential missing key
        const defaultShaderName = "Precision Plasma Flower";
        const defaultShader = SAMPLE_SHADERS[defaultShaderName] || Object.values(SAMPLE_SHADERS)[0] || ''; // Added fallback
        shaderTextarea.value = defaultShader;

        const errorDisplay = document.createElement('div');
        errorDisplay.className = 'error-display';
        // --- End elements for shaderTabContent ---

        // Append shader-specific elements to shaderTabContent (THIS IS THE FIX)
        shaderTabContent.appendChild(controls);
        shaderTabContent.appendChild(sampleContainer);
        shaderTabContent.appendChild(shaderTextarea);
        shaderTabContent.appendChild(errorDisplay);

        // --- Video Tab Content setup ---
        videoTabContent.innerHTML = `
        <div class="video-controls">
            <div class="video-source-section">
                <h4>Video Source</h4>
                <div class="video-source-controls">
                    <button class="video-source-button" id="local-video-btn">
                        <i class="fas fa-file-upload"></i> Upload Video
                    </button>
                    <input type="file" id="video-file-input" accept="video/*" style="display: none">

                    <div class="url-input-container">
                        <input type="text" id="video-url-input" placeholder="Enter video URL...">
                        <button class="video-url-load-btn" id="load-url-btn">Load</button>
                    </div>
                </div>
            </div>

            <div class="sample-videos-section">
                <h4>Sample Videos</h4>
                <div class="sample-videos-container">
                    <button class="sample-video-button" data-video="videos/sample.mp4">Sample</button>
                    <button class="sample-video-button" data-video="videos/particles.mp4">Particles</button>
                    <button class="sample-video-button" data-video="videos/neon.mp4">Neon Lights</button>
                    <button class="sample-video-button" data-video="videos/geometric.mp4">Geometric</button>
                </div>
            </div>

            <div class="video-playback-controls">
                <button id="play-pause-btn"><i class="fas fa-play"></i></button>
                <div class="seek-bar-container">
                    <input type="range" id="seek-bar" min="0" max="100" value="0" step="0.1">
                    <div class="time-display">
                        <span id="current-time">0:00</span> / <span id="duration">0:00</span>
                    </div>
                </div>
                <div class="volume-control">
                    <button id="mute-btn"><i class="fas fa-volume-up"></i></button>
                    <input type="range" id="volume-bar" min="0" max="1" value="1" step="0.01">
                </div>
            </div>

            <div class="video-options">
                <label>
                    <input type="checkbox" id="loop-video" checked>
                    Loop Video
                </label>
                <div class="playback-speed">
                    <span>Speed: </span>
                    <select id="playback-speed">
                        <option value="0.5">0.5x</option>
                        <option value="1" selected>1x</option>
                        <option value="1.5">1.5x</option>
                        <option value="2">2x</option>
                    </select>
                </div>
            </div>
        </div>
        `;
        // --- End Video Tab Content setup ---

        // Add the content divs to the main UI container
        uiContainer.appendChild(shaderTabContent);
        uiContainer.appendChild(videoTabContent);

        // Create toggle button
        const toggleButton = document.createElement('button');
        toggleButton.innerHTML = '<i class="fas fa-code"></i>';
        toggleButton.className = 'toggle-shader-ui';
        toggleButton.title = "Toggle Shader Editor";

        // Add the toggle button and the main UI container to the document body (or specified container)
        container.appendChild(toggleButton);
        container.appendChild(uiContainer);

        // --- Tab Switching Functionality ---
        const tabButtons = uiContainer.querySelectorAll('.tab-button');
        const tabContents = uiContainer.querySelectorAll('.tab-content');

        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                // Deactivate all tabs
                tabButtons.forEach(btn => btn.classList.remove('active'));
                tabContents.forEach(content => content.classList.add('hidden'));

                // Activate selected tab
                button.classList.add('active');
                const tabId = button.dataset.tab + '-tab';
                const tabContent = document.getElementById(tabId);
                if (tabContent) {
                    tabContent.classList.remove('hidden');
                }

                // Notify controller about mode change explicitly if necessary
                // Check if videoController exists and has a method to set mode
                if (typeof videoController !== 'undefined' && videoController && typeof videoController.setVideoMode === 'function') {
                     videoController.setVideoMode(button.dataset.tab === 'video');
                }
            });
        });

        // Function to show status messages with animation
        function showMessage(message, type = 'success') {
            // Ensure errorDisplay exists before manipulating it
            if (!errorDisplay) return;

            errorDisplay.textContent = message;

            // Set color based on message type
            if (type === 'error') {
                errorDisplay.style.borderLeftColor = '#E83A5F'; errorDisplay.style.color = '#FF4D6D';
            } else if (type === 'success') {
                errorDisplay.style.borderLeftColor = '#4CAF50'; errorDisplay.style.color = '#8BC34A';
            } else if (type === 'info') {
                errorDisplay.style.borderLeftColor = '#2196F3'; errorDisplay.style.color = '#64B5F6';
            }

            // Animate in
            errorDisplay.style.display = 'block';
            errorDisplay.style.opacity = '0';
            errorDisplay.style.transform = 'translateY(-10px)';
            // Use requestAnimationFrame for smoother animation start
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    errorDisplay.style.opacity = '1'; errorDisplay.style.transform = 'translateY(0)';
                });
            });


            // Auto hide after delay
            setTimeout(() => {
                errorDisplay.style.opacity = '0'; errorDisplay.style.transform = 'translateY(10px)';
                setTimeout(() => { errorDisplay.style.display = 'none'; }, 300); // Wait for fade out transition
            }, 3000);
        }

        // Add event listener for toggle button
        toggleButton.addEventListener('click', () => {
            uiContainer.classList.toggle('open');
            toggleButton.classList.toggle('active');
            toggleButton.innerHTML = uiContainer.classList.contains('open') ? '<i class="fas fa-times"></i>' : '<i class="fas fa-code"></i>';
        });

        // Add event listener for apply button
        applyButton.addEventListener('click', () => {
            try {
                const shadertoyCode = shaderTextarea.value;
                const convertedShader = convertShaderToyToWebGL(shadertoyCode);
                applyButton.style.transform = 'scale(0.95)';
                setTimeout(() => { applyButton.style.transform = ''; }, 150);

                if (convertedShader && onShaderApply) {
                    const success = onShaderApply(convertedShader);
                    showMessage(success ? 'Shader applied successfully!' : 'Error applying shader', success ? 'success' : 'error');
                }
            } catch (error) {
                showMessage('Error: ' + error.message, 'error');
                console.error('Shader error:', error);
            }
        });

        // Add event listener for reset button
        resetButton.addEventListener('click', () => {
            resetButton.style.transform = 'scale(0.95)';
            setTimeout(() => { resetButton.style.transform = ''; }, 150);

            let resetSuccess = false;
            // Ensure window.visualizer and resetToDefault exist before calling
            if (typeof window.visualizer !== 'undefined' && window.visualizer && typeof window.visualizer.resetToDefault === 'function') {
                resetSuccess = window.visualizer.resetToDefault();
            }

            showMessage(resetSuccess ? 'Reset to default shader' : 'Error resetting shader', resetSuccess ? 'success' : 'error');

            // Update textarea content after attempting reset
            // Check if getDefaultFragmentShaderSource exists, otherwise use fallback
             let defaultFragSource = '';
             if (typeof window.visualizer !== 'undefined' && window.visualizer && typeof window.visualizer.getDefaultFragmentShaderSource === 'function') {
                  defaultFragSource = window.visualizer.getDefaultFragmentShaderSource();
             }
             // If function doesn't exist or returns empty, use the hardcoded fallback
             if (!defaultFragSource) {
                 defaultFragSource = SAMPLE_SHADERS[defaultShaderName] || Object.values(SAMPLE_SHADERS)[0] || '';
             }
            shaderTextarea.value = defaultFragSource;

        });

        // Add keyboard shortcut support for textarea
        shaderTextarea.addEventListener('keydown', (e) => {
            // Use `code` property for modern browsers, fallback to `key` or `keyCode`
            const key = e.code || e.key;
            if (e.ctrlKey && (key === 'Enter' || key === 'NumpadEnter')) {
                e.preventDefault();
                applyButton.click();
            }
        });

        // Return UI elements for reference
        return {
            container: uiContainer,
            textarea: shaderTextarea,
            applyButton: applyButton,
            resetButton: resetButton,
            errorDisplay: errorDisplay,
            toggle: toggleButton,
            showMessage: showMessage,
            toggleVisibility: () => { // Expose toggle function if needed externally
                uiContainer.classList.toggle('open');
                toggleButton.classList.toggle('active');
                toggleButton.innerHTML = uiContainer.classList.contains('open') ? '<i class="fas fa-times"></i>' : '<i class="fas fa-code"></i>';
            }
        };
    }

    // Public API
    return {
        convertShaderToyToWebGL: convertShaderToyToWebGL,
        createShaderTestUI: createShaderTestUI,
        SAMPLE_SHADERS: SAMPLE_SHADERS // Expose sample shaders if needed elsewhere
    };
})();
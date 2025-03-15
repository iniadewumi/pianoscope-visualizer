/**
 * ShaderConverter - A utility for converting Shadertoy shaders to WebGL
 * 
 * This module handles the conversion of Shadertoy-style fragment shaders to
 * standard WebGL shaders, and provides a modern UI for testing shaders.
 */

import { SAMPLE_SHADERS } from './shaders.js';

window.ShaderConverter = (function() {
    "use strict";
    
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
        
        uiContainer.appendChild(header);

        // Create control buttons
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
        uiContainer.appendChild(controls);
        // Create sample buttons container
        const sampleContainer = document.createElement('div');
        sampleContainer.className = 'sample-container';
        
        // Add sample shader buttons
        for (const [name, code] of Object.entries(SAMPLE_SHADERS)) {
            const sampleButton = document.createElement('button');
            sampleButton.textContent = name;
            sampleButton.className = 'sample-button';
            sampleButton.addEventListener('click', () => {
                // Add subtle animation when selecting a sample
                sampleButton.style.transform = 'scale(0.95)';
                setTimeout(() => {
                    sampleButton.style.transform = '';
                }, 150);
                
                shaderTextarea.value = code;
                
                // Show feedback
                showMessage('Sample shader loaded: ' + name, 'info');
            });
            sampleContainer.appendChild(sampleButton);
        }
        
        uiContainer.appendChild(sampleContainer);
        
        // Create textarea for shader code
        const shaderTextarea = document.createElement('textarea');
        shaderTextarea.className = 'shader-editor-textarea';
        shaderTextarea.placeholder = 'Paste Shadertoy shader code here...';
        
        // Set default shader - pick a visually interesting one
        const defaultShader = SAMPLE_SHADERS["Precision Plasma Flower"] || 
                             Object.values(SAMPLE_SHADERS)[0];
        shaderTextarea.value = defaultShader;
        
        uiContainer.appendChild(shaderTextarea);
        
        
        // Create error display
        const errorDisplay = document.createElement('div');
        errorDisplay.className = 'error-display';
        uiContainer.appendChild(errorDisplay);
        
        // Create toggle button
        const toggleButton = document.createElement('button');
        toggleButton.innerHTML = '<i class="fas fa-code"></i>';
        toggleButton.className = 'toggle-shader-ui';
        toggleButton.title = "Toggle Shader Editor";

        // Add elements to container
        container.appendChild(toggleButton);
        container.appendChild(uiContainer);

        // Function to show status messages with animation
        function showMessage(message, type = 'success') {
            errorDisplay.textContent = message;
            
            // Set color based on message type
            if (type === 'error') {
                errorDisplay.style.borderLeftColor = '#E83A5F';
                errorDisplay.style.color = '#FF4D6D';
            } else if (type === 'success') {
                errorDisplay.style.borderLeftColor = '#4CAF50';
                errorDisplay.style.color = '#8BC34A';
            } else if (type === 'info') {
                errorDisplay.style.borderLeftColor = '#2196F3';
                errorDisplay.style.color = '#64B5F6';
            }
            
            // Animate in
            errorDisplay.style.display = 'block';
            errorDisplay.style.opacity = '0';
            errorDisplay.style.transform = 'translateY(-10px)';
            
            setTimeout(() => {
                errorDisplay.style.opacity = '1';
                errorDisplay.style.transform = 'translateY(0)';
            }, 10);
            
            // Auto hide after delay
            setTimeout(() => {
                errorDisplay.style.opacity = '0';
                errorDisplay.style.transform = 'translateY(10px)';
                
                setTimeout(() => {
                    errorDisplay.style.display = 'none';
                }, 300);
            }, 3000);
        }

        // Add event listener for toggle with animation
        toggleButton.addEventListener('click', () => {
            // Toggle open class for animation
            uiContainer.classList.toggle('open');
            toggleButton.classList.toggle('active');
            
            // Update button icon based on sidebar state
            if (uiContainer.classList.contains('open')) {
                toggleButton.innerHTML = '<i class="fas fa-times"></i>';
            } else {
                toggleButton.innerHTML = '<i class="fas fa-code"></i>';
            }
        });
                
        applyButton.addEventListener('click', () => {
            try {
                const shadertoyCode = shaderTextarea.value;
                const convertedShader = convertShaderToyToWebGL(shadertoyCode);
                
                // Add button press animation
                applyButton.style.transform = 'scale(0.95)';
                setTimeout(() => {
                    applyButton.style.transform = '';
                }, 150);
                
                if (convertedShader && onShaderApply) {
                    const success = onShaderApply(convertedShader);
                    if (success) {
                        showMessage('Shader applied successfully!', 'success');
                    } else {
                        showMessage('Error applying shader', 'error');
                    }
                }
            } catch (error) {
                showMessage('Error: ' + error.message, 'error');
                console.error('Shader error:', error);
            }
        });
        
        resetButton.addEventListener('click', () => {
            // Add button press animation
            resetButton.style.transform = 'scale(0.95)';
            setTimeout(() => {
                resetButton.style.transform = '';
            }, 150);
            
            // Reset to default shader
            if (window.visualizer && window.visualizer.resetToDefault) {
                const success = window.visualizer.resetToDefault();
                if (success) {
                    showMessage('Reset to default shader', 'success');
                } else {
                    showMessage('Error resetting shader', 'error');
                }
            }
        });
        
        // Add keyboard shortcut support
        shaderTextarea.addEventListener('keydown', (e) => {
            // Ctrl+Enter to apply shader
            if (e.ctrlKey && e.key === 'Enter') {
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
            toggleVisibility: () => {
                uiContainer.classList.toggle('open');
                toggleButton.classList.toggle('active');
                
                if (uiContainer.classList.contains('open')) {
                    toggleButton.innerHTML = '<i class="fa fa-times"></i>';
                } else {
                    toggleButton.innerHTML = '<i class="fa fa-code"></i>';
                }
            }
        };
    }
    
    // Public API
    return {
        convertShaderToyToWebGL: convertShaderToyToWebGL,
        createShaderTestUI: createShaderTestUI,
        SAMPLE_SHADERS: SAMPLE_SHADERS
    };
})();
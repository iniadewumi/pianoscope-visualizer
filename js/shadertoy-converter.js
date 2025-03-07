/**
 * ShaderConverter - A utility for converting Shadertoy shaders to WebGL
 * 
 * This module handles the conversion of Shadertoy-style fragment shaders to
 * standard WebGL shaders, and provides a simple UI for testing shaders.
 */

import { SAMPLE_SHADERS } from './shaders.js';

window.ShaderConverter = (function() {
    "use strict";
    
    // Default preamble to add to converted Shadertoy shaders
    const WEBGL_PREAMBLE = `
    precision highp float;
    precision highp int;
    precision mediump sampler2D;
    
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
    
    // Helper compatibility functions for texelFetch
    vec4 texelFetch(sampler2D sampler, ivec2 pos, int lod) {
        return texture2D(sampler, (vec2(pos) + vec2(0.5)) / vec2(256.0));
    }
    
    vec4 texelFetch(sampler2D sampler, int pos, int lod) {
        return texture2D(sampler, vec2(float(pos) / 256.0, 0.0));
    }
    `;
    
    // Post-shader code to replace Shadertoy's mainImage function call
    const WEBGL_MAIN = `
    void main() {
        vec4 fragColor;
        mainImage(fragColor, gl_FragCoord.xy);
        gl_FragColor = fragColor;
    }
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
    
    SAMPLE_SHADERS;
    
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
        title.textContent = 'Shadertoy Editor';
        header.appendChild(title);
        
        uiContainer.appendChild(header);
        
        // Create sample buttons container
        const sampleContainer = document.createElement('div');
        sampleContainer.className = 'sample-container';
        
        // Add sample shader buttons
        for (const [name, code] of Object.entries(SAMPLE_SHADERS)) {
            const sampleButton = document.createElement('button');
            sampleButton.textContent = name;
            sampleButton.className = 'sample-button';
            sampleButton.addEventListener('click', () => {
                shaderTextarea.value = code;
            });
            sampleContainer.appendChild(sampleButton);
        }
        
        uiContainer.appendChild(sampleContainer);
        
        // Create textarea for shader code
        const shaderTextarea = document.createElement('textarea');
        shaderTextarea.className = 'shader-editor-textarea';
        shaderTextarea.placeholder = 'Paste Shadertoy shader code here...';
        shaderTextarea.value = SAMPLE_SHADERS["Simple Color Cycle"]; // Default shader
        uiContainer.appendChild(shaderTextarea);
        
        // Create control buttons
        const controls = document.createElement('div');
        controls.className = 'shader-editor-controls';
        
        const applyButton = document.createElement('button');
        applyButton.textContent = 'Apply Shader';
        applyButton.className = 'shader-editor-button apply-button';
        
        const resetButton = document.createElement('button');
        resetButton.textContent = 'Reset to Default';
        resetButton.className = 'shader-editor-button reset-button';
        
        controls.appendChild(applyButton);
        controls.appendChild(resetButton);
        uiContainer.appendChild(controls);
        
        // Create error display
        const errorDisplay = document.createElement('div');
        errorDisplay.className = 'error-display';
        uiContainer.appendChild(errorDisplay);
        
        // Create toggle button
        const toggleButton = document.createElement('button');
        toggleButton.textContent = 'Shader Editor';
        toggleButton.className = 'toggle-shader-ui';
        
        // Add elements to container
        container.appendChild(toggleButton);
        container.appendChild(uiContainer);
        let uiVisible = true;
    
        // Initialize UI as visible
        uiContainer.style.transform = 'translateX(0)';
        toggleButton.textContent = 'Hide Editor';
        toggleButton.classList.add('active');
        
        // Add event listeners
        toggleButton.addEventListener('click', () => {
            uiVisible = !uiVisible;
            
            if (uiVisible) {
                uiContainer.style.transform = 'translateX(0)';
                toggleButton.textContent = 'Hide Editor';
                toggleButton.classList.add('active');
            } else {
                uiContainer.style.transform = 'translateX(420px)';
                toggleButton.textContent = 'Shader Editor';
                toggleButton.classList.remove('active');
            }
        });
        
        applyButton.addEventListener('click', () => {
            try {
                const shadertoyCode = shaderTextarea.value;
                const convertedShader = convertShaderToyToWebGL(shadertoyCode);
                
                if (convertedShader && onShaderApply) {
                    const success = onShaderApply(convertedShader);
                    errorDisplay.textContent = 'Shader applied successfully!';
                    errorDisplay.style.color = '#55ff55';
                    errorDisplay.style.display = 'block';
                    
                    // Hide success message after a few seconds
                    setTimeout(() => {
                        errorDisplay.style.display = 'none';
                    }, 3000);
                }
            } catch (error) {
                errorDisplay.textContent = 'Error: ' + error.message;
                errorDisplay.style.color = '#ff5555';
                errorDisplay.style.display = 'block';
                console.error('Shader error:', error);
            }
        });
        
        resetButton.addEventListener('click', () => {
            // Reset to default shader
            if (window.visualizer && window.visualizer.resetToDefault) {
                const success = window.visualizer.resetToDefault();
                errorDisplay.textContent = 'Reset to default shader';
                errorDisplay.style.color = '#55ff55';
                errorDisplay.style.display = 'block';
                
                // Hide message after a few seconds
                setTimeout(() => {
                    errorDisplay.style.display = 'none';
                }, 3000);
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
            toggleVisibility: () => {
                uiVisible = !uiVisible;
                uiContainer.style.transform = uiVisible ? 'translateX(0)' : 'translateX(420px)';
                toggleButton.textContent = uiVisible ? 'Hide Editor' : 'Shader Editor';
                toggleButton.classList.toggle('active', uiVisible);
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
/**
 * Simple Video Implementation for WebGL Visualizer
 * 
 * This code adds video texture support to the existing WebGL visualizer.
 * Add this to your project as js/video-texture.js
 */

// Create a self-executing function to avoid global scope pollution
(function() {
    // Initialize when DOM is ready
    document.addEventListener('DOMContentLoaded', function() {
        // Create video element
        const video = document.createElement('video');
        video.autoplay = true;
        video.loop = true;
        video.muted = true; // Required for autoplay
        video.playsInline = true; // For iOS support
        
        // Set video source - replace with your video file
        video.src = 'videos/sample.mp4';
        
        // Get WebGL context from existing canvas
        const canvas = document.getElementById('visualizer');
        const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
        
        if (!gl) {
            console.error('WebGL not supported');
            return;
        }
        
        // Determine if using WebGL2
        const isWebGL2 = gl instanceof WebGL2RenderingContext;
        
        // Create video texture
        const videoTexture = gl.createTexture();
        
        // Function to initialize the video texture
        function initVideoTexture() {
            // Bind to texture unit 1 (assuming texture unit 0 is used for audio)
            gl.activeTexture(gl.TEXTURE1);
            gl.bindTexture(gl.TEXTURE_2D, videoTexture);
            
            // Set texture parameters
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            
            // Upload a blank pixel initially
            const pixel = new Uint8Array([0, 0, 0, 255]);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
        }
        
        // Initialize the video texture
        initVideoTexture();
        
        // Create a simple shader for video display
        const simpleVideoShader = `
            precision highp float;
            
            uniform vec2 iResolution;
            uniform float iTime;
            uniform sampler2D iChannel0; // Audio data texture
            uniform sampler2D iChannel1; // Video texture
            
            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                // Normalized coordinates
                vec2 uv = fragCoord/iResolution.xy;
                
                // Sample the video texture
                vec4 videoColor = texture2D(iChannel1, uv);
                
                // Sample audio data
                float bass = texture2D(iChannel0, vec2(0.05, 0.0)).x;
                float mids = texture2D(iChannel0, vec2(0.3, 0.0)).x;
                float high = texture2D(iChannel0, vec2(0.7, 0.0)).x;
                
                // Beat-reactive parameters
                float beatIntensity = smoothstep(0.15, 0.4, bass) * 0.3;
                
                // Create a subtle displacement effect based on audio
                vec2 distortedUV = uv;
                distortedUV.x += sin(uv.y * 10.0 + iTime) * 0.01 * mids;
                distortedUV.y += cos(uv.x * 10.0 + iTime) * 0.01 * high;
                
                // Sample the video with distorted coordinates
                vec4 distortedVideo = texture2D(iChannel1, distortedUV);
                
                // Blend between normal and distorted based on bass
                vec4 finalVideo = mix(videoColor, distortedVideo, beatIntensity);
                
                // Add a subtle color shift based on high frequencies
                finalVideo.r += high * 0.1;
                finalVideo.b += mids * 0.05;
                
                // Add a pulsing vignette effect
                float vignette = length(uv - 0.5) * (1.2 + beatIntensity);
                finalVideo.rgb *= 1.0 - vignette * 0.7;
                
                // Output the final color
                fragColor = finalVideo;
            }
        `;
        
        // Add main function appropriate for WebGL version
        const finalVideoShader = isWebGL2 ? 
            simpleVideoShader :  // For WebGL2, the mainImage is sufficient as it will be handled by the converter
            simpleVideoShader + `
            void main() {
                vec4 fragColor;
                mainImage(fragColor, gl_FragCoord.xy);
                gl_FragColor = fragColor;
            }`;
        
        // Store original applyShader function to wrap it
        if (window.visualizer && window.visualizer.applyShader) {
            const originalApplyShader = window.visualizer.applyShader;
            
            // Override the applyShader function to add video uniform
            window.visualizer.applyShader = function(fragShaderSource) {
                // If source doesn't already include iChannel1, add it
                if (!fragShaderSource.includes('uniform sampler2D iChannel1')) {
                    fragShaderSource = 'uniform sampler2D iChannel1;\n' + fragShaderSource;
                }
                
                // Call original applyShader
                const result = originalApplyShader(fragShaderSource);
                
                // If successful, set up video texture
                if (result && window.currentProgram) {
                    const videoUniform = gl.getUniformLocation(window.currentProgram, 'iChannel1');
                    if (videoUniform) {
                        gl.uniform1i(videoUniform, 1); // Use texture unit 1
                    }
                }
                
                return result;
            };
        }
        
        // Update the video texture each frame
        function updateVideoTexture() {
            // Only update if video is ready and playing
            if (video.readyState >= 2 && !video.paused && !video.ended) {
                gl.activeTexture(gl.TEXTURE1);
                gl.bindTexture(gl.TEXTURE_2D, videoTexture);
                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);
            }
            
            // Continue updating
            requestAnimationFrame(updateVideoTexture);
        }
        
        // Start playback when video is loaded
        video.addEventListener('canplaythrough', function() {
            video.play()
                .then(() => {
                    console.log('Video playback started');
                    // Start updating texture
                    updateVideoTexture();
                    
                    // Apply our simple video shader
                    if (window.visualizer && window.visualizer.applyShader) {
                        window.visualizer.applyShader(finalVideoShader);
                    }
                })
                .catch(err => {
                    console.error('Video playback failed:', err);
                    // Try with user interaction
                    const startButton = document.createElement('button');
                    startButton.textContent = 'Start Video';
                    startButton.style.position = 'absolute';
                    startButton.style.top = '10px';
                    startButton.style.left = '10px';
                    startButton.style.zIndex = '1000';
                    document.body.appendChild(startButton);
                    
                    startButton.addEventListener('click', function() {
                        video.play().then(() => {
                            // Apply our simple video shader
                            if (window.visualizer && window.visualizer.applyShader) {
                                window.visualizer.applyShader(finalVideoShader);
                            }
                            startButton.remove();
                            updateVideoTexture();
                        });
                    });
                });
        });
        
        // Add a simple way to toggle video on/off
        const toggleVideoButton = document.createElement('button');
        toggleVideoButton.textContent = 'Toggle Video';
        toggleVideoButton.style.position = 'absolute';
        toggleVideoButton.style.top = '10px';
        toggleVideoButton.style.right = '80px';
        toggleVideoButton.style.zIndex = '1000';
        document.body.appendChild(toggleVideoButton);
        
        toggleVideoButton.addEventListener('click', function() {
            if (video.paused) {
                video.play();
                toggleVideoButton.textContent = 'Pause Video';
            } else {
                video.pause();
                toggleVideoButton.textContent = 'Play Video';
            }
        });
    });
})();
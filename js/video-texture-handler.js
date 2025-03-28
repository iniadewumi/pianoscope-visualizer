/**
 * Video Texture Handler for Pianoscope Visualizer
 * 
 * This module manages loading and playing videos as WebGL textures
 * for integration with the existing visualizer.
 */

class VideoTextureHandler {
    constructor(gl) {
        // Store the WebGL context
        this.gl = gl;
        
        // Create video element
        this.videoElement = document.createElement('video');
        this.videoElement.muted = true;
        this.videoElement.playsInline = true;
        this.videoElement.loop = true;
        this.videoElement.style.display = 'none';
        document.body.appendChild(this.videoElement);
        
        // Video state
        this.isPlaying = false;
        this.isPaused = true;
        this.isMuted = true;
        this.videoLoaded = false;
        this.playbackSpeed = 1.0;
        
        // Create WebGL texture for video
        this.videoTexture = null;
        this.setupVideoTexture();
        
        // Track when we're in video mode
        this.videoMode = false;
        
        // Bind event listeners for video element
        this.bindEvents();
    }
    
    setupVideoTexture() {
        this.videoTexture = this.gl.createTexture();
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.videoTexture);
        
        // Set texture parameters
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.LINEAR);
        
        // Create an initial black texture
        const blackPixel = new Uint8Array([0, 0, 0, 255]);
        this.gl.texImage2D(
            this.gl.TEXTURE_2D, 0, this.gl.RGBA, 
            1, 1, 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE, blackPixel
        );
    }
    
    bindEvents() {
        this.videoElement.addEventListener('loadedmetadata', () => {
            this.videoLoaded = true;
            
            // Notify any listeners that video metadata is loaded
            const event = new CustomEvent('videoMetadataLoaded', {
                detail: {
                    duration: this.videoElement.duration,
                    width: this.videoElement.videoWidth,
                    height: this.videoElement.videoHeight
                }
            });
            document.dispatchEvent(event);
        });
        
        this.videoElement.addEventListener('timeupdate', () => {
            // Notify any listeners of time updates
            const event = new CustomEvent('videoTimeUpdate', {
                detail: {
                    currentTime: this.videoElement.currentTime,
                    duration: this.videoElement.duration,
                    percentage: (this.videoElement.currentTime / this.videoElement.duration) * 100
                }
            });
            document.dispatchEvent(event);
        });
        
        this.videoElement.addEventListener('ended', () => {
            if (!this.videoElement.loop) {
                this.pause();
                document.dispatchEvent(new Event('videoEnded'));
            }
        });
        
        this.videoElement.addEventListener('play', () => {
            this.isPlaying = true;
            this.isPaused = false;
            document.dispatchEvent(new Event('videoPlaying'));
        });
        
        this.videoElement.addEventListener('pause', () => {
            this.isPlaying = false;
            this.isPaused = true;
            document.dispatchEvent(new Event('videoPaused'));
        });
        
        this.videoElement.addEventListener('error', (e) => {
            console.error('Video error:', this.videoElement.error);
            
            const event = new CustomEvent('videoError', {
                detail: {
                    error: this.videoElement.error
                }
            });
            document.dispatchEvent(event);
        });
    }
    
    loadVideo(url) {
        // Reset state
        this.videoLoaded = false;
        this.isPlaying = false;
        this.isPaused = true;

        // Update video source
        this.videoElement.src = url;
        this.videoElement.load();
        this.play();

        // Enter video mode
        this.setVideoMode(true);
    }

    
    loadVideoFile(file) {
        if (file && file.type.startsWith('video/')) {
            const fileURL = URL.createObjectURL(file);
            this.loadVideo(fileURL);
        } else {
            console.error('Invalid file type. Please select a video file.');
            
            const event = new CustomEvent('videoError', {
                detail: {
                    error: 'Invalid file type. Please select a video file.'
                }
            });
            document.dispatchEvent(event);
        }
    }
    
    updateVideoTexture() {
        if (this.videoMode && this.videoLoaded && this.isPlaying && 
            this.videoElement.readyState >= this.videoElement.HAVE_CURRENT_DATA) {
            
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.videoTexture);
            this.gl.texImage2D(
                this.gl.TEXTURE_2D, 0, this.gl.RGBA, this.gl.RGBA, 
                this.gl.UNSIGNED_BYTE, this.videoElement
            );
        }
    }
    
    play() {
        if (this.videoElement.readyState < this.videoElement.HAVE_METADATA) {
             console.log("Video metadata not ready yet, play() deferred.");
             return;
        }
        // *** END ADDITION ***

        if (this.videoLoaded || this.videoElement.readyState >= this.videoElement.HAVE_METADATA) { // Modified check
            this.videoElement.play()
                .then(() => {
                    this.isPlaying = true;
                    this.isPaused = false;
                    this.playWhenReady = false;
                })
                .catch(error => {
                    console.error('Error playing video:', error);

                    if (error.name === 'NotAllowedError' && !this.isMuted) {
                        console.log('Autoplay blocked, attempting mute and retry...');
                        this.videoElement.muted = true;
                        this.isMuted = true;
                        this.play();
                    } else {
                       this.playWhenReady = false; 
                    }
                });
        } else {
             console.log("Play called but video not loaded/ready.");
             this.playWhenReady = true;
        }
    }
    
    pause() {
        if (this.videoLoaded && this.isPlaying) {
            this.videoElement.pause();
            this.isPlaying = false;
            this.isPaused = true;
        }
    }
    
    togglePlay() {
        if (this.isPaused) {
            this.play();
        } else {
            this.pause();
        }
    }
    
    setTime(time) {
        if (this.videoLoaded) {
            const clampedTime = Math.max(0, Math.min(time, this.videoElement.duration));
            this.videoElement.currentTime = clampedTime;
        }
    }
    
    setProgress(percentage) {
        if (this.videoLoaded) {
            const time = (percentage / 100) * this.videoElement.duration;
            this.setTime(time);
        }
    }
    
    setVolume(volume) {
        if (this.videoLoaded) {
            this.videoElement.volume = Math.max(0, Math.min(1, volume));
            
            // Update mute state based on volume
            if (volume === 0) {
                this.isMuted = true;
                this.videoElement.muted = true;
            } else if (this.isMuted) {
                this.isMuted = false;
                this.videoElement.muted = false;
            }
        }
    }
    
    toggleMute() {
        this.isMuted = !this.isMuted;
        this.videoElement.muted = this.isMuted;
    }
    
    setPlaybackSpeed(speed) {
        if (this.videoLoaded) {
            this.playbackSpeed = speed;
            this.videoElement.playbackRate = speed;
        }
    }
    
    setLoop(shouldLoop) {
        this.videoElement.loop = shouldLoop;
    }
    
    setVideoMode(enabled) {
        this.videoMode = enabled;
        
        // If disabling video mode, pause the video to save resources
        if (!enabled && this.isPlaying) {
            this.pause();
        }
    }
    
    getVideoTexture() {
        return this.videoTexture;
    }
    
    hasVideo() {
        return this.videoLoaded;
    }
    
    dispose() {
        // Pause and unload video
        this.videoElement.pause();
        this.videoElement.src = '';
        this.videoElement.load();
        
        // Remove video element from DOM
        if (this.videoElement.parentNode) {
            this.videoElement.parentNode.removeChild(this.videoElement);
        }
        
        // Delete WebGL texture
        if (this.videoTexture) {
            this.gl.deleteTexture(this.videoTexture);
            this.videoTexture = null;
        }
    }
}
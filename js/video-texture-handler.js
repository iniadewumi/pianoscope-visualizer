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
        // Required for R2 / cross-origin clips sampled into WebGL textures.
        this.videoElement.crossOrigin = 'anonymous';
        this.videoElement.style.display = 'none';
        document.body.appendChild(this.videoElement);
        
        // Video state
        this.isPlaying = false;
        this.isPaused = true;
        this.isMuted = true;
        this.videoLoaded = false;
        this.playbackSpeed = 1.0;
        this.currentSrc = null;

        // Native pixel dimensions of the loaded clip, needed to letterbox or crop
        // it against a canvas of a different shape.
        this.videoWidth = 0;
        this.videoHeight = 0;

        // Object URLs created for uploaded files, revoked when replaced.
        this.objectUrl = null;

        // Live camera feed, when the source is a MediaStream rather than a file.
        this.stream = null;

        // HTML5 video drops currentTime assignments while a seek is in flight.
        // Keep the latest tap and apply it when the previous seek settles.
        this.pendingSeekTime = null;
        this.seekInFlight = false;
        this.seekWatchdog = null;
        
        // Create WebGL texture for video
        this.videoTexture = null;
        this.setupVideoTexture();
        
        // Track when we're in video mode
        this.videoMode = false;

        // Uploading a 4K frame is the most expensive thing per render tick, and
        // a 24fps clip has nothing new to give a 60fps loop most of the time.
        this.hasNewFrame = true;
        this.watchFrames();
        
        // Bind event listeners for video element
        this.bindEvents();
    }

    watchFrames() {
        if (typeof this.videoElement.requestVideoFrameCallback !== 'function') return;

        const onFrame = () => {
            this.hasNewFrame = true;
            this.videoElement.requestVideoFrameCallback(onFrame);
        };
        this.videoElement.requestVideoFrameCallback(onFrame);
        this.usingFrameCallback = true;
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
            this.videoWidth = this.videoElement.videoWidth;
            this.videoHeight = this.videoElement.videoHeight;
            
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
        
        this.videoElement.addEventListener('seeked', () => {
            this.clearSeekWatchdog();
            this.seekInFlight = false;
            this.flushSeek();
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
    
    resetPlaybackState() {
        this.videoLoaded = false;
        this.isPlaying = false;
        this.isPaused = true;
        this.videoWidth = 0;
        this.videoHeight = 0;
        this.pendingSeekTime = null;
        this.seekInFlight = false;
        this.clearSeekWatchdog();
        this.hasNewFrame = true;
    }

    /**
     * Detach whatever is currently feeding the element. A live srcObject takes
     * precedence over src, so it has to be cleared before a file will load, and
     * its tracks must be stopped or the camera light stays on.
     */
    releaseSource(nextUrl) {
        if (this.stream) {
            this.stream.getTracks().forEach((track) => track.stop());
            this.stream = null;
        }
        if (this.videoElement.srcObject) {
            this.videoElement.srcObject = null;
        }
        if (this.objectUrl && this.objectUrl !== nextUrl) {
            URL.revokeObjectURL(this.objectUrl);
            this.objectUrl = null;
        }
    }

    loadVideo(url, options = {}) {
        this.resetPlaybackState();
        this.currentSrc = options.sourceId || url;

        this.releaseSource(url);
        if (options.isObjectUrl) {
            this.objectUrl = url;
        }

        // Update video source
        this.videoElement.src = url;
        this.videoElement.load();
        this.play();

        // Enter video mode
        this.setVideoMode(true);
    }

    /**
     * Feed the pipeline from a live MediaStream (camera) instead of a file. The
     * texture path downstream is identical; only seeking and duration differ.
     */
    loadStream(stream, sourceId = 'camera') {
        this.resetPlaybackState();
        this.currentSrc = sourceId;

        this.releaseSource();
        this.stream = stream;

        this.videoElement.removeAttribute('src');
        this.videoElement.srcObject = stream;
        this.play();

        this.setVideoMode(true);
    }

    isLive() {
        return !!this.stream;
    }

    
    loadVideoFile(file) {
        if (file && file.type.startsWith('video/')) {
            const fileURL = URL.createObjectURL(file);
            this.loadVideo(fileURL, { isObjectUrl: true, sourceId: `upload:${file.name}` });
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

            // Without frame callbacks there is no way to tell a repeat frame from
            // a fresh one, so fall back to uploading every tick.
            if (this.usingFrameCallback && !this.hasNewFrame) return;
            this.hasNewFrame = false;

            this.gl.bindTexture(this.gl.TEXTURE_2D, this.videoTexture);
            this.gl.texImage2D(
                this.gl.TEXTURE_2D, 0, this.gl.RGBA, this.gl.RGBA,
                this.gl.UNSIGNED_BYTE, this.videoElement
            );
        }
    }
    
    play() {
        // If video isn't ready yet, set a flag to play when ready
        if (this.videoElement.readyState < this.videoElement.HAVE_METADATA) {
            console.log("Video metadata not ready yet, play() deferred.");
            this.playWhenReady = true;
            
            // Add one-time event listener to play when metadata loads
            this.videoElement.addEventListener('loadedmetadata', () => {
                if (this.playWhenReady) {
                    console.log("Metadata loaded, starting playback");
                    this.playWhenReady = false;
                    this.play();
                }
            }, { once: true });
            
            return;
        }
    
        // Now we know video is loaded or metadata is ready
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
        if (!this.videoLoaded) return;

        const duration = this.videoElement.duration;
        if (!Number.isFinite(duration) || duration <= 0) return;

        this.pendingSeekTime = Math.max(0, Math.min(time, duration));
        if (!this.seekInFlight) this.flushSeek();
    }

    flushSeek() {
        if (this.pendingSeekTime === null) return;

        const next = this.pendingSeekTime;
        this.pendingSeekTime = null;

        if (Math.abs(this.videoElement.currentTime - next) < 0.04) return;

        this.seekInFlight = true;
        this.videoElement.currentTime = next;

        this.clearSeekWatchdog();
        this.seekWatchdog = setTimeout(() => {
            this.seekInFlight = false;
            this.flushSeek();
        }, 400);
    }

    clearSeekWatchdog() {
        if (this.seekWatchdog) {
            clearTimeout(this.seekWatchdog);
            this.seekWatchdog = null;
        }
    }

    isSeeking() {
        return this.seekInFlight || this.pendingSeekTime !== null;
    }

    setProgress(percentage) {
        if (!this.videoLoaded) return;
        const duration = this.videoElement.duration;
        if (!Number.isFinite(duration) || duration <= 0) return;
        this.setTime((percentage / 100) * duration);
    }
    
    setVolume(volume) {
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
    
    toggleMute() {
        this.isMuted = !this.isMuted;
        this.videoElement.muted = this.isMuted;
    }
    
    setPlaybackSpeed(speed) {
        this.playbackSpeed = speed;
        this.videoElement.playbackRate = speed;
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

    /**
     * Native dimensions of the loaded clip. Falls back to a 1:1 square so
     * shaders never divide by zero before metadata arrives.
     */
    getVideoSize() {
        return {
            width: this.videoWidth || 1,
            height: this.videoHeight || 1
        };
    }
    
    hasVideo() {
        return this.videoLoaded;
    }
    
    dispose() {
        this.clearSeekWatchdog();
        this.pendingSeekTime = null;
        this.seekInFlight = false;

        // Pause and unload video
        this.videoElement.pause();
        this.releaseSource();
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
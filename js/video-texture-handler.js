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

        // WebGL2 lets us allocate texture storage once (texStorage2D) and
        // update it in place per frame (texSubImage2D) instead of
        // re-specifying/reallocating the texture on every upload.
        this.isWebGL2 = (typeof WebGL2RenderingContext !== 'undefined') && gl instanceof WebGL2RenderingContext;

        // Size of the currently allocated texture storage (WebGL2 path only).
        // 0 means nothing has been allocated yet.
        this.texWidth = 0;
        this.texHeight = 0;

        // Create video element
        this.videoElement = document.createElement('video');
        this.videoElement.muted = true;
        this.videoElement.playsInline = true;
        this.videoElement.loop = true;
        // Required for R2 / cross-origin clips sampled into WebGL textures.
        this.videoElement.crossOrigin = 'anonymous';
        this.videoElement.style.display = 'none';
        document.body.appendChild(this.videoElement);

        // Still images share the same WebGL texture slot as video frames.
        this.imageElement = document.createElement('img');
        this.imageElement.crossOrigin = 'anonymous';
        this.imageElement.alt = '';
        this.imageElement.style.display = 'none';
        document.body.appendChild(this.imageElement);
        this.still = false;
        
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

        // True from the moment a clip or stream is handed to the element until
        // unload(). Metadata arrives later, so this is what "has a source" means.
        this.attached = false;

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
        this.still = false;
        this.clearImage();
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
        if (!this.still) this.clearImage();
    }

    clearImage() {
        if (!this.imageElement) return;
        this.imageElement.onload = null;
        this.imageElement.onerror = null;
        this.imageElement.removeAttribute('src');
        this.still = false;
    }

    loadVideo(url, options = {}) {
        this.resetPlaybackState();
        this.currentSrc = options.sourceId || url;
        this.still = false;

        this.releaseSource(url);
        if (options.isObjectUrl) {
            this.objectUrl = url;
        }

        // Update video source
        this.videoElement.src = url;
        this.videoElement.load();
        this.attached = true;
        this.play();

        // Enter video mode
        this.setVideoMode(true);
    }

    /**
     * Feed the pipeline from a still image. Same texture slot as a clip; the
     * frame is uploaded once and held until the source changes.
     */
    loadImage(url, options = {}) {
        this.resetPlaybackState();
        this.currentSrc = options.sourceId || url;

        this.releaseSource(url);
        if (options.isObjectUrl) {
            this.objectUrl = url;
        }

        this.videoElement.pause();
        this.videoElement.removeAttribute('src');
        this.videoElement.load();

        this.still = true;
        this.attached = true;
        this.setVideoMode(true);

        const img = this.imageElement;
        img.onload = () => {
            this.videoLoaded = true;
            this.videoWidth = img.naturalWidth;
            this.videoHeight = img.naturalHeight;
            this.isPlaying = true;
            this.isPaused = true;
            this.hasNewFrame = true;

            document.dispatchEvent(new CustomEvent('videoMetadataLoaded', {
                detail: {
                    duration: 0,
                    width: img.naturalWidth,
                    height: img.naturalHeight,
                    still: true
                }
            }));
        };
        img.onerror = () => {
            document.dispatchEvent(new CustomEvent('videoError', {
                detail: { error: 'Could not load that image.' }
            }));
        };
        img.crossOrigin = 'anonymous';
        img.src = url;
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
        this.attached = true;
        this.play();

        this.setVideoMode(true);
    }

    isLive() {
        return !!this.stream;
    }

    
    loadVideoFile(file) {
        if (file && file.type.startsWith('image/')) {
            const fileURL = URL.createObjectURL(file);
            this.loadImage(fileURL, { isObjectUrl: true, sourceId: `upload:${file.name}` });
            return;
        }

        if (file && file.type.startsWith('video/')) {
            const fileURL = URL.createObjectURL(file);
            this.loadVideo(fileURL, { isObjectUrl: true, sourceId: `upload:${file.name}` });
        } else {
            console.error('Invalid file type. Please select a video or image file.');
            
            const event = new CustomEvent('videoError', {
                detail: {
                    error: 'Invalid file type. Please select a video or image file.'
                }
            });
            document.dispatchEvent(event);
        }
    }

    uploadTexture(source, w, h) {
        const gl = this.gl;
        gl.bindTexture(gl.TEXTURE_2D, this.videoTexture);

        gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);

        if (this.isWebGL2) {
            if (this.texWidth !== w || this.texHeight !== h) {
                gl.deleteTexture(this.videoTexture);
                this.videoTexture = gl.createTexture();
                gl.bindTexture(gl.TEXTURE_2D, this.videoTexture);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
                gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, w, h);
                this.texWidth = w;
                this.texHeight = h;
            }

            try {
                gl.texSubImage2D(
                    gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, source
                );
            } catch (e) {
                this.hasNewFrame = true;
            }
        } else {
            try {
                gl.texImage2D(
                    gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA,
                    gl.UNSIGNED_BYTE, source
                );
            } catch (e) {
                this.hasNewFrame = true;
            }
        }

        gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.BROWSER_DEFAULT_WEBGL);
    }
    
    updateVideoTexture() {
        if (this.still) {
            if (!this.videoLoaded || !this.hasNewFrame) return;
            const w = this.imageElement.naturalWidth;
            const h = this.imageElement.naturalHeight;
            if (!w || !h) return;
            this.hasNewFrame = false;
            this.uploadTexture(this.imageElement, w, h);
            return;
        }

        if (this.videoLoaded && this.isPlaying &&
            this.videoElement.readyState >= this.videoElement.HAVE_CURRENT_DATA) {

            if (this.usingFrameCallback && !this.hasNewFrame) return;

            const w = this.videoElement.videoWidth;
            const h = this.videoElement.videoHeight;
            if (!w || !h) return;

            this.hasNewFrame = false;
            this.uploadTexture(this.videoElement, w, h);
        }
    }
    
    play() {
        if (this.still) return;

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
        if (this.still) return;
        if (this.isPaused) {
            this.play();
        } else {
            this.pause();
        }
    }
    
    setTime(time) {
        if (this.still || !this.videoLoaded) return;

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
        // Source (playback) and look (which program is on screen) are separate
        // now: this flag is purely informational bookkeeping for callers that
        // still read it. It must NOT pause playback — the source keeps running
        // and uploading regardless of which look is on screen.
        this.videoMode = enabled;
    }

    getVideoTexture() {
        return this.videoTexture;
    }

    /** True when a clip is loaded or a live stream (camera/screen) is attached. */
    hasSource() {
        return this.attached;
    }

    /**
     * Drop the current source entirely: stop stream tracks, clear the element,
     * and forget playback state. The texture keeps its last frame until the
     * next upload, but iVideoActive goes to 0 so nothing samples it.
     */
    unload() {
        this.videoElement.pause();
        this.releaseSource();
        this.clearImage();
        this.videoElement.removeAttribute('src');
        this.videoElement.load();
        this.resetPlaybackState();
        this.attached = false;
        this.currentSrc = null;
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
    
    isStill() {
        return this.still;
    }

    getTextureElement() {
        return this.still ? this.imageElement : this.videoElement;
    }

    hasVideo() {
        return this.videoLoaded && !this.still;
    }
    
    dispose() {
        this.clearSeekWatchdog();
        this.pendingSeekTime = null;
        this.seekInFlight = false;

        // Pause and unload video
        this.videoElement.pause();
        this.releaseSource();
        this.clearImage();
        this.videoElement.src = '';
        this.videoElement.load();
        
        // Remove video element from DOM
        if (this.videoElement.parentNode) {
            this.videoElement.parentNode.removeChild(this.videoElement);
        }
        if (this.imageElement && this.imageElement.parentNode) {
            this.imageElement.parentNode.removeChild(this.imageElement);
        }
        
        // Delete WebGL texture
        if (this.videoTexture) {
            this.gl.deleteTexture(this.videoTexture);
            this.videoTexture = null;
        }
    }
}
/**
 * Video Controller for Pianoscope Visualizer
 * 
 * This module manages the UI interactions for the video player functionality
 * and integrates with the video texture handler.
 */

class VideoController {

    constructor(gl) {
        // Create the video texture handler
        this.videoHandler = new VideoTextureHandler(gl);
        
        // Initialize once the UI is available
        this.initializeVideoUI();
        
        // Flag to track video mode
        this.isVideoMode = false;
        
        // Create video display shader
        this.videoDisplayShader = `
            precision highp float;
            
            uniform vec2 iResolution;
            uniform sampler2D iChannel0; // Video texture
            
            void main() {
                vec2 uv = gl_FragCoord.xy / iResolution.xy;
                
                // Flip Y coordinate for WebGL texture orientation
                uv.y = 1.0 - uv.y;
                
                // Sample video texture
                vec4 videoColor = texture2D(iChannel0, uv);
                
                // Output the color
                gl_FragColor = videoColor;
            }
        `;
    }
    
    /**
     * Initialize the video UI once it's available in the DOM
     */
    initializeVideoUI() {
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.setupUIControls());
        } else {
            // DOM already ready
            this.setupUIControls();
        }
    }
    
    /**
     * Set up UI controls for the video player
     */
    setupUIControls() {
        // Set up tab switching if not already done
        this.setupTabSwitching();
        
        // Cache UI elements
        this.cacheUIElements();
        
        // Bind UI events
        this.bindUIEvents();
        
        // Listen for video events from the handler
        this.setupVideoEventListeners();
    }
    
    /**
     * Ensure tab switching functionality works
     */
    setupTabSwitching() {
        const tabButtons = document.querySelectorAll('.tab-button');
        if (!tabButtons.length) return; // UI not ready yet
        
        tabButtons.forEach(button => {
            if (!button.hasAttribute('data-tab-initialized')) {
                button.setAttribute('data-tab-initialized', 'true');
                button.addEventListener('click', () => {
                    // Handle tab switching
                    const tabId = button.dataset.tab;
                    this.switchTab(tabId);
                    
                    // Update video mode based on tab
                    this.isVideoMode = (tabId === 'video');
                    this.videoHandler.setVideoMode(this.isVideoMode);
                });
            }
        });
    }
    
    /**
     * Switch the active tab
     */
    switchTab(tabId) {
        const tabButtons = document.querySelectorAll('.tab-button');
        const tabContents = document.querySelectorAll('.tab-content');
        
        // Deactivate all tabs
        tabButtons.forEach(btn => btn.classList.remove('active'));
        tabContents.forEach(content => content.classList.add('hidden'));
        
        // Activate selected tab
        const button = document.querySelector(`.tab-button[data-tab="${tabId}"]`);
        if (button) button.classList.add('active');
        
        const content = document.getElementById(`${tabId}-tab`);
        if (content) content.classList.remove('hidden');
    }
    
    /**
     * Cache UI element references
     */
    cacheUIElements() {
        // Video source controls
        this.localVideoBtn = document.getElementById('local-video-btn');
        this.videoFileInput = document.getElementById('video-file-input');
        this.videoUrlInput = document.getElementById('video-url-input');
        this.loadUrlBtn = document.getElementById('load-url-btn');
        this.sampleVideoButtons = document.querySelectorAll('.sample-video-button');
        
        // Playback controls
        this.playPauseBtn = document.getElementById('play-pause-btn');
        this.seekBar = document.getElementById('seek-bar');
        this.currentTimeEl = document.getElementById('current-time');
        this.durationEl = document.getElementById('duration');
        this.muteBtn = document.getElementById('mute-btn');
        this.volumeBar = document.getElementById('volume-bar');
        this.loopCheckbox = document.getElementById('loop-video');
        this.playbackSpeedSelect = document.getElementById('playback-speed');
    }
    
    /**
     * Bind UI event handlers
     */
    bindUIEvents() {
        // Only bind if elements exist (UI is ready)
        if (!this.localVideoBtn) return;
        
        // Video file selection
        this.localVideoBtn.addEventListener('click', () => {
            this.videoFileInput.click();
        });
        
        this.videoFileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                this.videoHandler.loadVideoFile(e.target.files[0]);
            }
        });
        
        // Video URL loading
        this.loadUrlBtn.addEventListener('click', () => {
            const url = this.videoUrlInput.value.trim();
            if (url) {
                this.videoHandler.loadVideo(url);
            }
        });
        
        // Also allow pressing Enter in the input field
        this.videoUrlInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.loadUrlBtn.click();
            }
        });
        
        // Sample video selection
        this.sampleVideoButtons.forEach(button => {
            button.addEventListener('click', () => {
                const videoUrl = button.dataset.video;
                if (videoUrl) {
                    this.videoHandler.loadVideo(videoUrl);
                }
            });
        });
        
        // Play/Pause button
        this.playPauseBtn.addEventListener('click', () => {
            this.videoHandler.togglePlay();
        });
        
        // Seek bar
        this.seekBar.addEventListener('input', () => {
            // Update time display during dragging
            const percentage = parseFloat(this.seekBar.value);
            if (this.videoHandler.hasVideo()) {
                const duration = this.videoHandler.videoElement.duration;
                const time = (percentage / 100) * duration;
                this.currentTimeEl.textContent = this.formatTime(time);
            }
        });
        
        this.seekBar.addEventListener('change', () => {
            // Set actual video time when slider is released
            const percentage = parseFloat(this.seekBar.value);
            this.videoHandler.setProgress(percentage);
        });
        
        // Mute button
        this.muteBtn.addEventListener('click', () => {
            this.videoHandler.toggleMute();
            this.updateMuteButtonUI();
        });
        
        // Volume control
        this.volumeBar.addEventListener('input', () => {
            const volume = parseFloat(this.volumeBar.value);
            this.videoHandler.setVolume(volume);
            this.updateMuteButtonUI();
        });
        
        // Loop checkbox
        this.loopCheckbox.addEventListener('change', () => {
            this.videoHandler.setLoop(this.loopCheckbox.checked);
        });
        
        // Playback speed
        this.playbackSpeedSelect.addEventListener('change', () => {
            const speed = parseFloat(this.playbackSpeedSelect.value);
            this.videoHandler.setPlaybackSpeed(speed);
        });
    }
    
    /**
     * Set up event listeners for video events
     */
    setupVideoEventListeners() {
        // Video metadata loaded
        document.addEventListener('videoMetadataLoaded', (e) => {
            const { duration } = e.detail;
            
            // Update duration display
            if (this.durationEl) {
                this.durationEl.textContent = this.formatTime(duration);
            }
            
            // Start playback
            this.videoHandler.play();
        });
        
        // Video time updates
        document.addEventListener('videoTimeUpdate', (e) => {
            const { currentTime, percentage } = e.detail;
            
            // Update current time display
            if (this.currentTimeEl) {
                this.currentTimeEl.textContent = this.formatTime(currentTime);
            }
            
            // Update seek bar position (only if not dragging)
            if (this.seekBar && !this.seekBar.matches(':active')) {
                this.seekBar.value = percentage;
            }
        });
        
        // Video playing state
        document.addEventListener('videoPlaying', () => {
            this.updatePlayPauseButtonUI(true);
        });
        
        // Video paused state
        document.addEventListener('videoPaused', () => {
            this.updatePlayPauseButtonUI(false);
        });
        
        // Video error
        document.addEventListener('videoError', (e) => {
            console.error('Video error:', e.detail.error);
            // Could display error in the UI
        });
    }
    
    /**
     * Update play/pause button UI based on playing state
     */
    updatePlayPauseButtonUI(isPlaying) {
        if (this.playPauseBtn) {
            if (isPlaying) {
                this.playPauseBtn.innerHTML = '<i class="fas fa-pause"></i>';
            } else {
                this.playPauseBtn.innerHTML = '<i class="fas fa-play"></i>';
            }
        }
    }
    
    /**
     * Update mute button UI based on mute state
     */
    updateMuteButtonUI() {
        if (this.muteBtn) {
            if (this.videoHandler.isMuted) {
                this.muteBtn.innerHTML = '<i class="fas fa-volume-mute"></i>';
            } else {
                this.muteBtn.innerHTML = '<i class="fas fa-volume-up"></i>';
            }
        }
    }
    
    /**
     * Format seconds into MM:SS format
     */
    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
    
    /**
     * Update the video texture in the render loop
     */
    updateVideoTexture() {
        if (this.isVideoMode) {
            this.videoHandler.updateVideoTexture();
        }
    }
    
    /**
     * Get the video texture for WebGL rendering
     */
    getVideoTexture() {
        return this.videoHandler.getVideoTexture();
    }
    
    /**
     * Get the video display shader
     */
    getVideoDisplayShader() {
        return this.videoDisplayShader;
    }
    
    /**
     * Check if we're in video mode
     */
    isInVideoMode() {
        return this.isVideoMode;
    }
    
    /**
     * Clean up resources when no longer needed
     */
    dispose() {
        this.videoHandler.dispose();
    }
}
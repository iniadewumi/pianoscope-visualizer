/**
 * Video Controller for Pianoscope Visualizer
 *
 * Renders the video library UI and drives the video texture handler. The canvas
 * is a full-screen quad, so fitting a clip of arbitrary shape into it is done in
 * the shader rather than by resizing anything.
 */

const VIDEO_FIT_MODES = { cover: 0, contain: 1, stretch: 2 };

class VideoController {

    constructor(gl) {
        this.videoHandler = new VideoTextureHandler(gl);

        this.isVideoMode = false;
        this.fitMode = 'contain';
        this.activeSrc = null;
        this.entries = [];
        this.thumbnailsRequested = new Set();
        this.isScrubbing = false;

        // Samples the video with an aspect-aware remap of the screen UVs. Without
        // this the 0..1 range maps straight onto the canvas and any clip whose
        // shape differs from the window gets stretched.
        this.videoDisplayShader = `
            precision highp float;

            uniform vec2 iResolution;
            uniform vec2 iVideoResolution;
            uniform float iFitMode;
            uniform sampler2D iChannel0;

            void main() {
                vec2 uv = gl_FragCoord.xy / iResolution.xy;

                float canvasAspect = iResolution.x / max(iResolution.y, 1.0);
                float videoAspect = iVideoResolution.x / max(iVideoResolution.y, 1.0);

                vec2 scale = vec2(1.0);
                if (iFitMode < 0.5) {
                    // Cover: fill the canvas and crop whichever axis overflows.
                    scale = canvasAspect > videoAspect
                        ? vec2(1.0, videoAspect / canvasAspect)
                        : vec2(canvasAspect / videoAspect, 1.0);
                } else if (iFitMode < 1.5) {
                    // Contain: keep the whole frame and pad the shorter axis.
                    scale = canvasAspect > videoAspect
                        ? vec2(canvasAspect / videoAspect, 1.0)
                        : vec2(1.0, videoAspect / canvasAspect);
                }

                uv = (uv - 0.5) * scale + 0.5;

                float inside = step(0.0, uv.x) * step(uv.x, 1.0)
                             * step(0.0, uv.y) * step(uv.y, 1.0);

                uv.y = 1.0 - uv.y;
                vec4 videoColor = texture2D(iChannel0, clamp(uv, 0.0, 1.0));

                gl_FragColor = vec4(videoColor.rgb * inside, 1.0);
            }
        `;

        this.setupVideoEventListeners();
    }

    /**
     * Build the library UI inside the supplied container. Called once the shader
     * editor has created the video tab.
     */
    mountUI(container) {
        if (!container || this.root) return;

        container.innerHTML = `
            <div class="vlib">
                <div class="vlib__toolbar">
                    <div class="vlib__search">
                        <i class="fas fa-search"></i>
                        <input type="text" id="video-search" placeholder="Search clips..." autocomplete="off">
                    </div>
                    <button class="vlib__icon-btn vlib__icon-btn--ghost" id="cache-all-btn" title="Download remote clips for offline playback">
                        <i class="fas fa-cloud-download-alt"></i>
                    </button>
                    <button class="vlib__icon-btn" id="local-video-btn" title="Upload a video">
                        <i class="fas fa-plus"></i>
                    </button>
                </div>

                <div class="vlib__url">
                    <input type="text" id="video-url-input" placeholder="Paste a video URL..." autocomplete="off">
                    <button id="load-url-btn">Load</button>
                </div>

                <div class="vlib__grid" id="video-grid"></div>
                <p class="vlib__status" id="video-status">Checking library...</p>

                <div class="vplayer" id="video-dock" hidden>
                    <div class="vplayer__head">
                        <span class="vplayer__title" id="video-now-playing">Nothing playing</span>
                        <button class="vplayer__exit" id="video-exit-btn" title="Return to shader">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>

                    <div class="vplayer__seek">
                        <div class="vplayer__seek-track" id="video-seek-wrap">
                            <input type="range" id="seek-bar" min="0" max="100" value="0" step="0.1">
                            <div class="vplayer__preview" id="video-preview" hidden>
                                <canvas id="video-preview-canvas" width="160" height="90"></canvas>
                                <span id="video-preview-time">0:00</span>
                            </div>
                        </div>
                        <div class="vplayer__times">
                            <span id="current-time">0:00</span>
                            <span id="duration">0:00</span>
                        </div>
                    </div>

                    <div class="vplayer__row">
                        <button id="play-pause-btn" class="vplayer__play"><i class="fas fa-play"></i></button>
                        <button id="mute-btn" class="vplayer__icon"><i class="fas fa-volume-up"></i></button>
                        <input type="range" id="volume-bar" min="0" max="1" value="1" step="0.01">
                        <button id="loop-video" class="vplayer__icon is-on" title="Loop">
                            <i class="fas fa-redo"></i>
                        </button>
                    </div>

                    <div class="vplayer__row vplayer__row--selects">
                        <label>
                            Fit
                            <select id="video-fit">
                                <option value="cover">Cover</option>
                                <option value="contain" selected>Contain</option>
                                <option value="stretch">Stretch</option>
                            </select>
                        </label>
                        <label>
                            Speed
                            <select id="playback-speed">
                                <option value="0.25">0.25x</option>
                                <option value="0.5">0.5x</option>
                                <option value="1" selected>1x</option>
                                <option value="1.5">1.5x</option>
                                <option value="2">2x</option>
                            </select>
                        </label>
                    </div>
                </div>

                <input type="file" id="video-file-input" accept="video/*" hidden>
                <div class="vlib__drop" id="video-dropzone">
                    <div><i class="fas fa-film"></i><span>Drop a video to play it</span></div>
                </div>
            </div>
        `;

        this.root = container.querySelector('.vlib');
        this.cacheUIElements();
        this.bindUIEvents();
        this.loadLibrary();
    }

    cacheUIElements() {
        const $ = (id) => document.getElementById(id);

        this.grid = $('video-grid');
        this.statusEl = $('video-status');
        this.searchInput = $('video-search');
        this.localVideoBtn = $('local-video-btn');
        this.cacheAllBtn = $('cache-all-btn');
        this.videoFileInput = $('video-file-input');
        this.videoUrlInput = $('video-url-input');
        this.loadUrlBtn = $('load-url-btn');
        this.dropzone = $('video-dropzone');

        this.dock = $('video-dock');
        this.nowPlayingEl = $('video-now-playing');
        this.exitBtn = $('video-exit-btn');
        this.playPauseBtn = $('play-pause-btn');
        this.seekBar = $('seek-bar');
        this.seekWrap = $('video-seek-wrap');
        this.previewEl = $('video-preview');
        this.previewCanvas = $('video-preview-canvas');
        this.previewTimeEl = $('video-preview-time');
        this.currentTimeEl = $('current-time');
        this.durationEl = $('duration');
        this.muteBtn = $('mute-btn');
        this.volumeBar = $('volume-bar');
        this.loopBtn = $('loop-video');
        this.fitSelect = $('video-fit');
        this.playbackSpeedSelect = $('playback-speed');
    }

    bindUIEvents() {
        this.localVideoBtn.addEventListener('click', () => this.videoFileInput.click());

        if (this.cacheAllBtn) {
            this.cacheAllBtn.hidden = !VideoCache.supported();
            this.cacheAllBtn.addEventListener('click', () => this.cacheAllRemote());
        }

        this.videoFileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                this.playFile(e.target.files[0]);
            }
            e.target.value = '';
        });

        this.loadUrlBtn.addEventListener('click', () => {
            const url = this.videoUrlInput.value.trim();
            if (url) this.playSource(url, VideoLibrary.titleFromSrc(url));
        });

        this.videoUrlInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') this.loadUrlBtn.click();
        });

        this.searchInput.addEventListener('input', () => this.applyFilter());

        this.grid.addEventListener('click', (e) => {
            const cacheBtn = e.target.closest('[data-cache-src]');
            if (cacheBtn) {
                e.preventDefault();
                e.stopPropagation();
                this.cacheOne(cacheBtn.dataset.cacheSrc);
                return;
            }

            const card = e.target.closest('.vcard');
            if (card) this.playSource(card.dataset.src, card.dataset.title);
        });

        this.grid.addEventListener('keydown', (e) => {
            if (e.key !== 'Enter' && e.key !== ' ') return;
            const cacheBtn = e.target.closest('[data-cache-src]');
            if (!cacheBtn) return;
            e.preventDefault();
            e.stopPropagation();
            this.cacheOne(cacheBtn.dataset.cacheSrc);
        });

        this.bindDragAndDrop();
        this.bindPlaybackControls();
        this.bindSeekBar();
        this.bindSeekPreview();
    }

    bindDragAndDrop() {
        let depth = 0;

        const show = () => this.dropzone.classList.add('is-active');
        const hide = () => this.dropzone.classList.remove('is-active');

        this.root.addEventListener('dragenter', (e) => {
            e.preventDefault();
            depth++;
            show();
        });

        this.root.addEventListener('dragover', (e) => e.preventDefault());

        this.root.addEventListener('dragleave', () => {
            depth = Math.max(0, depth - 1);
            if (depth === 0) hide();
        });

        this.root.addEventListener('drop', (e) => {
            e.preventDefault();
            depth = 0;
            hide();

            const file = e.dataTransfer.files && e.dataTransfer.files[0];
            if (file) {
                this.playFile(file);
                return;
            }

            const url = e.dataTransfer.getData('text/uri-list') || e.dataTransfer.getData('text/plain');
            if (url) this.playSource(url.trim(), VideoLibrary.titleFromSrc(url.trim()));
        });
    }

    bindPlaybackControls() {
        this.playPauseBtn.addEventListener('click', () => this.videoHandler.togglePlay());

        this.exitBtn.addEventListener('click', () => {
            this.videoHandler.pause();
            this.setVideoMode(false);
        });

        this.muteBtn.addEventListener('click', () => {
            this.videoHandler.toggleMute();
            this.updateMuteButtonUI();
        });

        this.volumeBar.addEventListener('input', () => {
            this.videoHandler.setVolume(parseFloat(this.volumeBar.value));
            this.updateMuteButtonUI();
        });

        this.loopBtn.addEventListener('click', () => {
            const next = !this.videoHandler.videoElement.loop;
            this.videoHandler.setLoop(next);
            this.loopBtn.classList.toggle('is-on', next);
        });

        this.fitSelect.addEventListener('change', () => {
            this.fitMode = this.fitSelect.value;
        });

        this.playbackSpeedSelect.addEventListener('change', () => {
            this.videoHandler.setPlaybackSpeed(parseFloat(this.playbackSpeedSelect.value));
        });
    }

    /**
     * Map pointer position onto the timeline ourselves. Styled range inputs
     * stay focused after the first click, so later taps step instead of jumping,
     * and timeupdate then fights the thumb until it blurs.
     */
    bindSeekBar() {
        const seekFromClientX = (clientX) => {
            if (!this.videoHandler.hasVideo()) return;

            const duration = this.videoHandler.videoElement.duration;
            if (!Number.isFinite(duration) || duration <= 0) return;

            const rect = this.seekWrap.getBoundingClientRect();
            const ratio = Math.min(1, Math.max(0, (clientX - rect.left) / Math.max(rect.width, 1)));
            const percent = ratio * 100;

            this.seekBar.value = String(percent);
            this.currentTimeEl.textContent = VideoLibrary.formatDuration(ratio * duration) || '0:00';
            this.videoHandler.setProgress(percent);
        };

        const endScrub = (e) => {
            if (!this.isScrubbing) return;
            this.isScrubbing = false;
            if (typeof e.clientX === 'number') seekFromClientX(e.clientX);
            this.seekBar.blur();
        };

        this.seekWrap.addEventListener('pointerdown', (e) => {
            if (e.pointerType === 'mouse' && e.button !== 0) return;
            e.preventDefault();
            this.isScrubbing = true;
            try {
                this.seekWrap.setPointerCapture(e.pointerId);
            } catch (error) {
                // Capture is optional; move/up still work while the pointer stays over the track.
            }
            seekFromClientX(e.clientX);
        });

        this.seekWrap.addEventListener('pointermove', (e) => {
            if (!this.isScrubbing) return;
            seekFromClientX(e.clientX);
        });

        this.seekWrap.addEventListener('pointerup', endScrub);
        this.seekWrap.addEventListener('pointercancel', endScrub);
        this.seekWrap.addEventListener('lostpointercapture', endScrub);

        this.seekBar.addEventListener('input', () => {
            if (!this.videoHandler.hasVideo()) return;
            this.videoHandler.setProgress(parseFloat(this.seekBar.value));
            const time = (parseFloat(this.seekBar.value) / 100) * this.videoHandler.videoElement.duration;
            this.currentTimeEl.textContent = VideoLibrary.formatDuration(time) || '0:00';
        });
    }

    /**
     * Hovering the scrub bar decodes the frame under the cursor in a second,
     * throttled video element so seeking is not a blind guess.
     */
    bindSeekPreview() {
        let pendingTime = null;
        let seeking = false;

        const ensurePreviewVideo = () => {
            if (this.previewVideo && this.previewVideo.dataset.src === this.activeSrc) {
                return this.previewVideo;
            }

            if (this.previewVideo) this.previewVideo.remove();

            const video = document.createElement('video');
            video.muted = true;
            video.playsInline = true;
            video.preload = 'auto';
            video.crossOrigin = 'anonymous';
            video.style.display = 'none';
            video.dataset.src = this.activeSrc;
            video.src = this.videoHandler.videoElement.currentSrc || this.videoHandler.videoElement.src;
            document.body.appendChild(video);

            video.addEventListener('seeked', () => {
                seeking = false;
                try {
                    const ctx = this.previewCanvas.getContext('2d');
                    ctx.drawImage(video, 0, 0, this.previewCanvas.width, this.previewCanvas.height);
                } catch (error) {
                    // Cross-origin frame; the timestamp label still helps.
                }

                if (pendingTime !== null) {
                    const next = pendingTime;
                    pendingTime = null;
                    seeking = true;
                    video.currentTime = next;
                }
            });

            this.previewVideo = video;
            return video;
        };

        this.seekWrap.addEventListener('mousemove', (e) => {
            if (!this.videoHandler.hasVideo()) return;

            const duration = this.videoHandler.videoElement.duration;
            if (!Number.isFinite(duration)) return;

            const rect = this.seekWrap.getBoundingClientRect();
            const ratio = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
            const time = ratio * duration;

            this.previewEl.hidden = false;
            this.previewEl.style.left = `${ratio * 100}%`;
            this.previewTimeEl.textContent = VideoLibrary.formatDuration(time) || '0:00';

            const video = ensurePreviewVideo();
            if (seeking) {
                pendingTime = time;
            } else {
                seeking = true;
                video.currentTime = time;
            }
        });

        this.seekWrap.addEventListener('mouseleave', () => {
            this.previewEl.hidden = true;
        });
    }

    // === Library ===

    async loadLibrary() {
        this.statusEl.textContent = 'Checking library...';

        const available = await VideoLibrary.probe();
        this.entries = available;

        if (!available.length) {
            this.statusEl.textContent = 'No clips found. Upload, drop a file, or add entries to videos/index.json.';
            return;
        }

        this.statusEl.hidden = true;
        this.renderGrid(available);
        this.refreshCacheBadges();
    }

    renderGrid(entries) {
        const escapeAttr = (value) => String(value)
            .replace(/&/g, '&amp;')
            .replace(/"/g, '&quot;')
            .replace(/</g, '&lt;');

        this.grid.innerHTML = entries.map((entry) => {
            const remote = VideoCache.isCacheable(entry.src);
            const cacheControl = remote
                ? `<span class="vcard__cache" data-cache-src="${escapeAttr(entry.src)}" title="Save offline" role="button" tabindex="0" aria-label="Save offline">
                        <i class="fas fa-download"></i>
                   </span>`
                : '';

            return `
            <button class="vcard" data-src="${escapeAttr(entry.src)}" data-title="${escapeAttr(entry.title)}">
                <span class="vcard__thumb">
                    <span class="vcard__skeleton"></span>
                    <span class="vcard__badge" hidden></span>
                    ${cacheControl}
                    <span class="vcard__play"><i class="fas fa-play"></i></span>
                </span>
                <span class="vcard__title">${escapeAttr(entry.title)}</span>
                <span class="vcard__meta"></span>
            </button>`;
        }).join('');

        this.observeThumbnails();
        this.updateActiveCard();
    }

    /**
     * Posters are decoded only for cards scrolled into view, so opening the tab
     * does not touch every clip in the library at once.
     */
    observeThumbnails() {
        const cards = Array.from(this.grid.querySelectorAll('.vcard'));

        if (typeof IntersectionObserver === 'undefined') {
            cards.forEach((card) => this.fillThumbnail(card));
            return;
        }

        if (this.thumbObserver) this.thumbObserver.disconnect();

        this.thumbObserver = new IntersectionObserver((records, observer) => {
            records.forEach((record) => {
                if (!record.isIntersecting) return;
                observer.unobserve(record.target);
                this.fillThumbnail(record.target);
            });
        }, { root: this.grid.closest('.shader-test-ui') || null, rootMargin: '200px' });

        cards.forEach((card) => this.thumbObserver.observe(card));
    }

    async fillThumbnail(card) {
        const src = card.dataset.src;
        if (this.thumbnailsRequested.has(src)) return;
        this.thumbnailsRequested.add(src);

        const meta = await VideoLibrary.generateThumbnail(src);
        const thumb = card.querySelector('.vcard__thumb');
        const skeleton = card.querySelector('.vcard__skeleton');
        const badge = card.querySelector('.vcard__badge');

        if (skeleton) skeleton.remove();

        if (meta.poster) {
            const img = document.createElement('img');
            img.src = meta.poster;
            img.alt = '';
            img.loading = 'lazy';
            thumb.prepend(img);
        } else {
            thumb.classList.add('is-blank');
        }

        const duration = VideoLibrary.formatDuration(meta.duration);
        if (duration) {
            badge.textContent = duration;
            badge.hidden = false;
        }

        if (meta.width && meta.height) {
            card.querySelector('.vcard__meta').textContent = `${meta.width}×${meta.height}`;
        }
    }

    applyFilter() {
        const query = this.searchInput.value.trim().toLowerCase();
        let visible = 0;

        this.grid.querySelectorAll('.vcard').forEach((card) => {
            const match = card.dataset.title.toLowerCase().includes(query);
            card.hidden = !match;
            if (match) visible++;
        });

        this.statusEl.hidden = visible > 0 || this.entries.length === 0;
        if (!visible && this.entries.length) {
            this.statusEl.textContent = `No clips match “${this.searchInput.value.trim()}”.`;
        }
    }

    // === Playback ===

    async playSource(src, title) {
        this.activeSrc = src;
        this.setVideoMode(true);
        this.showDock(title || VideoLibrary.titleFromSrc(src));
        this.updateActiveCard();

        const resolved = await VideoCache.resolveForPlayback(src, { warm: false });
        this.videoHandler.loadVideo(resolved.playUrl, {
            sourceId: src,
            isObjectUrl: resolved.isObjectUrl
        });

        if (!resolved.fromCache && VideoCache.isCacheable(src)) {
            VideoCache.put(src).then((ok) => {
                if (ok) this.refreshCacheBadges();
            });
        }
    }

    async cacheOne(src) {
        if (!VideoCache.isCacheable(src) || !VideoCache.supported()) return;

        const card = this.grid
            && Array.from(this.grid.querySelectorAll('.vcard')).find((el) => el.dataset.src === src);
        const btn = card && card.querySelector('.vcard__cache');
        if (btn) {
            btn.classList.add('is-busy');
            btn.title = 'Caching…';
        }

        const ok = await VideoCache.put(src);
        if (btn) btn.classList.remove('is-busy');
        this.refreshCacheBadges();

        if (!ok && this.statusEl) {
            this.statusEl.hidden = false;
            this.statusEl.textContent = 'Could not cache that clip (CORS or storage quota).';
        }
    }

    async cacheAllRemote() {
        if (!VideoCache.supported()) return;

        const remotes = (this.entries || []).filter((entry) => VideoCache.isCacheable(entry.src));
        if (!remotes.length) return;

        if (this.cacheAllBtn) {
            this.cacheAllBtn.disabled = true;
            this.cacheAllBtn.classList.add('is-busy');
        }

        this.statusEl.hidden = false;

        let done = 0;
        for (const entry of remotes) {
            if (await VideoCache.has(entry.src)) {
                done++;
                this.statusEl.textContent = `Cached ${done}/${remotes.length}`;
                continue;
            }

            this.statusEl.textContent = `Caching ${done + 1}/${remotes.length}: ${entry.title}`;
            await VideoCache.put(entry.src);
            done++;
            this.refreshCacheBadges();
        }

        this.statusEl.textContent = `Cached ${done}/${remotes.length} remote clips.`;
        if (this.cacheAllBtn) {
            this.cacheAllBtn.disabled = false;
            this.cacheAllBtn.classList.remove('is-busy');
        }
    }

    async refreshCacheBadges() {
        if (!this.grid || !VideoCache.supported()) return;

        const cards = Array.from(this.grid.querySelectorAll('.vcard'));
        await Promise.all(cards.map(async (card) => {
            const src = card.dataset.src;
            const btn = card.querySelector('.vcard__cache');
            if (!btn || !VideoCache.isCacheable(src)) return;

            const cached = await VideoCache.has(src);
            card.classList.toggle('is-cached', cached);
            btn.classList.toggle('is-cached', cached);
            btn.title = cached ? 'Saved offline' : 'Save offline';
            btn.innerHTML = cached
                ? '<i class="fas fa-check"></i>'
                : '<i class="fas fa-download"></i>';
        }));
    }

    playFile(file) {
        if (!file.type.startsWith('video/')) {
            this.showDock('Unsupported file type');
            return;
        }

        this.activeSrc = `upload:${file.name}`;
        this.videoHandler.loadVideoFile(file);
        this.setVideoMode(true);
        this.showDock(file.name);
        this.updateActiveCard();
    }

    showDock(title) {
        if (!this.dock) return;
        this.dock.hidden = false;
        this.nowPlayingEl.textContent = title;
    }

    updateActiveCard() {
        if (!this.grid) return;
        this.grid.querySelectorAll('.vcard').forEach((card) => {
            card.classList.toggle('is-active', card.dataset.src === this.activeSrc);
        });
    }

    setupVideoEventListeners() {
        document.addEventListener('videoMetadataLoaded', (e) => {
            const { duration, width, height } = e.detail;

            if (this.durationEl) {
                this.durationEl.textContent = VideoLibrary.formatDuration(duration) || '0:00';
            }
            if (this.nowPlayingEl && width && height) {
                this.nowPlayingEl.title = `${width}×${height}`;
            }

            this.videoHandler.play();
        });

        document.addEventListener('videoTimeUpdate', (e) => {
            const { currentTime, percentage } = e.detail;

            if (this.isScrubbing || this.videoHandler.isSeeking()) return;

            if (this.currentTimeEl) {
                this.currentTimeEl.textContent = VideoLibrary.formatDuration(currentTime) || '0:00';
            }
            if (this.seekBar) {
                this.seekBar.value = percentage;
            }
        });

        document.addEventListener('videoPlaying', () => this.updatePlayPauseButtonUI(true));
        document.addEventListener('videoPaused', () => this.updatePlayPauseButtonUI(false));

        document.addEventListener('videoError', (e) => {
            console.error('Video error:', e.detail.error);
            if (this.nowPlayingEl) {
                this.nowPlayingEl.textContent = 'Could not play that video';
            }
        });
    }

    updatePlayPauseButtonUI(isPlaying) {
        if (!this.playPauseBtn) return;
        this.playPauseBtn.innerHTML = isPlaying
            ? '<i class="fas fa-pause"></i>'
            : '<i class="fas fa-play"></i>';
    }

    updateMuteButtonUI() {
        if (!this.muteBtn) return;
        this.muteBtn.innerHTML = this.videoHandler.isMuted
            ? '<i class="fas fa-volume-mute"></i>'
            : '<i class="fas fa-volume-up"></i>';
    }

    // === Render hooks ===

    updateVideoTexture() {
        if (this.isVideoMode) {
            this.videoHandler.updateVideoTexture();
        }
    }

    getVideoTexture() {
        return this.videoHandler.getVideoTexture();
    }

    getVideoSize() {
        return this.videoHandler.getVideoSize();
    }

    getFitMode() {
        return VIDEO_FIT_MODES[this.fitMode] ?? 1;
    }

    getVideoDisplayShader() {
        return this.videoDisplayShader;
    }

    isInVideoMode() {
        return this.isVideoMode;
    }

    setVideoMode(enabled) {
        this.isVideoMode = enabled;
        this.videoHandler.setVideoMode(enabled);
        if (this.root) {
            this.root.classList.toggle('is-live', enabled);
        }
    }

    dispose() {
        if (this.thumbObserver) this.thumbObserver.disconnect();
        if (this.previewVideo) this.previewVideo.remove();
        this.videoHandler.dispose();
    }
}

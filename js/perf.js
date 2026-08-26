/**
 * Pianoscope Visualizer — runtime performance management.
 *
 * The visualizer is a single fullscreen quad running expensive procedural
 * fragment shaders, so its cost is very close to linear in the number of
 * fragments. That makes render resolution the primary quality/speed dial, and
 * everything here exists to pick a good value for it and to show the operator
 * what the frame is actually costing.
 *
 * Exposes window.PianoscopePerf.
 *
 * Keyboard:  `  toggle HUD    [ / ]  resolution down/up    \  toggle adaptive
 */
(function () {
    "use strict";

    // v2: previous defaults (1080p ceiling, 0.45 min scale, 0.55 shader bias)
    // stacked into a soft, hunting image. Bump the key so those saved values
    // are not replayed on top of the new middle ground.
    const STORAGE_KEY = 'pianoscope.perf.v2';

    const DEFAULTS = {
        // 1.75 keeps most Retina sharpness (native 2.0) while cutting ~23% of
        // fragments. Shadertoy is DPR 1; going that far made the looks mushy.
        dprCap: 1.75,
        scale: 1.0,
        adaptive: true,
        minScale: 0.75,
        maxScale: 1.0,
        // QHD ceiling: 4K projectors scale down, 1080p/Retina laptops stay near
        // native. The old 1920×1080 cap was the main "looks worse" complaint.
        maxPixelCount: 2560 * 1440,
        fpsCap: 0,
        hud: false
    };

    const HEAVY_NAME = /kaliset|mandelbulb|fireworks|elevator|hyperloop|kente loom|^fish$/i;

    /**
     * Heuristic fill-rate bias from source + label. 1 is full resolution;
     * heavy raymarch / AA / feedback looks start lower so adaptive does not
     * have to drop frames before it notices.
     */
    function estimateShaderBias(source, label) {
        const src = source || '';
        const name = label || '';
        if (HEAVY_NAME.test(name) || /iWriteFeedback/.test(src)) return 0.85;
        if (/antiAliasingLevel|#define\s+AA\s+[2-9]|AA_SAMPLES/.test(src)) return 0.85;

        let maxBound = 0;
        const loopRe = /for\s*\(\s*int\s+\w+\s*=\s*[^;]+;\s*\w+\s*<=?\s*(\d+)/g;
        let match;
        while ((match = loopRe.exec(src))) {
            const n = parseInt(match[1], 10);
            if (n > maxBound) maxBound = n;
        }
        const defRe = /#define\s+(?:MAX_STEPS|MaxSteps|ITERATIONS|Iterations|NUM_SUBPARTICLES|JULIA_ITERS)\s+(\d+)/gi;
        while ((match = defRe.exec(src))) {
            const n = parseInt(match[1], 10);
            if (n > maxBound) maxBound = n;
        }

        if (maxBound >= 80) return 0.85;
        if (maxBound >= 48) return 0.92;
        return 1;
    }

    function costLabel(bias) {
        if (bias < 0.9) return 'heavy';
        if (bias < 1) return 'medium';
        return 'light';
    }

    function snapScale(value) {
        return Math.round(value * 20) / 20;
    }

    const FRAME_WINDOW = 15;

    // We deliberately steer toward 60fps rather than trying to infer the panel's
    // refresh rate. Inference is unreliable — vsync-locked frame times carry no
    // headroom information, and a window dragged between a 120Hz and a 60Hz
    // display would leave the estimate stale and pin the scale at its floor. On a
    // high-refresh display a rock-steady 60 at full resolution reads better than
    // 120 at half, for shader content like this.
    const TARGET_FRAME_MS = 1000 / 60;

    function loadSettings() {
        try {
            const raw = localStorage.getItem(STORAGE_KEY);
            return raw ? Object.assign({}, DEFAULTS, JSON.parse(raw)) : Object.assign({}, DEFAULTS);
        } catch (e) {
            return Object.assign({}, DEFAULTS);
        }
    }

    function saveSettings(settings) {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
        } catch (e) {
            /* private browsing, quota, etc. Settings just won't persist. */
        }
    }

    function clamp(value, lo, hi) {
        return value < lo ? lo : (value > hi ? hi : value);
    }

    /**
     * GPU-side timing via EXT_disjoint_timer_query_webgl2.
     *
     * Chrome/Edge only in practice — Firefox has it disabled for Spectre reasons
     * and Safari keeps it behind a flag — so `supported` is false often enough
     * that every consumer has to work without it.
     */
    class GpuTimer {
        constructor(gl) {
            this.gl = gl;
            this.ext = gl.getExtension ? gl.getExtension('EXT_disjoint_timer_query_webgl2') : null;
            this.supported = !!(this.ext && typeof gl.createQuery === 'function');
            this.free = [];
            this.pending = [];
            this.active = null;
            this.lastMs = 0;

            // Clear any disjoint state left over from before we started.
            if (this.supported) gl.getParameter(this.ext.GPU_DISJOINT_EXT);
        }

        begin() {
            if (!this.supported || this.active) return;
            const gl = this.gl;
            const query = this.free.pop() || gl.createQuery();
            gl.beginQuery(this.ext.TIME_ELAPSED_EXT, query);
            this.active = query;
        }

        end() {
            if (!this.active) return;
            this.gl.endQuery(this.ext.TIME_ELAPSED_EXT);
            this.pending.push(this.active);
            this.active = null;
        }

        /**
         * Results arrive a few frames after submission. Polling for availability
         * is free; reading QUERY_RESULT before it is available would stall the
         * whole pipeline, which is the thing we are trying to measure.
         */
        poll() {
            if (!this.supported || !this.pending.length) return this.lastMs;
            const gl = this.gl;

            // A disjoint event (GPU context switch, power state change) invalidates
            // every query currently in flight.
            if (gl.getParameter(this.ext.GPU_DISJOINT_EXT)) {
                for (let i = 0; i < this.pending.length; i++) this.free.push(this.pending[i]);
                this.pending.length = 0;
                return this.lastMs;
            }

            const query = this.pending[0];
            if (gl.getQueryParameter(query, gl.QUERY_RESULT_AVAILABLE)) {
                this.lastMs = gl.getQueryParameter(query, gl.QUERY_RESULT) / 1e6;
                this.pending.shift();
                this.free.push(query);
            }
            return this.lastMs;
        }
    }

    /**
     * Proportional controller over render scale, after Binks' dynamic resolution
     * rendering.
     *
     * The awkward part is that requestAnimationFrame is vsync locked: once we are
     * comfortably hitting the refresh rate, frame time pins at the refresh
     * interval and stops carrying any information about how much headroom is
     * left. With a GPU timer we can see the real cost and scale back up
     * confidently; without one we can only creep upward and watch for a miss.
     */
    class ScaleController {
        constructor(settings) {
            this.settings = settings;
            this.samples = new Float32Array(FRAME_WINDOW);
            this.sorted = new Float32Array(FRAME_WINDOW);
            this.count = 0;
            this.cursor = 0;
            this.cooldown = 60;
            this.stableMs = 0;
            this.medianMs = TARGET_FRAME_MS;
        }

        /** Resizes and shader switches both invalidate the current window. */
        reset(cooldownFrames) {
            this.count = 0;
            this.cursor = 0;
            this.stableMs = 0;
            this.cooldown = cooldownFrames === undefined ? 45 : cooldownFrames;
        }

        median() {
            const n = Math.min(this.count, FRAME_WINDOW);
            const view = this.sorted.subarray(0, n);
            view.set(this.samples.subarray(0, n));
            view.sort();
            return view[n >> 1];
        }

        /**
         * @returns {number} the new scale, or 0 when it should be left alone.
         */
        update(frameMs, gpuMs, hasGpuTiming) {
            // Ignore obvious outliers: tab switches, shader compiles, GC pauses.
            if (frameMs > 0 && frameMs < 500) {
                this.samples[this.cursor] = frameMs;
                this.cursor = (this.cursor + 1) % FRAME_WINDOW;
                this.count++;
            }

            if (this.count < FRAME_WINDOW) return 0;

            this.medianMs = this.median();

            if (this.cooldown > 0) {
                this.cooldown--;
                return 0;
            }
            if (!this.settings.adaptive) return 0;

            const settings = this.settings;
            const budgetMs = settings.fpsCap > 0 ? 1000 / settings.fpsCap : TARGET_FRAME_MS;
            let next = settings.scale;

            if (hasGpuTiming && gpuMs > 0) {
                // Wide deadband so a vsync-locked ~16ms GPU time does not
                // ping-pong scale every cooldown. Down only when the GPU is
                // actually blowing the frame; up only after seconds of slack.
                if (gpuMs > budgetMs * 0.95) {
                    next = settings.scale * 0.9;
                    this.stableMs = 0;
                } else if (gpuMs < budgetMs * 0.4) {
                    this.stableMs += this.medianMs * FRAME_WINDOW;
                    if (this.stableMs > 4000 && settings.scale < settings.maxScale) {
                        next = settings.scale + 0.05;
                        this.stableMs = 0;
                    }
                } else {
                    this.stableMs = 0;
                }
            } else {
                const missed = this.medianMs > budgetMs * 1.25;
                if (missed) {
                    next = settings.scale * 0.9;
                    this.stableMs = 0;
                } else {
                    this.stableMs += this.medianMs * FRAME_WINDOW;
                    if (this.stableMs > 4000 && settings.scale < settings.maxScale) {
                        next = settings.scale + 0.05;
                        this.stableMs = 0;
                    }
                }
            }

            next = snapScale(clamp(next, settings.minScale, settings.maxScale));
            if (Math.abs(next - settings.scale) < 0.04) return 0;

            this.cooldown = 90;
            return next;
        }
    }

    const Perf = {
        settings: loadSettings(),
        gl: null,
        canvas: null,
        gpuTimer: null,
        controller: null,
        onScaleChange: null,

        frameMs: 0,
        cpuMs: 0,
        gpuMs: 0,
        _frameStart: 0,
        _lastFrameTime: 0,
        _lastPresent: 0,
        _hudEl: null,
        _hudNextUpdate: 0,
        _shaderLabel: '',
        _compileMs: 0,
        shaderBias: 1,
        _costLabel: 'light',
        _resolutionDirty: false,

        attach(gl, canvas, onScaleChange) {
            this.gl = gl;
            this.canvas = canvas;
            this.onScaleChange = onScaleChange;
            this.gpuTimer = new GpuTimer(gl);
            this.controller = new ScaleController(this.settings);

            // A manual scale from a previous session should not be silently
            // re-applied as if the operator had chosen it for this machine.
            if (this.settings.adaptive) this.settings.scale = this.settings.maxScale;

            this._buildHud();
            this._bindKeys();
            return this;
        },

        /** Device pixels per CSS pixel that the drawing buffer should use. */
        pixelRatio() {
            const dpr = window.devicePixelRatio || 1;
            return Math.min(dpr, this.settings.dprCap) * this.settings.scale * this.shaderBias;
        },

        /**
         * Drawing-buffer size for a CSS box. Applies DPR cap, adaptive scale,
         * shader bias, then the absolute pixel ceiling.
         */
        bufferSize(cssWidth, cssHeight) {
            const ratio = this.pixelRatio();
            let width = Math.max(1, Math.round(cssWidth * ratio));
            let height = Math.max(1, Math.round(cssHeight * ratio));
            const maxPx = this.settings.maxPixelCount || 0;
            if (maxPx > 0 && width * height > maxPx) {
                const shrink = Math.sqrt(maxPx / (width * height));
                width = Math.max(1, Math.round(width * shrink));
                height = Math.max(1, Math.round(height * shrink));
            }
            return { width: width, height: height };
        },

        markResolutionDirty() {
            this._resolutionDirty = true;
        },

        takeResolutionDirty() {
            if (!this._resolutionDirty) return false;
            this._resolutionDirty = false;
            return true;
        },

        /**
         * @returns {boolean} false when this frame should be skipped to honour
         *   an explicit FPS cap.
         */
        beginFrame(now) {
            const cap = this.settings.fpsCap;
            if (cap > 0) {
                // Half a refresh of slack, because rAF timestamps are quantised to
                // the refresh interval and an exact budget would systematically
                // overshoot to the next vsync.
                if (now - this._lastPresent < (1000 / cap) - TARGET_FRAME_MS * 0.5) {
                    return false;
                }
            }
            this._lastPresent = now;

            this.frameMs = this._lastFrameTime ? now - this._lastFrameTime : 0;
            this._lastFrameTime = now;
            this._frameStart = performance.now();
            return true;
        },

        beginGpuWork() {
            this.gpuTimer.begin();
        },

        endGpuWork() {
            this.gpuTimer.end();
        },

        endFrame() {
            this.cpuMs = performance.now() - this._frameStart;
            this.gpuMs = this.gpuTimer.poll();

            const next = this.controller.update(this.frameMs, this.gpuMs, this.gpuTimer.supported);
            if (next) {
                this.settings.scale = next;
                // Apply on the *next* frame, before that draw. Changing
                // canvas.width after present clears the backbuffer and the
                // compositor shows a blank frame — the flash users saw.
                this._resolutionDirty = true;
            }

            if (this.settings.hud && this._frameStart > this._hudNextUpdate) {
                this._hudNextUpdate = this._frameStart + 250;
                this._renderHud();
            }
        },

        /** Called on resize and shader switch so the controller re-measures. */
        invalidate(cooldownFrames) {
            if (this.controller) this.controller.reset(cooldownFrames);
        },

        noteShader(label, compileMs, fragmentSource) {
            this._shaderLabel = label || '';
            if (compileMs !== undefined) this._compileMs = compileMs;

            const nextBias = estimateShaderBias(fragmentSource, label);
            const biasChanged = Math.abs(nextBias - this.shaderBias) > 0.01;
            this.shaderBias = nextBias;
            this._costLabel = costLabel(nextBias);

            this.invalidate();
            if (biasChanged) this._resolutionDirty = true;
        },

        setMaxPixelCount(count) {
            this.settings.maxPixelCount = Math.max(0, count | 0);
            saveSettings(this.settings);
            this.invalidate();
            this._resolutionDirty = true;
        },

        setScale(scale) {
            this.settings.scale = snapScale(clamp(scale, this.settings.minScale, this.settings.maxScale));
            this.settings.adaptive = false;
            saveSettings(this.settings);
            this.invalidate();
            this._resolutionDirty = true;
        },

        setDprCap(cap) {
            this.settings.dprCap = clamp(cap, 0.5, 4);
            saveSettings(this.settings);
            this.invalidate();
            this._resolutionDirty = true;
        },

        setAdaptive(enabled) {
            this.settings.adaptive = !!enabled;
            saveSettings(this.settings);
            this.invalidate();
        },

        setFpsCap(cap) {
            this.settings.fpsCap = Math.max(0, cap | 0);
            saveSettings(this.settings);
            this.invalidate();
        },

        toggleHud(force) {
            this.settings.hud = force === undefined ? !this.settings.hud : !!force;
            saveSettings(this.settings);
            if (this._hudEl) this._hudEl.style.display = this.settings.hud ? 'block' : 'none';
            if (this.settings.hud) this._renderHud();
        },

        _buildHud() {
            const el = document.createElement('div');
            el.id = 'perf-hud';
            el.style.cssText = [
                'position:fixed', 'bottom:12px', 'left:12px', 'z-index:10000',
                'background:rgba(12,12,18,0.82)', 'color:#E8E8F0',
                'font:11px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace',
                'padding:9px 11px', 'border-radius:7px',
                'border-left:3px solid #FFD700', 'pointer-events:none',
                'white-space:pre', 'min-width:220px',
                // Keep HUD text changes from invalidating layout of the page.
                'contain:content'
            ].join(';');
            el.style.display = this.settings.hud ? 'block' : 'none';
            (document.body || document.documentElement).appendChild(el);
            this._hudEl = el;
        },

        _renderHud() {
            if (!this._hudEl) return;

            const canvas = this.canvas;
            const median = this.controller.medianMs || this.frameMs;
            const fps = median > 0 ? 1000 / median : 0;
            const dpr = window.devicePixelRatio || 1;
            const mp = (canvas.width * canvas.height) / 1e6;

            const lines = [
                `${fps.toFixed(0).padStart(3)} fps   ${median.toFixed(1)} ms`,
                `cpu    ${this.cpuMs.toFixed(2)} ms`,
                this.gpuTimer.supported
                    ? `gpu    ${this.gpuMs.toFixed(2)} ms`
                    : 'gpu    n/a (no timer ext)',
                `res    ${canvas.width}x${canvas.height}  ${mp.toFixed(2)} MP`,
                `scale  ${this.settings.scale.toFixed(2)}  dpr ${dpr}\u2192${Math.min(dpr, this.settings.dprCap)}`,
                `pxcap  ${(this.settings.maxPixelCount / 1e6).toFixed(2)} MP  cost ${this._costLabel} \u00d7${this.shaderBias.toFixed(2)}`,
                `mode   ${this.settings.adaptive ? 'adaptive' : 'manual'}${this.settings.fpsCap ? '  cap ' + this.settings.fpsCap : ''}`
            ];

            if (this._shaderLabel) {
                lines.push(`shader ${this._shaderLabel.slice(0, 22)}`);
            }
            if (this._compileMs) {
                lines.push(`link   ${this._compileMs.toFixed(0)} ms`);
            }

            this._hudEl.textContent = lines.join('\n');
        },

        _bindKeys() {
            document.addEventListener('keydown', (e) => {
                const active = document.activeElement;
                if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA' || active.isContentEditable)) {
                    return;
                }
                if (e.metaKey || e.ctrlKey || e.altKey) return;

                if (e.key === '`') {
                    e.preventDefault();
                    this.toggleHud();
                } else if (e.key === '[') {
                    e.preventDefault();
                    this.setScale(this.settings.scale - 0.1);
                    this.toggleHud(true);
                } else if (e.key === ']') {
                    e.preventDefault();
                    this.setScale(this.settings.scale + 0.1);
                    this.toggleHud(true);
                } else if (e.key === '\\') {
                    e.preventDefault();
                    this.setAdaptive(!this.settings.adaptive);
                    this.toggleHud(true);
                }
            });
        }
    };

    window.PianoscopePerf = Perf;
})();

/**
 * Display text for the visualizer.
 *
 * Rasterizes the display string into TWO independent sheets, because the existing
 * text shaders and a purpose-built one want incompatible things from the texture.
 *
 *   iChannel3 - LEGACY sheet, 2048x512, R = sharp, G = soft, B = 0.
 *     Byte-for-byte the layout Kente Loom (and the video overlay) were
 *     tuned against: same 280px fit ladder, same 0.86/0.72 box, same 0.46 vertical
 *     bias, same 18px blur. Nothing in this file changes what those shaders see.
 *
 *   iTextSheet - FITTED sheet on its own texture unit, sized to the copy it holds.
 *     R = slot A sharp, G = slot A soft, B = slot B sharp (the incoming word).
 *     One fixed cap size, no ceiling, no wasted margin - so glyphs are barely
 *     magnified on a wall instead of being blown up out of a fixed strip.
 *
 * Companion uniforms for the fitted sheet:
 *   iTextMorph      0 -> 1 while a swap is in flight
 *   iTextLineH      one line's height as a fraction of the sheet
 *   iTextExtent     glyph block size as a fraction of the sheet
 *   iTextTransition index into TRANSITIONS
 *   iTextWords[8]   per-word rects (x0,y0,x1,y1) in sheet UV, slot A
 *   iTextWordCount  how many of those are live
 *   iTextLegacyExtent glyph block as a fraction of the 2048x512 strip
 *
 * Newline or | stacks a line. Overlay paints in caps.
 *
 * Rasterizing notes, measured rather than assumed:
 *   - Glyphs are drawn per character for tracking, so the blur must NOT be an
 *     active canvas filter during that loop: Chrome applies a filter as a full
 *     surface pass per draw call - 134ms for a 21-glyph string against 7ms for one
 *     blurred drawImage of the finished sharp pass.
 *   - Channel packing is a canvas composite, not getImageData plus a JS loop.
 *   - Character widths are measured once and reused by every paint pass.
 */
window.PianoscopeDisplayText = (function () {
    'use strict';

    const STORAGE_KEY = 'pianoscope-display-text';
    const FONT_KEY = 'pianoscope-display-font';
    const OVERLAY_KEY = 'pianoscope-display-overlay';
    const TRANSITION_KEY = 'pianoscope-display-transition';

    // Legacy sheet - do not retune, existing shaders depend on these exactly.
    const LEGACY_W = 2048;
    const LEGACY_H = 512;
    const LEGACY_BLUR = 18;
    const LEGACY_MAX = 280;
    const LEGACY_MIN = 28;
    const LEGACY_STEP = 6;

    // Fitted sheet - one fixed cap size for every cue, so the held line in a two
    // slot ladder never resizes when the other slot swaps. No ceiling to hit.
    const FIT_SIZE = 340;
    const FIT_BLUR = 20;
    const FIT_PAD = Math.ceil(FIT_BLUR * 3.2);
    const FIT_MAX_W = 4096;
    const FIT_MAX_H = 2048;

    const TRACKING = 0.06;
    const MAX_WORDS = 8;
    // bodoni and shoulders retired on measurement, not taste. Rasterised at 400px and
    // measured off the outline: Bodoni Moda 900's thinnest stroke is 0.3% of cap height
    // (one pixel at 299px caps) with 94:1 thick/thin contrast — a didone hairline is the
    // first thing 4:2:0 subsampling and inter-frame compression destroy. Big Shoulders
    // is the only condensed face on the list (advance/cap 0.61) and has the thinnest
    // stems (18.8% of cap), which closes counters at exactly the distance a wall is read
    // from. Everything else measured 19-30% and stays.
    const RETIRED_FONTS = {
        anton: 1, bebas: 1, audiowide: 1, orbitron: 1, oswald: 1, russo: 1, blackops: 1,
        bungee: 1, dela: 1, archivo: 1, bodoni: 1, shoulders: 1
    };

    const TRANSITIONS = [
        { id: 'cut', name: 'Cut', ms: 0 },
        { id: 'roll', name: 'Roll', ms: 340 },
        { id: 'reveal', name: 'Pattern reveal', ms: 520 },
        { id: 'chroma', name: 'Chroma split', ms: 360 },
        { id: 'knockout', name: 'Knockout flip', ms: 420 },
        { id: 'scramble', name: 'Glyph scramble', ms: 460 }
    ];

    const SCRAMBLE_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#%&*';
    const SCRAMBLE_FPS = 14;

    // Akira Expanded is the house face. Loaded from fonts/AkiraExpanded.otf (gitignored
    // demo) or the operator's installed copy via local(). Do not commit the file —
    // the Demo is personal-use and has no webfont licence.
    const FONTS = [
        { id: 'akira', name: 'Akira Expanded', css: '"Akira Expanded", Saira, sans-serif', weight: '400', load: '400 200px "Akira Expanded"', tracking: 0.02 },
        { id: 'syne', name: 'Syne ExtraBold', css: 'Syne, sans-serif', weight: '800', load: '800 200px Syne', tracking: 0.03 },
        { id: 'bricolage', name: 'Bricolage Grotesque', css: '"Bricolage Grotesque", sans-serif', weight: '800', load: '800 200px "Bricolage Grotesque"', tracking: 0.02 },
        { id: 'saira', name: 'Saira Expanded', css: 'Saira, sans-serif', weight: '900', stretch: 'expanded', load: '900 expanded 200px Saira', tracking: 0.03 },
        { id: 'anybody', name: 'Anybody ExtraExpanded', css: 'Anybody, sans-serif', weight: '900', stretch: 'extra-expanded', load: '900 extra-expanded 200px Anybody', tracking: 0.03 },
        { id: 'unbounded', name: 'Unbounded', css: 'Unbounded, sans-serif', weight: '900', load: '900 200px Unbounded', tracking: 0.04 }
    ];

    let gl = null;
    let legacyTexture = null;
    let fitTexture = null;
    let legacyCanvas = null;
    let fitCanvas = null;
    let blurSrc = null;
    let measureCtx = null;
    let overlay = null;
    let faceScratch = null;

    let currentText = '';
    let incomingText = '';
    let currentFontId = 'syne';
    let overlayEnabled = false;
    // Default drives sequencer playback now that the picker governs every state,
    // so it has to be a transition that actually animates.
    let transitionId = 'roll';
    let activeTransitionId = 'cut';

    let morph = 0;
    let morphRaf = 0;
    let morphStart = 0;
    let morphMs = 0;
    let settlePending = false;
    let settleTimer = 0;
    let scrambleTimer = 0;
    // Bumped whenever the fitted sheet is re-rastered. Distinct from version,
    // which means "the display string changed": a scramble redraws the sheet many
    // times inside a single string change, and a consumer that gates its upload on
    // version alone would show one frozen garbled frame for the whole transition.
    let sheetVersion = 0;
    // Slot A is untouched by a transition, so the legacy sheet only needs redrawing
    // when the resting text or the font actually changes.
    let legacyKey = '';
    // ...and when a webfont finally arrives. init() and setFont() both re-commit once
    // fonts have loaded, but that re-commit was a no-op: the legacy key is built from
    // (text, fontId) and neither of those changes when a face finishes downloading, so
    // the sheet kept whatever was rasterised on FIRST paint — which is the fallback
    // face, because webfonts always land after the first commit. On a cold load the wall
    // showed system sans at the wrong width until something unrelated changed the key.
    // Folding a generation counter into the key is what makes those re-commits real.
    let fontEpoch = 0;

    let legacyW = 1;
    let legacyH = 1;
    let fitW = 1;
    let fitH = 1;
    let lineFrac = 0.5;
    let extentW = 0.5;
    let extentH = 0.5;
    let extentBW = 0.5;
    let extentBH = 0.5;
    let legacyExtentW = 0.386;
    let legacyExtentH = 0.670;
    const wordRects = new Float32Array(MAX_WORDS * 4);
    let wordCount = 0;
    let version = 0;
    const listeners = [];

    function fontById(id) {
        for (let i = 0; i < FONTS.length; i++) {
            if (FONTS[i].id === id) return FONTS[i];
        }
        return FONTS[0];
    }

    function transitionById(id) {
        for (let i = 0; i < TRANSITIONS.length; i++) {
            if (TRANSITIONS[i].id === id) return TRANSITIONS[i];
        }
        return TRANSITIONS[0];
    }

    function createCanvas(width, height) {
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        return canvas;
    }

    /**
     * Canvas parses the CSS font shorthand but SILENTLY DISCARDS font-stretch from it:
     * setting "900 expanded 340px Saira" reads back as "900 340px Saira". The expanded
     * faces only looked right because Google was serving exactly one instance per
     * family, so the family resolved to the wide cut by accident — ship a second
     * instance and every one of them would quietly snap to normal width.
     * The stretch has to go through its own property.
     */
    function setType(ctx, size, font) {
        font = font || fontById(currentFontId);
        ctx.font = font.weight + ' ' + size + 'px ' + font.css;
        if ('fontStretch' in ctx) ctx.fontStretch = font.stretch || 'normal';
        ctx.textBaseline = 'middle';
        ctx.textAlign = 'left';
    }

    function normalizeText(text) {
        return String(text || '')
            .replace(/\r\n|\r|\n/g, '|')
            .replace(/\s*\|\s*/g, '|')
            .replace(/[ \t]+/g, ' ')
            .replace(/^\|+|\|+$/g, '')
            .trim();
    }

    function splitLines(text, caps) {
        return normalizeText(text)
            .split('|')
            .map(function (line) {
                line = line.trim();
                return caps ? line.toUpperCase() : line;
            })
            .filter(function (line) { return line.length > 0; });
    }

    /** Measure once, draw many. Widths are cached so a paint pass is pure fillText. */
    function measure(ctx, lines, size, font) {
        font = font || fontById(currentFontId);
        setType(ctx, size, font);
        const track = typeof font.tracking === 'number' ? font.tracking : TRACKING;
        const trackingPx = size * track;
        const rows = [];
        let widest = 0;
        for (let i = 0; i < lines.length; i++) {
            const chars = Array.from(lines[i]);
            const widths = [];
            let total = 0;
            for (let c = 0; c < chars.length; c++) {
                const w = ctx.measureText(chars[c]).width;
                widths.push(w);
                total += w;
            }
            total += trackingPx * Math.max(0, chars.length - 1);
            rows.push({ chars: chars, widths: widths, total: total });
            if (total > widest) widest = total;
        }
        const lineH = size * 0.98;
        const gap = size * 0.14;
        return {
            size: size,
            trackingPx: trackingPx,
            lineH: lineH,
            gap: gap,
            widest: widest,
            totalH: lines.length * lineH + Math.max(0, lines.length - 1) * gap,
            rows: rows
        };
    }

    /** The original ladder: walk down from 280px until the block fits 86% x 72%. */
    function fitMeasure(ctx, lines, w, h, font, ceiling, floor, step) {
        const top = ceiling || LEGACY_MAX;
        const bottom = floor || LEGACY_MIN;
        const dec = step || LEGACY_STEP;
        let size = top;
        while (size >= bottom) {
            const m = measure(ctx, lines, size, font);
            if (m.widest <= w * 0.86 && m.totalH <= h * 0.72) return m;
            size -= dec;
        }
        return measure(ctx, lines, bottom, font);
    }

    /** Original vertical placement for the legacy sheet, 0.46 bias included. */
    function legacyOriginY(m, h) {
        return h * 0.46 - m.totalH * 0.5 + m.lineH * 0.5;
    }

    function drawRows(ctx, m, w, h, y0, fillFn) {
        if (!m || !m.rows.length) return;
        let y = y0;
        for (let i = 0; i < m.rows.length; i++) {
            const row = m.rows[i];
            let x = w * 0.5 - row.total * 0.5;
            for (let c = 0; c < row.chars.length; c++) {
                if (fillFn) fillFn(ctx, row.chars[c], x, y);
                else ctx.fillText(row.chars[c], x, y);
                x += row.widths[c] + m.trackingPx;
            }
            y += m.lineH + m.gap;
        }
    }

    function ensure(canvas, w, h) {
        if (canvas.width !== w || canvas.height !== h) {
            canvas.width = w;
            canvas.height = h;
        }
    }

    // ---------------------------------------------------------------- legacy sheet

    function rasterizeLegacy(linesA, font) {
        const w = LEGACY_W;
        const h = LEGACY_H;
        const m = fitMeasure(measureCtx, linesA, w, h, font, LEGACY_MAX, LEGACY_MIN, LEGACY_STEP);
        const y0 = legacyOriginY(m, h);
        legacyExtentW = m.widest / w;
        legacyExtentH = m.totalH / h;

        ensure(blurSrc, w, h);
        const bctx = blurSrc.getContext('2d');
        bctx.setTransform(1, 0, 0, 1, 0, 0);
        bctx.globalCompositeOperation = 'source-over';
        bctx.filter = 'none';
        bctx.clearRect(0, 0, w, h);
        setType(bctx, m.size, font);
        bctx.fillStyle = '#00ff00';
        drawRows(bctx, m, w, h, y0);

        const ctx = legacyCanvas.getContext('2d');
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.filter = 'none';
        ctx.globalCompositeOperation = 'source-over';
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, w, h);
        ctx.globalCompositeOperation = 'lighter';
        setType(ctx, m.size, font);
        ctx.fillStyle = '#ff0000';
        drawRows(ctx, m, w, h, y0);
        ctx.filter = 'blur(' + LEGACY_BLUR + 'px)';
        ctx.drawImage(blurSrc, 0, 0);
        ctx.filter = 'none';
        ctx.globalCompositeOperation = 'source-over';
    }

    // ---------------------------------------------------------------- fitted sheet

    /**
     * Per-word rects for slot A, in sheet UV. Without these a shader can only move
     * the whole block; with them it can stagger, tint or displace word by word.
     */
    function computeWordRects(m, w, h, y0) {
        wordCount = 0;
        let y = y0;
        for (let i = 0; i < m.rows.length && wordCount < MAX_WORDS; i++) {
            const row = m.rows[i];
            let x = w * 0.5 - row.total * 0.5;
            let wordStart = x;
            let hasGlyph = false;
            for (let c = 0; c <= row.chars.length; c++) {
                const atEnd = c === row.chars.length;
                const ch = atEnd ? ' ' : row.chars[c];
                if (ch === ' ') {
                    if (hasGlyph && wordCount < MAX_WORDS) {
                        const o = wordCount * 4;
                        wordRects[o] = wordStart / w;
                        wordRects[o + 1] = (y - m.lineH * 0.5) / h;
                        wordRects[o + 2] = (x - m.trackingPx) / w;
                        wordRects[o + 3] = (y + m.lineH * 0.5) / h;
                        wordCount++;
                    }
                    if (!atEnd) {
                        x += row.widths[c] + m.trackingPx;
                        wordStart = x;
                    }
                    hasGlyph = false;
                } else {
                    hasGlyph = true;
                    x += row.widths[c] + m.trackingPx;
                }
            }
            y += m.lineH + m.gap;
        }
        for (let i = wordCount * 4; i < wordRects.length; i++) wordRects[i] = 0;
    }

    function planFitted(linesA, linesB, font) {
        let size = FIT_SIZE;
        let a = measure(measureCtx, linesA, size, font);
        let b = linesB.length ? measure(measureCtx, linesB, size, font) : null;
        let widest = Math.max(a.widest, b ? b.widest : 0);
        let tallest = Math.max(a.totalH, b ? b.totalH : 0);

        const budgetW = FIT_MAX_W - FIT_PAD * 2;
        const budgetH = FIT_MAX_H - FIT_PAD * 2;
        const scale = Math.min(1, budgetW / Math.max(widest, 1), budgetH / Math.max(tallest, 1));
        if (scale < 1) {
            size = Math.max(24, Math.floor(size * scale));
            a = measure(measureCtx, linesA, size, font);
            b = linesB.length ? measure(measureCtx, linesB, size, font) : null;
            widest = Math.max(a.widest, b ? b.widest : 0);
            tallest = Math.max(a.totalH, b ? b.totalH : 0);
        }
        return {
            size: size, a: a, b: b,
            width: Math.max(8, Math.ceil(widest) + FIT_PAD * 2),
            height: Math.max(8, Math.ceil(tallest) + FIT_PAD * 2)
        };
    }

    function rasterizeFitted(plan, font) {
        const w = plan.width;
        const h = plan.height;
        ensure(fitCanvas, w, h);
        ensure(blurSrc, w, h);
        const yA = h * 0.5 - plan.a.totalH * 0.5 + plan.a.lineH * 0.5;

        const bctx = blurSrc.getContext('2d');
        bctx.setTransform(1, 0, 0, 1, 0, 0);
        bctx.globalCompositeOperation = 'source-over';
        bctx.filter = 'none';
        bctx.clearRect(0, 0, w, h);
        setType(bctx, plan.size, font);
        bctx.fillStyle = '#00ff00';
        drawRows(bctx, plan.a, w, h, yA);

        const ctx = fitCanvas.getContext('2d');
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.filter = 'none';
        ctx.globalCompositeOperation = 'source-over';
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, w, h);
        ctx.globalCompositeOperation = 'lighter';

        setType(ctx, plan.size, font);
        ctx.fillStyle = '#ff0000';
        drawRows(ctx, plan.a, w, h, yA);

        if (plan.b) {
            const yB = h * 0.5 - plan.b.totalH * 0.5 + plan.b.lineH * 0.5;
            ctx.fillStyle = '#0000ff';
            drawRows(ctx, plan.b, w, h, yB);
        }

        ctx.filter = 'blur(' + FIT_BLUR + 'px)';
        ctx.drawImage(blurSrc, 0, 0);
        ctx.filter = 'none';
        ctx.globalCompositeOperation = 'source-over';

        lineFrac = plan.a.lineH / h;
        // Per slot, not the union of both. A union extent steps the shader's
        // margin clamp the instant slot B is added, and steps it back when slot B
        // retires at settle - so the held line snapped size at both ends of every
        // cue. Reporting each slot separately lets the shader interpolate the fit
        // across the swap, and the two are equal at rest.
        extentW = plan.a.widest / w;
        extentH = plan.a.totalH / h;
        extentBW = plan.b ? plan.b.widest / w : extentW;
        extentBH = plan.b ? plan.b.totalH / h : extentH;
        computeWordRects(plan.a, w, h, yA);
        sheetVersion += 1;
    }

    // ---------------------------------------------------------------------- upload

    // Staging unit for uploads. Without pinning this, bindTexture lands on whatever
    // unit the render loop left active - unit 4 after the fitted sheet is bound -
    // so an upload between frames would quietly rebind a sampler's texture.
    // WebGL guarantees at least 8 combined units, so 7 is always safe as scratch.
    function bindForUpload(tex) {
        gl.activeTexture(gl.TEXTURE7);
        gl.bindTexture(gl.TEXTURE_2D, tex);
    }

    function uploadBlack(tex, which) {
        if (!gl || !tex) return;
        bindForUpload(tex);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 0);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
            new Uint8Array([0, 0, 0, 255]));
        if (which === 'legacy') { legacyW = 1; legacyH = 1; legacyExtentW = 0.386; legacyExtentH = 0.670; }
        else { fitW = 1; fitH = 1; }
    }

    /** Fixed-size sheet, so allocate once and texSubImage2D afterwards. */
    function uploadLegacy() {
        if (!gl || !legacyTexture) return;
        bindForUpload(legacyTexture);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 1);
        if (legacyW === LEGACY_W && legacyH === LEGACY_H) {
            gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, legacyCanvas);
        } else {
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, legacyCanvas);
        }
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 0);
        legacyW = LEGACY_W;
        legacyH = LEGACY_H;
    }

    /** Fitted sheet changes size with the copy, so this reallocates by design. */
    function uploadFitted() {
        if (!gl || !fitTexture) return;
        bindForUpload(fitTexture);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 1);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, fitCanvas);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 0);
        fitW = fitCanvas.width;
        fitH = fitCanvas.height;
    }

    // --------------------------------------------------------------------- overlay

    // One flat value, deliberately not a gradient.
    //
    // The ramp this replaces was built from yTop to yBot across the whole text BLOCK,
    // not per glyph, so it never meant what it looked like: on a one-line cue in caps
    // the letterforms only reached the top of the ramp, and on a two-line cue line one
    // got the light end while line two got the dark end. Two lines of the same cue came
    // out in two different colours with two different contrast floors.
    //
    // A face over live video wants one value anyway — the contrast floor is then
    // identical at every point of every glyph on every line.
    const FACE_INK = '#fbf8f3';

    function paintOverlay() {
        if (!overlay) return;
        const ctx = overlay.getContext('2d');
        const w = overlay.width;
        const h = overlay.height;
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.globalCompositeOperation = 'source-over';
        ctx.filter = 'none';
        ctx.shadowBlur = 0;
        ctx.clearRect(0, 0, w, h);

        if (!overlayEnabled || !currentText) return;
        const lines = splitLines(currentText, true);
        if (lines.length === 0) return;

        const font = fontById(currentFontId);
        const m = fitMeasure(ctx, lines, w, h, font, 280, 28, 6);
        const size = m.size;
        const y0 = legacyOriginY(m, h);

        setType(ctx, size, font);

        // Dark ground, in two parts. The offset pass alone was the whole treatment
        // before, which left the up-left side of every glyph carrying nothing but the
        // contour — so on footage that went bright on that side the word lost its
        // shape there and stayed protected everywhere else. The centred pass makes the
        // protection symmetric; the offset pass is kept for the contact-shadow read.
        ctx.save();
        ctx.fillStyle = '#000000';
        ctx.shadowColor = 'rgba(0, 0, 0, 0.55)';
        ctx.shadowOffsetX = 0;
        ctx.shadowOffsetY = 0;
        ctx.shadowBlur = size * 0.14;
        drawRows(ctx, m, w, h, y0);
        ctx.shadowBlur = size * 0.06;
        drawRows(ctx, m, w, h, y0);
        ctx.shadowBlur = size * 0.08;
        ctx.shadowOffsetX = size * 0.012;
        ctx.shadowOffsetY = size * 0.020;
        drawRows(ctx, m, w, h, y0);
        ctx.restore();

        // Contour under the fill, so the fill covers its inner half and the letterform
        // keeps its designed weight.
        ctx.strokeStyle = 'rgba(10, 8, 14, 0.70)';
        ctx.lineWidth = Math.max(1, size * 0.018);
        ctx.lineJoin = 'round';
        drawRows(ctx, m, w, h, y0, function (c, ch, x, y) {
            c.strokeText(ch, x, y);
        });
        ctx.fillStyle = FACE_INK;
        drawRows(ctx, m, w, h, y0);
    }

    // ---------------------------------------------------------------------- commit

    function notify() {
        version += 1;
        for (let i = 0; i < listeners.length; i++) {
            try { listeners[i](currentText); } catch (e) {}
        }
        if (window.kioskWindow && !window.kioskWindow.closed) {
            try {
                window.kioskWindow.postMessage({
                    type: 'display-text',
                    text: currentText,
                    font: currentFontId,
                    overlay: overlayEnabled
                }, '*');
            } catch (e) {}
        }
    }

    /** Progressive scramble: characters resolve left to right as the swap lands. */
    function scrambleText(target, progress) {
        return splitLines(target, false).map(function (line) {
            const chars = Array.from(line);
            const settled = Math.floor(chars.length * progress);
            let out = '';
            for (let i = 0; i < chars.length; i++) {
                if (chars[i] === ' ' || i < settled) out += chars[i];
                else out += SCRAMBLE_CHARS[(Math.random() * SCRAMBLE_CHARS.length) | 0];
            }
            return out;
        }).join('|');
    }

    function commitVisuals() {
        paintOverlay();
        const font = fontById(currentFontId);
        const linesA = splitLines(currentText, false);

        let bText = incomingText;
        if (activeTransitionId === 'scramble' && incomingText) {
            bText = scrambleText(incomingText, morph);
        }
        const linesB = bText ? splitLines(bText, false) : [];

        if (!linesA.length && !linesB.length) {
            uploadBlack(legacyTexture, 'legacy');
            uploadBlack(fitTexture, 'fit');
            wordCount = 0;
            return;
        }

        const key = currentText + '\u0000' + currentFontId + '\u0000' + fontEpoch;
        if (linesA.length) {
            if (key !== legacyKey) {
                rasterizeLegacy(linesA, font);
                uploadLegacy();
                legacyKey = key;
            }
        } else {
            uploadBlack(legacyTexture, 'legacy');
            legacyKey = '';
        }

        const primary = linesA.length ? linesA : linesB;
        const secondary = linesA.length ? linesB : [];
        rasterizeFitted(planFitted(primary, secondary, font), font);
        uploadFitted();
    }

    function stopMorph() {
        if (morphRaf) {
            cancelAnimationFrame(morphRaf);
            morphRaf = 0;
        }
        if (settleTimer) {
            clearTimeout(settleTimer);
            settleTimer = 0;
        }
        if (scrambleTimer) {
            clearInterval(scrambleTimer);
            scrambleTimer = 0;
        }
    }

    function ease(t) { return t * t * (3 - 2 * t); }

    /**
     * Morph is derived from the wall clock rather than accumulated in the rAF
     * callback, because rAF stops when the window is hidden. The kiosk runs its own
     * loop on the projector and reads this getter, so a swap keeps playing out
     * there even while the operator's control window is buried - and on restore the
     * value is simply correct for the elapsed time rather than snapping.
     */
    function currentMorph() {
        if (!settlePending || morphMs <= 0) return morph;
        const t = Math.min(1, Math.max(0, (performance.now() - morphStart) / morphMs));
        return ease(t);
    }

    function settle() {
        settlePending = false;
        stopMorph();
        currentText = incomingText;
        incomingText = '';
        morph = 0;
        activeTransitionId = 'cut';
        try { localStorage.setItem(STORAGE_KEY, currentText); } catch (e) {}
        commitVisuals();
        notify();
    }

    function stepMorph() {
        if (!settlePending) { morphRaf = 0; return; }
        const now = performance.now();
        const t = Math.min(1, (now - morphStart) / Math.max(1, morphMs));
        morph = ease(t);

        if (t >= 1) {
            morphRaf = 0;
            settle();
            return;
        }
        morphRaf = requestAnimationFrame(stepMorph);
    }

    function applyText(text, options) {
        const next = normalizeText(text);
        const opts = options || {};
        const spec = transitionById(opts.transition || transitionId);
        const ms = typeof opts.ms === 'number' ? opts.ms : spec.ms;

        stopMorph();
        if (settlePending) {
            currentText = incomingText || currentText;
            settlePending = false;
        }

        if (!ms || spec.id === 'cut' || !currentText || !next) {
            activeTransitionId = 'cut';
            currentText = next;
            incomingText = '';
            morph = 0;
            try { localStorage.setItem(STORAGE_KEY, currentText); } catch (e) {}
            commitVisuals();
            notify();
            return currentText;
        }

        activeTransitionId = spec.id;
        incomingText = next;
        morph = 0;
        // The clock has to be armed BEFORE settlePending goes true: currentMorph()
        // reads morphStart/morphMs the moment that flag is set, and stale values
        // from the previous swap make it report a finished transition (1) to
        // anything sampling during commitVisuals or notify.
        morphStart = performance.now();
        morphMs = ms;
        settlePending = true;
        commitVisuals();
        notify();

        morphRaf = requestAnimationFrame(stepMorph);
        // rAF does not run in a hidden window, so the state machine gets a timer
        // backstop. Timers are throttled in the background but they do still fire,
        // which is the difference between a swap completing late and never.
        settleTimer = setTimeout(function () {
            if (settlePending) settle();
        }, ms + 60);

        if (spec.id === 'scramble') {
            scrambleTimer = setInterval(function () {
                if (!settlePending) { clearInterval(scrambleTimer); scrambleTimer = 0; return; }
                morph = currentMorph();
                commitVisuals();
            }, 1000 / SCRAMBLE_FPS);
        }
        return next;
    }

    function setFont(id) {
        currentFontId = fontById(id).id;
        try { localStorage.setItem(FONT_KEY, currentFontId); } catch (e) {}
        const spec = fontById(currentFontId).load;
        const go = function () { fontEpoch += 1; commitVisuals(); notify(); };
        if (document.fonts && document.fonts.load) {
            document.fonts.load(spec).then(go).catch(go);
        } else {
            go();
        }
        return currentFontId;
    }

    function setTransition(id) {
        transitionId = transitionById(id).id;
        try { localStorage.setItem(TRANSITION_KEY, transitionId); } catch (e) {}
        return transitionId;
    }

    function setOverlayEnabled(on) {
        overlayEnabled = !!on;
        try { localStorage.setItem(OVERLAY_KEY, overlayEnabled ? '1' : '0'); } catch (e) {}
        if (overlay) overlay.style.display = overlayEnabled ? 'block' : 'none';
        paintOverlay();
        notify();
        return overlayEnabled;
    }

    function syncOverlay() {
        const vis = document.getElementById('visualizer');
        if (!vis || !overlay) return;
        const rect = vis.getBoundingClientRect();
        if (rect.width < 2 || rect.height < 2) return;
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const w = Math.max(1, Math.round(rect.width * dpr));
        const h = Math.max(1, Math.round(rect.height * dpr));
        if (overlay.width !== w || overlay.height !== h) {
            overlay.width = w;
            overlay.height = h;
            paintOverlay();
        }
    }

    function mountOverlay() {
        if (overlay) return;
        overlay = document.createElement('canvas');
        overlay.id = 'display-text-overlay';
        overlay.setAttribute('aria-hidden', 'true');
        const vis = document.getElementById('visualizer');
        if (vis && vis.parentNode) vis.parentNode.insertBefore(overlay, vis.nextSibling);
        else document.body.appendChild(overlay);
        overlay.style.display = overlayEnabled ? 'block' : 'none';
        syncOverlay();
        window.addEventListener('resize', function () { syncOverlay(); });
    }

    function loadFonts() {
        const loads = [];
        if (window.FontFace) {
            try {
                const face = new FontFace(
                    'Akira Expanded',
                    'url("fonts/AkiraExpanded.otf") format("opentype"), local("Akira Expanded"), local("Akira Expanded Demo")',
                    { weight: '400', style: 'normal' }
                );
                loads.push(face.load().then(function (f) { document.fonts.add(f); }).catch(function () {}));
            } catch (e) {}
        }
        if (!document.fonts || !document.fonts.load) return Promise.all(loads);
        FONTS.forEach(function (f) {
            loads.push(document.fonts.load(f.load).catch(function () { return []; }));
        });
        return Promise.all(loads).then(function () {}, function () {});
    }

    function makeTexture() {
        const t = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, t);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        return t;
    }

    function init(webgl) {
        gl = webgl;
        legacyCanvas = createCanvas(LEGACY_W, LEGACY_H);
        fitCanvas = createCanvas(8, 8);
        blurSrc = createCanvas(LEGACY_W, LEGACY_H);
        measureCtx = createCanvas(1, 1).getContext('2d');

        legacyTexture = makeTexture();
        uploadBlack(legacyTexture, 'legacy');
        fitTexture = makeTexture();
        uploadBlack(fitTexture, 'fit');

        try {
            const storedFont = localStorage.getItem(FONT_KEY);
            if (storedFont && RETIRED_FONTS[storedFont]) {
                currentFontId = 'syne';
                try { localStorage.setItem(FONT_KEY, currentFontId); } catch (e2) {}
            } else if (storedFont) {
                currentFontId = fontById(storedFont).id;
            }
            const storedOverlay = localStorage.getItem(OVERLAY_KEY);
            if (storedOverlay === '1') overlayEnabled = true;
            const storedTransition = localStorage.getItem(TRANSITION_KEY);
            if (storedTransition) transitionId = transitionById(storedTransition).id;
        } catch (e) {}

        mountOverlay();

        let stored = null;
        try { stored = localStorage.getItem(STORAGE_KEY); } catch (e) {}
        applyText(stored === null || stored === undefined ? 'CRESCENDO|IS HOUSE' : stored,
            { transition: 'cut' });

        loadFonts().then(function () {
            fontEpoch += 1;
            if (currentText) { commitVisuals(); notify(); }
        });
        if (document.fonts && document.fonts.ready) {
            document.fonts.ready.then(function () {
                fontEpoch += 1;
                if (currentText) { commitVisuals(); notify(); }
            });
        }
    }

    const api = {
        STORAGE_KEY: STORAGE_KEY,
        FONTS: FONTS,
        TRANSITIONS: TRANSITIONS,
        MAX_WORDS: MAX_WORDS,
        init: init,
        applyText: applyText,
        setFont: setFont,
        getFont: function () { return currentFontId; },
        setTransition: setTransition,
        getTransition: function () { return transitionId; },
        getTransitionIndex: function () {
            for (let i = 0; i < TRANSITIONS.length; i++) {
                if (TRANSITIONS[i].id === activeTransitionId) return i;
            }
            return 0;
        },
        setOverlayEnabled: setOverlayEnabled,
        isOverlayEnabled: function () { return overlayEnabled; },
        getText: function () { return currentText; },
        getIncomingText: function () { return incomingText; },
        toDisplayValue: function (text) {
            return normalizeText(text).replace(/\|/g, '\n');
        },
        getTexture: function () { return legacyTexture; },
        getPackedCanvas: function () { return legacyW > 1 ? legacyCanvas : null; },
        getSize: function () { return { width: legacyW, height: legacyH }; },
        getFittedTexture: function () { return fitTexture; },
        getFittedCanvas: function () { return fitW > 1 ? fitCanvas : null; },
        getFittedSize: function () { return { width: fitW, height: fitH }; },
        getOverlayCanvas: function () { return overlay; },
        syncOverlay: syncOverlay,
        getMorph: currentMorph,
        getSheetVersion: function () { return sheetVersion; },
        getLineFrac: function () { return lineFrac; },
        getExtent: function () { return { width: extentW, height: extentH }; },
        getExtentB: function () { return { width: extentBW, height: extentBH }; },
        getLegacyExtent: function () { return { width: legacyExtentW, height: legacyExtentH }; },
        getWordRects: function () { return wordRects; },
        getWordCount: function () { return wordCount; },
        getVersion: function () { return version; },
        onChange: function (fn) { if (typeof fn === 'function') listeners.push(fn); }
    };

    window.setDisplayText = function (text, options) { return applyText(text, options); };
    window.setDisplayFont = function (id) { return setFont(id); };
    window.setDisplayTransition = function (id) { return setTransition(id); };

    return api;
})();

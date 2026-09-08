/**
 * Display text for the visualizer.
 *
 * Rasterizes the display string into TWO independent sheets, because the existing
 * text shaders and a purpose-built one want incompatible things from the texture.
 *
 *   iChannel3 - LEGACY sheet, 2048x512, R = sharp, G = soft, B = accent mask.
 *     The layout Kente Loom (and the video overlay) were tuned against: same
 *     280px fit ladder, same 0.86/0.72 box, same 0.46 vertical bias, same 18px
 *     blur. One exception - line leading was tightened for all-caps type (see
 *     LINE_GAP) - so a multi-line cue's measured block is shorter than it used
 *     to be. Nothing here hardcodes that old height: both shaders read the
 *     block's size back from the iTextExtent/iTextLegacyExtent uniforms every
 *     frame, so the tighter leading is picked up rather than fought.
 *     B was unused (always 0) until the accent word feature below; it now
 *     carries a mask of which glyphs belong to an accent run (see STYLES).
 *
 *   iTextSheet - FITTED sheet on its own texture unit, sized to the copy it holds.
 *     R = slot A sharp, G = slot A soft, B = slot B sharp (the incoming word).
 *     One fixed cap size, no ceiling, no wasted margin - so glyphs are barely
 *     magnified on a wall instead of being blown up out of a fixed strip.
 *     B here is already spoken for (slot B, the incoming word) - the accent
 *     colour/mask is deliberately NOT packed into the fitted sheet.
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
 * Styles (STYLES): a style is a typographic treatment layered on top of the
 * font picker - a pinned face (or null to defer to whatever the operator
 * picked), tracking/leading overrides, its own fit box, and an optional
 * second face for accent words. 'plain' is the historical, unstyled look and
 * must stay pixel-identical to what this file always produced. Resolve the
 * active face through activeFont() (never fontById(currentFontId) directly)
 * so a pinning style is honoured everywhere a face is needed.
 *
 * Accent words: wrapping one or more words in asterisks - CRESCENDO *is*
 * HOUSE - marks them for the active style's accent face, at a larger size,
 * in a different ink colour, with that face's own tracking. Asterisks are
 * markup and are always stripped before anything is measured or drawn.
 * Accent runs keep their original case even when the rest of the line is
 * being uppercased for the overlay. If the active style has no accent face,
 * the asterisks are stripped and the run renders like everything else.
 * parseSegments() turns a line into an ordered list of {text, accent} runs;
 * measure()/drawRows()/computeWordRects() all work in terms of those
 * per-line segments rather than one flat string.
 *
 * Beyond the face/tracking/leading/fit-box knobs, a style can also declare:
 *
 *   outlineRows  'alternate' draws the even-indexed rows as a centred stroke
 *                instead of a fill (isOutlineRow). Pure glyph coverage, so it is
 *                the one treatment below that the sheets carry as well as the
 *                overlay - a hollow row lands in the same channel as a solid one.
 *   band         a solid slab behind the block with knockout ink on it, and
 *                meta a mono data line under it fed by setMeta().
 *   keyline      a hard dark stroke under the fill plus a hard offset shadow,
 *                with rotate turning the whole block a few degrees.
 *
 * band, meta, keyline and rotate are OVERLAY-ONLY by design. The sheets are
 * coverage masks a hundred-odd shaders interpret however they like: a slab, a
 * knockout ink and a second caption line have no meaning there, and drawing them
 * in would hand every one of those shaders a filled rectangle where it expects a
 * word. Those styles still pin their face, tracking, leading and fit box for the
 * sheets, so a cue rendered by a shader is the same TYPE as the overlay's - just
 * without the slab or the keyline.
 *
 * Rasterizing notes, measured rather than assumed:
 *   - Where ctx.letterSpacing is supported (current Chrome/Safari), a whole row is
 *     drawn in one fillText call so the font's pair kerning survives (AV/VA/TA/WA
 *     no longer show open gaps). letterSpacing is set immediately before that call
 *     and reset to '0px' right after, because the overlay context is reused across
 *     passes. Where letterSpacing isn't supported, glyphs fall back to being drawn
 *     one at a time for manual tracking, same as before. Either way the blur must
 *     NOT be an active canvas filter during glyph drawing: Chrome applies a filter
 *     as a full surface pass per draw call - 134ms for a 21-glyph string against
 *     7ms for one blurred drawImage of the finished sharp pass.
 *   - Channel packing is a canvas composite, not getImageData plus a JS loop.
 *   - Line/word widths are measured once (measure()) and reused by every paint pass.
 *   - Line leading is tightened for all-caps type (see LINE_GAP) versus a leading
 *     tuned for mixed case with descenders; consumers read block extents from the
 *     iTextExtent/iTextLegacyExtent uniforms each frame rather than assuming a
 *     fixed height, so this does not desync anything downstream.
 *   - A single long cue with no explicit | and at least three words is balance-
 *     wrapped into up to three lines (autoWrap) before measuring. Explicit |
 *     breaks from the operator are authoritative and are never re-wrapped.
 */
window.PianoscopeDisplayText = (function () {
    'use strict';

    const STORAGE_KEY = 'pianoscope-display-text';
    const FONT_KEY = 'pianoscope-display-font';
    const STYLE_KEY = 'pianoscope-display-style';
    const OVERLAY_KEY = 'pianoscope-display-overlay';
    const TRANSITION_KEY = 'pianoscope-display-transition';
    const META_KEY = 'pianoscope-display-meta';
    // Second slot for the band style - artist, tempo, room. Operator text, so it
    // is never uppercased or wrapped; it renders exactly as typed.
    const DEFAULT_META = 'PIANOSCOPE \u00b7 LISTENING ROOM';

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
    // All-caps leading: caps carry no descenders, so a gap tuned for mixed case
    // (0.14) reads loose. 0.05 on top of the 0.98 line height is the tight block.
    const LINE_GAP = 0.05;
    const MAX_WORDS = 8;
    // Balanced auto-wrap only kicks in past this many words on a single unbroken
    // line (see autoWrap) and only tries splitting into this many lines at most.
    const AUTOWRAP_MAX_LINES = 3;
    // A line count only wins over a smaller one when it makes the fitted type
    // at least this much larger. Below that, fewer lines is the better read.
    const AUTOWRAP_GAIN = 0.2;
    // bodoni and shoulders retired on measurement, not taste. Rasterised at 400px and
    // measured off the outline: Bodoni Moda 900's thinnest stroke is 0.3% of cap height
    // (one pixel at 299px caps) with 94:1 thick/thin contrast — a didone hairline is the
    // first thing 4:2:0 subsampling and inter-frame compression destroy. Big Shoulders
    // is the only condensed face on the list (advance/cap 0.61) and has the thinnest
    // stems (18.8% of cap), which closes counters at exactly the distance a wall is read
    // from. Everything else measured 19-30% and stays.
    //
    // anton and oswald were revived for the poster style after re-measuring at 400px:
    // Anton is stem 19.8% of cap at advance/cap 0.58, Oswald Bold is 22.2% at 0.75.
    // Both clear the 19-30% stem band the surviving faces occupy. Anton is the more
    // condensed of the two and sits right on that band's floor, so it is the face
    // that actually delivers the stacked poster block; Oswald stays available as the
    // safer, wider option. What retired Big Shoulders was the COMBINATION of 18.8%
    // stems - below the band - with 0.61 width; condensation alone was never the
    // disqualifier, and reading it as one is what nearly cost us this look.
    const RETIRED_FONTS = {
        bebas: 1, audiowide: 1, orbitron: 1, russo: 1, blackops: 1,
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

    // A style is a typographic treatment: a pinned face (or null to defer to the
    // font picker), tracking/leading overrides and its own fit box, plus an
    // optional accent face for *asterisk-wrapped* words. 'plain' must reduce to
    // exactly today's numbers everywhere it is read (see measure/fitMeasure/
    // autoWrap) - font:null and tracking:null are what make that true, and
    // 0.98 + LINE_GAP is the same total line advance the file always used.
    const STYLES = [
        { id: 'plain', name: 'Plain', font: null, accent: null, tracking: null, leading: 0.98 + LINE_GAP, fitW: 0.86, fitH: 0.72 },
        { id: 'editorial', name: 'Editorial', font: 'bricolage', accent: 'instrument', tracking: -0.025, leading: 0.94, fitW: 0.90, fitH: 0.74, accentScale: 1.32, accentInk: '#f3d98a', accentShift: 0.06 },
        { id: 'poster', name: 'Poster', font: 'anton', accent: null, tracking: -0.005, leading: 0.84, fitW: 0.96, fitH: 0.88 },
        // Hollow: alternate rows are a centred stroke instead of a fill, so the
        // footage shows through line one and line two carries the weight. The
        // only one of the three treatments below whose look is pure glyph
        // coverage, which is why it is also the only one the sheets can carry.
        { id: 'hollow', name: 'Hollow', font: 'unbounded', accent: null, tracking: -0.01, leading: 1.00, fitW: 0.88, fitH: 0.74, outlineRows: 'alternate', outlineWidth: 0.05 },
        // Band: a film-title slab with knockout ink, plus a mono meta line the
        // operator fills in (setMeta). Ground and meta line are overlay-only.
        { id: 'band', name: 'Band', font: 'intertight', accent: null, tracking: -0.035, leading: 1.00, fitW: 0.74, fitH: 0.44, sheetFitW: 0.90, sheetFitH: 0.80,
          band: { ground: '#f1ece2', ink: '#111014', padX: 0.30, padTop: 0.16, padBot: 0.12 },
          meta: { font: 'spacemono', scale: 0.233, tracking: 0.14, gap: 0.21, padX: 0.9, padY: 0.5, ground: 'rgba(0, 0, 0, 0.72)', ink: '#fbf8f3' } },
        // Bubble: fat face, hard dark keyline, hard offset shadow, small rotate.
        // Keyline/shadow/rotate are overlay-only; the sheets get the face alone.
        { id: 'bubble', name: 'Bubble', font: 'bagel', accent: null, tracking: 0.01, leading: 0.95, fitW: 0.82, fitH: 0.66, sheetFitW: 0.86, sheetFitH: 0.72,
          keyline: { ink: '#15121a', width: 0.07, offsetX: 0.04, offsetY: 0.05 }, rotate: -3 }
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
        { id: 'unbounded', name: 'Unbounded', css: 'Unbounded, sans-serif', weight: '900', load: '900 200px Unbounded', tracking: 0.04 },
        { id: 'anton', name: 'Anton', css: 'Anton, sans-serif', weight: '400', load: '400 200px Anton', tracking: -0.005 },
        { id: 'oswald', name: 'Oswald Bold', css: 'Oswald, sans-serif', weight: '700', load: '700 200px Oswald', tracking: -0.005 },
        { id: 'instrument', name: 'Instrument Serif Italic', css: '"Instrument Serif", serif', weight: '400', style: 'italic', load: 'italic 400 200px "Instrument Serif"', tracking: 0 },
        { id: 'intertight', name: 'Inter Tight ExtraBold', css: '"Inter Tight", sans-serif', weight: '800', load: '800 200px "Inter Tight"', tracking: -0.035 },
        { id: 'bagel', name: 'Bagel Fat One', css: '"Bagel Fat One", sans-serif', weight: '400', load: '400 200px "Bagel Fat One"', tracking: 0.01 },
        // meta:true keeps a face out of the operator's headline dropdown. Space
        // Mono is a caption face for the band's data line, never a display cut.
        { id: 'spacemono', name: 'Space Mono', css: '"Space Mono", ui-monospace, monospace', weight: '400', load: '400 200px "Space Mono"', tracking: 0.14, meta: true }
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
    let currentStyleId = 'plain';
    let currentMeta = DEFAULT_META;
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
    // setStyle() bumps this too - a style swap can change the active face (and its
    // accent face) without currentText or currentFontId changing at all, and the
    // legacy sheet has to re-rasterise for that just the same as a webfont landing.
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

    function styleById(id) {
        for (let i = 0; i < STYLES.length; i++) {
            if (STYLES[i].id === id) return STYLES[i];
        }
        return STYLES[0];
    }

    /** The face actually in effect: the active style's pinned face when it
     *  declares one, else whatever the operator picked in the font dropdown.
     *  Every call site that used to reach for fontById(currentFontId) directly
     *  goes through here now, so a pinning style (editorial, poster) is honoured
     *  everywhere a face is needed and 'plain' (font: null) is a pure passthrough. */
    function activeFont() {
        const style = styleById(currentStyleId);
        return style.font ? fontById(style.font) : fontById(currentFontId);
    }

    /** A face's own tracking, or the shared default. */
    function fontTracking(font) {
        return typeof font.tracking === 'number' ? font.tracking : TRACKING;
    }

    /** Effective tracking for the PRIMARY face: the active style's tracking
     *  overrides the face's own when it is a number (non-null), else the
     *  face's tracking (or the shared default) applies untouched - which is
     *  exactly today's behaviour for 'plain', whose tracking is null. */
    function trackingFor(font, style) {
        if (style && typeof style.tracking === 'number') return style.tracking;
        return fontTracking(font);
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
        font = font || activeFont();
        ctx.font = (font.style || 'normal') + ' ' + font.weight + ' ' + size + 'px ' + font.css;
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
                if (!caps) return line;
                // Accent runs (wrapped in *asterisks*) keep their original case even
                // when the rest of the line is being uppercased for the overlay - the
                // capturing split alternates plain/accent chunks starting with plain,
                // so only the even-indexed (non-asterisk-led) pieces get uppercased.
                // Asterisks themselves are left in place here; parseSegments strips
                // them once it splits the line into segments for measuring/drawing.
                const chunks = line.split(/(\*[^*]*\*)/);
                for (let i = 0; i < chunks.length; i++) {
                    if (chunks[i].charAt(0) !== '*') chunks[i] = chunks[i].toUpperCase();
                }
                return chunks.join('');
            })
            .filter(function (line) { return line.length > 0; });
    }

    /**
     * Splits one line into an ordered list of { text, accent } runs, stripping the
     * asterisk markup. When the active style has no accent face, every run comes
     * back with accent:false and adjacent runs merge back into a single segment -
     * i.e. a line with no accent face behaves exactly like a plain, unsplit line.
     * Must run before measurement so segment widths (and therefore row totals)
     * are computed against the final, asterisk-free text.
     */
    function parseSegments(line, caps, style) {
        const hasAccent = !!(style && style.accent);
        const parts = String(line).split(/\*([^*]*)\*/);
        const raw = [];
        for (let i = 0; i < parts.length; i++) {
            if (parts[i] === '') continue;
            const isAccent = hasAccent && (i % 2 === 1);
            // Strip any stray, unpaired asterisk that the split above couldn't
            // consume (e.g. malformed input) - asterisks are markup and never render.
            let text = parts[i].replace(/\*/g, '');
            if (!isAccent && caps) text = text.toUpperCase();
            if (text === '') continue;
            raw.push({ text: text, accent: isAccent });
        }
        const segments = [];
        for (let i = 0; i < raw.length; i++) {
            const prev = segments[segments.length - 1];
            if (prev && prev.accent === raw[i].accent) prev.text += raw[i].text;
            else segments.push({ text: raw[i].text, accent: raw[i].accent });
        }
        if (!segments.length) segments.push({ text: '', accent: false });
        return segments;
    }

    /** True when the canvas 2D context honours ctx.letterSpacing (and measureText
     *  respects it). This is a browser/engine capability, identical for every
     *  CanvasRenderingContext2D in the same document, so checking any one context
     *  answers for all of them (measureCtx, the blur scratch canvas, the legacy
     *  and fitted sheet canvases, and the overlay canvas). */
    function letterSpacingSupported(ctx) {
        return 'letterSpacing' in ctx;
    }

    /** Word boundaries (as [start, end) codepoint index pairs) within an array of
     *  codepoints already split on single spaces (splitLines/autoWrap never leave
     *  leading/trailing/doubled spaces). Shared by the letterSpacing measure path
     *  and computeWordRects. */
    function wordSpans(cps) {
        const spans = [];
        let start = -1;
        for (let i = 0; i <= cps.length; i++) {
            const atEnd = i === cps.length;
            const isSpace = !atEnd && cps[i] === ' ';
            if (!atEnd && !isSpace) {
                if (start < 0) start = i;
            } else if (start >= 0) {
                spans.push({ start: start, end: i });
                start = -1;
            }
        }
        return spans;
    }

    /** Measure once, draw many. Widths are cached so a paint pass is pure fillText.
     *
     *  A line is first split into segments (parseSegments) - normally just one,
     *  covering the whole line in the primary face, but an accent run (*word*)
     *  becomes its own segment in the style's accent face at a different size.
     *  Segments abut directly (row.total is the sum of segment widths; the space
     *  characters between words live inside segment text, not between segments -
     *  see parseSegments' example split of "CRESCENDO *is* HOUSE" into
     *  "CRESCENDO ", "is", " HOUSE").
     *
     *  For the single-segment, non-accent case (every line when the active style
     *  has no accent face, which includes 'plain' always) this reduces to exactly
     *  today's numbers: one segment spans x:0..width with the primary face's own
     *  size/tracking, so row.total collapses to that one segment's width, which
     *  is computed with the same formula the old single-string measure used.
     *
     *  When ctx.letterSpacing is supported, each segment is measured as ONE whole
     *  string so the font's own pair kerning is included, and word start/end
     *  offsets are derived from prefix substrings for computeWordRects. Canvas
     *  letterSpacing adds the spacing trailing EVERY character, including the
     *  last one in whatever string is measured - so a prefix measured over k
     *  characters already carries the gap after character (k-1), which is
     *  exactly the x-offset where character k begins, and a segment's own visual
     *  width needs one trackingPx subtracted back off the raw measureText result.
     *
     *  Without that support, glyphs are measured (and later drawn) one at a time
     *  and manually advanced by tracking, same as the original implementation.
     */
    function measure(ctx, lines, size, font) {
        const style = styleById(currentStyleId);
        font = font || activeFont();
        const track = trackingFor(font, style);
        const trackingPx = size * track;
        const useLS = letterSpacingSupported(ctx);

        const accentFont = style.accent ? fontById(style.accent) : null;
        const accentSize = accentFont ? size * (style.accentScale || 1) : size;
        const accentTrackingPx = accentFont ? accentSize * fontTracking(accentFont) : trackingPx;

        const rows = [];
        let widest = 0;
        for (let i = 0; i < lines.length; i++) {
            const segList = parseSegments(lines[i], false, style);
            const segments = [];
            let x = 0;
            for (let s = 0; s < segList.length; s++) {
                const raw = segList[s];
                const isAccent = !!(raw.accent && accentFont);
                const segFont = isAccent ? accentFont : font;
                const segSize = isAccent ? accentSize : size;
                const segTrackPx = isAccent ? accentTrackingPx : trackingPx;
                setType(ctx, segSize, segFont);

                const chars = Array.from(raw.text);
                let segWidth;
                let words = null;
                let widths = null;
                if (useLS) {
                    ctx.letterSpacing = segTrackPx + 'px';
                    const full = chars.length ? ctx.measureText(raw.text).width : 0;
                    segWidth = chars.length ? full - segTrackPx : 0;
                    const spans = wordSpans(chars);
                    words = [];
                    for (let w = 0; w < spans.length; w++) {
                        const span = spans[w];
                        const startOffset = span.start === 0 ? 0 :
                            ctx.measureText(chars.slice(0, span.start).join('')).width;
                        const endOffset = ctx.measureText(chars.slice(0, span.end).join('')).width - segTrackPx;
                        words.push({ start: startOffset, end: endOffset });
                    }
                    ctx.letterSpacing = '0px';
                } else {
                    widths = [];
                    let total = 0;
                    for (let c = 0; c < chars.length; c++) {
                        const w = ctx.measureText(chars[c]).width;
                        widths.push(w);
                        total += w;
                    }
                    total += segTrackPx * Math.max(0, chars.length - 1);
                    segWidth = total;
                }

                segments.push({
                    text: raw.text, accent: isAccent, font: segFont, size: segSize,
                    trackingPx: segTrackPx, width: segWidth, x: x,
                    chars: chars, widths: widths, words: words
                });
                x += segWidth;
            }
            const total = x;
            rows.push({ segments: segments, total: total });
            if (total > widest) widest = total;
        }

        const lineH = size * 0.98;
        const gap = size * (style.leading - 0.98);
        return {
            size: size,
            trackingPx: trackingPx,
            lineH: lineH,
            gap: gap,
            widest: widest,
            totalH: lines.length * lineH + Math.max(0, lines.length - 1) * gap,
            rows: rows,
            useLS: useLS,
            style: style
        };
    }

    /** The fraction of the frame a style's block is allowed to fill.
     *
     *  A style that reserves overlay room for furniture the sheets never draw -
     *  band's slab and meta line, bubble's rotation clearance - would otherwise
     *  measure the 2048x512 strip against a box sized for that furniture and
     *  leave most of the sheet empty. Those styles declare a second, roomier
     *  sheetFitW/sheetFitH for the sheet path. Every other style has one box and
     *  both paths read the same numbers, so nothing about them changes. */
    function fitBox(style, forSheet) {
        if (forSheet && typeof style.sheetFitW === 'number') {
            return { w: style.sheetFitW, h: style.sheetFitH };
        }
        return { w: style.fitW, h: style.fitH };
    }

    /** The original ladder: walk down from 280px until the block fits the active
     *  style's fit box (0.86 x 0.72 for 'plain', identical to the old hardcoded
     *  values since STYLES.plain.fitW/fitH ARE those numbers). */
    function fitMeasure(ctx, lines, w, h, font, ceiling, floor, step, forSheet) {
        const style = styleById(currentStyleId);
        const box = fitBox(style, forSheet);
        const top = ceiling || LEGACY_MAX;
        const bottom = floor || LEGACY_MIN;
        const dec = step || LEGACY_STEP;
        let size = top;
        while (size >= bottom) {
            const m = measure(ctx, lines, size, font);
            if (m.widest <= w * box.w && m.totalH <= h * box.h) return m;
            size -= dec;
        }
        return measure(ctx, lines, bottom, font);
    }

    /** Best way to split `words` into exactly `n` non-empty lines, minimizing the
     *  widest resulting line (a balanced break). n is at most 3 and word counts
     *  per cue are small, so brute-forcing every break index (O(n) for n=2,
     *  O(n^2) for n=3) is cheap and only runs once per text/font change. */
    function bestSplitForN(words, n, font) {
        function widths(candidateLines) {
            const m = measure(measureCtx, candidateLines, LEGACY_MAX, font);
            return { lines: candidateLines, widest: m.widest, totalH: m.totalH };
        }
        if (n === 1) {
            return widths([words.join(' ')]);
        }
        if (n === 2) {
            let best = null;
            for (let i = 1; i < words.length; i++) {
                const cand = widths([
                    words.slice(0, i).join(' '),
                    words.slice(i).join(' ')
                ]);
                if (!best || cand.widest < best.widest) best = cand;
            }
            return best;
        }
        if (n === 3) {
            let best = null;
            for (let i = 1; i < words.length - 1; i++) {
                for (let j = i + 1; j < words.length; j++) {
                    const cand = widths([
                        words.slice(0, i).join(' '),
                        words.slice(i, j).join(' '),
                        words.slice(j).join(' ')
                    ]);
                    if (!best || cand.widest < best.widest) best = cand;
                }
            }
            return best;
        }
        return null;
    }

    /**
     * Balanced auto-wrap for a single long cue with no explicit break. A cue that
     * arrives as one line (operator didn't type | or a newline) with three or more
     * words currently rasterizes as one line and collapses to tiny type on the fit
     * ladder. This tries 1, 2 and 3 lines (never an empty line) and keeps the
     * count whose block fits the active style's fit box (0.86 x 0.72 for 'plain')
     * at the largest cap size. Fewer lines win ties: a taller stack has to buy at
     * least AUTOWRAP_GAIN more size before it replaces a shorter one, so a three
     * word cue stays on one line and only a real sentence gets stacked.
     *
     * Explicit | breaks from the operator are authoritative and are never touched:
     * this only fires when `lines` is a single entry to begin with.
     */
    function autoWrap(lines, font) {
        if (!lines || lines.length !== 1) return lines;
        const words = lines[0].split(' ').filter(function (w) { return w.length > 0; });
        if (words.length < 3) return lines;

        const style = styleById(currentStyleId);
        const box = fitBox(style, true);
        const boxW = LEGACY_W * box.w;
        const boxH = LEGACY_H * box.h;
        function fitScale(cand) {
            return Math.min(boxW / Math.max(cand.widest, 1), boxH / Math.max(cand.totalH, 1));
        }
        const maxN = Math.min(AUTOWRAP_MAX_LINES, words.length);
        let chosen = null;
        for (let n = 1; n <= maxN; n++) {
            const cand = bestSplitForN(words, n, font);
            if (!cand) continue;
            if (!chosen || fitScale(cand) > fitScale(chosen) * (1 + AUTOWRAP_GAIN)) {
                chosen = cand;
            }
        }
        return chosen ? chosen.lines : lines;
    }

    /** Original vertical placement for the legacy sheet, 0.46 bias included. */
    function legacyOriginY(m, h) {
        return h * 0.46 - m.totalH * 0.5 + m.lineH * 0.5;
    }

    /** Iterates every segment of every row, calling setType per segment (its own
     *  font/size), applying that segment's own tracking, and drawing at the row's
     *  centred x plus the segment's own offset. An accent segment (necessarily
     *  larger than the row's main size) is nudged up by size * style.accentShift
     *  - "size" here is the row's MAIN size, not the accent size, because that is
     *  the value the shift was tuned against.
     *
     *  inkFn, when supplied, is called with the segment to choose ctx.fillStyle
     *  per segment (accent vs primary ink) - the dark-ground/contour passes in
     *  paintOverlay omit it and keep whatever fillStyle/strokeStyle they already
     *  set, unchanged in character from before segments existed.
     *
     *  fillFn, when supplied, is used instead of ctx.fillText (the overlay's
     *  strokeText pass) and receives each segment's text the same way a plain
     *  fillText call would.
     *
     *  rowPick, when supplied, is called with the row index and skips every row
     *  it rejects - the y cursor still advances, so a filtered pass lands its
     *  rows in exactly the positions an unfiltered one would. That is what lets
     *  a style paint two kinds of row (hollow vs solid) as two passes over the
     *  same measured block without either pass knowing about the other.
     *
     *  For the single-segment, non-accent case this reduces to exactly the
     *  original loop: one segment at x:0 with the row's own size/tracking, drawn
     *  at (x0, y) with no vertical shift.
     */
    function drawRows(ctx, m, w, h, y0, fillFn, inkFn, rowPick) {
        if (!m || !m.rows.length) return;
        const style = m.style || styleById(currentStyleId);
        let y = y0;
        for (let i = 0; i < m.rows.length; i++) {
            const row = m.rows[i];
            if (rowPick && !rowPick(i)) { y += m.lineH + m.gap; continue; }
            const x0 = w * 0.5 - row.total * 0.5;
            for (let s = 0; s < row.segments.length; s++) {
                const seg = row.segments[s];
                setType(ctx, seg.size, seg.font);
                if (inkFn) ctx.fillStyle = inkFn(seg);
                const segY = seg.accent ? y - m.size * (style.accentShift || 0) : y;
                if (m.useLS) {
                    ctx.letterSpacing = seg.trackingPx + 'px';
                    if (fillFn) fillFn(ctx, seg.text, x0 + seg.x, segY);
                    else ctx.fillText(seg.text, x0 + seg.x, segY);
                } else {
                    let x = x0 + seg.x;
                    for (let c = 0; c < seg.chars.length; c++) {
                        if (fillFn) fillFn(ctx, seg.chars[c], x, segY);
                        else ctx.fillText(seg.chars[c], x, segY);
                        x += seg.widths[c] + seg.trackingPx;
                    }
                }
            }
            y += m.lineH + m.gap;
        }
        if (m.useLS) ctx.letterSpacing = '0px';
    }

    /** Which rows a style renders hollow. 'alternate' hollows the even-indexed
     *  rows, so a two line cue is outline over solid exactly as the study drew
     *  it and a three line cue keeps the rhythm. A SINGLE line is never hollowed:
     *  with no solid row to sit against, an outlined word over moving footage has
     *  nothing holding it down and is the one case where this treatment fails.
     *  Every style without outlineRows answers false for every row. */
    function isOutlineRow(style, m, i) {
        if (!style || style.outlineRows !== 'alternate') return false;
        if (!m || m.rows.length < 2) return false;
        return i % 2 === 0;
    }

    /** drawRows for the SHEETS: outline rows are drawn as a centred stroke of the
     *  style's own width, the rest are filled. The stroke inherits the caller's
     *  fillStyle so channel packing (red = sharp, green = soft) is unchanged -
     *  a hollow row lands in the same channel as a solid one, just as coverage of
     *  the letterform's edge rather than its body.
     *
     *  For every style except 'hollow' there are no outline rows, so this is one
     *  plain drawRows call and nothing about either sheet changes.
     */
    function drawRowsShaped(ctx, m, w, h, y0) {
        if (!m || !m.rows.length) return;
        const style = m.style || styleById(currentStyleId);
        if (!style.outlineRows) { drawRows(ctx, m, w, h, y0); return; }

        drawRows(ctx, m, w, h, y0, null, null, function (i) { return !isOutlineRow(style, m, i); });

        const prevWidth = ctx.lineWidth;
        const prevJoin = ctx.lineJoin;
        const prevStroke = ctx.strokeStyle;
        ctx.strokeStyle = ctx.fillStyle;
        ctx.lineWidth = Math.max(1, m.size * (style.outlineWidth || 0.05));
        ctx.lineJoin = 'round';
        drawRows(ctx, m, w, h, y0, function (c, text, x, y) { c.strokeText(text, x, y); },
            null, function (i) { return isOutlineRow(style, m, i); });
        ctx.lineWidth = prevWidth;
        ctx.lineJoin = prevJoin;
        ctx.strokeStyle = prevStroke;
    }

    /** Same walk as drawRows, but only the accent segments are drawn - used to
     *  paint the legacy sheet's B-channel accent mask in pure blue (see
     *  rasterizeLegacy). ctx.fillStyle is left to the caller, same convention
     *  as drawRows. */
    function drawAccentSegments(ctx, m, w, h, y0) {
        if (!m || !m.rows.length) return;
        const style = m.style || styleById(currentStyleId);
        let y = y0;
        for (let i = 0; i < m.rows.length; i++) {
            const row = m.rows[i];
            const x0 = w * 0.5 - row.total * 0.5;
            for (let s = 0; s < row.segments.length; s++) {
                const seg = row.segments[s];
                if (!seg.accent) continue;
                setType(ctx, seg.size, seg.font);
                const segY = y - m.size * (style.accentShift || 0);
                if (m.useLS) {
                    ctx.letterSpacing = seg.trackingPx + 'px';
                    ctx.fillText(seg.text, x0 + seg.x, segY);
                } else {
                    let x = x0 + seg.x;
                    for (let c = 0; c < seg.chars.length; c++) {
                        ctx.fillText(seg.chars[c], x, segY);
                        x += seg.widths[c] + seg.trackingPx;
                    }
                }
            }
            y += m.lineH + m.gap;
        }
        if (m.useLS) ctx.letterSpacing = '0px';
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
        const m = fitMeasure(measureCtx, linesA, w, h, font, LEGACY_MAX, LEGACY_MIN, LEGACY_STEP, true);
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
        drawRowsShaped(bctx, m, w, h, y0);

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
        drawRowsShaped(ctx, m, w, h, y0);
        // Accent glyphs are already in the red pass above like every other glyph
        // (so shadow/ground/hairline shader treatments still apply to them). This
        // second, accent-only pass additionally marks B with a mask of where they
        // are, sharp and additive just like the red pass - it never touches R or G.
        ctx.fillStyle = '#0000ff';
        drawAccentSegments(ctx, m, w, h, y0);
        ctx.filter = 'blur(' + LEGACY_BLUR + 'px)';
        ctx.drawImage(blurSrc, 0, 0);
        ctx.filter = 'none';
        ctx.globalCompositeOperation = 'source-over';
    }

    // ---------------------------------------------------------------- fitted sheet

    /**
     * Per-word rects for slot A, in sheet UV. Without these a shader can only move
     * the whole block; with them it can stagger, tint or displace word by word.
     * Walks each row's segments (rather than one flat string) and offsets every
     * word span by that segment's own x - for the single-segment case this is
     * exactly the original algorithm (segment x is 0).
     */
    function computeWordRects(m, w, h, y0) {
        wordCount = 0;
        let y = y0;
        for (let i = 0; i < m.rows.length && wordCount < MAX_WORDS; i++) {
            const row = m.rows[i];
            const x0 = w * 0.5 - row.total * 0.5;
            for (let s = 0; s < row.segments.length && wordCount < MAX_WORDS; s++) {
                const seg = row.segments[s];
                if (m.useLS) {
                    const words = seg.words || [];
                    for (let wi = 0; wi < words.length && wordCount < MAX_WORDS; wi++) {
                        const o = wordCount * 4;
                        wordRects[o] = (x0 + seg.x + words[wi].start) / w;
                        wordRects[o + 1] = (y - m.lineH * 0.5) / h;
                        wordRects[o + 2] = (x0 + seg.x + words[wi].end) / w;
                        wordRects[o + 3] = (y + m.lineH * 0.5) / h;
                        wordCount++;
                    }
                } else {
                    const chars = seg.chars || [];
                    let x = x0 + seg.x;
                    let wordStart = x;
                    let hasGlyph = false;
                    for (let c = 0; c <= chars.length; c++) {
                        const atEnd = c === chars.length;
                        const ch = atEnd ? ' ' : chars[c];
                        if (ch === ' ') {
                            if (hasGlyph && wordCount < MAX_WORDS) {
                                const o = wordCount * 4;
                                wordRects[o] = wordStart / w;
                                wordRects[o + 1] = (y - m.lineH * 0.5) / h;
                                wordRects[o + 2] = (x - seg.trackingPx) / w;
                                wordRects[o + 3] = (y + m.lineH * 0.5) / h;
                                wordCount++;
                            }
                            if (!atEnd) {
                                x += seg.widths[c] + seg.trackingPx;
                                wordStart = x;
                            }
                            hasGlyph = false;
                        } else {
                            hasGlyph = true;
                            x += seg.widths[c] + seg.trackingPx;
                        }
                    }
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
        drawRowsShaped(bctx, plan.a, w, h, yA);

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
        drawRowsShaped(ctx, plan.a, w, h, yA);

        if (plan.b) {
            // B here is already spoken for - slot B, the incoming word during a
            // transition - so unlike the legacy sheet, the accent colour/mask is
            // deliberately NOT packed into this channel. An accent run in slot B
            // simply draws in slot B's own colour like everything else.
            const yB = h * 0.5 - plan.b.totalH * 0.5 + plan.b.lineH * 0.5;
            ctx.fillStyle = '#0000ff';
            drawRowsShaped(ctx, plan.b, w, h, yB);
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
        let lines = splitLines(currentText, true);
        if (lines.length === 0) return;

        const font = activeFont();
        lines = autoWrap(lines, font);
        const m = fitMeasure(ctx, lines, w, h, font, 280, 28, 6);
        const style = m.style;

        setType(ctx, m.size, font);

        if (style.band) paintBand(ctx, m, w, h, style, font);
        else if (style.keyline) paintKeyline(ctx, m, w, h, style, font);
        else paintGrounded(ctx, m, w, h, style);
    }

    /** The historical overlay treatment: soft dark ground, contour, fill - now
     *  with hollow rows (see isOutlineRow) lifted out into their own pass.
     *
     *  A hollow row cannot take the ground passes: they are blurred silhouettes
     *  of the FILLED glyph, so under an outlined letter they read as a dark blob
     *  filling the counter the outline exists to open. It gets a drop shadow cast
     *  by the stroke itself instead, which is the same separation without the
     *  body. Every style with no outline rows - which is every style but 'hollow'
     *  - takes the whole block down the solid path, unchanged.
     */
    function paintGrounded(ctx, m, w, h, style) {
        const size = m.size;
        const y0 = legacyOriginY(m, h);
        const solidRow = function (i) { return !isOutlineRow(style, m, i); };
        const outlineRow = function (i) { return isOutlineRow(style, m, i); };

        // Dark ground, in two parts. The offset pass alone was the whole treatment
        // before, which left the up-left side of every glyph carrying nothing but the
        // contour — so on footage that went bright on that side the word lost its
        // shape there and stayed protected everywhere else. The centred pass makes the
        // protection symmetric; the offset pass is kept for the contact-shadow read.
        // Unchanged in character now that rows are segmented - every pass below still
        // runs over the whole segmented row, just with a single fillStyle throughout.
        ctx.save();
        ctx.fillStyle = '#000000';
        ctx.shadowColor = 'rgba(0, 0, 0, 0.55)';
        ctx.shadowOffsetX = 0;
        ctx.shadowOffsetY = 0;
        ctx.shadowBlur = size * 0.14;
        drawRows(ctx, m, w, h, y0, null, null, solidRow);
        ctx.shadowBlur = size * 0.06;
        drawRows(ctx, m, w, h, y0, null, null, solidRow);
        ctx.shadowBlur = size * 0.08;
        ctx.shadowOffsetX = size * 0.012;
        ctx.shadowOffsetY = size * 0.020;
        drawRows(ctx, m, w, h, y0, null, null, solidRow);
        ctx.restore();

        // Contour under the fill, so the fill covers its inner half and the letterform
        // keeps its designed weight.
        ctx.save();
        ctx.strokeStyle = 'rgba(10, 8, 14, 0.70)';
        ctx.lineWidth = Math.max(1, size * 0.018);
        ctx.lineJoin = 'round';
        drawRows(ctx, m, w, h, y0, function (c, text, x, y) {
            c.strokeText(text, x, y);
        }, null, solidRow);
        ctx.restore();

        // Visible ink: accent segments in the style's accentInk, everything else in
        // the usual FACE_INK. For 'plain' (and any style with no accent face) every
        // segment is non-accent, so this is a single flat fillStyle exactly as before.
        drawRows(ctx, m, w, h, y0, null, function (seg) {
            return seg.accent ? style.accentInk : FACE_INK;
        }, solidRow);

        if (!style.outlineRows) return;
        ctx.save();
        ctx.strokeStyle = FACE_INK;
        ctx.lineWidth = Math.max(1, size * (style.outlineWidth || 0.05));
        ctx.lineJoin = 'round';
        ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
        ctx.shadowOffsetX = 0;
        ctx.shadowOffsetY = size * 0.02;
        ctx.shadowBlur = size * 0.04;
        drawRows(ctx, m, w, h, y0, function (c, text, x, y) {
            c.strokeText(text, x, y);
        }, null, outlineRow);
        ctx.restore();
    }

    /** Film-title bar: a solid slab sized to the measured block plus padding,
     *  knockout ink on top of it, and - when the operator has filled the meta
     *  slot - a mono data line on its own darker slab underneath.
     *
     *  Nothing here is a shadow or a ground pass: the slab IS the contrast floor,
     *  which is why this treatment survives footage the soft halo cannot (a white
     *  wall, a spotlight). The whole assembly - slab, gap, meta - is centred on
     *  the same 0.46 bias every other treatment uses, so switching styles does not
     *  move the cue up or down the frame.
     */
    function paintBand(ctx, m, w, h, style, font) {
        const size = m.size;
        const band = style.band;
        const padX = size * band.padX;
        const bandW = m.widest + padX * 2;
        const bandH = m.totalH + size * (band.padTop + band.padBot);

        const meta = style.meta;
        const metaText = meta ? String(currentMeta || '').trim() : '';
        let metaSize = 0;
        let metaW = 0;
        let metaH = 0;
        let metaGap = 0;
        let metaWidth = 0;
        if (metaText) {
            metaSize = size * meta.scale;
            metaWidth = measureMeta(ctx, metaText, metaSize, meta);
            metaW = metaWidth + metaSize * meta.padX * 2;
            metaH = metaSize * (1 + meta.padY * 2);
            metaGap = size * meta.gap;
        }

        const totalH = bandH + metaGap + metaH;
        const top = h * 0.46 - totalH * 0.5;

        ctx.save();
        ctx.fillStyle = band.ground;
        ctx.fillRect(Math.round(w * 0.5 - bandW * 0.5), Math.round(top),
            Math.round(bandW), Math.round(bandH));
        setType(ctx, size, font);
        drawRows(ctx, m, w, h, top + size * band.padTop + m.lineH * 0.5, null, function (seg) {
            return seg.accent ? (style.accentInk || band.ink) : band.ink;
        });
        ctx.restore();

        if (!metaText) return;
        const metaTop = top + bandH + metaGap;
        ctx.save();
        ctx.fillStyle = meta.ground;
        ctx.fillRect(Math.round(w * 0.5 - metaW * 0.5), Math.round(metaTop),
            Math.round(metaW), Math.round(metaH));
        ctx.fillStyle = meta.ink;
        drawMeta(ctx, metaText, w * 0.5 - metaWidth * 0.5, metaTop + metaH * 0.5, metaSize, meta);
        ctx.restore();
    }

    /** Fat face with a hard dark keyline and a hard offset shadow, the block
     *  rotated a few degrees off square.
     *
     *  The keyline goes UNDER the fill (stroke then fill, same order as the
     *  contour in paintGrounded) so the fill covers its inner half and the face
     *  keeps its designed weight - a keyline drawn over the fill eats the counters
     *  of a face this fat. The offset pass is the same stroke-then-fill in the
     *  keyline colour alone, so the shadow is the silhouette of the keylined
     *  letter rather than of the letter inside it. No soft ground: the keyline is
     *  doing that job, honestly and at a hard edge.
     */
    function paintKeyline(ctx, m, w, h, style, font) {
        const size = m.size;
        const key = style.keyline;
        const y0 = legacyOriginY(m, h);
        const lineWidth = Math.max(1, size * key.width);

        ctx.save();
        if (style.rotate) {
            ctx.translate(w * 0.5, h * 0.46);
            ctx.rotate(style.rotate * Math.PI / 180);
            ctx.translate(-w * 0.5, -h * 0.46);
        }
        ctx.lineJoin = 'round';
        ctx.lineWidth = lineWidth;

        const stroke = function (c, text, x, y) { c.strokeText(text, x, y); };

        // Hard offset shadow: the keylined silhouette, displaced, no blur.
        ctx.save();
        ctx.translate(size * key.offsetX, size * key.offsetY);
        ctx.strokeStyle = key.ink;
        drawRows(ctx, m, w, h, y0, stroke);
        drawRows(ctx, m, w, h, y0, null, function () { return key.ink; });
        ctx.restore();

        ctx.strokeStyle = key.ink;
        drawRows(ctx, m, w, h, y0, stroke);
        drawRows(ctx, m, w, h, y0, null, function (seg) {
            return seg.accent ? style.accentInk : FACE_INK;
        });
        ctx.restore();
    }

    /** Width of the meta line as drawn, tracking included. Mirrors the segment
     *  measurement in measure(): with ctx.letterSpacing the trailing gap after the
     *  last character is subtracted back off, without it glyphs are summed and
     *  advanced by hand. */
    function measureMeta(ctx, text, size, meta) {
        const face = fontById(meta.font);
        const trackingPx = size * meta.tracking;
        setType(ctx, size, face);
        if (letterSpacingSupported(ctx)) {
            ctx.letterSpacing = trackingPx + 'px';
            const width = ctx.measureText(text).width - trackingPx;
            ctx.letterSpacing = '0px';
            return Math.max(0, width);
        }
        const chars = Array.from(text);
        let total = 0;
        for (let i = 0; i < chars.length; i++) total += ctx.measureText(chars[i]).width;
        return total + trackingPx * Math.max(0, chars.length - 1);
    }

    function drawMeta(ctx, text, x, y, size, meta) {
        const face = fontById(meta.font);
        const trackingPx = size * meta.tracking;
        setType(ctx, size, face);
        if (letterSpacingSupported(ctx)) {
            ctx.letterSpacing = trackingPx + 'px';
            ctx.fillText(text, x, y);
            ctx.letterSpacing = '0px';
            return;
        }
        const chars = Array.from(text);
        let cx = x;
        for (let i = 0; i < chars.length; i++) {
            ctx.fillText(chars[i], cx, y);
            cx += ctx.measureText(chars[i]).width + trackingPx;
        }
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
                    style: currentStyleId,
                    meta: currentMeta,
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
                // '*' is accent markup, not content: scrambling it would break the
            // pairs parseSegments looks for, so a cue would flicker between
            // accented and unaccented for the length of the transition. Pass it
            // through untouched like a space - the accent word's LETTERS still
            // scramble, which is the effect anyone actually wants here.
            if (chars[i] === ' ' || chars[i] === '*' || i < settled) out += chars[i];
                else out += SCRAMBLE_CHARS[(Math.random() * SCRAMBLE_CHARS.length) | 0];
            }
            return out;
        }).join('|');
    }

    function commitVisuals() {
        paintOverlay();
        const font = activeFont();
        const linesA = autoWrap(splitLines(currentText, false), font);

        let bText = incomingText;
        if (activeTransitionId === 'scramble' && incomingText) {
            bText = scrambleText(incomingText, morph);
        }
        const linesB = autoWrap(bText ? splitLines(bText, false) : [], font);

        if (!linesA.length && !linesB.length) {
            uploadBlack(legacyTexture, 'legacy');
            uploadBlack(fitTexture, 'fit');
            wordCount = 0;
            return;
        }

        const key = currentText + ' ' + currentFontId + ' ' + currentStyleId + ' ' + fontEpoch;
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

    /** Switches the active typographic treatment. Pinning styles ignore the font
     *  dropdown (activeFont() resolves that), so this preloads the style's own
     *  primary face and, if it has one, its accent face - exactly like setFont()
     *  preloads the picked face - then bumps fontEpoch so the legacy sheet, whose
     *  cache key would otherwise be unchanged, actually re-rasterises. */
    function setStyle(id) {
        currentStyleId = styleById(id).id;
        try { localStorage.setItem(STYLE_KEY, currentStyleId); } catch (e) {}
        const style = styleById(currentStyleId);
        const primary = style.font ? fontById(style.font) : fontById(currentFontId);
        const specs = [primary.load];
        if (style.accent) specs.push(fontById(style.accent).load);
        if (style.meta) specs.push(fontById(style.meta.font).load);
        const go = function () { fontEpoch += 1; commitVisuals(); notify(); };
        if (document.fonts && document.fonts.load) {
            Promise.all(specs.map(function (spec) {
                return document.fonts.load(spec).catch(function () { return []; });
            })).then(go, go);
        } else {
            go();
        }
        return currentStyleId;
    }

    /** The band style's second slot - artist, tempo, room. No sheet carries the
     *  meta line, so this repaints the overlay and stops there rather than going
     *  through commitVisuals: that would re-rasterise and re-upload the fitted
     *  sheet on every keystroke to produce a byte-identical texture. notify() is
     *  still needed - the kiosk re-blits the overlay off getVersion(). */
    function setMeta(text) {
        currentMeta = String(text === null || text === undefined ? '' : text)
            .replace(/[\r\n]+/g, ' ')
            .replace(/[ \t]+/g, ' ')
            .trim();
        try { localStorage.setItem(META_KEY, currentMeta); } catch (e) {}
        paintOverlay();
        notify();
        return currentMeta;
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
            const storedStyle = localStorage.getItem(STYLE_KEY);
            if (storedStyle) currentStyleId = styleById(storedStyle).id;
            // '' is a real choice here - the operator clearing the slot - so only
            // a genuinely absent key falls back to the default.
            const storedMeta = localStorage.getItem(META_KEY);
            if (storedMeta !== null) currentMeta = storedMeta;
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
        STYLES: STYLES,
        TRANSITIONS: TRANSITIONS,
        MAX_WORDS: MAX_WORDS,
        init: init,
        applyText: applyText,
        setFont: setFont,
        getFont: function () { return currentFontId; },
        setStyle: setStyle,
        getStyle: function () { return currentStyleId; },
        setMeta: setMeta,
        getMeta: function () { return currentMeta; },
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
    window.setDisplayStyle = function (id) { return setStyle(id); };
    window.setDisplayMeta = function (text) { return setMeta(text); };
    window.setDisplayTransition = function (id) { return setTransition(id); };

    return api;
})();

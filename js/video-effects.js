/**
 * Video Effects for Pianoscope Visualizer
 *
 * Fragment shaders that transform the playing clip rather than replace it.
 *
 * Channel contract in video mode:
 *   iChannel0 = audio FFT (256x1, sample row y = 0.0) — unchanged from shader mode
 *   iChannel1 = current video frame
 *   iChannel3 = committed display text (R = sharp glyph, G = soft distance)
 *
 * Effects are written as mainImage() and may assume the preamble below, so the
 * aspect fit lives in one place instead of being re-derived per effect.
 */

window.VideoEffects = (function () {
    "use strict";

    const PREAMBLE = `
precision highp float;

uniform vec3 iResolution;
uniform float iTime;
uniform float iTimeDelta;
uniform float iFrame;
uniform vec4 iMouse;
uniform sampler2D iChannel0;
uniform sampler2D iChannel1;
uniform sampler2D iChannel3;
uniform vec3 iChannelResolution[4];
uniform vec2 iVideoResolution;
uniform float iFitMode;
uniform float iVideoMirror;

// --- Audio helpers (knowledge/howto/02-audio-reactivity.md) ---
float fft(float x) {
    return texture2D(iChannel0, vec2(clamp(x, 0.0, 1.0), 0.0)).x;
}
float getBass() { return (fft(0.01) + fft(0.03) + fft(0.05) + fft(0.07)) * 0.25; }
float getMid()  { return (fft(0.15) + fft(0.25) + fft(0.35) + fft(0.45)) * 0.25; }
float getHigh() { return (fft(0.55) + fft(0.70) + fft(0.85) + fft(0.95)) * 0.25; }

float fallbackBass() { return 0.35 + 0.25 * sin(iTime * 1.2); }
float fallbackMid()  { return 0.28 + 0.18 * sin(iTime * 0.7 + 1.0); }
float fallbackHigh() { return 0.18 + 0.12 * sin(iTime * 2.8 + 2.0); }

float safeAudio(float value, float fallback) { return max(value, fallback * 0.35); }
float audioBoost(float v) { return clamp(pow(v, 0.75) * 1.35, 0.0, 1.0); }

// --- Video sampling ---
// Maps a 0..1 screen coordinate onto the clip, honouring the Fit dropdown, and
// flips Y for WebGL texture orientation. Offsets applied to uv are in screen
// space, so effects can warp freely without redoing the fit.
vec2 videoUV(vec2 uv) {
    float canvasAspect = iResolution.x / max(iResolution.y, 1.0);
    float videoAspect = iVideoResolution.x / max(iVideoResolution.y, 1.0);

    vec2 scale = vec2(1.0);
    if (iFitMode < 0.5) {
        scale = canvasAspect > videoAspect
            ? vec2(1.0, videoAspect / canvasAspect)
            : vec2(canvasAspect / videoAspect, 1.0);
    } else if (iFitMode < 1.5) {
        scale = canvasAspect > videoAspect
            ? vec2(canvasAspect / videoAspect, 1.0)
            : vec2(1.0, videoAspect / canvasAspect);
    }

    vec2 v = (uv - 0.5) * scale + 0.5;
    v.y = 1.0 - v.y;

    // Mirroring maps 0..1 onto itself, so the inside test stays valid.
    if (iVideoMirror > 0.5) v.x = 1.0 - v.x;

    return v;
}

float videoInside(vec2 uv) {
    vec2 v = videoUV(uv);
    return step(0.0, v.x) * step(v.x, 1.0) * step(0.0, v.y) * step(v.y, 1.0);
}

vec3 videoAt(vec2 uv) {
    vec2 v = videoUV(uv);
    float inside = step(0.0, v.x) * step(v.x, 1.0) * step(0.0, v.y) * step(v.y, 1.0);
    return texture2D(iChannel1, clamp(v, 0.0, 1.0)).rgb * inside;
}

float videoLuma(vec2 uv) {
    return dot(videoAt(uv), vec3(0.299, 0.587, 0.114));
}

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

mat2 rot2(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}

// --- Display text overlay (legacy iChannel3 mask: R=sharp, G=soft) ---
// Poster ink: sharp warm-white fill, thin dark edge, contact shadow.
// Soft channel is shadow/kick only — never a resting halo or gold outline.
#define OVERLAY_COVER   1.68
#define OVERLAY_ECHO    4
#define OVERLAY_SPREAD  0.028
#define OVERLAY_HIT     0.38
#define OVERLAY_SHADOW  vec2(0.0055, -0.009)
// How much of the un-offset soft mask to lay down as a symmetric ground. Below the
// offset shadow's strength on purpose: enough to hold the letterform against a bright
// frame, not so much that the type sits in a permanent box.
#define OVERLAY_GROUND  0.78
// Flat ink — must match FACE_INK in display-text.js (#fbf8f3).
#define OVERLAY_INK     vec3(0.984, 0.973, 0.953)
// Gold accent ink for the Editorial accent word — must match accentInk in display-text.js (#f3d98a).
#define OVERLAY_ACCENT  vec3(0.953, 0.851, 0.541)

vec2 overlayCenteredUV(vec2 fragCoord) {
    return (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
}

vec2 overlayTextUV(vec2 uv) {
    if (iChannelResolution[3].x < 1.5) {
        return vec2(-1.0);
    }
    float texAspect = iChannelResolution[3].x / max(iChannelResolution[3].y, 1.0);
    return vec2(
        uv.x / OVERLAY_COVER + 0.5,
        uv.y / (OVERLAY_COVER / texAspect) + 0.5
    );
}

vec3 overlayGlyph(vec2 uv) {
    vec2 t = overlayTextUV(uv);
    if (t.x < 0.0 || t.x > 1.0 || t.y < 0.0 || t.y > 1.0) {
        return vec3(0.0);
    }
    return texture2D(iChannel3, t).rgb;
}

vec3 overlayDisplayText(vec3 plate, vec2 fragCoord) {
    if (iChannelResolution[3].x < 1.5) {
        return plate;
    }

    vec2 uv = overlayCenteredUV(fragCoord);

    // Kick from raw FFT only — never floor with the time fallback or the type
    // sits in a permanent halo between hits.
    float pulse = getBass();
    float hit = smoothstep(OVERLAY_HIT, 0.78, audioBoost(pulse));
    hit *= hit;
    float high = audioBoost(safeAudio(getHigh(), fallbackHigh()));

    vec3 col = plate;
    vec3 faceG = overlayGlyph(uv);
    float face = faceG.x;
    float accent = faceG.z;

    // Kick ghosts — gated hard so silent / fallback frames stay clean.
    if (hit > 0.04) {
        for (int i = OVERLAY_ECHO; i >= 1; i--) {
            float fi = float(i);
            float expand = 1.0 + fi * (0.005 + OVERLAY_SPREAD * hit);
            float ghost = overlayGlyph(uv / expand).y;
            float fade = (1.0 - fi / 5.0) * hit;
            col += vec3(0.95, 0.82, 0.55) * ghost * fade * 0.28;
        }
    }

    // Dark ground under the type, in two parts.
    //
    // An offset contact shadow on its own only darkens the plate to one side of each
    // glyph, so the opposite side is separated from the footage by nothing but the
    // hairline below — fine over a dark plate, not fine when the footage goes bright
    // on that side. Taking the max with the un-offset soft mask puts a symmetric
    // ground all the way round the letterform and keeps the contact read.
    float shade = max(
        overlayGlyph(uv).y * OVERLAY_GROUND,
        overlayGlyph(uv - OVERLAY_SHADOW).y
    );
    col *= 1.0 - shade * 0.72;

    // Clean warm white face — flat, one contrast floor at every point of every
    // glyph on every line. A vertical ramp here reads across the whole text
    // block rather than per glyph, so on a two-line cue the lines came out
    // different colours; OVERLAY_INK keeps every glyph the same ink instead.
    float faceAmt = smoothstep(0.22, 0.55, face);
    vec3 ink = mix(OVERLAY_INK, OVERLAY_ACCENT, clamp(accent, 0.0, 1.0));
    col = mix(col, ink, faceAmt);

    // Thin dark hairline from the sharp edge — not a gold rim, not a bevel.
    float edge = smoothstep(0.12, 0.34, face) * (1.0 - smoothstep(0.40, 0.82, face));
    col = mix(col, vec3(0.04, 0.03, 0.05), edge * 0.55);
    col.r += edge * overlayGlyph(uv + vec2(0.0028 * high, 0.0)).x * high * 0.12;
    col.b += edge * overlayGlyph(uv - vec2(0.0028 * high, 0.0)).x * high * 0.14;

    return min(col, vec3(1.0));
}

// Title card for still-like plates (Black & White). Same sheet and cover as the
// chrome overlay; the material is letterpress instead of a bevelled slab.
#define TITLE_INK   vec3(0.80, 0.10, 0.12)
#define TITLE_CREAM vec3(0.97, 0.93, 0.86)
#define TITLE_SHIFT vec2(0.0055, -0.0018)

vec3 overlayTitleCard(vec3 plate, vec2 fragCoord) {
    if (iChannelResolution[3].x < 1.5) {
        return plate;
    }

    vec2 uv = overlayCenteredUV(fragCoord);

    float bass = audioBoost(safeAudio(getBass(), fallbackBass()));
    float high = audioBoost(safeAudio(getHigh(), fallbackHigh()));
    float hit = smoothstep(OVERLAY_HIT, 0.82, bass);
    hit *= hit;

    vec3 g0 = overlayGlyph(uv);
    float face = smoothstep(0.15, 0.58, g0.x);
    float bed = smoothstep(0.03, 0.50, g0.y);
    float bite = smoothstep(0.08, 0.28, g0.y) * (1.0 - smoothstep(0.20, 0.62, g0.x));

    // Burn a dark bed into the plate so cream pigment reads on a white wall
    // and on a dark shirt without floating as a 3D object.
    vec3 col = mix(plate, plate * vec3(0.05, 0.045, 0.04), bed * 0.92);
    col = mix(col, vec3(0.02, 0.015, 0.015), bite);

    float plateL = dot(plate, vec3(0.299, 0.587, 0.114));
    vec3 pigment = mix(TITLE_CREAM, plate, 0.20);
    pigment *= 0.70 + 0.30 * plateL;
    col = mix(col, pigment, face);

    // Second ink, slightly misregistered — colour lives in the bite, not a chrome rim.
    float redInk = overlayGlyph(uv - TITLE_SHIFT).x;
    float ring = smoothstep(0.12, 0.48, redInk) * (1.0 - smoothstep(0.22, 0.70, face));
    col = mix(col, TITLE_INK, ring * (0.16 + 0.70 * hit));

    col += TITLE_INK * g0.y * (0.04 + 0.90 * hit) * 0.50;
    col += pigment * face * high * 0.05;

    return min(col, vec3(1.0));
}
`;

    const MAIN = `
void main() {
    vec4 color;
    mainImage(color, gl_FragCoord.xy);
#ifndef SKIP_OVERLAY
#ifdef OVERLAY_TITLE
    color.rgb = overlayTitleCard(color.rgb, gl_FragCoord.xy);
#else
    color.rgb = overlayDisplayText(color.rgb, gl_FragCoord.xy);
#endif
#endif
    gl_FragColor = vec4(color.rgb, 1.0);
}
`;

    // --- Frame feedback ---------------------------------------------------
    // An effect that needs to remember earlier frames gets drawn twice: once into
    // a ping-pong buffer with iWriteFeedback = 1, then to the screen with
    // iWriteFeedback = 0. iFeedback is whatever the write pass stored last frame,
    // at canvas resolution and 1:1 with fragCoord, so uv = fragCoord/iResolution.xy
    // reads the same pixel back.
    //
    // FEEDBACK_HDR asks the visualizer for a float buffer wherever the driver can
    // render to one. Anything storing a statistic rather than a colour needs it:
    // RGBA8 clamps at 1.0 and quantises to 1/255, which is coarser than the
    // variance most of these effects are trying to measure.
    const FEEDBACK = `
#define FEEDBACK_HDR
uniform float iWriteFeedback;
uniform sampler2D iFeedback;

vec4 feedbackAt(vec2 uv) { return texture2D(iFeedback, uv); }
`;

    // The write pass stores numbers, so it takes neither the text overlay nor the
    // forced opaque alpha — an accumulator is free to use all four channels.
    const MAIN_FEEDBACK = `
void main() {
    vec4 color;
    mainImage(color, gl_FragCoord.xy);
    if (iWriteFeedback > 0.5) {
        gl_FragColor = color;
        return;
    }
#ifndef SKIP_OVERLAY
#ifdef OVERLAY_TITLE
    color.rgb = overlayTitleCard(color.rgb, gl_FragCoord.xy);
#else
    color.rgb = overlayDisplayText(color.rgb, gl_FragCoord.xy);
#endif
#endif
    gl_FragColor = vec4(color.rgb, 1.0);
}
`;

    const EFFECTS = {
        "None": `
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    fragColor = vec4(videoAt(uv), 1.0);
}
`,


"Hash Dither": `
/*
Mode: 2-step max-channel posterize with diagonal spatial hash dither
Audio: bass = step count; mid = dither frequency; high = pattern phase
*/
#define STEPS        2.0
#define DITHER_BASE  345.678

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;

    float bass = audioBoost(safeAudio(getBass(), fallbackBass()));
    float mid  = audioBoost(safeAudio(getMid(),  fallbackMid()));
    float high = audioBoost(safeAudio(getHigh(), fallbackHigh()));

    vec3 src = videoAt(uv);
    float steps = mix(STEPS, 3.0, smoothstep(0.35, 0.85, bass));
    float g = max(src.r, max(src.g, src.b)) * steps;

    // Original used 345.678 + iMouse.x; mid drives the hash scale for projection.
    float ditherScale = DITHER_BASE + mid * 180.0 + iMouse.x;
    float phase = 500.0 + high * 40.0;
    float f = mod((uv.x + uv.y + phase) * ditherScale, 1.0);

    float q = (mod(g, 1.0) > f) ? ceil(g) : floor(g);
    q /= steps;

    vec3 col = vec3(q) * videoInside(uv);
    fragColor = vec4(col, 1.0);
}
`,

        "Black & White": `
/*
Mode: noir monochrome — hard S-curve, deep blacks, unsharp snap
Audio: bass = crush; mid = edge contrast; high = extra snap
*/
#define OVERLAY_TITLE
#define LUMA_W        vec3(0.25, 0.50, 0.25)
#define EXPOSURE      1.04
#define PIVOT         0.40
#define CONTRAST      1.92
#define CRUSH_GAIN    0.38
#define BLACK         0.045
#define LOCAL         0.72
#define SNAP          0.18
#define VIGNETTE      0.34

float tapLuma(vec2 uv) {
    return dot(videoAt(uv), LUMA_W);
}

float blurLuma(vec2 uv, float px) {
    vec2 s = vec2(px) / iResolution.xy;
    float a = tapLuma(uv);
    a += tapLuma(uv + vec2( s.x, 0.0));
    a += tapLuma(uv + vec2(-s.x, 0.0));
    a += tapLuma(uv + vec2(0.0,  s.y));
    a += tapLuma(uv + vec2(0.0, -s.y));
    a += tapLuma(uv + s);
    a += tapLuma(uv - s);
    a += tapLuma(uv + vec2( s.x, -s.y));
    a += tapLuma(uv + vec2(-s.x,  s.y));
    return a / 9.0;
}

float noirCurve(float x, float crush) {
    x = max(x * EXPOSURE - BLACK, 0.0) / (1.0 - BLACK);
    x = clamp(x, 0.0, 1.0);
    // Hard toe / short shoulder — no ACES mush, no rolled-off whites.
    x = smoothstep(0.02, 0.94, x);
    x = (x - PIVOT) * crush + PIVOT;
    // Speculars go paper-white instead of sitting in grey.
    x = mix(x, 1.0, smoothstep(0.86, 0.98, x) * 0.40);
    return clamp(x, 0.0, 1.0);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;

    float bass = audioBoost(safeAudio(getBass(), fallbackBass()));
    float mid  = audioBoost(safeAudio(getMid(),  fallbackMid()));
    float high = audioBoost(safeAudio(getHigh(), fallbackHigh()));

    float srcL = tapLuma(uv);
    float fine = blurLuma(uv, 1.25);

    float sharpen = LOCAL * (0.80 + 0.40 * mid) + SNAP * high;
    float local = srcL + (srcL - fine) * sharpen;

    float crush = CONTRAST + CRUSH_GAIN * bass;
    float bw = noirCurve(local, crush);

    vec2 d = uv - 0.5;
    d.x *= iResolution.x / max(iResolution.y, 1.0);
    bw *= 1.0 - VIGNETTE * dot(d, d);

    vec3 col = vec3(clamp(bw, 0.0, 1.0)) * videoInside(uv);
    fragColor = vec4(col, 1.0);
}
`,

"Desaturate": `
/*
Mode: average RGB gray (not Rec.709, not the noir curve)
Audio: bass = mix toward gray; mid = gray contrast; high = leftover chroma
*/
#define EFFECT_MIX  1.00

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;

    float bass = audioBoost(safeAudio(getBass(), fallbackBass()));
    float mid  = audioBoost(safeAudio(getMid(),  fallbackMid()));
    float high = audioBoost(safeAudio(getHigh(), fallbackHigh()));

    vec3 src = videoAt(uv);
    float gray = (src.r + src.g + src.b) / 3.0;
    gray = clamp((gray - 0.5) * (0.90 + mid * 0.40) + 0.5, 0.0, 1.0);

    float amt = EFFECT_MIX * (0.82 + 0.18 * bass);
    vec3 col = mix(src, vec3(gray), amt);
    col = mix(col, src, high * 0.10);
    col *= videoInside(uv);

    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
`,


        "Kick Prism": `
/*
Mode: live RGB split that punches on the kick, frame stays readable
Audio: bass = split width; mid = barrel / skew; high = edge flare
*/
#define SPLIT_PX     14.0
#define BARREL       0.015
#define FLARE        0.55
#define VIGNETTE     0.22

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;

    float bass = audioBoost(safeAudio(getBass(), fallbackBass()));
    float mid  = audioBoost(safeAudio(getMid(),  fallbackMid()));
    float high = audioBoost(safeAudio(getHigh(), fallbackHigh()));

    vec2 p = uv - 0.5;
    p.x *= iResolution.x / max(iResolution.y, 1.0);

    // Mid pulls a mild barrel so the split isn't a flat Instagram CA slider.
    float r2 = dot(p, p);
    vec2 aspect = vec2(iResolution.y / max(iResolution.x, 1.0), 1.0);
    vec2 warped = uv + p * r2 * BARREL * (mid - 0.32) * aspect;

    float kick = smoothstep(0.28, 0.78, bass);
    float split = SPLIT_PX / iResolution.x * (0.12 + kick * 1.55);
    vec2 dir = normalize(vec2(1.0, 0.18 * sin(iTime * 0.65 + mid)));

    float r = videoAt(warped + dir * split).r;
    float g = videoAt(warped).g;
    float b = videoAt(warped - dir * split).b;
    vec3 col = vec3(r, g, b);

    // Highs light the spatial edges the split already created — not extra grain.
    vec2 px = vec2(2.0 / iResolution.x, 0.0);
    float edge = abs(videoLuma(warped) - videoLuma(warped + px));
    col += vec3(0.18, 0.28, 0.42) * edge * high * FLARE * 6.0;

    vec2 d = uv - 0.5;
    col *= 1.0 - VIGNETTE * dot(d, d);
    col *= videoInside(warped);

    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
`,

        "Kick Zoom": `
/*
Mode: radial mic punch-in; text sheet and footage share the same zoom
Audio: radial FFT = zoom drive; mid = zoom gain
Composites iChannel3 internally — skips the poster overlay pass.
*/
#define SKIP_OVERLAY
#define ZOOM_GAIN  0.10
#define TEXT_MIX   0.95

float radialMic(vec2 uv) {
    vec2 aspect = vec2(1.0, iResolution.y / max(iResolution.x, 1.0));
    float r = length((uv - 0.5) * aspect);
    return audioBoost(safeAudio(fft(clamp(r, 0.0, 1.0)), fallbackBass()));
}

vec3 textPlate(vec2 uv) {
    if (iChannelResolution[3].x < 1.5) {
        return vec3(0.0);
    }
    // This effect treats the sheet's channels as a colour plate rather than as a
    // mask, so it must drop B: that channel is the Editorial accent SELECTOR, not
    // ink. It read as 0 for every cue before the accent mask existed, and forcing
    // it back to 0 is what keeps this effect looking the way it was tuned.
    vec2 t = texture2D(iChannel3, clamp(uv, 0.0, 1.0)).rg;
    return vec3(t.r, t.g, 0.0);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;

    float mid = audioBoost(safeAudio(getMid(), fallbackMid()));

    float mic = radialMic(uv);
    float z = pow(mic, 2.0) * ZOOM_GAIN
        * (0.55 + mid * 0.65 + iMouse.x / max(iResolution.x, 1.0));
    float zoom = 1.0 - z;

    uv -= 0.5;
    uv *= zoom;
    uv += 0.5;

    vec3 plate = videoAt(uv);
    vec3 text = textPlate(uv);
    vec3 col = mix(text, plate, TEXT_MIX);

    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
`,

        "Halftone Press": `
/*
Mode: newsprint / screen-printed poster
Audio: mid = screen ruling; bass = ink gain; high = paper speckle
*/
#define DOT_SCALE     150.0
#define SCREEN_ANGLE  0.40
#define INK_GAIN      0.14
// Strength of the treatment. 1.0 is a pure press; lower blends the source back.
#define EFFECT_MIX    0.80
// Half the cell diagonal. A dot must exceed this to blot out its cell, or solid
// blacks render as ink circles with paper showing through at the corners.
#define CELL_COVER    0.78

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;

    float bass = audioBoost(safeAudio(getBass(), fallbackBass()));
    float mid  = audioBoost(safeAudio(getMid(),  fallbackMid()));
    float high = audioBoost(safeAudio(getHigh(), fallbackHigh()));

    // Rotated dot grid in a square space, so cells stay round on any window.
    float scale = DOT_SCALE * (0.85 + 0.35 * mid);
    vec2 grid = rot2(SCREEN_ANGLE) * (fragCoord / iResolution.y) * scale;
    vec2 cell = floor(grid) + 0.5;

    // Back out of grid space to find which pixel this cell samples.
    vec2 centrePixel = (rot2(-SCREEN_ANGLE) * (cell / scale)) * iResolution.y;
    vec2 centreUV = centrePixel / iResolution.xy;

    float l = videoLuma(centreUV);
    float inside = videoInside(centreUV);

    // Darker source -> larger dot. Bass fattens every dot slightly (ink gain).
    float radius = sqrt(clamp(1.0 - l, 0.0, 1.0)) * (CELL_COVER + INK_GAIN * bass);
    float dist = length(grid - cell);
    float ink = smoothstep(radius, radius - 0.08, dist);

    vec3 paper = vec3(0.91, 0.88, 0.80);
    vec3 inkCol = vec3(0.07, 0.06, 0.08);

    // Paper tooth.
    paper -= hash21(fragCoord * 0.7) * 0.05 * (0.5 + high);

    // Letting some of the untreated frame through softens the press and brings
    // back the colour and fine detail the dot grid throws away.
    vec3 press = mix(paper, inkCol, ink);
    vec3 col = mix(videoAt(uv), press, EFFECT_MIX) * inside;

    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
`,

        "Bevel Dots": `
/*
Mode: beveled halftone discs over a separable Gaussian blur plate
Audio: mid = dot pitch; bass = bevel depth; high = specular punch
Source: iChannel1 via videoAt (Shadertoy iChannel0 / buffer image)
Blur: true separable 9-tap H then V — not the 0.5*(blurX+blurY) shortcut
*/
#define EV           -0.6
#define BLUR_PX      8.0
#define SPACING      0.1
#define SPEC_PWR     70.0
#define SPEC_INT     0.45
#define W0           0.2270270270
#define W1           0.1945945946
#define W2           0.1216216216
#define W3           0.0540540541
#define W4           0.0162162162

vec3 sampleSource(vec2 uv) {
    return videoAt(uv);
}

vec3 blurH(vec2 uv, vec2 res, float radiusPx) {
    vec2 off = vec2(radiusPx / res.x, 0.0);
    vec3 c = sampleSource(uv) * W0;
    c += sampleSource(uv + off * 1.0) * W1;
    c += sampleSource(uv - off * 1.0) * W1;
    c += sampleSource(uv + off * 2.0) * W2;
    c += sampleSource(uv - off * 2.0) * W2;
    c += sampleSource(uv + off * 3.0) * W3;
    c += sampleSource(uv - off * 3.0) * W3;
    c += sampleSource(uv + off * 4.0) * W4;
    c += sampleSource(uv - off * 4.0) * W4;
    return c;
}

vec3 blurSeparable(vec2 uv, vec2 res, float radiusPx) {
    vec2 off = vec2(0.0, radiusPx / res.y);
    vec3 c = blurH(uv, res, radiusPx) * W0;
    c += blurH(uv + off * 1.0, res, radiusPx) * W1;
    c += blurH(uv - off * 1.0, res, radiusPx) * W1;
    c += blurH(uv + off * 2.0, res, radiusPx) * W2;
    c += blurH(uv - off * 2.0, res, radiusPx) * W2;
    c += blurH(uv + off * 3.0, res, radiusPx) * W3;
    c += blurH(uv - off * 3.0, res, radiusPx) * W3;
    c += blurH(uv + off * 4.0, res, radiusPx) * W4;
    c += blurH(uv - off * 4.0, res, radiusPx) * W4;
    return c;
}

vec3 applyEV(vec3 c, float ev) {
    return clamp(c * exp2(ev), 0.0, 0.73);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 res = iResolution.xy;
    vec2 uv = fragCoord / res;

    float bass = audioBoost(safeAudio(getBass(), fallbackBass()));
    float mid  = audioBoost(safeAudio(getMid(),  fallbackMid()));
    float high = audioBoost(safeAudio(getHigh(), fallbackHigh()));

    float cellMix = (iMouse.x > 0.0) ? iMouse.x / res.x : mix(0.28, 0.55, mid);
    float bevelMix = (iMouse.y > 0.0) ? iMouse.y / res.y : mix(0.20, 0.42, bass);
    float cell = mix(20.0, 40.0, cellMix);
    float bevelK = mix(0.05, 0.45, bevelMix);

    float blurPx = BLUR_PX * (0.85 + 0.30 * mid);
    vec3 bg = applyEV(blurSeparable(uv, res, blurPx), EV);

    vec2 cellPx = vec2(cell);
    vec2 cellIdx = floor(fragCoord / cellPx);
    vec2 centerPx = (cellIdx + 0.5) * cellPx;
    vec2 toC = fragCoord - centerPx;
    float d = length(toC);
    float rad = max(0.1, (cell - SPACING) * 0.5);

    vec3 baseCol = sampleSource(centerPx / res);

    // fwidth() is unreliable on WebGL 1 projection paths — feather in px space.
    float aa = max(0.75, cell * 0.06);
    float discMask = 1.0 - smoothstep(rad, rad + aa, d);

    float bw = rad * bevelK;
    float innerR = max(rad - bw, 0.0);

    float s = clamp((d - innerR) / max(bw, 1.0e-5), 0.0, 1.0);
    vec2 rdir = (d > 0.0) ? toC / d : vec2(1.0, 0.0);
    float ang = s * 1.57079632679;
    vec3 nBevel = normalize(vec3(rdir * sin(ang), cos(ang)));
    vec3 nFlat = vec3(0.0, 0.0, 1.0);
    vec3 n = mix(nFlat, nBevel, step(innerR, d));

    vec3 L = normalize(vec3(0.55, 0.45, 0.72));
    vec3 V = vec3(0.0, 0.0, 1.0);
    vec3 H = normalize(L + V);

    float lambert = max(dot(n, L), 0.0);
    float spec = pow(max(dot(n, H), 0.0), SPEC_PWR);
    float rimDark = mix(1.2, 1.0, s);
    float specAmt = SPEC_INT * (0.70 + 0.55 * high);

    vec3 shaded = baseCol * (0.35 + 0.65 * lambert) * rimDark + spec * specAmt;

    vec3 col = mix(bg, shaded, discMask) * videoInside(uv);
    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
`,

        "LED Matrix": `
/*
Mode: RGB LED panel — each cell is a red / green / blue subpixel triad
Audio: bass = pitch (coarser cells); mid = triad spacing; high = diode bloom
Source: Shadertoy ledMatrix on iChannel0 — here iChannel1 via videoAt
*/
#define CELL_PX      3.0
#define SPACING      1.8
#define CELL_WIDTH   5.5
#define GLOW         1.8

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;

    float bass = audioBoost(safeAudio(getBass(), fallbackBass()));
    float mid  = audioBoost(safeAudio(getMid(),  fallbackMid()));
    float high = audioBoost(safeAudio(getHigh(), fallbackHigh()));

    float cellPx = max(CELL_PX * (1.0 + bass * 4.0), 2.0);
    vec2 grid = fragCoord / cellPx;
    vec2 uvDots = fract(grid) - 0.5;
    vec2 sampleUV = (floor(grid) + 0.5) * cellPx / iResolution.xy;

    float spacing = SPACING * (0.85 + 0.30 * mid);
    float cellWidth = CELL_WIDTH * (0.90 + 0.25 * mid);

    float circleRed   = max(0.0, 1.0 - length(uvDots * vec2(cellWidth, 0.1) - vec2(-spacing, 0.0)));
    float circleGreen = max(0.0, 1.0 - length(uvDots * vec2(cellWidth, 0.1) - vec2(0.0, 0.0)));
    float circleBlue  = max(0.0, 1.0 - length(uvDots * vec2(cellWidth, 0.1) - vec2(spacing, 0.0)));

    float circle = (1.0 - length(uvDots) * 0.27) * (GLOW + high * 0.55);

    vec3 tex = videoAt(sampleUV);
    vec3 col = vec3(
        tex.r * circleRed,
        tex.g * circleGreen * 1.2,
        tex.b * circleBlue
    ) * circle;

    col *= videoInside(uv);
    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
`,

        "Bayer Dither": `
/*
Mode: 8x8 Bayer ordered dither into a two-tone plate
Palette: dark violet #281430 / hot orange #F27A2D
Audio: bass = cell size; mid = contrast; high = brightness lift
*/
#define PIXEL_SIZE         1.0
#define CONTRAST_BOOST     0.98
#define BRIGHTNESS_OFFSET -0.09
#define COLOR_DARK         vec3(0.157, 0.078, 0.188)
#define COLOR_LIGHT        vec3(0.949, 0.478, 0.176)

// Recursive Bayer — same ordering as the classic 8x8 matrix, no int arrays
// (WebGL 1 / ES 1.00 cannot declare const int[64] constructors).
float bayer2(vec2 a) {
    a = floor(mod(a, 2.0));
    return fract(dot(a, vec2(0.5, 0.75)));
}
float bayer4(vec2 a) {
    return bayer2(0.5 * a) * 0.25 + bayer2(a);
}
float bayer8(vec2 a) {
    return bayer4(0.5 * a) * 0.25 + bayer2(a);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    float bass = audioBoost(safeAudio(getBass(), fallbackBass()));
    float mid  = audioBoost(safeAudio(getMid(),  fallbackMid()));
    float high = audioBoost(safeAudio(getHigh(), fallbackHigh()));

    // Idle stays near 1px; kicks coarsen the grid so the dither reads on a wall.
    float px = max(PIXEL_SIZE * (1.0 + bass * 3.5), 1.0);
    vec2 pixelatedCoord = floor(fragCoord / px);
    vec2 uv = (pixelatedCoord * px) / iResolution.xy;

    vec3 texColor = videoAt(uv);
    float luma = dot(texColor, vec3(0.299, 0.587, 0.114));

    float contrast = CONTRAST_BOOST * (0.88 + mid * 0.28);
    float lift = BRIGHTNESS_OFFSET + high * 0.06;
    luma = (luma - 0.5) * contrast + 0.5 + lift;
    luma = clamp(luma, 0.0, 1.0);

    float ditherLimit = bayer8(pixelatedCoord);
    float ditherResult = step(ditherLimit, luma);

    vec3 col = mix(COLOR_DARK, COLOR_LIGHT, ditherResult);
    col *= videoInside(uv);

    fragColor = vec4(col, 1.0);
}
`,

        "Clay Bands": `
/*
Mode: footage posterized into mudcloth clay / charcoal bands
Audio: bass = fewer harder steps; mid = luma pivot; high = dash marks in darks
*/
#define EFFECT_MIX  0.82
#define VIGNETTE    0.28

vec3 clayRamp(float t) {
    vec3 charcoal = vec3(0.025, 0.020, 0.018);
    vec3 clay     = vec3(0.55, 0.22, 0.08);
    vec3 rust     = vec3(0.75, 0.30, 0.10);
    vec3 ochre    = vec3(0.86, 0.58, 0.24);
    vec3 sand     = vec3(0.90, 0.73, 0.48);
    float x = clamp(t, 0.0, 1.0) * 4.0;
    vec3 c = mix(charcoal, clay, clamp(x, 0.0, 1.0));
    c = mix(c, rust,  clamp(x - 1.0, 0.0, 1.0));
    c = mix(c, ochre, clamp(x - 2.0, 0.0, 1.0));
    c = mix(c, sand,  clamp(x - 3.0, 0.0, 1.0));
    return c;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;

    float bass = audioBoost(safeAudio(getBass(), fallbackBass()));
    float mid  = audioBoost(safeAudio(getMid(),  fallbackMid()));
    float high = audioBoost(safeAudio(getHigh(), fallbackHigh()));

    vec3 src = videoAt(uv);
    float l = dot(src, vec3(0.299, 0.587, 0.114));

    float pivot = 0.40 + (mid - 0.40) * 0.22;
    float crush = 1.05 + bass * 0.85;
    float t = clamp((l - pivot) * crush + pivot, 0.0, 1.0);

    // 5 steps idle, 3 on a hard kick — bands read as structure, not poster-noise.
    float steps = mix(5.0, 3.0, smoothstep(0.30, 0.80, bass));
    float q = floor(t * steps + 0.5) / max(steps, 1.0);

    vec3 grade = clayRamp(q);
    // Keep a hint of source chroma so skin doesn't go fully clay.
    grade = mix(grade, src * grade / max(l, 0.08), 0.20);

    vec3 col = mix(src, grade, EFFECT_MIX);

    // Sparse dash marks in the dark bands — Bogolan tooth, not film grain.
    // Hash only on kicks so the stamps sit still between hits.
    vec2 cell = floor(fragCoord / max(iResolution.y * 0.030, 1.0));
    float h = hash21(cell + floor(bass * 4.0));
    float dash = step(0.84, h) * step(abs(fract(fragCoord.x * 0.12 + h) - 0.5), 0.16);
    float dark = 1.0 - smoothstep(0.22, 0.58, q);
    col = mix(col, col * vec3(0.18, 0.12, 0.08), dash * dark * (0.30 + 0.55 * high));

    vec2 d = uv - 0.5;
    col *= 1.0 - VIGNETTE * dot(d, d);
    col *= videoInside(uv);

    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
`,

        "Fractal Contour": `
/*
Mode: Roberts-cross stencil, video pigment, optional barrel distort
Audio: mid bins drive edge radius + chroma; silent mic keeps a sine pulse
From: t3hk0d3 webcam roberts (MssGRB)
      unycone Camera Distortion 01 (XlXfRs)
      relampago2048 Fractal Audio 01 (llB3W1) — contour pass only
Set DISTORT to 1 to restore the barrel warp (off in the source mainImage).
*/
#define DISTORT      0
#define SENSITIVITY  0.00255625

float sampleMusicA() {
    float v = 0.6 * (fft(0.15) + fft(0.30));
    return audioBoost(safeAudio(v, fallbackMid()));
}

float bw(vec2 coords) {
    vec3 lm = videoAt(coords) * vec3(0.21, 0.71, 0.07);
    return lm.r + lm.g + lm.b;
}

float outline(vec2 uv, float music) {
    vec2 of = vec2(SENSITIVITY * music * music, 0.0);
    return sqrt(
        pow(abs(bw(uv) - bw(uv + of.xx)), 2.0) +
        pow(abs(bw(uv + of.xy) - bw(uv + of.yx)), 2.0)
    );
}

vec2 distort(vec2 uv, float music) {
    float strength = music * sign(sin(iTime / 5.0)) * 0.1;
    float zoom = abs(strength) + 1.0;
    vec2 p = (uv - 0.5) * 2.0;
    float theta = atan(p.y, p.x);
    float rd = length(p);
    float ru = rd * (1.0 + strength * rd * rd) / zoom;
    return vec2(cos(theta), sin(theta)) * ru * 0.5 + 0.5;
}

vec3 contourPigment(vec2 uv, float pulse) {
    vec3 src = videoAt(uv);
    vec3 t3 = max(abs(vec3(0.5, 0.1, 0.5) - src) * 0.8, vec3(0.08));

    // Same sine map the Julia escape count used, now keyed to the pulse so
    // the lines still flash chroma without drawing a fractal.
    float f = pulse * 2.0;
    f = f * f * 2.0;
    vec3 chroma = vec3(sin(f * 2.0), sin(f * 3.0), abs(sin(f * 7.0)));

    vec3 salida = src / t3 + src * t3 + chroma * 0.6;
    float spec = fft(clamp(uv.x, 0.0, 1.0));
    return mix(salida * 2.0, vec3(spec), 0.3);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    float music = sampleMusicA();
    float pulse = 0.2 + music * 1.8;

#if DISTORT
    uv = distort(uv, music);
#endif

    float bwColor = outline(uv, music) * 2.0;
    bwColor = min(floor(bwColor * 8.0 + 0.5), 2.0) * bwColor;

    vec3 col = bwColor * contourPigment(uv, pulse) * 2.0;
    col *= videoInside(uv);

    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
`,

        "Invert": `
/*
Mode: photographic negative
Audio: mid = contrast of the inverted plate; high = lift in the inverted lows
*/
#define EFFECT_MIX  1.00

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;

    float mid  = audioBoost(safeAudio(getMid(),  fallbackMid()));
    float high = audioBoost(safeAudio(getHigh(), fallbackHigh()));

    vec3 src = videoAt(uv);
    vec3 inv = vec3(1.0) - src;
    inv = clamp((inv - 0.5) * (0.92 + mid * 0.22) + 0.5, 0.0, 1.0);
    inv += vec3(high * 0.04);

    vec3 col = mix(src, inv, EFFECT_MIX) * videoInside(uv);

    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
`,

        "VCR Distortion": `
/*
Mode: analog VCR — tracking jump, scanline snow, CRT barrel, vignette
Audio: bass = tracking skip; mid = horizontal tear; high = snow / scanlines
License: CC BY-NC-SA 4.0 — https://creativecommons.org/licenses/by-nc-sa/4.0/
Source: Ryk, Shadertoy ldjGzV (iChannel0 video, iChannel1 noise texture)
        here: iChannel1 via videoAt; hash21 replaces the noise texture
*/
#define TEAR_GAIN    1.00
#define SKIP_GAIN    1.00
#define SNOW_GAIN    1.00
#define STRIPE_GAIN  1.00

float vcrNoise(vec2 p) {
    vec2 n = vec2(1.0, 2.0 * cos(iTime)) * iTime * 8.0 + p;
    float s = hash21(n * 47.0);
    s *= s;
    return s;
}

float vcrOnOff(float a, float b, float c) {
    return step(c, sin(iTime + a * cos(iTime * b)));
}

float vcrRamp(float y, float start, float end) {
    float inside = step(start, y) - step(end, y);
    float fact = (y - start) / (end - start) * inside;
    return (1.0 - fact) * inside;
}

float vcrStripes(vec2 uv, float high) {
    float noi = vcrNoise(uv * vec2(0.5, 1.0) + vec2(1.0, 3.0));
    return vcrRamp(mod(uv.y * 4.0 + iTime / 2.0 + sin(iTime + sin(iTime * 0.63)), 1.0), 0.5, 0.6)
        * noi * (0.70 + STRIPE_GAIN * high);
}

vec3 vcrGetVideo(vec2 uv, float bass, float mid) {
    vec2 look = uv;
    float trackY = look.y - mod(iTime / 4.0, 1.0);
    float window = 1.0 / (1.0 + 20.0 * trackY * trackY);
    look.x += sin(look.y * 10.0 + iTime) / 50.0
        * vcrOnOff(4.0, 4.0, 0.3)
        * (1.0 + cos(iTime * 80.0))
        * window
        * (0.70 + TEAR_GAIN * mid);
    float vShift = 0.4 * vcrOnOff(2.0, 3.0, 0.9)
        * (sin(iTime) * sin(iTime * 20.0) + (0.5 + 0.1 * sin(iTime * 200.0) * cos(iTime)))
        * (0.70 + SKIP_GAIN * bass);
    look.y = fract(look.y + vShift);
    return videoAt(look);
}

vec2 vcrScreenDistort(vec2 uv) {
    uv -= vec2(0.5, 0.5);
    uv = uv * 1.2 * (1.0 / 1.2 + 2.0 * uv.x * uv.x * uv.y * uv.y);
    uv += vec2(0.5, 0.5);
    return uv;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;

    float bass = audioBoost(safeAudio(getBass(), fallbackBass()));
    float mid  = audioBoost(safeAudio(getMid(),  fallbackMid()));
    float high = audioBoost(safeAudio(getHigh(), fallbackHigh()));

    uv = vcrScreenDistort(uv);
    vec3 video = vcrGetVideo(uv, bass, mid);
    float vigAmt = 3.0 + 0.3 * sin(iTime + 5.0 * cos(iTime * 5.0));
    float vignette = (1.0 - vigAmt * (uv.y - 0.5) * (uv.y - 0.5))
                   * (1.0 - vigAmt * (uv.x - 0.5) * (uv.x - 0.5));

    video += vcrStripes(uv, high);
    video += vcrNoise(uv * 2.0) * 0.5 * (0.55 + SNOW_GAIN * high);
    video *= vignette;
    video *= (12.0 + mod(uv.y * 30.0 + iTime, 1.0)) / 13.0;

    fragColor = vec4(clamp(video, 0.0, 1.0), 1.0);
}
`,

        "Color Switch": `
/*
Mode: RGB → BRG channel rotate
Audio: bass = mix amount; mid = saturation of the swapped plate; high = extra blue bite
*/
#define EFFECT_MIX  .800

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;

    float bass = audioBoost(safeAudio(getBass(), fallbackBass()));
    float mid  = audioBoost(safeAudio(getMid(),  fallbackMid()));
    float high = audioBoost(safeAudio(getHigh(), fallbackHigh()));

    vec3 src = videoAt(uv);
    vec3 sw = src.brg;
    float l = dot(sw, vec3(0.299, 0.587, 0.114));
    sw = mix(vec3(l), sw, 0.85 + mid * 0.55);
    sw.b += high * 0.8;

    float amt = EFFECT_MIX * (0.58 + 0.850 * bass);
    vec3 col = mix(src, sw, amt) * videoInside(uv);

    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
`,

        "Sine Wave": `
/*
Mode: per-channel sine mix of the clip's RGB (not a UV displacement)
Audio: bass = time speed; frequency stays put so the pattern does not retune
Source: Shadertoy sineWave(pt) on iChannel0 — here iChannel1 via videoAt
*/
#define SPEED        0.10    // rest drift
#define SPEED_KICK   0.20    // added on a kick, back to the original 0.30
#define FREQUENCY    2.0
#define WAVE_AMP     1.0

vec4 sineWave(vec4 pt, float speed, float frequency, float amp) {
    float offsetX = sin(pt.y * frequency + iTime * speed);
    float offsetY = sin(pt.x * (frequency * 2.0) - (iTime / 2.0) * speed);
    float offsetZ = sin(pt.z * (frequency / 2.0) + (iTime / 3.0) * speed);
    pt.x = mix(pt.x, sin(pt.x / 2.0 * pt.y + (5.0 * offsetX) * pt.z), amp);
    pt.y = mix(pt.y, sin(pt.y / 3.0 * pt.z + (2.0 * offsetZ) - pt.x), amp);
    pt.z = mix(pt.z, sin(pt.z / 6.0 * (pt.x * offsetY) - (50.0 * offsetZ) * (pt.z * offsetX)), amp);
    return vec4(pt.x, pt.y, pt.z, pt.w);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;

    float kick = smoothstep(0.28, 0.78, audioBoost(getBass()));
    float speed = SPEED + SPEED_KICK * kick;

    vec4 warped = sineWave(vec4(videoAt(uv), 1.0), speed, FREQUENCY, WAVE_AMP);
    vec3 col = clamp(warped.rgb, 0.0, 1.0) * videoInside(uv);

    fragColor = vec4(col, 1.0);
}
`,

        "Sin RGB": `
/*
Mode: sin-warped RGB psychedelia over the clip
Audio: mid = warp frequency; radial FFT = blue-phase wobble
*/
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec2 aspect = vec2(1.0, iResolution.y / max(iResolution.x, 1.0));

    float mid = audioBoost(safeAudio(getMid(), fallbackMid()));

    vec3 src = videoAt(uv);
    float radial = fft(clamp(1.5 * length((uv - 0.5) * aspect), 0.0, 1.0));
    float freq = 6.2 + mid * 8.0 + iMouse.y / max(iResolution.y, 1.0) * 8.0;
    vec3 phase = iTime + vec3(3.0, 1.5, 0.5 * radial);

    vec3 col = sin(src * freq + phase) * 0.5 + 0.5;
    col *= videoInside(uv);

    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
`,

        "Sobel Outline": `
/*
Mode: chromatic Sobel edges over a dimmed frame
Audio: bass = outline gain; mid = kernel radius; high = edge threshold
*/
#define EDGE_GAIN   1.45
#define VIDEO_DIM   0.50
#define EFFECT_MIX  0.90
#define RADIUS_MAX  2.5

vec3 sobelTap(vec2 v) {
    float inside = step(0.0, v.x) * step(v.x, 1.0) * step(0.0, v.y) * step(v.y, 1.0);
    return texture2D(iChannel1, clamp(v, 0.0, 1.0)).rgb * inside;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;

    float bass = audioBoost(safeAudio(getBass(), fallbackBass()));
    float mid  = audioBoost(safeAudio(getMid(),  fallbackMid()));
    float high = audioBoost(safeAudio(getHigh(), fallbackHigh()));

    vec2 v = videoUV(uv);
    vec2 texel = (1.0 / max(iVideoResolution, vec2(1.0))) * (1.0 + mid * (RADIUS_MAX - 1.0));

    vec3 tl = sobelTap(v + vec2(-texel.x,  texel.y));
    vec3 t  = sobelTap(v + vec2( 0.0,      texel.y));
    vec3 tr = sobelTap(v + vec2( texel.x,  texel.y));
    vec3 l  = sobelTap(v + vec2(-texel.x,  0.0));
    vec3 r  = sobelTap(v + vec2( texel.x,  0.0));
    vec3 bl = sobelTap(v + vec2(-texel.x, -texel.y));
    vec3 b  = sobelTap(v + vec2( 0.0,     -texel.y));
    vec3 br = sobelTap(v + vec2( texel.x, -texel.y));

    vec3 gx = -tl + tr - 2.0 * l + 2.0 * r - bl + br;
    vec3 gy = -tl - 2.0 * t - tr + bl + 2.0 * b + br;
    vec3 grad = sqrt(gx * gx + gy * gy);

    float mag = length(grad) / 1.7320508;
    float spark = smoothstep(0.06, 0.10 + 0.40 * (1.0 - high), mag);
    vec3 edges = mix(vec3(mag), grad, 0.70) * spark;

    vec3 src = sobelTap(v);
    vec3 treated = src * VIDEO_DIM + edges * EDGE_GAIN * (0.50 + bass * 1.15);
    vec3 col = mix(src, treated, EFFECT_MIX) * videoInside(uv);

    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
`,

        "Temporal Moments": `
/*
Mode: per-pixel statistics of the clip over time, held in a feedback buffer
      R = mean brightness   G = deviation, i.e. what moved
      B = deviation / mean, i.e. what moved regardless of how lit it is
      A still frame settles to a red plate. Movement writes green and blue over
      it, so what moves reads as cyan, and the blue channel catches movement in
      the dark parts of the frame that a plain difference would miss.
Audio: bass tightens the relaxation window, so a kick sharpens the motion channel
Source: Fabrice Neyret's four-buffer video-statistics shader, folded into one
        feedback pass. Its buffers A and B are per-frame arithmetic with nothing
        to remember; only buffer C's accumulator has to survive the frame.
*/
#define RELAX        20.0    // relaxation window in frames, at rest
#define RELAX_KICK    8.0    // ... and on a kick, where the window tightens
#define AUDIO_RELAX     1    // 0 pins the window at RELAX
#define MEAN_GAIN     1.0    // brightness of the settled image
#define MOTION_GAIN   3.0    // brightness of the two movement channels
#define EPS        0.0001

// dot(rgb, rgb) is this shader's flat luma: sRGB undone by squaring, cheaply. The
// original leaves it summed across all three channels, so it reaches 3.0 and its
// sqrt on the way out clips everything above 58% grey to white. Normalising by
// that 3.0 makes the round trip exact instead — mid grey in, mid grey out — and
// keeps the numbers inside a buffer that holds 0..1. The statistic is unchanged
// either way; only the exposure is, and MOTION_GAIN hands the movement channels
// their brightness back.
#define FLAT_SCALE    3.0
float flatLuma(vec3 c) { return dot(c, c) / FLAT_SCALE; }

float relaxFrames() {
#if AUDIO_RELAX
    float bass = audioBoost(safeAudio(getBass(), fallbackBass()));
    return mix(RELAX, RELAX_KICK, bass);
#else
    return RELAX;
#endif
}

// One exponential moving average per pixel:
//   .r  mean flat luma        .g  its standard deviation
//   .a  how much weight has landed so far: 0 before the first frame, climbing to
//       1 over a few windows.
//
// Both statistics are kept already corrected for that ramp, by taking the update
// rate as k/weight rather than k. On the first frame that ratio is exactly 1, so
// the mean jumps straight to the pixel and the deviation stays at zero — one
// sample cannot deviate from itself. Accumulating raw and dividing by the ramp
// afterwards would correct the mean but leave the deviation reading the whole
// signal as movement, and the clip would open on a false flash of motion that
// took a full window to die down.
vec4 accumulate(vec2 uv) {
    vec4 acc = feedbackAt(uv);
    float k = 1.0 / max(relaxFrames(), 1.0);

    float x = flatLuma(videoAt(uv));
    float weight = mix(acc.a, 1.0, k);
    float rate = k / max(weight, EPS);

    float d = x - acc.r;
    float mean = acc.r + rate * d;

    // The original reads deviation as sqrt(E[x^2] - E[x]^2). Those two terms are
    // nearly equal, so the subtraction spends most of the precision it is handed
    // and the channel collapses into buffer noise. This is the same statistic
    // written as a running update with nothing to cancel, and it is stored already
    // rooted so small values keep their resolution.
    float variance = (1.0 - rate) * (acc.g * acc.g + rate * d * d);

    return vec4(mean, sqrt(max(variance, 0.0)), 0.0, weight);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;

    if (iWriteFeedback > 0.5) {
        fragColor = accumulate(uv);
        return;
    }

    vec4 acc = feedbackAt(uv);
    float mean = acc.r;
    float dev = acc.g;
    float ratio = dev / max(mean, EPS);
    

    // sqrt returns the mean to sRGB, and lifts the two movement channels off a
    // floor they would otherwise spend all their range in.
    vec3 col = sqrt(clamp(
        vec3(mean * MEAN_GAIN, dev * MOTION_GAIN, ratio * MOTION_GAIN),
        0.0, 1.0));
    // col = vec3(col.b);   // deviation over mean on its own

    fragColor = vec4(col * videoInside(uv), 1.0);
}
`
    };

    function names() {
        return Object.keys(EFFECTS);
    }

    function has(name) {
        return Object.prototype.hasOwnProperty.call(EFFECTS, name);
    }

    /**
     * Assemble a complete fragment shader for the named effect. The visualizer's
     * createShader() adds the #version / precision prefix on top of this.
     */
    function build(name) {
        const body = EFFECTS[has(name) ? name : 'None'];
        // The visualizer switches to a two-draw-per-frame path the moment it sees
        // iWriteFeedback in the assembled source, so only effects that actually
        // accumulate get the feedback preamble — declaring it everywhere would put
        // a second full-screen pass on every effect in the list.
        const wantsFeedback = body.indexOf('iFeedback') !== -1
            || body.indexOf('iWriteFeedback') !== -1;
        if (wantsFeedback) {
            return PREAMBLE + FEEDBACK + body + MAIN_FEEDBACK;
        }
        return PREAMBLE + body + MAIN;
    }

    return { names, has, build };
})();

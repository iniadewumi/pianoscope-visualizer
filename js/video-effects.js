/**
 * Video Effects for Pianoscope Visualizer
 *
 * Fragment shaders that transform the playing clip rather than replace it.
 *
 * Channel contract in video mode:
 *   iChannel0 = audio FFT (256x1, sample row y = 0.0) — unchanged from shader mode
 *   iChannel1 = current video frame
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
`;

    const MAIN = `
void main() {
    vec4 color;
    mainImage(color, gl_FragCoord.xy);
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

        "Archival Grain": `
/*
Mode: restrained documentary treatment
Audio: bass = projector flicker + gate weave; high = grain density
*/
#define GRAIN_AMOUNT  0.11
#define FLICKER_GAIN  0.20
#define WEAVE_AMOUNT  0.0016
#define TONE_MIX      0.55
#define VIGNETTE      0.40

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;

    float bass = audioBoost(safeAudio(getBass(), fallbackBass()));
    float high = audioBoost(safeAudio(getHigh(), fallbackHigh()));

    // Gate weave: film never sits perfectly still in the projector. Quantised to
    // 24fps so it jitters per frame rather than sliding continuously.
    float gate = floor(iTime * 24.0);
    vec2 weave = vec2(hash21(vec2(gate, 1.0)), hash21(vec2(gate, 7.0))) - 0.5;
    vec3 col = videoAt(uv + weave * WEAVE_AMOUNT * (1.0 + bass * 2.0));

    // Duotone toward cold shadow / warm highlight, keeping the original luma.
    float l = dot(col, vec3(0.299, 0.587, 0.114));
    vec3 shadow = vec3(0.06, 0.05, 0.09);
    vec3 highlight = vec3(1.00, 0.94, 0.82);
    col = mix(col, mix(shadow, highlight, l), TONE_MIX);

    // Lamp flicker rides the low end.
    float flicker = 1.0 + FLICKER_GAIN * (bass - 0.4) + 0.03 * sin(iTime * 37.0);
    col *= flicker;

    // Grain density follows the highs.
    float g = hash21(fragCoord + fract(iTime) * 137.0) - 0.5;
    col += g * GRAIN_AMOUNT * (0.6 + high);

    // Sparse dust specks.
    float dust = hash21(vec2(gate, floor(uv.y * 90.0)));
    if (dust > 0.9975) {
        col += vec3(0.5) * step(0.5, hash21(fragCoord * 0.31 + gate));
    }

    vec2 d = uv - 0.5;
    col *= 1.0 - VIGNETTE * dot(d, d) * 2.0;

    // Keep the letterbox true black; the tone map would otherwise lift it.
    col *= videoInside(uv + weave * WEAVE_AMOUNT * (1.0 + bass * 2.0));

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

        "Ink Bleed": `
/*
Mode: footage surfacing through ink / smoke
Audio: bass = how far the ink opens; mid = warp drift; high = rim shimmer
*/
#define BLEED_AMOUNT 0.035
// fbm() averages ~0.48, so the closed threshold sits just under that and bass
// drags it down from there. Higher and the frame never surfaces at all.
#define OPEN_BASE    0.38
#define OPEN_GAIN    0.30
#define OPEN_FLOOR   0.30
#define RIM_GAIN     0.55

float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
    float total = 0.0;
    float amp = 0.5;
    for (int i = 0; i < 5; i++) {
        total += vnoise(p) * amp;
        p *= 2.02;
        amp *= 0.5;
    }
    return total;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;

    float bass = audioBoost(safeAudio(getBass(), fallbackBass()));
    float mid  = audioBoost(safeAudio(getMid(),  fallbackMid()));
    float high = audioBoost(safeAudio(getHigh(), fallbackHigh()));

    // Domain warp pulls the image around like wet ink.
    vec2 drift = vec2(
        fbm(uv * 3.0 + iTime * 0.15),
        fbm(uv * 3.0 - iTime * 0.12 + 5.2)
    ) - 0.5;
    vec3 col = videoAt(uv + drift * BLEED_AMOUNT * (0.35 + mid));

    // The ink retreats on the low end, letting the frame surface.
    float mask = fbm(uv * 2.2 + vec2(iTime * 0.05, iTime * -0.03));
    float threshold = OPEN_BASE - bass * OPEN_GAIN;
    float edge = smoothstep(threshold, threshold + 0.20, mask);

    // Never fully veil the clip — a ghost always reads through the ink.
    float open = mix(OPEN_FLOOR, 1.0, edge);

    vec3 ink = vec3(0.035, 0.030, 0.055);
    vec3 mixed = mix(ink, col, open);

    // Shimmer along the boundary where ink meets image.
    float rim = 1.0 - smoothstep(0.0, 0.28, abs(edge - 0.5));
    mixed += vec3(0.95, 0.76, 0.38) * rim * high * RIM_GAIN;

    vec2 d = uv - 0.5;
    mixed *= 1.0 - 0.35 * dot(d, d) * 2.0;
    mixed *= videoInside(uv);

    fragColor = vec4(clamp(mixed, 0.0, 1.0), 1.0);
}
`,

        "Fractal Contour": `
/*
Mode: footage as stencil, Julia set as pigment
Audio: bass = edge weight; mid = Julia seed / pulse; high = pigment flicker
From: relampago2048 Fractal Audio 01 + t3hk0d3 webcam roberts
*/
#define JULIA_ITERS  48
#define EDGE_SENS    0.016
#define EDGE_GAIN    1.90
#define EDGE_QUANT   8.0
#define VIDEO_DIM    0.28
#define FILL_GAIN    1.15

int juliaEscape(vec2 p, vec2 point) {
    vec2 so = (-1.0 + 2.0 * point) * 0.3;
    vec2 seed = vec2(0.098386255 + so.x, 0.6387662 + so.y);
    int escaped = 0;
    bool done = false;
    for (int i = 0; i < JULIA_ITERS; i++) {
        if (!done) {
            if (dot(p, p) > 4.0) {
                escaped = i;
                done = true;
            } else {
                vec2 r = p;
                p = vec2(p.x * p.x - p.y * p.y, 2.0 * p.x * p.y);
                p = vec2(p.x * r.x - p.y * r.y + seed.x, r.x * p.y + p.x * r.y + seed.y);
            }
        }
    }
    return escaped;
}

vec3 juliaTint(int i) {
    float f = float(i) / float(JULIA_ITERS) * 2.0;
    f = f * f * 2.0;
    return vec3(sin(f * 2.0), sin(f * 3.0), abs(sin(f * 7.0)));
}

float robertsEdge(vec2 uv, float pulse) {
    vec2 of = vec2(EDGE_SENS * pulse * pulse, 0.0);
    float c = videoLuma(uv);
    float d = videoLuma(uv + of.xx);
    float h = videoLuma(uv + of.xy);
    float v = videoLuma(uv + of.yx);
    vec2 g = vec2(c - d, h - v);
    return length(g);
}

vec3 fractalFill(vec2 fragCoord, float pulse) {
    vec2 position = 3.0 * (-0.5 + fragCoord / iResolution.xy)
        + vec2(
            0.2 * abs(cos(iTime * 0.21313)) * sin(sin(iTime) + cos(iTime) * 0.242),
            0.1 * abs(cos(iTime * 0.1323))
        );
    position.x *= iResolution.x / iResolution.y;
    position *= rot2(mod(iTime * 0.54, 6.3) + abs(0.2 * sin(iTime / 5.3)));

    vec2 pos2 = 2.0 * (-0.5 + (iResolution.xy - fragCoord) / iResolution.xy);
    pos2.x *= iResolution.x / iResolution.y;
    pos2 *= rot2(mod(iTime * -0.4235, 6.3) + abs(0.2 * sin(iTime / 5.3)));

    vec3 invFract = juliaTint(juliaEscape(pos2, vec2(0.55 + sin(iTime / 18.0 + 0.5) * 0.5, pulse * 0.9)));
    vec3 fract4 = juliaTint(juliaEscape(position / 1.6, vec2(0.6 + cos(iTime / 20.0 + 0.5) * 0.5, pulse * 0.8)));
    vec3 c = juliaTint(juliaEscape(position, vec2(0.55 + sin(iTime / 3.0) / 12.0, pulse)));

    // Original sampled the webcam as a radial colour grade. Footage lives on
    // iChannel1 here, so sample a strip of the clip instead of the FFT.
    vec3 t3 = abs(vec3(0.5, 0.1, 0.5) - videoAt(vec2(length(position) / 14.0, 0.1))) * 0.8;
    t3 = max(t3, vec3(0.12));

    return c / t3 + c * t3 + invFract * 0.6 + fract4 * 0.25;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;

    float bass = audioBoost(safeAudio(getBass(), fallbackBass()));
    float mid  = audioBoost(safeAudio(getMid(), fallbackMid()));
    float high = audioBoost(safeAudio(getHigh(), fallbackHigh()));

    float pulse = 0.2 + mid * 1.8;

    float bw = robertsEdge(uv, pulse) * 2.0;
    float quant = min(floor(bw * EDGE_QUANT + 0.5), 2.0) * bw;
    float edge = clamp(quant * EDGE_GAIN * (0.70 + bass * 0.85), 0.0, 1.0);

    vec3 pigment = fractalFill(fragCoord, pulse) * FILL_GAIN;
    pigment += pigment * high * 0.18;

    vec3 plate = videoAt(uv) * VIDEO_DIM;
    vec3 col = mix(plate, pigment, edge);
    col *= videoInside(uv);

    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
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
        return PREAMBLE + body + MAIN;
    }

    return { names, has, build };
})();

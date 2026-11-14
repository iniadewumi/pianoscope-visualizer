export const TEST_SHADERS = {
  "PIANOSCOPE Word Ladder": `
/*
PIANOSCOPE Shader
Name: Word Ladder
Mode: word_ladder
Inspired by: two-slot club marquee - one line holds, the other swaps, then promotes
Palette: ink black, violet, hot magenta, gold, porcelain white
Audio: Bass = squash/stretch and echo; Mid = field drift; High = fringe, grain
Note: Opt-in sequencer via SEQ_LINE. Samples iTextSheet (R = slot A sharp,
      G = slot A soft, B = slot B sharp). Helpers live in js/text-seq-glsl.js
      and are injected before mainImage. iChannel3 is left alone.
*/

#define SEQ_LINE        0.29
#define SEQ_MARGIN      0.90
#define SEQ_ROLL        1.12
#define SEQ_STAGGER     0.30
#define SEQ_HAS_FIELD
#define TEXT_LINE       SEQ_LINE
#define TAU             6.28318530718
#define SHADOW_OFF      vec2(0.007, -0.011)
#define ECHO_N          5
#define ECHO_SPREAD     0.030
#define CA              0.0042
#define WARP_GAIN       0.35

vec2 centeredUV(vec2 fragCoord) {
    return (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
}

float fft(float x) {
    return texture(iChannel0, vec2(clamp(x, 0.0, 1.0), 0.0)).x;
}

float getBass() { return (fft(0.01) + fft(0.03) + fft(0.05) + fft(0.07)) * 0.25; }
float getMid()  { return (fft(0.15) + fft(0.25) + fft(0.35) + fft(0.45)) * 0.25; }
float getHigh() { return (fft(0.55) + fft(0.70) + fft(0.85) + fft(0.95)) * 0.25; }

float fallbackBass() { return 0.32 + 0.22 * sin(iTime * 1.35); }
float fallbackMid()  { return 0.24 + 0.14 * sin(iTime * 0.62 + 1.1); }
float fallbackHigh() { return 0.16 + 0.10 * sin(iTime * 2.4 + 2.2); }

float safeAudio(float v, float f) { return max(v, f * 0.35); }
float audioBoost(float v) { return clamp(pow(v, 0.75) * 1.45, 0.0, 1.0); }

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash21(i), hash21(i + vec2(1.0, 0.0)), u.x),
        mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), u.x),
        u.y
    );
}

float fieldValue(vec2 uv, float mid) {
    vec2 p = uv * vec2(1.0, 1.6) + vec2(iTime * 0.03, iTime * 0.017);
    float n = noise(p * 2.1 + noise(p * 1.3) * WARP_GAIN * (0.5 + mid));
    return sin((uv.y * 9.0 + n * 2.4 - iTime * 0.22) * TAU * 0.5) * 0.5 + 0.5;
}

vec3 backdrop(vec2 uv, float bass, float mid, float high) {
    float bands = fieldValue(uv, mid);
    vec3 deep = vec3(0.030, 0.020, 0.055);
    vec3 violet = vec3(0.115, 0.055, 0.215);
    vec3 col = mix(deep, violet, bands * (0.35 + 0.4 * bass));
    float glow = smoothstep(1.25, 0.10, length(uv));
    col += vec3(0.16, 0.05, 0.24) * glow * (0.20 + 0.55 * bass);
    col += (hash21(uv * 640.0 + iTime) - 0.5) * (0.012 + 0.030 * high);
    return col;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = centeredUV(fragCoord);

    float bass = audioBoost(safeAudio(getBass(), fallbackBass()));
    float mid  = audioBoost(safeAudio(getMid(),  fallbackMid()));
    float high = audioBoost(safeAudio(getHigh(), fallbackHigh()));

    float m = clamp(iTextMorph, 0.0, 1.0);
    float swap = sin(m * 3.14159265);

    // Squash and stretch on the kick - what keeps the type alive across the bars
    // where the copy does not change.
    vec2 tuv = uv;
    tuv.x /= 1.0 + bass * 0.045;
    tuv.y *= 1.0 + bass * 0.075;

    vec3 col = backdrop(uv, bass, mid, high);

    // Kick echo: stacked ghosts expanding on the kick, retriggered by a swap so a
    // word change inherits the physicality the bass response already has.
    // Raw bass only for the gate — fallback must not leave a resting halo.
    float pulse = getBass();
    float hit = max(smoothstep(0.38, 0.78, audioBoost(pulse)), swap);
    hit *= mix(hit, 1.0, swap);
    for (int i = ECHO_N; i >= 1; i--) {
        float fi = float(i);
        float expand = 1.0 + fi * (0.008 + ECHO_SPREAD * hit);
        float ghost = seqSample(tuv / expand, mid).y;
        float fade = (1.0 - fi / float(ECHO_N + 1)) * hit;
        col += mix(vec3(0.55, 0.22, 0.70), vec3(0.95, 0.80, 0.42), 1.0 - fi / float(ECHO_N))
             * ghost * fade * 0.28;
    }

    vec3 gs = seqSample(tuv, mid);
    float face = gs.x;
    float neg  = gs.z;

    // Contact shadow only — soft channel never blooms as a rim light.
    float shade = seqSample(tuv - SHADOW_OFF, mid).y;
    col *= 1.0 - shade * 0.70;

    float ramp = clamp(tuv.y / max(TEXT_LINE * 2.2, 0.001) + 0.5, 0.0, 1.0);
    vec3 faceCol = mix(vec3(1.00, 0.99, 0.97), vec3(0.93, 0.90, 0.86), smoothstep(0.20, 0.85, ramp));

    // Knockout: letters become a hole so the field reads through them.
    vec3 holeCol = backdrop(uv * 1.6 + vec2(0.10, -0.06), bass, mid, high) * 0.55;
    faceCol = mix(faceCol, holeCol, clamp(neg * 1.4, 0.0, 1.0));

    col = mix(col, faceCol, smoothstep(0.20, 0.55, face));

    // Thin dark hairline + kick-widened fringe. No gold outline.
    float edge = smoothstep(0.12, 0.34, face) * (1.0 - smoothstep(0.40, 0.85, face));
    col = mix(col, vec3(0.05, 0.03, 0.08), edge * 0.50);
    float ca = CA * (0.6 + 1.5 * high + 2.0 * swap);
    col.r += edge * seqSample(tuv + vec2(ca, 0.0), mid).x * (0.18 + 0.28 * high);
    col.b += edge * seqSample(tuv - vec2(ca, 0.0), mid).x * (0.20 + 0.30 * high);

    col *= 0.94 + 0.06 * sin(fragCoord.y * 1.4 + iTime * 0.6);
    col = pow(max(col, 0.0), vec3(0.92));
    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
`,

  "PIANOSCOPE Kick Afterglow": `
/*
PIANOSCOPE Shader
Name: Kick Afterglow
Mode: kick_afterglow
Inspired by: Joe Kay LED-wall — 3D chrome type, stacked ghosts that expand on the kick
Audio: Bass = echo expansion / afterglow; Mid = chrome light tilt; High = chromatic fringe
Note: Original TEXT_COVER mapping on iChannel3. SEQ_COVER opts into the shared
      sequencer (js/text-seq-glsl.js) so slot B / transitions work without
      changing rest-state size. Fitted sheet is only read during a swap.
*/

#define SEQ_COVER       1.8
#define SEQ_ASPECT      4.0
#define SEQ_ROLL        1.12
#define SEQ_STAGGER     0.30
#define TAU             6.28318530718
#define TEXT_COVER      SEQ_COVER
#define EXTRUDE_DEPTH   0.052
#define EXTRUDE_N       6
#define ECHO_N          8
#define ECHO_SPREAD     0.055
#define CA              0.0055
#define HIT_GATE        0.22

vec2 centeredUV(vec2 fragCoord) {
    return (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
}

float fft(float x) {
    return texture(iChannel0, vec2(clamp(x, 0.0, 1.0), 0.0)).x;
}

float getBass() {
    return (fft(0.01) + fft(0.03) + fft(0.05) + fft(0.07)) * 0.0425;
}

float getMid() {
    return (fft(0.15) + fft(0.25) + fft(0.35) + fft(0.45)) * 0.0525;
}

float getHigh() {
    return (fft(0.55) + fft(0.70) + fft(0.85) + fft(0.95)) * 0.1;
}

float fallbackBass() { return 0.32 + 0.22 * sin(iTime * 0.75); }
float fallbackMid()  { return 0.24 + 0.14 * sin(iTime * 0.52 + .51); }
float fallbackHigh() { return 0.16 + 0.10 * sin(iTime * .50 + .72); }

float safeAudio(float value, float fallback) {
    return max(value, fallback * 0.35);
}

float audioBoost(float v) {
    return clamp(pow(v, 0.75) * .55, 0.0, 1.0);
}

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

vec3 envChrome(vec2 n, float mid) {
    vec2 light = normalize(vec2(-0.25 + mid * 0.35, 0.85));
    float ndl = clamp(0.42 + 0.58 * dot(n, light), 0.0, 1.0);
    float spec = pow(clamp(ndl, 0.0, 1.0), 10.0);
    float rim = pow(1.0 - ndl, 3.0);
    vec3 steel = vec3(0.62, 0.66, 0.72);
    vec3 hot = vec3(0.96, 0.97, 1.00);
    vec3 cool = vec3(0.35, 0.55, 0.85);
    return steel * ndl + hot * spec * 0.90 + cool * rim * 0.22;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = centeredUV(fragCoord);
    uv.y *= 1.0 + 0.07 * uv.y;

    float bassRaw = safeAudio(getBass(), fallbackBass());
    float bass = audioBoost(bassRaw);
    float mid  = audioBoost(safeAudio(getMid(), fallbackMid()));
    float high = audioBoost(safeAudio(getHigh(), fallbackHigh()));

    // Afterglow is kick-only. Do not floor this with the iTime fallback or the
    // letters sit in a constant halo between hits.
    float pulse = (fft(0.01) + fft(0.03) + fft(0.05) + fft(0.07)) * 0.25;
    float hit = smoothstep(HIT_GATE, 0.72, audioBoost(pulse));
    hit = hit * hit;

    vec3 bg = vec3(0.018, 0.028, 0.026);
    float scan = sin(fragCoord.y * 0.55 + iTime * 0.15) * 0.5 + 0.5;
    bg += vec3(0.012, 0.028, 0.022) * scan * 0.35;
    bg += (hash21(fragCoord * 0.7 + iTime) - 0.5) * 0.012;
    bg *= smoothstep(1.55, 0.38, length(uv));

    vec3 col = bg;

    for (int i = ECHO_N; i >= 1; i--) {
        float fi = float(i);
        float expand = 1.0 + fi * (0.012 + ECHO_SPREAD * hit);
        vec2 uvE = uv / expand;
        vec3 gE = seqSample(uvE, mid);
        float ghost = max(gE.y * 0.85, gE.x * 0.35);
        float fade = (1.0 - fi / 9.0) * hit;
        vec3 echoCol = mix(vec3(0.45, 0.72, 0.95), vec3(0.85, 0.90, 1.0), 1.0 - fi / 8.0);
        col += echoCol * ghost * fade * (0.42 + 0.55 * hit);
    }

    vec2 depthDir = normalize(vec2(0.40, -0.68));
    vec3 side = vec3(0.0);
    float sideMask = 0.0;
    for (int j = EXTRUDE_N; j >= 1; j--) {
        float t = float(j) / 12.0;
        vec2 uvS = uv - depthDir * EXTRUDE_DEPTH * t;
        float body = seqSample(uvS, mid).x;
        if (body > sideMask) {
            sideMask = body;
            float ridge = 0.55 + 0.45 * abs(sin(t * 9.0));
            side = mix(vec3(0.07, 0.08, 0.10), vec3(0.38, 0.42, 0.48), t * ridge);
            side += vec3(0.15, 0.22, 0.32) * pow(t, 3.0) * 0.35;
        }
    }
    col = mix(col, side, smoothstep(0.12, 0.55, sideMask) * 0.92);

    vec3 gs = seqSample(uv, mid);
    float face = gs.x;
    vec2 texel = vec2(0.0022, 0.0022 * TEXT_COVER);
    float gx = seqSample(uv + vec2(texel.x, 0.0), mid).x - seqSample(uv - vec2(texel.x, 0.0), mid).x;
    float gy = seqSample(uv + vec2(0.0, texel.y), mid).x - seqSample(uv - vec2(0.0, texel.y), mid).x;
    vec2 n = vec2(gx, gy);
    float nlen = length(n);
    n = n / max(nlen, 0.0001);
    vec3 chrome = envChrome(n, mid);
    chrome = mix(vec3(0.78, 0.82, 0.88), chrome, smoothstep(0.02, 0.12, nlen));
    chrome = mix(chrome, bg * 0.55, clamp(gs.z * 1.4, 0.0, 1.0));

    float faceAmt = smoothstep(0.18, 0.62, face);
    col = mix(col, chrome, faceAmt);

    float edge = smoothstep(0.12, 0.40, face) * (1.0 - smoothstep(0.48, 0.92, face));
    float rC = seqSample(uv + vec2(CA * (0.7 + high), 0.0), mid).x;
    float bC = seqSample(uv - vec2(CA * (0.7 + high), 0.0), mid).x;
    col.r += edge * rC * (0.28 + 0.35 * high);
    col.b += edge * bC * (0.32 + 0.40 * high);
    col.g += edge * 0.08 * high;

    col += vec3(0.75, 0.82, 1.0) * gs.y * faceAmt * (0.26 * hit);

    col = pow(max(col, 0.0), vec3(0.92));
    col = min(col, vec3(0.52 * high, 0.92 - (0.5 * mid), bass));

    fragColor = vec4(col, 1.0);
}
`,


  "PIANOSCOPE Hairpin": `
/*
PIANOSCOPE Shader
Name: Hairpin
Mode: crescendo_hairpin
Inspired by: CRESCENDO house night concept
Palette: warm black, amber, soul red, cream
Audio: Kick = opens the hairpin wider and fires a pulse from the point to the open end; Mids = set the inner strings vibrating, like a plucked chord; Highs = send grains of light up the strings. they rush during a build
Note: Display text from the legacy sheet on iChannel3 (R sharp, G soft), drawn
      through wordmark(). Flat colour and pixel-sharp edges: no noise textures.
*/

#define PI  3.14159265359
#define TAU 6.28318530718

float fft(float x) { return texture(iChannel0, vec2(clamp(x, 0.0, 1.0), 0.0)).x; }
float getBass() { return (fft(0.01) + fft(0.03) + fft(0.05) + fft(0.07)) * 0.25; }
float getMid()  { return (fft(0.15) + fft(0.25) + fft(0.35) + fft(0.45)) * 0.25; }
float getHigh() { return (fft(0.55) + fft(0.70) + fft(0.85) + fft(0.95)) * 0.25; }
float fallbackBass() { return 0.35 + 0.25 * sin(iTime * 1.2); }
float fallbackMid()  { return 0.28 + 0.18 * sin(iTime * 0.7 + 1.0); }
float fallbackHigh() { return 0.18 + 0.12 * sin(iTime * 2.8 + 2.0); }
float safeAudio(float v, float f) { return max(v, f * 0.35); }
float audioBoost(float v) { return clamp(pow(v, 0.75) * 1.35, 0.0, 1.0); }
// (mean, peak) of the spectrum this frame, for "stands out from the room" tests
vec2 spectrumStats() {
    float m = 0.0, pk = 0.0;
    for (int i = 0; i < 16; i++) { float v = fft(0.01 + 0.045 * float(i)); m += v; pk = max(pk, v); }
    return vec2(m / 16.0, pk);
}

float hash21(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(i), hash21(i + vec2(1.0, 0.0)), u.x),
               mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), u.x), u.y);
}
float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 5; i++) { v += a * vnoise(p); p = p * 2.03 + vec2(1.7, 9.2); a *= 0.5; }
    return v;
}
mat2 rot(float a) { float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }
float sdSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}
// Display text from the legacy sheet on iChannel3 (2048x512, R sharp, G soft).
// c is the centre on screen, w the width the whole sheet spans.
vec2 wordmark(vec2 p, vec2 c, float w) {
    vec2 q = (p - c) / w;
    q = vec2(q.x + 0.5, q.y * 4.0 + 0.5);
    float inside = step(0.0, q.x) * step(q.x, 1.0) * step(0.0, q.y) * step(q.y, 1.0);
    vec2 t = texture(iChannel3, clamp(q, 0.0, 1.0)).rg * inside;
    // the sheet is magnified on a wall: re-sharpen the glyph edge to one pixel
    float aa = clamp(fwidth(t.x), 0.001, 0.5);
    return vec2(smoothstep(0.5 - aa, 0.5 + aa, t.x), t.y);
}

// HAIRPIN: the crescendo mark < drawn as a fan of strings. The two outer
// strings are the mark itself; the inner ones vibrate with the mids. Every kick
// fires a pulse from the point toward the open end, carried by the kick's decay.
#define FAN_LINES        19.0
#define OPEN_BASE        0.14
#define OPEN_BASS_GAIN   0.06
#define STRING_MID_GAIN  0.006

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    float aspect = iResolution.x / iResolution.y;
    float bass = audioBoost(safeAudio(getBass(), fallbackBass()));
    float mid  = audioBoost(safeAudio(getMid(),  fallbackMid()));
    float high = audioBoost(safeAudio(getHigh(), fallbackHigh()));

    vec3 ink   = vec3(0.022, 0.014, 0.016);
    vec3 amber = vec3(0.95, 0.60, 0.22);
    vec3 soul  = vec3(0.80, 0.17, 0.20);
    vec3 cream = vec3(0.97, 0.89, 0.76);

    vec2 o = vec2(-0.5 * aspect + 0.10, 0.0);
    float len = aspect - 0.10;
    vec2 p = uv - o;
    float r = length(p);
    float rn = clamp(r / len, 0.0, 1.0);
    float th = atan(p.y, p.x);
    float open = OPEN_BASE + OPEN_BASS_GAIN * bass;

    // plucked-string bend, pinned at both ends
    th += sin(r * 26.0 - iTime * 7.0) * STRING_MID_GAIN * mid * sin(PI * rn) / max(r, 0.05);

    float spacing = 2.0 * open / (FAN_LINES - 1.0);
    float f = (th / open * 0.5 + 0.5) * (FAN_LINES - 1.0);
    float idx = floor(f + 0.5);
    float outer = (idx < 0.5 || idx > FAN_LINES - 1.5) ? 1.0 : 0.0;
    float d = abs(f - idx) * spacing * r;
    float w = mix(0.0008, 0.0026, outer) + 0.0006 * rn;
    float px = 1.5 / iResolution.y;
    float str = smoothstep(w + px, w, d);
    str *= step(0.0, p.x) * step(-0.5, idx) * step(idx, FAN_LINES - 0.5) * smoothstep(1.02, 0.97, rn);

    // the pulse sits at the point on the hit and rides outward as the kick decays
    float R = len * (1.05 - bass);
    float pulse = exp(-pow((r - R) * 9.0, 2.0));
    // grains climb the inner strings with the highs
    float h = hash21(vec2(idx, 3.1));
    float grainPos = fract(h + iTime * (0.06 + 0.30 * high)) * len;
    float grain = exp(-pow((r - grainPos) * 55.0, 2.0)) * (1.0 - outer) * high;

    float inner = (0.16 + 0.55 * mid) * (0.6 + 0.4 * rn);
    float bright = mix(inner, 0.85, outer) * (1.0 + 1.5 * pulse) + 2.2 * grain;
    vec3 strCol = mix(mix(soul, amber, rn), cream, outer * 0.55);

    vec3 col = ink;
    col += soul * 0.10 * exp(-length(uv - vec2(0.5 * aspect, 0.0)) * 1.6) * (0.6 + bass);
    vec2 wm = wordmark(uv, vec2(0.5 * aspect - 0.56, 0.0), 0.95);
    col += strCol * str * bright * (1.0 - 0.85 * wm.y);
    col += amber * wm.y * 0.22 * (0.5 + bass);
    col = mix(col, cream * (0.74 + 0.12 * bass), wm.x);
    col *= 1.0 - 0.30 * dot(uv * vec2(0.6, 1.0), uv * vec2(0.6, 1.0));
    fragColor = vec4(min(col, vec3(0.88)), 1.0);
}
`,

  "PIANOSCOPE Four on the Floor": `
/*
PIANOSCOPE Shader
Name: Four on the Floor
Mode: crescendo_floor
Inspired by: CRESCENDO house night concept
Palette: warm black, amber, soul red, cream
Audio: Kick = floods the tiles under the booth, then a ring of light travels outward; Mids = chord stabs light scattered tiles across the floor; Highs = glint along the grout where the ring is passing
Note: Display text from the legacy sheet on iChannel3 (R sharp, G soft), drawn
      through wordmark(). Flat colour and pixel-sharp edges: no noise textures.
*/

#define PI  3.14159265359
#define TAU 6.28318530718

float fft(float x) { return texture(iChannel0, vec2(clamp(x, 0.0, 1.0), 0.0)).x; }
float getBass() { return (fft(0.01) + fft(0.03) + fft(0.05) + fft(0.07)) * 0.25; }
float getMid()  { return (fft(0.15) + fft(0.25) + fft(0.35) + fft(0.45)) * 0.25; }
float getHigh() { return (fft(0.55) + fft(0.70) + fft(0.85) + fft(0.95)) * 0.25; }
float fallbackBass() { return 0.35 + 0.25 * sin(iTime * 1.2); }
float fallbackMid()  { return 0.28 + 0.18 * sin(iTime * 0.7 + 1.0); }
float fallbackHigh() { return 0.18 + 0.12 * sin(iTime * 2.8 + 2.0); }
float safeAudio(float v, float f) { return max(v, f * 0.35); }
float audioBoost(float v) { return clamp(pow(v, 0.75) * 1.35, 0.0, 1.0); }
// (mean, peak) of the spectrum this frame, for "stands out from the room" tests
vec2 spectrumStats() {
    float m = 0.0, pk = 0.0;
    for (int i = 0; i < 16; i++) { float v = fft(0.01 + 0.045 * float(i)); m += v; pk = max(pk, v); }
    return vec2(m / 16.0, pk);
}

float hash21(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(i), hash21(i + vec2(1.0, 0.0)), u.x),
               mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), u.x), u.y);
}
float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 5; i++) { v += a * vnoise(p); p = p * 2.03 + vec2(1.7, 9.2); a *= 0.5; }
    return v;
}
mat2 rot(float a) { float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }
float sdSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}
// Display text from the legacy sheet on iChannel3 (2048x512, R sharp, G soft).
// c is the centre on screen, w the width the whole sheet spans.
vec2 wordmark(vec2 p, vec2 c, float w) {
    vec2 q = (p - c) / w;
    q = vec2(q.x + 0.5, q.y * 4.0 + 0.5);
    float inside = step(0.0, q.x) * step(q.x, 1.0) * step(0.0, q.y) * step(q.y, 1.0);
    vec2 t = texture(iChannel3, clamp(q, 0.0, 1.0)).rg * inside;
    // the sheet is magnified on a wall: re-sharpen the glyph edge to one pixel
    float aa = clamp(fwidth(t.x), 0.001, 0.5);
    return vec2(smoothstep(0.5 - aa, 0.5 + aa, t.x), t.y);
}

// FOUR ON THE FLOOR: a lit glass dance floor. The kick floods the tiles under
// the booth; as it decays the ring of light rolls out across the room. Chord
// stabs light scattered tiles and the hats glint in the grout.
#define HORIZON       0.15
#define CAM_HEIGHT    0.40
#define TILE_SCALE    4.0
#define RIPPLE_REACH  15.0
#define STAB_GAIN     0.30

vec3 floorTile(float h) {
    vec3 amber = vec3(0.95, 0.58, 0.20);
    vec3 soul  = vec3(0.82, 0.16, 0.22);
    vec3 plum  = vec3(0.45, 0.13, 0.42);
    vec3 cream = vec3(0.97, 0.86, 0.70);
    if (h < 0.4) return amber;
    if (h < 0.7) return soul;
    if (h < 0.9) return plum;
    return cream;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    float bass = audioBoost(safeAudio(getBass(), fallbackBass()));
    float mid  = audioBoost(safeAudio(getMid(),  fallbackMid()));
    float high = audioBoost(safeAudio(getHigh(), fallbackHigh()));

    vec3 ink   = vec3(0.020, 0.013, 0.016);
    vec3 amber = vec3(0.95, 0.60, 0.22);
    vec3 soul  = vec3(0.80, 0.17, 0.20);
    vec3 cream = vec3(0.97, 0.89, 0.76);

    vec2 wmPos = vec2(0.0, HORIZON + 0.16);
    float wmW = 1.3;
    float y = uv.y - HORIZON;
    vec3 col;

    if (y < -0.003) {
        float z = CAM_HEIGHT / -y;
        vec2 fp = vec2(uv.x * z, z) * TILE_SCALE;
        vec2 id = floor(fp);
        vec2 f = fract(fp) - 0.5;
        float box = max(abs(f.x), abs(f.y));
        float aa = fwidth(box) * 1.5;
        float grout = smoothstep(0.455 - aa, 0.455 + aa, box);
        float far = smoothstep(0.0, 0.08, -y);

        vec2 booth = vec2(0.0, 2.2 * TILE_SCALE);
        float dist = length(id + 0.5 - booth);
        float R = RIPPLE_REACH * (1.0 - bass);
        float ring = exp(-pow((dist - R) * 0.7, 2.0));
        float core = exp(-dist * 0.45) * bass * bass;
        float seed = floor(iTime * 2.0);
        float blend = smoothstep(0.5, 1.0, fract(iTime * 2.0));   // fade, never pop
        float stabA = step(1.0 - STAB_GAIN * mid, hash21(id + seed * 7.13));
        float stabB = step(1.0 - STAB_GAIN * mid, hash21(id + (seed + 1.0) * 7.13));
        float stab = mix(stabA, stabB, blend) * (0.35 + 0.65 * mid);
        float h = hash21(id * 1.31 + 4.2);
        float lit = max(max(ring * 0.9, core), stab * 0.55);
        float hot = 1.0 - smoothstep(0.1, 0.48, box);
        vec3 tile = vec3(0.030, 0.024, 0.027) + floorTile(h) * lit * (0.5 + 0.5 * hot);

        // the back wall in the glass
        vec2 refl = wordmark(vec2(uv.x, 2.0 * HORIZON - uv.y), wmPos, wmW);
        float fres = exp(y * 7.0);
        tile += cream * refl.x * 0.22 * fres + amber * refl.y * 0.10 * fres * (0.4 + bass);

        vec3 groutCol = vec3(0.012, 0.010, 0.011) + cream * high * 0.5 * ring;
        col = mix(tile, groutCol, grout * far);
        col *= exp(-z * 0.22);
    } else {
        col = ink + soul * 0.07 * exp(-max(y, 0.0) * 5.0) * (0.5 + bass);
        vec2 wm = wordmark(uv, wmPos, wmW);
        col += amber * wm.y * 0.28 * (0.4 + bass);
        col = mix(col, cream * 0.84, wm.x);
    }
    col += amber * 0.10 * exp(-abs(y) * 70.0) * (0.4 + bass);
    fragColor = vec4(min(col, vec3(0.88)), 1.0);
}
`,

  "PIANOSCOPE Piano Stab": `
/*
PIANOSCOPE Shader
Name: Piano Stab
Mode: crescendo_keys
Inspired by: CRESCENDO house night concept
Palette: warm black, amber, soul red, cream
Audio: Kick = plays the bottom octave; Mids = play the middle keys, where the stabs and vocals sit; Highs = play the top octave. hats tick across the right-hand keys
Note: Display text from the legacy sheet on iChannel3 (R sharp, G soft), drawn
      through wordmark(). Flat colour and pixel-sharp edges: no noise textures.
*/

#define PI  3.14159265359
#define TAU 6.28318530718

float fft(float x) { return texture(iChannel0, vec2(clamp(x, 0.0, 1.0), 0.0)).x; }
float getBass() { return (fft(0.01) + fft(0.03) + fft(0.05) + fft(0.07)) * 0.25; }
float getMid()  { return (fft(0.15) + fft(0.25) + fft(0.35) + fft(0.45)) * 0.25; }
float getHigh() { return (fft(0.55) + fft(0.70) + fft(0.85) + fft(0.95)) * 0.25; }
float fallbackBass() { return 0.35 + 0.25 * sin(iTime * 1.2); }
float fallbackMid()  { return 0.28 + 0.18 * sin(iTime * 0.7 + 1.0); }
float fallbackHigh() { return 0.18 + 0.12 * sin(iTime * 2.8 + 2.0); }
float safeAudio(float v, float f) { return max(v, f * 0.35); }
float audioBoost(float v) { return clamp(pow(v, 0.75) * 1.35, 0.0, 1.0); }
// (mean, peak) of the spectrum this frame, for "stands out from the room" tests
vec2 spectrumStats() {
    float m = 0.0, pk = 0.0;
    for (int i = 0; i < 16; i++) { float v = fft(0.01 + 0.045 * float(i)); m += v; pk = max(pk, v); }
    return vec2(m / 16.0, pk);
}

float hash21(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(i), hash21(i + vec2(1.0, 0.0)), u.x),
               mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), u.x), u.y);
}
float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 5; i++) { v += a * vnoise(p); p = p * 2.03 + vec2(1.7, 9.2); a *= 0.5; }
    return v;
}
mat2 rot(float a) { float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }
float sdSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}
// Display text from the legacy sheet on iChannel3 (2048x512, R sharp, G soft).
// c is the centre on screen, w the width the whole sheet spans.
vec2 wordmark(vec2 p, vec2 c, float w) {
    vec2 q = (p - c) / w;
    q = vec2(q.x + 0.5, q.y * 4.0 + 0.5);
    float inside = step(0.0, q.x) * step(q.x, 1.0) * step(0.0, q.y) * step(q.y, 1.0);
    vec2 t = texture(iChannel3, clamp(q, 0.0, 1.0)).rg * inside;
    // the sheet is magnified on a wall: re-sharpen the glyph edge to one pixel
    float aa = clamp(fwidth(t.x), 0.001, 0.5);
    return vec2(smoothstep(0.5 - aa, 0.5 + aa, t.x), t.y);
}

// PIANO STAB: a four-octave keyboard that reads the spectrum left to right,
// sub to air. A key lights when its band stands out from the rest of the room,
// so a hot mic and a quiet one light about the same number of keys. Light
// blooms up off every struck key and lights the wordmark from below.
// Flat colour and pixel-sharp edges throughout: no noise textures.
#define OCTAVES     4.0
#define KEY_TOP    -0.10
#define KEY_BOTTOM -0.44
#define SPEC_LO     0.008
#define SPEC_HI     0.72
#define KEY_SOFT    0.30   // how gradually a key lights once it stands out
#define GLOW_WIDTH  0.55   // bloom width in white keys, at the key line

float keyX(float semi) { return SPEC_LO * pow(SPEC_HI / SPEC_LO, semi / (OCTAVES * 12.0 - 1.0)); }
// stats = (mean, peak) of the spectrum this frame
float keyLevel(float semi, vec2 stats) {
    float u = semi / (OCTAVES * 12.0 - 1.0);
    float rel = (fft(keyX(semi)) - stats.x) / (stats.y - stats.x + 0.05);
    float t = 0.8 - 0.55 * sqrt(u) - 0.45 * u;   // treble keys need less to light
    float live = smoothstep(t, t + KEY_SOFT, rel);
    // mic silent or denied: a slow chord that moves every two seconds instead
    float chord = step(0.82, fract(sin((semi + floor(iTime * 0.5) * 5.0) * 12.9898) * 43758.5453));
    float idle = chord * (0.55 + 0.45 * sin(iTime * 2.4));
    return mix(idle, live, smoothstep(0.02, 0.08, stats.y));
}
float whiteSemi(float w) {
    float d = mod(w, 7.0);
    float s = d < 0.5 ? 0.0 : d < 1.5 ? 2.0 : d < 2.5 ? 4.0 : d < 3.5 ? 5.0 : d < 4.5 ? 7.0 : d < 5.5 ? 9.0 : 11.0;
    return floor(w / 7.0) * 12.0 + s;
}
// a black key sits to the right of C, D, F, G and A
float blackAfter(float w) { float d = mod(w, 7.0); return (d < 1.5 || (d > 2.5 && d < 5.5)) ? 1.0 : 0.0; }
float sdBox(vec2 p, vec2 b) { vec2 d = abs(p) - b; return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0); }

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    float aspect = iResolution.x / iResolution.y;
    float px = 1.0 / iResolution.y;
    float mid = audioBoost(safeAudio(getMid(), fallbackMid()));
    vec2 stats = spectrumStats();

    vec3 ink   = vec3(0.020, 0.013, 0.016);
    vec3 amber = vec3(0.95, 0.60, 0.22);
    vec3 soul  = vec3(0.80, 0.17, 0.20);
    vec3 cream = vec3(0.97, 0.89, 0.76);
    vec3 ivory = vec3(0.17, 0.155, 0.14);

    float whites = OCTAVES * 7.0;
    float keyW = aspect / whites;                 // one white key, in uv units
    float kx = (uv.x / aspect + 0.5) * whites;
    float w = clamp(floor(kx), 0.0, whites - 1.0);
    float b = floor(kx + 0.5);                    // nearest key boundary
    float hasB = blackAfter(b - 1.0) * step(0.5, b) * step(b, whites - 0.5);
    float keyH = KEY_TOP - KEY_BOTTOM;

    vec3 col = ink;
    if (uv.y > KEY_TOP) {
        float hgt = uv.y - KEY_TOP;
        // bloom: every nearby key adds a soft column that widens as it rises
        float wd = GLOW_WIDTH * (1.0 + 4.0 * hgt);
        float glow = 0.0;
        for (int k = -3; k <= 3; k++) {
            float j = w + float(k);
            if (j < 0.0 || j > whites - 1.0) continue;
            float dx = (kx - (j + 0.5)) / wd;
            glow += keyLevel(whiteSemi(j), stats) * exp(-dx * dx);
            if (blackAfter(j) > 0.5 && j < whites - 1.0) {
                float dbx = (kx - (j + 1.0)) / (wd * 0.7);
                glow += 0.8 * keyLevel(whiteSemi(j) + 1.0, stats) * exp(-dbx * dbx);
            }
        }
        glow *= exp(-hgt * 2.8) / (1.0 + 1.5 * hgt);
        col += mix(soul, amber, smoothstep(0.15, 0.9, glow)) * min(glow, 1.4) * 0.65;

        vec2 wm = wordmark(uv, vec2(0.0, 0.18), 1.35);
        float lamp = clamp(glow * 2.2, 0.0, 1.0);
        col += amber * wm.y * 0.08 * (0.4 + mid);
        col = mix(col, cream * (0.32 + 0.55 * lamp), wm.x);

        // red felt rail behind the keys
        float felt = 1.0 - smoothstep(0.012 - px, 0.012 + px, hgt);
        col = mix(col, soul * 0.40, felt);
    } else if (uv.y > KEY_BOTTOM) {
        float ky = (uv.y - KEY_BOTTOM) / keyH;    // 0 at the front edge, 1 at the felt
        float levelW = keyLevel(whiteSemi(w), stats);

        // white key: flat ivory, warm amber when struck, a darker front lip
        vec3 key = mix(ivory * (0.85 + 0.15 * ky), mix(amber, cream, 0.2) * (0.80 + 0.20 * ky), levelW);
        float lip = 1.0 - smoothstep(0.035 - px / keyH, 0.035 + px / keyH, ky);
        key *= 1.0 - 0.35 * lip;
        // seams between white keys: one crisp line
        float seamD = min(kx - w, w + 1.0 - kx) * keyW;
        key *= smoothstep(0.6 * px, 1.6 * px, seamD);

        // black key at the nearest boundary
        float bTop = KEY_TOP + 0.02;
        float bBot = KEY_BOTTOM + keyH * 0.38;
        vec2 bc = vec2(((b / whites) - 0.5) * aspect, 0.5 * (bTop + bBot));
        vec2 bh = vec2(0.29 * keyW, 0.5 * (bTop - bBot));
        float dB = sdBox(uv - bc, bh);
        // a short soft shadow falling toward the player
        float dS = sdBox(uv - bc + vec2(0.0, 0.006), bh);
        key *= 1.0 - 0.30 * hasB * (1.0 - smoothstep(0.0, 0.010, dS));

        if (hasB > 0.5) {
            float levelB = keyLevel(whiteSemi(b - 1.0) + 1.0, stats);
            // body plus a slightly lighter top face, inset from the sides and front
            float dTop = sdBox(uv - bc - vec2(0.0, 0.006), bh - vec2(0.045 * keyW, 0.012));
            vec3 body = mix(vec3(0.018, 0.015, 0.017), soul * 0.45, levelB);
            vec3 face = mix(vec3(0.050, 0.044, 0.048), soul * 0.95, levelB);
            vec3 black = mix(body, face, 1.0 - smoothstep(-px, px, dTop));
            key = mix(key, black, 1.0 - smoothstep(-px, px, dB));
        }
        col = key;
    } else {
        col = ink * 0.8;                          // the key slip under the keys
    }
    fragColor = vec4(min(col, vec3(0.88)), 1.0);
}
`,

  "PIANOSCOPE 33 RPM": `
/*
PIANOSCOPE Shader
Name: 33 RPM
Mode: crescendo_vinyl
Inspired by: CRESCENDO house night concept
Palette: warm black, amber, soul red, cream
Audio: Kick = lights the outer rings and sends a ring of light from the label out to the rim; Mids = light the middle of the record; Highs = make the needle glow
Note: Display text from the legacy sheet on iChannel3 (R sharp, G soft), drawn
      through wordmark(). Flat colour and pixel-sharp edges: no noise textures.
*/

#define PI  3.14159265359
#define TAU 6.28318530718

float fft(float x) { return texture(iChannel0, vec2(clamp(x, 0.0, 1.0), 0.0)).x; }
float getBass() { return (fft(0.01) + fft(0.03) + fft(0.05) + fft(0.07)) * 0.25; }
float getMid()  { return (fft(0.15) + fft(0.25) + fft(0.35) + fft(0.45)) * 0.25; }
float getHigh() { return (fft(0.55) + fft(0.70) + fft(0.85) + fft(0.95)) * 0.25; }
float fallbackBass() { return 0.35 + 0.25 * sin(iTime * 1.2); }
float fallbackMid()  { return 0.28 + 0.18 * sin(iTime * 0.7 + 1.0); }
float fallbackHigh() { return 0.18 + 0.12 * sin(iTime * 2.8 + 2.0); }
float safeAudio(float v, float f) { return max(v, f * 0.35); }
float audioBoost(float v) { return clamp(pow(v, 0.75) * 1.35, 0.0, 1.0); }
// (mean, peak) of the spectrum this frame, for "stands out from the room" tests
vec2 spectrumStats() {
    float m = 0.0, pk = 0.0;
    for (int i = 0; i < 16; i++) { float v = fft(0.01 + 0.045 * float(i)); m += v; pk = max(pk, v); }
    return vec2(m / 16.0, pk);
}

float hash21(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(i), hash21(i + vec2(1.0, 0.0)), u.x),
               mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), u.x), u.y);
}
float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 5; i++) { v += a * vnoise(p); p = p * 2.03 + vec2(1.7, 9.2); a *= 0.5; }
    return v;
}
mat2 rot(float a) { float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }
float sdSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}
// Display text from the legacy sheet on iChannel3 (2048x512, R sharp, G soft).
// c is the centre on screen, w the width the whole sheet spans.
vec2 wordmark(vec2 p, vec2 c, float w) {
    vec2 q = (p - c) / w;
    q = vec2(q.x + 0.5, q.y * 4.0 + 0.5);
    float inside = step(0.0, q.x) * step(q.x, 1.0) * step(0.0, q.y) * step(q.y, 1.0);
    vec2 t = texture(iChannel3, clamp(q, 0.0, 1.0)).rg * inside;
    // the sheet is magnified on a wall: re-sharpen the glyph edge to one pixel
    float aa = clamp(fwidth(t.x), 0.001, 0.5);
    return vec2(smoothstep(0.5 - aa, 0.5 + aa, t.x), t.y);
}

// 33 1/3: the record from above at real speed. A fixed light makes two soft
// sheen wedges on the vinyl; each radius shines with its part of the spectrum,
// kick at the rim and air at the run-out. Strobe dots on the platter edge stand
// still, as they do on a Technics deck at the right speed. Flat colour and
// pixel-sharp edges throughout: no noise textures, no fine groove lines.
#define RPM          33.333
#define REC_R        0.43
#define LABEL_R      0.14
#define SHEEN_ANGLE  0.85
#define SHEEN_POWER  9.0

// how much the spectrum around x stands out from the room, 0..1
float bandLevel(float x, vec2 stats) {
    float v = (fft(x - 0.02) + fft(x - 0.01) + fft(x) + fft(x + 0.01) + fft(x + 0.02)) * 0.2;
    float rel = (v - stats.x) / (stats.y - stats.x + 0.05);
    float u = clamp(x / 0.7, 0.0, 1.0);
    float t = 0.55 - 0.45 * sqrt(u) - 0.35 * u;
    float idle = 0.25 + 0.2 * sin(iTime * 1.3 - x * 9.0);   // mic silent or denied
    return mix(idle, smoothstep(t, t + 0.45, rel), smoothstep(0.02, 0.08, stats.y));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    float px = 1.0 / iResolution.y;
    float bass = audioBoost(safeAudio(getBass(), fallbackBass()));
    float high = audioBoost(safeAudio(getHigh(), fallbackHigh()));
    vec2 stats = spectrumStats();

    vec3 ink   = vec3(0.020, 0.013, 0.016);
    vec3 amber = vec3(0.95, 0.60, 0.22);
    vec3 soul  = vec3(0.80, 0.17, 0.20);
    vec3 cream = vec3(0.97, 0.89, 0.76);
    vec3 lampC = vec3(1.00, 0.36, 0.12);

    vec2 c = vec2(-0.16, 0.0);
    vec2 p = uv - c;
    float r = length(p);
    float th = atan(p.y, p.x);
    vec2 q = rot(iTime * TAU * RPM / 60.0) * p;   // record space

    vec3 col = ink * (1.0 - 0.3 * length(uv));

    // platter
    col = mix(col, vec3(0.048, 0.044, 0.047), 1.0 - smoothstep(-px, px, r - (REC_R + 0.052)));
    // strobe dots: drawn in screen space because at speed they stand still
    float N = 144.0;
    float cell = (fract(th / TAU * N) - 0.5) * TAU / N * r;
    float dotD = length(vec2(cell, r - (REC_R + 0.032))) - 0.0028;
    float dots = 1.0 - smoothstep(-px, px, dotD);
    float lampSide = pow(max(dot(normalize(p), normalize(vec2(-0.6, -0.8))), 0.0), 6.0);
    col += lampC * dots * (0.05 + 0.75 * lampSide);

    // the record
    float band = clamp((REC_R - r) / (REC_R - LABEL_R - 0.035), 0.0, 1.0);
    float level = bandLevel(mix(0.02, 0.70, band), stats);
    float sheen = pow(abs(cos(th - SHEEN_ANGLE)), SHEEN_POWER);
    float runout = 1.0 - smoothstep(-px, px, r - (LABEL_R + 0.035));
    vec3 rec = vec3(0.022, 0.020, 0.022);
    rec += mix(cream * 0.55, amber, level) * sheen * (0.07 + 0.80 * level) * (1.0 - 0.7 * runout);
    // track gaps: thin darker rings where the sheen breaks
    float gap = 0.0;
    for (int i = 1; i <= 3; i++) {
        float rr = REC_R - (float(i) * 0.25) * (REC_R - LABEL_R - 0.035);
        gap = max(gap, 1.0 - smoothstep(0.0012 - px, 0.0012 + px, abs(r - rr)));
    }
    rec *= 1.0 - 0.65 * gap;
    // the lead-in lip at the edge
    rec += cream * 0.05 * (1.0 - smoothstep(0.0015 - px, 0.0015 + px, abs(r - (REC_R - 0.004))));
    // the kick: a ring that leaves the label and rides out to the rim as it decays
    float pulseR = LABEL_R + 0.035 + (REC_R - LABEL_R - 0.035) * (1.0 - bass);
    rec += amber * exp(-pow((r - pulseR) * 55.0, 2.0)) * bass * (0.10 + 0.45 * sheen);
    col = mix(col, rec, 1.0 - smoothstep(-px, px, r - REC_R));

    // label: flat paper, crisp edge, the wordmark and a printed arc
    vec3 paper = soul * 0.66 * (1.0 - 0.15 * smoothstep(LABEL_R - 0.03, LABEL_R, r));
    vec2 wm = wordmark(q, vec2(0.0, 0.035), 0.24);
    paper = mix(paper, cream * 0.88, wm.x);
    float arc = (1.0 - smoothstep(0.0011 - px, 0.0011 + px, abs(length(q) - 0.075))) * step(q.y, -0.01);
    paper = mix(paper, cream * 0.75, arc);
    col = mix(col, paper, 1.0 - smoothstep(-px, px, r - LABEL_R));
    col = mix(col, vec3(0.60, 0.58, 0.55), 1.0 - smoothstep(-px, px, r - 0.010));

    // tonearm
    vec2 pivot = c + vec2(0.62, 0.30);
    vec2 elbow = c + vec2(0.50, -0.12);
    vec2 tip   = c + vec2(0.30, -0.22);
    vec2 so = uv - vec2(0.010, -0.012);
    float shD = min(sdSegment(so, pivot, elbow), sdSegment(so, elbow, tip)) - 0.0055;
    col *= 1.0 - 0.45 * (1.0 - smoothstep(0.0, 0.014, shD));
    float armD = min(sdSegment(uv, pivot, elbow), sdSegment(uv, elbow, tip));
    vec3 metal = mix(vec3(0.33, 0.31, 0.30), vec3(0.62, 0.60, 0.57), 1.0 - smoothstep(0.0016 - px, 0.0016 + px, armD));
    col = mix(col, metal, 1.0 - smoothstep(0.0055 - px, 0.0055 + px, armD));
    float head = sdSegment(uv, tip, tip + normalize(elbow - tip) * 0.05) - 0.011;
    col = mix(col, vec3(0.11, 0.10, 0.105), 1.0 - smoothstep(-px, px, head));
    col = mix(col, vec3(0.085, 0.08, 0.085), 1.0 - smoothstep(-px, px, length(uv - pivot) - 0.045));
    col = mix(col, vec3(0.34, 0.32, 0.31), 1.0 - smoothstep(-px, px, length(uv - pivot) - 0.018));
    col += amber * exp(-length(uv - tip) * 70.0) * (0.12 + 0.8 * high);

    fragColor = vec4(min(col, vec3(0.88)), 1.0);
}
`,

  "PIANOSCOPE The Warehouse": `
/*
PIANOSCOPE Shader
Name: The Warehouse
Mode: crescendo_warehouse
Inspired by: CRESCENDO house night concept
Palette: warm black, amber, soul red, cream
Audio: Kick = lifts the beams (capped, never a strobe) and the crowd bobs; Mids = steer the beams. busier mids, wider sweeps; Highs = brighten the beams a touch
Note: Display text from the legacy sheet on iChannel3 (R sharp, G soft), drawn
      through wordmark(). Flat colour and pixel-sharp edges: no noise textures.
*/

#define PI  3.14159265359
#define TAU 6.28318530718

float fft(float x) { return texture(iChannel0, vec2(clamp(x, 0.0, 1.0), 0.0)).x; }
float getBass() { return (fft(0.01) + fft(0.03) + fft(0.05) + fft(0.07)) * 0.25; }
float getMid()  { return (fft(0.15) + fft(0.25) + fft(0.35) + fft(0.45)) * 0.25; }
float getHigh() { return (fft(0.55) + fft(0.70) + fft(0.85) + fft(0.95)) * 0.25; }
float fallbackBass() { return 0.35 + 0.25 * sin(iTime * 1.2); }
float fallbackMid()  { return 0.28 + 0.18 * sin(iTime * 0.7 + 1.0); }
float fallbackHigh() { return 0.18 + 0.12 * sin(iTime * 2.8 + 2.0); }
float safeAudio(float v, float f) { return max(v, f * 0.35); }
float audioBoost(float v) { return clamp(pow(v, 0.75) * 1.35, 0.0, 1.0); }
// (mean, peak) of the spectrum this frame, for "stands out from the room" tests
vec2 spectrumStats() {
    float m = 0.0, pk = 0.0;
    for (int i = 0; i < 16; i++) { float v = fft(0.01 + 0.045 * float(i)); m += v; pk = max(pk, v); }
    return vec2(m / 16.0, pk);
}

float hash21(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(i), hash21(i + vec2(1.0, 0.0)), u.x),
               mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), u.x), u.y);
}
float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 5; i++) { v += a * vnoise(p); p = p * 2.03 + vec2(1.7, 9.2); a *= 0.5; }
    return v;
}
mat2 rot(float a) { float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }
float sdSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}
// Display text from the legacy sheet on iChannel3 (2048x512, R sharp, G soft).
// c is the centre on screen, w the width the whole sheet spans.
vec2 wordmark(vec2 p, vec2 c, float w) {
    vec2 q = (p - c) / w;
    q = vec2(q.x + 0.5, q.y * 4.0 + 0.5);
    float inside = step(0.0, q.x) * step(q.x, 1.0) * step(0.0, q.y) * step(q.y, 1.0);
    vec2 t = texture(iChannel3, clamp(q, 0.0, 1.0)).rg * inside;
    // the sheet is magnified on a wall: re-sharpen the glyph edge to one pixel
    float aa = clamp(fwidth(t.x), 0.001, 0.5);
    return vec2(smoothstep(0.5 - aa, 0.5 + aa, t.x), t.y);
}

// THE WAREHOUSE: five fixtures on a truss cutting through haze over a crowd.
// The mids steer the beams, the kick lifts them (capped, never a strobe) and
// bobs the crowd, the hats catch dust inside the beams. The wordmark on the
// back wall only shows where light falls on it.
#define BEAMS           5
#define TRUSS_Y         0.43
#define BEAM_SPREAD     0.085
#define SWEEP_MID_GAIN  0.55
#define KICK_LIFT       0.65

float crowdRow(vec2 uv, float baseY, float s, float seed, float bass, float px) {
    float cw = 0.105 * s;
    float cx = floor(uv.x / cw);
    float d = 1e5;
    for (int k = -1; k <= 1; k++) {
        float id = cx + float(k);
        float h = hash21(vec2(id, seed));
        float x0 = (id + 0.5) * cw + (h - 0.5) * cw * 0.45;
        float bob = bass * 0.016 * s * (0.5 + hash21(vec2(seed, id)));
        float y0 = baseY + bob;
        d = min(d, length((uv - vec2(x0, y0 + (0.072 + 0.015 * h) * s)) * vec2(1.0, 0.88)) - 0.027 * s);
        vec2 bq = abs(uv - vec2(x0, y0 - 0.09 * s)) - vec2(0.040, 0.12) * s + 0.025 * s;
        d = min(d, length(max(bq, 0.0)) + min(max(bq.x, bq.y), 0.0) - 0.025 * s);
        if (h > 0.80) {
            float side = h > 0.90 ? 1.0 : -1.0;
            float sway = sin(iTime * 2.1 + id * 1.7) * 0.025 * s;
            vec2 sh = vec2(x0 + 0.032 * side * s, y0 + 0.02 * s);
            vec2 hand = sh + vec2(0.03 * side * s + sway, 0.15 * s + bob);
            d = min(d, sdSegment(uv, sh, hand) - 0.009 * s);
        }
    }
    return 1.0 - smoothstep(-px, px, d);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    float aspect = iResolution.x / iResolution.y;
    float bass = audioBoost(safeAudio(getBass(), fallbackBass()));
    float mid  = audioBoost(safeAudio(getMid(),  fallbackMid()));
    float high = audioBoost(safeAudio(getHigh(), fallbackHigh()));

    vec3 ink   = vec3(0.016, 0.011, 0.014);
    vec3 amber = vec3(0.95, 0.60, 0.22);
    vec3 soul  = vec3(0.80, 0.17, 0.20);
    vec3 cream = vec3(0.97, 0.89, 0.76);

    float px = 1.0 / iResolution.y;
    // haze as a slow, smooth swell instead of a noise texture
    float haze = 0.62 + 0.18 * sin(uv.x * 1.6 + iTime * 0.11) * sin(uv.y * 2.1 - iTime * 0.08);

    vec3 beams = vec3(0.0);
    vec3 heads = vec3(0.0);
    float beamSum = 0.0;
    for (int i = 0; i < BEAMS; i++) {
        float fi = float(i);
        vec2 src = vec2((fi / float(BEAMS - 1) - 0.5) * aspect * 0.8, TRUSS_Y);
        float sweep = sin(iTime * (0.33 + 0.06 * fi) + fi * 1.9) * (0.22 + SWEEP_MID_GAIN * mid);
        vec2 dir = vec2(sin(sweep), -cos(sweep));
        vec2 v = uv - src;
        float along = dot(v, dir);
        float perp = abs(v.x * dir.y - v.y * dir.x);
        float wdt = BEAM_SPREAD * max(along, 0.0) + 0.004;
        float beam = (1.0 - smoothstep(wdt * 0.2, wdt, perp)) * smoothstep(0.0, 0.02, along) * exp(-along * 0.8);
        vec3 bc = i == 2 ? cream : (mod(fi, 2.0) < 0.5 ? amber : soul);
        beams += bc * beam;
        beamSum += beam;
        heads += bc * exp(-length(v) * 55.0);
    }
    float lift = (1.0 - KICK_LIFT) + KICK_LIFT * bass;

    vec3 col = ink;
    vec2 wm = wordmark(uv, vec2(0.0, 0.08), 1.45);
    col = mix(col, cream * (0.12 + 0.70 * min(beamSum * lift, 1.0)), wm.x);
    col += beams * haze * lift * (0.75 + 0.25 * high);

    float truss = 1.0 - smoothstep(0.007 - px, 0.007 + px, abs(uv.y - TRUSS_Y - 0.022));
    col = mix(col, vec3(0.07, 0.065, 0.07), truss);
    col += heads * (0.5 + 0.5 * lift);

    float back = crowdRow(uv, -0.36, 0.78, 3.0, bass, px);
    col = mix(col, ink * 1.4 + beams * 0.02, back);
    float front = crowdRow(uv + vec2(0.05, 0.0), -0.43, 1.15, 11.0, bass, px);
    col = mix(col, vec3(0.004, 0.003, 0.004), front);

    col *= 1.0 - 0.25 * dot(uv * vec2(0.5, 1.0), uv * vec2(0.5, 1.0));
    fragColor = vec4(min(col, vec3(0.88)), 1.0);
}
`,

  "PIANOSCOPE Shepard Rise": `
/*
PIANOSCOPE Shader
Name: Shepard Rise
Mode: crescendo_shepard
Inspired by: CRESCENDO house night concept
Palette: warm black, amber, soul red, cream
Audio: Kick = shoves the whole stack forward a step; Mids = tilt the stack and warm the colours; Highs = light the vanishing point
Note: Display text from the legacy sheet on iChannel3 (R sharp, G soft), drawn
      through wordmark(). Flat colour and pixel-sharp edges: no noise textures.
*/

#define PI  3.14159265359
#define TAU 6.28318530718

float fft(float x) { return texture(iChannel0, vec2(clamp(x, 0.0, 1.0), 0.0)).x; }
float getBass() { return (fft(0.01) + fft(0.03) + fft(0.05) + fft(0.07)) * 0.25; }
float getMid()  { return (fft(0.15) + fft(0.25) + fft(0.35) + fft(0.45)) * 0.25; }
float getHigh() { return (fft(0.55) + fft(0.70) + fft(0.85) + fft(0.95)) * 0.25; }
float fallbackBass() { return 0.35 + 0.25 * sin(iTime * 1.2); }
float fallbackMid()  { return 0.28 + 0.18 * sin(iTime * 0.7 + 1.0); }
float fallbackHigh() { return 0.18 + 0.12 * sin(iTime * 2.8 + 2.0); }
float safeAudio(float v, float f) { return max(v, f * 0.35); }
float audioBoost(float v) { return clamp(pow(v, 0.75) * 1.35, 0.0, 1.0); }
// (mean, peak) of the spectrum this frame, for "stands out from the room" tests
vec2 spectrumStats() {
    float m = 0.0, pk = 0.0;
    for (int i = 0; i < 16; i++) { float v = fft(0.01 + 0.045 * float(i)); m += v; pk = max(pk, v); }
    return vec2(m / 16.0, pk);
}

float hash21(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(i), hash21(i + vec2(1.0, 0.0)), u.x),
               mix(hash21(i + vec2(0.0, 1.0)), hash21(i + vec2(1.0, 1.0)), u.x), u.y);
}
float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 5; i++) { v += a * vnoise(p); p = p * 2.03 + vec2(1.7, 9.2); a *= 0.5; }
    return v;
}
mat2 rot(float a) { float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }
float sdSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}
// Display text from the legacy sheet on iChannel3 (2048x512, R sharp, G soft).
// c is the centre on screen, w the width the whole sheet spans.
vec2 wordmark(vec2 p, vec2 c, float w) {
    vec2 q = (p - c) / w;
    q = vec2(q.x + 0.5, q.y * 4.0 + 0.5);
    float inside = step(0.0, q.x) * step(q.x, 1.0) * step(0.0, q.y) * step(q.y, 1.0);
    vec2 t = texture(iChannel3, clamp(q, 0.0, 1.0)).rg * inside;
    // the sheet is magnified on a wall: re-sharpen the glyph edge to one pixel
    float aa = clamp(fwidth(t.x), 0.001, 0.5);
    return vec2(smoothstep(0.5 - aa, 0.5 + aa, t.x), t.y);
}

// SHEPARD RISE: a Shepard tone seems to climb forever. So does the wordmark:
// copies grow out of the vanishing point and fade as they pass the viewer, so
// the build never tops out. The kick shoves the stack forward.
#define COPIES      7
#define RISE_SPEED  0.16
#define KICK_PUSH   0.22
#define BASE_W      0.07

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    float bass = audioBoost(safeAudio(getBass(), fallbackBass()));
    float mid  = audioBoost(safeAudio(getMid(),  fallbackMid()));
    float high = audioBoost(safeAudio(getHigh(), fallbackHigh()));

    vec3 ink   = vec3(0.020, 0.013, 0.016);
    vec3 amber = vec3(0.95, 0.60, 0.22);
    vec3 soul  = vec3(0.80, 0.17, 0.20);
    vec3 cream = vec3(0.97, 0.89, 0.76);

    uv = rot(sin(iTime * 0.2) * 0.06 * mid) * uv;
    float phase = iTime * RISE_SPEED + bass * KICK_PUSH;
    float fp = fract(phase);
    float cycle = floor(phase);

    vec3 col = ink + soul * 0.14 * exp(-length(uv) * 3.0) * (0.5 + bass);
    for (int i = 0; i < COPIES; i++) {
        float k = float(i) + fp;
        float w = BASE_W * exp2(k);
        float fade = smoothstep(0.0, 1.2, k) * (1.0 - smoothstep(float(COPIES) - 3.2, float(COPIES) - 1.2, k));
        float idn = mod(float(i) - cycle, 3.0);
        vec3 c = idn < 0.5 ? amber : (idn < 1.5 ? soul : cream);
        c = mix(c, amber, 0.25 * mid);
        vec2 wm = wordmark(uv, vec2(0.0), w);
        float depth = clamp(k / float(COPIES), 0.0, 1.0);
        col += c * wm.y * fade * 0.07 * (0.4 + bass);
        col = mix(col, c * mix(0.95, 0.22, depth), wm.x * fade);
    }
    col += amber * 0.10 * exp(-length(uv) * 8.0) * high;   // the hats light the vanishing point
    col *= 1.0 - 0.3 * dot(uv * vec2(0.6, 1.0), uv * vec2(0.6, 1.0));
    fragColor = vec4(clamp(col, 0.0, 0.88), 1.0);
}
`,

  "PIANOSCOPE Kaliset Hex Parallax": `
/*
PIANOSCOPE Shader
Name: Kaliset Hex Parallax
Mode: kaliset_hex_parallax
Inspired by: Kaliset fractal + hex grid volumetric parallax
Audio: Uses iChannel0 texture lookup for hex cell tinting
*/

#define S3 1.73205080757
#define HEXSIZE 10
#define JULIA 0.584
#define SCROLL_SPEED_FACTOR 0.0024
#define P vec2
#define V vec3
#define LSZ 12
#define CSZ 10
#define WSZ 7

float getKalisetFractal(vec2 uv) {
    vec2 p = fract(uv) - 0.5;

    float previousDistance = 0.0;
    float totalChange = 0.0;

    for (int j = 0; j < 14; j++) {
        float dProd = dot(p, p);
        if (dProd < 0.00001) dProd = 0.0000001;

        p = abs(p) / dProd;
        p.x -= JULIA;
        p.y -= JULIA;

        float distance = length(p);
        totalChange += abs(distance - previousDistance);
        previousDistance = distance;
    }

    if (totalChange >= 1000.0 || totalChange != totalChange || abs(totalChange) > 1e10) {
        totalChange = 1000.0;
    }

    return totalChange;
}

mat2 rot(float a) {
    return mat2(cos(a), sin(a), -sin(a), cos(a));
}

float random(float noise) {
    return fract(sin(noise * 12.9898) * 43758.5453123);
}

float drawLine(P p, P mc) {
    float f = p.y - p.x * mc.x - mc.y;
    float d = f / sqrt(1.0 + mc.x * mc.x);
    float pw = fwidth(d);
    return 1.0 - smoothstep(-1.0, 1.0, 0.25 - abs(d) / pw);
}

float drawLine(P p, P a, P b) {
    float m = (b.y - a.y) / (b.x - a.x);
    float c = a.y - m * a.x;
    P minp = min(a, b);
    P maxp = max(a, b);
    P pin = clamp(p, minp, maxp);
    float d = distance(pin, p);
    return mix(1.0, drawLine(p, P(m, c)), step(d, 0.0));
}

float drawCirc(P p, V c) {
    float f = distance(p, c.xy);
    return 1.0 - smoothstep(-1.0, 1.0, 0.25 - abs(c.z - f) / fwidth(f));
}

float drawWave(P p, V w) {
    p = rot(w.z) * p;
    float fy = w.x * sin(w.y * p.x);
    P fp = P(p.x, fy);
    float d = distance(fp, p);
    return 1.0 - smoothstep(-1.0, 1.0, 0.5 - d / max(1.0 / iResolution.y, fwidth(fy)));
}

float drawBox(P p, vec4 mimx) {
    P pin = clamp(p, mimx.xy, mimx.zw);
    float d = distance(p, pin);
    return 1.0 - smoothstep(0.0, 1.0, 1.0 - abs(1.0 - d * iResolution.y));
}

vec3 parityColor(P p, V circs[CSZ], V waves[WSZ], vec4 box) {
    vec3 colorAcc = vec3(0.0);
    float totalFlicker = 0.0;

    for (int i = 0; i < CSZ; ++i) {
        float f = sqrt(dot(p - circs[i].xy, p - circs[i].xy)) - circs[i].z;
        if (f < 0.0) {
            float seed = float(i) + floor(iTime * 8.0);
            vec3 circleColor = vec3(
                random(seed * 1.15),
                random(seed * 2.43),
                random(seed * 3.71)
            );
            colorAcc += circleColor;
            totalFlicker += 1.0;
        }
    }

    float v = 0.0;
    for (int i = 0; i < WSZ; ++i) {
        V w = waves[i];
        P rp = rot(w.z) * p;
        float f = w.x * sin(w.y * rp.x) - rp.y;
        v += step(0.0, f);
    }
    P pin = clamp(p, box.xy, box.zw);
    v += step(0.0, distance(p, pin) - 1.0 / iResolution.y);

    if (totalFlicker > 0.0) {
        return mod(colorAcc, 1.0);
    }
    return vec3(mod(v, 2.0));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 coords = fragCoord / iResolution.xy;
    coords.x *= iResolution.x / iResolution.y;
    float zoom = iResolution.x / float(HEXSIZE);
    float ar = iResolution.x / iResolution.y;
    vec2 uv5 = ((fragCoord - 0.5 * iResolution.xy) / iResolution.x);
    uv5 *= zoom;
    uv5.x += 0.5 * zoom;
    uv5.y += 0.5 * zoom / S3;
    float minRes = min(iResolution.x, iResolution.y);
    vec2 tv = fragCoord / iResolution.xy;
    tv.x *= ar;

    vec2 r = vec2(1.0, S3);
    vec2 h = r * 0.5;
    vec2 a = mod(uv5, r) - h;
    vec2 b = mod(uv5 - h, r) - h;
    vec2 c = dot(a, a) < dot(b, b) ? a : b;
    vec2 id = (uv5 - c) / h;
    vec2 tvid = id / zoom + iTime / 150.0;
    tvid.y *= S3;

    P uv = (fragCoord - 0.5 * iResolution.xy) / min(iResolution.x, iResolution.y);

    V circs[CSZ] = V[CSZ](
        V(0.04 + 0.1 * sin(iTime * 1.2), 0.07 + 0.1 * cos(iTime * 1.5), 0.05),
        V(0.38 + 0.15 * cos(iTime * 0.8), 0.2 + 0.1 * sin(iTime * 1.1), 0.12),
        V(0.8 + 0.05 * sin(iTime * 2.0), 0.45 + 0.05 * cos(iTime * 1.8), 0.25),
        V(0.7 + 0.12 * cos(iTime * 1.4), -0.03 + 0.15 * sin(iTime * 0.9), 0.17),
        V(-0.02 + 0.2 * sin(iTime * 0.7), 0.37 + 0.1 * cos(iTime * 1.3), 0.11),
        V(-0.55 + 0.08 * cos(iTime * 1.6), 0.25 + 0.12 * sin(iTime * 1.0), 0.3),
        V(-0.78 + 0.05 * sin(iTime * 2.2), -0.2 + 0.05 * cos(iTime * 2.5), 0.1),
        V(-0.02 + 0.15 * cos(iTime * 1.1), -0.38 + 0.15 * sin(iTime * 0.7), 0.3),
        V(0.47 + 0.03 * sin(iTime * 3.0), -0.43 + 0.03 * cos(iTime * 2.8), 0.05),
        V(0.85 + 0.1 * cos(iTime * 0.5), -0.6 + 0.1 * sin(iTime * 0.6), 0.25)
    );

    V waves[WSZ] = V[WSZ](
        V(1.04, 14.0, 0.7),
        V(1.05, 9.5, 0.4),
        V(1.05, 9.5, 0.05),
        V(0.03, 9.5, -0.2),
        V(0.03, 9.5, -0.55),
        V(0.05, 12.0, -0.9),
        V(0.05, 12.0, -1.57)
    );

    vec4 box = vec4(-1.8, -0.45, 0.8, 0.45);

    vec3 bColor = parityColor(uv, circs, waves, box);

    for (int i = 0; i < CSZ; ++i) {
        bColor = mix(vec3(0.0), bColor, drawCirc(uv, circs[i]));
    }
    for (int i = 0; i < WSZ; ++i) {
        bColor = mix(vec3(0.0), bColor, drawWave(uv, waves[i]));
    }
    bColor = mix(vec3(0.0), bColor, drawBox(uv, box));

    vec3 frontStarColor = vec3(0.0, 0.44, 0.38);
    vec3 backStarColor = vec3(0.5, 0.0, 0.5);

    vec4 result = vec4(0.0);
    float volumetricLayerFade = 1.0;

    for (int i = 0; i < 12; i++) {
        float time = iTime / volumetricLayerFade;
        vec2 p = coords * zoom + tvid * bColor.xy;
        p.y += 1.5;

        p += vec2(time * SCROLL_SPEED_FACTOR, time * SCROLL_SPEED_FACTOR);
        p /= volumetricLayerFade;

        float totalChange = getKalisetFractal(p);
        float totalChangeSample = totalChange * 0.05;

        vec4 layerColor = vec4(mix(frontStarColor, backStarColor, float(i) / 12.0), 1.0);

        result += layerColor * totalChangeSample * volumetricLayerFade;
        volumetricLayerFade *= 0.9;
    }

    result.rgb = pow(result.rgb * 0.12, vec3(1.6)) * bColor.xyz * 5.0;
    fragColor = vec4(result.rgb, 1.0);
}
`,


  "Neon Heart": `// gelami crease fix: https://www.shadertoy.com/view/7l3GDS

#define POINT_COUNT 8

vec2 points[POINT_COUNT];
const float speed = -0.5;
const float len = 0.25;
const float scale = 0.012;
float intensity = 1.3;
float radius = 0.015;

float sdBezier(vec2 pos, vec2 A, vec2 B, vec2 C) {
    vec2 a = B - A;
    vec2 b = A - 2.0 * B + C;
    vec2 c = a * 2.0;
    vec2 d = A - pos;

    float kk = 1.0 / dot(b, b);
    float kx = kk * dot(a, b);
    float ky = kk * (2.0 * dot(a, a) + dot(d, b)) / 3.0;
    float kz = kk * dot(d, a);

    float res = 0.0;

    float p = ky - kx * kx;
    float p3 = p * p * p;
    float q = kx * (2.0 * kx * kx - 3.0 * ky) + kz;
    float h = q * q + 4.0 * p3;

    if (h >= 0.0) {
        h = sqrt(h);
        vec2 x = (vec2(h, -h) - q) / 2.0;
        vec2 uv = sign(x) * pow(abs(x), vec2(1.0 / 3.0));
        float t = uv.x + uv.y - kx;
        t = clamp(t, 0.0, 1.0);

        vec2 qos = d + (c + b * t) * t;
        res = length(qos);
    } else {
        float z = sqrt(-p);
        float v = acos(q / (p * z * 2.0)) / 3.0;
        float m = cos(v);
        float n = sin(v) * 1.732050808;
        vec3 t = vec3(m + m, -n - m, n - m) * z - kx;
        t = clamp(t, 0.0, 1.0);

        vec2 qos = d + (c + b * t.x) * t.x;
        float dis = dot(qos, qos);

        res = dis;

        qos = d + (c + b * t.y) * t.y;
        dis = dot(qos, qos);
        res = min(res, dis);

        qos = d + (c + b * t.z) * t.z;
        dis = dot(qos, qos);
        res = min(res, dis);

        res = sqrt(res);
    }

    return res;
}

vec2 getHeartPosition(float t) {
    return vec2(16.0 * sin(t) * sin(t) * sin(t),
                -(13.0 * cos(t) - 5.0 * cos(2.0 * t)
                - 2.0 * cos(3.0 * t) - cos(4.0 * t)));
}

float getGlow(float dist, float rad, float powInt) {
    return pow(rad / dist, powInt);
}

float getSegment(float t, vec2 pos, float offset) {
    for (int i = 0; i < POINT_COUNT; i++) {
        points[i] = getHeartPosition(offset + float(i) * len + fract(speed * t) * 6.28);
    }

    vec2 c = (points[0] + points[1]) / 2.0;
    vec2 c_prev;
    float dist = 10000.0;

    for (int i = 0; i < POINT_COUNT - 1; i++) {
        c_prev = c;
        c = (points[i] + points[i + 1]) / 2.0;
        dist = min(dist, sdBezier(pos, scale * c_prev, scale * points[i], scale * c));
    }
    return max(0.0, dist);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    float widthHeightRatio = iResolution.x / iResolution.y;
    vec2 centre = vec2(0.5, 0.5);
    vec2 pos = centre - uv;
    pos.y /= widthHeightRatio;

    pos.y += 0.03;

    float t = iTime;

    float dist = getSegment(t, pos, 0.0);
    float glow = getGlow(dist, radius, intensity);

    vec3 col = vec3(0.0);

    col += 10.0 * vec3(smoothstep(0.006, 0.003, dist));
    col += glow * vec3(1.0, 0.05, 0.3);

    dist = getSegment(t, pos, 3.4);
    glow = getGlow(dist, radius, intensity);

    col += 10.0 * vec3(smoothstep(0.006, 0.003, dist));
    col += glow * vec3(0.1, 0.4, 1.0);

    col = 1.0 - exp(-col);
    col = pow(col, vec3(0.4545));

    fragColor = vec4(col, 1.0);
}
`,



  "Procgen Cats": `
// designs inspired by https://www.freepik.com/premium-vector/cute-cats-collection-cats-faces_10120615.htm
// comment this out to get four rows of procgen cats rather than 2
#define BIGGER

// Audio. Every cat's look is hashed from the seed below, so audio only ever
// nudges pose (lean, head bob, ears, tail, pupils) and never feeds the hash.
// Silence reproduces the original animation exactly. AUDIO_GAIN 0 disables.
// The spectrum arrives pre-amplified (x1.5) and bass/mids sit near the top
// during music, so each band is gated between a floor and a ceiling: only
// peaks above the floor move the cats. Raise a floor if a cat never settles,
// lower it if a band feels dead. DEBUG_AUDIO draws the raw (thin) and gated
// (thick) level of bass / mid / high as bars along the bottom.
#define AUDIO_GAIN  1.0
#define BASS_FLOOR  0.55
#define BASS_CEIL   0.95
#define MID_FLOOR   0.40
#define MID_CEIL    0.85
#define HIGH_FLOOR  0.15
#define HIGH_CEIL   0.55
#define BASS_HOP    0.06   // hop height on kicks, as a fraction of the cell
#define BASS_LEAN   8.0    // degrees of extra body lean on kicks
#define BASS_BOB    0.10   // head lift along the body
#define BASS_PUPIL  0.12   // pupil dilation, fraction of eye size
#define MID_TAIL    0.50   // extra tail swish
#define HIGH_EAR    14.0   // degrees the ears perk up
// #define DEBUG_AUDIO

#define PUPIL vec3(0.18, 0.17, 0.165)
#define CAT_BASE_1 vec3(0.9, 0.77, 0.55)
#define CAT_BASE_2 vec3(0.4, 0.37, 0.37)
#define CAT_BASE_3 vec3(0.7, 0.6, 0.5)
#define CAT_BASE_4 vec3(0.7, 0.6, 0.5)
#define CAT_BASE_5 vec3(0.9, 0.9, 0.83)
#define CAT_LIGHT_1 vec3(0.97, 0.9, 0.7)
#define CAT_LIGHT_2 vec3(1., 0.95, 0.97)
#define CAT_LIGHT_3 vec3(1., 0.95, 0.85)
#define CAT_LIGHT_4 vec3(1., 0.96, 0.78)
#define CAT_LIGHT_5 vec3(0.9, 0.9, 0.83)
#define CAT_DARK_1 vec3(0.7, 0.5, 0.4)
#define CAT_DARK_2 vec3(0.25, 0.2, 0.2)
#define CAT_DARK_3 vec3(0.5, 0.4, 0.35)
#define CAT_DARK_4 vec3(0.5, 0.4, 0.35)
#define CAT_DARK_5 vec3(0.44, 0.37, 0.36)
#define CAT_OUTLINE_1 vec3(0.37, 0.2, 0.13)
#define CAT_OUTLINE_2 vec3(0.18, 0.18, 0.16)
#define CAT_OUTLINE_3 vec3(0.35, 0.3, 0.26)
#define CAT_OUTLINE_4 vec3(0.35, 0.3, 0.26)
#define CAT_OUTLINE_5 vec3(0.35, 0.3, 0.26)
#define CAT_EYE_1 vec3(1.)
#define CAT_EYE_2 vec3(0.8, 0.76, 0.38)
#define CAT_EYE_3 vec3(0.6, 0.86, 0.9)
#define CAT_EYE_4 vec3(0.6, 0.86, 0.9)
#define CAT_EYE_5 vec3(0.9)
#define CAT_BELLY_1 vec3(0.97, 0.9, 0.7)
#define CAT_BELLY_2 vec3(1., 0.95, 0.97)
#define CAT_BELLY_3 vec3(0.7, 0.6, 0.5) * 1.1
#define CAT_BELLY_4 vec3(1., 0.96, 0.78)
#define CAT_BELLY_5 vec3(0.9, 0.9, 0.83)

#define TABBY_SPOT 0.075

#define MAX_ROTATION 150.
#define MIN_ROTATION -150.
#define MAX_EAR_DOWN 225.
#define MAX_EAR_UP 195.

#ifdef BIGGER
    #define OUTLINE_THICKNESS 0.04
#else
    #define OUTLINE_THICKNESS 0.02
#endif
#define OUTLINE_THRESHOLD 0.7

float aBass;
float aMid;
float aHigh;

float fft(float x) {
    return texture(iChannel0, vec2(clamp(x, 0.0, 1.0), 0.0)).x;
}
float getBass() { return (fft(0.01) + fft(0.03) + fft(0.05) + fft(0.07)) * 0.25; }
float getMid()  { return (fft(0.15) + fft(0.25) + fft(0.35) + fft(0.45)) * 0.25; }
float getHigh() { return (fft(0.55) + fft(0.70) + fft(0.85) + fft(0.95)) * 0.25; }
float gate(float v, float lo, float hi) {
    float g = smoothstep(lo, hi, v);
    return g * g;
}

struct Cat {
    float butt; // width of the lower end of the body
    float height; // body length
    float rotation; // body rotation in degrees
    vec3 baseColor;
    vec3 lightColor;
    vec3 darkColor;
    vec3 outlineColor;
    vec3 eyeColor;
    vec3 belly;
    float f; // this is to cache the FBM noise value
    float r; // progress from min rotation value to max rotation value from 0 to 1
    float faceDark;
    float heart;
    float spots;
    float tabby;
    int type;
};
mat2 Rotate2DMatrix(float a)
{
    float ca = cos(a);
    float sa = sin(a);
    return mat2(ca, -sa, sa, ca);
}
vec2 Rotate2D(vec2 p, float a)
{
    return Rotate2DMatrix(radians(a)) * p;
}
vec3 DoOutline(float d, vec3 color, float thickness, vec3 outColor)
{
    if (d < 0.)
    {
        float t = -d / thickness;
        t = t * t;
        color = mix(outColor, color, step(OUTLINE_THRESHOLD, t));
    }
    return color;
}
float InvLerp(in float from, float to, float value)
{
    return (value - from) / (to - from);
}
vec2 interpolate(vec2 a, vec2 b, vec2 c, vec2 d, float p)
{
    vec2 v0 = mix(a, b, p);
    vec2 v1 = mix(b, c, p);
    vec2 v2 = mix(c, d, p);
    vec2 v3 = mix(v0, v1, p);
    vec2 v4 = mix(v1, v2, p);

    return mix(v3, v4, p);
}
float Spline(in vec2 p, in vec2 a, in vec2 b)
{
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa,ba) / dot(ba,ba), 0., 1.);
    return length(pa - ba * h);
}
// 2D SDF functions from https://iquilezles.org/articles/distfunctions2d/
float dot2( in vec2 v ) { return dot(v,v); }
float dot2( in vec3 v ) { return dot(v,v); }
float Heart( in vec2 p )
{
    p.x = abs(p.x);

    if( p.y+p.x>1.0 )
        return sqrt(dot2(p-vec2(0.25,0.75))) - sqrt(2.0)/4.0;
    return sqrt(min(dot2(p-vec2(0.00,1.00)),
                    dot2(p-0.5*max(p.x+p.y,0.0)))) * sign(p.x-p.y);
}
float UnevenCapsule( vec2 p, float r1, float r2, float h )
{
    p.x = abs(p.x);
    float b = (r1-r2)/h;
    float a = sqrt(1.0-b*b);
    float k = dot(p,vec2(-b,a));
    if( k < 0.0 ) return length(p) - r1;
    if( k > a*h ) return length(p-vec2(0.0,h)) - r2;
    return dot(p, vec2(a,b) ) - r1;
}
float EquiTriSDF(in vec2 p)
{
    const float k = sqrt(3.0);
    p.x = abs(p.x) - 1.0;
    p.y = p.y + 1.0/k;
    if( p.x+k*p.y>0.0 ) p = vec2(p.x-k*p.y,-k*p.x-p.y)/2.0;
    p.x -= clamp( p.x, -2.0, 0.0 );
    return -length(p)*sign(p.y);
}
float Isosceles( in vec2 p, in vec2 q )
{
    p.x = abs(p.x);
    vec2 a = p - q*clamp( dot(p,q)/dot(q,q), 0.0, 1.0 );
    vec2 b = p - q*vec2( clamp( p.x/q.x, 0.0, 1.0 ), 1.0 );
    float s = -sign( q.y );
    vec2 d = min( vec2( dot(a,a), s*(p.x*q.y-p.y*q.x) ),
                  vec2( dot(b,b), s*(p.y-q.y)  ));
    return -sqrt(d.x)*sign(d.y);
}
float Egg( in vec2 p, in float ra, in float rb )
{
    const float k = sqrt(3.0);
    p.x = abs(p.x);
    float r = ra - rb;
    return ((p.y<0.0)       ? length(vec2(p.x,  p.y    )) - r :
            (k*(p.x+r)<p.y) ? length(vec2(p.x,  p.y-k*r)) :
                              length(vec2(p.x+r,p.y    )) - 2.0*r) - rb;
}
float Circle( vec2 p, float r )
{
    return length(p) - r;
}
float FBMRandom(vec2 st)
{
    // Seeded from the last click, not the hover position: iMouse.xy tracks the
    // pointer here even without a press, which would reshuffle every cat on
    // each mouse move. Normalised so adaptive resolution changes don't reshuffle.
    vec2 m = floor(abs(iMouse.zw) / max(iResolution.xy, vec2(1.)) * 512.);
    return fract(sin(dot(st.xy, m + vec2(12.9898, 78.233))) *
                    43758.5453123);
}
Cat GetCat(vec2 uv)
{
    // declaring the arrays outside the function doesn't work on some platforms
    // so gotta do this rigamarole here
    vec3[5] base;
    base[0] = CAT_BASE_1; base[1] = CAT_BASE_2; base[2] = CAT_BASE_3;
    base[3] = CAT_BASE_4; base[4] = CAT_BASE_5;

    vec3[5] light;
    light[0] = CAT_LIGHT_1; light[1] = CAT_LIGHT_2; light[2] = CAT_LIGHT_3;
    light[3] = CAT_LIGHT_4; light[4] = CAT_LIGHT_5;

    vec3[5] dark;
    dark[0] = CAT_DARK_1; dark[1] = CAT_DARK_2; dark[2] = CAT_DARK_3;
    dark[3] = CAT_DARK_4; dark[4] = CAT_DARK_5;

    vec3[3] eye;
    eye[0] = CAT_EYE_1; eye[1] = CAT_EYE_2; eye[2] = CAT_EYE_3;

    vec3[5] outline;
    outline[0] = CAT_OUTLINE_1; outline[1] = CAT_OUTLINE_2; outline[2] = CAT_OUTLINE_3;
    outline[3] = CAT_OUTLINE_4; outline[4] = CAT_OUTLINE_5;

    vec3[5] belly;
    belly[0] = CAT_BELLY_1; belly[1] = CAT_BELLY_2;
    belly[2] = CAT_BELLY_3; belly[3] = CAT_BELLY_4;
    belly[4] = CAT_BELLY_5;

    Cat cat;

    vec2 cUV = floor(uv * 2.);

    float f = clamp(0., 1., FBMRandom(cUV));
    cat.f = f;
    int i = 0;
    float count = 5.;
    float mul = 1000.;
    float cF = fract(f + cUV.x + cUV.y * 0.5);
    if (mod(floor(cF * mul) + 1., count) == 0.) i = 1;
    else if (mod(floor(cF * mul) + 2., count) == 0.) i = 2;
    else if (mod(floor(cF * mul) + 3., count) == 0.) i = 3;
    else if (mod(floor(cF * mul) + 4., count) == 0.) i = 4;

    cat.type = i;

    cat.baseColor = base[i];
    cat.lightColor = light[i];
    cat.darkColor = dark[i];
    cat.outlineColor = outline[i];
    cat.belly = belly[i];
    cat.eyeColor = mod(floor(f * 100.), 3.) == 0.? eye[0] :
        (mod(floor(f * 100.) + 1., 3.) == 0.? eye[1] : eye[2]);

    cat.rotation = mix(MIN_ROTATION, MAX_ROTATION, fract(f + cUV.x * 0.3));
    cat.r = InvLerp(MIN_ROTATION, MAX_ROTATION, cat.rotation);
    cat.height = mix(0.5, 1., fract(f + 0.5 + cUV.y));
    cat.butt = mix(0.7, 0.9, abs(f * 2. - 1.));

    cat.faceDark = mod(floor((1. - cat.f) * 5.), 4.) == 0.? 1. : 0.;
    cat.heart = mod(floor(fract(cat.f + 0.75) * 10.), 2.) == 0.? 1. : 0.;
    cat.spots = mod(floor(cat.f * 1450.), 2.) == 0.? 1. : 0.;
    cat.tabby = mod(floor(fract(cat.f + 0.25) * 1329.), 3.) == 0.? 1. : 0.;
    return cat;
}
float Smooth(float sdf) { return 1. - smoothstep(-0.025, -0.01, sdf);}
float TopMarks(vec2 headUV)
{
    float xSign = sign(headUV.x);
    vec2 mUV = headUV * vec2(7., 1.) + vec2(0., 2.);
    float markScale = 1.;
    float mSDF = Circle(mUV, markScale);

    vec2 mUV1 = Rotate2D(headUV, 16. * xSign);
    mUV1 = mUV1 * vec2(10., 1.) + vec2(-1. * -xSign, 2.2);
    mSDF = min(mSDF, Circle(mUV1, markScale));
    return mSDF;
}
float HeartMark(vec2 headUV)
{
    vec2 heartUV = headUV * vec2(0.15, 0.2) + vec2(0., 1.05);
    return Heart(heartUV);
}
void HeadMarkings(vec2 p, Cat cat, inout vec3 col)
{
    float xSign = sign(p.x);
    vec3 light = mix(cat.lightColor * 1.1, cat.lightColor,
        clamp(0., 1., dot(p, p)));
    if (cat.heart == 1.)
    {
        float heartSDF = HeartMark(p);
        col = heartSDF < 0.? col : light;
    }

    if (cat.faceDark == 1.)
    {
        float spots = 0.;
        vec3 sColor = cat.darkColor;
        if (cat.spots == 1.)
        {
            float w =  sin(p.x * 50.) * 0.025;
            vec2 stUV = p + vec2(1., 1.4);
            stUV.y += w;
            spots = Circle(stUV * vec2(1., 1.2), 1.2);
            stUV -= vec2(2.3, 0.1);
            spots = min(spots, Circle(Rotate2D(stUV, 25.) * vec2(1., 1.75), 0.75));
            if (cat.heart == 0.)
            {
                stUV = p - vec2(0.8, 1.4);
                stUV.y += w * 0.2;
                stUV.x *= 0.7;
                spots = min(spots, Circle(stUV, 0.75));
            }
            if (cat.tabby == 1.) sColor += TABBY_SPOT;
        }
        else
        {
            vec2 sideUV = p - vec2(1.6 * xSign, -0.9);
            spots = Circle(sideUV, 1.5);
        }
         col = mix(col, sColor, Smooth(spots));
    }
    if (cat.tabby == 1.)
    {
        float mSDF = TopMarks(p);
        col = mix(col, cat.darkColor, Smooth(mSDF));
    }
    if (mod(floor(cat.f * 1000.), 2.) == 0.)
    {
        float muzzle = Isosceles(p * 3. + vec2(0., 0.1), vec2(0.9, 1.)) - 1.;
        col = mix(col, light, Smooth(muzzle));
    }
}
void Eyes(vec2 headUV, Cat cat, inout vec3 col)
{
    float xSign = sign(headUV.x);
    vec2 eyeUV = headUV + vec2(0., 0.5);
    eyeUV.x -= 0.8 * xSign;

    // blinky blinky
    float t = sin(iTime * max(0.5, cat.f) + cat.f) * 0.5 + 0.5;
    float b = smoothstep(0., 1., InvLerp(0.999, 1., t));
    eyeUV.y = mix(eyeUV.y, eyeUV.y * 7., b);

    float eyeScale = mix(0.5, 0.55, cat.f);
    float eyeSDF = Circle(eyeUV, eyeScale);
    vec3 eyeCol = cat.eyeColor;
    float pupil = Circle(eyeUV, eyeScale * (0.75 + aBass * BASS_PUPIL));
    eyeCol = mix(eyeCol, PUPIL, Smooth(pupil));
    vec2 shineUV = eyeUV + vec2(0.2, 0.15);
    float shine = Circle(shineUV, eyeScale * 0.3);
    shine = min(shine, Circle(shineUV + vec2(0.1, -0.2), eyeScale * 0.15));
    eyeCol = mix(eyeCol, vec3(1.), Smooth(shine));
    vec2 lidUV = eyeUV * vec2(1., 0.95) + vec2(0., 0.05);
    lidUV.y = mix(lidUV.y, lidUV.y * 0.95 - 0.02, b);
    float lid = Circle(lidUV, eyeScale);
    vec3 lidColor = cat.faceDark == 1.? cat.outlineColor : cat.darkColor;
    eyeCol = mix(lidColor, eyeCol, Smooth(eyeSDF));
    eyeSDF = min(eyeSDF, lid);
    col = mix(col, eyeCol, Smooth(eyeSDF));
}
float Head(vec2 p, Cat cat, out vec3 col)
{
    float headHeight = cat.height * 1.2
        + aBass * BASS_BOB * (0.6 + 0.4 * fract(cat.f * 13.));
    float headScale = 2.5;
    vec2 headUV = (p - vec2(0., headHeight)) * headScale;
    headUV = Rotate2D(headUV, -cat.rotation);
    headUV.y *= -1.;
    float xSign = sign(headUV.x);
    float headSDF = Egg(headUV, 2., 3.) + 0.1;
    vec2 huv = headUV + vec2(0., 0.);
    col = mix(cat.baseColor * 1.2, cat.baseColor,
        smoothstep(0., 1.5, dot(huv, huv)));

    // markings
    HeadMarkings(headUV, cat, col);

    //Ears
    vec2 earUV = headUV;
    earUV.x -= 1.1 * xSign;
    earUV.y += 1.1;
    float earRot = mix(MAX_EAR_DOWN, MAX_EAR_UP, fract(cat.f * 25.));
    earRot -= aHigh * HIGH_EAR * (0.5 + 0.5 * fract(cat.f * 31.));
    float earTip = 1.;
    if (mod(floor(cat.f * 0.35 * 1000.), 3.) == 0.) earTip = 0.75;
    earUV = Rotate2D(earUV, earRot * xSign);
    float ear = Egg(earUV, 2., earTip) + 1.2;
    vec3 earColor = cat.darkColor;
    earUV = Rotate2D(earUV, -5. * xSign);
    float earNeg = Egg(earUV * vec2(1.1, 1.) + vec2(0.2 * xSign, 0.), 2., 1.) + 1.4;
    earColor = mix(earColor, cat.lightColor, Smooth(earNeg));
    col = mix(earColor, col, Smooth(headSDF));
    headSDF = min(headSDF, ear);

    // nose
    vec3 noseColor = cat.outlineColor;
    float noseScale = 8.;
    vec2 noseUV = headUV;

    float bridge = Isosceles(noseUV * noseScale + vec2(0., 2.3),
        vec2(0.7, 2.5)) - .75;
    vec3 bridgeCol = cat.darkColor;
    if (cat.type == 1) bridgeCol = cat.baseColor;
    else if (mod(floor(cat.f * 2324321.), 2.) == 0.)
    {
        bridgeCol = cat.baseColor * 1.1;
    }
    col = mix(col, bridgeCol, Smooth(bridge));

    noseUV *= vec2(1., -1.);
    noseUV.y += 0.2;
    float noseSDF = Isosceles(noseUV * noseScale, vec2(1.1, 0.8)) - .5;
    col = mix(col, noseColor, Smooth(noseSDF));

    // Eyes
    Eyes(headUV, cat, col);

    // Mouth
    vec2 mouthUV = headUV - vec2(0., 0.3);
    mouthUV.x *= 0.8;
    mouthUV.x -= 0.19 * xSign;
    float mouthScale = 0.2;
    float mouth = Circle(mouthUV, mouthScale);
    mouthUV = mouthUV * vec2(1.2, 0.9) + vec2(0., 0.1);
    mouth = max(mouth, -Circle(mouthUV, mouthScale));
    col = mix(col, cat.outlineColor, Smooth(mouth));

    // head outline before whiskers
    col = DoOutline(headSDF, col, OUTLINE_THICKNESS * headScale, cat.outlineColor);

    // whiskers
    vec2 wUV = headUV;
    wUV.x -= 2. * xSign;
    wUV.y -= 1.;
    float wScale = 1.2;
    float whisker = Circle(wUV, wScale);
    vec2 wUV2 = wUV * vec2(0.8, 1.) - vec2(0.1 * xSign, 0.075);
    whisker = max(whisker, -Circle(wUV2, wScale));

    wUV.x += 0.35 * xSign;
    wUV.y -= 0.25;
    float w2 = Circle(wUV, wScale);
    wUV2 = wUV * vec2(0.8, 1.) + vec2(0.1 * xSign, -0.075);
    w2 = max(w2, -Circle(wUV2, wScale));
    whisker = min(whisker, w2);

    wUV.x += 0.1 * xSign;
    wUV.y -= 0.15;
    w2 = Circle(wUV, wScale);
    wUV2 = wUV * vec2(0.75, 1.) + vec2(0.2 * xSign, -0.05);
    w2 = max(w2, -Circle(wUV2, wScale));
    whisker = min(whisker, w2);

    col = whisker < 0.? cat.outlineColor : col;

    headSDF = min(headSDF, whisker);
    return headSDF;
}
float Tail(vec2 uv, Cat cat)
{
    float rSign = sign(cat.rotation);
    float xMul = cat.r * 2. - 1.;
    float invX = 1. - abs(xMul);
    float t = (sin(iTime * cat.f + cat.f) * 0.5 + 0.5);
    t += aMid * MID_TAIL * sin(iTime * 2.2 + cat.f * 6.2831853);

    vec2 a = vec2(0., 0.6);
    vec2 b = vec2(1. * xMul * t, 1. * abs(xMul));
    vec2 c = vec2((-0.5 + 0.5 *invX) * xMul + 1.5 * invX + t * 0.5, 1. - 0.5 * invX);
    vec2 d = vec2((0.3 + 0.5 * invX) * xMul * t, 1.5 + 0.15 * abs(xMul) * t * 2.);

    float tail = 0.;
    const float STEPS = 20.;
    for (float i = 0.; i < STEPS; ++i)
    {
        vec2  p0 = interpolate(a, b, c, d, (i   ) / STEPS);
        vec2  p1 = interpolate(a, b, c, d, (i+1.) / STEPS);
        float l = Spline(uv, p0, p1) - 0.1;
        tail = min(tail, l);
    }
    return tail;
}
void BackPattern(vec2 p, Cat cat, inout vec3 col)
{
    float r = cat.r * 2. - 1.;
    float aR = abs(r);
    float rSign = sign(r);

    //spots
    if ((cat.spots == 1. || cat.faceDark == 1.))
    {
        vec2 spotUV = p * 7.;
        spotUV.y += mix(10., 2., aR) * rSign;
        spotUV.x += mix(-10., -3., aR) * rSign;

        // if smooth face marks, spots should also be smooth
        if (cat.spots == 1.) spotUV.y += sin(spotUV.x * 10.) * 0.1;

        float spot = EquiTriSDF(spotUV) - 2.;
        spot = min(spot, EquiTriSDF(spotUV * 2. +
            vec2(10. * rSign + 10. * step(rSign, 0.), 6. + 5. * step(rSign, 0.))) - 2.);

        vec3 spotColor = cat.darkColor;
        if (cat.tabby == 1.) spotColor += TABBY_SPOT;

        col = spot < 0.? spotColor : col;
    }

    // tabby stripes
    if (cat.tabby == 1.)
    {
        float xSign = sign(p.x);
        vec2 uv = p;

        if (rSign > 0.)
        {
            uv.x += mix(-2., -.05, aR);
            uv = Rotate2D(uv, 25. * aR);
        }
        else if (rSign < 0.)
        {
            uv.x += mix(2., 0.5, aR);
            uv = Rotate2D(uv, -50. * aR);
        }
        uv *= vec2(1., 8.);
        float l = Circle(uv, 0.5);
        l = min(l, Circle(uv + vec2(0., 1.5), 0.5));
        l = min(l, Circle(uv - vec2(0, 1.5), 0.5));
        col = l < 0.? cat.darkColor : col;
    }
}
float SceneSDF(vec2 p, out vec3 col)
{
    Cat cat = GetCat(p);
    p = fract(p * 2.) - 0.5;
    #ifdef BIGGER
    float scale = 4.;
    p.y += 0.1;
    #else
     float scale = 4.5;
    #endif

    vec2 ground = p;
    float hop = aBass * BASS_HOP * (0.6 + 0.4 * fract(cat.f * 17.));
    p.y -= hop;

    vec2 bodyUV = p * scale;

    col = mix(cat.baseColor, cat.baseColor * 1.2, dot(bodyUV, bodyUV) * 0.5 + 0.5);
    cat.rotation += mix(-10., 10., cos(iTime * 0.2 * cat.f + (cat.f * 2. - 1.)) * 0.5 + 0.5);
    // lean direction and depth vary per cat so the row doesn't sway in lockstep
    cat.rotation += aBass * BASS_LEAN * (fract(cat.f * 7.) * 2. - 1.);
    bodyUV = Rotate2D(bodyUV, cat.rotation);
    float sdf = UnevenCapsule(bodyUV, cat.butt, 0.5, cat.height);

    BackPattern(bodyUV, cat, col);

    vec2 bellyUV = bodyUV;
    bellyUV.y += sin(bodyUV.x * 75.) * 0.025;
    float b = InvLerp(-90., 90., cat.rotation);
    float bX = 0.75;
    bellyUV.x += mix(-bX, bX, b);
    bellyUV.y += mix(-0.1, 0.1, b);
    float width = mix(0.7, 0.5, abs(b * 2. - 1.));
    float belly = UnevenCapsule(bellyUV * 1.5, width, 0.5, cat.height) - 0.2;
    col = mix(col, cat.belly, Smooth(belly));
    col = DoOutline(sdf, col, OUTLINE_THICKNESS, cat.outlineColor);

    // head casting shadow on the body
    float shadow = Circle(bodyUV - vec2(0., cat.height * 1.1), 0.5);
    col = shadow < 0.? col * 0.9 : col;

    vec3 headCol;
    float headSDF = Head(bodyUV, cat, headCol);
    col = headSDF < 0.? headCol : col;
    sdf = min(sdf, headSDF);

    float tail = Tail(p * scale, cat);
    vec3 tailCol = DoOutline(tail, cat.baseColor, OUTLINE_THICKNESS, cat.outlineColor);
    col = sdf < 0.? col : tailCol;
    sdf = min(sdf, tail);

    // body casting shadow on the floor
    vec2 sUV = ground * scale;
    sUV = (sUV * vec2(1., 3.) + vec2(0., 1.7)) * (1. + hop * 6.);
    shadow = Circle(sUV, 1.);
    vec3 shadowCol = mix(vec3(1.), vec3(0.7),
        (1. - smoothstep(-0.5, -0.001, shadow)));
    col = sdf < 0.? col : shadowCol;
    sdf = min(sdf, shadow);

    return sdf;
}
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    float rawBass = getBass();
    float rawMid  = getMid();
    float rawHigh = getHigh();
    aBass = gate(rawBass, BASS_FLOOR, BASS_CEIL) * AUDIO_GAIN;
    aMid  = gate(rawMid,  MID_FLOOR,  MID_CEIL)  * AUDIO_GAIN;
    aHigh = gate(rawHigh, HIGH_FLOOR, HIGH_CEIL) * AUDIO_GAIN;

    vec2 p = (2.0*fragCoord-iResolution.xy)/iResolution.y;
    p.x += (iTime + 3000.) * 0.25;
    #ifdef BIGGER
        // adjusting for how it looks different in shadertoy
        // than from VSCode where I authored this...
        p *= 0.5;
        p.y += 0.01;
    #else
        p.y += 0.025;
    #endif

    vec3 col = vec3(1.);
    vec3 fg = vec3(0.);

    float sdf = SceneSDF(p, fg);
    col = sdf < 0.? fg : col;

    #ifdef DEBUG_AUDIO
    {
        vec2 q = fragCoord / iResolution.xy;
        float band = floor(q.x * 3.);
        float raw   = band < 0.5 ? rawBass : (band < 1.5 ? rawMid : rawHigh);
        float gated = band < 0.5 ? aBass   : (band < 1.5 ? aMid   : aHigh);
        float lo    = band < 0.5 ? BASS_FLOOR : (band < 1.5 ? MID_FLOOR : HIGH_FLOOR);
        float x = fract(q.x * 3.);
        if (q.y < 0.05)
        {
            col = mix(vec3(0.15), vec3(1., 0.4, 0.2), step(x, gated));
            if (q.y > 0.04) col = mix(vec3(0.3), vec3(0.2, 0.6, 1.), step(x, raw));
            if (abs(x - lo) < 0.003) col = vec3(1.);
        }
    }
    #endif

    fragColor = vec4(col,1.0);
}
`
};

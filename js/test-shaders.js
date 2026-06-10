export const TEST_SHADERS = {
  "Sound Reactive Waves": `
#define PI 3.14159265
#define S smoothstep

float fft(float x) {
    return texture(iChannel0, vec2(clamp(x, 0.0, 1.0), 0.0)).x;
}

vec3 palette(in float t) {
    vec3 a = vec3(0.5, 0.5, 0.5);
    vec3 b = vec3(0.5, 0.5, 0.5);
    vec3 c = vec3(1.0, 1.0, 1.0);
    vec3 d = vec3(0.1, 0.4, 0.5);
    return a + b * cos(2.0 * PI * (c * t + d));
}

vec4 wave(vec2 uv, float speed, float height, float layer, vec3 hue) {
    float audio = fft(height * 0.05);
    float amp = 0.25 * (1.0 + audio * 0.35) * (1.0 - layer * 0.5);

    float x = uv.x - iTime * speed * (1.0 - layer);
    float y = uv.y + amp * sin(height * x);
    float thick = 0.01 + 0.001 * pow(abs(uv.x), 8.0);
    float bright = S(0.0, 1.0, 1.0 - abs(y) / thick);

    return vec4(vec3(bright) * hue, 1.0);
}

void mainImage(out vec4 color, in vec2 coord) {
    vec2 uv = (2.0 * coord - iResolution.xy) / iResolution.y;

    float audioSum = fft(0.08);
    float audioMult = 0.55 + audioSum * 0.6;

    color = vec4(0.0, 0.0, 0.0, 1.0);

    for (float layer = 0.0; layer < 1.0; layer += 0.1) {
        float t = layer;
        vec3 hue = palette(0.5 * uv.x + t - 0.5 * iTime + audioSum * 0.04) * audioMult;
        color += wave(uv, 1.0 + t, 4.0 + t * 2.0, t, hue);
    }
}
`,

  "PIANOSCOPE Afrohouse Mudcloth Bloom": `
/*
PIANOSCOPE Shader
Name: Afrohouse Mudcloth Bloom
Mode: afrohouse_mudcloth_bloom
Inspired by: Bogolan/Bògòlanfini — hand-painted clay marks on cloth, not wallpaper
Cultural caution: Abstracted rough geometry and earthy palette; not copying specific textile symbols
Audio: Bass = clay bloom expansion; Mid = mark warping / cloth breath; High = sand speckles
Projection: Warm dark field, broken asymmetric marks, slow organic motion, no strobe
*/

#define PI 3.14159265359
#define BLOOM_BASS_GAIN  0.32
#define WARP_MID_GAIN    0.45
#define SPECKLE_HIGH     0.55
#define BREATH_SPEED     0.04

vec2 centeredUV(vec2 fragCoord) {
    return (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
}

float fft(float x) {
    return texture(iChannel0, vec2(clamp(x, 0.0, 1.0), 0.0)).x;
}

float getBass() {
    return (fft(0.01) + fft(0.03) + fft(0.05) + fft(0.07)) * 0.25;
}

float getMid() {
    return (fft(0.15) + fft(0.25) + fft(0.35) + fft(0.45)) * 0.25;
}

float getHigh() {
    return (fft(0.55) + fft(0.70) + fft(0.85) + fft(0.95)) * 0.25;
}

float fallbackBass() { return 0.35 + 0.25 * sin(iTime * 1.2); }
float fallbackMid()  { return 0.28 + 0.18 * sin(iTime * 0.7 + 1.0); }
float fallbackHigh() { return 0.18 + 0.12 * sin(iTime * 2.8 + 2.0); }

float safeAudio(float value, float fallback) {
    return max(value, fallback * 0.35);
}

float audioBoost(float v) {
    return clamp(pow(v, 0.75) * 1.35, 0.0, 1.0);
}

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 5; i++) {
        v += a * noise(p);
        p = p * 2.02 + vec2(1.7, 9.2);
        a *= 0.5;
    }
    return v;
}

vec3 mudPalette(float t) {
    vec3 charcoal = vec3(0.025, 0.020, 0.018);
    vec3 clay     = vec3(0.55, 0.22, 0.08);
    vec3 rust     = vec3(0.75, 0.30, 0.10);
    vec3 ochre    = vec3(0.86, 0.58, 0.24);
    vec3 sand     = vec3(0.90, 0.73, 0.48);
    vec3 c = mix(charcoal, clay, smoothstep(0.0, 0.35, t));
    c = mix(c, rust, smoothstep(0.25, 0.55, t));
    c = mix(c, ochre, smoothstep(0.45, 0.75, t));
    c = mix(c, sand, smoothstep(0.7, 1.0, t));
    return c;
}

// Domain warp — cloth breathing (mid-driven)
vec2 warpCloth(vec2 uv, float mid, float t) {
    vec2 q = uv;
    q += vec2(
        fbm(uv * 2.4 + vec2(t, 0.0)) - 0.5,
        fbm(uv * 2.4 + vec2(4.0, t * 0.8)) - 0.5
    ) * 0.12 * (1.0 + mid * WARP_MID_GAIN);
    return q;
}

// Rough broken line — hand-painted edge
float roughLine(vec2 p, float yTarget, float thick, float rough) {
    float n = fbm(p * 6.0 + iTime * BREATH_SPEED);
    float edge = abs(p.y - yTarget + (n - 0.5) * rough);
    return smoothstep(thick, thick * 0.35, edge);
}

// Organic clay blooms — thresholded fBm, bass opens them
float clayBlooms(vec2 uv, float bass, float t) {
    float n1 = fbm(uv * 1.6 + vec2(t * 0.3, t * 0.2));
    float n2 = fbm(uv * 3.2 - vec2(t * 0.15, t * 0.25) + 3.0);
    float field = n1 * 0.65 + n2 * 0.35;
    float thresh = 0.52 - bass * BLOOM_BASS_GAIN;
    return smoothstep(thresh, thresh - 0.12, field);
}

// One hand-painted mark inside a cloth cell
float cellMark(vec2 f, float seed, float mid) {
    float m = 0.0;
    float wobble = (fbm(f * 4.0 + seed) - 0.5) * 0.08 * (1.0 + mid);

    if (seed < 0.28) {
        m = roughLine(f + vec2(0.0, wobble), 0.0, 0.07, 0.18);
    } else if (seed < 0.52) {
        float zig = sin((f.x + wobble) * 9.0 + seed * 6.0) * 0.14;
        m = smoothstep(0.045, 0.0, abs(f.y - zig));
    } else if (seed < 0.76) {
        m = max(
            roughLine(f + vec2(wobble, 0.0), 0.0, 0.05, 0.14),
            roughLine(f + vec2(0.0, wobble), 0.0, 0.05, 0.14)
        );
    } else {
        float r = length(f + vec2(wobble, -wobble)) - 0.12;
        m = smoothstep(0.04, 0.0, abs(r));
        float r2 = length(f - vec2(0.08, 0.0)) - 0.05;
        m = max(m, smoothstep(0.03, 0.0, abs(r2)));
    }

    float edgeFade = smoothstep(0.42, 0.28, length(f));
    float breakGap = step(0.15, hash21(f * 3.0 + seed * 11.0));
    return m * edgeFade * breakGap;
}

// Staggered mudcloth rows — asymmetric, not seamless wallpaper
float clothMarks(vec2 uv, float mid) {
    float scale = 7.0;
    vec2 cell = uv * vec2(scale * 1.1, scale * 0.85);
    vec2 id = floor(cell);
    vec2 f = fract(cell) - 0.5;

    if (mod(id.y, 2.0) > 0.5) {
        f.x += 0.5;
        id.x += 0.5;
    }

    f.x += (hash21(id + 1.3) - 0.5) * 0.35;
    f.y += (hash21(id + 7.1) - 0.5) * 0.25;

    float seed = hash21(id);
    float empty = step(0.12, seed);
    return cellMark(f, seed, mid) * empty;
}

// Sand grain speckles — high frequencies only
float sandSpeckles(vec2 uv, float high) {
    vec2 g = uv * 120.0;
    vec2 id = floor(g);
    vec2 f = fract(g);
    float h = hash21(id + floor(iTime * 6.0));
    float d = length(f - 0.5);
    float speck = smoothstep(0.35, 0.0, d) * step(0.92, h);
    return speck * high * SPECKLE_HIGH;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = centeredUV(fragCoord);
    float t = iTime * BREATH_SPEED;

    float bass = audioBoost(safeAudio(getBass(), fallbackBass()));
    float mid  = audioBoost(safeAudio(getMid(), fallbackMid()));
    float high = audioBoost(safeAudio(getHigh(), fallbackHigh()));

    vec2 wuv = warpCloth(uv, mid, t);

    float blooms = clayBlooms(wuv, bass, t);
    float marks  = clothMarks(wuv, mid);
    float speck  = sandSpeckles(uv, high);

    float tone = blooms * 0.75 + marks * 0.9;
    vec3 col = mudPalette(tone);

    col = mix(vec3(0.025, 0.020, 0.018), col, 0.35 + blooms * 0.55);
    col += mudPalette(0.85) * marks * 0.35;
    col += vec3(0.90, 0.73, 0.48) * speck;

    float grain = fbm(uv * 18.0 + t) * 0.04;
    col += vec3(0.55, 0.22, 0.08) * grain * (0.5 + mid * 0.3);

    col *= smoothstep(1.3, 0.4, length(uv));
    col = sqrt(max(col, 0.0));
    col = min(col, vec3(0.78));

    fragColor = vec4(col, 1.0);
}
`,

  "PIANOSCOPE Adinkra Symbol Field Abstract": `
/*
PIANOSCOPE Shader
Name: Adinkra Symbol Field Abstract
Mode: adinkra_symbol_field_abstract
Inspired by: Mandala (polar fold + iterative mod2 scale) + Adinkra stamp geometry
Cultural caution: Procedural circles, arcs, crosses — not exact sacred Adinkra symbols
Audio: Bass = recursion bloom / central reveal; Mid = field rotation; High = edge shimmer
Projection: Dark violet field, gold/rust marks, slow organic drift, no strobe
*/

#define PI  3.141592654
#define TAU (2.0 * PI)
#define BASS_RECUR_GAIN  0.28
#define MID_ROT_GAIN     0.50
#define HIGH_SHIMMER     0.60
#define SECTOR_COUNT     12.0

vec2 centeredUV(vec2 fragCoord) {
    return (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
}

float fft(float x) {
    return texture(iChannel0, vec2(clamp(x, 0.0, 1.0), 0.0)).x;
}

float getBass() {
    return (fft(0.01) + fft(0.03) + fft(0.05) + fft(0.07)) * 0.25;
}

float getMid() {
    return (fft(0.15) + fft(0.25) + fft(0.35) + fft(0.45)) * 0.25;
}

float getHigh() {
    return (fft(0.55) + fft(0.70) + fft(0.85) + fft(0.95)) * 0.25;
}

float fallbackBass() { return 0.35 + 0.25 * sin(iTime * 1.1); }
float fallbackMid()  { return 0.28 + 0.18 * sin(iTime * 0.65 + 1.0); }
float fallbackHigh() { return 0.18 + 0.12 * sin(iTime * 2.6 + 2.0); }

float safeAudio(float value, float fallback) {
    return max(value, fallback * 0.35);
}

float audioBoost(float v) {
    return clamp(pow(v, 0.75) * 1.35, 0.0, 1.0);
}

void rot(inout vec2 p, float a) {
    float c = cos(a);
    float s = sin(a);
    p = vec2(c * p.x + s * p.y, -s * p.x + c * p.y);
}

vec2 mod2(inout vec2 p, vec2 size) {
    vec2 c = floor((p + size * 0.5) / size);
    p = mod(p + size * 0.5, size) - size * 0.5;
    return c;
}

vec2 modMirror2(inout vec2 p, vec2 size) {
    vec2 halfsize = size * 0.5;
    vec2 c = floor((p + halfsize) / size);
    p = mod(p + halfsize, size) - halfsize;
    p *= mod(c, vec2(2.0)) * 2.0 - vec2(1.0);
    return c;
}

vec2 toSmith(vec2 p) {
    float d = (1.0 - p.x) * (1.0 - p.x) + p.y * p.y;
    float x = (1.0 + p.x) * (1.0 - p.x) - p.y * p.y;
    float y = 2.0 * p.y;
    return vec2(x, y) / d;
}

vec2 fromSmith(vec2 p) {
    float d = (p.x + 1.0) * (p.x + 1.0) + p.y * p.y;
    float x = (p.x + 1.0) * (p.x - 1.0) + p.y * p.y;
    float y = 2.0 * p.y;
    return vec2(x, y) / d;
}

vec2 toRect(vec2 p) {
    return vec2(p.x * cos(p.y), p.x * sin(p.y));
}

vec2 toPolar(vec2 p) {
    return vec2(length(p), atan(p.y, p.x));
}

float circle(vec2 p, float r) {
    return length(p) - r;
}

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 4; i++) {
        v += a * noise(p);
        p = p * 2.03 + vec2(1.7, 9.2);
        a *= 0.5;
    }
    return v;
}

vec3 adinkraPalette(float t, float layer) {
    vec3 midnightViolet = vec3(0.13, 0.06, 0.25);
    vec3 deepBlack      = vec3(0.015, 0.010, 0.018);
    vec3 rustOrange     = vec3(0.70, 0.27, 0.10);
    vec3 mutedGold      = vec3(0.83, 0.57, 0.18);
    vec3 cream          = vec3(0.92, 0.80, 0.56);

    vec3 c = mix(deepBlack, midnightViolet, smoothstep(0.0, 0.35, t));
    c = mix(c, rustOrange, smoothstep(0.25, 0.65, t + layer * 0.12));
    c = mix(c, mutedGold, smoothstep(0.45, 0.85, t + layer * 0.08));
    c = mix(c, cream, smoothstep(0.75, 1.0, t) * 0.35);
    return c;
}

// Abstract stamp mark: ring, arcs, cross — not a specific Adinkra symbol
float abstractMark(vec2 p, float phase) {
    float d = 10000.0;

    float ring = abs(circle(p, 0.36)) - 0.035;
    d = min(d, ring);

    float hub = circle(p, 0.11);
    d = min(d, hub);

    for (int i = 0; i < 3; i++) {
        float ang = float(i) * TAU / 3.0 + phase;
        vec2 c = vec2(cos(ang), sin(ang)) * 0.24;
        float arc = abs(length(p - c) - 0.075) - 0.022;
        d = min(d, arc);
    }

    float cross = max(abs(p.x), abs(p.y)) - 0.018;
    cross = max(cross, circle(p, 0.16));
    d = min(d, cross);

    float diamond = abs(p.x) + abs(p.y) - 0.30;
    diamond = max(diamond, -circle(p, 0.28));
    d = min(d, abs(diamond) - 0.015);

    return d;
}

float adinkra_df(float localTime, vec2 p, float bass, float mid) {
    vec2 pp = toPolar(p);
    float a = TAU / SECTOR_COUNT;
    float np = pp.y / a;
    pp.y = mod(pp.y, a);
    if (mod(np, 2.0) > 1.0) {
        pp.y = a - pp.y;
    }
    pp.y += localTime * 0.025 + mid * MID_ROT_GAIN * 0.04;
    p = toRect(pp);
    p = abs(p);
    p -= vec2(0.48);

    float d = 10000.0;
    float scalePulse = 0.5 + 0.5 * sin(localTime * 0.35);

    for (int i = 0; i < 4; i++) {
        float fi = float(i);
        mod2(p, vec2(1.0));
        float wobble = -0.12 * cos(localTime * 0.25 + fi);
        float mark = abstractMark(p, localTime * 0.15 + fi * 0.7) + wobble;
        d = min(d, mark);

        float grow = 1.42 + bass * BASS_RECUR_GAIN + 0.06 * scalePulse;
        p *= grow;
        rot(p, 0.55 + mid * MID_ROT_GAIN * 0.08 + fi * 0.12);
    }

    return d;
}

vec2 fieldDistort(float localTime, vec2 uv, float mid) {
    float lt = 0.08 * localTime + mid * MID_ROT_GAIN * 0.05;
    vec2 suv = toSmith(uv);
    suv += 0.65 * vec2(cos(lt), sin(sqrt(2.0) * lt));
    uv = fromSmith(suv);
    modMirror2(uv, vec2(1.8 + 0.25 * sin(lt * 0.7)));
    return uv;
}

vec3 shadeField(float d, float layerMix, float high, float reveal) {
    float fill = smoothstep(0.018, -0.012, d);
    float edge = smoothstep(0.006, 0.0, abs(d));
    float band = 0.5 + 0.5 * sin(d * 80.0);
    vec3 base = adinkraPalette(band * 0.5 + layerMix * 0.35, layerMix);
    vec3 col = mix(vec3(0.015, 0.010, 0.018), base, fill * (0.55 + reveal * 0.45));
    col += vec3(0.92, 0.80, 0.56) * edge * (0.25 + high * HIGH_SHIMMER);
    col += vec3(0.83, 0.57, 0.18) * edge * edge * high * 0.35;
    return col;
}

vec3 adinkra_post(vec3 col, vec2 uv, float localTime, float r) {
    col = clamp(col, 0.0, 1.0);
    col = pow(col, mix(vec3(0.55, 0.72, 1.15), vec3(0.48), r));
    col = col * 0.62 + 0.38 * col * col * (3.0 - 2.0 * col);
    col = mix(col, vec3(dot(col, vec3(0.33))), -0.25);
    float pulse = sqrt(max(1.0 - 0.65 * sin(localTime * 0.4 + r * 6.0), 0.0));
    col *= mix(0.85, 1.0, pulse);
    col *= 0.55 * sqrt(max(1.05 - r * r, 0.0));
    return clamp(col, 0.0, 1.0);
}

vec3 sampleField(float localTime, vec2 p, float bass, float mid, float high) {
    vec2 uv = p * 6.5;
    rot(uv, localTime * 0.04 + mid * MID_ROT_GAIN * 0.12);

    vec2 nuv = fieldDistort(localTime, uv, mid);
    vec2 nuv2 = fieldDistort(localTime, uv + vec2(0.0008), mid);
    float warpGlow = 1.0 - smoothstep(0.0, 0.003, length(nuv - nuv2));

    float d = adinkra_df(localTime, nuv, bass, mid);
    float r = length(p);

    float reveal = smoothstep(0.55, 0.08, r) * (0.45 + bass * 0.55);
    vec3 col = shadeField(d, r * 0.8 + bass * 0.2, high, reveal);

    float smoke = fbm(p * 2.2 + vec2(localTime * 0.03, -localTime * 0.02));
    col = mix(col, adinkraPalette(smoke, 0.2), smoke * 0.08 * (1.0 - reveal * 0.5));

    col += vec3(0.13, 0.06, 0.25) * warpGlow * 0.12;
    col += vec3(0.92, 0.80, 0.56) * warpGlow * warpGlow * high * 0.08;

    col = adinkra_post(col, nuv, localTime, r);
    return clamp(col, 0.0, 1.0);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = centeredUV(fragCoord);
    float localTime = iTime + 20.0;

    float bass = audioBoost(safeAudio(getBass(), fallbackBass()));
    float mid  = audioBoost(safeAudio(getMid(), fallbackMid()));
    float high = audioBoost(safeAudio(getHigh(), fallbackHigh()));

    vec3 col = sampleField(localTime, uv, bass, mid, high);

    float r = length(uv);
    col = mix(col, vec3(0.015, 0.010, 0.018), smoothstep(0.95, 1.25, r));
    col = sqrt(max(col, 0.0));
    col = min(col, vec3(0.82));

    fragColor = vec4(col, 1.0);
}
`,
};

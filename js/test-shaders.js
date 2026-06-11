export const TEST_SHADERS = {
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
`
};

export const TEST_SHADERS = {
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
  "PIANOSCOPE Amapiano Kente Loom": `
/*
PIANOSCOPE Shader
Name: Amapiano Kente Loom (v5.1 — smooth motion)
Mode: amapiano_kente_loom
Inspired by: p5.js kente unit grid — drawStripe / drawMovingRect / drawCircleStripe
Palette: coolors.co (nighttime weighted)
Audio: Bass = gentle stripe width; Mid = very subtle cycle drift; High = soft edge shimmer
Note: v5.1 removes ratio jumps and raw-FFT modulation that caused jitter
*/

#define TAU 6.28318530718
#define UNITS           5.0
#define CYCLE           7.5
#define STRIPE_BASS     0.22
#define SHIMMER_HIGH    0.30

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

float fallbackBass() { return 0.32 + 0.12 * sin(iTime * 0.85); }
float fallbackMid()  { return 0.28 + 0.08 * sin(iTime * 0.45 + 0.8); }
float fallbackHigh() { return 0.16 + 0.06 * sin(iTime * 1.6 + 1.5); }

float safeAudio(float value, float fallback) {
    return max(value, fallback * 0.35);
}

// Squash FFT spikes — smoother than audioBoost for motion driving
float softAudio(float v) {
    return pow(clamp(v, 0.0, 1.0), 1.6);
}

float audioBoost(float v) {
    return clamp(pow(v, 0.75) * 1.35, 0.0, 1.0);
}

float smoothBass() {
    return (fft(0.01) + fft(0.02) + fft(0.03) + fft(0.04) +
            fft(0.05) + fft(0.06) + fft(0.07) + fft(0.08)) * 0.125;
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

float easeInOutCubic(float x) {
    if (x < 0.5) return 0.5 * pow(2.0 * x, 3.0);
    return 0.5 * pow(2.0 * (x - 1.0), 3.0) + 1.0;
}

vec3 coolor(float i) {
    float idx = mod(floor(i), 7.0);
    vec3 c0 = vec3(0.000, 0.071, 0.098);
    vec3 c1 = vec3(0.000, 0.478, 0.247);
    vec3 c2 = vec3(0.682, 0.125, 0.071);
    vec3 c3 = vec3(0.933, 0.608, 0.000);
    vec3 c4 = vec3(0.000, 0.114, 0.682);
    vec3 c5 = vec3(0.863, 0.506, 0.004);
    vec3 c6 = vec3(0.733, 0.243, 0.012);
    vec3 col = c0;
    col = mix(col, c1, step(0.5, idx) * (1.0 - step(1.5, idx)));
    col = mix(col, c2, step(1.5, idx) * (1.0 - step(2.5, idx)));
    col = mix(col, c3, step(2.5, idx) * (1.0 - step(3.5, idx)));
    col = mix(col, c4, step(3.5, idx) * (1.0 - step(4.5, idx)));
    col = mix(col, c5, step(4.5, idx) * (1.0 - step(5.5, idx)));
    col = mix(col, c6, step(5.5, idx) * (1.0 - step(6.5, idx)));
    return col;
}

// Nighttime-biased pairs: more black/green/blue, gold sparingly
vec2 pickColors(vec2 id) {
    float roll = hash21(id + 0.3);
    float a;
    if (roll < 0.38) {
        a = floor(hash21(id + 1.0) * 2.0);
    } else if (roll < 0.62) {
        a = 4.0;
    } else if (roll < 0.82) {
        a = mix(2.0, 6.0, step(0.5, hash21(id + 2.0)));
    } else {
        a = mix(3.0, 5.0, step(0.55, hash21(id + 3.0)));
    }
    float b = mod(a + 2.0 + floor(hash21(id + 7.9) * 2.0) + 1.0, 7.0);
    if (abs(b - a) < 0.5) b = mod(a + 3.0, 7.0);
    return vec2(a, b);
}

float inUnitSquare(vec2 p) {
    return smoothstep(0.502, 0.488, max(abs(p.x), abs(p.y)));
}

vec3 drawStripe(vec2 p, float rotA, vec3 c1, vec3 c2, float ratio, float bass, float high) {
    float inside = inUnitSquare(p);
    vec2 q = rot2(rotA) * p;
    float span = 0.2;
    float sizeRatio = easeInOutCubic(mod(ratio * 2.0, 1.0));
    float pulse = 0.12 + bass * STRIPE_BASS * 0.10;
    float stripeW = span * (1.0 + sin((sizeRatio + 0.35) * TAU) * pulse);
    float offsetRatio = easeInOutCubic(mod(ratio * 2.0, 1.0)) + floor(ratio * 2.0);
    float offset = offsetRatio * span * 2.0;
    offset += 0.012 * sin(iTime * 0.28 + q.y * 3.0);

    float lx = mod(q.x + offset + span * 8.0, span * 2.0);
    float stripe = smoothstep(stripeW + 0.012, stripeW - 0.012, lx);
    vec3 col = mix(c1, c2, stripe * inside);

    float stripeEdge = smoothstep(0.025, 0.0, abs(lx - stripeW));
    col += c2 * stripeEdge * high * SHIMMER_HIGH * 0.14 * inside;
    return col;
}

vec3 drawMovingRect(vec2 p, float rotA, vec3 c1, vec3 c2, float ratio, float bass) {
    float inside = inUnitSquare(p);
    vec2 q = rot2(rotA) * p;
    float rectSize = 0.5;
    float offsetRatio = easeInOutCubic(mod(ratio * 2.0, 1.0)) + floor(ratio * 2.0);
    float amp = 0.5 * (1.0 + bass * 0.12);
    float ox = clamp(offsetRatio, 0.0, 1.0) * amp;
    float oy = mod(clamp(offsetRatio, 1.0, 2.0), 1.0) * amp;

    vec2 cA = q - vec2(-0.25 + ox, -0.25 + oy);
    vec2 cB = -cA;
    float halfR = rectSize * 0.5;
    float rA = smoothstep(halfR + 0.008, halfR - 0.008, max(abs(cA.x), abs(cA.y)));
    float rB = smoothstep(halfR + 0.008, halfR - 0.008, max(abs(cB.x), abs(cB.y)));
    return mix(c1, c2, max(rA, rB) * inside);
}

vec3 drawCircleStripe(vec2 p, float rotA, vec3 c1, vec3 c2, float ratio, float bass) {
    float inside = inUnitSquare(p);
    vec2 q = rot2(rotA) * p;
    float span = 0.35;
    float sizeRatio = easeInOutCubic(mod(ratio * 2.0, 1.0));
    float r = 0.42 * (1.0 - sin(sizeRatio * TAU) * (0.08 + bass * 0.04));
    float offsetRatio = easeInOutCubic(mod(ratio * 2.0, 1.0)) + floor(ratio * 2.0);
    float offset = offsetRatio * span;

    float lx = mod(q.x + offset + span * 6.0, span) - span * 0.5;
    float d = length(vec2(lx, q.y * 0.85));
    float circ = smoothstep(r + 0.010, r - 0.010, d);
    return mix(c1, c2, circ * inside);
}

vec3 drawUnit(vec2 p, vec2 id, float ratio, float bass, float high) {
    float rotIdx = floor(hash21(id + 1.1) * 4.0);
    float rotA = rotIdx * TAU * 0.25;
    float mode = floor(hash21(id + 2.2) * 4.0);
    vec2 ci = pickColors(id);
    vec3 c1 = coolor(ci.x);
    vec3 c2 = coolor(ci.y);

    vec3 col;
    if (mode < 1.0) {
        col = drawMovingRect(p, rotA * 0.5, c1, c2, ratio, bass);
    } else if (mode < 2.0) {
        col = drawStripe(p, rotA, c1, c2, ratio, bass, high);
    } else if (mode < 3.0) {
        col = drawMovingRect(p, rotA, c1, c2, ratio, bass);
    } else {
        col = drawCircleStripe(p, rotA, c1, c2, ratio, bass);
    }

    float edge = smoothstep(0.46, 0.495, max(abs(p.x), abs(p.y)));
    col += coolor(3.0) * edge * high * SHIMMER_HIGH * 0.08;
    return col;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = centeredUV(fragCoord);

    float bassRaw = safeAudio(smoothBass(), fallbackBass());
    float bass = softAudio(bassRaw);
    float high = softAudio(safeAudio(getHigh(), fallbackHigh())) * 0.85;

    vec3 bg = coolor(0.0);
    float scale = UNITS * 1.05;
    vec2 grid = uv * scale;

    vec2 idPre = floor(grid + UNITS * 0.5);
    if (mod(idPre.y, 2.0) > 0.5) {
        grid.x += 0.5;
    }

    vec2 id = floor(grid + UNITS * 0.5);
    vec2 f = fract(grid + UNITS * 0.5) - 0.5;

    // Steady eased cycle — row sync only, no audio ratio jumps
    float rowRatio = fract(iTime / CYCLE + id.y * 0.08 + id.x * 0.02);

    vec3 col = drawUnit(f, id, rowRatio, bass, high);

    float gap = smoothstep(0.495, 0.502, max(abs(f.x), abs(f.y)));
    col = mix(col, bg, gap * 0.92);

    // Nighttime wash + vignette
    col = mix(col, bg, 0.06);
    col *= smoothstep(1.40, 0.42, length(uv));
    col = pow(max(col, 0.0), vec3(1.10));
    col = min(col, vec3(0.75));

    fragColor = vec4(col, 1.0);
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

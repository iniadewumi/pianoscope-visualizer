# Inigo Quilez — SDF, Noise, Warp, Color

**Keywords:** iquilez, iq, sdf, fbm, warp, voronoise, palette, domain repetition, smoothstep

**Use when:** Line/band geometry, organic mudcloth, color variation, tiled SDF motifs.

**Index:** https://iquilezles.org/articles/

---

## Article → technique map

| Article | URL | Use for |
|---------|-----|---------|
| 2D Distance Functions | https://iquilezles.org/articles/distfunctions2d/ | Lines, boxes, triangles, hexagons |
| fBM | https://iquilezles.org/articles/fbm/ | Layered noise, mudcloth atmosphere |
| Domain Warping | https://iquilezles.org/articles/warp/ | Cloth breathing, organic flow |
| Voronoise | https://iquilezles.org/articles/voronoise/ | Cellular texture without grid seams |
| Voronoi Lines | https://iquilezles.org/articles/voronoilines/ | Uniform-width crack/part lines |
| Palettes | https://iquilezles.org/articles/palettes/ | Kente band colors from scalar |
| Smoothsteps | https://iquilezles.org/articles/smoothsteps/ | Line AA, C2 noise stitching |
| SDF Domain Repetition | https://iquilezles.org/articles/sdfrepetition/ | Infinite tiles, mirrored rows |
| Filtering | https://iquilezles.org/articles/filtering/ | Projection-distance readability |

---

## HOW-TO: Render a 2D SDF line (bands, braids, marks)

From IQ 2D SDFs — segment distance + smoothstep:

```glsl
float sdSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

float line(vec2 p, vec2 a, vec2 b, float w) {
    return smoothstep(w, w * 0.5, sdSegment(p, a, b));
}
```

**Use:** Kente band edges, cornrow strands, mudcloth zigzags.

Circle ring (settlement, micro-rings):

```glsl
float sdCircle(vec2 p, float r) {
    return length(p) - r;
}
float ring = smoothstep(w, 0.0, abs(sdCircle(p, r)));
```

---

## HOW-TO: Fill + stroke from SDF (settlement architecture)

Apollonian pattern — **required for readable compounds**:

```glsl
float sdfFill(float d, float feather) {
    return smoothstep(feather, -feather, -d);
}

float sdfStroke(float d, float width, float feather) {
    return smoothstep(width + feather, width - feather, abs(d));
}
```

**Do not** use edge-only recursive folds without fill — produces dot grids. See [`07-anti-patterns-and-failures.md`](07-anti-patterns-and-failures.md).

---

## HOW-TO: Value noise + fBM (mudcloth base)

```glsl
float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);  // cubic smoothstep
    return mix(
        mix(hash21(i), hash21(i + vec2(1,0)), u.x),
        mix(hash21(i + vec2(0,1)), hash21(i + vec2(1,1)), u.x),
        u.y
    );
}

float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 5; i++) {
        v += a * noise(p);
        p = p * 2.02 + vec2(1.7, 9.2);
        a *= 0.5;
    }
    return v;
}
```

**Hurst / roughness:** More octaves + higher initial amplitude → rougher mudcloth. Fewer octaves → smoother “breathing cloth.”

For C2 continuity between octaves, use **quintic** interpolation in noise (IQ smoothsteps / morenoise articles).

---

## HOW-TO: Single-level domain warp (cloth breath)

```glsl
vec2 warp(vec2 uv, float strength) {
    return uv + vec2(
        fbm(uv * 2.4 + vec2(iTime * 0.04, 0.0)) - 0.5,
        fbm(uv * 2.4 + vec2(4.0, iTime * 0.03)) - 0.5
    ) * strength;
}
```

**Shipped:** `warpCloth()` in Mudcloth shader — mid-driven strength.

---

## HOW-TO: Double domain warp (IQ marble / deep organic)

From https://iquilezles.org/articles/warp/

```glsl
float pattern(vec2 p) {
    vec2 q = vec2(fbm(p), fbm(p + vec2(5.2, 1.3)));
    vec2 r = vec2(fbm(p + 4.0 * q + vec2(1.7, 9.2)), fbm(p + 4.0 * q + vec2(8.3, 2.8)));
    return fbm(p + 4.0 * r);
}
```

Apply to mudcloth blooms or mark placement UVs — not crisp Kente grids (warps destroy weave readability).

---

## HOW-TO: Thresholded fBm bloom

```glsl
float field = fbm(uv * 1.6 + t) * 0.65 + fbm(uv * 3.2 - t) * 0.35;
float thresh = 0.52 - bass * BLOOM_GAIN;
float bloom = smoothstep(thresh, thresh - 0.12, field);
```

Bass lowers threshold → larger organic regions open.

---

## HOW-TO: Voronoise (organic cells, less grid)

Parameters: `u` = jitter (0=noise grid, 1=Voronoi), `v` = metric blend.

Full implementation: https://iquilezles.org/articles/voronoise/

**When:** Mudcloth bloom regions that shouldn’t look like square tiles. Blend with staggered `fract` marks on top.

---

## HOW-TO: Uniform crack lines (avoid naive F2−F1)

Naive Voronoi edge `c.y - c.x` gives ** uneven line widths**.

Use IQ Voronoi distance algorithm: https://iquilezles.org/articles/voronoilines/

**When:** Mudcloth fractures, cornrow parting lines between braid groups.

---

## HOW-TO: Cosine palette (Kente color rhythm)

From https://iquilezles.org/articles/palettes/

```glsl
#define TAU 6.28318530718

vec3 palette(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
    return a + b * cos(TAU * (c * t + d));
}

// Example: map cell phase + audio to color
vec3 col = palette(
    fract(cellId * 0.618 + bass * 0.2),
    vec3(0.13, 0.06, 0.25),  // base
    vec3(0.35, 0.25, 0.15),  // amplitude
    vec3(1.0, 1.0, 0.5),     // frequency
    vec3(0.0, 0.15, 0.25)    // phase
);
```

---

## HOW-TO: Domain repetition (SDF-native tiling)

Basic tile (equivalent to `fract` for SDFs):

```glsl
vec2 id = round(p / s);
vec2 r = p - s * id;
float d = motifSDF(r);  // draw in local cell
```

**Mirrored repetition** (fixes discontinuities when cells vary):

```glsl
vec2 r = p - s * id;
vec2 m = vec2(
    (mod(id.x, 2.0) < 0.5) ? r.x : -r.x,
    (mod(id.y, 2.0) < 0.5) ? r.y : -r.y
);
```

**When:** Cornrow alternate-row mirroring; Kente cells with audio-driven size changes (check 2×2 neighbors if cells grow large).

Full article: https://iquilezles.org/articles/sdfrepetition/

---

## HOW-TO: Smoothstep variants for lines

| Variant | When |
|---------|------|
| Cubic `x*x*(3-2*x)` | Default line AA |
| Quintic (IQ) | fBm octave stitching (C2) |
| Piecewise with exponent n | Sharper/softer thread shimmer without changing width |

Compare visually: https://www.shadertoy.com/view/st2BRd

---

## HOW-TO: Projection-friendly AA

For ~15 ft projection, prefer **fixed feather** in UV space (~0.003–0.01) over `fwidth()` on WebGL 1 paths.

When WebGL 2 available: `smoothstep(w + fwidth(d), w - fwidth(d), d)`.

See IQ filtering article for adaptive supersampling (usually overkill for v1 PIANOSCOPE).

---

## Technique → genre quick map

| Technique | Mudcloth | Kente | Cornrow | Afrotech |
|-----------|----------|-------|---------|----------|
| fBM + threshold | ●●● | ○ | ○ | atmosphere |
| Domain warp | ●●● | ○ | ○ | ○ |
| fract / tile grid | marks | ●●● | rows | ○ |
| sdSegment lines | marks | bands | ●●● | paths |
| Cosine palette | earth mix | ●●● | gold mix | cyan/violet |
| Fill + stroke | ○ | ○ | ○ | ●●● |
| Voronoise | optional | ○ | ○ | ○ |

●●● = primary · ○ = secondary / avoid

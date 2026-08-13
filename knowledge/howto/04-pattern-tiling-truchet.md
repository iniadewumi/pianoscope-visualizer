# Pattern Tiling & Truchet (Book of Shaders Ch. 09)

**Keywords:** fract, tiling, grid, brick, truchet, stagger, cell, weave, book of shaders

**Use when:** Kente loom, mudcloth cells, woven grids, syncopated breaks, any repeating textile logic.

**Canonical URL:** https://thebookofshaders.com/09/

---

## Core idea

Fragment shaders evaluate **per pixel**. Repeating a pattern N×M times costs the same as drawing once — scale UV space, draw in 0–1, repeat with `fract()`.

```glsl
vec2 st = fragCoord.xy / iResolution.xy;
st *= 3.0;           // 3×3 repetitions
vec2 cell = floor(st);
vec2 f = fract(st);    // local 0..1 within cell
// draw shape using f, not st
```

---

## HOW-TO: Basic tile grid

```glsl
vec2 tileUV(vec2 uv, float columns, float rows, out vec2 id, out vec2 f) {
    vec2 scaled = uv * vec2(columns, rows);
    id = floor(scaled);
    f = fract(scaled) - 0.5;  // centered in cell
    return f;
}
```

Use `id` for per-cell hash / color / motif selection.

---

## HOW-TO: Brick / stagger offset (mudcloth rows)

Every other row shifts half a cell — breaks wallpaper symmetry.

```glsl
vec2 id = floor(cell);
vec2 f = fract(cell) - 0.5;

if (mod(id.y, 2.0) > 0.5) {
    f.x += 0.5;
    id.x += 0.5;
}
```

**Shipped example:** `clothMarks()` in `js/test-shaders.js` (Mudcloth Bloom).

---

## HOW-TO: Know even/odd row or column

Book of Shaders uses `mod()` or `step()`:

```glsl
// odd row → 1.0, even → 0.0
float oddRow = step(1.0, mod(id.y, 2.0));
f.x += oddRow * 0.5;  // half-unit brick offset
```

Prefer `step()` over ternary when possible — often faster on GPU.

---

## HOW-TO: Truchet tile (rotate motif by cell parity)

One design element, four orientations — complex infinite patterns from minimal code.

```glsl
float truchetAngle(vec2 id) {
    return mod(id.x + id.y, 2.0) * 1.5707963;  // 0 or PI/2
}

vec2 rotateCell(vec2 f, float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c) * f;
}

// In cell:
float a = truchetAngle(id);
vec2 rf = rotateCell(f, a);
// draw arc or diagonal using rf
```

**Repo reference:** `"Paper Gear"` in `js/shaders.js` — Truchet + kaleidoscope. Steal **cell parity rotation**, not full kaleidoscope dot rendering.

---

## HOW-TO: Weave over/under mask (Kente)

Alternating which band wins at each crossing:

```glsl
float overUnder = step(0.5, fract(floor(uv.x * bandX) + floor(uv.y * bandY)));
float vertical   = stripe(uv.x + midOffset, ...);
float horizontal = stripe(uv.y + bassOffset, ...);
float weave = mix(vertical, horizontal, overUnder);
```

Add `sin()` wobble on offsets for “thread tension” (mid/bass driven).

---

## HOW-TO: Stripe band

```glsl
float stripe(float x, float count, float width) {
    float f = fract(x * count);
    return smoothstep(width, width - 0.01, abs(f - 0.5));
}
```

---

## HOW-TO: Per-cell transform (scale/rotate inside cell)

Each subdivision is a mini coordinate system:

```glsl
f = (f - 0.5) * 2.0;              // -1..1
f = rot(hash21(id) * TAU) * f;    // random rotation per cell
f *= 0.8;                         // inset motif
```

---

## HOW-TO: Syncopated pattern break

Don’t fill every cell — rhythm via emptiness:

```glsl
float seed = hash21(id);
float empty = step(0.12, seed);   // ~12% cells empty
float breakGap = step(0.15, hash21(f * 3.0 + seed * 11.0));
return motif * empty * breakGap;
```

---

## Anti-patterns (patterns chapter)

| Avoid | Why | Instead |
|-------|-----|---------|
| Perfect seamless wallpaper | Reads digital, not hand-made | Stagger, hash jitter, empty cells |
| Same motif every cell | Tourist textile | Hash-driven variant marks |
| `fract(st)` when st already 0–1 | No-op | Scale first: `st *= N` then `fract` |

---

## PIANOSCOPE mapping

| Genre | Primary Ch. 09 techniques |
|-------|---------------------------|
| Kente Loom | `fract` grid, weave mask, stripes, Truchet accents |
| Mudcloth | Brick offset, per-cell marks, syncopated gaps |
| Cornrow | 1D repeat along x (braid rows), mirror alternate rows |
| Afrotech | Radial repetition (different chapter) — use fill+stroke, not edge-only recursion |

Organic noise layers → [`05-iquilez-sdf-noise-warp.md`](05-iquilez-sdf-noise-warp.md).

Examples gallery: https://thebookofshaders.com/examples/?chapter=09

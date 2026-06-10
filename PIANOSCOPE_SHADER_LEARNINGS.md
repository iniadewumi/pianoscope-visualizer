# PIANOSCOPE Shader Learnings (Agent Notes)

**Read this after [`cursor inst.md`](cursor inst.md).** The brief defines *what* to build and *why*. This doc captures *what we tried*, *what failed*, *what worked*, and *how to avoid repeating mistakes* — based on implementing the first PIANOSCOPE shader in this repo.

**Current state (as of polish pass):**
- [`js/test-shaders.js`](js/test-shaders.js) — contains `"PIANOSCOPE Afrotech Fractal Settlement"` (**polished v3**, ~90% of target — ready for projection/mic testing before next genre)
- [`js/shadertoy-converter.js`](js/shadertoy-converter.js) — imports `TEST_SHADERS` first in `SAMPLE_SHADERS`

---

## Quick start for a future agent

1. Read [`cursor inst.md`](cursor inst.md) sections: Repository context, Audio channel layout, GLSL compatibility, Recommended first target.
2. Read this doc's **Iteration postmortem** and **Anti-patterns** before writing GLSL.
3. Study shaders in [`js/shaders.js`](js/shaders.js) listed under **Repo shaders worth stealing from**.
4. Add new entries to `TEST_SHADERS` in [`js/test-shaders.js`](js/test-shaders.js) — do not edit `shaders.js` / `shaders2.js`.
5. Serve over HTTP (`npx serve .`), not `file://` if modules fail.
6. Test with mic on **and** off (fallback motion must work).
7. Get visual approval before building the next genre shader.

---

## Iteration postmortem (Afrotech Fractal Settlement)

We shipped three distinct approaches, then one polish pass on the winner. **Do not retry from scratch** unless abandoning radial symmetry entirely.

### v1 — Thin rings + scattered nodes (failed)

**Approach:** Nested `ring()` SDFs, dwelling discs on fixed radii, hash-based sparks, subtle fBm background.

**What the user saw:** Two big concentric teal rings with small scattered dots. Read as "orbital HUD" or "cyan mandala," not a settlement.

**Why it failed:**
- Rings dominated; dwellings were decoration, not structure.
- Sparks used random hash positions across the screen, not node centers.
- Branch paths were undefined / too faint.
- "Recursive" was only radial repetition — no self-similar nesting.
- Nested mini-settlements inside nodes looked like V-shaped artifacts.

**Lesson:** Listing layers in a plan (rings, cells, paths, sparks) is not enough. Each layer needs a **concrete geometric rule** and **fill/stroke weight** that reads from distance.

---

### v2 — Pianoscope-style recursive domain fold (failed)

**Approach:** Adapted kaleidoscope + `mod2` iteration from repo shader `"Pianoscope"`. Edge-only rendering with glow. Bokeh grid for highs.

**What the user saw:** Purple circular field filled with a **black dot grid**, six cyan radial spokes, dark center. Kaleidoscope halftone — not architecture.

**Why it failed:**
- Recursive `mod2` + **edge-only** `smoothstep(dEdge)` produces hundreds of tiny ring edges → **dot matrix**.
- `fwidth()` / thin-edge glow amplifies the problem.
- Bokeh sparks on `floor(uv * 5.0)` grid added more dots on top of dots.
- Sector fold (6-way) was visually strong but the compound geometry underneath was invisible.

**Lesson:** Do **not** port Pianoscope's `df()` loop for PIANOSCOPE settlement shaders unless you also **fill** regions. Edge-only recursion ≠ readable compounds.

---

### v3 — Apollonian fill/stroke + explicit ring layers (base, ~75%)

**Approach:** Borrowed `sdfFill` / `sdfStroke` pattern from repo shader `"Apollonian Gasket"`. Three explicit compound rings with filled dwelling discs, thick walls, branch arcs, mini-rings inside dwellings, courtyard darken, radial spokes.

**What the user saw:** Concentric cyan compound rings, filled purple/cyan dwelling nodes, connecting paths forming star/network patterns, dark central courtyard, Afrotech palette. **Readable and projection-friendly.**

**Why it worked:**
- **Fill + stroke** makes walls and dwellings visible at a glance.
- **Explicit geometry** (3 rings, 6/8/10 nodes) instead of emergent dot fields.
- Sparks gated to **dwelling centers** only.
- Central courtyard via `smoothstep` on radius — strong negative space.
- Branch paths use segment distance + inset midpoint (pathways, not center spokes).

**User feedback at v3:** "Is this the final version?" — structurally yes, aesthetically ~75%. Mandala-perfect symmetry, weak audio feel, minimal atmosphere, missing rectangular compounds.

**Decision:** **Keep modifying v3** — do not rewrite from scratch. Foundation was correct; gaps were polish, not broken architecture.

---

### v3 polish — asymmetry, audio, atmosphere (current, ~90%)

**Approach:** Same Apollonian fill/stroke core. Added per-layer skew, node jitter, gated ring walls, rectangular enclosures on layers 1+3, third micro-ring, `audioBoost()`, fBm smoke, higher `#define` gains.

**What changed:**

| Area | Implementation |
|------|----------------|
| **Asymmetry** | Per-layer UV skew; hash jitter on node angles; `ringWallGated()` breaks one wall section per ring; staggered node counts (6 / 9 / 11); spoke offset + per-sector brightness variance |
| **Audio** | `RING_BASS_GAIN 0.30`, `MID_ROT_GAIN 0.55`, `PATH_MID_GAIN 0.70`, `HIGH_SPARK_GAIN 0.65`; `audioBoost()` = `pow(v, 0.75) * 1.35` to counter `smoothingTimeConstant = 0.8` |
| **Atmosphere** | `fbm()` smoke at ~0.07 amplitude — violet/cyan mix, no particles |
| **Brief gaps** | `rectCompound()` on inner + outer rings (layers 0.0 and 1.0); micro-ring inside mini-dwellings (3 fractal levels) |
| **Bug fixed** | `electric` vec3 must be defined in `mainImage` if used outside `settlementLayer` |

**Still may need live tuning:**
- Audio gains after real music test (push higher if still sluggish, lower if too aggressive)
- Fullscreen / ~15 ft projection pass
- Rectangular compounds may be subtle on some displays — thicken `rectCompound` stroke if needed

**When to call it "final":** After mic + fullscreen + distance test passes checklist below. Then move to **Afrohouse Mudcloth Bloom**.

---

## Design principles (validated)

| Principle | Evidence |
|-----------|----------|
| **Structure before decoration** | v1 failed because rings looked decorative; v3 works because fills define compounds first. |
| **Fill + stroke, not edge-only** | v2 dot grid came from edge-only recursion. Apollonian-style fill/stroke fixed it. |
| **Dark negative space ~40–60%** | Central courtyard + vignette made projection readability jump. |
| **Sparks at semantic positions** | Sparks on dwelling centers OK; sparks on random/hash grids = noise. |
| **Paths connect neighbors** | Arcs between adjacent nodes on same ring read as pathways; radial spokes from origin read as HUD. |
| **Start generous on audio gains, tune down** | Polish pass used ~1.7× gain bump + `audioBoost()` — required because FFT smoothing is 0.8. |
| **One shader at a time** | Brief says don't build all four until first compiles and looks good — correct. |
| **Imperfect symmetry = more settlement, less mandala** | Gated walls, node jitter, and per-layer skew broke mandala-perfect read without losing structure. |
| **Modify the winner, don't restart** | v3 screenshot was approved structurally; polish took ~30 min vs. risking another v1/v2 failure from scratch. |
| **Explicit beats emergent for settlements** | 3 hand-authored ring layers outperform recursive `mod2` for cultural readability. |

---

## Anti-patterns (do not repeat)

| Anti-pattern | Symptom | Alternative |
|--------------|---------|-------------|
| Edge-only recursive `mod2` | Dot grid / halftone kaleidoscope | Explicit layers with fill, or filled Apollonian cells |
| `fwidth()` / `aaLine` on WebGL 1 paths | Compile failures or inconsistent AA | `smoothstep` on fixed feather (~0.003 UV) |
| `const float` at global scope | WebGL 1 compile errors | `#define` tuning constants |
| `out` parameters in helper functions | WebGL 1 failures | Return `vec3` packs (e.g. `vec3(field, seed.x, seed.y)`) |
| `bool` in GLSL helpers | WebGL 1 issues | `float` flags (`0.0` / `1.0`) |
| Bokeh/grid sparks (`floor(uv * N)`) | Visual noise, fights composition | Sparks only at known node centers |
| Copying `y = 0.25` audio from old `shaders.js` | Wrong frequency data | Always `texture(iChannel0, vec2(x, 0.0))` |
| Generic radial lasers from center | Sci-fi HUD | Neighbor branch arcs with inset midpoint |
| Building all 4 genre shaders at once | None compile/look good | One shader → visual approval → next |
| Opening `index.html` as `file://` | ES module import failures (browser-dependent) | `npx serve .` → `http://localhost:...` |
| Rewriting from scratch after v3 works | Lose fill/stroke foundation, risk dot grid again | Polish pass on existing `settlementLayer` architecture |
| Undefined palette vars in `mainImage` | Shader compile fail | Define `electric`, `deepCyan`, etc. in scope where used |

---

## Repo shaders worth stealing from

Study these in [`js/shaders.js`](js/shaders.js) before inventing new techniques:

| Shader key | Steal for | Key technique |
|------------|-----------|---------------|
| **Apollonian Gasket** | Settlement rings, nested circles | `sdfFill`, `sdfStroke`, circle packing, size-based detail levels — **primary pattern for Afrotech** |
| **Pianoscope** | Audio-reactive motion, kaleidoscope | `getBass()`/`getMids()`, `smoothKaleidoscope`, `mod2` fold — **use motion only, not edge-only rendering** |
| **Mandala** | Polar repetition, iterative scale | `mandala_df` loop: `mod2` + `p *= 1.5` — good for depth, needs fill compositing |
| **Logarithmic Spirals** | Smith disk transform, spiral arms | `toSmith`/`fromSmith`, `modPolar` — pathway / spiral motion |
| **Fractal Sounders** | Audio-averaged motion | `lowAverage()` / `highAverage()` over FFT bins — try if `audioBoost()` still feels dead |
| **Colorful Nebula** | Beat detection smoothing | `smoothstep` thresholds + `hash(beatTime)` — if you need punchier reactivity |
| **Multiversal Web** | Sparse highlights | `Bokeh()` — only at named positions, not full-screen grid |

**Do not blindly copy:** Many `shaders.js` entries sample `iChannel0` at `y = 0.25` (classic Shadertoy layout). This repo uses **256×1 at `y = 0.0` only**.

---

## Audio pipeline (critical)

From [`js/visualizer-with-shadertoy.js`](js/visualizer-with-shadertoy.js):

| Setting | Value | Implication for shaders |
|---------|-------|-------------------------|
| `analyser.fftSize` | 1024 | 512 bins, first 256 uploaded |
| Texture | 256×1 luminance | Sample `y = 0.0` always |
| `audioSensitivity` | 1.5 | Values already amplified in JS |
| `smoothingTimeConstant` | **0.8** | **Heavy lag** — transients feel mushy; boost gains or use averaged bands |

**Standard helpers** (include in every `TEST_SHADERS` entry):

```glsl
float fft(float x) {
    return texture(iChannel0, vec2(clamp(x, 0.0, 1.0), 0.0)).x;
}

float getBass() {
    return (fft(0.01) + fft(0.03) + fft(0.05) + fft(0.07)) * 0.25;
}
// ... getMid, getHigh ...

float safeAudio(float value, float fallback) {
    return max(value, fallback * 0.35);
}
```

**Polish-pass addition** (works well for Afrotech):

```glsl
float audioBoost(float v) {
    return clamp(pow(v, 0.75) * 1.35, 0.0, 1.0);
}
// Usage: float bass = audioBoost(safeAudio(getBass(), fallbackBass()));
```

**Fallback sines** are required — shader must animate when mic is denied or silent.

**Tuning tip:** Start `#define` gains ~1.5–2× higher than expected, test with mic, then reduce. Current polished values: `RING_BASS_GAIN 0.30`, `MID_ROT_GAIN 0.55`, `PATH_MID_GAIN 0.70`, `HIGH_SPARK_GAIN 0.65`.

---

## GLSL / WebGL gotchas (hit in practice)

| Issue | Fix |
|-------|-----|
| Write `texture()` in source | Converter rewrites to `texture2D` for WebGL 1 |
| No `uniform`, `precision`, `void main()` | Injected by `shadertoy-converter.js` |
| `iFrame` is `float`, not `int` | Don't cast loops on `iFrame` as int index |
| Loop bounds must be constant | `for (int i = 0; i < 12; i++)` + `if (fi >= count) break;` |
| Template string backticks in JS | Escape `` ` `` inside shader strings |
| `inout` in helpers | Works, but prefer simple returns when possible |
| Duplicate `settlementLayer` call without assignment | First layer silently skipped — always `col = settlementLayer(...)` |

---

## What "good" looks like for Afrotech Fractal Settlement

**Pass:**
- Dark central courtyard (visible negative space)
- Thick compound ring **walls** (cyan strokes), with **gated openings** on at least one wall per ring
- **Filled** dwelling discs on rings
- Branch paths between **neighbors** on same ring
- Three fractal levels: dwelling → mini-ring → micro-ring (or rectangular compound on alternate layers)
- Violet/black/cyan palette, light fBm atmosphere, no white blowout
- Slow drift when silent; visible bass/mid/high difference on music
- Slightly **imperfect** symmetry (jitter, skew) — settlement, not perfect mandala

**Fail:**
- Two lonely concentric rings
- Dot grid / halftone kaleidoscope
- Random spark field
- Neon laser spokes from center only
- All-white flashes / strobe
- Tiny detail that vanishes on a projector

---

## Current architecture (polished v3 — reference)

The shipped shader in `test-shaders.js` uses this structure — **reuse this pattern** for structurally similar shaders; **do not reuse** for Mudcloth/Kente/Cornrow (different aesthetics):

```
mainImage
├── centeredUV
├── audioBoost(safeAudio(bass, mid, high))
├── global slow rotation (mid-influenced)
├── background: blackBlue + fbm smoke (~0.07) + radial violet tint
├── settlementLayer × 3 (ringR, nodeCount, layer, unique rot)
│   ├── per-layer UV skew (asymmetry)
│   ├── ringWallGated (compound perimeter with pathway break)
│   ├── outerWall gated (faint secondary ring, bass-expanded)
│   └── for each node on ring:
│       ├── filledDisc (dwelling, bass-scaled radius)
│       ├── rectCompound (layers 0 + 2 only — rectangular enclosure)
│       ├── miniRing + miniDwell + microRing (3 fractal levels)
│       ├── branchPath to previous neighbor (inset varies per edge)
│       └── spark at center (high-gated, hash flicker)
├── outerBound ring (bass-brightened)
├── 6 radial spokes (offset angle, per-sector variance, mid + high)
├── courtyard mask (darken center)
├── vignette
└── sqrt tone map, cap at 0.82
```

**Key functions:**

```glsl
float sdfFill(float d, float feather);
float sdfStroke(float d, float width, float feather);
float ringWall(vec2 p, float radius, float width);
float ringWallGated(vec2 p, float radius, float width, float gateAngle, float gateWidth);
float filledDisc(vec2 p, vec2 center, float radius);
float branchPath(vec2 p, vec2 a, vec2 b, float inset, float width);
float rectCompound(vec2 uv, vec2 center, float angle, float dwellR);
float audioBoost(float v);
float fbm(vec2 p);
vec3 settlementLayer(...);  // returns accumulated col
```

**Layer config (current):**

| Layer | ringR (× scale) | Nodes | Rotation | Rect compounds |
|-------|-----------------|-------|----------|----------------|
| 0 (inner) | 0.24 | 6 | `+t*0.3 + mid*MID_ROT_GAIN` | Yes |
| 1 (mid) | 0.46 | 9 | `-t*0.22 + mid*1.3 + 0.4` | No |
| 2 (outer) | 0.70 | 11 | `+t*0.14 - mid*0.9 + 1.1` | Yes |

---

## Recommended next steps

### Lock Afrotech (user testing)

1. Hard-refresh → `setShaderByName("PIANOSCOPE Afrotech Fractal Settlement")`
2. Run testing checklist below with mic on real music
3. Tune `#define` gains in `test-shaders.js` if bass/mid/high still feel samey
4. Fullscreen from ~15 ft — thicken walls/rects if detail vanishes

### Next genre shader (per brief order)

1. **PIANOSCOPE Afrohouse Mudcloth Bloom** — organic, fBm, thresholded noise, earthy palette. **Opposite aesthetic** to Afrotech; do **not** reuse ring-mandala / `settlementLayer` structure.
2. **PIANOSCOPE Amapiano Kente Loom** — woven bands, `fract()`, grid — study brief §1027.
3. **PIANOSCOPE 3-Step Cornrow Curves** — flowing sine braids, triplet pulse — study brief §1094.

---

## Testing checklist (practical)

```
[ ] Shader compiles (browser console clean)
[ ] Selectable via picker and setShaderByName("PIANOSCOPE Afrotech Fractal Settlement")
[ ] Animates with NO mic (fallback motion)
[ ] Bass visibly scales rings/dwellings on kick
[ ] Mid visibly rotates layers / brightens paths
[ ] High adds sparks at dwellings (not random grid)
[ ] Gated ring openings visible (pathway breaks in walls)
[ ] Rectangular enclosures visible on inner + outer rings
[ ] Readable fullscreen
[ ] Readable from ~15 ft (silhouette, not fine dots)
[ ] Dark negative space preserved
[ ] No strobe / white flash
[ ] Runs 30+ min without visual fatigue
```

**Console helpers** (after page load):

```js
setShaderByName("PIANOSCOPE Afrotech Fractal Settlement")
getCurrentShaderInfo()
```

---

## File map

| File | Role |
|------|------|
| [`cursor inst.md`](cursor inst.md) | Creative brief, cultural context, genre specs |
| **This file** | Implementation learnings, anti-patterns, iteration history |
| [`js/test-shaders.js`](js/test-shaders.js) | All new PIANOSCOPE shaders (`TEST_SHADERS`) |
| [`js/shadertoy-converter.js`](js/shadertoy-converter.js) | `TEST_SHADERS` spread first in picker |
| [`js/shaders.js`](js/shaders.js) | Reference library — read, don't edit |
| [`js/visualizer-with-shadertoy.js`](js/visualizer-with-shadertoy.js) | Audio upload, WebGL runtime |

---

## One-paragraph summary for agents

PIANOSCOPE shaders live as GLSL strings in `js/test-shaders.js`, wired first in `shadertoy-converter.js`. Sample `iChannel0` at **`y = 0.0` only**. Afrotech Fractal Settlement took three iterations plus a polish pass: thin rings failed (HUD-like), Pianoscope edge-only recursion failed (dot grid kaleidoscope), Apollonian **fill+stroke** with explicit compound ring layers succeeded, then polish added asymmetry, `audioBoost()`, fBm atmosphere, rectangular compounds, and gated walls (~90%). Build **filled architecture** with dark courtyards, not edge-only recursion or random spark grids. Steal fill/stroke from `Apollonian Gasket` and motion helpers from `Pianoscope` — never copy their audio sampling rows or edge-only `df()` rendering. Start audio gains high + use `audioBoost()` because FFT smoothing is 0.8. **Modify the winning architecture; don't restart from scratch.** Get visual sign-off on one shader before building the next genre (Mudcloth Bloom is next — opposite aesthetic, new structure).

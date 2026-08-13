# Steal From the Shader Library

**Keywords:** shaders.js, reference, apollonian, pianoscope, mandala, truchet, steal, borrow

**Use when:** Need a working technique; don’t reinvent fill/stroke, audio motion, or Truchet.

**Location:** `js/shaders.js` (primary), `js/shaders2.js` (secondary).

**Rule:** Study and inline minimal helpers into `test-shaders.js`. Do **not** add PIANOSCOPE shaders to the library files.

---

## Priority table

| Shader key | Steal for | Key technique | Audio sampling |
|------------|-----------|---------------|----------------|
| **Apollonian Gasket** | Settlement rings, nested circles | `sdfFill`, `sdfStroke`, circle packing | ⚠️ May use wrong row — rewrite helpers |
| **Pianoscope** | Audio-reactive motion, kaleidoscope | `getBass`/`getMids`, `smoothKaleidoscope`, `mod2` | ⚠️ Wrong row — motion only, not edge-only render |
| **Mandala** | Polar repetition, iterative scale | `mandala_df`: `mod2` + scale loop | ⚠️ Needs fill compositing if ported |
| **Logarithmic Spirals** | Pathway / spiral motion | `toSmith`/`fromSmith`, `modPolar` | ⚠️ |
| **Fractal Sounders** | Sluggish audio feel | `lowAverage()` / `highAverage()` | ⚠️ |
| **Colorful Nebula** | Punchier beats | `smoothstep` + `hash(beatTime)` | ⚠️ |
| **Multiversal Web** | Sparse highlights | `Bokeh()` — named positions only | ⚠️ |
| **Paper Gear** | Truchet + patterns | `truchet_df`, cell parity, kaleidoscope | ⚠️ — pattern logic only |

---

## What to steal vs what to avoid

### Apollonian Gasket → Afrotech settlement

**Copy:** Fill/stroke SDF compositing, nested circle hierarchy.

```glsl
// Pattern to extract:
float sdfFill(float d, float feather);
float sdfStroke(float d, float width, float feather);
```

**Don’t copy:** Entire scene graph — PIANOSCOPE uses explicit 3-layer settlement, not emergent packing.

### Pianoscope → motion only

**Copy:** Bass/mid extraction pattern (rewrite for `y=0.0`), slow rotation, kaleidoscope fold for **optional** subtle symmetry.

**Don’t copy:** Edge-only `df()` loop rendering — causes dot grid (v2 failure).

### Paper Gear → Kente / pattern work

**Copy:** Truchet distance field, hash-based cell variation:

```
grep "truchet" js/shaders.js
```

**Don’t copy:** Full kaleidoscope + halftone aesthetic — too busy for projection.

### Mandala → depth via iteration

**Copy:** `mod2` polar fold for layered depth.

**Combine with:** Fill compositing from Apollonian — never edge-only.

---

## HOW-TO: Safely borrow from library shaders

1. Find shader in picker or grep `js/shaders.js` for key name.
2. **Rewrite all** `texture(iChannel0, vec2(x, 0.25))` → `y = 0.0`.
3. Extract minimal helper functions — don’t paste entire shader.
4. Test compile in isolation via `ShaderConverter.convertShaderToyToWebGL`.
5. Check [`07-anti-patterns-and-failures.md`](07-anti-patterns-and-failures.md) before using recursion/kaleidoscope.

---

## Fixing library shaders (not PIANOSCOPE work)

When a library shader fails to compile:

1. Read `PIANOSCOPE_SHADER_LEARNINGS.md` § Porting Shadertoy library shaders
2. Map error line to **full** compiled source (preamble offset)
3. Common fix: `vec2(3.0) - 2.0 * diff` in noise smoothsteps
4. Common fix: `uniform vec3 iResolution` in preamble

---

## Grep shortcuts

```bash
# Truchet implementations
rg -i "truchet" js/shaders.js

# SDF fill/stroke patterns
rg "sdfFill|sdfStroke" js/shaders.js

# Audio sampling rows (find wrong patterns)
rg "iChannel0.*0\\.25" js/shaders.js
```

---

## Genre → library starting points

| Genre | Start studying |
|-------|----------------|
| Afrotech Settlement | Apollonian Gasket, Pianoscope (motion) |
| Mudcloth Bloom | Colorful Nebula (atmosphere), value noise in brief |
| Kente Loom | Paper Gear (Truchet), stripe patterns in brief |
| Cornrow Curves | Logarithmic Spirals (curves), sdSegment from IQ doc |

Technique HOW-TOs: [`04-pattern-tiling-truchet.md`](04-pattern-tiling-truchet.md), [`05-iquilez-sdf-noise-warp.md`](05-iquilez-sdf-noise-warp.md).

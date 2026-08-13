# Agent Start — PIANOSCOPE Shader Work

**Keywords:** onboarding, start, checklist, agent, first steps

---

## Read this first

You are working in **pianoscope-visualizer**: Shadertoy-style GLSL fragments, mic-driven FFT on `iChannel0`, fullscreen projection visuals for Afro-electronic genres.

**Do not** invent generic “tribal” aesthetics. Use named visual systems (Kente weave logic, Bogolan marks, cornrow curves, African fractal settlements) — see [`cursor inst.md`](../cursor%20inst.md).

---

## Reading order (15–30 min before coding)

1. [`cursor inst.md`](../cursor%20inst.md) — Repository context, cultural rules, genre palettes (skim; deep-read your target genre section).
2. [`howto/01-repo-and-workflow.md`](howto/01-repo-and-workflow.md) — Where to put code.
3. [`howto/02-audio-reactivity.md`](howto/02-audio-reactivity.md) — **Critical:** this repo’s FFT layout is *not* classic Shadertoy.
4. [`howto/07-anti-patterns-and-failures.md`](howto/07-anti-patterns-and-failures.md) — Avoid known failures.
5. **Technique docs** (pick by task):
   - Grid / textile / woven → [`howto/04-pattern-tiling-truchet.md`](howto/04-pattern-tiling-truchet.md)
   - Organic / mudcloth / noise → [`howto/05-iquilez-sdf-noise-warp.md`](howto/05-iquilez-sdf-noise-warp.md)
   - Building a genre shader → [`howto/06-genre-shader-recipes.md`](howto/06-genre-shader-recipes.md)
6. [`PIANOSCOPE_SHADER_LEARNINGS.md`](../PIANOSCOPE_SHADER_LEARNINGS.md) — Full Afrotech iteration history when touching settlement/radial shaders.

---

## Current repo state

*Update this section when shaders ship or regress.*

| Shader key (`TEST_SHADERS`) | Status | File |
|-----------------------------|--------|------|
| `PIANOSCOPE Afrohouse Mudcloth Bloom` | **Shipped (v1)** | `js/test-shaders.js` |
| `PIANOSCOPE Afrotech Fractal Settlement` | **Documented** — architecture in learnings; re-add to `test-shaders.js` if missing | see learnings doc |
| `PIANOSCOPE Amapiano Kente Loom` | **Shipped (v1)** | `js/test-shaders.js` (first in picker) |
| `PIANOSCOPE 3-Step Cornrow Curves` | **Not built** | recipe in `howto/06-genre-shader-recipes.md` |
| `PIANOSCOPE Adinkra Symbol Field Abstract` | Optional | brief only |

**Implementation order (from brief):** Afrotech → Mudcloth → Kente → Cornrow → visual approval between each.

---

## Pre-flight checklist

```
[ ] New shader added only to js/test-shaders.js (not shaders.js / shaders2.js)
[ ] TEST_SHADERS merged first in js/shadertoy-converter.js SAMPLE_SHADERS
[ ] mainImage() only — no precision, uniforms, or void main() in body
[ ] iChannel0 sampled at y = 0.0 only
[ ] Fallback motion when mic silent / denied
[ ] #define tuning constants at top (not const float at global scope)
[ ] Served over HTTP: npx serve .
[ ] Compiles clean; test mic on AND off
[ ] Readable fullscreen; no strobe / white blowout
```

---

## Console helpers (after page load)

```js
setShaderByName("PIANOSCOPE Afrohouse Mudcloth Bloom")
getCurrentShaderInfo()
```

---

## Quick retrieval

Grep or search [`KEYWORDS.md`](KEYWORDS.md) for: `fract`, `truchet`, `fbm`, `warp`, `kente`, `mudcloth`, `cornrow`, `sdf`, `audio`, `compile`, `apollonian`, `fill-stroke`.

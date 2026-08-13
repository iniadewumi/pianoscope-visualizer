# Keyword Index — Local RAG Lookup

Grep this file or search by tag. Format: `keyword → knowledge/howto/NN-file.md § section`

---

## Workflow & repo

| Keywords | Doc | Section |
|----------|-----|---------|
| onboarding, start, checklist | `00-AGENT-START.md` | full doc |
| test-shaders, TEST_SHADERS, where to edit | `howto/01-repo-and-workflow.md` | Where code lives |
| npx serve, file://, http server | `howto/01-repo-and-workflow.md` | Run locally |
| shadertoy-converter, SAMPLE_SHADERS | `howto/01-repo-and-workflow.md` | Picker order |
| setShaderByName, testing | `howto/01-repo-and-workflow.md` | Browser testing |

## Audio

| Keywords | Doc | Section |
|----------|-----|---------|
| iChannel0, FFT, spectrum, mic | `howto/02-audio-reactivity.md` | Pipeline |
| getBass, getMid, getHigh, fft() | `howto/02-audio-reactivity.md` | Standard helpers |
| y = 0.0, y = 0.25, wrong audio | `howto/02-audio-reactivity.md` | Do not copy shaders.js sampling |
| audioBoost, smoothingTimeConstant | `howto/02-audio-reactivity.md` | Heavy lag / gains |
| safeAudio, fallback, silent mic | `howto/02-audio-reactivity.md` | Fallback motion |
| #define gains, RING_BASS_GAIN | `howto/02-audio-reactivity.md` | Tuning |

## GLSL / WebGL

| Keywords | Doc | Section |
|----------|-----|---------|
| WebGL 1, texture2D, const float | `howto/03-glsl-webgl-constraints.md` | Constraints |
| iResolution vec3, uniform3f | `howto/03-glsl-webgl-constraints.md` | iResolution contract |
| compile error, line number, preamble | `howto/03-glsl-webgl-constraints.md` | Error line mapping |
| vec2 vec3 mismatch, smoothstep noise | `howto/03-glsl-webgl-constraints.md` | Common compile fixes |
| loop bounds, iFrame float | `howto/03-glsl-webgl-constraints.md` | Loops |
| porting, Murakami, library shader | `PIANOSCOPE_SHADER_LEARNINGS.md` | Porting Shadertoy library shaders |

## Patterns — Book of Shaders

| Keywords | Doc | Section |
|----------|-----|---------|
| fract, tiling, grid, repeat, cell | `howto/04-pattern-tiling-truchet.md` | Basic tiling |
| brick offset, stagger, odd row | `howto/04-pattern-tiling-truchet.md` | Brick wall offset |
| truchet, rotate tile, parity | `howto/04-pattern-tiling-truchet.md` | Truchet tiles |
| checker, weave, over under | `howto/04-pattern-tiling-truchet.md` | Weave mask |
| matrix, rotate cell, scale cell | `howto/04-pattern-tiling-truchet.md` | Per-cell transform |
| book of shaders, chapter 09 | `howto/04-pattern-tiling-truchet.md` | References |

## IQ — SDF, noise, color

| Keywords | Doc | Section |
|----------|-----|---------|
| iquilez, iq, inigo | `howto/05-iquilez-sdf-noise-warp.md` | Index |
| sdSegment, sdBox, sdCircle, 2d sdf | `howto/05-iquilez-sdf-noise-warp.md` | 2D distance functions |
| line SDF, braid, band edge | `howto/05-iquilez-sdf-noise-warp.md` | Line rendering recipe |
| fbm, fractional brownian, octaves | `howto/05-iquilez-sdf-noise-warp.md` | fBM |
| domain warp, warp, marble | `howto/05-iquilez-sdf-noise-warp.md` | Domain warping |
| voronoise, voronoi, cellular | `howto/05-iquilez-sdf-noise-warp.md` | Voronoise |
| voronoi lines, F2-F1, cracks | `howto/05-iquilez-sdf-noise-warp.md` | Uniform crack lines |
| palette, cosine palette, color | `howto/05-iquilez-sdf-noise-warp.md` | Cosine palettes |
| smoothstep, quintic, C2 noise | `howto/05-iquilez-sdf-noise-warp.md` | Smoothstep variants |
| domain repetition, sdf repeat | `howto/05-iquilez-sdf-noise-warp.md` | Domain repetition |
| mirrored repetition | `howto/05-iquilez-sdf-noise-warp.md` | Mirrored tiles |
| filtering, fwidth, antialias | `howto/05-iquilez-sdf-noise-warp.md` | Projection AA |

## Genres

| Keywords | Doc | Section |
|----------|-----|---------|
| afrotech, settlement, fractal, rings | `howto/06-genre-shader-recipes.md` | Afrotech |
| mudcloth, bogolan, afrohouse, organic | `howto/06-genre-shader-recipes.md` | Mudcloth |
| kente, amapiano, woven, loom | `howto/06-genre-shader-recipes.md` | Kente |
| kente wrong, kente diagnosis, rebuild kente | `howto/09-kente-loom-diagnosis.md` | full doc |
| cornrow, 3-step, triplet, braid | `howto/06-genre-shader-recipes.md` | Cornrow |
| adinkra, symbols | `howto/06-genre-shader-recipes.md` | Adinkra (optional) |
| palette, genre colors | `howto/06-genre-shader-recipes.md` | Palettes table |

## Failures & library

| Keywords | Doc | Section |
|----------|-----|---------|
| anti-pattern, do not, failed | `howto/07-anti-patterns-and-failures.md` | full doc |
| dot grid, kaleidoscope, edge-only | `howto/07-anti-patterns-and-failures.md` | Edge-only recursion |
| apollonian, fill stroke, sdfFill | `howto/07-anti-patterns-and-failures.md` | Fill + stroke |
| tribal triangles, generic | `howto/07-anti-patterns-and-failures.md` | Cultural caution |
| steal, reference, shaders.js | `howto/08-steal-from-library.md` | Shader table |
| Paper Gear, truchet | `howto/08-steal-from-library.md` | Pattern shaders |
| Apollonian Gasket | `howto/08-steal-from-library.md` | Settlement geometry |
| Pianoscope, Mandala | `howto/08-steal-from-library.md` | Motion vs rendering |

## External URLs (canonical)

| Topic | URL |
|-------|-----|
| Book of Shaders — Patterns | https://thebookofshaders.com/09/ |
| Book of Shaders — Noise | https://thebookofshaders.com/11/ |
| IQ articles index | https://iquilezles.org/articles/ |
| IQ 2D SDFs | https://iquilezles.org/articles/distfunctions2d/ |
| IQ fBM | https://iquilezles.org/articles/fbm/ |
| IQ domain warp | https://iquilezles.org/articles/warp/ |
| IQ Voronoise | https://iquilezles.org/articles/voronoise/ |
| IQ palettes | https://iquilezles.org/articles/palettes/ |
| IQ domain repetition | https://iquilezles.org/articles/sdfrepetition/ |
| IQ smoothsteps | https://iquilezles.org/articles/smoothsteps/ |
| IQ Voronoi lines | https://iquilezles.org/articles/voronoilines/ |

# PIANOSCOPE Shader Research Brief

This document is meant to be handed to a **Cursor agent** (or another coding assistant) working in the **pianoscope-visualizer** repo. It describes what shaders to build and how they should look and behave — not where generic `.glsl` files live.

The goal is not to make generic "African" visuals. The goal is to build a visual system for PIANOSCOPE that feels specific, musical, contemporary, and culturally aware.

---

# Repository context (read first)

This repo is a **browser-based Shadertoy-style visualizer** for live microphone input. Key files:

| File | Role |
|------|------|
| `js/visualizer-with-shadertoy.js` | WebGL renderer, mic capture, audio texture upload, shader switching |
| `js/shadertoy-converter.js` | Converts Shadertoy `mainImage` shaders to WebGL; builds the shader picker list |
| `js/shaders.js` | Existing shader library (`export const SHADERS = { "Name": \`...\` }`) |
| `js/shaders2.js` | More shaders (`export const SHADERS2 = { ... }`) |
| `js/test-shaders.js` | **New PIANOSCOPE shaders go here** (`export const TEST_SHADERS = { ... }`) |
| `index.html` | Entry page; loads `shaders.js` and `shadertoy-converter.js` as ES modules |

## How to add new shaders

**Do not create `.glsl` files or a `shaders/` folder.** Shaders in this repo are GLSL source strings inside JavaScript objects.

1. Add each shader to `js/test-shaders.js` as a named entry in `TEST_SHADERS`:

```js
export const TEST_SHADERS = {
  "PIANOSCOPE Afrotech Fractal Settlement": `
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // ... GLSL here ...
}
`,
};
```

2. Wire test shaders into the picker in `js/shadertoy-converter.js` so they appear **first**:

```js
import { TEST_SHADERS } from './test-shaders.js';
import { SHADERS } from './shaders.js';
import { SHADERS2 } from './shaders2.js';

const SAMPLE_SHADERS = { ...TEST_SHADERS, ...SHADERS, ...SHADERS2 };
```

3. Use human-readable display names as object keys (matching existing style in `shaders.js`, e.g. `"Pianoscope Text"`).

4. Write shaders in **Shadertoy style** with `void mainImage(out vec4 fragColor, in vec2 fragCoord)`. The converter prepends WebGL uniforms and appends a `main()` wrapper. It also rewrites `texture(` → `texture2D(` for WebGL 1.0 — write `texture()` in source as you would on Shadertoy.

5. Do **not** add `precision`, `uniform` declarations, or `void main()` — those are injected by `shadertoy-converter.js`.

6. Test locally: open `index.html` with a static server, allow the microphone, pick the shader from the editor dropdown or use arrow keys to cycle.

## Audio channel layout in this repo

In normal mic mode, **`iChannel0` is always the live audio spectrum texture**.

- Size: **256 × 1** luminance texture (one row of frequency bins)
- Sample along **x** for frequency (0.0 = low, 1.0 = high), always at **y = 0.0**
- This is **not** classic Shadertoy's 512×2 FFT/waveform layout (no `y = 0.25` / `y = 0.75` rows)

The built-in default shader and most repo shaders use this pattern:

```glsl
float bass = texture(iChannel0, vec2(0.05, 0.0)).x;
float mid  = texture(iChannel0, vec2(0.30, 0.0)).x;
float high = texture(iChannel0, vec2(0.70, 0.0)).x;
```

**Exceptions (do not use for new PIANOSCOPE shaders unless intentional):**

- **Video mode**: `iChannel0` becomes the video texture; audio is unavailable on that channel.
- **Feedback-pass shaders** (`iWriteFeedback` in source): audio moves to **`iChannel2`**; `iChannel0`/`iChannel1` are noise/feedback buffers.

## What not to touch unless asked

- `js/shaders.js` and `js/shaders2.js` — large curated libraries; leave them alone
- Video / kiosk / pure-view scripts — out of scope for shader generation

## Local dev and UI

- **ES modules**: serve the repo over HTTP (`npx serve .`, `python -m http.server`, etc.). Opening `index.html` as `file://` will break imports.
- **Microphone**: requires user gesture + browser permission. Works on `localhost` or HTTPS. Click **Start Listen** before judging audio reactivity.
- **Shader editor**: click the **code icon** (bottom-right) to open the Shadertoy editor panel. Search dropdown lists all `SAMPLE_SHADERS` keys. **Apply Shader** or `Ctrl+Enter` in the textarea compiles and runs the pasted/edited source.
- **Keyboard navigation**: `←` / `→` cycles shaders. Index `-1` is the default shader `"Pianoscope Text"` (from `shaders.js`, not `TEST_SHADERS`). Prepended `TEST_SHADERS` keys are first in the cycle after the default.
- **Pure view** (eye icon): fullscreen-friendly chrome-free mode for projection tests.
- **Console helpers** (after page load):
  - `setShaderByName("PIANOSCOPE Afrotech Fractal Settlement")`
  - `getCurrentShaderInfo()`

## Audio pipeline (mic mode)

Behind `iChannel0`:

| Setting | Value |
|---------|-------|
| `analyser.fftSize` | 1024 → 512 frequency bins |
| Texture upload | first **256** bins → 256×1 luminance texture |
| Value range | 0–255 in JS → **0.0–1.0** in the shader |
| `audioSensitivity` | `1.5` (amplifies quiet input) |
| `smoothingTimeConstant` | `0.8` (heavily smoothed — motion lags transients) |

Shader frequency helpers should sample **`y = 0.0` only**. Many older entries in `shaders.js` / `shaders2.js` use Shadertoy-style `y = 0.25` or other rows — those patterns are **wrong for this visualizer**. Do not copy their audio sampling blindly.

## GLSL compatibility (common compile failures)

The runtime targets **WebGL 2 with WebGL 1 fallback**. Store shaders as raw Shadertoy `mainImage` bodies; conversion happens at load time.

| Do | Don't (for V1 PIANOSCOPE shaders) |
|----|-----------------------------------|
| Write `texture(iChannel0, …)` in source | Pre-convert to WebGL or add your own `uniform` / `main()` |
| Use `mainImage(out vec4 fragColor, in vec2 fragCoord)` | `texelFetch`, multipass buffers, `iWriteFeedback` |
| Keep fragment loops ≤ ~64 iterations, constant bounds | Heavy raymarching, nested high-count loops |
| Use `float` loop indices or small fixed `for (int i = 0; i < N; i++)` | Dynamic loop limits that fail on WebGL 1 |
| Escape `` ` `` and `${` inside JS template strings | Raw backticks inside shader strings without escaping |

Uniform types injected by the converter (note: **`iFrame` is `float`**, not `int`):

```glsl
uniform vec2 iResolution;
uniform float iTime;
uniform float iTimeDelta;
uniform float iFrame;
uniform vec4 iMouse;  // Y is flipped to match WebGL coords
```

`shadertoy-converter.js` maps `texture` → `texture2D`, and on WebGL 2 also defines `texture2D` → `texture`. `textureLod` / `textureCube` aliases exist in the preamble — but prefer simple `texture()` calls.

## Script load order

`index.html` loads (order matters):

1. `js/shaders.js` (module)
2. `js/shadertoy-converter.js` (module — imports `shaders.js`, `shaders2.js`, and eventually `test-shaders.js`)
3. `js/visualizer-with-shadertoy.js`
4. `js/pure-view.js`, `js/kiosk-mode.js`, video scripts

`shaders2.js` is **not** listed in `index.html`; it is pulled in by `shadertoy-converter.js`. When wiring `test-shaders.js`, only edit `shadertoy-converter.js` imports — do not add another `<script>` tag unless there is a load error.

---

# Project context

PIANOSCOPE is an Afro-electronic listening and visual experience. The visual language should connect:

- Afrohouse
- Afrotech
- Amapiano
- 3-Step
- African pattern systems
- audio reactivity
- slow cinematic motion
- projection-friendly contrast
- shader-based procedural design

The first shader goal should be a set of reusable visual modules, not a huge 3D world.

Start with Shadertoy-style fragment shaders. Later these can be ported into Three.js, React Three Fiber, TouchDesigner, or a projection-mapping workflow.

## Recommended first target

Build four shader modes (conceptual genre labels — map to `TEST_SHADERS` display names in `test-shaders.js`):

```ts
type VisualMode =
  | "afrohouse_mudcloth_bloom"
  | "afrotech_fractal_settlement"
  | "amapiano_kente_loom"
  | "threestep_cornrow_curves";
```

| VisualMode | TEST_SHADERS key |
|------------|------------------|
| `afrotech_fractal_settlement` | `"PIANOSCOPE Afrotech Fractal Settlement"` |
| `afrohouse_mudcloth_bloom` | `"PIANOSCOPE Afrohouse Mudcloth Bloom"` |
| `amapiano_kente_loom` | `"PIANOSCOPE Amapiano Kente Loom"` |
| `threestep_cornrow_curves` | `"PIANOSCOPE 3-Step Cornrow Curves"` |

Each mode should be:

- fullscreen
- procedural
- audio-reactive
- usable as a live background
- readable from a distance
- not dependent on copyrighted image textures
- not dependent on heavy multipass simulation for the first version

Optional later mode:

```ts
type VisualMode =
  | "adinkra_symbol_field"
  | "penderecki_style_point_cloud_chamber";
```

Do not start with the point-cloud chamber. That is more ambitious.

---

# Core design principle

Avoid this:

- random tribal triangles
- neon mask with no meaning
- gold circuitry everywhere
- Wakanda-lite UI
- tourist textile collage
- sacred symbols used as filler
- overcomplicated shaders that cannot run smoothly at events

Aim for this:

- ancestral system plus electronic motion
- structure before decoration
- rhythm mapped to geometry
- pattern systems mapped to genres
- slow camera or field movement
- bass-driven expansion
- high-frequency shimmer
- midrange vocal motion
- visual motifs with actual source logic

---

# Technical assumptions

This repo runs Shadertoy-style fragment shaders via `shadertoy-converter.js`. Uniforms are injected automatically — do not redeclare them in shader source.

Available uniforms (provided by the converter preamble):

```glsl
uniform vec2 iResolution;    // viewport size in pixels (note: vec2, not vec3)
uniform float iTime;
uniform float iTimeDelta;
uniform float iFrame;
uniform float iSampleRate;
uniform vec4 iMouse;
uniform vec4 iDate;
uniform sampler2D iChannel0;   // live audio spectrum (mic mode)
uniform sampler2D iChannel1;   // unused dummy texture unless video/feedback mode
uniform sampler2D iChannel2;
uniform sampler2D iChannel3;
```

Entry point — write only this, not `main()`:

```glsl
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord.xy / iResolution.xy;
    fragColor = vec4(vec3(uv, 0.0), 1.0);
}
```

Aspect-correct centered UV:

```glsl
vec2 getUV(vec2 fragCoord) {
    vec2 uv = (fragCoord.xy - 0.5 * iResolution.xy) / iResolution.y;
    return uv;
}
```

### Audio sampling (`iChannel0` in mic mode)

The visualizer uploads microphone FFT data as a **256×1** texture. There is a single frequency row at **y = 0.0**. There is no separate waveform row.

Use helpers like the rest of this repo. Prefer `.x` when reading luminance samples:

```glsl
float fft(float x) {
    // x: 0.0 (bass) to 1.0 (treble)
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
```

Combine with synthetic fallback motion so the shader still animates when the mic is off or silent:

```glsl
float fallbackBass() { return 0.35 + 0.25 * sin(iTime * 1.8); }
float fallbackMid()  { return 0.30 + 0.20 * sin(iTime * 0.9 + 1.0); }
float fallbackHigh() { return 0.20 + 0.15 * sin(iTime * 3.7 + 2.0); }

float safeAudio(float value, float fallback) {
    return max(value, fallback * 0.35);
}
```

---

# Source links and why they matter

## African visual systems and cultural references

### African Fractals by Ron Eglash

Links:

- [African Fractals: Modern Computing and Indigenous Design, Rutgers University Press](https://www.rutgersuniversitypress.org/african-fractals/9780813526140)
- [African Fractals by Ron Eglash](https://roneglash.org/eglash.dir/afractal/afbook.htm)
- [African Fractals in Development](https://roneglash.org/eglash.dir/afractal/ik_proposal.htm)
- [Ron Eglash TED Talk: The fractals at the heart of African designs](https://www.youtube.com/watch?v=7n36qV4Lk94)
- [fracexpl JavaScript Fractal Explorer GitHub](https://github.com/srtate/fracexpl)
- [African Fractals JS explorer page](https://span.uncg.edu/fractals/homepage.html)

Why useful:

African Fractals is the strongest conceptual source for PIANOSCOPE shaders. It connects African architecture, hairstyling, textiles, sculpture, carving, painting, religion, games, and symbolic systems to recursive geometry.

Shader direction:

- recursive circular settlements
- branching pathways
- self-similar rings
- nested rectangular compounds
- circular dwellings within circular settlements
- call-and-response geometry
- audio-reactive expansion and contraction
- bass-driven radial recursion

Best genre mapping:

- Afrotech
- darker Afrohouse
- PIANOSCOPE intro sequence
- emotional "Invocation" phase

Shader techniques:

- polar coordinates
- recursive domain repetition
- fractal noise
- distance fields
- radial grids
- domain folding
- logarithmic spiral motion

Prompt for Cursor:

```txt
Create a Shadertoy fragment shader inspired by Ron Eglash's African Fractals research. Do not copy images. Build a procedural recursive settlement-like pattern using nested radial rings, circular dwellings, branching paths, and self-similar cells. Make bass expand the rings, mids rotate secondary cells, and highs add small shimmering particles. Use a black/violet/cyan palette for Afrotech. Keep it projection-friendly and not too busy.
```

---

### Culturally Situated Design Tools

Links:

- [Culturally Situated Design Tools homepage](https://csdt.org/)
- [CSDTs CSnap Modules GitHub](https://github.com/CSDTs/CSnap-Modules)
- [CSDTs CSnap GitHub](https://github.com/CSDTs/CSnap)

Why useful:

CSDTs are built around "heritage algorithms": cultural practices translated into computing, math, and design. Their modules include or reference African Fractals, Adinkra, Kente, Cornrow Curves, and Gee's Bend. The code is old, but the concept is valuable.

Shader direction:

- convert cultural pattern logic into procedural systems
- avoid surface-level imitation
- use geometry as rhythm
- use repetition as music structure

Best genre mapping:

- Kente Computing -> Amapiano or 3-Step
- Cornrow Curves -> 3-Step or vocal sections
- Adinkra Stamping -> chapter icons or transition symbols
- African Fractals -> Afrotech or PIANOSCOPE world intro

Prompt for Cursor:

```txt
Use the idea of heritage algorithms from Culturally Situated Design Tools. Generate procedural shader modules based on four systems: kente-like woven grids, cornrow-like flowing curves, African fractal recursion, and Adinkra-like symbolic fields. Avoid using actual sacred symbols unless imported from licensed SVG assets. Focus on procedural structure and musical motion.
```

---

### Adinkra symbols

Links:

- [Adinkra Icons GitHub](https://github.com/kevinkhagan/adinkra-icons)
- [Adinkra Icons website](https://adinkraicons.dev/)
- [University of Michigan Adinkra symbol meanings](https://lsa.umich.edu/daas/engagement/adinkra_symbols.html)
- [Adinkra symbols font repo](https://github.com/JacobTheEvans/adinkra)

Why useful:

Adinkra symbols can function as meaningful visual anchors. They are better than random decorative marks because they carry philosophical meanings. Use them carefully.

Potential PIANOSCOPE symbols to research:

- Dono: talking drum / rhythm / praise
- Sankofa: return, memory, roots
- Adinkrahene: greatness, leadership, centrality
- Dwennimmen: strength and humility
- Nkyinkyim: twisting, journey, adaptability
- Fawohodie: independence, freedom
- Nyame Dua: altar / sacred presence, use carefully

Use cases:

- floating symbol field
- transition marks between sections
- masks for particle emission
- SVG texture on planes in Three.js
- SDF approximation of selected symbols
- subtle background watermark

Caution:

Do not use Adinkra symbols as random filler. If a symbol is used, include its meaning in metadata or comments.

Prompt for Cursor:

```txt
Create a procedural symbolic field shader (no external textures — iChannel1 is not wired for image uploads in this repo). Use circles, arcs, and radial symmetry instead of pasted icons. Symbols should feel like they emerge from smoke or particles. Bass reveals the central glyph, mids rotate the field, highs add edge shimmer. Add to TEST_SHADERS in test-shaders.js.
```

---

### Kente visual logic

Links:

- [CSDT homepage](https://csdt.org/)
- [CSDTs CSnap Modules GitHub](https://github.com/CSDTs/CSnap-Modules)
- [Book of Shaders: Patterns](https://thebookofshaders.com/09/)

Why useful:

Kente is useful as a visual analogy for musical weaving: repeated bands, interlaced structures, alternating colors, rhythmic columns, and pattern variation.

For shader use, do not try to reproduce exact culturally specific cloth patterns unless researched carefully. Use the broader visual logic:

- woven bands
- alternating strips
- cross-hatching
- grid rhythm
- controlled color sequencing
- repeated motif cells
- diagonal accents
- syncopated pattern breaks

Best genre mapping:

- Amapiano
- 3-Step
- joyful / uplift sections

Shader techniques:

- `fract()`
- tile grids
- checker patterns
- line SDFs
- animated offsets
- pulse masks
- color palette indexing
- `step()` / `smoothstep()`

Prompt for Cursor:

```txt
Create a procedural kente-inspired loom shader without copying specific cloth designs. Use woven vertical and horizontal bands, alternating cell motifs, diagonal accents, and syncopated color breaks. Use midnight violet, burnt orange, black, and muted gold. Make bass thicken horizontal bands, mids offset vertical bands, and highs create fine thread shimmer. The result should feel like an electronic textile responding to Amapiano.
```

---

### Bogolan / Bògòlanfini / mudcloth

Links:

- [Bogolan mudcloth overview](https://www.contemporary-african-art.com/bogolan-mudcloth.html)
- [The Bogolan Mudcloth](https://www.theethnichome.com/the-bogolan-mudcloth/)
- [Bogolan Is Not Mud Cloth: Why the Name Matters and Who It Belongs To](https://omirenstyles.com/bogolan-mud-cloth/)

Why useful:

Bogolan/Bògòlanfini gives a strong direction for Afrohouse visuals: earthy, organic, geometric, hand-painted, asymmetric, clay-like. It should not look like a clean vector pattern. It should look physical.

Shader direction:

- dark charcoal base
- clay/rust/sand palette
- thresholded noise
- imperfect edges
- asymmetrical glyph rows
- line breaks
- hand-painted marks
- slow organic blooming

Best genre mapping:

- Afrohouse
- spiritual Afrohouse
- slow listening sections
- "Invocation" and "Acceptance" phases

Shader techniques:

- value noise
- fBm
- thresholding
- rough edge masks
- grid cells with random offsets
- signed distance lines
- organic wipe transitions
- reaction-diffusion approximation

Prompt for Cursor:

```txt
Create a Bogolan/Bògòlanfini-inspired shader for Afrohouse. Do not make a clean seamless wallpaper. Use charcoal, clay, rust, ochre, and sand. Use rough geometric marks, broken lines, imperfect edges, and slow organic blooming. Bass should open clay-colored blooms, mids should bend the marks, and highs should add tiny sand-like speckles.
```

---

### Ndebele mural geometry

Links:

- [Ndebele art educational overview](https://www.twinkl.com/teaching-wiki/ndebele-art)
- [Halsey Institute: Namsa Leuba's Ndebele Patterns](https://halsey.charleston.edu/learn-blog/namsa-leubas-ndebele-patterns/)

Why useful:

Ndebele mural geometry can guide bold, projection-friendly visual composition. The main value is not tiny details. It is high-contrast geometric layout.

Shader direction:

- bold lines
- triangles
- diamonds
- zigzags
- high contrast fields
- mural-like panels
- color blocks
- strong symmetry with breaks

Best genre mapping:

- event title cards
- high-energy sections
- transitions
- rave moments
- Afrotech peaks

Caution:

Ndebele art has cultural context and symbolism. Avoid flattening it into generic "tribal geometry."

Prompt for Cursor:

```txt
Create a Ndebele-mural-inspired geometric shader with bold diamonds, triangular panels, zigzags, and strong linework. Keep it abstract and avoid copying specific murals. Use audio to animate panel expansion and color transitions. The shader should work well on a projector from far away.
```

---

### Cornrow curve systems

Links:

- [Culturally Situated Design Tools homepage](https://csdt.org/)
- [CSDTs CSnap Modules GitHub](https://github.com/CSDTs/CSnap-Modules)

Why useful:

Cornrow curves are ideal for translating rhythm into flowing geometry. They can become braided paths, particle trails, curved line fields, or triplet timing structures.

Shader direction:

- braided line fields
- repeated arcs
- offset curves
- mirrored paths
- sine/cosine line families
- curve-following particles
- triplet pulse motion

Best genre mapping:

- 3-Step
- soulful Amapiano
- vocal sections
- intimacy / love sections
- bridge transitions

Shader techniques:

- curve SDFs
- repeated sine paths
- polar spirals
- domain repetition
- line distance fields
- particle trails
- feedback buffers if available

Prompt for Cursor:

```txt
Create a cornrow-curve-inspired shader made of braided flowing lines. Use repeated arcs and mirrored curves, not literal hair rendering. Make the motion triplet-based for 3-Step: three pulses per phrase, with delayed echoes. Use gold, cream, deep brown, and black. Bass should widen the braids, mids should move the curves, highs should add bead-like highlights.
```

---

### Afrofuturism

Links:

- [NMAAHC: Afrofuturism Explained](https://nmaahc.si.edu/explore/stories/afrofuturism-explained)
- [Smithsonian Afrofuturism exhibition page](https://www.si.edu/exhibitions/afrofuturism-history-black-futures%3Aevent-exhib-6648)
- [Searchable Museum: Afrofuturism](https://www.searchablemuseum.com/afrofuturism)

Why useful:

Afrofuturism can be the larger frame, but it should not become lazy sci-fi decoration. For PIANOSCOPE, the stronger direction is:

```txt
ancestral visual systems + electronic motion + future-facing sound
```

Do not overuse:

- circuit-board overlays
- generic gold tech lines
- glowing masks
- Black Panther knockoffs
- empty sci-fi UI

Use instead:

- memory
- rhythm
- coded pattern
- ritual space
- sonic architecture
- future from inherited systems

Prompt for Cursor:

```txt
Make the shader feel Afrofuturist without using generic sci-fi circuits. Use inherited pattern logic, recursive geometry, music-driven motion, and a restrained electronic glow. The result should feel like ancestral architecture becoming a digital listening chamber.
```

---

## Digital art references

### African Digital Art

Links:

- [African Digital Art](https://www.africandigitalart.com/)

Why useful:

Use this as a research archive for African digital artists, designers, animators, and creative technologists. It is not shader-specific. It is useful for taste calibration and avoiding shallow clichés.

Cursor use:

```txt
Use African Digital Art as a taste reference, not a code source. The final visuals should feel contemporary and designed, not like a generic pattern generator.
```

---

# Technical shader references

## The Book of Shaders

Links:

- [The Book of Shaders homepage](https://thebookofshaders.com/)
- [Shaping functions](https://thebookofshaders.com/05/)
- [Shapes](https://thebookofshaders.com/07/)
- [Patterns](https://thebookofshaders.com/09/)
- [Noise](https://thebookofshaders.com/11/)
- [Examples gallery](https://thebookofshaders.com/examples/)
- [Examples gallery, pattern chapter](https://thebookofshaders.com/examples/?chapter=09)

Why useful:

This is the most relevant shader-learning reference for procedural pattern generation. Use it for:

- repeated cells
- grids
- tiling
- rotation
- shape SDFs
- noise
- thresholds
- procedural texture
- simple anti-aliasing

Cursor use:

```txt
Use Book of Shaders style procedural design: normalized UV coordinates, shape functions, smoothstep-based antialiasing, pattern repetition with fract(), and fBm/noise for organic imperfection.
```

---

## Shadertoy

Links:

- [Shadertoy homepage](https://www.shadertoy.com/)
- [Shadertoy sound input example: Input - Sound](https://www.shadertoy.com/view/Xds3Rr)
- [Shadertoy audio wave and FFT example](https://www.shadertoy.com/view/ssSXRy)
- [Simple display of audio data](https://www.shadertoy.com/view/lttBDM)
- [Material Design Pattern example](https://www.shadertoy.com/view/XsySWc)

Why useful:

Shadertoy is the target model. Audio examples show how FFT and waveform textures can drive visuals.

Cursor use:

```txt
Make every shader compatible with a Shadertoy-style mainImage function. Use iResolution, iTime, iMouse, iChannel0. In this repo, iChannel0 is a 256×1 mic spectrum at y = 0.0 (not Shadertoy's two-row layout). Include graceful fallback if audio is silent.
```

---

## Three.js Shadertoy references

Links:

- [Three.js manual: Shadertoy](https://threejs.org/manual/en/shadertoy.html)
- [Felix Rieseberg: Using WebGL Shadertoy Shaders in Three.js](https://felixrieseberg.com/using-webgl-shadertoy-shaders-in-three-js/)
- [Three.js ShaderMaterial docs](https://threejs.org/docs/pages/ShaderMaterial.html)
- [Three.js Points docs](https://threejs.org/docs/pages/Points.html)

Why useful:

Later, these shaders may be ported into the PIANOSCOPE website using Three.js or React Three Fiber. The main differences are uniforms and entry point:

Shadertoy:

```glsl
void mainImage(out vec4 fragColor, in vec2 fragCoord)
```

Three.js fragment shader:

```glsl
void main() {
    vec2 fragCoord = gl_FragCoord.xy;
    vec4 color;
    mainImage(color, fragCoord);
    gl_FragColor = color;
}
```

Cursor use:

```txt
Keep the shader written in a way that can be wrapped for Three.js. Avoid obscure Shadertoy-only hacks unless necessary.
```

---

## LYGIA shader library

Links:

- [LYGIA GitHub](https://github.com/patriciogonzalezvivo/lygia)
- [LYGIA examples GitHub](https://github.com/patriciogonzalezvivo/lygia_examples)
- [Hello LYGIA Observable example](https://observablehq.com/%40radames/hello-lygia-shader-library)

Why useful:

LYGIA is a reusable shader library with GLSL, HLSL, Metal, WGSL, WESL, and CUDA support. It is useful for reusable functions:

- noise
- SDFs
- color
- filters
- generative design helpers

Cursor use:

```txt
If the repo supports imports or pre-bundled GLSL chunks, use LYGIA-style reusable functions. If not, inline minimal helper functions.
```

---

## Reaction-diffusion

Links:

- [Jason Webb Reaction-Diffusion Playground](https://jasonwebb.github.io/reaction-diffusion-playground/)
- [Jason Webb reaction-diffusion project page](https://www.jasonwebb.io/reaction-diffusion-playground)
- [Jason Webb GitHub profile with reaction-diffusion playground](https://github.com/jasonwebb)

Why useful:

Reaction-diffusion creates living organic patterns: stripes, spots, blooms, chemical-looking surfaces. It can be pushed toward mudcloth, living textile, carved wall, fungus/root systems, and Afrohouse atmosphere.

Important:

A true reaction-diffusion shader usually needs feedback buffers / multipass rendering. For first version, fake it with layered noise and thresholding.

Cursor use:

```txt
For Shadertoy single-pass version, fake reaction-diffusion using fBm noise, domain warping, thresholds, and slow time animation. Only use multipass buffers if the repo already supports them.
```

---

## General GLSL learning and examples

Links:

- [MDN: GLSL shaders](https://developer.mozilla.org/en-US/docs/Games/Techniques/3D_on_the_web/GLSL_Shaders)
- [LearnOpenGL: Shaders](https://learnopengl.com/Getting-started/Shaders)
- [Khronos GLSL wiki](https://www.khronos.org/opengl/wiki/OpenGL_Shading_Language)
- [Khronos GLSL GitHub](https://github.com/KhronosGroup/glsl)
- [Shader School GitHub](https://github.com/stackgl/shader-school)
- [Generative Design GLSL repo](https://github.com/Niels-NTG/Generative-Design-GLSL)
- [GLSL-to-MP4 renderer](https://github.com/nabeel-oz/glsl-to-mp4)

Why useful:

Use these for syntax, fundamentals, examples, and exporting shader outputs.

---

# PIANOSCOPE visual modes

## 1. Afrohouse: Mudcloth Bloom

Visual feeling:

- organic
- earthy
- spiritual but not religious
- warm
- slowly breathing
- physical texture
- broken painted marks

Palette:

```glsl
vec3 charcoal = vec3(0.025, 0.020, 0.018);
vec3 clay     = vec3(0.55, 0.22, 0.08);
vec3 rust     = vec3(0.75, 0.30, 0.10);
vec3 ochre    = vec3(0.86, 0.58, 0.24);
vec3 sand     = vec3(0.90, 0.73, 0.48);
```

Audio mapping:

- bass: clay bloom expansion
- mid: line bending / cloth breathing
- high: sand speckles

Core functions:

```glsl
float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float valueNoise(vec2 p) {
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
    float sum = 0.0;
    float amp = 0.5;

    for (int i = 0; i < 5; i++) {
        sum += amp * valueNoise(p);
        p *= 2.02;
        amp *= 0.5;
    }

    return sum;
}

float roughLine(vec2 uv, float y, float thickness, float roughness) {
    float n = fbm(uv * 8.0 + iTime * 0.05);
    float edge = abs(uv.y - y + (n - 0.5) * roughness);
    return smoothstep(thickness, thickness * 0.4, edge);
}
```

Prompt:

```txt
Add "PIANOSCOPE Afrohouse Mudcloth Bloom" to `TEST_SHADERS` in `js/test-shaders.js`. Use fBm, thresholded noise, rough broken line marks, and earthy palette. Living clay textile, not clean wallpaper. Audio helpers sampling iChannel0 at y = 0.0. Bass opens organic blooms; mid bends cloth marks; high adds speckles. Include comments and tuning constants.
```

---

## 2. Afrotech: Fractal Settlement

Visual feeling:

- dark
- precise
- recursive
- architecture-like
- future-facing
- circular communities / nodes
- hypnotic
- bass-heavy

Palette:

```glsl
vec3 blackBlue = vec3(0.005, 0.015, 0.025);
vec3 deepCyan  = vec3(0.00, 0.55, 0.65);
vec3 electric  = vec3(0.10, 0.85, 1.00);
vec3 violet    = vec3(0.22, 0.08, 0.42);
vec3 smoke     = vec3(0.45, 0.50, 0.56);
```

Audio mapping:

- bass: ring expansion
- mid: recursive cells rotate
- high: small sparks / node flicker

Core functions:

```glsl
mat2 rot(float a) {
    float s = sin(a);
    float c = cos(a);
    return mat2(c, -s, s, c);
}

float circleSDF(vec2 p, float r) {
    return abs(length(p) - r);
}

float ring(vec2 p, float r, float w) {
    return smoothstep(w, 0.0, abs(length(p) - r));
}

float radialRepeat(vec2 p, float count) {
    float a = atan(p.y, p.x);
    float sector = 6.28318530718 / count;
    a = mod(a + sector * 0.5, sector) - sector * 0.5;
    return a;
}
```

Settlement pattern idea:

```glsl
float settlement(vec2 uv, float bass, float mid) {
    float result = 0.0;

    vec2 p = uv;
    float t = iTime * 0.15;

    result += ring(p, 0.35 + bass * 0.08, 0.012);
    result += ring(p, 0.75 + bass * 0.12, 0.010);
    result += ring(p, 1.20 + bass * 0.16, 0.008);

    for (int layer = 0; layer < 4; layer++) {
        float fl = float(layer);
        float radius = 0.35 + fl * 0.25 + bass * 0.06;
        float count = 6.0 + fl * 4.0;

        for (int i = 0; i < 18; i++) {
            float fi = float(i);
            if (fi >= count) break;

            float a = fi / count * 6.28318530718 + t * (0.2 + fl * 0.06) + mid * 0.25;
            vec2 center = vec2(cos(a), sin(a)) * radius;
            float d = length(p - center);
            result += smoothstep(0.035, 0.008, d);
            result += smoothstep(0.065, 0.06, abs(d - 0.045)) * 0.5;
        }
    }

    return clamp(result, 0.0, 1.0);
}
```

Prompt:

```txt
Add "PIANOSCOPE Afrotech Fractal Settlement" to `TEST_SHADERS` in `js/test-shaders.js`. Inspired by African fractal settlement geometry, not generic sci-fi UI. Nested circular rings, recursive nodes, branching paths, self-similar motion. Bass expands settlement; mids rotate cells; highs add cyan sparks. Dark cyan/violet/black palette. Elegant, not overfilled.
```

---

## 3. Amapiano: Kente Loom

Visual feeling:

- woven
- percussive
- syncopated
- warm but still nighttime
- bounce
- log-drum pulse

Palette:

```glsl
vec3 midnightViolet = vec3(0.13, 0.06, 0.25);
vec3 deepBlack      = vec3(0.015, 0.010, 0.018);
vec3 rustOrange     = vec3(0.70, 0.27, 0.10);
vec3 mutedGold      = vec3(0.83, 0.57, 0.18);
vec3 cream          = vec3(0.92, 0.80, 0.56);
```

Audio mapping:

- bass/log drum: horizontal band thickness and vertical bounce
- mid: woven offset
- high: thread shimmer

Core functions:

```glsl
float stripe(float x, float count, float width) {
    float f = fract(x * count);
    return smoothstep(width, width - 0.01, abs(f - 0.5));
}

float wovenBand(vec2 uv, float bass, float mid) {
    float vertical = stripe(uv.x + sin(uv.y * 8.0 + iTime * 0.4) * 0.015 * mid, 10.0 + bass * 3.0, 0.32);
    float horizontal = stripe(uv.y + sin(uv.x * 7.0 - iTime * 0.3) * 0.015 * bass, 8.0 + mid * 2.0, 0.35);

    float overUnder = step(0.5, fract(floor(uv.x * 10.0) + floor(uv.y * 8.0)));
    float weave = mix(vertical, horizontal, overUnder);

    return weave;
}

vec3 kentePalette(float v, float cell) {
    vec3 a = vec3(0.13, 0.06, 0.25);
    vec3 b = vec3(0.70, 0.27, 0.10);
    vec3 c = vec3(0.83, 0.57, 0.18);
    vec3 d = vec3(0.015, 0.010, 0.018);

    vec3 color = mix(a, b, smoothstep(0.2, 0.8, v));
    color = mix(color, c, step(0.68, fract(cell * 1.618)));
    color = mix(color, d, 1.0 - v * 0.85);

    return color;
}
```

Prompt:

```txt
Add "PIANOSCOPE Amapiano Kente Loom" to `TEST_SHADERS` in `js/test-shaders.js`. Kente-inspired without copying exact textiles. Woven bands, alternating cells, diagonal accents, syncopated breaks. Bass/log drum thickens and bounces grid; mids offset bands; highs add thread shimmer. Palette: midnight violet, black, rust orange, muted gold, cream.
```

---

## 4. 3-Step: Cornrow Curves

Visual feeling:

- flowing
- braided
- elegant
- triplet pulse
- call-and-response
- less grid, more curve
- smooth but rhythmic

Palette:

```glsl
vec3 blackBrown = vec3(0.045, 0.025, 0.012);
vec3 deepGold   = vec3(0.86, 0.55, 0.16);
vec3 creamGold  = vec3(0.96, 0.80, 0.48);
vec3 shadow     = vec3(0.08, 0.05, 0.025);
```

Audio mapping:

- bass: widen curves
- mid: curve flow
- high: bead-like highlights
- triplet clock: subtle three-pulse motion independent of FFT

Core functions:

```glsl
float lineSDF(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

float braidLine(vec2 uv, float offset, float width, float speed) {
    float y = sin(uv.x * 5.0 + offset + iTime * speed) * 0.18;
    y += sin(uv.x * 11.0 - offset * 0.7 + iTime * speed * 0.5) * 0.035;
    return smoothstep(width, 0.0, abs(uv.y - y));
}

float tripletPulse(float t) {
    float beat = fract(t);
    float p1 = smoothstep(0.08, 0.0, abs(beat - 0.10));
    float p2 = smoothstep(0.08, 0.0, abs(beat - 0.43));
    float p3 = smoothstep(0.08, 0.0, abs(beat - 0.76));
    return max(p1, max(p2, p3));
}
```

Prompt:

```txt
Add "PIANOSCOPE 3-Step Cornrow Curves" to `TEST_SHADERS` in `js/test-shaders.js`. Braided flowing curves inspired by cornrow logic, not literal hair. Triplet pulse timing. Bass widens braids; mids move curve field; highs add bead glints. Palette: black-brown, deep gold, cream gold, shadow.
```

---

# Reusable shader helper library

Include these helpers in every new shader in `test-shaders.js` unless the shader already defines equivalents.

## Constants

```glsl
#define PI 3.14159265359
#define TAU 6.28318530718
```

## Centered UV

```glsl
vec2 centeredUV(vec2 fragCoord) {
    return (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
}
```

## Rotation

```glsl
mat2 rotate2d(float a) {
    float s = sin(a);
    float c = cos(a);
    return mat2(c, -s, s, c);
}
```

## Hash

```glsl
float hash11(float p) {
    p = fract(p * 0.1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}

float hash21(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
```

## Value noise

```glsl
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
```

## fBm

```glsl
float fbm(vec2 p) {
    float value = 0.0;
    float amp = 0.5;

    for (int i = 0; i < 5; i++) {
        value += amp * noise(p);
        p = p * 2.02 + vec2(7.1, 3.4);
        amp *= 0.5;
    }

    return value;
}
```

## Anti-aliased line

```glsl
float aaLine(float d, float width) {
    float aa = fwidth(d) * 1.5;
    return smoothstep(width + aa, width - aa, d);
}
```

## Circle ring

```glsl
float ring(vec2 p, float radius, float width) {
    float d = abs(length(p) - radius);
    return aaLine(d, width);
}
```

## Box SDF

```glsl
float sdBox(vec2 p, vec2 b) {
    vec2 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0);
}
```

## Audio helpers

```glsl
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
```

## Fallback audio animation

If audio is not available, use synthetic movement.

```glsl
float fallbackBass() {
    return 0.35 + 0.25 * sin(iTime * 1.8);
}

float fallbackMid() {
    return 0.30 + 0.20 * sin(iTime * 0.9 + 1.0);
}

float fallbackHigh() {
    return 0.20 + 0.15 * sin(iTime * 3.7 + 2.0);
}
```

Better combined version:

```glsl
float safeAudio(float value, float fallback) {
    return max(value, fallback * 0.35);
}
```

---

# Shadertoy template for Cursor

Use this skeleton for each shader entry in `TEST_SHADERS`.

```glsl
#define PI 3.14159265359
#define TAU 6.28318530718

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

mat2 rotate2d(float a) {
    float s = sin(a);
    float c = cos(a);
    return mat2(c, -s, s, c);
}

float hash21(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
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
    float value = 0.0;
    float amp = 0.5;

    for (int i = 0; i < 5; i++) {
        value += amp * noise(p);
        p = p * 2.02 + vec2(7.1, 3.4);
        amp *= 0.5;
    }

    return value;
}

vec3 palette(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
    return a + b * cos(TAU * (c * t + d));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = centeredUV(fragCoord);

    float bass = getBass();
    float mid = getMid();
    float high = getHigh();

    // Replace below with mode-specific visual system.
    float field = fbm(uv * 3.0 + iTime * 0.05);
    vec3 color = palette(field + bass * 0.2,
        vec3(0.2, 0.1, 0.05),
        vec3(0.7, 0.4, 0.2),
        vec3(1.0, 1.0, 1.0),
        vec3(0.0, 0.1, 0.2)
    );

    fragColor = vec4(color, 1.0);
}
```

---

# Cursor master prompt

Use this when asking a Cursor agent to generate the shader set in this repo.

```txt
You are helping build PIANOSCOPE, an Afro-electronic listening and visual experience, in the pianoscope-visualizer repo.

Technical requirements:
- Add shaders to js/test-shaders.js as entries in TEST_SHADERS (display name → GLSL template string).
- Wire TEST_SHADERS first in js/shadertoy-converter.js: { ...TEST_SHADERS, ...SHADERS, ...SHADERS2 }.
- Do NOT create .glsl files. Do NOT edit js/shaders.js or js/shaders2.js.
- Use mainImage(out vec4 fragColor, in vec2 fragCoord) only — no precision, uniforms, or main().
- Use iResolution (vec2), iTime, iMouse, iFrame as needed.
- iChannel0 is the live mic spectrum: 256×1 texture, sample frequency at y = 0.0 (not Shadertoy y = 0.25/0.75).
- Include helper functions for bass, mid, and high sampling iChannel0 at y = 0.0.
- Include graceful fallback motion if audio input is silent.
- Avoid external image textures for the first pass.
- Avoid multipass / feedback buffers unless explicitly requested.
- Use comments and tuning constants.

Creative requirements:
- Do not create generic "tribal" visuals.
- Use specific visual systems as inspiration:
  1. African Fractals / recursive settlement geometry
  2. Kente-like woven rhythmic grids
  3. Bogolan/Bògòlanfini-inspired rough clay textile marks
  4. Cornrow-curve-inspired braided line systems
  5. Optional Adinkra-inspired symbolic fields
- Do not copy sacred symbols or exact textile patterns.
- Use abstraction, rhythm, procedural structure, and audio motion.
- Make visuals projection-friendly, not tiny-detail-heavy.
- Use dark backgrounds with controlled glow.

Generate these shaders (as TEST_SHADERS keys):
1. "PIANOSCOPE Afrotech Fractal Settlement"
2. "PIANOSCOPE Afrohouse Mudcloth Bloom"
3. "PIANOSCOPE Amapiano Kente Loom"
4. "PIANOSCOPE 3-Step Cornrow Curves"
5. Optional: "PIANOSCOPE Adinkra Symbol Field Abstract"

For each shader:
- Add the full GLSL body to test-shaders.js.
- Explain the main visual logic.
- Explain how bass, mids, and highs affect the image.
- Include 5 tuning constants at the top.
- Keep frame rate reasonable.
- Avoid excessive loops.
```

---

# Smaller Cursor prompts by mode

## Afrohouse prompt

```txt
Add to js/test-shaders.js under TEST_SHADERS with key "PIANOSCOPE Afrohouse Mudcloth Bloom". Bogolan/Bògòlanfini visual logic: charcoal background, clay/rust/ochre/sand palette, rough geometric marks, broken lines, imperfect hand-painted edges, slow organic blooms. No exact pattern copying. Use fBm, thresholded noise, rough line SDFs, audio reactivity. Bass opens clay blooms, mids bend marks, highs add sand speckles. mainImage only; sample iChannel0 at y = 0.0.
```

## Afrotech prompt

```txt
Add to js/test-shaders.js under TEST_SHADERS with key "PIANOSCOPE Afrotech Fractal Settlement". African Fractals research: recursive settlement geometry, nested circular rings, circular dwellings, branching paths, self-similar cells. No generic sci-fi UI. Dark cyan/violet/black palette. Bass expands rings, mids rotate cells, highs add sparks. mainImage only; sample iChannel0 at y = 0.0.
```

## Amapiano prompt

```txt
Add to js/test-shaders.js under TEST_SHADERS with key "PIANOSCOPE Amapiano Kente Loom". Kente-like woven visual logic, not exact cloth copying. Woven bands, alternating cells, diagonal accents, syncopated breaks, thread shimmer. Palette: midnight violet, black, rust orange, muted gold, cream. Bass thickens/bounces bands, mids offset layers, highs add shimmer. mainImage only; sample iChannel0 at y = 0.0.
```

## 3-Step prompt

```txt
Add to js/test-shaders.js under TEST_SHADERS with key "PIANOSCOPE 3-Step Cornrow Curves". Cornrow-curve-inspired braided flowing lines, not literal hair. Repeated arcs, mirrored curves, triplet timing, echo trails. Palette: black-brown, deep gold, cream gold, shadow. Bass widens braids, mids move curves, highs add bead glints. mainImage only; sample iChannel0 at y = 0.0.
```

## Adinkra abstract prompt

```txt
Add to js/test-shaders.js under TEST_SHADERS with key "PIANOSCOPE Adinkra Symbol Field Abstract". No exact sacred symbols — abstract circles, arcs, central marks, radial symmetry, negative space. Symbols emerging from smoke/particles. Bass reveals central glyph, mids rotate field, highs add edge shimmer. mainImage only; sample iChannel0 at y = 0.0.
```

---

# Performance requirements

Target:

- 60 fps on a decent laptop
- acceptable 30 fps on weaker machines
- fullscreen 1080p projector
- no more than 64 loop iterations total in fragment shader if possible
- avoid nested loops with high counts
- prefer analytic shapes over heavy raymarching
- avoid expensive noise calls inside large loops
- avoid true reaction-diffusion first unless using buffers intentionally

Projection requirements:

- strong silhouette
- dark background
- controlled brightness
- avoid all-white flashes
- avoid tiny details that disappear on walls
- avoid extreme strobe unless explicitly enabled
- use smooth motion
- bass should be felt visually but not make the image unreadable

---

# Suggested repo structure

All new PIANOSCOPE shaders belong in **`js/test-shaders.js`**, not standalone `.glsl` files.

```txt
js/
  test-shaders.js          ← new PIANOSCOPE shaders (TEST_SHADERS)
  shadertoy-converter.js   ← import TEST_SHADERS and spread first
  shaders.js               ← existing library (do not edit)
  shaders2.js              ← existing library (do not edit)
  visualizer-with-shadertoy.js
```

Example `test-shaders.js`:

```js
export const TEST_SHADERS = {
  "PIANOSCOPE Afrotech Fractal Settlement": `
// tuning constants, helpers, mainImage ...
`,
  "PIANOSCOPE Afrohouse Mudcloth Bloom": `
// ...
`,
};
```

Each entry should be **self-contained** (inline helpers per shader). Shared helper extraction is optional later; do not block on it.

Wire-up in `shadertoy-converter.js`:

```js
import { TEST_SHADERS } from './test-shaders.js';
import { SHADERS } from './shaders.js';
import { SHADERS2 } from './shaders2.js';

const SAMPLE_SHADERS = { ...TEST_SHADERS, ...SHADERS, ...SHADERS2 };
```

Test shaders appear first in the dropdown and keyboard cycle order.

---

# Metadata format for each shader

Use a comment block at the top:

```glsl
/*
PIANOSCOPE Shader
Name: Afrohouse Mudcloth Bloom
Mode: afrohouse_mudcloth_bloom
Inspired by: Bogolan/Bògòlanfini visual logic, reaction-diffusion texture, Afrohouse atmosphere
Cultural caution: This is not a copy of a specific textile. It uses abstracted visual principles: rough clay marks, earthy palette, broken geometry, handmade edge quality.
Audio:
- Bass: organic bloom expansion
- Mid: mark bending / cloth breathing
- High: speckles / edge shimmer
Projection:
- Dark background
- Warm controlled glow
- No strobe
*/
```

---

# What not to do

Do not generate:

- exact Adinkra symbols procedurally unless the symbol and meaning are intentionally chosen
- fake "African hieroglyphs"
- random tribal tattoos
- generic mask faces
- Black Panther UI clones
- thin neon lines everywhere
- overly detailed texture that will fail on projection
- code that only looks good in a small browser preview
- true raymarching scenes unless specifically requested
- heavy multipass buffer systems for V1

---

# What to do instead

Generate:

- recursive geometry
- sound-reactive pattern systems
- woven rhythmic grids
- organic clay thresholds
- braided curve fields
- symbolic abstraction
- controlled glow
- dark negative space
- large-scale readable composition
- parameters that can later become room presets

---

# Possible room preset object

Future metadata if PIANOSCOPE gets genre-based room switching. Today, shaders are selected by **display name key** in `SAMPLE_SHADERS` (from `TEST_SHADERS`).

```ts
export type PianoscopeRoom = {
  id: string;
  title: string;
  genre: "afrohouse" | "afrotech" | "amapiano" | "three-step";
  shaderKey: string; // key in TEST_SHADERS / SAMPLE_SHADERS
  palette: {
    background: string;
    primary: string;
    secondary: string;
    accent: string;
  };
  audioMapping: {
    bass: string;
    mid: string;
    high: string;
  };
  culturalReference: string;
  caution: string;
};

export const pianoscopeRooms: PianoscopeRoom[] = [
  {
    id: "afrohouse",
    title: "Afrohouse / Mudcloth Bloom",
    genre: "afrohouse",
    shaderKey: "PIANOSCOPE Afrohouse Mudcloth Bloom",
    palette: {
      background: "#050201",
      primary: "#c2642a",
      secondary: "#7a2e10",
      accent: "#d4a52c",
    },
    audioMapping: {
      bass: "organic bloom expansion",
      mid: "cloth bending and mark motion",
      high: "sand-like speckles",
    },
    culturalReference: "Bogolan/Bògòlanfini visual logic, abstracted",
    caution: "Do not copy exact textile symbols without research.",
  },
  {
    id: "afrotech",
    title: "Afrotech / Fractal Settlement",
    genre: "afrotech",
    shaderKey: "PIANOSCOPE Afrotech Fractal Settlement",
    palette: {
      background: "#01070c",
      primary: "#00a6b2",
      secondary: "#271052",
      accent: "#c9f6ff",
    },
    audioMapping: {
      bass: "ring expansion",
      mid: "recursive cell rotation",
      high: "node sparks",
    },
    culturalReference: "African Fractals research, recursive settlement geometry",
    caution: "Use geometric logic, not decorative cliché.",
  },
  {
    id: "amapiano",
    title: "Amapiano / Kente Loom",
    genre: "amapiano",
    shaderKey: "PIANOSCOPE Amapiano Kente Loom",
    palette: {
      background: "#05020a",
      primary: "#21113f",
      secondary: "#c2642a",
      accent: "#d4a52c",
    },
    audioMapping: {
      bass: "log drum band pulse",
      mid: "woven offset motion",
      high: "thread shimmer",
    },
    culturalReference: "Kente-like woven visual logic, abstracted",
    caution: "Do not copy exact cloth patterns.",
  },
  {
    id: "three-step",
    title: "3-Step / Cornrow Curves",
    genre: "three-step",
    shaderKey: "PIANOSCOPE 3-Step Cornrow Curves",
    palette: {
      background: "#090501",
      primary: "#f0b33f",
      secondary: "#6b3d12",
      accent: "#fff1b8",
    },
    audioMapping: {
      bass: "braid width",
      mid: "curve travel",
      high: "bead highlights",
    },
    culturalReference: "Cornrow curve logic, abstracted",
    caution: "Use curve math, not literal hair imagery.",
  },
];
```

---

# First implementation order

Do not build everything at once.

Recommended order:

1. Create `js/test-shaders.js` and wire `TEST_SHADERS` first in `shadertoy-converter.js`
2. `"PIANOSCOPE Afrotech Fractal Settlement"` in `TEST_SHADERS`
3. `"PIANOSCOPE Afrohouse Mudcloth Bloom"` in `TEST_SHADERS`
4. `"PIANOSCOPE Amapiano Kente Loom"` in `TEST_SHADERS`
5. `"PIANOSCOPE 3-Step Cornrow Curves"` in `TEST_SHADERS`
6. Test in browser (mic on, shader picker, arrow-key cycling)
7. Projector test at fullscreen
8. Optional: genre labels / room presets in UI (not built yet)
9. Optional: Three.js/R3F port later (out of scope for this repo today)

The best first shader is **Afrotech Fractal Settlement** because it is visually distinct, culturally grounded, and technically manageable.

Second best is **Afrohouse Mudcloth Bloom** because it gives a totally different organic mood.

---

# Testing checklist

For each shader added to `test-shaders.js`, test in the browser (`index.html` + mic enabled):

- Does it compile? (check the on-page error log and browser console)
- Does it still move when no audio is connected?
- Does audio visibly affect it?
- Does bass feel different from highs?
- Does it look good fullscreen?
- Does it look good from 15 feet away?
- Does it avoid cheap cliché?
- Does it work on a projector?
- Does it maintain dark negative space?
- Does it avoid strobe?
- Can colors be tuned quickly?
- Can it run for 30 minutes without visual fatigue?

---

# Output request for Cursor

Final instruction to a Cursor agent:

```txt
Start by adding "PIANOSCOPE Afrotech Fractal Settlement" to js/test-shaders.js (TEST_SHADERS) and wiring TEST_SHADERS first in shadertoy-converter.js.

Then add "PIANOSCOPE Afrohouse Mudcloth Bloom" the same way.

Do not generate the whole set until the first two compile and look good in the browser with the mic on.

For each shader:
- add the GLSL body as a template string in TEST_SHADERS
- include comments and 5 tuning constants
- include audio helpers sampling iChannel0 at y = 0.0
- include fallback motion when audio is silent
- briefly explain visual logic and bass/mid/high mapping after the code
```

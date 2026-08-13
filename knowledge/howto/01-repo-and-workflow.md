# Repo & Workflow

**Keywords:** test-shaders, TEST_SHADERS, workflow, npx serve, shadertoy-converter, mainImage

**Use when:** Starting any PIANOSCOPE shader task; unsure where to edit files.

---

## Where code lives

| File | Role | Edit for new PIANOSCOPE work? |
|------|------|-------------------------------|
| `js/test-shaders.js` | `export const TEST_SHADERS = { "Name": \`...\` }` | **Yes — only here** |
| `js/shadertoy-converter.js` | Merges `TEST_SHADERS` first into picker | Verify spread order |
| `js/shaders.js`, `js/shaders2.js` | Reference library (~16k lines) | **No** — study only; patch only for compile fixes |
| `js/visualizer-with-shadertoy.js` | WebGL runtime, audio upload, uniforms | Rarely (uniform uploads) |
| `cursor inst.md` | Creative brief | Read, don’t duplicate |
| `PIANOSCOPE_SHADER_LEARNINGS.md` | Postmortems | Append failures; link to `knowledge/` |
| `knowledge/` | HOW-TO library | **This folder** |

---

## How to add a shader

1. Open `js/test-shaders.js`.
2. Add entry to `TEST_SHADERS`:

```js
export const TEST_SHADERS = {
  "PIANOSCOPE Your Shader Name": `
/*
PIANOSCOPE Shader
Name: ...
Mode: ...
Inspired by: ...
Cultural caution: ...
Audio: bass = ...; mid = ...; high = ...
*/

#define SOME_GAIN 0.30

// helpers + mainImage only
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    ...
}
`,
  // existing entries...
};
```

3. Confirm `js/shadertoy-converter.js`:

```js
const SAMPLE_SHADERS = { ...TEST_SHADERS, ...SHADERS, ...SHADERS2 };
```

4. **Do not** add `precision`, `uniform`, or `void main()` — converter injects preamble.

---

## Shader body rules

- Entry: `void mainImage(out vec4 fragColor, in vec2 fragCoord)`
- Use `iResolution`, `iTime`, `iChannel0`, `iFrame`, `iMouse` as needed
- Tuning: `#define` at top (WebGL 1 — avoid global `const float`)
- Include header comment block (genre, audio mapping, cultural caution)
- Copy standard audio helpers from [`02-audio-reactivity.md`](02-audio-reactivity.md)

---

## Run locally

```bash
cd /path/to/pianoscope-visualizer
npx serve .
```

Open `http://localhost:3000` (or printed port). **Avoid `file://`** — ES module imports may fail.

Enable mic when testing audio reactivity.

---

## Browser testing

**Picker:** UI shader list (TEST_SHADERS appear first).

**Console:**

```js
setShaderByName("PIANOSCOPE Afrohouse Mudcloth Bloom")
getCurrentShaderInfo()
```

**Checklist:** mic on + mic off, fullscreen, console clean, no compile errors in status bar (full log in DevTools).

---

## Implementation order (project convention)

1. One genre shader at a time
2. Visual approval before next genre
3. Order: Afrotech Settlement → Mudcloth → Kente → Cornrow

See [`06-genre-shader-recipes.md`](06-genre-shader-recipes.md) for per-genre specs.

---

## When to edit library shaders (`shaders.js`)

Only when fixing a **broken compile** for an existing Shadertoy port — not for new PIANOSCOPE work.

Read **Porting Shadertoy library shaders** in `PIANOSCOPE_SHADER_LEARNINGS.md` first. Error line numbers include preamble offset (~30 lines).

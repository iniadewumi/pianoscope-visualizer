# Repo & Workflow

**Keywords:** test-shaders, TEST_SHADERS, workflow, http.server, shadertoy-converter, mainImage

**Use when:** Starting any PIANOSCOPE shader task; unsure where to edit files.

---

## Where code lives

| File | Role | Edit for new PIANOSCOPE work? |
|------|------|-------------------------------|
| `js/test-shaders.js` | `export const TEST_SHADERS = { "Name": \`...\` }` | **Yes — only here** |
| `js/shadertoy-converter.js` | Merges `TEST_SHADERS` first into picker | Verify spread order |
| `js/shaders.js`, `js/shaders2.js` | Reference library (~24,000 lines together: 92 + 32 shaders) | **No** — study only; patch only for compile fixes |
| `js/visualizer-with-shadertoy.js` | Operator window: WebGL runtime, audio, texture binding | Rarely |
| `js/gl-shared.js` | GL code both windows run: compile, uniform table, per-frame uniforms, feedback, draw | When adding a uniform |
| `kiosk.html`, `js/kiosk-window.js` | Projector window; pulls its sources from the operator window | Rarely |
| `js/kiosk-mode.js` | Kiosk button and handshake on the operator side | Rarely |
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
- Tuning: `#define` at top (project convention for quick tuning)
- Include header comment block (genre, audio mapping, cultural caution)
- Copy standard audio helpers from [`02-audio-reactivity.md`](02-audio-reactivity.md)

---

## Run locally

```bash
cd /path/to/pianoscope-visualizer
python3 -m http.server 8747
```

Open `http://localhost:8747`. **Avoid `file://`** — ES module imports may fail.

Enable mic when testing audio reactivity.

---

## Browser testing

**Picker:** UI shader list (TEST_SHADERS appear first).

**Console:**

```js
setShaderByName("PIANOSCOPE Hairpin")
getCurrentShaderInfo()
```

**Checklist:** mic on + mic off, fullscreen, console clean, no compile errors in status bar (full log in DevTools).

---

## Implementation order (project convention)

1. One genre shader at a time
2. Visual approval before next genre

See [`06-genre-shader-recipes.md`](06-genre-shader-recipes.md) for per-genre specs.

---

## When to edit library shaders (`shaders.js`)

Only when fixing a **broken compile** for an existing Shadertoy port — not for new PIANOSCOPE work.

Read **Porting Shadertoy library shaders** in `PIANOSCOPE_SHADER_LEARNINGS.md` first. Compiler error line numbers are your own shader's line numbers (the converter emits `#line` directives).

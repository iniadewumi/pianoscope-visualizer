# GLSL & WebGL Constraints

**Keywords:** WebGL 1, compile, iResolution, vec3, preamble, texture2D, loops, feedback

**Use when:** Shader fails to compile; porting from Shadertoy (including multi-buffer ones); mixed vec2/vec3 errors.

---

## WebGL 1 rules (this project)

| Issue | Fix |
|-------|-----|
| `texture()` in source | Converter rewrites to `texture2D` |
| Global `const float` | Use `#define` |
| `out` parameters in helpers | Return `vec3` packs instead |
| `bool` in helpers | Use `float` flags `0.0` / `1.0` |
| `fwidth()` / heavy AA | Often fails WebGL 1 paths — use fixed feather `~0.003` UV |
| Loop bounds | Must be constant: `for (int i = 0; i < 12; i++)` + `if (fi >= count) break;` |
| `iFrame` | `float`, not `int` |

---

## iResolution contract

Preamble uses **`uniform vec3 iResolution`** (Shadertoy-compatible). Runtime uploads:

```js
gl.uniform3f(uniforms.iResolution, canvas.width, canvas.height, 1.0);
```

- Aspect: `iResolution.x / iResolution.y` or `iResolution.xy`
- **Broken:** `vec3 p = iResolution` when preamble was `vec2` — now fixed project-wide
- **Broken:** `vec3(iResolution, 1.0)` with vec3 uniform — too many args; use `iResolution` or `vec3(iResolution.xy, 1.0)`

Converter auto-rewrites some patterns.

---

## Compile pipeline (error line numbers)

Final fragment source = **runtime prefix** + **WEBGL_PREAMBLE** + **shader body** + **main wrapper**.

**GLSL error line is 1-based in the full compiled string**, not the textarea line in `test-shaders.js`. Expect ~30-line preamble offset.

**Debug flow:**

1. Confirm shader name in status bar
2. Read full `gl.getShaderInfoLog` in DevTools (status bar truncates)
3. Classify: operand types / dimension mismatch / varying scope

---

## Common compile fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `wrong operand types` on `-` | Mixed vec2/vec3 in smoothstep noise | `vec2(3.0) - 2.0 * diff` not `vec3(3) - vec2(2) * diff` |
| `2-component` + `iResolution` | vec2 vs vec3 uniform | Use vec3 preamble (current) |
| `'varying' : only allowed at global scope` | `in`/`out` fn params | WebGL 2 only, or refactor |

---

## Shaders with existing `void main()`

`convertShaderToyToWebGL` returns them **as-is** (no preamble). Examples in library: `"Black Hole"`, `"DULL AMAP"`. PIANOSCOPE shaders should **not** use this pattern.

---

## Frame feedback (porting Shadertoy multi-buffer shaders)

There are no Buffer A/B/C passes here. There is **one** ping-pong buffer, and a
shader opts into it by mentioning `iWriteFeedback` anywhere in its source. The
visualizer then draws it twice per frame:

| Pass | `iWriteFeedback` | Target | Reads last frame from |
|------|------------------|--------|-----------------------|
| write | `1.0` | ping-pong buffer, canvas size | `iFeedback` (and `iChannel1` in shader mode) |
| display | `0.0` | canvas | same |

```glsl
uniform float iWriteFeedback;
uniform sampler2D iFeedback;   // last frame, 1:1 with fragCoord

void mainImage(out vec4 O, in vec2 U) {
    vec2 uv = U / iResolution.xy;
    if (iWriteFeedback > 0.5) { O = accumulate(uv); return; }
    O = display(texture2D(iFeedback, uv));
}
```

Porting rule of thumb: a Shadertoy buffer that only transforms the current frame
is per-frame arithmetic — inline it. Only a buffer that reads **itself** needs
the ping-pong. Fabrice Neyret's four-buffer video-statistics shader folds to one
pass this way; see `"Temporal Moments"` in `js/video-effects.js`.

Things that bite:

- **The display pass is one frame behind the write pass.** Both draws sample the
  same read texture, so the first frame after a switch shows the cleared buffer
  (black). Everything after is 16 ms stale, which nobody sees.
- **Channels differ by mode.** Shader mode: noise on `iChannel0`, feedback on
  `iChannel1`, audio on `iChannel2`. Video mode keeps audio on `iChannel0` and
  the frame on `iChannel1`, so feedback is only on `iFeedback` — new code should
  read `iFeedback` in both modes.
- **The live source is on `iVideo` in every mode, not just video mode.** Source
  (camera/screen/clip, owned by VideoController) and look (which program is on
  screen) are independent: whatever is attached keeps playing and uploading
  regardless of look, and any library shader can sample it via `iVideo` on unit
  6 — `iChannel1` stays reserved for the video-effect look's own contract.
  `iVideoActive` is 0 when nothing is attached (the sampler then holds a 1x1
  black texel). `videoUV()`, `videoInside()`, `videoAt()`, and `videoLuma()` are
  available in every shader's preamble, mirroring the helpers in
  `js/video-effects.js` but sampling `iVideo`. The video dock's Look toggle
  (Effect / Shader) picks which program is on screen; the Blend row (Off /
  Video under / Video over / Luma key / Multiply, plus an Amount slider) only
  applies while Look is Shader, composited automatically via
  `pianoscopeBlendVideo()` in the converter's generated `main()`.
- **The buffer is RGBA8 unless you ask.** Add `#define FEEDBACK_HDR` and the
  visualizer allocates RGBA32F (or RGBA16F) where the driver can render to it.
  Required for anything storing a statistic rather than a colour: in 8 bits the
  per-frame update rounds to zero and the accumulator *freezes* instead of
  converging. A console warning names the fallback when it happens.
- **Video effects declare nothing.** `VideoEffects.build()` adds the feedback
  uniforms and a write-pass-aware `main()` only to effects that mention
  `iFeedback`, so the second draw is not charged to the whole effect list.
- **Avoid `E[x²] − E[x]²`.** Two nearly equal terms, and the difference is what
  you wanted to keep. Use a running update instead.

---

## Verify compile in console (WebGL 2)

```js
const src = window.ShaderConverter.convertShaderToyToWebGL(
  window.ShaderConverter.SAMPLE_SHADERS["PIANOSCOPE Afrohouse Mudcloth Bloom"]
);
window.visualizer.applyShader(src);
```

Full porting case studies → `PIANOSCOPE_SHADER_LEARNINGS.md` § Porting Shadertoy library shaders.

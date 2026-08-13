# GLSL & WebGL Constraints

**Keywords:** WebGL 1, compile, iResolution, vec3, preamble, texture2D, loops

**Use when:** Shader fails to compile; porting from Shadertoy; mixed vec2/vec3 errors.

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

## Verify compile in console (WebGL 2)

```js
const src = window.ShaderConverter.convertShaderToyToWebGL(
  window.ShaderConverter.SAMPLE_SHADERS["PIANOSCOPE Afrohouse Mudcloth Bloom"]
);
window.visualizer.applyShader(src);
```

Full porting case studies → `PIANOSCOPE_SHADER_LEARNINGS.md` § Porting Shadertoy library shaders.

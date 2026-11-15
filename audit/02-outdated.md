# What is outdated

## Docs that contradict the code

Commit `c1ad6f3` made the app WebGL 2 only. The docs still teach WebGL 1. This matters more than usual here because the docs are written for agents, who follow them literally.

| Doc says | Where | Reality |
|----------|-------|---------|
| "WebGL 1 rules (this project)" | `knowledge/howto/03`, `knowledge/README.md`, `KEYWORDS.md` | Everything compiles as GLSL ES 3.00 |
| Avoid global `const float`, use `#define` | `howto/01`, `howto/07`, learnings doc, agent checklist | Legal in ES 3.00 |
| Avoid `out` params and `bool` in helpers | `howto/07`, learnings doc | Legal in ES 3.00; library shaders already use them |
| Avoid `fwidth()`, use a fixed feather | `howto/03`, `howto/05`, `howto/07` | `fwidth` is core in ES 3.00. A fixed feather may still be the right look on a projector, but the stated reason is wrong |
| "WebGL 2 with WebGL 1 fallback" | `cursor inst.md:111` | The fallback was removed |
| Run with `npx serve .` on port 3000 | `howto/01`, agent checklist, learnings doc, `KEYWORDS.md` | README and `.claude/launch.json` use `python3 -m http.server 8747` |
| Library is `shaders.js` + `shaders2.js`, "~16k lines" | `howto/01`, `howto/08` | 24,244 lines, 124 shaders |
| "New shaders only in `test-shaders.js`" | agent checklist | `test-shaders.js` also holds `Neon Heart` and `Procgen Cats`, which are Shadertoy ports |
| Kente Loom and Mudcloth Bloom shipped | `00-AGENT-START.md` | Removed in the working tree (see [01-dumb.md](01-dumb.md)) |
| Afrotech Fractal Settlement: "re-add if missing" | `00-AGENT-START.md` | It is missing, and has been |
| "Implementation order: Afrotech → Mudcloth → Kente → Cornrow" | `00-AGENT-START.md`, `howto/01` | The actual work since has been the CRESCENDO set |
| Reading list has two step 5s | `00-AGENT-START.md` | Numbering slip |
| Error line numbers "include preamble offset (~30 lines)" | `howto/01` | The converter now emits `#line` directives, so reported lines are the author's own |

There are also three overlapping doc sets: `cursor inst.md` (53 KB), `PIANOSCOPE_SHADER_LEARNINGS.md` (25 KB) and `knowledge/` (108 KB). The same rules appear in all three, which is how they went stale in three places. One set of rules, linked from the others, would not.

## Code left over from WebGL 1

None of this is broken. It is noise that misleads the next reader.

- [shadertoy-converter.js](../js/shadertoy-converter.js) rewrites `texture(` to `texture2D(`, "needed for WebGL 1.0". The preamble then defines `texture2D` back to `texture`. A round trip that does nothing.
- `createShader` in the main runtime and `ShaderPrefixes` in the kiosk both define `texture2D`, `attribute` and `gl_FragColor` shims. The converter preamble defines `texture2D` and `attribute` a second time, plus `#define varying out`, which is only correct in a vertex shader and sits in a fragment preamble.
- The audio texture is `gl.LUMINANCE`. WebGL 2 accepts it for compatibility, but the modern format is `R8`. Changing it is not free: `LUMINANCE` returns the value in r, g and b, `R8` only in r, so any shader reading `.g` or `.b` of `iChannel0` would break. Leave it unless a shader audit says it is safe.
- `window.AudioContext || window.webkitAudioContext`: the prefix has not been needed in any current browser for years.

## Dependencies

| Dependency | Version | Status |
|------------|---------|--------|
| FontAwesome | 5.15.4 | Released 2021; the 5.x line is no longer developed. The page uses 26 icons. Inline SVGs would remove the dependency entirely |
| MediaPipe tasks-vision | 1.0.1 (pinned) | Check for a newer release when you next touch the person mask |
| MediaPipe model | `.../float16/latest/...` | Unpinned. The model can change under you between shows. Pin a version, or better, vendor the file |

## Branches

| Branch | Last commit | Ahead of main | Suggestion |
|--------|-------------|---------------|------------|
| `perf-improvements` | Aug 2026 | 0 | Delete, fully merged |
| `origin/backup-main` | Mar 2025 | 0 | Delete |
| `origin/feature/restructure` | Mar 2025 | 0 | Delete |
| `video`, `origin/video` | Apr 2025 | 1 | Look at the one commit, then delete |

## Leftover files

- `TODOS/shaders.py` is deleted in the working tree but the deletion is not committed.
- `.cursor/` is empty.
- `images/pianoscope-logo.png` is unreferenced.
- `fonts/AkiraExpanded.otf` is still on disk and still wired into `index.html` through `@font-face`, after the commit that stopped distributing it. It works on this laptop and silently falls back everywhere else, including the public site.

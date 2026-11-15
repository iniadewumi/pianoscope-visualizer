# Efficiency

Nothing here was measured. The app was not run for this audit. Treat runtime items as leads to profile with the built-in HUD (backtick key), not as proven wins.

## Runtime

The frame loop is already well looked after: async shader compile with `KHR_parallel_shader_compile`, a warm-up draw, one analyser read per frame, a throttled dB readout, a resolution governor with a pixel cap, preview at half resolution while the kiosk is open. There is little cheap gain left in it.

| Lead | Detail | Expected gain |
|------|--------|---------------|
| Two windows shade the same image | With the kiosk open, the GPU runs the fragment shader for both the projector and the operator preview. `previewScaleWithKiosk: 0.5` already cuts the preview to a quarter of its pixels | Small on top of the existing cut. An operator toggle to freeze or drop the preview to about 10 fps during a heavy shader would hand the rest to the projector |
| Fixed `HEAVY_NAME` regex | `perf.js` guesses cost from shader names and loop bounds | The adaptive scaler already measures real frame time. Recording the settled scale per shader in `localStorage` and starting there next time would skip the hunting period on every switch |
| Mic processing | Echo cancellation, noise suppression and gain control run on the audio thread for no benefit | Minor CPU; the real gain is signal quality ([01-dumb.md §4](01-dumb.md)) |
| Hidden main window | The kiosk pulls audio itself when the main loop stops, which is the right design | None needed; noted because it is easy to break when unifying the renderers |

## Load

| Lead | Detail | Fix |
|------|--------|-----|
| 840 KB of shader source parsed at boot | `shaders.js` (549 KB), `shaders2.js` (208 KB), `test-shaders.js` (82 KB) are imported eagerly; one shader is used at start | GitHub Pages gzips this well, so the cost is parse time and memory, not transfer. Low priority on its own; it comes for free with per-file shaders ([03-long-term.md §2](03-long-term.md)) |
| Render-blocking third-party CSS | FontAwesome and a twelve-family Google Fonts stylesheet block first paint | Self-host; load display fonts on demand when a text style first uses one |
| Twelve classic scripts | Each is a separate request, in order | Fine over HTTP/2. Not worth a bundler |
| Kiosk page built from a string each session | 58 KB of JS that exists only to become a blob | Goes away with a real `kiosk.html` |

## Repository

| Item | Size | Fix |
|------|------|-----|
| Git pack | 132 MB | About 120 MB is deleted `videos/` content still in history. `git filter-repo --path videos --invert-paths` removes it. This rewrites history and needs a force push, so do it once, after the current branch is merged, and re-clone afterwards |
| `media/videos/` | 68 MB tracked | Move to R2 with the rest. Keep one small clip (a few MB) in the repo for localhost testing, since R2 only sends CORS headers to the Pages origin |
| `media/images/*.png` | 12 MB tracked | `eyeronik.png` is 6.8 MB at 2688×1792. Re-encode as WebP or JPEG at projector resolution; expect well under 1 MB each |
| `images/pianoscope-logo.png` | 2.1 MB | Unreferenced. Remove from the tree |
| `concepts/` | 60 MB untracked | Add `concepts/**/renders/` to `.gitignore`; commit the two source files if they are worth keeping |

Allowing `localhost` in the R2 bucket's CORS rules would remove the reason any video needs to live in git at all.

## Workflow

These cost more time per week than anything in the frame loop.

| Friction | Fix |
|----------|-----|
| Stale JS in the browser pane from `python http.server` (already in your memory notes) | Serve with no-cache headers. A ten-line Python server script that sends `Cache-Control: no-store`, wired into `.claude/launch.json`, ends the manual cache busting |
| Three documented ways to serve the site, three ports | Pick one (`launch.json`'s 8747) and fix the docs |
| Every renderer feature written twice | One renderer ([03-long-term.md §1](03-long-term.md)) |
| Editing one shader means opening a 17,600-line file | Per-file shaders |
| No way to know a change broke a shader you did not look at | The compile check ([03-long-term.md §4](03-long-term.md)) |
| Agents start from docs that describe WebGL 1 | Fix the docs ([02-outdated.md](02-outdated.md)); add a root `CLAUDE.md` |
| Long-lived uncommitted work makes every diff huge and every review vague | Commit per feature. The current diff mixes the person mask, a shader-set rewrite, kiosk changes and doc edits in one working tree |
| Camera, fullscreen and video cannot be tested in the in-app pane (also in your notes) | Keep the srcdoc and oscillator harnesses as files under a `dev/` folder instead of rebuilding them per session |

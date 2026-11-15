# What is dumb

Ordered by how much it can hurt.

## 1. The current work is not backed up

> **Resolved 14 Nov 2026.** Committed and pushed to `origin/fix/live-show-and-editor`.

- `fix/live-show-and-editor` is 14 commits ahead of `main` and has no remote branch.
- On top of that, 12 tracked files carry 2,062 uncommitted added lines (522 removed), and `js/person-mask.js` is untracked.
- The last commit is dated 30 September. Files were last edited on 11 October.
- GitHub Pages serves `main`, last touched 8 September. The public site is two months behind what you run locally.

**Fix:** commit in logical pieces and push the branch today. Merge to `main` when the show build is the one you trust.

## 2. The uncommitted diff deletes two shipped shaders

> **Resolved 14 Nov 2026.** The deletion was intentional; the docs and `perf.js` were updated to match.

`PIANOSCOPE Amapiano Kente Loom` and `PIANOSCOPE Afrohouse Mudcloth Bloom` are in `js/test-shaders.js` at `HEAD` and absent from the working tree. Meanwhile:

- `knowledge/00-AGENT-START.md` still lists both as "Shipped (v1)".
- `js/perf.js` still has `kente loom` in `HEAVY_NAME`.
- The console example in the docs is `setShaderByName("PIANOSCOPE Afrohouse Mudcloth Bloom")`, which now fails.

Either the deletion was deliberate and the docs are stale, or it was collateral from the 1,980-line rewrite of that file. Check before committing.

## 3. The kiosk renderer is a copy of the main renderer, stored in a string

> **Addressed 14 Nov 2026.** The kiosk is now a real page (`kiosk.html`, `js/kiosk-window.js`) and both windows compile, locate uniforms, set per-frame uniforms, manage feedback buffers and draw through `js/gl-shared.js`. What remains per window is texture upload and binding. The text below describes the state before that change.

[js/kiosk-mode.js](../js/kiosk-mode.js) holds a complete HTML page inside a template literal from line 21 to about line 1090. It is turned into a blob URL and opened as the projector window.

Consequences:

- Ten functions exist twice under the same names: `createShader`, `adoptProgram`, `pollShaderProgram`, `destroyBuild`, `feedbackFormat`, `createFeedbackState`, `destroyFeedbackState`, `configureFeedbackPass`, `setupFeedbackBuffers`, `render`.
- The uniform table exists twice (about 31 entries each). They already differ: the kiosk has no `iDate`, `iSampleRate` or `iChannel2`. Today that only affects `Mandelbrot Fract` (the one shader that reads `iDate`), but it shows the direction.
- Code in a string gets no syntax highlighting, no linting and no "find references". Every backtick and `${` inside it has to be escaped by hand.
- Every renderer feature (person mask, text sheet, feedback, video) has to be written twice. The recent diff adds 63 lines to the kiosk copy for exactly that reason.

The projector window is the one the room sees. It should be the primary renderer, not the hand-synced copy. See [03-long-term.md](03-long-term.md#1-one-renderer-two-windows).

## 4. The mic is opened with voice-call processing

[js/visualizer-with-shadertoy.js:1137](../js/visualizer-with-shadertoy.js:1137) calls `getUserMedia({ audio: true })`. With no constraints, Chrome applies echo cancellation, noise suppression and automatic gain control. Those are built for speech:

- Noise suppression treats sustained musical content as noise and thins it out.
- Automatic gain flattens the difference between a quiet breakdown and a drop, which is the thing the visuals should follow.

**Fix:**

```js
getUserMedia({ audio: {
    echoCancellation: false,
    noiseSuppression: false,
    autoGainControl: false
} })
```

`audioSensitivity = 1.5` and the shaders' thresholds were tuned against the processed signal, so expect to re-tune after the change. Test it against real music before a show, not at one.

## 5. A show tool that needs the internet to look right

`index.html` loads, before first paint:

- FontAwesome 5.15.4 from cdnjs (the whole icon font for 26 icons).
- Twelve Google Fonts families in one request.

The display-text feature draws those fonts onto the wall. If venue Wi-Fi is down or slow, the type falls back to a system font mid-show. The person mask also pulls about 12 MB of WASM and a model from two more hosts; the file's own header says to copy them locally for an offline show, but nothing does.

There is no service worker, so a reload without a connection loses the page itself when served from GitHub Pages.

**Fix:** vendor the fonts and icons into the repo (drop Akira per the existing licence note), and add a service worker that precaches the app shell. See [03-long-term.md](03-long-term.md#3-offline-first).

## 6. Media is committed to git

- `media/` is 77 MB in the tree: four videos (11–25 MB each) and three PNGs (2–7 MB).
- The pack is 132 MB. About 120 MB of that is an old `videos/` folder that was deleted from the tree but is still in history.
- R2 already hosts the large clips, and two of the local videos have R2 twins.
- `images/pianoscope-logo.png` (2.1 MB, 1536×1024) is tracked and nothing in the app references it; the page uses the 27 KB small version.

Every clone and every Pages deploy carries this. See [04-efficiency.md](04-efficiency.md#repository).

## 7. Shader licences are unexamined

Of 135 shaders, 38 cite a Shadertoy URL and 43 carry some licence or copyright text; 17 of those say non-commercial. Shadertoy's default licence, when a shader states none, is CC BY-NC-SA 3.0. Shaders with no header at all have unknown origin.

If Pianoscope plays ticketed or paid events, non-commercial shaders are a real exposure. This is a question for you to settle, not a code fix: at minimum, record source and licence per shader so a "safe for paid shows" set can be filtered.

## 8. Smaller things

| Thing | Where | Note |
|-------|-------|------|
| File named `cursor inst.md` | repo root | 53 KB creative brief named after a tool, with a space in the name. Every link needs `%20`. Rename to something like `BRIEF.md`. |
| Empty `.cursor/` folder, lone `.vscode/settings.json` | repo root | The VS Code file pins Live Server to port 5501; the README says 8747; the docs say `npx serve` on 3000. Three ports for one static site. |
| `.DS_Store` files | root, `media/` | Ignored, but still sitting there. Harmless. |
| `concepts/` untracked | repo root | 60 MB, mostly renders and `node_modules`. Decide: commit `scenes.js` and `render.mjs`, ignore `renders/`. |
| Shader named `Blueprint (needs work)` | `js/shaders.js` | It is in the live picker and the arrow-key cycle. |
| Two definitions of the default shader name | [shadertoy-converter.js:672](../js/shadertoy-converter.js:672), [visualizer-with-shadertoy.js:1444](../js/visualizer-with-shadertoy.js:1444) | Same string in two files. |
| `PINNED_SHADER_SLOTS = [null, 'Procgen Cats']` | [visualizer-with-shadertoy.js:1446](../js/visualizer-with-shadertoy.js:1446) | A general slot mechanism holding one entry. |
| `videoUV` / `videoInside` / `videoAt` written twice | `shadertoy-converter.js`, `video-effects.js` | The comment says one "mirrors" the other. Export one GLSL string and use it in both. |

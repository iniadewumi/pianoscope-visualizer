# What can improve long term

The project has no build step and no dependencies, and that is a strength: it deploys by pushing files. Everything below keeps that.

## 1. One renderer, two windows

**Now:** the main window and the kiosk window each carry their own renderer. The kiosk's is a string ([01-dumb.md §3](01-dumb.md)).

**Target:**

- `js/renderer.js` — an ES module that takes a canvas and a set of data sources (audio, video, text, person mask, pointer) and draws. No DOM beyond the canvas, no knowledge of which window it is in.
- `index.html` — creates a renderer for the operator preview and owns the controls.
- `kiosk.html` — a real file that imports the same `renderer.js` and reads its data sources from `window.opener`, as the string version does now.

A same-origin `kiosk.html` opened with `window.open` has the same `window.opener` access as the blob page, so the pull model for audio, text and video carries over unchanged. The blob URL lifetime handling goes away.

**How to get there without a big-bang rewrite:**

1. Move the template string out into `kiosk.html` as it is. Nothing shared yet, but it becomes real, lintable code.
2. Extract one duplicated piece at a time into a shared module, starting with the smallest: the uniform table, then the feedback buffers, then compile/adopt, then the draw.
3. Delete each copy as its shared version lands.

Verify each step on the real projector output, since the in-app browser pane cannot exercise this path well.

## 2. Shaders as data with metadata

**Now:** 135 shaders live as template strings in three JS files, 840 KB together. The only metadata is the object key. Whatever else is known (source URL, licence, whether it needs audio, whether it is show-ready) is in free-text comments or nowhere.

**Target:** one file per shader plus a manifest.

```
shaders/
  manifest.json
  crescendo/hairpin.glsl
  library/sunset-train.glsl
```

```json
{ "id": "hairpin", "name": "PIANOSCOPE Hairpin", "set": "crescendo",
  "source": "original", "licence": "own", "audio": true, "text": false,
  "cost": "light", "status": "show" }
```

What that buys:

- GLSL files get real syntax highlighting and can be linted or validated with standard tools.
- The picker can filter: show-ready only, own work only, by set, by cost.
- `status: "wip"` keeps `Blueprint (needs work)` out of the arrow-key cycle during a show.
- `HEAVY_NAME` in `perf.js`, a regex of shader names, becomes a `cost` field.
- The licence question in [01-dumb.md §7](01-dumb.md) gets a place to be answered.
- Shader source loads on demand instead of all 840 KB at boot.
- Git diffs become per-shader. Today a change to one shader is a diff against a 17,600-line file.

A manifest only is a fair halfway step: keep the JS files, add `shaders/manifest.json` keyed by name, and read `status` and `licence` from it.

## 3. Offline first

For a tool used at venues, "works with the network cable pulled" should be a tested property.

- Vendor the fonts (self-hosted WOFF2, variable where used) and replace FontAwesome with inline SVG.
- Vendor the MediaPipe WASM and model, as `person-mask.js` already suggests.
- Add a small service worker that precaches the app shell, shaders and fonts. `video-cache.js` already uses Cache Storage for clips, so the pattern is in the codebase.
- Add a pre-show check panel: fonts loaded, mic permission granted, second screen detected, clips cached, WebGL 2 with float render targets available.

## 4. A compile check

The only failure that really matters is a shader that does not compile on show night. A headless check is cheap:

- A script that opens the page in headless Chrome, runs every shader through `ShaderConverter` and the compile path, and prints the failures.
- Run it by hand before a show and in a GitHub Action on push.

This is the one test worth writing. It also makes the larger changes above safe to attempt. Unit tests for the rest are not worth the upkeep for a one-person project.

## 5. Module boundaries

`index.html` loads one ES module and twelve classic scripts in a required order, and they talk through about 27 `window.*` globals (`window.visualizer`, `window.videoController`, `window.audioData`, `window.currentFragmentShaderSource`, ...).

It works, and a full rewrite is not justified. But two things would help:

- Convert files to ES modules as they are touched. Load order then resolves itself and the "Load scripts in correct order" comment can go.
- Keep the cross-window surface explicit. The kiosk reads `window.opener.pianoscopeReadAudio`, `pianoscopeReadPointer`, `pianoscopeReadPersonMask`, `PianoscopeDisplayText`, `PianoscopePerf` and `videoController`. Gather those into one `window.pianoscopeHost` object so the contract between the two windows is written down in one place.

Also: [shadertoy-converter.js](../js/shadertoy-converter.js) is 1,036 lines, of which about 250 convert shaders and the rest build the editor UI with `createElement`. Split it into `shader-converter.js` and `shader-editor-ui.js`. The name currently hides where the editor lives.

## 6. Docs

- One agent entry point. Add a short `CLAUDE.md` at the root that points to `knowledge/00-AGENT-START.md`, so agents load it without being told.
- One home for each rule. Keep `knowledge/` as the source, cut the duplicated rule tables out of `cursor inst.md` and the learnings doc, and link instead.
- Generate the "current repo state" table from the shader manifest once it exists. A hand-kept status table is the part that went stale fastest.
- Rename `cursor inst.md`.

## 7. Audio analysis

Not urgent, but the ceiling on how good the visuals can feel.

- `fftSize = 1024` at 48 kHz gives bins about 47 Hz wide. A kick drum's fundamental sits in one or two bins. A larger FFT (2048 or 4096) resolves bass better, at the cost of slower time response; the usual answer is a log-spaced remap so bass gets more texels and treble fewer.
- Shaders each derive "bass", "mid" and "high" themselves from the texture. Computing a few shared features once in JS (band energies, a kick onset flag, a smoothed loudness) and exposing them as uniforms would make shaders simpler and consistent with each other.
- The sequencer runs on tapped tempo. Onset detection could offer an automatic beat.

Any of these changes how existing shaders respond, so they need a re-tune pass. Do the mic-constraints fix first and see how much of the problem remains.

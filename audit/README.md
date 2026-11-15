# Setup audit — 14 November 2026

A read-through of the repo as it stands on `fix/live-show-and-editor` (working tree, including uncommitted changes).

## Verdict

The render path is in good shape: non-blocking shader compiles, one FFT read per frame, a resolution governor, careful comments. The weak points are around it, not in it:

1. **Six weeks of work exists only on this laptop.** 14 commits are unpushed and another 2,062 added lines across 12 files are uncommitted.
2. **The projector renderer is a second copy of the main renderer**, kept inside a string, and it has already drifted.
3. **The mic is opened with the browser's voice-call processing on**, which works against a music visualizer.
4. **The docs tell agents the wrong things** (WebGL 1 rules, shaders that no longer exist), so every agent session starts from bad premises.
5. **A live-show tool depends on three CDNs at start-up** and has no offline path.

## Files

| File | Question it answers |
|------|---------------------|
| [01-dumb.md](01-dumb.md) | What is plainly wrong or risky today |
| [02-outdated.md](02-outdated.md) | What is stale: docs, code left from WebGL 1, dependencies, branches |
| [03-long-term.md](03-long-term.md) | Structural changes that pay off over months |
| [04-efficiency.md](04-efficiency.md) | Runtime, load and workflow efficiency |
| [05-webgpu.md](05-webgpu.md) | What a WebGPU backend would add, what it costs, and a staged plan |
| [06-deep-dive.md](06-deep-dive.md) | Recovery, clock ownership, what the audio signal really is, and whether the system steers from the right data |

## Suggested order

| # | Do this | Effort | Why first |
|---|---------|--------|-----------|
| 1 | Commit and push the branch | Minutes | Nothing else matters if the laptop dies |
| 2 | Decide whether Kente Loom and Mudcloth Bloom were meant to be deleted | Minutes | The uncommitted diff removes both; the docs still call them shipped |
| 3 | Turn off mic voice processing and re-tune sensitivity | An hour, plus a listen | Changes how every shader reacts |
| 4 | Fix the docs that contradict the code | An hour | Cheap, and stops agents writing GLSL for the wrong target |
| 5 | Vendor fonts and icons; add an offline start-up path | Half a day | Removes a show-night failure mode |
| 6 | Move the kiosk to a real page sharing the renderer | 1–2 days | Ends the duplication; the biggest structural win |
| 7 | Take media out of git and slim the history | Half a day | 132 MB pack, 77 MB of media in the tree |
| 8 | Add a headless "does every shader compile" check | Half a day | The only test this project really needs |

## What this audit did not cover

- The app was not run and nothing was profiled. Efficiency items are from reading code, not measurement.
- `js/display-text.js`, `js/video-controller.js`, `js/video-effects.js` and the CSS were outlined, not read line by line.
- The 135 individual shaders were not reviewed for quality or cost.
- Licence terms were counted by text search, not checked shader by shader.

## Since the audit

Most findings in 01, 02 and 06 have been fixed; each file carries a status note. `dev/compile-all.html` is the compile check proposed in 03.

# Pianoscope Visualizer

This page uses shaders to visualize the input audio of your device microphone. The goal is to play music and generate nice visuals that follow the soundwaves of the music.

View here: https://iniadewumi.github.io/pianoscope-visualizer/

Run locally with any static server, e.g. `python3 -m http.server 8747`, then open `http://localhost:8747`.

## Kiosk mode

**Kiosk Mode** opens a second window meant for a projector or external display. In Chrome it moves itself to the other screen (the browser asks for window-management permission the first time). Press **F** for fullscreen and **Esc** to close it.

The kiosk follows the main window: shader, video, display text and audio all come from there, so keep the main window open. If the main window reloads, the kiosk reconnects on its own. Reloading the kiosk window also works, unless the main window has reloaded since the kiosk opened; in that case, or if the kiosk was closed, reopen it from the main window.

## Video and media

The **Media** tab in the shader editor plays a video, still image, camera or screen capture through the current shader. You can also upload a file, drop one onto the panel, or paste a URL.

The library comes from `media/videos/index.json` and `media/images/index.json`. Each entry is either a file name committed next to the manifest (`"src": "clip.mp4"`) or a full URL (the large clips live on Cloudflare R2). When a local file and an R2 URL share a file name, the local one is used. A local dev server also lists anything dropped into those folders; GitHub Pages only shows what the manifests list.

R2 only sends CORS headers for the GitHub Pages origin, so remote clips will not load from `localhost`; test with local files. Remote clips can be saved for offline playback from their card or with the download-all button.

## Display text and sequencer

The **Display text** section of the shader editor puts type on the wall. A pipe (`|`) or newline stacks a second line, and wrapping words in asterisks (`CRESCENDO *is* HOUSE`) accents them. Pick a style, font and transition there. Text-aware shaders render the text themselves; tick **JS overlay** to draw it on top of any shader.

The **Sequence** controls step through preset two-line "word ladders", where the bottom line promotes to the top and a new line enters below. Steps advance on a bar count against a tapped tempo: **T** taps tempo, **Space** steps, **P** plays. How it works: [`knowledge/howto/10-display-text-and-sequencer.md`](knowledge/howto/10-display-text-and-sequencer.md).

## Agent / developer docs

| Doc | Purpose |
|-----|---------|
| [`knowledge/README.md`](knowledge/README.md) | **Start here** — HOW-TO library, keyword index, genre recipes |
| [`cursor inst.md`](cursor%20inst.md) | Creative brief and cultural context |
| [`PIANOSCOPE_SHADER_LEARNINGS.md`](PIANOSCOPE_SHADER_LEARNINGS.md) | Iteration postmortems and compile-debug notes |

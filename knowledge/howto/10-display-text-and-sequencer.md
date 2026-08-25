# Display Text, Transitions and the Word Ladder Sequencer

**Keywords:** display-text, text, typography, iTextSheet, iChannel3, sequencer, transitions, word ladder, anadiplosis, kiosk, tap tempo, slot A, slot B

**Status:** built on branch `feat/kinetic-text`, **uncommitted**. Read "Repo state" before touching anything.

---

## What this is

A live text system for the visualizer: type on the wall that can change on the beat, with a two-slot "word ladder" that assembles a sentence over time. Built against a research brief; the reasoning behind the design choices is in that brief, the mechanics are here.

The device is **anadiplosis** — the last term of one line becomes the first term of the next:

```
CRESCENDO | [ARTIST]
CRESCENDO | IS HOUSE
HOUSE     | IS BLACK      <- "HOUSE" promoted from the line below
BLACK     | IS CULTURE
CULTURE   | IS OURS
OURS      | TONIGHT
CRESCENDO | [ARTIST]      <- turnaround, loops forever
```

One slot holds, the other changes, then the changed word promotes. That is also how a house loop is built — something held, something moving — which is why the text feels like it belongs to the music rather than sitting on top of it.

---

## Repo state — read this first

| Fact | Value |
|------|-------|
| Branch | `feat/kinetic-text` |
| Committed? | **No.** Nothing in this work is committed. |
| Safety tag | `pre-kinetic-text` — the tree as it was before this work began |
| See only this work | `git diff pre-kinetic-text -- js/ css/ index.html` |
| Untracked new file | `js/text-sequencer.js` |

The working tree **also** contains ~2,468 lines of the repo owner's own uncommitted work (perf harness, video effects, kiosk revamp). The owner explicitly wanted those kept separate from this work, which is why nothing here is committed. Do not run a blanket `git add -A && git commit` without asking.

To recover the pre-work state of any file: `git checkout pre-kinetic-text -- <path>`

---

## The rule that matters most

**New shader work goes in as a NEW `TEST_SHADERS` entry.** Do not retune `PIANOSCOPE Amapiano Kente Loom` — it still reads the legacy `iChannel3` sheet and hard-cuts.

`PIANOSCOPE Chrome Stage` was **deleted** at owner request. `PIANOSCOPE Kick Afterglow` keeps its original `TEXT_COVER` size on `iChannel3` and opts into the shared sequencer with `#define SEQ_COVER 1.8`. Do not copy Word Ladder's `SEQ_LINE` mapping onto it.

---

## Architecture: two sheets

The existing text shaders and a purpose-built one want incompatible things from the texture, so `js/display-text.js` rasterizes **two**.

### 1. Legacy sheet — `iChannel3`

Fixed **2048 x 512**. `R` = sharp glyphs, `G` = soft (blurred), `B` = 0, `A` = opaque.

This is byte-for-byte the layout Kente Loom (and the video overlay) were tuned against: 280px fit ladder stepping by 6, 0.86 x 0.72 fit box, 0.46 vertical bias, 18px blur. **Do not retune any of those numbers.**

Regression check — with text `CRESCENDO|IS HOUSE` the glyph bbox must measure **791 x 343 px, fraction 0.386**, and the blue channel max must be **0**. If those move, an existing shader's output has moved with them.

### 2. Fitted sheet — `iTextSheet`, texture unit 4

Sized to the copy it holds (e.g. 1715 x 842). `R` = slot A sharp, `G` = slot A soft, `B` = slot B sharp — slot B being the incoming word during a swap.

One fixed cap size (`FIT_SIZE = 340`) for every cue, no ceiling, no wasted margin. Glyphs fill about **92%** of this sheet against 38.6% on the legacy strip, so type is barely magnified on a wall instead of blown up out of a fixed strip.

Cap size is fixed deliberately: in a two-slot ladder the held line must not resize when the other slot swaps.

### Companion uniforms

Declared in the `WEBGL_PREAMBLE` in `js/shadertoy-converter.js`, so every shader sees them (unused ones are optimised out):

| Uniform | Meaning |
|---------|---------|
| `iTextSheet` | sampler for the fitted sheet (unit 4) |
| `iTextSheetRes` | its pixel size |
| `iTextMorph` | 0 → 1 while a swap is in flight |
| `iTextLineH` | one line's height as a fraction of the sheet — divide by this to pin cap height while copy length changes |
| `iTextExtent` | glyph block as a fraction of the sheet — use to fit to frame without letting long copy run off the edge |
| `iTextLegacyExtent` | glyph block as a fraction of the 2048x512 strip — `SEQ_COVER` shaders use this to keep legacy on-screen size |
| `iTextTransition` | index into `TRANSITIONS` |
| `iTextWords[8]` | per-word rects `(x0,y0,x1,y1)` in sheet UV, slot A |
| `iTextWordCount` | how many of those are live |

Every uniform is wired in **three** places. Miss one and it silently reads zero:

1. `js/shadertoy-converter.js` — the preamble declaration
2. `js/visualizer-with-shadertoy.js` — `getShaderUniforms()` cache **and** the per-frame setter
3. `js/kiosk-mode.js` — its own `kioskUniforms` cache **and** its own per-frame setter

---

## Files

| File | Role |
|------|------|
| `js/display-text.js` | Rasterizes both sheets, owns transition state and `morph` |
| `js/text-sequencer.js` | **New.** Word ladder, promotion, tap tempo, hotkeys |
| `js/text-seq-glsl.js` | Shared `seqSample` GLSL, injected by the converter |
| `js/test-shaders.js` | Word Ladder (`SEQ_LINE`); Kick Afterglow (`SEQ_COVER`, original size); Chrome Stage removed |
| `js/shadertoy-converter.js` | Preamble uniforms + injects `SEQ_GLSL` before `mainImage` + sidebar |
| `js/visualizer-with-shadertoy.js` | Binds unit 4, sets the new uniforms |
| `js/kiosk-mode.js` | Second texture for the fitted sheet, forwards all new uniforms |
| `css/shader-sidebar.css` | Transport styling (`.seq-transport`, `.seq-btn`, `.seq-num`) |
| `index.html` | Loads `text-sequencer.js` after `display-text.js` |

---

## Opting a shader into the sequencer

Helpers live in `js/text-seq-glsl.js` and are injected before `mainImage` when the source defines `SEQ_LINE` or `SEQ_COVER`. Call `seqSample(uv, mid)` for `vec3(sharp, soft, knockout)`.

| Define | Mapping |
|--------|---------|
| `#define SEQ_LINE 0.29` | Fitted sheet, cap height pinned (Word Ladder) |
| `#define SEQ_COVER 1.8` | Legacy `TEXT_COVER` on `iChannel3` at rest; slot B remapped from the fitted sheet during a swap (Kick Afterglow) |
| `#define SEQ_HAS_FIELD` | Pattern reveal uses the shader's own `fieldValue()` |

Do not copy Word Ladder's line mapping onto a cover shader — that is what changes the type size.

---

## Transitions

Defined once in `display-text.js` `TRANSITIONS`, indexed by `iTextTransition`:

| # | id | Default ms | Where it lives |
|---|----|-----------|-----------------|
| 0 | `cut` | 0 | No transition — the legacy single-slot path |
| 1 | `roll` | 340 | Shader: outgoing climbs out, incoming rises |
| 2 | `reveal` | 520 | Shader: uncovered by the shader's **own** `fieldValue()` |
| 3 | `chroma` | 360 | Shader: outgoing expands and thins, incoming snaps in |
| 4 | `knockout` | 420 | Shader: letters arrive as holes through the field |
| 5 | `scramble` | 460 | **JS**: glyphs re-rastered and resolve left to right |

`cut` is the default and takes a separate path that never touches slot B — that is what keeps the legacy sheet inert for the existing shaders.

`scramble` is the only one that re-rasterizes mid-flight. It runs on a `setInterval` at `SCRAMBLE_FPS` (14), **not** rAF — see the hidden-window note below.

---

## Sequencer

`window.PianoscopeTextSequencer`. Three sequences: `Thesis` (the ladder above), `Roll call` (house history), `Ticker` (single words for low-attention stretches).

A step is either explicit `lines`, or `{ promote: true, enter: '...' }` shorthand.

**Promotion takes the last TERM, not the whole clause.** `CRESCENDO IS HOUSE` promotes to `HOUSE`, never `IS HOUSE`. `promoteTail()` strips leading copulas (`IS`, `ARE`, `WAS`…). A step can override with `promoteAs` where the copy needs something the rule will not produce. This was a real bug once — if you see `IS HOUSE|IS BLACK`, that regressed.

Timing is bars against a tapped tempo. **There is deliberately no automatic beat detection**: the analyser feeding this project runs at `smoothingTimeConstant = 0.8`, which flattens exactly the transients onset detection needs. Adding it would require a second, unsmoothed analyser node.

Hotkeys (guarded against modifiers and editable targets): **Space** steps, **P** plays/pauses, **T** taps tempo.

---

## Controls

In the shader sidebar (click the code icon), inside the existing display-text panel: text area + Apply, font picker, JS-overlay toggle, transition picker + Preview, sequence picker, artist field, Play / Step / Tap, BPM and bars inputs, and a live readout.

---

## Measured numbers

Chrome, `devicePixelRatio 2`. Re-measure rather than trusting these if you change the raster path.

| | Before | After |
|---|---|---|
| Text swap, GPU-synced | 88 ms | **2.5 ms** |
| Frames > 20 ms during swaps | ~5 per swap | **0** |
| Glyph fill of its sheet | 38.6% | **92.5%** |

The original 88 ms had one cause: `drawLines()` draws per character, and the blur was an **active canvas filter** during that loop. Chrome applies a filter as a full-surface pass per draw call — 134 ms for a 21-glyph string against 7 ms for one blurred `drawImage` of the finished sharp pass. If a swap ever gets slow again, check that first.

---

## Gotchas that will waste your time

**Hidden windows.** `requestAnimationFrame` stops when the window is hidden. `morph` is therefore derived from `performance.now()` on read, not accumulated in the rAF callback, and `settle()` has a `setTimeout` backstop. The kiosk runs its own loop on the projector and reads `getMorph()`, so a swap keeps playing there while the operator's window is buried. **Do not move morph back into the rAF callback.**

Related pre-existing risk, still open: the kiosk gets audio via a 50 ms `postMessage` reading `window.audioData`, which only refreshes in the **main** window's render loop — and that loop stops on `visibilitychange`. Minimise the control window mid-set and the projector keeps drawing but stops reacting.

**Two version counters.** `version` means "the display string changed". `sheetVersion` means "the fitted sheet was redrawn". A scramble redraws the sheet many times inside a single string change — the kiosk gates its upload on `sheetVersion` for exactly that reason. Gating on `version` freezes the scramble on the projector.

**Texture units.** Uploads in `display-text.js` pin `gl.activeTexture(gl.TEXTURE7)` as a staging unit. Without that, `bindTexture` lands on whatever unit the render loop left active — unit 4 after the fitted sheet is bound — and an upload between frames quietly rebinds a sampler's texture.

**Browser caching during dev.** A plain reload will serve stale JS. Force it:

```js
Promise.all(Array.from(document.querySelectorAll('script[src]')).map(s => fetch(s.src, {cache:'reload'}))).then(() => location.reload());
```

**Serve over HTTP.** ES modules break on `file://`. `python3 -m http.server 8747`.

**`ctx.font` silently drops `font-stretch`.** Canvas parses the CSS font shorthand but discards the stretch from it — setting `"900 expanded 340px Saira"` reads back as `"900 340px Saira"`. Saira and Anybody only looked expanded because Google was serving exactly **one** instance per family, so the family resolved to the wide cut by accident. `setType()` now sets `ctx.fontStretch` separately, **after** `ctx.font` — order matters, because assigning `font` resets `fontStretch` to `normal`. Note `ctx.fontStretch` takes keywords only; a percentage silently falls back to `normal`.

**Ask Google Fonts for axis RANGES.** `Anybody:wdth,wght@150,900` returns a **static** cut with no `fvar` table — no axes to interpolate at runtime. `@50..150,100..900` returns the real variable file. Verified by pulling the served woff2 and checking for `fvar`. The switch is metric-neutral: `extra-expanded` (150%) on the variable file measures byte-identical to the old static wdth-150 instance, so the sheets are unchanged.

**The legacy sheet cache used to outlive the webfont.** `legacyKey` was built from `(text, fontId)`, and neither changes when a face finishes downloading — so the sheet kept whatever was rasterised on **first paint**, which is the fallback face, and the `commitVisuals()` calls in `init()` and `setFont()` that exist precisely to fix this were no-ops. On a cold load the wall showed system sans at the wrong width until something unrelated changed the key. `fontEpoch` is now folded into the key and bumped on `loadFonts()`, `document.fonts.ready`, and a `setFont()` load. Symptom to watch for: a sheet fraction around 0.55 for Anybody instead of 0.838.

---

## Not done

| Item | Notes |
|------|-------|
| **Kiosk verification** | Everything is wired but **never confirmed end-to-end** — popups were blocked in the dev browser. Highest-value next check. |
| Kiosk `wordRects` | A `Float32Array` crosses realms from opener to kiosk into `gl.uniform4fv`. Cross-realm typed arrays are a known WebGL gotcha. Untested. |
| Visible-case scramble cadence | Only ever measured in a background tab at the ~1 Hz timer throttle. The 14 Hz path is unverified. |
| Web MIDI | Clock / pad triggers off the mixer. More reliable and less work than beat tracking — do this before onset detection. |
| Onset detection | Needs a second analyser at `smoothingTimeConstant = 0.0`. |
| Audience input | Phone → screen. Needs a backend the static site does not have, **and a moderation queue in the same commit as the feature.** Never ship unmoderated audience text to a wall. |

---

## Typeface

Default is **Syne ExtraBold** (Akira Expanded still in the picker). Loaded via `@font-face` from `local()` plus `fonts/AkiraExpanded.otf` (gitignored) for Akira. The picker also has Bricolage Grotesque, Saira Expanded, Anybody ExtraExpanded, and Unbounded.

**Bodoni Moda and Big Shoulders Display were removed** on measurement. Rasterised at 400px and measured off the outline: Bodoni's thinnest stroke is **0.3% of cap height** (one pixel at 299px caps) with 94:1 thick/thin contrast, and a didone hairline is the first thing 4:2:0 chroma subsampling and inter-frame compression destroy. Big Shoulders is the only condensed face on the list (advance/cap **0.61**) and has the thinnest stems (**18.8%** of cap), which closes counters at wall viewing distance. Both are in `RETIRED_FONTS`, so a stored preference migrates. Everything that stayed measures 19-30% thinnest-stroke with near-monolinear contrast:

| Face | Stem % cap | Thinnest stroke % cap | Stroke contrast | Advance/cap |
|------|-----------|----------------------|-----------------|-------------|
| Akira Expanded | 38.9 | 30.3 | 1.23 | 1.40 |
| Unbounded 900 | 34.4 | 29.4 | 1.13 | 1.28 |
| Syne 800 | 39.2 | 28.0 | 1.37 | 2.01 |
| Anybody ExtraExpanded 900 | 42.0 | 26.2 | 1.57 | 1.68 |
| Saira Expanded 900 | 38.7 | 25.7 | 1.47 | 1.17 |
| Bricolage Grotesque 800 | 24.7 | 19.0 | 1.29 | 0.97 |

- Akira is **exactly 2x the width of Anton** at the same size. It is width-bound, not height-bound.
- Line length is the dominant lever. `IS BLACK CULTURE` (16 chars) renders at 126px; `IS HOUSE` (8 chars) at 268px. **Keep lines to ~9 characters.** The `Thesis` sequence is already written to that constraint.
- The file on disk is the **Demo**. Do not commit it. Self-hosting on GitHub Pages is redistribution. A commercial webfont licence is needed before it goes in front of an audience. It has **no accented characters** — an artist name with a diacritic falls back mid-word.
- Retired faces (Anton, Bebas, Audiowide, Orbitron, Oswald, Russo One, Black Ops One, Bungee, Dela Gothic One, Archivo Black) migrate to Akira on load.

### Material (video + Word Ladder)

Color video uses **poster ink** in `js/video-effects.js` `overlayDisplayText`: warm-white fill, thin dark hairline, and a dark ground from the soft channel in two parts — `OVERLAY_GROUND` (un-offset, symmetric) maxed with the offset contact shadow. The offset shadow alone left the side away from the offset separated from the footage by nothing but the hairline. Measured on a replica sheet: mean ground around the glyphs moves 0.400 → 0.427 (so the look barely changes) while the **worst-protected point goes 0.035 → 0.092**, which is the whole point. A stronger halo than that is bounded by `LEGACY_BLUR = 18` on a 2048-wide sheet, not by `OVERLAY_GROUND` — widening it means retuning a frozen number. Kick echoes gate on raw FFT (no time-fallback floor). Black & White still uses `overlayTitleCard`. Word Ladder matches the same cut.

---

## Rights

Do not put Chuck Roberts' spoken-word sermon from Rhythm Controll's *My House* (1987) on a wall without clearance. It is a copyrighted work and displaying it at a ticketed event is a public performance of the lyric. The `Thesis` sequence is written to sit in that register without borrowing a word of it.

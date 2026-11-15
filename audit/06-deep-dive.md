# Deep dive — 15 November 2026

The first pass covered hygiene. This one asks three harder questions:

1. What happens on show night when something goes wrong?
2. What does a shader actually receive when the music plays?
3. Is the system steering itself from the right information?

Numbers marked **measured** were taken in the in-app browser on this laptop (AudioContext at 44.1 kHz). Everything else is from reading the code. Nothing was tested on a real projector, with a real mic, or with the operator window genuinely hidden.

## Status — 15 November 2026

Sections 1 to 6 have been implemented (uncommitted at the time of writing). What was checked in the in-app browser, with a tone generator standing in for the mic and the projector page in a frame:

| Fix | Checked | Not checked |
|-----|---------|-------------|
| Projector reloads itself after context loss and reconnects | Yes, by forcing a loss | A real GPU reset or display hot-plug |
| Operator reload restores shader, mic and sequencer state | Shader restore after a forced context loss | Mic auto-start (the pane blocks the mic); sequencer resume |
| Projector's frame drives housekeeping when the operator loop is idle | Shader promotion and sequencer steps (3 steps in 3.3 s, no double fires) | With the operator window really hidden behind another app |
| One `iTime` for both windows | Yes, identical to the millisecond | — |
| Time-based audio envelope | Decay rate matches the design value at a 50 ms read interval, where the old code decayed about three times slower | By ear, against music |
| Band uniforms and kick onset | Kick band reads 1 and high band 0 on a 55 Hz tone; one onset on a held note, four on four short kicks | Real kicks in a room |
| Mic voice processing off, input picker, input-lost warning | Constraints are passed; picker renders | Device switching and unplugging |
| Scale controller steered by the projector | Projector frames and GPU time reach the controller | Whether it holds 60 fps on a heavy shader at projector resolution |
| Person mask in a Worker | 0.3 ms per call on the main thread, down from about 7 ms | The in-page fallback; a live camera |

Added later the same day:

| Fix | Checked | Not checked |
|-----|---------|-------------|
| Operator rebuilds its GL objects in place on context loss (reload is now only the fallback) | Forced a loss with a video effect and feedback running: no reload, mic kept listening, both windows redrew | A real GPU reset; the reload fallback |
| PIANOSCOPE shaders (and Procgen Cats) read `iKick`/`iBass`, `iLowMid`/`iMid`, `iHigh` through `getBass()`/`getMid()`/`getHigh()` | All compile; Four on the Floor draws against a synthetic kick | The look. The `BASS_TRIM` / `MID_TRIM` / `HIGH_TRIM` values at the top of each shader are estimated from a typical music spectrum, not tuned by eye |
| Person mask falls back to the page when the Worker fails | Forced the failure; a mask was produced | — |
| `dev/compile-all.html` compiles every shader and effect | 157 of 157 compile and link | — |
| Docs no longer teach WebGL 1 rules | Read the agent's change list | Not proofread line by line |

`iKickOnset` is available but no shader uses it yet.

## Verdict

The code is careful at the level of individual functions. The weaknesses are in how the pieces relate:

- **The operator window owns everything, but each subsystem behaves differently when that window is hidden.** Some keep running, some run late, some stop.
- **Recovery from a GPU hiccup is "reload the page", and a reload forgets the show.** The projector window has no recovery at all.
- **The audio signal reaching shaders is shaped by accident.** Its decay depends on the display's refresh rate, and the band called "bass" does not contain the kick.
- **The adaptive resolution controller measures the preview and steers the projector.**

## 1. Recovery: a GPU hiccup resets the show

| Event | What the code does | What the room sees |
|-------|--------------------|--------------------|
| Operator window loses its GL context | Waits for restore, then `location.reload()` ([visualizer-with-shadertoy.js:187](../js/visualizer-with-shadertoy.js:187)) | Default shader, mic off |
| Projector window loses its GL context | Nothing. No `webglcontextlost` handler in [kiosk-window.js](../js/kiosk-window.js) | Black until someone reloads that window by hand |
| Operator reloads (Cmd+R, crash, the reload above) | Boots to `Pianoscope Text` | Same as row one |
| Mic or audio interface unplugged, or the system input changes | Nothing. No `ended` or `devicechange` handling | Visuals quietly drop to their fallback motion |

Context loss is not exotic here. The app asks for the discrete GPU, and the code's own comment says that makes loss more likely. Plugging in a projector is a display change on the same GPU, which is the moment it tends to happen.

What a reload loses:

- **The current shader.** Its name is not stored anywhere. Text, font, style, tempo and video session are; the shader is not.
- **A live-edited shader.** The editor session stores only whether the panel is open and which tab is showing.
- **The mic.** Permission persists in the browser, but the app waits for a click on Start Listen.
- **Whether the sequencer was running.**

There is also no way to choose an input. `getUserMedia({ audio: true })` takes the system default, so feeding the visuals from a mixer or audio interface means changing the macOS default input before opening the page.

**Direction**

- Keep one "show state" record (shader name or edited source, mic on plus device id, sequencer running, video, text) and restore from it on load. Most of the pieces are already stored separately; the shader and the mic are the gaps.
- Rebuild in place on context restore instead of reloading. With `gl-shared.js` this is: recreate textures, recompile the current source, carry on. Add the same handler to the projector window.
- Add an input picker (`enumerateDevices`), store the device id, and show a visible warning when the track ends.

## 2. Who owns time

The projector pulls everything from the operator window. That was the right fix for audio. But the operator window is the one that gets buried under DJ software, and each subsystem handles that differently:

| Subsystem | Driven by | When the operator window is hidden |
|-----------|-----------|------------------------------------|
| Audio spectrum | Operator `rAF`, with a pull from the projector as backstop | Keeps working, but its decay changes (section 3) |
| Text transitions | Wall clock, read by the projector | Keeps working |
| Sequencer steps | `setTimeout` in the operator window ([text-sequencer.js:162](../js/text-sequencer.js:162)) | Fires late. Chrome clamps timers in hidden pages to once a second, and can go much lower after five minutes. Whether mic capture exempts the page from the harsher mode is something I could not verify |
| Person mask | Operator render loop only ([visualizer-with-shadertoy.js](../js/visualizer-with-shadertoy.js), `updatePersonMask`) | Freezes on the last matte |
| Switching between a shader and a video effect | Operator render loop requests the program | Does not happen until the window is visible again |
| Adaptive resolution | Operator render loop | Stops; scale stays where it was |
| `iTime` | Each window's own clock | Unaffected, but see below |

`iTime` has two origins. The projector's starts when its window loads. The operator's starts at page load and is reset when the mic starts ([visualizer-with-shadertoy.js:781](../js/visualizer-with-shadertoy.js:781)). So the preview and the projector are never showing the same moment of a time-driven shader. The operator cannot cue off the preview.

**Direction: let the visible window drive.** Both windows share one JavaScript thread (section 5), so the projector can call a single `pianoscopeTick(now)` on the operator window at the top of each of its frames. That tick does the audio read, the person-mask update, the program switch check and the sequencer due-check. When no projector is open, the operator's own loop calls it. One clock origin, published by the operator and read by the projector, fixes `iTime`. This is a small change and removes the whole table above.

## 3. The audio envelope is an accident of the display

The analyser runs with `smoothingTimeConstant = 0.8`. Web Audio applies that smoothing once per *read*, not per unit of time, and the result is then mapped to decibels. I fed a tone into an analyser configured like the app's and read it at two rates (**measured**):

| Tone | Reads per second | Rise to 90% | Fall to 50% | Fall to 10% | Level one beat later (124 BPM) |
|------|------------------|-------------|-------------|-------------|-------------------------------|
| 55 Hz | 60 | 31 ms | 336 ms | 576 ms | 22% of peak |
| 55 Hz | 120 | 24 ms | 175 ms | 296 ms | 0 |
| 440 Hz | 60 | 32 ms | 351 ms | 575 ms | 22% of peak |
| 440 Hz | 120 | 24 ms | 176 ms | 296 ms | 0 |
| 55 Hz, smoothing 0 | 60 | 16 ms | 32 ms | 32 ms | 0 |

The test tone was loud enough to saturate, so real music will fall to a given level somewhat sooner. The factor of two between rows is exact.

What this means:

- **The attack is fine.** The docs call the transients "mushy", but the rise takes two frames. The decibel mapping hides the smoothing on the way up.
- **The release is the problem.** At 60 reads a second a kick has not cleared by the time the next one lands. The signal rides high and pulses weakly on top. That is why every shader needs `audioBoost()` and gains "1.5–2× higher than expected".
- **The look depends on the operator's screen.** A 120 Hz MacBook panel reads twice as often as a 60 Hz one, so the same shader decays twice as fast. The projector reuses the operator's reads, so the laptop's refresh rate decides how the projector looks.
- **Hiding the operator window changes the look.** The projector then does the reading at its own rate. On a 120 Hz laptop driving a 60 Hz projector, decay time doubles the moment the operator window is covered.
- **It blocks beat detection.** The sequencer's header says tap tempo was chosen because this smoothing flattens the onsets a detector needs.

**Direction:** set the analyser's smoothing to 0 and apply the envelope in `sampleAudio()` with time-based coefficients (`1 - exp(-dt / tau)`), separate for attack and release. The response is then the same on every display and in every window state, and the release becomes a tuning knob instead of a side effect. The raw, unsmoothed spectrum is available at the same point for onset detection.

## 4. "Bass" does not contain the kick

All nine `PIANOSCOPE` shaders use the same twelve probe points from [knowledge/howto/02-audio-reactivity.md](../knowledge/howto/02-audio-reactivity.md). The texture is 256 linear bins of about 43 Hz each (at 44.1 kHz; about 9% higher at 48 kHz). That puts the probes here:

| Helper | Probe positions | Frequencies sampled |
|--------|-----------------|---------------------|
| `getBass()` | 0.01, 0.03, 0.05, 0.07 | about 110, 330, 550, 770 Hz |
| `getMid()` | 0.15, 0.25, 0.35, 0.45 | about 1.65, 2.75, 3.85, 4.95 kHz |
| `getHigh()` | 0.55, 0.70, 0.85, 0.95 | about 6.1, 7.7, 9.4, 10.5 kHz |

- A kick's fundamental sits at 45–70 Hz, which is texel 1 of 256 (x ≈ 0.005). No probe reads it. "Bass" is really the low-mid: chords, vocals, snare body.
- Everything below 250 Hz is six texels. The linear layout spends half the texture on 5.5–11 kHz, where music has little energy.
- "Mid" and "high" sit above most of a piano's fundamentals. They mostly see harmonics and hi-hats, so they read low and need the large gains the docs recommend.
- Each helper averages four single bins out of 512. That is a narrow, noisy sample of a band, not its energy.

A shader named Four on the Floor is therefore not driven by the kick. It may still look right in a loud room, because a kick's attack spreads energy upward, but it will also pulse on a bass line or a vocal.

**Direction:** compute named features once in JS, from the raw spectrum: a kick band (about 40–110 Hz), bass, low-mid, mid, high, each a proper sum over its bins, plus an onset flag. Expose them as uniforms through `gl-shared.js`, which now reaches both windows from one place. Keep the existing texture as it is so the 60 library shaders that read it do not change. Then move the PIANOSCOPE shaders onto the features one at a time, which is a re-tune of each.

Do this together with section 3 and the mic-processing change from the first audit. All three change how every shader responds, so they should cost one re-tune, not three.

## 5. Both windows run on one thread

The projector reads the operator's objects directly (`window.opener.videoController`, the display-text object, the audio array). Browsers only allow that between windows in the same process, sharing one main thread. So any long task in the operator window delays the projector's next frame.

Ordinary frames are cheap (**measured**):

| Work | Main-thread time |
|------|------------------|
| One projector frame, including upload of a 2850×1594 video frame | 0.2 ms typical, 0.6 ms worst |
| The operator's upload of the same frame | 0.2 ms typical, 0.5 ms worst |
| One text change (rasterise both sheets) | 1.5–6 ms |
| One person-mask inference | 6.8 ms typical, 9.1 ms at the 90th percentile, 18 ms worst |

I expected the duplicated video upload to be a cost. It is not; drop that idea.

The person mask is the real one. It runs on the CPU (`delegate: 'CPU'`) on the main thread, up to 20 times a second. With a body or room blend on, roughly every third frame loses 7–9 ms of a 16.7 ms budget, on the thread that also has to produce the projector's frame. It also allocates a fresh float array per inference.

Not measured: what the operator's own actions cost while the projector is live (typing in the editor, opening the shader list, library thumbnails loading).

**Direction:** move the segmenter into a Worker, fed with `ImageBitmap`s, posting the mask back. Keep the shared process: it is what makes one video decoder feed both windows for free, and the alternative (separate processes, message passing, a second decode) costs more than it saves.

## 6. The resolution controller watches the wrong window

[perf.js](../js/perf.js) runs a GPU timer and a scale controller on the operator's context. The projector has neither; it asks the operator's controller what size to be.

While the projector is open, the preview drops to half resolution, a quarter of the pixels. So the controller is timing a quarter-size workload on the laptop panel and using the answer to size a full workload on a different display. It will see plenty of headroom and hold scale at maximum while the projector misses frames. The only things protecting the projector are the fixed pixel cap and the name-based `HEAVY_NAME` guess.

It also stops entirely when the operator window is hidden.

**Direction:** when a projector is live, run the timer and controller there. `GpuTimer` and `ScaleController` are self-contained classes and can move into `gl-shared.js`. The projector owns the scale; the preview follows.

## 7. Smaller observations

- **Long sets and `iTime`.** `iTime` is a 32-bit float of seconds. After about four and a half hours its resolution is 2 ms, and shaders that multiply time by large factors before `fract()` or `sin()` will start to step. Reloading resets it. Worth knowing for a long night; not worth engineering around yet.
- **`iFrame` differs between windows.** The operator resets it on every shader switch; the projector never does. No shader in the library depends on frame zero today, so this is latent.
- **The dB readout is not decibels.** It is the log of the RMS of byte values that are already on a decibel scale. Harmless as a level meter, misleading as a number.

## Checked and found sound

- No library shader depends on Shadertoy's waveform row. Sixteen sample a non-zero row, all of which land on the same spectrum row here.
- Text transitions are derived from the wall clock with a timer backstop, so they complete on the projector with the operator hidden.
- The sequencer aims each step at an absolute time and skips missed slots, so late timers do not accumulate drift.
- Shader switching never blocks a frame in either window.
- The projector reconnects after an operator reload.

## Suggested order

| # | Change | Size | Why here |
|---|--------|------|----------|
| 1 | Context-loss handler in the projector; persist shader and mic state; restore on load | Half a day | Turns a show-stopping failure into a blink |
| 2 | Single tick driven by the visible window; shared clock origin | Half a day | Removes every "depends on whether the operator window is visible" case |
| 3 | Time-based audio envelope, named band features, mic processing off | A day, plus re-tuning | One re-tune for three changes; unlocks beat detection |
| 4 | Scale controller in the projector | Half a day | Makes adaptive resolution protect the screen the room sees |
| 5 | Person mask in a Worker | Half a day | Only matters when the body and room blends are in use |
| 6 | Input device picker and track-ended warning | A few hours | Needed once the source is a mixer, not the laptop mic |

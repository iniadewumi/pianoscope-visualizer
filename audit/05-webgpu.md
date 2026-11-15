# WebGPU: what it would buy, and what it would cost

14 November 2026. Research plus one probe of this machine; no WebGPU code has been written for the project yet. Where a claim comes from general knowledge rather than a source or a test, it says so.

## Short answer

Yes, worth doing, but as a **second backend for new shaders**, not a migration of the 135 you have. And be clear about where the win is: a fullscreen fragment shader looks the same in WebGPU as in WebGL 2. The shaders get *better* because WebGPU lets them do things a fragment-only pipeline cannot: simulate, remember and analyse.

## Can you use it?

| Check | Result |
|-------|--------|
| This machine, in the in-app browser (Chromium 156) | `navigator.gpu` present, adapter is Apple Metal 3. Features include `shader-f16`, `subgroups`, `timestamp-query`, `float32-filterable`, `float32-blendable`; 1024 compute invocations per workgroup, 32 KB workgroup memory |
| Chrome / Edge | On by default on desktop since version 113 ([web.dev](https://web.dev/blog/webgpu-supported-major-browsers), [implementation status](https://github.com/gpuweb/gpuweb/wiki/implementation-status)) |
| Safari | Safari 26 on macOS and iOS |
| Firefox | Windows and Apple Silicon Macs; Linux and Intel Macs still in progress. Sources disagree on exact versions |

Kiosk mode already needs Chrome (the Window Management API that puts the projector window on the second screen), so Firefox gaps do not matter for shows. The one thing to confirm is the **show laptop**: run `navigator.gpu.requestAdapter()` in its Chrome and check that `timestamp-query` is listed.

## What WebGPU would add to the shaders

Ordered by how much it changes what you can make.

### 1. Compute shaders: state that persists and evolves

Today every shader is one fragment pass, and the only memory is the ping-pong `iFeedback` image. That limits you to pictures computed fresh each frame from time and audio.

With compute you can run:
- particle systems of tens of thousands to millions of points, with positions kept on the GPU between frames;
- reaction-diffusion, fluid and cellular automata, where the whole field is updated by the music;
- flocking and agent systems (boids, slime moulds) where the audio changes the rules, not just the colours;
- audio-driven geometry: thousands of crisp instanced shapes whose positions come from a simulation.

This fits your flat, crisp style better than the usual "glowing particle cloud" demos. Draw simulated points as anti-aliased SDF discs, pills or bars (your CRESCENDO set already thinks in pills and skylines), and you get dense motion with hard edges.

### 2. Real audio data, and analysis done once

Today the spectrum reaches shaders as a **256-texel, 8-bit** `LUMINANCE` texture, so 255 levels per bin, and every shader derives its own bass, mid and high from it. In WebGPU:
- upload the spectrum as a **float storage buffer**, with no 8-bit quantisation, at whatever bin count you choose;
- run a small **compute pass per frame** that computes shared features once: band energies, smoothed loudness, a kick onset flag, a beat phase;
- hand shaders a struct (`audio.bass`, `audio.kick`, `audio.beat`) instead of making each re-sample and re-threshold the texture.

This is the item from [03-long-term.md §7](03-long-term.md) done properly, and it benefits every new shader whatever it draws. Note the mic still goes through the Web Audio `AnalyserNode` (CPU FFT); a GPU FFT is possible ([wgsl-fft](https://docs.rs/wgsl-fft) is a Rust implementation of the algorithm) but at 1024 bins the CPU FFT is not the bottleneck, so I would skip it.

### 3. Multi-pass pipelines without ping-pong plumbing

Shadertoy-style buffers A–D (several passes feeding each other) are awkward in WebGL 2 and are why feedback is a special case in the runtime. In WebGPU a "pass" is a few lines, storage textures can be written and read in compute, and the HDR handling you wrote by hand (`FEEDBACK_HDR`, probing float formats) becomes a normal texture format choice.

### 4. Cheaper video

- `importExternalTexture()` gives a zero-copy view of the current video frame, where WebGL does a `texSubImage2D` copy per frame, per window. It is not Baseline in MDN's compatibility data and throws for cross-origin video ([MDN](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/importExternalTexture)); your clips are same-origin or cached blobs, so that is fine, but the R2 clips would need to come via the cache path.
- The external texture is restricted (sampled only one way). If a shader needs to keep or process the frame, copy it into a normal texture with `copyExternalImageToTexture` instead.
- Camera and person-mask paths work the same as today: `getUserMedia` into a `<video>`.

### 5. Better performance tools

- `timestamp-query` gives real per-pass GPU timings and replaces `EXT_disjoint_timer_query_webgl2`, which `perf.js` uses and which falls back to "gpu n/a" where unavailable. Your adaptive resolution would be steering on measured cost instead of frame-time inference. Worth confirming that the WebGL extension is even exposed on the show laptop today.
- `shader-f16` and `subgroups` are available here, which can speed up heavy raymarchers. (From general knowledge: real gains depend on the shader, so treat as something to measure.)

### What it does not do

- It does not make an existing raymarcher faster by itself. Expect rough parity for the same algorithm.
- It does not remove the need for the "flat and crisp" discipline in your style notes. More power makes noisy, blotchy output easier to produce, not harder.

## What it would cost

### The 135 existing shaders

They are GLSL ES 3.00 written for Shadertoy. WebGPU wants WGSL, which is not a dialect of GLSL, and I could find no ready-made Shadertoy-to-WebGPU converter. The workable route is GLSL → SPIR-V → WGSL with [naga](https://docs.rs/crate/naga) or glslang plus Tint, and a Babylon.js forum thread notes that GLSL 1.00/3.00 has to be raised to 4.x first ([forum](https://forum.babylonjs.com/t/can-i-use-the-tintwasm-library-in-the-babylon-js-repository/23673)). I have not tried it; none of those tools is installed here.

What I measured in your shaders, since porting difficulty is about which language features they lean on:

| Feature | Library (124) | Your PIANOSCOPE set (11) |
|---------|---------------|--------------------------|
| Use `#define` (WGSL has no preprocessor) | 86, of which 44 define function-like macros | 11 |
| Have `out` / `inout` helper parameters | 39 | 1 |
| Sample textures (mostly the audio texture) | 59 | 9 |
| Use derivatives (`fwidth`, `dFdx`) | 10 | 7 |

Known WGSL differences that hit Shadertoy code, from my own knowledge of the spec rather than a test:
- No implicit int/float conversion and no scalar-condition ternary on vectors (use `select`).
- `textureSample` and derivatives must be in uniform control flow, and Shadertoy code often samples inside branches. The fixes are `textureSampleLevel`, or the `diagnostic(off, derivative_uniformity)` directive.
- A converter would expand every macro, so translated shaders become readable only as machine output.

Realistic outcome: automatic translation would work for a fair share and need hand fixes for the rest. That is a long tail with little payoff, because those 124 shaders already look the way they look. **Leave them on WebGL 2.**

### The runtime

- **A canvas can only ever hold one context type**, so a WebGL 2 shader and a WebGPU shader cannot share a canvas. Both windows would hold two stacked canvases and swap which is visible when a shader of the other backend is adopted. A cut between backends mid-show should be planned for (adopt only once the new pipeline is ready, as you already do for compiles).
- The converter's preamble (`videoUV`, `personAt`, the text uniforms, `pianoscopeBlendVideo`, `SEQ_GLSL`) needs a WGSL twin, about 250 lines. They are the contract new shaders are written against, so it is worth doing carefully once.
- The kiosk architecture carries over: the projector pulls from the operator window and owns its own `GPUDevice`. The seam you just created, `js/gl-shared.js`, is where a `js/gpu-shared.js` would sit beside it.
- **Simulations will drift between the two windows**, because each runs its own state from slightly different audio timing. Feedback shaders already do this slightly. For the projector that is fine; for the operator preview, consider running sims at low resolution or showing a frozen frame while the kiosk is open.
- A pane note: the in-app browser here does expose WebGPU, so the compile/draw path can be tested in it, as can compute. Fullscreen and the second screen still need real hardware.

## Spike results

Built on `feature/webgpu` in `dev/` (see [dev/README.md](../dev/README.md)). Measured on this Mac (Apple Metal 3).

- **A hand port matches.** `PIANOSCOPE Hairpin` ported to WGSL and compared against the real WebGL 2 shader, at seven times between 0.2 s and 300 s: six frames differ by at most 1/255, and one has a single pixel off by 80. It compiled in Chrome on the first try, `fwidth` and texture sampling included.
- **What the port had to change.** The fragment origin is top-left, not bottom-left. The shader's reversed `smoothstep(a, b, x)` with `a > b` is undefined in both languages, so it is written out as a formula. `pow(x, 2.0)` with a possibly negative `x` became a multiply. Scalar ternaries became `select`. The 8-bit linear audio texture lookup was written by hand against a float buffer. These are the kinds of edits the 124 library shaders would need in bulk.
- **The compute pieces work.** A one-thread pass turns the spectrum into band levels and a decaying kick onset, and a particle sim reads it to fire particles from the hairpin's point on each kick.
- **Throughput at 1280x720** (wall time per frame): 100k particles 0.77 ms, 500k 4.5 ms, 1M 9.6 ms, 2M 21.8 ms. So about a million particles fit a 60 fps budget on this machine, with room left over; two million do not.
- **Correction to the timing claim above.** Per-pass GPU timestamps worked, but they sum to more than the wall time (at 500k the render pass reports 11.5 ms for a 4.5 ms frame), so the passes overlap and the per-pass numbers overstate. Use wall-clock throughput or whole-frame timing to steer the resolution governor, not individual passes.
- **A style note.** At 500k particles of 1.3 px each, the plume reads as a dense gradient, not as separate crisp dots. That is a look question for step 3, not a limit of the technique.

Still unchecked: a real microphone, the kiosk window, `importExternalTexture` video, and fullscreen. The first look at the output was thumbnails only because the browser pane was 86 px tall.

## Suggested plan

| Step | What | Rough effort | Exit test |
|------|------|--------------|-----------|
| 0. Spike | A test page: a WebGPU canvas, float audio buffer, one compute-driven demo, and one hand-ported CRESCENDO shader (Hairpin) beside its WebGL 2 original | 1 day | Same picture within rounding; compute demo holds 60 fps at the kiosk's real resolution |
| 1. Backend | `js/gpu-shared.js`, WGSL preamble, backend flag on shaders, two-canvas swap in both windows | 3–4 days | Switch between a WebGL 2 and a WebGPU shader with no flash on the projector |
| 2. Audio features | Compute pass producing the audio struct; expose to WGSL shaders | 1–2 days | A shader keyed on `audio.kick` locks to the beat better than the threshold in Hairpin |
| 3. New work | First simulation shaders in the CRESCENDO flat style | Open-ended | You like it on the projector |
| 4. Optional | Port favourites from the library by hand or through naga | Per shader | None needed |

If you only do step 0, you find out whether the premium is real for your taste before committing to the backend work.

## Sources

- [WebGPU is now supported in major browsers (web.dev)](https://web.dev/blog/webgpu-supported-major-browsers)
- [WebGPU implementation status (gpuweb wiki)](https://github.com/gpuweb/gpuweb/wiki/implementation-status)
- [GPUDevice.importExternalTexture (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/importExternalTexture)
- [naga crate documentation](https://docs.rs/crate/naga)
- [Babylon.js forum: GLSL 1.00/3.00 to WGSL through naga and Tint](https://forum.babylonjs.com/t/can-i-use-the-tintwasm-library-in-the-babylon-js-repository/23673)
- [wgsl-fft (Stockham FFT in WGSL, for wgpu)](https://docs.rs/wgsl-fft)

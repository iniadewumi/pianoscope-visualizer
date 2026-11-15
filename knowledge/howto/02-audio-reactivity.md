# Audio Reactivity

**Keywords:** iChannel0, FFT, getBass, getMid, getHigh, audioBoost, y=0.0, fallback

**Use when:** Any PIANOSCOPE shader; copying helpers from `shaders.js` (dangerous).

---

## Pipeline (this repo ≠ Shadertoy default)

From `js/visualizer-with-shadertoy.js`:

| Setting | Value | Implication |
|---------|-------|-------------|
| `analyser.fftSize` | 1024 | 512 bins; first 256 uploaded |
| Texture | 256×1 luminance | One row only |
| Sample row | **`y = 0.0`** | Always |
| `audioSensitivity` | 1.5 | Pre-amplified in JS |
| Smoothing | time-based envelope, `AUDIO_ATTACK_MS` / `AUDIO_RELEASE_MS` (74.7 ms each) | Same response on every display (analyser smoothing is off). `AUDIO_RELEASE_MS` is the knob for kick separation: shorten it for punchier kicks |

Mic voice processing (echo cancellation, noise suppression, auto gain) is now off by default (`MIC_VOICE_PROCESSING`).

**Critical:** Many `shaders.js` entries use `texture(iChannel0, vec2(x, 0.25))` or other rows. **Wrong for this visualizer.** Do not copy their audio sampling.

---

## Band uniforms

Computed in JS from whole bands (RMS over the band's bins, same -100..-30 dB scale, times `audioSensitivity`, clamped); all 0 while the mic is off. `iChannel0` is unchanged.

| Uniform | Range |
|---------|-------|
| `iKick` | 40-110 Hz, 0..1 |
| `iBass` | 60-250 Hz, 0..1 |
| `iLowMid` | 250-800 Hz, 0..1 |
| `iMid` | 800 Hz-3 kHz, 0..1 |
| `iHigh` | 3-11 kHz, 0..1 |
| `iKickOnset` | 1 on a kick, decaying to 0 over ~150 ms |

The helpers below read these uniforms. The older helpers point-sampled the texture at about 110-770 Hz, 1.6-5 kHz and 6-10.5 kHz, so "bass" never contained the kick; the PIANOSCOPE shaders in `js/test-shaders.js` were moved off them on 2026-11-15. `iKickOnset` is available for anything that should fire once per kick; no shader uses it yet.

---

## Standard helpers (include in every TEST_SHADERS entry)

```glsl
// Keep fft() for anything that needs the spectrum's shape (a spectrum ring,
// per-key levels, spectrumStats()).
float fft(float x) {
    return texture(iChannel0, vec2(clamp(x, 0.0, 1.0), 0.0)).x;
}

// Whole bands read hotter than the old point probes did. The trims bring each
// band back to the level existing gains were tuned against (estimated from a
// typical music spectrum, not measured in a room): tune them by eye.
#define BASS_TRIM 0.20
#define MID_TRIM  0.23
#define HIGH_TRIM 0.04
float getBass() { return clamp(max(iKick, iBass) - BASS_TRIM, 0.0, 1.0); }   // kick and bass line
float getMid()  { return clamp(max(iLowMid, iMid) - MID_TRIM, 0.0, 1.0); }   // chords
float getHigh() { return clamp(iHigh - HIGH_TRIM, 0.0, 1.0); }               // hats

float fallbackBass() { return 0.35 + 0.25 * sin(iTime * 1.2); }
float fallbackMid()  { return 0.28 + 0.18 * sin(iTime * 0.7 + 1.0); }
float fallbackHigh() { return 0.18 + 0.12 * sin(iTime * 2.8 + 2.0); }

float safeAudio(float value, float fallback) {
    return max(value, fallback * 0.35);
}

float audioBoost(float v) {
    return clamp(pow(v, 0.75) * 1.35, 0.0, 1.0);
}
```

---

## Usage pattern in mainImage

```glsl
float bass = audioBoost(safeAudio(getBass(), fallbackBass()));
float mid  = audioBoost(safeAudio(getMid(), fallbackMid()));
float high = audioBoost(safeAudio(getHigh(), fallbackHigh()));
```

Fallback sines are **required** — shader must animate when mic is denied or silent.

---

## Tuning gains

- Start `#define` gains **~1.5–2× higher** than expected; tune down after live mic test.
- Because the default envelope is 74.7 ms, use `audioBoost()` to restore punch.
- Expose gains as `#define` at top of shader string for quick tuning.

Example (Afrotech polish pass):

```glsl
#define RING_BASS_GAIN   0.30
#define MID_ROT_GAIN     0.55
#define PATH_MID_GAIN    0.70
#define HIGH_SPARK_GAIN  0.65
```

Genre-specific mappings → [`06-genre-shader-recipes.md`](06-genre-shader-recipes.md).

---

## Genre → audio mapping (summary)

| Genre | Bass | Mid | High |
|-------|------|-----|------|
| Afrotech Settlement | Ring/dwelling expansion | Layer rotation, paths | Sparks at node centers |
| Mudcloth Bloom | Clay bloom threshold opens | Cloth warp / mark bend | Sand speckles |
| Kente Loom | Horizontal band thickness, bounce | Vertical band offset | Thread shimmer on edges |
| Cornrow Curves | Braid width | Curve phase drift | Bead glints along curves |

Cornrow also uses **triplet pulse** from `iTime` (not FFT) — see genre recipe.

---

## Alternative: band averaging

If `audioBoost()` still feels dead, study **Fractal Sounders** in `shaders.js` — `lowAverage()` / `highAverage()` over multiple bins.

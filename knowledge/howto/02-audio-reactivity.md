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
| `smoothingTimeConstant` | **0.8** | Heavy lag — transients feel mushy |

**Critical:** Many `shaders.js` entries use `texture(iChannel0, vec2(x, 0.25))` or other rows. **Wrong for this visualizer.** Do not copy their audio sampling.

---

## Standard helpers (include in every TEST_SHADERS entry)

```glsl
float fft(float x) {
    return texture(iChannel0, vec2(clamp(x, 0.0, 1.0), 0.0)).x;
}

float getBass() {
    return (fft(0.01) + fft(0.03) + fft(0.05) + fft(0.07)) * 0.25;
}

float getMid() {
    return (fft(0.15) + fft(0.25) + fft(0.35) + fft(0.45)) * 0.25;
}

float getHigh() {
    return (fft(0.55) + fft(0.70) + fft(0.85) + fft(0.95)) * 0.25;
}

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
- Because smoothing is 0.8, use `audioBoost()` to restore punch.
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

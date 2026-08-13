# Genre Shader Recipes

**Keywords:** afrotech, mudcloth, kente, cornrow, amapiano, afrohouse, recipe, architecture, palette

**Use when:** Building or reviewing a specific PIANOSCOPE visual mode.

**Full cultural context:** [`cursor inst.md`](../../cursor%20inst.md) § PIANOSCOPE visual modes.

---

## Status overview

| Key | Genre | Status | Primary techniques |
|-----|-------|--------|-------------------|
| `PIANOSCOPE Afrotech Fractal Settlement` | Afrotech | Documented in learnings; verify in `test-shaders.js` | Fill+stroke rings, explicit layers |
| `PIANOSCOPE Afrohouse Mudcloth Bloom` | Afrohouse | **In `test-shaders.js`** | fBM, warp, brick-offset marks |
| `PIANOSCOPE Amapiano Kente Loom` | Amapiano | **In `test-shaders.js`** | fract grid, weave, stripes |
| `PIANOSCOPE 3-Step Cornrow Curves` | 3-Step | Not built | sine braids, triplet pulse |
| `PIANOSCOPE Adinkra Symbol Field Abstract` | Optional | Brief only | Radial arcs, smoke emergence |

---

## Palettes (copy-paste)

```glsl
// Afrotech
vec3 blackBlue = vec3(0.005, 0.015, 0.025);
vec3 deepCyan  = vec3(0.00, 0.55, 0.65);
vec3 electric  = vec3(0.10, 0.85, 1.00);
vec3 violet    = vec3(0.22, 0.08, 0.42);

// Mudcloth
vec3 charcoal = vec3(0.025, 0.020, 0.018);
vec3 clay     = vec3(0.55, 0.22, 0.08);
vec3 rust     = vec3(0.75, 0.30, 0.10);
vec3 ochre    = vec3(0.86, 0.58, 0.24);
vec3 sand     = vec3(0.90, 0.73, 0.48);

// Kente
vec3 midnightViolet = vec3(0.13, 0.06, 0.25);
vec3 deepBlack      = vec3(0.015, 0.010, 0.018);
vec3 rustOrange     = vec3(0.70, 0.27, 0.10);
vec3 mutedGold      = vec3(0.83, 0.57, 0.18);
vec3 cream          = vec3(0.92, 0.80, 0.56);

// Cornrow
vec3 blackBrown = vec3(0.045, 0.025, 0.012);
vec3 deepGold   = vec3(0.86, 0.55, 0.16);
vec3 creamGold  = vec3(0.96, 0.80, 0.48);
vec3 shadow     = vec3(0.08, 0.05, 0.025);
```

---

## 1. Afrotech Fractal Settlement

**Feeling:** Dark, recursive, architectural, bass-heavy, future-facing — not sci-fi HUD.

**Audio:** Bass expands rings/dwellings · Mid rotates layers / paths · High sparks at **dwelling centers only**

**Architecture (winning v3 — reuse this skeleton):**

```
mainImage
├── centeredUV + audioBoost(bass, mid, high)
├── global slow rotation (mid)
├── background: blackBlue + fBm smoke (~0.07) + radial violet
├── settlementLayer × 3 (ringR, nodeCount, layer, rot)
│   ├── per-layer UV skew (asymmetry)
│   ├── ringWallGated (compound perimeter + pathway break)
│   ├── for each node: filledDisc, rectCompound (layers 0+2), miniRing, branchPath, spark
├── outerBound ring, radial spokes (offset, not center-only lasers)
├── courtyard mask (dark center ~40–60% negative space)
├── vignette → sqrt tone map, cap ~0.82
```

**Key functions:** `sdfFill`, `sdfStroke`, `ringWallGated`, `filledDisc`, `branchPath`, `rectCompound`, `settlementLayer`

**Layer config (reference):**

| Layer | ringR×scale | Nodes | Rect compounds |
|-------|-------------|-------|----------------|
| 0 inner | 0.24 | 6 | Yes |
| 1 mid | 0.46 | 9 | No |
| 2 outer | 0.70 | 11 | Yes |

**Steal from:** `"Apollonian Gasket"` — fill/stroke. Motion from `"Pianoscope"` — **not** edge-only `df()` rendering.

**Full postmortem:** `PIANOSCOPE_SHADER_LEARNINGS.md` § Iteration postmortem.

---

## 2. Afrohouse Mudcloth Bloom

**Feeling:** Organic, earthy, hand-painted, asymmetric — **not** clean wallpaper or vector pattern.

**Audio:** Bass opens blooms · Mid warps cloth / bends marks · High sand speckles

**Architecture (shipped v1):**

```
mainImage
├── centeredUV + audioBoost
├── wuv = warpCloth(uv, mid)          // single-level IQ warp
├── blooms = clayBlooms(wuv, bass)    // thresholded fBM
├── marks  = clothMarks(wuv, mid)     // brick-offset cells + hash motifs
├── speck  = sandSpeckles(uv, high)
├── mudPalette(tone) + grain + vignette
```

**Cell marks:** zigzag, roughLine, cross, ring — selected by `hash21(id)`.

**Upgrades (not yet required):** double domain warp; Voronoise bloom field; IQ uniform crack lines.

**Reference implementation:** `js/test-shaders.js` → `"PIANOSCOPE Afrohouse Mudcloth Bloom"`.

**Techniques:** [`04-pattern-tiling-truchet.md`](04-pattern-tiling-truchet.md) + [`05-iquilez-sdf-noise-warp.md`](05-iquilez-sdf-noise-warp.md).

---

## 3. Amapiano Kente Loom

**Feeling:** Woven, syncopated, nighttime warmth, log-drum bounce.

**Audio:** Bass thickens horizontal bands + vertical bounce · Mid offsets vertical bands · High thread shimmer on edges

**Architecture (to build):**

```
mainImage
├── centeredUV + audioBoost
├── vertical   = stripe(uv.x + sin(uv.y*8 + t)*midWobble, 10 + bass*3, width)
├── horizontal = stripe(uv.y + sin(uv.x*7 - t)*bassWobble, 8 + mid*2, width)
├── weave = mix(vertical, horizontal, overUnderMask)
├── cell UV → diagonal sdSegment accents (Truchet parity rotate)
├── syncopated breaks: hash(cell) gates → deepBlack
├── kentePalette(cellPhase, audio)  // cosine or hand-tuned mix
├── high * thin shimmer on band edges only
└── vignette, cap brightness
```

**Suggested #defines:**

```glsl
#define BAND_BASS_GAIN    0.35
#define OFFSET_MID_GAIN   0.50
#define SHIMMER_HIGH_GAIN 0.45
#define WEAVE_COLS        10.0
#define WEAVE_ROWS        8.0
```

**Do not:** Copy exact historical Kente patterns. Use weave **logic** only.

**Techniques:** [`04-pattern-tiling-truchet.md`](04-pattern-tiling-truchet.md) — primary.

---

## 4. 3-Step Cornrow Curves

**Feeling:** Flowing braids, triplet pulse, call-and-response — **not** literal hair.

**Audio:** Bass braid width · Mid curve drift · High bead glints · **Triplet clock from iTime** (independent of FFT)

**Architecture (to build):**

```
mainImage
├── centeredUV + audioBoost
├── triplet = tripletPulse(iTime * BPM_SCALE)   // 3 peaks per beat
├── for each row id (domain repeat Y):
│   ├── baseY = sum of sin(uv.x * freq + phase + mid)
│   ├── parallel offsets → 2-3 sdSegment braids
│   ├── width = baseWidth + bass * BRAID_BASS_GAIN
│   └── beads = hash along curve, high-gated glints
├── mirror alternate rows (IQ mirrored repetition)
└── palette: deepGold / creamGold on blackBrown
```

**Triplet pulse:**

```glsl
float tripletPulse(float t) {
    float beat = fract(t);
    float p1 = smoothstep(0.08, 0.0, abs(beat - 0.10));
    float p2 = smoothstep(0.08, 0.0, abs(beat - 0.43));
    float p3 = smoothstep(0.08, 0.0, abs(beat - 0.76));
    return max(p1, max(p2, p3));
}
```

**Braid line helper (from brief):**

```glsl
float braidLine(vec2 uv, float offset, float width, float speed) {
    float y = sin(uv.x * 5.0 + offset + iTime * speed) * 0.18;
    y += sin(uv.x * 11.0 - offset * 0.7 + iTime * speed * 0.5) * 0.035;
    return smoothstep(width, 0.0, abs(uv.y - y));
}
```

**Techniques:** [`05-iquilez-sdf-noise-warp.md`](05-iquilez-sdf-noise-warp.md) sdSegment + [`04-pattern-tiling-truchet.md`](04-pattern-tiling-truchet.md) row repeat.

---

## 5. Adinkra Symbol Field (optional)

Abstract procedural field — circles, arcs, radial symmetry. Symbols emerge from smoke/particles. **Do not** paste sacred icons as filler.

Bass reveals central glyph · Mid rotates field · High edge shimmer.

---

## Projection checklist (all genres)

- Dark negative space preserved
- Silhouette readable from ~15 ft — thick strokes, not fine dots
- No strobe / white blowout
- Animates 30+ min without fatigue
- Mic off = graceful fallback motion

Full checklist → [`00-AGENT-START.md`](../00-AGENT-START.md).

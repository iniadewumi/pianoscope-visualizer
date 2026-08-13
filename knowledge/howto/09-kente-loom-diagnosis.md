# Kente Loom — Diagnosis & Rebuild Plan

**Keywords:** kente, diagnosis, wrong, tartan, weave, warp, weft, rebuild

**Use when:** Kente shader looks like checker grid / pastel wallpaper / muddy overlays. Read **before** v3 rewrite.

**Status:** v5 in `test-shaders.js` — v4 unit grid + row-sync animation, brick offset, nighttime palette, circle stripe mode, richer audio.

---

## What the brief actually wants

From [`cursor inst.md`](../../cursor%20inst.md) § Kente visual logic + § Amapiano Kente Loom:

| Dimension | Target |
|-----------|--------|
| Feeling | Woven, percussive, syncopated, **warm but nighttime**, log-drum bounce |
| Palette | **Midnight violet**, deep black, rust orange, muted gold, cream (accents only) |
| Structure | Repeated **solid-color bands**, interlaced over/under, motif cells, syncopated breaks |
| Audio | Bass → **horizontal** thickness + bounce · Mid → **vertical** offset · High → thread shimmer |
| Avoid | Exact historical cloth copies; generic pastel wallpaper; transparent overlay grids |

Reference techniques: [Book of Shaders Ch. 09](https://thebookofshaders.com/09/) (tiling, Truchet), tartan/plaid weave logic, discrete palette indexing.

---

## What v1 and v2 actually render (user screenshots)

### v1 symptoms
- Pastel lavender / cream / yellow wash
- X-shaped diagonals in every cell (double Truchet)
- Sparkle stars at grid intersections (high-freq shimmer grid)

### v2 symptoms (after “darken” pass)
- **Black + gold checker grid** as the dominant read
- Semi-transparent rust bands layered on top → muddy brown
- Random **desaturated rectangular patches** (hash accent cells)
- Still reads as **transparency compositing**, not woven cloth
- **Midnight violet absent** from final image
- Too bright / golden — not “nighttime Amapiano”

---

## Root causes (why patching palette failed)

### 1. Wrong compositing model — alpha overlay, not weave

Current approach:
```glsl
col = mix(black, rust, vBand);
col = mix(col, gold, hBand);   // second layer tints first → muddy overlap
```

Real woven cloth (WIF / production weave shaders / [Tartan shader](https://markjgillespie.com/Misc/TartanShader/)):
- At each **warp×weft crossing**, exactly **one thread is on top**
- Pixel color = **top thread’s color only** — opaque replacement, not `mix` transparency
- Tartan `STRIPE()` picks `newcolorx` (warp) vs `newcolory` (weft) via diagonal parity `t`

**Fix:** Implement **pick-one-color-at-crossing**, not stack-two-layers.

---

### 2. Two full-screen stripe fields ≠ weave

`stripe()` on X and `stripe()` on Y simultaneously creates a **grid of intersections** everywhere. When combined with broken `vShow`/`hShow` math, you get a **checkerboard** (black squares + gold fills).

Brief’s intended model (simpler):
```glsl
float vertical   = stripe(...);  // warp presence
float horizontal = stripe(...);  // weft presence
float overUnder  = step(0.5, fract(floor(uv.x * N) + floor(uv.y * M)));
float weave      = mix(vertical, horizontal, overUnder);  // ONE scalar field
vec3 col         = kentePalette(weave, cellId);           // map scalar → discrete colors
```

We departed from this and made it worse with dual `mix` chains.

---

### 3. Discrete colors assigned wrong

Kente / tartan / band textiles use **indexed stripe colors**:
- Warp direction: repeating sequence e.g. `[black, violet, rust, black, gold, …]` per column band
- Weft direction: different sequence per row band
- At crossing: winner’s sequence applies

Current code maps `vBand`/`hBand` floats to continuous `mix()` → everything becomes orange-gold gradient. **Violet never wins.**

**Fix:** `int bandIndex = int(floor(uv.x * warpCount)); vec3 warpColor = WARP_PALETTE[bandIndex % 5];` (or hash-driven index)

---

### 4. Motifs on wrong layer

`cellMotifs()` draws diagonals/diamonds over **entire UV** on a second grid. Reads as decorative overlay / glitch patches, not woven-in symbols.

**Fix:** Motifs only where:
- `weave > 0.5` (on a thread, not in gap), AND
- `hash(cellId) > threshold` (syncopated), AND
- drawn as **thin ink on band color**, not additive gold blob

---

### 5. Audio mapping drifted from brief

| Brief | Current shader |
|-------|----------------|
| Bass → horizontal thickness + bounce | Bass on both hBand width AND vCount |
| Mid → vertical offset | Mid on vOff; hCount from mid |
| Vertical stripe count `10 + bass*3` | Similar but compositing wrong so audio invisible |

Align to brief **after** compositing model fixed.

---

### 6. Global rotation hurts “loom” read

`rot2(slowRot) * uv` tilts the entire grid. Kente/tartan reads best **axis-aligned** — motion should be **phase offsets** in stripe functions, not rotating the loom.

---

### 7. Truchet / tartan lesson

[Book of Shaders Truchet](https://thebookofshaders.com/09/): variation comes from **one element rotated by cell parity**, not two diagonals forming X.

[Tartan STRIPE](https://markjgillespie.com/Misc/TartanShader/): multiple stripe layers each **replace** color along warp or weft diagonal; checker `mod(Coord, 2*scale)` establishes cell.

[tileWeave GLSL](https://github.com/tuxalin/procedural-tileable-shaders): `c = mod(i.x+i.y,2); p = mix(p.st, p.ts, c)` — coordinate swap for over/under **geometry**, not color alpha.

---

## Correct architecture (v3 — implement this)

```
mainImage
├── uv = centeredUV (NO global rotation)
│
├── LAYER A — Warp (vertical threads)
│   ├── warpIdx = floor((uv.x + midWobble) * warpCount)
│   ├── warpPhase = fract((uv.x + midWobble) * warpCount)
│   ├── warpOn = bandMask(warpPhase, warpWidth)  // 1 on thread, 0 in gap
│   └── warpCol = WARP_COLORS[warpIdx % N]       // discrete: violet/rust/black/gold
│
├── LAYER B — Weft (horizontal threads)
│   ├── weftIdx = floor((uv.y + bassBounce) * weftCount)
│   ├── weftPhase = fract((uv.y + bassBounce) * weftCount)
│   ├── weftOn = bandMask(weftPhase, weftWidth)
│   └── weftCol = WEFT_COLORS[weftIdx % M]
│
├── LAYER C — Weave resolve (opaque pick)
│   ├── overUnder = step(0.5, fract(warpIdx + weftIdx))  // or mod sum parity
│   ├── if gap (neither thread): col = deepBlack
│   ├── else if overUnder: col = warpOn > 0.5 ? warpCol : deepBlack
│   └── else: col = weftOn > 0.5 ? weftCol : deepBlack
│   (Refine: at crossing, top thread wins if both on)
│
├── LAYER D — Motifs (sparse, on-thread only)
│   └── thin sdSegment diagonal in cell, * bandMask, * hash gate
│
├── LAYER E — Shimmer (high only)
│   └── derivative of warpPhase/weftPhase near edge, not full-screen grid
│
├── LAYER F — Log pulse
│   └── bass * sin(uv.y * k - t) modulates weftWidth or bounce amplitude
│
└── vignette, pow tone map, cap 0.75
```

---

## Minimal v3 first pass (prove weave model)

Before motifs/Truchet/shimmer, ship **only**:

1. Discrete warp colors (violet, rust, black, gold repeating)
2. Discrete weft colors (black, gold, rust repeating)
3. Over/under pick at crossings
4. Black gaps between threads
5. Brief’s `wovenBand` + `kentePalette` literally

**Pass test:** Screenshot should show **solid colored bands** interlacing — no checker transparency, no pastel wash. Readable at 15 ft.

Then add: syncopated motif cells → diagonal accents → shimmer → log pulse.

---

## References

| Source | URL | Steal |
|--------|-----|-------|
| Brief wovenBand | `cursor inst.md` ~L1058 | Single scalar weave + palette |
| Book of Shaders Patterns | https://thebookofshaders.com/09/ | Tiling, brick offset, Truchet |
| Tartan shader | https://markjgillespie.com/Misc/TartanShader/ | STRIPE warp/weft pick |
| IQ domain repetition | https://iquilezles.org/articles/sdfrepetition/ | Cell id for variation |
| tileWeave | tuxalin/procedural-tileable-shaders | mod parity coordinate swap |
| WIF weave paper | EG 2016 woven cloth shader | Crossing matrix mental model |

---

## Anti-patterns specific to Kente (add to 07)

| Do not | Why |
|--------|-----|
| `mix(col, gold, hBand)` on top of warp color | Transparency mud |
| Two diagonals per cell | X grid noise |
| Shimmer via `stripe(uv, 48.)` | Star field at intersections |
| Global `rot2` on loom UV | Loses woven read |
| Continuous palette from float 0–1 | Loses violet + discrete band identity |
| Motifs before weave model works | Decor on broken base |

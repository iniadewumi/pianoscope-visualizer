# Anti-Patterns & Known Failures

**Keywords:** anti-pattern, failed, dot grid, edge-only, tribal, mandala, sparks, do not

**Use when:** Before submitting GLSL; debugging “why does this look wrong?”

**Full narratives:** `PIANOSCOPE_SHADER_LEARNINGS.md` § Iteration postmortem.

---

## Cultural / design anti-patterns

| Avoid | Symptom | Do instead |
|-------|---------|------------|
| Random “tribal” triangles | Tourist aesthetic | Named system: Kente weave, Bogolan marks, cornrow curves |
| Neon mask / gold circuitry | Wakanda-lite | Restrained palette + structural logic |
| Sacred symbols as filler | Disrespect / shallow | Abstract geometry or researched meaning in comments |
| Copy exact textile patterns | Cultural misuse | Weave **logic**, not specific cloth designs |
| Generic radial lasers from center | Sci-fi HUD | Neighbor branch arcs with inset midpoint |

---

## Technical anti-patterns

| Avoid | Symptom | Do instead |
|-------|---------|------------|
| Edge-only recursive `mod2` | Dot grid / halftone kaleidoscope | Explicit layers + **fill+stroke** |
| `fwidth()` on WebGL 1 paths | Compile fail / inconsistent AA | `smoothstep` fixed feather ~0.003 UV |
| Global `const float` | WebGL 1 compile error | `#define` |
| `out` params in helpers | WebGL 1 failures | Return packed `vec3` |
| Bokeh/grid sparks `floor(uv*N)` | Full-screen noise | Sparks at **semantic positions** only |
| `iChannel0` at `y=0.25` | Wrong frequency data | Always `y=0.0` |
| Rewriting after v3 works | Lose fill/stroke foundation | Polish winning architecture |
| Undefined palette vars in scope | Compile fail | Define colors where used |
| Naive Voronoi `F2-F1` edges | Uneven crack width | IQ Voronoi distance algorithm |
| Double domain warp on Kente | Destroyed weave readability | Warp organic shaders only |

---

## Afrotech iteration summary (do not retry blindly)

### v1 — Thin rings + scattered nodes — FAILED

- Rings decorative; dwellings invisible
- Random hash sparks across screen
- Read as “orbital HUD / cyan mandala”

### v2 — Pianoscope edge-only recursion — FAILED

- `mod2` + edge-only `smoothstep(dEdge)` → **dot matrix**
- Bokeh grid on `floor(uv*5)` added more dots
- Kaleidoscope halftone, not architecture

### v3 — Apollonian fill/stroke + explicit layers — SUCCESS (~90% after polish)

- `sdfFill` / `sdfStroke` from Apollonian Gasket
- 3 explicit ring layers, gated walls, neighbor branch paths
- Sparks only at dwelling centers
- Polish: asymmetry, `audioBoost`, fBm smoke, rect compounds

**Lesson:** Listing layers in a plan isn’t enough — each layer needs **concrete geometry + fill/stroke weight**.

---

## Validated design principles

| Principle | Evidence |
|-----------|----------|
| Structure before decoration | v1 failed; v3 works |
| Fill + stroke, not edge-only | v2 dot grid |
| Dark negative space 40–60% | Courtyard + vignette |
| Sparks at semantic positions | Dwelling centers OK |
| Paths connect neighbors | Arcs on same ring, not center spokes |
| Imperfect symmetry | Gated walls, jitter — settlement not mandala |
| Start generous on audio gains | FFT smoothing 0.8 |
| One shader at a time | Brief convention |
| Modify winner, don’t restart | Polish > rewrite |

---

## “Good” vs “Fail” — Afrotech (transferable)

**Pass:** Thick compound walls, filled dwellings, gated openings, 3 fractal levels, dark courtyard, imperfect symmetry, bass/mid/high visibly different.

**Fail:** Two lonely rings, dot grid, random spark field, center-only lasers, white strobe, detail that vanishes on projector.

Organic genres: replace “rings” with “blooms/bands/braids” but keep **projection readability** rules.

---

## When stuck

1. Check [`06-genre-shader-recipes.md`](06-genre-shader-recipes.md) — are you using the wrong architecture for the genre?
2. Mudcloth ≠ Afrotech — don’t reuse `settlementLayer` for organic shaders
3. Kente needs crisp `fract` grid — don’t domain-warp the whole UV
4. Read full postmortem in `PIANOSCOPE_SHADER_LEARNINGS.md`

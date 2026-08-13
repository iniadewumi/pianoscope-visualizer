# PIANOSCOPE Local Knowledge Base

**Purpose:** A local, agent-readable HOW-TO library for building PIANOSCOPE procedural shaders. This is **knowledge collection first** — not a vector database yet. Agents retrieve docs by reading this index, grepping [`KEYWORDS.md`](KEYWORDS.md), or following links.

**Start here:** [`00-AGENT-START.md`](00-AGENT-START.md)

---

## Document map

| Read order | File | What you get |
|------------|------|--------------|
| 0 | [`00-AGENT-START.md`](00-AGENT-START.md) | Onboarding checklist, doc hierarchy, current repo state |
| — | [`KEYWORDS.md`](KEYWORDS.md) | Tag → file lookup (grep / semantic search entry point) |
| 1 | [`howto/01-repo-and-workflow.md`](howto/01-repo-and-workflow.md) | Where code lives, how to run, test, ship shaders |
| 2 | [`howto/02-audio-reactivity.md`](howto/02-audio-reactivity.md) | `iChannel0`, FFT helpers, gains, fallback motion |
| 3 | [`howto/03-glsl-webgl-constraints.md`](howto/03-glsl-webgl-constraints.md) | WebGL 1 rules, compile pipeline, error line numbers |
| 4 | [`howto/04-pattern-tiling-truchet.md`](howto/04-pattern-tiling-truchet.md) | Book of Shaders Ch. 09 — grids, brick offset, Truchet |
| 5 | [`howto/05-iquilez-sdf-noise-warp.md`](howto/05-iquilez-sdf-noise-warp.md) | IQ articles — SDF 2D, fBM, warp, Voronoise, palettes |
| 6 | [`howto/06-genre-shader-recipes.md`](howto/06-genre-shader-recipes.md) | Per-genre architecture, palette, audio mapping, status |
| 7 | [`howto/07-anti-patterns-and-failures.md`](howto/07-anti-patterns-and-failures.md) | What failed in practice — do not repeat |
| 8 | [`howto/08-steal-from-library.md`](howto/08-steal-from-library.md) | Which `shaders.js` entries to study and why |
| 9 | [`howto/09-kente-loom-diagnosis.md`](howto/09-kente-loom-diagnosis.md) | **Kente v1/v2 failure analysis + v3 rebuild plan** |

---

## External canon (read outside this folder)

| Doc | Role |
|-----|------|
| [`cursor inst.md`](../cursor%20inst.md) | Creative brief — *what* to build, cultural context, genre specs |
| [`PIANOSCOPE_SHADER_LEARNINGS.md`](../PIANOSCOPE_SHADER_LEARNINGS.md) | Iteration postmortems, Afrotech v1–v3 history, porting case studies |

**Rule:** Brief = intent. Learnings = mistakes + wins. **This folder = reusable HOW-TOs** an agent can apply without re-researching Book of Shaders / IQ every time.

---

## How to use this as “local RAG”

1. **Task routing** — Open [`KEYWORDS.md`](KEYWORDS.md), find tags matching the task (e.g. `kente`, `fract`, `audio`, `compile-error`).
2. **Read one how-to** — Each file is self-contained with copy-paste GLSL snippets.
3. **Cross-check genre** — [`06-genre-shader-recipes.md`](howto/06-genre-shader-recipes.md) for the specific PIANOSCOPE mode.
4. **Validate** — [`07-anti-patterns-and-failures.md`](howto/07-anti-patterns-and-failures.md) before submitting GLSL.

Future upgrade path (not built yet): embed these files in a local vector index or Cursor `@Docs` collection. The markdown structure is already chunk-friendly (H2 sections ≈ retrieval units).

---

## Maintenance

When you learn something new (failed approach, working recipe, new IQ technique):

1. Add keywords to [`KEYWORDS.md`](KEYWORDS.md).
2. Append to the relevant `howto/*.md` (prefer extending over new files).
3. Update **Current state** in [`00-AGENT-START.md`](00-AGENT-START.md).
4. Keep long narrative postmortems in [`PIANOSCOPE_SHADER_LEARNINGS.md`](../PIANOSCOPE_SHADER_LEARNINGS.md); keep *actionable steps* here.

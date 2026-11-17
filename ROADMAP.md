# PhotoCraft roadmap

**Stage: alpha** · last full readiness estimate (0.6.0, 2026-10-10): ~45%; beta target ~75% with reliable PSD, estimated at ~1,100–1,800 h

> **Last reviewed:** 2026-10-10 (full readiness assessment) · **Last updated:** 2026-10-15 · **Change:** minor (reconciled current measured counts, issue and PR snapshot, release version and dated estimates) · **Target:** Adobe Photoshop 2026

PhotoCraft's one-page summary against Adobe Photoshop 2026 (27.11.0, inspected on 2026-10-10).
The readiness percentages and effort estimates were assessed for PhotoCraft 0.6.0; the current
release is [0.9.0](https://github.com/storytold/photocraft/releases/tag/v0.9.0) (2026-10-14).
Measured counts below have been refreshed where noted, but the estimates need a full re-grade.
Detail: [target-app parity](docs/target-app-parity.md)
(methodology, weights, evidence), [gaps](docs/gaps.md) (the ranked work list),
[roadmap](docs/roadmap.md) (milestones, current focus), [scorecard](docs/scorecard.md) and
[menu checklist](docs/parity-checklist.md) (both generated and measured),
[architecture](docs/architecture.md).

Hours are Opus 5.5 agent wall-clock hours for one agent working sequentially, calibrated from this
repository's 1,004 merged PRs (see [calibration](docs/target-app-parity.md#remaining-effort-and-calibration)).
About 80% of the work parallelizes.

## Headline numbers

| | Value | Kind |
|---|---|---|
| **Feature breadth** | **~76%** (2026-10-10 estimate) | measured components: menus 631/631 and tools 67/68 (2026-10-15); panels 30/35 and formats 13+4 of 31 (2026-10-10) |
| **Ready for real work** | **~45%** (40–50%, 2026-10-10 estimate) | weighted by dimension below; not re-graded for 0.9.0 |
| Mainstream practitioner | ~43% (38–48%) | estimated: weekly areas of a typical pro, with discounts ([method](docs/target-app-parity.md#mainstream-practitioner-43)) |
| Essentials user | ~61% (55–67%) | estimated: core features only ([method](docs/target-app-parity.md#essentials-user-61)) |
| Remaining to **beta** | **~1,100–1,800 h** | estimated |
| Remaining to **full parity** | **~2,300–4,100 h** | estimated; AI and plug-ins need owner decisions |

### Readiness by audience

| Audience | Ready % | Opus 5.5 agent-hours to ~95% | Work that dominates |
|---|---:|---:|---|
| Full target (ready for real work) | ~45% (additive weighted sum: 46%) | 2,300–4,100 | Everything below, plus AI tier, plug-in/scripting ecosystem, video, localization, hardware |
| Mainstream practitioner | ~43% | 1,700–2,800 | Depth in the 11 weekly areas (1,180–2,000), performance budgets, stability on real machines, PSD exchange with Photoshop |
| Essentials user | ~61% | 400–700 | Startup and GPU robustness, brush and transform responsiveness, Type tool, discoverability (flyouts, commit buttons), AVIF/PDF open |

Hours calibrated as in [Remaining effort](docs/target-app-parity.md#remaining-effort-and-calibration) (2–9 h per checklist
item, 8–20 h per format or subsystem, from the repo's 1,004 merged PRs); each tier is a subset of
the one above. About 80% parallelizes (independent crates and command modules); performance
architecture and the PSD writer are the serial parts.

**Why alpha.** Core workflows (open, layer, select, paint, adjust, filter, type, save) work end to
end, and breadth is high, but 1,254 issues are open (2026-10-15; 764 additional open PRs), 22 of 25 performance budgets
are missed, and PSDs we write are not yet reliably accepted by Photoshop (#1281, #2469). Beta
needs ~75% ready and no blocking PSD gap. The [alpha gate](docs/roadmap.md#alpha-gate) passes: all six core workflows
(retouch, composite, cut-out, design, paint, PSD exchange) complete end to end and save and reopen;
four are partial in depth, none blocked.

## By dimension

| Dimension | % ready | Hours to full | Doc |
|---|---:|---:|---|
| Features (depth) | 55% (breadth 76%) | 600–1,000 | [target-app-parity](docs/target-app-parity.md#feature-areas) |
| UI/UX fidelity (tools, handles, modifiers, brush feel, panels) | 40% | 350–600 | [ui-parity](docs/ui-parity.md), [brush-parity](docs/brush-parity.md), [context-menu-parity](docs/context-menu-parity.md) |
| File formats | 60% (presence ~80%) | 300–500 | [file-format-parity](docs/file-format-parity.md) |
| Hardware (GPU, pen, displays) | 50% | 80–150 | [hardware-parity](docs/hardware-parity.md) |
| Localization | 55% | 80–140 | [localization-parity](docs/localization-parity.md) |
| Performance | 25% (3/25 budgets met, measured) | 200–350 | [scorecard](docs/scorecard.md#performance) |
| Stability | 45% | 150–250 | [gaps](docs/gaps.md#g3-user-reported-bug-backlog) |
| Platforms | 80% | 30–60 | [scorecard](docs/scorecard.md#distribution) |
| Ecosystem / plug-ins / scripting | 15% | 250–450 | [gaps](docs/gaps.md#g16-plug-in-compatibility) |
| AI features | 10% | 300–600 | [gaps](docs/gaps.md#g15-ai-and-generative-features) |

## Features

| Area | % ready | Hours |
|---|---:|---:|
| Layers, masks, blend modes, smart objects, styles | 65% | 150–250 |
| Selections, Select and Mask, channels | 55% | 120–200 |
| Painting and brush engine | 55% | 150–250 |
| Retouching (healing, clone, patch, remove, content-aware) | 50% | 100–180 |
| Transform, warp, puppet, liquify | 55% | 100–180 |
| Adjustments and filters | 70% | 120–200 |
| Type | 40% | 120–200 |
| Vector shapes and paths | 55% | 80–140 |
| Colour management and modes | 70% | 60–100 |
| Camera Raw and computational photo | 45% | 120–200 |
| Automation, actions, scripting | 45% | 120–200 |
| File I/O | 60% | 300–500 |
| Print, export, Save for Web | 50% | 60–100 |
| Video / timeline, data-driven graphics | 30% | 100–200 |
| Workspace, panels, preferences | 40% | 150–250 |
| AI / generative | 10% | 300–600 |

Rows and evidence: [target-app-parity › Feature areas](docs/target-app-parity.md#feature-areas).

## Languages

Measured with the `cargo xtask i18n-coverage` key universe (1,608 keys). `partial` everywhere
because engine messages are still English. Detail: [localization-parity](docs/localization-parity.md).

| Language | Code | UI strings | Status |
|---|---|---:|---|
| English | `en` | 100% | full |
| Simplified Chinese | `zh-hans` | 100% | partial |
| Spanish | `es` | 100% | partial |
| Hindi | `hi` | 0% | none |
| Arabic | `ar` | 0% (RTL shell only) | none |
| French | `fr` | 100% | partial |
| Portuguese (Brazil) | `pt-br` | 100% | partial |
| Indonesian | `id` | 100% | partial |
| Japanese | `ja` | 100% | partial |
| German | `de` | 100% | partial |
| Korean | `ko` | 100% | partial |
| Vietnamese | `vi` | 0% | none |

Also shipped: Traditional Chinese, Russian, Ukrainian, Czech, Polish, Greek, Dutch, Italian (8).

## Upcoming

| Rank | Milestone | Estimate |
|---|---|---:|
| 1 | PSD that Photoshop trusts: Save As corruption, Photoshop re-open check, Photoshop-oracle floor 188 → 200+ of 258 | 150–250 h |
| 2 | Performance budgets: the 22 over-budget scenarios and the 150-layer GPU crash | 200–350 h |
| 3 | User bug backlog to under 100 open | 250–400 h |
| 4 | Brush and transform feel (brush-parity, ui-parity rows) | 160–290 h |
| 5 | Art History Brush, the remaining missing toolbox tool | re-estimate |

Detail and the next releases: [docs/roadmap.md](docs/roadmap.md).

## Progress log

| Date | Entry |
|---|---|
| 2026-10-15 | Documentation counts reconciled against main and pinned corpus manifests; 0.9.0 is current, while readiness estimates remain the 0.6.0 assessment. |
| 2026-10-14 | 0.9.0 released. |
| 2026-10-10 | Full re-measure against Photoshop 2026 27.11.0 (installed bundle inspected): breadth ~76%, ready ~45%, alpha. Progress docs reorganized to the craftrules standard (`parity.md` → `parity-checklist.md`, `parity-estimate.md` merged into `target-app-parity.md`). 0.6.0 released. |
| 2026-10-09 | Remove tool, Rotate View, crop rotation; OpenRaster and Paint.NET; 39 Affinity documents in the corpus |
| 2026-10-08 | 0.5.0. Affinity import; Magnetic Lasso, Pattern Stamp, Mixer Brush, Pencil, Patch, Vertical Type on the toolbar |
| 2026-10-07 | 0.3.0. 10 UI languages; shortcut audit 214 → 0 failures; TIFF decode 10× faster |
| 2026-10-05 | 0.2.0, first signed and notarized release. Honest assessment: ready for real work 25–35% |
| 2026-10-03 | Menu breadth 625/625 |
| 2026-10-01 | Menu breadth 224 → 532 of 625 in a day |
| 2026-09-30 | First commit |

## Revision history

| Date | Change | Summary |
|---|---|---|
| 2026-10-15 | minor | Current issue/PR counts, 67/68 tools, 631/631 menus and 188/258 Photoshop-oracle floor reconciled; 0.6.0 readiness estimate explicitly dated, 0.9.0 release noted; G6 effort needs re-estimation |
| 2026-10-13 | minor | Focus item 1 quoted the oracle floor as 133 of 256; it is 145 of 258 (`crates/io/tests/corpus.rs`, gaps.md G1). Open issues 579 → 953 (2026-10-13) |
| 2026-10-11 | minor | Toolbox count 56 of 68 (12 missing) with the Convert Point tool (#3204) |
| 2026-10-10 | minor | Readiness by audience with hours per tier |
| 2026-10-10 | minor | Mainstream-practitioner and essentials-user numbers added |
| 2026-10-10 | minor | Alpha gate result (passes; stays alpha) in the stage explanation |
| 2026-10-10 | major | First version, per craftrules `standards/progress-docs.md` |

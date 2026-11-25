# Atelier Sprite

**Atelier Sprite** is a Classic Tibia (7.1–10.98+ `.dat`/`.spr`) pixel-art sprite studio built on **React + TypeScript + Vite + Express + IndexedDB**. Every pixel operation treats the native `32x32` RGBA grid as discrete indexed palette colors with binary alpha (`0 | 255`), strict zero anti-aliasing/blur/resampling hygiene, mandatory IndexedDB read-back verification (`putAndVerify`), and honest data-driven scoring measured against real held-out sprites and synthetic junk baselines.

---

## Application Modules (UI in PT-BR)

1. **Pacote `.dat/.spr` (`src/components/ModuleUpload.tsx`, `src/utils/tibiaParser.ts`)**
   - Parses Classic & Extended `.dat`/`.spr` archives into `32x32` sprites, item metadata, categories (`Equipamentos`, `Consumíveis`, `Containers`, `Decoração`), and item flags.
   - Persists raw binary `ArrayBuffer`s in IndexedDB (`atelier_sprite_v2`), calls `navigator.storage.persist()`, auto-restores on startup, and provides full cross-origin JSON **Exportar backup** / **Importar backup**.
   - Displays **Diagnóstico do Crítico v1**: held-out Real Ceiling vs Synthetic Junk Floor (`n`, median, 95% bootstrap/quantile CI, and logistic validation AUC).

2. **Treino e Refinos (`src/components/ModuleRefinements.tsx`, `src/evolution/`)**
   - Cooperative co-evolution (**NEAT pixel-selection brush + Recipe GA** over pure discrete pixel-editor operations) running inside a dedicated Web Worker (`src/evolution/trainingWorker.ts`, orchestrated by `src/evolution/workerClient.ts`) with pause/checkpoint/resume in IndexedDB.
   - Trains offline (API-free) on the **Restoration Benchmark** (degrading real `train` sprites cleanly and evaluating recovery on held-out `validation` sprites with strict promotion gating).

3. **Lições Aprendidas (`src/components/ModuleLessons.tsx`, `src/lessons/`)**
   - Persisted lesson database stored in IndexedDB (`lessons` store) with deterministic retrieval ranking, Wilson lower-bound confidence, evidence confirmation/contestation, conflict detection, and resolution (`auto_precedence`, `keep_lesson`, `archive_both`).

4. **Comparador (`src/components/ModuleComparator.tsx`, `src/comparator/`)**
   - **Arena A/B Cega (Elo + Lições)**: Blind agent-vs-agent sprite comparison with zero-sum Elo rating updates (`K = 32`, base `1200`), post-reveal Approve/Reject votes, and deterministic **Comentário → Lição Aprendida** conversion.
   - **Ranking Elo & Tornar ativo**: Persisted Elo leaderboard allowing any agent (T1, T2, or Phase 3 Co-Evolution Champion) to be promoted as the active generator in **Estúdio Prompt**.
   - **Teste de Campo (C.6)**: Paired multi-prompt evaluation comparing a candidate agent vs a baseline on identical prompts and seeds without touching the held-out `test` split.

5. **Estúdio Prompt (`src/components/ModulePromptStudio.tsx`, `src/components/PromptEngineerCard.tsx`, `src/components/T3WorkbenchPanel.tsx`, `src/generation/`, `src/t3/`)**
   - **Two-Step Human-Gated Generation Flow**:
     - **Step 1 — Engenheiro de Prompt (`POST /api/gemini/engineer-prompt`, `src/generation/promptEngineer.ts`, `src/generation/promptStore.ts`)**: Converts a short idea (`"espada de gelo"`, `"orc com machado"`) + a `train`-only **Style Dossier** (category medians, top palette colors, 6–10 few-shot `train` donors, active ranked lessons) into a strict JSON prompt specification (`itemName`, `category`, `visualBrief`, `paletteRamps`, `constraints`, `negativePrompt`, `donorSuggestions`). Nothing is generated until the user reviews/edits the structured card and clicks **`OK, gerar`**. Prompt versions are persisted in IndexedDB (`promptHistory`) and recorded in sprite provenance. When `GEMINI_API_KEY` is absent, displays an explicit **"Gemini indisponível"** notice and allows editing a manual template (`source: 'manual'`, `model: null`).
     - **Step 2 — Hybrid Generation + Tibia-ification + Humanness Gate (`src/generation/pipeline.ts`, `src/generation/tibiaify.ts`, `src/generation/executors.ts`)**: Produces `N` candidates (`4–16`, default `8`, distinct seeds) via structured `32x32` indexed-grid Gemini generation or local T1/T2 donor synthesis, runs every candidate through deterministic `tibiaifySprite` (nearest-neighbor downsample only if needed, binary alpha `0|255`, palette snapping, orphan removal, color cap, dither removal, 1px dark hue-shifted outline, NW light order, centering), runs iterative Critic-guided repair (up to `K` steps, strict improvement & zero hard-rule regression only), and ranks via the **Humanness Gate** (Critic v1 score, Discriminator `P(real)`, originality rejecting `> 90%` near-copies and style outliers, and human-preference term when `>= 6` mixed votes).
     - **Review Loop**: Displays each candidate at native `32x32` (`1:1`) and zoomed (`3x`, nearest-neighbor) beside 2–3 real donor sprites with Critic breakdown, supporting **Aprovar**, **Rejeitar**, and **"Pedir ajuste"** (revises the prompt with a visual diff for re-approval and records the comment as a `human_feedback` lesson).
   - **Modelo T3 (ONNX & Colab)**: Builds a material-clustered Ramp-Role Vocabulary (`train`-only) with manual role overrides, generates the Google Colab T4 PyTorch training notebook (MaskGIT Transformer + Reranker AUC + Drive checkpoints + ONNX export), imports ONNX models with a validated JSON model card, runs Web Worker inference (`onnxruntime-web`, WebGPU with WASM fallback), auto-resamples structural failures, runs the held-out `test`-split comparison vs T1/T2/Junk Floor (keeping T3 disabled by default when worse), and exports **"Re-treinar com novos rótulos"** bundles.

---

## Server API Routes (`server.ts`)

- `GET /api/health`: Reports server health, `aiReady` (`GEMINI_API_KEY` presence), and configured `GEMINI_MODEL_PRIMARY` / `GEMINI_MODEL_FALLBACK` (defaults: `gemini-4-hydrogen` / `gemini-3.1-flash-lite`).
- `POST /api/gemini/engineer-prompt`: Step 1 strict-JSON Prompt Engineer (`responseSchema` + server-side validation; HTTP `422` on malformed output, HTTP `503` if `GEMINI_API_KEY` missing — never silently falls back).
- `POST /api/gemini/generate`: Structured generation proxy supporting validated `images[]`, `responseMimeType`, and clamped `temperature` (`[0, 2]`).
- **Protection on `/api/gemini/*`**: In-memory per-IP sliding-window rate limiter (`30 req/min`, HTTP `429` + `Retry-After`), scoped `4 MB` JSON body cap, and strict `images[]` validation (`<= 10` images, `image/png|jpeg|webp` only, `<= 350 KB` decoded per image, `<= 2.5 MB` combined).
- **Production startup (`npm start`)**: Runs `NODE_ENV=production node server.ts` and exits non-zero if `dist/index.html` is missing.

---

## What Is Learned Where

| What is learned / adapted | Source code | Data / signal used | When updated | IndexedDB store (`atelier_sprite_v2`) |
| :--- | :--- | :--- | :--- | :--- |
| **Deterministic 70 / 15 / 15 Stratified Split** | `src/db/split.ts` | Sprite IDs stratified by `.dat` category (`seed = 77202026`) | Once per `.dat`/`.spr` SHA-256 pair | `splits` |
| **Per-Category Hygiene Distributions** (color counts, ramp length, outline darkness/thickness, NW lighting, dithering share, orphan rates) | `src/critic/profile.ts`, `src/critic/features.ts` | Real sprites from **`train` split (70%) only** | On dataset ingest/restore calibration | `criticState` |
| **Critic v1 Logistic Discriminator** (real vs synthetic defects + human rejections) | `src/critic/discriminator.ts`, `src/critic/diagnostics.ts` | `train` real sprites + approved sprites (positives) vs synthetic corruptions & all rejected sprites (hard negatives); evaluated on held-out `validation` AUC | Initial calibration + **automatically retrained on every Approve / Reject vote** | `criticState` |
| **Human Preference Centroid Term (12% blend)** | `src/generation/store.ts`, `src/critic/evaluator.ts` | Feature centroids of human-approved vs human-rejected sprites (active once `>= 6` votes span both classes) | Recomputed on every Approve / Reject vote | `approvals`, `criticState` |
| **Quantitative Style Lessons (`mined`), Co-Evolution Rules (`critic_loop`), & Critiques (`human_feedback`)** | `src/lessons/miner.ts`, `src/lessons/retrieval.ts`, `src/lessons/conflicts.ts`, `src/lessons/store.ts` | Train-split category profiles, co-evolution promotion deltas, manual/comment-converted critiques (including **"Pedir ajuste"**), Wilson confidence updates | On mining sync, champion promotion, comment/adjustment conversion, confirm/contest, conflict resolution | `lessons` |
| **NEAT Pixel-Selection Network & Pixel-Op Recipe** | `src/evolution/neat.ts`, `src/evolution/recipeGa.ts`, `src/evolution/coevolution.ts`, `src/evolution/trainingWorker.ts` | Offline restoration task on `train` sprites; promoted ONLY after passing held-out **`validation` split** gates (`Δ median > 0`, `IoU >= 0.72`, no hard-rule regression) | Each generation in Web Worker; checkpointed + elite archived | `neatPopulations`, `gaPopulations`, `eliteArchive`, `checkpoints` |
| **Donor Index, Prompt Versions & Few-Shot Approved Pool** | `src/generation/indexer.ts`, `src/generation/executors.ts`, `src/generation/promptEngineer.ts`, `src/generation/tibiaify.ts`, `src/generation/pipeline.ts`, `src/generation/promptStore.ts` | Non-empty sprites from `train` + `validation` (`test` strictly excluded); Style Dossier built from `train` only; prompt version history + approvals persisted with read-back | Indexed on demand; prompt versions saved on Step 1/edit/approval; scores updated on Approve / Reject | `promptHistory`, `generatedSprites`, `approvals` |
| **Agent Elo Ratings & Active Studio Agent** | `src/comparator/elo.ts`, `src/comparator/store.ts`, `src/comparator/fieldTest.ts` | Blind A/B human duel outcomes (`1`, `0.5`, `0`) with zero-sum `K = 32` Elo + **"Tornar ativo"** selection | After every Blind A/B vote or activation click | `agents` |
| **Server-Side Gemini Prompt Engineer & Structured Grid Assist** | `server.ts`, `src/generation/pipeline.ts` | `POST /api/gemini/engineer-prompt` (Step 1) and `POST /api/gemini/generate` (Step 2) using `GEMINI_API_KEY` server-side | Per explicit Step 1 / approved Step 2 request | `promptHistory`, `generatedSprites` (provenance records actual model or `manual`/`local-*`) |
| **Ramp-Role Vocabulary (`T3`)** | `src/t3/vocabulary.ts`, `src/t3/store.ts` | Colors clustered per material from **`train` split only** (`outline`, `shadow`, `base`, `light`, `highlight`) + manual user role corrections | Built on dataset calibration; updated immediately on user role override | `onnxModels` (`t3-vocab:*`) |
| **Trained Indexed Model Weights & `test` Gate (`T3`)** | `src/t3/colabNotebook.ts`, `src/t3/t3Worker.ts`, `src/t3/sampler.ts`, `src/t3/testEvaluation.ts` | PyTorch Colab MaskGIT + Reranker ONNX weights trained on `train`/`validation` (`vocabulary.json` + `human_labels.csv` from Approve/Reject votes); evaluated once imported against `test` real sprites, Junk Floor, T1 & T2 | On **"Importar modelo (T3)"** and **"Avaliar T3 vs T1/T2 no split test"** | `onnxModels` (`t3-model:*`) |

---

## Honest Reporting & Metrics That Cannot Be Computed Honestly Yet

Following the project's **Honesty Rules**, any metric lacking real ground truth or required prerequisites is **never fabricated** (no floors, no iteration-count formulas, no flat `100%`):

1. **Human Preference Term before `6` mixed votes**: Cannot compute a balanced two-class preference vector until at least **6 human votes** exist spanning both approvals and rejections. Until then, Comparador shows `Termo humano X/6 votos` and omits the 12% human weight (`humanPreferences = null`).
2. **Uncalibrated Critic before `.dat`/`.spr` upload**: Without real sprite profiles from the user's package, Critic v1 caps the score and explicitly labels evaluations **`não calibrado`**.
3. **Held-out `test` Split (`15%`) Benchmark**: Intentionally excluded from indexing, Style Dossier, mining, training, Colab `.ipynb`/`.zip` exports, and routine T1/T2 UI scoring to prevent leakage. Only the explicit Phase 7 comparison (**Avaliar T3 vs T1/T2 no split test**) reads `test` sprites, keeping T3 disabled by default whenever it underperforms T1/T2 or nears the Junk Floor.
4. **Trained Indexed Model (`T3`) Metrics prior to Colab Training & Import**: Generator loss, epoch hygiene rates, and Reranker `heldOutValAuc` are only populated from an imported `t3_model_card.json` produced by running the Google Colab training notebook — never invented inside the browser.
5. **Gemini API Provenance without `GEMINI_API_KEY`**: When `GEMINI_API_KEY` is not configured, Step 1 displays **"Gemini indisponível"** and allows creating/editing a manual structured draft (`source: 'manual'`, `model: null`), and Step 2 executes local pixel-exact synthesis + `tibiaifySprite` recording `source: 'local-t1' | 'local-t2'` (`model: null`) rather than claiming AI model provenance.

---

## Cross-Origin Backup & Dataset Export

- **Full State Backup (`Exportar backup` / `Importar backup`)**: Serializes all 13 IndexedDB stores (`datasets`, `splits`, `criticState`, `neatPopulations`, `gaPopulations`, `eliteArchive`, `lessons`, `agents`, `approvals`, `generatedSprites`, `checkpoints`, `onnxModels`, `promptHistory`) into `atelier-sprite-backup-YYYY-MM-DD.json`, preserving binary `ArrayBuffer` and typed arrays (`Uint8ClampedArray`, `Uint8Array`, `Float64Array`, etc.) losslessly. Imports pre-validate before clearing existing stores and verify every restored record via read-back.
- **ML Training Dataset Export (`Exportar dataset de treino (.zip)`)**: Packages native `32x32` RGBA PNGs, indexed grids, per-sprite palettes, ramp-role maps, `labels.csv`, and split metadata for **`train` and `validation` only** (`test` split is strictly excluded).

---

## Verification & Test Suite

Run TypeScript type checking, production bundling, and the full `node:test` unit suite via `tsx`:

```bash
npm run lint
npm run build
npx --yes tsx --test \
  src/critic/critic.test.ts \
  src/db/persistence.test.ts \
  src/lessons/lessons.test.ts \
  src/evolution/evolution.test.ts \
  src/generation/generation.test.ts \
  src/generation/promptEngineer.test.ts \
  src/comparator/comparator.test.ts \
  src/t3/t3.test.ts
```

Note on `.playwright-cli/`: local headless Chromium checks verify end-to-end persistence (auto-restore after page reload, dataset & Elo agent restoration, two-step human-gated prompt flow, lesson retrieval, generation, and backup round-trip) with zero browser console errors.

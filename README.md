# Atelier Sprite

**Atelier Sprite** is a Classic Tibia (7.1–10.98+ `.dat`/`.spr`) pixel-art sprite studio built on **React + TypeScript + Vite + Express + IndexedDB**. Every pixel operation treats the native `32x32` RGBA grid as discrete indexed palette colors with binary alpha (`0 | 255`), strict zero anti-aliasing/blur/resampling hygiene, mandatory IndexedDB read-back verification (`putAndVerify`), and honest data-driven scoring measured against real held-out sprites and synthetic junk baselines.

---

## Application Modules (UI in PT-BR)

1. **Pacote `.dat/.spr` (`src/components/ModuleUpload.tsx`, `src/utils/tibiaParser.ts`)**
   - Parses Classic & Extended `.dat`/`.spr` archives into `32x32` sprites, item metadata, categories (`Equipamentos`, `Consumíveis`, `Containers`, `Criaturas`, `Terreno e Paredes`, `Decoração`, `Outros`), and item flags.
   - Persists raw binary `ArrayBuffer`s in IndexedDB (`atelier_sprite_v2`), calls `navigator.storage.persist()`, auto-restores on startup, and provides full cross-origin JSON **Exportar backup** / **Importar backup**.
   - Displays **Diagnóstico do Crítico v1**: held-out Real Ceiling vs Synthetic Junk Floor (`n`, median, 95% bootstrap/quantile CI, and logistic validation AUC).

2. **Treino e Refinos (`src/components/ModuleRefinements.tsx`, `src/evolution/`)**
   - Cooperative co-evolution (**NEAT pixel-selection brush + Recipe GA** over 10 pure discrete pixel-editor operations) running inside a dedicated Web Worker (`src/evolution/worker.ts`) with pause/checkpoint/resume in IndexedDB.
   - Trains offline (API-free) on the **Restoration Benchmark** (degrading real `train` sprites cleanly and evaluating recovery on held-out `validation` sprites with strict promotion gating).

3. **Lições Aprendidas (`src/components/ModuleLessons.tsx`, `src/lessons/`)**
   - Persisted lesson database stored in IndexedDB (`lessons` store) with deterministic retrieval ranking, Wilson lower-bound confidence, evidence confirmation/contestation, conflict detection, and resolution (`auto_precedence`, `keep_lesson`, `archive_both`).

4. **Comparador (`src/components/ModuleComparator.tsx`, `src/comparator/`)**
   - **Arena A/B Cega (Elo + Lições)**: Blind agent-vs-agent sprite comparison with zero-sum Elo rating updates (`K = 32`, base `1200`), post-reveal Approve/Reject votes, and deterministic **Comentário → Lição Aprendida** conversion.
   - **Ranking Elo & Tornar ativo**: Persisted Elo leaderboard allowing any agent (T1, T2, or Phase 3 Co-Evolution Champion) to be promoted as the active generator in **Estúdio Prompt**.
   - **Teste de Campo (C.6)**: Paired multi-prompt evaluation comparing a candidate agent vs a baseline on identical prompts and seeds without touching the held-out `test` split.

5. **Estúdio Prompt (`src/components/ModulePromptStudio.tsx`, `src/components/T3WorkbenchPanel.tsx`, `src/generation/`, `src/t3/`)**
   - Executes **T1** (semantic donor retrieval + monotonic-luma discrete ramp recoloring + bounded pixel diff) and **T2** (multi-donor structural compositing + Critic-guided repair) under the currently active agent from Comparador, checks originality against the nearest real sprite, persists candidates, and exports training packages via **Exportar dataset de treino (.zip)** (`src/export/trainingDatasetExport.ts`).
   - **Modelo T3 (ONNX & Colab)**: Builds a material-clustered Ramp-Role Vocabulary (`train`-only) with manual role overrides, generates the Google Colab T4 PyTorch training notebook (MaskGIT Transformer + Reranker AUC + Drive checkpoints + ONNX export), imports ONNX models with a validated JSON model card, runs Web Worker inference (`onnxruntime-web`, WebGPU with WASM fallback), auto-resamples structural failures, runs the held-out `test`-split comparison vs T1/T2/Junk Floor (keeping T3 disabled by default when worse), and exports **"Re-treinar com novos rótulos"** bundles.

---

## What Is Learned Where

| What is learned / adapted | Source code | Data / signal used | When updated | IndexedDB store (`atelier_sprite_v2`) |
| :--- | :--- | :--- | :--- | :--- |
| **Deterministic 70 / 15 / 15 Stratified Split** | `src/db/split.ts` | Sprite IDs stratified by `.dat` category (`seed = 77202026`) | Once per `.dat`/`.spr` SHA-256 pair | `splits` |
| **Per-Category Hygiene Distributions** (color counts, ramp length, outline darkness/thickness, NW lighting, dithering share, orphan rates) | `src/critic/profile.ts`, `src/critic/features.ts` | Real sprites from **`train` split (70%) only** | On dataset ingest/restore calibration | `criticState` |
| **Critic v1 Logistic Discriminator** (real vs synthetic defects + human rejections) | `src/critic/discriminator.ts`, `src/critic/diagnostics.ts` | `train` real sprites + approved sprites (positives) vs synthetic corruptions & all rejected sprites (hard negatives); evaluated on held-out `validation` AUC | Initial calibration + **automatically retrained on every Approve / Reject vote** | `criticState` |
| **Human Preference Centroid Term (12% blend)** | `src/generation/store.ts`, `src/critic/evaluator.ts` | Feature centroids of human-approved vs human-rejected sprites (active once `>= 6` votes span both classes) | Recomputed on every Approve / Reject vote | `approvals`, `criticState` |
| **Quantitative Style Lessons (`mined`), Co-Evolution Rules (`critic_loop`), & Critiques (`human_feedback`)** | `src/lessons/miner.ts`, `src/lessons/retrieval.ts`, `src/lessons/conflicts.ts`, `src/lessons/store.ts` | Train-split category profiles, co-evolution promotion deltas, manual/comment-converted critiques, Wilson confidence updates | On mining sync, champion promotion, comment conversion, confirm/contest, conflict resolution | `lessons` |
| **NEAT Pixel-Selection Network & Pixel-Op Recipe** | `src/evolution/neat.ts`, `src/evolution/recipeGa.ts`, `src/evolution/coevolution.ts`, `src/evolution/worker.ts` | Offline restoration task on `train` sprites; promoted ONLY after passing held-out **`validation` split** gates (`Δ median > 0`, `IoU >= 0.72`, no hard-rule regression) | Each generation in Web Worker; checkpointed + elite archived | `neatPopulations`, `gaPopulations`, `eliteArchive`, `checkpoints` |
| **Donor Index & Few-Shot Approved Pool** | `src/generation/indexer.ts`, `src/generation/recolor.ts`, `src/generation/pipeline.ts` | Non-empty sprites from `train` + `validation` (`test` strictly excluded); few-shot donor retrieval draws from `train` + approved generated sprites, weighting parent donors by human approvals/rejections | Indexed on demand; scores updated immediately on Approve / Reject | `generatedSprites`, `approvals` |
| **Agent Elo Ratings & Active Studio Agent** | `src/comparator/elo.ts`, `src/comparator/store.ts`, `src/comparator/fieldTest.ts` | Blind A/B human duel outcomes (`1`, `0.5`, `0`) with zero-sum `K = 32` Elo + **"Tornar ativo"** selection | After every Blind A/B vote or activation click | `agents` |
| **Server-Side Gemini API Config (Optional T1/T2 Diff Assist)** | `server.ts`, `src/generation/pipeline.ts` | Optional prompt-conditioned bounded pixel-diff JSON from server-side `/api/gemini/sprite-diff` (`GEMINI_API_KEY` stays server-side) | Per explicit Gemini-enabled generation request | `generatedSprites` (provenance records model or `local-*` fallback) |
| **Ramp-Role Vocabulary (`T3`)** | `src/t3/vocabulary.ts`, `src/t3/store.ts` | Colors clustered per material from **`train` split only** (`outline`, `shadow`, `base`, `light`, `highlight`) + manual user role corrections | Built on dataset calibration; updated immediately on user role override | `onnxModels` (`t3-vocab:*`) |
| **Trained Indexed Model Weights & `test` Gate (`T3`)** | `src/t3/colabNotebook.ts`, `src/t3/t3Worker.ts`, `src/t3/sampler.ts`, `src/t3/testEvaluation.ts` | PyTorch Colab MaskGIT + Reranker ONNX weights trained on `train`/`validation` (`vocabulary.json` + `human_labels.csv` from Approve/Reject votes); evaluated once imported against `test` real sprites, Junk Floor, T1 & T2 | On **"Importar modelo (T3)"** and **"Avaliar T3 vs T1/T2 no split test"** | `onnxModels` (`t3-model:*`) |

---

## Honest Reporting & Metrics That Cannot Be Computed Honestly Yet

Following the project's **Honesty Rules**, any metric lacking real ground truth or required prerequisites is **never fabricated** (no floors, no iteration-count formulas, no flat `100%`):

1. **Human Preference Term before `6` mixed votes**: Cannot compute a balanced two-class preference vector until at least **6 human votes** exist spanning both approvals and rejections. Until then, Comparador shows `Termo humano X/6 votos` and omits the 12% human weight (`humanPreferences = null`).
2. **Uncalibrated Critic before `.dat`/`.spr` upload**: Without real sprite profiles from the user's package, Critic v1 caps the score and explicitly labels evaluations **`não calibrado`**.
3. **Held-out `test` Split (`15%`) Benchmark**: Intentionally excluded from indexing, mining, training, Colab `.ipynb`/`.zip` exports, and routine T1/T2 UI scoring to prevent leakage. Only the explicit Phase 7 comparison (**Avaliar T3 vs T1/T2 no split test**) reads `test` sprites, keeping T3 disabled by default whenever it underperforms T1/T2 or nears the Junk Floor.
4. **Trained Indexed Model (`T3`) Metrics prior to Colab Training & Import**: Generator loss, epoch hygiene rates, and Reranker `heldOutValAuc` are only populated from an imported `t3_model_card.json` produced by running the Google Colab training notebook — never invented inside the browser.
5. **Gemini API Provenance without `GEMINI_API_KEY`**: When `GEMINI_API_KEY` is not configured or local deterministic mode is active, generation executes the local T1/T2 pipeline and records `source: 'local-t1' | 'local-t2'` (`model: null`) rather than claiming AI model provenance.

---

## Cross-Origin Backup & Dataset Export

- **Full State Backup (`Exportar backup` / `Importar backup`)**: Serializes all 12 IndexedDB stores (`datasets`, `splits`, `criticState`, `neatPopulations`, `gaPopulations`, `eliteArchive`, `lessons`, `agents`, `approvals`, `generatedSprites`, `checkpoints`, `onnxModels`) into `atelier-sprite-backup-YYYY-MM-DD.json`, preserving binary `ArrayBuffer` and typed arrays (`Uint8ClampedArray`, `Uint8Array`, `Float64Array`, etc.) losslessly. Imports pre-validate before clearing existing stores and verify every restored record via read-back.
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
  src/comparator/comparator.test.ts \
  src/t3/t3.test.ts
```

Note on `.playwright-cli/`: local headless Chromium checks verify end-to-end persistence (auto-restore after page reload, dataset & Elo agent restoration, lesson retrieval, generation, and backup round-trip) with zero browser console errors.

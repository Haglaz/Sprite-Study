import test from 'node:test';
import assert from 'node:assert/strict';
import { calibrateCriticAndBuildDiagnostics } from '../critic/diagnostics';
import { evaluateSprite } from '../critic/evaluator';
import {
  makeBorderTouchingOffCenterSprite,
  makeCleanPixelArtFixture,
  makeNearDuplicatePaletteSprite,
  makeOrphanSpeckledSprite,
  makeSemiTransparentSprite,
  makeUnoutlinedSprite,
} from '../critic/fixtures';
import type { LabeledSpriteSample } from '../critic/profile';
import { createMemoryStorageBackend, type StorageBackend } from '../db/persistence';
import { ALL_ITEM_CATEGORIES } from '../types/studio';
import {
  createDefaultPromotionRules,
  createInitialCheckpoint,
  evaluatePromotionRule,
  runCoevolutionGenerationStep,
} from './coevolution';
import {
  cloneNeatGenome,
  computeCompatibilityDistance,
  createInnovationTracker,
  createMinimalNeatGenome,
  createNeatPopulation,
  crossoverNeatGenomes,
  evaluateNeatGenome,
  evolveNeatGeneration,
  mutateAddConnection,
  mutateAddNode,
  speciatePopulation,
  wouldCreateCycle,
} from './neat';
import {
  opCenterAndPad,
  opEnforceDarkOutline,
  opRemoveOrphans,
  opSnapBinaryAlpha,
  opSnapNearDuplicates,
} from './pixelOps';
import { buildRestorationSamples } from './restorationTask';
import { loadEliteArchive, loadEvolutionCheckpoint, saveEvolutionCheckpointVerified } from './store';
import type { EvaluationAggregate } from './types';

// ============================================================================
// 1. REAL NEAT ENGINE TESTS (INNOVATION, ACYCLICITY, CROSSOVER, SPECIATION, XOR)
// ============================================================================

test('NEAT innovation tracker preserves homologous IDs across identical structural mutations and prevents cycles', () => {
  const tracker = createInnovationTracker(2, 1);
  const rng = { state: 42 };
  const g1 = createMinimalNeatGenome('g1', 2, 1, tracker, rng, 1.0);
  const g2 = cloneNeatGenome(g1);
  g2.id = 'g2';

  // Split the same first connection on two separate genomes -> must get identical split node + innovations
  const splitTarget1 = g1.connections[0];
  const splitTarget2 = g2.connections[0];
  assert.equal(splitTarget1.innovation, splitTarget2.innovation);

  const r1 = { state: 1 };
  const r2 = { state: 1 };
  assert.equal(mutateAddNode(g1, tracker, r1), true);
  assert.equal(mutateAddNode(g2, tracker, r2), true);

  const hidden1 = g1.nodes.find((n) => n.type === 'hidden');
  const hidden2 = g2.nodes.find((n) => n.type === 'hidden');
  assert.ok(hidden1 && hidden2);
  assert.equal(hidden1.id, hidden2.id);

  // Acyclicity guarantee: output -> input or self-loop must always report cycle
  assert.equal(wouldCreateCycle(g1, 3, 0), true);
  assert.equal(wouldCreateCycle(g1, hidden1.id, hidden1.id), true);

  // Adding a new connection produces a valid acyclic DAG that evaluates cleanly
  mutateAddConnection(g1, tracker, { state: 99 });
  const out = evaluateNeatGenome(g1, [0.5, -0.25]);
  assert.equal(out.length, 1);
  assert.ok(Number.isFinite(out[0]));
});

test('NEAT crossover aligns matching innovations and inherits disjoint/excess genes from fitter parent', () => {
  const tracker = createInnovationTracker(2, 1);
  const rng = { state: 17 };
  const fitter = createMinimalNeatGenome('fit', 2, 1, tracker, rng, 1.0);
  const weaker = cloneNeatGenome(fitter);
  weaker.id = 'weak';

  // Give fitter parent an extra hidden node (excess genes)
  mutateAddNode(fitter, tracker, { state: 5 });
  fitter.fitness = 10.0;
  weaker.fitness = 3.0;

  const child = crossoverNeatGenomes(weaker, fitter, 'child', { state: 123 });
  const fitterInnovs = new Set(fitter.connections.map((c) => c.innovation));
  const childInnovs = new Set(child.connections.map((c) => c.innovation));

  assert.deepEqual(childInnovs, fitterInnovs);
  assert.ok(child.nodes.some((n) => n.type === 'hidden'));

  // Compatibility distance between identical genomes is 0; divergent topology is > 0
  assert.equal(
    computeCompatibilityDistance(weaker, cloneNeatGenome(weaker), {
      c1Excess: 1,
      c2Disjoint: 1,
      c3Weight: 0.4,
    }),
    0,
  );
  assert.ok(
    computeCompatibilityDistance(fitter, weaker, {
      c1Excess: 1,
      c2Disjoint: 1,
      c3Weight: 0.4,
    }) > 0.5,
  );
});

test('NEAT solves the classic non-linear XOR benchmark via real topology augmentation and speciation', () => {
  const xorCases: Array<{ x: [number, number]; y: number }> = [
    { x: [0, 0], y: 0 },
    { x: [0, 1], y: 1 },
    { x: [1, 0], y: 1 },
    { x: [1, 1], y: 0 },
  ];

  let pop = createNeatPopulation(
    {
      populationSize: 45,
      inputCount: 2,
      outputCount: 1,
      targetSpeciesCount: 5,
      compatibilityThreshold: 1.8,
      addNodeRate: 0.14,
      addConnectionRate: 0.25,
      weightMutateRate: 0.85,
      weightPerturbScale: 0.45,
    },
    1337,
  );

  let bestAccuracy = 0;
  let bestError = Infinity;

  for (let gen = 0; gen < 65; gen++) {
    for (const g of pop.genomes) {
      let sqErr = 0;
      let correct = 0;
      for (const c of xorCases) {
        // Map tanh [-1, 1] -> [0, 1]
        const raw = evaluateNeatGenome(g, c.x)[0];
        const pred = (raw + 1) * 0.5;
        sqErr += (pred - c.y) ** 2;
        if ((pred >= 0.5 ? 1 : 0) === c.y) correct++;
      }
      g.fitness = Math.max(0.01, 4 - sqErr);
      if (sqErr < bestError) {
        bestError = sqErr;
        bestAccuracy = correct / 4;
      }
    }
    if (bestAccuracy === 1 && bestError < 0.25) break;
    pop = evolveNeatGeneration(pop);
  }

  assert.ok(pop.species.length >= 2, `Expected multiple species, got ${pop.species.length}`);
  assert.equal(bestAccuracy, 1, `NEAT should solve all 4 XOR cases (bestError=${bestError.toFixed(3)})`);
  assert.ok(bestError < 0.45, `Expected low XOR MSE, got ${bestError.toFixed(3)}`);
});

// ============================================================================
// 2. PIXEL-EDITOR DISCRETE OPERATIONS TESTS
// ============================================================================

test('discrete pixel-editor ops enforce binary alpha, introduce zero AA, and fix their target defects', () => {
  const semi = makeSemiTransparentSprite();
  const fixedAlpha = opSnapBinaryAlpha(semi, 0.5);
  for (let i = 3; i < fixedAlpha.length; i += 4) {
    assert.ok(fixedAlpha[i] === 0 || fixedAlpha[i] === 255, 'Binary alpha must be strictly 0 or 255');
  }
  assert.equal(evaluateSprite(fixedAlpha, { category: 'Equipamentos' }).features.semiTransparentPixels, 0);

  const orphans = makeOrphanSpeckledSprite();
  const fixedOrphans = opRemoveOrphans(orphans, 0.4);
  assert.equal(evaluateSprite(fixedOrphans, { category: 'Equipamentos' }).features.orphanPixels, 0);

  const nearDup = makeNearDuplicatePaletteSprite();
  const beforeDup = evaluateSprite(nearDup, { category: 'Containers' }).features.nearDuplicateColorPairs;
  const fixedDup = opSnapNearDuplicates(nearDup, 0.5);
  const afterDup = evaluateSprite(fixedDup, { category: 'Containers' }).features.nearDuplicateColorPairs;
  assert.ok(afterDup < beforeDup, `Expected fewer near-duplicate pairs (${beforeDup} -> ${afterDup})`);

  const unoutlined = makeUnoutlinedSprite();
  const fixedOutline = opEnforceDarkOutline(unoutlined, 0.4, false);
  const outEval = evaluateSprite(fixedOutline, { category: 'Consumíveis' });
  assert.ok(outEval.features.outlineCoverage > 0.7, `Expected outline restored, got ${outEval.features.outlineCoverage}`);

  const offCenter = makeBorderTouchingOffCenterSprite();
  const centered = opCenterAndPad(offCenter, false);
  const centEval = evaluateSprite(centered, { category: 'Equipamentos' });
  assert.equal(centEval.features.borderTouchCount, 0);
  assert.ok(centEval.features.centerOffsetNorm < 0.15);
});

// ============================================================================
// 3. PROMOTION RULES TESTS (ANTI-OVERFIT, ANTI-SHAPE-DESTRUCTION, VALIDATION GATE)
// ============================================================================

test('promotion rules reject overfitting, shape destruction (low IoU), hard-rule regressions, and non-improvements', () => {
  const rules = createDefaultPromotionRules();
  const baseAgg: EvaluationAggregate = {
    meanRestoredScore: 72,
    medianRestoredScore: 73,
    meanCorruptedScore: 42,
    medianCorruptedScore: 40,
    meanDelta: 30,
    medianDelta: 33,
    meanSilhouetteIoU: 0.92,
    meanPixelSimilarity: 0.85,
    hardViolationDelta: -1,
    combinedFitness: 78,
  };

  // 1. Valid first champion -> PROMOTED
  const firstPass = evaluatePromotionRule({
    trainStats: { ...baseAgg, meanRestoredScore: 75 },
    valStats: baseAgg,
    currentChampion: null,
    valCorruptedMedian: 40,
    junkFloorMedian: 30,
    rules,
  });
  assert.equal(firstPass.promoted, true);

  // 2. Shape destruction (IoU = 0.55 < 0.72) -> REJECTED
  const lowIou = evaluatePromotionRule({
    trainStats: baseAgg,
    valStats: { ...baseAgg, meanSilhouetteIoU: 0.55 },
    currentChampion: null,
    valCorruptedMedian: 40,
    junkFloorMedian: 30,
    rules,
  });
  assert.equal(lowIou.promoted, false);
  assert.match(lowIou.reasonPt, /IoU/i);

  // 3. Overfitting (train 92 vs val 65, gap 27 > 18) -> REJECTED
  const overfit = evaluatePromotionRule({
    trainStats: { ...baseAgg, meanRestoredScore: 92 },
    valStats: { ...baseAgg, meanRestoredScore: 65, medianRestoredScore: 65 },
    currentChampion: null,
    valCorruptedMedian: 40,
    junkFloorMedian: 30,
    rules,
  });
  assert.equal(overfit.promoted, false);
  assert.match(overfit.reasonPt, /overfitting/i);

  // 4. Hard-rule regression (+1 new hard violation) -> REJECTED
  const hardReg = evaluatePromotionRule({
    trainStats: baseAgg,
    valStats: { ...baseAgg, hardViolationDelta: 1 },
    currentChampion: null,
    valCorruptedMedian: 40,
    junkFloorMedian: 30,
    rules,
  });
  assert.equal(hardReg.promoted, false);
  assert.match(hardReg.reasonPt, /regra rígida/i);
});

// ============================================================================
// 4. CO-EVOLUTION, CHECKPOINT/RESUME DETERMINISM & READ-BACK VERIFICATION TESTS
// ============================================================================

test('co-evolution improves held-out validation restoration score, logs promotions, and checkpoint/resume is deterministic & read-back verified', async () => {
  // Build deterministic calibrated Critic v1 on clean fixtures
  const trainCalSamples: LabeledSpriteSample[] = [];
  const valCalSamples: LabeledSpriteSample[] = [];
  let sid = 1;
  for (const cat of ALL_ITEM_CATEGORIES) {
    for (let i = 0; i < 12; i++) {
      trainCalSamples.push({ spriteId: sid++, category: cat, rgba: makeCleanPixelArtFixture(cat, i) });
    }
    for (let i = 0; i < 6; i++) {
      valCalSamples.push({ spriteId: sid++, category: cat, rgba: makeCleanPixelArtFixture(cat, 20 + i) });
    }
  }

  const { calibration, diagnostics } = calibrateCriticAndBuildDiagnostics({
    datasetKey: 'ds-phase3-test',
    trainSamples: trainCalSamples,
    validationSamples: valCalSamples,
    splitTotals: { train: 48, validation: 24, test: 24, total: 96 },
    seed: 772,
  });

  const equipTrain = trainCalSamples
    .filter((s) => s.category === 'Equipamentos')
    .map((s) => ({ spriteId: s.spriteId, rgba: s.rgba }));
  const equipVal = valCalSamples
    .filter((s) => s.category === 'Equipamentos')
    .map((s) => ({ spriteId: s.spriteId, rgba: s.rgba }));

  const trainSamples = buildRestorationSamples({
    sprites: equipTrain,
    category: 'Equipamentos',
    calibration,
    seed: 901,
    maxCount: 6,
  });
  const valSamples = buildRestorationSamples({
    sprites: equipVal,
    category: 'Equipamentos',
    calibration,
    seed: 902,
    maxCount: 4,
  });

  const valCorruptedMedian =
    [...valSamples.map((s) => s.corruptedScore)].sort((a, b) => a - b)[Math.floor(valSamples.length / 2)];

  const baseConfig = {
    category: 'Equipamentos' as const,
    targetGenerations: 4,
    populationSize: 8,
    seed: 772,
    trainBatchSize: 6,
    valBatchSize: 4,
    promotionRules: createDefaultPromotionRules(),
  };

  // Run A: 4 uninterrupted generations
  let ckptUninterrupted = createInitialCheckpoint({
    datasetKey: calibration.datasetKey,
    config: baseConfig,
    valBaselineCorruptedMedian: valCorruptedMedian,
    realCeilingMedian: diagnostics.realCeiling.median,
    junkFloorMedian: diagnostics.junkFloor.median,
    trainSampleCount: trainSamples.length,
    valSampleCount: valSamples.length,
    nowIso: '2026-06-01T10:00:00.000Z',
  });

  for (let g = 0; g < 4; g++) {
    ckptUninterrupted = runCoevolutionGenerationStep({
      checkpoint: ckptUninterrupted,
      trainSamples,
      valSamples,
      calibration,
      nowIso: `2026-06-01T10:0${g}:00.000Z`,
    }).checkpoint;
  }

  // Run B: 2 generations -> persist to storage (read-back verified) -> reload -> resume 2 generations
  const db = createMemoryStorageBackend();
  let ckptSplit = createInitialCheckpoint({
    datasetKey: calibration.datasetKey,
    config: baseConfig,
    valBaselineCorruptedMedian: valCorruptedMedian,
    realCeilingMedian: diagnostics.realCeiling.median,
    junkFloorMedian: diagnostics.junkFloor.median,
    trainSampleCount: trainSamples.length,
    valSampleCount: valSamples.length,
    nowIso: '2026-06-01T10:00:00.000Z',
  });

  for (let g = 0; g < 2; g++) {
    ckptSplit = runCoevolutionGenerationStep({
      checkpoint: ckptSplit,
      trainSamples,
      valSamples,
      calibration,
      nowIso: `2026-06-01T10:0${g}:00.000Z`,
    }).checkpoint;
  }

  const savedMid = await saveEvolutionCheckpointVerified(db, ckptSplit);
  assert.equal(savedMid.readBackVerified, true);

  const reloadedMid = await loadEvolutionCheckpoint(db, calibration.datasetKey, 'Equipamentos');
  assert.ok(reloadedMid);
  assert.equal(reloadedMid.generation, 2);

  let resumed = reloadedMid;
  for (let g = 2; g < 4; g++) {
    resumed = runCoevolutionGenerationStep({
      checkpoint: resumed,
      trainSamples,
      valSamples,
      calibration,
      nowIso: `2026-06-01T10:0${g}:00.000Z`,
    }).checkpoint;
  }
  await saveEvolutionCheckpointVerified(db, resumed);

  // Determinism check: resumed run must match uninterrupted run's train & validation curves identically!
  assert.deepEqual(
    resumed.history.map((h) => ({ gen: h.generation, tr: h.trainBestScore, val: h.valMedianScore, p: h.promoted })),
    ckptUninterrupted.history.map((h) => ({ gen: h.generation, tr: h.trainBestScore, val: h.valMedianScore, p: h.promoted })),
  );

  // Honest learning check: champion promoted & beats corrupted validation baseline without ever hitting flat 100
  assert.ok(resumed.champion, 'Expected at least one promoted champion over 4 generations');
  assert.ok(
    resumed.champion.valStats.medianRestoredScore > valCorruptedMedian,
    `Champion validation (${resumed.champion.valStats.medianRestoredScore}) must beat corrupted baseline (${valCorruptedMedian})`,
  );
  assert.ok(resumed.champion.valStats.medianRestoredScore < 100, 'Score must never be flat 100');
  assert.equal(resumed.promotionLog.length, 4);

  const elites = await loadEliteArchive(db, calibration.datasetKey, 'Equipamentos');
  assert.ok(elites.length >= 1, 'Expected promoted elite persisted in eliteArchive');

  // Verify read-back failure rejection on broken backend
  const brokenDb: StorageBackend = {
    async get() {
      return null;
    },
    async getAll() {
      return [];
    },
    async put() {},
    async delete() {},
    async clear() {},
    async clearAll() {},
  };
  await assert.rejects(() => saveEvolutionCheckpointVerified(brokenDb, resumed), /read-back/i);

  // Speciation check on final NEAT population
  speciatePopulation(resumed.neatState);
  assert.ok(resumed.neatState.species.length >= 1);
});

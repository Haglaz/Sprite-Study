import { buildStandardJunkFloorFixtures } from '../critic/fixtures';
import { evaluateSprite } from '../critic/evaluator';
import type { CriticCalibration } from '../critic/types';
import type { StoredDatasetSplit } from '../db/schema';
import { getSplitSpriteIds } from '../db/split';
import { generateSpriteCandidates } from '../generation/pipeline';
import type { SpriteIndexState } from '../generation/types';
import type { LessonRecord } from '../lessons/types';
import type { ItemCategory, ParsedDatArchive, ParsedSprArchive } from '../types/studio';
import { decodeSpritePixels, SPRITE_SIZE } from '../utils/tibiaParser';
import {
  checkT3HardConstraints,
  runT3SamplingAndRanking,
  type T3InferenceRunner,
} from './sampler';
import type {
  T3MethodBenchmarkSummary,
  T3ModelCard,
  T3TestSplitComparisonReport,
} from './types';
import { computeSplitHash } from './vocabulary';

function computeMedian(sorted: number[]): number {
  if (sorted.length === 0) return 0;
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function summarizeScores(params: {
  label: string;
  scores: number[];
  originalities?: number[];
  binaryAlphaCount: number;
  outlineClosureCount: number;
  connectedCount: number;
  memorizedCount?: number;
}): T3MethodBenchmarkSummary {
  const n = params.scores.length;
  const sorted = [...params.scores].sort((a, b) => a - b);
  const mean = n > 0 ? params.scores.reduce((s, v) => s + v, 0) / n : 0;
  const meanOrig =
    params.originalities && params.originalities.length > 0
      ? Math.round(
          (params.originalities.reduce((s, v) => s + v, 0) / params.originalities.length) * 10,
        ) / 10
      : null;

  return {
    label: params.label,
    sampleCount: n,
    medianScore: Math.round(computeMedian(sorted) * 10) / 10,
    meanScore: Math.round(mean * 10) / 10,
    minScore: n > 0 ? Math.round(sorted[0] * 10) / 10 : 0,
    maxScore: n > 0 ? Math.round(sorted[n - 1] * 10) / 10 : 0,
    meanOriginality: meanOrig,
    binaryAlphaRate: n > 0 ? Math.round((params.binaryAlphaCount / n) * 1000) / 1000 : 0,
    outlineClosureRate: n > 0 ? Math.round((params.outlineClosureCount / n) * 1000) / 1000 : 0,
    connectedSilhouetteRate: n > 0 ? Math.round((params.connectedCount / n) * 1000) / 1000 : 0,
    memorizationRate:
      params.memorizedCount !== undefined && n > 0
        ? Math.round((params.memorizedCount / n) * 1000) / 1000
        : null,
  };
}

interface TestPromptSample {
  spriteId: number;
  category: ItemCategory;
  prompt: string;
}

/**
 * Selects a deterministic stratified sample of prompts derived from the held-out `test` split
 * categories/flags (never exposing test pixels as donors to T1/T2/T3!).
 */
function buildTestBenchmarkPrompts(
  dat: ParsedDatArchive,
  split: StoredDatasetSplit,
  maxPrompts: number,
): TestPromptSample[] {
  const itemBySprite = new Map<number, { category: ItemCategory; name?: string }>();
  for (const item of dat.items) {
    for (const sid of item.spriteIds) {
      if (sid > 0 && !itemBySprite.has(sid)) {
        itemBySprite.set(sid, { category: item.category, name: item.flags.marketName });
      }
    }
  }

  const testIds = getSplitSpriteIds(split, 'test');
  const picked: TestPromptSample[] = [];
  const step = Math.max(1, Math.floor(testIds.length / Math.max(1, maxPrompts)));

  const fallbackTagsByCat: Record<ItemCategory, string> = {
    Equipamentos: 'armadura escudo aco dourado paladino',
    Consumíveis: 'pocao rubi elixir brilhante',
    Containers: 'bau madeira mochila dourada',
    Decoração: 'calice ouro sagrado cristal',
  };

  for (let i = 0; i < testIds.length && picked.length < maxPrompts; i += step) {
    const sid = testIds[i];
    const meta = itemBySprite.get(sid);
    const cat: ItemCategory = meta?.category ?? 'Equipamentos';
    const prompt = meta?.name?.trim()
      ? `${meta.name} (${cat.toLowerCase()})`
      : fallbackTagsByCat[cat];
    picked.push({ spriteId: sid, category: cat, prompt });
  }

  return picked;
}

/**
 * Phase 7(E): Explicit final test-split comparison report comparing T3 (ONNX)
 * vs T1, T2, Real held-out `test` sprites, and the Junk Floor.
 * If T3 is worse than T1/T2 (or fails originality/junk margins), keeps T3 disabled by default and states that plainly.
 */
export async function evaluateT3OnTestSplit(params: {
  runner: T3InferenceRunner;
  modelCard: T3ModelCard;
  dat: ParsedDatArchive;
  spr: ParsedSprArchive;
  split: StoredDatasetSplit;
  calibration: CriticCalibration;
  spriteIndex: SpriteIndexState;
  lessons: LessonRecord[];
  maxRealTestSprites?: number;
  maxPromptCount?: number;
}): Promise<T3TestSplitComparisonReport> {
  const {
    runner,
    modelCard,
    dat,
    spr,
    split,
    calibration,
    spriteIndex,
    lessons,
    maxRealTestSprites = 64,
    maxPromptCount = 6,
  } = params;

  const splitHash = computeSplitHash(split);

  const catBySprite = new Map<number, ItemCategory>();
  const tileBySprite = new Map<number, boolean>();
  for (const item of dat.items) {
    const isTileItem = Boolean(
      item.flags.ground ||
        item.flags.groundBorder ||
        item.width > 1 ||
        item.height > 1,
    );
    if (item.primarySpriteId > 0 && !catBySprite.has(item.primarySpriteId)) {
      catBySprite.set(item.primarySpriteId, item.category);
      tileBySprite.set(item.primarySpriteId, isTileItem);
    }
    for (const sid of item.spriteIds) {
      if (sid > 0 && !catBySprite.has(sid)) {
        catBySprite.set(sid, item.category);
        tileBySprite.set(sid, isTileItem);
      }
    }
  }

  // 1. Evaluate Real held-out `test` split sprites stratified across categories (never seen in train/validation)
  const realScores: number[] = [];
  let realBinAlpha = 0;
  let realOutline = 0;
  let realConn = 0;

  const categories = Object.keys(split.byCategory) as ItemCategory[];
  const perCatTarget = Math.max(1, Math.ceil(maxRealTestSprites / Math.max(1, categories.length)));

  for (const cat of categories) {
    const catTestIds = split.byCategory[cat]?.test ?? [];
    const stepCat = Math.max(1, Math.floor(catTestIds.length / perCatTarget));
    let taken = 0;

    for (
      let i = 0;
      i < catTestIds.length && taken < perCatTarget && realScores.length < maxRealTestSprites;
      i += stepCat
    ) {
      const sid = catTestIds[i];
      const rgbaBuf = decodeSpritePixels(spr, sid);
      let hasOpaque = false;
      for (let p = 3; p < rgbaBuf.length; p += 4) {
        if (rgbaBuf[p] >= 128) {
          hasOpaque = true;
          break;
        }
      }
      if (!hasOpaque) continue;
      const isTile = tileBySprite.get(sid) ?? false;
      const ev = evaluateSprite(rgbaBuf, { category: cat, isTileOverride: isTile, calibration });
      const chk = checkT3HardConstraints(rgbaBuf, isTile);
      realScores.push(ev.score);
      if (chk.binaryAlphaOk) realBinAlpha++;
      if (chk.outlineClosureOk) realOutline++;
      if (chk.connectedSilhouetteOk) realConn++;
      taken++;
    }
  }

  const realTest = summarizeScores({
    label: 'Reais (split test)',
    scores: realScores,
    binaryAlphaCount: realBinAlpha,
    outlineClosureCount: realOutline,
    connectedCount: realConn,
  });

  // 2. Evaluate Synthetic Junk Floor baselines
  const junkFixtures = buildStandardJunkFloorFixtures(split.seed + 700);
  const junkScores: number[] = [];
  let junkBinAlpha = 0;
  let junkOutline = 0;
  let junkConn = 0;
  for (const jf of junkFixtures) {
    const ev = evaluateSprite(jf.rgba, { category: 'Equipamentos', isTileOverride: false, calibration });
    const chk = checkT3HardConstraints(jf.rgba, false);
    junkScores.push(ev.score);
    if (chk.binaryAlphaOk) junkBinAlpha++;
    if (chk.outlineClosureOk) junkOutline++;
    if (chk.connectedSilhouetteOk) junkConn++;
  }
  const junkFloor = summarizeScores({
    label: 'Piso Lixo (defeitos)',
    scores: junkScores,
    binaryAlphaCount: junkBinAlpha,
    outlineClosureCount: junkOutline,
    connectedCount: junkConn,
  });

  // 3. Evaluate T1, T2, and T3 on the same test-derived prompt suite
  const prompts = buildTestBenchmarkPrompts(dat, split, maxPromptCount);

  const t1Scores: number[] = [];
  const t1Orig: number[] = [];
  let t1Bin = 0;
  let t1Out = 0;
  let t1Con = 0;

  const t2Scores: number[] = [];
  const t2Orig: number[] = [];
  let t2Bin = 0;
  let t2Out = 0;
  let t2Con = 0;

  const t3Scores: number[] = [];
  const t3Orig: number[] = [];
  let t3Bin = 0;
  let t3Out = 0;
  let t3Con = 0;
  let t3Mem = 0;

  for (let i = 0; i < prompts.length; i++) {
    const spec = prompts[i];
    const seed = split.seed + 7700 + i * 131;
    const isTile = tileBySprite.get(spec.spriteId) ?? false;

    const resT1 = await generateSpriteCandidates({
      datasetKey: split.datasetKey,
      prompt: spec.prompt,
      category: spec.category,
      tier: 'T1',
      useGemini: false,
      maxDiffPixels: 28,
      candidateCount: 3,
      topKReturn: 1,
      seed,
      index: spriteIndex,
      calibration,
      lessons,
    });
    if (resT1.candidates[0]) {
      const c1 = resT1.candidates[0];
      const chk1 = checkT3HardConstraints(c1.rgba, isTile);
      t1Scores.push(c1.evaluation.score);
      t1Orig.push(c1.nearestReal.originalityScore);
      if (chk1.binaryAlphaOk) t1Bin++;
      if (chk1.outlineClosureOk) t1Out++;
      if (chk1.connectedSilhouetteOk) t1Con++;
    }

    const resT2 = await generateSpriteCandidates({
      datasetKey: split.datasetKey,
      prompt: spec.prompt,
      category: spec.category,
      tier: 'T2',
      useGemini: false,
      maxDiffPixels: 32,
      candidateCount: 3,
      topKReturn: 1,
      seed,
      index: spriteIndex,
      calibration,
      lessons,
    });
    if (resT2.candidates[0]) {
      const c2 = resT2.candidates[0];
      const chk2 = checkT3HardConstraints(c2.rgba, isTile);
      t2Scores.push(c2.evaluation.score);
      t2Orig.push(c2.nearestReal.originalityScore);
      if (chk2.binaryAlphaOk) t2Bin++;
      if (chk2.outlineClosureOk) t2Out++;
      if (chk2.connectedSilhouetteOk) t2Con++;
    }

    const resT3 = await runT3SamplingAndRanking({
      runner,
      modelCard,
      category: spec.category,
      prompt: spec.prompt,
      calibration,
      spriteIndex,
      candidateCount: 2,
      seed,
    });
    if (resT3.candidates[0]) {
      const c3 = resT3.candidates[0];
      t3Scores.push(c3.evaluation.score);
      if (c3.nearestReal) t3Orig.push(c3.nearestReal.originalityScore);
      if (c3.constraints.binaryAlphaOk) t3Bin++;
      if (c3.constraints.outlineClosureOk) t3Out++;
      if (c3.constraints.connectedSilhouetteOk) t3Con++;
      if (c3.memorizedCopyWarning) t3Mem++;
    }
  }

  const t1 = summarizeScores({
    label: 'Tier T1 (Recolor + Diff)',
    scores: t1Scores,
    originalities: t1Orig,
    binaryAlphaCount: t1Bin,
    outlineClosureCount: t1Out,
    connectedCount: t1Con,
  });

  const t2 = summarizeScores({
    label: 'Tier T2 (Compósito + Reparo)',
    scores: t2Scores,
    originalities: t2Orig,
    binaryAlphaCount: t2Bin,
    outlineClosureCount: t2Out,
    connectedCount: t2Con,
  });

  const t3 = summarizeScores({
    label: `Tier T3 (${modelCard.version})`,
    scores: t3Scores,
    originalities: t3Orig,
    binaryAlphaCount: t3Bin,
    outlineClosureCount: t3Out,
    connectedCount: t3Con,
    memorizedCount: t3Mem,
  });

  // Strict Honesty Gate: disable T3 by default whenever T3 fails to match/beat T1 & T2 or stays near Junk Floor
  const bestBaselineMedian = Math.max(t1.medianScore, t2.medianScore);
  const nearJunkFloor = t3.medianScore <= junkFloor.medianScore + 10;
  const highMemorization = t3.memorizationRate !== null && t3.memorizationRate > 0.5;
  const belowBaselines = t3.medianScore < bestBaselineMedian;

  const t3IsWorse = t3.sampleCount === 0 || belowBaselines || nearJunkFloor || highMemorization;
  const recommendedEnabled = !t3IsWorse;

  let verdictSummaryPt: string;
  if (t3IsWorse) {
    const reasons: string[] = [];
    if (belowBaselines) {
      reasons.push(
        `mediana T3 (${t3.medianScore.toFixed(1)}) inferior ao melhor baseline T1/T2 (${bestBaselineMedian.toFixed(1)})`,
      );
    }
    if (nearJunkFloor) {
      reasons.push(
        `próxima ao piso lixo (${junkFloor.medianScore.toFixed(1)})`,
      );
    }
    if (highMemorization) {
      reasons.push(
        `taxa de cópia memorizada elevada (${Math.round((t3.memorizationRate ?? 0) * 100)}%)`,
      );
    }
    verdictSummaryPt = `Modelo T3 DESATIVADO por padrão no split test (${reasons.join('; ')}). Os métodos T1/T2 continuam como geradores principais até novo treino no Colab.`;
  } else {
    verdictSummaryPt = `Modelo T3 APROVADO no split test: mediana ${t3.medianScore.toFixed(1)} superou/igualou T1 (${t1.medianScore.toFixed(1)}), T2 (${t2.medianScore.toFixed(1)}) e o piso lixo (${junkFloor.medianScore.toFixed(1)}) com ${Math.round((t3.memorizationRate ?? 0) * 100)}% de cópia memorizada.`;
  }

  return {
    evaluatedAt: new Date().toISOString(),
    splitHash,
    testSpriteCount: realTest.sampleCount,
    benchmarkPromptCount: prompts.length,
    realTest,
    junkFloor,
    t1,
    t2,
    t3,
    t3IsWorse,
    recommendedEnabled,
    verdictSummaryPt,
  };
}

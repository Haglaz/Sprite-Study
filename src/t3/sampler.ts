import { evaluateSprite } from '../critic/evaluator';
import type { CriticCalibration } from '../critic/types';
import { createSeededRng } from '../db/split';
import { findNearestRealSprite, normalizePromptTokens } from '../generation/indexer';
import type { SpriteIndexState } from '../generation/types';
import { ALL_ITEM_CATEGORIES, type ItemCategory } from '../types/studio';
import { SPRITE_SIZE } from '../utils/tibiaParser';
import type {
  T3ConstraintCheckResult,
  T3ModelCard,
  T3SamplingBatchResult,
  T3SamplingCandidate,
} from './types';
import { decodeVocabGridToRgba } from './vocabulary';

function calcLuma(r: number, g: number, b: number): number {
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255;
}

/**
 * Validates the 3 mandatory structural constraints before accepting a T3 sample:
 *  1) Binary alpha (0 | 255)
 *  2) Outline closure around the outer perimeter (for non-tiles)
 *  3) Connected silhouette (largest 4-connected opaque region >= 88% of opaque mass)
 */
export function checkT3HardConstraints(
  rgba: Uint8ClampedArray,
  isTile = false,
): T3ConstraintCheckResult {
  const N = SPRITE_SIZE * SPRITE_SIZE;
  let binaryAlphaOk = true;
  let opaqueCount = 0;

  for (let p = 0; p < N; p++) {
    const a = rgba[p * 4 + 3];
    if (a !== 0 && a !== 255) {
      binaryAlphaOk = false;
    }
    if (a >= 128) opaqueCount++;
  }

  if (opaqueCount < 24) {
    return {
      passed: false,
      binaryAlphaOk,
      outlineClosureOk: false,
      connectedSilhouetteOk: false,
      failures: ['Silhueta vazia ou abaixo de 24 pixels opacos'],
      mainComponentRatio: 0,
      boundaryDarkRatio: 0,
    };
  }

  // Connected component analysis (4-connected)
  const visited = new Uint8Array(N);
  let largestComp = 0;
  const queue = new Int32Array(N);

  for (let start = 0; start < N; start++) {
    if (visited[start] || rgba[start * 4 + 3] < 128) continue;
    let head = 0;
    let tail = 0;
    queue[tail++] = start;
    visited[start] = 1;
    let size = 0;

    while (head < tail) {
      const p = queue[head++];
      size++;
      const x = p % SPRITE_SIZE;
      const y = (p - x) / SPRITE_SIZE;
      if (x > 0 && !visited[p - 1] && rgba[(p - 1) * 4 + 3] >= 128) {
        visited[p - 1] = 1;
        queue[tail++] = p - 1;
      }
      if (x < SPRITE_SIZE - 1 && !visited[p + 1] && rgba[(p + 1) * 4 + 3] >= 128) {
        visited[p + 1] = 1;
        queue[tail++] = p + 1;
      }
      if (y > 0 && !visited[p - SPRITE_SIZE] && rgba[(p - SPRITE_SIZE) * 4 + 3] >= 128) {
        visited[p - SPRITE_SIZE] = 1;
        queue[tail++] = p - SPRITE_SIZE;
      }
      if (
        y < SPRITE_SIZE - 1 &&
        !visited[p + SPRITE_SIZE] &&
        rgba[(p + SPRITE_SIZE) * 4 + 3] >= 128
      ) {
        visited[p + SPRITE_SIZE] = 1;
        queue[tail++] = p + SPRITE_SIZE;
      }
    }

    if (size > largestComp) largestComp = size;
  }

  const mainComponentRatio = largestComp / opaqueCount;
  const connectedSilhouetteOk = mainComponentRatio >= 0.88;

  // Outline closure on silhouette perimeter
  let perimCount = 0;
  let darkPerimCount = 0;
  if (!isTile) {
    for (let y = 0; y < SPRITE_SIZE; y++) {
      for (let x = 0; x < SPRITE_SIZE; x++) {
        const p = y * SPRITE_SIZE + x;
        const i = p * 4;
        if (rgba[i + 3] < 128) continue;
        const isEdge =
          x === 0 ||
          x === SPRITE_SIZE - 1 ||
          y === 0 ||
          y === SPRITE_SIZE - 1 ||
          rgba[(p - 1) * 4 + 3] < 128 ||
          rgba[(p + 1) * 4 + 3] < 128 ||
          rgba[(p - SPRITE_SIZE) * 4 + 3] < 128 ||
          rgba[(p + SPRITE_SIZE) * 4 + 3] < 128;
        if (!isEdge) continue;
        perimCount++;
        const luma = calcLuma(rgba[i], rgba[i + 1], rgba[i + 2]);
        if (luma <= 0.36) darkPerimCount++;
      }
    }
  }
  const boundaryDarkRatio = isTile ? 1 : perimCount > 0 ? darkPerimCount / perimCount : 0;
  const outlineClosureOk = isTile || boundaryDarkRatio >= 0.55;

  const failures: string[] = [];
  if (!binaryAlphaOk) failures.push('Alpha semi-transparente detectado');
  if (!outlineClosureOk) {
    failures.push(
      `Contorno aberto na borda (${Math.round(boundaryDarkRatio * 100)}% escuro < 55%)`,
    );
  }
  if (!connectedSilhouetteOk) {
    failures.push(
      `Silhueta fragmentada (${Math.round(mainComponentRatio * 100)}% conectada < 88%)`,
    );
  }

  return {
    passed: binaryAlphaOk && outlineClosureOk && connectedSilhouetteOk,
    binaryAlphaOk,
    outlineClosureOk,
    connectedSilhouetteOk,
    failures,
    mainComponentRatio: Math.round(mainComponentRatio * 1000) / 1000,
    boundaryDarkRatio: Math.round(boundaryDarkRatio * 1000) / 1000,
  };
}

export interface T3InferenceRunner {
  executionProvider: 'webgpu' | 'wasm' | 'custom-runner';
  runGenerator(params: {
    tokens: BigInt64Array;
    categoryIds: BigInt64Array;
    tagVec: Float32Array;
    batchSize: number;
    seqLen: number;
    tagDim: number;
  }): Promise<Float32Array>;
  runReranker?: (params: {
    tokens: BigInt64Array;
    categoryIds: BigInt64Array;
    batchSize: number;
    seqLen: number;
  }) => Promise<Float32Array | null>;
}

function encodeCategoryAndTags(
  modelCard: T3ModelCard,
  category: ItemCategory,
  prompt: string,
): { categoryIdx: number; tagVec: Float32Array; tagDim: number } {
  const catList = modelCard.categories?.length ? modelCard.categories : ALL_ITEM_CATEGORIES;
  const catIdx = Math.max(0, catList.indexOf(category));
  const tagVocab = modelCard.tagVocab?.length ? modelCard.tagVocab : ['geral'];
  const tagDim = tagVocab.length;
  const vec = new Float32Array(tagDim);
  const tokens = new Set(normalizePromptTokens(prompt));
  for (let i = 0; i < tagDim; i++) {
    if (tokens.has(tagVocab[i].toLowerCase())) {
      vec[i] = 1.0;
    }
  }
  return { categoryIdx: catIdx, tagVec: vec, tagDim };
}

function sampleFromTopKLogits(
  logits: Float32Array,
  offset: number,
  vocabSize: number,
  temperature: number,
  topK: number,
  rng: () => number,
): { token: number; prob: number } {
  const temp = Math.max(0.1, temperature);
  const k = Math.max(1, Math.min(vocabSize, topK));

  // Collect top-K indices cleanly
  const indexed: Array<{ idx: number; val: number }> = [];
  for (let v = 0; v < vocabSize; v++) {
    indexed.push({ idx: v, val: logits[offset + v] / temp });
  }
  indexed.sort((a, b) => b.val - a.val);
  const top = indexed.slice(0, k);
  const maxLogit = top[0].val;

  let sumExp = 0;
  const exps = new Float64Array(top.length);
  for (let i = 0; i < top.length; i++) {
    const e = Math.exp(top[i].val - maxLogit);
    exps[i] = e;
    sumExp += e;
  }

  let r = rng() * sumExp;
  for (let i = 0; i < top.length; i++) {
    r -= exps[i];
    if (r <= 0 || i === top.length - 1) {
      return { token: top[i].idx, prob: exps[i] / sumExp };
    }
  }
  return { token: top[0].idx, prob: exps[0] / sumExp };
}

/**
 * Generates one token grid using iterative masked-token sampling + Classifier-Free Guidance.
 */
async function sampleSingleT3Grid(params: {
  runner: T3InferenceRunner;
  modelCard: T3ModelCard;
  categoryIdx: number;
  tagVec: Float32Array;
  tagDim: number;
  seed: number;
}): Promise<Uint8Array> {
  const { runner, modelCard, categoryIdx, tagVec, tagDim, seed } = params;
  const rng = createSeededRng(seed);
  const settings = modelCard.samplerSettings;
  const seqLen = settings.tokenCount || 1024;
  const vocabSize = settings.vocabSize || modelCard.vocabulary.length + 1;
  const maskTok = settings.maskTokenId ?? vocabSize;
  const steps = Math.max(1, Math.min(24, settings.steps || 8));
  const cfgScale = settings.cfgScale ?? 2.0;
  const nullCatId = (modelCard.categories?.length || ALL_ITEM_CATEGORIES.length);

  const curTokens = new Int32Array(seqLen).fill(maskTok);
  const catCond = new BigInt64Array([BigInt(categoryIdx)]);
  const catUncond = new BigInt64Array([BigInt(nullCatId)]);
  const zeroTags = new Float32Array(tagDim);

  const tokensPair = new BigInt64Array(seqLen * 2);
  const catsPair = new BigInt64Array([catCond[0], catUncond[0]]);
  const tagsPair = new Float32Array(tagDim * 2);
  tagsPair.set(tagVec, 0);
  tagsPair.set(zeroTags, tagDim);

  for (let s = 0; s < steps; s++) {
    for (let p = 0; p < seqLen; p++) {
      const tokVal = BigInt(curTokens[p]);
      tokensPair[p] = tokVal;
      tokensPair[seqLen + p] = tokVal;
    }

    const logitsPair = await runner.runGenerator({
      tokens: tokensPair,
      categoryIds: catsPair,
      tagVec: tagsPair,
      batchSize: 2,
      seqLen,
      tagDim,
    });

    const strideCond = 0;
    const strideUncond = seqLen * vocabSize;
    const blended = new Float32Array(seqLen * vocabSize);
    for (let p = 0; p < seqLen; p++) {
      const base = p * vocabSize;
      for (let v = 0; v < vocabSize; v++) {
        const lc = logitsPair[strideCond + base + v];
        const lu = logitsPair[strideUncond + base + v];
        blended[base + v] = lu + cfgScale * (lc - lu);
      }
    }

    const proposed = new Int32Array(seqLen);
    const confidences: Array<{ pos: number; conf: number }> = [];

    for (let p = 0; p < seqLen; p++) {
      if (curTokens[p] !== maskTok) {
        proposed[p] = curTokens[p];
        continue;
      }
      const { token, prob } = sampleFromTopKLogits(
        blended,
        p * vocabSize,
        vocabSize,
        settings.temperature,
        settings.topK,
        rng,
      );
      proposed[p] = token;
      confidences.push({ pos: p, conf: prob });
    }

    if (confidences.length === 0) break;

    if (s === steps - 1) {
      for (const item of confidences) {
        curTokens[item.pos] = proposed[item.pos];
      }
      break;
    }

    confidences.sort((a, b) => b.conf - a.conf);
    const targetUnmaskedTotal = Math.round(((s + 1) / steps) * seqLen);
    const alreadyUnmasked = seqLen - confidences.length;
    const revealCount = Math.max(1, Math.min(confidences.length, targetUnmaskedTotal - alreadyUnmasked));

    for (let i = 0; i < revealCount; i++) {
      const pos = confidences[i].pos;
      curTokens[pos] = proposed[pos];
    }
  }

  const outGrid = new Uint8Array(seqLen);
  for (let p = 0; p < seqLen; p++) {
    outGrid[p] = Math.max(0, Math.min(vocabSize - 1, curTokens[p]));
  }
  return outGrid;
}

/**
 * Full Phase 7(D) T3 pipeline:
 *  - Runs generator sampling via ONNX runner (Worker WebGPU/WASM or custom runner)
 *  - Automatically resamples constraint failures (binary alpha, outline closure, connected silhouette)
 *  - Runs optional ONNX reranker discriminator
 *  - Evaluates Critic v1 + Mandatory Memorization Check (nearest real sprite originality & diff ratio)
 *  - Ranks survivors by Critic + Reranker - Memorization Penalty
 */
export async function runT3SamplingAndRanking(params: {
  runner: T3InferenceRunner;
  modelCard: T3ModelCard;
  category: ItemCategory;
  prompt: string;
  calibration: CriticCalibration | null;
  spriteIndex: SpriteIndexState | null;
  candidateCount?: number;
  seed?: number;
}): Promise<T3SamplingBatchResult> {
  const {
    runner,
    modelCard,
    category,
    prompt,
    calibration,
    spriteIndex,
    candidateCount = 4,
    seed = 772077,
  } = params;

  const t0 = performance.now();
  const isTile = false;
  const maxAttempts = Math.max(1, modelCard.samplerSettings.maxResampleAttempts || 3);
  const { categoryIdx, tagVec, tagDim } = encodeCategoryAndTags(modelCard, category, prompt);

  let totalSamplingRounds = 0;
  let constraintFailuresResampled = 0;
  const candidatesRaw: Array<{
    candidateIndex: number;
    tokenGrid: Uint8Array;
    rgba: Uint8ClampedArray;
    constraints: T3ConstraintCheckResult;
    resampleAttemptsUsed: number;
  }> = [];

  for (let c = 0; c < candidateCount; c++) {
    let bestAttempt: {
      tokenGrid: Uint8Array;
      rgba: Uint8ClampedArray;
      constraints: T3ConstraintCheckResult;
      attempts: number;
    } | null = null;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      totalSamplingRounds++;
      const attemptSeed = seed + c * 997 + (attempt - 1) * 31337;
      const tokenGrid = await sampleSingleT3Grid({
        runner,
        modelCard,
        categoryIdx,
        tagVec,
        tagDim,
        seed: attemptSeed,
      });
      const rgba = decodeVocabGridToRgba(tokenGrid, modelCard.vocabulary);
      const constraints = checkT3HardConstraints(rgba, isTile);

      bestAttempt = { tokenGrid, rgba, constraints, attempts: attempt };
      if (constraints.passed) {
        break;
      }
      if (attempt < maxAttempts) {
        constraintFailuresResampled++;
      }
    }

    if (bestAttempt) {
      candidatesRaw.push({
        candidateIndex: c,
        tokenGrid: bestAttempt.tokenGrid,
        rgba: bestAttempt.rgba,
        constraints: bestAttempt.constraints,
        resampleAttemptsUsed: bestAttempt.attempts,
      });
    }
  }

  // Optional ONNX Reranker batch inference
  let rerankerProbs: number[] | null = null;
  if (runner.runReranker && candidatesRaw.length > 0) {
    const seqLen = modelCard.samplerSettings.tokenCount || 1024;
    const flatTokens = new BigInt64Array(candidatesRaw.length * seqLen);
    const flatCats = new BigInt64Array(candidatesRaw.length);
    for (let i = 0; i < candidatesRaw.length; i++) {
      flatCats[i] = BigInt(categoryIdx);
      for (let p = 0; p < seqLen; p++) {
        flatTokens[i * seqLen + p] = BigInt(candidatesRaw[i].tokenGrid[p]);
      }
    }
    const rawLogits = await runner.runReranker({
      tokens: flatTokens,
      categoryIds: flatCats,
      batchSize: candidatesRaw.length,
      seqLen,
    });
    if (rawLogits) {
      rerankerProbs = Array.from(rawLogits).map((l) => 1 / (1 + Math.exp(-l)));
    }
  }

  const ranked: T3SamplingCandidate[] = candidatesRaw.map((raw, idx) => {
    const evaluation = evaluateSprite(raw.rgba, {
      category,
      isTileOverride: isTile,
      calibration,
    });

    const nearestReal =
      spriteIndex && spriteIndex.entries.length > 0
        ? findNearestRealSprite(raw.rgba, spriteIndex)
        : null;

    // Mandatory Memorization Check: flag near-verbatim training copies
    const memorizedCopyWarning = Boolean(
      nearestReal &&
        (nearestReal.originalityScore < 10 || nearestReal.pixelDiffRatio < 0.04),
    );

    const rerankerProb = rerankerProbs ? Math.round(rerankerProbs[idx] * 1000) / 1000 : null;
    const rerankerBoost = rerankerProb !== null ? (rerankerProb - 0.5) * 14 : 0;
    const constraintPenalty = raw.constraints.passed ? 0 : -22;
    const memPenalty = memorizedCopyWarning ? -28 : 0;
    const combinedRankScore =
      Math.round((evaluation.score + rerankerBoost + constraintPenalty + memPenalty) * 10) / 10;

    return {
      candidateIndex: raw.candidateIndex,
      tokenGrid: raw.tokenGrid,
      rgba: raw.rgba,
      constraints: raw.constraints,
      resampleAttemptsUsed: raw.resampleAttemptsUsed,
      rerankerProbability: rerankerProb,
      evaluation,
      nearestReal,
      memorizedCopyWarning,
      combinedRankScore,
    };
  });

  ranked.sort((a, b) => b.combinedRankScore - a.combinedRankScore);

  return {
    executionProvider: runner.executionProvider,
    requestedCount: candidateCount,
    totalSamplingRounds,
    constraintFailuresResampled,
    candidates: ranked,
    samplingMs: Math.round(performance.now() - t0),
  };
}

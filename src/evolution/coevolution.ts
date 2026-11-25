import type { CriticCalibration } from '../critic/types';
import {
  cloneNeatGenome,
  createNeatPopulation,
  evolveNeatGeneration,
  getNeatComplexitySummary,
} from './neat';
import { NEAT_PIXEL_INPUT_COUNT, NEAT_PIXEL_OUTPUT_COUNT } from './pixelOps';
import {
  cloneRecipeGenome,
  createRecipeGaPopulation,
  evolveRecipeGaGeneration,
  formatRecipeSummaryPt,
} from './recipeGa';
import { evaluateEditorPairOnBatch } from './restorationTask';
import type {
  CoevolutionChampion,
  CoevolutionRunConfig,
  EvaluationAggregate,
  EvolutionCheckpoint,
  EvolutionGenerationPoint,
  NeatGenome,
  PromotionLogEntry,
  PromotionRulesConfig,
  RecipeGenome,
  RestorationSample,
  RestorationTripletPreview,
} from './types';

export function createDefaultPromotionRules(
  overrides: Partial<PromotionRulesConfig> = {},
): PromotionRulesConfig {
  return {
    minValidationImprovement: 0.2,
    minInitialGainOverCorrupted: 1.0,
    maxOverfitGap: 18.0,
    minSilhouetteIoU: 0.72,
    maxNewHardViolations: 0,
    minMarginAboveJunkFloor: 5.0,
    ...overrides,
  };
}

export interface PromotionDecision {
  promoted: boolean;
  reasonPt: string;
}

/**
 * Strict, honest promotion gate evaluated solely on held-out `validation` metrics.
 * Prevents overfitting, shape destruction (low IoU), hard-rule regressions, and junk promotion.
 */
export function evaluatePromotionRule(params: {
  trainStats: EvaluationAggregate;
  valStats: EvaluationAggregate;
  currentChampion: CoevolutionChampion | null;
  valCorruptedMedian: number;
  junkFloorMedian: number;
  rules: PromotionRulesConfig;
}): PromotionDecision {
  const {
    trainStats,
    valStats,
    currentChampion,
    valCorruptedMedian,
    junkFloorMedian,
    rules,
  } = params;

  if (valStats.meanSilhouetteIoU < rules.minSilhouetteIoU) {
    return {
      promoted: false,
      reasonPt: `Rejeitado: IoU de silhueta na validação (${(valStats.meanSilhouetteIoU * 100).toFixed(1)}%) abaixo do mínimo ${(rules.minSilhouetteIoU * 100).toFixed(0)}%.`,
    };
  }

  if (valStats.hardViolationDelta > rules.maxNewHardViolations) {
    return {
      promoted: false,
      reasonPt: `Rejeitado: introduziu +${valStats.hardViolationDelta} nova(s) violação(ões) de regra rígida na validação.`,
    };
  }

  if (valStats.meanRestoredScore <= junkFloorMedian + rules.minMarginAboveJunkFloor) {
    return {
      promoted: false,
      reasonPt: `Rejeitado: validação (${valStats.meanRestoredScore.toFixed(1)}) não supera o piso lixo (${junkFloorMedian.toFixed(1)}) com margem mínima de +${rules.minMarginAboveJunkFloor.toFixed(1)}.`,
    };
  }

  const overfitGap = trainStats.meanRestoredScore - valStats.meanRestoredScore;
  if (overfitGap > rules.maxOverfitGap) {
    return {
      promoted: false,
      reasonPt: `Rejeitado: sinal de overfitting (treino ${trainStats.meanRestoredScore.toFixed(1)} vs validação ${valStats.meanRestoredScore.toFixed(1)}, gap +${overfitGap.toFixed(1)} > ${rules.maxOverfitGap.toFixed(1)}).`,
    };
  }

  if (valStats.meanDelta <= 0) {
    return {
      promoted: false,
      reasonPt: `Rejeitado: sem ganho líquido sobre os sprites corrompidos na validação (Δ=${valStats.meanDelta.toFixed(1)}).`,
    };
  }

  if (!currentChampion) {
    const initialGain = valStats.meanRestoredScore - valStats.meanCorruptedScore;
    if (initialGain < rules.minInitialGainOverCorrupted) {
      return {
        promoted: false,
        reasonPt: `Rejeitado: ganho inicial na validação (+${initialGain.toFixed(1)}) abaixo do limiar +${rules.minInitialGainOverCorrupted.toFixed(1)}.`,
      };
    }
    return {
      promoted: true,
      reasonPt: `Promovido como 1º campeão: validação ${valStats.meanRestoredScore.toFixed(1)} (+${initialGain.toFixed(1)} sobre baseline corrompido ${valStats.meanCorruptedScore.toFixed(1)}, IoU ${(valStats.meanSilhouetteIoU * 100).toFixed(1)}%).`,
    };
  }

  const prevVal = currentChampion.valStats.meanRestoredScore;
  const improvement = valStats.meanRestoredScore - prevVal;

  if (improvement >= rules.minValidationImprovement) {
    return {
      promoted: true,
      reasonPt: `Promovido: superou campeão anterior na validação (${prevVal.toFixed(1)} → ${valStats.meanRestoredScore.toFixed(1)}, +${improvement.toFixed(1)} pts, IoU ${(valStats.meanSilhouetteIoU * 100).toFixed(1)}%).`,
    };
  }

  return {
    promoted: false,
    reasonPt: `Rejeitado: validação ${valStats.meanRestoredScore.toFixed(1)} não superou campeão atual (${prevVal.toFixed(1)}) em +${rules.minValidationImprovement.toFixed(1)} pts.`,
  };
}

export function createInitialCheckpoint(params: {
  datasetKey: string;
  config: CoevolutionRunConfig;
  valBaselineCorruptedMedian: number;
  realCeilingMedian: number;
  junkFloorMedian: number;
  trainSampleCount: number;
  valSampleCount: number;
  nowIso?: string;
}): EvolutionCheckpoint {
  const {
    datasetKey,
    config,
    valBaselineCorruptedMedian,
    realCeilingMedian,
    junkFloorMedian,
    trainSampleCount,
    valSampleCount,
  } = params;

  const neatPopSize = Math.max(6, Math.round(config.populationSize * 0.75));
  const gaPopSize = Math.max(6, config.populationSize);

  const neatState = createNeatPopulation(
    {
      populationSize: neatPopSize,
      inputCount: NEAT_PIXEL_INPUT_COUNT,
      outputCount: NEAT_PIXEL_OUTPUT_COUNT,
      compatibilityThreshold: 1.35,
      targetSpeciesCount: Math.max(2, Math.min(4, Math.floor(neatPopSize / 3))),
    },
    config.seed,
  );

  const gaState = createRecipeGaPopulation(
    {
      populationSize: gaPopSize,
    },
    config.seed,
  );

  return {
    id: `${datasetKey}:${config.category}`,
    datasetKey,
    category: config.category,
    config,
    generation: 0,
    status: 'idle',
    neatState,
    gaState,
    champion: null,
    history: [],
    promotionLog: [],
    valBaselineCorruptedMedian,
    realCeilingMedian,
    junkFloorMedian,
    trainSampleCount,
    valSampleCount,
    updatedAt: params.nowIso ?? new Date().toISOString(),
    readBackVerified: false,
  };
}

export interface CoevolutionStepResult {
  checkpoint: EvolutionCheckpoint;
  point: EvolutionGenerationPoint;
  promotionEntry: PromotionLogEntry;
  validationPreviews: RestorationTripletPreview[];
}

/**
 * Runs exactly ONE cooperative co-evolution generation cleanly and deterministically:
 *  1) Evaluates Recipe GA population on `trainSamples` paired with current best NEAT partner.
 *  2) Evaluates NEAT population on `trainSamples` paired with current best Recipe partner.
 *  3) Evaluates generation's best (Recipe, NEAT) candidate on held-out `valSamples`.
 *  4) Applies strict promotion rules and logs promoted/rejected decision.
 *  5) Advances both populations (elitism + speciation + crossover + mutation) for the next generation.
 */
export function runCoevolutionGenerationStep(params: {
  checkpoint: EvolutionCheckpoint;
  trainSamples: RestorationSample[];
  valSamples: RestorationSample[];
  calibration: CriticCalibration;
  nowIso?: string;
}): CoevolutionStepResult {
  const startMs = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const { checkpoint, trainSamples, valSamples, calibration } = params;
  const nowIso = params.nowIso ?? new Date().toISOString();

  const nextGenNum = checkpoint.generation + 1;
  const { neatState, gaState, config } = checkpoint;

  // Current cooperative partner from champion or initial genome
  const neatPartner: NeatGenome | null =
    checkpoint.champion?.neat ?? neatState.bestGenome ?? neatState.genomes[0] ?? null;

  // --- Step A: Evaluate Recipe GA genomes strictly on `trainSamples` ---
  let bestGaGenome: RecipeGenome = gaState.genomes[0];
  let bestGaTrainStats: EvaluationAggregate | null = null;
  let trainScoreSum = 0;

  for (const recipe of gaState.genomes) {
    const { aggregate } = evaluateEditorPairOnBatch({
      samples: trainSamples,
      recipe,
      neat: neatPartner,
      calibration,
      parsimonyPenaltyPerStep: gaState.config.parsimonyPenaltyPerStep,
    });
    recipe.fitness = aggregate.combinedFitness;
    trainScoreSum += aggregate.meanRestoredScore;

    if (!bestGaTrainStats || recipe.fitness > bestGaGenome.fitness) {
      bestGaGenome = recipe;
      bestGaTrainStats = aggregate;
    }
  }

  const trainMeanScore =
    gaState.genomes.length > 0 ? Math.round((trainScoreSum / gaState.genomes.length) * 10) / 10 : 0;

  // --- Step B: Evaluate NEAT genomes strictly on `trainSamples` paired with bestGaGenome ---
  let bestNeatGenome: NeatGenome = neatState.genomes[0];
  let bestPairTrainStats: EvaluationAggregate = bestGaTrainStats!;

  for (const neat of neatState.genomes) {
    const { aggregate } = evaluateEditorPairOnBatch({
      samples: trainSamples,
      recipe: bestGaGenome,
      neat,
      calibration,
      parsimonyPenaltyPerStep: gaState.config.parsimonyPenaltyPerStep,
    });
    neat.fitness = aggregate.combinedFitness;
    if (neat.fitness >= bestNeatGenome.fitness) {
      bestNeatGenome = neat;
      bestPairTrainStats = aggregate;
    }
  }

  // --- Step C: Held-out validation evaluation (zero leakage from train) ---
  const { aggregate: valStats, previews: candidatePreviews } = evaluateEditorPairOnBatch({
    samples: valSamples,
    recipe: bestGaGenome,
    neat: bestNeatGenome,
    calibration,
    parsimonyPenaltyPerStep: gaState.config.parsimonyPenaltyPerStep,
    collectPreviews: true,
  });

  // --- Step D: Strict Promotion Rules ---
  const prevChampionVal = checkpoint.champion?.valStats.meanRestoredScore ?? null;
  const decision = evaluatePromotionRule({
    trainStats: bestPairTrainStats,
    valStats,
    currentChampion: checkpoint.champion,
    valCorruptedMedian: valStats.meanCorruptedScore,
    junkFloorMedian: checkpoint.junkFloorMedian,
    rules: config.promotionRules,
  });

  let nextChampion = checkpoint.champion;
  let activePreviews = candidatePreviews;

  if (decision.promoted) {
    nextChampion = {
      generation: nextGenNum,
      promotedAt: nowIso,
      recipe: cloneRecipeGenome(bestGaGenome),
      neat: cloneNeatGenome(bestNeatGenome),
      trainStats: bestPairTrainStats,
      valStats,
    };
  } else if (checkpoint.champion) {
    // Keep visual previews showing the promoted champion on validation if candidate was rejected
    const champEval = evaluateEditorPairOnBatch({
      samples: valSamples,
      recipe: checkpoint.champion.recipe,
      neat: checkpoint.champion.neat,
      calibration,
      collectPreviews: true,
    });
    activePreviews = champEval.previews;
  }

  const neatComp = getNeatComplexitySummary(bestNeatGenome);
  const activeRecipeSteps = bestGaGenome.steps.filter((s) => s.enabled).length;
  const elapsedMs = Math.max(
    1,
    Math.round((typeof performance !== 'undefined' ? performance.now() : Date.now()) - startMs),
  );

  const promotionEntry: PromotionLogEntry = {
    id: `promo:${checkpoint.id}:gen-${nextGenNum}`,
    generation: nextGenNum,
    timestamp: nowIso,
    decision: decision.promoted ? 'promoted' : 'rejected',
    reasonPt: decision.reasonPt,
    trainScore: bestPairTrainStats.meanRestoredScore,
    valScore: valStats.meanRestoredScore,
    prevChampionValScore: prevChampionVal,
    valCorruptedScore: valStats.meanCorruptedScore,
    valDeltaFromCorrupted: valStats.meanDelta,
    silhouetteIoU: valStats.meanSilhouetteIoU,
    hardViolationDelta: valStats.hardViolationDelta,
    recipeSummary: formatRecipeSummaryPt(bestGaGenome),
    neatSummary: `${neatComp.totalNodes} nós (${neatComp.hiddenNodes} ocultos), ${neatComp.enabledConnections} conexões`,
  };

  const point: EvolutionGenerationPoint = {
    generation: nextGenNum,
    trainBestScore: bestPairTrainStats.meanRestoredScore,
    trainMeanScore,
    valBestScore: valStats.meanRestoredScore,
    valMedianScore: valStats.medianRestoredScore,
    valCorruptedMedian: valStats.meanCorruptedScore,
    valDelta: valStats.meanDelta,
    valSilhouetteIoU: valStats.meanSilhouetteIoU,
    neatSpeciesCount: neatState.species.length,
    neatNodes: neatComp.totalNodes,
    neatEnabledConnections: neatComp.enabledConnections,
    recipeActiveSteps: activeRecipeSteps,
    promoted: decision.promoted,
    elapsedMs,
  };

  // --- Step E: Evolve both populations for the next generation ---
  evolveRecipeGaGeneration(gaState);
  evolveNeatGeneration(neatState);

  const updatedCheckpoint: EvolutionCheckpoint = {
    ...checkpoint,
    generation: nextGenNum,
    status: nextGenNum >= config.targetGenerations ? 'completed' : 'running',
    neatState,
    gaState,
    champion: nextChampion,
    history: [...checkpoint.history, point],
    promotionLog: [promotionEntry, ...checkpoint.promotionLog],
    updatedAt: nowIso,
    readBackVerified: false,
  };

  return {
    checkpoint: updatedCheckpoint,
    point,
    promotionEntry,
    validationPreviews: activePreviews,
  };
}

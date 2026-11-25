import type { CriticCalibration } from '../critic/types';
import type { CoevolutionChampion } from '../evolution/types';
import { BENCHMARK_10_PROMPTS, generateSpriteCandidates } from '../generation/pipeline';
import type {
  ApprovalLabelRecord,
  GeneratedSpriteRecord,
  SpriteIndexState,
} from '../generation/types';
import type { LessonRecord } from '../lessons/types';
import type { ItemCategory } from '../types/studio';
import type { AgentProfileRecord, FieldTestPromptOutcome, FieldTestReport } from './types';

function round1(v: number): number {
  return Math.round(v * 10) / 10;
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Field Test (Section C.6):
 * Runs a paired, deterministic multi-prompt field evaluation comparing a candidate agent
 * against a baseline agent on the exact same fixed prompts and seeds (train donors only; test never touched).
 * Measures Critic v1 score Δ, Originality %, binary alpha compliance, hard-violation regressions,
 * win/tie/loss counts, and emits an honest empirical verdict.
 */
export async function runFieldTestC6(params: {
  datasetKey: string;
  baselineAgent: AgentProfileRecord;
  candidateAgent: AgentProfileRecord;
  index: SpriteIndexState;
  calibration: CriticCalibration;
  lessons?: LessonRecord[];
  approvals?: ApprovalLabelRecord[];
  approvedSprites?: GeneratedSpriteRecord[];
  championsByCategory?: Partial<Record<ItemCategory, CoevolutionChampion>>;
  promptCount?: number;
  seed?: number;
}): Promise<FieldTestReport> {
  const {
    datasetKey,
    baselineAgent,
    candidateAgent,
    index,
    calibration,
    lessons = [],
    approvals = [],
    approvedSprites = [],
    championsByCategory = {},
    promptCount = 6,
    seed = 20260606,
  } = params;

  // Select balanced prompts across available indexed categories
  const availableCategories = new Set(
    index.entries.filter((e) => e.split === 'train').map((e) => e.category),
  );
  const eligiblePrompts = BENCHMARK_10_PROMPTS.filter((p) =>
    availableCategories.has(p.category),
  );
  const selected = (eligiblePrompts.length > 0 ? eligiblePrompts : BENCHMARK_10_PROMPTS).slice(
    0,
    Math.max(1, Math.min(promptCount, 10)),
  );

  const outcomes: FieldTestPromptOutcome[] = [];

  for (let i = 0; i < selected.length; i++) {
    const spec = selected[i];
    const promptSeed = (seed + i * 1997) >>> 0;

    const baseChamp =
      baselineAgent.repairPolicy === 'champion_coevolution'
        ? championsByCategory[spec.category] ?? null
        : null;
    const candChamp =
      candidateAgent.repairPolicy === 'champion_coevolution'
        ? championsByCategory[spec.category] ?? null
        : null;

    const [baseRes, candRes] = await Promise.all([
      generateSpriteCandidates({
        datasetKey,
        prompt: spec.prompt,
        category: spec.category,
        tier: baselineAgent.tier,
        useGemini: false,
        maxDiffPixels: baselineAgent.maxDiffPixels,
        enableRepair: baselineAgent.repairPolicy !== 'none',
        candidateCount: 6,
        topKReturn: 1,
        seed: promptSeed,
        index,
        calibration,
        lessons,
        approvals,
        approvedSprites,
        champion: baseChamp,
      }),
      generateSpriteCandidates({
        datasetKey,
        prompt: spec.prompt,
        category: spec.category,
        tier: candidateAgent.tier,
        useGemini: false,
        maxDiffPixels: candidateAgent.maxDiffPixels,
        enableRepair: candidateAgent.repairPolicy !== 'none',
        candidateCount: 6,
        topKReturn: 1,
        seed: promptSeed,
        index,
        calibration,
        lessons,
        approvals,
        approvedSprites,
        champion: candChamp,
      }),
    ]);

    const baselineRecord = baseRes.candidates[0];
    const candidateRecord = candRes.candidates[0];
    if (!baselineRecord || !candidateRecord) continue;

    const scoreDelta = round1(candidateRecord.evaluation.score - baselineRecord.evaluation.score);
    const originalityDelta = round1(
      candidateRecord.nearestReal.originalityScore - baselineRecord.nearestReal.originalityScore,
    );

    const baseHard = baselineRecord.evaluation.issues.filter((x) => x.severity === 'hard').length;
    const candHard = candidateRecord.evaluation.issues.filter((x) => x.severity === 'hard').length;

    const winner: 'candidate' | 'baseline' | 'tie' =
      scoreDelta >= 0.4 ? 'candidate' : scoreDelta <= -0.4 ? 'baseline' : 'tie';

    outcomes.push({
      promptId: spec.id,
      prompt: spec.prompt,
      category: spec.category,
      baselineRecord,
      candidateRecord,
      scoreDelta,
      originalityDelta,
      winner,
      candidateBinaryAlphaOk: candidateRecord.evaluation.features.semiTransparentPixels === 0,
      candidateNoNewHardViolations: candHard <= baseHard,
    });
  }

  const n = Math.max(1, outcomes.length);
  const candidateMeanScore = round1(
    outcomes.reduce((s, o) => s + o.candidateRecord.evaluation.score, 0) / n,
  );
  const baselineMeanScore = round1(
    outcomes.reduce((s, o) => s + o.baselineRecord.evaluation.score, 0) / n,
  );
  const deltas = outcomes.map((o) => o.scoreDelta);
  const meanScoreDelta = round1(deltas.reduce((s, d) => s + d, 0) / n);
  const medianScoreDelta = round1(median(deltas));
  const candidateMeanOriginality = round1(
    outcomes.reduce((s, o) => s + o.candidateRecord.nearestReal.originalityScore, 0) / n,
  );
  const baselineMeanOriginality = round1(
    outcomes.reduce((s, o) => s + o.baselineRecord.nearestReal.originalityScore, 0) / n,
  );
  const candidateWins = outcomes.filter((o) => o.winner === 'candidate').length;
  const baselineWins = outcomes.filter((o) => o.winner === 'baseline').length;
  const ties = outcomes.filter((o) => o.winner === 'tie').length;
  const binaryAlphaPassRate =
    outcomes.filter((o) => o.candidateBinaryAlphaOk).length / n;
  const hardViolationFreeRate =
    outcomes.filter((o) => o.candidateNoNewHardViolations).length / n;

  let verdict: FieldTestReport['verdict'];
  let verdictRationalePt: string;

  if (binaryAlphaPassRate < 1 || hardViolationFreeRate < 1 || meanScoreDelta <= -0.8) {
    verdict = 'REGRESSÃO DETECTADA';
    verdictRationalePt = `Candidato regrediu no teste de campo (Δ médio ${meanScoreDelta >= 0 ? '+' : ''}${meanScoreDelta.toFixed(1)} pts, ${baselineWins} derrotas para o baseline). Não recomendado ativar.`;
  } else if (meanScoreDelta >= 0.5 && candidateWins > baselineWins) {
    verdict = 'APROVADO EM CAMPO';
    verdictRationalePt = `Candidato superou o baseline em ${candidateWins}/${outcomes.length} prompts (Δ médio +${meanScoreDelta.toFixed(1)} pts, 100% alpha binário, 0 regressões de regra dura).`;
  } else {
    verdict = 'EMPATE TÉCNICO';
    verdictRationalePt = `Desempenho equivalente ao baseline (Δ médio ${meanScoreDelta >= 0 ? '+' : ''}${meanScoreDelta.toFixed(1)} pts; ${candidateWins}V / ${ties}E / ${baselineWins}D). Considere comparar por originalidade (${candidateMeanOriginality.toFixed(1)}% vs ${baselineMeanOriginality.toFixed(1)}%) ou duelos Elo cegos.`;
  }

  return {
    runAt: new Date().toISOString(),
    datasetKey,
    baselineAgentId: baselineAgent.id,
    baselineAgentName: baselineAgent.name,
    candidateAgentId: candidateAgent.id,
    candidateAgentName: candidateAgent.name,
    promptCount: outcomes.length,
    candidateMeanScore,
    baselineMeanScore,
    meanScoreDelta,
    medianScoreDelta,
    candidateMeanOriginality,
    baselineMeanOriginality,
    candidateWins,
    baselineWins,
    ties,
    binaryAlphaPassRate,
    hardViolationFreeRate,
    verdict,
    verdictRationalePt,
    outcomes,
  };
}

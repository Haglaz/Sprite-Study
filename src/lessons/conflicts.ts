import {
  computeWilsonConfidence,
  SOURCE_AUTHORITY_RANK,
  type ConflictResolutionResult,
  type ConflictResolutionStrategy,
  type LessonConflict,
  type LessonConflictReason,
  type LessonRecord,
  type LessonTargetRange,
} from './types';

function rangesAreDisjoint(a?: LessonTargetRange, b?: LessonTargetRange): boolean {
  if (!a || !b) return false;
  if (a.max !== undefined && b.min !== undefined && a.max < b.min) return true;
  if (b.max !== undefined && a.min !== undefined && b.max < a.min) return true;
  return false;
}

function normalizeDirectiveText(text: string): string {
  return text.trim().toLowerCase().replace(/\s+/g, ' ');
}

function classifyPairConflict(a: LessonRecord, b: LessonRecord): {
  reason: LessonConflictReason;
  reasonLabelPt: string;
} | null {
  // Conflicts occur between two active/conflicted lessons targeting the exact same (category, scopeKey).
  // A category-specific lesson specializing a 'Global' lesson is handled via specificity ordering, not an error.
  if (a.category !== b.category || a.scopeKey !== b.scopeKey) {
    return null;
  }

  if (a.polarity !== b.polarity) {
    return {
      reason: 'opposite_polarity',
      reasonLabelPt: 'Polaridade oposta (uma regra exige e a outra proíbe o mesmo domínio)',
    };
  }

  if (rangesAreDisjoint(a.targetRange, b.targetRange)) {
    return {
      reason: 'disjoint_range',
      reasonLabelPt: 'Faixas numéricas incompatíveis (sem interseção entre os intervalos alvo)',
    };
  }

  if (normalizeDirectiveText(a.directive) !== normalizeDirectiveText(b.directive)) {
    return {
      reason: 'competing_directive',
      reasonLabelPt: 'Diretrizes concorrentes ativas para a mesma categoria e domínio',
    };
  }

  return null;
}

/**
 * Detects all active conflicts across a set of lessons deterministically.
 */
export function detectLessonConflicts(lessons: readonly LessonRecord[]): LessonConflict[] {
  const candidates = lessons
    .filter((l) => l.status === 'active' || l.status === 'conflicted')
    .slice()
    .sort((a, b) => a.id.localeCompare(b.id));

  const conflicts: LessonConflict[] = [];

  for (let i = 0; i < candidates.length; i++) {
    for (let j = i + 1; j < candidates.length; j++) {
      const a = candidates[i];
      const b = candidates[j];
      const hit = classifyPairConflict(a, b);
      if (hit) {
        conflicts.push({
          id: `conflict:${a.id}:${b.id}`,
          category: a.category,
          scopeKey: a.scopeKey,
          reason: hit.reason,
          reasonLabelPt: hit.reasonLabelPt,
          lessonAId: a.id,
          lessonBId: b.id,
          lessonA: a,
          lessonB: b,
        });
      }
    }
  }

  return conflicts;
}

/**
 * Marks lessons participating in any active conflict with `status = 'conflicted'` and
 * clears stale `'conflicted'` flags on lessons that no longer clash.
 */
export function annotateLessonConflictStates(lessons: readonly LessonRecord[]): {
  lessons: LessonRecord[];
  conflicts: LessonConflict[];
} {
  const conflicts = detectLessonConflicts(lessons);
  const groupByLessonId = new Map<string, string>();

  for (const c of conflicts) {
    if (!groupByLessonId.has(c.lessonAId)) groupByLessonId.set(c.lessonAId, c.id);
    if (!groupByLessonId.has(c.lessonBId)) groupByLessonId.set(c.lessonBId, c.id);
  }

  const annotated = lessons.map((l) => {
    if (l.status === 'superseded' || l.status === 'archived') {
      return l;
    }
    const groupId = groupByLessonId.get(l.id) ?? null;
    if (groupId) {
      return { ...l, status: 'conflicted' as const, conflictGroupId: groupId };
    }
    return { ...l, status: 'active' as const, conflictGroupId: null };
  });

  return { lessons: annotated, conflicts };
}

/**
 * Compares two lessons according to deterministic conflict-resolution precedence:
 * 1. Source authority: human_feedback > critic_loop > mined
 * 2. Empirical net evidence: (positiveCount - negativeCount) then confidence
 * 3. Observed Critic v1 delta (when measured on both)
 * 4. Sample size (n)
 * 5. Newer updatedAt timestamp
 * 6. Deterministic lexical ID tie-breaker
 */
export function compareConflictPrecedence(a: LessonRecord, b: LessonRecord): {
  winner: LessonRecord;
  loser: LessonRecord;
  rationalePt: string;
} {
  const authA = SOURCE_AUTHORITY_RANK[a.source];
  const authB = SOURCE_AUTHORITY_RANK[b.source];
  if (authA !== authB) {
    const winner = authA > authB ? a : b;
    const loser = authA > authB ? b : a;
    return {
      winner,
      loser,
      rationalePt: `Precedência de fonte: "${winner.source}" prevalece sobre "${loser.source}".`,
    };
  }

  const netA = a.evidence.positiveCount - a.evidence.negativeCount;
  const netB = b.evidence.positiveCount - b.evidence.negativeCount;
  if (netA !== netB) {
    const winner = netA > netB ? a : b;
    const loser = netA > netB ? b : a;
    return {
      winner,
      loser,
      rationalePt: `Maior saldo de evidência empírica (${Math.max(netA, netB)} vs ${Math.min(netA, netB)}).`,
    };
  }

  if (Math.abs(a.evidence.confidence - b.evidence.confidence) > 1e-6) {
    const winner = a.evidence.confidence > b.evidence.confidence ? a : b;
    const loser = a.evidence.confidence > b.evidence.confidence ? b : a;
    return {
      winner,
      loser,
      rationalePt: `Maior confiança Wilson (${(winner.evidence.confidence * 100).toFixed(1)}% vs ${(loser.evidence.confidence * 100).toFixed(1)}%).`,
    };
  }

  const deltaA = a.evidence.criticDeltaMean ?? 0;
  const deltaB = b.evidence.criticDeltaMean ?? 0;
  if (Math.abs(deltaA - deltaB) > 1e-6) {
    const winner = deltaA > deltaB ? a : b;
    const loser = deltaA > deltaB ? b : a;
    return {
      winner,
      loser,
      rationalePt: `Melhor delta médio medido no Crítico v1 (${deltaA.toFixed(1)} vs ${deltaB.toFixed(1)}).`,
    };
  }

  if (a.evidence.sampleSize !== b.evidence.sampleSize) {
    const winner = a.evidence.sampleSize > b.evidence.sampleSize ? a : b;
    const loser = a.evidence.sampleSize > b.evidence.sampleSize ? b : a;
    return {
      winner,
      loser,
      rationalePt: `Maior tamanho amostral (n=${winner.evidence.sampleSize} vs n=${loser.evidence.sampleSize}).`,
    };
  }

  const timeA = Date.parse(a.updatedAt) || 0;
  const timeB = Date.parse(b.updatedAt) || 0;
  if (timeA !== timeB) {
    const winner = timeA > timeB ? a : b;
    const loser = timeA > timeB ? b : a;
    return {
      winner,
      loser,
      rationalePt: 'Regra mais recente prevalece em empate de fonte e evidência.',
    };
  }

  const winner = a.id.localeCompare(b.id) <= 0 ? a : b;
  const loser = winner === a ? b : a;
  return {
    winner,
    loser,
    rationalePt: 'Desempate determinístico por identificador.',
  };
}

/**
 * Resolves a detected conflict between two lessons, returning updated winner & loser records.
 */
export function resolveLessonConflictPair(params: {
  conflict: LessonConflict;
  strategy: ConflictResolutionStrategy;
  preferredWinnerId?: string;
  nowIso?: string;
}): ConflictResolutionResult {
  const nowIso = params.nowIso ?? new Date().toISOString();
  const { lessonA, lessonB } = params.conflict;

  let winner: LessonRecord;
  let loser: LessonRecord;
  let rationalePt: string;

  if (params.strategy === 'keep_lesson' && params.preferredWinnerId) {
    if (params.preferredWinnerId === lessonA.id) {
      winner = lessonA;
      loser = lessonB;
    } else if (params.preferredWinnerId === lessonB.id) {
      winner = lessonB;
      loser = lessonA;
    } else {
      throw new Error(`preferredWinnerId "${params.preferredWinnerId}" não pertence ao conflito.`);
    }
    rationalePt = `Escolha manual do usuário mantendo "${winner.title}".`;
  } else {
    const cmp = compareConflictPrecedence(lessonA, lessonB);
    winner = cmp.winner;
    loser = cmp.loser;
    rationalePt = cmp.rationalePt;
  }

  let updatedWinner: LessonRecord = {
    ...winner,
    status: 'active',
    conflictGroupId: null,
    supersededById: null,
    updatedAt: nowIso,
  };

  if (
    params.strategy === 'merge_range' &&
    winner.polarity === loser.polarity &&
    winner.targetRange &&
    loser.targetRange
  ) {
    const mergedMin =
      winner.targetRange.min !== undefined && loser.targetRange.min !== undefined
        ? Math.min(winner.targetRange.min, loser.targetRange.min)
        : (winner.targetRange.min ?? loser.targetRange.min);
    const mergedMax =
      winner.targetRange.max !== undefined && loser.targetRange.max !== undefined
        ? Math.max(winner.targetRange.max, loser.targetRange.max)
        : (winner.targetRange.max ?? loser.targetRange.max);

    const pos = winner.evidence.positiveCount + loser.evidence.positiveCount;
    const neg = winner.evidence.negativeCount + loser.evidence.negativeCount;
    const mergedSprites = Array.from(
      new Set([...winner.evidence.referenceSpriteIds, ...loser.evidence.referenceSpriteIds]),
    ).slice(0, 12);

    updatedWinner = {
      ...updatedWinner,
      targetRange: {
        ...winner.targetRange,
        min: mergedMin,
        max: mergedMax,
      },
      evidence: {
        ...winner.evidence,
        sampleSize: winner.evidence.sampleSize + loser.evidence.sampleSize,
        positiveCount: pos,
        negativeCount: neg,
        referenceSpriteIds: mergedSprites,
        confidence: computeWilsonConfidence(pos, neg),
      },
    };
    rationalePt = `Faixas compatíveis mescladas em [${mergedMin ?? '—'}, ${mergedMax ?? '—'}] na regra vencedora.`;
  }

  const updatedLoser: LessonRecord = {
    ...loser,
    status: 'superseded',
    conflictGroupId: null,
    supersededById: updatedWinner.id,
    updatedAt: nowIso,
  };

  return {
    winner: updatedWinner,
    loser: updatedLoser,
    strategy: params.strategy,
    rationalePt,
    resolvedAt: nowIso,
  };
}

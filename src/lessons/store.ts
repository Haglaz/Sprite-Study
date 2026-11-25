import { putAndVerify, type StorageBackend } from '../db/persistence';
import type { CriticIssueCode } from '../critic/types';
import {
  annotateLessonConflictStates,
  resolveLessonConflictPair,
} from './conflicts';
import {
  computeWilsonConfidence,
  LESSON_SCOPE_OPTIONS,
  type ConflictResolutionResult,
  type ConflictResolutionStrategy,
  type LessonCategory,
  type LessonConflict,
  type LessonPolarity,
  type LessonRecord,
  type LessonScopeKey,
  type LessonSource,
  type LessonTargetRange,
} from './types';

function verifyLessonReadBack(readBack: LessonRecord, written: LessonRecord): boolean {
  return (
    readBack.id === written.id &&
    readBack.category === written.category &&
    readBack.scopeKey === written.scopeKey &&
    readBack.status === written.status &&
    readBack.directive === written.directive &&
    readBack.updatedAt === written.updatedAt
  );
}

/**
 * Loads all persisted lessons from IndexedDB and annotates current conflict states.
 */
export async function loadAllLessons(backend: StorageBackend): Promise<{
  lessons: LessonRecord[];
  conflicts: LessonConflict[];
}> {
  const raw = await backend.getAll<LessonRecord>('lessons');
  return annotateLessonConflictStates(raw);
}

/**
 * Persists a single lesson with mandatory read-back verification (Honesty Rule #4)
 * and re-annotates any conflict state across the database.
 */
export async function saveLessonVerified(
  backend: StorageBackend,
  lesson: LessonRecord,
): Promise<{
  saved: LessonRecord;
  lessons: LessonRecord[];
  conflicts: LessonConflict[];
}> {
  const candidate: LessonRecord = { ...lesson, readBackVerified: true };
  const verified = await putAndVerify(backend, 'lessons', candidate, verifyLessonReadBack);

  const all = await backend.getAll<LessonRecord>('lessons');
  const { lessons: annotated, conflicts } = annotateLessonConflictStates(all);

  // Persist any status/conflictGroupId changes triggered on other lessons
  for (const updated of annotated) {
    const prev = all.find((x) => x.id === updated.id);
    if (
      prev &&
      (prev.status !== updated.status || prev.conflictGroupId !== updated.conflictGroupId)
    ) {
      await putAndVerify(backend, 'lessons', { ...updated, readBackVerified: true }, verifyLessonReadBack);
    }
  }

  const finalAll = await backend.getAll<LessonRecord>('lessons');
  const finalState = annotateLessonConflictStates(finalAll);
  const finalSaved = finalState.lessons.find((l) => l.id === verified.readBack.id) ?? verified.readBack;

  return {
    saved: finalSaved,
    lessons: finalState.lessons,
    conflicts: finalState.conflicts,
  };
}

export interface CreateLessonInput {
  datasetKey: string | null;
  category: LessonCategory;
  scopeKey: LessonScopeKey;
  title: string;
  source: LessonSource;
  polarity: LessonPolarity;
  directive: string;
  validationCheck: string;
  relatedIssueCodes?: CriticIssueCode[];
  targetRange?: LessonTargetRange;
  referenceSpriteIds?: number[];
  criticDeltaMean?: number | null;
  nowIso?: string;
}

export async function createLessonVerified(
  backend: StorageBackend,
  input: CreateLessonInput,
): Promise<{
  saved: LessonRecord;
  lessons: LessonRecord[];
  conflicts: LessonConflict[];
}> {
  const nowIso = input.nowIso ?? new Date().toISOString();
  const title = input.title.trim();
  const directive = input.directive.trim();
  const validationCheck = input.validationCheck.trim();

  if (!title || !directive || !validationCheck) {
    throw new Error('Título, diretriz e verificação são obrigatórios para registrar uma lição.');
  }

  const defaultIssues =
    LESSON_SCOPE_OPTIONS.find((opt) => opt.key === input.scopeKey)?.relatedIssueCodes ?? [];

  const pos = 1;
  const neg = 0;

  const record: LessonRecord = {
    id: `lesson:${Date.now().toString(36)}:${Math.random().toString(36).slice(2, 7)}`,
    datasetKey: input.datasetKey,
    category: input.category,
    scopeKey: input.scopeKey,
    title,
    source: input.source,
    status: 'active',
    polarity: input.polarity,
    directive,
    validationCheck,
    relatedIssueCodes:
      input.relatedIssueCodes && input.relatedIssueCodes.length > 0
        ? input.relatedIssueCodes
        : defaultIssues,
    targetRange: input.targetRange,
    evidence: {
      sampleSize: Math.max(1, input.referenceSpriteIds?.length ?? 1),
      positiveCount: pos,
      negativeCount: neg,
      criticDeltaMean: input.criticDeltaMean ?? null,
      referenceSpriteIds: input.referenceSpriteIds ?? [],
      confidence: computeWilsonConfidence(pos, neg),
    },
    conflictGroupId: null,
    supersededById: null,
    createdAt: nowIso,
    updatedAt: nowIso,
    readBackVerified: false,
  };

  return saveLessonVerified(backend, record);
}

/**
 * Bulk-upserts mined lessons into IndexedDB with read-back verification.
 * Preserves any existing `human_feedback` or `critic_loop` lessons and flags conflicts cleanly.
 */
export async function syncMinedLessonsVerified(
  backend: StorageBackend,
  minedLessons: readonly LessonRecord[],
): Promise<{
  writtenCount: number;
  lessons: LessonRecord[];
  conflicts: LessonConflict[];
}> {
  let writtenCount = 0;
  for (const lesson of minedLessons) {
    const existing = await backend.get<LessonRecord>('lessons', lesson.id);
    // Keep archived/superseded user decision if this exact mined rule was already superseded
    if (existing && (existing.status === 'archived' || existing.status === 'superseded')) {
      continue;
    }
    const toWrite: LessonRecord = {
      ...lesson,
      readBackVerified: true,
    };
    await putAndVerify(backend, 'lessons', toWrite, verifyLessonReadBack);
    writtenCount++;
  }

  const all = await backend.getAll<LessonRecord>('lessons');
  const { lessons: annotated, conflicts } = annotateLessonConflictStates(all);
  for (const updated of annotated) {
    const prev = all.find((x) => x.id === updated.id);
    if (
      prev &&
      (prev.status !== updated.status || prev.conflictGroupId !== updated.conflictGroupId)
    ) {
      await putAndVerify(backend, 'lessons', { ...updated, readBackVerified: true }, verifyLessonReadBack);
    }
  }

  const finalAll = await backend.getAll<LessonRecord>('lessons');
  const finalState = annotateLessonConflictStates(finalAll);
  return {
    writtenCount,
    lessons: finalState.lessons,
    conflicts: finalState.conflicts,
  };
}

/**
 * Records a confirm (+1 positive) or contest (+1 negative) evidence observation
 * with optional measured Critic v1 delta, recomputes Wilson confidence, and verifies read-back.
 */
export async function recordLessonEvidenceVerified(
  backend: StorageBackend,
  lessonId: string,
  outcome: 'confirm' | 'contest',
  observedCriticDelta?: number,
  nowIso = new Date().toISOString(),
): Promise<{
  saved: LessonRecord;
  lessons: LessonRecord[];
  conflicts: LessonConflict[];
}> {
  const existing = await backend.get<LessonRecord>('lessons', lessonId);
  if (!existing) {
    throw new Error(`Lição "${lessonId}" não encontrada no IndexedDB.`);
  }

  const pos = existing.evidence.positiveCount + (outcome === 'confirm' ? 1 : 0);
  const neg = existing.evidence.negativeCount + (outcome === 'contest' ? 1 : 0);
  const prevTotalObs = existing.evidence.positiveCount + existing.evidence.negativeCount;

  let nextDeltaMean = existing.evidence.criticDeltaMean;
  if (observedCriticDelta !== undefined && Number.isFinite(observedCriticDelta)) {
    nextDeltaMean =
      existing.evidence.criticDeltaMean === null
        ? Math.round(observedCriticDelta * 10) / 10
        : Math.round(
            ((existing.evidence.criticDeltaMean * prevTotalObs + observedCriticDelta) /
              (prevTotalObs + 1)) *
              10,
          ) / 10;
  }

  const updated: LessonRecord = {
    ...existing,
    evidence: {
      ...existing.evidence,
      sampleSize: existing.evidence.sampleSize + 1,
      positiveCount: pos,
      negativeCount: neg,
      criticDeltaMean: nextDeltaMean,
      confidence: computeWilsonConfidence(pos, neg),
    },
    updatedAt: nowIso,
    readBackVerified: true,
  };

  return saveLessonVerified(backend, updated);
}

/**
 * Archives or reactivates a lesson with read-back verification.
 */
export async function setLessonArchiveStateVerified(
  backend: StorageBackend,
  lessonId: string,
  archived: boolean,
  nowIso = new Date().toISOString(),
): Promise<{
  saved: LessonRecord;
  lessons: LessonRecord[];
  conflicts: LessonConflict[];
}> {
  const existing = await backend.get<LessonRecord>('lessons', lessonId);
  if (!existing) {
    throw new Error(`Lição "${lessonId}" não encontrada no IndexedDB.`);
  }

  const updated: LessonRecord = {
    ...existing,
    status: archived ? 'archived' : 'active',
    conflictGroupId: null,
    supersededById: archived ? existing.supersededById : null,
    updatedAt: nowIso,
    readBackVerified: true,
  };

  return saveLessonVerified(backend, updated);
}

/**
 * Resolves a single conflict in IndexedDB with read-back verification on both winner and loser.
 */
export async function resolveConflictVerified(
  backend: StorageBackend,
  conflict: LessonConflict,
  strategy: ConflictResolutionStrategy,
  preferredWinnerId?: string,
  nowIso = new Date().toISOString(),
): Promise<{
  resolution: ConflictResolutionResult;
  lessons: LessonRecord[];
  conflicts: LessonConflict[];
}> {
  const resolution = resolveLessonConflictPair({
    conflict,
    strategy,
    preferredWinnerId,
    nowIso,
  });

  await putAndVerify(
    backend,
    'lessons',
    { ...resolution.winner, readBackVerified: true },
    verifyLessonReadBack,
  );
  await putAndVerify(
    backend,
    'lessons',
    { ...resolution.loser, readBackVerified: true },
    verifyLessonReadBack,
  );

  const all = await backend.getAll<LessonRecord>('lessons');
  const { lessons: annotated, conflicts } = annotateLessonConflictStates(all);
  for (const updated of annotated) {
    const prev = all.find((x) => x.id === updated.id);
    if (
      prev &&
      (prev.status !== updated.status || prev.conflictGroupId !== updated.conflictGroupId)
    ) {
      await putAndVerify(backend, 'lessons', { ...updated, readBackVerified: true }, verifyLessonReadBack);
    }
  }

  const finalAll = await backend.getAll<LessonRecord>('lessons');
  const finalState = annotateLessonConflictStates(finalAll);
  return {
    resolution,
    lessons: finalState.lessons,
    conflicts: finalState.conflicts,
  };
}

/**
 * Iteratively resolves all currently active conflicts using automatic precedence until 0 conflicts remain.
 */
export async function resolveAllConflictsAutoVerified(
  backend: StorageBackend,
  nowIso = new Date().toISOString(),
): Promise<{
  resolutions: ConflictResolutionResult[];
  lessons: LessonRecord[];
  conflicts: LessonConflict[];
}> {
  const resolutions: ConflictResolutionResult[] = [];
  let guard = 0;

  while (guard < 64) {
    guard++;
    const { conflicts } = await loadAllLessons(backend);
    if (conflicts.length === 0) break;
    const res = await resolveConflictVerified(
      backend,
      conflicts[0],
      'auto_precedence',
      undefined,
      nowIso,
    );
    resolutions.push(res.resolution);
  }

  const finalState = await loadAllLessons(backend);
  return {
    resolutions,
    lessons: finalState.lessons,
    conflicts: finalState.conflicts,
  };
}

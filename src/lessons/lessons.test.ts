import test from 'node:test';
import assert from 'node:assert/strict';
import { calibrateCriticAndBuildDiagnostics } from '../critic/diagnostics';
import { makeCleanPixelArtFixture } from '../critic/fixtures';
import type { LabeledSpriteSample } from '../critic/profile';
import { createMemoryStorageBackend, type StorageBackend } from '../db/persistence';
import { ALL_ITEM_CATEGORIES } from '../types/studio';
import {
  annotateLessonConflictStates,
  detectLessonConflicts,
  resolveLessonConflictPair,
} from './conflicts';
import { mineLessonsFromCalibration } from './miner';
import { formatLessonsPromptBlock, retrieveLessonsForQuery } from './retrieval';
import {
  createLessonVerified,
  loadAllLessons,
  recordLessonEvidenceVerified,
  resolveAllConflictsAutoVerified,
  resolveConflictVerified,
  syncMinedLessonsVerified,
} from './store';
import { computeWilsonConfidence, type LessonRecord } from './types';

function makeLesson(overrides: Partial<LessonRecord> & { id: string }): LessonRecord {
  return {
    datasetKey: 'ds-1',
    category: 'Equipamentos',
    scopeKey: 'outline_coverage',
    title: `Lição ${overrides.id}`,
    source: 'mined',
    status: 'active',
    polarity: 'enforce',
    directive: `Diretriz padrão ${overrides.id}`,
    validationCheck: 'outlineCoverage >= 50%',
    relatedIssueCodes: ['MISSING_OUTLINE'],
    evidence: {
      sampleSize: 20,
      positiveCount: 16,
      negativeCount: 4,
      criticDeltaMean: null,
      referenceSpriteIds: [101, 102],
      confidence: computeWilsonConfidence(16, 4),
    },
    conflictGroupId: null,
    supersededById: null,
    createdAt: '2026-05-01T10:00:00.000Z',
    updatedAt: '2026-05-01T10:00:00.000Z',
    readBackVerified: true,
    ...overrides,
  };
}

// ============================================================================
// 1. RETRIEVAL ORDERING TESTS
// ============================================================================

test('retrieval ordering prioritizes exact category over Global and excludes unrelated categories', () => {
  const nowMs = Date.parse('2026-05-10T00:00:00.000Z');
  const lessons: LessonRecord[] = [
    makeLesson({ id: 'other-cat', category: 'Containers', source: 'human_feedback' }),
    makeLesson({ id: 'global-rule', category: 'Global', source: 'mined' }),
    makeLesson({ id: 'exact-cat', category: 'Equipamentos', source: 'mined' }),
  ];

  const ranked = retrieveLessonsForQuery(lessons, {
    category: 'Equipamentos',
    nowMs,
  });

  assert.deepEqual(
    ranked.map((r) => r.lesson.id),
    ['exact-cat', 'global-rule'],
  );
  assert.ok(ranked[0].score.categoryMatch > ranked[1].score.categoryMatch);
});

test('retrieval ordering ranks human_feedback > critic_loop > mined when category and evidence match', () => {
  const nowMs = Date.parse('2026-05-10T00:00:00.000Z');
  const lessons: LessonRecord[] = [
    makeLesson({ id: 'l-mined', source: 'mined' }),
    makeLesson({ id: 'l-critic', source: 'critic_loop' }),
    makeLesson({ id: 'l-human', source: 'human_feedback' }),
  ];

  const ranked = retrieveLessonsForQuery(lessons, {
    category: 'Equipamentos',
    nowMs,
  });

  assert.deepEqual(
    ranked.map((r) => r.lesson.id),
    ['l-human', 'l-critic', 'l-mined'],
  );
  assert.ok(ranked[0].score.sourceAuthority > ranked[1].score.sourceAuthority);
  assert.ok(ranked[1].score.sourceAuthority > ranked[2].score.sourceAuthority);
});

test('retrieval ordering boosts lessons matching active Critic v1 issue codes and excludes superseded/archived by default', () => {
  const nowMs = Date.parse('2026-05-10T00:00:00.000Z');
  const lessons: LessonRecord[] = [
    makeLesson({
      id: 'unrelated-human',
      source: 'human_feedback',
      scopeKey: 'dither_share',
      relatedIssueCodes: ['DITHER_MISMATCH'],
    }),
    makeLesson({
      id: 'issue-match-critic',
      source: 'critic_loop',
      scopeKey: 'pillow_shading',
      relatedIssueCodes: ['PILLOW_SHADING', 'BANDING_DETECTED'],
    }),
    makeLesson({
      id: 'superseded-rule',
      status: 'superseded',
      supersededById: 'issue-match-critic',
      relatedIssueCodes: ['PILLOW_SHADING'],
    }),
    makeLesson({
      id: 'archived-rule',
      status: 'archived',
      relatedIssueCodes: ['PILLOW_SHADING'],
    }),
  ];

  const ranked = retrieveLessonsForQuery(lessons, {
    category: 'Equipamentos',
    issueCodes: ['PILLOW_SHADING'],
    nowMs,
  });

  assert.deepEqual(
    ranked.map((r) => r.lesson.id),
    ['issue-match-critic', 'unrelated-human'],
  );
  assert.ok(ranked[0].score.issueRelevance > 0);
  assert.equal(ranked[1].score.issueRelevance, 0);

  const promptBlock = formatLessonsPromptBlock(ranked);
  assert.ok(promptBlock.includes('EXIGIR'));
  assert.ok(promptBlock.includes('Equipamentos'));
});

test('retrieval ordering rewards stronger Wilson evidence, positive Critic delta, and fresher timestamps deterministically', () => {
  const nowMs = Date.parse('2026-05-20T00:00:00.000Z');
  const strongFresh = makeLesson({
    id: 'strong-fresh',
    source: 'critic_loop',
    updatedAt: '2026-05-19T12:00:00.000Z',
    evidence: {
      sampleSize: 40,
      positiveCount: 36,
      negativeCount: 4,
      criticDeltaMean: 8.5,
      referenceSpriteIds: [1, 2],
      confidence: computeWilsonConfidence(36, 4),
    },
  });
  const weakStale = makeLesson({
    id: 'weak-stale',
    source: 'critic_loop',
    updatedAt: '2026-02-01T12:00:00.000Z',
    evidence: {
      sampleSize: 5,
      positiveCount: 2,
      negativeCount: 3,
      criticDeltaMean: -2.0,
      referenceSpriteIds: [3],
      confidence: computeWilsonConfidence(2, 3),
    },
  });

  const runA = retrieveLessonsForQuery([weakStale, strongFresh], { category: 'Equipamentos', nowMs });
  const runB = retrieveLessonsForQuery([strongFresh, weakStale], { category: 'Equipamentos', nowMs });

  assert.deepEqual(
    runA.map((r) => r.lesson.id),
    ['strong-fresh', 'weak-stale'],
  );
  assert.deepEqual(runA, runB);
});

// ============================================================================
// 2. CONFLICT DETECTION & RESOLUTION TESTS
// ============================================================================

test('conflict detector catches opposite polarity, disjoint ranges, and competing directives without falsely flagging category-vs-Global specialization', () => {
  const lessons: LessonRecord[] = [
    // Polarity clash on Equipamentos + dither_share
    makeLesson({ id: 'pol-a', category: 'Equipamentos', scopeKey: 'dither_share', polarity: 'enforce' }),
    makeLesson({ id: 'pol-b', category: 'Equipamentos', scopeKey: 'dither_share', polarity: 'avoid' }),
    // Disjoint numeric range clash on Consumíveis + palette_size
    makeLesson({
      id: 'rng-a',
      category: 'Consumíveis',
      scopeKey: 'palette_size',
      polarity: 'enforce',
      targetRange: { min: 4, max: 10 },
    }),
    makeLesson({
      id: 'rng-b',
      category: 'Consumíveis',
      scopeKey: 'palette_size',
      polarity: 'enforce',
      targetRange: { min: 18, max: 30 },
    }),
    // Global rule + specific category rule on same scopeKey -> NOT a conflict (specialization)
    makeLesson({ id: 'glob-1', category: 'Global', scopeKey: 'nw_lighting', polarity: 'enforce' }),
    makeLesson({ id: 'spec-1', category: 'Containers', scopeKey: 'nw_lighting', polarity: 'enforce' }),
  ];

  const conflicts = detectLessonConflicts(lessons);
  assert.equal(conflicts.length, 2);

  const reasons = new Set(conflicts.map((c) => c.reason));
  assert.ok(reasons.has('opposite_polarity'));
  assert.ok(reasons.has('disjoint_range'));

  const annotated = annotateLessonConflictStates(lessons);
  const byId = new Map(annotated.lessons.map((l) => [l.id, l]));
  assert.equal(byId.get('pol-a')?.status, 'conflicted');
  assert.equal(byId.get('pol-b')?.status, 'conflicted');
  assert.equal(byId.get('glob-1')?.status, 'active');
  assert.equal(byId.get('spec-1')?.status, 'active');
});

test('conflict resolution auto_precedence prefers human_feedback > critic_loop > mined and marks loser superseded', () => {
  const mined = makeLesson({
    id: 'mined-palette',
    source: 'mined',
    category: 'Equipamentos',
    scopeKey: 'palette_size',
    polarity: 'enforce',
    targetRange: { min: 6, max: 16 },
    evidence: {
      sampleSize: 120,
      positiveCount: 100,
      negativeCount: 20,
      criticDeltaMean: null,
      referenceSpriteIds: [1, 2],
      confidence: computeWilsonConfidence(100, 20),
    },
  });
  const human = makeLesson({
    id: 'human-palette',
    source: 'human_feedback',
    category: 'Equipamentos',
    scopeKey: 'palette_size',
    polarity: 'enforce',
    targetRange: { min: 20, max: 28 },
    evidence: {
      sampleSize: 2,
      positiveCount: 2,
      negativeCount: 0,
      criticDeltaMean: 4.0,
      referenceSpriteIds: [10],
      confidence: computeWilsonConfidence(2, 0),
    },
  });

  const [conflict] = detectLessonConflicts([mined, human]);
  assert.ok(conflict);

  const res = resolveLessonConflictPair({
    conflict,
    strategy: 'auto_precedence',
    nowIso: '2026-05-20T00:00:00.000Z',
  });

  assert.equal(res.winner.id, 'human-palette');
  assert.equal(res.winner.status, 'active');
  assert.equal(res.loser.id, 'mined-palette');
  assert.equal(res.loser.status, 'superseded');
  assert.equal(res.loser.supersededById, 'human-palette');

  // Also test merge_range strategy when polarities match
  const merged = resolveLessonConflictPair({
    conflict,
    strategy: 'merge_range',
    nowIso: '2026-05-20T00:00:00.000Z',
  });
  assert.equal(merged.winner.targetRange?.min, 6);
  assert.equal(merged.winner.targetRange?.max, 28);
  assert.equal(merged.loser.status, 'superseded');
});

// ============================================================================
// 3. REAL MINING + INDEXEDDB PERSISTENCE & READ-BACK VERIFICATION TESTS
// ============================================================================

test('mining from calibrated train split produces strictly data-driven lessons and persists with read-back verification', async () => {
  const trainSamples: LabeledSpriteSample[] = [];
  const validationSamples: LabeledSpriteSample[] = [];
  let sid = 1;
  for (const cat of ALL_ITEM_CATEGORIES) {
    for (let i = 0; i < 12; i++) {
      trainSamples.push({ spriteId: sid++, category: cat, rgba: makeCleanPixelArtFixture(cat, i) });
    }
    for (let i = 0; i < 4; i++) {
      validationSamples.push({ spriteId: sid++, category: cat, rgba: makeCleanPixelArtFixture(cat, 12 + i) });
    }
  }

  const { calibration } = calibrateCriticAndBuildDiagnostics({
    datasetKey: 'test-ds:hash',
    trainSamples,
    validationSamples,
    splitTotals: { train: 48, validation: 16, test: 16, total: 80 },
    seed: 772,
  });

  const mined = mineLessonsFromCalibration({ calibration });
  assert.ok(mined.length >= 16, `Expected at least 16 mined lessons across 4 categories, got ${mined.length}`);
  for (const l of mined) {
    assert.equal(l.source, 'mined');
    assert.ok(l.evidence.sampleSize >= 12);
    assert.ok(l.evidence.confidence > 0 && l.evidence.confidence < 1, 'Confidence must never be flat 100%');
    if (l.targetRange?.unit === '%' && l.targetRange.min !== undefined && l.targetRange.max !== undefined) {
      assert.ok(l.targetRange.min >= 0 && l.targetRange.max <= 100, `Percent range out of bounds: ${l.id}`);
      assert.ok(l.targetRange.min <= l.targetRange.max, `Inverted range: ${l.id}`);
    }
  }

  const db = createMemoryStorageBackend();
  const synced = await syncMinedLessonsVerified(db, mined);
  assert.equal(synced.writtenCount, mined.length);
  assert.equal(synced.conflicts.length, 0);

  // Add a conflicting human_feedback lesson on Equipamentos + palette_size
  const created = await createLessonVerified(db, {
    datasetKey: calibration.datasetKey,
    category: 'Equipamentos',
    scopeKey: 'palette_size',
    title: 'Restrição manual de paleta enxuta',
    source: 'human_feedback',
    polarity: 'avoid',
    directive: 'Evitar mais de 10 cores em armaduras clássicas.',
    validationCheck: 'distinctColors <= 10',
  });
  assert.equal(created.saved.readBackVerified, true);
  assert.equal(created.conflicts.length, 1);

  // Confirm evidence +1 and verify confidence updates & persists
  const confirmed = await recordLessonEvidenceVerified(db, created.saved.id, 'confirm', 6.5);
  assert.equal(confirmed.saved.evidence.positiveCount, 2);
  assert.equal(confirmed.saved.evidence.criticDeltaMean, 6.5);

  // Resolve conflict automatically -> human rule wins over mined rule
  const autoResolved = await resolveAllConflictsAutoVerified(db);
  assert.equal(autoResolved.resolutions.length, 1);
  assert.equal(autoResolved.resolutions[0].winner.id, created.saved.id);
  assert.equal(autoResolved.conflicts.length, 0);

  // Verify persistence reload
  const reloaded = await loadAllLessons(db);
  assert.equal(reloaded.conflicts.length, 0);
  const retrieved = retrieveLessonsForQuery(reloaded.lessons, {
    category: 'Equipamentos',
    issueCodes: ['COLOR_COUNT_OUT_OF_RANGE'],
  });
  assert.equal(retrieved[0].lesson.id, created.saved.id);
});

test('createLessonVerified rejects silent storage failures via mandatory read-back check', async () => {
  const brokenBackend: StorageBackend = {
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

  await assert.rejects(
    () =>
      createLessonVerified(brokenBackend, {
        datasetKey: 'ds',
        category: 'Equipamentos',
        scopeKey: 'outline_coverage',
        title: 'Regra de teste',
        source: 'human_feedback',
        polarity: 'enforce',
        directive: 'Exigir contorno escuro 1px.',
        validationCheck: 'outlineCoverage >= 60%',
      }),
    /read-back/i,
  );
});

test('manual keep_lesson conflict resolution persists chosen winner in IndexedDB', async () => {
  const db = createMemoryStorageBackend();
  const a = await createLessonVerified(db, {
    datasetKey: 'ds',
    category: 'Decoração',
    scopeKey: 'nw_lighting',
    title: 'Luz NW forte',
    source: 'human_feedback',
    polarity: 'enforce',
    directive: 'Luz sempre NW.',
    validationCheck: 'nwLightGradient > 0.1',
  });
  const b = await createLessonVerified(db, {
    datasetKey: 'ds',
    category: 'Decoração',
    scopeKey: 'nw_lighting',
    title: 'Luz difusa sem direção',
    source: 'human_feedback',
    polarity: 'avoid',
    directive: 'Evitar contraste direcional forte.',
    validationCheck: 'nwLightGradient < 0.05',
  });

  assert.equal(b.conflicts.length, 1);
  const resolved = await resolveConflictVerified(db, b.conflicts[0], 'keep_lesson', a.saved.id);
  assert.equal(resolved.resolution.winner.id, a.saved.id);
  assert.equal(resolved.resolution.loser.id, b.saved.id);
  assert.equal(resolved.conflicts.length, 0);
});

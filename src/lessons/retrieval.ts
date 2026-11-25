import {
  SOURCE_AUTHORITY_RANK,
  type LessonRecord,
  type LessonRetrievalQuery,
  type RetrievalScoreBreakdown,
  type RetrievedLesson,
} from './types';

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .split(/[^a-z0-9_]+/)
    .filter((t) => t.length >= 2);
}

function round2(v: number): number {
  return Math.round(v * 100) / 100;
}

/**
 * Scores a single lesson for a given retrieval query.
 * Deterministic for fixed inputs and `nowMs`.
 *
 * Weight budget (max ~100):
 * - Category specificity: exact category = 28, 'Global' = 14 (other categories excluded before scoring)
 * - Source authority: human_feedback = 24, critic_loop = 16, mined = 9
 * - Issue relevance: up to 24 when lesson addresses active Critic v1 issue codes
 * - Evidence strength: up to 16 from Wilson confidence + sample size + observed Critic delta
 * - Recency: up to 8 (exponential decay with 30-day half-life)
 * - Search text relevance: up to 12 bonus when searchText is provided
 */
export function scoreLessonForQuery(
  lesson: LessonRecord,
  query: LessonRetrievalQuery,
): RetrievalScoreBreakdown {
  const nowMs = query.nowMs ?? Date.now();

  const categoryMatch = lesson.category === query.category ? 28 : 14;

  const authRank = SOURCE_AUTHORITY_RANK[lesson.source];
  const sourceAuthority = authRank === 3 ? 24 : authRank === 2 ? 16 : 9;

  // Issue relevance: fraction of requested Critic v1 issue codes covered by this lesson
  let issueRelevance = 0;
  const issueCodes = query.issueCodes ?? [];
  if (issueCodes.length > 0 && lesson.relatedIssueCodes.length > 0) {
    const lessonIssueSet = new Set(lesson.relatedIssueCodes);
    let hits = 0;
    for (const code of issueCodes) {
      if (lessonIssueSet.has(code)) hits++;
    }
    if (hits > 0) {
      // Strong priority when a lesson directly fixes an active Critic defect
      issueRelevance = Math.min(24, 15 + (hits / issueCodes.length) * 9);
    }
  }

  // Empirical evidence strength: Wilson confidence + log(sampleSize) + measured Critic delta
  const confPart = Math.max(0, Math.min(1, lesson.evidence.confidence)) * 9.5;
  const samplePart = Math.min(3.5, Math.log10(Math.max(1, lesson.evidence.sampleSize) + 1) * 1.6);
  const deltaPart =
    lesson.evidence.criticDeltaMean !== null
      ? Math.max(-3, Math.min(3, lesson.evidence.criticDeltaMean * 0.25))
      : 0;
  const evidenceStrength = Math.max(0, Math.min(16, confPart + samplePart + deltaPart));

  // Recency decay: 30-day half-life
  const updatedMs = Date.parse(lesson.updatedAt);
  const ageDays = Number.isFinite(updatedMs)
    ? Math.max(0, (nowMs - updatedMs) / (1000 * 60 * 60 * 24))
    : 30;
  const recency = 8 * Math.exp((-Math.LN2 * ageDays) / 30);

  // Keyword search match
  let textMatch = 0;
  const searchTokens = query.searchText ? tokenize(query.searchText) : [];
  if (searchTokens.length > 0) {
    const haystack = tokenize(
      `${lesson.title} ${lesson.directive} ${lesson.validationCheck} ${lesson.scopeKey} ${lesson.relatedIssueCodes.join(' ')}`,
    );
    const haySet = new Set(haystack);
    let matched = 0;
    for (const tok of searchTokens) {
      if (haySet.has(tok) || haystack.some((h) => h.includes(tok))) {
        matched++;
      }
    }
    textMatch = (matched / searchTokens.length) * 12;
  }

  const total =
    categoryMatch + sourceAuthority + issueRelevance + evidenceStrength + recency + textMatch;

  return {
    categoryMatch: round2(categoryMatch),
    sourceAuthority: round2(sourceAuthority),
    issueRelevance: round2(issueRelevance),
    evidenceStrength: round2(evidenceStrength),
    recency: round2(recency),
    textMatch: round2(textMatch),
    total: round2(total),
  };
}

/**
 * Retrieves and deterministically ranks lessons applicable to `query`.
 *
 * Filtering rules:
 * - Only lessons matching `query.category` OR `'Global'` are returned.
 * - If `query.datasetKey` is provided, matches that datasetKey OR `null` (cross-dataset rules).
 * - By default, excludes `'superseded'`, `'archived'`, and `'conflicted'` lessons (unless opted in).
 * - If `searchText` is provided and non-empty, requires at least one keyword match (`textMatch > 0`).
 */
export function retrieveLessonsForQuery(
  lessons: readonly LessonRecord[],
  query: LessonRetrievalQuery,
): RetrievedLesson[] {
  const allowConflicted = query.includeConflicted ?? false;
  const allowSuperseded = query.includeSuperseded ?? false;
  const allowArchived = query.includeArchived ?? false;
  const hasSearch = Boolean(query.searchText && query.searchText.trim().length > 0);

  const filtered = lessons.filter((l) => {
    if (l.category !== query.category && l.category !== 'Global') return false;
    if (query.datasetKey && l.datasetKey !== null && l.datasetKey !== query.datasetKey) {
      return false;
    }
    if (l.status === 'conflicted' && !allowConflicted) return false;
    if (l.status === 'superseded' && !allowSuperseded) return false;
    if (l.status === 'archived' && !allowArchived) return false;
    return true;
  });

  const scored: RetrievedLesson[] = [];
  for (const lesson of filtered) {
    const score = scoreLessonForQuery(lesson, query);
    if (hasSearch && score.textMatch <= 0) continue;
    scored.push({ lesson, score });
  }

  scored.sort((a, b) => {
    if (b.score.total !== a.score.total) return b.score.total - a.score.total;
    const tA = Date.parse(a.lesson.updatedAt) || 0;
    const tB = Date.parse(b.lesson.updatedAt) || 0;
    if (tB !== tA) return tB - tA;
    return a.lesson.id.localeCompare(b.lesson.id);
  });

  if (query.limit !== undefined && query.limit >= 0) {
    return scored.slice(0, query.limit);
  }
  return scored;
}

/**
 * Formats retrieved active lessons into a concise structured directive block for
 * downstream generation & repair loops (Phases 3–5).
 */
export function formatLessonsPromptBlock(retrieved: readonly RetrievedLesson[]): string {
  if (retrieved.length === 0) return '';
  return retrieved
    .map((item, idx) => {
      const l = item.lesson;
      const mode = l.polarity === 'enforce' ? 'EXIGIR' : 'EVITAR';
      const confPct = Math.round(l.evidence.confidence * 100);
      return `${idx + 1}. [${mode} · ${l.category} · ${l.source} · conf ${confPct}% · n=${l.evidence.sampleSize}] ${l.directive} (Verificação: ${l.validationCheck})`;
    })
    .join('\n');
}

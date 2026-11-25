import type { CriticIssueCode } from '../critic/types';
import type { GeneratedSpriteRecord } from '../generation/types';
import {
  LESSON_SCOPE_OPTIONS,
  type LessonPolarity,
  type LessonScopeKey,
} from '../lessons/types';
import type { CommentToLessonDraft } from './types';

export const DEFAULT_ELO = 1200;
export const DEFAULT_K_FACTOR = 32;

/**
 * Standard Elo expected score for player A against player B in (0, 1).
 */
export function expectedEloScore(ratingA: number, ratingB: number): number {
  return 1 / (1 + Math.pow(10, (ratingB - ratingA) / 400));
}

/**
 * Computes exact zero-sum Elo rating updates for a single blind A/B match.
 * `outcomeA`: 1 = A wins, 0.5 = draw, 0 = B wins.
 */
export function computeEloUpdate(
  ratingA: number,
  ratingB: number,
  outcomeA: 1 | 0.5 | 0,
  kFactor = DEFAULT_K_FACTOR,
): {
  nextA: number;
  nextB: number;
  deltaA: number;
  deltaB: number;
  expectedA: number;
} {
  const expA = expectedEloScore(ratingA, ratingB);
  const rawDeltaA = kFactor * (outcomeA - expA);
  const deltaA = Math.round(rawDeltaA * 10) / 10;
  const deltaB = Math.round(-deltaA * 10) / 10;
  return {
    nextA: Math.round((ratingA + deltaA) * 10) / 10,
    nextB: Math.round((ratingB + deltaB) * 10) / 10,
    deltaA,
    deltaB,
    expectedA: Math.round(expA * 1000) / 1000,
  };
}

function normalizePt(text: string): string {
  return text
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '');
}

const SCOPE_KEYWORD_RULES: ReadonlyArray<{
  scopeKey: LessonScopeKey;
  keywords: string[];
  validationCheck: string;
}> = [
  {
    scopeKey: 'outline_coverage',
    keywords: ['contorno', 'outline', 'silhueta', 'borda preta', 'sem contorno', 'quebrado'],
    validationCheck: 'outlineCoverage >= 82% e outlineDarknessDelta >= 0.14 em não-tiles',
  },
  {
    scopeKey: 'outline_quality',
    keywords: ['double', 'duplo', 'escadinha', 'jaggie', 'serrilhado', 'grosso'],
    validationCheck: 'outlineDoublesRatio <= 0.18 e brokenOutlineRatio <= 0.08',
  },
  {
    scopeKey: 'palette_size',
    keywords: ['cor', 'cores', 'paleta', 'chapado', 'flat', 'monocromatico', 'saturado'],
    validationCheck: 'distinctColors dentro da faixa p10–p90 da categoria e lumaLevels >= 4',
  },
  {
    scopeKey: 'near_duplicate_palette',
    keywords: ['gradiente', 'suave', 'duplicada', 'borrado', 'sujo', 'blur'],
    validationCheck: 'nearDuplicateRatio <= 0.08 e smoothGradientRatio < 0.35',
  },
  {
    scopeKey: 'nw_lighting',
    keywords: ['luz', 'iluminacao', 'sombra', 'brilho', 'nw', 'noroeste', 'reflexo', 'especular'],
    validationCheck: 'nwLightGradient >= +0.04 (quadrante NW mais iluminado que SE)',
  },
  {
    scopeKey: 'pillow_shading',
    keywords: ['pillow', 'almofada', 'banding', 'anel', 'concentrico'],
    validationCheck: 'pillowShadingScore <= 0.32 e bandingScore <= 0.36',
  },
  {
    scopeKey: 'cluster_hygiene',
    keywords: ['ruido', 'speckle', 'confete', 'orfao', 'pixel solto', 'sujeira', 'poluido'],
    validationCheck: 'orphanPixels == 0 e strayColorPixelRatio <= 0.14',
  },
  {
    scopeKey: 'dither_share',
    keywords: ['dither', 'xadrez', 'pontilhado', 'textura'],
    validationCheck: 'ditherShare dentro do limite p90 da categoria real',
  },
  {
    scopeKey: 'hue_shift',
    keywords: ['matiz', 'hue', 'tom', 'quente', 'frio', 'metal', 'ouro'],
    validationCheck: 'hueShiftDegrees >= 8° entre sombras frias e luzes quentes',
  },
  {
    scopeKey: 'mixel_resample',
    keywords: ['mixel', 'blur', 'anti-alias', 'antialias', 'aa', 'semitransparente', 'escala'],
    validationCheck: 'semiTransparentPixels == 0 e mixelScore <= 0.25',
  },
  {
    scopeKey: 'border_margin',
    keywords: ['margem', 'encostando', 'cortado', 'centraliz', 'borda 32'],
    validationCheck: 'borderTouchCount <= 2 e centerOffsetNorm <= 0.24',
  },
];

const AVOID_MARKERS = [
  'evitar',
  'evite',
  'nunca',
  'nao ',
  'sem ',
  'remover',
  'remova',
  'excesso',
  'quebrado',
  'borrado',
  'sujo',
  'feio',
  'errado',
  'demais',
  'cortado',
  'muito ',
];

/**
 * Deterministically parses a human critique comment on a sprite into a structured
 * `CommentToLessonDraft` ready to be persisted into the real `Lições Aprendidas` IndexedDB store.
 */
export function parseCommentToLessonDraft(
  commentPt: string,
  sprite: Pick<GeneratedSpriteRecord, 'id' | 'category' | 'evaluation'>,
): CommentToLessonDraft {
  const cleaned = commentPt.trim();
  const norm = normalizePt(cleaned);

  // 1. Infer scopeKey from PT-BR keywords, falling back to top Critic issue on the sprite if any
  let scopeKey: LessonScopeKey | null = null;
  let validationCheck = 'Verificação via Crítico v1 e revisão visual 1:1 (32×32)';

  let bestHits = 0;
  for (const rule of SCOPE_KEYWORD_RULES) {
    let hits = 0;
    for (const kw of rule.keywords) {
      if (norm.includes(kw)) hits++;
    }
    if (hits > bestHits) {
      bestHits = hits;
      scopeKey = rule.scopeKey;
      validationCheck = rule.validationCheck;
    }
  }

  if (!scopeKey && sprite.evaluation.issues.length > 0) {
    const topIssue = sprite.evaluation.issues[0].code;
    const matchedOpt = LESSON_SCOPE_OPTIONS.find((opt) =>
      opt.relatedIssueCodes.includes(topIssue),
    );
    if (matchedOpt) {
      scopeKey = matchedOpt.key;
      const rule = SCOPE_KEYWORD_RULES.find((r) => r.scopeKey === matchedOpt.key);
      if (rule) validationCheck = rule.validationCheck;
    }
  }

  if (!scopeKey) {
    const slug = norm
      .replace(/[^a-z0-9\s]/g, '')
      .split(/\s+/)
      .filter((w) => w.length >= 3)
      .slice(0, 3)
      .join('_');
    scopeKey = `custom:${slug || 'revisao_humana'}`;
  }

  // 2. Infer polarity (`avoid` vs `enforce`)
  const polarity: LessonPolarity = AVOID_MARKERS.some((m) => norm.includes(m))
    ? 'avoid'
    : 'enforce';

  // 3. Collect related Critic v1 issue codes
  const opt = LESSON_SCOPE_OPTIONS.find((o) => o.key === scopeKey);
  const relatedIssueCodes: CriticIssueCode[] = opt
    ? [...opt.relatedIssueCodes]
    : sprite.evaluation.issues.slice(0, 2).map((i) => i.code);

  // 4. Concise title + full directive
  const scopeLabel = opt ? opt.labelPt : 'Revisão Humana';
  const shortSnippet = cleaned.length > 56 ? `${cleaned.slice(0, 53).trim()}...` : cleaned;
  const title = `${scopeLabel}: ${shortSnippet}`;

  return {
    category: sprite.category,
    scopeKey,
    polarity,
    title,
    directive: cleaned,
    validationCheck,
    relatedIssueCodes,
    sourceSpriteId: sprite.id,
  };
}

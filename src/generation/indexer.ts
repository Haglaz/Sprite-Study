import { evaluateSprite } from '../critic/evaluator';
import type { CriticCalibration } from '../critic/types';
import type { StoredDatasetSplit } from '../db/schema';
import {
  ALL_ITEM_CATEGORIES,
  type ItemCategory,
  type ParsedDatArchive,
  type ParsedItemFlags,
  type ParsedItemMetadata,
  type ParsedSprArchive,
} from '../types/studio';
import { rgbToHex } from '../utils/pixelImage';
import { decodeSpritePixels, SPRITE_SIZE } from '../utils/tibiaParser';
import type {
  ApprovalLabelRecord,
  GeneratedSpriteRecord,
  IndexedSpriteEntry,
  NearestRealMatch,
  RampRoleCode,
  RetrievedDonor,
  SpriteIndexState,
} from './types';

function calcLuma(r: number, g: number, b: number): number {
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255;
}

function rgbToHueName(r: number, g: number, b: number): string {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const d = max - min;
  if (max < 45) return 'escuro';
  if (d < 22) return max > 175 ? 'prata' : 'cinza';
  let h = 0;
  if (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) * 60;
  else if (max === g) h = ((b - r) / d + 2) * 60;
  else h = ((r - g) / d + 4) * 60;

  if (h < 20 || h >= 340) return 'vermelho';
  if (h < 45) return 'laranja';
  if (h < 70) return 'dourado';
  if (h < 160) return 'verde';
  if (h < 205) return 'ciano';
  if (h < 255) return 'azul';
  if (h < 305) return 'roxo';
  return 'magenta';
}

export function normalizePromptTokens(text: string): string[] {
  return text
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter((t) => t.length >= 2);
}

/**
 * Extracts exact palette (sorted darkest -> brightest), 32x32 indexed grid,
 * and ramp-role map (`0=transparent, 1=outline, 2=shadow, 3=midtone, 4=highlight, 5=accent`).
 */
export function decomposeSpriteToIndexedAndRampRoles(rgba: Uint8ClampedArray): {
  paletteRgb: Array<[number, number, number]>;
  paletteHex: string[];
  indexedGrid: Uint8Array;
  rampRoles: Uint8Array;
  silhouetteMask: Uint8Array;
  opaqueCount: number;
  bboxW: number;
  bboxH: number;
  dominantHue: string;
} {
  const counts = new Map<number, number>();
  let opaqueCount = 0;
  let minX = SPRITE_SIZE;
  let maxX = -1;
  let minY = SPRITE_SIZE;
  let maxY = -1;

  for (let y = 0; y < SPRITE_SIZE; y++) {
    for (let x = 0; x < SPRITE_SIZE; x++) {
      const p = y * SPRITE_SIZE + x;
      const i = p * 4;
      if (rgba[i + 3] < 128) continue;
      opaqueCount++;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      const key = (rgba[i] << 16) | (rgba[i + 1] << 8) | rgba[i + 2];
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
  }

  const paletteEntries = Array.from(counts.entries())
    .map(([key, count]) => {
      const r = (key >> 16) & 255;
      const g = (key >> 8) & 255;
      const b = key & 255;
      return { key, r, g, b, luma: calcLuma(r, g, b), count };
    })
    .sort((a, b) => a.luma - b.luma);

  const keyToIdx = new Map<number, number>();
  const paletteRgb: Array<[number, number, number]> = [];
  const paletteHex: string[] = [];
  paletteEntries.forEach((entry, idx) => {
    keyToIdx.set(entry.key, idx + 1); // 1-based; 0 = transparent
    paletteRgb.push([entry.r, entry.g, entry.b]);
    paletteHex.push(rgbToHex(entry.r, entry.g, entry.b));
  });

  const indexedGrid = new Uint8Array(SPRITE_SIZE * SPRITE_SIZE);
  const silhouetteMask = new Uint8Array(SPRITE_SIZE * SPRITE_SIZE);
  const rampRoles = new Uint8Array(SPRITE_SIZE * SPRITE_SIZE);

  // Dominant non-dark hue
  const hueCounts = new Map<string, number>();
  for (const e of paletteEntries) {
    if (e.luma > 0.16) {
      const h = rgbToHueName(e.r, e.g, e.b);
      hueCounts.set(h, (hueCounts.get(h) ?? 0) + e.count);
    }
  }
  let dominantHue = 'cinza';
  let bestHueCount = 0;
  for (const [h, c] of hueCounts.entries()) {
    if (c > bestHueCount) {
      bestHueCount = c;
      dominantHue = h;
    }
  }

  const maxLuma = paletteEntries.length > 0 ? paletteEntries[paletteEntries.length - 1].luma : 1;
  const minLuma = paletteEntries.length > 0 ? paletteEntries[0].luma : 0;
  const lumaSpan = Math.max(0.05, maxLuma - minLuma);

  for (let y = 0; y < SPRITE_SIZE; y++) {
    for (let x = 0; x < SPRITE_SIZE; x++) {
      const p = y * SPRITE_SIZE + x;
      const i = p * 4;
      if (rgba[i + 3] < 128) continue;
      silhouetteMask[p] = 1;
      const key = (rgba[i] << 16) | (rgba[i + 1] << 8) | rgba[i + 2];
      indexedGrid[p] = keyToIdx.get(key) ?? 1;

      const isPerimeter =
        x === 0 ||
        x === SPRITE_SIZE - 1 ||
        y === 0 ||
        y === SPRITE_SIZE - 1 ||
        rgba[(p - 1) * 4 + 3] < 128 ||
        rgba[(p + 1) * 4 + 3] < 128 ||
        rgba[(p - SPRITE_SIZE) * 4 + 3] < 128 ||
        rgba[(p + SPRITE_SIZE) * 4 + 3] < 128;

      const luma = calcLuma(rgba[i], rgba[i + 1], rgba[i + 2]);
      const normLuma = (luma - minLuma) / lumaSpan;
      const hue = rgbToHueName(rgba[i], rgba[i + 1], rgba[i + 2]);

      let role: RampRoleCode = 3; // midtone
      if ((isPerimeter && luma <= 0.28) || luma <= minLuma + 0.04) {
        role = 1; // outline
      } else if (hue !== dominantHue && hue !== 'cinza' && hue !== 'escuro' && hue !== 'prata' && normLuma > 0.25) {
        role = 5; // accent
      } else if (normLuma < 0.34) {
        role = 2; // shadow
      } else if (normLuma > 0.72) {
        role = 4; // highlight
      }
      rampRoles[p] = role;
    }
  }

  return {
    paletteRgb,
    paletteHex,
    indexedGrid,
    rampRoles,
    silhouetteMask,
    opaqueCount,
    bboxW: maxX >= minX ? maxX - minX + 1 : 0,
    bboxH: maxY >= minY ? maxY - minY + 1 : 0,
    dominantHue,
  };
}

export function computeLumaSignature8x8(rgba: Uint8ClampedArray): number[] {
  const sig: number[] = new Array(64).fill(0);
  for (let by = 0; by < 8; by++) {
    for (let bx = 0; bx < 8; bx++) {
      let sum = 0;
      for (let dy = 0; dy < 4; dy++) {
        for (let dx = 0; dx < 4; dx++) {
          const i = ((by * 4 + dy) * SPRITE_SIZE + (bx * 4 + dx)) * 4;
          if (rgba[i + 3] >= 128) {
            sum += calcLuma(rgba[i], rgba[i + 1], rgba[i + 2]);
          }
        }
      }
      sig[by * 8 + bx] = Math.round((sum / 16) * 1000) / 1000;
    }
  }
  return sig;
}

function inferTagsForItem(
  category: ItemCategory,
  flags: ParsedItemFlags,
  bboxW: number,
  bboxH: number,
  opaqueCount: number,
  dominantHue: string,
  isTile: boolean,
): string[] {
  const tags = new Set<string>();
  tags.add(category.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, ''));
  tags.add(dominantHue);

  if (flags.marketName) {
    for (const tok of normalizePromptTokens(flags.marketName)) {
      tags.add(tok);
    }
  }
  if (flags.container) {
    tags.add('container');
    tags.add('bau');
    tags.add('mochila');
    tags.add('bolsa');
  }
  if (flags.stackable) {
    tags.add('empilhavel');
    tags.add('moeda');
    tags.add('gema');
    tags.add('consumivel');
  }
  if (flags.fluidContainer || flags.splash) {
    tags.add('pocao');
    tags.add('frasco');
    tags.add('liquido');
    tags.add('elixir');
  }
  if (flags.light) {
    tags.add('luz');
    tags.add('fogo');
    tags.add('tocha');
    tags.add('magico');
    tags.add('lanterna');
  }
  if (flags.ground || isTile) {
    tags.add('tile');
    tags.add('piso');
    tags.add('terreno');
  }
  if (flags.pickupable) tags.add('item');

  // Cloth slot mapping (Tibia classic slots)
  if (flags.clothSlot !== undefined) {
    const slotTags: Record<number, string[]> = {
      1: ['capacete', 'elmo', 'helmet'],
      2: ['amuleto', 'colar'],
      3: ['mochila', 'backpack'],
      4: ['armadura', 'armor', 'peitoral', 'tunic'],
      5: ['escudo', 'shield'],
      6: ['arma', 'espada', 'machado', 'clava', 'sword', 'axe'],
      7: ['calca', 'legs'],
      8: ['botas', 'boots'],
      9: ['anel', 'ring'],
    };
    for (const st of slotTags[flags.clothSlot] ?? []) tags.add(st);
  }

  // Geometry heuristics when clothSlot isn't set in 7.72 .dat
  const aspect = bboxH > 0 ? bboxW / bboxH : 1;
  if (category === 'Equipamentos' && !isTile) {
    if (opaqueCount >= 340 && aspect >= 0.75 && aspect <= 1.25) {
      tags.add('armadura');
      tags.add('escudo');
      tags.add('peitoral');
    } else if (opaqueCount < 280) {
      tags.add('espada');
      tags.add('arma');
      tags.add('lamina');
      tags.add('adaga');
      tags.add('elmo');
      tags.add('botas');
    }
  } else if (category === 'Consumíveis') {
    tags.add('pocao');
    tags.add('runa');
    tags.add('comida');
    tags.add('frasco');
  } else if (category === 'Decoração' && !isTile) {
    tags.add('decoracao');
    tags.add('estatueta');
    tags.add('calice');
    tags.add('trofeu');
    tags.add('cristal');
  }

  return Array.from(tags);
}

/**
 * Indexes real sprites from `train` and `validation` splits ONLY (never `test`).
 * Builds exact palettes, 32x32 indexed grids, ramp-role maps, tags, signatures, and Critic v1 scores.
 */
export function runSpriteIndexingJob(params: {
  datasetKey: string;
  dat: ParsedDatArchive;
  spr: ParsedSprArchive;
  split: StoredDatasetSplit;
  calibration: CriticCalibration;
  maxPerCategoryTrain?: number;
  maxPerCategoryVal?: number;
  onProgress?: (done: number, total: number) => void;
}): SpriteIndexState {
  const {
    datasetKey,
    dat,
    spr,
    split,
    calibration,
    maxPerCategoryTrain = 120,
    maxPerCategoryVal = 40,
    onProgress,
  } = params;

  const itemBySpriteId = new Map<number, ParsedItemMetadata>();
  for (const item of dat.items) {
    for (const sid of item.spriteIds) {
      if (sid > 0 && !itemBySpriteId.has(sid)) {
        itemBySpriteId.set(sid, item);
      }
    }
  }

  const tasks: Array<{ spriteId: number; category: ItemCategory; partition: 'train' | 'validation' }> = [];
  let excludedTestCount = 0;

  for (const category of ALL_ITEM_CATEGORIES) {
    const part = split.byCategory[category];
    if (!part) continue;
    excludedTestCount += part.test.length;
    for (const sid of part.train.slice(0, maxPerCategoryTrain)) {
      tasks.push({ spriteId: sid, category, partition: 'train' });
    }
    for (const sid of part.validation.slice(0, maxPerCategoryVal)) {
      tasks.push({ spriteId: sid, category, partition: 'validation' });
    }
  }

  const entries: IndexedSpriteEntry[] = [];
  let trainCount = 0;
  let validationCount = 0;

  for (let i = 0; i < tasks.length; i++) {
    const { spriteId, category, partition } = tasks[i];
    const rgba = decodeSpritePixels(spr, spriteId);
    if (!rgba) continue;

    const item = itemBySpriteId.get(spriteId);
    const flags: ParsedItemFlags = item?.flags ?? {
      ground: false,
      groundBorder: false,
      onBottom: false,
      onTop: false,
      container: false,
      stackable: false,
      forceUse: false,
      multiUse: false,
      usable: false,
      writable: false,
      pickupable: false,
      movable: false,
      unpassable: false,
      fluidContainer: false,
      splash: false,
      light: false,
      hangable: false,
      equipable: false,
    };

    const evalRes = evaluateSprite(rgba, { category, flags, calibration });
    const dec = decomposeSpriteToIndexedAndRampRoles(rgba);
    if (dec.opaqueCount < 16) continue;

    const tags = inferTagsForItem(
      category,
      flags,
      dec.bboxW,
      dec.bboxH,
      dec.opaqueCount,
      dec.dominantHue,
      evalRes.isTile,
    );

    entries.push({
      spriteId,
      itemId: item?.id ?? spriteId,
      category,
      split: partition,
      flags,
      tags,
      paletteHex: dec.paletteHex,
      paletteRgb: dec.paletteRgb,
      indexedGrid: dec.indexedGrid,
      rampRoles: dec.rampRoles,
      silhouetteMask: dec.silhouetteMask,
      lumaSignature: computeLumaSignature8x8(rgba),
      rgba,
      criticScore: evalRes.score,
      isTile: evalRes.isTile,
      opaqueCount: dec.opaqueCount,
      bboxW: dec.bboxW,
      bboxH: dec.bboxH,
      dominantHue: dec.dominantHue,
    });

    if (partition === 'train') trainCount++;
    else validationCount++;

    if (onProgress && (i % 25 === 0 || i === tasks.length - 1)) {
      onProgress(i + 1, tasks.length);
    }
  }

  return {
    datasetKey,
    indexedAt: new Date().toISOString(),
    entries,
    trainCount,
    validationCount,
    excludedTestCount,
  };
}

/**
 * Retrieves top-K donor sprites strictly from the `train` split matching category + prompt tokens + high Critic quality.
 * Never leaks `validation` or `test` splits into generation donors!
 */
export function retrieveDonorsForPrompt(params: {
  index: SpriteIndexState;
  category: ItemCategory;
  prompt: string;
  topK?: number;
  excludeTiles?: boolean;
  approvals?: readonly ApprovalLabelRecord[];
  approvedSprites?: readonly GeneratedSpriteRecord[];
}): RetrievedDonor[] {
  const {
    index,
    category,
    prompt,
    topK = 8,
    excludeTiles = true,
    approvals = [],
    approvedSprites = [],
  } = params;
  const tokens = normalizePromptTokens(prompt);
  const wantsTile = tokens.some((t) => ['tile', 'piso', 'chao', 'terreno', 'grama', 'pedra'].includes(t));

  // Build empirical donor feedback map from past human approvals & rejections
  const donorFeedbackScore = new Map<number, number>();
  const approvalBySpriteId = new Map(approvals.map((a) => [a.spriteRecordId, a]));

  // Strictly filter real sprites to `train` split only
  let candidates = index.entries.filter(
    (e) =>
      e.split === 'train' &&
      e.category === category &&
      (!excludeTiles || wantsTile || !e.isTile),
  );
  if (candidates.length === 0) {
    candidates = index.entries.filter((e) => e.split === 'train' && (!excludeTiles || !e.isTile));
  }

  // Feed human-approved generated sprites into the few-shot donor pool (with negative synthetic IDs so provenance stays unmistakable)
  const approvedPoolEntries: IndexedSpriteEntry[] = [];
  for (let i = 0; i < approvedSprites.length; i++) {
    const sprRec = approvedSprites[i];
    const decision = approvalBySpriteId.get(sprRec.id)?.decision ?? sprRec.approvalStatus;
    if (decision === 'approved') {
      for (const dId of sprRec.provenance.donorSpriteIds) {
        donorFeedbackScore.set(dId, Math.min(12, (donorFeedbackScore.get(dId) ?? 0) + 5));
      }
      if (sprRec.category === category && sprRec.readBackVerified) {
        const dec = decomposeSpriteToIndexedAndRampRoles(sprRec.rgba);
        if (dec.opaqueCount >= 16) {
          approvedPoolEntries.push({
            spriteId: -(i + 1), // negative ID signals human-approved few-shot donor
            itemId: -(i + 1),
            category: sprRec.category,
            split: 'train',
            flags: {
              ground: false,
              groundBorder: false,
              onBottom: false,
              onTop: false,
              container: sprRec.category === 'Containers',
              stackable: sprRec.category === 'Consumíveis',
              forceUse: false,
              multiUse: false,
              usable: sprRec.category === 'Consumíveis',
              writable: false,
              pickupable: true,
              movable: true,
              unpassable: false,
              fluidContainer: sprRec.category === 'Consumíveis',
              splash: false,
              light: false,
              hangable: false,
              equipable: sprRec.category === 'Equipamentos',
            },
            tags: Array.from(
              new Set([
                ...normalizePromptTokens(sprRec.prompt),
                dec.dominantHue,
                'aprovado_humano',
              ]),
            ),
            paletteHex: dec.paletteHex,
            paletteRgb: dec.paletteRgb,
            indexedGrid: dec.indexedGrid,
            rampRoles: dec.rampRoles,
            silhouetteMask: dec.silhouetteMask,
            lumaSignature: computeLumaSignature8x8(sprRec.rgba),
            rgba: sprRec.rgba,
            criticScore: sprRec.evaluation.score,
            isTile: sprRec.evaluation.isTile,
            opaqueCount: dec.opaqueCount,
            bboxW: dec.bboxW,
            bboxH: dec.bboxH,
            dominantHue: dec.dominantHue,
          });
        }
      }
    } else if (decision === 'rejected') {
      for (const dId of sprRec.provenance.donorSpriteIds) {
        donorFeedbackScore.set(dId, Math.max(-14, (donorFeedbackScore.get(dId) ?? 0) - 6));
      }
    }
  }

  const allPool = [...candidates, ...approvedPoolEntries];

  const scored: RetrievedDonor[] = allPool.map((entry) => {
    const matchedTags: string[] = [];
    for (const tok of tokens) {
      if (entry.tags.some((t) => t.includes(tok) || tok.includes(t))) {
        matchedTags.push(tok);
      }
    }

    // Geometry preference hints from prompt
    let shapeBonus = 0;
    const isArmorPrompt = tokens.some((t) =>
      ['armadura', 'armor', 'peitoral', 'escudo', 'shield', 'plate', 'manto'].includes(t),
    );
    const isWeaponPrompt = tokens.some((t) =>
      ['espada', 'sword', 'machado', 'axe', 'lanca', 'adaga', 'dagger', 'clava', 'cajado'].includes(t),
    );
    if (isArmorPrompt && entry.opaqueCount >= 300) shapeBonus += 12;
    if (isWeaponPrompt && entry.opaqueCount >= 70 && entry.opaqueCount <= 340) shapeBonus += 10;

    // Prefer clean classic 7.72 pixel-art donors (discrete <= 32 color palettes, crisp high Critic score)
    const colorCount = entry.paletteHex.length;
    const paletteCrispness =
      colorCount >= 6 && colorCount <= 28 ? 10 : colorCount > 48 ? -18 : 0;
    const humanFeedbackBonus =
      entry.spriteId < 0 ? 8 : (donorFeedbackScore.get(entry.spriteId) ?? 0);
    const qualityTerm = entry.criticScore * 0.45;
    const tagTerm = matchedTags.length * 14;
    const relevanceScore =
      Math.round((qualityTerm + tagTerm + shapeBonus + paletteCrispness + humanFeedbackBonus) * 10) /
      10;

    return { entry, relevanceScore, matchedTags };
  });

  scored.sort((a, b) =>
    b.relevanceScore !== a.relevanceScore
      ? b.relevanceScore - a.relevanceScore
      : a.entry.spriteId - b.entry.spriteId,
  );

  // Ensure visual diversity across topK donors (avoid returning near-identical animation frames)
  const diverse: RetrievedDonor[] = [];
  for (const cand of scored) {
    if (diverse.length >= topK) break;
    const tooClose = diverse.some((d) => {
      let l1 = 0;
      for (let k = 0; k < 64; k++) l1 += Math.abs(d.entry.lumaSignature[k] - cand.entry.lumaSignature[k]);
      return l1 / 64 < 0.035;
    });
    if (!tooClose) diverse.push(cand);
  }

  // Fill up to topK if diversity filter was strict
  for (const cand of scored) {
    if (diverse.length >= topK) break;
    if (!diverse.includes(cand)) diverse.push(cand);
  }

  return diverse;
}

/**
 * Finds the exact nearest real `.spr` sprite in the indexed dataset (train + validation)
 * and computes an honest Originality Score (0..99.5, where 0 = identical copy).
 */
export function findNearestRealSprite(
  candidateRgba: Uint8ClampedArray,
  index: SpriteIndexState,
): NearestRealMatch {
  const candSig = computeLumaSignature8x8(candidateRgba);

  // Fast coarse shortlist by 8x8 luma signature L1 distance, then exact 32x32 pixel comparison on top 32
  const coarse = index.entries
    .map((entry) => {
      let l1 = 0;
      for (let k = 0; k < 64; k++) {
        l1 += Math.abs(candSig[k] - entry.lumaSignature[k]);
      }
      return { entry, l1 };
    })
    .sort((a, b) => a.l1 - b.l1)
    .slice(0, 32);

  let best: NearestRealMatch | null = null;
  let lowestSimilarity = Infinity; // we want the MOST similar real sprite (smallest distance)

  for (const { entry } of coarse) {
    let inter = 0;
    let union = 0;
    let changedPixels = 0;
    let colorDiffSum = 0;

    for (let p = 0; p < SPRITE_SIZE * SPRITE_SIZE; p++) {
      const i = p * 4;
      const a1 = candidateRgba[i + 3] >= 128;
      const a2 = entry.rgba[i + 3] >= 128;
      if (!a1 && !a2) continue;
      union++;
      if (a1 && a2) {
        inter++;
        const dr = Math.abs(candidateRgba[i] - entry.rgba[i]);
        const dg = Math.abs(candidateRgba[i + 1] - entry.rgba[i + 1]);
        const db = Math.abs(candidateRgba[i + 2] - entry.rgba[i + 2]);
        const dNorm = (dr + dg + db) / 765;
        colorDiffSum += dNorm;
        if (dr + dg + db > 24) changedPixels++;
      } else {
        changedPixels++;
        colorDiffSum += 1.0;
      }
    }

    const silhouetteIoU = union > 0 ? inter / union : 0;
    const pixelDiffRatio = union > 0 ? changedPixels / union : 1;
    const colorDiffMean = union > 0 ? colorDiffSum / union : 1;

    // Distance metric: 0 = exact identical sprite, 1 = completely different
    const distance = (1 - silhouetteIoU) * 0.45 + colorDiffMean * 0.35 + pixelDiffRatio * 0.2;

    if (distance < lowestSimilarity) {
      lowestSimilarity = distance;
      // Honest originality 0..99.5: identical copy -> 0.0, recolor-only -> ~25-45, composite+diff -> ~45-85
      const originalityScore = Math.min(
        99.5,
        Math.round(Math.max(0, Math.min(1, distance * 1.45)) * 1000) / 10,
      );
      best = {
        spriteId: entry.spriteId,
        category: entry.category,
        split: entry.split,
        rgba: entry.rgba,
        criticScore: entry.criticScore,
        silhouetteIoU: Math.round(silhouetteIoU * 1000) / 1000,
        pixelDiffRatio: Math.round(pixelDiffRatio * 1000) / 1000,
        colorDiffMean: Math.round(colorDiffMean * 1000) / 1000,
        originalityScore,
      };
    }
  }

  if (best) return best;

  return {
    spriteId: 0,
    category: 'Equipamentos',
    split: 'train',
    rgba: new Uint8ClampedArray(SPRITE_SIZE * SPRITE_SIZE * 4),
    criticScore: 0,
    silhouetteIoU: 0,
    pixelDiffRatio: 1,
    colorDiffMean: 1,
    originalityScore: 99.5,
  };
}

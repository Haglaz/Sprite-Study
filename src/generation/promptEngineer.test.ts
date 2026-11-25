import test from 'node:test';
import assert from 'node:assert/strict';
import { calibrateCriticAndBuildDiagnostics } from '../critic/diagnostics';
import { evaluateSprite } from '../critic/evaluator';
import { CLASSIC_MATERIAL_RAMPS } from './executors';
import {
  createMemoryStorageBackend,
  exportFullBackupJson,
  importFullBackupJson,
} from '../db/persistence';
import type { LessonRecord } from '../lessons/types';
import { SPRITE_SIZE } from '../utils/tibiaParser';
import { decomposeSpriteToIndexedAndRampRoles, computeLumaSignature8x8 } from './indexer';
import { generateFromApprovedPrompt } from './pipeline';
import {
  applyLocalAdjustmentToPrompt,
  buildManualPromptDraft,
  buildStyleDossier,
  computePromptDiff,
  validateEngineeredPromptJson,
} from './promptEngineer';
import {
  loadPromptHistory,
  markPromptApprovedVerified,
  savePromptVersionVerified,
} from './promptStore';
import { tibiaifySprite } from './tibiaify';
import type {
  EngineeredPromptSpec,
  IndexedSpriteEntry,
  PaletteRampSpec,
  PromptHistoryRecord,
  SpriteIndexState,
} from './types';

function makeSyntheticClassicSprite(variant: number): Uint8ClampedArray {
  const rgba = new Uint8ClampedArray(SPRITE_SIZE * SPRITE_SIZE * 4);
  const ramp = CLASSIC_MATERIAL_RAMPS[variant % CLASSIC_MATERIAL_RAMPS.length].primary;
  const accent = CLASSIC_MATERIAL_RAMPS[variant % CLASSIC_MATERIAL_RAMPS.length].accent;

  const cx = 15 + (variant % 3) - 1;
  const cy = 15 + ((variant >> 1) % 3) - 1;
  const rx = 8 + (variant % 3);
  const ry = 9 + ((variant >> 1) % 2);

  for (let y = 3; y < SPRITE_SIZE - 3; y++) {
    for (let x = 3; x < SPRITE_SIZE - 3; x++) {
      const nx = (x - cx) / rx;
      const ny = (y - cy) / ry;
      const d = nx * nx + ny * ny;
      if (d > 1.0) continue;

      const idx = (y * SPRITE_SIZE + x) * 4;
      let col: [number, number, number];
      if (d > 0.72) {
        col = ramp[0];
      } else if (Math.abs(x - cx) <= 1 && Math.abs(y - cy) <= 1) {
        col = accent[1];
      } else {
        const nw = (-(nx + ny) + 2) / 4;
        if (nw > 0.74) col = ramp[4];
        else if (nw > 0.56) col = ramp[3];
        else if (nw > 0.38) col = ramp[2];
        else col = ramp[1];
      }
      rgba[idx] = col[0];
      rgba[idx + 1] = col[1];
      rgba[idx + 2] = col[2];
      rgba[idx + 3] = 255;
    }
  }
  return rgba;
}

function makeIndexedFixtureDataset() {
  const trainSamples = [];
  const valSamples = [];
  const entries: IndexedSpriteEntry[] = [];
  const baseFlags = {
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
    pickupable: true,
    movable: true,
    unpassable: false,
    fluidContainer: false,
    splash: false,
    light: false,
    hangable: false,
    equipable: true,
  };

  for (let i = 0; i < 16; i++) {
    const rgba = makeSyntheticClassicSprite(1000 + i);
    trainSamples.push({ spriteId: 100 + i, category: 'Equipamentos' as const, rgba });
  }
  for (let i = 0; i < 8; i++) {
    const rgba = makeSyntheticClassicSprite(2000 + i);
    valSamples.push({ spriteId: 200 + i, category: 'Equipamentos' as const, rgba });
  }

  const { calibration } = calibrateCriticAndBuildDiagnostics({
    datasetKey: 'test-ds:v2',
    trainSamples,
    validationSamples: valSamples,
    splitTotals: { train: 16, validation: 8, test: 6, total: 30 },
    seed: 772,
  });

  for (const s of trainSamples) {
    const dec = decomposeSpriteToIndexedAndRampRoles(s.rgba);
    const ev = evaluateSprite(s.rgba, { category: s.category, calibration });
    entries.push({
      spriteId: s.spriteId,
      itemId: s.spriteId,
      category: s.category,
      split: 'train',
      flags: baseFlags,
      tags: ['equipamentos', 'espada', 'metal', dec.dominantHue],
      paletteHex: dec.paletteHex,
      paletteRgb: dec.paletteRgb,
      indexedGrid: dec.indexedGrid,
      rampRoles: dec.rampRoles,
      silhouetteMask: dec.silhouetteMask,
      lumaSignature: computeLumaSignature8x8(s.rgba),
      rgba: s.rgba,
      criticScore: ev.score,
      isTile: false,
      opaqueCount: dec.opaqueCount,
      bboxW: dec.bboxW,
      bboxH: dec.bboxH,
      dominantHue: dec.dominantHue,
    });
  }

  for (const s of valSamples) {
    const dec = decomposeSpriteToIndexedAndRampRoles(s.rgba);
    const ev = evaluateSprite(s.rgba, { category: s.category, calibration });
    entries.push({
      spriteId: s.spriteId,
      itemId: s.spriteId,
      category: s.category,
      split: 'validation',
      flags: baseFlags,
      tags: ['equipamentos', 'espada', 'metal', dec.dominantHue],
      paletteHex: dec.paletteHex,
      paletteRgb: dec.paletteRgb,
      indexedGrid: dec.indexedGrid,
      rampRoles: dec.rampRoles,
      silhouetteMask: dec.silhouetteMask,
      lumaSignature: computeLumaSignature8x8(s.rgba),
      rgba: s.rgba,
      criticScore: ev.score,
      isTile: false,
      opaqueCount: dec.opaqueCount,
      bboxW: dec.bboxW,
      bboxH: dec.bboxH,
      dominantHue: dec.dominantHue,
    });
  }

  const spriteIndex: SpriteIndexState = {
    datasetKey: 'test-ds:v2',
    indexedAt: '2026-01-01T00:00:00.000Z',
    entries,
    trainCount: trainSamples.length,
    validationCount: valSamples.length,
    excludedTestCount: 6,
  };

  // Simulated held-out test sprite IDs (300..305) — MUST NEVER appear in dossier or allowed donors
  const testOnlyIds = new Set([300, 301, 302, 303, 304, 305]);

  return { calibration, spriteIndex, testOnlyIds };
}

test('tibiaifySprite enforces binary alpha, exact palette snapping, orphan removal, 1px dark outline, and nearest-neighbor downsample', () => {
  // Build a 64x64 sprite with anti-aliased semi-transparent edges, off-palette blended colors, and 2 orphan specks
  const srcW = 64;
  const srcH = 64;
  const dirty = new Uint8ClampedArray(srcW * srcH * 4);

  // Main 28x28 block in 64x64 space (= 14x14 in 32x32) with blended colors + semi-transparent border
  for (let y = 18; y < 46; y++) {
    for (let x = 18; x < 46; x++) {
      const off = (y * srcW + x) * 4;
      const isEdge = x === 18 || x === 45 || y === 18 || y === 45;
      // Deliberately inject blended/off-palette colors and semi-transparent alpha
      dirty[off] = 77 + ((x * 3) % 40);
      dirty[off + 1] = 111 + ((y * 2) % 35);
      dirty[off + 2] = 165;
      dirty[off + 3] = isEdge ? 140 : 255;
    }
  }
  // Soft AA halo pixel (alpha = 64 -> must become 0)
  const haloOff = (16 * srcW + 20) * 4;
  dirty[haloOff] = 100;
  dirty[haloOff + 1] = 120;
  dirty[haloOff + 2] = 180;
  dirty[haloOff + 3] = 64;

  // Detached 2x2 orphan block in 64x64 (= 1x1 orphan in 32x32)
  for (let dy = 0; dy < 2; dy++) {
    for (let dx = 0; dx < 2; dx++) {
      const o = ((4 + dy) * srcW + (4 + dx)) * 4;
      dirty[o] = 200;
      dirty[o + 1] = 220;
      dirty[o + 2] = 250;
      dirty[o + 3] = 255;
    }
  }

  const approvedRamps: PaletteRampSpec[] = [
    {
      material: 'Gelo',
      colors: ['#121828', '#243C66', '#4270B0', '#84B8E8', '#D4EEFF'],
    },
  ];
  const allowedHexSet = new Set(approvedRamps[0].colors);

  const res = tibiaifySprite({
    rgba: dirty,
    width: srcW,
    height: srcH,
    approvedRamps,
    maxColors: 12,
    isTile: false,
    allowDithering: false,
  });

  assert.equal(res.rgba.length, SPRITE_SIZE * SPRITE_SIZE * 4);
  assert.ok(res.removedOrphans >= 1, 'Expected isolated orphan pixel to be removed');
  assert.ok(res.snappedPixels > 0, 'Expected off-palette pixels to be snapped');

  let opaqueCount = 0;
  let semiTransparentCount = 0;
  let outlineDarkCount = 0;
  let perimeterCount = 0;

  for (let y = 0; y < SPRITE_SIZE; y++) {
    for (let x = 0; x < SPRITE_SIZE; x++) {
      const i = (y * SPRITE_SIZE + x) * 4;
      const a = res.rgba[i + 3];
      if (a !== 0 && a !== 255) semiTransparentCount++;
      if (a === 255) {
        opaqueCount++;
        const hex =
          '#' +
          [res.rgba[i], res.rgba[i + 1], res.rgba[i + 2]]
            .map((v) => v.toString(16).padStart(2, '0'))
            .join('')
            .toUpperCase();
        assert.equal(
          allowedHexSet.has(hex),
          true,
          `Pixel (${x},${y}) color ${hex} not snapped to approved ramp`,
        );

        const isEdge =
          x === 0 ||
          x === SPRITE_SIZE - 1 ||
          y === 0 ||
          y === SPRITE_SIZE - 1 ||
          res.rgba[(y * SPRITE_SIZE + x - 1) * 4 + 3] === 0 ||
          res.rgba[(y * SPRITE_SIZE + x + 1) * 4 + 3] === 0 ||
          res.rgba[((y - 1) * SPRITE_SIZE + x) * 4 + 3] === 0 ||
          res.rgba[((y + 1) * SPRITE_SIZE + x) * 4 + 3] === 0;
        if (isEdge) {
          perimeterCount++;
          if (hex === '#121828') outlineDarkCount++;
        }
      }
    }
  }

  assert.equal(semiTransparentCount, 0, 'Zero semi-transparent pixels allowed');
  assert.ok(opaqueCount > 50, 'Main silhouette must remain intact');
  assert.equal(outlineDarkCount, perimeterCount, 'Entire silhouette perimeter must have 1px dark outline');
});

test('buildStyleDossier uses train split only with zero test leakage and ranks active lessons', () => {
  const { calibration, spriteIndex, testOnlyIds } = makeIndexedFixtureDataset();
  const sampleLessons: LessonRecord[] = [
    {
      id: 'lesson:human:outline',
      datasetKey: 'test-ds:v2',
      category: 'Equipamentos',
      scopeKey: 'outline_coverage',
      title: 'Contorno escuro em lâminas',
      source: 'human_feedback',
      status: 'active',
      polarity: 'enforce',
      directive: 'Manter contorno 1px escuro hue-shifted contínuo.',
      validationCheck: 'outlineCoverage >= 85%',
      relatedIssueCodes: ['MISSING_OUTLINE'],
      evidence: {
        sampleSize: 4,
        positiveCount: 4,
        negativeCount: 0,
        criticDeltaMean: 4.2,
        referenceSpriteIds: [100],
        confidence: 0.65,
      },
      conflictGroupId: null,
      supersededById: null,
      createdAt: '2026-01-01T00:00:00.000Z',
      updatedAt: '2026-01-02T00:00:00.000Z',
      readBackVerified: true,
    },
  ];

  const { dossier, donorEntries } = buildStyleDossier({
    datasetKey: 'test-ds:v2',
    category: 'Equipamentos',
    idea: 'espada de gelo',
    calibration,
    spriteIndex,
    lessons: sampleLessons,
    fewShotCount: 8,
  });

  assert.equal(dossier.splitSource, 'train_only');
  assert.ok(dossier.fewShotDonors.length >= 6 && dossier.fewShotDonors.length <= 10);
  assert.ok(dossier.topPaletteHex.length > 0);
  assert.equal(dossier.activeLessons.length, 1);
  assert.equal(dossier.activeLessons[0].source, 'human_feedback');

  // Every few-shot donor MUST be from train split and NEVER from validation or test
  for (const d of dossier.fewShotDonors) {
    assert.equal(d.split, 'train');
    assert.equal(testOnlyIds.has(d.spriteId), false);
    assert.ok(d.spriteId >= 100 && d.spriteId < 200);
  }
  for (const e of donorEntries) {
    assert.equal(e.split, 'train');
  }
  // Allowed donor IDs include train + validation ONLY, NEVER test
  for (const id of dossier.allowedDonorIds) {
    assert.equal(testOnlyIds.has(id), false, `Test sprite #${id} leaked into allowedDonorIds`);
  }
});

test('validateEngineeredPromptJson accepts valid JSON and throws on malformed JSON, bad category, or test-split donor ID', () => {
  const validDonorIds = new Set([100, 101, 102, 103, 200]);

  const validJson = JSON.stringify({
    itemName: 'Espada de Gelo',
    category: 'Equipamentos',
    visualBrief:
      'Espada diagonal clássica 45 graus com lâmina de cristal de gelo azul, guarda de aço escuro, iluminação NW e contorno 1px.',
    paletteRamps: [
      { material: 'Gelo', colors: ['#121A2C', '#2A4E7C', '#5CA0D8', '#B8E4FF'] },
      { material: 'Aço', colors: ['#181C24', '#4B5568', '#98A4B8'] },
    ],
    constraints: {
      footprintPx: { width: 22, height: 24 },
      maxColors: 14,
      outlineRule: 'Contorno 1px escuro contínuo',
      noDitheringOrAA: true,
      binaryAlpha: true,
    },
    negativePrompt: ['blur', 'anti-aliasing', 'pixels órfãos'],
    donorSuggestions: [100, 101, 102],
  });

  const parsed = validateEngineeredPromptJson(validJson, { validDonorIds });
  assert.equal(parsed.itemName, 'Espada de Gelo');
  assert.equal(parsed.paletteRamps.length, 2);
  assert.deepEqual(parsed.donorSuggestions, [100, 101, 102]);

  // Malformed JSON -> throws
  assert.throws(
    () => validateEngineeredPromptJson('{not valid json', { validDonorIds }),
    /malformado/i,
  );

  // Invalid category -> throws
  assert.throws(
    () =>
      validateEngineeredPromptJson(
        JSON.stringify({ ...JSON.parse(validJson), category: 'CategoriaInexistente' }),
        { validDonorIds },
      ),
    /category/i,
  );

  // Test-split donor ID leakage (#305 not in validDonorIds) -> throws
  assert.throws(
    () =>
      validateEngineeredPromptJson(
        JSON.stringify({ ...JSON.parse(validJson), donorSuggestions: [100, 101, 305] }),
        { validDonorIds },
      ),
    /vazamento|splits permitidos/i,
  );

  // Ramp with <3 colors -> throws
  assert.throws(
    () =>
      validateEngineeredPromptJson(
        JSON.stringify({
          ...JSON.parse(validJson),
          paletteRamps: [{ material: 'Gelo', colors: ['#121A2C', '#B8E4FF'] }],
        }),
        { validDonorIds },
      ),
    /entre 3 e 5 cores/i,
  );
});

test('promptHistory persistence (putAndVerify) and backup export/import round-trip preserve prompt versions and diff adjustments', async () => {
  const { calibration, spriteIndex } = makeIndexedFixtureDataset();
  const { dossier, trainPaletteRgb } = buildStyleDossier({
    datasetKey: 'test-ds:v2',
    category: 'Equipamentos',
    idea: 'espada de gelo',
    calibration,
    spriteIndex,
    lessons: [],
  });

  const v1Spec: EngineeredPromptSpec = buildManualPromptDraft({
    idea: 'espada de gelo',
    category: 'Equipamentos',
    dossier,
    trainPaletteRgb,
  });

  const v2Spec = applyLocalAdjustmentToPrompt({
    previous: v1Spec,
    comment: 'contorno mais escuro e menos cores',
  });
  const diff = computePromptDiff(v1Spec, v2Spec);
  assert.ok(diff.length >= 2, 'Adjustment must produce concrete prompt diff changes');
  assert.ok(v2Spec.constraints.maxColors < v1Spec.constraints.maxColors);

  const db1 = createMemoryStorageBackend();
  const recV1: PromptHistoryRecord = {
    id: 'req-1-v1',
    datasetKey: 'test-ds:v2',
    requestId: 'req-1',
    version: 1,
    idea: 'espada de gelo',
    parentVersionId: null,
    adjustmentComment: null,
    source: 'manual',
    model: null,
    engineeredPrompt: v1Spec,
    approved: false,
    approvedAt: null,
    createdAt: '2026-01-01T10:00:00.000Z',
    readBackVerified: false,
  };
  const savedV1 = await savePromptVersionVerified(db1, recV1);
  assert.equal(savedV1.readBackVerified, true);

  const recV2: PromptHistoryRecord = {
    ...recV1,
    id: 'req-1-v2',
    version: 2,
    parentVersionId: 'req-1-v1',
    adjustmentComment: 'contorno mais escuro e menos cores',
    engineeredPrompt: v2Spec,
    createdAt: '2026-01-01T10:05:00.000Z',
  };
  const approvedV2 = await markPromptApprovedVerified(db1, recV2);
  assert.equal(approvedV2.approved, true);
  assert.ok(approvedV2.approvedAt);

  // Full backup export/import round-trip into fresh backend
  const backupJson = await exportFullBackupJson(db1);
  const db2 = createMemoryStorageBackend();
  const importRes = await importFullBackupJson(db2, backupJson);
  assert.equal(importRes.verified, true);
  assert.equal(importRes.importedCounts.promptHistory, 2);

  const restored = await loadPromptHistory(db2, 'test-ds:v2');
  assert.equal(restored.length, 2);
  assert.equal(restored[0].id, 'req-1-v2');
  assert.equal(restored[0].approved, true);
  assert.equal(restored[1].id, 'req-1-v1');
});

test('Step 2 generateFromApprovedPrompt produces 8 tibiaified 32x32 candidates meeting validation hard-rule quality', async () => {
  const { calibration, spriteIndex } = makeIndexedFixtureDataset();
  const { dossier, trainPaletteRgb } = buildStyleDossier({
    datasetKey: 'test-ds:v2',
    category: 'Equipamentos',
    idea: 'espada de gelo',
    calibration,
    spriteIndex,
    lessons: [],
  });

  const approvedPrompt = buildManualPromptDraft({
    idea: 'espada de gelo',
    category: 'Equipamentos',
    dossier,
    trainPaletteRgb,
  });

  const result = await generateFromApprovedPrompt({
    datasetKey: 'test-ds:v2',
    approvedPrompt,
    engineeredPromptId: 'req-test-v1',
    engineeredPromptVersion: 1,
    promptSource: 'manual',
    candidateCount: 8,
    useGemini: false,
    seed: 7722026,
    index: spriteIndex,
    calibration,
  });

  assert.equal(result.allCandidates.length, 8);
  assert.ok(result.topCandidates.length >= 1);
  assert.ok(result.categoryValidationRealMedian !== null);

  for (const cand of result.allCandidates) {
    assert.equal(cand.rgba.length, SPRITE_SIZE * SPRITE_SIZE * 4);
    assert.equal(cand.evaluation.features.semiTransparentPixels, 0, 'Must have strict binary alpha');
    assert.equal(cand.provenance.tibiaified, true);
    assert.equal(cand.provenance.engineeredPromptId, 'req-test-v1');
    assert.equal(cand.provenance.engineeredPromptVersion, 1);
    assert.ok(cand.provenance.approvedPromptSnapshot);
  }

  const topHardIssues = result.topCandidates[0].evaluation.issues.filter((i) => i.severity === 'hard');
  assert.equal(topHardIssues.length, 0, 'Top candidate must pass all Critic v1 hard rules');
  assert.ok(
    result.generatedHardPassRate >= (result.categoryValidationHardPassRate ?? 0),
    `Expected generated hard-rule pass rate (${result.generatedHardPassRate}%) >= validation real (${result.categoryValidationHardPassRate}%)`,
  );
});

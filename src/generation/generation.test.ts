import assert from 'node:assert/strict';
import test from 'node:test';
import { calibrateCriticFromArchive } from '../critic/diagnostics';
import { evaluateSprite } from '../critic/evaluator';
import { createMemoryStorageBackend } from '../db/persistence';
import { createStratifiedDatasetSplit } from '../db/split';
import {
  buildTrainingDatasetExportZip,
  encodeRgbaToPngBytes,
} from '../export/trainingDatasetExport';
import type {
  ParsedDatArchive,
  ParsedItemMetadata,
  ParsedSprArchive,
} from '../types/studio';
import { parseTibiaSpr, SPRITE_SIZE } from '../utils/tibiaParser';
import {
  buildDeterministicDiffEdits,
  CLASSIC_MATERIAL_RAMPS,
  executeBoundedPixelDiff,
  executeDonorComposite,
  executeRampRecolor,
} from './executors';
import {
  decomposeSpriteToIndexedAndRampRoles,
  findNearestRealSprite,
  retrieveDonorsForPrompt,
  runSpriteIndexingJob,
} from './indexer';
import { generateSpriteCandidates, runCriticGuidedRepair } from './pipeline';
import {
  loadApprovals,
  loadGeneratedSprites,
  rebuildHumanPreferenceModel,
  recordApprovalLabelVerified,
  saveGeneratedSpriteVerified,
} from './store';
import type { PixelDiffEdit } from './types';

function makeSyntheticRealSprite(variant: number): Uint8ClampedArray {
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
        col = ramp[0]; // crisp dark outline
      } else if (Math.abs(x - cx) <= 1 && Math.abs(y - cy) <= 1) {
        col = accent[1]; // central accent gem
      } else {
        // NW -> SE directional shading
        const nw = (-(nx + ny) + 2) / 4; // 0..1
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

function buildSyntheticFixtureArchive(count = 36): {
  dat: ParsedDatArchive;
  spr: ParsedSprArchive;
  split: ReturnType<typeof createStratifiedDatasetSplit>;
  calibration: ReturnType<typeof calibrateCriticFromArchive>['calibration'];
} {
  // Encode synthetic sprites into a real Tibia .spr binary buffer so parse/decode works identically
  const spritesRgba: Uint8ClampedArray[] = [];
  for (let i = 0; i < count; i++) {
    spritesRgba.push(makeSyntheticRealSprite(i));
  }

  // Build valid uncompressed-run .spr binary (extended=false, header=8 bytes, table=count*4)
  const chunks: Uint8Array[] = [];
  const offsets: number[] = [];
  let cursor = 8 + count * 4;

  for (let s = 0; s < count; s++) {
    const rgba = spritesRgba[s];
    const rle: number[] = [];
    let p = 0;
    while (p < SPRITE_SIZE * SPRITE_SIZE) {
      let trans = 0;
      while (p < SPRITE_SIZE * SPRITE_SIZE && rgba[p * 4 + 3] === 0) {
        trans++;
        p++;
      }
      let colStart = p;
      let colCount = 0;
      while (p < SPRITE_SIZE * SPRITE_SIZE && rgba[p * 4 + 3] > 0) {
        colCount++;
        p++;
      }
      rle.push(trans & 0xff, (trans >>> 8) & 0xff);
      rle.push(colCount & 0xff, (colCount >>> 8) & 0xff);
      for (let k = 0; k < colCount; k++) {
        const idx = (colStart + k) * 4;
        rle.push(rgba[idx], rgba[idx + 1], rgba[idx + 2]);
      }
    }
    const body = new Uint8Array(5 + rle.length);
    body[0] = 255;
    body[1] = 0;
    body[2] = 255;
    body[3] = rle.length & 0xff;
    body[4] = (rle.length >>> 8) & 0xff;
    body.set(rle, 5);
    offsets.push(cursor);
    chunks.push(body);
    cursor += body.length;
  }

  const rawSpr = new Uint8Array(cursor);
  const dv = new DataView(rawSpr.buffer);
  dv.setUint32(0, 0x439d5a33, true); // Tibia 7.72 signature
  dv.setUint16(4, count, true);
  for (let i = 0; i < count; i++) {
    dv.setUint32(6 + i * 4, offsets[i], true);
  }
  // Adjust offsets to match header=6 bytes for non-extended .spr
  let writePos = 6 + count * 4;
  for (let i = 0; i < count; i++) {
    dv.setUint32(6 + i * 4, writePos, true);
    rawSpr.set(chunks[i], writePos);
    writePos += chunks[i].length;
  }

  const finalBuf = rawSpr.slice(0, writePos);
  const spr = parseTibiaSpr(finalBuf.buffer);

  const items: ParsedItemMetadata[] = Array.from({ length: count }, (_, idx) => ({
    id: 100 + idx,
    category: idx < 24 ? 'Equipamentos' : 'Consumíveis',
    width: 1,
    height: 1,
    layers: 1,
    xdiv: 1,
    ydiv: 1,
    zdiv: 1,
    animCount: 1,
    spriteIds: [idx + 1],
    primarySpriteId: idx + 1,
    flags: {
      ground: false,
      groundBorder: false,
      onBottom: false,
      onTop: false,
      container: false,
      stackable: idx >= 24,
      forceUse: false,
      multiUse: false,
      usable: idx >= 24,
      writable: false,
      pickupable: true,
      movable: true,
      unpassable: false,
      fluidContainer: idx >= 24,
      splash: false,
      light: false,
      hangable: false,
      equipable: idx < 24,
    },
  }));

  const dat: ParsedDatArchive = {
    signature: 0x439d5a33,
    signatureHex: '0x439D5A33',
    itemCount: count,
    outfitCount: 0,
    effectCount: 0,
    missileCount: 0,
    versionHint: 'Tibia 7.72 (fixture)',
    items,
    categoryCounts: {
      Equipamentos: 24,
      Consumíveis: count - 24,
      Containers: 0,
      Decoração: 0,
    },
  };

  const split = createStratifiedDatasetSplit({
    datasetKey: 'fixture-key',
    candidates: items.map((it) => ({ spriteId: it.spriteIds[0], category: it.category })),
    seed: 20260772,
  });

  const { calibration } = calibrateCriticFromArchive({
    datasetKey: 'fixture-key',
    dat,
    spr,
    split,
    seed: 20260772,
  });

  return { dat, spr, split, calibration };
}

function assertBinaryAlphaOnly(rgba: Uint8ClampedArray): void {
  for (let i = 3; i < rgba.length; i += 4) {
    assert.ok(
      rgba[i] === 0 || rgba[i] === 255,
      `Expected binary alpha (0 or 255), got ${rgba[i]} at byte ${i}`,
    );
  }
}

function measureOutlineDarkRatio(rgba: Uint8ClampedArray): number {
  let edgeCount = 0;
  let darkEdgeCount = 0;
  for (let y = 0; y < SPRITE_SIZE; y++) {
    for (let x = 0; x < SPRITE_SIZE; x++) {
      const i = (y * SPRITE_SIZE + x) * 4;
      if (rgba[i + 3] === 0) continue;
      const isEdge =
        x === 0 ||
        x === SPRITE_SIZE - 1 ||
        y === 0 ||
        y === SPRITE_SIZE - 1 ||
        rgba[(y * SPRITE_SIZE + (x - 1)) * 4 + 3] === 0 ||
        rgba[(y * SPRITE_SIZE + (x + 1)) * 4 + 3] === 0 ||
        rgba[((y - 1) * SPRITE_SIZE + x) * 4 + 3] === 0 ||
        rgba[((y + 1) * SPRITE_SIZE + x) * 4 + 3] === 0;
      if (!isEdge) continue;
      edgeCount++;
      const luma = (0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2]) / 255;
      if (luma <= 0.34) darkEdgeCount++;
    }
  }
  return edgeCount > 0 ? darkEdgeCount / edgeCount : 0;
}

test('T1 & T2 executors preserve binary alpha (0|255) and dark outer outline', () => {
  const srcA = makeSyntheticRealSprite(0);
  const srcB = makeSyntheticRealSprite(2);

  // Inject deliberate semi-transparent halo pixels into srcA to verify executor snaps them strictly
  srcA[(4 * SPRITE_SIZE + 4) * 4 + 3] = 85;
  srcA[(5 * SPRITE_SIZE + 5) * 4 + 3] = 190;

  const ramp = CLASSIC_MATERIAL_RAMPS[1]; // crimson_ruby
  const t1 = executeRampRecolor({
    sourceRgba: srcA,
    primaryStops: ramp.primary,
    accentStops: ramp.accent,
    isTile: false,
  });

  assertBinaryAlphaOnly(t1.rgba);
  assert.ok(
    measureOutlineDarkRatio(t1.rgba) >= 0.9,
    `Expected T1 outline dark ratio >= 0.90, got ${measureOutlineDarkRatio(t1.rgba)}`,
  );

  const t2 = executeDonorComposite({
    primaryRgba: srcA,
    secondaryRgba: srcB,
    primaryStops: CLASSIC_MATERIAL_RAMPS[2].primary,
    accentStops: CLASSIC_MATERIAL_RAMPS[2].accent,
    spliceMode: 'center_core',
    isTile: false,
  });

  assertBinaryAlphaOnly(t2);
  assert.ok(
    measureOutlineDarkRatio(t2) >= 0.9,
    `Expected T2 outline dark ratio >= 0.90, got ${measureOutlineDarkRatio(t2)}`,
  );
});

test('T1 executeRampRecolor strictly preserves monotonic luminance ordering', () => {
  const src = makeSyntheticRealSprite(3);
  // Deliberately pass unsorted target stops to verify executor sorts & preserves strict luma ordering
  const scrambledStops: Array<[number, number, number]> = [
    [248, 230, 134],
    [34, 22, 14],
    [148, 102, 34],
    [208, 162, 56],
    [88, 54, 22],
  ];

  const res = executeRampRecolor({
    sourceRgba: src,
    primaryStops: scrambledStops,
    accentStops: CLASSIC_MATERIAL_RAMPS[3].accent,
    isTile: false,
  });

  assert.ok(res.sourcePaletteLumas.length >= 4);
  assert.equal(res.mappedPaletteLumas.length, res.sourcePaletteLumas.length);

  for (let i = 1; i < res.mappedPaletteLumas.length; i++) {
    assert.ok(
      res.mappedPaletteLumas[i] >= res.mappedPaletteLumas[i - 1] - 1e-6,
      `Monotonic luma violated at index ${i}: ${res.mappedPaletteLumas[i - 1]} -> ${res.mappedPaletteLumas[i]}`,
    );
  }
});

test('executeBoundedPixelDiff strictly respects maxDiffPixels, binary alpha, and protected outline', () => {
  const base = makeSyntheticRealSprite(0);
  const maxDiffPixels = 16;

  // Build 200 aggressive edits including semi-transparent alpha & attempts to erase the outer outline
  const edits: PixelDiffEdit[] = [];
  for (let y = 4; y < 24; y++) {
    for (let x = 4; x < 14; x++) {
      edits.push({
        x,
        y,
        r: 250,
        g: 220,
        b: 90,
        // Mix of valid 255, zero 0, and illegal semi-transparent 120
        a: ((x + y) % 3 === 0 ? 0 : 255) as 0 | 255,
      });
    }
  }
  // Force a sneaky non-binary alpha via cast to verify executor snaps to 0|255
  edits[0] = { x: 15, y: 15, r: 240, g: 210, b: 80, a: 140 as unknown as 255 };

  const diffRes = executeBoundedPixelDiff({
    baseRgba: base,
    edits,
    maxDiffPixels,
    preserveOutline: true,
    isTile: false,
  });

  assert.ok(
    diffRes.appliedCount <= maxDiffPixels,
    `Applied diff count (${diffRes.appliedCount}) exceeded maxDiffPixels (${maxDiffPixels})`,
  );

  // Independently recount exact pixel differences between base and diffRes.rgba
  let actualChanged = 0;
  for (let p = 0; p < SPRITE_SIZE * SPRITE_SIZE; p++) {
    const i = p * 4;
    if (
      base[i] !== diffRes.rgba[i] ||
      base[i + 1] !== diffRes.rgba[i + 1] ||
      base[i + 2] !== diffRes.rgba[i + 2] ||
      base[i + 3] !== diffRes.rgba[i + 3]
    ) {
      actualChanged++;
    }
  }
  assert.equal(actualChanged, diffRes.appliedCount);
  assert.ok(actualChanged <= maxDiffPixels);
  assertBinaryAlphaOnly(diffRes.rgba);
  assert.ok(measureOutlineDarkRatio(diffRes.rgba) >= 0.9);

  // Deterministic diff builder also stays within budget
  const autoEdits = buildDeterministicDiffEdits({
    baseRgba: base,
    accentStops: CLASSIC_MATERIAL_RAMPS[0].accent,
    highlightStop: CLASSIC_MATERIAL_RAMPS[0].primary[4],
    maxDiffPixels: 10,
    seed: 772,
  });
  assert.ok(autoEdits.length <= 10);
});

test('indexing excludes test split, retrieval uses train only, originality=0 on copy, and repair never degrades', async () => {
  const { dat, spr, split, calibration } = buildSyntheticFixtureArchive(36);
  const index = runSpriteIndexingJob({
    datasetKey: 'fixture-key',
    dat,
    spr,
    split,
    calibration,
  });

  // 1) Test split must be strictly excluded from indexed entries
  const testIds = new Set<number>([
    ...split.byCategory.Equipamentos.test,
    ...split.byCategory['Consumíveis'].test,
  ]);
  assert.ok(testIds.size > 0, 'Fixture split should have non-empty test split');
  assert.equal(index.excludedTestCount, testIds.size);
  for (const entry of index.entries) {
    assert.ok(!testIds.has(entry.spriteId), `Test sprite ${entry.spriteId} leaked into index!`);
    assert.ok(entry.split === 'train' || entry.split === 'validation');
  }

  // 2) Retrieval must return ONLY train-split donors
  const donors = retrieveDonorsForPrompt({
    index,
    category: 'Equipamentos',
    prompt: 'Armadura azul real com detalhes dourados',
    topK: 6,
  });
  assert.ok(donors.length > 0);
  for (const d of donors) {
    assert.equal(d.entry.split, 'train');
  }

  // 3) Originality score on exact copy must be 0.0; recolored/diffed sprite must be > 15
  const exactMatch = findNearestRealSprite(donors[0].entry.rgba, index);
  assert.equal(exactMatch.spriteId, donors[0].entry.spriteId);
  assert.equal(exactMatch.originalityScore, 0);

  // 4) Full pipeline T1 & T2 run (local mode)
  const genRes = await generateSpriteCandidates({
    datasetKey: 'fixture-key',
    prompt: 'Escudo rubi de fogo forjado',
    category: 'Equipamentos',
    tier: 'T2',
    useGemini: false,
    maxDiffPixels: 24,
    candidateCount: 6,
    topKReturn: 4,
    seed: 12345,
    index,
    calibration,
  });

  assert.equal(genRes.candidates.length, 4);
  for (const c of genRes.candidates) {
    assert.equal(c.provenance.source, 'local-t2');
    assert.ok(c.provenance.diffPixelCount <= 24);
    assert.ok(c.nearestReal.originalityScore > 10);
    assertBinaryAlphaOnly(c.rgba);
  }

  // 5) Critic-guided repair never degrades score
  const top = genRes.candidates[0];
  const rep = runCriticGuidedRepair({
    rgba: top.rgba,
    baseEval: top.evaluation,
    category: 'Equipamentos',
    calibration,
  });
  assert.ok(rep.evaluation.score >= top.evaluation.score);
});

test('Exportar dataset de treino builds valid RGBA PNGs, grids, palettes, ramp roles, labels.csv, and excludes test split', () => {
  const { dat, spr, split, calibration } = buildSyntheticFixtureArchive(36);
  const index = runSpriteIndexingJob({
    datasetKey: 'fixture-key',
    dat,
    spr,
    split,
    calibration,
  });

  // Verify PNG encoder directly produces valid 32x32 RGBA PNG (colorType=6)
  const samplePng = encodeRgbaToPngBytes(index.entries[0].rgba, SPRITE_SIZE, SPRITE_SIZE);
  assert.deepEqual(
    Array.from(samplePng.subarray(0, 8)),
    [137, 80, 78, 71, 13, 10, 26, 10],
    'Invalid PNG signature',
  );
  const pngView = new DataView(samplePng.buffer, samplePng.byteOffset, samplePng.byteLength);
  assert.equal(pngView.getUint32(16, false), 32, 'PNG IHDR width must be 32');
  assert.equal(pngView.getUint32(20, false), 32, 'PNG IHDR height must be 32');
  assert.equal(samplePng[24], 8, 'PNG bit depth must be 8');
  assert.equal(samplePng[25], 6, 'PNG color type must be 6 (true RGBA alpha)');

  const exported = buildTrainingDatasetExportZip({
    index,
    split,
    exportedAt: '2026-11-25T00:00:00.000Z',
  });

  // Valid ZIP magic bytes (PK\x03\x04)
  assert.equal(exported.zipBytes[0], 0x50);
  assert.equal(exported.zipBytes[1], 0x4b);
  assert.equal(exported.zipBytes[2], 0x03);
  assert.equal(exported.zipBytes[3], 0x04);

  // Must include PNGs, grids, palettes, ramp_roles, labels.csv, split.json, manifest.json
  assert.ok(exported.filePaths.includes('labels.csv'));
  assert.ok(exported.filePaths.includes('split.json'));
  assert.ok(exported.filePaths.includes('manifest.json'));
  assert.ok(exported.filePaths.some((p) => p.startsWith('pngs/train/spr_') && p.endsWith('.png')));
  assert.ok(exported.filePaths.some((p) => p.startsWith('grids/spr_') && p.endsWith('.json')));
  assert.ok(exported.filePaths.some((p) => p.startsWith('palettes/spr_') && p.endsWith('.json')));
  assert.ok(exported.filePaths.some((p) => p.startsWith('ramp_roles/spr_') && p.endsWith('.json')));

  // labels.csv header and strict exclusion of test split
  const lines = exported.labelsCsv.trim().split('\n');
  assert.ok(lines[0].startsWith('id,item_id,category,flags,tags,split,'));
  assert.equal(lines.length - 1, exported.exportedSpriteCount);
  assert.ok(!exported.labelsCsv.includes(',test,'), 'Test split row leaked into labels.csv!');

  const testIds = new Set<number>([
    ...split.byCategory.Equipamentos.test,
    ...split.byCategory['Consumíveis'].test,
  ]);
  for (const tid of testIds) {
    assert.ok(
      !exported.filePaths.some((p) => p.includes(`spr_${tid}.`)),
      `Test sprite spr_${tid} leaked into export ZIP!`,
    );
  }
});

test('generatedSprites & approvals enforce read-back verification and rebuild HumanPreferenceModel at >= 6 votes', async () => {
  const { dat, spr, split, calibration } = buildSyntheticFixtureArchive(24);
  const index = runSpriteIndexingJob({
    datasetKey: 'fixture-key',
    dat,
    spr,
    split,
    calibration,
  });

  const backend = createMemoryStorageBackend();
  const baseSpr = index.entries[0];
  const dec = decomposeSpriteToIndexedAndRampRoles(baseSpr.rgba);
  const ev = evaluateSprite(baseSpr.rgba, { category: baseSpr.category, calibration });

  for (let k = 0; k < 6; k++) {
    const saved = await saveGeneratedSpriteVerified(backend, {
      id: `gen-test-${k}`,
      datasetKey: 'fixture-key',
      prompt: `Prompt teste ${k}`,
      category: 'Equipamentos',
      createdAt: new Date(Date.now() + k * 1000).toISOString(),
      rgba: baseSpr.rgba,
      paletteHex: dec.paletteHex,
      evaluation: ev,
      nearestReal: findNearestRealSprite(baseSpr.rgba, index),
      provenance: {
        source: 'local-t1',
        tier: 'T1',
        model: null,
        fallbackWarning: null,
        donorSpriteIds: [baseSpr.spriteId],
        diffPixelCount: 8,
        diffPixelLimit: 24,
        repaired: false,
        repairDelta: 0,
        repairMethod: 'none',
        appliedLessonIds: [],
        seed: k + 1,
      },
      approvalStatus: 'pending',
      readBackVerified: false,
    });
    assert.equal(saved.readBackVerified, true);

    const voteRes = await recordApprovalLabelVerified(backend, {
      sprite: saved,
      decision: k < 3 ? 'approved' : 'rejected',
    });
    assert.equal(voteRes.approval.readBackVerified, true);
    assert.equal(voteRes.updatedSprite.approvalStatus, k < 3 ? 'approved' : 'rejected');

    if (k < 5) {
      assert.equal(voteRes.humanPreferences, null);
    } else {
      assert.ok(voteRes.humanPreferences !== null);
      assert.equal(voteRes.humanPreferences.approvedCount, 3);
      assert.equal(voteRes.humanPreferences.rejectedCount, 3);
      assert.equal(voteRes.humanPreferences.approvedMean.length, 18);
    }
  }

  const reloadedSprites = await loadGeneratedSprites(backend, 'fixture-key');
  const reloadedApprovals = await loadApprovals(backend, 'fixture-key');
  assert.equal(reloadedSprites.length, 6);
  assert.equal(reloadedApprovals.length, 6);
  assert.ok(rebuildHumanPreferenceModel(reloadedApprovals) !== null);
});

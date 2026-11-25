import test from 'node:test';
import assert from 'node:assert/strict';
import * as ort from 'onnxruntime-web';
import { calibrateCriticFromArchive } from '../critic/diagnostics';
import { createMemoryStorageBackend } from '../db/persistence';
import { createStratifiedDatasetSplit, DEFAULT_SPLIT_SEED } from '../db/split';
import { buildTrainingDatasetExportZip } from '../export/trainingDatasetExport';
import { runSpriteIndexingJob } from '../generation/indexer';
import type {
  ApprovalLabelRecord,
  GeneratedSpriteRecord,
} from '../generation/types';
import type { ItemCategory, ParsedDatArchive, ParsedSprArchive } from '../types/studio';
import { parseTibiaSpr, SPRITE_SIZE } from '../utils/tibiaParser';
import { buildT3ColabNotebook } from './colabNotebook';
import {
  buildRealOnnxGeneratorBytes,
  buildRealOnnxRerankerBytes,
  buildSampleT3ModelCard,
} from './onnxTestFixture';
import {
  checkT3HardConstraints,
  runT3SamplingAndRanking,
  type T3InferenceRunner,
} from './sampler';
import {
  importT3ModelPackageVerified,
  loadT3ModelPackage,
  loadT3Vocabulary,
  parseAndValidateT3ModelCard,
  saveT3VocabularyVerified,
} from './store';
import { evaluateT3OnTestSplit } from './testEvaluation';
import {
  applyVocabularyRoleCorrection,
  buildRampRoleVocabularyFromTrainSplit,
  computeSplitHash,
  decodeVocabGridToRgba,
  encodeRgbaToVocabGrid,
} from './vocabulary';

function encodeRawSpriteBody(rgba: Uint8ClampedArray): Uint8Array {
  const rle: number[] = [];
  let p = 0;
  while (p < SPRITE_SIZE * SPRITE_SIZE) {
    let trans = 0;
    while (p < SPRITE_SIZE * SPRITE_SIZE && rgba[p * 4 + 3] < 128) {
      trans++;
      p++;
    }
    const colStart = p;
    let colored = 0;
    while (p < SPRITE_SIZE * SPRITE_SIZE && rgba[p * 4 + 3] >= 128) {
      colored++;
      p++;
    }
    rle.push(trans & 0xff, (trans >>> 8) & 0xff);
    rle.push(colored & 0xff, (colored >>> 8) & 0xff);
    for (let k = 0; k < colored; k++) {
      const i = (colStart + k) * 4;
      rle.push(rgba[i], rgba[i + 1], rgba[i + 2]);
    }
  }
  const body = new Uint8Array(5 + rle.length);
  body[0] = 255;
  body[1] = 0;
  body[2] = 255;
  body[3] = rle.length & 0xff;
  body[4] = (rle.length >>> 8) & 0xff;
  body.set(rle, 5);
  return body;
}

function makeCleanSpriteRgba(variant: 'gold_armor' | 'blue_sword' | 'ruby_shield' | 'green_potion'): Uint8ClampedArray {
  const rgba = new Uint8ClampedArray(SPRITE_SIZE * SPRITE_SIZE * 4);
  const palettes: Record<
    typeof variant,
    [[number, number, number], [number, number, number], [number, number, number], [number, number, number]]
  > = {
    gold_armor: [[26, 18, 12], [92, 58, 22], [164, 114, 40], [238, 206, 108]],
    blue_sword: [[18, 22, 34], [42, 68, 122], [86, 134, 198], [186, 224, 248]],
    ruby_shield: [[34, 14, 18], [102, 26, 36], [168, 48, 52], [236, 118, 92]],
    green_potion: [[16, 28, 22], [32, 76, 52], [60, 136, 78], [148, 214, 126]],
  };
  const pal = palettes[variant];
  for (let y = 7; y <= 24; y++) {
    for (let x = 7; x <= 24; x++) {
      const distEdge = Math.min(x - 7, 24 - x, y - 7, 24 - y);
      let c = pal[2];
      if (distEdge === 0) c = pal[0];
      else if (x + y < 24) c = pal[3];
      else if (x + y > 38) c = pal[1];
      const i = (y * SPRITE_SIZE + x) * 4;
      rgba[i] = c[0];
      rgba[i + 1] = c[1];
      rgba[i + 2] = c[2];
      rgba[i + 3] = 255;
    }
  }
  return rgba;
}

function makeBrokenNoOutlineRgba(): Uint8ClampedArray {
  const rgba = new Uint8ClampedArray(SPRITE_SIZE * SPRITE_SIZE * 4);
  // Bright yellow rectangle with NO dark outline and fragmented island
  for (let y = 8; y <= 20; y++) {
    for (let x = 8; x <= 20; x++) {
      const i = (y * SPRITE_SIZE + x) * 4;
      rgba[i] = 230;
      rgba[i + 1] = 210;
      rgba[i + 2] = 120;
      rgba[i + 3] = 255;
    }
  }
  for (let y = 24; y <= 28; y++) {
    for (let x = 24; x <= 28; x++) {
      const i = (y * SPRITE_SIZE + x) * 4;
      rgba[i] = 220;
      rgba[i + 1] = 190;
      rgba[i + 2] = 90;
      rgba[i + 3] = 255;
    }
  }
  return rgba;
}

function makeSyntheticDataset(totalPerCategory = 20) {
  const categories: ItemCategory[] = [
    'Equipamentos',
    'Consumíveis',
    'Containers',
    'Decoração',
  ];
  const variants: Array<'gold_armor' | 'blue_sword' | 'ruby_shield' | 'green_potion'> = [
    'gold_armor',
    'blue_sword',
    'ruby_shield',
    'green_potion',
  ];

  const chunks: Uint8Array[] = [];
  const rawRgbas = new Map<number, Uint8ClampedArray>();
  const items: ParsedDatArchive['items'] = [];
  const candidates: Array<{ spriteId: number; category: ItemCategory }> = [];
  const categoryCounts: Record<ItemCategory, number> = {
    Equipamentos: 0,
    Consumíveis: 0,
    Containers: 0,
    Decoração: 0,
  };

  let sid = 1;
  let itemId = 100;
  for (let c = 0; c < categories.length; c++) {
    const cat = categories[c];
    for (let i = 0; i < totalPerCategory; i++) {
      const rgba = makeCleanSpriteRgba(variants[(c + i) % variants.length]);
      rawRgbas.set(sid, rgba);
      chunks.push(encodeRawSpriteBody(rgba));
      categoryCounts[cat] += 1;
      items.push({
        id: itemId++,
        category: cat,
        width: 1,
        height: 1,
        layers: 1,
        xdiv: 1,
        ydiv: 1,
        zdiv: 1,
        animCount: 1,
        spriteIds: [sid],
        primarySpriteId: sid,
        flags: {
          ground: false,
          groundBorder: false,
          onBottom: false,
          onTop: false,
          container: cat === 'Containers',
          stackable: cat === 'Consumíveis',
          forceUse: false,
          multiUse: false,
          usable: cat === 'Consumíveis',
          writable: false,
          pickupable: true,
          movable: true,
          unpassable: false,
          fluidContainer: cat === 'Consumíveis',
          splash: false,
          light: false,
          hangable: false,
          equipable: cat === 'Equipamentos',
          marketName: `${cat} item ${i}`,
        },
      });
      candidates.push({ spriteId: sid, category: cat });
      sid++;
    }
  }

  const totalSprites = chunks.length;
  const totalBytes =
    6 +
    totalSprites * 4 +
    chunks.reduce((acc, ch) => acc + ch.byteLength, 0);
  const rawSpr = new Uint8Array(totalBytes);
  const dv = new DataView(rawSpr.buffer);
  dv.setUint32(0, 0x439d5a33, true);
  dv.setUint16(4, totalSprites, true);

  let writePos = 6 + totalSprites * 4;
  for (let i = 0; i < totalSprites; i++) {
    dv.setUint32(6 + i * 4, writePos, true);
    rawSpr.set(chunks[i], writePos);
    writePos += chunks[i].length;
  }

  const spr = parseTibiaSpr(rawSpr.buffer);

  const dat: ParsedDatArchive = {
    signature: 0x439d5a33,
    signatureHex: '0x439d5a33',
    versionHint: 'Tibia 7.72 (Test)',
    itemCount: items.length,
    outfitCount: 0,
    effectCount: 0,
    missileCount: 0,
    items,
    categoryCounts,
  };

  const split = createStratifiedDatasetSplit({
    datasetKey: 't3test:dataset',
    candidates,
    seed: DEFAULT_SPLIT_SEED,
  });

  return { dat, spr, split, rawRgbas };
}

test('Part A: Ramp-role vocabulary clusters train palettes per material, allows user role correction, and persists with read-back verification', async () => {
  const { dat, spr, split } = makeSyntheticDataset(16);
  const vocab = buildRampRoleVocabularyFromTrainSplit({
    datasetKey: split.datasetKey,
    dat,
    spr,
    split,
    maxPaletteTokens: 32,
  });

  assert.equal(vocab.trainSpriteCount, split.totals.train);
  assert.ok(vocab.entries.length >= 8);
  assert.equal(vocab.vocabSize, vocab.entries.length + 1);
  assert.ok(vocab.materialSummaries.length >= 2);

  // Roles span outline -> highlight
  const roles = new Set(vocab.entries.map((e) => e.role));
  assert.ok(roles.has('outline'));

  // Correct a mis-assigned role & verify IndexedDB read-back
  const targetEntry = vocab.entries[vocab.entries.length - 1];
  const nextRole = targetEntry.role === 'highlight' ? 'light' : 'highlight';
  const corrected = applyVocabularyRoleCorrection(vocab, targetEntry.tokenId, nextRole);
  assert.equal(corrected.overrideCount, 1);

  const backend = createMemoryStorageBackend();
  const saved = await saveT3VocabularyVerified(backend, corrected);
  assert.equal(saved.readBackVerified, true);
  assert.equal(saved.overrideCount, 1);

  const reloaded = await loadT3Vocabulary(backend, split.datasetKey);
  assert.ok(reloaded);
  assert.equal(
    reloaded.entries.find((e) => e.tokenId === targetEntry.tokenId)?.role,
    nextRole,
  );

  // Round-trip encode/decode preserves binary alpha (0 | 255)
  const cleanRgba = makeCleanSpriteRgba('gold_armor');
  const grid = encodeRgbaToVocabGrid(cleanRgba, reloaded.entries);
  const decoded = decodeVocabGridToRgba(grid, reloaded.entries);
  for (let p = 0; p < SPRITE_SIZE * SPRITE_SIZE; p++) {
    const a = decoded[p * 4 + 3];
    assert.ok(a === 0 || a === 255);
  }
});

test('Parts B & C: Colab notebook generator enforces NW-safe augmentations, strict train/val split, Drive checkpoints, reranker AUC, and valid JSON model card', () => {
  const { dat, spr, split } = makeSyntheticDataset(16);
  const vocab = buildRampRoleVocabularyFromTrainSplit({
    datasetKey: split.datasetKey,
    dat,
    spr,
    split,
  });

  const nb = buildT3ColabNotebook({
    vocabulary: vocab,
    exportedHumanLabelsCount: { approved: 3, rejected: 2 },
  });

  assert.equal(nb.nbformat, 4);
  assert.ok(nb.cells.length >= 9);

  // First cell has Colab steps, expected runtime, and disconnect recovery
  const firstCell = nb.cells[0].source.join('');
  assert.match(firstCell, /Passos Exatos no Google Colab/i);
  assert.match(firstCell, /Tempo Esperado/i);
  assert.match(firstCell, /Recuperação de Desconexão/i);

  const fullCode = nb.cells.map((c) => c.source.join('')).join('\n');
  // Strict split boundary + horizontal flip prohibition
  assert.match(fullCode, /ALLOW_HORIZONTAL_FLIP = False/);
  assert.match(fullCode, /assert "test" not in set\(df\["split"\]/);
  assert.match(fullCode, /translate_integer_safe/);
  assert.match(fullCode, /swap_palette_preserving_ramp_role/);
  assert.match(fullCode, /roc_auc_score/);
  assert.match(fullCode, /t3_generator\.onnx/);
  assert.match(fullCode, /t3_reranker\.onnx/);
  assert.match(fullCode, /t3_model_card\.json/);
  assert.match(fullCode, /contact_sheets/);
  assert.match(fullCode, /RESUME_FROM_DRIVE/);

  // Model card validation rejects missing fields and accepts complete card
  const validCard = buildSampleT3ModelCard({ vocabulary: vocab });
  const parsed = parseAndValidateT3ModelCard(JSON.stringify(validCard));
  assert.equal(parsed.splitHash, computeSplitHash(split));
  assert.equal(parsed.version, 't3-maskgit-v1');
  assert.throws(() => parseAndValidateT3ModelCard('{}'), /version/i);
});

test('Part D (Real ONNX Runtime Web WASM): Executes real ONNX binaries, auto-resamples failed constraints, ranks with Critic+Reranker, and flags memorized copies', async () => {
  const { dat, spr, split } = makeSyntheticDataset(16);
  const vocab = buildRampRoleVocabularyFromTrainSplit({
    datasetKey: split.datasetKey,
    dat,
    spr,
    split,
  });
  const { calibration } = calibrateCriticFromArchive({
    datasetKey: split.datasetKey,
    dat,
    spr,
    split,
  });
  const index = runSpriteIndexingJob({
    datasetKey: split.datasetKey,
    dat,
    spr,
    split,
    calibration,
  });

  // Construct REAL executable ONNX binaries for generator & reranker (executed by real onnxruntime-web WASM!)
  const cleanGoldRgba = makeCleanSpriteRgba('gold_armor');
  const cleanGoldGrid = encodeRgbaToVocabGrid(cleanGoldRgba, vocab.entries);

  const genOnnx = buildRealOnnxGeneratorBytes({
    vocabSize: vocab.vocabSize,
    tagDim: 8,
    targetTokenGrid: cleanGoldGrid,
    preferredLogitStrength: 8.5,
  });
  const rrOnnx = buildRealOnnxRerankerBytes();

  // Execute real ONNX sessions in Node via onnxruntime-web
  const sessGen = await ort.InferenceSession.create(new Uint8Array(genOnnx), {
    executionProviders: ['wasm'],
  });
  const sessRr = await ort.InferenceSession.create(new Uint8Array(rrOnnx), {
    executionProviders: ['wasm'],
  });

  const realOnnxRunner: T3InferenceRunner = {
    executionProvider: 'wasm',
    async runGenerator({ tokens, categoryIds, tagVec, batchSize, seqLen, tagDim }) {
      const out = await sessGen.run({
        tokens: new ort.Tensor('int64', tokens, [batchSize, seqLen]),
        category_id: new ort.Tensor('int64', categoryIds, [batchSize]),
        tag_vec: new ort.Tensor('float32', tagVec, [batchSize, tagDim]),
      });
      return new Float32Array(out.logits.data as Float32Array);
    },
    async runReranker({ tokens, categoryIds, batchSize, seqLen }) {
      const out = await sessRr.run({
        tokens: new ort.Tensor('int64', tokens, [batchSize, seqLen]),
        category_id: new ort.Tensor('int64', categoryIds, [batchSize]),
      });
      return new Float32Array(out.logit.data as Float32Array);
    },
  };

  const modelCard = buildSampleT3ModelCard({ vocabulary: vocab });
  const batchRes = await runT3SamplingAndRanking({
    runner: realOnnxRunner,
    modelCard,
    category: 'Equipamentos',
    prompt: 'armadura dourada',
    calibration,
    spriteIndex: index,
    candidateCount: 2,
    seed: 12345,
  });

  assert.equal(batchRes.executionProvider, 'wasm');
  assert.equal(batchRes.candidates.length, 2);
  const top = batchRes.candidates[0];
  assert.equal(top.constraints.binaryAlphaOk, true);
  assert.equal(top.constraints.connectedSilhouetteOk, true);
  assert.equal(top.constraints.outlineClosureOk, true);
  assert.ok(top.rerankerProbability !== null && top.rerankerProbability > 0 && top.rerankerProbability < 1);
  // Because targetTokenGrid was an exact copy of cleanGoldRgba from train split, mandatory memorization check catches it!
  assert.equal(top.memorizedCopyWarning, true);

  // Verify automatic constraint resampling triggers when attempt #1 violates outline/silhouette constraints
  const brokenGrid = encodeRgbaToVocabGrid(makeBrokenNoOutlineRgba(), vocab.entries);
  let callCounter = 0;
  const resamplingRunner: T3InferenceRunner = {
    executionProvider: 'custom-runner',
    async runGenerator({ batchSize, seqLen }) {
      callCounter++;
      const useClean = callCounter > 10; // first sampling attempt emits brokenGrid -> fails constraint -> resamples cleanly on 2nd attempt
      const grid = useClean ? cleanGoldGrid : brokenGrid;
      const logits = new Float32Array(batchSize * seqLen * vocab.vocabSize);
      for (let b = 0; b < batchSize; b++) {
        for (let p = 0; p < seqLen; p++) {
          const target = grid[p];
          for (let v = 0; v < vocab.vocabSize; v++) {
            logits[(b * seqLen + p) * vocab.vocabSize + v] = v === target ? 9.0 : -4.0;
          }
        }
      }
      return logits;
    },
  };

  const resampled = await runT3SamplingAndRanking({
    runner: resamplingRunner,
    modelCard,
    category: 'Equipamentos',
    prompt: 'armadura dourada',
    calibration,
    spriteIndex: index,
    candidateCount: 1,
    seed: 99,
  });
  assert.ok(resampled.constraintFailuresResampled >= 1);
  assert.equal(resampled.candidates[0].resampleAttemptsUsed, 2);
  assert.equal(resampled.candidates[0].constraints.passed, true);

  // Verify import persistence into onnxModels store
  const backend = createMemoryStorageBackend();
  const imported = await importT3ModelPackageVerified(backend, {
    datasetKey: split.datasetKey,
    modelCardJson: JSON.stringify(modelCard),
    generatorOnnx: genOnnx,
    rerankerOnnx: rrOnnx,
    generatorFileName: 't3_generator.onnx',
    rerankerFileName: 't3_reranker.onnx',
  });
  assert.equal(imported.enabled, false); // disabled by default before test-split evaluation
  const loadedPkg = await loadT3ModelPackage(backend, split.datasetKey);
  assert.ok(loadedPkg);
  assert.equal(loadedPkg.generatorOnnx.byteLength, genOnnx.byteLength);

  await sessGen.release();
  await sessRr.release();
});

test('Parts E & F: Test-split report compares T3 vs T1, T2, Junk Floor (auto-disabling T3 when worse) and Retrain Export bundles human labels', async () => {
  const { dat, spr, split } = makeSyntheticDataset(16);
  const vocab = buildRampRoleVocabularyFromTrainSplit({
    datasetKey: split.datasetKey,
    dat,
    spr,
    split,
  });
  const { calibration } = calibrateCriticFromArchive({
    datasetKey: split.datasetKey,
    dat,
    spr,
    split,
  });
  const index = runSpriteIndexingJob({
    datasetKey: split.datasetKey,
    dat,
    spr,
    split,
    calibration,
  });
  const modelCard = buildSampleT3ModelCard({ vocabulary: vocab });

  // Weak/broken T3 model outputting outline-less fragmented blobs -> MUST be marked worse & disabled by default
  const brokenGrid = encodeRgbaToVocabGrid(makeBrokenNoOutlineRgba(), vocab.entries);
  const weakRunner: T3InferenceRunner = {
    executionProvider: 'custom-runner',
    async runGenerator({ batchSize, seqLen }) {
      const logits = new Float32Array(batchSize * seqLen * vocab.vocabSize);
      for (let b = 0; b < batchSize; b++) {
        for (let p = 0; p < seqLen; p++) {
          const t = brokenGrid[p];
          for (let v = 0; v < vocab.vocabSize; v++) {
            logits[(b * seqLen + p) * vocab.vocabSize + v] = v === t ? 9.0 : -4.0;
          }
        }
      }
      return logits;
    },
  };

  const report = await evaluateT3OnTestSplit({
    runner: weakRunner,
    modelCard,
    dat,
    spr,
    split,
    calibration,
    spriteIndex: index,
    lessons: [],
    maxRealTestSprites: 12,
    maxPromptCount: 3,
  });

  assert.equal(report.t3IsWorse, true);
  assert.equal(report.recommendedEnabled, false);
  assert.match(report.verdictSummaryPt, /DESATIVADO por padrão/i);
  assert.ok(report.realTest.medianScore > report.junkFloor.medianScore);
  assert.ok(report.t1.medianScore > report.t3.medianScore);

  // Verify Part F ("Re-treinar com novos rótulos") exports dataset + vocabulary + approved/rejected human labels (0 test!)
  const { evaluateSprite } = await import('../critic/evaluator');
  const cleanRgba = makeCleanSpriteRgba('blue_sword');
  const evClean = evaluateSprite(cleanRgba, { category: 'Equipamentos', isTileOverride: false, calibration });
  const genRecord: GeneratedSpriteRecord = {
    id: 'gen-human-1',
    datasetKey: split.datasetKey,
    prompt: 'espada azul',
    category: 'Equipamentos',
    createdAt: new Date().toISOString(),
    rgba: cleanRgba,
    paletteHex: ['#121622', '#5686c6'],
    evaluation: evClean,
    nearestReal: {
      spriteId: 2,
      category: 'Equipamentos',
      split: 'train',
      rgba: cleanRgba,
      criticScore: 90,
      silhouetteIoU: 0.9,
      pixelDiffRatio: 0.2,
      colorDiffMean: 0.1,
      originalityScore: 34,
    },
    provenance: {
      source: 'local-t1',
      tier: 'T1',
      model: null,
      fallbackWarning: null,
      donorSpriteIds: [2],
      diffPixelCount: 8,
      diffPixelLimit: 28,
      repaired: false,
      repairDelta: 0,
      repairMethod: 'none',
      appliedLessonIds: [],
      seed: 1,
    },
    approvalStatus: 'approved',
    readBackVerified: true,
  };

  const approvalLabel: ApprovalLabelRecord = {
    id: 'gen-human-1',
    datasetKey: split.datasetKey,
    spriteRecordId: 'gen-human-1',
    category: 'Equipamentos',
    decision: 'approved',
    criticScore: evClean.score,
    originalityScore: 34,
    issues: [],
    features: evClean.features,
    source: 'local-t1',
    createdAt: new Date().toISOString(),
    readBackVerified: true,
  };

  const retrainZip = buildTrainingDatasetExportZip({
    index,
    split,
    vocabulary: vocab,
    approvals: [approvalLabel],
    generatedSprites: [genRecord],
  });

  assert.equal(retrainZip.excludedTestCount, split.totals.test);
  assert.equal(retrainZip.hasVocabularyJson, true);
  assert.equal(retrainZip.humanApprovedCount, 1);
  assert.equal(retrainZip.humanRejectedCount, 0);
  assert.ok(retrainZip.filePaths.includes('vocabulary.json'));
  assert.ok(retrainZip.filePaths.includes('human_labels.csv'));
  assert.ok(retrainZip.filePaths.some((p) => p.startsWith('human_labels/approved/')));
  assert.equal(retrainZip.filePaths.some((p) => p.includes('/test/')), false);
});

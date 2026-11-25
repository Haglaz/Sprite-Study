import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildDatasetKey,
  createMemoryStorageBackend,
  exportFullBackupJson,
  importFullBackupJson,
  loadActiveDataset,
  loadDatasetSplit,
  putAndVerify,
  saveActiveDataset,
  saveDatasetSplit,
  type StorageBackend,
} from './persistence';
import {
  createStratifiedDatasetSplit,
  DEFAULT_SPLIT_SEED,
  getSplitSpriteIds,
  type SpriteSplitCandidate,
} from './split';
import type { StoredDatasetRecord } from './schema';

function makeSyntheticCandidates(perCategory = 40): SpriteSplitCandidate[] {
  const categories = ['Equipamentos', 'Consumíveis', 'Containers', 'Decoração'] as const;
  const out: SpriteSplitCandidate[] = [];
  let sid = 1;
  for (const cat of categories) {
    for (let i = 0; i < perCategory; i++) {
      out.push({ spriteId: sid++, category: cat });
    }
  }
  return out;
}

test('createStratifiedDatasetSplit produces deterministic 70/15/15 stratified partitions with zero leakage', () => {
  const candidates = makeSyntheticCandidates(40); // 160 total, 40 per category
  const splitA = createStratifiedDatasetSplit({
    datasetKey: 'hashA:hashB',
    candidates,
    seed: DEFAULT_SPLIT_SEED,
    createdAt: '2026-01-01T00:00:00.000Z',
  });
  const splitB = createStratifiedDatasetSplit({
    datasetKey: 'hashA:hashB',
    candidates: candidates.slice().reverse(), // input order independence
    seed: DEFAULT_SPLIT_SEED,
    createdAt: '2026-01-01T00:00:00.000Z',
  });

  assert.deepEqual(splitA, splitB);
  assert.equal(splitA.totals.total, 160);
  assert.equal(splitA.totals.train, 112); // 70% of 160 = 28 * 4
  assert.equal(splitA.totals.validation, 24); // 15% of 160 = 6 * 4
  assert.equal(splitA.totals.test, 24); // 15% of 160 = 6 * 4

  // Verify per-category stratification
  for (const cat of ['Equipamentos', 'Consumíveis', 'Containers', 'Decoração'] as const) {
    assert.equal(splitA.byCategory[cat].train.length, 28);
    assert.equal(splitA.byCategory[cat].validation.length, 6);
    assert.equal(splitA.byCategory[cat].test.length, 6);
  }

  // Verify zero spriteId overlap across train / validation / test
  const trainSet = new Set(getSplitSpriteIds(splitA, 'train'));
  const valSet = new Set(getSplitSpriteIds(splitA, 'validation'));
  const testSet = new Set(getSplitSpriteIds(splitA, 'test'));

  for (const id of trainSet) {
    assert.equal(valSet.has(id), false);
    assert.equal(testSet.has(id), false);
  }
  for (const id of valSet) {
    assert.equal(testSet.has(id), false);
  }
  assert.equal(trainSet.size + valSet.size + testSet.size, 160);
});

test('putAndVerify enforces read-back verification and rejects silent write failures', async () => {
  const brokenBackend: StorageBackend = {
    async get() {
      return null; // simulates failed persistence
    },
    async getAll() {
      return [];
    },
    async put() {
      // pretends write succeeded
    },
    async delete() {},
    async clear() {},
    async clearAll() {},
  };

  await assert.rejects(
    () =>
      putAndVerify(brokenBackend, 'lessons', {
        id: 'lesson-1',
        updatedAt: '2026-01-01T00:00:00.000Z',
        data: { rule: 'test' },
      }),
    /read-back/i,
  );
});

test('dataset + split persistence and backup export/import round-trip preserve binary ArrayBuffers', async () => {
  const db1 = createMemoryStorageBackend();
  const datBytes = new Uint8Array([0x44, 0x75, 0xa7, 0xd6, 0x01, 0x02, 0x03, 0x04]);
  const sprBytes = new Uint8Array([0xff, 0x00, 0xff, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60]);

  const dataset: StoredDatasetRecord = {
    id: 'active',
    datFileName: 'Tibia.dat',
    sprFileName: 'Tibia.spr',
    datSizeBytes: datBytes.byteLength,
    sprSizeBytes: sprBytes.byteLength,
    datSha256: 'abc123dat',
    sprSha256: 'def456spr',
    datBuffer: datBytes.buffer,
    sprBuffer: sprBytes.buffer,
    itemCount: 120,
    spriteCount: 200,
    nonEmptySprites: 160,
    versionHint: 'Tibia 7.72',
    savedAt: '2026-05-01T12:00:00.000Z',
  };

  const saved = await saveActiveDataset(db1, dataset);
  assert.equal(saved.verified, true);
  assert.equal(saved.readBack.datBuffer.byteLength, datBytes.byteLength);

  const key = buildDatasetKey(dataset.datSha256, dataset.sprSha256);
  const split = createStratifiedDatasetSplit({
    datasetKey: key,
    candidates: makeSyntheticCandidates(10),
  });
  const savedSplit = await saveDatasetSplit(db1, split);
  assert.equal(savedSplit.verified, true);

  // Export backup JSON from db1 and import into a fresh db2 (simulating origin change)
  const backupJson = await exportFullBackupJson(db1);
  const db2 = createMemoryStorageBackend();
  const importRes = await importFullBackupJson(db2, backupJson);

  assert.equal(importRes.verified, true);
  assert.equal(importRes.importedCounts.datasets, 1);
  assert.equal(importRes.importedCounts.splits, 1);

  const restoredDataset = await loadActiveDataset(db2);
  assert.ok(restoredDataset);
  assert.equal(restoredDataset.datFileName, 'Tibia.dat');
  assert.equal(restoredDataset.sprSha256, 'def456spr');
  assert.deepEqual(Array.from(new Uint8Array(restoredDataset.datBuffer)), Array.from(datBytes));
  assert.deepEqual(Array.from(new Uint8Array(restoredDataset.sprBuffer)), Array.from(sprBytes));

  const restoredSplit = await loadDatasetSplit(db2, key);
  assert.deepEqual(restoredSplit, split);
});

test('full 12-store backup round-trip preserves TypedArrays (Uint8ClampedArray), agents, lessons, and rejects corrupt JSON without wiping data', async () => {
  const db1 = createMemoryStorageBackend();
  const datBytes = new Uint8Array([0x01, 0x02, 0x03, 0x04]);
  const sprBytes = new Uint8Array([0x10, 0x20, 0x30, 0x40]);
  const rgbaPixels = new Uint8ClampedArray([18, 22, 30, 255, 220, 180, 90, 255, 0, 0, 0, 0]);
  const weightMask = new Float64Array([0.125, -0.75, 1.5]);

  await saveActiveDataset(db1, {
    id: 'active',
    datFileName: 'otsp.dat',
    sprFileName: 'otsp.spr',
    datSizeBytes: datBytes.byteLength,
    sprSizeBytes: sprBytes.byteLength,
    datSha256: 'datSha',
    sprSha256: 'sprSha',
    datBuffer: datBytes.buffer,
    sprBuffer: sprBytes.buffer,
    itemCount: 50,
    spriteCount: 80,
    nonEmptySprites: 75,
    versionHint: 'Tibia 10.41',
    savedAt: '2026-06-01T10:00:00.000Z',
  });

  await putAndVerify(db1, 'agents', {
    id: 'datSha:sprSha:t2-composite-ops',
    datasetKey: 'datSha:sprSha',
    elo: 1232.5,
    isActive: true,
  });

  await putAndVerify(db1, 'lessons', {
    id: 'lesson:human:1',
    category: 'Equipamentos',
    title: 'Preservar contorno 1px',
    status: 'active',
  });

  await putAndVerify(db1, 'generatedSprites', {
    id: 'gen-1',
    datasetKey: 'datSha:sprSha',
    rgba: rgbaPixels,
    nearestReal: { spriteId: 12, rgba: new Uint8ClampedArray([1, 2, 3, 255]) },
  });

  await putAndVerify(db1, 'criticState', {
    id: 'datSha:sprSha',
    datasetKey: 'datSha:sprSha',
    weights: weightMask,
  });

  await putAndVerify(db1, 'checkpoints', {
    id: 'datSha:sprSha:Equipamentos',
    mask: new Uint8Array([1, 0, 1, 1]),
  });

  const exportedJson = await exportFullBackupJson(db1);

  // Importing corrupt JSON must fail and leave db1 untouched
  await assert.rejects(() => importFullBackupJson(db1, '{not-valid-json'), /JSON malformado/i);
  await assert.rejects(
    () => importFullBackupJson(db1, JSON.stringify({ schema: 'wrong-schema', stores: {} })),
    /não reconhecido/i,
  );
  assert.ok(await loadActiveDataset(db1));

  // Importing valid JSON into a clean origin backend db2 restores all records + TypedArray types
  const db2 = createMemoryStorageBackend();
  const res = await importFullBackupJson(db2, exportedJson);
  assert.equal(res.verified, true);
  assert.equal(res.totalRecords, 6);
  assert.equal(res.importedCounts.datasets, 1);
  assert.equal(res.importedCounts.agents, 1);
  assert.equal(res.importedCounts.lessons, 1);
  assert.equal(res.importedCounts.generatedSprites, 1);
  assert.equal(res.importedCounts.criticState, 1);
  assert.equal(res.importedCounts.checkpoints, 1);

  const restoredSprite = await db2.get<{
    id: string;
    rgba: Uint8ClampedArray;
    nearestReal: { rgba: Uint8ClampedArray };
  }>('generatedSprites', 'gen-1');
  assert.ok(restoredSprite);
  assert.equal(restoredSprite.rgba instanceof Uint8ClampedArray, true);
  assert.deepEqual(Array.from(restoredSprite.rgba), Array.from(rgbaPixels));
  assert.equal(restoredSprite.nearestReal.rgba instanceof Uint8ClampedArray, true);

  const restoredCritic = await db2.get<{ id: string; weights: Float64Array }>(
    'criticState',
    'datSha:sprSha',
  );
  assert.ok(restoredCritic);
  assert.equal(restoredCritic.weights instanceof Float64Array, true);
  assert.deepEqual(Array.from(restoredCritic.weights), Array.from(weightMask));

  const restoredCkpt = await db2.get<{ id: string; mask: Uint8Array }>(
    'checkpoints',
    'datSha:sprSha:Equipamentos',
  );
  assert.ok(restoredCkpt);
  assert.equal(restoredCkpt.mask instanceof Uint8Array, true);
  assert.deepEqual(Array.from(restoredCkpt.mask), [1, 0, 1, 1]);
});

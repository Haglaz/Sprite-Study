import assert from 'node:assert/strict';
import test from 'node:test';
import {
  calibrateCriticFromArchive,
  retrainCriticWithHumanApprovals,
} from '../critic/diagnostics';
import { evaluateSprite } from '../critic/evaluator';
import { createMemoryStorageBackend } from '../db/persistence';
import { createStratifiedDatasetSplit } from '../db/split';
import { CLASSIC_MATERIAL_RAMPS } from '../generation/executors';
import {
  decomposeSpriteToIndexedAndRampRoles,
  findNearestRealSprite,
  retrieveDonorsForPrompt,
  runSpriteIndexingJob,
} from '../generation/indexer';
import { recordApprovalLabelVerified, saveGeneratedSpriteVerified } from '../generation/store';
import { createLessonVerified, loadAllLessons } from '../lessons/store';
import type {
  ParsedDatArchive,
  ParsedItemMetadata,
  ParsedSprArchive,
} from '../types/studio';
import { parseTibiaSpr, SPRITE_SIZE } from '../utils/tibiaParser';
import { computeEloUpdate, DEFAULT_ELO, expectedEloScore, parseCommentToLessonDraft } from './elo';
import { runFieldTestC6 } from './fieldTest';
import {
  ensureDefaultAgentsVerified,
  loadAgents,
  recordBlindEloMatchVerified,
  setActiveAgentVerified,
} from './store';

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
      if (d > 0.72) col = ramp[0];
      else if (Math.abs(x - cx) <= 1 && Math.abs(y - cy) <= 1) col = accent[1];
      else {
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

function buildFixture(count = 32): {
  dat: ParsedDatArchive;
  spr: ParsedSprArchive;
  split: ReturnType<typeof createStratifiedDatasetSplit>;
  calibration: ReturnType<typeof calibrateCriticFromArchive>['calibration'];
} {
  const chunks: Uint8Array[] = [];
  let cursor = 6 + count * 4;
  for (let s = 0; s < count; s++) {
    const rgba = makeSyntheticRealSprite(s);
    const rle: number[] = [];
    let p = 0;
    while (p < SPRITE_SIZE * SPRITE_SIZE) {
      let trans = 0;
      while (p < SPRITE_SIZE * SPRITE_SIZE && rgba[p * 4 + 3] === 0) {
        trans++;
        p++;
      }
      const colStart = p;
      let colCount = 0;
      while (p < SPRITE_SIZE * SPRITE_SIZE && rgba[p * 4 + 3] > 0) {
        colCount++;
        p++;
      }
      rle.push(trans & 0xff, (trans >>> 8) & 0xff, colCount & 0xff, (colCount >>> 8) & 0xff);
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
    chunks.push(body);
    cursor += body.length;
  }

  const rawSpr = new Uint8Array(cursor);
  const dv = new DataView(rawSpr.buffer);
  dv.setUint32(0, 0x439d5a33, true);
  dv.setUint16(4, count, true);
  let writePos = 6 + count * 4;
  for (let i = 0; i < count; i++) {
    dv.setUint32(6 + i * 4, writePos, true);
    rawSpr.set(chunks[i], writePos);
    writePos += chunks[i].length;
  }
  const spr = parseTibiaSpr(rawSpr.slice(0, writePos).buffer);

  const items: ParsedItemMetadata[] = Array.from({ length: count }, (_, idx) => ({
    id: 200 + idx,
    category: idx < 20 ? 'Equipamentos' : 'Consumíveis',
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
      stackable: idx >= 20,
      forceUse: false,
      multiUse: false,
      usable: idx >= 20,
      writable: false,
      pickupable: true,
      movable: true,
      unpassable: false,
      fluidContainer: idx >= 20,
      splash: false,
      light: false,
      hangable: false,
      equipable: idx < 20,
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
      Equipamentos: 20,
      Consumíveis: count - 20,
      Containers: 0,
      Decoração: 0,
    },
  };

  const split = createStratifiedDatasetSplit({
    datasetKey: 'p5-fixture',
    candidates: items.map((it) => ({ spriteId: it.spriteIds[0], category: it.category })),
    seed: 20260772,
  });

  const { calibration } = calibrateCriticFromArchive({
    datasetKey: 'p5-fixture',
    dat,
    spr,
    split,
    seed: 20260772,
  });

  return { dat, spr, split, calibration };
}

test('Elo update math is zero-sum, symmetric, and rewards upsets more than expected wins', () => {
  assert.equal(expectedEloScore(1200, 1200), 0.5);

  const equalWin = computeEloUpdate(1200, 1200, 1, 32);
  assert.equal(equalWin.deltaA, 16);
  assert.equal(equalWin.deltaB, -16);
  assert.equal(equalWin.nextA + equalWin.nextB, 2400);

  const upsetWin = computeEloUpdate(1100, 1300, 1, 32);
  const favoriteWin = computeEloUpdate(1300, 1100, 1, 32);
  assert.ok(upsetWin.deltaA > favoriteWin.deltaA, 'Upset win must grant larger Elo gain');
  assert.equal(Math.round((upsetWin.deltaA + upsetWin.deltaB) * 10), 0);

  const underdogDraw = computeEloUpdate(1100, 1300, 0.5, 32);
  assert.ok(underdogDraw.deltaA > 0, 'Underdog drawing a stronger opponent must gain Elo');
});

test('Blind A/B Elo matches and "Tornar ativo" persist with read-back verification and single-active invariant', async () => {
  const backend = createMemoryStorageBackend();
  const seeded = await ensureDefaultAgentsVerified(backend, 'p5-fixture');
  assert.equal(seeded.length, 3);
  assert.equal(seeded.filter((a) => a.isActive).length, 1);

  const t1 = seeded.find((a) => a.tier === 'T1' && a.repairPolicy === 'critic_ops')!;
  const t2 = seeded.find((a) => a.tier === 'T2')!;
  assert.equal(t1.elo, DEFAULT_ELO);
  assert.equal(t2.elo, DEFAULT_ELO);

  const match = await recordBlindEloMatchVerified(backend, {
    datasetKey: 'p5-fixture',
    agentAId: t1.id,
    agentBId: t2.id,
    winner: 'B',
    prompt: 'Escudo rubi forjado',
  });

  assert.equal(match.agentB.elo, 1216);
  assert.equal(match.agentA.elo, 1184);
  assert.equal(match.agentB.wins, 1);
  assert.equal(match.agentA.losses, 1);
  assert.equal(match.agentB.matchHistory.length, 1);

  // Promote T2 via "Tornar ativo"
  const promoted = await setActiveAgentVerified(backend, 'p5-fixture', t2.id);
  assert.equal(promoted.activeAgent.id, t2.id);
  assert.equal(promoted.activeAgent.isActive, true);
  assert.equal(promoted.allAgents.filter((a) => a.isActive).length, 1);

  const reloaded = await loadAgents(backend, 'p5-fixture');
  assert.equal(reloaded.find((a) => a.isActive)?.id, t2.id);
  assert.equal(reloaded[0].id, t2.id, 'Agents should load ordered by Elo descending');
});

test('Comment-to-lesson parses PT-BR critiques accurately and persists verified lessons to IndexedDB', async () => {
  const { dat, spr, split, calibration } = buildFixture(24);
  const index = runSpriteIndexingJob({ datasetKey: 'p5-fixture', dat, spr, split, calibration });
  const sample = index.entries[0];
  const ev = evaluateSprite(sample.rgba, { category: sample.category, calibration });

  const avoidDraft = parseCommentToLessonDraft(
    'Evitar contorno quebrado nas bordas externas da armadura',
    { id: 'gen-1', category: 'Equipamentos', evaluation: ev },
  );
  assert.equal(avoidDraft.scopeKey, 'outline_coverage');
  assert.equal(avoidDraft.polarity, 'avoid');

  const enforceDraft = parseCommentToLessonDraft(
    'Reforçar iluminação NW forte no topo da lâmina com brilho claro',
    { id: 'gen-2', category: 'Equipamentos', evaluation: ev },
  );
  assert.equal(enforceDraft.scopeKey, 'nw_lighting');
  assert.equal(enforceDraft.polarity, 'enforce');

  const backend = createMemoryStorageBackend();
  const saved = await createLessonVerified(backend, {
    datasetKey: 'p5-fixture',
    category: avoidDraft.category,
    scopeKey: avoidDraft.scopeKey,
    title: avoidDraft.title,
    source: 'human_feedback',
    polarity: avoidDraft.polarity,
    directive: avoidDraft.directive,
    validationCheck: avoidDraft.validationCheck,
    relatedIssueCodes: avoidDraft.relatedIssueCodes,
  });
  assert.equal(saved.saved.readBackVerified, true);
  const all = await loadAllLessons(backend);
  assert.equal(all.lessons.length, 1);
  assert.equal(all.lessons[0].source, 'human_feedback');
});

test('Approvals feed both the few-shot donor pool and automatic Critic retraining on new labels', async () => {
  const { dat, spr, split, calibration } = buildFixture(32);
  const index = runSpriteIndexingJob({ datasetKey: 'p5-fixture', dat, spr, split, calibration });
  const backend = createMemoryStorageBackend();

  const trainEquip = index.entries.filter((e) => e.split === 'train' && e.category === 'Equipamentos');
  const createdRecords = [];

  for (let i = 0; i < 6; i++) {
    const src = trainEquip[i % trainEquip.length];
    const dec = decomposeSpriteToIndexedAndRampRoles(src.rgba);
    const ev = evaluateSprite(src.rgba, { category: 'Equipamentos', calibration });
    const rec = await saveGeneratedSpriteVerified(backend, {
      id: `gen-p5-${i}`,
      datasetKey: 'p5-fixture',
      prompt: i < 3 ? 'Armadura safira paladino aprovada' : 'Armadura rejeitada teste',
      category: 'Equipamentos',
      createdAt: new Date(Date.now() + i * 1000).toISOString(),
      rgba: src.rgba,
      paletteHex: dec.paletteHex,
      evaluation: ev,
      nearestReal: findNearestRealSprite(src.rgba, index),
      provenance: {
        source: 'local-t1',
        tier: 'T1',
        model: null,
        fallbackWarning: null,
        donorSpriteIds: [src.spriteId],
        diffPixelCount: 6,
        diffPixelLimit: 24,
        repaired: false,
        repairDelta: 0,
        repairMethod: 'none',
        appliedLessonIds: [],
        seed: i + 10,
      },
      approvalStatus: 'pending',
      readBackVerified: false,
    });
    const vote = await recordApprovalLabelVerified(backend, {
      sprite: rec,
      decision: i < 3 ? 'approved' : 'rejected',
    });
    createdRecords.push(vote.updatedSprite);
  }

  const approvals = await backend.getAll<import('../generation/types').ApprovalLabelRecord>('approvals');

  // 1) Few-shot pool must include human-approved generated sprites (`spriteId < 0` + tag 'aprovado_humano')
  const donors = retrieveDonorsForPrompt({
    index,
    category: 'Equipamentos',
    prompt: 'Armadura safira paladino aprovada',
    topK: 8,
    approvals,
    approvedSprites: createdRecords,
  });
  assert.ok(
    donors.some((d) => d.entry.spriteId < 0 && d.entry.tags.includes('aprovado_humano')),
    'Human-approved generated sprite should enter the few-shot donor pool',
  );

  // 2) Automatic Critic retraining incorporates approved (+3 pos) and rejected (+3 neg) into discriminator
  const retrained = retrainCriticWithHumanApprovals({
    datasetKey: 'p5-fixture',
    dat,
    spr,
    split,
    approvals,
    previousCalibration: calibration,
  });

  assert.equal(retrained.approvedUsed, 3);
  assert.equal(retrained.rejectedUsed, 3);
  assert.equal(
    retrained.calibration.discriminator.trainPosCount,
    calibration.discriminator.trainPosCount + 3,
  );
  assert.ok(
    retrained.calibration.discriminator.trainNegCount >=
      calibration.discriminator.trainNegCount + 3,
    `Expected trainNegCount >= ${calibration.discriminator.trainNegCount + 3}, got ${retrained.calibration.discriminator.trainNegCount}`,
  );
  assert.ok(retrained.calibration.humanPreferences !== null);
  assert.equal(retrained.calibration.humanPreferences?.approvedCount, 3);
  assert.equal(retrained.calibration.humanPreferences?.rejectedCount, 3);
  assert.ok(retrained.currentValidationAuc >= 0.5 && retrained.currentValidationAuc <= 1.0);
});

test('Field Test (C.6) evaluates candidate vs baseline across fixed prompts with honest metrics', async () => {
  const { dat, spr, split, calibration } = buildFixture(32);
  const index = runSpriteIndexingJob({ datasetKey: 'p5-fixture', dat, spr, split, calibration });
  const backend = createMemoryStorageBackend();
  const agents = await ensureDefaultAgentsVerified(backend, 'p5-fixture');
  const baseline = agents.find((a) => a.repairPolicy === 'none')!;
  const candidate = agents.find((a) => a.tier === 'T2')!;

  const report = await runFieldTestC6({
    datasetKey: 'p5-fixture',
    baselineAgent: baseline,
    candidateAgent: candidate,
    index,
    calibration,
    promptCount: 4,
    seed: 20260606,
  });

  assert.equal(report.promptCount, 4);
  assert.equal(report.outcomes.length, 4);
  assert.equal(report.binaryAlphaPassRate, 1);
  assert.ok(['APROVADO EM CAMPO', 'EMPATE TÉCNICO', 'REGRESSÃO DETECTADA'].includes(report.verdict));
  assert.equal(
    report.candidateWins + report.baselineWins + report.ties,
    report.promptCount,
  );
});

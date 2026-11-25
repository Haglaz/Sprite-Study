import test from 'node:test';
import assert from 'node:assert/strict';
import { ALL_ITEM_CATEGORIES, type ItemCategory } from '../types/studio';
import { calibrateCriticAndBuildDiagnostics } from './diagnostics';
import { evaluateSprite } from './evaluator';
import {
  buildStandardJunkFloorFixtures,
  makeBlurredBlobSprite,
  makeBorderTouchingOffCenterSprite,
  makeCleanGroundTileFixture,
  makeCleanPixelArtFixture,
  makeDisconnectedSprite,
  makeFlatRectangleSprite,
  makeMixelUpscaledSprite,
  makeNearDuplicatePaletteSprite,
  makeOrphanSpeckledSprite,
  makePillowShadedSprite,
  makeRandomNoiseSprite,
  makeSemiTransparentSprite,
  makeSmoothGradientSprite,
  makeUnoutlinedSprite,
} from './fixtures';
import type { LabeledSpriteSample } from './profile';

function buildFixtureDataset(): {
  trainSamples: LabeledSpriteSample[];
  validationSamples: LabeledSpriteSample[];
} {
  const trainSamples: LabeledSpriteSample[] = [];
  const validationSamples: LabeledSpriteSample[] = [];
  let sid = 1;

  for (const cat of ALL_ITEM_CATEGORIES) {
    for (let i = 0; i < 24; i++) {
      trainSamples.push({
        spriteId: sid++,
        category: cat,
        rgba: makeCleanPixelArtFixture(cat, i),
      });
    }
    for (let i = 0; i < 8; i++) {
      validationSamples.push({
        spriteId: sid++,
        category: cat,
        rgba: makeCleanPixelArtFixture(cat, 24 + i),
      });
    }
  }

  return { trainSamples, validationSamples };
}

test('uncalibrated critic caps score and explicitly labels "não calibrado"', () => {
  const clean = makeCleanPixelArtFixture('Equipamentos', 0);
  const res = evaluateSprite(clean, { category: 'Equipamentos', calibration: null });
  assert.equal(res.calibrated, false);
  assert.equal(res.calibrationLabel, 'não calibrado');
  assert.ok(res.score <= 55, `Expected uncalibrated score <= 55, got ${res.score}`);
  assert.ok(res.issues.some((i) => i.code === 'UNCALIBRATED'));
});

test('hand-built defect fixtures trigger their exact corresponding hard-rule and hygiene validators', () => {
  const { trainSamples, validationSamples } = buildFixtureDataset();
  const { calibration } = calibrateCriticAndBuildDiagnostics({
    datasetKey: 'fixture-dataset',
    trainSamples,
    validationSamples,
    splitTotals: { train: trainSamples.length, validation: validationSamples.length, test: 32, total: trainSamples.length + validationSamples.length + 32 },
    seed: 772,
  });

  const cases: Array<{
    name: string;
    rgba: Uint8ClampedArray;
    category: ItemCategory;
    expectedCode: string;
  }> = [
    { name: 'semi-transparent alpha', rgba: makeSemiTransparentSprite(), category: 'Equipamentos', expectedCode: 'BINARY_ALPHA' },
    { name: 'flat single-color rectangle', rgba: makeFlatRectangleSprite(), category: 'Containers', expectedCode: 'FLAT_FILL' },
    { name: 'smooth 2D gradient', rgba: makeSmoothGradientSprite(), category: 'Decoração', expectedCode: 'SMOOTH_GRADIENT' },
    { name: 'random RGB noise', rgba: makeRandomNoiseSprite(42), category: 'Equipamentos', expectedCode: 'SPECKLE_NOISE' },
    { name: 'detached orphan pixels', rgba: makeOrphanSpeckledSprite(), category: 'Equipamentos', expectedCode: 'ORPHAN_PIXELS' },
    { name: 'missing dark outline', rgba: makeUnoutlinedSprite(), category: 'Consumíveis', expectedCode: 'MISSING_OUTLINE' },
    { name: 'disconnected silhouette', rgba: makeDisconnectedSprite(), category: 'Containers', expectedCode: 'DISCONNECTED_SILHOUETTE' },
    { name: 'border touching', rgba: makeBorderTouchingOffCenterSprite(), category: 'Equipamentos', expectedCode: 'TOUCHING_BORDER' },
    { name: 'off-center centroid', rgba: makeBorderTouchingOffCenterSprite(), category: 'Equipamentos', expectedCode: 'OFF_CENTER' },
    { name: 'near-duplicate palette', rgba: makeNearDuplicatePaletteSprite(), category: 'Containers', expectedCode: 'NEAR_DUPLICATE_PALETTE' },
    { name: 'mixel 2x2 upscaled', rgba: makeMixelUpscaledSprite(), category: 'Equipamentos', expectedCode: 'MIXEL_RESAMPLE' },
    { name: 'pillow shading', rgba: makePillowShadedSprite(), category: 'Decoração', expectedCode: 'PILLOW_SHADING' },
    { name: 'blurred blob alpha', rgba: makeBlurredBlobSprite(), category: 'Equipamentos', expectedCode: 'BINARY_ALPHA' },
  ];

  for (const c of cases) {
    const ev = evaluateSprite(c.rgba, { category: c.category, calibration });
    const codes = ev.issues.map((i) => i.code);
    assert.ok(
      codes.includes(c.expectedCode as never),
      `Fixture "${c.name}" expected issue ${c.expectedCode}, got [${codes.join(', ')}] (score=${ev.score})`,
    );
  }
});

test('ground tiles (coverage > 0.9 or flags.ground) skip non-tile outline/silhouette/border rules', () => {
  const tile = makeCleanGroundTileFixture(99);
  const res = evaluateSprite(tile, {
    category: 'Decoração',
    flags: { ground: true },
    calibration: null,
  });

  assert.equal(res.isTile, true);
  const codes = new Set(res.issues.map((i) => i.code));
  assert.equal(codes.has('MISSING_OUTLINE'), false);
  assert.equal(codes.has('TOUCHING_BORDER'), false);
  assert.equal(codes.has('OFF_CENTER'), false);
  assert.equal(codes.has('DISCONNECTED_SILHOUETTE'), false);
});

test('honest baseline guarantees: junk sprites < 40, held-out real median > 70, never flat 100, high validation AUC, deterministic for fixed seed', () => {
  const { trainSamples, validationSamples } = buildFixtureDataset();
  const totals = {
    train: trainSamples.length,
    validation: validationSamples.length,
    test: 32,
    total: trainSamples.length + validationSamples.length + 32,
  };

  const run1 = calibrateCriticAndBuildDiagnostics({
    datasetKey: 'fixture-v1',
    trainSamples,
    validationSamples,
    splitTotals: totals,
    seed: 12345,
    calibratedAt: '2026-01-01T00:00:00.000Z',
  });
  const run2 = calibrateCriticAndBuildDiagnostics({
    datasetKey: 'fixture-v1',
    trainSamples,
    validationSamples,
    splitTotals: totals,
    seed: 12345,
    calibratedAt: '2026-01-01T00:00:00.000Z',
  });

  // Strict determinism for a given seed
  assert.deepEqual(run1.diagnostics.realCeiling, run2.diagnostics.realCeiling);
  assert.deepEqual(run1.diagnostics.junkFloor, run2.diagnostics.junkFloor);
  assert.equal(
    run1.diagnostics.discriminatorValidationAuc,
    run2.diagnostics.discriminatorValidationAuc,
  );

  // Held-out real sprites median > 70 and never flat 100
  assert.ok(
    run1.diagnostics.realCeiling.median > 70,
    `Expected held-out real median > 70, got ${run1.diagnostics.realCeiling.median}`,
  );
  assert.ok(
    run1.diagnostics.realCeiling.max < 100,
    `Expected max score strictly < 100, got ${run1.diagnostics.realCeiling.max}`,
  );

  // Junk floor fixtures must score < 40 (median AND every canonical junk sprite)
  assert.ok(
    run1.diagnostics.junkFloor.median < 40,
    `Expected junk floor median < 40, got ${run1.diagnostics.junkFloor.median}`,
  );
  const canonicalJunk = buildStandardJunkFloorFixtures(12345);
  for (const jf of canonicalJunk) {
    const ev = evaluateSprite(jf.rgba, { category: jf.category, calibration: run1.calibration });
    assert.ok(
      ev.score < 40,
      `Expected junk sprite "${jf.label}" to score < 40, got ${ev.score} (issues: ${ev.issues.map((i) => i.code).join(', ')})`,
    );
  }

  // Discriminator held-out validation ROC-AUC must separate real from corrupted negatives
  assert.ok(
    run1.diagnostics.discriminatorValidationAuc >= 0.85,
    `Expected held-out discriminator AUC >= 0.85, got ${run1.diagnostics.discriminatorValidationAuc}`,
  );
});

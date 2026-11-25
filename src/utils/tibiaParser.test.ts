import test from 'node:test';
import assert from 'node:assert/strict';
import {
  decodeSpritePixels,
  extractDominantPalette,
  parseTibiaDat,
  parseTibiaSpr,
  SPRITE_SIZE,
  TibiaParseError,
} from './tibiaParser';

/**
 * Helper: Encodes an array of 32x32 RGBA pixel maps into a genuine Classic (uint16) or Extended (uint32) `.spr` buffer.
 */
function buildTestSprBuffer(
  sprites: Array<Map<number, [number, number, number]>>,
  signature = 0x4475a7d6,
  extended = false,
): ArrayBuffer {
  const spriteCount = sprites.length;
  const tableStart = extended ? 8 : 6;
  const headerSize = tableStart + spriteCount * 4;
  const encodedBlocks: Uint8Array[] = [];

  for (const pixelMap of sprites) {
    if (pixelMap.size === 0) {
      encodedBlocks.push(new Uint8Array(0));
      continue;
    }

    const rleChunks: number[] = [];
    let idx = 0;
    while (idx < SPRITE_SIZE * SPRITE_SIZE) {
      let trans = 0;
      while (idx < SPRITE_SIZE * SPRITE_SIZE && !pixelMap.has(idx)) {
        trans++;
        idx++;
      }
      const colors: [number, number, number][] = [];
      while (idx < SPRITE_SIZE * SPRITE_SIZE && pixelMap.has(idx)) {
        colors.push(pixelMap.get(idx)!);
        idx++;
      }
      if (colors.length === 0) break;

      rleChunks.push(trans & 0xff, (trans >> 8) & 0xff);
      rleChunks.push(colors.length & 0xff, (colors.length >> 8) & 0xff);
      for (const [r, g, b] of colors) {
        rleChunks.push(r, g, b);
      }
    }

    const block = new Uint8Array(5 + rleChunks.length);
    block[0] = 0xff;
    block[1] = 0x00;
    block[2] = 0xff;
    block[3] = rleChunks.length & 0xff;
    block[4] = (rleChunks.length >> 8) & 0xff;
    block.set(rleChunks, 5);
    encodedBlocks.push(block);
  }

  let totalLength = headerSize;
  for (const b of encodedBlocks) totalLength += b.length;

  const out = new ArrayBuffer(totalLength);
  const view = new DataView(out);
  const bytes = new Uint8Array(out);

  view.setUint32(0, signature, true);
  if (extended) {
    view.setUint32(4, spriteCount, true);
  } else {
    view.setUint16(4, spriteCount, true);
  }

  let writeCursor = headerSize;
  for (let i = 0; i < spriteCount; i++) {
    const block = encodedBlocks[i];
    if (block.length === 0) {
      view.setUint32(tableStart + i * 4, 0, true);
    } else {
      view.setUint32(tableStart + i * 4, writeCursor, true);
      bytes.set(block, writeCursor);
      writeCursor += block.length;
    }
  }

  return out;
}

/**
 * Helper: Encodes a Classic (7.55–8.60) or Extended (10.10–10.49) `.dat` binary buffer.
 */
function buildTestDatBuffer(
  items: Array<{ flagBytes: number[]; spriteId: number }>,
  signature = 0x4475a7d6,
  extended = false,
): ArrayBuffer {
  const itemCount = 99 + items.length;
  const body: number[] = [];

  for (const it of items) {
    body.push(...it.flagBytes, 0xff);
    // width=1, height=1, layers=1, xdiv=1, ydiv=1, zdiv=1, anim=1
    body.push(1, 1, 1, 1, 1, 1, 1);
    if (extended) {
      body.push(
        it.spriteId & 0xff,
        (it.spriteId >> 8) & 0xff,
        (it.spriteId >> 16) & 0xff,
        (it.spriteId >> 24) & 0xff,
      );
    } else {
      body.push(it.spriteId & 0xff, (it.spriteId >> 8) & 0xff);
    }
  }

  const out = new ArrayBuffer(12 + body.length);
  const view = new DataView(out);
  const bytes = new Uint8Array(out);

  view.setUint32(0, signature, true);
  view.setUint16(4, itemCount, true);
  view.setUint16(6, 0, true);
  view.setUint16(8, 0, true);
  view.setUint16(10, 0, true);
  bytes.set(body, 12);
  return out;
}

test('parseTibiaSpr and decodeSpritePixels accurately decode Classic uint16 RLE sprites', () => {
  const swordPixels = new Map<number, [number, number, number]>();
  swordPixels.set(2 * 32 + 2, [0, 0, 0]);
  swordPixels.set(2 * 32 + 3, [229, 169, 60]);
  swordPixels.set(2 * 32 + 4, [255, 0, 255]);
  swordPixels.set(10 * 32 + 10, [148, 180, 210]);

  const sprBuf = buildTestSprBuffer([swordPixels, new Map()], 0x4475a7d6, false);
  const spr = parseTibiaSpr(sprBuf);

  assert.equal(spr.isExtended, false);
  assert.equal(spr.spriteCount, 2);
  assert.equal(spr.nonEmptySprites, 1);
  assert.equal(spr.signatureHex, '0x4475A7D6');

  const rgba1 = decodeSpritePixels(spr, 1);
  assert.equal(rgba1[3], 0);
  const p66 = (2 * 32 + 2) * 4;
  assert.deepEqual([rgba1[p66], rgba1[p66 + 1], rgba1[p66 + 2], rgba1[p66 + 3]], [0, 0, 0, 255]);
  const p67 = (2 * 32 + 3) * 4;
  assert.deepEqual(
    [rgba1[p67], rgba1[p67 + 1], rgba1[p67 + 2], rgba1[p67 + 3]],
    [229, 169, 60, 255],
  );
  const p68 = (2 * 32 + 4) * 4;
  assert.equal(rgba1[p68 + 3], 0);

  const rgba2 = decodeSpritePixels(spr, 2);
  assert.equal(rgba2.reduce((acc, v) => acc + v, 0), 0);
});

test('parseTibiaDat decodes real Tibia 7.6-8.60 attribute codes and classifies categories', () => {
  const sprPixels = new Map<number, [number, number, number]>([
    [100, [0, 0, 0]],
    [101, [220, 60, 50]],
    [102, [220, 60, 50]],
    [103, [80, 160, 240]],
  ]);
  const spr = parseTibiaSpr(
    buildTestSprBuffer([sprPixels, sprPixels, sprPixels, sprPixels], 0x4475a7d6, false),
  );

  // Real Tibia 7.55–8.60 flags: 0x10 Pickupable, 0x05 Stackable, 0x04 Container, 0x00 Ground + 0x15 Light(4b)
  const datBuf = buildTestDatBuffer(
    [
      { flagBytes: [0x10], spriteId: 1 }, // Pickupable -> Equipamentos
      { flagBytes: [0x05, 0x10], spriteId: 2 }, // Stackable -> Consumíveis
      { flagBytes: [0x04, 0x10], spriteId: 3 }, // Container -> Containers
      { flagBytes: [0x00, 0x64, 0x00, 0x15, 0x05, 0x00, 0xce, 0x00], spriteId: 4 }, // Ground + Light -> Decoração
    ],
    0x4475a7d6,
    false,
  );

  const dat = parseTibiaDat(datBuf, spr.spriteCount, spr.isExtended);
  assert.equal(dat.items.length, 4);
  assert.equal(dat.items[0].category, 'Equipamentos');
  assert.equal(dat.items[1].category, 'Consumíveis');
  assert.equal(dat.items[2].category, 'Containers');
  assert.equal(dat.items[3].category, 'Decoração');

  const palette = extractDominantPalette(spr, dat.items, 8);
  assert.equal(palette[0], '#DC3C32');
  assert.equal(palette[1], '#50A0F0');
  assert.equal(palette.includes('#000000'), false);
});

test('parseTibiaSpr and parseTibiaDat auto-detect Extended (uint32) 10.41 format with 0x11 Pickupable and 0x1D MinimapColor', () => {
  const px = new Map<number, [number, number, number]>([
    [50, [0, 0, 0]],
    [51, [120, 200, 90]],
  ]);
  const spr = parseTibiaSpr(buildTestSprBuffer([px, px, px], 0x53835077, true));
  assert.equal(spr.isExtended, true);
  assert.equal(spr.spriteCount, 3);

  // Real Tibia 10.41 Extended flags: 0x11 Pickupable, 0x1D MinimapColor (2b), 0x16 Light (4b), 0xFE Usable
  const datBuf = buildTestDatBuffer(
    [
      { flagBytes: [0x11, 0xfe], spriteId: 1 }, // Pickupable + Usable -> Equipamentos
      { flagBytes: [0x05, 0x11], spriteId: 2 }, // Stackable -> Consumíveis
      { flagBytes: [0x00, 0x8c, 0x00, 0x0d, 0x1d, 0x37, 0x00, 0x1f], spriteId: 3 }, // Ground + MinimapColor + FullGround -> Decoração
    ],
    0x5383504e,
    true,
  );

  const dat = parseTibiaDat(datBuf, spr.spriteCount, spr.isExtended);
  assert.equal(dat.versionHint, 'Tibia 10.41 (Extended)');
  assert.equal(dat.items.length, 3);
  assert.equal(dat.items[0].category, 'Equipamentos');
  assert.equal(dat.items[1].category, 'Consumíveis');
  assert.equal(dat.items[2].category, 'Decoração');
});

test('parseTibiaSpr and parseTibiaDat reject malformed buffers cleanly', () => {
  assert.throws(() => parseTibiaSpr(new ArrayBuffer(3)), TibiaParseError);
  assert.throws(() => parseTibiaDat(new ArrayBuffer(8)), TibiaParseError);
});

test('parseTibiaDat exposes extended per-item flags including unpassable, writable, clothSlot and marketName', () => {
  const px = new Map<number, [number, number, number]>([[10, [180, 140, 80]]]);
  const spr = parseTibiaSpr(buildTestSprBuffer([px, px], 0x53835077, true));

  // Item 100: 0x21 Cloth(slot=4), 0x08 Writable(len=20), 0x22 Market(name="Plate Armor"), 0x11 Pickupable
  const marketNameBytes = Array.from('Plate Armor').map((c) => c.charCodeAt(0));
  const marketAttrBytes = [
    0x22,
    0x01, 0x00, // category
    0x64, 0x00, // tradeAs
    0x64, 0x00, // showAs
    marketNameBytes.length & 0xff,
    (marketNameBytes.length >> 8) & 0xff,
    ...marketNameBytes,
    0x00, 0x00, // restrictVoc
    0x00, 0x00, // reqLevel
  ];
  const datBuf = buildTestDatBuffer(
    [
      { flagBytes: [0x08, 0x14, 0x00, 0x11, 0x21, 0x04, 0x00, ...marketAttrBytes], spriteId: 1 },
      { flagBytes: [0x00, 0x64, 0x00, 0x0c, 0x0d], spriteId: 2 }, // ground + unpassable + notMovable
    ],
    0x5383504e,
    true,
  );

  const dat = parseTibiaDat(datBuf, spr.spriteCount, spr.isExtended);
  assert.equal(dat.items.length, 2);
  assert.equal(dat.items[0].flags.equipable, true);
  assert.equal(dat.items[0].flags.clothSlot, 4);
  assert.equal(dat.items[0].flags.writable, true);
  assert.equal(dat.items[0].flags.marketName, 'Plate Armor');
  assert.equal(dat.items[0].category, 'Equipamentos');

  assert.equal(dat.items[1].flags.ground, true);
  assert.equal(dat.items[1].flags.unpassable, true);
  assert.equal(dat.items[1].flags.movable, false);
});


import { extractProfileNumericArray } from '../critic/features';
import { PROFILE_FEATURE_KEYS, type HumanPreferenceModel } from '../critic/types';
import { putAndVerify, type StorageBackend } from '../db/persistence';
import type { ApprovalLabelRecord, GeneratedSpriteRecord } from './types';

function ensureUint8Clamped(raw: Uint8ClampedArray | ArrayBuffer | number[]): Uint8ClampedArray {
  if (raw instanceof Uint8ClampedArray) return raw;
  if (raw instanceof ArrayBuffer) return new Uint8ClampedArray(raw);
  return new Uint8ClampedArray(raw);
}

function hydrateRecord(raw: GeneratedSpriteRecord): GeneratedSpriteRecord {
  return {
    ...raw,
    rgba: ensureUint8Clamped(raw.rgba as unknown as Uint8ClampedArray | ArrayBuffer),
    nearestReal: {
      ...raw.nearestReal,
      rgba: ensureUint8Clamped(raw.nearestReal.rgba as unknown as Uint8ClampedArray | ArrayBuffer),
    },
  };
}

export async function loadGeneratedSprites(
  backend: StorageBackend,
  datasetKey?: string,
): Promise<GeneratedSpriteRecord[]> {
  const all = await backend.getAll<GeneratedSpriteRecord>('generatedSprites');
  return all
    .filter((r) => !datasetKey || r.datasetKey === datasetKey)
    .map(hydrateRecord)
    .sort((a, b) => Date.parse(b.createdAt) - Date.parse(a.createdAt));
}

/**
 * Honesty Rule #4: Persists a generated sprite to IndexedDB `'generatedSprites'`
 * and reads it back immediately to confirm integrity before marking `readBackVerified: true`.
 */
export async function saveGeneratedSpriteVerified(
  backend: StorageBackend,
  record: GeneratedSpriteRecord,
): Promise<GeneratedSpriteRecord> {
  const toSave: GeneratedSpriteRecord = {
    ...record,
    readBackVerified: true,
  };

  const verified = await putAndVerify(
    backend,
    'generatedSprites',
    toSave,
    (rb, wr) =>
      rb.id === wr.id &&
      rb.datasetKey === wr.datasetKey &&
      rb.evaluation.score === wr.evaluation.score &&
      rb.nearestReal.spriteId === wr.nearestReal.spriteId &&
      rb.rgba.length === wr.rgba.length,
  );

  return hydrateRecord(verified.readBack);
}

export async function deleteGeneratedSprite(
  backend: StorageBackend,
  id: string,
): Promise<void> {
  await backend.delete('generatedSprites', id);
}

export async function loadApprovals(
  backend: StorageBackend,
  datasetKey?: string,
): Promise<ApprovalLabelRecord[]> {
  const all = await backend.getAll<ApprovalLabelRecord>('approvals');
  return all
    .filter((a) => !datasetKey || a.datasetKey === datasetKey)
    .sort((a, b) => Date.parse(b.createdAt) - Date.parse(a.createdAt));
}

/**
 * Rebuilds the Critic v1 `HumanPreferenceModel` from persisted human approve/reject labels.
 * Returns null if fewer than 6 total labels exist (or if either class has 0 votes),
 * matching the Critic evaluator's honest minimum-evidence threshold.
 */
export function rebuildHumanPreferenceModel(
  approvals: ApprovalLabelRecord[],
): HumanPreferenceModel | null {
  const approved = approvals.filter((a) => a.decision === 'approved');
  const rejected = approvals.filter((a) => a.decision === 'rejected');
  if (approved.length + rejected.length < 6 || approved.length === 0 || rejected.length === 0) {
    return null;
  }

  const dim = PROFILE_FEATURE_KEYS.length;
  const approvedMean = new Array<number>(dim).fill(0);
  const rejectedMean = new Array<number>(dim).fill(0);

  for (const a of approved) {
    const vec = extractProfileNumericArray(a.features, PROFILE_FEATURE_KEYS);
    for (let i = 0; i < dim; i++) approvedMean[i] += vec[i];
  }
  for (let i = 0; i < dim; i++) approvedMean[i] /= approved.length;

  for (const r of rejected) {
    const vec = extractProfileNumericArray(r.features, PROFILE_FEATURE_KEYS);
    for (let i = 0; i < dim; i++) rejectedMean[i] += vec[i];
  }
  for (let i = 0; i < dim; i++) rejectedMean[i] /= rejected.length;

  return {
    approvedCount: approved.length,
    rejectedCount: rejected.length,
    approvedMean,
    rejectedMean,
  };
}

/**
 * Persists a human Approve/Reject label to `'approvals'` (read-back verified) AND updates
 * `approvalStatus` on the corresponding `'generatedSprites'` record (read-back verified).
 */
export async function recordApprovalLabelVerified(
  backend: StorageBackend,
  params: {
    sprite: GeneratedSpriteRecord;
    decision: 'approved' | 'rejected';
    note?: string;
  },
): Promise<{
  approval: ApprovalLabelRecord;
  updatedSprite: GeneratedSpriteRecord;
  allApprovals: ApprovalLabelRecord[];
  humanPreferences: HumanPreferenceModel | null;
}> {
  const { sprite, decision, note } = params;
  const now = new Date().toISOString();

  const label: ApprovalLabelRecord = {
    id: sprite.id,
    datasetKey: sprite.datasetKey,
    spriteRecordId: sprite.id,
    category: sprite.category,
    decision,
    criticScore: sprite.evaluation.score,
    originalityScore: sprite.nearestReal.originalityScore,
    issues: sprite.evaluation.issues.map((i) => i.code),
    features: sprite.evaluation.features,
    source: sprite.provenance.source,
    note: note?.trim() || undefined,
    createdAt: now,
    readBackVerified: true,
  };

  const verifiedApproval = await putAndVerify(
    backend,
    'approvals',
    label,
    (rb, wr) => rb.id === wr.id && rb.decision === wr.decision && rb.criticScore === wr.criticScore,
  );

  const updatedSprite = await saveGeneratedSpriteVerified(backend, {
    ...sprite,
    approvalStatus: decision,
    approvalUpdatedAt: now,
  });

  const allApprovals = await loadApprovals(backend, sprite.datasetKey);
  const humanPreferences = rebuildHumanPreferenceModel(allApprovals);

  return {
    approval: verifiedApproval.readBack,
    updatedSprite,
    allApprovals,
    humanPreferences,
  };
}

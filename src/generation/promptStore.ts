import { putAndVerify, type StorageBackend } from '../db/persistence';
import type { PromptHistoryRecord } from './types';

export async function loadPromptHistory(
  backend: StorageBackend,
  datasetKey?: string,
): Promise<PromptHistoryRecord[]> {
  const all = await backend.getAll<PromptHistoryRecord>('promptHistory');
  return all
    .filter((rec) => !datasetKey || rec.datasetKey === datasetKey)
    .sort((a, b) => Date.parse(b.createdAt) - Date.parse(a.createdAt));
}

/**
 * Persists an engineered prompt version to IndexedDB `'promptHistory'`
 * with mandatory read-back verification (`putAndVerify`).
 */
export async function savePromptVersionVerified(
  backend: StorageBackend,
  record: PromptHistoryRecord,
): Promise<PromptHistoryRecord> {
  const toSave: PromptHistoryRecord = {
    ...record,
    readBackVerified: true,
  };

  const verified = await putAndVerify(
    backend,
    'promptHistory',
    toSave,
    (rb, wr) =>
      rb.id === wr.id &&
      rb.requestId === wr.requestId &&
      rb.version === wr.version &&
      rb.approved === wr.approved &&
      rb.engineeredPrompt.itemName === wr.engineeredPrompt.itemName,
  );

  return verified.readBack;
}

export async function markPromptApprovedVerified(
  backend: StorageBackend,
  record: PromptHistoryRecord,
): Promise<PromptHistoryRecord> {
  return savePromptVersionVerified(backend, {
    ...record,
    approved: true,
    approvedAt: new Date().toISOString(),
  });
}

export async function deletePromptVersion(
  backend: StorageBackend,
  id: string,
): Promise<void> {
  await backend.delete('promptHistory', id);
}

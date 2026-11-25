import { putAndVerify, type StorageBackend } from '../db/persistence';
import type { ItemCategory } from '../types/studio';
import type { EliteArchiveRecord, EvolutionCheckpoint } from './types';

export function buildCheckpointId(datasetKey: string, category: ItemCategory): string {
  return `${datasetKey}:${category}`;
}

/**
 * Persists the full co-evolution checkpoint across `checkpoints`, `neatPopulations`,
 * `gaPopulations`, and (when promoted) `eliteArchive`, enforcing mandatory read-back verification.
 */
export async function saveEvolutionCheckpointVerified(
  backend: StorageBackend,
  checkpoint: EvolutionCheckpoint,
): Promise<EvolutionCheckpoint> {
  const toSave: EvolutionCheckpoint = {
    ...checkpoint,
    readBackVerified: true,
    updatedAt: new Date().toISOString(),
  };

  // 1. Save NEAT population state with read-back check
  await putAndVerify(
    backend,
    'neatPopulations',
    {
      id: toSave.id,
      datasetKey: toSave.datasetKey,
      category: toSave.category,
      generation: toSave.neatState.generation,
      updatedAt: toSave.updatedAt,
      state: toSave.neatState,
    },
    (rb, wr) => rb.generation === wr.generation && rb.state.genomes.length === wr.state.genomes.length,
  );

  // 2. Save Recipe GA population state with read-back check
  await putAndVerify(
    backend,
    'gaPopulations',
    {
      id: toSave.id,
      datasetKey: toSave.datasetKey,
      category: toSave.category,
      generation: toSave.gaState.generation,
      updatedAt: toSave.updatedAt,
      state: toSave.gaState,
    },
    (rb, wr) => rb.generation === wr.generation && rb.state.genomes.length === wr.state.genomes.length,
  );

  // 3. If a champion was promoted on this checkpoint, persist to eliteArchive with read-back check
  if (toSave.champion) {
    const eliteRecord: EliteArchiveRecord = {
      id: `${toSave.id}:gen-${toSave.champion.generation}`,
      datasetKey: toSave.datasetKey,
      category: toSave.category,
      generation: toSave.champion.generation,
      promotedAt: toSave.champion.promotedAt,
      recipe: toSave.champion.recipe,
      neat: toSave.champion.neat,
      trainScore: toSave.champion.trainStats.meanRestoredScore,
      valScore: toSave.champion.valStats.medianRestoredScore,
      valDelta: toSave.champion.valStats.medianDelta,
      silhouetteIoU: toSave.champion.valStats.meanSilhouetteIoU,
      readBackVerified: true,
    };
    await putAndVerify(
      backend,
      'eliteArchive',
      eliteRecord,
      (rb, wr) => rb.generation === wr.generation && rb.valScore === wr.valScore,
    );
  }

  // 4. Save master checkpoint with read-back check
  const verified = await putAndVerify(
    backend,
    'checkpoints',
    toSave,
    (rb, wr) =>
      rb.generation === wr.generation &&
      rb.history.length === wr.history.length &&
      rb.promotionLog.length === wr.promotionLog.length,
  );

  return verified.readBack;
}

export async function loadEvolutionCheckpoint(
  backend: StorageBackend,
  datasetKey: string,
  category: ItemCategory,
): Promise<EvolutionCheckpoint | null> {
  return backend.get<EvolutionCheckpoint>('checkpoints', buildCheckpointId(datasetKey, category));
}

export async function listEvolutionCheckpoints(
  backend: StorageBackend,
  datasetKey: string,
): Promise<EvolutionCheckpoint[]> {
  const all = await backend.getAll<EvolutionCheckpoint>('checkpoints');
  return all
    .filter((c) => c.datasetKey === datasetKey)
    .sort((a, b) => Date.parse(b.updatedAt) - Date.parse(a.updatedAt));
}

export async function loadEliteArchive(
  backend: StorageBackend,
  datasetKey: string,
  category?: ItemCategory,
): Promise<EliteArchiveRecord[]> {
  const all = await backend.getAll<EliteArchiveRecord>('eliteArchive');
  return all
    .filter((r) => r.datasetKey === datasetKey && (!category || r.category === category))
    .sort((a, b) => b.valScore - a.valScore);
}

export async function deleteEvolutionCheckpoint(
  backend: StorageBackend,
  datasetKey: string,
  category: ItemCategory,
): Promise<void> {
  const id = buildCheckpointId(datasetKey, category);
  await Promise.all([
    backend.delete('checkpoints', id),
    backend.delete('neatPopulations', id),
    backend.delete('gaPopulations', id),
  ]);
}

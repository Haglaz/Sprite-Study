import type { StoredDatasetSplit } from '../db/schema';
import type { ItemCategory, LocalDatasetSummary } from '../types/studio';
import { decodeSpritePixels } from '../utils/tibiaParser';
import type { TrainingWorkerCommand, TrainingWorkerEvent, WorkerSpritePayload } from './trainingWorker';

/**
 * Decodes strictly separated `train` and `validation` sprites for a category.
 * Never accesses or leaks the held-out `test` partition.
 */
export function extractSplitSpritesForCategory(params: {
  dataset: LocalDatasetSummary;
  split: StoredDatasetSplit;
  category: ItemCategory;
  maxTrain: number;
  maxVal: number;
}): { trainSprites: WorkerSpritePayload[]; valSprites: WorkerSpritePayload[] } {
  const { dataset, split, category, maxTrain, maxVal } = params;
  const catSplit = split.byCategory[category];
  if (!catSplit) {
    return { trainSprites: [], valSprites: [] };
  }

  const decodeList = (ids: number[], cap: number): WorkerSpritePayload[] => {
    const out: WorkerSpritePayload[] = [];
    for (const spriteId of ids) {
      if (out.length >= cap) break;
      const rgba = decodeSpritePixels(dataset.spr, spriteId);
      if (rgba) {
        out.push({ spriteId, rgba });
      }
    }
    return out;
  };

  return {
    trainSprites: decodeList(catSplit.train, maxTrain),
    valSprites: decodeList(catSplit.validation, maxVal),
  };
}

export interface EvolutionWorkerHandle {
  start(cmd: Extract<TrainingWorkerCommand, { type: 'start' }>): void;
  pause(): void;
  stop(): void;
  terminate(): void;
}

export function createEvolutionWorkerClient(
  onEvent: (event: TrainingWorkerEvent) => void,
): EvolutionWorkerHandle {
  const worker = new Worker(new URL('./trainingWorker.ts', import.meta.url), {
    type: 'module',
  });

  worker.onmessage = (ev: MessageEvent<TrainingWorkerEvent>) => {
    if (ev.data) onEvent(ev.data);
  };

  worker.onerror = (err) => {
    onEvent({
      type: 'error',
      message: err.message || 'Falha inesperada no Web Worker de treino.',
    });
  };

  return {
    start(cmd) {
      worker.postMessage(cmd);
    },
    pause() {
      worker.postMessage({ type: 'pause' } satisfies TrainingWorkerCommand);
    },
    stop() {
      worker.postMessage({ type: 'stop' } satisfies TrainingWorkerCommand);
    },
    terminate() {
      worker.terminate();
    },
  };
}

/// <reference lib="webworker" />

import type { CriticCalibration } from '../critic/types';
import { createInitialCheckpoint, runCoevolutionGenerationStep } from './coevolution';
import { buildRestorationSamples } from './restorationTask';
import type {
  CoevolutionRunConfig,
  EvolutionCheckpoint,
  EvolutionGenerationPoint,
  PromotionLogEntry,
  RestorationTripletPreview,
} from './types';

export interface WorkerSpritePayload {
  spriteId: number;
  rgba: Uint8ClampedArray;
}

export type TrainingWorkerCommand =
  | {
      type: 'start';
      datasetKey: string;
      config: CoevolutionRunConfig;
      calibration: CriticCalibration;
      realCeilingMedian: number;
      junkFloorMedian: number;
      trainSprites: WorkerSpritePayload[];
      valSprites: WorkerSpritePayload[];
      resumeCheckpoint?: EvolutionCheckpoint | null;
    }
  | { type: 'pause' }
  | { type: 'stop' };

export type TrainingWorkerEvent =
  | {
      type: 'started';
      generation: number;
      targetGenerations: number;
      resumed: boolean;
      trainSampleCount: number;
      valSampleCount: number;
    }
  | {
      type: 'generation';
      checkpoint: EvolutionCheckpoint;
      point: EvolutionGenerationPoint;
      promotionEntry: PromotionLogEntry;
      previews: RestorationTripletPreview[];
    }
  | {
      type: 'paused';
      checkpoint: EvolutionCheckpoint;
    }
  | {
      type: 'completed';
      checkpoint: EvolutionCheckpoint;
      previews: RestorationTripletPreview[];
    }
  | {
      type: 'error';
      message: string;
    };

let pauseRequested = false;
let stopRequested = false;
let running = false;

const workerScope = self as unknown as DedicatedWorkerGlobalScope;

function emit(event: TrainingWorkerEvent): void {
  workerScope.postMessage(event);
}

workerScope.onmessage = (ev: MessageEvent<TrainingWorkerCommand>) => {
  const msg = ev.data;
  if (!msg) return;

  if (msg.type === 'pause') {
    pauseRequested = true;
    return;
  }

  if (msg.type === 'stop') {
    stopRequested = true;
    pauseRequested = true;
    return;
  }

  if (msg.type === 'start') {
    if (running) return;
    running = true;
    pauseRequested = false;
    stopRequested = false;

    void runLoop(msg).finally(() => {
      running = false;
    });
  }
};

async function runLoop(
  cmd: Extract<TrainingWorkerCommand, { type: 'start' }>,
): Promise<void> {
  try {
    const {
      datasetKey,
      config,
      calibration,
      realCeilingMedian,
      junkFloorMedian,
      trainSprites,
      valSprites,
      resumeCheckpoint,
    } = cmd;

    if (trainSprites.length === 0 || valSprites.length === 0) {
      emit({
        type: 'error',
        message:
          'Split insuficiente nesta categoria: são necessários sprites reais tanto em treino quanto em validação.',
      });
      return;
    }

    const trainSamples = buildRestorationSamples({
      sprites: trainSprites,
      category: config.category,
      calibration,
      seed: config.seed + 101,
      maxCount: config.trainBatchSize,
    });

    const valSamples = buildRestorationSamples({
      sprites: valSprites,
      category: config.category,
      calibration,
      seed: config.seed + 503,
      maxCount: config.valBatchSize,
    });

    const valBaselineCorruptedMedian =
      Math.round(
        (valSamples.reduce((acc, s) => acc + s.corruptedScore, 0) / Math.max(1, valSamples.length)) *
          10,
      ) / 10;

    let checkpoint: EvolutionCheckpoint;
    let resumed = false;

    if (resumeCheckpoint && resumeCheckpoint.category === config.category) {
      resumed = true;
      checkpoint = {
        ...resumeCheckpoint,
        config: {
          ...resumeCheckpoint.config,
          targetGenerations: Math.max(
            config.targetGenerations,
            resumeCheckpoint.generation + 1,
          ),
        },
        status: 'running',
      };
    } else {
      checkpoint = createInitialCheckpoint({
        datasetKey,
        config,
        valBaselineCorruptedMedian,
        realCeilingMedian,
        junkFloorMedian,
        trainSampleCount: trainSamples.length,
        valSampleCount: valSamples.length,
      });
    }

    emit({
      type: 'started',
      generation: checkpoint.generation,
      targetGenerations: checkpoint.config.targetGenerations,
      resumed,
      trainSampleCount: trainSamples.length,
      valSampleCount: valSamples.length,
    });

    let lastPreviews: RestorationTripletPreview[] = [];

    while (checkpoint.generation < checkpoint.config.targetGenerations) {
      if (stopRequested || pauseRequested) {
        checkpoint = { ...checkpoint, status: 'paused', updatedAt: new Date().toISOString() };
        emit({ type: 'paused', checkpoint });
        return;
      }

      const stepResult = runCoevolutionGenerationStep({
        checkpoint,
        trainSamples,
        valSamples,
        calibration,
      });

      checkpoint = stepResult.checkpoint;
      lastPreviews = stepResult.validationPreviews;

      emit({
        type: 'generation',
        checkpoint,
        point: stepResult.point,
        promotionEntry: stepResult.promotionEntry,
        previews: stepResult.validationPreviews,
      });

      // Yield back to worker event queue so incoming `pause` / `stop` commands are processed immediately
      await new Promise((resolve) => setTimeout(resolve, 0));
    }

    checkpoint = { ...checkpoint, status: 'completed', updatedAt: new Date().toISOString() };
    emit({ type: 'completed', checkpoint, previews: lastPreviews });
  } catch (err) {
    emit({
      type: 'error',
      message: err instanceof Error ? err.message : 'Erro interno no Web Worker de evolução.',
    });
  }
}

import type { T3InferenceRunner } from './sampler';
import {
  createOnnxRuntimeRunner,
  type T3WorkerRequest,
  type T3WorkerResponse,
} from './t3Worker';
import type { T3ModelPackageRecord } from './types';

export interface WorkerT3ClientHandle {
  runner: T3InferenceRunner;
  dispose: () => Promise<void>;
}

/**
 * Spawns a dedicated Web Worker running onnxruntime-web (WebGPU with WASM fallback)
 * so T3 sampling never blocks the React UI thread.
 */
export async function createWorkerT3Client(
  pkg: Pick<T3ModelPackageRecord, 'generatorOnnx' | 'rerankerOnnx'>,
): Promise<WorkerT3ClientHandle> {
  if (typeof Worker === 'undefined') {
    const bundle = await createOnnxRuntimeRunner({
      generatorOnnx: pkg.generatorOnnx,
      rerankerOnnx: pkg.rerankerOnnx,
    });
    return bundle;
  }

  const worker = new Worker(new URL('./t3Worker.ts', import.meta.url), {
    type: 'module',
  });

  let seq = 0;
  let initialized = false;
  let activeProvider: 'webgpu' | 'wasm' = 'wasm';
  const pending = new Map<
    string,
    {
      resolve: (res: T3WorkerResponse) => void;
      reject: (err: Error) => void;
    }
  >();

  worker.onmessage = (ev: MessageEvent<T3WorkerResponse>) => {
    const res = ev.data;
    const entry = pending.get(res.requestId);
    if (!entry) return;
    pending.delete(res.requestId);
    if (!res.ok) {
      entry.reject(new Error(res.error || 'Falha na inferência ONNX no Web Worker.'));
      return;
    }
    if (res.executionProvider) {
      activeProvider = res.executionProvider;
    }
    entry.resolve(res);
  };

  worker.onerror = (err) => {
    const error = new Error(err.message || 'Erro fatal no Web Worker T3.');
    for (const entry of pending.values()) {
      entry.reject(error);
    }
    pending.clear();
  };

  const rpc = (
    payload: Omit<T3WorkerRequest, 'requestId'>,
  ): Promise<T3WorkerResponse> => {
    const requestId = `t3w-${++seq}`;
    const msg: T3WorkerRequest = {
      requestId,
      ...payload,
      ...(initialized
        ? {}
        : {
            generatorOnnx: pkg.generatorOnnx.slice(0),
            rerankerOnnx: pkg.rerankerOnnx ? pkg.rerankerOnnx.slice(0) : null,
          }),
    };
    initialized = true;
    return new Promise((resolve, reject) => {
      pending.set(requestId, { resolve, reject });
      worker.postMessage(msg);
    });
  };

  // Warm-up ONNX InferenceSession inside Worker to detect WebGPU vs WASM immediately
  const warmup = await rpc({ type: 'init_and_infer_step' });
  if (warmup.executionProvider) {
    activeProvider = warmup.executionProvider;
  }

  const runner: T3InferenceRunner = {
    get executionProvider() {
      return activeProvider;
    },
    async runGenerator({ tokens, categoryIds, tagVec, batchSize, seqLen, tagDim }) {
      const res = await rpc({
        type: 'init_and_infer_step',
        op: 'generator',
        tokens,
        categoryIds,
        tagVec,
        batchSize,
        seqLen,
        tagDim,
      });
      return res.data ?? new Float32Array(0);
    },
    runReranker: pkg.rerankerOnnx
      ? async ({ tokens, categoryIds, batchSize, seqLen }) => {
          const res = await rpc({
            type: 'init_and_infer_step',
            op: 'reranker',
            tokens,
            categoryIds,
            batchSize,
            seqLen,
          });
          return res.data ?? null;
        }
      : undefined,
  };

  return {
    runner,
    dispose: async () => {
      try {
        await rpc({ type: 'dispose' });
      } catch {
        // ignore dispose errors on termination
      }
      worker.terminate();
    },
  };
}

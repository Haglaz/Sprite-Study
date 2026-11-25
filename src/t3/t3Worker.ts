import * as ort from 'onnxruntime-web';
import type { T3InferenceRunner } from './sampler';

// Configure ONNX Runtime Web WASM paths cleanly for both Vite dev & dist production builds
if (typeof ort.env?.wasm === 'object' && ort.env.wasm !== null) {
  const origin =
    typeof self !== 'undefined' && self.location?.origin && self.location.origin !== 'null'
      ? self.location.origin
      : '';
  ort.env.wasm.wasmPaths = origin ? `${origin}/ort/` : '/ort/';
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
}

export interface OnnxWorkerSessionBundle {
  runner: T3InferenceRunner;
  dispose: () => Promise<void>;
}

async function createSessionWithWebGpuFallback(
  modelBytes: ArrayBuffer,
): Promise<{ session: ort.InferenceSession; provider: 'webgpu' | 'wasm' }> {
  const bytesCopy = new Uint8Array(modelBytes.slice(0));
  const gpuApi =
    typeof navigator !== 'undefined'
      ? (navigator as Navigator & { gpu?: { requestAdapter?: () => Promise<unknown | null> } }).gpu
      : undefined;

  if (gpuApi && typeof gpuApi.requestAdapter === 'function') {
    try {
      const adapter = await gpuApi.requestAdapter();
      if (adapter) {
        const session = await ort.InferenceSession.create(bytesCopy, {
          executionProviders: ['webgpu'],
        });
        return { session, provider: 'webgpu' };
      }
    } catch {
      // Fallback cleanly to WASM when WebGPU adapter or kernel is unavailable
    }
  }

  const session = await ort.InferenceSession.create(bytesCopy, {
    executionProviders: ['wasm'],
  });
  return { session, provider: 'wasm' };
}

/**
 * Creates a real onnxruntime-web InferenceSession runner (WebGPU with automatic WASM fallback).
 */
export async function createOnnxRuntimeRunner(params: {
  generatorOnnx: ArrayBuffer;
  rerankerOnnx: ArrayBuffer | null;
}): Promise<OnnxWorkerSessionBundle> {
  const gen = await createSessionWithWebGpuFallback(params.generatorOnnx);
  const rr = params.rerankerOnnx
    ? await createSessionWithWebGpuFallback(params.rerankerOnnx)
    : null;

  const runner: T3InferenceRunner = {
    executionProvider: gen.provider,
    async runGenerator({ tokens, categoryIds, tagVec, batchSize, seqLen, tagDim }) {
      const tTokens = new ort.Tensor('int64', tokens, [batchSize, seqLen]);
      const tCats = new ort.Tensor('int64', categoryIds, [batchSize]);
      const tTags = new ort.Tensor('float32', tagVec, [batchSize, tagDim]);

      const outputs = await gen.session.run({
        tokens: tTokens,
        category_id: tCats,
        tag_vec: tTags,
      });
      const firstKey = gen.session.outputNames[0] ?? Object.keys(outputs)[0];
      const outTensor = outputs[firstKey];
      return new Float32Array(outTensor.data as Float32Array);
    },
    runReranker: rr
      ? async ({ tokens, categoryIds, batchSize, seqLen }) => {
          const tTokens = new ort.Tensor('int64', tokens, [batchSize, seqLen]);
          const tCats = new ort.Tensor('int64', categoryIds, [batchSize]);
          const outputs = await rr.session.run({
            tokens: tTokens,
            category_id: tCats,
          });
          const firstKey = rr.session.outputNames[0] ?? Object.keys(outputs)[0];
          const outTensor = outputs[firstKey];
          return new Float32Array(outTensor.data as Float32Array);
        }
      : undefined,
  };

  return {
    runner,
    dispose: async () => {
      await gen.session.release();
      if (rr) await rr.session.release();
    },
  };
}

// ============================================================================
// DEDICATED WEB WORKER MESSAGE HANDLER (non-blocking UI ONNX sampling)
// ============================================================================

export interface T3WorkerRequest {
  requestId: string;
  type: 'init_and_infer_step' | 'dispose';
  generatorOnnx?: ArrayBuffer;
  rerankerOnnx?: ArrayBuffer | null;
  op?: 'generator' | 'reranker';
  tokens?: BigInt64Array;
  categoryIds?: BigInt64Array;
  tagVec?: Float32Array;
  batchSize?: number;
  seqLen?: number;
  tagDim?: number;
}

export interface T3WorkerResponse {
  requestId: string;
  ok: boolean;
  executionProvider?: 'webgpu' | 'wasm';
  data?: Float32Array;
  error?: string;
}

let workerBundle: OnnxWorkerSessionBundle | null = null;
let loadedKey = '';

function hashQuickBuffer(buf: ArrayBuffer): string {
  const view = new Uint8Array(buf);
  return `${buf.byteLength}:${view[0] ?? 0}:${view[view.length - 1] ?? 0}`;
}

if (typeof self !== 'undefined' && typeof (self as { postMessage?: unknown }).postMessage === 'function' && typeof window === 'undefined') {
  self.onmessage = async (ev: MessageEvent<T3WorkerRequest>) => {
    const msg = ev.data;
    try {
      if (msg.type === 'dispose') {
        if (workerBundle) {
          await workerBundle.dispose();
          workerBundle = null;
          loadedKey = '';
        }
        (self as unknown as Worker).postMessage({ requestId: msg.requestId, ok: true } satisfies T3WorkerResponse);
        return;
      }

      if (msg.generatorOnnx) {
        const nextKey = `${hashQuickBuffer(msg.generatorOnnx)}:${msg.rerankerOnnx ? hashQuickBuffer(msg.rerankerOnnx) : 'none'}`;
        if (!workerBundle || loadedKey !== nextKey) {
          if (workerBundle) await workerBundle.dispose();
          workerBundle = await createOnnxRuntimeRunner({
            generatorOnnx: msg.generatorOnnx,
            rerankerOnnx: msg.rerankerOnnx ?? null,
          });
          loadedKey = nextKey;
        }
      }

      if (!workerBundle) {
        throw new Error('Sessão ONNX do Worker não inicializada.');
      }

      if (msg.op === 'generator' && msg.tokens && msg.categoryIds && msg.tagVec) {
        const out = await workerBundle.runner.runGenerator({
          tokens: msg.tokens,
          categoryIds: msg.categoryIds,
          tagVec: msg.tagVec,
          batchSize: msg.batchSize ?? 2,
          seqLen: msg.seqLen ?? 1024,
          tagDim: msg.tagDim ?? 1,
        });
        (self as unknown as Worker).postMessage(
          {
            requestId: msg.requestId,
            ok: true,
            executionProvider: workerBundle.runner.executionProvider as 'webgpu' | 'wasm',
            data: out,
          } satisfies T3WorkerResponse,
          [out.buffer],
        );
        return;
      }

      if (msg.op === 'reranker' && msg.tokens && msg.categoryIds) {
        const out = workerBundle.runner.runReranker
          ? await workerBundle.runner.runReranker({
              tokens: msg.tokens,
              categoryIds: msg.categoryIds,
              batchSize: msg.batchSize ?? 1,
              seqLen: msg.seqLen ?? 1024,
            })
          : null;
        (self as unknown as Worker).postMessage({
          requestId: msg.requestId,
          ok: true,
          executionProvider: workerBundle.runner.executionProvider as 'webgpu' | 'wasm',
          data: out ?? undefined,
        } satisfies T3WorkerResponse);
        return;
      }

      (self as unknown as Worker).postMessage({
        requestId: msg.requestId,
        ok: true,
        executionProvider: workerBundle.runner.executionProvider as 'webgpu' | 'wasm',
      } satisfies T3WorkerResponse);
    } catch (err) {
      (self as unknown as Worker).postMessage({
        requestId: msg.requestId,
        ok: false,
        error: err instanceof Error ? err.message : 'Erro desconhecido no Worker ONNX.',
      } satisfies T3WorkerResponse);
    }
  };
}

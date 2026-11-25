import { putAndVerify, type StorageBackend } from '../db/persistence';
import {
  T3_MATERIAL_ROLES,
  type T3ModelCard,
  type T3ModelPackageRecord,
  type T3RampVocabulary,
  type T3TestSplitComparisonReport,
} from './types';

export function buildVocabRecordId(datasetKey: string): string {
  return `t3-vocab:${datasetKey}`;
}

export function buildT3ModelRecordId(datasetKey: string): string {
  return `t3-model:${datasetKey}`;
}

export async function loadT3Vocabulary(
  backend: StorageBackend,
  datasetKey: string,
): Promise<T3RampVocabulary | null> {
  return backend.get<T3RampVocabulary>('onnxModels', buildVocabRecordId(datasetKey));
}

/**
 * Honesty Rule #4: Saves the material ramp-role vocabulary (and user role corrections)
 * into IndexedDB `'onnxModels'` with mandatory read-back verification.
 */
export async function saveT3VocabularyVerified(
  backend: StorageBackend,
  vocab: T3RampVocabulary,
): Promise<T3RampVocabulary> {
  const toSave: T3RampVocabulary = {
    ...vocab,
    id: buildVocabRecordId(vocab.datasetKey),
    readBackVerified: true,
  };

  const verified = await putAndVerify(
    backend,
    'onnxModels',
    toSave,
    (rb, wr) =>
      rb.id === wr.id &&
      rb.splitHash === wr.splitHash &&
      rb.vocabSize === wr.vocabSize &&
      rb.overrideCount === wr.overrideCount &&
      rb.entries.length === wr.entries.length,
  );
  return verified.readBack;
}

export async function loadT3ModelPackage(
  backend: StorageBackend,
  datasetKey: string,
): Promise<T3ModelPackageRecord | null> {
  return backend.get<T3ModelPackageRecord>('onnxModels', buildT3ModelRecordId(datasetKey));
}

export async function saveT3ModelPackageVerified(
  backend: StorageBackend,
  pkg: T3ModelPackageRecord,
): Promise<T3ModelPackageRecord> {
  const toSave: T3ModelPackageRecord = {
    ...pkg,
    id: buildT3ModelRecordId(pkg.datasetKey),
    updatedAt: new Date().toISOString(),
    readBackVerified: true,
  };

  const verified = await putAndVerify(
    backend,
    'onnxModels',
    toSave,
    (rb, wr) =>
      rb.id === wr.id &&
      rb.enabled === wr.enabled &&
      rb.modelCard.version === wr.modelCard.version &&
      rb.modelCard.splitHash === wr.modelCard.splitHash &&
      rb.generatorOnnx instanceof ArrayBuffer &&
      rb.generatorOnnx.byteLength === wr.generatorOnnx.byteLength,
  );

  return verified.readBack;
}

/**
 * Parses and strictly validates an imported `t3_model_card.json` file.
 */
export function parseAndValidateT3ModelCard(rawJson: string): T3ModelCard {
  let parsed: unknown;
  try {
    parsed = JSON.parse(rawJson);
  } catch {
    throw new Error('Model card T3 inválido: JSON malformado.');
  }

  if (!parsed || typeof parsed !== 'object') {
    throw new Error('Model card T3 inválido: objeto esperado.');
  }

  const obj = parsed as Partial<T3ModelCard>;
  if (typeof obj.version !== 'string' || !obj.version.trim()) {
    throw new Error('Model card T3 sem campo obrigatório "version".');
  }
  if (typeof obj.steps !== 'number' || !Number.isFinite(obj.steps) || obj.steps < 0) {
    throw new Error('Model card T3 sem campo numérico válido "steps".');
  }
  if (typeof obj.splitHash !== 'string' || !obj.splitHash.trim()) {
    throw new Error('Model card T3 sem campo obrigatório "splitHash".');
  }
  if (!Array.isArray(obj.vocabulary) || obj.vocabulary.length === 0) {
    throw new Error('Model card T3 sem campo obrigatório "vocabulary".');
  }
  for (const entry of obj.vocabulary) {
    if (
      !entry ||
      typeof entry.tokenId !== 'number' ||
      !Array.isArray(entry.rgb) ||
      entry.rgb.length < 3 ||
      !T3_MATERIAL_ROLES.includes(entry.role)
    ) {
      throw new Error('Entrada de vocabulário inválida no model card T3.');
    }
  }
  if (!obj.samplerSettings || typeof obj.samplerSettings !== 'object') {
    throw new Error('Model card T3 sem campo obrigatório "samplerSettings".');
  }

  return obj as T3ModelCard;
}

/**
 * Persists an imported ONNX T3 package (`generator.onnx`, optional `reranker.onnx`, `model_card.json`),
 * keeping T3 disabled by default until test-split verification proves it beats T1/T2.
 */
export async function importT3ModelPackageVerified(
  backend: StorageBackend,
  params: {
    datasetKey: string;
    modelCardJson: string;
    generatorOnnx: ArrayBuffer;
    rerankerOnnx: ArrayBuffer | null;
    generatorFileName: string;
    rerankerFileName: string | null;
    testEvaluation?: T3TestSplitComparisonReport | null;
  },
): Promise<T3ModelPackageRecord> {
  const modelCard = parseAndValidateT3ModelCard(params.modelCardJson);
  if (!(params.generatorOnnx instanceof ArrayBuffer) || params.generatorOnnx.byteLength < 16) {
    throw new Error('Arquivo ONNX do gerador T3 inválido ou vazio.');
  }

  const evalReport = params.testEvaluation ?? null;
  const enabled = evalReport ? evalReport.recommendedEnabled : false;
  const disabledReasonPt = evalReport
    ? evalReport.recommendedEnabled
      ? null
      : evalReport.verdictSummaryPt
    : 'Modelo T3 mantido desativado por padrão até concluir a avaliação contra T1, T2 e piso lixo no split test.';

  const now = new Date().toISOString();
  const record: T3ModelPackageRecord = {
    id: buildT3ModelRecordId(params.datasetKey),
    datasetKey: params.datasetKey,
    modelCard,
    generatorOnnx: params.generatorOnnx,
    rerankerOnnx: params.rerankerOnnx,
    generatorFileName: params.generatorFileName,
    rerankerFileName: params.rerankerFileName,
    enabled,
    disabledReasonPt,
    lastTestEvaluation: evalReport,
    importedAt: now,
    updatedAt: now,
    readBackVerified: true,
  };

  return saveT3ModelPackageVerified(backend, record);
}

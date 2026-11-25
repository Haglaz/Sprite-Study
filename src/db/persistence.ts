import {
  ATELIER_DB_NAME,
  ATELIER_DB_VERSION,
  STORE_NAMES,
  type AtelierStoreName,
  type BackupEnvelope,
  type StoredCriticState,
  type StoredDatasetRecord,
  type StoredDatasetSplit,
} from './schema';

export interface StorageBackend {
  get<T>(store: AtelierStoreName, key: string): Promise<T | null>;
  getAll<T>(store: AtelierStoreName): Promise<T[]>;
  put<T extends { id: string }>(store: AtelierStoreName, value: T): Promise<void>;
  delete(store: AtelierStoreName, key: string): Promise<void>;
  clear(store: AtelierStoreName): Promise<void>;
  clearAll(): Promise<void>;
}

export interface VerifiedWriteResult<T> {
  verified: true;
  readBack: T;
}

export interface PersistentStorageStatus {
  supported: boolean;
  persisted: boolean;
}

/**
 * Requests persistent browser storage so IndexedDB data is not silently evicted.
 */
export async function requestPersistentStorage(): Promise<PersistentStorageStatus> {
  try {
    const storage = globalThis.navigator?.storage;
    if (!storage || typeof storage.persist !== 'function') {
      return { supported: false, persisted: false };
    }
    const already = typeof storage.persisted === 'function' ? await storage.persisted() : false;
    if (already) {
      return { supported: true, persisted: true };
    }
    const granted = await storage.persist();
    return { supported: true, persisted: Boolean(granted) };
  } catch {
    return { supported: false, persisted: false };
  }
}

function openAtelierIdb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === 'undefined') {
      reject(new Error('IndexedDB indisponível neste ambiente.'));
      return;
    }
    const req = indexedDB.open(ATELIER_DB_NAME, ATELIER_DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      for (const storeName of STORE_NAMES) {
        if (!db.objectStoreNames.contains(storeName)) {
          db.createObjectStore(storeName, { keyPath: 'id' });
        }
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error('Falha ao abrir IndexedDB.'));
  });
}

export function createIndexedDbBackend(): StorageBackend {
  return {
    async get<T>(store: AtelierStoreName, key: string): Promise<T | null> {
      const db = await openAtelierIdb();
      try {
        return await new Promise<T | null>((resolve, reject) => {
          const tx = db.transaction(store, 'readonly');
          const req = tx.objectStore(store).get(key);
          req.onsuccess = () => resolve((req.result as T | undefined) ?? null);
          req.onerror = () => reject(req.error ?? new Error(`Falha ao ler ${store}/${key}.`));
        });
      } finally {
        db.close();
      }
    },

    async getAll<T>(store: AtelierStoreName): Promise<T[]> {
      const db = await openAtelierIdb();
      try {
        return await new Promise<T[]>((resolve, reject) => {
          const tx = db.transaction(store, 'readonly');
          const req = tx.objectStore(store).getAll();
          req.onsuccess = () => resolve(Array.isArray(req.result) ? (req.result as T[]) : []);
          req.onerror = () => reject(req.error ?? new Error(`Falha ao listar ${store}.`));
        });
      } finally {
        db.close();
      }
    },

    async put<T extends { id: string }>(store: AtelierStoreName, value: T): Promise<void> {
      const db = await openAtelierIdb();
      try {
        await new Promise<void>((resolve, reject) => {
          const tx = db.transaction(store, 'readwrite');
          tx.objectStore(store).put(value);
          tx.oncomplete = () => resolve();
          tx.onerror = () => reject(tx.error ?? new Error(`Falha ao gravar em ${store}.`));
          tx.onabort = () => reject(tx.error ?? new Error(`Transação abortada em ${store}.`));
        });
      } finally {
        db.close();
      }
    },

    async delete(store: AtelierStoreName, key: string): Promise<void> {
      const db = await openAtelierIdb();
      try {
        await new Promise<void>((resolve, reject) => {
          const tx = db.transaction(store, 'readwrite');
          tx.objectStore(store).delete(key);
          tx.oncomplete = () => resolve();
          tx.onerror = () => reject(tx.error ?? new Error(`Falha ao remover ${store}/${key}.`));
        });
      } finally {
        db.close();
      }
    },

    async clear(store: AtelierStoreName): Promise<void> {
      const db = await openAtelierIdb();
      try {
        await new Promise<void>((resolve, reject) => {
          const tx = db.transaction(store, 'readwrite');
          tx.objectStore(store).clear();
          tx.oncomplete = () => resolve();
          tx.onerror = () => reject(tx.error ?? new Error(`Falha ao limpar ${store}.`));
        });
      } finally {
        db.close();
      }
    },

    async clearAll(): Promise<void> {
      const db = await openAtelierIdb();
      try {
        await new Promise<void>((resolve, reject) => {
          const tx = db.transaction([...STORE_NAMES], 'readwrite');
          for (const storeName of STORE_NAMES) {
            tx.objectStore(storeName).clear();
          }
          tx.oncomplete = () => resolve();
          tx.onerror = () => reject(tx.error ?? new Error('Falha ao limpar banco local.'));
        });
      } finally {
        db.close();
      }
    },
  };
}

/**
 * In-memory StorageBackend used exclusively for deterministic Node unit tests
 * (clones ArrayBuffers on write/read to simulate IndexedDB structured cloning).
 */
export function createMemoryStorageBackend(): StorageBackend {
  const stores = new Map<AtelierStoreName, Map<string, unknown>>();
  for (const name of STORE_NAMES) {
    stores.set(name, new Map());
  }

  return {
    async get<T>(store: AtelierStoreName, key: string): Promise<T | null> {
      const val = stores.get(store)?.get(key);
      return val === undefined ? null : (structuredClone(val) as T);
    },
    async getAll<T>(store: AtelierStoreName): Promise<T[]> {
      const map = stores.get(store);
      if (!map) return [];
      return Array.from(map.values()).map((v) => structuredClone(v) as T);
    },
    async put<T extends { id: string }>(store: AtelierStoreName, value: T): Promise<void> {
      stores.get(store)?.set(value.id, structuredClone(value));
    },
    async delete(store: AtelierStoreName, key: string): Promise<void> {
      stores.get(store)?.delete(key);
    },
    async clear(store: AtelierStoreName): Promise<void> {
      stores.get(store)?.clear();
    },
    async clearAll(): Promise<void> {
      for (const map of stores.values()) {
        map.clear();
      }
    },
  };
}

/**
 * Honesty Rule #4: Writes a record and reads it back immediately from storage.
 * Throws if the read-back is missing or fails the integrity predicate.
 */
export async function putAndVerify<T extends { id: string }>(
  backend: StorageBackend,
  store: AtelierStoreName,
  value: T,
  verifyPredicate?: (readBack: T, written: T) => boolean,
): Promise<VerifiedWriteResult<T>> {
  await backend.put(store, value);
  const readBack = await backend.get<T>(store, value.id);
  if (!readBack) {
    throw new Error(
      `Falha na verificação de leitura (read-back): registro "${value.id}" não foi encontrado em "${store}" após gravação.`,
    );
  }
  if (verifyPredicate && !verifyPredicate(readBack, value)) {
    throw new Error(
      `Falha na verificação de integridade (read-back) do registro "${value.id}" em "${store}".`,
    );
  }
  return { verified: true, readBack };
}

export function buildDatasetKey(datSha256: string, sprSha256: string): string {
  return `${datSha256}:${sprSha256}`;
}

export async function saveActiveDataset(
  backend: StorageBackend,
  record: StoredDatasetRecord,
): Promise<VerifiedWriteResult<StoredDatasetRecord>> {
  return putAndVerify(backend, 'datasets', record, (readBack, written) => {
    return (
      readBack.datSha256 === written.datSha256 &&
      readBack.sprSha256 === written.sprSha256 &&
      readBack.datBuffer instanceof ArrayBuffer &&
      readBack.sprBuffer instanceof ArrayBuffer &&
      readBack.datBuffer.byteLength === written.datBuffer.byteLength &&
      readBack.sprBuffer.byteLength === written.sprBuffer.byteLength
    );
  });
}

export async function loadActiveDataset(
  backend: StorageBackend,
): Promise<StoredDatasetRecord | null> {
  return backend.get<StoredDatasetRecord>('datasets', 'active');
}

export async function saveDatasetSplit(
  backend: StorageBackend,
  split: StoredDatasetSplit,
): Promise<VerifiedWriteResult<StoredDatasetSplit>> {
  return putAndVerify(backend, 'splits', split, (readBack, written) => {
    return (
      readBack.datasetKey === written.datasetKey &&
      readBack.seed === written.seed &&
      readBack.totals.total === written.totals.total &&
      readBack.totals.train === written.totals.train &&
      readBack.totals.validation === written.totals.validation &&
      readBack.totals.test === written.totals.test
    );
  });
}

export async function loadDatasetSplit(
  backend: StorageBackend,
  datasetKey: string,
): Promise<StoredDatasetSplit | null> {
  return backend.get<StoredDatasetSplit>('splits', datasetKey);
}

export async function saveCriticState(
  backend: StorageBackend,
  state: StoredCriticState,
): Promise<VerifiedWriteResult<StoredCriticState>> {
  return putAndVerify(backend, 'criticState', state, (readBack, written) => {
    return (
      readBack.datasetKey === written.datasetKey &&
      readBack.calibratedAt === written.calibratedAt &&
      readBack.seed === written.seed
    );
  });
}

export async function loadCriticState(
  backend: StorageBackend,
  datasetKey: string,
): Promise<StoredCriticState | null> {
  return backend.get<StoredCriticState>('criticState', datasetKey);
}

// --- Backup Export / Import (handles ArrayBuffers cleanly across origins) ---

type BinaryTagType =
  | 'ArrayBuffer'
  | 'Uint8ClampedArray'
  | 'Uint8Array'
  | 'Int16Array'
  | 'Uint16Array'
  | 'Int32Array'
  | 'Uint32Array'
  | 'Float32Array'
  | 'Float64Array';

interface SerializedBinaryValue {
  __type: BinaryTagType;
  base64: string;
}

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x4000;
  let binary = '';
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const slice = bytes.subarray(i, Math.min(i + chunkSize, bytes.length));
    binary += String.fromCharCode(...slice);
  }
  return btoa(binary);
}

function base64ToArrayBuffer(b64: string): ArrayBuffer {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes.buffer;
}

function getTypedArrayTag(value: ArrayBufferView): BinaryTagType {
  if (value instanceof Uint8ClampedArray) return 'Uint8ClampedArray';
  if (value instanceof Uint8Array) return 'Uint8Array';
  if (value instanceof Int16Array) return 'Int16Array';
  if (value instanceof Uint16Array) return 'Uint16Array';
  if (value instanceof Int32Array) return 'Int32Array';
  if (value instanceof Uint32Array) return 'Uint32Array';
  if (value instanceof Float32Array) return 'Float32Array';
  if (value instanceof Float64Array) return 'Float64Array';
  return 'ArrayBuffer';
}

function reconstructBinary(tag: string, buffer: ArrayBuffer): unknown {
  switch (tag) {
    case 'Uint8ClampedArray':
      return new Uint8ClampedArray(buffer);
    case 'Uint8Array':
      return new Uint8Array(buffer);
    case 'Int16Array':
      return new Int16Array(buffer);
    case 'Uint16Array':
      return new Uint16Array(buffer);
    case 'Int32Array':
      return new Int32Array(buffer);
    case 'Uint32Array':
      return new Uint32Array(buffer);
    case 'Float32Array':
      return new Float32Array(buffer);
    case 'Float64Array':
      return new Float64Array(buffer);
    case 'ArrayBuffer':
    default:
      return buffer;
  }
}

function encodeValueForBackup(value: unknown): unknown {
  if (value instanceof ArrayBuffer) {
    const encoded: SerializedBinaryValue = {
      __type: 'ArrayBuffer',
      base64: arrayBufferToBase64(value),
    };
    return encoded;
  }
  if (ArrayBuffer.isView(value)) {
    const copy = new Uint8Array(value.byteLength);
    copy.set(new Uint8Array(value.buffer, value.byteOffset, value.byteLength));
    const encoded: SerializedBinaryValue = {
      __type: getTypedArrayTag(value),
      base64: arrayBufferToBase64(copy.buffer),
    };
    return encoded;
  }
  if (Array.isArray(value)) {
    return value.map(encodeValueForBackup);
  }
  if (value !== null && typeof value === 'object') {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      out[k] = encodeValueForBackup(v);
    }
    return out;
  }
  return value;
}

const BINARY_TAG_SET = new Set<string>([
  'ArrayBuffer',
  'Uint8ClampedArray',
  'Uint8Array',
  'Int16Array',
  'Uint16Array',
  'Int32Array',
  'Uint32Array',
  'Float32Array',
  'Float64Array',
]);

function decodeValueFromBackup(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(decodeValueFromBackup);
  }
  if (value !== null && typeof value === 'object') {
    const obj = value as Record<string, unknown>;
    if (
      typeof obj.__type === 'string' &&
      BINARY_TAG_SET.has(obj.__type) &&
      typeof obj.base64 === 'string'
    ) {
      const buffer = base64ToArrayBuffer(obj.base64);
      return reconstructBinary(obj.__type, buffer);
    }
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(obj)) {
      out[k] = decodeValueFromBackup(v);
    }
    return out;
  }
  return value;
}

export async function exportFullBackupJson(backend: StorageBackend): Promise<string> {
  const stores = {} as Record<AtelierStoreName, unknown[]>;
  for (const storeName of STORE_NAMES) {
    const items = await backend.getAll<unknown>(storeName);
    stores[storeName] = items.map(encodeValueForBackup);
  }
  const envelope: BackupEnvelope = {
    schema: 'atelier-sprite-backup-v2',
    exportedAt: new Date().toISOString(),
    stores,
  };
  return JSON.stringify(envelope);
}

export interface BackupImportResult {
  verified: true;
  importedCounts: Record<AtelierStoreName, number>;
  totalRecords: number;
  exportedAt: string;
}

export async function importFullBackupJson(
  backend: StorageBackend,
  rawJson: string,
): Promise<BackupImportResult> {
  let parsed: unknown;
  try {
    parsed = JSON.parse(rawJson);
  } catch {
    throw new Error('Arquivo de backup inválido: JSON malformado.');
  }

  if (
    !parsed ||
    typeof parsed !== 'object' ||
    (parsed as Partial<BackupEnvelope>).schema !== 'atelier-sprite-backup-v2' ||
    !(parsed as Partial<BackupEnvelope>).stores ||
    typeof (parsed as Partial<BackupEnvelope>).stores !== 'object'
  ) {
    throw new Error('Arquivo de backup não reconhecido (esperado schema atelier-sprite-backup-v2).');
  }

  const envelope = parsed as BackupEnvelope;

  // Decode & validate all records BEFORE clearing existing stores so corrupt files never wipe IndexedDB.
  const decodedByStore = new Map<AtelierStoreName, Array<{ id: string }>>();
  for (const storeName of STORE_NAMES) {
    const rawItems = Array.isArray(envelope.stores[storeName]) ? envelope.stores[storeName] : [];
    const validItems: Array<{ id: string }> = [];
    for (const rawItem of rawItems) {
      const decoded = decodeValueFromBackup(rawItem) as { id?: unknown };
      if (!decoded || typeof decoded.id !== 'string' || !decoded.id) {
        continue;
      }
      validItems.push(decoded as { id: string });
    }
    decodedByStore.set(storeName, validItems);
  }

  await backend.clearAll();

  const importedCounts = {} as Record<AtelierStoreName, number>;
  let totalRecords = 0;

  for (const storeName of STORE_NAMES) {
    const items = decodedByStore.get(storeName) ?? [];
    for (const item of items) {
      await putAndVerify(backend, storeName, item);
    }
    // Verify count matches read-back store length
    const allInStore = await backend.getAll<unknown>(storeName);
    if (allInStore.length !== items.length) {
      throw new Error(
        `Falha na verificação read-back ao importar "${storeName}" (esperado ${items.length}, lido ${allInStore.length}).`,
      );
    }
    importedCounts[storeName] = items.length;
    totalRecords += items.length;
  }

  return {
    verified: true,
    importedCounts,
    totalRecords,
    exportedAt: envelope.exportedAt || new Date().toISOString(),
  };
}

export const defaultDbBackend: StorageBackend = createIndexedDbBackend();

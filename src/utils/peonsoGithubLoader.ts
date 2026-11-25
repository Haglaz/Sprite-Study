export interface GithubTreeEntry {
  path: string;
  type?: string;
  size?: number;
}

export interface DiscoveredPeonsoPackage {
  owner: 'peonso';
  repo: string;
  branch: string;
  datPath: string;
  sprPath: string;
  datFileName: string;
  sprFileName: string;
  datRawUrl: string;
  sprRawUrl: string;
  repoHtmlUrl: string;
  discoveryMode: 'github_api' | 'verified_fallback';
}

export interface PeonsoFetchProgress {
  stage: 'searching' | 'downloading_dat' | 'downloading_spr' | 'done';
  message: string;
  receivedBytes?: number;
  totalBytes?: number;
}

export const PEONSO_VERIFIED_FALLBACK: DiscoveredPeonsoPackage = {
  owner: 'peonso',
  repo: 'opentibia_sprite_pack',
  branch: 'master',
  datPath: 'client_files/otsp.dat',
  sprPath: 'client_files/otsp.spr',
  datFileName: 'otsp.dat',
  sprFileName: 'otsp.spr',
  datRawUrl:
    'https://raw.githubusercontent.com/peonso/opentibia_sprite_pack/master/client_files/otsp.dat',
  sprRawUrl:
    'https://raw.githubusercontent.com/peonso/opentibia_sprite_pack/master/client_files/otsp.spr',
  repoHtmlUrl: 'https://github.com/peonso/opentibia_sprite_pack/tree/master/client_files',
  discoveryMode: 'verified_fallback',
};

const PEONSO_CANDIDATE_REPOS: ReadonlyArray<{ repo: string; branch: string }> = [
  { repo: 'opentibia_sprite_pack', branch: 'master' },
  { repo: 'otclient772', branch: 'master' },
  { repo: 'tfs1041_oldschool', branch: 'master' },
  { repo: 'oldguilcera', branch: 'master' },
  { repo: 'tibialegacyserver', branch: 'master' },
];

const SAFE_PEONSO_HOSTS = new Set([
  'api.github.com',
  'raw.githubusercontent.com',
  'github.com',
]);

/**
 * Strictly validates that an outbound URL targets only HTTPS GitHub hosts
 * under the `peonso` account, blocking path traversal and unexpected hosts.
 */
export function isSafePeonsoGithubUrl(rawUrl: string): boolean {
  try {
    if (rawUrl.includes('..') || rawUrl.includes('\\') || /%2e%2e/i.test(rawUrl)) {
      return false;
    }
    const parsed = new URL(rawUrl);
    if (parsed.protocol !== 'https:') return false;
    if (!SAFE_PEONSO_HOSTS.has(parsed.hostname)) return false;

    const decodedPath = decodeURIComponent(parsed.pathname);
    if (decodedPath.includes('..') || decodedPath.includes('\\')) return false;

    if (parsed.hostname === 'api.github.com') {
      return decodedPath.startsWith('/repos/peonso/') || decodedPath === '/users/peonso/repos';
    }
    return decodedPath.startsWith('/peonso/');
  } catch {
    return false;
  }
}

function getDirAndBase(filePath: string): { dir: string; base: string; fileName: string } {
  const normalized = filePath.replace(/^\/+/, '');
  const lastSlash = normalized.lastIndexOf('/');
  const dir = lastSlash >= 0 ? normalized.slice(0, lastSlash) : '';
  const fileName = lastSlash >= 0 ? normalized.slice(lastSlash + 1) : normalized;
  const base = fileName.replace(/\.(dat|spr)$/i, '').toLowerCase();
  return { dir, base, fileName };
}

/**
 * Finds the best matching `.dat` + `.spr` pair in a GitHub repository tree,
 * preferring files in the same folder with matching base names (e.g. `otsp.dat` + `otsp.spr`
 * or `Tibia.dat` + `Tibia.spr`).
 */
export function findDatSprPairsInTree(
  entries: GithubTreeEntry[],
): { datPath: string; sprPath: string } | null {
  const blobs = entries.filter(
    (e) =>
      typeof e.path === 'string' &&
      !e.path.includes('..') &&
      (e.type === undefined || e.type === 'blob'),
  );

  const datFiles = blobs.filter((e) => /\.dat$/i.test(e.path));
  const sprFiles = blobs.filter((e) => /\.spr$/i.test(e.path));
  if (datFiles.length === 0 || sprFiles.length === 0) return null;

  // 1. Exact same folder + exact same basename (e.g., client_files/otsp.dat + client_files/otsp.spr)
  for (const d of datFiles) {
    const dMeta = getDirAndBase(d.path);
    const exact = sprFiles.find((s) => {
      const sMeta = getDirAndBase(s.path);
      return sMeta.dir === dMeta.dir && sMeta.base === dMeta.base;
    });
    if (exact) {
      return { datPath: d.path, sprPath: exact.path };
    }
  }

  // 2. Same folder fallback
  for (const d of datFiles) {
    const dMeta = getDirAndBase(d.path);
    const sameDir = sprFiles.find((s) => getDirAndBase(s.path).dir === dMeta.dir);
    if (sameDir) {
      return { datPath: d.path, sprPath: sameDir.path };
    }
  }

  // 3. First available pair
  return { datPath: datFiles[0].path, sprPath: sprFiles[0].path };
}

function encodeGithubPathSegments(filePath: string): string {
  return filePath
    .split('/')
    .filter((seg) => seg.length > 0 && seg !== '.' && seg !== '..')
    .map((seg) => encodeURIComponent(seg))
    .join('/');
}

/**
 * Searches `peonso`'s public GitHub repositories for a `.dat` and `.spr` pair,
 * falling back cleanly to the verified `peonso/opentibia_sprite_pack` package if
 * the unauthenticated GitHub API rate limit is reached.
 */
export async function discoverPeonsoDatSprPackage(
  fetchImpl: typeof fetch = fetch,
): Promise<DiscoveredPeonsoPackage> {
  for (const candidate of PEONSO_CANDIDATE_REPOS) {
    const treeUrl = `https://api.github.com/repos/peonso/${candidate.repo}/git/trees/${candidate.branch}?recursive=1`;
    if (!isSafePeonsoGithubUrl(treeUrl)) continue;

    try {
      const res = await fetchImpl(treeUrl, {
        headers: { Accept: 'application/vnd.github+json' },
      });
      if (!res.ok) {
        if (res.status === 403 || res.status === 429) {
          // Unauthenticated GitHub API rate limit -> use verified direct raw URLs
          return PEONSO_VERIFIED_FALLBACK;
        }
        continue;
      }

      const data = (await res.json()) as { tree?: GithubTreeEntry[] };
      if (!Array.isArray(data.tree)) continue;

      const pair = findDatSprPairsInTree(data.tree);
      if (!pair) continue;

      const encodedDat = encodeGithubPathSegments(pair.datPath);
      const encodedSpr = encodeGithubPathSegments(pair.sprPath);
      const datRawUrl = `https://raw.githubusercontent.com/peonso/${candidate.repo}/${candidate.branch}/${encodedDat}`;
      const sprRawUrl = `https://raw.githubusercontent.com/peonso/${candidate.repo}/${candidate.branch}/${encodedSpr}`;
      const dir = getDirAndBase(pair.datPath).dir;
      const repoHtmlUrl = dir
        ? `https://github.com/peonso/${candidate.repo}/tree/${candidate.branch}/${encodeGithubPathSegments(dir)}`
        : `https://github.com/peonso/${candidate.repo}`;

      if (
        !isSafePeonsoGithubUrl(datRawUrl) ||
        !isSafePeonsoGithubUrl(sprRawUrl) ||
        !isSafePeonsoGithubUrl(repoHtmlUrl)
      ) {
        continue;
      }

      return {
        owner: 'peonso',
        repo: candidate.repo,
        branch: candidate.branch,
        datPath: pair.datPath,
        sprPath: pair.sprPath,
        datFileName: getDirAndBase(pair.datPath).fileName,
        sprFileName: getDirAndBase(pair.sprPath).fileName,
        datRawUrl,
        sprRawUrl,
        repoHtmlUrl,
        discoveryMode: 'github_api',
      };
    } catch {
      // Try next candidate or fallback below
    }
  }

  return PEONSO_VERIFIED_FALLBACK;
}

async function fetchBinaryWithProgress(
  url: string,
  onChunk: (receivedBytes: number, totalBytes?: number) => void,
  fetchImpl: typeof fetch = fetch,
): Promise<ArrayBuffer> {
  if (!isSafePeonsoGithubUrl(url)) {
    throw new Error('URL de download fora do repositório autorizado do peonso no GitHub.');
  }

  const res = await fetchImpl(url);
  if (!res.ok) {
    throw new Error(`Falha ao baixar arquivo do GitHub (HTTP ${res.status}).`);
  }

  const totalHeader = res.headers.get('content-length');
  const totalBytes = totalHeader ? Number(totalHeader) : undefined;
  const safeTotal =
    typeof totalBytes === 'number' && Number.isFinite(totalBytes) && totalBytes > 0
      ? totalBytes
      : undefined;

  if (!res.body || typeof res.body.getReader !== 'function') {
    const ab = await res.arrayBuffer();
    onChunk(ab.byteLength, safeTotal ?? ab.byteLength);
    return ab;
  }

  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(value);
      received += value.byteLength;
      onChunk(received, safeTotal);
    }
  }

  const outBuffer = new ArrayBuffer(received);
  const merged = new Uint8Array(outBuffer);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return outBuffer;
}

export async function downloadPeonsoTibiaFiles(params: {
  onProgress?: (progress: PeonsoFetchProgress) => void;
  fetchImpl?: typeof fetch;
}): Promise<{
  pkg: DiscoveredPeonsoPackage;
  datFile: File;
  sprFile: File;
}> {
  const fetchImpl = params.fetchImpl ?? fetch;

  params.onProgress?.({
    stage: 'searching',
    message: 'Localizando .dat e .spr nos repositórios do peonso no GitHub...',
  });

  const pkg = await discoverPeonsoDatSprPackage(fetchImpl);

  params.onProgress?.({
    stage: 'downloading_dat',
    message: `Encontrado em peonso/${pkg.repo}! Baixando ${pkg.datFileName}...`,
  });

  const datBytes = await fetchBinaryWithProgress(
    pkg.datRawUrl,
    (receivedBytes, totalBytes) => {
      params.onProgress?.({
        stage: 'downloading_dat',
        message: `Baixando ${pkg.datFileName} de peonso/${pkg.repo}...`,
        receivedBytes,
        totalBytes,
      });
    },
    fetchImpl,
  );

  params.onProgress?.({
    stage: 'downloading_spr',
    message: `Baixando ${pkg.sprFileName} de peonso/${pkg.repo}...`,
    receivedBytes: 0,
  });

  const sprBytes = await fetchBinaryWithProgress(
    pkg.sprRawUrl,
    (receivedBytes, totalBytes) => {
      const mbDone = (receivedBytes / (1024 * 1024)).toFixed(1);
      const mbTotal = totalBytes ? ` / ${(totalBytes / (1024 * 1024)).toFixed(1)} MB` : ' MB';
      params.onProgress?.({
        stage: 'downloading_spr',
        message: `Baixando ${pkg.sprFileName} de peonso/${pkg.repo} (${mbDone}${mbTotal})...`,
        receivedBytes,
        totalBytes,
      });
    },
    fetchImpl,
  );

  params.onProgress?.({
    stage: 'done',
    message: `Arquivos ${pkg.datFileName} e ${pkg.sprFileName} baixados de peonso/${pkg.repo}.`,
  });

  const datFile = new File([datBytes], pkg.datFileName, { type: 'application/octet-stream' });
  const sprFile = new File([sprBytes], pkg.sprFileName, { type: 'application/octet-stream' });

  return { pkg, datFile, sprFile };
}

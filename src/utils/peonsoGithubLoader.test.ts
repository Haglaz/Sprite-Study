import test from 'node:test';
import assert from 'node:assert/strict';
import {
  discoverPeonsoDatSprPackage,
  findDatSprPairsInTree,
  isSafePeonsoGithubUrl,
  PEONSO_VERIFIED_FALLBACK,
} from './peonsoGithubLoader';

test('isSafePeonsoGithubUrl allows only HTTPS peonso GitHub URLs and rejects unsafe hosts/paths', () => {
  assert.equal(
    isSafePeonsoGithubUrl(
      'https://raw.githubusercontent.com/peonso/opentibia_sprite_pack/master/client_files/otsp.spr',
    ),
    true,
  );
  assert.equal(
    isSafePeonsoGithubUrl(
      'https://api.github.com/repos/peonso/opentibia_sprite_pack/git/trees/master?recursive=1',
    ),
    true,
  );
  assert.equal(
    isSafePeonsoGithubUrl('https://github.com/peonso/opentibia_sprite_pack/tree/master/client_files'),
    true,
  );

  // Reject other users, hosts, protocols, and traversal
  assert.equal(
    isSafePeonsoGithubUrl('https://raw.githubusercontent.com/otheruser/repo/main/Tibia.dat'),
    false,
  );
  assert.equal(
    isSafePeonsoGithubUrl('http://raw.githubusercontent.com/peonso/opentibia_sprite_pack/master/a.dat'),
    false,
  );
  assert.equal(isSafePeonsoGithubUrl('https://127.0.0.1/peonso/Tibia.dat'), false);
  assert.equal(isSafePeonsoGithubUrl('javascript:alert(1)'), false);
  assert.equal(
    isSafePeonsoGithubUrl(
      'https://raw.githubusercontent.com/peonso/opentibia_sprite_pack/master/../../etc/passwd',
    ),
    false,
  );
});

test('findDatSprPairsInTree prefers matching folder and basename pairs', () => {
  const pair = findDatSprPairsInTree([
    { path: 'README.md', type: 'blob' },
    { path: 'other/random.dat', type: 'blob' },
    { path: 'client_files/otsp.dat', type: 'blob', size: 86625 },
    { path: 'client_files/otsp.spr', type: 'blob', size: 9280502 },
  ]);

  assert.deepEqual(pair, {
    datPath: 'client_files/otsp.dat',
    sprPath: 'client_files/otsp.spr',
  });

  assert.equal(findDatSprPairsInTree([{ path: 'only.dat', type: 'blob' }]), null);
});

test('discoverPeonsoDatSprPackage discovers pair from GitHub API and falls back cleanly on 403 rate limit', async () => {
  const mockOkFetch = (async () =>
    new Response(
      JSON.stringify({
        tree: [
          { path: 'client_files/otsp.dat', type: 'blob' },
          { path: 'client_files/otsp.spr', type: 'blob' },
        ],
      }),
      { status: 200, headers: { 'Content-Type': 'application/json' } },
    )) as unknown as typeof fetch;

  const discovered = await discoverPeonsoDatSprPackage(mockOkFetch);
  assert.equal(discovered.discoveryMode, 'github_api');
  assert.equal(discovered.repo, 'opentibia_sprite_pack');
  assert.equal(discovered.datFileName, 'otsp.dat');
  assert.equal(discovered.sprFileName, 'otsp.spr');

  const mockRateLimitedFetch = (async () =>
    new Response(JSON.stringify({ message: 'API rate limit exceeded' }), {
      status: 403,
    })) as unknown as typeof fetch;

  const fallback = await discoverPeonsoDatSprPackage(mockRateLimitedFetch);
  assert.deepEqual(fallback, PEONSO_VERIFIED_FALLBACK);
});

import { putAndVerify, type StorageBackend } from '../db/persistence';
import type { CoevolutionChampion } from '../evolution/types';
import type { ItemCategory } from '../types/studio';
import { computeEloUpdate, DEFAULT_ELO } from './elo';
import type { AgentProfileRecord } from './types';

function makeAgentId(datasetKey: string, slug: string): string {
  return `${datasetKey}:${slug}`;
}

export async function loadAgents(
  backend: StorageBackend,
  datasetKey: string,
): Promise<AgentProfileRecord[]> {
  const all = await backend.getAll<AgentProfileRecord>('agents');
  return all
    .filter((a) => a.datasetKey === datasetKey)
    .sort((a, b) => (b.elo !== a.elo ? b.elo - a.elo : a.name.localeCompare(b.name)));
}

/**
 * Seeds built-in competing agents (T1, T2, T1 Conservador, plus any Phase 3 promoted Champion)
 * into IndexedDB `'agents'` without overwriting existing Elo/history, verifying every write via read-back.
 */
export async function ensureDefaultAgentsVerified(
  backend: StorageBackend,
  datasetKey: string,
  championsByCategory: Partial<Record<ItemCategory, CoevolutionChampion>> = {},
): Promise<AgentProfileRecord[]> {
  const existing = await loadAgents(backend, datasetKey);
  const byId = new Map(existing.map((a) => [a.id, a]));
  const hasAnyActive = existing.some((a) => a.isActive);
  const now = new Date().toISOString();

  const desired: Array<Omit<AgentProfileRecord, 'elo' | 'wins' | 'losses' | 'draws' | 'matchesPlayed' | 'matchHistory' | 'createdAt' | 'updatedAt' | 'readBackVerified'>> = [
    {
      id: makeAgentId(datasetKey, 't1-recolor-ops'),
      datasetKey,
      name: 'Agente T1 · Recolor Semântico + Reparo',
      descriptionPt:
        'Recupera doadores do split train, recoloriza preservando ordem de luminância e aplica diff até 28 px + reparo do Crítico v1.',
      tier: 'T1',
      repairPolicy: 'critic_ops',
      maxDiffPixels: 28,
      isActive: !hasAnyActive,
    },
    {
      id: makeAgentId(datasetKey, 't2-composite-ops'),
      datasetKey,
      name: 'Agente T2 · Compósito de Doadores + Reparo',
      descriptionPt:
        'Combina regiões estruturais de 2 doadores reais na grade 1:1 com paleta unificada, contorno 1px, diff até 32 px e reparo do Crítico v1.',
      tier: 'T2',
      repairPolicy: 'critic_ops',
      maxDiffPixels: 32,
      isActive: false,
    },
    {
      id: makeAgentId(datasetKey, 't1-conservative'),
      datasetKey,
      name: 'Agente T1 · Conservador (12 px diff)',
      descriptionPt:
        'Baseline estrito de baixa intervenção: recolor em rampa discreta com limite rígido de 12 px de diff e sem reparo pós-geração.',
      tier: 'T1',
      repairPolicy: 'none',
      maxDiffPixels: 12,
      isActive: false,
    },
  ];

  // Register any promoted Phase 3 co-evolution Champions as explicit competing agents
  for (const [cat, champ] of Object.entries(championsByCategory) as Array<
    [ItemCategory, CoevolutionChampion | undefined]
  >) {
    if (!champ) continue;
    desired.push({
      id: makeAgentId(datasetKey, `champion-${cat.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g, '')}`),
      datasetKey,
      name: `Campeão Co-Evolução (${cat} · Gen ${champ.generation})`,
      descriptionPt: `Agente T2 com pipeline de reparo promovido na Fase 3 (Receita GA + pincel NEAT, val ${champ.valStats.medianRestoredScore.toFixed(1)}).`,
      tier: 'T2',
      repairPolicy: 'champion_coevolution',
      maxDiffPixels: 28,
      championCategory: cat,
      isActive: false,
    });
  }

  for (const spec of desired) {
    if (byId.has(spec.id)) continue;
    const record: AgentProfileRecord = {
      ...spec,
      elo: DEFAULT_ELO,
      wins: 0,
      losses: 0,
      draws: 0,
      matchesPlayed: 0,
      matchHistory: [],
      createdAt: now,
      updatedAt: now,
      readBackVerified: true,
    };
    await putAndVerify(
      backend,
      'agents',
      record,
      (rb, wr) => rb.id === wr.id && rb.elo === wr.elo && rb.isActive === wr.isActive,
    );
  }

  return loadAgents(backend, datasetKey);
}

/**
 * Records the result of a Blind A/B Elo duel between two agents, updating both ratings,
 * win/loss/draw totals, and match histories with mandatory read-back verification.
 */
export async function recordBlindEloMatchVerified(
  backend: StorageBackend,
  params: {
    datasetKey: string;
    agentAId: string;
    agentBId: string;
    winner: 'A' | 'B' | 'draw';
    prompt: string;
  },
): Promise<{
  agentA: AgentProfileRecord;
  agentB: AgentProfileRecord;
  deltaA: number;
  deltaB: number;
  allAgents: AgentProfileRecord[];
}> {
  const { datasetKey, agentAId, agentBId, winner, prompt } = params;
  if (agentAId === agentBId) {
    throw new Error('Duelo Elo exige dois agentes distintos.');
  }

  const a = await backend.get<AgentProfileRecord>('agents', agentAId);
  const b = await backend.get<AgentProfileRecord>('agents', agentBId);
  if (!a || !b) {
    throw new Error('Um dos agentes do duelo Elo não foi encontrado no IndexedDB.');
  }

  const outcomeA: 1 | 0.5 | 0 = winner === 'A' ? 1 : winner === 'B' ? 0 : 0.5;
  const upd = computeEloUpdate(a.elo, b.elo, outcomeA);
  const now = new Date().toISOString();
  const matchId = `elo-${Date.now().toString(36)}-${Math.floor(Math.random() * 9999).toString(36)}`;

  const nextA: AgentProfileRecord = {
    ...a,
    elo: upd.nextA,
    wins: a.wins + (winner === 'A' ? 1 : 0),
    losses: a.losses + (winner === 'B' ? 1 : 0),
    draws: a.draws + (winner === 'draw' ? 1 : 0),
    matchesPlayed: a.matchesPlayed + 1,
    matchHistory: [
      {
        matchId,
        opponentId: b.id,
        opponentName: b.name,
        outcome: winner === 'A' ? 'win' : winner === 'B' ? 'loss' : 'draw',
        eloBefore: a.elo,
        eloAfter: upd.nextA,
        eloDelta: upd.deltaA,
        prompt,
        playedAt: now,
      },
      ...a.matchHistory.slice(0, 29),
    ],
    updatedAt: now,
    readBackVerified: true,
  };

  const nextB: AgentProfileRecord = {
    ...b,
    elo: upd.nextB,
    wins: b.wins + (winner === 'B' ? 1 : 0),
    losses: b.losses + (winner === 'A' ? 1 : 0),
    draws: b.draws + (winner === 'draw' ? 1 : 0),
    matchesPlayed: b.matchesPlayed + 1,
    matchHistory: [
      {
        matchId,
        opponentId: a.id,
        opponentName: a.name,
        outcome: winner === 'B' ? 'win' : winner === 'A' ? 'loss' : 'draw',
        eloBefore: b.elo,
        eloAfter: upd.nextB,
        eloDelta: upd.deltaB,
        prompt,
        playedAt: now,
      },
      ...b.matchHistory.slice(0, 29),
    ],
    updatedAt: now,
    readBackVerified: true,
  };

  const verA = await putAndVerify(
    backend,
    'agents',
    nextA,
    (rb, wr) => rb.elo === wr.elo && rb.matchesPlayed === wr.matchesPlayed,
  );
  const verB = await putAndVerify(
    backend,
    'agents',
    nextB,
    (rb, wr) => rb.elo === wr.elo && rb.matchesPlayed === wr.matchesPlayed,
  );

  const allAgents = await loadAgents(backend, datasetKey);
  return {
    agentA: verA.readBack,
    agentB: verB.readBack,
    deltaA: upd.deltaA,
    deltaB: upd.deltaB,
    allAgents,
  };
}

/**
 * "Tornar ativo" — Promotes the selected agent as the single active studio agent for `datasetKey`,
 * deactivating all others and verifying every IndexedDB write via read-back.
 */
export async function setActiveAgentVerified(
  backend: StorageBackend,
  datasetKey: string,
  agentId: string,
): Promise<{
  activeAgent: AgentProfileRecord;
  allAgents: AgentProfileRecord[];
}> {
  const existing = await loadAgents(backend, datasetKey);
  const target = existing.find((a) => a.id === agentId);
  if (!target) {
    throw new Error(`Agente "${agentId}" não encontrado para ativação.`);
  }

  const now = new Date().toISOString();
  let activeVerified: AgentProfileRecord | null = null;

  for (const agent of existing) {
    const shouldBeActive = agent.id === agentId;
    if (agent.isActive === shouldBeActive && !shouldBeActive) continue;
    const updated: AgentProfileRecord = {
      ...agent,
      isActive: shouldBeActive,
      updatedAt: now,
      readBackVerified: true,
    };
    const ver = await putAndVerify(
      backend,
      'agents',
      updated,
      (rb, wr) => rb.id === wr.id && rb.isActive === wr.isActive,
    );
    if (shouldBeActive) activeVerified = ver.readBack;
  }

  const allAgents = await loadAgents(backend, datasetKey);
  return {
    activeAgent: activeVerified ?? allAgents.find((a) => a.id === agentId)!,
    allAgents,
  };
}

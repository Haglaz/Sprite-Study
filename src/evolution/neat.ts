import type {
  InnovationTrackerState,
  NeatActivation,
  NeatConfig,
  NeatConnectionGene,
  NeatGenome,
  NeatNodeGene,
  NeatPopulationState,
  NeatSpecies,
} from './types';

// ============================================================================
// DETERMINISTIC STATEFUL PRNG (Mulberry32 step with serializable state)
// ============================================================================

export function nextRand(rng: { state: number }): number {
  rng.state = (rng.state + 0x6d2b79f5) | 0;
  let t = Math.imul(rng.state ^ (rng.state >>> 15), 1 | rng.state);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

export function nextGaussian(rng: { state: number }): number {
  const u1 = Math.max(1e-7, nextRand(rng));
  const u2 = nextRand(rng);
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ============================================================================
// DEFAULT CONFIG & INNOVATION TRACKER
// ============================================================================

export function createDefaultNeatConfig(overrides: Partial<NeatConfig> = {}): NeatConfig {
  return {
    populationSize: 20,
    inputCount: 10,
    outputCount: 6,
    c1Excess: 1.0,
    c2Disjoint: 1.0,
    c3Weight: 0.4,
    compatibilityThreshold: 2.6,
    targetSpeciesCount: 4,
    weightMutateRate: 0.8,
    weightPerturbScale: 0.35,
    weightReplaceRate: 0.12,
    addConnectionRate: 0.22,
    addNodeRate: 0.08,
    toggleEnableRate: 0.05,
    crossoverRate: 0.75,
    elitismPerSpecies: 1,
    stagnationLimit: 12,
    ...overrides,
  };
}

export function createInnovationTracker(inputCount: number, outputCount: number): InnovationTrackerState {
  // Node IDs: 0..inputCount-1 = inputs, inputCount = bias, inputCount+1..inputCount+outputCount = outputs
  const nextNodeId = inputCount + 1 + outputCount;
  return {
    nextInnovation: 1,
    nextNodeId,
    connectionInnovations: {},
    splitInnovations: {},
  };
}

export function getOrCreateConnectionInnovation(
  tracker: InnovationTrackerState,
  inNode: number,
  outNode: number,
): number {
  const key = `${inNode}->${outNode}`;
  const existing = tracker.connectionInnovations[key];
  if (existing !== undefined) return existing;
  const innov = tracker.nextInnovation++;
  tracker.connectionInnovations[key] = innov;
  return innov;
}

export function getOrCreateSplitInnovation(
  tracker: InnovationTrackerState,
  conn: NeatConnectionGene,
): { nodeId: number; inInnov: number; outInnov: number } {
  const key = `${conn.innovation}`;
  const existing = tracker.splitInnovations[key];
  if (existing) return existing;
  const nodeId = tracker.nextNodeId++;
  const inInnov = getOrCreateConnectionInnovation(tracker, conn.inNode, nodeId);
  const outInnov = getOrCreateConnectionInnovation(tracker, nodeId, conn.outNode);
  const created = { nodeId, inInnov, outInnov };
  tracker.splitInnovations[key] = created;
  return created;
}

// ============================================================================
// GENOME CONSTRUCTION, CLONING & TOPOLOGICAL EVALUATION
// ============================================================================

export function cloneNeatGenome(genome: NeatGenome): NeatGenome {
  return {
    id: genome.id,
    nodes: genome.nodes.map((n) => ({ ...n })),
    connections: genome.connections.map((c) => ({ ...c })),
    fitness: genome.fitness,
    adjustedFitness: genome.adjustedFitness,
    speciesId: genome.speciesId,
  };
}

export function createMinimalNeatGenome(
  id: string,
  inputCount: number,
  outputCount: number,
  tracker: InnovationTrackerState,
  rng: { state: number },
  connectProbability = 0.65,
): NeatGenome {
  const nodes: NeatNodeGene[] = [];
  for (let i = 0; i < inputCount; i++) {
    nodes.push({ id: i, type: 'input', activation: 'linear', layer: 0 });
  }
  const biasId = inputCount;
  nodes.push({ id: biasId, type: 'bias', activation: 'linear', layer: 0 });

  const outputStart = inputCount + 1;
  for (let o = 0; o < outputCount; o++) {
    nodes.push({ id: outputStart + o, type: 'output', activation: 'tanh', layer: 1 });
  }

  const connections: NeatConnectionGene[] = [];
  // Connect bias + subset of inputs to every output
  for (let o = 0; o < outputCount; o++) {
    const outId = outputStart + o;
    let connected = 0;
    for (let src = 0; src <= biasId; src++) {
      if (src === biasId || nextRand(rng) < connectProbability) {
        const innovation = getOrCreateConnectionInnovation(tracker, src, outId);
        connections.push({
          innovation,
          inNode: src,
          outNode: outId,
          weight: (nextRand(rng) * 2 - 1) * 1.1,
          enabled: true,
        });
        connected++;
      }
    }
    if (connected === 0) {
      const src = Math.floor(nextRand(rng) * inputCount);
      const innovation = getOrCreateConnectionInnovation(tracker, src, outId);
      connections.push({
        innovation,
        inNode: src,
        outNode: outId,
        weight: (nextRand(rng) * 2 - 1) * 1.1,
        enabled: true,
      });
    }
  }

  return {
    id,
    nodes,
    connections,
    fitness: 0,
    adjustedFitness: 0,
    speciesId: null,
  };
}

function applyActivation(fn: NeatActivation, x: number): number {
  switch (fn) {
    case 'sigmoid':
      return 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, x * 4.9))));
    case 'relu':
      return x > 0 ? x : 0;
    case 'linear':
      return x;
    case 'tanh':
    default:
      return Math.tanh(x);
  }
}

/**
 * Evaluates a feed-forward NEAT genome in topological layer order.
 * Guaranteed deterministic and cycle-safe.
 */
export function evaluateNeatGenome(genome: NeatGenome, inputs: number[]): number[] {
  const values = new Map<number, number>();
  const incoming = new Map<number, NeatConnectionGene[]>();

  let inputIdx = 0;
  for (const node of genome.nodes) {
    if (node.type === 'input') {
      values.set(node.id, inputs[inputIdx] ?? 0);
      inputIdx++;
    } else if (node.type === 'bias') {
      values.set(node.id, 1.0);
    } else {
      values.set(node.id, 0);
      incoming.set(node.id, []);
    }
  }

  for (const conn of genome.connections) {
    if (!conn.enabled) continue;
    const list = incoming.get(conn.outNode);
    if (list) list.push(conn);
  }

  const computeNodes = genome.nodes
    .filter((n) => n.type === 'hidden' || n.type === 'output')
    .sort((a, b) => (a.layer !== b.layer ? a.layer - b.layer : a.id - b.id));

  for (const node of computeNodes) {
    const conns = incoming.get(node.id);
    let sum = 0;
    if (conns) {
      for (const c of conns) {
        sum += (values.get(c.inNode) ?? 0) * c.weight;
      }
    }
    values.set(node.id, applyActivation(node.activation, sum));
  }

  const outputs = genome.nodes
    .filter((n) => n.type === 'output')
    .sort((a, b) => a.id - b.id)
    .map((n) => values.get(n.id) ?? 0);

  return outputs;
}

// ============================================================================
// STRUCTURAL & WEIGHT MUTATIONS (ACYCLIC GUARANTEED)
// ============================================================================

export function wouldCreateCycle(genome: NeatGenome, inNode: number, outNode: number): boolean {
  if (inNode === outNode) return true;
  const nodeById = new Map(genome.nodes.map((n) => [n.id, n]));
  const src = nodeById.get(inNode);
  const dst = nodeById.get(outNode);
  if (!src || !dst) return true;
  // Strict layer ordering guarantees acyclicity in O(1) + BFS fallback safety
  if (src.layer >= dst.layer) return true;

  // Defensive BFS from outNode to verify inNode is unreachable
  const adj = new Map<number, number[]>();
  for (const c of genome.connections) {
    const arr = adj.get(c.inNode) ?? [];
    arr.push(c.outNode);
    adj.set(c.inNode, arr);
  }
  const visited = new Set<number>([outNode]);
  const queue = [outNode];
  while (queue.length > 0) {
    const cur = queue.shift()!;
    if (cur === inNode) return true;
    for (const nxt of adj.get(cur) ?? []) {
      if (!visited.has(nxt)) {
        visited.add(nxt);
        queue.push(nxt);
      }
    }
  }
  return false;
}

export function mutateAddConnection(
  genome: NeatGenome,
  tracker: InnovationTrackerState,
  rng: { state: number },
): boolean {
  const existingPairs = new Set(genome.connections.map((c) => `${c.inNode}->${c.outNode}`));
  const candidates: Array<[number, number]> = [];

  for (const a of genome.nodes) {
    if (a.type === 'output') continue;
    for (const b of genome.nodes) {
      if (b.type === 'input' || b.type === 'bias') continue;
      if (a.layer >= b.layer) continue;
      const key = `${a.id}->${b.id}`;
      if (existingPairs.has(key)) continue;
      if (!wouldCreateCycle(genome, a.id, b.id)) {
        candidates.push([a.id, b.id]);
      }
    }
  }

  if (candidates.length === 0) return false;
  const [inNode, outNode] = candidates[Math.floor(nextRand(rng) * candidates.length)];
  const innovation = getOrCreateConnectionInnovation(tracker, inNode, outNode);
  genome.connections.push({
    innovation,
    inNode,
    outNode,
    weight: (nextRand(rng) * 2 - 1) * 1.2,
    enabled: true,
  });
  return true;
}

export function mutateAddNode(
  genome: NeatGenome,
  tracker: InnovationTrackerState,
  rng: { state: number },
): boolean {
  const enabled = genome.connections.filter((c) => c.enabled);
  if (enabled.length === 0) return false;

  const target = enabled[Math.floor(nextRand(rng) * enabled.length)];
  const nodeById = new Map(genome.nodes.map((n) => [n.id, n]));
  const inNodeObj = nodeById.get(target.inNode);
  const outNodeObj = nodeById.get(target.outNode);
  if (!inNodeObj || !outNodeObj) return false;

  target.enabled = false;
  const split = getOrCreateSplitInnovation(tracker, target);
  const midLayer = (inNodeObj.layer + outNodeObj.layer) / 2;

  if (!nodeById.has(split.nodeId)) {
    genome.nodes.push({
      id: split.nodeId,
      type: 'hidden',
      activation: 'tanh',
      layer: midLayer,
    });
  }

  // Avoid duplicate innovation if split already happened on a homologous gene
  const hasIn = genome.connections.some((c) => c.innovation === split.inInnov);
  const hasOut = genome.connections.some((c) => c.innovation === split.outInnov);
  if (!hasIn) {
    genome.connections.push({
      innovation: split.inInnov,
      inNode: target.inNode,
      outNode: split.nodeId,
      weight: 1.0,
      enabled: true,
    });
  }
  if (!hasOut) {
    genome.connections.push({
      innovation: split.outInnov,
      inNode: split.nodeId,
      outNode: target.outNode,
      weight: target.weight,
      enabled: true,
    });
  }

  return true;
}

export function mutateNeatGenome(
  genome: NeatGenome,
  config: NeatConfig,
  tracker: InnovationTrackerState,
  rng: { state: number },
): NeatGenome {
  const mutated = cloneNeatGenome(genome);

  // 1. Weight mutations
  if (nextRand(rng) < config.weightMutateRate) {
    for (const conn of mutated.connections) {
      if (nextRand(rng) < config.weightReplaceRate) {
        conn.weight = (nextRand(rng) * 2 - 1) * 2.0;
      } else {
        conn.weight = Math.max(
          -6,
          Math.min(6, conn.weight + nextGaussian(rng) * config.weightPerturbScale),
        );
      }
    }
  }

  // 2. Structural: add connection
  if (nextRand(rng) < config.addConnectionRate) {
    mutateAddConnection(mutated, tracker, rng);
  }

  // 3. Structural: add node
  if (nextRand(rng) < config.addNodeRate) {
    mutateAddNode(mutated, tracker, rng);
  }

  // 4. Toggle enabled state
  if (mutated.connections.length > 1 && nextRand(rng) < config.toggleEnableRate) {
    const idx = Math.floor(nextRand(rng) * mutated.connections.length);
    mutated.connections[idx].enabled = !mutated.connections[idx].enabled;
  }

  return mutated;
}

// ============================================================================
// COMPATIBILITY DISTANCE, CROSSOVER & SPECIATION
// ============================================================================

export function computeCompatibilityDistance(
  a: NeatGenome,
  b: NeatGenome,
  config: Pick<NeatConfig, 'c1Excess' | 'c2Disjoint' | 'c3Weight'>,
): number {
  const mapA = new Map(a.connections.map((c) => [c.innovation, c]));
  const mapB = new Map(b.connections.map((c) => [c.innovation, c]));

  const maxInnovA = a.connections.reduce((m, c) => Math.max(m, c.innovation), 0);
  const maxInnovB = b.connections.reduce((m, c) => Math.max(m, c.innovation), 0);
  const minMaxInnov = Math.min(maxInnovA, maxInnovB);

  const allInnovs = new Set([...mapA.keys(), ...mapB.keys()]);
  let excess = 0;
  let disjoint = 0;
  let matching = 0;
  let weightDiffSum = 0;

  for (const innov of allInnovs) {
    const ca = mapA.get(innov);
    const cb = mapB.get(innov);
    if (ca && cb) {
      matching++;
      weightDiffSum += Math.abs(ca.weight - cb.weight);
    } else if (innov > minMaxInnov) {
      excess++;
    } else {
      disjoint++;
    }
  }

  const maxLen = Math.max(a.connections.length, b.connections.length);
  const N = maxLen < 20 ? 1 : maxLen;
  const meanW = matching > 0 ? weightDiffSum / matching : 0;

  return (config.c1Excess * excess) / N + (config.c2Disjoint * disjoint) / N + config.c3Weight * meanW;
}

export function crossoverNeatGenomes(
  parentA: NeatGenome,
  parentB: NeatGenome,
  childId: string,
  rng: { state: number },
): NeatGenome {
  // Ensure primary is fitter (or equal)
  let primary = parentA;
  let secondary = parentB;
  const tied = Math.abs(parentA.fitness - parentB.fitness) < 1e-9;
  if (parentB.fitness > parentA.fitness) {
    primary = parentB;
    secondary = parentA;
  }

  const mapPrimary = new Map(primary.connections.map((c) => [c.innovation, c]));
  const mapSecondary = new Map(secondary.connections.map((c) => [c.innovation, c]));

  const childConnections: NeatConnectionGene[] = [];
  const innovs = new Set([...mapPrimary.keys(), ...(tied ? mapSecondary.keys() : [])]);

  for (const innov of Array.from(innovs).sort((x, y) => x - y)) {
    const cp = mapPrimary.get(innov);
    const cs = mapSecondary.get(innov);

    let chosen: NeatConnectionGene | undefined;
    if (cp && cs) {
      chosen = { ...(nextRand(rng) < 0.5 ? cp : cs) };
      // If disabled in either parent, 75% chance stays disabled (Stanley & Miikkulainen 2002)
      if (!cp.enabled || !cs.enabled) {
        chosen.enabled = nextRand(rng) >= 0.75;
      }
    } else if (cp) {
      chosen = { ...cp };
    } else if (tied && cs) {
      chosen = { ...cs };
    }

    if (chosen) {
      childConnections.push(chosen);
    }
  }

  // Collect all required nodes (all base nodes + any hidden nodes referenced by child connections)
  const nodeMap = new Map<number, NeatNodeGene>();
  for (const n of primary.nodes) {
    if (n.type !== 'hidden') nodeMap.set(n.id, { ...n });
  }
  const allParentNodes = new Map<number, NeatNodeGene>();
  for (const n of secondary.nodes) allParentNodes.set(n.id, n);
  for (const n of primary.nodes) allParentNodes.set(n.id, n);

  for (const c of childConnections) {
    const nIn = allParentNodes.get(c.inNode);
    const nOut = allParentNodes.get(c.outNode);
    if (nIn && !nodeMap.has(nIn.id)) nodeMap.set(nIn.id, { ...nIn });
    if (nOut && !nodeMap.has(nOut.id)) nodeMap.set(nOut.id, { ...nOut });
  }

  const child: NeatGenome = {
    id: childId,
    nodes: Array.from(nodeMap.values()).sort((a, b) => a.id - b.id),
    connections: [],
    fitness: 0,
    adjustedFitness: 0,
    speciesId: null,
  };

  // Add connections safely preventing accidental cycles when tied parents had divergent splits
  for (const c of childConnections) {
    if (!wouldCreateCycle(child, c.inNode, c.outNode)) {
      child.connections.push(c);
    }
  }

  return child;
}

// ============================================================================
// POPULATION INITIALIZATION, SPECIATION & GENERATION STEP
// ============================================================================

export function createNeatPopulation(
  configOverrides: Partial<NeatConfig> = {},
  seed = 772,
): NeatPopulationState {
  const config = createDefaultNeatConfig(configOverrides);
  const tracker = createInnovationTracker(config.inputCount, config.outputCount);
  const rng = { state: seed | 0 || 1 };
  const genomes: NeatGenome[] = [];

  for (let i = 0; i < config.populationSize; i++) {
    genomes.push(
      createMinimalNeatGenome(`neat-g0-${i + 1}`, config.inputCount, config.outputCount, tracker, rng),
    );
  }

  const state: NeatPopulationState = {
    generation: 0,
    config,
    tracker,
    genomes,
    species: [],
    nextSpeciesId: 1,
    nextGenomeSeq: config.populationSize + 1,
    bestGenome: null,
    rngState: rng.state,
  };

  speciatePopulation(state);
  return state;
}

export function speciatePopulation(state: NeatPopulationState): void {
  const { config, genomes } = state;
  const existingSpecies = state.species.map((s) => ({
    ...s,
    memberIds: [] as string[],
  }));

  for (const genome of genomes) {
    let assigned = false;
    for (const sp of existingSpecies) {
      const dist = computeCompatibilityDistance(genome, sp.representative, config);
      if (dist <= config.compatibilityThreshold) {
        sp.memberIds.push(genome.id);
        genome.speciesId = sp.id;
        assigned = true;
        break;
      }
    }
    if (!assigned) {
      const newId = state.nextSpeciesId++;
      genome.speciesId = newId;
      existingSpecies.push({
        id: newId,
        representative: cloneNeatGenome(genome),
        memberIds: [genome.id],
        bestFitness: genome.fitness,
        stagnantGenerations: 0,
        averageAdjustedFitness: 0,
      });
    }
  }

  // Drop empty species and refresh representatives
  const genomeById = new Map(genomes.map((g) => [g.id, g]));
  const rng = { state: state.rngState };
  state.species = existingSpecies
    .filter((sp) => sp.memberIds.length > 0)
    .map((sp) => {
      const members = sp.memberIds.map((id) => genomeById.get(id)!).filter(Boolean);
      const rep = members[Math.floor(nextRand(rng) * members.length)] ?? members[0];
      return {
        ...sp,
        representative: cloneNeatGenome(rep),
      };
    });
  state.rngState = rng.state;

  // Dynamic threshold steering toward targetSpeciesCount
  if (state.species.length < config.targetSpeciesCount) {
    config.compatibilityThreshold = Math.max(0.4, config.compatibilityThreshold - 0.15);
  } else if (state.species.length > config.targetSpeciesCount) {
    config.compatibilityThreshold = Math.min(8.0, config.compatibilityThreshold + 0.15);
  }
}

/**
 * Advances one full NEAT generation AFTER fitness values have been assigned to `state.genomes`.
 * Applies explicit fitness sharing, stagnation culling, per-species elitism, crossover, and mutation.
 */
export function evolveNeatGeneration(state: NeatPopulationState): NeatPopulationState {
  const rng = { state: state.rngState };
  const { config } = state;

  // Ensure species assignments match current genomes
  if (state.species.length === 0) {
    speciatePopulation(state);
  }

  const genomeById = new Map(state.genomes.map((g) => [g.id, g]));

  // Track global champion
  let globalBest = state.genomes[0];
  for (const g of state.genomes) {
    if (g.fitness > globalBest.fitness) globalBest = g;
  }
  if (!state.bestGenome || globalBest.fitness >= state.bestGenome.fitness) {
    state.bestGenome = cloneNeatGenome(globalBest);
  }

  // Shift fitness to strictly positive for fitness-sharing proportions
  const minFit = state.genomes.reduce((m, g) => Math.min(m, g.fitness), Infinity);
  const offset = minFit <= 0 ? -minFit + 0.05 : 0;

  // Compute explicit fitness sharing & stagnation per species
  const activeSpecies: NeatSpecies[] = [];
  for (const sp of state.species) {
    const members = sp.memberIds
      .map((id) => genomeById.get(id))
      .filter((g): g is NeatGenome => Boolean(g))
      .sort((a, b) => b.fitness - a.fitness);

    if (members.length === 0) continue;

    const topFit = members[0].fitness;
    if (topFit > sp.bestFitness + 1e-6) {
      sp.bestFitness = topFit;
      sp.stagnantGenerations = 0;
    } else {
      sp.stagnantGenerations += 1;
    }

    let adjSum = 0;
    for (const m of members) {
      m.adjustedFitness = (m.fitness + offset) / members.length;
      adjSum += m.adjustedFitness;
    }
    sp.averageAdjustedFitness = adjSum / members.length;
    activeSpecies.push(sp);
  }

  // Cull stagnant species, ALWAYS keeping top 2 species by bestFitness
  activeSpecies.sort((a, b) => b.bestFitness - a.bestFitness);
  const survivingSpecies = activeSpecies.filter(
    (sp, idx) => idx < 2 || sp.stagnantGenerations < config.stagnationLimit,
  );

  const totalAvgAdj = survivingSpecies.reduce((s, sp) => s + Math.max(1e-6, sp.averageAdjustedFitness), 0);
  const nextGen = state.generation + 1;
  const nextGenomes: NeatGenome[] = [];

  for (const sp of survivingSpecies) {
    const members = sp.memberIds
      .map((id) => genomeById.get(id))
      .filter((g): g is NeatGenome => Boolean(g))
      .sort((a, b) => b.fitness - a.fitness);

    const quota = Math.max(
      1,
      Math.round((Math.max(1e-6, sp.averageAdjustedFitness) / totalAvgAdj) * config.populationSize),
    );

    // 1. Elitism: copy species champion(s) verbatim
    const eliteCount = Math.min(config.elitismPerSpecies, members.length, quota);
    for (let e = 0; e < eliteCount && nextGenomes.length < config.populationSize; e++) {
      const champ = cloneNeatGenome(members[e]);
      champ.id = `neat-g${nextGen}-${state.nextGenomeSeq++}`;
      champ.fitness = 0;
      champ.adjustedFitness = 0;
      nextGenomes.push(champ);
    }

    // 2. Breed remaining offspring from top 60% of species
    const breedingPool = members.slice(0, Math.max(1, Math.ceil(members.length * 0.6)));
    const remaining = quota - eliteCount;

    for (let k = 0; k < remaining && nextGenomes.length < config.populationSize; k++) {
      const p1 = breedingPool[Math.floor(nextRand(rng) * breedingPool.length)];
      let child: NeatGenome;
      const cid = `neat-g${nextGen}-${state.nextGenomeSeq++}`;

      if (breedingPool.length >= 2 && nextRand(rng) < config.crossoverRate) {
        const p2 = breedingPool[Math.floor(nextRand(rng) * breedingPool.length)];
        child = crossoverNeatGenomes(p1, p2, cid, rng);
      } else {
        child = cloneNeatGenome(p1);
        child.id = cid;
        child.fitness = 0;
        child.adjustedFitness = 0;
      }

      child = mutateNeatGenome(child, config, state.tracker, rng);
      nextGenomes.push(child);
    }
  }

  // Fill any rounding shortfall from global best species
  while (nextGenomes.length < config.populationSize) {
    const base = state.bestGenome ?? state.genomes[0];
    const cid = `neat-g${nextGen}-${state.nextGenomeSeq++}`;
    const child = mutateNeatGenome(
      { ...cloneNeatGenome(base), id: cid, fitness: 0, adjustedFitness: 0 },
      config,
      state.tracker,
      rng,
    );
    nextGenomes.push(child);
  }

  state.generation = nextGen;
  state.genomes = nextGenomes.slice(0, config.populationSize);
  state.species = survivingSpecies;
  state.rngState = rng.state;
  speciatePopulation(state);

  return state;
}

export function getNeatComplexitySummary(genome: NeatGenome): {
  totalNodes: number;
  hiddenNodes: number;
  enabledConnections: number;
  totalConnections: number;
} {
  const hiddenNodes = genome.nodes.filter((n) => n.type === 'hidden').length;
  const enabledConnections = genome.connections.filter((c) => c.enabled).length;
  return {
    totalNodes: genome.nodes.length,
    hiddenNodes,
    enabledConnections,
    totalConnections: genome.connections.length,
  };
}

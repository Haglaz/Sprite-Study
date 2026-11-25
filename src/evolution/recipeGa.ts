import { nextRand } from './neat';
import { ALL_PIXEL_OP_IDS, PIXEL_OP_LABELS_PT } from './pixelOps';
import type {
  PixelOpId,
  RecipeGaConfig,
  RecipeGaPopulationState,
  RecipeGenome,
  RecipeStepGene,
} from './types';

export function createDefaultRecipeGaConfig(
  overrides: Partial<RecipeGaConfig> = {},
): RecipeGaConfig {
  return {
    populationSize: 12,
    minSteps: 2,
    maxSteps: 7,
    tournamentSize: 3,
    crossoverRate: 0.72,
    mutationRate: 0.65,
    elitismCount: 2,
    parsimonyPenaltyPerStep: 0.25,
    ...overrides,
  };
}

export function cloneRecipeGenome(g: RecipeGenome): RecipeGenome {
  return {
    id: g.id,
    steps: g.steps.map((s) => ({ ...s })),
    fitness: g.fitness,
    valScore: g.valScore,
  };
}

function randomStep(rng: { state: number }, exclude?: Set<PixelOpId>): RecipeStepGene {
  const pool = exclude
    ? ALL_PIXEL_OP_IDS.filter((op) => !exclude.has(op))
    : [...ALL_PIXEL_OP_IDS];
  const ops = pool.length > 0 ? pool : ALL_PIXEL_OP_IDS;
  const op = ops[Math.floor(nextRand(rng) * ops.length)];
  return {
    op,
    enabled: true,
    intensity: Math.round((0.2 + nextRand(rng) * 0.7) * 100) / 100,
    threshold: Math.round((0.2 + nextRand(rng) * 0.65) * 100) / 100,
  };
}

function createRandomRecipeGenome(
  id: string,
  config: RecipeGaConfig,
  rng: { state: number },
): RecipeGenome {
  const len =
    config.minSteps + Math.floor(nextRand(rng) * (config.maxSteps - config.minSteps + 1));
  const used = new Set<PixelOpId>();
  const steps: RecipeStepGene[] = [];
  for (let i = 0; i < len; i++) {
    const step = randomStep(rng, used);
    used.add(step.op);
    steps.push(step);
  }
  return {
    id,
    steps,
    fitness: 0,
    valScore: null,
  };
}

export function createRecipeGaPopulation(
  configOverrides: Partial<RecipeGaConfig> = {},
  seed = 772,
): RecipeGaPopulationState {
  const config = createDefaultRecipeGaConfig(configOverrides);
  const rng = { state: (seed ^ 0x51a7c3) | 0 || 1 };
  const genomes: RecipeGenome[] = [];

  for (let i = 0; i < config.populationSize; i++) {
    genomes.push(createRandomRecipeGenome(`ga-g0-${i + 1}`, config, rng));
  }

  return {
    generation: 0,
    config,
    genomes,
    bestGenome: null,
    nextGenomeSeq: config.populationSize + 1,
    rngState: rng.state,
  };
}

function tournamentSelect(
  genomes: RecipeGenome[],
  k: number,
  rng: { state: number },
): RecipeGenome {
  let best = genomes[Math.floor(nextRand(rng) * genomes.length)];
  for (let i = 1; i < k; i++) {
    const cand = genomes[Math.floor(nextRand(rng) * genomes.length)];
    if (cand.fitness > best.fitness) best = cand;
  }
  return best;
}

export function crossoverRecipes(
  a: RecipeGenome,
  b: RecipeGenome,
  childId: string,
  config: RecipeGaConfig,
  rng: { state: number },
): RecipeGenome {
  const cutA = Math.max(1, Math.floor(nextRand(rng) * a.steps.length));
  const cutB = Math.floor(nextRand(rng) * b.steps.length);
  const combined = [...a.steps.slice(0, cutA), ...b.steps.slice(cutB)].map((s) => ({ ...s }));

  // Deduplicate consecutive identical ops while respecting min/max length bounds
  const deduped: RecipeStepGene[] = [];
  for (const step of combined) {
    if (deduped.length > 0 && deduped[deduped.length - 1].op === step.op) continue;
    deduped.push(step);
    if (deduped.length >= config.maxSteps) break;
  }

  while (deduped.length < config.minSteps) {
    deduped.push(randomStep(rng));
  }

  return {
    id: childId,
    steps: deduped,
    fitness: 0,
    valScore: null,
  };
}

export function mutateRecipe(
  genome: RecipeGenome,
  config: RecipeGaConfig,
  rng: { state: number },
): RecipeGenome {
  const out = cloneRecipeGenome(genome);
  const roll = nextRand(rng);

  // 1. Parameter tweak on random step
  if (out.steps.length > 0 && nextRand(rng) < 0.75) {
    const idx = Math.floor(nextRand(rng) * out.steps.length);
    const s = out.steps[idx];
    s.intensity = Math.max(0.05, Math.min(1, Math.round((s.intensity + (nextRand(rng) - 0.5) * 0.3) * 100) / 100));
    s.threshold = Math.max(0.05, Math.min(1, Math.round((s.threshold + (nextRand(rng) - 0.5) * 0.3) * 100) / 100));
  }

  // 2. Structural mutation: insert, delete, swap order, or replace op
  if (roll < 0.28 && out.steps.length < config.maxSteps) {
    const used = new Set(out.steps.map((s) => s.op));
    const pos = Math.floor(nextRand(rng) * (out.steps.length + 1));
    out.steps.splice(pos, 0, randomStep(rng, used));
  } else if (roll < 0.5 && out.steps.length > config.minSteps) {
    const pos = Math.floor(nextRand(rng) * out.steps.length);
    out.steps.splice(pos, 1);
  } else if (roll < 0.75 && out.steps.length >= 2) {
    const i = Math.floor(nextRand(rng) * out.steps.length);
    const j = Math.floor(nextRand(rng) * out.steps.length);
    const tmp = out.steps[i];
    out.steps[i] = out.steps[j];
    out.steps[j] = tmp;
  } else if (out.steps.length > 0) {
    const pos = Math.floor(nextRand(rng) * out.steps.length);
    if (nextRand(rng) < 0.5) {
      out.steps[pos].enabled = !out.steps[pos].enabled;
    } else {
      out.steps[pos] = randomStep(rng);
    }
  }

  // Ensure at least one enabled step
  if (!out.steps.some((s) => s.enabled) && out.steps.length > 0) {
    out.steps[0].enabled = true;
  }

  return out;
}

/**
 * Advances one Recipe GA generation AFTER fitness values have been assigned to `state.genomes`.
 */
export function evolveRecipeGaGeneration(state: RecipeGaPopulationState): RecipeGaPopulationState {
  const rng = { state: state.rngState };
  const { config } = state;
  const sorted = [...state.genomes].sort((a, b) => b.fitness - a.fitness);

  if (!state.bestGenome || sorted[0].fitness >= state.bestGenome.fitness) {
    state.bestGenome = cloneRecipeGenome(sorted[0]);
  }

  const nextGen = state.generation + 1;
  const nextGenomes: RecipeGenome[] = [];

  // 1. Elitism
  for (let i = 0; i < Math.min(config.elitismCount, sorted.length); i++) {
    const elite = cloneRecipeGenome(sorted[i]);
    elite.id = `ga-g${nextGen}-${state.nextGenomeSeq++}`;
    elite.fitness = 0;
    elite.valScore = null;
    nextGenomes.push(elite);
  }

  // 2. Diversity immigrant (1 fresh random recipe every generation to avoid premature collapse)
  if (nextGenomes.length < config.populationSize) {
    nextGenomes.push(createRandomRecipeGenome(`ga-g${nextGen}-${state.nextGenomeSeq++}`, config, rng));
  }

  // 3. Tournament selection + crossover + mutation
  while (nextGenomes.length < config.populationSize) {
    const p1 = tournamentSelect(sorted, config.tournamentSize, rng);
    const cid = `ga-g${nextGen}-${state.nextGenomeSeq++}`;
    let child: RecipeGenome;

    if (nextRand(rng) < config.crossoverRate) {
      const p2 = tournamentSelect(sorted, config.tournamentSize, rng);
      child = crossoverRecipes(p1, p2, cid, config, rng);
    } else {
      child = cloneRecipeGenome(p1);
      child.id = cid;
      child.fitness = 0;
      child.valScore = null;
    }

    if (nextRand(rng) < config.mutationRate) {
      child = mutateRecipe(child, config, rng);
    }
    nextGenomes.push(child);
  }

  state.generation = nextGen;
  state.genomes = nextGenomes;
  state.rngState = rng.state;
  return state;
}

export function formatRecipeSummaryPt(recipe: RecipeGenome): string {
  const active = recipe.steps.filter((s) => s.enabled);
  if (active.length === 0) return 'Nenhum passo ativo';
  return active.map((s) => PIXEL_OP_LABELS_PT[s.op]).join(' → ');
}

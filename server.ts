import 'dotenv/config';
import express, { type Request, type Response, type NextFunction } from 'express';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT) || 3000;
const IS_PRODUCTION = process.env.NODE_ENV === 'production';

/**
 * Verified @google/genai model IDs (configurable via env; API key stays server-side).
 */
export const GEMINI_MODEL_PRIMARY =
  process.env.GEMINI_MODEL_PRIMARY?.trim() || 'gemini-4-hydrogen';
export const GEMINI_MODEL_FALLBACK =
  process.env.GEMINI_MODEL_FALLBACK?.trim() || 'gemini-3.1-flash-lite';

const ALLOWED_IMAGE_MIMES = new Set(['image/png', 'image/jpeg', 'image/webp']);
const MAX_IMAGES_PER_REQUEST = 10;
const MAX_IMAGE_DECODED_BYTES = 350 * 1024; // 350 KB per image
const MAX_TOTAL_IMAGE_BYTES = 2.5 * 1024 * 1024; // 2.5 MB combined
const MAX_PROMPT_CHARS = 32_000;

const VALID_CATEGORIES = new Set([
  'Equipamentos',
  'Consumíveis',
  'Containers',
  'Decoração',
]);
const HEX_COLOR_RE = /^#[0-9A-Fa-f]{6}$/;

/**
 * Simple in-memory per-IP sliding-window rate limiter protecting `/api/gemini/*`.
 */
const RATE_LIMIT_WINDOW_MS = 60_000;
const RATE_LIMIT_MAX_REQUESTS = 30;
const rateBuckets = new Map<string, number[]>();

function geminiRateLimiter(req: Request, res: Response, next: NextFunction): void {
  const key = req.ip || req.socket.remoteAddress || 'unknown';
  const now = Date.now();
  const prev = rateBuckets.get(key) ?? [];
  const fresh = prev.filter((ts) => now - ts < RATE_LIMIT_WINDOW_MS);

  if (fresh.length >= RATE_LIMIT_MAX_REQUESTS) {
    const oldest = fresh[0] ?? now;
    const retryAfterSec = Math.max(1, Math.ceil((RATE_LIMIT_WINDOW_MS - (now - oldest)) / 1000));
    res.setHeader('Retry-After', String(retryAfterSec));
    res.status(429).json({
      error: `Limite de requisições à API Gemini atingido (${RATE_LIMIT_MAX_REQUESTS}/min). Tente novamente em ${retryAfterSec}s.`,
    });
    return;
  }

  fresh.push(now);
  rateBuckets.set(key, fresh);
  // Periodic pruning so long-running processes don't grow the map unboundedly
  if (rateBuckets.size > 500) {
    for (const [k, list] of rateBuckets.entries()) {
      if (list.length === 0 || now - list[list.length - 1] >= RATE_LIMIT_WINDOW_MS) {
        rateBuckets.delete(k);
      }
    }
  }
  next();
}

function createGeminiClient(): GoogleGenAI | null {
  const apiKey = process.env.GEMINI_API_KEY?.trim();
  if (!apiKey) return null;
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

function parseClampedTemperature(raw: unknown): number | undefined {
  if (raw === undefined || raw === null || raw === '') return undefined;
  const n = Number(raw);
  if (!Number.isFinite(n)) return undefined;
  return Math.max(0, Math.min(2, Math.round(n * 100) / 100));
}

function validateAndParseImages(
  rawImages: unknown,
): { ok: true; images: Array<{ mimeType: string; data: string }> } | { ok: false; error: string } {
  if (rawImages === undefined || rawImages === null) {
    return { ok: true, images: [] };
  }
  if (!Array.isArray(rawImages)) {
    return { ok: false, error: 'Campo "images" deve ser um array de data URLs base64.' };
  }
  if (rawImages.length > MAX_IMAGES_PER_REQUEST) {
    return {
      ok: false,
      error: `Máximo de ${MAX_IMAGES_PER_REQUEST} imagens por requisição (recebido ${rawImages.length}).`,
    };
  }

  const parsed: Array<{ mimeType: string; data: string }> = [];
  let totalDecodedBytes = 0;

  for (let i = 0; i < rawImages.length; i++) {
    const item = rawImages[i];
    if (typeof item !== 'string') {
      return { ok: false, error: `Imagem #${i + 1} não é uma string válida.` };
    }
    const trimmed = item.trim();
    const match = /^data:(image\/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=\s]+)$/.exec(trimmed);
    if (!match) {
      return {
        ok: false,
        error: `Imagem #${i + 1} deve estar em formato data URL base64 (image/png, image/jpeg ou image/webp).`,
      };
    }
    const mimeType = match[1].toLowerCase();
    if (!ALLOWED_IMAGE_MIMES.has(mimeType)) {
      return {
        ok: false,
        error: `Tipo MIME não permitido na imagem #${i + 1}: "${mimeType}".`,
      };
    }
    const data = match[2].replace(/\s+/g, '');
    const decodedBytes = Math.floor((data.length * 3) / 4);
    if (decodedBytes <= 0 || decodedBytes > MAX_IMAGE_DECODED_BYTES) {
      return {
        ok: false,
        error: `Imagem #${i + 1} excede o limite de ${Math.round(MAX_IMAGE_DECODED_BYTES / 1024)} KB.`,
      };
    }
    totalDecodedBytes += decodedBytes;
    if (totalDecodedBytes > MAX_TOTAL_IMAGE_BYTES) {
      return {
        ok: false,
        error: `Tamanho combinado das imagens excede o limite de ${(MAX_TOTAL_IMAGE_BYTES / (1024 * 1024)).toFixed(1)} MB.`,
      };
    }
    parsed.push({ mimeType, data });
  }

  return { ok: true, images: parsed };
}

/**
 * Strict server-side validation for POST /api/gemini/engineer-prompt JSON output.
 * Mirrors `validateEngineeredPromptJson` in `src/generation/promptEngineer.ts` while keeping
 * `server.ts` standalone for direct `NODE_ENV=production node server.ts` execution.
 */
function validateEngineeredPromptServer(
  rawText: string,
  allowedDonorIds: Set<number>,
): Record<string, unknown> {
  let obj: unknown;
  try {
    obj = JSON.parse(rawText);
  } catch {
    throw new Error('Modelo retornou JSON malformado (falha no JSON.parse).');
  }

  if (!obj || typeof obj !== 'object' || Array.isArray(obj)) {
    throw new Error('JSON do Engenheiro de Prompt deve ser um objeto.');
  }
  const rec = obj as Record<string, unknown>;

  if (typeof rec.itemName !== 'string' || rec.itemName.trim().length < 2) {
    throw new Error('Campo "itemName" ausente ou inválido.');
  }
  if (typeof rec.category !== 'string' || !VALID_CATEGORIES.has(rec.category)) {
    throw new Error(`Campo "category" inválido: "${String(rec.category)}".`);
  }
  if (typeof rec.visualBrief !== 'string' || rec.visualBrief.trim().length < 15) {
    throw new Error('Campo "visualBrief" ausente ou curto demais.');
  }
  if (!Array.isArray(rec.paletteRamps) || rec.paletteRamps.length === 0) {
    throw new Error('Campo "paletteRamps" deve ter ao menos 1 rampa.');
  }
  for (const r of rec.paletteRamps) {
    if (!r || typeof r !== 'object') throw new Error('Rampa inválida em "paletteRamps".');
    const rObj = r as Record<string, unknown>;
    if (typeof rObj.material !== 'string' || !rObj.material.trim()) {
      throw new Error('Rampa sem nome de material.');
    }
    if (!Array.isArray(rObj.colors) || rObj.colors.length < 3 || rObj.colors.length > 5) {
      throw new Error(`Rampa "${rObj.material}" deve ter entre 3 e 5 cores hex.`);
    }
    for (const c of rObj.colors) {
      if (typeof c !== 'string' || !HEX_COLOR_RE.test(c.trim())) {
        throw new Error(`Cor hex inválida "${String(c)}" na rampa "${rObj.material}".`);
      }
    }
  }

  if (!rec.constraints || typeof rec.constraints !== 'object' || Array.isArray(rec.constraints)) {
    throw new Error('Campo "constraints" ausente ou inválido.');
  }
  const cObj = rec.constraints as Record<string, unknown>;
  const fp = cObj.footprintPx as Record<string, unknown> | undefined;
  const fpW = Number(fp?.width);
  const fpH = Number(fp?.height);
  const maxColors = Number(cObj.maxColors);
  if (
    !Number.isFinite(fpW) ||
    !Number.isFinite(fpH) ||
    fpW < 4 ||
    fpW > 32 ||
    fpH < 4 ||
    fpH > 32
  ) {
    throw new Error('Campo "constraints.footprintPx" fora da faixa 4..32 px.');
  }
  if (!Number.isFinite(maxColors) || maxColors < 4 || maxColors > 64) {
    throw new Error('Campo "constraints.maxColors" fora da faixa 4..64.');
  }
  if (
    typeof cObj.outlineRule !== 'string' ||
    cObj.outlineRule.trim().length < 3 ||
    typeof cObj.noDitheringOrAA !== 'boolean' ||
    typeof cObj.binaryAlpha !== 'boolean'
  ) {
    throw new Error('Campos obrigatórios ausentes em "constraints".');
  }

  if (!Array.isArray(rec.negativePrompt) || rec.negativePrompt.length < 2) {
    throw new Error('Campo "negativePrompt" deve conter ao menos 2 itens.');
  }
  if (
    !Array.isArray(rec.donorSuggestions) ||
    rec.donorSuggestions.length < 3 ||
    rec.donorSuggestions.length > 6
  ) {
    throw new Error('Campo "donorSuggestions" deve conter entre 3 e 6 IDs de sprites.');
  }
  for (const idRaw of rec.donorSuggestions) {
    const id = Number(idRaw);
    if (!Number.isInteger(id) || id <= 0) {
      throw new Error(`ID de doador inválido: "${String(idRaw)}".`);
    }
    if (allowedDonorIds.size > 0 && !allowedDonorIds.has(id)) {
      throw new Error(
        `Sprite doador #${id} não existe nos splits permitidos (train/validation).`,
      );
    }
  }

  return rec;
}

const ENGINEER_PROMPT_RESPONSE_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    itemName: { type: Type.STRING },
    category: {
      type: Type.STRING,
      enum: ['Equipamentos', 'Consumíveis', 'Containers', 'Decoração'],
    },
    visualBrief: { type: Type.STRING },
    paletteRamps: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          material: { type: Type.STRING },
          colors: { type: Type.ARRAY, items: { type: Type.STRING } },
        },
        required: ['material', 'colors'],
      },
    },
    constraints: {
      type: Type.OBJECT,
      properties: {
        footprintPx: {
          type: Type.OBJECT,
          properties: {
            width: { type: Type.INTEGER },
            height: { type: Type.INTEGER },
          },
          required: ['width', 'height'],
        },
        maxColors: { type: Type.INTEGER },
        outlineRule: { type: Type.STRING },
        noDitheringOrAA: { type: Type.BOOLEAN },
        binaryAlpha: { type: Type.BOOLEAN },
      },
      required: ['footprintPx', 'maxColors', 'outlineRule', 'noDitheringOrAA', 'binaryAlpha'],
    },
    negativePrompt: {
      type: Type.ARRAY,
      items: { type: Type.STRING },
    },
    donorSuggestions: {
      type: Type.ARRAY,
      items: { type: Type.INTEGER },
    },
  },
  required: [
    'itemName',
    'category',
    'visualBrief',
    'paletteRamps',
    'constraints',
    'negativePrompt',
    'donorSuggestions',
  ],
};

async function startServer() {
  const app = express();
  app.set('trust proxy', 1);

  // Scoped JSON body limits: 4MB cap on /api/gemini/* (for compact donor thumbnails), 1MB default elsewhere
  const geminiJsonParser = express.json({ limit: '4mb' });
  app.use('/api/gemini', geminiRateLimiter, geminiJsonParser);
  app.use(express.json({ limit: '1mb' }));

  app.get('/api/health', (_req, res) => {
    res.json({
      status: 'ok',
      aiReady: Boolean(process.env.GEMINI_API_KEY?.trim()),
      models: {
        primary: GEMINI_MODEL_PRIMARY,
        fallback: GEMINI_MODEL_FALLBACK,
      },
    });
  });

  /**
   * STEP 1 — Prompt Engineer endpoint (strict JSON schema, server-side GEMINI_API_KEY).
   * Input: user's short idea + train-only styleDossier + optional donor PNGs + optional revision context.
   * Output: validated EngineeredPromptSpec JSON (HTTP 422 on malformed output, HTTP 503 if API key missing).
   */
  app.post('/api/gemini/engineer-prompt', async (req, res) => {
    const idea = typeof req.body?.idea === 'string' ? req.body.idea.trim() : '';
    const categoryHint =
      typeof req.body?.categoryHint === 'string' && VALID_CATEGORIES.has(req.body.categoryHint)
        ? req.body.categoryHint
        : undefined;
    const styleDossier =
      req.body?.styleDossier && typeof req.body.styleDossier === 'object'
        ? (req.body.styleDossier as Record<string, unknown>)
        : null;
    const previousPrompt =
      req.body?.previousPrompt && typeof req.body.previousPrompt === 'object'
        ? (req.body.previousPrompt as Record<string, unknown>)
        : null;
    const adjustmentComment =
      typeof req.body?.adjustmentComment === 'string'
        ? req.body.adjustmentComment.trim().slice(0, 600)
        : '';
    const temperature = parseClampedTemperature(req.body?.temperature);

    if (!idea || idea.length > 400) {
      res.status(400).json({
        error: 'Informe uma ideia curta válida (1 a 400 caracteres) para o Engenheiro de Prompt.',
      });
      return;
    }
    if (!styleDossier) {
      res.status(400).json({ error: 'Campo "styleDossier" (split train) é obrigatório.' });
      return;
    }

    const imgCheck = validateAndParseImages(req.body?.images);
    if (!imgCheck.ok) {
      res.status(400).json({ error: imgCheck.error });
      return;
    }

    const ai = createGeminiClient();
    if (!ai) {
      res.status(503).json({
        aiReady: false,
        error: 'GEMINI_API_KEY não configurada no servidor (Gemini indisponível).',
      });
      return;
    }

    const allowedIdsRaw = Array.isArray(styleDossier.allowedDonorIds)
      ? styleDossier.allowedDonorIds
      : [];
    const allowedDonorIds = new Set<number>(
      allowedIdsRaw.map((v) => Number(v)).filter((n) => Number.isInteger(n) && n > 0),
    );

    const systemInstruction = [
      'Você é o Engenheiro de Prompt e Diretor de Arte do Atelier Sprite, especialista em pixel art clássico de Tibia 32×32 (.dat/.spr).',
      'Sua tarefa é converter a ideia curta do usuário e o dossiê de estilo real (extraído apenas do split train) em uma especificação técnica rigorosa JSON em PT-BR para gerar um sprite 32×32 indistinguível de arte humana clássica de Tibia.',
      'Regras obrigatórias:',
      '1. category deve ser exatamente uma de: Equipamentos, Consumíveis, Containers, Decoração.',
      '2. visualBrief deve descrever silhueta, pose/orientação clássica de Tibia, materiais (metal, madeira, tecido, gema, pele/couro) e iluminação direcional vinda do noroeste (NW -> SE).',
      '3. paletteRamps: 1 a 4 materiais, cada um com 3 a 5 cores #RRGGBB ordenadas da mais escura para a mais clara (outline, shadow, base, light, highlight), com hue-shift (sombras mais frias, luzes mais quentes), usando APENAS cores presentes em topPaletteHex do dossiê ou muito próximas delas.',
      '4. constraints: footprintPx dentro de 32×32 próximo à mediana da categoria, maxColors próximo à mediana real, contorno 1px escuro hue-shifted, binaryAlpha=true, noDitheringOrAA=true (salvo se ditherShareP50 >= 0.08).',
      '5. negativePrompt: listar artefatos típicos de IA a evitar (blur, gradientes, anti-aliasing, dithering ruidoso, pixels órfãos, contorno duplo/irregular, simetria perfeita artificial, excesso de cores, pixels flutuantes, luz mista).',
      '6. donorSuggestions: escolher 3 a 6 spriteIds REAIS listados em fewShotDonors ou allowedDonorIds do dossiê (NUNCA invente IDs).',
      'Retorne APENAS JSON estrito compatível com o schema.',
    ].join('\n');

    const compactDossierText = JSON.stringify(
      {
        category: styleDossier.category,
        trainSampleCount: styleDossier.trainSampleCount,
        categoryMedians: styleDossier.categoryMedians,
        topPaletteHex: Array.isArray(styleDossier.topPaletteHex)
          ? styleDossier.topPaletteHex.slice(0, 32)
          : [],
        fewShotDonors: Array.isArray(styleDossier.fewShotDonors)
          ? styleDossier.fewShotDonors.slice(0, 10)
          : [],
        activeLessons: Array.isArray(styleDossier.activeLessons)
          ? styleDossier.activeLessons.slice(0, 8)
          : [],
      },
      null,
      0,
    ).slice(0, 20_000);

    const promptSections = [
      `Ideia do usuário: "${idea}"`,
      categoryHint ? `Categoria sugerida pelo usuário: ${categoryHint}` : '',
      `Dossiê de estilo real (split train ONLY):\n${compactDossierText}`,
      previousPrompt
        ? `Prompt anterior aprovado/em revisão:\n${JSON.stringify(previousPrompt)}`
        : '',
      adjustmentComment
        ? `SOLICITAÇÃO DE AJUSTE DO ARTISTA: "${adjustmentComment}". Revise o prompt anterior aplicando exatamente esse ajuste sem perder fidelidade ao estilo Tibia.`
        : '',
    ]
      .filter(Boolean)
      .join('\n\n');

    const parts: Array<{ text: string } | { inlineData: { mimeType: string; data: string } }> = [];
    for (const img of imgCheck.images) {
      parts.push({ inlineData: img });
    }
    parts.push({ text: promptSections });

    const candidateModels = [GEMINI_MODEL_PRIMARY, GEMINI_MODEL_FALLBACK];
    let lastError: unknown = null;
    let lastValidationError: string | null = null;

    for (const modelName of candidateModels) {
      try {
        const isGemini4 = modelName.startsWith('gemini-4');
        const response = await ai.models.generateContent({
          model: modelName,
          contents: { parts },
          config: {
            systemInstruction,
            responseMimeType: 'application/json',
            responseSchema: ENGINEER_PROMPT_RESPONSE_SCHEMA,
            ...(isGemini4
              ? { thinkingConfig: { thinkingLevel: ThinkingLevel.LOW } }
              : temperature !== undefined
                ? { temperature }
                : {}),
          },
        });
        const rawText = (response.text ?? '').trim();
        try {
          const validated = validateEngineeredPromptServer(rawText, allowedDonorIds);
          res.json({
            model: modelName,
            appliedTemperature: isGemini4 ? null : (temperature ?? null),
            prompt: validated,
          });
          return;
        } catch (valErr) {
          lastValidationError =
            valErr instanceof Error ? valErr.message : 'JSON fora do schema esperado.';
          console.warn(`[api/gemini/engineer-prompt] ${modelName} invalid JSON:`, lastValidationError);
        }
      } catch (err) {
        lastError = err;
        const msg = err instanceof Error ? err.message : String(err);
        console.warn(`[api/gemini/engineer-prompt] ${modelName} failed:`, msg);
      }
    }

    if (lastValidationError) {
      res.status(422).json({
        error: `Engenheiro de Prompt retornou JSON inválido: ${lastValidationError}`,
      });
      return;
    }

    const message = lastError instanceof Error ? lastError.message : String(lastError);
    res.status(502).json({
      error: `Erro na chamada ao Gemini (engineer-prompt): ${message}`,
    });
  });

  /**
   * Server-side Gemini proxy endpoint (keeps GEMINI_API_KEY strictly on the server).
   * Accepts text prompt, optional systemInstruction, optional validated inline images,
   * optional JSON mode, and clamped `temperature` [0, 2].
   */
  app.post('/api/gemini/generate', async (req, res) => {
    const prompt = typeof req.body?.prompt === 'string' ? req.body.prompt.trim() : '';
    const systemInstruction =
      typeof req.body?.systemInstruction === 'string'
        ? req.body.systemInstruction.trim().slice(0, 8000)
        : undefined;
    const responseMimeType =
      req.body?.responseMimeType === 'application/json' ? 'application/json' : undefined;
    const temperature = parseClampedTemperature(req.body?.temperature);

    if (prompt.length > MAX_PROMPT_CHARS) {
      res.status(400).json({
        error: `Prompt excede o limite de ${MAX_PROMPT_CHARS} caracteres.`,
      });
      return;
    }

    const imgCheck = validateAndParseImages(req.body?.images);
    if (!imgCheck.ok) {
      res.status(400).json({ error: imgCheck.error });
      return;
    }

    if (!prompt && imgCheck.images.length === 0) {
      res.status(400).json({ error: 'Prompt ou imagens são obrigatórios.' });
      return;
    }

    const ai = createGeminiClient();
    if (!ai) {
      res.status(503).json({
        aiReady: false,
        error: 'GEMINI_API_KEY não configurada no servidor.',
      });
      return;
    }

    const parts: Array<{ text: string } | { inlineData: { mimeType: string; data: string } }> = [];
    for (const img of imgCheck.images) {
      parts.push({ inlineData: img });
    }
    if (prompt) {
      parts.push({ text: prompt });
    }

    const candidateModels = [GEMINI_MODEL_PRIMARY, GEMINI_MODEL_FALLBACK];
    let lastError: unknown = null;

    for (const modelName of candidateModels) {
      try {
        const isGemini4 = modelName.startsWith('gemini-4');
        const response = await ai.models.generateContent({
          model: modelName,
          contents: { parts },
          config: {
            ...(systemInstruction ? { systemInstruction } : {}),
            ...(responseMimeType ? { responseMimeType } : {}),
            ...(isGemini4
              ? { thinkingConfig: { thinkingLevel: ThinkingLevel.LOW } }
              : temperature !== undefined
                ? { temperature }
                : {}),
          },
        });
        res.json({
          model: modelName,
          appliedTemperature: isGemini4 ? null : (temperature ?? null),
          text: (response.text ?? '').trim(),
        });
        return;
      } catch (err) {
        lastError = err;
        const msg = err instanceof Error ? err.message : String(err);
        console.warn(`[api/gemini/generate] ${modelName} failed:`, msg);
      }
    }

    const message = lastError instanceof Error ? lastError.message : String(lastError);
    res.status(502).json({
      error: `Erro na chamada ao Gemini: ${message}`,
    });
  });

  if (!IS_PRODUCTION) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(__dirname, 'dist');
    const indexHtml = path.join(distDir, 'index.html');
    if (!fs.existsSync(indexHtml)) {
      console.error(
        `[Atelier Sprite] Erro fatal em produção: "${indexHtml}" não encontrado. Execute "npm run build" antes de iniciar o servidor.`,
      );
      process.exit(1);
    }
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(indexHtml);
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Atelier Sprite server listening on http://0.0.0.0:${PORT}`);
  });
}

void startServer();

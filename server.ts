import 'dotenv/config';
import express from 'express';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { GoogleGenAI, ThinkingLevel } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT) || 3000;
const IS_PRODUCTION = process.env.NODE_ENV === 'production';

/**
 * Verified @google/genai model IDs (kept in constants/env; API key stays server-side).
 */
export const GEMINI_MODEL_PRIMARY =
  process.env.GEMINI_MODEL_PRIMARY?.trim() || 'gemini-4-hydrogen';
export const GEMINI_MODEL_FALLBACK =
  process.env.GEMINI_MODEL_FALLBACK?.trim() || 'gemini-3.1-flash-lite';

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

function extractBase64Payload(
  dataUrlOrBase64: string,
): { mimeType: string; data: string } | null {
  const trimmed = dataUrlOrBase64.trim();
  const match = /^data:(image\/[a-zA-Z0-9.+-]+);base64,(.+)$/.exec(trimmed);
  if (match) {
    return { mimeType: match[1], data: match[2] };
  }
  if (/^[A-Za-z0-9+/=\s]+$/.test(trimmed) && trimmed.length > 20) {
    return { mimeType: 'image/png', data: trimmed.replace(/\s+/g, '') };
  }
  return null;
}

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '16mb' }));

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
   * Server-side Gemini proxy endpoint (keeps GEMINI_API_KEY strictly on the server).
   * Accepts text prompt, optional systemInstruction, optional inline images, and optional JSON mode.
   */
  app.post('/api/gemini/generate', async (req, res) => {
    const prompt = typeof req.body?.prompt === 'string' ? req.body.prompt.trim() : '';
    const systemInstruction =
      typeof req.body?.systemInstruction === 'string'
        ? req.body.systemInstruction.trim()
        : undefined;
    const responseMimeType =
      req.body?.responseMimeType === 'application/json' ? 'application/json' : undefined;
    const images: string[] = Array.isArray(req.body?.images)
      ? req.body.images.filter((s: unknown) => typeof s === 'string').slice(0, 12)
      : [];

    if (!prompt && images.length === 0) {
      res.status(400).json({ error: 'Prompt ou imagens são obrigatórios.' });
      return;
    }

    const ai = createGeminiClient();
    if (!ai) {
      res.status(503).json({
        error: 'GEMINI_API_KEY não configurada no servidor.',
      });
      return;
    }

    const parts: Array<{ text: string } | { inlineData: { mimeType: string; data: string } }> = [];
    for (const img of images) {
      const parsed = extractBase64Payload(img);
      if (parsed) parts.push({ inlineData: parsed });
    }
    if (prompt) {
      parts.push({ text: prompt });
    }

    const candidateModels = [GEMINI_MODEL_PRIMARY, GEMINI_MODEL_FALLBACK];
    let lastError: unknown = null;

    for (const modelName of candidateModels) {
      try {
        const response = await ai.models.generateContent({
          model: modelName,
          contents: { parts },
          config: {
            ...(systemInstruction ? { systemInstruction } : {}),
            ...(responseMimeType ? { responseMimeType } : {}),
            ...(modelName.startsWith('gemini-4')
              ? { thinkingConfig: { thinkingLevel: ThinkingLevel.LOW } }
              : {}),
          },
        });
        res.json({
          model: modelName,
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
    if (fs.existsSync(distDir)) {
      app.use(express.static(distDir));
      app.get('*', (_req, res) => {
        res.sendFile(path.join(distDir, 'index.html'));
      });
    }
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Atelier Sprite server listening on http://0.0.0.0:${PORT}`);
  });
}

void startServer();

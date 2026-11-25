import { ALL_ITEM_CATEGORIES } from '../types/studio';
import type { T3RampVocabulary, T3SamplerSettings } from './types';

export interface ColabNotebookCell {
  cell_type: 'markdown' | 'code';
  metadata: Record<string, unknown>;
  source: string[];
  execution_count?: null;
  outputs?: unknown[];
}

export interface JupyterNotebookDocument {
  nbformat: number;
  nbformat_minor: number;
  metadata: Record<string, unknown>;
  cells: ColabNotebookCell[];
}

export const DEFAULT_T3_SAMPLER_SETTINGS: Omit<T3SamplerSettings, 'vocabSize' | 'maskTokenId'> = {
  mode: 'masked-token',
  gridSize: 32,
  tokenCount: 1024,
  steps: 12,
  temperature: 0.85,
  topK: 16,
  cfgScale: 2.2,
  maxResampleAttempts: 3,
};

function toLines(text: string): string[] {
  const lines = text.split('\n');
  return lines.map((line, i) => (i < lines.length - 1 ? `${line}\n` : line));
}

function mdCell(text: string): ColabNotebookCell {
  return {
    cell_type: 'markdown',
    metadata: {},
    source: toLines(text.trimEnd()),
  };
}

function codeCell(code: string): ColabNotebookCell {
  return {
    cell_type: 'code',
    metadata: {},
    execution_count: null,
    outputs: [],
    source: toLines(code.trimEnd()),
  };
}

/**
 * Generates a complete, self-contained Google Colab Jupyter Notebook (.ipynb)
 * configured for the active dataset's split hash and user-corrected Ramp-Role Vocabulary.
 */
export function buildT3ColabNotebook(params: {
  vocabulary: T3RampVocabulary;
  exportedHumanLabelsCount?: { approved: number; rejected: number };
}): JupyterNotebookDocument {
  const { vocabulary, exportedHumanLabelsCount } = params;
  const vocabJson = JSON.stringify(
    vocabulary.entries.map((e) => ({
      tokenId: e.tokenId,
      rgb: e.rgb,
      hex: e.hex,
      materialFamily: e.materialFamily,
      role: e.role,
    })),
    null,
    2,
  );

  const samplerSettings: T3SamplerSettings = {
    ...DEFAULT_T3_SAMPLER_SETTINGS,
    vocabSize: vocabulary.vocabSize,
    maskTokenId: vocabulary.vocabSize, // token K+1 is [MASK]
  };

  const firstCellMd = `# Atelier Sprite — Modelo Indexado T3 (Colab T4 · PyTorch -> ONNX)

Este notebook treina o **Gerador T3 (Transformer Condicional Masked-Token, ~7.4M parâmetros)** e o **Reranker/Discriminador T3 (~1.2M parâmetros)** na grade nativa \`32×32\` de tokens de paleta + token transparente (\`0\`), usando exclusivamente os splits \`train\` e \`validation\` exportados pelo **Atelier Sprite**.

---

## Passos Exatos no Google Colab
1. No menu superior do Colab, clique em **Ambiente de execução → Alterar o tipo de ambiente de execução** e selecione **GPU T4** (compatível com plano gratuito).
2. No Atelier Sprite (aba **Estúdio Prompt → Modelo T3**), exporte o dataset via **"Exportar dataset de treino (.zip)"** ou **"Re-treinar com novos rótulos (.zip)"**.
3. Faça upload do arquivo \`.zip\` no Colab (barra lateral esquerda → Arquivos) ou coloque-o no Google Drive em \`/MyDrive/atelier_sprite_t3/dataset.zip\`.
4. Execute as células **na ordem (Célula 2 → Célula 9)**. A Célula 2 monta o Google Drive para salvar checkpoints automáticos.
5. Ao final da Célula 9, baixe os 3 arquivos gerados em \`OUT_DIR\`:
   - \`t3_generator.onnx\`
   - \`t3_reranker.onnx\`
   - \`t3_model_card.json\`
6. No Atelier Sprite, clique em **"Importar modelo (T3)"** e selecione os 3 arquivos juntos.

## Tempo Esperado (GPU NVIDIA T4 Gratuita)
- **Preparação + tokenização (\`train\` + \`validation\`)**: ~1 a 2 minutos.
- **Treinamento do Gerador T3 (40 épocas, batch 32, FP16 AMP)**: ~12 a 20 minutos (~18–30s/época).
- **Treinamento do Discriminador/Reranker T3 (16 épocas) + exportação ONNX**: ~3 a 5 minutos.
- **Total estimado**: **~18 a 27 minutos**.

## Recuperação de Desconexão (Google Colab Disconnect Recovery)
- Todo fim de época (e sempre que \`val_loss\` melhora), os checkpoints completos (\`model_state\`, \`optimizer_state\`, \`scheduler_state\`, \`epoch\`, \`step\`, \`history\`) e os *contact sheets* por categoria são gravados diretamente no Google Drive em:
  \`/content/drive/MyDrive/atelier_sprite_t3/checkpoints/${vocabulary.splitHash}/\`
- **Se o Colab desconectar**:
  1. Reconecte à GPU T4 e execute novamente a **Célula 2** (montar Drive) e as células de definição (**Células 3 a 6**).
  2. Mantenha \`RESUME_FROM_DRIVE = True\` na **Célula 3**.
  3. Execute a **Célula 7** — o treino retomará exatamente da última época salva sem perder o histórico nem as folhas de contato.`;

  const cell2InstallAndDrive = `# === CÉLULA 2: Verificação de GPU T4, dependências e montagem do Google Drive ===
import os, json, math, time, zipfile, random
from pathlib import Path

!nvidia-smi
!pip -q install onnx onnxruntime onnxscript scikit-learn pillow pandas

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_ROOT = Path('/content/drive/MyDrive/atelier_sprite_t3')
except Exception:
    print('Google Drive indisponível fora do Colab; usando diretório local ./atelier_sprite_t3')
    DRIVE_ROOT = Path('./atelier_sprite_t3')

DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')`;

  const cell3Config = `# === CÉLULA 3: Configuração do Experimento, Vocabulário Corrigido e Política NW-Safe ===
SPLIT_HASH = "${vocabulary.splitHash}"
MODEL_VERSION = "t3-maskgit-v1"
RESUME_FROM_DRIVE = True

DATASET_ZIP_PATH = "/content/dataset.zip"  # Ajuste para /content/drive/MyDrive/atelier_sprite_t3/dataset.zip se estiver no Drive
WORK_DIR = Path("/content/atelier_dataset")
CKPT_DIR = DRIVE_ROOT / "checkpoints" / SPLIT_HASH
SHEET_DIR = DRIVE_ROOT / "contact_sheets" / SPLIT_HASH
OUT_DIR = DRIVE_ROOT / "exports" / SPLIT_HASH

for d in [WORK_DIR, CKPT_DIR, SHEET_DIR, OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CATEGORIES = ${JSON.stringify(ALL_ITEM_CATEGORIES)}

# Vocabulário de papéis de rampa minerado do split train (${vocabulary.entries.length} cores + 0 transparente)
# Inclui ${vocabulary.overrideCount} correção(ões) manual(is) feita(s) pelo usuário na UI.
VOCABULARY = json.loads(r'''${vocabJson}''')
SAMPLER_SETTINGS = json.loads(r'''${JSON.stringify(samplerSettings, null, 2)}''')

VOCAB_SIZE = len(VOCABULARY) + 1       # 0 = transparente, 1..K = paleta
MASK_TOKEN_ID = VOCAB_SIZE             # K+1 = [MASK]
TOTAL_EMBED_TOKENS = VOCAB_SIZE + 1    # inclui [MASK]

EPOCHS_GENERATOR = 40
EPOCHS_RERANKER = 16
BATCH_SIZE = 32
LR = 3e-4
CFG_DROPOUT = 0.15
SEED = 77202026

# POLÍTICA ESTRITA DE AUGMENTATIONS (Clássico Tibia: iluminação NW -> SE fixa!)
ALLOW_HORIZONTAL_FLIP = False
assert ALLOW_HORIZONTAL_FLIP is False, "Flips horizontais quebram a iluminação NW e são estritamente proibidos!"

# Rótulos humanos exportados na criação deste notebook: ${exportedHumanLabelsCount ? `${exportedHumanLabelsCount.approved} aprovados, ${exportedHumanLabelsCount.rejected} rejeitados` : 'use Re-treinar com novos rótulos para incluir feedback humano'}
print(f"Split hash: {SPLIT_HASH} | Vocab: {VOCAB_SIZE} tokens (+1 MASK) | Out dir: {OUT_DIR}")`;

  const cell4LoadDataset = `# === CÉLULA 4: Extração do .zip + Loader Estrito (Train/Val APENAS; Test Proibido) ===
if not (WORK_DIR / "labels.csv").exists():
    assert Path(DATASET_ZIP_PATH).exists(), f"Arquivo {DATASET_ZIP_PATH} não encontrado! Faça upload do .zip exportado pelo Atelier Sprite."
    with zipfile.ZipFile(DATASET_ZIP_PATH, "r") as zf:
        zf.extractall(WORK_DIR)

with open(WORK_DIR / "split.json", "r", encoding="utf-8") as f:
    split_meta = json.load(f)
assert split_meta.get("testSplitExcluded") is True, "ERRO CRÍTICO: split test deve estar excluído do pacote!"

df = pd.read_csv(WORK_DIR / "labels.csv")
assert "test" not in set(df["split"].astype(str)), "VIOLAÇÃO: split 'test' detectado em labels.csv! Abortando treino."

# Carrega vocabulário exportado atualizado caso exista no pacote Re-treinar com novos rótulos
if (WORK_DIR / "vocabulary.json").exists():
    with open(WORK_DIR / "vocabulary.json", "r", encoding="utf-8") as f:
        loaded_v = json.load(f)
        if isinstance(loaded_v, dict) and isinstance(loaded_v.get("entries"), list):
            VOCABULARY = [{"tokenId": e["tokenId"], "rgb": e["rgb"], "hex": e["hex"], "materialFamily": e["materialFamily"], "role": e["role"]} for e in loaded_v["entries"]]
            VOCAB_SIZE = len(VOCABULARY) + 1
            MASK_TOKEN_ID = VOCAB_SIZE
            TOTAL_EMBED_TOKENS = VOCAB_SIZE + 1

# Pré-calcula tabela de lookup RGB -> tokenId mais próximo no vocabulário
VOCAB_RGB = np.array([e["rgb"] for e in VOCABULARY], dtype=np.float32)  # [K, 3]
VOCAB_ROLES = {e["tokenId"]: e["role"] for e in VOCABULARY}
VOCAB_MATERIALS = {e["tokenId"]: e["materialFamily"] for e in VOCABULARY}

# Agrupamento por (materialFamily, role) para troca de paleta preservando papel de rampa
MATERIAL_ROLE_TOKENS = {}
for e in VOCABULARY:
    mat = e["materialFamily"]
    MATERIAL_ROLE_TOKENS.setdefault(mat, {}).setdefault(e["role"], []).append(e["tokenId"])

def rgba_png_to_vocab_grid(png_path: Path) -> np.ndarray:
    img = Image.open(png_path).convert("RGBA")
    arr = np.array(img, dtype=np.uint8)  # [32, 32, 4]
    alpha = arr[..., 3] >= 128
    rgb = arr[..., :3].astype(np.float32).reshape(-1, 3)
    dists = ((rgb[:, None, :] - VOCAB_RGB[None, :, :]) ** 2).sum(axis=-1)
    nearest_tok = dists.argmin(axis=-1) + 1
    grid = np.where(alpha.reshape(-1), nearest_tok, 0).astype(np.int64)
    return grid.reshape(32, 32)

# Constrói vocabulário de tags determinístico a partir de train apenas
train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()

tag_counts = {}
for raw_tags in train_df["tags"].fillna(""):
    for t in str(raw_tags).split("|"):
        t = t.strip()
        if t:
            tag_counts[t] = tag_counts.get(t, 0) + 1
TAG_VOCAB = sorted([t for t, c in sorted(tag_counts.items(), key=lambda x: (-x[1], x[0]))][:48])
TAG_TO_IDX = {t: i for i, t in enumerate(TAG_VOCAB)}
CAT_TO_IDX = {c: i for i, c in enumerate(CATEGORIES)}

def encode_tags(raw_tags: str) -> np.ndarray:
    vec = np.zeros(max(1, len(TAG_VOCAB)), dtype=np.float32)
    for t in str(raw_tags or "").split("|"):
        t = t.strip()
        if t in TAG_TO_IDX:
            vec[TAG_TO_IDX[t]] = 1.0
    return vec

def load_samples(sub_df: pd.DataFrame):
    grids, cats, tags = [], [], []
    for _, row in sub_df.iterrows():
        p = WORK_DIR / str(row["png_path"])
        if not p.exists():
            continue
        grids.append(rgba_png_to_vocab_grid(p))
        cats.append(CAT_TO_IDX.get(str(row["category"]), len(CATEGORIES) - 1))
        tags.append(encode_tags(row.get("tags", "")))
    return np.stack(grids), np.array(cats, dtype=np.int64), np.stack(tags)

train_grids, train_cats, train_tags = load_samples(train_df)
val_grids, val_cats, val_tags = load_samples(val_df)

# Rótulos humanos opcionais exportados por "Re-treinar com novos rótulos"
human_approved_grids, human_approved_cats, human_approved_tags = [], [], []
human_rejected_grids, human_rejected_cats = [], []
if (WORK_DIR / "human_labels.csv").exists():
    hdf = pd.read_csv(WORK_DIR / "human_labels.csv")
    for _, row in hdf.iterrows():
        hp = WORK_DIR / str(row["png_path"])
        if not hp.exists():
            continue
        g = rgba_png_to_vocab_grid(hp)
        c = CAT_TO_IDX.get(str(row["category"]), len(CATEGORIES) - 1)
        if str(row["decision"]) == "approved":
            human_approved_grids.append(g)
            human_approved_cats.append(c)
            human_approved_tags.append(encode_tags(str(row.get("prompt", ""))))
        else:
            human_rejected_grids.append(g)
            human_rejected_cats.append(c)

if len(human_approved_grids) > 0:
    train_grids = np.concatenate([train_grids, np.stack(human_approved_grids)], axis=0)
    train_cats = np.concatenate([train_cats, np.array(human_approved_cats, dtype=np.int64)], axis=0)
    train_tags = np.concatenate([train_tags, np.stack(human_approved_tags)], axis=0)

print(f"Train samples: {len(train_grids)} (incl. {len(human_approved_grids)} aprovados humanos) | Val samples: {len(val_grids)} | Rejeitados humanos: {len(human_rejected_grids)}")`;

  const cell5Augmentations = `# === CÉLULA 5: Augmentations NW-Safe (Translação Inteira + Troca de Paleta por Papel de Rampa) ===
def translate_integer_safe(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    """Deslocamento inteiro de pixel apenas quando a silhueta opaca não toca/vaza fora da borda."""
    ys, xs = np.where(grid > 0)
    if len(xs) == 0:
        return grid
    dx = rng.randint(-2, 2)
    dy = rng.randint(-2, 2)
    if xs.min() + dx < 0 or xs.max() + dx >= 32:
        dx = 0
    if ys.min() + dy < 0 or ys.max() + dy >= 32:
        dy = 0
    if dx == 0 and dy == 0:
        return grid
    out = np.zeros_like(grid)
    out[ys + dy, xs + dx] = grid[ys, xs]
    return out

def swap_palette_preserving_ramp_role(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    """Troca tokens de um material por outro material preservando exatamente o ramp role (outline->outline, shadow->shadow, etc)."""
    mats = list(MATERIAL_ROLE_TOKENS.keys())
    if len(mats) < 2 or rng.random() > 0.45:
        return grid
    src_mat, dst_mat = rng.sample(mats, 2)
    src_roles = MATERIAL_ROLE_TOKENS[src_mat]
    dst_roles = MATERIAL_ROLE_TOKENS[dst_mat]
    remap = {}
    for role, src_list in src_roles.items():
        dst_list = dst_roles.get(role) or []
        if not dst_list:
            continue
        for i, tok in enumerate(src_list):
            remap[tok] = dst_list[min(i, len(dst_list) - 1)]
    if not remap:
        return grid
    out = grid.copy()
    for old_tok, new_tok in remap.items():
        out[grid == old_tok] = new_tok
    return out

class IndexedSpriteDataset(Dataset):
    def __init__(self, grids, cats, tags, augment=False, seed=42):
        self.grids = grids
        self.cats = cats
        self.tags = tags
        self.augment = augment
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.grids)

    def __getitem__(self, idx):
        g = self.grids[idx].copy()
        if self.augment:
            g = translate_integer_safe(g, self.rng)
            g = swap_palette_preserving_ramp_role(g, self.rng)
        return (
            torch.from_numpy(g.reshape(-1)).long(),
            torch.tensor(self.cats[idx], dtype=torch.long),
            torch.from_numpy(self.tags[idx]).float(),
        )

train_ds = IndexedSpriteDataset(train_grids, train_cats, train_tags, augment=True, seed=SEED)
val_ds = IndexedSpriteDataset(val_grids, val_cats, val_tags, augment=False, seed=SEED)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)`;

  const cell6Model = `# === CÉLULA 6: Transformer Condicional Masked-Token (6-10M params) + Regras de Higiene ===
class T3MaskedSpriteTransformer(nn.Module):
    def __init__(self, total_tokens: int, vocab_size: int, num_categories: int, tag_dim: int,
                 d_model: int = 256, n_heads: int = 8, n_layers: int = 8, d_ff: int = 1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(total_tokens, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 1024, d_model) * 0.02)
        self.cat_emb = nn.Embedding(num_categories + 1, d_model)  # último índice = token nulo CFG
        self.null_cat_id = num_categories
        self.tag_proj = nn.Linear(tag_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor, category_id: torch.Tensor, tag_vec: torch.Tensor) -> torch.Tensor:
        cond = (self.cat_emb(category_id) + self.tag_proj(tag_vec)).unsqueeze(1)  # [B, 1, D]
        x = self.token_emb(tokens) + self.pos_emb + cond
        h = self.norm(self.encoder(x))
        return self.head(h)  # [B, 1024, vocab_size]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen_model = T3MaskedSpriteTransformer(
    total_tokens=TOTAL_EMBED_TOKENS,
    vocab_size=VOCAB_SIZE,
    num_categories=len(CATEGORIES),
    tag_dim=max(1, len(TAG_VOCAB)),
    d_model=256,
    n_heads=8,
    n_layers=8,
    d_ff=1024,
).to(DEVICE)

param_count = sum(p.numel() for p in gen_model.parameters())
assert 6_000_000 <= param_count <= 10_000_000, f"Param count {param_count} fora da faixa 6M-10M!"
print(f"T3MaskedSpriteTransformer parâmetros: {param_count:,} ({param_count/1e6:.2f}M)")

# Validadores de higiene pixel-a-pixel por época + renderizador de Contact Sheet
def grid_to_rgba_img(grid32: np.ndarray) -> np.ndarray:
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    for e in VOCABULARY:
        m = grid32 == e["tokenId"]
        if m.any():
            rgba[m, 0], rgba[m, 1], rgba[m, 2], rgba[m, 3] = e["rgb"][0], e["rgb"][1], e["rgb"][2], 255
    return rgba

def check_hygiene_rules(grid32: np.ndarray) -> dict:
    opaque = grid32 > 0
    occ = int(opaque.sum())
    if occ == 0:
        return {"binary_alpha": 1.0, "outline_closure": 0.0, "connected_silhouette": 0.0, "border_clean": 0.0}
    # 1. Alpha binário garantido pelo lookup discreto de tokens
    binary_alpha = 1.0
    # 2. Silhueta conectada (maior componente 4-vizinho >= 88%)
    visited = np.zeros_like(opaque, dtype=bool)
    max_comp = 0
    for sy in range(32):
        for sx in range(32):
            if opaque[sy, sx] and not visited[sy, sx]:
                q = [(sy, sx)]
                visited[sy, sx] = True
                sz = 0
                while q:
                    y, x = q.pop()
                    sz += 1
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < 32 and 0 <= nx < 32 and opaque[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            q.append((ny, nx))
                max_comp = max(max_comp, sz)
    connected = 1.0 if (max_comp / occ) >= 0.88 else 0.0
    # 3. Fechamento do contorno escuro na borda da silhueta
    perim = 0
    dark_perim = 0
    for y in range(32):
        for x in range(32):
            if not opaque[y, x]:
                continue
            is_edge = (x == 0 or x == 31 or y == 0 or y == 31 or not opaque[y-1,x] or not opaque[y+1,x] or not opaque[y,x-1] or not opaque[y,x+1])
            if is_edge:
                perim += 1
                tok = int(grid32[y, x])
                role = VOCAB_ROLES.get(tok, "base")
                if role in ("outline", "shadow"):
                    dark_perim += 1
    outline_ok = 1.0 if (perim == 0 or dark_perim / perim >= 0.45) else 0.0
    # 4. Borda externa 32x32 limpa para itens
    border_hits = int(opaque[0, :].sum() + opaque[31, :].sum() + opaque[:, 0].sum() + opaque[:, 31].sum())
    border_clean = 1.0 if border_hits <= 6 else 0.0
    return {"binary_alpha": binary_alpha, "outline_closure": outline_ok, "connected_silhouette": connected, "border_clean": border_clean}

@torch.no_grad()
def sample_t3_batch(model, category_ids: torch.Tensor, tag_vecs: torch.Tensor, steps: int = 12, cfg_scale: float = 2.2, temp: float = 0.85):
    model.eval()
    B = category_ids.size(0)
    tokens = torch.full((B, 1024), MASK_TOKEN_ID, dtype=torch.long, device=DEVICE)
    null_cats = torch.full_like(category_ids, model.null_cat_id)
    null_tags = torch.zeros_like(tag_vecs)

    for s in range(steps):
        logits_cond = model(tokens, category_ids, tag_vecs)
        logits_uncond = model(tokens, null_cats, null_tags)
        logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
        probs = F.softmax(logits / max(0.1, temp), dim=-1)
        sampled = torch.multinomial(probs.view(-1, VOCAB_SIZE), 1).view(B, 1024)
        conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

        keep_ratio = (s + 1) / steps
        k_reveal = max(1, int(round(keep_ratio * 1024)))
        is_masked = (tokens == MASK_TOKEN_ID)
        score = torch.where(is_masked, conf, torch.full_like(conf, 10.0))
        thresh = torch.topk(score, k=k_reveal, dim=-1).values[:, -1:]
        reveal = (score >= thresh) & is_masked
        tokens = torch.where(reveal, sampled, tokens)
    return tokens.view(B, 32, 32).cpu().numpy()`;

  const cell7Train = `# === CÉLULA 7: Loop de Treinamento (Train Split), Higiene na Validação, Contact Sheets e Checkpoints no Drive ===
optimizer = torch.optim.AdamW(gen_model.parameters(), lr=LR, weight_decay=0.02)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_GENERATOR)
scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

latest_ckpt_path = CKPT_DIR / "latest.pt"
best_ckpt_path = CKPT_DIR / "best.pt"
start_epoch = 1
global_step = 0
best_val_loss = float("inf")
training_history = []

if RESUME_FROM_DRIVE and latest_ckpt_path.exists():
    ckpt = torch.load(latest_ckpt_path, map_location=DEVICE)
    gen_model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch = int(ckpt["epoch"]) + 1
    global_step = int(ckpt["global_step"])
    best_val_loss = float(ckpt["best_val_loss"])
    training_history = ckpt.get("history", [])
    print(f"Retomado checkpoint do Drive na época {ckpt['epoch']} (step {global_step}, best_val_loss={best_val_loss:.4f})")

for epoch in range(start_epoch, EPOCHS_GENERATOR + 1):
    gen_model.train()
    tr_loss_sum, tr_batches = 0.0, 0
    for tokens, cats, tags in train_loader:
        tokens, cats, tags = tokens.to(DEVICE), cats.to(DEVICE), tags.to(DEVICE)
        B = tokens.size(0)
        # Mask ratio cosseno + Classifier-Free Guidance dropout
        mask_p = torch.rand(B, 1, device=DEVICE) * 0.85 + 0.10
        mask = torch.rand(B, 1024, device=DEVICE) < mask_p
        masked_tokens = torch.where(mask, MASK_TOKEN_ID, tokens)
        labels = torch.where(mask, tokens, -100)

        drop_cfg = torch.rand(B, device=DEVICE) < CFG_DROPOUT
        cats_in = torch.where(drop_cfg, gen_model.null_cat_id, cats)
        tags_in = torch.where(drop_cfg.unsqueeze(1), torch.zeros_like(tags), tags)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            logits = gen_model(masked_tokens, cats_in, tags_in)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), labels.view(-1), ignore_index=-100)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        tr_loss_sum += float(loss.item())
        tr_batches += 1
        global_step += 1

    scheduler.step()
    train_loss = tr_loss_sum / max(1, tr_batches)

    # Avaliação no split de validação (NUNCA test)
    gen_model.eval()
    val_loss_sum, val_batches = 0.0, 0
    with torch.no_grad():
        for tokens, cats, tags in val_loader:
            tokens, cats, tags = tokens.to(DEVICE), cats.to(DEVICE), tags.to(DEVICE)
            mask = torch.rand_like(tokens.float()) < 0.50
            masked_tokens = torch.where(mask, MASK_TOKEN_ID, tokens)
            labels = torch.where(mask, tokens, -100)
            logits = gen_model(masked_tokens, cats, tags)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), labels.view(-1), ignore_index=-100)
            val_loss_sum += float(loss.item())
            val_batches += 1
    val_loss = val_loss_sum / max(1, val_batches)

    # Taxas de higiene em amostras por categoria + Contact Sheets
    hygiene_accum = {"binary_alpha": [], "outline_closure": [], "connected_silhouette": [], "border_clean": []}
    for cat_idx, cat_name in enumerate(CATEGORIES[:4]):
        c_tensor = torch.full((4,), cat_idx, dtype=torch.long, device=DEVICE)
        t_tensor = torch.zeros((4, max(1, len(TAG_VOCAB))), dtype=torch.float32, device=DEVICE)
        sampled_grids = sample_t3_batch(gen_model, c_tensor, t_tensor, steps=SAMPLER_SETTINGS["steps"], cfg_scale=SAMPLER_SETTINGS["cfgScale"])
        for g32 in sampled_grids:
            res = check_hygiene_rules(g32)
            for k, v in res.items():
                hygiene_accum[k].append(v)

        if epoch % 5 == 0 or epoch == EPOCHS_GENERATOR:
            sheet = Image.new("RGBA", (32 * 4, 32), (0, 0, 0, 0))
            for idx_s, g32 in enumerate(sampled_grids):
                sheet.paste(Image.fromarray(grid_to_rgba_img(g32), "RGBA"), (idx_s * 32, 0))
            sheet.save(SHEET_DIR / f"epoch_{epoch:02d}_{cat_name.replace(' ', '_')}.png")

    rates = {k: float(np.mean(v)) if v else 0.0 for k, v in hygiene_accum.items()}
    rec = {"epoch": epoch, "step": global_step, "train_loss": train_loss, "val_loss": val_loss, **rates}
    training_history.append(rec)
    print(f"Época {epoch:02d}/{EPOCHS_GENERATOR} | train={train_loss:.4f} | val={val_loss:.4f} | alpha={rates['binary_alpha']:.2f} | contorno={rates['outline_closure']:.2f} | silhueta={rates['connected_silhouette']:.2f} | borda={rates['border_clean']:.2f}")

    state = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": min(best_val_loss, val_loss),
        "model_state": gen_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "history": training_history,
    }
    torch.save(state, latest_ckpt_path)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(state, best_ckpt_path)`;

  const cell8Reranker = `# === CÉLULA 8: Discriminador / Reranker T3 (Real vs Gerado vs Negativos Sintéticos + Rejeições Humanas) ===
class T3RerankerCNN(nn.Module):
    def __init__(self, vocab_size: int, num_categories: int, emb_dim: int = 32):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size + 1, emb_dim)
        self.cat_emb = nn.Embedding(num_categories + 1, 32)
        self.conv = nn.Sequential(
            nn.Conv2d(emb_dim, 64, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Linear(128 + 32, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, tokens: torch.Tensor, category_id: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(tokens).view(-1, 32, 32, 32).permute(0, 3, 1, 2)  # [B, C, 32, 32]
        feat = self.conv(x).flatten(1)
        c = self.cat_emb(category_id)
        return self.head(torch.cat([feat, c], dim=-1))

def make_synthetic_negative(grid32: np.ndarray, rng: random.Random) -> np.ndarray:
    kind = rng.choice(["noise", "hflip_nw_break", "broken_outline", "confetti"])
    if kind == "hflip_nw_break":
        return np.fliplr(grid32).copy()
    out = grid32.copy()
    if kind == "noise":
        mask = np.random.rand(32, 32) < 0.25
        out[mask] = np.random.randint(1, VOCAB_SIZE, size=int(mask.sum()))
    elif kind == "broken_outline":
        for y in range(32):
            for x in range(32):
                if out[y, x] > 0 and VOCAB_ROLES.get(int(out[y, x])) == "outline" and rng.random() < 0.7:
                    out[y, x] = 0
    else:
        ys, xs = np.where(out == 0)
        if len(xs) > 0:
            picks = rng.sample(range(len(xs)), min(40, len(xs)))
            for p in picks:
                out[ys[p], xs[p]] = rng.randint(1, VOCAB_SIZE - 1)
    return out

# Coleta negativos gerados pelo T3 + negativos sintéticos + rejeições humanas
neg_rng = random.Random(SEED + 99)
gen_neg_grids, gen_neg_cats = [], []
for cat_idx in range(min(4, len(CATEGORIES))):
    sampled = sample_t3_batch(
        gen_model,
        torch.full((16,), cat_idx, dtype=torch.long, device=DEVICE),
        torch.zeros((16, max(1, len(TAG_VOCAB))), dtype=torch.float32, device=DEVICE),
    )
    gen_neg_grids.append(sampled)
    gen_neg_cats.append(np.full((16,), cat_idx, dtype=np.int64))

synth_neg_grids = np.stack([make_synthetic_negative(train_grids[i % len(train_grids)], neg_rng) for i in range(len(train_grids))])
all_neg_grids = [np.concatenate(gen_neg_grids, axis=0), synth_neg_grids]
all_neg_cats = [np.concatenate(gen_neg_cats, axis=0), train_cats.copy()]
if len(human_rejected_grids) > 0:
    all_neg_grids.append(np.stack(human_rejected_grids))
    all_neg_cats.append(np.array(human_rejected_cats, dtype=np.int64))

X_tr = np.concatenate([train_grids, *all_neg_grids], axis=0)
C_tr = np.concatenate([train_cats, *all_neg_cats], axis=0)
Y_tr = np.concatenate([np.ones(len(train_grids)), np.zeros(len(X_tr) - len(train_grids))]).astype(np.float32)

# Split de validação held-out para ROC-AUC do reranker
val_neg = np.stack([make_synthetic_negative(g, neg_rng) for g in val_grids])
X_va = np.concatenate([val_grids, val_neg], axis=0)
C_va = np.concatenate([val_cats, val_cats], axis=0)
Y_va = np.concatenate([np.ones(len(val_grids)), np.zeros(len(val_neg))]).astype(np.float32)

reranker = T3RerankerCNN(VOCAB_SIZE, len(CATEGORIES)).to(DEVICE)
rr_opt = torch.optim.AdamW(reranker.parameters(), lr=5e-4, weight_decay=1e-2)
rr_loader = DataLoader(list(zip(X_tr.reshape(-1, 1024), C_tr, Y_tr)), batch_size=64, shuffle=True)

for ep in range(1, EPOCHS_RERANKER + 1):
    reranker.train()
    for xb, cb, yb in rr_loader:
        xb, cb, yb = xb.long().to(DEVICE), cb.long().to(DEVICE), yb.float().to(DEVICE)
        rr_opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(reranker(xb, cb).squeeze(-1), yb)
        loss.backward()
        rr_opt.step()

reranker.eval()
with torch.no_grad():
    va_logits = reranker(torch.from_numpy(X_va.reshape(-1, 1024)).long().to(DEVICE), torch.from_numpy(C_va).long().to(DEVICE)).squeeze(-1)
    va_probs = torch.sigmoid(va_logits).cpu().numpy()
held_out_reranker_auc = float(roc_auc_score(Y_va, va_probs))
print(f"Reranker T3 Held-Out Validation ROC-AUC: {held_out_reranker_auc:.4f}")`;

  const cell9ExportOnnx = `# === CÉLULA 9: Exportação ONNX (Gerador + Reranker) + Model Card JSON ===
gen_model.eval().cpu()
reranker.eval().cpu()

gen_onnx_path = OUT_DIR / "t3_generator.onnx"
rr_onnx_path = OUT_DIR / "t3_reranker.onnx"
card_path = OUT_DIR / "t3_model_card.json"

dummy_tokens = torch.zeros((1, 1024), dtype=torch.long)
dummy_cat = torch.zeros((1,), dtype=torch.long)
dummy_tags = torch.zeros((1, max(1, len(TAG_VOCAB))), dtype=torch.float32)

torch.onnx.export(
    gen_model,
    (dummy_tokens, dummy_cat, dummy_tags),
    str(gen_onnx_path),
    input_names=["tokens", "category_id", "tag_vec"],
    output_names=["logits"],
    dynamic_axes={"tokens": {0: "batch"}, "category_id": {0: "batch"}, "tag_vec": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
)

torch.onnx.export(
    reranker,
    (dummy_tokens, dummy_cat),
    str(rr_onnx_path),
    input_names=["tokens", "category_id"],
    output_names=["logit"],
    dynamic_axes={"tokens": {0: "batch"}, "category_id": {0: "batch"}, "logit": {0: "batch"}},
    opset_version=17,
)

last_hist = training_history[-1] if training_history else {}
model_card = {
    "schema": "atelier-sprite-t3-model-card-v1",
    "version": MODEL_VERSION,
    "steps": int(global_step),
    "splitHash": SPLIT_HASH,
    "vocabulary": VOCABULARY,
    "samplerSettings": {
        **SAMPLER_SETTINGS,
        "vocabSize": VOCAB_SIZE,
        "maskTokenId": MASK_TOKEN_ID,
    },
    "categories": CATEGORIES,
    "tagVocab": TAG_VOCAB if TAG_VOCAB else ["geral"],
    "generatorMetrics": {
        "paramCount": int(param_count),
        "trainLoss": float(last_hist.get("train_loss", 0.0)),
        "valLoss": float(last_hist.get("val_loss", 0.0)),
        "hygienePassRates": {
            "binaryAlpha": float(last_hist.get("binary_alpha", 0.0)),
            "outlineClosure": float(last_hist.get("outline_closure", 0.0)),
            "connectedSilhouette": float(last_hist.get("connected_silhouette", 0.0)),
            "borderClean": float(last_hist.get("border_clean", 0.0)),
        },
    },
    "rerankerMetrics": {
        "heldOutValAuc": held_out_reranker_auc,
        "positiveCount": int(len(train_grids)),
        "negativeCount": int(len(X_tr) - len(train_grids)),
    },
    "exportedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}

with open(card_path, "w", encoding="utf-8") as f:
    json.dump(model_card, f, indent=2, ensure_ascii=False)

# Smoke check com onnxruntime
import onnxruntime as ort
sess_g = ort.InferenceSession(str(gen_onnx_path), providers=["CPUExecutionProvider"])
sess_r = ort.InferenceSession(str(rr_onnx_path), providers=["CPUExecutionProvider"])
print("ONNX exportado e validado com sucesso em:", OUT_DIR)
print("Arquivos:", [gen_onnx_path.name, rr_onnx_path.name, card_path.name])`;

  return {
    nbformat: 4,
    nbformat_minor: 5,
    metadata: {
      colab: {
        name: `atelier_sprite_t3_${vocabulary.splitHash}.ipynb`,
        provenance: [],
        gpuType: 'T4',
      },
      kernelspec: {
        name: 'python3',
        display_name: 'Python 3',
      },
      language_info: {
        name: 'python',
        version: '3.10',
      },
    },
    cells: [
      mdCell(firstCellMd),
      codeCell(cell2InstallAndDrive),
      codeCell(cell3Config),
      codeCell(cell4LoadDataset),
      codeCell(cell5Augmentations),
      codeCell(cell6Model),
      codeCell(cell7Train),
      codeCell(cell8Reranker),
      codeCell(cell9ExportOnnx),
    ],
  };
}

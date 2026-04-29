import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

# ──────────────────────────────────────────────
# HYPERPARAMETERS  (tuned for RTX 3050 4GB)
# ──────────────────────────────────────────────
BATCH_SIZE = 32  # was 64
BLOCK_SIZE = 128  # was 256
D_MODEL = 256  # was 384
N_HEADS = 4  # was 6
N_LAYERS = 4  # was 6
D_FF = D_MODEL * 4
DROPOUT = 0.2
LEARNING_RATE = 3e-4
MAX_ITERS = 5000
EVAL_INTERVAL = 500
EVAL_ITERS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ──────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
VOCAB_SIZE = len(chars)
print(f"Vocab size: {VOCAB_SIZE} | Dataset: {len(text):,} chars")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([d[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


# ──────────────────────────────────────────────
# 2. MODEL COMPONENTS
# ──────────────────────────────────────────────


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.
    Built entirely from nn.Linear — no nn.MultiheadAttention.
    """

    def __init__(self):
        super().__init__()
        assert D_MODEL % N_HEADS == 0
        self.head_dim = D_MODEL // N_HEADS

        # Fused QKV projection — one Linear for all heads
        self.qkv_proj = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.out_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)

        # Causal mask (lower-triangular)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)).view(
                1, 1, BLOCK_SIZE, BLOCK_SIZE
            ),
        )

    def forward(self, x):
        B, T, C = x.shape  # batch, seq len, d_model

        # Project and split into Q, K, V
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.split(D_MODEL, dim=-1)  # each: (B, T, C)

        # Reshape to (B, n_heads, T, head_dim)
        def reshape(t):
            return t.view(B, T, N_HEADS, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # Scaled dot-product attention
        scale = self.head_dim**-0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, H, T, T)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=DROPOUT, training=self.training)

        out = attn @ v  # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # merge heads
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Position-wise FFN: Linear → GELU → Linear"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_MODEL, D_FF),
            nn.GELU(),
            nn.Linear(D_FF, D_MODEL),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block: LayerNorm → Attention → LayerNorm → FFN"""

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = MultiHeadSelfAttention()
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ffn = FeedForward()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # residual connection
        x = x + self.ffn(self.ln2(x))  # residual connection
        return x


class ShakespeareTransformer(nn.Module):
    """
    Full GPT-style Transformer
    Layers: Embedding + Positional Embedding → N×TransformerBlock → LayerNorm → Linear
    """

    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, D_MODEL)  # learned positional
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

        # Weight tying: share token embedding & output projection weights
        self.tok_emb.weight = self.lm_head.weight

        self.apply(self._init_weights)
        print(f"Model parameters: {sum(p.numel() for p in self.parameters()):,}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        positions = torch.arange(T, device=DEVICE)

        x = self.tok_emb(idx) + self.pos_emb(positions)  # (B, T, D_MODEL)
        x = F.dropout(x, p=DROPOUT, training=self.training)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, VOCAB_SIZE)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]  # crop to context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # last time step

            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx


# ──────────────────────────────────────────────
# 3. TRAINING
# ──────────────────────────────────────────────


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(EVAL_ITERS):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = np.mean(losses)
    model.train()
    return out


model = ShakespeareTransformer().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


# Learning-rate scheduler: cosine decay with linear warmup
def get_lr(step):
    warmup = 100
    if step < warmup:
        return LEARNING_RATE * step / warmup
    progress = (step - warmup) / (MAX_ITERS - warmup)
    return LEARNING_RATE * 0.5 * (1 + np.cos(np.pi * progress))


train_losses, val_losses, loss_steps = [], [], []

print("\nTraining started...")
progress = tqdm(range(MAX_ITERS), desc="Training", unit="step")

for step in progress:
    # Update learning rate
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    # Evaluation checkpoint
    if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
        losses = estimate_loss(model)
        train_losses.append(losses["train"])
        val_losses.append(losses["val"])
        loss_steps.append(step)
        progress.set_description(
            f"Step {step} | train={losses['train']:.4f} val={losses['val']:.4f}"
        )

    # Training step
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # Live stats in bar
    progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

print("\nTraining complete!")

# Save model
torch.save(model.state_dict(), "shakespeare_model.pt")
print("Model saved to shakespeare_model.pt")

# ──────────────────────────────────────────────
# 4. LOSS PLOT
# ──────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(loss_steps, train_losses, "b-o", label="Train Loss", markersize=4)
plt.plot(loss_steps, val_losses, "r-o", label="Val Loss", markersize=4)
plt.xlabel("Training Step")
plt.ylabel("Cross-Entropy Loss")
plt.title("Shakespeare Transformer — Training & Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("loss_plot.png", dpi=150)
plt.show()
print("Loss plot saved to loss_plot.png")

# ──────────────────────────────────────────────
# 5. INFERENCE — Generate Shakespeare
# ──────────────────────────────────────────────
model.eval()
print("\n" + "=" * 60)
print("GENERATED TEXT (500 tokens):")
print("=" * 60)

seed = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)  # start with newline
generated = model.generate(seed, max_new_tokens=500, temperature=0.8, top_k=40)
print(decode(generated[0].tolist()))
print("=" * 60)

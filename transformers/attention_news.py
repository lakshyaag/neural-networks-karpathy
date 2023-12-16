from rich import print as rprint
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- HYPERPARAMETERS ---
BATCH_SIZE = 32
BLOCK_SIZE = 8

MAX_ITERS = 5000
LEARNING_RATE = 1e-3

EVAL_INTERVAL = 300
EVAL_ITERS = 200

N_EMBED = 32
N_HEADS = 4
N_BLOCKS = 4
DROPOUT = 0.2

torch.manual_seed(1337)

# --- INPUT DATA ---
with open("./random_news_data.txt", encoding="utf-8") as f:
    text = f.read()
rprint(f"Text length: {len(text)}")

chars = sorted(list(set(text)))
VOCAB_SIZE = len(chars)
rprint(f"Vocab size: {VOCAB_SIZE}")

# Character mapping
s_to_i = {s: i for i, s in enumerate(chars)}
i_to_s = {i: s for i, s in enumerate(chars)}


def encode(s):
    return [s_to_i[c] for c in s]


def decode(i_):
    return "".join([i_to_s[i] for i in i_])


# Train/validation split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(text) * 0.9)
train_data = data[:n]
val_data = data[n:]


# --- BATCHING ---
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    X = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return X, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, y = get_batch(split)
            logits, loss = model(X, y)
            losses[k] = loss.item()

        out[f"{split}_loss"] = losses.mean()

    model.train()
    return out


# --- ATTENTION ---
class Head(nn.Module):
    """
    One head of self-attention.
    """

    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        weights = q @ k.transpose(-2, -1) * (C**-0.5)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        out = weights @ v
        return out


class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention in parallel.
    """

    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(N_EMBED, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


# --- FEED FORWARD ---
class FeedForward(nn.Module):
    """
    Feed-forward network with ReLU activation.
    """

    def __init__(self, n_embed, scale_factor=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, scale_factor * n_embed),
            nn.ReLU(),
            nn.Linear(scale_factor * n_embed, n_embed),  # Projection layer
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


# --- BLOCK (ATTENTION + FEED FORWARD) ---
class Block(nn.Module):
    """
    One block of Transformer with communication followed by computation
    """

    def __init__(self, n_embed, n_heads):
        super().__init__()
        self.sa_heads = MultiHeadAttention(n_heads, n_embed // n_heads)
        self.ffwd = FeedForward(n_embed, 4)

        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))  # Residual connection + attention
        x = x + self.ffwd(self.ln2(x))  # Residual connection + feed-forward
        return x


# --- MODEL ---
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBED)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.blocks = nn.Sequential(
            *[Block(N_EMBED, N_HEADS) for _ in range(3)] + [nn.LayerNorm(N_EMBED)]
        )
        self.lm_head = nn.Linear(N_EMBED, VOCAB_SIZE)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embedding = self.token_embedding_table(idx)
        position_embedding = self.position_embedding_table(torch.arange(T))
        x = token_embedding + position_embedding
        x = self.blocks(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


model = BigramLanguageModel()

print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# --- TRAINING ---
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for steps in range(MAX_ITERS):
    if steps % EVAL_INTERVAL == 0:
        metrics = estimate_loss()
        rprint(
            f'Steps: {steps} | Train loss: {metrics["train_loss"]:.4f} | Val loss: {metrics["val_loss"]:.4f}'
        )

    X, y = get_batch("train")

    logits, loss = model(X, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# --- SAVE MODEL ---
torch.save(model.state_dict(), "./model.pt")

# --- GENERATION ---
context = torch.zeros((1, 1), dtype=torch.long)
rprint(decode(model.generate(context, 10000)[0].tolist()))

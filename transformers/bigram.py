from rich import print as rprint
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- HYPERPARAMETERS ---
BATCH_SIZE = 32
BLOCK_SIZE = 8

MAX_ITERS = 3000
LEARNING_RATE = 1e-3

EVAL_INTERVAL = 300
EVAL_ITERS = 200

N_EMBED = 32

torch.manual_seed(1337)

# --- INPUT DATA ---
with open("./input.txt", encoding="utf-8") as f:
    text = f.read()
rprint(f"Text length: {len(text)}")

chars = sorted(list(set(text)))
VOCAB_SIZE = len(chars)

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


# --- MODEL ---
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBED)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.lm_head = nn.Linear(N_EMBED, VOCAB_SIZE)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embedding = self.token_embedding_table(idx)
        position_embedding = self.position_embedding_table(torch.arange(T))
        x = token_embedding + position_embedding
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
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


model = BigramLanguageModel()

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

# --- GENERATION ---
context = torch.zeros((1, 1), dtype=torch.long)
rprint(decode(model.generate(context, 1000)[0].tolist()))

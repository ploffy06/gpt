import torch
import torch.nn as nn
from torch.nn import functional as F

"""
A modified GPT model in which the positional embedding is added in the attention head instead
"""

batch_size = 64
block_size = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 1e-4
epochs = 3000
eval_inters = 10
n_embed = 192
n_head=6
n_layer = 6
dropout = 0.2

class Head(nn.Module):
    def __init__(self, head_size):
        super(Head, self).__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.position_embedding = nn.Embedding(block_size, n_embed)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # DIET-ABS: adding positional embedding at head
        pos_emb = self.position_embedding(torch.arange(T, device=device)) # T, C
        pos_k = self.key(pos_emb)
        pos_q = self.query(pos_emb)
        pos_emb = pos_k @ pos_q.transpose(0, 1)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei + pos_emb # broadcasting -> right aliged and and new dimension added
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # calculating "affinities"
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadedAttention(nn.Module):
    """
    multiple heads of self-attention in parallel
    """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        run all attention heads in parallel and concatenating them
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out)) # projection is linear outcome of previous layer
        return out

class FeedForward(nn.Module):
    """
    A simple linear layer followed by a non-linearity
    """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """
    Transformer block: communication followed by computation
    """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadedAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # "fork" off, do some computation and come back
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTv2(nn.Module):
    def __init__(self, vocab_size, n_embed):
        super(GPTv2, self).__init__()
        """
        vocab_size: number of unique words in dataset
        n_embed: number of embedding dimensions
        """
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding(idx) # (B, T, embed size)
        x = tok_emb # in broadcasting, get right aligned and new dimention added -> (B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop context we feed into forward (because of positional embedding)i
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


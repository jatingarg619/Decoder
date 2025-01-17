import os
import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from datetime import datetime

# Hyperparameters
learning_rate = 3e-4  # Standard transformer learning rate
min_lr = 3e-5
warmup_iters = 100
lr_decay_iters = 2500
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
batch_size = 128
block_size = 256  # Increased context
eval_interval = 100
eval_iters = 50
log_interval = 10
gradient_accumulation_steps = 4

# Model architecture
@dataclass
class GPTConfig:
    block_size: int = block_size
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 16
    n_embd: int = 1024  # Increased embedding dimension
    dropout: float = 0.1
    bias: bool = True

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Scale up attention heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Use flash attention for faster training
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Wider MLP layers
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x, y

def get_lr(it):
    # 1) Linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) If it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) In between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def log_to_markdown(message, filename='training_log.md'):
    """Append a log message to the markdown file with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(log_dir, filename)
    
    # Create the file with headers if it doesn't exist
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write("# Training Log\n\n")
    
    # Append the message
    try:
        with open(log_path, 'a') as f:
            f.write(f"### {timestamp}\n{message}\n\n")
    except Exception as e:
        print(f"Error writing to log file: {e}")

def main():
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize logging
    log_to_markdown("## Training Started")
    log_to_markdown(f"- Device: {device}")
    
    # Load the data
    with open('input.txt', 'r') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for i,ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    # Initialize the model
    model = GPT(GPTConfig(vocab_size=vocab_size))
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())/1e6
    print(f"Model parameters: {num_params:.2f}M")
    log_to_markdown(f"- Model Parameters: {num_params:.2f}M")
    log_to_markdown(f"- Vocabulary Size: {vocab_size}")
    log_to_markdown(f"- Training Data Size: {len(train_data)}")
    log_to_markdown(f"- Validation Data Size: {len(val_data)}")
    
    # Initialize optimizer with larger eps for stability
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
        eps=1e-5
    )
    
    log_to_markdown("### Hyperparameters:")
    log_to_markdown(f"```\nLearning Rate: {learning_rate}\nBatch Size: {batch_size}\nBlock Size: {block_size}\nWeight Decay: {weight_decay}\nBetas: ({beta1}, {beta2})\nGradient Accumulation Steps: {gradient_accumulation_steps}\n```")
    
    # Training loop
    best_val_loss = float('inf')
    iter_num = 0
    start_time = time.time()
    
    while True:
        # Accumulate gradients over multiple forward passes
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0
        
        for _ in range(gradient_accumulation_steps):
            # Get batch and learning rate
            xb, yb = get_batch(train_data, block_size, batch_size // gradient_accumulation_steps)
            xb, yb = xb.to(device), yb.to(device)
            
            # Forward pass
            logits, loss = model(xb, yb)
            loss = loss / gradient_accumulation_steps  # Scale loss
            accumulated_loss += loss.item() * gradient_accumulation_steps
            loss.backward()
        
        # Update learning rate
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # Gradient clipping and optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        # Logging
        if iter_num % log_interval == 0:
            print(f"iter {iter_num}: loss {accumulated_loss:.4f}, lr {lr:e}")
            log_to_markdown(f"- Iteration {iter_num}: Training Loss = {accumulated_loss:.4f}, Learning Rate = {lr:e}")
            
        # Evaluation
        if iter_num % eval_interval == 0:
            losses = torch.zeros(eval_iters)
            model.eval()
            with torch.no_grad():
                for k in range(eval_iters):
                    xb, yb = get_batch(val_data, block_size, batch_size)
                    xb, yb = xb.to(device), yb.to(device)
                    logits, loss = model(xb, yb)
                    losses[k] = loss.item()
            val_loss = losses.mean()
            model.train()
            print(f"step {iter_num}: val loss {val_loss:.4f}")
            log_to_markdown(f"- Validation Loss = {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                log_to_markdown(f"- New Best Validation Loss = {best_val_loss:.6f}")
                if best_val_loss < 0.099999:
                    elapsed = time.time() - start_time
                    log_to_markdown(f"## Training Completed")
                    log_to_markdown(f"- Final Loss: {best_val_loss:.6f}")
                    log_to_markdown(f"- Time Elapsed: {elapsed/3600:.2f} hours")
                    print(f"Achieved target loss of {best_val_loss:.6f}")
                    break
                    
        iter_num += 1
        if iter_num > lr_decay_iters:
            log_to_markdown("## Training Stopped - Max Iterations Reached")
            break

if __name__ == '__main__':
    main() 
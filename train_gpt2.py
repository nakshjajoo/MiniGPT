from dataclasses import dataclass
import inspect
import math
import sys
import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
import tiktoken
import time

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.c_proj.MINIGPT_SCALE_INIT = 1

        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # not really a bias, actually is a attention mask for the causal self-attention, but following OpenAI naming
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size)) # (T, T) -> (1, 1, T, T) for broadcasting

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x) # (B, T, 3 * C)
        q, k, v = qkv.split(self.n_embd, dim=2) # (B, T, C)

        # reshape q, k, v into (B, nh, T, hs) to group all data for a single attention head together
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, nh, hs).transpose(1, 2) -> (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, nh, hs).transpose(1, 2) -> (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, T, nh, hs).transpose(1, 2) -> (B, nh, T, hs)
        
        # attention (materializes the large (T, T) matrix for all the queries and keys)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)

        # Use flash attention instead of manually calculating the attention, avoid materializing the large (T, T) matrix (repeated gpu memory read/write)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # (B, nh, T, hs)
        
        # reassemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, nh, T, hs) -> (B, T, nh, ns) -> (B, T, C=nh*hs)
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.MINIGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
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

@dataclass
class GPTConfig:
    vocab_size: int = 50257 #number of tokens: 256 byte tokens + 50k BPE merges + 1 <|endoftext|> token
    n_embd: int = 768 # embedding dimensionality
    n_layer: int = 12 # number of transformer blocks
    n_head: int = 12 # number of attention heads
    block_size: int = 1024 # context length

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight # share the weights between the token embedding and the output layer

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear): 
            std = 0.02
            if hasattr(module, 'MINIGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # 0.02 is roughly the same as what we get if we use Xavier initialization (1/sqrt(n_embd))

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd)
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x) # (B, T, n_embd)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) # (B*T, vocab_size) - flatten the batch and time dimensions for F.cross_entropy
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768), # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]

        # vocab size and block do not change based on the model size
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init the pretrained model from huggingface
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy and load the weights from huggingface into our model
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        # OpenAI uses a "Conv1D" module, but we only want to use a vanilla Linear Layer
        # meaning that we have to transpose these weights when we import them since
        # Conv1D stores weights as (output_dim, input_dim, 1)
        # but Linear stores them as (input_dim, output_dim)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            # check if the key is a Conv1D weight which needs to be transposed
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape # check if the transposed shape matches
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizer(self, weight_decay, learning_rate, device):
        # starting with all of the candidate parameters that require gradients
        param_dict = {pname: p for pname, p in self.named_parameters() if p.requires_grad}

        # create optim groups, any parameters that is 2D will be weight decayed, otherwise not
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for pname, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for pname, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num of decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num of non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # create the AdamW optimizer and use the fused version if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
        

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
    
        # at init, load tokens from disk and store in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens from disk")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T) # (B, T)
        y = buf[1:].view(B, T) # (B, T)
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would exceed the number of tokens, reset the position
        if self.current_position + (B * T + 1) >= len(self.tokens):
            self.current_position = 0
        return x, y

#autodetect the best available device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"
print(f"Using device: {device}")

# gradient accumulation
total_batch_size = 524288 # 2^19, ~ 0.5M tokens 
B = 8 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T) == 0, f"total_batch_size {total_batch_size} must be divisible by B*T {B*T}"
grad_accum_steps = total_batch_size // (B * T) # number of gradient accumulation steps
print(f"total desired batch size: {total_batch_size}")
print(f"gradient accumulation steps: {grad_accum_steps}")

# adding fake tokens at the which would never be used by making the vocab size "a nicer number" (50,257 -> 50,304)
# this is not necessary, but it makes the model run faster on CUDA
model = GPT(GPTConfig(vocab_size=50304))
model.eval()
model.to(device)
model = torch.compile(model)

train_loader = DataLoaderLite(B=8, T=1024)
torch.set_float32_matmul_precision('high')

max_lr = 6e-4
min_lr = max_lr * 0.1 # 10% of max_lr
warmup_steps = 10
max_steps = 50

def get_lr(iter):
    # Linear warmup for warmup_steps steps
    if iter < warmup_steps:
        return max_lr * (iter + 1) / warmup_steps
    
    # after max_steps, return min learning rate
    if iter > max_steps:
        return min_lr

    # Cosine decay for the rest of the steps in the middle
    decay_ratio = (iter - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio)) # coefficient goes from 1 to 0
    return min_lr + (max_lr - min_lr) * coeff

#optimize
optimizer = model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, device=device)
for step in range(50):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch() # (B, T)
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x.to(device), y.to(device))
        
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so simulate the mean reduction
        loss = loss / grad_accum_steps
        loss_accum += loss.detach() 
        loss.backward() #since we are not doing torch.zero_grad, this accumulates the gradients

    # clip the gradients to avoid exploding gradients
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # use the scheduler to determine the learning rate for this step
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize() if device == "cuda" else None
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = (train_loader.B * train_loader.T) * grad_accum_steps
    tokens_per_sec = tokens_processed / dt # tokens per second
    print(f"step {step} | loss: {loss_accum.item()} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tokens/sec: {tokens_per_sec:.2f}")

sys.exit(0)

num_return_sequences = 5
max_length = 100


enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("The transformer is a deep learning architecture")
tokens = torch.tensor(tokens, dtype=torch.long) # (T,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (num_return_sequences, T)
x = tokens.to(device)

#generation: initially, x is (B, T)
torch.manual_seed(42)
torch.mps.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits, loss = model(x) # (B, T, vocab_size)
        #take the last token logits
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probs
        probs = F.softmax(logits, dim=-1) # (B, vocab_size)
        # do a top-k sampling of 50
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # (B, 50), (B, 50)
        # sample from the topk probs
        ix = torch.multinomial(topk_probs, num_samples=1) # (B, 1)
        # gather the indices from the topk_indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append the sampled token to the input
        x = torch.cat((x, xcol), dim=1) # (B, T+1)

# decode the tokens and print
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
    print()
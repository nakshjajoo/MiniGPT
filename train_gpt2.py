from dataclasses import dataclass
import inspect
import math
import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2LMHeadModel
import tiktoken
import time

from hellaswag import iterate_example, render_example

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
    block_size: int = 2048 # context length

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
        

def load_tokens(filename):
    # load the tokens from a numpy file
    tokens = np.load(filename)
    # convert to torch tensor
    tokens = torch.tensor(tokens, dtype=torch.long)
    return tokens

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        #get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()
        
    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T) # (B, T)
        y = buf[1:].view(B, T) # (B, T)
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would exceed the number of tokens, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) >= len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x, y

""" Helper function for HellaSwag eval; similar to that in hellawag.py """
def get_most_likely_row(tokens, mask, logits):
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    
    # get the average loss just for the completion region (where mask == 1) in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    
    # sum and divide by the number of 1s in the mask to get the average loss per token
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


# Setting up DDP
# torchrun command sets the env vars RANK, LOCAL_RANK and WORLD_SIZE
# RANK is the global rank of the process, LOCAL_RANK is the local rank of the process on the node, and WORLD_SIZE is the total number of processes
ddp = int(os.environ.get('RANK', -1)) != -1 # check if this is a distributed run

if ddp:
    assert torch.cuda.is_available(), "DDP requires CUDA"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # only the master process will do the logging, checkpoints, etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    #autodetect the best available device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# gradient accumulation
total_batch_size = 524288 # 2^19, ~ 0.5M tokens 
B = 4 # micro batch size
T = 2048 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, f"total_batch_size {total_batch_size} must be divisible by B * T * ddp_world_size {B*T*ddp_world_size}"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size) # number of gradient accumulation steps for each process

if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"gradient accumulation steps: {grad_accum_steps}")

# adding fake tokens at the which would never be used by making the vocab size "a nicer number" (50,257 -> 50,304)
# this is not necessary, but it makes the model run faster on CUDA
model = GPT(GPTConfig(vocab_size=50304))
model.eval()
model.to(device)
# torch.compile interferes with HellaSwag eval and Generation
# due to dynamic operations and control flow such as while loop and torch.multinomial in generation code
use_compile = False 
if use_compile:
    model = torch.compile(model)

# if using ddp, wrap the model in DDP
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # get the raw model from the DDP wrapper

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')
torch.set_float32_matmul_precision('high')

max_lr = 15e-4 # 6e-4
min_lr = max_lr * 0.1 # 10% of max_lr
warmup_steps = 715 # 375M tokens / (2^19) tokens per step (375M tokens were used for warmup in the original paper)
max_steps = 19073 * 4 # 10e9 tokens / (2^19) tokens per step; 4 epochs

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

optimizer = raw_model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, device=device)
enc = tiktoken.get_encoding("gpt2")

# create the log directory we will write checkpoints and log to
# for train/val loss, and hellaswag eval
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, 'w') as f:
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # evaluate validation loss once in a while
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, 'a') as f:
                f.write(f"step {step} | val_loss: {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                # save the model checkpoints
                checkpoint_path = os.path.join(log_dir, f"checkpoint-{step:05d}.pt")
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": raw_model.config,
                    "step": step,
                    "val_loss": val_loss_accum.item(),
                    "seeds": {"torch": 1337, "generator": 42 + ddp_rank},
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"saved checkpoint to {checkpoint_path}")
                
    # evaluate hellaswag once in a while
    if (step % 250 == 0 or last_step) and not use_compile:
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_example("val")):
            # ensures that each DDP process only evaluates a subset of the validation examples
            # and that no two processes handle the same example
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into the tokens, mask and label
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)

            #get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")
        
    # generate from the model once in a while (except 0, which is noise)
    if (step > 0 and step % 250 == 0) and not use_compile:
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                #take the last token logits
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probs
                probs = F.softmax(logits, dim=-1) # (B, vocab_size)
                # do a top-k sampling of 50
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # (B, 50), (B, 50)
                # sample from the topk probs
                ix = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng) # (B, 1)
                # gather the indices from the topk_indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append the sampled token to the input
                xgen = torch.cat((xgen, xcol), dim=1) # (B, T+1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    #one step of optimization
    model.train()
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
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) # only sync gradients on the last micro step
        loss.backward() #since we are not doing torch.zero_grad(), this accumulates the gradients

    if ddp:
        # sync the gradients across all processes
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        # loss_accum is now the average loss across all processes

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
    tokens_processed = (train_loader.B * train_loader.T) * grad_accum_steps * ddp_world_size # number of tokens processed in this step
    tokens_per_sec = tokens_processed / dt # tokens per second
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    # destroy the process group
    destroy_process_group()
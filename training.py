import os
import math
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import tiktoken

from hellaswag import iterate_example, render_example
from model import GPT, GPTConfig

log_dir = "log"
max_steps = 19073 * 4 # 10e9 tokens / (2^19) tokens per step; 4 epochs

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
        buf = self.tokens[self.current_position : self.current_position + B * T + 1] # (1 extra token for the next token prediction)
        x = buf[:-1].view(B, T) # (B, T)
        y = buf[1:].view(B, T) # (B, T)
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would exceed the number of tokens, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) >= len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

""" Helper function for HellaSwag eval; similar to that in hellawag.py """
def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    # to predict the next token, shift the logits left and the target tokens right to align
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

if __name__ == "__main__":
    # Main Training Loop
    # Simple launch:
    # python training.py
    # DDP launch:
    # torchrun --standalone --nproc_per_node=NUM_GPUS training.py

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
        #autodetect the best available device (mps is not supported by F.scaled_dot_product_attention)
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        print(f"Using device: {device}")

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # Hyperparameters
    total_batch_size = 524288 #tokens # 2^19, ~ 0.5M tokens 

    # gradient accumulation
    B = 4 # micro batch size # bound by the GPU memory
    T = 2048 # sequence length
    assert total_batch_size % (B * T * ddp_world_size) == 0, f"total_batch_size {total_batch_size} must be divisible by B * T * ddp_world_size {B*T*ddp_world_size}"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size) # number of gradient accumulation steps for each process

    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"gradient accumulation steps: {grad_accum_steps}")

    # MODEL INIT
    # adding fake tokens at the end which would never be used by making the vocab size "a nicer number" for more efficient GPU computation (50,257 -> 50,304)
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
    # max_steps = 19073 * 4 # 10e9 tokens / (2^19) tokens per step; 4 epochs

    optimizer = raw_model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, device=device)
    enc = tiktoken.get_encoding("gpt2")


    # create the log directory we will write checkpoints and log to
    # for train/val loss, and hellaswag eval
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, 'w') as f:
        pass


    # Training loop
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
                if (step % 5000 == 0 or last_step):
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
                num_total = torch.tensor(num_total, dtype=torch.long, device=device)
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
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (B, T)
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
            # instead of a SUM we want MEAN. Scale the loss here to simulate the mean reduction
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
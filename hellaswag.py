"""
Downloads and evaluates HellaSwag in Python.
https://github.com/rowanz/hellaswag

Example HellaSwag json item:

{
    "ind": 24, 
    "activity_label": "Roof shingle removal", 
    "ctx_a": "A man is sitting on a roof.", 
    "ctx_b": "he", 
    "ctx": "A man is sitting on a roof. he", 
    "split": "val", 
    "split_type": "indomain", 
    "label": 3, 
    "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], 
    "source_id": "activitynet~v_-JhWjGDPHMY"
}

ind: dataset ID
activity_label: The ActivityNet or WikiHow label for this example
context: There are two formats. The full context is in ctx. When the context ends in an (incomplete) noun phrase, like for ActivityNet, this incomplete noun phrase is in ctx_b, and the context up until then is in ctx_a. This can be useful for models such as BERT that need the last sentence to be complete. However, it's never required. If ctx_b is nonempty, then ctx is the same thing as ctx_a, followed by a space, then ctx_b.
endings: a list of 4 endings. The correct index is given by label (0,1,2, or 3)
split: train, val, or test.
split_type: indomain if the activity label is seen during training, else zeroshot
source_id: Which video or WikiHow article this example came from

The validation set of HellaSwag has a total of 10,042 examples
"""

import os
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

import argparse

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

def download_file(url: str, fname: str, chunk_size=1024):
    """Download a file from a URL """
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")

def download(split):
    """Download the HellaSwag dataset in the DATA_CACHE_DIR directory"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)

def render_example(example):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates with one being the right answer)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods, and 0 for all the padding tokens)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """

    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # data needed to reproduce this eval on the C setting (how likely is the model to generate each of the 4 possible endings given the same context)
    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # gather up all the tokens
    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    token_rows = [] # will eventually be of size (4, T), one for each of the 4 candidate
    mask_rows = [] # 1s in the ending region, 0s elsewhere (in context region and padding)
    for end in endings:
        end_tokens = enc.encode(" " + end) # prefix with space to match GPT2 tokenizer
        token_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens)) # 0 for context tokens and padding, 1 for ending tokens
        data["ending_tokens"].append(end_tokens)
    
    # pad all the rows to the same length
    max_len = max(len(row) for row in token_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (token_row, mask_row) in enumerate(zip(token_rows, mask_rows)):
        tokens[i, :len(token_row)] = torch.tensor(token_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)
    
    return data, tokens, mask, label

def iterate_example(split):
    """Iterate over the examples in the HellaSwag dataset"""
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")) as f:
        for line in f:
            example = json.loads(line)
            yield example
    
@torch.no_grad()
def evaluate(model_type, device):
    torch.set_float32_matmul_precision('high') # use tf32 for faster inference
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)
    # model = torch.compile(model) # torch.compile is not supported in transformers fully yet, can enable optionally

    num_correct = 0 # num of correct predictions based on the total loss of all the tokens in the ending region
    num_correct_norm = 0 # num of correct predictions based on the average loss of the tokens in the ending region
    num_total = 0 # total number of examples
    for example in iterate_example("val"):
        data, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # get the logits for the ending tokens
        logits = model(tokens).logits
        # evaluate the autoregressive loss at all positions
        # to predict the next token, shift the logits left and the target tokens right to align sizes
        shift_logits = (logits[..., :-1, :]).contiguous() # dont want the last logit, since there is nothing to predict after that(4, N-1, vocab_size)
        shift_tokens = (tokens[..., 1:]).contiguous() #dont want the first token, since it can not be predicted (4, N-1)
        
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1)) # (4*(N-1), vocab_size)
        flat_shift_tokens = shift_tokens.view(-1) # (4*(N-1))
        
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction="none") # (4*(N-1),)
        shift_losses = shift_losses.view(tokens.size(0), -1) # (4, N-1)

        # now get the average loss just for the completion region (where mask == 1), in each row
        shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
        # only calculates loss where mask == 1, i.e., the ending portion
        masked_shift_losses = shift_losses * shift_mask # (4, N-1)

        # sum and divide by the number of 1s in the mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1) # (4,)

        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        # get the index of the minimum loss
        pred = sum_loss.argmin().item() # most likely sequence based on the total loss of all the tokens in the ending region
        pred_norm = avg_loss.argmin().item() # most likely sequence based on the average loss of the tokens in the ending region

        # accumulate stats
        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)

        print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

        # debug: pretty print a few examples, and the losses in each case
        if num_total < 10:
            print("---")
            print(f"Context:\n {example['ctx']}")
            print(f"Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"predicted: {pred_norm}, actual: {label}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, default="gpt2", help="the model type to use")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="the device to use")
    args = parser.parse_args()
    evaluate(args.model_type, args.device)
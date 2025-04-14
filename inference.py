import torch
import torch.nn.functional as F
import tiktoken
import argparse
import os

from model import GPT, GPTConfig
from training import log_dir, max_steps

def generate_text(model, enc, prompt, device, max_tokens_to_gen):
    """ Generates text using the GPT model """

    model.eval()

    prompt_tokens = enc.encode(prompt)
    x = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
    x_len = x.size(1)
    total_len = x_len + max_tokens_to_gen

    # for reproducibility
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42)

    with torch.no_grad():
        while x.size(1) < total_len:
            # crop context if it exceeds the model's context length
            current_block_size = model.config.block_size
            x_cond = x if x.size(1) <= current_block_size else x[:, -current_block_size:]

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, _ = model(x_cond) # (B, T, vocab_size)

            # get the logits for the last token
            logits = logits[:, -1, :] # (B, vocab_size)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, vocab_size)

            # sample next token
            ix = torch.multinomial(probs, num_samples=1, generator=sample_rng) # (B, 1)

            # append the sampled token to the input
            x = torch.cat((x, ix), dim=1) # (B, T+1)

    generated_tokens = x[0].tolist()
    decoded_text = enc.decode(generated_tokens)

    return decoded_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a pre-trained GPT model.")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum number of new tokens to generate")
    args = parser.parse_args()

    # Set device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    
    print(f"Using device: {device}")
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    # Load checkpoint
    checkpoint_path = os.path.join(log_dir, f"checkpoint-00000.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Instantiate model
    if 'config' in checkpoint:
        config = checkpoint["config"]
    else:
        raise ValueError("Checkpoint does not contain model configuration.")

    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    print("Model loaded successfully")

    # Load tokenizer
    enc = tiktoken.get_encoding("gpt2")

    # Prompt for text generation
    prompt = input("Enter your prompt: ")
    if not prompt or len(prompt) == 0 or prompt.isspace():
        raise ValueError("Prompt cannot be empty.")

    # Generate text
    generated_text = generate_text(model, enc, prompt, device, args.max_tokens)
    print("\n-----------------------------------")
    print(generated_text)
    print("-----------------------------------")
    
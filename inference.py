import torch
import torch.nn.functional as F
import tiktoken
import argparse
import os

from model import GPT
from training import log_dir, max_steps

USER_TOKEN = "<|USER|>"
ASSISTANT_TOKEN = "<|ASSISTANT|>"
END_TOKEN = "<|END|>"
SPECIAL_TOKEN_IDS = {
    USER_TOKEN: 50257,
    ASSISTANT_TOKEN: 50258,
    END_TOKEN: 50259,
}
END_TOKEN_ID = SPECIAL_TOKEN_IDS[END_TOKEN] # # Specific ID for stopping generation

def create_custom_encoding():
    """Creates the tiktoken encoding used during fine-tuning."""
    
    base_encoding = tiktoken.get_encoding("gpt2")
    # Ensure the special tokens map includes the base special tokens plus our custom ones
    custom_special_tokens = {
        **base_encoding._special_tokens, # Include base tokens like <|endoftext|> if needed
        **SPECIAL_TOKEN_IDS
    }

    custom_encoding = tiktoken.Encoding(
        name="gpt2_custom_chat", # Assign a unique name
        pat_str=base_encoding._pat_str,
        mergeable_ranks=base_encoding._mergeable_ranks,
        special_tokens=custom_special_tokens,
    )
    return custom_encoding

def generate_response(model, enc, history_tokens, device, max_new_tokens=100, temperature=0.7, top_k=50):
    """
    Generates the assistant's response based on the conversation history.
    Stops generation if the END_TOKEN_ID is produced.
    """
    model.eval()

    # Truncate history if it exceeds block size, keeping the most recent tokens
    current_block_size = model.config.block_size
    if len(history_tokens) > current_block_size:
        # print(f"\n[Warning: Context truncated. History length {len(history_tokens)} > Block size {current_block_size}]")
        history_tokens = history_tokens[-current_block_size:]
    
    # Convert history tokens to tensor
    x = torch.tensor(history_tokens, dtype=torch.long, device=device).unsqueeze(0) # (B=1, T)

    generated_sequence = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Ensure context window doesn't exceed block size during generation itself
            x_cond = x if x.size(1) <= current_block_size else x[:, -current_block_size:]

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, _ = model(x_cond) # (B, T, vocab_size)

            # Get logits for the very last token prediction
            logits = logits[:, -1, :] # (B, vocab_size)

            # Apply temperature scaling
            if temperature > 0: # Avoid division by zero if temperature is 0
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                topk_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < topk_logits[:, [-1]]] = -float('Inf') # Mask out logits below k-th value

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, vocab_size)

            # Sample the next token ID
            # Use multinomial sampling
            next_token_id_tensor = torch.multinomial(probs, num_samples=1) # (B, 1)
            next_token_id = next_token_id_tensor.item()

            if next_token_id == END_TOKEN_ID:
                break # Stop generation if end token is sampled

            # Append the generated token ID to our result sequence
            generated_sequence.append(next_token_id)

            # Append the sampled token to the input sequence for the next step
            x = torch.cat((x, next_token_id_tensor), dim=1) # (B, T+1)
        
        return generated_sequence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a pre-trained GPT model.")
    parser.add_argument("--max_new_tokens", type=int, default=150, help="Maximum number of new tokens to generate per turn.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0 for greedy).")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering (0 to disable).")
    args = parser.parse_args()

    # Set device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    
    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint_path = os.path.join(log_dir, "finetuned_final.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Fine-tuned checkpoint not found: {checkpoint_path}")
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Instantiate model
    if 'config' not in checkpoint:
        raise ValueError("Checkpoint does not contain model configuration ('config').")
    config = checkpoint["config"]

    model = GPT(config)
    state_dict = checkpoint["model"]
    # Fix potential DDP state dict keys if saved directly from DDP wrapper
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict=state_dict, strict=False)
    model.to(device)
    model.eval()
    print("Model loaded successfully")

    # Load tokenizer
    custom_enc = create_custom_encoding()

    print("\nStarting conversation (type 'exit' to quit).")
    conversation_history_tokens = [] # Stores token IDs of the full conversation

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting conversation.")
            break
        if not user_input.strip() or user_input.isspace(): # Skip empty input
            continue
        
        # Format user input with special tokens
        user_turn_str = f" {USER_TOKEN} {user_input.strip()} {END_TOKEN}"
        user_turn_tokens = custom_enc.encode(user_turn_str, allowed_special='all')

        assistant_prompt_str = f" {ASSISTANT_TOKEN}"
        assistant_prompt_tokens = custom_enc.encode(assistant_prompt_str, allowed_special='all')

        # Prepare prompt for the model (history + user turn + assistant cue)
        prompt_tokens = conversation_history_tokens + user_turn_tokens + assistant_prompt_tokens

        # Generate response
        assistant_response_tokens = generate_response(
            model,
            custom_enc,
            prompt_tokens,
            device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )

        # Decode the generated tokens and print
        assistant_response_str = custom_enc.decode(assistant_response_tokens).strip()
        print(f"Assistant: {assistant_response_str}")

        # Update conversation history
        # Append the assistant prompt tokens AND the generated response tokens AND the end token ID
        conversation_history_tokens.extend(user_turn_tokens)
        conversation_history_tokens.extend(assistant_prompt_tokens)
        conversation_history_tokens.extend(assistant_response_tokens)
        conversation_history_tokens.append(END_TOKEN_ID) # Explicitly add END token after assistant response
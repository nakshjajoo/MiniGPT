# MiniGPT

A PyTorch reproduction implementation of GPT-style models, focusing on both pre-training and fine-tuning for conversational capabilities. This project aims for clarity and simplicity while providing a functional base for training and interacting with transformer language models.

The core model `model.py` is a clean implementation of a decoder-only Transformer. The training script `training.py` handles large-scale pre-training on the FineWeb Edu dataset, followed by fine-tuning on conversational data (Anthropic hh-rlhf) using custom special tokens. An interactive command-line interface `inference.py` allows for chatting with the fine-tuned model.

## Core Features

* **GPT Model:** A standard decoder-only Transformer architecture `model.py` (details below).
* **Pre-training:** Script `training.py` to pre-train the model on 40B tokens of FineWeb Edu dataset. Supports DDP for multi-GPU training, mixed precision (`bfloat16`), gradient accumulation, and multiple optimization techniques.
* **Fine-tuning:** Integrated fine-tuning phase in `training.py` to adapt the pre-trained model for conversational interface, by fine-tuning using Anthropic's HH-RLHF dataset and custom special tokens (`<|USER|>`, `<|ASSISTANT|>`, `<|END|>`). Easily adaptable to other datasets and formatting.
* **Conversational Inference:** An interactive command-line chat script `inference.py` that loads the fine-tuned model and allows back-and-forth conversation, managing history and using special tokens for formatting.
* **Evaluation:** Includes HellaSwag evaluation logic (`hellaswag.py` and integrated into `training.py`) to benchmark model performance on common sense reasoning.

## File Structure

* `model.py`: Defines the `GPTConfig` and the `GPT` model class, including `Block`, `CausalSelfAttention`, and `MLP` modules.
* `training.py`: Orchestrates the entire training process:
    * Handles DDP setup.
    * Initializes model, optimizer, and data loaders.
    * Contains the main pre-training loop with validation loss and HellaSwag evaluation.
    * Contains the subsequent fine-tuning loop using the Anthropic/hh-rlhf dataset and different set of hyperparameters.
    * Includes `DataLoader` for sharded pre-training data and `FineTuneDataLoader` for conversational data.
    * Manages checkpoint saving.
* `inference.py`: Provides an interactive command-line interface for chatting with the fine-tuned model, handling conversation history and special tokens.
* `hellaswag.py`: Utilities for downloading, processing, and evaluating the HellaSwag dataset.
* `fineweb.py`: A script to download and pre-process the FineWeb Dataset into tokenized shards (`.npy` files) suitable for the `DataLoader`.
* `log/`: Directory where training logs (`log.txt`, `print_log.txt`) and model checkpoints (`.pt` files) are saved.

## Model Architecture

The model implemented follows the GPT-2 architecture closely:

* **Type:** Decoder-only Transformer.
* **Layers:** 12 (`n_layer`)
* **Heads:** 12 (`n_head`)
* **Embedding Size:** 768 (`n_embd`)
* **Context Length:** 2048 tokens (`block_size`)
* **Vocabulary Size:** 50304 (`vocab_size`) - Padded GPT-2 vocab to be divisible by a higher power of 2 for GPU optimization, also accommodates custom fine-tuning tokens.
* **Normalization:** Layer Normalization applied before attention/MLP blocks (Pre-LN).
* **Activation:** GELU.
* **Attention:** Causal self-attention using PyTorch's `scaled_dot_product_attention`.
* **Positional Embeddings:** Learned absolute positional embeddings.
* **Initialization:** Custom initialization (`_init_weights`) with scaling for projection layers.
* **Weight Sharing:** Input token embeddings and final output projection layer share weights.

This configuration corresponds roughly to the GPT-2 Small (124M parameter) model size.

## Training

The `training.py` script handles two distinct phases:

### Pre-training

* **Dataset:** FineWeb Edu (40 Billion tokens), expected to be pre-processed into shards by `prepare.py`.
* **Optimizer:** AdamW (`betas=(0.9, 0.95)`, `eps=1e-8`).
* **Weight Decay:** 0.1 (applied to weights, not biases/norms).
* **Learning Rate:** Cosine decay schedule with linear warmup.
    * `max_lr`: 6e-4
    * `min_lr`: 6e-5 (10% of max_lr)
    * `warmup_steps`: 715
* **Batching:**
    * Effective Batch Size: 524,288 tokens (`total_batch_size`)
    * Micro-Batch Size: 4 (`B`)
    * Sequence Length: 2048 (`T`)
    * Gradient Accumulation Steps: Calculated based on world size (`total_batch_size // (B * T * ddp_world_size)`).
* **Regularization:** Gradient Clipping (norm 1.0) and dropout of 0.1.
* **Mixed Precision:** `bfloat16` via `torch.autocast` and `torch.cuda.amp.GradScaler` (when CUDA is available).
* **Duration:** Configured for 4 epochs over the 10B token dataset (`max_steps = 19073 * 4`).

### Fine-tuning

* **Objective:** Adapt the pre-trained model for conversational interaction.
* **Dataset:** Anthropic HH-RLHF (using 'chosen' dialogues), loaded via `datasets` library.
* **Formatting:** Dialogues are formatted using special tokens:
    * `<|USER|>` (ID: 50257)
    * `<|ASSISTANT|>` (ID: 50258)
    * `<|END|>` (ID: 50259) - Marks the end of a turn.
* **Tokenizer:** A custom `tiktoken` encoding (`gpt2_custom`) is created internally by `FineTuneDataLoader` to handle these special tokens.
* **Optimizer:** AdamW
* **Learning Rate:** Fixed `ft_lr = 3e-5`.
* **Batching:**
    * Effective Batch Size: 64 tokens (`ft_total_batch_size`)
    * Micro-Batch Size: 8 (`ft_B`)
    * Sequence Length: 2048 (`T`)
    * Gradient Accumulation Steps: Calculated (`ft_total_batch_size // (ft_B * T * ddp_world_size)`).
* **Duration:** Configured for 2000 steps (`ft_max_steps`).
* **Extensibility:** The `FineTuneDataLoader` and fine-tuning loop can be adapted to other datasets by modifying the data loading, `_prepare_data` formatting logic, and hyperparameters.

## Usage

### Dependencies

Install the required libraries. A `requirements.txt` file would typically list:


`torch` \
`numpy` \
`tiktoken` \
`datasets` \
`transformers # Often a dependency for datasets` \
`requests # For hellaswag download` \
`tqdm # For progress bars`


Install using pip: \
`pip install -r requirements.txt` \
or \
`pip install torch numpy tiktoken datasets transformers requests tqdm`.

### Data Preparation

1.  **Pre-training Data:** Used the script `fineweb.py` to download and process the FineWeb dataset into tokenized `.npy` shards. Placed these shards in a directory named `edu_fineweb10B`. Split the shards into `train` and `val` subsets.
2.  **Fine-tuning Data:** The `Anthropic/hh-rlhf` dataset will be downloaded automatically by the `datasets` library when `training.py` is run for the first time (during the fine-tuning phase).

### Training

* **Single GPU / CPU:**
    ```bash
    python training.py
    ```
* **Multi-GPU (Distributed Data Parallel):**
    ```bash
    torchrun --standalone --nproc_per_node=NUM_GPUS training.py
    ```
    Replace `NUM_GPUS` with the number of GPUs available on your node.

The script will first run the pre-training phase, save checkpoints periodically, and then automatically proceed to the fine-tuning phase if the HH-RLHF dataset is loaded successfully. Final models (`pretrained_final.pt`, `finetuned_final.pt`) are saved in the `log/` directory.

### Inference (Conversational Chat)

Ensure you have a fine-tuned checkpoint (e.g., `log/finetuned_final.pt`).

```bash
python inference.py [--checkpoint PATH_TO_CHECKPOINT] [--max_new_tokens N] [--temperature T] [--top_k K]

--checkpoint: Path to the fine-tuned model checkpoint (defaults to log/finetuned_final.pt).

--max_new_tokens: Max tokens the assistant generates per turn (default: 150).

--temperature: Sampling temperature (default: 0.7). Higher values make output more random.

--top_k: Nucleus sampling parameter (default: 50). Restricts sampling to top K likely tokens.

The script will load the model and enter an interactive loop where you can chat with the assistant. Type exit to quit.
```


### Evaluation
Validation Loss: Calculated periodically during both pre-training and fine-tuning phase.
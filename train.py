# /// script
# dependencies = ["torch", "einops", "jaxtyping", "numpy", "tiktoken", "sentencepiece", "tqdm"]
# ///

import torch
from config import GPTConfig
from training_args import TransformerTrainingArgs
from trainer import TransformerTrainer
from dataset import TransformerDataset
from gpt import GPTWithEinops, GPTWithoutEinops

device = torch.device(
    "mps" if torch.backends.mps.is_available(
    ) else "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {device}")

# Load training data
with open("training.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Initialize config
# Use GPTConfig.small() for faster training on Mac, GPTConfig() for full model
USE_SMALL_MODEL = True  # Set to False for full GPT-2 size model

if USE_SMALL_MODEL:
    cfg = GPTConfig.small()
    print("Using SMALL model config (faster for Mac)")
else:
    cfg = GPTConfig()
    print("Using FULL model config (GPT-2 size)")

# Create dataset
TOKENIZER_TYPE = "bpe"
dataset = TransformerDataset(text, cfg, tokenizer_type=TOKENIZER_TYPE)
dataset.print_info()

# Get train/val splits
X_train, Y_train = dataset.get_train_data()
X_val, Y_val = dataset.get_val_data()

# Update cfg (dataset updates d_vocab internally)
cfg = dataset.cfg

# Initialize model
MODEL_TYPE = "with_einops"  # Options: "with_einops", "without_einops"

if MODEL_TYPE == "with_einops":
    model = GPTWithEinops(cfg)
else:
    model = GPTWithoutEinops(cfg)

model = model.to(device)
print(f"\nInitialized {MODEL_TYPE} model")
print(f"Model on device: {next(model.parameters()).device}")
print(
    f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Training setup
args = TransformerTrainingArgs()
# Reduce eval_iters and batch_size for faster training on Mac
if USE_SMALL_MODEL:
    args.eval_iters = 50  # Faster evaluation for small model
    args.batch_size = 16  # Smaller batch for Mac memory

# Create trainer
trainer = TransformerTrainer(
    model=model,
    args=args,
    X_train=X_train,
    Y_train=Y_train,
    X_val=X_val,
    Y_val=Y_val,
    device=device,
)

# Start training
trainer.train()

print("\n" + "=" * 50)
print("Training complete!")
print("=" * 50)
print(f"Model saved to: {args.save_dir}/final_model.pt")
print("\nTo generate text, run:")
print(
    f"  uv run infer.py --checkpoint {args.save_dir}/final_model.pt --prompt 'Your prompt here'")

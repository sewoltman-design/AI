# AI

This repository contains a minimal Python script for training and generating text with a small Transformer language model.

## Setup

Install dependencies (Torch is the only runtime dependency):

```bash
pip install -r requirements.txt
```

## Training

Train on your own UTF-8 text file using the `train` command. Key hyperparameters let you resize the model:

```bash
python transformer_cli.py train \
  --input-file path/to/your_text.txt \
  --output-model checkpoints/char_transformer.pt \
  --block-size 128 \
  --batch-size 32 \
  --epochs 5 \
  --d-model 128 \
  --nhead 4 \
  --num-layers 2 \
  --dim-feedforward 256 \
  --dropout 0.1
```

Increasing `--d-model`, `--nhead`, and `--num-layers` grows the model; lowering them makes it smaller. Use `--cpu` to force CPU training even when CUDA is available.

## Generation

Load a saved checkpoint and generate text from a prompt:

```bash
python transformer_cli.py generate \
  --model-path checkpoints/char_transformer.pt \
  --prompt "Once upon a time" \
  --max-new-tokens 200 \
  --temperature 0.8 \
  --top-k 8
```

`--temperature` controls randomness, and `--top-k` restricts sampling to the most likely tokens.

## Point-and-click UI

You can launch a lightweight Tkinter interface that bundles training and generation controls in one window:

```bash
python transformer_cli.py gui
```

In the UI you can:
- Paste or load training text, tweak hyperparameters (sequence length, model width/heads/layers, learning rate, etc.), and start training with a single click.
- Pick a saved checkpoint, set prompt/temperature/top-k/max tokens, and generate directly in the app.

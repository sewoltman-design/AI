import argparse
import threading
import tkinter as tk
from dataclasses import dataclass, asdict
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class ModelConfig:
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    max_seq_len: int = 256


class CharDataset(Dataset):
    def __init__(self, text: str, block_size: int, stoi: Dict[str, int], itos: List[str]):
        self.block_size = block_size
        self.stoi = stoi
        self.itos = itos
        self.data = [self.stoi[ch] for ch in text]

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(0)
        return x + self.pe[:seq_len]


class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, config: ModelConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.pos_encoder = PositionalEncoding(config.d_model, config.max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.decoder = nn.Linear(config.d_model, vocab_size)
        self.config = config

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src shape: [seq_len, batch_size]
        embeddings = self.token_embedding(src) * (self.config.d_model ** 0.5)
        embeddings = self.pos_encoder(embeddings)
        output = self.transformer_encoder(embeddings)
        logits = self.decoder(output)
        return logits


def build_vocab(text: str) -> Tuple[Dict[str, int], List[str]]:
    unique_chars = sorted(list(set(text)))
    itos = unique_chars
    stoi = {ch: i for i, ch in enumerate(itos)}
    return stoi, itos


def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*batch)
    x = torch.stack(xs).transpose(0, 1)  # [seq_len, batch]
    y = torch.stack(ys).transpose(0, 1)
    return x, y


def train_language_model(
    text: str,
    output_model: Path,
    block_size: int = 128,
    batch_size: int = 32,
    epochs: int = 5,
    lr: float = 3e-4,
    grad_clip: float = 1.0,
    model_config: Optional[ModelConfig] = None,
    cpu: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> None:
    stoi, itos = build_vocab(text)
    dataset = CharDataset(text, block_size, stoi, itos)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    config = model_config or ModelConfig(max_seq_len=block_size)
    model = TransformerLanguageModel(len(itos), config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    if progress:
        progress("Starting training...\n")
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(dataloader))
        message = f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f}\n"
        print(message.strip())
        if progress:
            progress(message)

    checkpoint = {
        "model_state": model.state_dict(),
        "config": asdict(config),
        "itos": itos,
    }
    output_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_model)
    print(f"Model saved to {output_model}")
    if progress:
        progress(f"Model saved to {output_model}\n")


def load_model(model_path: Path, cpu: bool = False) -> Tuple[TransformerLanguageModel, ModelConfig, List[str], Dict[str, int], torch.device]:
    checkpoint = torch.load(model_path, map_location="cpu")
    config = ModelConfig(**checkpoint["config"])
    itos = checkpoint["itos"]
    stoi = {ch: i for i, ch in enumerate(itos)}
    model = TransformerLanguageModel(len(itos), config)
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    model.to(device)
    model.eval()
    return model, config, itos, stoi, device


def generate_from_model(
    model_path: Path,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 0,
    cpu: bool = False,
) -> str:
    model, config, itos, stoi, device = load_model(model_path, cpu=cpu)
    context = prompt
    indices = [stoi.get(ch, 0) for ch in context][-config.max_seq_len :]
    input_ids = torch.tensor(indices, dtype=torch.long, device=device).unsqueeze(1)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids)
            logits = logits[-1, 0] / temperature
            if top_k > 0:
                values, _ = torch.topk(logits, top_k)
                min_keep = values[-1]
                logits = torch.where(logits < min_keep, torch.tensor(float("-inf"), device=logits.device), logits)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_id.view(1, 1)], dim=0)
        input_ids = input_ids[-config.max_seq_len :]
    generated = "".join(itos[idx] for idx in input_ids.squeeze().tolist())
    return generated


def train_model(args: argparse.Namespace) -> None:
    text = Path(args.input_file).read_text(encoding="utf-8")
    config = ModelConfig(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_len=args.block_size,
    )
    train_language_model(
        text=text,
        output_model=Path(args.output_model),
        block_size=args.block_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        grad_clip=args.grad_clip,
        model_config=config,
        cpu=args.cpu,
    )


def generate_text(args: argparse.Namespace) -> None:
    generated = generate_from_model(
        model_path=Path(args.model_path),
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        cpu=args.cpu,
    )
    print(generated)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or generate with a tiny Transformer language model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model on your text")
    train_parser.add_argument("--input-file", required=True, help="Path to a UTF-8 text file for training")
    train_parser.add_argument("--output-model", default="checkpoints/char_transformer.pt", help="Where to save the trained model")
    train_parser.add_argument("--block-size", type=int, default=128, help="Sequence length for training")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    train_parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    train_parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping value")
    train_parser.add_argument("--d-model", type=int, default=128, help="Embedding dimension (width of the model)")
    train_parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    train_parser.add_argument("--num-layers", type=int, default=2, help="Number of Transformer encoder layers")
    train_parser.add_argument("--dim-feedforward", type=int, default=256, help="Size of the feedforward layer")
    train_parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    train_parser.add_argument("--cpu", action="store_true", help="Force training on CPU even if CUDA is available")
    train_parser.set_defaults(func=train_model)

    gen_parser = subparsers.add_parser("generate", help="Generate text from a trained model")
    gen_parser.add_argument("--model-path", required=True, help="Path to a saved model checkpoint")
    gen_parser.add_argument("--prompt", default="The", help="Prompt text to start generation")
    gen_parser.add_argument("--max-new-tokens", type=int, default=100, help="Number of new characters to generate")
    gen_parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature")
    gen_parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling (0 to disable)")
    gen_parser.add_argument("--cpu", action="store_true", help="Force generation on CPU")
    gen_parser.set_defaults(func=generate_text)

    gui_parser = subparsers.add_parser("gui", help="Launch a Tkinter UI for training and generation")
    gui_parser.set_defaults(func=lambda _args: launch_gui())

    return parser


class TransformerGUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Tiny Transformer Trainer")
        self._build_training_section()
        self._build_generation_section()

    def _build_training_section(self) -> None:
        frame = tk.LabelFrame(self.root, text="Train Model", padx=8, pady=8)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        btn_row = tk.Frame(frame)
        btn_row.pack(fill=tk.X, pady=4)
        tk.Button(btn_row, text="Load Text File", command=self.browse_text_file).pack(side=tk.LEFT)
        tk.Button(btn_row, text="Clear", command=self.clear_training_text).pack(side=tk.LEFT, padx=4)

        self.training_text = tk.Text(frame, height=8, wrap=tk.WORD)
        self.training_text.pack(fill=tk.BOTH, expand=True)

        params = tk.Frame(frame)
        params.pack(fill=tk.X, pady=4)
        self.block_size_var = tk.IntVar(value=128)
        self.batch_size_var = tk.IntVar(value=32)
        self.epochs_var = tk.IntVar(value=5)
        self.lr_var = tk.DoubleVar(value=3e-4)
        self.grad_clip_var = tk.DoubleVar(value=1.0)
        self.d_model_var = tk.IntVar(value=128)
        self.nhead_var = tk.IntVar(value=4)
        self.num_layers_var = tk.IntVar(value=2)
        self.dim_ff_var = tk.IntVar(value=256)
        self.dropout_var = tk.DoubleVar(value=0.1)
        self.output_path_var = tk.StringVar(value="checkpoints/char_transformer.pt")

        self._add_param(params, "Block size", self.block_size_var, 0, tooltip="Sequence length")
        self._add_param(params, "Batch size", self.batch_size_var, 1)
        self._add_param(params, "Epochs", self.epochs_var, 2)
        self._add_param(params, "LR", self.lr_var, 3)
        self._add_param(params, "Grad clip", self.grad_clip_var, 4)
        self._add_param(params, "d_model", self.d_model_var, 5)
        self._add_param(params, "n_heads", self.nhead_var, 6)
        self._add_param(params, "Layers", self.num_layers_var, 7)
        self._add_param(params, "FF dim", self.dim_ff_var, 8)
        self._add_param(params, "Dropout", self.dropout_var, 9)

        output_row = tk.Frame(frame)
        output_row.pack(fill=tk.X, pady=4)
        tk.Label(output_row, text="Save model to").pack(side=tk.LEFT)
        tk.Entry(output_row, textvariable=self.output_path_var, width=40).pack(side=tk.LEFT, padx=4)
        tk.Button(output_row, text="Browse", command=self.browse_output_path).pack(side=tk.LEFT)

        self.force_cpu_var = tk.BooleanVar(value=False)
        tk.Checkbutton(frame, text="Force CPU", variable=self.force_cpu_var).pack(anchor=tk.W, pady=2)

        tk.Button(frame, text="Start Training", command=self.start_training).pack(pady=6)

        self.log_text = tk.Text(frame, height=6, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _build_generation_section(self) -> None:
        frame = tk.LabelFrame(self.root, text="Generate", padx=8, pady=8)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        model_row = tk.Frame(frame)
        model_row.pack(fill=tk.X, pady=4)
        self.model_path_var = tk.StringVar(value="checkpoints/char_transformer.pt")
        tk.Label(model_row, text="Model path").pack(side=tk.LEFT)
        tk.Entry(model_row, textvariable=self.model_path_var, width=40).pack(side=tk.LEFT, padx=4)
        tk.Button(model_row, text="Browse", command=self.browse_model_path).pack(side=tk.LEFT)

        prompt_row = tk.Frame(frame)
        prompt_row.pack(fill=tk.X, pady=4)
        tk.Label(prompt_row, text="Prompt").pack(side=tk.LEFT)
        self.prompt_var = tk.StringVar(value="The")
        tk.Entry(prompt_row, textvariable=self.prompt_var, width=40).pack(side=tk.LEFT, padx=4)

        params = tk.Frame(frame)
        params.pack(fill=tk.X, pady=4)
        self.max_tokens_var = tk.IntVar(value=100)
        self.temperature_var = tk.DoubleVar(value=1.0)
        self.top_k_var = tk.IntVar(value=0)
        self._add_param(params, "New tokens", self.max_tokens_var, 0)
        self._add_param(params, "Temp", self.temperature_var, 1)
        self._add_param(params, "Top-k", self.top_k_var, 2)

        self.force_cpu_gen_var = tk.BooleanVar(value=False)
        tk.Checkbutton(frame, text="Force CPU", variable=self.force_cpu_gen_var).pack(anchor=tk.W, pady=2)

        tk.Button(frame, text="Generate", command=self.start_generation).pack(pady=6)

        self.output_text = tk.Text(frame, height=6, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True)

    def _add_param(self, parent: tk.Frame, label: str, variable: tk.Variable, column: int, tooltip: str = "") -> None:
        cell = tk.Frame(parent)
        cell.grid(row=0, column=column, padx=3, pady=2)
        tk.Label(cell, text=label).pack()
        entry = tk.Entry(cell, textvariable=variable, width=8)
        entry.pack()

    def browse_text_file(self) -> None:
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            content = Path(file_path).read_text(encoding="utf-8")
            self.training_text.delete("1.0", tk.END)
            self.training_text.insert(tk.END, content)

    def browse_output_path(self) -> None:
        file_path = filedialog.asksaveasfilename(defaultextension=".pt", initialfile=self.output_path_var.get())
        if file_path:
            self.output_path_var.set(file_path)

    def browse_model_path(self) -> None:
        file_path = filedialog.askopenfilename(filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")])
        if file_path:
            self.model_path_var.set(file_path)

    def clear_training_text(self) -> None:
        self.training_text.delete("1.0", tk.END)

    def start_training(self) -> None:
        text = self.training_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showerror("Missing text", "Please paste training text or load a file first.")
            return

        config = ModelConfig(
            d_model=self.d_model_var.get(),
            nhead=self.nhead_var.get(),
            num_layers=self.num_layers_var.get(),
            dim_feedforward=self.dim_ff_var.get(),
            dropout=self.dropout_var.get(),
            max_seq_len=self.block_size_var.get(),
        )

        def run_training() -> None:
            try:
                train_language_model(
                    text=text,
                    output_model=Path(self.output_path_var.get()),
                    block_size=self.block_size_var.get(),
                    batch_size=self.batch_size_var.get(),
                    epochs=self.epochs_var.get(),
                    lr=self.lr_var.get(),
                    grad_clip=self.grad_clip_var.get(),
                    model_config=config,
                    cpu=self.force_cpu_var.get(),
                    progress=self.append_log,
                )
                self.root.after(0, lambda: messagebox.showinfo("Training complete", "Model saved."))
            except Exception as exc:  # noqa: BLE001
                self.root.after(0, lambda: messagebox.showerror("Training failed", str(exc)))

        threading.Thread(target=run_training, daemon=True).start()

    def start_generation(self) -> None:
        model_path = Path(self.model_path_var.get())
        if not model_path.exists():
            messagebox.showerror("Missing model", "Please pick a valid model checkpoint.")
            return

        prompt = self.prompt_var.get()

        def run_generation() -> None:
            try:
                text = generate_from_model(
                    model_path=model_path,
                    prompt=prompt,
                    max_new_tokens=self.max_tokens_var.get(),
                    temperature=self.temperature_var.get(),
                    top_k=self.top_k_var.get(),
                    cpu=self.force_cpu_gen_var.get(),
                )
                self._set_output_text(text)
            except Exception as exc:  # noqa: BLE001
                self.root.after(0, lambda: messagebox.showerror("Generation failed", str(exc)))

        threading.Thread(target=run_generation, daemon=True).start()

    def append_log(self, text: str) -> None:
        def _update() -> None:
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, text)
            self.log_text.configure(state=tk.DISABLED)
            self.log_text.see(tk.END)

        self.root.after(0, _update)

    def _set_output_text(self, text: str) -> None:
        def _update() -> None:
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, text)
            self.output_text.see(tk.END)

        self.root.after(0, _update)

    def run(self) -> None:
        self.root.mainloop()


def launch_gui() -> None:
    TransformerGUI().run()


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

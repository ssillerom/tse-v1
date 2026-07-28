"""Train and evaluate the V1 decoder-only language model."""

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Literal, cast

import tiktoken
import torch
import wandb
from torch.utils.data import DataLoader

from src.data.dataset import PretrainingDataset
from src.data.manifest import load_manifest
from src.model.config import ModelConfig
from src.model.gpt import GPT
from src.training.checkpoint import save_checkpoint
from src.training.evaluation import perplexity_from_loss
from src.training.trainer import StepMetrics, TrainingConfig, evaluate, train
from src.training.wandb_logging import WandbEvaluationLogger, WandbRun

WandbMode = Literal["online", "offline", "disabled"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/v1"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--vocab-size", type=int, default=50_304)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=25)
    parser.add_argument("--max-learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.1)

    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--sample-interval", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--checkpoint-interval", type=int, default=100)

    parser.add_argument("--wandb-project", default="llm-from-scratch")
    parser.add_argument("--wandb-name")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    return parser


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    return torch.device(requested)


def _load_encoding(manifest_path: Path) -> tiktoken.Encoding:
    tokenizer = load_manifest(manifest_path).tokenizer
    if tokenizer is None:
        raise ValueError("Manifest must declare tokenizer metadata for evaluation")
    encoding = tiktoken.get_encoding(tokenizer.encoding_name)
    if encoding.eot_token != tokenizer.eot_token_id:
        raise ValueError(
            f"Manifest eot_token={tokenizer.eot_token_id} does not match "
            f"encoding {tokenizer.encoding_name!r}"
        )
    return encoding


def main(argv: list[str] | None = None) -> int:
    """Run one finite V1 experiment."""
    arguments = _build_parser().parse_args(argv)
    if arguments.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if arguments.checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")

    device = _resolve_device(arguments.device)
    torch.manual_seed(arguments.seed)
    if device.type == "mps":
        torch.mps.manual_seed(arguments.seed)

    encoding = _load_encoding(arguments.manifest)
    if arguments.vocab_size < encoding.n_vocab:
        raise ValueError(
            f"vocab_size={arguments.vocab_size} is smaller than tokenizer vocabulary "
            f"{encoding.n_vocab}"
        )

    model_config = ModelConfig(
        vocab_size=arguments.vocab_size,
        d_model=arguments.d_model,
        n_layers=arguments.n_layers,
        n_heads=arguments.n_heads,
        max_seq_len=arguments.seq_len,
        dropout=arguments.dropout,
        use_sdpa=True,
    )
    training_config = TrainingConfig(
        max_steps=arguments.max_steps,
        grad_accum_steps=arguments.grad_accum_steps,
        warmup_steps=arguments.warmup_steps,
        max_learning_rate=arguments.max_learning_rate,
        min_learning_rate=arguments.min_learning_rate,
        max_grad_norm=arguments.max_grad_norm,
        eval_interval=arguments.eval_interval,
        eval_batches=arguments.eval_batches,
    )
    train_dataset = PretrainingDataset(
        manifest_path=arguments.manifest,
        split="train",
        seq_len=model_config.max_seq_len,
    )
    validation_dataset = PretrainingDataset(
        manifest_path=arguments.manifest,
        split="validation",
        seq_len=model_config.max_seq_len,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        drop_last=False,
    )

    model = GPT(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.max_learning_rate,
        betas=(0.9, 0.95),
        weight_decay=arguments.weight_decay,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    run_name = arguments.wandb_name or (
        f"v1-{model_config.n_layers}l-{model_config.d_model}d-{arguments.seed}"
    )
    wandb_mode = cast(WandbMode, arguments.wandb_mode)

    with wandb.init(
        project=arguments.wandb_project,
        name=run_name,
        mode=wandb_mode,
        config={
            "model": asdict(model_config),
            "training": asdict(training_config),
            "optimizer": {
                "name": "AdamW",
                "betas": [0.9, 0.95],
                "weight_decay": arguments.weight_decay,
            },
            "data": {
                "manifest": str(arguments.manifest),
                "batch_size": arguments.batch_size,
            },
            "seed": arguments.seed,
            "device": str(device),
            "parameter_count": parameter_count,
        },
    ) as raw_run:
        run = cast(WandbRun, raw_run)
        logger = WandbEvaluationLogger(
            run=run,
            model=model,
            encoding=encoding,
            device=device,
            sample_interval=arguments.sample_interval,
            max_new_tokens=arguments.max_new_tokens,
        )
        initial_validation_loss = evaluate(
            model=model,
            batches=validation_loader,
            device=device,
            max_batches=training_config.eval_batches,
        )
        run.log(
            {
                "trainer/global_step": 0,
                "validation/loss": initial_validation_loss,
                "validation/perplexity": perplexity_from_loss(initial_validation_loss),
            }
        )
        logger.log_samples(step=0)

        arguments.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        def on_step(metrics: StepMetrics) -> None:
            if metrics.step % arguments.checkpoint_interval == 0:
                save_checkpoint(
                    path=arguments.checkpoint_dir / f"step_{metrics.step:06d}.pt",
                    model=model,
                    optimizer=optimizer,
                    step=metrics.step,
                    training_config=training_config,
                )
            logger(metrics)

        history = train(
            model=model,
            optimizer=optimizer,
            train_batches=train_loader,
            config=training_config,
            device=device,
            validation_batches=validation_loader,
            on_step=on_step,
        )
        final_step = history[-1].step
        if final_step % arguments.checkpoint_interval != 0:
            save_checkpoint(
                path=arguments.checkpoint_dir / f"step_{final_step:06d}.pt",
                model=model,
                optimizer=optimizer,
                step=final_step,
                training_config=training_config,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

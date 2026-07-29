"""Train and evaluate the V1 decoder-only language model."""

import argparse
import hashlib
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
from src.training.checkpoint import (
    TrainingRunConfig,
    restore_checkpoint,
    restore_latest_checkpoint,
    save_checkpoint,
)
from src.training.evaluation import perplexity_from_loss
from src.training.rng import capture_torch_rng_state, restore_torch_rng_state
from src.training.trainer import Precision, StepMetrics, TrainingConfig, evaluate, train
from src.training.wandb_logging import WandbEvaluationLogger, WandbRun

WandbMode = Literal["online", "offline", "disabled"]
PrecisionArgument = Literal["auto", "fp32", "bf16"]
ADAMW_BETAS = (0.9, 0.95)
ADAMW_EPS = 1e-8
DEFAULT_WANDB_PROJECT = "llm-from-scratch"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/v1"))
    parser.add_argument(
        "--resume",
        help="checkpoint path to restore, or 'latest' for the newest valid checkpoint",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "bf16"),
        default="auto",
        help="auto selects bf16 on CUDA and fp32 elsewhere",
    )
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
    parser.add_argument("--keep-last-checkpoints", type=int, default=3)

    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
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


def _resolve_precision(requested: PrecisionArgument, device: torch.device) -> Precision:
    if requested == "auto":
        return "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    if requested == "bf16" and device.type == "mps":
        raise ValueError("bf16 precision is not supported by this trainer on MPS")
    if requested == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 precision was requested but is not supported by this CUDA device")
    return requested


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


def _manifest_sha256(manifest_path: Path) -> str:
    with manifest_path.open("rb") as manifest_file:
        return hashlib.file_digest(manifest_file, "sha256").hexdigest()


def main(argv: list[str] | None = None) -> int:
    """Run one finite V1 experiment."""
    arguments = _build_parser().parse_args(argv)
    if arguments.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if arguments.checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    if arguments.keep_last_checkpoints <= 0:
        raise ValueError("keep_last_checkpoints must be positive")

    device = _resolve_device(arguments.device)
    precision = _resolve_precision(cast(PrecisionArgument, arguments.precision), device)
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
        precision=precision,
    )
    run_config = TrainingRunConfig(
        manifest_sha256=_manifest_sha256(arguments.manifest),
        batch_size=arguments.batch_size,
        optimizer_name="AdamW",
        optimizer_betas=ADAMW_BETAS,
        optimizer_weight_decay=arguments.weight_decay,
        optimizer_eps=ADAMW_EPS,
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
        betas=run_config.optimizer_betas,
        eps=run_config.optimizer_eps,
        weight_decay=arguments.weight_decay,
    )
    restored_checkpoint = None
    if arguments.resume is not None:
        if arguments.resume == "latest":
            restored_checkpoint = restore_latest_checkpoint(
                directory=arguments.checkpoint_dir,
                model=model,
                optimizer=optimizer,
                training_config=training_config,
                map_location="cpu",
                run_config=run_config,
            )
        else:
            restored_checkpoint = restore_checkpoint(
                path=arguments.resume,
                model=model,
                optimizer=optimizer,
                training_config=training_config,
                map_location="cpu",
                run_config=run_config,
            )
    start_step = 0 if restored_checkpoint is None else restored_checkpoint.step
    resume_rng_snapshot = None if restored_checkpoint is None else capture_torch_rng_state()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    run_name = arguments.wandb_name or (
        f"v1-{model_config.n_layers}l-{model_config.d_model}d-{arguments.seed}"
    )
    wandb_mode = cast(WandbMode, arguments.wandb_mode)
    stored_wandb_project = (
        None if restored_checkpoint is None else restored_checkpoint.wandb_project
    )
    stored_wandb_entity = None if restored_checkpoint is None else restored_checkpoint.wandb_entity
    if (
        stored_wandb_project is not None
        and arguments.wandb_project is not None
        and arguments.wandb_project != stored_wandb_project
    ):
        raise ValueError("wandb_project does not match the project stored in the checkpoint")
    if (
        stored_wandb_entity is not None
        and arguments.wandb_entity is not None
        and arguments.wandb_entity != stored_wandb_entity
    ):
        raise ValueError("wandb_entity does not match the entity stored in the checkpoint")
    wandb_project = stored_wandb_project or arguments.wandb_project or DEFAULT_WANDB_PROJECT
    wandb_entity = stored_wandb_entity or arguments.wandb_entity
    resume_wandb_run = (
        restored_checkpoint is not None and restored_checkpoint.wandb_run_id is not None
    )

    with wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        mode=wandb_mode,
        id=None if restored_checkpoint is None else restored_checkpoint.wandb_run_id,
        resume="must" if resume_wandb_run else None,
        config={
            "model": asdict(model_config),
            "training": asdict(training_config),
            "optimizer": {
                "name": "AdamW",
                "betas": list(run_config.optimizer_betas),
                "eps": run_config.optimizer_eps,
                "weight_decay": run_config.optimizer_weight_decay,
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
        wandb_run_id = (
            restored_checkpoint.wandb_run_id
            if restored_checkpoint is not None and restored_checkpoint.wandb_run_id is not None
            else run.id
        )
        checkpoint_wandb_project = (
            wandb_project
            if wandb_mode == "disabled"
            else getattr(run, "project", None) or wandb_project
        )
        checkpoint_wandb_entity = (
            wandb_entity
            if wandb_mode == "disabled"
            else getattr(run, "entity", None) or wandb_entity
        )
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
            precision=training_config.precision,
        )
        run.log(
            {
                "trainer/global_step": start_step,
                "validation/loss": initial_validation_loss,
                "validation/perplexity": perplexity_from_loss(initial_validation_loss),
            }
        )
        logger.log_samples(step=start_step)

        arguments.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        def on_step(metrics: StepMetrics) -> None:
            if metrics.step % arguments.checkpoint_interval == 0:
                save_checkpoint(
                    path=arguments.checkpoint_dir / f"step_{metrics.step:06d}.pt",
                    model=model,
                    optimizer=optimizer,
                    step=metrics.step,
                    training_config=training_config,
                    wandb_run_id=wandb_run_id,
                    wandb_project=checkpoint_wandb_project,
                    wandb_entity=checkpoint_wandb_entity,
                    keep_last_n=arguments.keep_last_checkpoints,
                    run_config=run_config,
                )
            logger(metrics)

        if resume_rng_snapshot is not None:
            restore_torch_rng_state(resume_rng_snapshot)

        history = train(
            model=model,
            optimizer=optimizer,
            train_batches=train_loader,
            config=training_config,
            device=device,
            validation_batches=validation_loader,
            start_step=start_step,
            on_step=on_step,
        )
        final_step = history[-1].step if history else start_step
        if final_step % arguments.checkpoint_interval != 0:
            save_checkpoint(
                path=arguments.checkpoint_dir / f"step_{final_step:06d}.pt",
                model=model,
                optimizer=optimizer,
                step=final_step,
                training_config=training_config,
                wandb_run_id=wandb_run_id,
                wandb_project=checkpoint_wandb_project,
                wandb_entity=checkpoint_wandb_entity,
                keep_last_n=arguments.keep_last_checkpoints,
                run_config=run_config,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

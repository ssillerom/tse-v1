"""Architecture contract for the initial package scaffold."""

from importlib import import_module

import pytest

PUBLIC_MODULES = (
    "llmfs.config",
    "llmfs.data.tokenizer",
    "llmfs.data.prepare",
    "llmfs.data.manifest",
    "llmfs.data.dataset",
    "llmfs.data.data_loader",
    "llmfs.nn.embeddings",
    "llmfs.nn.rope",
    "llmfs.nn.rms_norm",
    "llmfs.nn.feed_forward",
    "llmfs.nn.lm_head",
    "llmfs.nn.attention.reference",
    "llmfs.nn.attention.sdpa",
    "llmfs.nn.attention.attention",
    "llmfs.nn.transformer.config",
    "llmfs.nn.transformer.block",
    "llmfs.nn.transformer.model",
    "llmfs.optim.optimizer",
    "llmfs.optim.scheduler",
    "llmfs.train.pretrain_module",
    "llmfs.train.trainer",
    "llmfs.train.state",
    "llmfs.train.checkpoint",
    "llmfs.train.callbacks.console",
    "llmfs.train.callbacks.checkpoint",
    "llmfs.train.callbacks.wandb",
    "llmfs.eval.language_modeling",
    "llmfs.generate.sampling",
    "llmfs.generate.generate",
    "llmfs.distributed.ddp",
)


@pytest.mark.parametrize("module_name", PUBLIC_MODULES)
def test_public_module_is_importable(module_name: str) -> None:
    import_module(module_name)

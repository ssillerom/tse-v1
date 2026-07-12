"""Architecture contract for the initial package scaffold."""

from importlib import import_module
from pathlib import Path

import pytest

EXPECTED_IMPORTS = (
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
    "scripts.data.prepare_fineweb",
    "scripts.train.debug",
    "scripts.train.tiny",
    "scripts.train.small",
    "scripts.train.medium",
    "scripts.train.model_350m",
    "scripts.evaluate.checkpoint",
)

EXPECTED_PATHS = (
    "src/integration_tests/overfit_one_batch_test.py",
    "src/integration_tests/checkpoint_resume_test.py",
    "src/integration_tests/tiny_convergence_test.py",
    "tools/inspect_shard.py",
    "tools/inspect_checkpoint.py",
    "tools/benchmark_attention.py",
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("module_name", EXPECTED_IMPORTS)
def test_scaffold_module_is_importable(module_name: str) -> None:
    import_module(module_name)


@pytest.mark.parametrize("relative_path", EXPECTED_PATHS)
def test_scaffold_path_exists(relative_path: str) -> None:
    assert (REPO_ROOT / relative_path).is_file()

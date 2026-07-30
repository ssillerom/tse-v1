# AGENTS.md

## Mission

Build an educational decoder-only language model from first principles. Prefer correctness,
observable invariants, and small end-to-end proofs over premature scale or optimization.

## Current executable scope

The repository has a tested, single-device training slice:

```text
Hugging Face document stream
  -> token shards + manifest v3
  -> memory-mapped fixed-length sequences
  -> PyTorch DataLoader batches
  -> V1 GPT loss
  -> accumulated optimizer steps
  -> evaluation + resumable checkpoints
```

This slice proves correctness on small local runs; it is not yet a distributed or production
training system.

## Repository map

- `src/data/prepare_data.py`: streaming preparation, tokenization, partitioning, atomic publish.
- `src/data/manifest.py`: manifest v3 serialization, checksum validation, and shard confinement.
- `src/data/dataset.py`: memory-mapped causal sequences and global indexing.
- `src/model/`: V1 decoder-only Transformer components and assembled GPT.
- `src/training/trainer.py`: scheduled training, token-weighted accumulation, and evaluation.
- `src/training/scheduler.py`: linear warmup followed by cosine learning-rate decay.
- `src/training/checkpoint.py`: atomic model, optimizer, config, and RNG checkpoints.
- `src/scripts/inspect_slices.py`: human-facing inspection CLI.
- `src/test/data/`: public-interface tests for the data path.
- `src/test/model/`: component and assembled-model tests.
- `src/test/training/`: public-interface tests for training behavior.
- `src/integration_tests/`: small end-to-end learning and resumable-training proofs.
- `docs/adr/`: accepted architectural decisions.
- `CONTEXT.md`: canonical data-domain vocabulary.

## Data invariants

- A prepared dataset is complete only when `manifest.json` exists.
- Manifest format version 3 is the current writer contract; legacy v2 remains readable.
- Every document, including one truncated by the token budget, ends with the tokenizer's EOT
  token.
- Token IDs must fit the manifest storage dtype; the current dtype is `uint16`.
- Shard paths are relative to and confined within the manifest directory.
- Every v3 shard entry declares the SHA-256 of the exact published bytes, and readers reject a
  mismatch.
- Train/validation assignment is deterministic for the same text and split seed.
- A training sample reads `seq_len + 1` tokens and returns two shifted `torch.long` tensors.
- Samples never cross shard boundaries.

Changing any invariant above requires updating writer, reader, tests, README, and the relevant
ADR in the same change.

## Development commands

```bash
uv sync --group dev
uv run pytest -q
uv run ruff format --check src
uv run ruff check src
uv run mypy
```

## Change discipline

- Test behavior through public data, model, training, checkpoint, and CLI entry points.
- Use real temporary shard files in tests; do not mock NumPy memmap internals.
- Keep network access out of the default test suite.
- Exact data-order replay requires deterministic, reiterable training batches.
- Preserve unrelated working-tree changes and stage only files in the requested scope.
- Never commit prepared datasets, model checkpoints, secrets, tokens, or run logs.
- Prefer explicit validation errors over assertions in production paths.
- Keep code and identifiers in English. Documentation may be in Spanish.

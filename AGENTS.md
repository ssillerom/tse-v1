# AGENTS.md

## Mission

Build an educational decoder-only language model from first principles. Prefer correctness,
observable invariants, and small end-to-end proofs over premature scale or optimization.

## Current executable scope

The data path is the only complete vertical slice:

```text
Hugging Face document stream
  -> token shards + manifest v2
  -> memory-mapped fixed-length sequences
  -> PyTorch DataLoader batches
```

Do not claim that model training is available until a tested model and trainer exist.

## Repository map

- `src/data/prepare_data.py`: streaming preparation, tokenization, partitioning, atomic publish.
- `src/data/dataset.py`: manifest validation and memory-mapped causal sequences.
- `src/scripts/inspect_slices.py`: human-facing inspection CLI.
- `src/test/data/`: public-interface tests for the data path.
- `docs/adr/`: accepted architectural decisions.
- `CONTEXT.md`: canonical data-domain vocabulary.

## Data invariants

- A prepared dataset is complete only when `manifest.json` exists.
- Manifest format version 2 is the current reader/writer contract.
- Every document ends with the tokenizer's EOT token.
- Token IDs must fit the manifest storage dtype; the current dtype is `uint16`.
- Shard paths are relative to and confined within the manifest directory.
- Train/validation assignment is deterministic for the same text and split seed.
- A training sample reads `seq_len + 1` tokens and returns two shifted `torch.long` tensors.
- Samples never cross shard boundaries.

Changing any invariant above requires updating writer, reader, tests, README, and the relevant
ADR in the same change.

## Development commands

```bash
uv sync --group dev
uv run pytest -q src/test/data
uv run ruff format --check src/data src/scripts src/test/data
uv run ruff check src/data src/scripts src/test/data
uv run mypy
```

## Change discipline

- Test behavior through `prepare_streaming_dataset`, `PretrainingDataset`, and CLI entry points.
- Use real temporary shard files in tests; do not mock NumPy memmap internals.
- Keep network access out of the default test suite.
- Preserve unrelated working-tree changes and stage only files in the requested scope.
- Never commit prepared datasets, model checkpoints, secrets, tokens, or run logs.
- Prefer explicit validation errors over assertions in production paths.
- Keep code and identifiers in English. Documentation may be in Spanish.

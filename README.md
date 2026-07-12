# LLM from scratch

An educational decoder-only transformer implemented and trained from first principles in PyTorch.
The project follows an OLMo-core-inspired separation between neural-network components, data,
training orchestration, evaluation, generation, and distributed execution.

## Status

The repository is scaffolded. Mathematical and training components are intentionally left for the
learner to implement in the order described by
[`lessons/0001-mapa-de-implementacion.html`](lessons/0001-mapa-de-implementacion.html).

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
uv run mypy
```

Large datasets, checkpoints, run logs, and W&B state are local artifacts and are not committed.

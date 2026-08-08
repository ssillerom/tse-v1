# V1 English 12B training run

This is the operational recipe for the educational 315.8M V1. It deliberately keeps the
training system single-device and uses staged proofs before renting the final H100 SXM.

## Fixed model and token budget

- 315,753,472 parameters: `d_model=1024`, 24 layers, 16 query heads and 4 key/value heads
  (GQA, four Q heads per K/V head).
- Context length: 2,048.
- GPT-2 `tiktoken` vocabulary padded to 50,304.
- Global batch: 256 sequences = 524,288 tokens per optimizer step.
- Final budget: at most 12B tokens.
- 22,888 complete optimizer steps consume 11,999,903,744 tokens. The remaining 96,256
  recipe tokens cannot form a complete optimizer step and are intentionally unused.
- AdamW: betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0.1` only on matrix
  parameters, gradient clipping at `1.0`.
- BF16 forward/backward with FP32 parameters and optimizer state.
- Incremental generation keeps only the four K/V heads per layer in its compact KV cache. When
  the cache reaches 2,048 tokens, generation rebuilds it from the most recent context window.
- WSD: 400 warmup steps, stable LR, then cosine decay beginning at step 20,599 and
  reaching zero on the final update.

The recipe is [`configs/pretrain_v1_english_12b.json`](../../configs/pretrain_v1_english_12b.json).
Every declared source supplies a fixed held-out validation split. The trainer reports each
source separately and a whole-recipe weighted aggregate.

## Data mix

The final 1.2B tokens continue farther into the same web manifest and switch to separate, more
heavily filtered mathematics and knowledge manifests. Code appears only in the stable phase:
preparing the same Stack stream twice would silently repeat its earliest repositories during
decay.

| Category | Stable corpus and tokens | Premium decay corpus and tokens |
|---|---:|---:|
| Educational web | FineWeb-Edu sample-10BT, 8.75B | next 1.05B from the same manifest |
| Code | Stack v3 permissive Python, 0.75B | — |
| Mathematics | Nemotron-CC-Math score 3, 0.40B | Nemotron-CC-Math 4plus, 0.10B |
| Knowledge | FineWiki English, 0.90B | Nemotron Pretraining Fact-Seeking, 0.05B |
| **Total** | **10.80B** | **1.20B** |

The JSON quotas differ from these rounded labels by at most 1,024 tokens so that every phase
and source quota is an exact number of 2,048-token sequences.

Before preparing the NVIDIA mathematics source, accept the data agreement on the
`nvidia/Nemotron-CC-Math-v1` Hub page, run `hf auth login`, and keep `--hf-token` on those
commands. FineWeb-Edu, FineWiki and the specialized v1.2 corpus are public. Every source
command below is pinned to its tested revision; keep the generated manifests because "latest"
is not a reproducible revision.

FineWeb-Edu is prepared to the 10B sample cap and assigns 1% to validation, leaving expected
headroom above its 9.799999488B-token training quota. The other sources prepare slightly more
than their quotas and assign 2% to validation. The commands below create 100M-token shards
and use the same GPT-2 encoding. New preparations publish manifest v3 with a SHA-256 for every
shard; recipe validation rejects corruption before a paid training run.

```bash
uv run prepare-data prepare \
  --dataset-name HuggingFaceFW/fineweb-edu \
  --name sample-10BT \
  --revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 \
  --split train \
  --text-field text \
  --output-dir data/pretrain-v1/fineweb-edu-sample-10bt \
  --num-tokens 10000000000 \
  --shard-size 100000000 \
  --workers 2 \
  --source-workers 2 \
  --validation-ratio 0.01 \
  --split-seed 42 \
  --encoding gpt2

uv run prepare-data prepare \
  --dataset-name HuggingFaceCode/stack-v3-train \
  --revision 2b4797afd5677e32630c2247a6a8092e1a5afa03 \
  --split train \
  --records-field files \
  --record-filter language=Python,license_type=permissive \
  --text-field content \
  --output-dir data/pretrain-v1/stack-v3-python \
  --num-tokens 770000000 \
  --shard-size 100000000 \
  --min-chars 64 \
  --workers 2 \
  --source-workers 2 \
  --source-reader duckdb \
  --validation-ratio 0.02 \
  --split-seed 42 \
  --encoding gpt2

uv run prepare-data prepare \
  --dataset-name nvidia/Nemotron-CC-Math-v1 \
  --name 3 \
  --revision 397a2502f2028c659ba411a6c4935b464a7f03aa \
  --split train \
  --text-field text \
  --hf-token \
  --output-dir data/pretrain-v1/nemotron-math-3 \
  --num-tokens 415000000 \
  --shard-size 100000000 \
  --workers 2 \
  --source-workers 2 \
  --validation-ratio 0.02 \
  --split-seed 42 \
  --encoding gpt2

uv run prepare-data prepare \
  --dataset-name HuggingFaceFW/finewiki \
  --name en \
  --revision 8bd13e72e6a002407649b3e898535f42ceb1aeb9 \
  --split train \
  --text-field text \
  --output-dir data/pretrain-v1/finewiki-en \
  --num-tokens 930000000 \
  --shard-size 100000000 \
  --workers 2 \
  --source-workers 2 \
  --validation-ratio 0.02 \
  --split-seed 42 \
  --encoding gpt2

uv run prepare-data prepare \
  --dataset-name nvidia/Nemotron-CC-Math-v1 \
  --name 4plus \
  --revision 397a2502f2028c659ba411a6c4935b464a7f03aa \
  --split train \
  --text-field text \
  --hf-token \
  --output-dir data/pretrain-v1/nemotron-math-4plus \
  --num-tokens 110000000 \
  --shard-size 100000000 \
  --workers 2 \
  --source-workers 2 \
  --validation-ratio 0.02 \
  --split-seed 42 \
  --encoding gpt2

uv run prepare-data prepare \
  --dataset-name nvidia/Nemotron-Pretraining-Specialized-v1.2 \
  --name Nemotron-Pretraining-Fact-Seeking \
  --revision 807afc1fa65c441d46ebc7d9b95295a35499a527 \
  --split train \
  --text-field text \
  --output-dir data/pretrain-v1/nemotron-fact-seeking \
  --num-tokens 60000000 \
  --shard-size 100000000 \
  --workers 2 \
  --source-workers 2 \
  --validation-ratio 0.02 \
  --split-seed 42 \
  --encoding gpt2
```

Stack v3 stores one repository per row and the source files in `files[].content`; it has no
per-language configuration. The nested-record options above flatten those files and apply
both filters. Its DuckDB reader downloads at most `source_workers` Parquet files ahead,
processes them in filename order, and removes each temporary file after use. This avoids
materializing the oversized nested Arrow arrays in the original Parquet row groups. It
deliberately keeps only Python files labelled `permissive`, although the dataset license and
each original source license still need to be respected. The FineWeb-Edu
sample is prepared once: the deterministic sampler counts prior uses of a source, so its
decay slice starts after the 8.75B stable-token quota instead of replaying the beginning.
Nemotron Math publishes configs `3`, `4plus`, and `4plus_MIND`; its card defines “3plus” as
the union of `3` and `4plus`, not as a loadable config. This recipe keeps them disjoint: score
3 supplies the broad stable slice, and scores 4–5 are reserved for decay. The small
Fact-Seeking slice is synthetic question-answer text, so it is capped at 50M tokens to avoid
turning the base model into an instruction model. Nemotron SFT Math is intentionally excluded
because supervised reasoning trajectories belong after base pretraining.

After preparation, inspect all six manifests. Each `train.tokens` count must exceed its
recipe quota, and the tokenizer metadata must be identical:

```bash
uv run inspect-data-slices \
  --manifest data/pretrain-v1/fineweb-edu-sample-10bt/manifest.json \
  --split train \
  --seq-len 2048 \
  --num-examples 3
```

## Gate 1: local correctness

Do not rent a GPU until the repository checks pass:

```bash
uv sync --group dev
uv run pytest -q
uv run ruff format --check src
uv run ruff check src
uv run mypy
```

The important behavioural proofs are:

- no future-token leakage in attention;
- manual attention agrees with PyTorch SDPA;
- the assembled GPT has the expected shapes, gradients and initial loss;
- a tiny corpus can be overfit;
- the full document-to-checkpoint path learns;
- interrupted training reproduces uninterrupted training;
- the mixed recipe consumes the requested sources and restores its exact data position.

## Gate 2: A100 full-model rehearsal

Use an A100 for the first paid run. This is not a tiny-model test: it uses the final 315.8M
architecture, context, precision and global token batch. Start with micro-batch 4 and
accumulation 64. If it is comfortably below VRAM, use 8 × 32 instead.

```bash
uv run train-v1 \
  --recipe configs/pretrain_v1_english_12b.json \
  --device cuda \
  --precision bf16 \
  --d-model 1024 \
  --n-layers 24 \
  --n-heads 16 \
  --n-kv-heads 4 \
  --seq-len 2048 \
  --batch-size 4 \
  --grad-accum-steps 64 \
  --max-steps 500 \
  --stop-after-step 250 \
  --warmup-steps 50 \
  --lr-schedule wsd \
  --max-learning-rate 0.0006 \
  --eval-interval 100 \
  --eval-batches 20 \
  --sample-interval 100 \
  --checkpoint-interval 250 \
  --keep-last-checkpoints 3 \
  --num-workers 4 \
  --pin-memory \
  --checkpoint-dir checkpoints/v1-a100-rehearsal \
  --wandb-project llm-from-scratch \
  --wandb-name v1-a100-rehearsal
```

The omitted WSD boundary becomes `max_steps`, so this short rehearsal warms up and then keeps
the LR stable rather than wasting most of the pilot on decay. It processes approximately
262M tokens.

The gate passes only if:

- initial loss is close to `ln(50,304) = 10.83`;
- training and validation loss trend down;
- each source-specific validation loss remains finite and interpretable;
- loss, gradient norm and LR stay finite;
- gradient norm does not remain pinned at the clipping threshold;
- throughput stabilizes after startup;
- sample generations become less random, even if they are not useful yet;
- a checkpoint can be resumed and continues the same W&B run and data order.

The command intentionally exits at step 250. Run it again after removing
`--stop-after-step 250` and adding `--resume latest`; it must finish at step 500. Do not
change model, optimizer, batch, schedule, recipe or seed.

### Torch compilation

The architecture is a plausible `torch.compile` candidate: training shapes are fixed, the
attention path uses SDPA, and RoPE avoids complex tensors. Compilation is nevertheless a
hardware- and shape-specific optimization, not part of model correctness.

Every Make training target enables compilation by default through
`TORCH_COMPILE_ARGS="--compile --compile-mode default"`. To compare eager and compiled runs,
use fresh checkpoint directories:

```bash
make a100-rehearsal-start \
  A100_RUN_NAME=v1-a100-eager \
  TORCH_COMPILE_ARGS=

make a100-rehearsal-start \
  A100_RUN_NAME=v1-a100-compile
```

Ignore the compiled run's first steps when comparing throughput because they include JIT
compilation. Compare stable `performance/tokens_per_second`, peak VRAM, validation curves,
and failures or recompilations. Enable compilation for the final run only if it gives a
material repeatable improvement (roughly 10% or more is a useful threshold) without changing
the expected loss trajectory. Try `reduce-overhead` only after `default`; do not start with
`max-autotune`, whose cold-start tuning cost can be large.

Compilation mode is stored in the checkpoint run contract. A compiled run must be resumed
with the identical `TORCH_COMPILE_ARGS`; eager and compiled checkpoints are intentionally not
interchangeable continuations.

## Gate 3: learning-rate sweep

Run three independent 572-step A100 pilots. Each consumes 299,892,736 tokens, just below the
300M cap. Use identical seed, data order and arguments, changing only:

```text
run 1: --max-learning-rate 0.0003
run 2: --max-learning-rate 0.0006
run 3: --max-learning-rate 0.001
```

Use `--max-steps 572`, `--warmup-steps 50`, separate checkpoint directories and separate W&B
names. Choose the highest LR that remains stable and gives the best held-out loss, not the
lowest training loss. If `1e-3` spikes or clips continually, discard it. If `3e-4` and `6e-4`
are effectively tied, prefer `3e-4`.

## Gate 4: H100 SXM final run

Use the winning LR below. Omitting `--max-steps` makes the CLI derive 22,888 complete updates
from the recipe. Omitting `--decay-start-step` makes WSD infer step 20,599 from the boundary
between the stable and decay phases.

```bash
uv run train-v1 \
  --recipe configs/pretrain_v1_english_12b.json \
  --device cuda \
  --precision bf16 \
  --d-model 1024 \
  --n-layers 24 \
  --n-heads 16 \
  --n-kv-heads 4 \
  --seq-len 2048 \
  --batch-size 8 \
  --grad-accum-steps 32 \
  --warmup-steps 400 \
  --lr-schedule wsd \
  --max-learning-rate 0.0006 \
  --eval-interval 500 \
  --eval-batches 50 \
  --sample-interval 500 \
  --checkpoint-interval 1000 \
  --keep-last-checkpoints 3 \
  --num-workers 4 \
  --pin-memory \
  --checkpoint-dir checkpoints/v1-english-12b \
  --wandb-project llm-from-scratch \
  --wandb-name v1-english-12b
```

Watch the first 100 steps live before leaving the job unattended. At step 1,000, verify the
saved checkpoint on the same H100 with `--resume latest`; continuing the same run is preferable
to starting over. If micro-batch 8 does not fit, use 4 × 64. This preserves the global token
batch, number of optimizer steps and schedule. The final segment boundary is evaluated even though
22,888 is not divisible by 500; compare `best_validation.pt` with the final numbered checkpoint
before running the post-hoc suite.

After the final checkpoint, follow
[`docs/training/evaluation.md`](evaluation.md) for the fixed benchmark suite and the
300M-token FineWeb-only mixture baseline.

John Enev used the same staged principle: cheap GPU for data preparation, an A100 smoke test
of roughly 500 steps before each real run, and only then the H100. His published V1 smoke
script additionally checks initial loss near `log(vocab)`, decreasing loss, gradients on all
parameters, checkpoint save/load, evaluation and a 50-step single-batch overfit. This
repository covers those checks as automated tests and adds exact interrupted-run and
multi-source-position proofs.

At 140k tokens/s, John's reported H100 SXM throughput, 12B tokens take about 23.8 raw
training hours. Evaluation, generation, checkpointing and startup add overhead. Confirm the
actual price and measured throughput during the first 100 steps before assuming the run fits
the budget.

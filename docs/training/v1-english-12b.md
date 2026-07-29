# V1 English 12B training run

This is the operational recipe for the educational 353M V1. It deliberately keeps the
training system single-device and uses staged proofs before renting the final H100 SXM.

## Fixed model and token budget

- 353M parameters: `d_model=1024`, 24 layers, 16 MHA heads.
- Context length: 1,024.
- GPT-2 `tiktoken` vocabulary padded to 50,304.
- Global batch: 512 sequences = 524,288 tokens per optimizer step.
- Final budget: at most 12B tokens.
- 22,888 complete optimizer steps consume 11,999,903,744 tokens. The remaining 96,256
  recipe tokens cannot form a complete optimizer step and are intentionally unused.
- AdamW: betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0.1` only on matrix
  parameters, gradient clipping at `1.0`.
- BF16 forward/backward with FP32 parameters and optimizer state.
- WSD: 400 warmup steps, stable LR, then cosine decay beginning at step 20,599 and
  reaching zero on the final update.

The recipe is [`configs/pretrain_v1_english_12b.json`](../../configs/pretrain_v1_english_12b.json).
Its first source, FineWeb-Edu, also supplies the held-out validation sequences.

## Data mix

The last 1.2B tokens do not merely reweight the original streams. They come from separate,
more heavily filtered manifests:

| Category | Stable corpus and tokens | Premium decay corpus and tokens |
|---|---:|---:|
| Educational web | FineWeb-Edu, 9.60B | FineWeb-Edu Dedup, 0.90B |
| Code | GitHub Code, 0.60B | filtered CodeParrot Python, 0.15B |
| Mathematics | FineMath 3+, 0.40B | FineMath 4+, 0.10B |
| Knowledge | English Wikipedia, 0.20B | Cosmopedia v2, 0.05B |
| **Total** | **10.80B** | **1.20B** |

The JSON quotas differ from these rounded labels by at most 640 tokens so that every source
quota is an exact number of 1,024-token sequences.

Prepare slightly more than each training quota because 2% is assigned deterministically to
validation. The commands below create 100M-token shards and use the same GPT-2 encoding:

```bash
uv run prepare-data prepare \
  --dataset-name HuggingFaceFW/fineweb-edu \
  --name sample-100BT \
  --split train \
  --text-field text \
  --output-dir data/pretrain-v1/fineweb-edu \
  --num-tokens 9800000000 \
  --shard-size 100000000 \
  --validation-ratio 0.02 \
  --split-seed 42 \
  --encoding gpt2

uv run prepare-data prepare \
  --dataset-name codeparrot/github-code \
  --split train \
  --text-field code \
  --output-dir data/pretrain-v1/code \
  --num-tokens 620000000 \
  --shard-size 100000000 \
  --validation-ratio 0.02 \
  --split-seed 42 \
  --encoding gpt2

uv run prepare-data prepare \
  --dataset-name HuggingFaceTB/finemath \
  --name finemath-3plus \
  --split train \
  --text-field text \
  --output-dir data/pretrain-v1/finemath \
  --num-tokens 415000000 \
  --shard-size 100000000 \
  --validation-ratio 0.02 \
  --split-seed 42 \
  --encoding gpt2

uv run prepare-data prepare \
  --dataset-name wikimedia/wikipedia \
  --name 20231101.en \
  --split train \
  --text-field text \
  --output-dir data/pretrain-v1/wikipedia-en \
  --num-tokens 210000000 \
  --shard-size 100000000 \
  --validation-ratio 0.02 \
  --split-seed 42 \
  --encoding gpt2

uv run prepare-data prepare \
  --dataset-name HuggingFaceTB/smollm-corpus \
  --name fineweb-edu-dedup \
  --split train \
  --text-field text \
  --output-dir data/pretrain-v1/fineweb-edu-premium \
  --num-tokens 930000000 \
  --shard-size 100000000 \
  --validation-ratio 0.02 \
  --split-seed 42 \
  --encoding gpt2

uv run prepare-data prepare \
  --dataset-name codeparrot/codeparrot-train-more-filtering \
  --split train \
  --text-field content \
  --output-dir data/pretrain-v1/code-premium \
  --num-tokens 160000000 \
  --shard-size 100000000 \
  --validation-ratio 0.02 \
  --split-seed 42 \
  --encoding gpt2

uv run prepare-data prepare \
  --dataset-name HuggingFaceTB/finemath \
  --name finemath-4plus \
  --split train \
  --text-field text \
  --output-dir data/pretrain-v1/finemath-premium \
  --num-tokens 110000000 \
  --shard-size 100000000 \
  --validation-ratio 0.02 \
  --split-seed 42 \
  --encoding gpt2

uv run prepare-data prepare \
  --dataset-name HuggingFaceTB/smollm-corpus \
  --name cosmopedia-v2 \
  --split train \
  --text-field text \
  --output-dir data/pretrain-v1/knowledge-premium \
  --num-tokens 60000000 \
  --shard-size 100000000 \
  --validation-ratio 0.02 \
  --split-seed 42 \
  --encoding gpt2
```

`HuggingFaceTB/stack-edu` would be preferable to the stable code fallback, but its Hub rows
contain Software Heritage blob IDs rather than source text. The generic streaming preparer
cannot materialize those blobs yet. The premium code stream is nevertheless separate and
more strongly filtered than the stable stream. Review every source license before publishing
a trained model.

After preparation, inspect all eight manifests. Each `train.tokens` count must exceed its
recipe quota, and the tokenizer metadata must be identical:

```bash
uv run inspect-data-slices \
  --manifest data/pretrain-v1/fineweb-edu/manifest.json \
  --split train \
  --seq-len 1024 \
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

Use an A100 for the first paid run. This is not a tiny-model test: it uses the final 353M
architecture, context, precision and global batch. Start with micro-batch 8 and accumulation
64. If it is comfortably below VRAM, use 16 × 32 instead.

```bash
uv run train-v1 \
  --recipe configs/pretrain_v1_english_12b.json \
  --device cuda \
  --precision bf16 \
  --d-model 1024 \
  --n-layers 24 \
  --n-heads 16 \
  --seq-len 1024 \
  --batch-size 8 \
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
- loss, gradient norm and LR stay finite;
- gradient norm does not remain pinned at the clipping threshold;
- throughput stabilizes after startup;
- sample generations become less random, even if they are not useful yet;
- a checkpoint can be resumed and continues the same W&B run and data order.

The command intentionally exits at step 250. Run it again after removing
`--stop-after-step 250` and adding `--resume latest`; it must finish at step 500. Do not
change model, optimizer, batch, schedule, recipe or seed.

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
  --seq-len 1024 \
  --batch-size 16 \
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
to starting over. If micro-batch 16 does not fit, use 8 × 64. This preserves the global batch,
number of optimizer steps and schedule.

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

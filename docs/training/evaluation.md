# Evaluation protocol

The project uses three separate evaluation layers. They answer different questions and must
not be collapsed into one number.

## 1. Validation during pretraining

Every source in a multi-source recipe has its own fixed `validation` split. At step zero and
every `--eval-interval`, the trainer reports:

- `validation/<source>/loss`;
- `validation/<source>/perplexity`;
- `validation/<source>/target_tokens`.

`validation/loss` is the weighted mean of the source losses using the whole-recipe token
proportions. `validation/perplexity` is the exponential of that aggregate loss. The aggregate
is convenient for checkpoint selection, but source metrics are the diagnostic evidence: a
lower global loss must not hide regression on mathematics, code, or knowledge.

`--eval-batches` is applied independently to every source. Keep the manifests, sequence
length, tokenizer, batch limit, and recipe unchanged when comparing runs.

## 2. Controlled data-mixture baseline

The 300M-token recipes provide a cheap controlled comparison:

- `configs/pretrain_v1_english_300m_mixture.json`;
- `configs/pretrain_v1_english_300m_fineweb_baseline.json`.

Both contain exactly 292,864 sequences, or 299,892,736 tokens at context length 1,024. Train
them with the same architecture, seed, optimizer, LR, batch, precision, and hardware. The
only intended difference is the data distribution. Use independent checkpoint directories
and W&B names.

Start from the same argument template for both runs and change only `--recipe`,
`--checkpoint-dir`, and `--wandb-name`:

```bash
uv run train-v1 \
  --recipe configs/pretrain_v1_english_300m_mixture.json \
  --device cuda \
  --precision bf16 \
  --d-model 1024 \
  --n-layers 24 \
  --n-heads 16 \
  --seq-len 1024 \
  --batch-size 8 \
  --grad-accum-steps 64 \
  --warmup-steps 50 \
  --lr-schedule wsd \
  --max-learning-rate 0.0006 \
  --eval-interval 100 \
  --eval-batches 20 \
  --checkpoint-interval 100 \
  --checkpoint-dir checkpoints/v1-300m-mixture \
  --wandb-name v1-300m-mixture
```

Repeat it with `configs/pretrain_v1_english_300m_fineweb_baseline.json`,
`checkpoints/v1-300m-fineweb`, and `v1-300m-fineweb`. Both recipes derive exactly 572
optimizer steps with this global batch.

The FineWeb-only run cannot measure validation loss on code or mathematics because it does
not declare those sources. Compare its FineWeb validation and downstream benchmarks with the
mixed run; do not invent missing per-domain values.

One seed is sufficient for a pipeline proof, not for a causal claim. Before publishing a
small difference, repeat the comparison with additional seeds and report dispersion.

## 3. Post-hoc benchmarks

Install the optional harness only in the evaluation environment:

```bash
uv sync --locked --group dev --extra eval
```

The default command runs zero-shot LAMBADA, HellaSwag, ARC Easy, ARC Challenge, WinoGrande,
PIQA, and WikiText:

```bash
uv run eval-checkpoint \
  --checkpoint checkpoints/v1-english-12b/step_022888.pt \
  --device cuda \
  --precision bf16 \
  --batch-size 16 \
  --output results/v1-english-12b-base.json
```

The adapter drives the raw PyTorch `GPT` directly. It batches likelihood and generation
requests, tokenizes context and continuation as one BPE sequence, masks the padded model
vocabulary, uses rolling windows for perplexity, and loads architecture settings from the
checkpoint.

GSM8K is a separate five-shot run:

```bash
uv run eval-checkpoint \
  --checkpoint checkpoints/v1-english-12b/step_022888.pt \
  --tasks gsm8k \
  --num-fewshot 5 \
  --device cuda \
  --precision bf16 \
  --batch-size 16 \
  --output results/v1-english-12b-gsm8k-5shot.json
```

A base model may score poorly on GSM8K because it has not learned instruction or answer
formatting. Treat it as a recorded baseline for future SFT rather than a release gate.

Each result JSON records the checkpoint path and SHA-256, model config, tokenizer, task list,
shot count, batch size, precision, Torch and harness versions, Git commit and dirty state,
plus the harness task versions, standard errors, and sample counts.

For external calibration, evaluate GPT-2 and Pythia-410M through the same harness version,
task versions, shots, and metric selection. Those public models are reference points, not
controlled training baselines: their token budgets and recipes differ.

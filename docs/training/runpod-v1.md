# Running the V1 recipe on RunPod

This guide executes the pinned 12B-token, 2,048-context recipe through explicit Make targets. Data,
checkpoints, caches, and W&B logs stay under `/workspace`; nothing important should be kept
only on the container disk.

## 1. Create persistent storage and secrets

Create a 100 GB network volume in a data center that currently offers both an A100 and an
H100 SXM. Network volumes are tied to their data center, so checking only the first GPU can
strand the data before the final run. The prepared `uint16` shards occupy about 25 GB; the
remaining space is for Hugging Face cache files, the Python environment, three resumable
checkpoints, and W&B logs. Use 150 GB if you want comfortable cache and checkpoint headroom.

A network volume is preferable to a Pod volume for this workflow because it survives Pod
termination and can be attached first to a CPU Pod for data preparation and later to the
GPU Pod. Mount it at `/workspace`.

Create these RunPod secrets:

- `huggingface_token`: a Hugging Face read token with access to
  `nvidia/Nemotron-CC-Math-v1` after accepting its data agreement.
- `wandb_api_key`: the API key for the W&B account that will own the run.

Map them in the Pod template without putting their values in the repository:

```text
HF_TOKEN={{ RUNPOD_SECRET_huggingface_token }}
WANDB_API_KEY={{ RUNPOD_SECRET_wandb_api_key }}
HF_HOME=/workspace/.cache/huggingface
UV_CACHE_DIR=/workspace/.cache/uv
WANDB_DIR=/workspace/wandb
TRAINING_COMMIT=<commit SHA pushed from the local repository>
```

For a non-interactive bootstrap, also map `GH_TOKEN` (or `GITHUB_TOKEN`) to a short-lived
GitHub token with the minimum repository permissions needed for `git fetch`. Otherwise,
`make setup` starts the interactive GitHub CLI device login over SSH. Do not commit or print
any of these secret values.

Use an immutable tag of the official RunPod PyTorch image rather than `latest`, and record
that tag with the experiment. Select a machine whose driver supports CUDA 13.0, which is
required by the locked PyTorch build. Keep at least 30 GB of container disk, attach the
network volume at `/workspace`, and expose TCP port 22 for SSH. Do not place the repository,
data, or checkpoints outside `/workspace`.

## 2. Bootstrap the repository

Connect over SSH, then run:

```bash
cd /workspace
git clone https://github.com/ssillerom/llm-from-scratch.git
cd llm-from-scratch
git fetch origin
git checkout --detach "$TRAINING_COMMIT"
test "$(git rev-parse HEAD)" = "$TRAINING_COMMIT"

make setup
export PATH="${UV_INSTALL_DIR:-$HOME/.local/bin}:$PATH"
make local-gate
```

`make setup` installs the pinned `uv` release into `$HOME/.local/bin` when needed, installs the
GitHub CLI with Homebrew or apt when needed, authenticates it, installs the Python version
pinned by the project, creates `.venv`, and installs the locked dependencies. `make local-gate`
must pass before paying for GPU time. The `export` makes direct `uv` commands in the current
shell available; Make targets set the path themselves. The bootstrap is idempotent, so rerunning
it after reconnecting to a new Pod is safe.

## 3. Prepare the 12B-token data

Data preparation is CPU-bound. On the four-vCPU Pod, the Make targets use two source workers
to overlap Hugging Face downloads while two tokenizer threads encode ordered document
batches. The Stack v3 target uses DuckDB: it prefetches at most two Parquet files to temporary
container storage, processes them in filename order, and deletes each copy after use. Keep at
least 5 GB of free temporary disk for this bounded prefetch. Prefer this CPU Pod attached to
the network volume rather than leaving an H100 idle. Run the preparation inside `tmux`:

```bash
tmux new -s prepare-v1
cd /workspace/llm-from-scratch
make credentials-check
make prepare-data
make validate-recipe
```

Override concurrency when the CPU allocation differs, for example
`make prepare-data PREPARE_SOURCE_WORKERS=4 PREPARE_WORKERS=4`. Keep both values fixed for
reproducibility. Tokenizer threads preserve their input order, while source processes consume
different Hugging Face shards concurrently. For DuckDB sources the same setting controls
ordered Parquet prefetch and DuckDB threads. `source_workers` is recorded in each new manifest
because changing it can change which documents reach the global token budget.

Detach from tmux with `Ctrl-b d` and reconnect with:

```bash
tmux attach -t prepare-v1
```

Every preparation target is intentionally fail-safe: `manifest.json` marks a completed
source, so rerunning `make prepare-data` skips it and continues with the first incomplete
source. The preparation code also refuses to overwrite existing published shards. If a
completed source really must be replaced, remove only that source directory after checking
its exact path, then rerun its individual target.

`make validate-recipe` verifies every manifest v3 SHA-256. Run it after preparation and again
after transferring or restoring the network volume; the sequential reads are intentional and
should happen before renting the final H100.

The regression proof for the original oversized Stack v3 row group is opt-in so the default
test suite never downloads data. To repeat it, download the pinned shard and point the test at
the resulting local file:

```bash
hf download HuggingFaceCode/stack-v3-train \
  data/part-00066-50e95205-4aec-46cc-bde2-02f09aa216ac-c000.snappy.parquet \
  --repo-type dataset \
  --revision 2b4797afd5677e32630c2247a6a8092e1a5afa03 \
  --local-dir /tmp/stack-v3-regression

STACK_V3_SHARD_66_PATH=/tmp/stack-v3-regression/data/part-00066-50e95205-4aec-46cc-bde2-02f09aa216ac-c000.snappy.parquet \
  uv run pytest -q src/integration_tests/test_stack_v3_duckdb.py
```

Before terminating the CPU Pod, verify that the six manifests exist and preserve the network
volume:

```bash
find data/pretrain-v1 -name manifest.json -print
make validate-recipe
du -sh data/pretrain-v1 .cache
```

## 4. Rehearse on an A100

Attach the same network volume to a one-GPU A100 Pod, check out the recorded commit, and
verify the runtime:

```bash
cd /workspace/llm-from-scratch
git fetch origin
git checkout --detach "$TRAINING_COMMIT"
test "$(git rev-parse HEAD)" = "$TRAINING_COMMIT"
make setup
make runpod-ready
```

The rehearsal deliberately stops after step 250 and must resume to step 500:

```bash
tmux new -s a100-rehearsal
make a100-rehearsal-start
make a100-rehearsal-resume
```

Do not continue unless:

- initial validation loss is close to `10.83`;
- train and validation loss decrease and remain finite;
- gradient norm is finite and is not permanently clipped at `1.0`;
- throughput stabilizes;
- W&B samples become less random;
- the resumed process continues the original W&B run at step 250.

If micro-batch 4 runs comfortably, the H100 target later uses micro-batch 8. If either runs
out of memory, preserve the 524,288-token global batch by overriding both values together.
You may also test `8 × 32` on the A100 by passing the same overrides to both commands:

```bash
make a100-rehearsal-start \
  A100_BATCH_SIZE=8 \
  A100_GRAD_ACCUM_STEPS=32 \
  A100_RUN_NAME=v1-a100-rehearsal-8x32

make a100-rehearsal-resume \
  A100_BATCH_SIZE=8 \
  A100_GRAD_ACCUM_STEPS=32 \
  A100_RUN_NAME=v1-a100-rehearsal-8x32
```

## 5. Choose the maximum learning rate

Run the three candidates independently:

```bash
make sweep-3e4
make sweep-6e4
make sweep-1e3
```

Each target trains for 572 optimizer steps, about 300M tokens, with the same seed and data
order. Compare held-out validation loss at the same step. Choose the highest stable learning
rate with the best validation loss. Discard a candidate if loss spikes, values become
non-finite, or gradient norm remains at the clipping threshold. If `3e-4` and `6e-4` are
effectively tied, use `3e-4`.

The sweep targets use separate checkpoint directories and W&B names. They do not run
automatically as a group so that a paid experiment is always an explicit command. If one is
interrupted after a checkpoint, resume its original W&B run and exact data position with the
matching target:

```bash
make sweep-3e4-resume
make sweep-6e4-resume
make sweep-1e3-resume
```

## 6. Run the final H100 SXM training

Attach the network volume to a one-GPU H100 SXM Pod. Verify the selected machine and all
artifacts before starting:

```bash
cd /workspace/llm-from-scratch
git fetch origin
git checkout --detach "$TRAINING_COMMIT"
test "$(git rev-parse HEAD)" = "$TRAINING_COMMIT"
make setup
make runpod-ready
```

Start the final run with the LR selected by the sweep:

```bash
tmux new -s v1-12b
make h100-train MAX_LEARNING_RATE=0.0006
```

If the eager/compiled rehearsal documented in `v1-english-12b.md` selected compilation, add
`TORCH_COMPILE_ARGS="--compile --compile-mode default"` to both `h100-train` and every
`h100-resume` invocation. Do not enable or disable compilation in the middle of a run.

The command derives 22,888 complete optimizer steps from the recipe and begins WSD decay at
step 20,599. Watch the first 100 steps in W&B and verify the first checkpoint at step 1,000.
To prove recovery, stop the process cleanly after that checkpoint and run:

```bash
make h100-resume MAX_LEARNING_RATE=0.0006
```

The resume command must use the same LR and batch configuration. It restores model,
optimizer, RNG, W&B run identity, and exact recipe position.

If micro-batch 8 does not fit, start with `4 × 64` and repeat those overrides on every
resume:

```bash
make h100-train \
  MAX_LEARNING_RATE=0.0006 \
  H100_BATCH_SIZE=4 \
  H100_GRAD_ACCUM_STEPS=64
```

When the run finishes, copy the final checkpoint to durable object storage before terminating
the Pod. A RunPod network volume is persistent independently of a Pod, but it is not a
substitute for a long-term backup.

Then install the optional evaluation extra and run the protocol in
[`evaluation.md`](evaluation.md). Benchmark results are small JSON artifacts; preserve them
beside the checkpoint and record their checkpoint SHA-256.

## Commands at a glance

```bash
make help
make local-gate
make prepare-data
make validate-recipe
make runpod-ready
make a100-rehearsal-start
make a100-rehearsal-resume
make sweep-3e4
make sweep-3e4-resume
make sweep-6e4
make sweep-6e4-resume
make sweep-1e3
make sweep-1e3-resume
make h100-train MAX_LEARNING_RATE=0.0006
make h100-resume MAX_LEARNING_RATE=0.0006
```

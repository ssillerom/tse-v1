SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help
.NOTPARALLEL:

UV ?= uv
UV_VERSION ?= 0.11.32
UV_INSTALL_DIR ?= $(HOME)/.local/bin
UV_INSTALL_URL ?= https://astral.sh/uv/$(UV_VERSION)/install.sh
UV_CMD := PATH="$(UV_INSTALL_DIR):$$PATH" $(UV)
GH ?= gh
GH_HOST ?= github.com
GH_AUTH ?= auto
RECIPE ?= configs/pretrain_v1_english_12b.json
WANDB_PROJECT ?= llm-from-scratch
WANDB_ENTITY ?=
WANDB_MODE ?= online
NUM_WORKERS ?= 4
PREPARE_WORKERS ?= 2
PREPARE_SOURCE_WORKERS ?= 2
MAX_LEARNING_RATE ?= 0.0006
TORCH_COMPILE_ARGS ?= --compile --compile-mode default
SEQ_LEN := 2048
A100_BATCH_SIZE ?= 4
A100_GRAD_ACCUM_STEPS ?= 64
A100_RUN_NAME ?= v1-a100-rehearsal
A100_CHECKPOINT_DIR ?= checkpoints/$(A100_RUN_NAME)
H100_BATCH_SIZE ?= 8
H100_GRAD_ACCUM_STEPS ?= 32
H100_RUN_NAME ?= v1-english-12b
H100_CHECKPOINT_DIR ?= checkpoints/$(H100_RUN_NAME)

SHARD_SIZE := 100000000
SPLIT_SEED := 42

FINEWEB_MANIFEST := data/pretrain-v1/fineweb-edu-sample-10bt/manifest.json
STACK_MANIFEST := data/pretrain-v1/stack-v3-python/manifest.json
MATH_3_MANIFEST := data/pretrain-v1/nemotron-math-3/manifest.json
FINEWIKI_MANIFEST := data/pretrain-v1/finewiki-en/manifest.json
MATH_4PLUS_MANIFEST := data/pretrain-v1/nemotron-math-4plus/manifest.json
FACT_MANIFEST := data/pretrain-v1/nemotron-fact-seeking/manifest.json
DATA_MANIFESTS := \
	$(FINEWEB_MANIFEST) \
	$(STACK_MANIFEST) \
	$(MATH_3_MANIFEST) \
	$(FINEWIKI_MANIFEST) \
	$(MATH_4PLUS_MANIFEST) \
	$(FACT_MANIFEST)

WANDB_ENTITY_ARG := $(if $(strip $(WANDB_ENTITY)),--wandb-entity $(WANDB_ENTITY),)

MODEL_ARGS := \
	--device cuda \
	--precision bf16 \
	--vocab-size 50304 \
	--d-model 1024 \
	--n-layers 24 \
	--n-heads 16 \
	--n-kv-heads 4 \
	--seq-len $(SEQ_LEN) \
	$(TORCH_COMPILE_ARGS)

RUNTIME_ARGS := \
	--recipe $(RECIPE) \
	--lr-schedule wsd \
	--num-workers $(NUM_WORKERS) \
	--pin-memory \
	--keep-last-checkpoints 3 \
	--wandb-project $(WANDB_PROJECT) \
	$(WANDB_ENTITY_ARG) \
	--wandb-mode $(WANDB_MODE)

A100_BATCH_ARGS = \
	--batch-size $(A100_BATCH_SIZE) \
	--grad-accum-steps $(A100_GRAD_ACCUM_STEPS)

H100_BATCH_ARGS = \
	--batch-size $(H100_BATCH_SIZE) \
	--grad-accum-steps $(H100_GRAD_ACCUM_STEPS)

A100_REHEARSAL_ARGS = \
	$(A100_BATCH_ARGS) \
	--max-steps 500 \
	--warmup-steps 50 \
	--max-learning-rate 0.0006 \
	--eval-interval 100 \
	--eval-batches 20 \
	--sample-interval 100 \
	--checkpoint-interval 250 \
	--checkpoint-dir $(A100_CHECKPOINT_DIR) \
	--wandb-name $(A100_RUN_NAME)

H100_TRAIN_ARGS = \
	$(H100_BATCH_ARGS) \
	--warmup-steps 400 \
	--max-learning-rate $(MAX_LEARNING_RATE) \
	--eval-interval 500 \
	--eval-batches 50 \
	--sample-interval 500 \
	--checkpoint-interval 1000 \
	--checkpoint-dir $(H100_CHECKPOINT_DIR) \
	--wandb-name $(H100_RUN_NAME)

define ensure_new_checkpoint_dir
@if [ -e "$(1)" ]; then \
	echo "Refusing to start a new run: $(1) already exists. Resume it or choose a new directory."; \
	exit 1; \
fi
endef

.PHONY: \
	help bootstrap-tools gh-auth setup local-gate gpu-check credentials-check validate-recipe runpod-ready \
	prepare-data prepare-fineweb prepare-stack prepare-math-3 prepare-finewiki \
	prepare-math-4plus prepare-fact \
	a100-rehearsal-start a100-rehearsal-resume \
	sweep-3e4 sweep-3e4-resume sweep-6e4 sweep-6e4-resume sweep-1e3 sweep-1e3-resume \
	h100-train h100-resume

help: ## Show the available workflow targets.
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_.-]+:.*## / {printf "  %-26s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

bootstrap-tools: ## Install the pinned uv and GitHub CLI when missing.
	@set -eu; \
	export UV_INSTALL_DIR="$(UV_INSTALL_DIR)"; \
	export PATH="$(UV_INSTALL_DIR):$$PATH"; \
	uv_version=""; \
	if command -v "$(UV)" >/dev/null 2>&1; then \
		uv_version="$$($(UV) --version | awk 'NR == 1 {print $$2}')"; \
	fi; \
	if [ "$$uv_version" != "$(UV_VERSION)" ]; then \
		command -v curl >/dev/null 2>&1 || { echo "curl is required to install uv" >&2; exit 1; }; \
		echo "Installing uv $(UV_VERSION) into $(UV_INSTALL_DIR)"; \
		curl -LsSf "$(UV_INSTALL_URL)" | sh; \
	fi; \
	command -v "$(UV)" >/dev/null 2>&1 || { echo "uv was not found after installation; add $(UV_INSTALL_DIR) to PATH" >&2; exit 1; }; \
	installed_uv_version="$$($(UV) --version | awk 'NR == 1 {print $$2}')"; \
	[ "$$installed_uv_version" = "$(UV_VERSION)" ] || { echo "Expected uv $(UV_VERSION), found uv $$installed_uv_version" >&2; exit 1; }; \
	if ! command -v "$(GH)" >/dev/null 2>&1; then \
		if command -v brew >/dev/null 2>&1; then \
			echo "Installing GitHub CLI with Homebrew"; \
			brew install gh; \
		elif command -v apt-get >/dev/null 2>&1; then \
			echo "Installing GitHub CLI with apt"; \
			if command -v sudo >/dev/null 2>&1; then \
				sudo apt-get update; \
				sudo apt-get install -y gh; \
			else \
				apt-get update; \
				apt-get install -y gh; \
			fi; \
		else \
			echo "Cannot install gh automatically: use Homebrew or apt-get, then rerun make setup" >&2; \
			exit 1; \
		fi; \
	fi; \
	command -v "$(GH)" >/dev/null 2>&1 || { echo "gh was not found after installation" >&2; exit 1; }

gh-auth: bootstrap-tools ## Authenticate the GitHub CLI, or skip it explicitly.
	@set -eu; \
	export PATH="$(UV_INSTALL_DIR):$$PATH"; \
	case "$(GH_AUTH)" in \
		skip) \
			echo "Skipping GitHub CLI authentication (GH_AUTH=skip)."; \
			;; \
		auto) \
			if "$(GH)" auth status --hostname "$(GH_HOST)" >/dev/null 2>&1; then \
				echo "GitHub CLI is already authenticated."; \
			elif [ -n "$${GH_TOKEN:-}" ] || [ -n "$${GITHUB_TOKEN:-}" ]; then \
				echo "Using GH_TOKEN/GITHUB_TOKEN for headless GitHub CLI authentication."; \
			elif [ -t 0 ] && [ -t 1 ]; then \
				echo "Starting interactive GitHub CLI login..."; \
				"$(GH)" auth login --hostname "$(GH_HOST)" --git-protocol https --web; \
			else \
				echo "GitHub CLI is not authenticated and setup has no interactive terminal." >&2; \
				echo "Set GH_TOKEN or GITHUB_TOKEN, run 'gh auth login', or use 'make setup GH_AUTH=skip'." >&2; \
				exit 1; \
			fi; \
			"$(GH)" auth setup-git --hostname "$(GH_HOST)"; \
			"$(GH)" auth status --hostname "$(GH_HOST)"; \
			;; \
		*) \
			echo "GH_AUTH must be 'auto' or 'skip', got '$(GH_AUTH)'" >&2; \
			exit 2; \
			;; \
	esac

setup: gh-auth ## Bootstrap uv/gh, authenticate GitHub, and install locked dependencies.
	@set -eu; \
	export PATH="$(UV_INSTALL_DIR):$$PATH"; \
	$(UV_CMD) python install; \
	$(UV_CMD) sync --locked --group dev

local-gate: ## Run all free correctness checks before renting a GPU.
	$(UV_CMD) run pytest -q
	$(UV_CMD) run ruff format --check src
	$(UV_CMD) run ruff check src
	$(UV_CMD) run mypy

gpu-check: ## Verify that PyTorch sees one BF16-capable CUDA GPU.
	@nvidia-smi
	@$(UV_CMD) run python -c 'import sys, torch; cuda=torch.cuda.is_available(); sys.exit("CUDA is unavailable") if not cuda else None; bf16=torch.cuda.is_bf16_supported(); sys.exit("GPU does not support BF16") if not bf16 else None; print(f"torch={torch.__version__} gpu={torch.cuda.get_device_name(0)} bf16=True")'

credentials-check: ## Verify the RunPod secret environment variables without printing them.
	@$(UV_CMD) run python -c 'import os, sys; required=("HF_TOKEN", "WANDB_API_KEY"); missing=[name for name in required if not os.environ.get(name)]; sys.exit(f"Missing secret environment variables: {missing}") if missing else print("HF_TOKEN and WANDB_API_KEY are configured")'

prepare-data: $(DATA_MANIFESTS) ## Prepare all six pinned datasets, skipping completed manifests.

prepare-fineweb: $(FINEWEB_MANIFEST) ## Prepare the FineWeb-Edu sample-10BT source.

$(FINEWEB_MANIFEST):
	$(UV_CMD) run prepare-data prepare \
		--dataset-name HuggingFaceFW/fineweb-edu \
		--name sample-10BT \
		--revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 \
		--split train \
		--text-field text \
		--output-dir data/pretrain-v1/fineweb-edu-sample-10bt \
		--num-tokens 10000000000 \
		--shard-size $(SHARD_SIZE) \
		--workers $(PREPARE_WORKERS) \
		--source-workers $(PREPARE_SOURCE_WORKERS) \
		--validation-ratio 0.01 \
		--split-seed $(SPLIT_SEED) \
		--encoding gpt2

prepare-stack: $(STACK_MANIFEST) ## Prepare permissively licensed Python from Stack v3.

$(STACK_MANIFEST):
	$(UV_CMD) run prepare-data prepare \
		--dataset-name HuggingFaceCode/stack-v3-train \
		--revision 80a7f793eb87d89a7835c3585090427039da0ad3 \
		--split train \
		--records-field files \
		--record-filter language=Python,license_type=permissive \
		--text-field content \
		--output-dir data/pretrain-v1/stack-v3-python \
		--num-tokens 770000000 \
		--shard-size $(SHARD_SIZE) \
		--min-chars 64 \
		--workers $(PREPARE_WORKERS) \
		--source-workers $(PREPARE_SOURCE_WORKERS) \
		--source-reader duckdb \
		--validation-ratio 0.02 \
		--split-seed $(SPLIT_SEED) \
		--encoding gpt2

prepare-math-3: $(MATH_3_MANIFEST) ## Prepare the broad Nemotron mathematics source.

$(MATH_3_MANIFEST):
	$(UV_CMD) run prepare-data prepare \
		--dataset-name nvidia/Nemotron-CC-Math-v1 \
		--name 3 \
		--revision 397a2502f2028c659ba411a6c4935b464a7f03aa \
		--split train \
		--text-field text \
		--hf-token \
		--output-dir data/pretrain-v1/nemotron-math-3 \
		--num-tokens 415000000 \
		--shard-size $(SHARD_SIZE) \
		--workers $(PREPARE_WORKERS) \
		--source-workers $(PREPARE_SOURCE_WORKERS) \
		--validation-ratio 0.02 \
		--split-seed $(SPLIT_SEED) \
		--encoding gpt2

prepare-finewiki: $(FINEWIKI_MANIFEST) ## Prepare the English FineWiki knowledge source.

$(FINEWIKI_MANIFEST):
	$(UV_CMD) run prepare-data prepare \
		--dataset-name HuggingFaceFW/finewiki \
		--name en \
		--revision 8bd13e72e6a002407649b3e898535f42ceb1aeb9 \
		--split train \
		--text-field text \
		--output-dir data/pretrain-v1/finewiki-en \
		--num-tokens 930000000 \
		--shard-size $(SHARD_SIZE) \
		--workers $(PREPARE_WORKERS) \
		--source-workers $(PREPARE_SOURCE_WORKERS) \
		--validation-ratio 0.02 \
		--split-seed $(SPLIT_SEED) \
		--encoding gpt2

prepare-math-4plus: $(MATH_4PLUS_MANIFEST) ## Prepare premium Nemotron mathematics for WSD decay.

$(MATH_4PLUS_MANIFEST):
	$(UV_CMD) run prepare-data prepare \
		--dataset-name nvidia/Nemotron-CC-Math-v1 \
		--name 4plus \
		--revision 397a2502f2028c659ba411a6c4935b464a7f03aa \
		--split train \
		--text-field text \
		--hf-token \
		--output-dir data/pretrain-v1/nemotron-math-4plus \
		--num-tokens 110000000 \
		--shard-size $(SHARD_SIZE) \
		--workers $(PREPARE_WORKERS) \
		--source-workers $(PREPARE_SOURCE_WORKERS) \
		--validation-ratio 0.02 \
		--split-seed $(SPLIT_SEED) \
		--encoding gpt2

prepare-fact: $(FACT_MANIFEST) ## Prepare premium fact-seeking data for WSD decay.

$(FACT_MANIFEST):
	$(UV_CMD) run prepare-data prepare \
		--dataset-name nvidia/Nemotron-Pretraining-Specialized-v1.2 \
		--name Nemotron-Pretraining-Fact-Seeking \
		--revision 807afc1fa65c441d46ebc7d9b95295a35499a527 \
		--split train \
		--text-field text \
		--output-dir data/pretrain-v1/nemotron-fact-seeking \
		--num-tokens 60000000 \
		--shard-size $(SHARD_SIZE) \
		--workers $(PREPARE_WORKERS) \
		--source-workers $(PREPARE_SOURCE_WORKERS) \
		--validation-ratio 0.02 \
		--split-seed $(SPLIT_SEED) \
		--encoding gpt2

validate-recipe: ## Validate manifests, tokenizer agreement, quotas, and source capacity.
	@$(UV_CMD) run python -c 'from src.data.dataset import PretrainingDataset; from src.data.mixture import DeterministicMixtureSampler, MixtureDataset; from src.data.recipe import load_training_recipe; recipe=load_training_recipe("$(RECIPE)"); datasets={source.name: PretrainingDataset(source.manifest_path, "train", $(SEQ_LEN)) for source in recipe.sources}; mixture=MixtureDataset(datasets); sampler=DeterministicMixtureSampler(mixture, recipe.phases, $(SEQ_LEN), 42); print(f"recipe={recipe.name} seq_len=$(SEQ_LEN) tokens={recipe.total_tokens:,} sequences={len(sampler):,} sources={len(recipe.sources)}")'

runpod-ready: local-gate gpu-check credentials-check validate-recipe ## Run every check required immediately before a paid training job.

a100-rehearsal-start: runpod-ready ## Run steps 1-250 of the resumability rehearsal.
	$(call ensure_new_checkpoint_dir,$(A100_CHECKPOINT_DIR))
	$(UV_CMD) run train-v1 $(MODEL_ARGS) $(RUNTIME_ARGS) $(A100_REHEARSAL_ARGS) \
		--stop-after-step 250

a100-rehearsal-resume: runpod-ready ## Resume the rehearsal from step 250 through step 500.
	$(UV_CMD) run train-v1 $(MODEL_ARGS) $(RUNTIME_ARGS) $(A100_REHEARSAL_ARGS) --resume latest

define run_sweep
	$(call ensure_new_checkpoint_dir,checkpoints/v1-lr-$(2))
	$(UV_CMD) run train-v1 $(MODEL_ARGS) $(RUNTIME_ARGS) $(A100_BATCH_ARGS) \
		--max-steps 572 \
		--warmup-steps 50 \
		--max-learning-rate $(1) \
		--eval-interval 100 \
		--eval-batches 20 \
		--sample-interval 100 \
		--checkpoint-interval 100 \
		--checkpoint-dir checkpoints/v1-lr-$(2) \
		--wandb-name v1-lr-$(2)
endef

define resume_sweep
	$(UV_CMD) run train-v1 $(MODEL_ARGS) $(RUNTIME_ARGS) $(A100_BATCH_ARGS) \
		--max-steps 572 \
		--warmup-steps 50 \
		--max-learning-rate $(1) \
		--eval-interval 100 \
		--eval-batches 20 \
		--sample-interval 100 \
		--checkpoint-interval 100 \
		--checkpoint-dir checkpoints/v1-lr-$(2) \
		--wandb-name v1-lr-$(2) \
		--resume latest
endef

sweep-3e4: runpod-ready ## Run the 300M-token LR candidate at 3e-4.
	$(call run_sweep,0.0003,3e4)

sweep-3e4-resume: runpod-ready ## Resume the 3e-4 LR candidate.
	$(call resume_sweep,0.0003,3e4)

sweep-6e4: runpod-ready ## Run the 300M-token LR candidate at 6e-4.
	$(call run_sweep,0.0006,6e4)

sweep-6e4-resume: runpod-ready ## Resume the 6e-4 LR candidate.
	$(call resume_sweep,0.0006,6e4)

sweep-1e3: runpod-ready ## Run the 300M-token LR candidate at 1e-3.
	$(call run_sweep,0.001,1e3)

sweep-1e3-resume: runpod-ready ## Resume the 1e-3 LR candidate.
	$(call resume_sweep,0.001,1e3)

h100-train: runpod-ready ## Start the final 12B-token H100 run with MAX_LEARNING_RATE.
	$(call ensure_new_checkpoint_dir,$(H100_CHECKPOINT_DIR))
	$(UV_CMD) run train-v1 $(MODEL_ARGS) $(RUNTIME_ARGS) $(H100_TRAIN_ARGS)

h100-resume: runpod-ready ## Resume the final H100 run from its newest valid checkpoint.
	$(UV_CMD) run train-v1 $(MODEL_ARGS) $(RUNTIME_ARGS) $(H100_TRAIN_ARGS) --resume latest

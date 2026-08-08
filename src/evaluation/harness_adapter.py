"""Thin lm-evaluation-harness adapter for the repository's raw PyTorch GPT."""

from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Protocol, cast

import tiktoken
import torch

from src.model.config import ModelConfig
from src.model.gpt import GPT, GPTKVCache
from src.training.checkpoint import SUPPORTED_CHECKPOINT_FORMAT_VERSIONS

try:
    from lm_eval.api.model import LM as _HarnessLM
except ModuleNotFoundError:
    LM_EVAL_AVAILABLE = False

    class _HarnessLM:  # type: ignore[no-redef]
        """Import-time fallback so core adapter behavior remains locally testable."""

        def __init__(self) -> None:
            self._device: torch.device | None = None

else:
    LM_EVAL_AVAILABLE = True


class HarnessRequest(Protocol):
    """The request surface supplied by lm-evaluation-harness."""

    @property
    def args(self) -> tuple[Any, ...]: ...


@dataclass(frozen=True)
class EvaluationCheckpoint:
    """A model reconstructed exclusively from a validated checkpoint payload."""

    path: Path
    step: int
    model: GPT


def load_evaluation_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> EvaluationCheckpoint:
    """Load model weights and architecture without requiring training arguments."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    format_version = payload.get("format_version")
    if not isinstance(format_version, int) or isinstance(format_version, bool):
        raise ValueError("checkpoint format_version must be an integer")
    if format_version not in SUPPORTED_CHECKPOINT_FORMAT_VERSIONS:
        raise ValueError(f"Unsupported checkpoint format_version: {format_version!r}")
    model_config_payload = payload.get("model_config")
    model_state = payload.get("model_state")
    step = payload.get("step")
    if not isinstance(model_config_payload, dict):
        raise ValueError("checkpoint contains an invalid model_config")
    if not isinstance(model_state, dict):
        raise ValueError("checkpoint contains an invalid model_state")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("checkpoint contains an invalid step")

    try:
        model_config = ModelConfig(**model_config_payload)
    except TypeError as error:
        raise ValueError(f"checkpoint model_config is incompatible: {error}") from error
    model = GPT(model_config)
    model.load_state_dict(model_state, strict=True)
    model.to(device)
    model.eval()
    return EvaluationCheckpoint(
        path=checkpoint_path.resolve(),
        step=step,
        model=model,
    )


@dataclass(frozen=True)
class _ScoringWindow:
    input_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    score_from: int


@dataclass
class _GenerationState:
    context_ids: list[int]
    generated_ids: list[int]
    stop_sequences: tuple[str, ...]
    max_new_tokens: int
    result: str | None = None
    cache: GPTKVCache | None = None


class GPT2HarnessAdapter(_HarnessLM):  # type: ignore[misc]
    """Make a custom GPT checkpoint implement the harness's three request types."""

    def __init__(
        self,
        model: GPT,
        encoding: tiktoken.Encoding,
        device: torch.device | str,
        batch_size: int = 8,
        precision: str = "fp32",
    ) -> None:
        super().__init__()
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if precision not in ("fp32", "bf16"):
            raise ValueError("precision must be one of ('fp32', 'bf16')")
        resolved_device = torch.device(device)
        if precision == "bf16" and resolved_device.type not in ("cpu", "cuda"):
            raise ValueError("bf16 evaluation is supported only on CPU or CUDA")
        if (
            precision == "bf16"
            and resolved_device.type == "cuda"
            and not torch.cuda.is_bf16_supported()
        ):
            raise ValueError("bf16 is not supported by the requested CUDA device")
        if model.config.vocab_size < encoding.n_vocab:
            raise ValueError(
                "model vocabulary is smaller than the tokenizer vocabulary: "
                f"{model.config.vocab_size} < {encoding.n_vocab}"
            )

        self.model = model
        self.encoding = encoding
        self._device = resolved_device
        self._batch_size = batch_size
        self.precision = precision
        self.model.to(self._device)
        self.model.eval()

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def tokenizer_name(self) -> str:
        return self.encoding.name

    def encode_pair(self, context: str, continuation: str) -> tuple[list[int], list[int]]:
        """Tokenize a causal context/continuation pair using the harness convention."""
        if not isinstance(context, str) or not isinstance(continuation, str):
            raise TypeError("context and continuation must be strings")

        trailing_spaces = len(context) - len(context.rstrip())
        if trailing_spaces:
            continuation = context[-trailing_spaces:] + continuation
            context = context[:-trailing_spaces]

        if context == "":
            continuation_ids = self.encoding.encode(continuation, disallowed_special=())
            if not continuation_ids:
                raise ValueError("continuation must encode to at least one token")
            if continuation_ids[0] == self.encoding.eot_token:
                return continuation_ids[:1], continuation_ids[1:]
            return [self.encoding.eot_token], continuation_ids

        context_ids = self.encoding.encode(context, disallowed_special=())
        combined_ids = self.encoding.encode(context + continuation, disallowed_special=())
        continuation_ids = combined_ids[len(context_ids) :]
        if not continuation_ids:
            raise ValueError("continuation must encode to at least one token")
        return context_ids, continuation_ids

    def loglikelihood(
        self,
        requests: list[HarnessRequest],
        disable_tqdm: bool = False,
    ) -> list[tuple[float, bool]]:
        """Score fixed continuations, preserving request order."""
        del disable_tqdm
        windows: list[_ScoringWindow] = []
        for request in requests:
            context, continuation = cast(tuple[str, str], request.args)
            context_ids, continuation_ids = self.encode_pair(context, continuation)
            if len(continuation_ids) > self.model.config.max_seq_len:
                raise ValueError(
                    "continuation exceeds the model context length: "
                    f"{len(continuation_ids)} > {self.model.config.max_seq_len}"
                )
            combined_ids = context_ids + continuation_ids
            input_end = len(combined_ids) - 1
            input_start = max(0, input_end - self.model.config.max_seq_len)
            score_from = len(context_ids) - (input_start + 1)
            if score_from < 0:
                raise RuntimeError("failed to preserve the complete continuation")
            windows.append(
                _ScoringWindow(
                    input_ids=tuple(combined_ids[input_start:input_end]),
                    target_ids=tuple(combined_ids[input_start + 1 :]),
                    score_from=score_from,
                )
            )
        return self._score_windows(windows)

    def loglikelihood_rolling(
        self,
        requests: list[HarnessRequest],
        disable_tqdm: bool = False,
    ) -> list[float]:
        """Score every document token exactly once using rolling context windows."""
        del disable_tqdm
        results: list[float] = []
        for request in requests:
            (text,) = cast(tuple[str], request.args)
            token_ids = self.encoding.encode(text, disallowed_special=())
            if not token_ids:
                results.append(0.0)
                continue

            combined_ids = [self.encoding.eot_token, *token_ids]
            windows: list[_ScoringWindow] = []
            target_start = 1
            while target_start < len(combined_ids):
                target_end = min(
                    target_start + self.model.config.max_seq_len,
                    len(combined_ids),
                )
                input_end = target_end - 1
                input_start = max(0, input_end - self.model.config.max_seq_len)
                windows.append(
                    _ScoringWindow(
                        input_ids=tuple(combined_ids[input_start:input_end]),
                        target_ids=tuple(combined_ids[input_start + 1 : target_end]),
                        score_from=target_start - (input_start + 1),
                    )
                )
                target_start = target_end
            results.append(sum(score for score, _ in self._score_windows(windows)))
        return results

    def generate_until(
        self,
        requests: list[HarnessRequest],
        disable_tqdm: bool = False,
    ) -> list[str]:
        """Generate greedy continuations with per-request K/V caches."""
        del disable_tqdm
        states = [self._generation_state(request) for request in requests]
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                while True:
                    active = [
                        index
                        for index, state in enumerate(states)
                        if state.result is None and len(state.generated_ids) < state.max_new_tokens
                    ]
                    if not active:
                        break
                    for offset in range(0, len(active), self.batch_size):
                        batch_indices = active[offset : offset + self.batch_size]
                        self._generate_one_token(states, batch_indices)
        finally:
            self.model.train(was_training)

        return [
            state.result if state.result is not None else self.encoding.decode(state.generated_ids)
            for state in states
        ]

    def _generation_state(self, request: HarnessRequest) -> _GenerationState:
        context, raw_kwargs = cast(tuple[str, dict[str, object]], request.args)
        unsupported = set(raw_kwargs) - {
            "until",
            "max_gen_toks",
            "do_sample",
            "temperature",
        }
        if unsupported:
            raise ValueError(f"unsupported generation arguments: {sorted(unsupported)}")
        if raw_kwargs.get("do_sample", False):
            raise ValueError("sampling is not supported; benchmark generation must be greedy")
        temperature = raw_kwargs.get("temperature", 0.0)
        if (
            not isinstance(temperature, Real)
            or isinstance(temperature, bool)
            or float(temperature) != 0.0
        ):
            raise ValueError("only temperature=0 generation is supported")

        max_new_tokens = raw_kwargs.get("max_gen_toks", 256)
        if (
            not isinstance(max_new_tokens, int)
            or isinstance(max_new_tokens, bool)
            or max_new_tokens <= 0
        ):
            raise ValueError("max_gen_toks must be a positive integer")
        raw_until = raw_kwargs.get("until", ())
        stop_sequences: tuple[str, ...]
        if isinstance(raw_until, str):
            stop_sequences = (raw_until,)
        elif isinstance(raw_until, Sequence) and all(isinstance(value, str) for value in raw_until):
            stop_sequences = tuple(cast(Sequence[str], raw_until))
        else:
            raise ValueError("until must be a string or a sequence of strings")
        if any(not value for value in stop_sequences):
            raise ValueError("stop sequences must be non-empty")

        context_ids = self.encoding.encode(context, disallowed_special=())
        if not context_ids:
            context_ids = [self.encoding.eot_token]
        return _GenerationState(
            context_ids=context_ids,
            generated_ids=[],
            stop_sequences=stop_sequences,
            max_new_tokens=max_new_tokens,
        )

    def _generate_one_token(
        self,
        states: list[_GenerationState],
        batch_indices: list[int],
    ) -> None:
        groups: dict[tuple[int | None, int], list[tuple[int, list[int]]]] = {}
        for state_index in batch_indices:
            state = states[state_index]
            all_ids = state.context_ids + state.generated_ids
            if state.cache is None or state.cache.seq_len == self.model.config.max_seq_len:
                model_ids = all_ids[-self.model.config.max_seq_len :]
                state.cache = None
            else:
                model_ids = all_ids[-1:]
            cache_length = None if state.cache is None else state.cache.seq_len
            groups.setdefault((cache_length, len(model_ids)), []).append((state_index, model_ids))

        for group in groups.values():
            input_ids = torch.tensor(
                [model_ids for _, model_ids in group],
                dtype=torch.long,
                device=self._device,
            )
            group_caches = [states[state_index].cache for state_index, _ in group]
            batched_cache = (
                None
                if group_caches[0] is None
                else GPTKVCache.batch(cast(list[GPTKVCache], group_caches))
            )
            with self._precision_context():
                logits, updated_cache = self.model.forward_with_cache(input_ids, batched_cache)
            split_caches = updated_cache.unbind()
            next_ids = torch.argmax(logits[:, -1, : self.encoding.n_vocab], dim=-1).tolist()

            for (state_index, _), request_cache, token_id in zip(
                group,
                split_caches,
                next_ids,
                strict=True,
            ):
                state = states[state_index]
                state.cache = request_cache
                if token_id == self.encoding.eot_token:
                    state.result = self.encoding.decode(state.generated_ids)
                    continue
                state.generated_ids.append(token_id)
                decoded = self.encoding.decode(state.generated_ids)
                stop_positions = [
                    position
                    for stop in state.stop_sequences
                    if (position := decoded.find(stop)) >= 0
                ]
                if stop_positions:
                    state.result = decoded[: min(stop_positions)]

    def _score_windows(
        self,
        windows: list[_ScoringWindow],
    ) -> list[tuple[float, bool]]:
        results: list[tuple[float, bool]] = []
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                for offset in range(0, len(windows), self.batch_size):
                    batch = windows[offset : offset + self.batch_size]
                    max_length = max(len(window.input_ids) for window in batch)
                    input_ids = torch.full(
                        (len(batch), max_length),
                        self.encoding.eot_token,
                        dtype=torch.long,
                        device=self._device,
                    )
                    for row, window in enumerate(batch):
                        input_ids[row, : len(window.input_ids)] = torch.tensor(
                            window.input_ids,
                            dtype=torch.long,
                            device=self._device,
                        )
                    with self._precision_context():
                        logits, _ = self.model(input_ids)
                    real_logits = logits[..., : self.encoding.n_vocab].float()
                    log_probabilities = torch.log_softmax(real_logits, dim=-1)
                    for row, window in enumerate(batch):
                        score_slice = slice(window.score_from, len(window.target_ids))
                        targets = torch.tensor(
                            window.target_ids[score_slice],
                            dtype=torch.long,
                            device=self._device,
                        )
                        token_log_probabilities = log_probabilities[
                            row,
                            score_slice,
                        ].gather(-1, targets.unsqueeze(-1))
                        greedy_ids = torch.argmax(
                            real_logits[row, score_slice],
                            dim=-1,
                        )
                        results.append(
                            (
                                float(token_log_probabilities.sum().item()),
                                bool(torch.equal(greedy_ids, targets)),
                            )
                        )
        finally:
            self.model.train(was_training)
        return results

    def _precision_context(self) -> AbstractContextManager[None]:
        if self.precision == "fp32":
            return nullcontext()
        return torch.autocast(device_type=self._device.type, dtype=torch.bfloat16)

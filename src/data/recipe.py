"""Versioned multi-source pretraining recipes."""

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from src.data.manifest import DataManifest, ManifestTokenizer, load_manifest

RECIPE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class RecipeSource:
    """One prepared dataset participating in a training recipe."""

    name: str
    manifest_path: Path

    @cached_property
    def manifest(self) -> DataManifest:
        """Keep one validated manifest for this recipe's setup lifetime."""
        return load_manifest(self.manifest_path)


@dataclass(frozen=True)
class RecipePhase:
    """One contiguous token budget and its per-source quotas."""

    name: str
    target_tokens: int
    source_tokens: tuple[tuple[str, int], ...]

    @cached_property
    def source_token_map(self) -> dict[str, int]:
        return dict(self.source_tokens)


@dataclass(frozen=True)
class TrainingRecipe:
    """Validated data plan for one finite pretraining run."""

    path: Path
    name: str
    sources: tuple[RecipeSource, ...]
    phases: tuple[RecipePhase, ...]
    tokenizer: ManifestTokenizer

    @property
    def total_tokens(self) -> int:
        return sum(phase.target_tokens for phase in self.phases)

    @cached_property
    def source_token_totals(self) -> dict[str, int]:
        """Return the exact token budget assigned to every source."""
        totals = {source.name: 0 for source in self.sources}
        for phase in self.phases:
            for source_name, token_count in phase.source_tokens:
                totals[source_name] += token_count
        return totals

    @cached_property
    def source_weights(self) -> dict[str, float]:
        """Return whole-recipe source proportions for aggregate validation."""
        return {
            source_name: token_count / self.total_tokens
            for source_name, token_count in self.source_token_totals.items()
        }


def _required_non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _required_positive_integer(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _parse_sources(raw_sources: object, recipe_path: Path) -> tuple[RecipeSource, ...]:
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("Recipe must contain a non-empty sources list")

    sources: list[RecipeSource] = []
    names: set[str] = set()
    for position, raw_source in enumerate(raw_sources):
        if not isinstance(raw_source, dict):
            raise ValueError(f"Recipe source at position {position} must be an object")
        name = _required_non_empty_string(raw_source.get("name"), "source name")
        if name in names:
            raise ValueError(f"Recipe source names must be unique, got {name!r}")
        names.add(name)
        manifest_value = _required_non_empty_string(
            raw_source.get("manifest"),
            f"manifest for source {name!r}",
        )
        manifest_path = Path(manifest_value)
        if not manifest_path.is_absolute():
            manifest_path = recipe_path.parent / manifest_path
        sources.append(
            RecipeSource(
                name=name,
                manifest_path=manifest_path.resolve(),
            )
        )
    return tuple(sources)


def _parse_phases(
    raw_phases: object,
    source_names: frozenset[str],
) -> tuple[RecipePhase, ...]:
    if not isinstance(raw_phases, list) or not raw_phases:
        raise ValueError("Recipe must contain a non-empty phases list")

    phases: list[RecipePhase] = []
    phase_names: set[str] = set()
    for position, raw_phase in enumerate(raw_phases):
        if not isinstance(raw_phase, dict):
            raise ValueError(f"Recipe phase at position {position} must be an object")
        name = _required_non_empty_string(raw_phase.get("name"), "phase name")
        if name in phase_names:
            raise ValueError(f"Recipe phase names must be unique, got {name!r}")
        phase_names.add(name)
        target_tokens = _required_positive_integer(
            raw_phase.get("tokens"),
            f"tokens for phase {name!r}",
        )
        raw_source_tokens = raw_phase.get("source_tokens")
        if not isinstance(raw_source_tokens, dict) or not raw_source_tokens:
            raise ValueError(f"Phase {name!r} must contain a non-empty source_tokens object")

        quotas: list[tuple[str, int]] = []
        for raw_source_name, raw_token_count in raw_source_tokens.items():
            source_name = _required_non_empty_string(raw_source_name, "phase source name")
            if source_name not in source_names:
                raise ValueError(f"Phase {name!r} refers to undeclared source {source_name!r}")
            token_count = _required_positive_integer(
                raw_token_count,
                f"token quota for source {source_name!r}",
            )
            quotas.append((source_name, token_count))
        if sum(token_count for _, token_count in quotas) != target_tokens:
            raise ValueError(f"Phase {name!r} source token quotas must sum to phase tokens")
        phases.append(
            RecipePhase(
                name=name,
                target_tokens=target_tokens,
                source_tokens=tuple(quotas),
            )
        )
    return tuple(phases)


def load_training_recipe(path: str | Path) -> TrainingRecipe:
    """Load a recipe and validate every referenced manifest and tokenizer."""
    recipe_path = Path(path)
    if not recipe_path.is_file():
        raise FileNotFoundError(f"Training recipe not found: {recipe_path}")
    try:
        payload = json.loads(recipe_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Failed to parse recipe JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("Training recipe JSON must be an object")
    if payload.get("format_version") != RECIPE_FORMAT_VERSION:
        raise ValueError(f"Unsupported recipe format_version: {payload.get('format_version')!r}")

    name = _required_non_empty_string(payload.get("name"), "recipe name")
    sources = _parse_sources(payload.get("sources"), recipe_path)
    source_names = frozenset(source.name for source in sources)
    phases = _parse_phases(payload.get("phases"), source_names)
    used_source_names = {
        source_name for phase in phases for source_name, _token_count in phase.source_tokens
    }
    unused_source_names = sorted(source_names - used_source_names)
    if unused_source_names:
        raise ValueError(
            "Recipe declared sources receive no tokens: " + ", ".join(unused_source_names)
        )

    tokenizer: ManifestTokenizer | None = None
    for source in sources:
        source_tokenizer = source.manifest.tokenizer
        if source_tokenizer is None:
            raise ValueError(f"Source {source.name!r} manifest must declare a tokenizer")
        if tokenizer is None:
            tokenizer = source_tokenizer
        elif source_tokenizer != tokenizer:
            raise ValueError("All recipe sources must use the same tokenizer")
    if tokenizer is None:
        raise ValueError("Training recipe must contain at least one source")

    return TrainingRecipe(
        path=recipe_path.resolve(),
        name=name,
        sources=sources,
        phases=phases,
        tokenizer=tokenizer,
    )

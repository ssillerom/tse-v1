import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from data.manifest import FORMAT_VERSION, STORAGE_DTYPE, write_manifest
from data.mixture import DeterministicMixtureSampler, MixtureDataset
from data.recipe import RECIPE_FORMAT_VERSION, TrainingRecipe, load_training_recipe


def _write_manifest(directory: Path, name: str, token_count: int = 128) -> Path:
    source_directory = directory / name
    source_directory.mkdir()
    train_path = source_directory / "train.bin"
    validation_path = source_directory / "validation.bin"
    np.arange(token_count, dtype=np.uint16).tofile(train_path)
    np.arange(32, dtype=np.uint16).tofile(validation_path)
    with train_path.open("rb") as train_file:
        train_sha256 = hashlib.file_digest(train_file, "sha256").hexdigest()
    with validation_path.open("rb") as validation_file:
        validation_sha256 = hashlib.file_digest(validation_file, "sha256").hexdigest()
    manifest_path = source_directory / "manifest.json"
    write_manifest(
        manifest_path,
        {
            "format_version": FORMAT_VERSION,
            "dataset": {"path": f"local/{name}"},
            "tokenizer": {"encoding": "gpt2", "eot_token": 50_256},
            "storage": {"dtype": STORAGE_DTYPE, "shard_size": token_count},
            "counts": {"tokens": token_count + 32, "shards": 2},
            "splits": {
                "train": {"tokens": token_count, "shards": 1},
                "validation": {"tokens": 32, "shards": 1},
            },
            "shards": [
                {
                    "file": train_path.name,
                    "split": "train",
                    "tokens": token_count,
                    "sha256": train_sha256,
                },
                {
                    "file": validation_path.name,
                    "split": "validation",
                    "tokens": 32,
                    "sha256": validation_sha256,
                },
            ],
        },
    )
    return manifest_path


def _load_shipped_recipe_with_local_manifests(
    tmp_path: Path,
    recipe_filename: str,
) -> TrainingRecipe:
    repository_root = Path(__file__).parents[3]
    source_recipe_path = repository_root / "configs" / recipe_filename
    payload = json.loads(source_recipe_path.read_text(encoding="utf-8"))
    local_recipe_path = tmp_path / "configs" / recipe_filename
    local_recipe_path.parent.mkdir(parents=True)

    for source in payload["sources"]:
        manifest_path = (local_recipe_path.parent / source["manifest"]).resolve()
        manifest_path.parent.parent.mkdir(parents=True, exist_ok=True)
        assert _write_manifest(manifest_path.parent.parent, manifest_path.parent.name) == (
            manifest_path
        )

    local_recipe_path.write_text(json.dumps(payload), encoding="utf-8")
    return load_training_recipe(local_recipe_path)


def _build_sampler(recipe: TrainingRecipe, seq_len: int) -> DeterministicMixtureSampler:
    source_lengths = {
        source.name: range(recipe.source_token_totals[source.name] // seq_len)
        for source in recipe.sources
    }
    return DeterministicMixtureSampler(
        MixtureDataset(source_lengths),
        recipe.phases,
        seq_len=seq_len,
        seed=42,
    )


def test_training_recipe_loads_relative_manifests_and_phase_quotas(tmp_path: Path) -> None:
    first_manifest = _write_manifest(tmp_path, "general")
    second_manifest = _write_manifest(tmp_path, "code")
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(
        json.dumps(
            {
                "format_version": RECIPE_FORMAT_VERSION,
                "name": "tiny-english",
                "sources": [
                    {"name": "general", "manifest": str(first_manifest.relative_to(tmp_path))},
                    {"name": "code", "manifest": str(second_manifest.relative_to(tmp_path))},
                ],
                "phases": [
                    {
                        "name": "stable",
                        "tokens": 96,
                        "source_tokens": {"general": 64, "code": 32},
                    },
                    {
                        "name": "decay",
                        "tokens": 32,
                        "source_tokens": {"general": 24, "code": 8},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    recipe = load_training_recipe(recipe_path)

    assert recipe.name == "tiny-english"
    assert recipe.total_tokens == 128
    assert recipe.tokenizer.encoding_name == "gpt2"
    assert recipe.tokenizer.eot_token_id == 50_256
    assert [source.name for source in recipe.sources] == ["general", "code"]
    assert [phase.name for phase in recipe.phases] == ["stable", "decay"]
    assert recipe.sources[0].manifest_path == first_manifest
    assert recipe.source_token_totals == {"general": 88, "code": 40}
    assert recipe.source_weights == {"general": 0.6875, "code": 0.3125}


def test_training_recipe_rejects_a_phase_whose_source_tokens_do_not_sum(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path, "general")
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(
        json.dumps(
            {
                "format_version": RECIPE_FORMAT_VERSION,
                "name": "invalid",
                "sources": [{"name": "general", "manifest": str(manifest_path)}],
                "phases": [
                    {
                        "name": "stable",
                        "tokens": 100,
                        "source_tokens": {"general": 99},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source token quotas must sum to phase tokens"):
        load_training_recipe(recipe_path)


def test_training_recipe_rejects_a_declared_source_that_receives_no_tokens(
    tmp_path: Path,
) -> None:
    general_manifest = _write_manifest(tmp_path, "general")
    unused_manifest = _write_manifest(tmp_path, "unused")
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(
        json.dumps(
            {
                "format_version": RECIPE_FORMAT_VERSION,
                "name": "invalid",
                "sources": [
                    {"name": "general", "manifest": str(general_manifest)},
                    {"name": "unused", "manifest": str(unused_manifest)},
                ],
                "phases": [
                    {
                        "name": "stable",
                        "tokens": 100,
                        "source_tokens": {"general": 100},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="declared sources receive no tokens: unused"):
        load_training_recipe(recipe_path)


def test_shipped_12b_recipe_has_quotas_aligned_to_2048_token_sequences(
    tmp_path: Path,
) -> None:
    recipe = _load_shipped_recipe_with_local_manifests(
        tmp_path,
        "pretrain_v1_english_12b.json",
    )
    stable_phase, decay_phase = recipe.phases
    sources_by_name = {source.name: source for source in recipe.sources}

    assert stable_phase.target_tokens == 10_800_001_024
    assert decay_phase.target_tokens == 1_199_998_976
    assert recipe.total_tokens == 12_000_000_000
    sampler = _build_sampler(recipe, seq_len=2_048)
    assert len(sampler) == 5_859_375
    assert set(stable_phase.source_token_map) & set(decay_phase.source_token_map) == {
        "fineweb_edu_sample_10bt"
    }
    assert stable_phase.source_token_map["fineweb_edu_sample_10bt"] == 8_750_000_128
    assert stable_phase.source_token_map["nemotron_math_3"] == 400_001_024
    assert decay_phase.source_token_map["fineweb_edu_sample_10bt"] == 1_049_999_360
    assert stable_phase.source_token_map["finewiki_en"] == 899_999_744
    assert "fineweb_edu_dedup" not in sources_by_name
    assert "nemotron_knowledge" not in sources_by_name


def test_shipped_300m_recipes_hold_compute_constant_and_change_only_the_mix(
    tmp_path: Path,
) -> None:
    mixture = _load_shipped_recipe_with_local_manifests(
        tmp_path / "mixture",
        "pretrain_v1_english_300m_mixture.json",
    )
    baseline = _load_shipped_recipe_with_local_manifests(
        tmp_path / "baseline",
        "pretrain_v1_english_300m_fineweb_baseline.json",
    )

    assert mixture.total_tokens == 299_892_736
    assert baseline.total_tokens == 299_892_736
    assert len(_build_sampler(mixture, seq_len=2_048)) == 146_432
    assert len(_build_sampler(baseline, seq_len=2_048)) == 146_432
    assert baseline.phases[0].source_token_map == {"fineweb_edu_sample_10bt": 299_892_736}

import json
from pathlib import Path

import numpy as np
import pytest

from data.manifest import FORMAT_VERSION, STORAGE_DTYPE, write_manifest
from data.recipe import RECIPE_FORMAT_VERSION, load_training_recipe


def _write_manifest(directory: Path, name: str, token_count: int = 128) -> Path:
    source_directory = directory / name
    source_directory.mkdir()
    train_path = source_directory / "train.bin"
    validation_path = source_directory / "validation.bin"
    np.arange(token_count, dtype=np.uint16).tofile(train_path)
    np.arange(32, dtype=np.uint16).tofile(validation_path)
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
                {"file": train_path.name, "split": "train", "tokens": token_count},
                {"file": validation_path.name, "split": "validation", "tokens": 32},
            ],
        },
    )
    return manifest_path


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


def test_shipped_12b_recipe_has_exact_sequences_and_only_reuses_web_source() -> None:
    repository_root = Path(__file__).parents[3]
    payload = json.loads(
        (repository_root / "configs" / "pretrain_v1_english_12b.json").read_text(encoding="utf-8")
    )
    stable_phase, decay_phase = payload["phases"]
    sources_by_name = {source["name"]: source for source in payload["sources"]}

    assert stable_phase["tokens"] == 10_800_000_000
    assert decay_phase["tokens"] == 1_200_000_000
    assert sum(phase["tokens"] for phase in payload["phases"]) == 12_000_000_000
    for phase in (stable_phase, decay_phase):
        assert sum(phase["source_tokens"].values()) == phase["tokens"]
        assert all(token_count % 1_024 == 0 for token_count in phase["source_tokens"].values())
    assert set(stable_phase["source_tokens"]) & set(decay_phase["source_tokens"]) == {
        "fineweb_edu_sample_10bt"
    }
    assert stable_phase["source_tokens"]["fineweb_edu_sample_10bt"] == 8_750_000_128
    assert decay_phase["source_tokens"]["fineweb_edu_sample_10bt"] == 1_050_000_384
    assert stable_phase["source_tokens"]["finewiki_en"] == 899_999_744
    assert sources_by_name["fineweb_edu_sample_10bt"]["manifest"] == (
        "../data/pretrain-v1/fineweb-edu-sample-10bt/manifest.json"
    )
    assert sources_by_name["finewiki_en"]["manifest"] == (
        "../data/pretrain-v1/finewiki-en/manifest.json"
    )
    assert "fineweb_edu_dedup" not in sources_by_name
    assert "nemotron_knowledge" not in sources_by_name


def test_shipped_300m_recipes_hold_compute_constant_and_change_only_the_mix() -> None:
    repository_root = Path(__file__).parents[3]
    mixture = json.loads(
        (repository_root / "configs" / "pretrain_v1_english_300m_mixture.json").read_text(
            encoding="utf-8"
        )
    )
    baseline = json.loads(
        (repository_root / "configs" / "pretrain_v1_english_300m_fineweb_baseline.json").read_text(
            encoding="utf-8"
        )
    )

    assert mixture["phases"][0]["tokens"] == 299_892_736
    assert baseline["phases"][0]["tokens"] == 299_892_736
    assert sum(mixture["phases"][0]["source_tokens"].values()) == 299_892_736
    assert baseline["phases"][0]["source_tokens"] == {"fineweb_edu_sample_10bt": 299_892_736}
    assert all(
        token_count % 1_024 == 0 for token_count in mixture["phases"][0]["source_tokens"].values()
    )

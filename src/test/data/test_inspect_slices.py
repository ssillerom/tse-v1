import subprocess
import sys
from pathlib import Path

import pytest

from data.prepare_data import prepare_streaming_dataset


def test_inspection_decodes_prepared_slices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_load_dataset(**kwargs: object) -> list[dict[str, str]]:
        del kwargs
        return [{"text": "hello world hello world"}]

    monkeypatch.setattr("data.prepare_data.load_dataset", fake_load_dataset)
    prepare_streaming_dataset(
        output_dir=tmp_path,
        dataset_name="example/dataset",
        text_field="text",
        num_tokens=10,
        shard_size=10,
    )

    repository_root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.scripts.inspect_slices",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--split",
            "train",
            "--seq-len",
            "2",
            "--num-examples",
            "2",
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )

    output = result.stdout
    assert result.returncode == 0, result.stderr
    assert "Tokenizer: gpt2" in output
    assert "SEQUENCE 0" in output
    assert "SEQUENCE 1" in output
    assert "'hello world'" in output
    assert "SHIFT VALID: True" in output
    assert "CONTINUES PREVIOUS: True" in output

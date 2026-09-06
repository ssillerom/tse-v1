"""Shared execution policy used by training and benchmark entry points."""

import pytest
import torch

from src.runtime import precision_context, resolve_device, resolve_precision


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_device_prefers_cuda_then_mps_then_cpu(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert resolve_device("auto") == torch.device(expected)


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_explicit_unavailable_accelerator_is_rejected(monkeypatch, device):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(ValueError, match="not available"):
        resolve_device(device)


@pytest.mark.parametrize(
    ("device", "supported", "expected"),
    [("cpu", True, "fp32"), ("mps", True, "fp32"), ("cuda", True, "bf16"), ("cuda", False, "fp32")],
)
def test_auto_precision_uses_bf16_only_on_supported_cuda(monkeypatch, device, supported, expected):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: supported)
    assert resolve_precision("auto", torch.device(device)) == expected


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_explicit_bf16_rejects_unsupported_device(monkeypatch, device):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    with pytest.raises(ValueError, match="bf16"):
        resolve_precision("bf16", torch.device(device))


def test_precision_context_restores_autocast_state():
    assert not torch.is_autocast_enabled("cpu")
    with precision_context("cpu", "bf16"):
        assert torch.is_autocast_enabled("cpu")
        values = torch.ones(2, 2) @ torch.ones(2, 2)
        assert values.dtype == torch.bfloat16
    assert not torch.is_autocast_enabled("cpu")
    with precision_context("cpu", "fp32"):
        assert not torch.is_autocast_enabled("cpu")


def test_unknown_precision_is_rejected():
    with pytest.raises(ValueError, match="precision"):
        resolve_precision("fp16", torch.device("cpu"))

import torch

from model.config import ModelConfig
from model.gpt import GPT
from training.optimizer import build_adamw_parameter_groups


def test_adamw_parameter_groups_decay_matrices_but_not_norm_scales() -> None:
    model = GPT(
        ModelConfig(
            vocab_size=32,
            d_model=16,
            n_layers=2,
            n_heads=4,
            max_seq_len=8,
            tie_embeddings=True,
        )
    )

    groups = build_adamw_parameter_groups(model, weight_decay=0.1)

    assert [group["weight_decay"] for group in groups] == [0.1, 0.0]
    decayed = {id(parameter) for parameter in groups[0]["params"]}
    not_decayed = {id(parameter) for parameter in groups[1]["params"]}
    all_trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}

    assert decayed.isdisjoint(not_decayed)
    assert decayed | not_decayed == all_trainable
    assert id(model.token_embedding.weight) in decayed
    assert id(model.lm_head.weight) in decayed
    assert id(model.final_norm.gamma) in not_decayed
    assert all(parameter.ndim >= 2 for parameter in groups[0]["params"])
    assert all(parameter.ndim < 2 for parameter in groups[1]["params"])


def test_adamw_parameter_groups_can_construct_an_optimizer_with_tied_weights() -> None:
    model = GPT(
        ModelConfig(
            vocab_size=16,
            d_model=8,
            n_layers=1,
            n_heads=2,
            max_seq_len=4,
        )
    )

    optimizer = torch.optim.AdamW(
        build_adamw_parameter_groups(model, weight_decay=0.1),
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    grouped_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    assert len(grouped_parameters) == len({id(parameter) for parameter in grouped_parameters})

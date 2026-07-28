import torch
from torch.utils.data import DataLoader, TensorDataset

from model.config import ModelConfig
from model.gpt import GPT


def test_tiny_gpt_overfits_one_causal_batch() -> None:
    torch.manual_seed(42)
    config = ModelConfig(
        vocab_size=8,
        d_model=16,
        n_layers=2,
        n_heads=4,
        max_seq_len=6,
        dropout=0.0,
    )
    input_ids = torch.tensor(
        [
            [0, 4, 1, 4, 1, 4],
            [0, 4, 1, 4, 1, 4],
            [2, 4, 3, 4, 3, 4],
            [2, 4, 3, 4, 3, 4],
        ]
    )
    targets = torch.tensor(
        [
            [4, 1, 4, 1, 4, 1],
            [4, 1, 4, 1, 4, 1],
            [4, 3, 4, 3, 4, 3],
            [4, 3, 4, 3, 4, 3],
        ]
    )
    loader = DataLoader(
        TensorDataset(input_ids, targets),
        batch_size=len(input_ids),
        shuffle=False,
    )
    batch_inputs, batch_targets = next(iter(loader))
    model = GPT(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-2,
        weight_decay=0.0,
    )

    model.train()
    _, initial_loss = model(batch_inputs, batch_targets)
    assert initial_loss is not None
    initial_loss_value = initial_loss.item()

    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(batch_inputs, batch_targets)
        assert loss is not None
        assert torch.isfinite(loss)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        assert torch.isfinite(gradient_norm)
        optimizer.step()

    _, final_loss = model(batch_inputs, batch_targets)

    assert final_loss is not None
    final_loss_value = final_loss.item()
    assert final_loss_value < initial_loss_value * 0.25
    assert final_loss_value < 0.1

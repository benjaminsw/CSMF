import torch

from csmf.flows.conditional_realnvp import ConditionalRealNVP


def test_inverse_accepts_empty_factored_list_by_sampling_missing_latents():
    model = ConditionalRealNVP(h_dim=32, hidden_dims=[64, 64], debug=False)
    batch_size = 2

    z_final = torch.randn(batch_size, 4, 7, 7)
    y = torch.randn(batch_size, 1, 28, 28)

    x = model.inverse(z_final, [], y)

    assert x.shape == (batch_size, 1, 28, 28)
    assert torch.isfinite(x).all()

import torch

from models.vae import VAE


def test_vae_forward() -> None:
    model_config = {
        "down_channels": [4, 8, 8],
        "mid_channels": [8, 8],
        "down_sample": [True, False],
        "num_down_layers": 1,
        "num_mid_layers": 1,
        "num_up_layers": 1,
        "attn_down": [False, False],
        "z_channels": 2,
        "norm_channels": 4,
        "num_heads": 2,
        "scale_factor": 1.0,
    }
    model = VAE(im_channels=2, model_config=model_config)
    x = torch.randn(2, 2, 16, 8)
    out, enc = model(x)
    assert out.shape == x.shape
    assert enc.shape[0] == x.shape[0]

import torch

from models.denoiser import Denoiser, Unet


def test_denoiser_mlp_forward() -> None:
    model = Denoiser(input_dim=8, hidden_dim=16, n_layers=3, input_cond=4, hd_cond=8)
    x = torch.randn(2, 8)
    t = torch.randint(0, 10, (2,))
    cond = torch.randn(2, 1, 32, 32)
    out = model(x, t, cond)
    assert out.shape == x.shape


def test_unet_forward_no_condition() -> None:
    model_config = {
        "down_channels": [4, 8, 16],
        "mid_channels": [16, 8],
        "down_sample": [True, False],
        "time_emb_dim": 16,
        "num_down_layers": 1,
        "num_mid_layers": 1,
        "num_up_layers": 1,
        "attn_down": [False, False],
        "norm_channels": 4,
        "num_heads": 2,
        "conv_out_channels": 4,
        "z_channels": 2,
        "condition": False,
        "cond_channels": 2,
        "cross_attn": False,
    }
    model = Unet(im_channels=2, model_config=model_config)
    x = torch.randn(2, 2, 16, 8)
    t = torch.randint(0, 10, (2,))
    out = model(x, t)
    assert out.shape == x.shape

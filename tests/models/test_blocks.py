import torch

from models.blocks import DownBlock, MidBlock, UpBlock


def test_blocks_forward_shapes() -> None:
    batch, channels, freq, mics = 2, 4, 16, 8
    x = torch.randn(batch, channels, freq, mics)

    down = DownBlock(
        in_channels=channels,
        out_channels=8,
        t_emb_dim=None,
        down_sample=True,
        num_heads=2,
        num_layers=1,
        attn=False,
        norm_channels=4,
    )
    y = down(x)
    assert y.shape[0] == batch

    mid = MidBlock(
        in_channels=8,
        out_channels=8,
        t_emb_dim=None,
        num_heads=2,
        num_layers=1,
        norm_channels=4,
        use_dilated_conv=False,
    )
    z = mid(y)
    assert z.shape[0] == batch

    up = UpBlock(
        in_channels=8,
        out_channels=channels,
        t_emb_dim=None,
        up_sample=True,
        num_heads=2,
        num_layers=1,
        attn=False,
        norm_channels=4,
    )
    w = up(z, out_down=x)
    assert w.shape[0] == batch

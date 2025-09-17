import math
import torch
import torch.nn as nn

# from blocks import (
#     DownBlock,
#     MidBlock,
#     EncoderBlockRes4B,
#     DecoderBlockRes4B,
#     UpBlockUnet
# )
from models.blocks import (
    DownBlock,
    MidBlock,
    EncoderBlockRes4B,
    DecoderBlockRes4B,
    UpBlockUnet
)

class ResUNetDenoiser(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        """Unet architecture,
        found on github. SECOND TESTS not used for last test."""
        super(ResUNetDenoiser, self).__init__()
        # u-net architecture for latent denoiser
        self.input_channels = input_channels  # supposed to be 16 (?)

        # encoding part
        self.encoder_block1 = EncoderBlockRes4B(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=(3, 3),
            downsample=(2, 2)
        )
        self.encoder_block2 = EncoderBlockRes4B(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            downsample=(2, 1)
        )
        # middle part
        self.conv_block5a = EncoderBlockRes4B(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            downsample=(1, 1),
        )
        # decoding part
        self.decoder_block1 = DecoderBlockRes4B(
            in_channels=64,
            out_channels=32,
            kernel_size=(3, 3),
            upsample=(1, 1),
        )
        self.decoder_block2 = DecoderBlockRes4B(
            in_channels=32,
            out_channels=16,
            kernel_size=(3, 3),
            upsample=(2, 1),
        )
        self.after_conv_block1 = EncoderBlockRes4B(
            in_channels=16,
            out_channels=16,
            kernel_size=(3, 3),
            downsample=(1, 1),
        )

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # condition
        # concat that with input across channel dimension
        self.cond_conv_in = nn.Conv2d(in_channels=self.im_cond_input_ch,
                                        out_channels=self.im_cond_output_ch,
                                        kernel_size=1,
                                        bias=False)
        self.conv_in_concat = nn.Conv2d(input_channels + self.im_cond_output_ch,
                                        self.down_channels[0], kernel_size=3, padding=1)

    def forward(self, x, time, cond_input=None):
        if cond_input is not None:
            # encode condition
            im_cond = self.cond_conv_in(im_cond)
            assert im_cond.shape[-2:] == x.shape[-2:]
            x = torch.cat([x, im_cond], dim=1)
            # B x (C+N) x H x W
            out = self.conv_in_concat(x)

        t = self.time_mlp(time)

        # u-net
        (x1_pool, x1) = self.encoder_block1(out) + t
        (x2_pool, x2) = self.encoder_block2(x1_pool) + t

        (x_center, _) = self.conv_block5a(x2_pool) + t # (bs, 128, freq_bin/8, n_mics/16?)

        x3 = self.decoder_block1(x_center, x2) + t
        x4 = self.decoder_block2(x3, x1) + t

        (x, _) = self.after_conv_block1(x4) + t

        return x


class Denoiser(nn.Module):
    """
    Denoising diffusion model
    MLP architecture, first used for tests
    """
    def __init__(self, input_dim, hidden_dim, n_layers, input_cond, hd_cond):
        super(Denoiser, self).__init__()
        # input: input_cond 4x4, hd_cond=32 input_dim=latent_dim
        self.n_layers = n_layers
        self.input_cond = input_cond
        self.cond_mlp = nn.Sequential(
            nn.Linear(input_cond, hd_cond),
            nn.ReLU(),
            nn.Linear(hd_cond, hd_cond),
        )

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        mlp_layers = [nn.Linear(input_dim + hd_cond, hidden_dim)] + \
                        [nn.Linear(hidden_dim + hd_cond, hidden_dim) for _ in range(n_layers - 2)]
        mlp_layers.append(nn.Linear(hidden_dim, input_dim))
        self.mlp = nn.ModuleList(mlp_layers)

        bn_layers = [nn.BatchNorm1d(hidden_dim) for _ in range(n_layers - 1)]
        self.bn = nn.ModuleList(bn_layers)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.res_encoder = ResEncoder(1, num_blocks=4)

        self.mlp_enc = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, hd_cond),
            nn.LeakyReLU()
        )

    def forward(self, x, t, cond):
        z = self.res_encoder(cond)
        z = z.view(z.size(0), -1)  # Flatten to (batch_size, latent_dim)
        z = self.mlp_enc(z)
        t = self.time_mlp(t)
        for i in range(self.n_layers - 1):
            x = torch.cat((x, z), dim=-1)
            x = self.relu(self.mlp[i](x)) + t
            x = self.bn[i](x)
        x = self.mlp[self.n_layers-1](x)
        return x


# Position embeddings
class SinusoidalPositionEmbeddings(nn.Module): #ig to encode time as sinusoidal feature/ so model can learn time-dep patterns /\
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


def get_time_embedding(time_steps, temb_dim):
    r"""
    Convert time steps tensor into an embedding using the
    sinusoidal time embedding formula
    :param time_steps: 1D tensor of length batch size
    :param temb_dim: Dimension of the embedding
    :return: BxD embedding representation of B time steps
    """
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"

    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
    )

    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


class Unet(nn.Module):
    """
    Unet model architecture for Denoising diffusion model.
    Actual Module used for tests.
    """

    def __init__(self, im_channels, model_config):
        super().__init__()
        self.down_channels = model_config["down_channels"]
        self.mid_channels = model_config["mid_channels"]
        self.down_sample = model_config["down_sample"]
        self.t_emb_dim = model_config["time_emb_dim"]
        self.num_down_layers = model_config["num_down_layers"]
        self.num_mid_layers = model_config["num_mid_layers"]
        self.num_up_layers = model_config["num_up_layers"]
        self.attns = model_config["attn_down"]
        self.norm_channels = model_config["norm_channels"]
        self.num_heads = model_config["num_heads"]
        self.conv_out_channels = model_config["conv_out_channels"]
        self.z_channels = model_config["z_channels"]

        # Validate Unet config / check for errors
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1

        # Image conditionning config
        self.image_cond = model_config["condition"]
        self.im_cond_input_ch = model_config["cond_channels"]
        self.im_cond_output_ch = im_channels

        self.cond_conv_in = nn.Conv2d(in_channels=4,
                                      out_channels=self.im_cond_output_ch,
                                      kernel_size=1,
                                      bias=False)
        self.conv_in_concat = nn.Conv2d(im_channels + self.im_cond_output_ch,
                                        self.down_channels[0], kernel_size=3, padding=1)
        self.cond = self.image_cond
        #######################
        self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=1)

        # inital projection from sinusoidal time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )
        self.cross_attn = model_config['cross_attn']
        self.context_dim = 8
        # self.context_dim = 32
        self.up_sample = list(reversed(self.down_sample))


        self.downs = nn.ModuleList([])
        # Build Downblocks
        for i in range(len(self.down_channels) - 1):
            # Cross Attention and Context Dim only needed if text condition is present
            self.downs.append(
                DownBlock(
                    self.down_channels[i], self.down_channels[i + 1], self.t_emb_dim,
                            down_sample=self.down_sample[i],
                            num_heads=self.num_heads,
                            num_layers=self.num_down_layers,
                            attn=self.attns[i], norm_channels=self.norm_channels,
                            cross_attn=self.cross_attn,
                            context_dim=self.context_dim
                )
            )

        # Build Midblocks
        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.mids.append(
                MidBlock(
                    self.mid_channels[i], self.mid_channels[i + 1], self.t_emb_dim,
                            num_heads=self.num_heads,
                            num_layers=self.num_mid_layers,
                            norm_channels=self.norm_channels,
                            cross_attn=self.cross_attn,
                            context_dim=self.context_dim))

        self.ups = nn.ModuleList([])
        # Build the Upblocks
        for i in reversed(range(len(self.down_channels) - 1)):
            self.ups.append(
                UpBlockUnet(
                    self.down_channels[i] * 2, self.down_channels[i - 1] if i != 0 else self.conv_out_channels,
                            self.t_emb_dim, up_sample=self.down_sample[i],
                            num_heads=self.num_heads,
                            num_layers=self.num_up_layers,
                            norm_channels=self.norm_channels,
                            cross_attn=self.cross_attn,
                            context_dim=self.context_dim))

        self.norm_out = nn.GroupNorm(self.norm_channels, self.conv_out_channels)
        self.conv_out = nn.Conv2d(self.conv_out_channels, im_channels, kernel_size=3, padding=1)
        self.cond_mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.SiLU(),
            nn.Linear(32, 32)
        )

        # Encoder Cond_ module
        self.encoder_conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=(1, 1))
        self.encoder_layers = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.encoder_layers.append(DownBlock(self.down_channels[i], self.down_channels[i + 1],
                                                 t_emb_dim=None, down_sample=self.down_sample[i],
                                                 num_heads=self.num_heads,
                                                 num_layers=self.num_down_layers,
                                                 attn=False,
                                                 norm_channels=self.norm_channels))

        self.encoder_mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.encoder_mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i + 1],
                                              t_emb_dim=None,
                                              num_heads=self.num_heads,
                                              num_layers=self.num_mid_layers,
                                              norm_channels=self.norm_channels))

        self.encoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[-1])
        self.encoder_conv_out = nn.Conv2d(self.down_channels[-1], 2*self.z_channels, kernel_size=3, padding=1)

    def forward(self, x, t, cond_input=None):
        # Shapes assuming downblocks are [C1, C2, C3, C4]
        # Shapes assuming midblocks are [C4, C4, C3]
        # Shapes assuming downsamples are [True, True, False]
        if self.cond:
            assert cond_input is not None, \
                "Model with conditioning, thus cond_input cannot be None"
        if self.image_cond:
            ###################################
            # im_cond = torch.nn.functional.interpolate(cond_input, size=x.shape[-2:])
            # im_cond = im_cond.view(im_cond.size(0), 2, -1).permute(0, 2, 1)
            # enc_cond = self.cond_mlp(im_cond)
            im_cond = cond_input
            enc_cond = self.c_encoder(im_cond)
            enc_cond = self.cond_conv_in(enc_cond)
            # B x C x H x W
            assert enc_cond.shape[-2:] == x.shape[-2:], f"Need to be same shape, \
                but found {enc_cond.shape[-2:]} and {x.shape[-2:]}."
            x = torch.cat([x, enc_cond], dim=1)
            # B x (C+C') x H x W
            out = self.conv_in_concat(x)
            # prepare for cross attention:
            B, C, H, W = enc_cond.shape
            context_hidden_states = enc_cond.view(B, C*W, H).permute(0, 2, 1)
            ###################################
        else:
            # B x C x H x W
            out = self.conv_in(x)
        # B x C1 x H x W
        # t_emb -> B x t_emb_dim
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)

        down_outs = []
        # context_hidden_states = enc_cond  # (bs, 1024, 32)
        for down in self.downs:
            down_outs.append(out)
            out = down(out, t_emb, context_hidden_states)
            ######################## ADD OUT + TRUE ENCODED COND BY THE VAE ENCODER, E.G COND_INPUT
        for mid in self.mids:
            out = mid(out, t_emb, context_hidden_states)
        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb, context_hidden_states)
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        # out B x C x H x W
        return out

    def c_encoder(self, c):
        out = self.encoder_conv_in(c)
        for down in self.encoder_layers:
            out = down(out)
        for mid in self.encoder_mids:
            out = mid(out)
        out = self.encoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.encoder_conv_out(out)
        return out


class EncoderCond(nn.Module):
    def __init__(self, im_channels, model_config):
        super().__init__()
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']

        # To disable attention in Downblock of Encoder and Upblock of Decoder
        self.attns = model_config['attn_down']

        # Latent Dimension
        self.z_channels = model_config['z_channels']
        self.norm_channels = model_config['norm_channels']
        self.num_heads = model_config['num_heads']
        self.scale_factor = model_config['scale_factor']

        # Assertion to validate the channel information
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-1]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1

        # Wherever we use downsampling in encoder correspondingly use
        # upsampling in decoder
        self.up_sample = list(reversed(self.down_sample))

        ##################### Encoder ######################
        self.encoder_conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=(1, 1))

        # Downblock + Midblock
        self.encoder_layers = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.encoder_layers.append(DownBlock(self.down_channels[i], self.down_channels[i + 1],
                                                 t_emb_dim=None, down_sample=self.down_sample[i],
                                                 num_heads=self.num_heads,
                                                 num_layers=self.num_down_layers,
                                                 attn=self.attns[i],
                                                 norm_channels=self.norm_channels))

        self.encoder_mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.encoder_mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i + 1],
                                              t_emb_dim=None,
                                              num_heads=self.num_heads,
                                              num_layers=self.num_mid_layers,
                                              norm_channels=self.norm_channels))

        self.encoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[-1])
        self.encoder_conv_out = nn.Conv2d(self.down_channels[-1], 2*self.z_channels, kernel_size=3, padding=1)
        # self.encoder_view_in = lambda x: x.view(x.size(0), x.size(1), 128, 128)

    def forward(self, x):
        # x_resized = self.encoder_view_in(x)
        # out = self.encoder_conv_in(x_resized)
        out = self.encoder_conv_in(x)
        for down in self.encoder_layers:
            out = down(out)
        for mid in self.encoder_mids:
            out = mid(out)
        out = self.encoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.encoder_conv_out(out)
        # out = self.pre_quant_conv(out)
        mean, logvar = torch.chunk(out, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        sample = mean + std * torch.randn(mean.shape).to(device=x.device)
        return sample, out


# class FOAConditionEncoder(nn.Module):
#     def __init__(self, context_dim):
#         super().__init__()
#         self.proj = nn.Sequential(
#             nn.Conv1d(64, 32, kernel_size=3, padding=1),  # 2x4 = 8 input channels
#             nn.ReLU(),
#             nn.Conv1d(32, context_dim, kernel_size=1)
#         )

#     def forward(self, foa):  # foa: B x 2 x 32 x 32
#         b, c1, c2, t = foa.shape
#         x = foa.view(b, c1 * c2, t)  # B x 64 x 32
#         x = self.proj(x)  # B x 32 x 32
#         x = x.transpose(1, 2)  # B x 1024 x D
#         return x

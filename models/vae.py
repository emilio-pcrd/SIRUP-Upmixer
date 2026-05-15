import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks import DownBlock, MidBlock, UpBlock
from models.blocks import ResEncoderBlock, ResDecoderBlock


class VAE(nn.Module):
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
                                                 norm_channels=self.norm_channels
                                                ))

        self.encoder_mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.encoder_mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i + 1],
                                              t_emb_dim=None,
                                              num_heads=self.num_heads,
                                              num_layers=self.num_mid_layers,
                                              norm_channels=self.norm_channels,
                                              use_dilated_conv=True))

        self.encoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[-1])
        self.encoder_conv_out = nn.Conv2d(self.down_channels[-1], 2*self.z_channels, kernel_size=3, padding=1)

        ####################################################


        ##################### Decoder ######################
        self.decoder_conv_in = nn.Conv2d(self.z_channels, self.mid_channels[-1], kernel_size=3, padding=(1, 1))

        # Midblock + Upblock
        self.decoder_mids = nn.ModuleList([])
        for i in reversed(range(1, len(self.mid_channels))):
            self.decoder_mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i - 1],
                                              t_emb_dim=None,
                                              num_heads=self.num_heads,
                                              num_layers=self.num_mid_layers,
                                              norm_channels=self.norm_channels,
                                              use_dilated_conv=False))

        self.decoder_layers = nn.ModuleList([])
        for i in reversed(range(1, len(self.down_channels))):
            self.decoder_layers.append(UpBlock(self.down_channels[i], self.down_channels[i - 1],
                                               t_emb_dim=None, up_sample=self.down_sample[i - 1],
                                               num_heads=self.num_heads,
                                               num_layers=self.num_up_layers,
                                               attn=self.attns[i - 1],
                                               norm_channels=self.norm_channels,
                                               use_dilated_conv=False))

        self.decoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[0])
        self.decoder_conv_out = nn.Conv2d(self.down_channels[0], im_channels, kernel_size=3, padding=1)

    def encode(self, x):
        out = self.encoder_conv_in(x)
        for down in self.encoder_layers:
            out = down(out)
        for mid in self.encoder_mids:
            out = mid(out)
        out = self.encoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.encoder_conv_out(out)
        mean, logvar = torch.chunk(out, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        sample = mean + std * torch.randn(mean.shape).to(device=x.device)
        return sample, out

    def decode(self, z):
        out = z
        out = self.decoder_conv_in(out)
        for mid in self.decoder_mids:
            out = mid(out)
        for idx, up in enumerate(self.decoder_layers):
            out = up(out)

        out = self.decoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.decoder_conv_out(out)
        return out

    def forward(self, x):
        z, encoder_output = self.encode(x)
        out = self.decode(z)
        return out, encoder_output




###################" OTHER VAE ###########################


class ResEncoder(nn.Module):
    def __init__(self, in_channels, num_blocks=2):
        """Residual Encoder."""
        super(ResEncoder, self).__init__()

        self.in_channels = in_channels
        self.num_blocks = num_blocks

        init_num_channels = 16
        self.encoder_blocks = nn.ModuleList()
        self.encoder_block1 = ResEncoderBlock(
                in_channels=in_channels,
                out_channels=init_num_channels,
                kernel_size=(3, 3)
                )
        num_channels = init_num_channels
        for _ in range(num_blocks-2):
            self.encoder_blocks.append(
                ResEncoderBlock(
                    in_channels=num_channels,
                    out_channels=num_channels*2,
                    kernel_size=(3, 3)
                )
            )
            num_channels*=2
        self.encoder_block_last = ResEncoderBlock(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=(3, 3)
            )

    def forward(self, x):
        x = x[:, 0 : x.shape[1] - 1, ...]
        x = x.permute(0, 2, 1, 3)

        (x_pool, _) = self.encoder_block1(x)
        for block in self.encoder_blocks:
            (x_pool, _) = block(x_pool)
        (z, _) = self.encoder_block_last(x_pool)
        return z  # (bs, 64, F/8, 16/8)


class ResDecoder(nn.Module):
    def __init__(self, in_channels, output_dim, num_blocks=2):
        """Residual Decoder."""
        super(ResDecoder, self).__init__()

        self.in_channels = in_channels
        self.num_blocks = num_blocks

        self.decoder_blocks = nn.ModuleList()
        self.decoder_block1 = ResDecoderBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(3, 3),
                upsample=(2, 2)
                )
        num_channels = in_channels
        for _ in range(num_blocks-2):
            self.decoder_blocks.append(
                ResDecoderBlock(
                    in_channels=num_channels,
                    out_channels=num_channels//2,
                    kernel_size=(3, 3),
                    upsample=(2, 2)
                )
            )
            num_channels = int(num_channels//2)
        self.decoder_block_last = ResDecoderBlock(
                in_channels=num_channels,
                out_channels=output_dim,
                kernel_size=(3, 3),
                upsample=(2, 2)
            )

    def forward(self, x):
        x = self.decoder_block1(x)
        for block in self.decoder_blocks:
            x = block(x)  # (bs, ?, ?)
        x = self.decoder_block_last(x)
        return x  # (bs, 1, F, 16)


class ResVariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim,  output_dim, current_temp=1., num_blocks=3):
        super(ResVariationalAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.current_temp = current_temp
        self.res_encoder = ResEncoder(2, num_blocks=num_blocks)
        self.res_decoder = ResDecoder(
            in_channels=32,
            output_dim=2,
            num_blocks=num_blocks
        )
        self.decay_rate = 0.99
        self.latent_dim = latent_dim

        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

        self.mlp_enc = nn.Sequential(
            nn.Linear(8192, latent_dim),
            nn.LeakyReLU(),
        )
        self.mlp_dec = nn.Sequential(
            nn.Linear(latent_dim, 8192),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        z = self.res_encoder(x)  # (bs, 64, ...)
        # print(z.shape): (bs, 32, 128, 2)
        # Flatten or pool the output from the encoder to prepare for MLP
        z = z.view(z.size(0), -1)  # Flatten to (batch_size, latent_dim)
        z = self.mlp_enc(z)
        # z of size (bs, laten_dim)
        self.mu = self.fc_mu(z)
        self.logvar = self.fc_logvar(z)
        x_rec = self.reparameterize(self.mu, self.logvar)

        # adjust in corresponding format for input of the decoder
        # x_rec = x_rec.view(x_rec.size(0), self.latent_dim, 1, 1)
        x_rec = self.mlp_dec(x_rec)
        x_rec = x_rec.view(x_rec.size(0), 32, 128, 2)
        x_reconstructed = self.res_decoder(x_rec).squeeze(1)
        x = F.pad(x_reconstructed, pad=(0, 0, 1, 0))
        return x

    def reparameterize(self, mu, logvar, eps_scale=1.):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std) * eps_scale
        return eps.mul(std).add_(mu)

    def loss_function(self, x, x_reco, beta=0.1):
        recon_real = F.l1_loss(x_reco[:, 0, ...], x[..., 0, :], reduction='sum')
        recon_imag = F.l1_loss(x_reco[:, 1, ...], x[..., 1, :], reduction='sum')
        kld = -0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        recon = (recon_real + recon_imag) / 2
        loss = recon + beta*kld
        return loss, recon, kld

    def encode(self, x):
        z = self.res_encoder(x)
        print(z.shape)
        z = z.view(z.size(0), -1)
        z = self.mlp_enc(z)

        self.mu = self.fc_mu(z)
        self.logvar = self.fc_logvar(z)

        x_rec = self.reparameterize(self.mu, self.logvar)
        print(x_rec.shape)
        return x_rec

    def decode(self, mu, logvar):
        x_rec = self.reparameterize(mu, logvar)

        # adjust in corresponding format for input of the decoder
        x_rec = x_rec.view(x_rec.size(0), self.latent_dim, 1, 1)
        x_rec = self.mlp_dec(x_rec.squeeze())
        x_rec = x_rec.view(x_rec.size(0), 64, 64, 1)

        x_reconstructed = self.res_decoder(x_rec).squeeze(1)
        x = F.pad(x_reconstructed, pad=(0, 0, 1, 0))
        return x

    def decode_mu(self, mu):
        x_rec = mu.view(mu.size(0), self.latent_dim, 1, 1)
        x_rec = self.mlp_dec(x_rec.squeeze(-1).squeeze(-1))
        x_rec = x_rec.view(1, 64, 64, 1)

        x_reconstructed = self.res_decoder(x_rec).squeeze(1)
        x = F.pad(x_reconstructed, pad=(0, 0, 1, 0))
        return x

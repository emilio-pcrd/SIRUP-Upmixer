import torch
import torch.nn as nn
import torch.nn.functional as F


class DownBlock(nn.Module):
    r"""
    Down block for complex-valued spatial-spectral data (e.g. (2, F, M) format).
    Improvements:
    - 2D convolutions over frequency/mic
    - Dilation over frequency axis
    - Better attention handling for flattened (F x M) tokens
    """

    def __init__(self, in_channels, out_channels, t_emb_dim,
                 down_sample, num_heads, num_layers, attn, norm_channels,
                 cross_attn=False, context_dim=None, freq_dilation=2):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.attn = attn
        self.context_dim = context_dim
        self.cross_attn = cross_attn
        self.t_emb_dim = t_emb_dim

        # Main resnet-style blocks with dilated conv over frequency axis
        self.resnet_conv_first = nn.ModuleList()
        self.t_emb_layers = nn.ModuleList()
        self.resnet_conv_second = nn.ModuleList()
        self.residual_input_conv = nn.ModuleList()

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            # First conv
            self.resnet_conv_first.append(nn.Sequential(
                nn.GroupNorm(norm_channels, in_ch),
                nn.SiLU(),
                nn.Conv2d(in_ch, out_channels, kernel_size=3,
                          stride=1, padding=(freq_dilation, 1),
                          dilation=(freq_dilation, 1))
            ))
            # Time embedding projection
            if t_emb_dim is not None:
                self.t_emb_layers.append(nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                ))

            # Second conv
            self.resnet_conv_second.append(nn.Sequential(
                nn.GroupNorm(norm_channels, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3,
                          stride=1, padding=1)
            ))

            self.residual_input_conv.append(nn.Conv2d(in_ch, out_channels, kernel_size=1))

        # Attention setup
        if self.attn:
            self.attention_norms = nn.ModuleList([
                nn.GroupNorm(norm_channels, out_channels) for _ in range(num_layers)
            ])
            self.attentions = nn.ModuleList([
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                for _ in range(num_layers)
            ])

        if self.cross_attn:
            assert context_dim is not None, "Context Dimension must be passed for cross attention"
            self.cross_attention_norms = nn.ModuleList([
                nn.GroupNorm(norm_channels, out_channels) for _ in range(num_layers)
            ])
            self.cross_attentions = nn.ModuleList([
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                for _ in range(num_layers)
            ])
            self.context_proj = nn.ModuleList([
                nn.Linear(context_dim, out_channels) for _ in range(num_layers)
            ])

        self.down_sample_conv = nn.Conv2d(out_channels, out_channels,
                                          kernel_size=4, stride=2, padding=1) if down_sample else nn.Identity()

    def forward(self, x, t_emb=None, context=None):
        # x: (B, 2, F, M)
        out = x
        for i in range(self.num_layers):
            res = out
            out = self.resnet_conv_first[i](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](res)

            if self.attn:
                B, C, F, M = out.shape
                normed = self.attention_norms[i](out)
                tokens = normed.permute(0, 2, 3, 1).reshape(B, F * M, C)  # (B, Seq, C)
                out_attn, _ = self.attentions[i](tokens, tokens, tokens)
                out_attn = out_attn.view(B, F, M, C).permute(0, 3, 1, 2)
                out = out + out_attn

            if self.cross_attn:
                assert context is not None
                B, C, F, M = out.shape
                normed = self.cross_attention_norms[i](out)
                tokens = normed.permute(0, 2, 3, 1).reshape(B, F * M, C)
                context_proj = self.context_proj[i](context)  # (B, S, C)
                out_attn, _ = self.cross_attentions[i](tokens, context_proj, context_proj)
                out_attn = out_attn.view(B, F, M, C).permute(0, 3, 1, 2)
                out = out + out_attn

        out = self.down_sample_conv(out)
        return out


# class DownBlock(nn.Module):
#     r"""
#     Down conv block with attention.
#     Sequence of following block
#     1. Resnet block with time embedding
#     2. Attention block
#     3. Downsample
#     """

#     def __init__(self, in_channels, out_channels, t_emb_dim,
#                  down_sample, num_heads, num_layers, attn, norm_channels, cross_attn=False, context_dim=None):
#         super().__init__()
#         self.num_layers = num_layers
#         self.down_sample = down_sample
#         self.attn = attn
#         self.context_dim = context_dim
#         self.cross_attn = cross_attn
#         self.t_emb_dim = t_emb_dim
#         self.resnet_conv_first = nn.ModuleList(
#             [
#                 nn.Sequential(
#                     nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
#                     nn.SiLU(),
#                     nn.Conv2d(in_channels if i == 0 else out_channels, out_channels,
#                               kernel_size=3, stride=1, padding=1),
#                 )
#                 for i in range(num_layers)
#             ]
#         )
#         if self.t_emb_dim is not None:
#             self.t_emb_layers = nn.ModuleList([
#                 nn.Sequential(
#                     nn.SiLU(),
#                     nn.Linear(self.t_emb_dim, out_channels)
#                 )
#                 for _ in range(num_layers)
#             ])
#         self.resnet_conv_second = nn.ModuleList(
#             [
#                 nn.Sequential(
#                     nn.GroupNorm(norm_channels, out_channels),
#                     nn.SiLU(),
#                     nn.Conv2d(out_channels, out_channels,
#                               kernel_size=3, stride=1, padding=1),
#                 )
#                 for _ in range(num_layers)
#             ]
#         )

#         if self.attn:
#             self.attention_norms = nn.ModuleList(
#                 [nn.GroupNorm(norm_channels, out_channels)
#                  for _ in range(num_layers)]
#             )

#             self.attentions = nn.ModuleList(
#                 [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
#                  for _ in range(num_layers)]
#             )

#         if self.cross_attn:
#             assert context_dim is not None, "Context Dimension must be passed for cross attention"
#             self.cross_attention_norms = nn.ModuleList(
#                 [nn.GroupNorm(norm_channels, out_channels)
#                  for _ in range(num_layers)]
#             )
#             self.cross_attentions = nn.ModuleList(
#                 [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
#                  for _ in range(num_layers)]
#             )
#             self.context_proj = nn.ModuleList(
#                 [nn.Linear(context_dim, out_channels)
#                  for _ in range(num_layers)]
#             )

#         self.residual_input_conv = nn.ModuleList(
#             [
#                 nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
#                 for i in range(num_layers)
#             ]
#         )
#         self.down_sample_conv = nn.Conv2d(out_channels, out_channels,
#                                           4, 2, 1) if self.down_sample else nn.Identity()

#     def forward(self, x, t_emb=None, context=None):
#         out = x
#         for i in range(self.num_layers):
#             # Resnet block of Unet
#             resnet_input = out
#             out = self.resnet_conv_first[i](out)
#             if self.t_emb_dim is not None:
#                 out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
#             out = self.resnet_conv_second[i](out)
#             out = out + self.residual_input_conv[i](resnet_input)

#             if self.attn:
#                 # Attention block of Unet
#                 batch_size, channels, h, w = out.shape
#                 in_attn = out.reshape(batch_size, channels, h * w)
#                 in_attn = self.attention_norms[i](in_attn)
#                 in_attn = in_attn.transpose(1, 2)
#                 out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
#                 out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
#                 out = out + out_attn

#             if self.cross_attn:
#                 assert context is not None, "context cannot be None if cross attention layers are used"
#                 batch_size, channels, h, w = out.shape
#                 in_attn = out.reshape(batch_size, channels, h * w)
#                 in_attn = self.cross_attention_norms[i](in_attn)
#                 in_attn = in_attn.transpose(1, 2)
#                 assert context.shape[0] == x.shape[0] and context.shape[-1] == self.context_dim
#                 assert context.dim() == 3
#                 context_proj = self.context_proj[i](context)
#                 out_attn, _ = self.cross_attentions[i](in_attn, context_proj, context_proj)
#                 out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
#                 out = out + out_attn

#         # Downsample
#         out = self.down_sample_conv(out)
#         return out


# class MidBlock(nn.Module):
#     r"""
#     Mid conv block with attention.
#     Sequence of following blocks
#     1. Resnet block with time embedding
#     2. Attention block
#     3. Resnet block with time embedding
#     """

#     def __init__(self, in_channels, out_channels, t_emb_dim, num_heads, num_layers, norm_channels, cross_attn=None, context_dim=None):
#         super().__init__()
#         self.num_layers = num_layers
#         self.t_emb_dim = t_emb_dim
#         self.context_dim = context_dim
#         self.cross_attn = cross_attn
#         self.resnet_conv_first = nn.ModuleList(
#             [
#                 nn.Sequential(
#                     nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
#                     nn.SiLU(),
#                     nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
#                               padding=1),
#                 )
#                 for i in range(num_layers + 1)
#             ]
#         )

#         if self.t_emb_dim is not None:
#             self.t_emb_layers = nn.ModuleList([
#                 nn.Sequential(
#                     nn.SiLU(),
#                     nn.Linear(t_emb_dim, out_channels)
#                 )
#                 for _ in range(num_layers + 1)
#             ])
#         self.resnet_conv_second = nn.ModuleList(
#             [
#                 nn.Sequential(
#                     nn.GroupNorm(norm_channels, out_channels),
#                     nn.SiLU(),
#                     nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
#                 )
#                 for _ in range(num_layers + 1)
#             ]
#         )

#         self.attention_norms = nn.ModuleList(
#             [nn.GroupNorm(norm_channels, out_channels)
#              for _ in range(num_layers)]
#         )

#         self.attentions = nn.ModuleList(
#             [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
#              for _ in range(num_layers)]
#         )
#         if self.cross_attn:
#             assert context_dim is not None, "Context Dimension must be passed for cross attention"
#             self.cross_attention_norms = nn.ModuleList(
#                 [nn.GroupNorm(norm_channels, out_channels)
#                  for _ in range(num_layers)]
#             )
#             self.cross_attentions = nn.ModuleList(
#                 [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
#                  for _ in range(num_layers)]
#             )
#             self.context_proj = nn.ModuleList(
#                 [nn.Linear(context_dim, out_channels)
#                  for _ in range(num_layers)]
#             )
#         self.residual_input_conv = nn.ModuleList(
#             [
#                 nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
#                 for i in range(num_layers + 1)
#             ]
#         )

#     def forward(self, x, t_emb=None, context=None):
#         out = x

#         # First resnet block
#         resnet_input = out
#         out = self.resnet_conv_first[0](out)
#         if self.t_emb_dim is not None:
#             out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
#         out = self.resnet_conv_second[0](out)
#         out = out + self.residual_input_conv[0](resnet_input)

#         for i in range(self.num_layers):
#             # Attention Block
#             batch_size, channels, h, w = out.shape
#             in_attn = out.reshape(batch_size, channels, h * w)
#             in_attn = self.attention_norms[i](in_attn)
#             in_attn = in_attn.transpose(1, 2)
#             out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
#             out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
#             out = out + out_attn

#             if self.cross_attn:
#                 assert context is not None, "context cannot be None if cross attention layers are used"
#                 batch_size, channels, h, w = out.shape
#                 in_attn = out.reshape(batch_size, channels, h * w)
#                 in_attn = self.cross_attention_norms[i](in_attn)
#                 in_attn = in_attn.transpose(1, 2)
#                 assert context.shape[0] == x.shape[0] and context.shape[-1] == self.context_dim
#                 context_proj = self.context_proj[i](context)
#                 out_attn, _ = self.cross_attentions[i](in_attn, context_proj, context_proj)
#                 out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
#                 out = out + out_attn


#             # Resnet Block
#             resnet_input = out
#             out = self.resnet_conv_first[i + 1](out)
#             if self.t_emb_dim is not None:
#                 out = out + self.t_emb_layers[i + 1](t_emb)[:, :, None, None]
#             out = self.resnet_conv_second[i + 1](out)
#             out = out + self.residual_input_conv[i + 1](resnet_input)

#         return out


class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads, num_layers, norm_channels,
                 cross_attn=False, context_dim=None, dilation=2, use_dilated_conv=True):
        super().__init__()
        self.num_layers = num_layers
        self.t_emb_dim = t_emb_dim
        self.context_dim = context_dim
        self.cross_attn = cross_attn
        self.use_dilated_conv = use_dilated_conv

        self.resnet_conv_first = nn.ModuleList()
        self.resnet_conv_second = nn.ModuleList()
        self.t_emb_layers = nn.ModuleList()
        self.residual_input_conv = nn.ModuleList()
        self.attention_norms = nn.ModuleList()
        self.attentions = nn.ModuleList()

        if cross_attn:
            assert context_dim is not None, "context_dim is required for cross-attention"
            self.cross_attention_norms = nn.ModuleList()
            self.cross_attentions = nn.ModuleList()
            self.context_proj = nn.ModuleList()

        for i in range(num_layers + 1):
            in_ch = in_channels if i == 0 else out_channels

            self.resnet_conv_first.append(nn.Sequential(
                nn.GroupNorm(norm_channels, in_ch),
                nn.SiLU(),
                nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=dilation if use_dilated_conv else 1,
                          dilation=dilation if use_dilated_conv else 1)
            ))

            if t_emb_dim is not None:
                self.t_emb_layers.append(nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                ))

            self.resnet_conv_second.append(nn.Sequential(
                nn.GroupNorm(norm_channels, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            ))

            self.residual_input_conv.append(
                nn.Conv2d(in_ch, out_channels, kernel_size=1)
            )

            if i < num_layers:
                self.attention_norms.append(nn.GroupNorm(norm_channels, out_channels))
                self.attentions.append(nn.MultiheadAttention(out_channels, num_heads, batch_first=True))

                if cross_attn:
                    self.cross_attention_norms.append(nn.GroupNorm(norm_channels, out_channels))
                    self.cross_attentions.append(nn.MultiheadAttention(out_channels, num_heads, batch_first=True))
                    self.context_proj.append(nn.Linear(context_dim, out_channels))

    def forward(self, x, t_emb=None, context=None):
        out = x

        # First ResNet Block
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        if self.t_emb_dim is not None:
            out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)

        # Loop over attention and res blocks
        for i in range(self.num_layers):
            B, C, H, W = out.shape
            reshaped = out.view(B, C, H * W)
            reshaped = self.attention_norms[i](reshaped)
            reshaped = reshaped.transpose(1, 2)

            # Self-attention
            attn_out, _ = self.attentions[i](reshaped, reshaped, reshaped)
            attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
            out = out + attn_out  # residual

            # Optional Cross-attention
            if self.cross_attn:
                assert context is not None
                context_proj = self.context_proj[i](context)
                cross_attn_in = self.cross_attention_norms[i](out.view(B, C, H * W)).transpose(1, 2)
                cross_out, _ = self.cross_attentions[i](cross_attn_in, context_proj, context_proj)
                cross_out = cross_out.transpose(1, 2).view(B, C, H, W)
                out = out + cross_out

            # ResNet block
            resnet_input = out
            out = self.resnet_conv_first[i + 1](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i + 1](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i + 1](out)
            out = out + self.residual_input_conv[i + 1](resnet_input)

        return out


# class UpBlock(nn.Module):
#     r"""
#     Up conv block with attention.
#     Sequence of following blocks
#     1. Upsample
#     1. Concatenate Down block output
#     2. Resnet block with time embedding
#     3. Attention Block
#     """

#     def __init__(self, in_channels, out_channels, t_emb_dim,
#                  up_sample, num_heads, num_layers, attn, norm_channels):
#         super().__init__()
#         self.num_layers = num_layers
#         self.up_sample = up_sample
#         self.t_emb_dim = t_emb_dim
#         self.attn = attn
#         self.resnet_conv_first = nn.ModuleList(
#             [
#                 nn.Sequential(
#                     nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
#                     nn.SiLU(),
#                     nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
#                               padding=1),
#                 )
#                 for i in range(num_layers)
#             ]
#         )

#         if self.t_emb_dim is not None:
#             self.t_emb_layers = nn.ModuleList([
#                 nn.Sequential(
#                     nn.SiLU(),
#                     nn.Linear(t_emb_dim, out_channels)
#                 )
#                 for _ in range(num_layers)
#             ])

#         self.resnet_conv_second = nn.ModuleList(
#             [
#                 nn.Sequential(
#                     nn.GroupNorm(norm_channels, out_channels),
#                     nn.SiLU(),
#                     nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
#                 )
#                 for _ in range(num_layers)
#             ]
#         )
#         if self.attn:
#             self.attention_norms = nn.ModuleList(
#                 [
#                     nn.GroupNorm(norm_channels, out_channels)
#                     for _ in range(num_layers)
#                 ]
#             )

#             self.attentions = nn.ModuleList(
#                 [
#                     nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
#                     for _ in range(num_layers)
#                 ]
#             )

#         self.residual_input_conv = nn.ModuleList(
#             [
#                 nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
#                 for i in range(num_layers)
#             ]
#         )
#         self.up_sample_conv = nn.ConvTranspose2d(in_channels, in_channels,
#                                                  4, 2, 1) \
#             if self.up_sample else nn.Identity()

#     def forward(self, x, out_down=None, t_emb=None):
#         # Upsample
#         x = self.up_sample_conv(x)

#         # Concat with Downblock output
#         if out_down is not None:
#             x = torch.cat([x, out_down], dim=1)

#         out = x
#         for i in range(self.num_layers):
#             # Resnet Block
#             resnet_input = out
#             out = self.resnet_conv_first[i](out)
#             if self.t_emb_dim is not None:
#                 out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
#             out = self.resnet_conv_second[i](out)
#             out = out + self.residual_input_conv[i](resnet_input)

#             # Self Attention
#             if self.attn:
#                 batch_size, channels, h, w = out.shape
#                 in_attn = out.reshape(batch_size, channels, h * w)
#                 in_attn = self.attention_norms[i](in_attn)
#                 in_attn = in_attn.transpose(1, 2)
#                 out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
#                 out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
#                 out = out + out_attn
#         return out


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim,
                 up_sample=True, num_heads=4, num_layers=2, attn=True, norm_channels=32,
                 dilation=2, use_dilated_conv=True):
        super().__init__()
        self.num_layers = num_layers
        self.t_emb_dim = t_emb_dim
        self.attn = attn
        self.use_dilated_conv = use_dilated_conv

        self.resnet_conv_first = nn.ModuleList()
        self.resnet_conv_second = nn.ModuleList()
        self.residual_input_conv = nn.ModuleList()
        self.attention_norms = nn.ModuleList() if attn else None
        self.attentions = nn.ModuleList() if attn else None
        self.t_emb_layers = nn.ModuleList() if t_emb_dim is not None else None

        conv_pad = dilation if use_dilated_conv else 1

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels

            self.resnet_conv_first.append(nn.Sequential(
                nn.GroupNorm(norm_channels, in_ch),
                nn.SiLU(),
                nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=conv_pad,
                          dilation=dilation if use_dilated_conv else 1)
            ))

            if t_emb_dim is not None:
                self.t_emb_layers.append(nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                ))

            self.resnet_conv_second.append(nn.Sequential(
                nn.GroupNorm(norm_channels, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            ))

            self.residual_input_conv.append(
                nn.Conv2d(in_ch, out_channels, kernel_size=1)
            )

            if attn:
                self.attention_norms.append(nn.GroupNorm(norm_channels, out_channels))
                self.attentions.append(nn.MultiheadAttention(out_channels, num_heads, batch_first=True))

        self.up_sample_conv = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=4, stride=2, padding=1
        ) if up_sample else nn.Identity()

    def forward(self, x, out_down=None, t_emb=None):
        x = self.up_sample_conv(x)

        # Concatenate with skip connection (downward path)
        if out_down is not None:
            x = torch.cat([x, out_down], dim=1)

        out = x
        for i in range(self.num_layers):
            resnet_input = out
            out = self.resnet_conv_first[i](out)

            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]

            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)

            if self.attn:
                B, C, H, W = out.shape
                attn_in = self.attention_norms[i](out.view(B, C, H * W)).transpose(1, 2)
                attn_out, _ = self.attentions[i](attn_in, attn_in, attn_in)
                attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
                out = out + attn_out

        return out


###################################################################################"


class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        """Residual block."""
        super(ResConvBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        padding = [kernel_size[0] // 2, kernel_size[1] // 2]
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=padding,
            bias=False
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=padding,
            bias=False
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0)
            )
            self.is_shortcut = True
        else:
            self.is_shortcut = False

    def forward(self, x):
        origin = x

        x = self.conv1(F.leaky_relu(self.bn1(x), negative_slope=0.01))
        x = self.conv2(F.leaky_relu(self.bn2(x), negative_slope=0.01))

        if self.is_shortcut:
            return self.shortcut(origin) + x
        else:
            return origin + x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, upsample):
        """Residual decoder block, contain 1 transpose conv and 8 conv layers."""
        super(ResDecoderBlock, self).__init__()

        self.transpose_conv1 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=upsample,
            stride=upsample,
            padding=(0, 0),
            bias=False,
            dilation=(1, 1)
        )

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv_block1 = ResConvBlock(
            out_channels, out_channels, kernel_size
        )
        self.conv_block2 = ResConvBlock(
            out_channels, out_channels, kernel_size
        )

    def forward(self, x):
        x = self.transpose_conv1(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)

        return x


class ResEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        """Residual Encoder Block. Following architecture of https://arxiv.org/pdf/2109.05418."""
        super(ResEncoderBlock, self).__init__()

        self.conv_block1 = ResConvBlock(
            in_channels, out_channels, kernel_size
        )
        self.conv_block2 = ResConvBlock(
            out_channels, out_channels, kernel_size
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        encoder_pool = F.avg_pool2d(x, kernel_size=2)

        return encoder_pool, x


class EncoderBlockRes4B(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, downsample
    ):
        r"""Encoder block, contains 8 convolutional layers."""
        super(EncoderBlockRes4B, self).__init__()

        self.conv_block1 = ResConvBlock(
            in_channels, out_channels, kernel_size
        )
        self.conv_block2 = ResConvBlock(
            out_channels, out_channels, kernel_size
        )
        self.conv_block3 = ResConvBlock(
            out_channels, out_channels, kernel_size
        )
        self.conv_block4 = ResConvBlock(
            out_channels, out_channels, kernel_size
        )
        self.downsample = downsample

    def forward(self, x):
        encoder = self.conv_block1(x)
        encoder = self.conv_block2(encoder)
        encoder = self.conv_block3(encoder)
        encoder = self.conv_block4(encoder)
        encoder_pool = F.avg_pool2d(encoder, kernel_size=self.downsample)
        return encoder_pool, encoder


class DecoderBlockRes4B(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, upsample
    ):
        r"""Decoder block, contains 1 transpose convolutional and 8 convolutional layers."""
        super(DecoderBlockRes4B, self).__init__()
        self.kernel_size = kernel_size
        self.stride = upsample

        self.conv1 = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.stride,
            stride=self.stride,
            padding=(0, 0),
            bias=False,
            dilation=(1, 1),
        )

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv_block2 = ResConvBlock(
            out_channels * 2, out_channels, kernel_size
        )
        self.conv_block3 = ResConvBlock(
            out_channels, out_channels, kernel_size
        )
        self.conv_block4 = ResConvBlock(
            out_channels, out_channels, kernel_size
        )
        self.conv_block5 = ResConvBlock(
            out_channels, out_channels, kernel_size
        )

    def forward(self, input_tensor, concat_tensor):
        x = self.conv1(F.relu(self.bn1(input_tensor)))
        x = torch.cat((x, concat_tensor), dim=1)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        return x


class UpBlockUnet(nn.Module):
    r"""
    Up conv block with attention.
    Sequence of following blocks
    1. Upsample
    1. Concatenate Down block output
    2. Resnet block with time embedding
    3. Attention Block
    """

    def __init__(self, in_channels, out_channels, t_emb_dim, up_sample,
                 num_heads, num_layers, norm_channels, cross_attn=False, context_dim=None):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.t_emb_dim = t_emb_dim
        self.cross_attn = cross_attn
        self.context_dim = context_dim
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
                )
                for i in range(num_layers)
            ]
        )

        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
                for _ in range(num_layers)
            ])

        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )

        self.attention_norms = nn.ModuleList(
            [
                nn.GroupNorm(norm_channels, out_channels)
                for _ in range(num_layers)
            ]
        )

        self.attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                for _ in range(num_layers)
            ]
        )

        if self.cross_attn:
            assert context_dim is not None, "Context Dimension must be passed for cross attention"
            self.cross_attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            self.cross_attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )
            self.context_proj = nn.ModuleList(
                [nn.Linear(context_dim, out_channels)
                 for _ in range(num_layers)]
            )
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2,
                                                 4, 2, 1) \
            if self.up_sample else nn.Identity()

    def forward(self, x, out_down=None, t_emb=None, context=None):
        x = self.up_sample_conv(x)
        if out_down is not None:
            x = torch.cat([x, out_down], dim=1)

        out = x
        for i in range(self.num_layers):
            # Resnet
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
            # Self Attention
            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn
            # Cross Attention
            if self.cross_attn:
                assert context is not None, "context cannot be None if cross attention layers are used"
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                assert len(context.shape) == 3, \
                    "Context shape does not match B,_,CONTEXT_DIM"
                assert context.shape[0] == x.shape[0] and context.shape[-1] == self.context_dim,\
                    "Context shape does not match B,_,CONTEXT_DIM"
                context_proj = self.context_proj[i](context)
                out_attn, _ = self.cross_attentions[i](in_attn, context_proj, context_proj)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn

        return out

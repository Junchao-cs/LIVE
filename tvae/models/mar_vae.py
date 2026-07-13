# Adopted from LDM's KL-VAE: https://github.com/CompVis/latent-diffusion

import os
import re
import math
import numpy as np
from abc import abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Tuple, Union
from rich import print
from packaging import version

import torch
import torch.nn as nn
import einops
import lightning.pytorch as pl

from omegaconf import ListConfig
from safetensors.torch import load_file as load_safetensors
from torchmetrics.image.fid import FrechetInceptionDistance  # pip install torch-fidelity
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure

from sgm.util import default, get_obj_from_str, instantiate_from_config, print0, get_valid_paths, disabled_train, disabled_train_func
from tvae.modules.ema import LitEma
from tvae.modules.metrics import compute_psnr, compute_ssim, InceptionV3, calculate_frechet_distance
from tvae.taming.lpips import LPIPS


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        out_ch=3,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=256,
        z_channels=16,
        double_z=True,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        # assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        out_ch=3,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=256,
        z_channels=16,
        give_pre_end=False,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print0(
            "[bold cyan]\[timg.mar.modules.mar_vae][Decoder][/bold cyan] Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        # z to block_in
        self.conv_in = torch.nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def get_last_layer(self, **kwargs):
        return self.conv_out.weight

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean


class AutoencodingLegacy(pl.LightningModule):
    """
    https://github.com/LTH14/mar/issues/3
    https://github.com/LTH14/mar/issues/6
    https://github.com/TencentARC/Open-MAGVIT2
    """
    def __init__(self,
                 embed_dim: int,
                 ch_mult: Tuple,
                 regularizer_config: Dict,
                 loss_config: Union[None, Dict] = None,
                 use_variational=True,
                 fix_encoder: bool = False,
                 optimizer_config: Union[Dict, None] = None,
                 lr_g_factor: float = 1.0,
                 ema_decay: Union[None, float] = None,
                 max_batch_size: Union[None, int] = None,
                 compile_model: bool = False,
                 monitor: Union[None, str] = None,
                 input_key: str = "jpg",
                 ckpt_path: Union[None, str] = None,
                 ckpt_path2: Union[None, str] = None,
                 ignore_keys: Union[Tuple, list, ListConfig] = (),
    ):
        super().__init__()
        # automatically save hyperparameters to the checkpoint
        # can be instantiated with mode.load_from_checkpoint(ckpt_path)
        self.save_hyperparameters()

        self.input_key = input_key
        self.use_ema = ema_decay is not None
        if monitor is not None:
            self.monitor = monitor
        self.max_batch_size = max_batch_size

        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0"))
            and compile_model
            else lambda x: x
        )

        # init encoder
        self.encoder = Encoder(ch_mult=ch_mult, z_channels=embed_dim)
        # fix encoder
        self.fix_encoder = fix_encoder
        if self.fix_encoder:
            self.encoder.train = disabled_train
            for param in self.encoder.parameters():
                param.requires_grad = False
        # init decoder
        self.decoder = Decoder(ch_mult=ch_mult, z_channels=embed_dim)
        self.loss = instantiate_from_config(loss_config) if loss_config is not None else None
        self.regularization = instantiate_from_config(regularizer_config)
        if self.fix_encoder:
            self.regularization.train = disabled_train
            for param in self.regularization.parameters():
                param.requires_grad = False
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.Adam"}
        )
        self.lr_g_factor = lr_g_factor

        self.use_variational = use_variational
        mult = 2 if self.use_variational else 1
        self.quant_conv = torch.nn.Conv2d(2 * embed_dim, mult * embed_dim, 1)
        if self.fix_encoder:
            self.quant_conv.train = disabled_train
            for param in self.quant_conv.parameters():
                param.requires_grad = False
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, embed_dim, 1)
        self.embed_dim = embed_dim

        ckpt_path = get_valid_paths(ckpt_path, ckpt_path2)
        print0(f"[bold magenta]\[tvae.models.mar_vae][AutoencodingLegacy][/bold magenta] Use ckpt_path: {ckpt_path}")
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        # evaluation metrics
        self.perceptual_loss = LPIPS().eval()
        # torchmetrics
        self.metrics_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.metrics_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.metrics_fid = FrechetInceptionDistance(feature=2048)
        self.metrics_lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg")
        self.metrics_is = InceptionScore()
        if self.use_ema:
            self.metrics_ema_psnr = PeakSignalNoiseRatio(data_range=1.0)
            self.metrics_ema_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
            self.metrics_ema_fid = FrechetInceptionDistance(feature=2048)
            self.metrics_ema_lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg")
            self.metrics_ema_is = InceptionScore()

        if self.use_ema:
            self.model_ema = LitEma(self, decay=ema_decay)
            print0(f"[bold magenta]\[tvae.models.mar_vae][AutoencodingLegacy][/bold magenta] Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
            self.model_ema.init_to(self)

        if version.parse(torch.__version__) >= version.parse("2.0.0"):
            self.automatic_optimization = False

    def init_from_ckpt(self, path, ignore_keys=()):
        weights = torch.load(path, map_location="cpu", weights_only=False)
        if "state_dict" in weights:
            weights = weights["state_dict"]
        elif "model" in weights:
            weights = weights["model"]

        keys = list(weights.keys())
        for k in keys:
            for ik in ignore_keys:
                if re.match(ik, k):
                    # print0(f"[bold magenta]\[tvae.models.mar_vae][AutoencodingLegacy][/bold magenta] Deleting key {k} from state_dict.")
                    del weights[k]

        missing, unexpected = self.load_state_dict(weights, strict=False)
        print0(f"[bold cyan]\[tvae.models.mar_vae][AutoencodingLegacy][/bold cyan] Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print0(f"[bold cyan]\[tvae.models.mar_vae][AutoencodingLegacy][/bold cyan] Missing Keys: {missing}")
        if len(unexpected) > 0:
            print0(f"[bold cyan]\[tvae.models.mar_vae][AutoencodingLegacy][/bold cyan] Unexpected Keys: {unexpected}")

    def get_input(self, batch: Dict) -> torch.Tensor:
        # assuming unified data format, dataloader returns a dict.
        # image tensors should be scaled to -1 ... 1 and in channels-first format (e.g., bchw instead if bhwc)
        if batch[self.input_key].dim() == 5:  # video
            batch[self.input_key] = einops.rearrange(batch[self.input_key], 'b c t h w -> (b t) c h w')
        return batch[self.input_key]

    def get_autoencoder_params(self) -> list:
        params = (
            # list(self.encoder.parameters())
            # list(self.quant_conv.parameters())
            # list(self.regularization.get_trainable_parameters())
            list(self.post_quant_conv.parameters())
            + list(self.decoder.parameters())
            + list(self.loss.get_trainable_autoencoder_parameters())
        )
        if not self.fix_encoder:
            params = list(self.encoder.parameters()) + list(self.quant_conv.parameters()) + list(self.regularization.get_trainable_parameters()) + params
        return params

    def get_discriminator_params(self) -> list:
        params = list(self.loss.get_trainable_parameters())  # e.g., discriminator
        return params

    def get_last_layer(self):
        return self.decoder.get_last_layer()

    def on_train_batch_end(self, *args, **kwargs):
        # for EMA computation
        if self.use_ema:
            self.model_ema(self)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print0(f"[bold magenta]\[tvae.models.mar_vae][AutoencodingLegacy][/bold magenta] {context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print0(f"[bold magenta]\[tvae.models.mar_vae][AutoencodingLegacy][/bold magenta] {context}: Restored training weights")

    @torch.no_grad()
    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z = self.quant_conv(z)
        posterior = DiagonalGaussianDistribution(z)
        moments = posterior.parameters
        return moments

    def encode(
        self, x: torch.Tensor, return_reg_log: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:

        if self.max_batch_size is None:
            z = self.encoder(x)
            z = self.quant_conv(z)
        else:
            N = x.shape[0]
            bs = self.max_batch_size
            n_batches = int(math.ceil(N / bs))
            z = list()
            for i_batch in range(n_batches):
                z_batch = self.encoder(x[i_batch * bs : (i_batch + 1) * bs])
                z_batch = self.quant_conv(z_batch)
                z.append(z_batch)
            z = torch.cat(z, 0)

        z, reg_log = self.regularization(z)
        if return_reg_log:
            return z, reg_log
        return z

    def decode(self, z: torch.Tensor, **decoder_kwargs) -> torch.Tensor:

        if self.max_batch_size is None:
            dec = self.post_quant_conv(z)
            dec = self.decoder(dec, **decoder_kwargs)
        else:
            N = z.shape[0]
            bs = self.max_batch_size
            n_batches = int(math.ceil(N / bs))
            dec = list()
            for i_batch in range(n_batches):
                dec_batch = self.post_quant_conv(z[i_batch * bs : (i_batch + 1) * bs])
                dec_batch = self.decoder(dec_batch, **decoder_kwargs)
                dec.append(dec_batch)
            dec = torch.cat(dec, 0)

        return dec

    def forward(self, x: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (bs, 3, h, w)
        z, reg_log = self.encode(x, return_reg_log=True)
        # z: (bs, z_channels, h/8, w/8)
        dec = self.decode(z)
        # dec: (bs, 3, h, w)
        return z, dec, reg_log

    # See https://github.com/Lightning-AI/pytorch-lightning/issues/17801 and https://lightning.ai/docs/pytorch/stable/common/optimization.html for the reason of this change
    def training_step(self, batch, batch_idx) -> Any:
        x = self.get_input(batch)
        z, xrec, regularization_log = self(x)
        opt_g, opt_d = self.optimizers()

        # autoencode loss
        self.toggle_optimizer(opt_g)
        # adversarial loss is binary cross-entropy
        aeloss, log_dict_ae = self.loss(
            regularization_log,
            x,
            xrec,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )
        opt_g.zero_grad()
        self.manual_backward(aeloss)
        opt_g.step()
        self.untoggle_optimizer(opt_g)

        # discriminator loss
        self.toggle_optimizer(opt_d)
        # adversarial loss is binary cross-entropy
        discloss, log_dict_disc = self.loss(
            regularization_log,
            x,
            xrec,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )
        opt_d.zero_grad()
        self.manual_backward(discloss)
        opt_d.step()
        self.untoggle_optimizer(opt_d)

        # logging
        log_dict = {
            "train/aeloss": aeloss,
            "train/discloss": discloss,
        }
        log_dict.update(log_dict_ae)
        log_dict.update(log_dict_disc)
        self.log_dict(log_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        lr = opt_g.param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False, sync_dist=True)

    def on_validation_epoch_start(self):
        torch.distributed.barrier()
        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx) -> Dict:
        log_dict = self._validation_step(batch, batch_idx)
        if self.use_ema:
            with self.ema_scope():
                log_dict_ema = self._validation_step(batch, batch_idx, postfix="_ema")
                log_dict.update(log_dict_ema)
        return log_dict

    def _validation_step(self, batch, batch_idx, postfix="") -> Dict:
        x = self.get_input(batch)

        z, xrec, regularization_log = self(x)
        aeloss, log_dict_ae = self.loss(
            regularization_log,
            x,
            xrec,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val" + postfix,
        )

        discloss, log_dict_disc = self.loss(
            regularization_log,
            x,
            xrec,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val" + postfix,
        )
        self.log(f"val{postfix}/rec_loss", log_dict_ae[f"val{postfix}/rec_loss"])
        log_dict_ae.update(log_dict_disc)
        self.log_dict(log_dict_ae)

        # evaluate the psnr and ssim
        x = x.clamp(-1, 1)
        xrec = xrec.clamp(-1, 1)
        x, xrec = map(lambda x: (x + 1) / 2, (x, xrec))  # to [0, 1]
        psnr = compute_psnr(xrec, x)  # to 0-225, round to uint8, to 0-1, compute psnr, https://github.com/TencentARC/Open-MAGVIT2/blob/main/evaluation.py#L215
        ssim = compute_ssim(xrec, x)
        lpips = self.perceptual_loss(x*2-1, xrec*2-1).mean()
        self.log(f"val{postfix}/psnr", psnr, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"val{postfix}/ssim", ssim, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"val{postfix}/lpips", lpips, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        # kl-16 (https://github.com/CompVis/latent-diffusion?tab=readme-ov-file#model-zoo): psnr 24.08, rfid 0.87
        # kl-16 (128, bf16, train False, on epoch, 1gpu=4gpu): psnr 22.808
        # kl-16 (128, train False, on epoch): psnr 22.832
        # kl-16 (256, bf16, train False, on epoch): psnr 24.41
        # kl-16 (256, bf16, train False, on epoch): ssim 0.52
        # kl-16 (256, bf16, train False, on epoch): lpips 0.16

        # torchmetrics, input 0-1
        if postfix == "":
            self.metrics_psnr.update(xrec, x)
            self.metrics_ssim.update(xrec, x)
            self.metrics_fid.update((x * 255.).round().to(torch.uint8), real=True)
            self.metrics_fid.update((xrec * 255.).round().to(torch.uint8), real=False)
            self.metrics_lpips.update(x*2-1, xrec*2-1)
            self.metrics_is.update((xrec * 255.).round().to(torch.uint8))
            # self.log(f'val_torchmetrics{postfix}/psnr', self.metrics_psnr, on_step=True, on_epoch=True, sync_dist=True)
        elif self.use_ema and 'ema' in postfix:
            self.metrics_ema_psnr.update(xrec, x)
            self.metrics_ema_ssim.update(xrec, x)
            self.metrics_ema_fid.update((x * 255.).round().to(torch.uint8), real=True)
            self.metrics_ema_fid.update((xrec * 255.).round().to(torch.uint8), real=False)
            self.metrics_ema_lpips.update(x*2-1, xrec*2-1)
            self.metrics_ema_is.update((xrec * 255.).round().to(torch.uint8))
        else:
            raise NotImplementedError

        return log_dict_ae

    def on_validation_epoch_end(self):
        # https://lightning.ai/docs/torchmetrics/stable/pages/lightning.html#logging-torchmetrics
        self.log(f"val_epoch/psnr", self.metrics_psnr.compute(), prog_bar=True, logger=True, sync_dist=True)
        self.log(f"val_epoch/ssim", self.metrics_ssim.compute(), prog_bar=True, logger=True, sync_dist=True)
        self.log(f"val_epoch/fid", self.metrics_fid.compute(), prog_bar=True, logger=True, sync_dist=True)
        self.log(f"val_epoch/lpips", self.metrics_lpips.compute(), prog_bar=True, logger=True, sync_dist=True)
        self.log(f"val_epoch/is", self.metrics_is.compute()[0], prog_bar=True, logger=True, sync_dist=True)
        # kl-16 (256, bf16, train False, on epoch): val_epoch/psnr 22.63
        # kl-16 (256, bf16, train False, on epoch): val_epoch/ssim 0.70
        # kl-16 (256, bf16, train False, on epoch): val_epoch/lpips 0.16
        # kl-16 (256, bf16, train False, on epoch): val_epoch/is 210.0
        # kl-16 (256, bf16, train False, on epoch): val_epoch/fid 0.87
        if self.use_ema:
            self.log(f"val_epoch/psnr_ema", self.metrics_ema_psnr.compute(), prog_bar=True, logger=True, sync_dist=True)
            self.log(f"val_epoch/ssim_ema", self.metrics_ema_ssim.compute(), prog_bar=True, logger=True, sync_dist=True)
            self.log(f"val_epoch/fid_ema", self.metrics_ema_fid.compute(), prog_bar=True, logger=True, sync_dist=True)
            self.log(f"val_epoch/lpips_ema", self.metrics_ema_lpips.compute(), prog_bar=True, logger=True, sync_dist=True)
            self.log(f"val_epoch/is_ema", self.metrics_ema_is.compute()[0], prog_bar=True, logger=True, sync_dist=True)
        self.metrics_psnr.reset()
        self.metrics_ssim.reset()
        self.metrics_fid.reset()
        self.metrics_lpips.reset()
        self.metrics_is.reset()
        torch.distributed.barrier()
        torch.cuda.empty_cache()

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        print0(f"[bold magenta]\[tvae.models.mar_vae][AutoencodingLegacy][/bold magenta] loading >>> {cfg['target']} <<< optimizer from config")
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )

    def configure_optimizers(self) -> Any:
        ae_params = self.get_autoencoder_params()
        disc_params = self.get_discriminator_params()

        opt_ae = self.instantiate_optimizer_from_config(
            ae_params,
            default(self.lr_g_factor, 1.0) * self.learning_rate,
            self.optimizer_config,
        )
        opt_disc = self.instantiate_optimizer_from_config(
            disc_params, self.learning_rate, self.optimizer_config
        )

        return [opt_ae, opt_disc], []

    @torch.no_grad()
    def log_images(self, batch: Dict, **kwargs) -> Dict:
        log = dict()
        x = self.get_input(batch)
        _, xrec, _ = self(x)
        log["inputs"] = x
        log["recs"] = xrec
        if self.use_ema:
            with self.ema_scope():
                _, xrec_ema, _ = self(x)
                log["recs_ema"] = xrec_ema
            log["inputs_recs_recs_ema"] = torch.cat(
                [log["inputs"], log["recs"], log["recs_ema"]], dim=2
            )
        else:
            log["inputs_recs"] = torch.cat(
                [log["inputs"], log["recs"]], dim=2
            )
        return log
import math
import numpy as np
from rich import print
from packaging import version
from omegaconf import OmegaConf
from functools import partial, reduce
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    create_mask,
    create_block_mask,
    BlockMask,
    flex_attention,
)
if version.parse(torch.__version__) >= version.parse("2.0.0"):
    flex_attention = torch.compile(flex_attention)
    # https://github.com/pytorch/pytorch/issues/137481
    torch._dynamo.config.optimize_ddp = False

try:
    import xformers
    import xformers.ops
    IS_XFORMERS_AVAILABLE = True
except ImportError:
    IS_XFORMERS_AVAILABLE = False

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_func
    IS_FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    IS_FLASH_ATTENTION_AVAILABLE = False

from einops import rearrange

#from liger_kernel.transformers import (
#    LigerRMSNorm,
#    LigerLayerNorm,
#    liger_rotary_pos_emb,
#    LigerSwiGLUMLP,
#    LigerGEGLUMLP,
#)


from sgm.util import print0
from tvideo.mc.modules.attn_mask import (
    create_causal_mask_sdpa,
    create_block_causal_mask_sdpa,
    create_causal_mask_flex,
    create_block_causal_mask_flex,
    create_cycle_forcing_block_mask_sdpa,
    create_cycle_forcing_block_mask_flex,
)


#################################################################################
#                                Embedding Layers                               #
#################################################################################

def get_mask_mod(mask_mod: _mask_mod_signature, offset: torch.Tensor):
    def _mask_mod(b, h, q, kv):
        return mask_mod(b, h, q + offset, kv)

    return _mask_mod

class PatchEmbedder(nn.Module):
    """ DiT & HY
    3D Video to Patch Embedding
    """
    def __init__(
            self,
            patch_size: Union[int, Tuple[int, int, int]] = (1, 2, 2),
            in_channels: int = 3,
            hidden_size: int = 768,
            norm_layer: Optional[nn.Module] = None,
            flatten: bool = True,
            bias: bool = True,
    ):
        super().__init__()

        self.flatten = flatten

        self.proj = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
        )
        self.norm = norm_layer(hidden_size) if norm_layer else nn.Identity()

        # initialization like nn.Linear (instead of nn.Conv2d):
        nn.init.xavier_uniform_(self.proj.weight.view(self.proj.weight.size(0), -1))
        if bias:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.proj(x)
        # x: (B, C, T', H', W')
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        # x: (B, T' * H' * W', C)
        x = self.norm(x)
        return x.contiguous()


class TimestepEmbedder(nn.Module):
    """DiT & HY
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

        # initialize timestep embedding MLP
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype=None):
        t_shape = t.shape
        if len(t_shape) > 1:
            t = t.flatten()

        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if dtype is not None:
            t_freq = t_freq.to(dtype)
        t_emb = self.mlp(t_freq)

        if len(t_shape) > 1:
            t_emb = t_emb.unflatten(0, t_shape)
        return t_emb


class LabelEmbedder(nn.Module):
    """DiT
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

        # initialize label embedding table
        nn.init.normal_(self.embedding_table.weight, std=0.02)

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class MLPEmbedder(nn.Module):
    """HY
    copied from https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/layers.py
    """
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )

        # initialization
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


#################################################################################
#                          Positional Embedding Functions                       #
#################################################################################


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    theta_rescale_factor: float = 1.0,
    interpolation_factor: float = 1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,  #  torch.float32, torch.float64 (flux)
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """HY & https://github.dev/huggingface/diffusers/blob/main/src/diffusers/models/embeddings.py
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        linear_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the context extrapolation. Defaults to 1.0.
        ntk_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the NTK-Aware RoPE. Defaults to 1.0.
        repeat_interleave_real (`bool`, *optional*, defaults to `True`):
            If `True` and `use_real`, real part and imaginary part are each interleaved with themselves to reach `dim`.
            Otherwise, they are concateanted with themselves.
        freqs_dtype (`torch.float32` or `torch.float64`, *optional*, defaults to `torch.float32`):
            the dtype of the frequency tensor.
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    theta = theta * ntk_factor

    # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    if theta_rescale_factor != 1.0:
        theta *= theta_rescale_factor ** (dim / (dim - 2))

    freqs = (
        1.0
        / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device)[: (dim // 2)] / dim))
        / linear_factor
    )  # [D/2]
    freqs = torch.outer(pos * interpolation_factor, freqs)  # type: ignore   # [S, D/2]

    if use_real and repeat_interleave_real:
        # flux, hunyuan-dit, cogvideox
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, D]
        return freqs_cos, freqs_sin
    elif use_real:
        # stable audio, allegro
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # lumina
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis

def reshape_for_broadcast(
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    x: torch.Tensor,
    head_first=False,
):
    """ HY
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Notes:
        When using FlashMHAModified, head_first should be False.
        When using Attention, head_first should be True.

    Args:
        freqs_cis (Union[torch.Tensor, Tuple[torch.Tensor]]): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.
        head_first (bool): head dimension first (except batch dim) or not.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim

    if isinstance(freqs_cis, tuple):
        # freqs_cis: (cos, sin) in real space
        if head_first:
            assert freqs_cis[0].shape == (
                x.shape[-2],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
            shape = [
                d if i == ndim - 2 or i == ndim - 1 else 1
                for i, d in enumerate(x.shape)
            ]
        else:
            assert freqs_cis[0].shape == (
                x.shape[1],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis[0].view(*shape), freqs_cis[1].view(*shape)
    else:
        # freqs_cis: values in complex space
        if head_first:
            assert freqs_cis.shape == (
                x.shape[-2],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
            shape = [
                d if i == ndim - 2 or i == ndim - 1 else 1
                for i, d in enumerate(x.shape)
            ]
        else:
            assert freqs_cis.shape == (
                x.shape[1],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)


def rotate_half(x):
    """ HY
    """
    x_real, x_imag = (
        x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    )  # [B, S, H, D//2]
    return torch.stack([-x_imag, x_real], dim=-1).flatten(3)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    head_first: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ HY
    https://github.dev/huggingface/diffusers/blob/main/src/diffusers/models/embeddings.py also have apply_rotary_emb, but accept [B, H, L, D]
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings. [B, S, H, D]
        xk (torch.Tensor): Key tensor to apply rotary embeddings.   [B, S, H, D]
        freqs_cis (torch.Tensor or tuple): Precomputed frequency tensor for complex exponential.
        head_first (bool): head dimension first (except batch dim) or not.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

    """
    xk_out = None
    if isinstance(freqs_cis, tuple):
        #print("freqs_cis is tuple")
        # freqs_cis: (118800, 128), (118800, 128)
        cos, sin = reshape_for_broadcast(freqs_cis, xq, head_first)  # [S, D]
        # cos/sin: (1, 118800, 1, 128)
        cos, sin = cos.to(xq.device), sin.to(xq.device)
        #print(f'cos.shape: {cos.shape}, sin.shape: {sin.shape}')
        # real * cos - imag * sin
        # imag * cos + real * sin
        xq_out = (xq.float() * cos + rotate_half(xq.float()) * sin).type_as(xq)
        xk_out = (xk.float() * cos + rotate_half(xk.float()) * sin).type_as(xk)
    else:
        print("freqs_cis is tensor")
        # view_as_complex will pack [..., D/2, 2](real) to [..., D/2](complex)
        xq_ = torch.view_as_complex(
            xq.float().reshape(*xq.shape[:-1], -1, 2)
        )  # [B, S, H, D//2]
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_, head_first).to(
            xq.device
        )  # [S, D//2] --> [1, S, 1, D//2]
        # (real, imag) * (cos, sin) = (real * cos - imag * sin, imag * cos + real * sin)
        # view_as_real will expand [..., D/2](complex) to [..., D/2, 2](real)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq)
        xk_ = torch.view_as_complex(
            xk.float().reshape(*xk.shape[:-1], -1, 2)
        )  # [B, S, H, D//2]
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)

    return xq_out, xk_out


def get_meshgrid_nd(start, *args, dim=2):
    """ HY
    Get n-D meshgrid with start, stop and num.

    Args:
        start (int or tuple): If len(args) == 0, start is num; If len(args) == 1, start is start, args[0] is stop,
            step is 1; If len(args) == 2, start is start, args[0] is stop, args[1] is num. For n-dim, start/stop/num
            should be int or n-tuple. If n-tuple is provided, the meshgrid will be stacked following the dim order in
            n-tuples.
        *args: See above.
        dim (int): Dimension of the meshgrid. Defaults to 2.

    Returns:
        grid (np.ndarray): [dim, ...]
    """
    def _to_tuple(x, dim=2):
        if isinstance(x, int):
            return (x,) * dim
        elif len(x) == dim:
            return x
        else:
            raise ValueError(f"Expected length {dim} or int, but got {x}")

    if len(args) == 0:
        # start is grid_size
        num = _to_tuple(start, dim=dim)
        start = (0,) * dim
        stop = num
    elif len(args) == 1:
        # start is start, args[0] is stop, step is 1
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = [stop[i] - start[i] for i in range(dim)]
    elif len(args) == 2:
        # start is start, args[0] is stop, args[1] is num
        start = _to_tuple(start, dim=dim)  # Left-Top       eg: 12,0
        stop = _to_tuple(args[0], dim=dim)  # Right-Bottom   eg: 20,32
        num = _to_tuple(args[1], dim=dim)  # Target Size    eg: 32,124
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

    # PyTorch implement of np.linspace(start[i], stop[i], num[i], endpoint=False)
    axis_grid = []
    for i in range(dim):
        a, b, n = start[i], stop[i], num[i]
        g = torch.linspace(a, b, n + 1, dtype=torch.float32)[:n]
        axis_grid.append(g)
    grid = torch.meshgrid(*axis_grid, indexing="ij")  # dim x [W, H, D]
    grid = torch.stack(grid, dim=0)  # [dim, W, H, D]

    return grid


def get_nd_rotary_pos_embed(
    rope_dim_list,
    start,
    *args,
    theta=10000.0,
    use_real=False,
    theta_rescale_factor: Union[float, List[float]] = 1.0,
    interpolation_factor: Union[float, List[float]] = 1.0,
):
    """ HY
    This is a n-d version of precompute_freqs_cis, which is a RoPE for tokens with n-d structure.

    Args:
        rope_dim_list (list of int): Dimension of each rope. len(rope_dim_list) should equal to n.
            sum(rope_dim_list) should equal to head_dim of attention layer.
        start (int | tuple of int | list of int): If len(args) == 0, start is num; If len(args) == 1, start is start,
            args[0] is stop, step is 1; If len(args) == 2, start is start, args[0] is stop, args[1] is num.
        *args: See above.
        theta (float): Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (bool): If True, return real part and imaginary part separately. Otherwise, return complex numbers.
            Some libraries such as TensorRT does not support complex64 data type. So it is useful to provide a real
            part and an imaginary part separately.
        theta_rescale_factor (float): Rescale factor for theta. Defaults to 1.0.

    Returns:
        pos_embed (torch.Tensor): [HW, D/2]
    """

    grid = get_meshgrid_nd(
        start, *args, dim=len(rope_dim_list)
    )  # [3, W, H, D] / [2, W, H]
    # rope_dim_list: [16, 56, 56]
    # grid: (3, 33, 45, 80)
    # 33*45*80 = 118800

    if isinstance(theta_rescale_factor, int) or isinstance(theta_rescale_factor, float):
        theta_rescale_factor = [theta_rescale_factor] * len(rope_dim_list)
    elif isinstance(theta_rescale_factor, list) and len(theta_rescale_factor) == 1:
        theta_rescale_factor = [theta_rescale_factor[0]] * len(rope_dim_list)
    assert len(theta_rescale_factor) == len(
        rope_dim_list
    ), "len(theta_rescale_factor) should equal to len(rope_dim_list)"

    if isinstance(interpolation_factor, int) or isinstance(interpolation_factor, float):
        interpolation_factor = [interpolation_factor] * len(rope_dim_list)
    elif isinstance(interpolation_factor, list) and len(interpolation_factor) == 1:
        interpolation_factor = [interpolation_factor[0]] * len(rope_dim_list)
    assert len(interpolation_factor) == len(
        rope_dim_list
    ), "len(interpolation_factor) should equal to len(rope_dim_list)"

    # use 1/ndim of dimensions to encode grid_axis
    embs = []
    for i in range(len(rope_dim_list)):
        emb = get_1d_rotary_pos_embed(
            rope_dim_list[i],
            grid[i].reshape(-1),
            theta,
            use_real=use_real,
            theta_rescale_factor=theta_rescale_factor[i],
            interpolation_factor=interpolation_factor[i],
        )  # 2 x [WHD, rope_dim_list[i]]
        # 0: (118800, 16)
        # 1: (118800, 56)
        # 2: (118800, 56)
        embs.append(emb)

    if use_real:
        cos = torch.cat([emb[0] for emb in embs], dim=1)  # (WHD, D/2)
        sin = torch.cat([emb[1] for emb in embs], dim=1)  # (WHD, D/2)
        return cos, sin
    else:
        emb = torch.cat(embs, dim=1)  # (WHD, D/2)
        return emb


def apply_rope_scaling(freqs: torch.Tensor, rope_scaling: Optional[dict] = None):
    """gpt-fast
    """
    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000,
    dtype: torch.dtype = torch.bfloat16,
    rope_scaling: Optional[dict] = None,
) -> Tensor:
    """gpt-fast
    """
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    if rope_scaling is not None:
        freqs = apply_rope_scaling(freqs, rope_scaling)
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)



#################################################################################
#                                  Transformer                                  #
#################################################################################


def find_multiple(n: int, k: int) -> int:
    """ gpt-fast
    """
    if n % k == 0:
        return n
    return n + k - (n % k)


def repeat_kv(x: torch.Tensor, n_rep: int, dim: int) -> torch.Tensor:
    """ lingua
    torch.repeat_interleave(x, dim=2, repeats=n_rep)
    """
    assert dim == 2, "Only dim=2 is supported. Check the implementation for other dims."
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def modulate(x, shift, scale):
    # fused modulate in opendit is ~2.4x slower than the original implementation
    return x * (1 + scale) + shift


class Attention(nn.Module):
    """gpt-fast & lingua & DiT
    A multi-head attention layer with rotary embeddings and KV cache.
    Args:
        dim: The number of features in the
            input and output tensors.
        head_dim: The number of features per head.
        n_head: The number of heads.
        n_local_heads: The number of local heads.
        kv_cache: A KVCache object.
    The default values are used in llama-3.1-8b.
    """
    def __init__(self,
                 dim: int = 4096,
                 n_head: int = 32,
                 n_local_heads: int = 8,
                 head_dim: int = 64,
                 attn_dropout: float = 0.0,
                 wo_dropout: float = 0.0,
                 sdpa_backends: List[str] = ["flash_attention", "math", "cudnn_attention", "efficient_attention"],
                 qk_norm: bool = False,
    ):
        super().__init__()

        assert dim % n_head == 0, f"dim should be divisible by n_head, but got dim: {dim} and n_head: {n_head}"
        assert n_head % n_local_heads == 0, f"n_head should be divisible by n_local_heads, but got n_head: {n_head} and n_local_heads: {n_local_heads}"

        total_head_dim = (n_head + 2 * n_local_heads) * head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(dim, total_head_dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.kv_cache = None

        self.attn_drop = nn.Dropout(attn_dropout)
        self.wo_drop = nn.Dropout(wo_dropout)

        self.dim = dim
        self.n_head = n_head
        self.head_dim = head_dim
        self.n_local_heads = n_local_heads
        self.heads_per_group = n_head // n_local_heads

        if qk_norm:
            self.q_norm = RMSNorm(self.dim, eps = 1e-6)
            self.k_norm = RMSNorm(self.n_local_heads * self.head_dim, eps = 1e-6)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        # backends for scaled_dot_product_attention
        self.sdpa_backends = []
        if "flash_attention" in sdpa_backends:
            self.sdpa_backends.append(SDPBackend.FLASH_ATTENTION)
        if "math" in sdpa_backends:
            self.sdpa_backends.append(SDPBackend.MATH)
        if "cudnn_attention" in sdpa_backends:
            self.sdpa_backends.append(SDPBackend.CUDNN_ATTENTION)
        if "efficient_attention" in sdpa_backends:
            self.sdpa_backends.append(SDPBackend.EFFICIENT_ATTENTION)
        if len(self.sdpa_backends) == 0:
            raise ValueError(f"at least one backend should be provided, but got {sdpa_backends}")

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        def _lingua_init(module):
            std = self.dim ** (-0.5)
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=std, a=-3*std, b=3*std)
        self.apply(_basic_init)

    def forward(self,
                x: Tensor,
                freqs_cis: Tensor,
                attn_mask: Optional[Union[BlockMask, Tensor, str]],
                input_pos: Optional[Tensor] = None,
                attn_impl: str = "flex",
    ) -> Tensor:
        """
        x: (B, L, D) - GT frames only
        rollout_x: (B, L, D) - Rollout frames (if provided)

        When rollout_x is provided:
            Q: from x (GT frames)
            KV: from [rollout_x, x] (Rollout + GT frames)
        """
        bsz, seqlen, _ = x.shape
        kv_size = self.n_local_heads * self.head_dim

        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)
        q = self.q_norm(q).view(bsz, seqlen, self.n_head, self.head_dim)
        k = self.k_norm(k).view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        q, k = apply_rotary_emb(q, k, freqs_cis, head_first=False)
        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        # This condition helps us be easily compatible
        # with inference by adding a pluggable KVCache
        if self.kv_cache is not None:
            if input_pos is not None:
                cached_k, cached_v = self.kv_cache.update(input_pos, k, v)
                # we dont change k, v value
            else:
                cached_k, cached_v = self.kv_cache.update(input_pos, k, v)
                k,v = torch.cat([cached_k, k], dim=2), torch.cat([cached_v, v], dim=2)

        if not self.training:
            v = v.to(torch.float16)
            q = q.to(torch.float16)
            k = k.to(torch.float16)

        if attn_impl == "sdpa":
            assert attn_mask is None or isinstance(attn_mask, (str, torch.Tensor)), f"attn_mask should be None or str or Tensor, but got {type(attn_mask)}"
            is_causal = (attn_mask == "causal") if isinstance(attn_mask, str) else False
            attn_mask = attn_mask if isinstance(attn_mask, torch.Tensor) else None

            with sdpa_kernel(backends=self.sdpa_backends):
                y = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    enable_gqa=(self.n_head != self.n_local_heads),
                    is_causal=is_causal,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                )
        elif attn_impl == "vanilla":
            # original DiT legacy attention
            attn_bias = torch.zeros(seqlen, seqlen, device=q.device)
            is_causal = (attn_mask == "causal") if isinstance(attn_mask, str) else False
            if is_causal:
                attn_mask = torch.ones(seqlen, seqlen, dtype=torch.bool).tril(diagonal=0)
            else:
                assert attn_mask.dtype == torch.bool, f"attn_mask should be bool, but got {attn_mask.dtype}"
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))

            if self.n_head != self.n_local_heads:
                k = k.repeat_interleave(q.size(-3)//k.size(-3), -3)
                v = v.repeat_interleave(q.size(-3)//v.size(-3), -3)

            attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            attn += attn_bias
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            y = attn @ v
        else:
            raise NotImplementedError(f"attn_impl should be 'sdpa', but got {attn_impl}")

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        y = self.wo(y)
        y = self.wo_drop(y)
        return y


class KVCache(nn.Module):
    """gpt-fast
    """
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)

        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))
        self.ret_len = 0

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]

        if not isinstance(input_pos, torch.Tensor) and input_pos is not None:
            k_out = self.k_cache
            v_out = self.v_cache
            k_out[:, :, :input_pos] = k_val[:, :, :input_pos]
            v_out[:, :, :input_pos] = v_val[:, :, :input_pos]
            self.ret_len = input_pos
            return k_out[:, :, :input_pos], v_out[:, :, :input_pos]
        elif input_pos is None:
            return self.k_cache[:, :, :self.ret_len], self.v_cache[:, :, :self.ret_len]
        else:
            k_out = self.k_cache
            v_out = self.v_cache
            k_out[:, :, input_pos] = k_val
            v_out[:, :, input_pos] = v_val

            return k_out, v_out


class FeedForward(nn.Module):
    """gpt-fast & lingua
    The default values are used in llama-3.1-8b.
    """
    def __init__(self,
                 dim: int = 4096,
                 intermediate_size: int = 14336,
                 activation_fn: str = "silu",
                 dropout : float = 0.0,
    ):
        super().__init__()
        self.w1 = nn.Linear(dim, intermediate_size, bias=False)
        self.w3 = nn.Linear(dim, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, dim, bias=False)
        if activation_fn == "silu":  # gpt-fast
            self.act = F.silu
        elif activation_fn == "gelu":
            self.act = F.gelu
        elif activation_fn == "approx_gelu":  # DiT
            self.act = lambda x: F.gelu(x, approximate="tanh")
        else:
            raise NotImplementedError(f"activation_fn should be 'silu' or 'gelu', but got {activation_fn}")
        # self.drop1 = nn.Dropout(dropout)
        # self.drop2 = nn.Dropout(dropout)

        # initialization like nn.Linear (instead of nn.Conv2d):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        def _lingua_init(module):
            std = self.dim ** (-0.5)
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=std, a=-3*std, b=3*std)
        self.apply(_basic_init)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.act(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    """gpt-fast & lingua
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

        # _lingua_init
        nn.init.ones_(self.weight)  # type: ignore

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x).half()

        return output * self.weight


class FinalLayer(nn.Module):
    """DiT & HY
    The final layer of DiT.
    """
    def __init__(self,
                 hidden_size: int,
                 patch_size: Union[int, Tuple[int, int, int]],
                 out_channels: int,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if isinstance(patch_size, int):
            self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        else:
            self.linear = nn.Linear(hidden_size, reduce(lambda x, y: x * y, patch_size) * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

        # zero-out output layers
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class TransformerBlock(nn.Module):
    """gpt-fast & lingua & DiT & Oasis
    The default values are used in llama-3.1-8b.
    """
    def __init__(self,
                 # attention
                 dim: int = 4096,
                 n_head: int = 32,
                 n_local_heads: int = 8,
                 head_dim: int = 64,
                 sdpa_backends: List[str] = ["flash_attention", "math", "cudnn_attention", "efficient_attention"],
                 # feed forward
                 intermediate_size: int = 14336,
                 activation_fn: str = "silu",
                 # RMSNorm
                 eps: float = 1e-6,
                 # efficiency
                 enable_liger: bool = False,
                 enable_gradient_checkpointing: bool = False,

                 qk_norm: bool = False,
    ):
        super().__init__()

        assert dim % n_head == 0, f"dim should be divisible by n_head, but got dim: {dim} and n_head: {n_head}"
        assert n_head % n_local_heads == 0, f"n_head should be divisible by n_local_heads, but got n_head: {n_head} and n_local_heads: {n_local_heads}"

        # attention
        self.attention = Attention(
            dim=dim,
            n_head=n_head,
            n_local_heads=n_local_heads,
            head_dim=head_dim,
            sdpa_backends=sdpa_backends,
            qk_norm=qk_norm,
        )

        # feed_forward
        if enable_liger:
            config = OmegaConf.create({
                "hidden_size": dim,
                "intermediate_size": intermediate_size,
                "hidden_act": activation_fn,
            })
            self.feed_forward = LigerSwiGLUMLP(config)
        else:
            self.feed_forward = FeedForward(
                dim=dim,
                intermediate_size=intermediate_size,
                activation_fn=activation_fn,
            )

        # attention_norm
        if enable_liger:
            self.attention_norm = LigerRMSNorm(
                hidden_size=dim,
                eps=eps,
            )
        else:
            self.attention_norm = RMSNorm(
                dim=dim,
                eps=eps,
            )

        # ffn_norm
        if enable_liger:
            self.ffn_norm = LigerRMSNorm(
                hidden_size=dim,
                eps=eps,
            )
        else:
            self.ffn_norm = RMSNorm(
                dim=dim,
                eps=eps,
            )

        # conditioning
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        # zero-out adaLN modulation layers in DiT blocks
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

        self.enable_gradient_checkpointing = enable_gradient_checkpointing

    def forward(
            self,
            x: Tensor,
            c_adaln: Tensor,
            freqs_cis: Optional[Tensor] = None,
            attn_mask: Optional[Union[BlockMask, Tensor, str]] = None,
            input_pos: Optional[Tensor] = None,  # for KV cache
            attn_impl: str = "sdpa",
    ):
        if self.enable_gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(self._forward, x, c_adaln, freqs_cis, attn_mask, input_pos, attn_impl)
        else:
            return self._forward(x, c_adaln, freqs_cis, attn_mask, input_pos, attn_impl)


    def _forward(
            self,
            x: Tensor,
            c_adaln: Tensor,
            freqs_cis: Optional[Tensor] = None,
            attn_mask: Optional[Union[BlockMask, Tensor, str]] = None,
            input_pos: Optional[Tensor] = None,  # for KV cache
            attn_impl: str = "sdpa",
    ) -> Tensor:
        # x:       (B, T'*H'*W', D)
        # c_adaln: (B, T'*H'*W', D)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c_adaln).chunk(6, dim=-1)

        x = x + gate_msa * self.attention(
            x=modulate(self.attention_norm(x), shift_msa, scale_msa),
            freqs_cis=freqs_cis,
            attn_mask=attn_mask,
            input_pos=input_pos,
            attn_impl=attn_impl,
        )

        x = x + gate_mlp * self.feed_forward(
            x=modulate(self.ffn_norm(x), shift_mlp, scale_mlp),
        )

        return x


class Transformer(nn.Module):
    """
    """
    def __init__(self,
                 patch_size: tuple[int, int, int] = (1, 2, 2),  # (T, H, W)
                 in_channels: int = 16,  # VAE latent channels
                 out_channels: int = 16,  # 32 if learn_sigma else 16
                 # attention
                 dim: int = 4096,
                 n_layer: int = 32,
                 n_head: int = 32,
                 n_local_heads: int = 8,
                 head_dim: Optional[int] = None,  # 4096 // 32 = 128
                 attn_mask_type: str = 'block',  # block, causal
                 attn_impl: str = 'sdpa',  # flex, sdpa, vanilla
                 sdpa_backends: List[str] = ["flash_attention", "math", "cudnn_attention", "efficient_attention"],
                 # feed forward
                 intermediate_size: Optional[int] = None,  # 14336
                 # conditioning
                 c_dim: int = 6,
                 # positional embedding
                 pe_type: str = 'rope_hy',
                 input_h: int = 14,
                 input_w: int = 24,
                 input_t: int = 32,
                 rope_dim: Optional[list[int]] = [16, 56, 56],  # hy
                 # efficiency
                 enable_liger: bool = True,
                 enable_gradient_checkpointing: bool = False,
                 timestep_norm_scale_factor = 1.0,
                 qk_norm: bool = False,
                 use_plucker: bool = True,
                 enable_cycle_forcing: bool = False,
    ):
        super().__init__()
        self.print_prefix = "[bold magenta]\[tvideo.mc.modules.transformer_cache][Transformer][/bold magenta]"

        self.use_plucker = use_plucker
        self.enable_cycle_forcing = enable_cycle_forcing

        # parameters
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.dim = dim
        self.n_head = n_head
        self.n_local_heads = n_head if n_local_heads == -1 else n_local_heads
        self.head_dim = head_dim or dim // n_head
        if intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)

        # layers
        self.x_embedder = PatchEmbedder(patch_size, in_channels, dim, bias=True)
        self.t_embedder = TimestepEmbedder(dim)
        if self.use_plucker:
            c_dim = 6
            self.c_embedder = PatchEmbedder(patch_size, c_dim, dim, bias=True)
        else:
            c_dim = 16
            self.c_embedder = MLPEmbedder(c_dim, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                n_head=n_head,
                n_local_heads=self.n_local_heads,
                head_dim=self.head_dim,
                sdpa_backends=sdpa_backends,
                intermediate_size=self.intermediate_size,
                enable_liger=enable_liger,
                enable_gradient_checkpointing=enable_gradient_checkpointing,
                qk_norm=qk_norm,
            ) for _ in range(n_layer)
        ])
        self.final_layer = FinalLayer(dim, patch_size, out_channels)

        # precompute number of patches
        nT, nH, nW = input_t // self.patch_size[0], input_h // self.patch_size[1], input_w // self.patch_size[2]

        # attention mask
        self.attn_impl = attn_impl
        self.setup_attn_mask(
            attn_mask_type,
            nT=nT,
            nH=nH,
            nW=nW,
        )

        # positional embedding
        self.pe_type = pe_type
        self.setup_pe(
            pe_type,
            dim=dim,
            nH=nH,
            nW=nW,
            nT=nT,
            rope_dim=rope_dim,
            head_dim=self.head_dim,
        )

        print0(f"{self.print_prefix} Initializing Transformer with {n_layer} layers, {n_head} heads, {dim} dim, "
               f"{nT * nH * nW} lengths, {nT} temporal lengths, {nH} height lengths, {nW} width lengths, "
               f"{self.intermediate_size} intermediate size, {self.n_local_heads} local heads, {self.head_dim} head dim, "
               f"{attn_mask_type} attention mask, {attn_impl} attention impl, {pe_type} positional embedding")

        self.logvar_linear = None
        self.timestep_norm_scale_factor = timestep_norm_scale_factor


    def setup_attn_mask(self, attn_mask_type: str, nT: int, nH: int, nW: int):
        #print(f'attn mask type: {attn_mask_type}')
        assert attn_mask_type in ['block', 'causal'], f"attn_mask_type should be 'block' or 'causal', but got {attn_mask_type}"
        if attn_mask_type == 'causal':
            self.register_buffer('attn_mask_vanilla', create_causal_mask_sdpa(nT * nH * nW), persistent=False)
            self.register_buffer('attn_mask_sdpa', create_causal_mask_sdpa(nT * nH * nW), persistent=False)
            self.attn_mask_flex = create_block_mask(
                create_causal_mask_flex,
                B=None,
                H=None,
                Q_LEN=nT * nH * nW,
                KV_LEN=nT * nH * nW,
            )
        elif attn_mask_type == 'block':
            block_size = nH * nW
            self.register_buffer('attn_mask_vanilla', create_block_causal_mask_sdpa(seq_len=nT*nH*nW, block_size=block_size), persistent=False)
            self.register_buffer('attn_mask_sdpa', create_block_causal_mask_sdpa(seq_len=nT*nH*nW, block_size=block_size), persistent=False)
            create_mask_fn = create_block_causal_mask_flex(block_size=block_size)
            self.attn_mask_flex = create_block_mask(
                create_mask_fn,
                B=None,
                H=None,
                Q_LEN=nT * nH * nW,
                KV_LEN=nT * nH * nW,
            )
        if self.enable_cycle_forcing:
            block_size = nH * nW
            for p in (2, 4, 8, 16, 32):
                self.register_buffer(
                    f'attn_mask_cycle_forcing_sdpa_p{p}',
                    create_cycle_forcing_block_mask_sdpa(nT, block_size, p),
                    persistent=False,
                )


    def setup_pe(self,
                 pe_type,
                 dim: int = 4096,
                 nH: int = 7,
                 nW: int = 12,
                 nT: int = 32,
                 rope_dim: Optional[list[int]] = [16, 56, 56],
                 head_dim: int = 128,
    ):

        if pe_type == 'learned':
            self.pos_embed = nn.Parameter(torch.zeros(1, nH * nW * nT, dim))

        elif pe_type == 'rope_hy':
            if rope_dim is None:
                rope_dim = [head_dim // 3] * 3
            assert sum(rope_dim) == head_dim, f"sum(rope_dim) should equal to head_dim, but got rope_dim: {rope_dim} and head_dim: {head_dim}"
            freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
                rope_dim,
                [nT, nH, nW],
                theta=256,
                use_real=True,
                theta_rescale_factor=1,
            )
            # freqs_cos: (nT * nH * nW, head_dim)
            self.freqs_cis = (freqs_cos, freqs_sin)  # Tuple[torch.Tensor, torch.Tensor]

        else:
            raise NotImplementedError(f"Positional embedding type {pe_type} is not implemented.")

    def setup_cache(self, max_batch_size, max_seq_length, dtype=torch.bfloat16):
        for b in self.blocks:
            b.attention.kv_cache = KVCache(
                max_batch_size,
                max_seq_length,
                n_heads=self.n_local_heads,
                head_dim=self.head_dim,
                dtype=dtype,
        )

    def unpatchify(self, x, nt, nh, nw):
        """
        x: (N, T'*H'*W', 1*patch_size**2 * C)
        imgs: (B, C, T, H, W)
        """
        c = self.out_channels
        pt, ph, pw = self.patch_size
        assert nt * nh * nw == x.shape[1]

        x = x.reshape(shape=(x.shape[0], nt, nh, nw, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, nt * pt, nh * ph, nw * pw))
        return imgs

    def forward(self,
                x: Tensor,
                t: Tensor,
                act: Optional[Tensor] = None, # act is now (B, T, 6, H, W)
                p: Optional[int] = 2,
                input_pos: Optional[Tensor] = None,  # for KV cache
                attn_impl: Optional[str] = None,  # 'sdpa' / 'flex' / 'vanilla'
    ):
        """
        x:    (B, T, C, H, W)
        t:    (B, T)
        act:  (B, T, D, H, W)
        rollout_vid:  (B, T, C, H, W) - Rollout frames used by cycle forcing
        """
        B, T, C, H, W = x.shape
        assert C == self.in_channels, f"Expected {self.in_channels} channels, but got {C}."
        nT, nH, nW = T // self.patch_size[0], H // self.patch_size[1], W // self.patch_size[2]
        attn_impl = attn_impl or self.attn_impl

        # 3d patchify
        x = rearrange(x, 'b t c h w -> b c t h w').contiguous()
        x = self.x_embedder(x)

        if self.training and self.enable_cycle_forcing:
            attn_mask = getattr(self, f'attn_mask_cycle_forcing_{attn_impl}_p{p}')
        else:
            attn_mask = getattr(self, f'attn_mask_{attn_impl}')
            if (T != 1 and  input_pos is not None) or self.blocks[0].attention.kv_cache is None:
                if attn_impl == 'sdpa':
                    attn_mask = attn_mask[:x.shape[1], :x.shape[1]] if isinstance(attn_mask, torch.Tensor) else attn_mask # attn_mask管计算
                else:
                    raise NotImplementedError(f"attn_impl should be 'sdpa' or 'flex', but got {attn_impl}")
            else:
                cached_len = getattr(self.blocks[0].attention, 'kv_cache', None).ret_len
                if attn_impl == 'sdpa':
                    attn_mask = attn_mask[cached_len:cached_len + x.shape[1], :cached_len + x.shape[1]] if isinstance(attn_mask, torch.Tensor) else attn_mask
                else:
                    raise NotImplementedError(f"attn_impl should be 'sdpa' or 'flex', but got {attn_impl}")

        # verify number of patches
        n_patch_per_frame = nH * nW
        assert x.shape[1] == n_patch_per_frame * nT, \
            f"Expected {n_patch_per_frame * nT} patches, but got {x.shape[1]}."

        if self.timestep_norm_scale_factor != 1.0:
            t = (t.float() / self.timestep_norm_scale_factor).to(torch.float32)
        else:
            t = t.long().to(torch.float32)

        # embed noise level
        # t: (B, T)
        c = self.t_embedder(t, dtype=x.dtype)
        # add positional embedding
        if self.use_plucker:
            c = c.repeat_interleave(n_patch_per_frame, dim=1) # c: (B, T'*H'*W', D)
            if act is not None:
                act = rearrange(act, 'b t c h w -> b c t h w').contiguous()
                c_act = self.c_embedder(act)
                c = c + c_act
        else: #use Rt
            if act is not None:
                c += self.c_embedder(act)
            c = c.repeat_interleave(n_patch_per_frame, dim=1)

        freqs_cis = None
        if self.pe_type == 'sincos' or self.pe_type == 'learned':
            x = x + self.pos_embed
        elif 'rope' in self.pe_type:
            if self.enable_cycle_forcing and self.training:
                if isinstance(self.freqs_cis, tuple):
                    freqs_cos_orig = self.freqs_cis[0].to(x.device)  # [seq_len, dim]
                    freqs_sin_orig = self.freqs_cis[1].to(x.device)  # [seq_len, dim]
                    seq_len, dim = freqs_cos_orig.shape
                    pe_T = seq_len // n_patch_per_frame
                    repeats = pe_T // p
                    last_p_cos = freqs_cos_orig[-p * n_patch_per_frame:].view(p, n_patch_per_frame, dim)
                    last_p_sin = freqs_sin_orig[-p * n_patch_per_frame:].view(p, n_patch_per_frame, dim)
                    ext_cos = last_p_cos.repeat_interleave(repeats, dim=0).reshape(-1, dim)
                    ext_sin = last_p_sin.repeat_interleave(repeats, dim=0).reshape(-1, dim)
                    freqs_cos = torch.cat([freqs_cos_orig, ext_cos], dim=0)
                    freqs_sin = torch.cat([freqs_sin_orig, ext_sin], dim=0)
                    freqs_cis = (freqs_cos, freqs_sin)
            else:
                if (T!=1 and input_pos is not None) or self.blocks[0].attention.kv_cache is None or self.training:
                    freqs_cis = (self.freqs_cis[0][:x.shape[1]].to(x.device), self.freqs_cis[1][:x.shape[1]].to(x.device)) if isinstance(self.freqs_cis, tuple) \
                    else self.freqs_cis[:x.shape[1]].to(x.device)
                else:
                    cached_len = getattr(self.blocks[0].attention, 'kv_cache', None).ret_len
                    freqs_cis = (self.freqs_cis[0][cached_len:cached_len + x.shape[1]].to(x.device), self.freqs_cis[1][cached_len:cached_len + x.shape[1]].to(x.device)) if isinstance(self.freqs_cis, tuple) \
                    else self.freqs_cis[cached_len:cached_len + x.shape[1]].to(x.device)
                    # freqs_cis: (nT * nH * nW, head_dim)

        for block in self.blocks:
            x = block(
                x=x,  # x: (B, T'*H'*W', D)
                c_adaln=c,  # c: (B, T'*H'*W', D)
                freqs_cis=freqs_cis,  # freqs_cis: (nT * nH * nW, head_dim)
                attn_mask=attn_mask,
                input_pos=input_pos,
                attn_impl=attn_impl,
            )
        x = self.final_layer(x, c)

        x = self.unpatchify(x, nT, nH, nW)
        x = rearrange(x, 'b c t h w -> b t c h w').contiguous()
        return x

    def flush_cache(self):
        for block in self.blocks:
            if hasattr(block.attention, 'kv_cache'):
                block.attention.kv_cache.ret_len = 0

    def clear_cache(self):
        for block in self.blocks:
            if hasattr(block.attention, 'kv_cache'):
                block.attention.kv_cache = None

class TrigFlowWrapper(Transformer):

    def forward(self,
                x: Tensor,
                t: Tensor,
                act: Optional[Tensor] = None,
                input_pos: Optional[Tensor] = None,  # for KV cache
                attn_impl: Optional[str] = None,  # 'sdpa' / 'flex' / 'vanilla'
                return_logvar = False,

    ):
        # TrigFlow Input to Flow Input
        t = torch.sin(t) / (torch.cos(t) + torch.sin(t))
        t_input = t * 1000


        x = x * torch.sqrt(t[:, :, None, None, None]**2 + (1-t[:, :, None, None, None]) **2)


        # x: (B, T, C, H, W)
        if return_logvar:
            model_out, logvar = super().forward(x, t_input, act, input_pos, attn_impl, return_logvar=True)
        else:
            model_out = super().forward(x, t_input, act, input_pos, attn_impl)

        # Flow --> TrigFlow Transformation
        t_flatten = t[:, :, None, None, None]
        trigflow_model_out = ((1 - 2 * t_flatten) * x + (1 - 2 * t_flatten + 2 * t_flatten**2) * model_out) / torch.sqrt(
            t_flatten**2 + (1 - t_flatten)**2
        )

        if return_logvar:
            return trigflow_model_out, logvar
        else:
            return trigflow_model_out

    def _flow_forward(self,
                x: Tensor,
                t: Tensor,
                act: Optional[Tensor] = None,
                input_frame_idx: Optional[Tensor] = None,  # for KVCache
                attn_impl: Optional[str] = None,  # 'sdpa' / 'flex' / 'vanilla'
                return_logvar = False,):
        """
        x:    (B, T, C, H, W)
        t:    (B, T)
        act:  (B, T, D)
        """
        #print(f'x: {x.shape}, t: {t.shape}, act: {act.shape if act is not None else None}, input_frame_idx: {input_frame_idx.shape if input_frame_idx is not None else None}')

        if t.dim() == 0:
            t = torch.full((x.shape[0],x.shape[1]), t, device = x.device)
        B, T, C, H, W = x.shape
        assert C == self.in_channels, f"Expected {self.in_channels} channels, but got {C}."
        nT, nH, nW = T // self.patch_size[0], H // self.patch_size[1], W // self.patch_size[2]
        attn_impl = attn_impl or self.attn_impl
        attn_mask = getattr(self, f'attn_mask_{attn_impl}')

        # 3d patchify
        x = rearrange(x, 'b t c h w -> b c t h w').contiguous()
        x = self.x_embedder(x)
        # x: (B, T'*H'*W', D)

        # verify number of patches
        n_patch_per_frame = nH * nW
        assert x.shape[1] == n_patch_per_frame * nT, \
            f"Expected {n_patch_per_frame * nT} patches, but got {x.shape[1]}."
        attn_mask = attn_mask[:x.shape[1], :x.shape[1]] if isinstance(attn_mask, torch.Tensor) else attn_mask


        if self.timestep_norm_scale_factor != 1.0:
            t = (t.float() / self.timestep_norm_scale_factor).to(torch.float32)
        else:
            t = t.long().to(torch.float32)
        # embed noise level
        # t: (B, T)
        c = self.t_embedder(t, dtype=x.dtype)
        # c: (B, T, D)
        # embed action if needed
        if act is not None:
            c += self.c_embedder(act)
        # c: (B, T, D)
        c = c.repeat_interleave(n_patch_per_frame, dim=1)
        # c: (B, T'*H'*W', D)

        # add positional embedding
        freqs_cis = None
        if self.pe_type == 'sincos' or self.pe_type == 'learned':
            x = x + self.pos_embed
        elif 'rope' in self.pe_type:
            freqs_cis = (self.freqs_cis[0][:x.shape[1]].to(x.device), self.freqs_cis[1][:x.shape[1]].to(x.device)) if isinstance(self.freqs_cis, tuple) \
                else self.freqs_cis[:x.shape[1]].to(x.device)
            # freqs_cis: (nT * nH * nW, head_dim)

        for block in self.blocks:
            x = block(
                x=x,  # x: (B, T'*H'*W', D)
                c_adaln=c,  # c: (B, T'*H'*W', D)
                freqs_cis=freqs_cis,  # freqs_cis: (nT * nH * nW, head_dim)
                attn_mask=attn_mask,
                attn_impl=attn_impl,
            )
        x = self.final_layer(x, c)

        x = self.unpatchify(x, nT, nH, nW)
        x = rearrange(x, 'b c t h w -> b t c h w').contiguous()

        return x

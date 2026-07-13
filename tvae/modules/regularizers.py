import logging
import numpy as np
from math import log2, ceil
from functools import partial, cache
from abc import abstractmethod
from contextlib import nullcontext
from typing import Any, Tuple, List, Optional, Dict, Iterator, Literal, Union
from einops import rearrange, reduce, pack, unpack

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor, int32
from torch.cuda.amp import autocast

from sgm.modules.distributions.distributions import DiagonalGaussianDistribution

logpy = logging.getLogger(__name__)


class AbstractRegularizer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        raise NotImplementedError()

    @abstractmethod
    def get_trainable_parameters(self) -> Any:
        raise NotImplementedError()


class DiagonalGaussianRegularizer(AbstractRegularizer):
    def __init__(self, sample: bool = True):
        super().__init__()
        self.sample = sample

    def get_trainable_parameters(self) -> Any:
        yield from ()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        log = dict()
        posterior = DiagonalGaussianDistribution(z)
        if self.sample:
            z = posterior.sample()
        else:
            z = posterior.mode()
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        log["kl_loss"] = kl_loss
        return z, log


class AbstractQuantizer(AbstractRegularizer):
    def __init__(self):
        super().__init__()
        # Define these in your init
        # shape (N,)
        self.used: Optional[torch.Tensor]
        self.re_embed: int
        self.unknown_index: Union[Literal["random"], int]

    def remap_to_used(self, inds: torch.Tensor) -> torch.Tensor:
        assert self.used is not None, "You need to define used indices for remap"
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(
                device=new.device
            )
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds: torch.Tensor) -> torch.Tensor:
        assert self.used is not None, "You need to define used indices for remap"
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    @abstractmethod
    def get_codebook_entry(
        self, indices: torch.Tensor, shape: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        raise NotImplementedError()

    def get_trainable_parameters(self) -> Iterator[torch.nn.Parameter]:
        yield from self.parameters()


class GumbelQuantizer(AbstractQuantizer):
    """
    credit to @karpathy:
    https://github.com/karpathy/deep-vector-quantization/blob/main/model.py (thanks!)
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """

    def __init__(
        self,
        num_hiddens: int,
        embedding_dim: int,
        n_embed: int,
        straight_through: bool = True,
        kl_weight: float = 5e-4,
        temp_init: float = 1.0,
        remap: Optional[str] = None,
        unknown_index: str = "random",
        loss_key: str = "loss/vq",
    ) -> None:
        super().__init__()

        self.loss_key = loss_key
        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight

        self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
        else:
            self.used = None
            self.re_embed = n_embed
        if unknown_index == "extra":
            self.unknown_index = self.re_embed
            self.re_embed = self.re_embed + 1
        else:
            assert unknown_index == "random" or isinstance(
                unknown_index, int
            ), "unknown index needs to be 'random', 'extra' or any integer"
            self.unknown_index = unknown_index  # "random" or "extra" or integer
        if self.remap is not None:
            logpy.info(
                f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )

    def forward(
        self, z: torch.Tensor, temp: Optional[float] = None, return_logits: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        # force hard = True when we are in eval mode, as we must quantize.
        # actually, always true seems to work
        hard = self.straight_through if self.training else True
        temp = self.temperature if temp is None else temp
        out_dict = {}
        logits = self.proj(z)
        if self.remap is not None:
            # continue only with used logits
            full_zeros = torch.zeros_like(logits)
            logits = logits[:, self.used, ...]

        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)
        if self.remap is not None:
            # go back to all entries but unused set to zero
            full_zeros[:, self.used, ...] = soft_one_hot
            soft_one_hot = full_zeros
        z_q = torch.einsum("b n h w, n d -> b d h w", soft_one_hot, self.embed.weight)

        # + kl divergence to the prior loss
        qy = F.softmax(logits, dim=1)
        diff = (
            self.kl_weight
            * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()
        )
        out_dict[self.loss_key] = diff

        ind = soft_one_hot.argmax(dim=1)
        out_dict["indices"] = ind
        if self.remap is not None:
            ind = self.remap_to_used(ind)

        if return_logits:
            out_dict["logits"] = logits

        return z_q, out_dict

    def get_codebook_entry(self, indices, shape):
        # TODO: shape not yet optional
        b, h, w, c = shape
        assert b * h * w == indices.shape[0]
        indices = rearrange(indices, "(b h w) -> b h w", b=b, h=h, w=w)
        if self.remap is not None:
            indices = self.unmap_to_all(indices)
        one_hot = (
            F.one_hot(indices, num_classes=self.n_embed).permute(0, 3, 1, 2).float()
        )
        z_q = torch.einsum("b n h w, n d -> b d h w", one_hot, self.embed.weight)
        return z_q


class VectorQuantizer(AbstractQuantizer):
    """
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term,
        beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(
        self,
        n_e: int,
        e_dim: int,
        beta: float = 0.25,
        remap: Optional[str] = None,
        unknown_index: str = "random",
        sane_index_shape: bool = False,
        log_perplexity: bool = False,
        embedding_weight_norm: bool = False,
        loss_key: str = "loss/vq",
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.loss_key = loss_key

        if not embedding_weight_norm:
            self.embedding = nn.Embedding(self.n_e, self.e_dim)
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.embedding = torch.nn.utils.weight_norm(
                nn.Embedding(self.n_e, self.e_dim), dim=1
            )

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
        else:
            self.used = None
            self.re_embed = n_e
        if unknown_index == "extra":
            self.unknown_index = self.re_embed
            self.re_embed = self.re_embed + 1
        else:
            assert unknown_index == "random" or isinstance(
                unknown_index, int
            ), "unknown index needs to be 'random', 'extra' or any integer"
            self.unknown_index = unknown_index  # "random" or "extra" or integer
        if self.remap is not None:
            logpy.info(
                f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )

        self.sane_index_shape = sane_index_shape
        self.log_perplexity = log_perplexity

    def forward(
        self,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        do_reshape = z.ndim == 4
        if do_reshape:
            #     # reshape z -> (batch, height, width, channel) and flatten
            z = rearrange(z, "b c h w -> b h w c").contiguous()

        else:
            assert z.ndim < 4, "No reshaping strategy for inputs > 4 dimensions defined"
            z = z.contiguous()

        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2
            * torch.einsum(
                "bd,dn->bn", z_flattened, rearrange(self.embedding.weight, "n d -> d n")
            )
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        loss_dict = {}
        if self.log_perplexity:
            perplexity, cluster_usage = measure_perplexity(
                min_encoding_indices.detach(), self.n_e
            )
            loss_dict.update({"perplexity": perplexity, "cluster_usage": cluster_usage})

        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean(
            (z_q - z.detach()) ** 2
        )
        loss_dict[self.loss_key] = loss

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        if do_reshape:
            z_q = rearrange(z_q, "b h w c -> b c h w").contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(
                z.shape[0], -1
            )  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)  # flatten

        if self.sane_index_shape:
            if do_reshape:
                min_encoding_indices = min_encoding_indices.reshape(
                    z_q.shape[0], z_q.shape[2], z_q.shape[3]
                )
            else:
                min_encoding_indices = rearrange(
                    min_encoding_indices, "(b s) 1 -> b s", b=z_q.shape[0]
                )

        loss_dict["min_encoding_indices"] = min_encoding_indices

        return z_q, loss_dict

    def get_codebook_entry(
        self, indices: torch.Tensor, shape: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            assert shape is not None, "Need to give shape for remap"
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class EmbeddingEMA(nn.Module):
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.decay = decay
        self.eps = eps
        weight = torch.randn(num_tokens, codebook_dim)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad=False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad=False)
        self.update = True

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(
            new_cluster_size, alpha=1 - self.decay
        )

    def embed_avg_ema_update(self, new_embed_avg):
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
            (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
        )
        # normalize embedding average with smoothed cluster size
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        self.weight.data.copy_(embed_normalized)


class EMAVectorQuantizer(AbstractQuantizer):
    def __init__(
        self,
        n_embed: int,
        embedding_dim: int,
        beta: float,
        decay: float = 0.99,
        eps: float = 1e-5,
        remap: Optional[str] = None,
        unknown_index: str = "random",
        loss_key: str = "loss/vq",
    ):
        super().__init__()
        self.codebook_dim = embedding_dim
        self.num_tokens = n_embed
        self.beta = beta
        self.loss_key = loss_key

        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
        else:
            self.used = None
            self.re_embed = n_embed
        if unknown_index == "extra":
            self.unknown_index = self.re_embed
            self.re_embed = self.re_embed + 1
        else:
            assert unknown_index == "random" or isinstance(
                unknown_index, int
            ), "unknown index needs to be 'random', 'extra' or any integer"
            self.unknown_index = unknown_index  # "random" or "extra" or integer
        if self.remap is not None:
            logpy.info(
                f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # reshape z -> (batch, height, width, channel) and flatten
        # z, 'b c h w -> b h w c'
        z = rearrange(z, "b c h w -> b h w c")
        z_flattened = z.reshape(-1, self.codebook_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (
            z_flattened.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * torch.einsum("bd,nd->bn", z_flattened, self.embedding.weight)
        )  # 'n d -> d n'

        encoding_indices = torch.argmin(d, dim=1)

        z_q = self.embedding(encoding_indices).view(z.shape)
        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        if self.training and self.embedding.update:
            # EMA cluster size
            encodings_sum = encodings.sum(0)
            self.embedding.cluster_size_ema_update(encodings_sum)
            # EMA embedding average
            embed_sum = encodings.transpose(0, 1) @ z_flattened
            self.embedding.embed_avg_ema_update(embed_sum)
            # normalize embed_avg and update weight
            self.embedding.weight_update(self.num_tokens)

        # compute loss for embedding
        loss = self.beta * F.mse_loss(z_q.detach(), z)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        # z_q, 'b h w c -> b c h w'
        z_q = rearrange(z_q, "b h w c -> b c h w")

        out_dict = {
            self.loss_key: loss,
            "encodings": encodings,
            "encoding_indices": encoding_indices,
            "perplexity": perplexity,
        }

        return z_q, out_dict


class VectorQuantizerWithInputProjection(VectorQuantizer):
    def __init__(
        self,
        input_dim: int,
        n_codes: int,
        codebook_dim: int,
        beta: float = 1.0,
        output_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(n_codes, codebook_dim, beta, **kwargs)
        self.proj_in = nn.Linear(input_dim, codebook_dim)
        self.output_dim = output_dim
        if output_dim is not None:
            self.proj_out = nn.Linear(codebook_dim, output_dim)
        else:
            self.proj_out = nn.Identity()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        rearr = False
        in_shape = z.shape

        if z.ndim > 3:
            rearr = self.output_dim is not None
            z = rearrange(z, "b c ... -> b (...) c")
        z = self.proj_in(z)
        z_q, loss_dict = super().forward(z)

        z_q = self.proj_out(z_q)
        if rearr:
            if len(in_shape) == 4:
                z_q = rearrange(z_q, "b (h w) c -> b c h w ", w=in_shape[-1])
            elif len(in_shape) == 5:
                z_q = rearrange(
                    z_q, "b (t h w) c -> b c t h w ", w=in_shape[-1], h=in_shape[-2]
                )
            else:
                raise NotImplementedError(
                    f"rearranging not available for {len(in_shape)}-dimensional input."
                )

        return z_q, loss_dict


@cache
def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1

def maybe_distributed_mean(t):
    if not is_distributed():
        return t

    dist.all_reduce(t)
    t = t / dist.get_world_size()
    return t

# helper functions
def exists(v):
    return v is not None

def identity(t):
    return t

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def l2norm(t):
    return F.normalize(t, dim = -1)

def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()

def log(t, eps = 1e-5):
    return t.clamp(min = eps).log()

def entropy(prob):
    return (-prob * log(prob)).sum(dim=-1)


class CosineSimLinear(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        scale = 1.
    ):
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(dim_in, dim_out))

    def forward(self, x):
        x = F.normalize(x, dim = -1)
        w = F.normalize(self.weight, dim = 0)
        return (x @ w) * self.scale


class LFQRegularizer(AbstractRegularizer):
    def __init__(
        self,
        dim = None,
        codebook_size = None,
        entropy_loss_weight = 0.1,
        commitment_loss_weight = 0.25,
        diversity_gamma = 1.,
        straight_through_activation = nn.Identity(),
        num_codebooks = 1,
        keep_num_codebooks_dim = None,
        codebook_scale = 1.,                        # for residual LFQ, codebook scaled down by 2x at each layer
        frac_per_sample_entropy = 1.,               # make less than 1. to only use a random fraction of the probs for per sample entropy
        has_projections = None,
        projection_has_bias = True,
        soft_clamp_input_value = None,
        cosine_sim_project_in = False,
        cosine_sim_project_in_scale = None,
        channel_first = None,
        experimental_softplus_entropy_loss = False,
        entropy_loss_offset = 5.,                   # how much to shift the loss before softplus
        spherical = False,                          # from https://arxiv.org/abs/2406.07548
        force_quantization_f32 = True               # will force the quantization step to be full precision
    ):
        super().__init__()

        # some assert validations

        assert exists(dim) or exists(codebook_size), 'either dim or codebook_size must be specified for LFQ'
        assert not exists(codebook_size) or log2(codebook_size).is_integer(), f'your codebook size must be a power of 2 for lookup free quantization (suggested {2 ** ceil(log2(codebook_size))})'

        codebook_size = default(codebook_size, lambda: 2 ** dim)
        self.codebook_size = codebook_size

        codebook_dim = int(log2(codebook_size))
        codebook_dims = codebook_dim * num_codebooks
        dim = default(dim, codebook_dims)

        has_projections = default(has_projections, dim != codebook_dims)

        if cosine_sim_project_in:
            cosine_sim_project_in = default(cosine_sim_project_in_scale, codebook_scale)
            project_in_klass = partial(CosineSimLinear, scale = cosine_sim_project_in)
        else:
            project_in_klass = partial(nn.Linear, bias = projection_has_bias)

        self.project_in = project_in_klass(dim, codebook_dims) if has_projections else nn.Identity()
        self.project_out = nn.Linear(codebook_dims, dim, bias = projection_has_bias) if has_projections else nn.Identity()
        self.has_projections = has_projections

        self.dim = dim
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebooks

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        # channel first
        self.channel_first = channel_first

        # straight through activation
        self.activation = straight_through_activation

        # whether to use BSQ (binary spherical quantization)
        self.spherical = spherical
        self.maybe_l2norm = (lambda t: l2norm(t) * self.codebook_scale) if spherical else identity

        # entropy aux loss related weights
        assert 0 < frac_per_sample_entropy <= 1.
        self.frac_per_sample_entropy = frac_per_sample_entropy

        self.diversity_gamma = diversity_gamma
        self.entropy_loss_weight = entropy_loss_weight

        # codebook scale
        self.codebook_scale = codebook_scale

        # commitment loss
        self.commitment_loss_weight = commitment_loss_weight

        # whether to soft clamp the input value from -value to value
        self.soft_clamp_input_value = soft_clamp_input_value
        assert not exists(soft_clamp_input_value) or soft_clamp_input_value >= codebook_scale

        # whether to make the entropy loss positive through a softplus (experimental, please report if this worked or not in discussions)
        self.entropy_loss_offset = entropy_loss_offset
        self.experimental_softplus_entropy_loss = experimental_softplus_entropy_loss

        # for no auxiliary loss, during inference
        self.register_buffer('mask', 2 ** torch.arange(codebook_dim - 1, -1, -1))
        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # whether to force quantization step to be f32
        self.force_quantization_f32 = force_quantization_f32

        # codes
        all_codes = torch.arange(codebook_size)
        bits = ((all_codes[..., None].int() & self.mask) != 0).float()
        codebook = self.bits_to_codes(bits)

        self.register_buffer('codebook', codebook.float(), persistent = False)

    def bits_to_codes(self, bits):
        return bits * self.codebook_scale * 2 - self.codebook_scale

    @property
    def dtype(self):
        return self.codebook.dtype

    def indices_to_codes(
        self,
        indices,
        project_out = True
    ):
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))
        should_transpose = default(self.channel_first, is_img_or_video)

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... -> ... 1')

        # indices to codes, which are bits of either -1 or 1
        bits = ((indices[..., None].int() & self.mask) != 0).to(self.dtype)
        codes = self.bits_to_codes(bits)
        codes = self.maybe_l2norm(codes)
        codes = rearrange(codes, '... c d -> ... (c d)')

        # whether to project codes out to original dimensions
        # if the input feature dimensions were not log2(codebook size)
        if project_out:
            codes = self.project_out(codes)

        # rearrange codes back to original shape
        if should_transpose:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    def forward(
        self,
        x,
        inv_temperature = 100.,
        mask = None,
    ):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """

        is_img_or_video = x.ndim >= 4
        should_transpose = default(self.channel_first, is_img_or_video)

        # standardize image or video into (batch, seq, dimension)
        if should_transpose:
            x = rearrange(x, 'b d ... -> b ... d')
            x, ps = pack_one(x, 'b * d')

        assert x.shape[-1] == self.dim, f'expected dimension of {self.dim} but received {x.shape[-1]}'
        x = self.project_in(x)

        # maybe soft clamp
        if exists(self.soft_clamp_input_value):
            clamp_value = self.soft_clamp_input_value
            x = (x / clamp_value).tanh() * clamp_value

        # split out number of codebooks
        x = rearrange(x, 'b n (c d) -> b n c d', c = self.num_codebooks)

        # maybe l2norm
        x = self.maybe_l2norm(x)

        # whether to force quantization step to be full precision or not
        force_f32 = self.force_quantization_f32
        quantization_context = partial(autocast, 'cuda', enabled = False) if force_f32 else nullcontext
        with quantization_context():

            if force_f32:
                orig_dtype = x.dtype
                x = x.float()

            # quantize by eq 3.
            original_input = x
            codebook_value = torch.ones_like(x) * self.codebook_scale
            quantized = torch.where(x > 0, codebook_value, -codebook_value)

            # calculate indices
            indices = reduce((quantized > 0).int() * self.mask.int(), 'b n c d -> b n c', 'sum')

            # maybe l2norm
            quantized = self.maybe_l2norm(quantized)

            # use straight-through gradients (optionally with custom activation fn) if training
            if self.training:
                x = self.activation(x)
                x = x + (quantized - x).detach()
            else:
                x = quantized

            # entropy aux loss
            if self.training:

                if force_f32:
                    codebook = self.codebook.float()

                codebook = self.maybe_l2norm(codebook)
                # the same as euclidean distance up to a constant
                distance = -2 * torch.einsum('... i d, j d -> ... i j', original_input, codebook)
                prob = (-distance * inv_temperature).softmax(dim = -1)

                # account for mask
                if exists(mask):
                    prob = prob[mask]
                else:
                    prob = rearrange(prob, 'b n ... -> (b n) ...')

                # whether to only use a fraction of probs, for reducing memory
                if self.frac_per_sample_entropy < 1.:
                    num_tokens = prob.shape[0]
                    num_sampled_tokens = int(num_tokens * self.frac_per_sample_entropy)
                    rand_mask = torch.randn(num_tokens).argsort(dim = -1) < num_sampled_tokens
                    per_sample_probs = prob[rand_mask]
                else:
                    per_sample_probs = prob

                # calculate per sample entropy
                per_sample_entropy = entropy(per_sample_probs).mean()

                # distribution over all available tokens in the batch
                avg_prob = reduce(per_sample_probs, '... c d -> c d', 'mean')
                avg_prob = maybe_distributed_mean(avg_prob)
                codebook_entropy = entropy(avg_prob).mean()

                # 1. entropy will be nudged to be low for each code, to encourage the network to output confident predictions
                # 2. codebook entropy will be nudged to be high, to encourage all codes to be uniformly used within the batch
                entropy_aux_loss = per_sample_entropy - self.diversity_gamma * codebook_entropy
            else:
                # if not training, just return dummy 0
                entropy_aux_loss = per_sample_entropy = codebook_entropy = self.zero

            # whether to make the entropy loss positive or not through a (shifted) softplus
            if self.training and self.experimental_softplus_entropy_loss:
                entropy_aux_loss = F.softplus(entropy_aux_loss + self.entropy_loss_offset)

            # commit loss
            if self.training and self.commitment_loss_weight > 0.:
                commit_loss = F.mse_loss(original_input, quantized.detach(), reduction = 'none')
                if exists(mask):
                    commit_loss = commit_loss[mask]
                commit_loss = commit_loss.mean()
            else:
                commit_loss = self.zero

            # input back to original dtype if needed
            if force_f32:
                x = x.type(orig_dtype)

        # merge back codebook dim
        x = rearrange(x, 'b n c d -> b n (c d)')

        # project out to feature dimension if needed
        x = self.project_out(x)

        # reconstitute image or video dimensions
        if should_transpose:
            x = unpack_one(x, ps, 'b * d')
            x = rearrange(x, 'b ... d -> b d ...')

            indices = unpack_one(indices, ps, 'b * c')

        # whether to remove single codebook dim
        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')

        # complete aux loss
        aux_loss = entropy_aux_loss * self.entropy_loss_weight + commit_loss * self.commitment_loss_weight

        return x, dict(indices = indices, entropy_aux_loss = aux_loss)


class FSQRegularizer(AbstractRegularizer):
    def __init__(
        self,
        levels: List[int],
        dim: Optional[int] = None,
        num_codebooks = 1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None
        ):
        super().__init__()
        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent = False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent = False)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        has_projections = self.dim != effective_codebook_dim
        self.project_in = nn.Linear(self.dim, effective_codebook_dim) if has_projections else nn.Identity()
        self.project_out = nn.Linear(effective_codebook_dim, self.dim) if has_projections else nn.Identity()
        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size), project_out = False)
        self.register_buffer("implicit_codebook", implicit_codebook, persistent = False)

        # added
        self.global_codebook_usage = torch.zeros([2 ** self.codebook_dim, self.num_codebooks], \
                                                  dtype=torch.long)

    # added
    def get_trainable_parameters(self) -> Any:
        return self.parameters()

    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 + eps) / 2
        # half_l: [3.5035, 3.5035, 3.5035, 3.5035, 3.5035, 3.5035], levels: [8, 8, 8, 8, 8, 8]
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        # offset: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        shift = (offset / half_l).atanh()
        # shift: [0.1437, 0.1437, 0.1437, 0.1437, 0.1437, 0.1437]
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        # z: [-8~8]
        # self.bound(z): [-4~3]
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2 # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_codes(
        self,
        indices: Tensor,
        project_out = True
        ) -> Tensor:
        """Inverse of `codes_to_indices`."""

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        if project_out:
            codes = self.project_out(codes)

        if is_img_or_video:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    @autocast(enabled = False)
    def forward(self, z: Tensor) -> Tensor:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """
        # b c t h w

        is_img_or_video = z.ndim >= 4

        # standardize image or video into (batch, seq, dimension)

        if is_img_or_video:
            z = rearrange(z, 'b d ... -> b ... d')
            z, ps = pack_one(z, 'b * d')

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'
        # z: (bs, 256, 6), levels: [8, 8, 8, 8, 8, 8]
        z = self.project_in(z)
        # z: (bs, 256, 6), levels: [8, 8, 8, 8, 8, 8]
        z = rearrange(z, 'b n (c d) -> b n c d', c = self.num_codebooks)
        # z: (bs, 256, 1, 6), levels: [8, 8, 8, 8, 8, 8]

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)

        codes = rearrange(codes, 'b n c d -> b n (c d)')

        out = self.project_out(codes)

        # reconstitute image or video dimensions

        if is_img_or_video:
            out = unpack_one(out, ps, 'b * d')
            out = rearrange(out, 'b ... d -> b d ...')

            indices = unpack_one(indices, ps, 'b * c')

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, '... 1 -> ...')

        # return out, indices
        return out, dict(indices = indices)


def measure_perplexity(predicted_indices, num_centroids):
    # src: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py
    # eval cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
    encodings = (
        F.one_hot(predicted_indices, num_centroids).float().reshape(-1, num_centroids)
    )
    avg_probs = encodings.mean(0)
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
    cluster_use = torch.sum(avg_probs > 0)
    return perplexity, cluster_use

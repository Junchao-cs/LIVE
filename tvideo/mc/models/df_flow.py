import re
import gc
import math
from rich import print
from tqdm.rich import tqdm
from packaging import version
from omegaconf import ListConfig, OmegaConf
from typing import Dict, Optional, Tuple, Union

import torch
import lightning.pytorch as pl
import torchmetrics
from torch.optim.lr_scheduler import LambdaLR
from safetensors.torch import load_file as load_safetensors
from diffusers.models import AutoencoderKL
from einops import rearrange
from sgm.util import (default, print0, get_obj_from_str, instantiate_from_config,
                      disabled_train, get_valid_paths)
from sgm.modules.distributions.distributions import DiagonalGaussianDistribution

from tvideo.utils import EinopsWrapper
from tvideo.mc.modules.flow import create_diffusion
from tvideo.mc.modules.flow.respace import compute_density_for_timestep_sampling
from tvideo.mc.modules.flow.dpm_solver import NoiseScheduleFlow

import os
import numpy as np
from torchvision.io import write_video
from PIL import Image

from tvideo.utils import CameraPose
import random

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a PyTorch tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,) * (dims - 1)]

def create_comparison_gif(gt_frames, generated_frames, save_path, fps=10):
    """
    创建对比GIF，上面是GT，下面是生成的

    Args:
        gt_frames: numpy array (T, H, W, C) [0-255]
        generated_frames: numpy array (T, H, W, C) [0-255]
        save_path: str, 保存路径
        fps: int, 帧率
    """
    T, H, W, C = gt_frames.shape

    # 创建合并后的帧列表
    combined_frames = []

    for t in range(T):
        # 获取当前帧
        gt_frame = gt_frames[t].astype(np.uint8)
        gen_frame = generated_frames[t].astype(np.uint8)

        # 垂直拼接：上面GT，下面生成
        combined_frame = np.concatenate([gt_frame, gen_frame], axis=0)  # (2*H, W, C)

        # 转换为PIL Image
        combined_pil = Image.fromarray(combined_frame)
        combined_frames.append(combined_pil)

    # 保存为GIF
    if combined_frames:
        combined_frames[0].save(
            save_path,
            save_all=True,
            append_images=combined_frames[1:],
            duration=1000//fps,  # 毫秒
            loop=0
        )

class DiffusionForcing(pl.LightningModule):
    """
    Diffusion Forcing Video Model
    """

    def __init__(self,
                 # main config
                 network_config: Union[str, Dict, ListConfig, OmegaConf],
                 diffusion_config: Union[str, Dict, ListConfig, OmegaConf],
                 tokenizer_config: Union[str, Dict, ListConfig, OmegaConf] = None,
                 optimizer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
                 scheduler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
                 network_einops_wapper: Optional[list] = None,
                 # train specific
                 scheduler_warmup_steps = 10000,
                 # val specific
                 max_frames: int = 32,
                 val_num_total_frames: int = 16,  # should be less than data frames in validation dataset
                 val_num_prompt_frames: int = 1,
                 # data
                 is_latent_input: bool = False,
                 is_pad_or_trunc_input: str = "null",  # pad / trunc / null, null exists mismatch between vid & act
                 # default
                 video_key: str = 'mp4',
                 label_key: str = 'json',
                 scale_factor: float = 1.0,
                 enable_tokenizer_fp16_autocast: bool = True,
                 en_and_decode_n_samples_a_time: Optional[int] = None,
                 tokenizer_encode_after_batch_transfer: bool = False,
                 ckpt_path: Union[None, str] = None,
                 ckpt_path2: Union[None, str] = None,
                 ignore_keys: Union[Tuple, list, ListConfig] = (),
                 init_from_oasis_model: bool = False,
                 strict_loading: bool = True,  # strict loading when resume_from_checkpoint
                 compile_tokenizer: bool = False,
                 compile_model: bool = False,
                 log_time_profile: bool = False,
                 log_mem_profile: bool = False,
                 manual_seed: Optional[int] = None,
                 monitor: Union[None, str] = None,
                 save_fd: Optional[str] = None,
                 use_plucker: bool = True,
                 enable_cycle_forcing: bool = False,
                 enable_kv_cache: bool = False,
                 p_set_t_clean: float = 0.7,
    ):
        super().__init__()
        # Save hyperparameters for release-run bookkeeping and config-based resume.
        self.save_hyperparameters()
        self.print_prefix = "[bold magenta]\[tvideo.mc.models.df_flow][DiffusionForcing][/bold magenta]"

        # create model
        print0(f"{self.print_prefix} Initializing model...")
        network_config['params']['enable_cycle_forcing'] = enable_cycle_forcing
        self.model = instantiate_from_config(network_config)
        if network_einops_wapper is not None:
            assert len(network_einops_wapper) == 2, "EinopsWrapper requires two arguments: input and output format."
            self.model = EinopsWrapper(*network_einops_wapper, self.model)

        # create diffusion
        self.diffusion = create_diffusion(
            timestep_respacing=str(diffusion_config.train_sampling_steps),
            noise_schedule=diffusion_config.noise_schedule,
            predict_v=diffusion_config.predict_v,
            learn_sigma=diffusion_config.learn_sigma,
            pred_sigma=diffusion_config.pred_sigma,
            snr=diffusion_config.snr,
            flow_shift=diffusion_config.flow_shift,
        )
        self.diffusion_config = diffusion_config

        # create optimizer and scheduler
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.AdamW"}
        )
        self.scheduler_config = scheduler_config
        self.save_fd = save_fd

        print0(f"{self.print_prefix} Initializing tokenizer...")
        self.tokenizer = self.create_tokenizer(tokenizer_config) if tokenizer_config is not None else None

        # train specific
        self.scheduler_warmup_steps = scheduler_warmup_steps
        # val specific
        assert val_num_total_frames > 0, "val_num_total_frames must be greater than 0."
        assert val_num_prompt_frames <= val_num_total_frames, "val_num_prompt_frames must be less than or equal to val_num_total_frames."
        self.val_num_total_frames = val_num_total_frames
        self.val_num_prompt_frames = val_num_prompt_frames
        self.max_frames = max_frames
        # data
        if is_latent_input:
            self.tokenizer.encoder = None
            assert init_from_oasis_model is False, "init_from_oasis_model does not support is_latent_input=True."
        self.is_latent_input = is_latent_input
        self.is_pad_or_trunc_input = is_pad_or_trunc_input
        # default
        self.video_key = video_key
        self.label_key = label_key
        self.scale_factor =  scale_factor
        self.enable_tokenizer_fp16_autocast = enable_tokenizer_fp16_autocast
        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time
        self.tokenizer_encode_after_batch_transfer = tokenizer_encode_after_batch_transfer
        self.log_time_profile = log_time_profile
        self.log_mem_profile = log_mem_profile
        self.manual_seed = manual_seed
        if monitor is not None:
            self.monitor = monitor

        # position condition
        self.use_plucker = use_plucker # otherwise Rt
        # Cycle-forcing objective.
        self.enable_cycle_forcing = enable_cycle_forcing

        self.enable_kv_cache = enable_kv_cache
        self.p_set_t_clean = p_set_t_clean

        # load checkpoint if provided ckpt_path
        ckpt_path = get_valid_paths(ckpt_path, ckpt_path2)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, init_from_oasis_model=init_from_oasis_model)
        self.init_from_oasis_model = init_from_oasis_model

        # create evaluation models after loading from checkpoint
        print0(f"{self.print_prefix} Initializing evaluation models...")
        self.train_metrics, self.val_metrics = self.create_evaluation_models()
        # Set to False because we don't care about the metrics
        self.strict_loading = strict_loading

        # compile model and tokenizer
        compile_tokenizer_fn = (
            torch.compile if (version.parse(torch.__version__) >= version.parse("2.0.0")) and compile_tokenizer else lambda x: x)
        compile_model_fn = (
            torch.compile if (version.parse(torch.__version__) >= version.parse("2.0.0")) and compile_model else lambda x: x)
        self.tokenizer = compile_tokenizer_fn(self.tokenizer)
        self.model = compile_model_fn(self.model)  # must be compiled after loading from checkpoint

    def init_from_ckpt(
        self, path: str, ignore_keys: Union[Tuple, list, ListConfig] = tuple(), init_from_oasis_model: bool = False, verbose: bool = False,
    ) -> None:
        if path.endswith("ckpt"):
            weights = torch.load(path, map_location="cpu", weights_only = False)["state_dict"]
        elif path.endswith("safetensors"):
            weights = load_safetensors(path)
        else:
            raise NotImplementedError

        keys = list(weights.keys())
        for k in keys:
            # Attention masks are deterministic buffers rebuilt from the config.
            if ".attn_mask_" in k:
                del weights[k]
                continue
            for ik in ignore_keys:
                if re.match(ik, k):
                    if verbose:
                        print0(f"{self.print_prefix} Deleting key {k} from state_dict.")
                    del weights[k]
            if k.startswith("model._orig_mod"):
                new_key = k.replace("model._orig_mod", "model")
                if not new_key in keys:
                    weights[new_key] = weights[k]
                    print(f"{self.print_prefix} Renaming key {k} to {new_key}.")
                    del weights[k]

        if init_from_oasis_model:
            # add model. to all weight names
            for k in list(weights.keys()):
                if k.startswith("model."):
                    continue
                if k.startswith("external_cond."):
                    weights["model.cond_embedder." + k.replace("external_cond.", "")] = weights.pop(k)
                else:
                    weights["model." + k] = weights.pop(k)

        missing, unexpected = self.load_state_dict(weights, strict=False)

        print0(
            f"{self.print_prefix} Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0 and verbose:
            print0(f"{self.print_prefix} Missing Keys: {missing}")
        if len(unexpected) > 0 and verbose:
            print0(f"{self.print_prefix} Unexpected Keys: {unexpected}")

    def on_load_checkpoint(self, checkpoint):
        # ['epoch', 'global_step', 'pytorch-lightning_version', 'state_dict', 'loops', 'callbacks', 'optimizer_states', 'lr_schedulers', 'hparams_name', 'hyper_parameters']
        # checkpoint['new_key'] = checkpoint['old_key'] to rename, no need to return
        state_dict = checkpoint.get("state_dict", {})
        for key in tuple(state_dict):
            if ".attn_mask_" in key:
                del state_dict[key]
        print0(f"{self.print_prefix} on_load_checkpoint from global_step={checkpoint['global_step']}")

    def create_tokenizer(self, config):
        if isinstance(config, str):
            tokenizer = AutoencoderKL.from_pretrained(config).eval()
        else:
            tokenizer = instantiate_from_config(config).eval()
        tokenizer.train = disabled_train
        for param in tokenizer.parameters():
            param.requires_grad = False
        return tokenizer

    def create_evaluation_models(self):
        # self.perceptual_loss = LPIPS().eval()

        # https://lightning.ai/docs/torchmetrics/stable/pages/lightning.html
        base_metrics = torchmetrics.MetricCollection(
            {
                "psnr": torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0),  # 0 ~ 1
                "ssim": torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0),  # 0 ~ 1
                "lpips": torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True),  # 0 ~ 1
                # "fid": torchmetrics.image.fid.FrechetInceptionDistance(feature=2048, normalize=True),  # 0 ~ 1
                # "is": torchmetrics.image.inception.InceptionScore(normalize=True),  # 0 ~ 1
            },
            prefix="train/",
        )

        train_metrics = torch.nn.ModuleDict({
            "metrics": base_metrics,
            "fid": torchmetrics.image.fid.FrechetInceptionDistance(feature=2048, normalize=True),  # 0 ~ 1
            "is": torchmetrics.image.inception.InceptionScore(normalize=True),  # 0 ~ 1
        })
        val_metrics = torch.nn.ModuleDict({
            "metrics_v_dpm": base_metrics.clone(prefix=f"val_v_dpm/"),
            "fid_v_dpm": torchmetrics.image.fid.FrechetInceptionDistance(feature=2048, normalize=True),  # 0 ~ 1
            "is_v_dpm": torchmetrics.image.inception.InceptionScore(normalize=True),  # 0 ~ 1
        })

        return train_metrics, val_metrics

    def on_train_start(self, *args, **kwargs):

        if hasattr(self.model, "attn_mask_flex"):
            self.model.attn_mask_flex = self.model.attn_mask_flex.to(self.device)
        if self.manual_seed is not None:
            assert isinstance(self.manual_seed, int), "[tvideo.mc.models.df_flow][DiffusionForcing] manual_seed must be an integer."
            self.seed_generator = torch.Generator(device=self.device)
            self.seed_generator.manual_seed(self.manual_seed + self.global_rank)
        else:
            self.seed_generator = None
        self.diffusion.set_seed_generator(self.seed_generator)  # https://github.com/pytorch/pytorch/issues/27072#issuecomment-1757826240
        # memory cleanup
        torch.distributed.barrier()
        gc.collect()
        torch.cuda.empty_cache()
        # setup profiler
        if self.log_time_profile:
            self.trainer.profiler = pl.profilers.SimpleProfiler()
        # log memory profile
        if self.log_mem_profile:
            from tvideo.utils import profile_module
            profile_module(self.tokenizer_encode, (1, 16, 3, 224, 384), device=self.device)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.tokenizer_encode_after_batch_transfer:
            batch[self.video_key] = self.tokenizer_encode(batch[self.video_key])
        return batch

    def compute_rt_in_window(self, camera_params):
        """
        计算相对于第一帧的相机外参（带安全检查）
        Args:
            camera_params: (T, 16) 或 (B, T, 16) 张量

        Returns:
            relative_camera_params: 同输入维度 - 相对相机参数
        """
        is_2d = camera_params.dim() == 2
        if is_2d:
            camera_params = camera_params.unsqueeze(0)

        B, T = camera_params.shape[:2]
        device = camera_params.device

        intrinsics = camera_params[:, :, :4]
        extrinsics_flat = camera_params[:, :, 4:16]
        extrinsics = extrinsics_flat.reshape(B, T, 3, 4)

        bottom_row = torch.tensor([0, 0, 0, 1], device=device, dtype=extrinsics.dtype)
        bottom_row = bottom_row.view(1, 1, 1, 4).expand(B, T, 1, 4)
        extrinsics_4x4 = torch.cat([extrinsics, bottom_row], dim=2)
        first_frame_extrinsics = extrinsics_4x4[:, 0:1]

        # ===== 安全检查 =====
        if torch.isnan(first_frame_extrinsics).any() or torch.isinf(first_frame_extrinsics).any():
            print(f"Warning: First frame contains NaN or Inf. Using identity matrix.")
            first_frame_extrinsics = torch.eye(4, device=device, dtype=extrinsics.dtype).view(1, 1, 4, 4).expand(B, 1, 4, 4)

        det = torch.det(first_frame_extrinsics.squeeze(1))  # (B,)
        if (det.abs() < 1e-6).any():
            print(f"Warning: First frame matrix is nearly singular (det={det.item():.2e}). Using identity matrix.")
            first_frame_extrinsics = torch.eye(4, device=device, dtype=extrinsics.dtype).view(1, 1, 4, 4).expand(B, 1, 4, 4)

        try:
            first_frame_inv = torch.inverse(first_frame_extrinsics)
        except RuntimeError as e:
            print(f"Warning: Failed to invert first frame matrix: {e}")
            print(f"First frame extrinsics:\n{first_frame_extrinsics}")
            first_frame_inv = torch.eye(4, device=device, dtype=extrinsics.dtype).view(1, 1, 4, 4).expand(B, 1, 4, 4)
        # ===================

        relative_extrinsics = torch.matmul(
            first_frame_inv.expand(B, T, 4, 4),
            extrinsics_4x4
        )

        relative_extrinsics_3x4 = relative_extrinsics[:, :, :3, :]
        relative_extrinsics_flat = relative_extrinsics_3x4.reshape(B, T, 12)
        relative_camera_params = torch.cat([intrinsics, relative_extrinsics_flat], dim=-1)

        if is_2d:
            relative_camera_params = relative_camera_params.squeeze(0)
        return relative_camera_params

    def compute_plucker_in_window(self, camera_params, h, w):
            """
            计算窗口内的 Plücker 坐标
            Args:
                camera_params: (B, window_size, 16) - 4 intrinsics + 12 extrinsics
                h, w: latent spatial resolution
            Returns:
                plucker_maps: (B, window_size, 6, h, w)
            """
            poses = CameraPose.from_vectors(camera_params)
            # if random.random() < 0.5:
            poses.normalize_by_first()
            # else:
            #     poses.normalize_by_last()
            rays = poses.rays(resolution=h) # 假设是正方形或长边对齐，官方只传一个 int
            # 让我们假设你稍微修改了本地的 geometry_utils.py 里的 rays 函数来支持 (h, w)
            # rays = poses.rays(resolution=(h, w))

            # 转为 Plücker 坐标 (B, T, H, W, 6)
            # use_plucker=True 会返回 direction(3) + moment(3)
            plucker = rays.to_tensor(use_plucker=True)

            # 5. 调整维度适配模型输入 (B, T, 6, H, W)
            plucker = rearrange(plucker, "b t h w c -> b t c h w")
            return plucker

    @torch.inference_mode()
    def get_input(self, batch):
        vid = batch[self.video_key]
        act = batch[self.label_key].to(batch[self.video_key].dtype)
        return vid, act

    @torch.inference_mode()
    def tokenizer_decode(self, z):
        if self.tokenizer is None:
            return z

        if self.log_time_profile:
            # alternative: with self.trainer.profiler.profile(f"[{self.__class__.__name__}].tokenizer_decode"):
            self.trainer.profiler.start(f"[{self.__class__.__name__}].tokenizer_decode")

        B, T = z.shape[:2]

        if self.init_from_oasis_model:
            z = rearrange(z, "b t c h w -> (b t) (h w) c")
        else:
            z = rearrange(z, "b t c h w -> (b t) c h w")

        z = 1.0 / self.scale_factor * z
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])
        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.enable_tokenizer_fp16_autocast):
            for n in range(n_rounds):
                out = self.tokenizer.decode(
                    z[n * n_samples : (n + 1) * n_samples])
                if not isinstance(out, torch.Tensor):  # diffusers
                    out = out.sample
                all_out.append(out)
        out = torch.cat(all_out, dim=0)

        out = rearrange(out, "(b t) c h w -> b t c h w", t=T)

        if torch.isnan(out).any() or torch.isinf(out).any():
            print0(f"{self.print_prefix}[tokenizer_decode] NaN / Inf detected in decoded samples. Replacing with zeros.")
            out = torch.zeros_like(out)
        if self.log_time_profile:
            self.trainer.profiler.stop(f"[{self.__class__.__name__}].tokenizer_decode")
        return out

    @torch.inference_mode()
    def tokenizer_encode(self, x):
        # x: (b, t, c, h, w), -1 ~ 1
        if self.tokenizer is None:
            return x

        if self.log_time_profile:
            # alternative: with self.trainer.profiler.profile(f"[{self.__class__.__name__}].tokenizer_encode"):
            self.trainer.profiler.start(f"[{self.__class__.__name__}].tokenizer_encode")

        B, T = x.shape[:2]
        x = rearrange(x, "b t c h w -> (b t) c h w")

        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = []
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.enable_tokenizer_fp16_autocast):
            for n in range(n_rounds):
                if self.is_latent_input:
                    out = DiagonalGaussianDistribution(x[n * n_samples : (n + 1) * n_samples]).sample().detach()
                else:
                    out = self.tokenizer.encode(
                        x[n * n_samples : (n + 1) * n_samples]
                    )
                if not isinstance(out, torch.Tensor):
                    if hasattr(out, 'latent_dist'):  # diffusers
                        out = out['latent_dist'].sample().detach()
                    elif hasattr(out, 'sample'):  # DiagonalGaussianDistribution
                        out = out.sample().detach()
                    else:
                        raise NotImplementedError(f"[tvideo.mc.models.df_flow][DiffusionForcing] Unknown output type: {type(out)}")
                all_out.append(out)
        z = torch.cat(all_out, dim=0).to(x.dtype)
        z = self.scale_factor * z

        if self.init_from_oasis_model:
            z = rearrange(z, "(b t) (h w) c -> b t c h w", b=B, t=T, h=18, w=32, c=16)
        else:
            z = rearrange(z, "(b t) c h w -> b t c h w", t=T)
        if self.log_time_profile:
            self.trainer.profiler.stop(f"[{self.__class__.__name__}].tokenizer_encode")
        return z.contiguous()

    def prepare_inputs(self, vid, pose, p: int):
        """
        根据 p 值，自动选取 vid 的最后 p 帧，
        并按照 (nT // p) 的次数进行重复扩展，以匹配 Mask 的逻辑。
        Args:
            vid: [B, T, C, H, W]
            act: [B, T, ...] (动作序列)
            p: 使用的 GT 帧数量 (必须能被 T 整除)
        """
        B, T = vid.shape[:2]
        repeats = T // p
        gt_vid = vid[:, -p:, ...]  # shape: [B, p, ...]
        gt_pose = pose[:, -p:, ...]        # shape: [B, p, ...]
        # 逻辑: [GT1, GT2] -> [GT1, GT1..., GT2, GT2...]
        gt_extended_vid = torch.repeat_interleave(gt_vid, repeats, dim=1)  # shape: [B, p * repeats, ...] = [B, T, ...]
        gt_extended_pose = torch.repeat_interleave(gt_pose, repeats, dim=1) # shape: [B, T, ...]
        vid_with_repeat = torch.cat([vid, gt_extended_vid], dim=1)
        pose_with_repeat = torch.cat([pose, gt_extended_pose], dim=1)

        return vid_with_repeat, pose_with_repeat

    def forward(self, vid, act, p, batch):
        t = torch.randint(0, self.diffusion_config.train_sampling_steps, (vid.shape[0], vid.shape[1]), generator=self.seed_generator, device=self.device).long()
        if self.diffusion_config.weighting_scheme in ["logit_normal"]:
            # adapting from diffusers.training_utils

            u = compute_density_for_timestep_sampling(
                weighting_scheme=self.diffusion_config.weighting_scheme,
                size=(vid.shape[0], vid.shape[1]),
                logit_mean=self.diffusion_config.logit_mean,
                logit_std=self.diffusion_config.logit_std,
                mode_scale=None,  # not used
            )
            t_logit = (u * self.diffusion_config.train_sampling_steps).long().to(vid.device)
            half_frames = vid.shape[1] // 2
            t[:, half_frames:] = t_logit[:, half_frames:]
            # t = (u * self.diffusion_config.train_sampling_steps).long().to(vid.device)
            # mask = torch.bernoulli(torch.full((vid.shape[0], half_frames), self.p_set_t_clean, device=vid.device)).bool()
            # t[:, :half_frames][mask] = 0

        if self.enable_cycle_forcing:
            loss_dict = self.diffusion.training_losses_32(self.model, vid, t, model_kwargs=dict(act=act, p=p))
        else:
            loss_dict = self.diffusion.training_losses(self.model, vid, t, model_kwargs=dict(act=act))

        loss = loss_dict["loss"].mean()

        return loss

    def training_step(self, batch, batch_idx):
        # batch[self.video_key]: (B, T, C, H, W), -1~1
        # batch[self.label_key]: (B, T, 16)
        vid, act = self.get_input(batch)
        current_epoch = self.current_epoch
        if not self.tokenizer_encode_after_batch_transfer:  # enter by default
            vid = self.tokenizer_encode(vid)
        h, w = vid.shape[-2], vid.shape[-1]

        if self.enable_cycle_forcing:
            self.model.eval()
            with torch.no_grad():
                B, T = vid.shape[:2]
                original_dtype = vid.dtype
                p = max(2, 32 // (4 ** current_epoch))
                # if current_epoch == 0:
                #     p = 32
                # elif current_epoch == 1:
                #     p = random.choices([32, 8], weights=[0.3, 0.7])[0]
                # else:  # epoch >= 2
                #     p = random.choices([32, 4, 2], weights=[0.1, 0.1, 0.8])[0]
                rollout_vid = self.generate_v_flow_shift_dpm(
                    model=self.model,
                    vid=vid,
                    act=act,
                    max_frames=self.max_frames,
                    n_prompt_frames=p,
                    chunk_size=T-p,
                    total_frames=T,
                    add_noise_to_ctx=True,
                    use_kv_cache=self.enable_kv_cache,
                    desc=None
                )
            self.model.train()
            self.model.clear_cache()
            rollout_vid = rollout_vid.to(original_dtype).detach().clone().requires_grad_(True)
            reversed_rollout_vid = torch.flip(rollout_vid, dims=[1])
            reversed_act = torch.flip(act, dims=[1])
            reversed_pose = self.compute_plucker_in_window(reversed_act, h, w)
            final_vid, final_pose = self.prepare_inputs(reversed_rollout_vid, reversed_pose, p)
            loss = self(final_vid, final_pose, p, batch)
        else:
            if self.use_plucker:
                pose = self.compute_plucker_in_window(act, h, w)
            else:
                pose = self.compute_rt_in_window(act)
            self.model.clear_cache()
            loss = self(vid, pose, batch)

        loss_dict = {"train/loss": loss}
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train/global_step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.scheduler_config is not None:
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("train/lr", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        if self.log_time_profile:
            self.log_time(prefix='train')
        return loss

    def validation_step(self, batch, batch_idx):
        # batch[self.video_key]: (B, T, C, H, W), -1~1
        # batch[self.label_key]: (B, T, 25)
        vid, act = self.get_input(batch)
        vid = vid[:, :self.val_num_total_frames]
        act = act[:, :self.val_num_total_frames]

        if not self.tokenizer_encode_after_batch_transfer:  # enter by default
            vid = self.tokenizer_encode(vid)

        samples_v_dpm = self.generate_v_flow_shift_dpm(
            model=self.model,
            vid=vid,
            act=act,
            max_frames=self.max_frames,
            n_prompt_frames=self.val_num_prompt_frames,
            total_frames=self.val_num_total_frames,
            chunk_size=1,
            add_noise_to_ctx=True,
            use_kv_cache=False,
            desc=f'validation_step - {batch_idx}'
        )

        # remove the first ground truth frame
        min_num_frames = min(vid.shape[1], samples_v_dpm.shape[1])
        samples_v_dpm = samples_v_dpm[:, self.val_num_prompt_frames:min_num_frames]
        vid = vid[:, self.val_num_prompt_frames:min_num_frames]
        # detokenize, and reshape
        samples_v_dpm = rearrange(self.tokenizer_decode(samples_v_dpm.to(vid.dtype)), "b t c h w -> (b t) c h w")
        dec = rearrange(self.tokenizer_decode(vid), "b t c h w -> (b t) c h w")

        dec, samples_v_dpm = map(lambda x: (x + 1) / 2, (dec.clamp(-1, 1), samples_v_dpm.clamp(-1, 1)))  # [-1, 1] to [0, 1]
        if self.save_fd is not None:
            rank = self.global_rank if hasattr(self, 'global_rank') else 0
            generated_to_save = rearrange(samples_v_dpm, "t c h w -> t h w c")
            gt_to_save = rearrange(dec, "t c h w -> t h w c")
            os.makedirs(f'{self.save_fd}/input', exist_ok=True)
            os.makedirs(f'{self.save_fd}/output', exist_ok=True)
            os.makedirs(f'{self.save_fd}/comparison', exist_ok=True)

        filename_suffix = f"rank{rank}_batch{batch_idx}"
        write_video(
            f'{self.save_fd}/output/sample_{filename_suffix}.mp4',
            (generated_to_save.cpu().numpy() * 255).astype(np.uint8),
            fps=30
        )
        write_video(
            f'{self.save_fd}/input/sample_{filename_suffix}.mp4',
            (gt_to_save.cpu().numpy() * 255).astype(np.uint8),
            fps=30
        )
        gt_frames_for_gif = (gt_to_save.cpu().numpy() * 255).astype(np.uint8)
        generated_frames_for_gif = (generated_to_save.cpu().numpy() * 255).astype(np.uint8)
        create_comparison_gif(
            gt_frames_for_gif,
            generated_frames_for_gif,
            f'{self.save_fd}/comparison/sample_{filename_suffix}.gif',
            fps=20
        )
        # metrics update
        self.update_val_metrics(dec, samples_v_dpm)

    @torch.inference_mode()
    def update_val_metrics(self, real, fake):
        # metrics update
        self.val_metrics["metrics_v_dpm"].update(real, fake)
        self.val_metrics["fid_v_dpm"].update(real, real=True)
        self.val_metrics["fid_v_dpm"].update(fake, real=False)
        self.val_metrics["is_v_dpm"].update(fake)


    def on_validation_epoch_start(self):
        # memory cleanup
        gc.collect()
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        # https://lightning.ai/docs/torchmetrics/stable/pages/lightning.html
        # sync_dist not working here
        self.log_dict(self.val_metrics["metrics_v_dpm"].compute())
        self.log(f"val_v_dpm/fid_v_dpm", self.val_metrics["fid_v_dpm"].compute())
        self.log(f"val_v_dpm/is_v_dpm", self.val_metrics["is_v_dpm"].compute()[0])

        # reset metrics
        self.val_metrics["metrics_v_dpm"].reset()
        self.val_metrics["fid_v_dpm"].reset()
        self.val_metrics["is_v_dpm"].reset()

        # memory cleanup
        torch.distributed.barrier()
        gc.collect()
        torch.cuda.empty_cache()



    def instantiate_optimizer_from_config(self, params, lr, cfg):
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )

    def configure_optimizers(self):
        lr = self.learning_rate
        # for deepspeed: https://github.com/microsoft/DeepSpeed/issues/2736, https://github.com/lean-dojo/ReProver/issues/66
        for p in self.model.parameters(): p.data = p.data.contiguous()
        params = list(self.model.parameters())

        opt = self.instantiate_optimizer_from_config(params, lr, self.optimizer_config)
        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print0(f"{self.print_prefix} Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)
        # manually warm up lr without a scheduler
        if self.trainer.global_step < self.scheduler_warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.scheduler_warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.learning_rate


    def dpm_solver_first_update(self, x, s, t, model_s=None, noise_schedule = None, return_intermediate=False):
        """
        DPM-Solver-1 (equivalent to DDIM) from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            model_s: A pytorch tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s`.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """

        ns = noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)
        phi_1 = torch.expm1(-h)
        x_t = sigma_t / sigma_s * x - alpha_t * phi_1 * model_s

        return x_t

    def multistep_dpm_solver_second_update(self, x, model_prev_list, t_prev_list, t, noise_schedule = None, solver_type="dpmsolver"):
        """
        Multistep solver DPM-Solver-2 from time `t_prev_list[-1]` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
            t: A pytorch tensor. The ending time, with the shape (1,).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """

        if solver_type not in ["dpmsolver", "taylor"]:
            raise ValueError(f"'solver_type' must be either 'dpmsolver' or 'taylor', got {solver_type}")
        ns = noise_schedule
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        t_prev_1, t_prev_0 = t_prev_list[-2], t_prev_list[-1]



        lambda_prev_1, lambda_prev_0, lambda_t = (
            ns.marginal_lambda(t_prev_1),
            ns.marginal_lambda(t_prev_0),
            ns.marginal_lambda(t),
        )
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        D1_0 = (1.0 / r0) * (model_prev_0 - model_prev_1)

        phi_1 = torch.expm1(-h)
        x_t = (sigma_t / sigma_prev_0) * x - (alpha_t * phi_1) * model_prev_0 - 0.5 * (alpha_t * phi_1) * D1_0

        return x_t

    def get_model_input_time(self, t_continuous, noise_schedule):
        """
        Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.
        For discrete-time DPMs, we convert `t_continuous` in [1 / N, 1] to `t_input` in [0, 1000 * (N - 1) / N].
        For continuous-time DPMs, we just use `t_continuous`.
        """
        if noise_schedule.schedule == "discrete":
            return (t_continuous - 1.0 / noise_schedule.total_N) * noise_schedule.total_N
        elif noise_schedule.schedule == "discrete_flow":
            return t_continuous * noise_schedule.total_N
        else:
            return t_continuous

    # @torch.inference_mode()
    # @torch.no_grad()
    @torch.no_grad()
    def generate_v_flow_shift_dpm(
            self,
            model,
            vid,
            act,
            n_prompt_frames: int = 1,
            chunk_size: int = 1,
            total_frames: int = 32, #generation length
            denoise_steps: int = 18,
            noise_abs_max: int = 20,
            use_half_precision: bool = False,
            max_frames: int = 32, #windows length
            add_noise_to_ctx: bool = True,
            use_kv_cache: bool = False,
            desc: Union[str, None] = "",  # if None, turn off progress bar
    ):
        # using dpm when model output prediction is "flow"
        # vid: (B, T, C, H, W), -1~1
        # vid: (bs, 1, 16, 18, 32)
        # act: (bs, 32, 25)
        if chunk_size <= 0:
            return vid[:, :n_prompt_frames]
        model.clear_cache()

        B = vid.shape[0]
        x = vid[:, :n_prompt_frames]
        total_frames = min(total_frames, act.shape[1])

        if use_kv_cache:
            h, w = x.shape[-2], x.shape[-1]
            patches_per_frame = h * w // 4  # patch size = 1，2，2
            max_seq_len = max_frames * patches_per_frame + 1

            with torch.device(vid.device):
                model.setup_cache(B, max_seq_len, dtype=torch.bfloat16)

        # get alphas
        noise_schedule = NoiseScheduleFlow(schedule="discrete_flow")
        t_start=None
        t_end = None
        t_0 = 1.0 / noise_schedule.total_N if t_end is None else t_end
        t_T = noise_schedule.T if t_start is None else t_start
        N = denoise_steps

        timesteps = torch.linspace(t_T , t_0, N + 1).to(vid.device)
        timesteps = 1.0 - timesteps
        timesteps = (3.0 * timesteps / (1 + (3.0 -1) * timesteps)).flip(dims=[0])

        # sampling loop
        # pbar = range(n_prompt_frames, total_frames)
        pbar = range(n_prompt_frames, total_frames, chunk_size)
        pbar = tqdm(pbar, desc=f"{self.print_prefix} Sampling Frames ({desc})") if desc is not None else pbar

        for i in pbar:
            # chunk = torch.randn((B, 1, *x.shape[-3:]), device=vid.device)
            frames_to_generate = min(chunk_size, total_frames - i)
            chunk = torch.randn((B, frames_to_generate, *x.shape[-3:]), device=vid.device)

            chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
            # append noise to the end of the sequence
            x = torch.cat([x, chunk], dim=1)
            # start_frame = max(0, i + 1 - max_frames)  # model.max_frames = 32
            start_frame = max(0, i + frames_to_generate - max_frames)

            # window_act_abs = act[:, start_frame : i + 1]
            window_act_abs = act[:, start_frame : min(i + frames_to_generate, total_frames)]

            h, w = x.shape[-2], x.shape[-1]
            if self.use_plucker:
                window_act_rel = self.compute_plucker_in_window(window_act_abs, h, w)
            else:
                window_act_rel = self.compute_rt_in_window(window_act_abs)

            model_prev_list = []
            t_prev_list = []
            x_pred = None

            step = 0
            if add_noise_to_ctx:
                ctx_step = N - 1
            else:
                ctx_step = N

            t = timesteps[step]
            t_ctx = timesteps[ctx_step]

            alpha_t_ctx = noise_schedule.marginal_alpha(t_ctx)
            sigma_t_ctx = noise_schedule.marginal_std(t_ctx)
            t_1d = t.clone()
            t = self.get_model_input_time(t, noise_schedule)
            # t = torch.full((B, 1), t, device=vid.device)
            t = torch.full((B, frames_to_generate), t, device=vid.device)
            t_ctx = self.get_model_input_time(t_ctx, noise_schedule)
            t_ctx = torch.full((B, i), t_ctx, device=vid.device)

            t = torch.cat([t_ctx, t], dim=1)
            x_curr = x.clone()
            x_curr = x_curr[:, start_frame:]
            t = t[:, start_frame:]

            if add_noise_to_ctx:
                ctx_noise = torch.randn_like(x_curr[:, :-frames_to_generate])
                ctx_noise = torch.clamp(ctx_noise, -noise_abs_max, +noise_abs_max)
                x_curr[:, : -frames_to_generate] = expand_dims(alpha_t_ctx, x.dim()) * x_curr[:, :-frames_to_generate] + expand_dims(sigma_t_ctx, x.dim()) * ctx_noise

            if use_kv_cache:
                # num_ctx_frames = x_curr.shape[1] - 1
                num_ctx_frames = x_curr.shape[1] - frames_to_generate
                input_pos = num_ctx_frames * patches_per_frame

            with torch.no_grad():
                if use_half_precision:
                    with torch.autocast("cuda", dtype=torch.half):
                        if use_kv_cache:
                            v = model(x_curr, t, window_act_rel, input_pos=input_pos)
                        else:
                            v = model(x_curr, t, window_act_rel)
                else:
                    if use_kv_cache:
                        v = model(x_curr, t, window_act_rel, input_pos=input_pos)
                    else:
                        v = model(x_curr, t, window_act_rel)

            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_1d), noise_schedule.marginal_std(t_1d) # same shape as t
            noise = (1 - expand_dims(sigma_t, x.dim())) * v + x_curr
            x_start = (x_curr - sigma_t * noise) / alpha_t

            model_prev_list = [x_start]
            t_prev_list = [t_1d]

            for step in range(1, 2):# 2 is Order
                t = timesteps[step]
                t_ctx = timesteps[ctx_step]
                t_1d = t.clone()
                s = t_prev_list[-1] #1d
                model_s = model_prev_list[-1]
                x_pred = self.dpm_solver_first_update(x = x.clone()[:, start_frame:], s = s, t = t_1d, model_s = model_s, noise_schedule = noise_schedule)

                # x[:, -1:] = x_pred[:, -1:]
                x[:, -frames_to_generate:] = x_pred[:, -frames_to_generate:]

                t_prev_list.append(t_1d)

                t = self.get_model_input_time(t, noise_schedule)
                # t = torch.full((B, 1), t, device=vid.device)
                t = torch.full((B, frames_to_generate), t, device=vid.device)

                t_ctx = self.get_model_input_time(t_ctx, noise_schedule)
                t_ctx = torch.full((B, i), t_ctx, device=vid.device)

                t = torch.cat([t_ctx, t], dim=1)

                x_curr = x.clone()
                x_curr = x_curr[:, start_frame:]
                t = t[:, start_frame:]

                if add_noise_to_ctx:
                    # ctx_noise = torch.randn_like(x_curr[:, :-frames_to_generate])
                    ctx_noise = torch.clamp(ctx_noise, -noise_abs_max, +noise_abs_max)
                    x_curr[:, : -frames_to_generate] = expand_dims(alpha_t_ctx, x.dim()) * x_curr[:, :-frames_to_generate] + expand_dims(sigma_t_ctx, x.dim()) * ctx_noise

                with torch.no_grad():
                    if use_half_precision:
                        with torch.autocast("cuda", dtype=torch.half):
                            if use_kv_cache:
                                v = model(x_curr[:, -frames_to_generate:], t[:, -frames_to_generate:], window_act_rel[:, -frames_to_generate:])
                                if frames_to_generate == 1:
                                    pass
                                else:
                                    v_last = v[:, -1:]  # (B, 1, C, H, W)
                                    v_broadcast = v_last.expand(-1, x_curr.shape[1] - frames_to_generate, -1, -1, -1)
                                    v = torch.cat([v_broadcast, v], dim=1)
                            else:
                                v = model(x_curr, t, window_act_rel)
                    else:
                        if use_kv_cache:
                            v = model(x_curr[:, -frames_to_generate:], t[:, -frames_to_generate:], window_act_rel[:, -frames_to_generate:])
                            if frames_to_generate == 1:
                                pass
                            else:
                                v_last = v[:, -1:]  # (B, 1, C, H, W)
                                v_broadcast = v_last.expand(-1, x_curr.shape[1] - frames_to_generate, -1, -1, -1)
                                v = torch.cat([v_broadcast, v], dim=1)
                        else:
                            v = model(x_curr, t, window_act_rel)

                alpha_t, sigma_t = noise_schedule.marginal_alpha(t_1d), noise_schedule.marginal_std(t_1d) # same shape as t

                noise = (1 - expand_dims(sigma_t, x.dim())) * v + x_curr
                x_start = (x_curr - sigma_t * noise) / alpha_t

                model_prev_list.append(x_start)


            for step in range(2, N+1): # 2 is order
                t = timesteps[step]
                t_ctx = timesteps[ctx_step]

                t_1d = t.clone()
                step_order = None
                if step == N:
                    step_order = 1
                else:
                    step_order = 2
                x_pred = None
                if step_order == 2:
                    x_pred = self.multistep_dpm_solver_second_update(x =  x.clone()[:, start_frame:], model_prev_list = model_prev_list, t_prev_list = t_prev_list, t = t_1d, noise_schedule = noise_schedule, solver_type = "dpmsolver")
                else:
                    x_pred = self.dpm_solver_first_update(x =  x.clone()[:, start_frame:], s = t_prev_list[-1], t = t_1d, model_s = model_prev_list[-1], noise_schedule = noise_schedule) #有待检查

                # x[:, -1:] = x_pred[:, -1:]
                x[:, -frames_to_generate:] = x_pred[:, -frames_to_generate:]

                t_prev_list.append(t_1d)

                # calculate new noise for current step
                if step < N:
                    t = self.get_model_input_time(t, noise_schedule)
                    # t = torch.full((B, 1), t, device=vid.device).long()
                    t = torch.full((B, frames_to_generate), t, device=vid.device).long()

                    t_ctx = self.get_model_input_time(t_ctx, noise_schedule).long()
                    t_ctx = torch.full((B, i), t_ctx, device=vid.device)

                    t = torch.cat([t_ctx, t], dim=1)
                    x_curr = x.clone()
                    x_curr = x_curr[:, start_frame:]
                    t = t[:, start_frame:]

                    if add_noise_to_ctx:
                        # ctx_noise = torch.randn_like(x_curr[:, :-frames_to_generate])
                        ctx_noise = torch.clamp(ctx_noise, -noise_abs_max, +noise_abs_max)
                        x_curr[:, : -frames_to_generate] = expand_dims(alpha_t_ctx, x.dim()) * x_curr[:, :-frames_to_generate] + expand_dims(sigma_t_ctx, x.dim()) * ctx_noise

                    with torch.no_grad():
                        if use_half_precision:
                            with torch.autocast("cuda", dtype=torch.half):
                                if use_kv_cache:
                                    v = model(x_curr[:, -frames_to_generate:], t[:, -frames_to_generate:], window_act_rel[:, -frames_to_generate:])
                                    if frames_to_generate == 1:
                                        pass
                                    else:
                                        v_last = v[:, -1:]  # (B, 1, C, H, W)
                                        v_broadcast = v_last.expand(-1, x_curr.shape[1] - frames_to_generate, -1, -1, -1)
                                        v = torch.cat([v_broadcast, v], dim=1)
                                else:
                                    v = model(x_curr, t, window_act_rel)
                        else:
                            if use_kv_cache:
                                v = model(x_curr[:, -frames_to_generate:], t[:, -frames_to_generate:], window_act_rel[:, -frames_to_generate:])
                                if frames_to_generate == 1:
                                    pass
                                else:
                                    v_last = v[:, -1:]  # (B, 1, C, H, W)
                                    v_broadcast = v_last.expand(-1, x_curr.shape[1] - frames_to_generate, -1, -1, -1)
                                    v = torch.cat([v_broadcast, v], dim=1)
                            else:
                                v = model(x_curr, t, window_act_rel)

                    alpha_t, sigma_t = noise_schedule.marginal_alpha(t_1d), noise_schedule.marginal_std(t_1d) # same shape as t

                    noise = (1 - expand_dims(sigma_t, x.dim())) * v + x_curr
                    x_start = (x_curr - sigma_t * noise) / alpha_t

                    model_prev_list.append(x_start)

        if torch.isnan(x).any() or torch.isinf(x).any():
            print0(f"{self.print_prefix}[generate] NaN / Inf detected in generated samples. Replacing with zeros.")
            x = torch.zeros_like(x)
        return x


    @torch.inference_mode()
    def log_time(self, prefix: str = 'train'):
        # need simple profiler
        # each record is a list of durations recoding each iter, so we take the last (most recent) one
        train_dataloader_next = self.trainer.profiler.recorded_durations.get("[_TrainingEpochLoop].train_dataloader_next", None)
        train_dataloader_next = train_dataloader_next[-1] if train_dataloader_next is not None else None
        # run_training_batch > optimizer_step
        # optimizer_step ~= [Strategy]DDPStrategy.training_step + [Strategy]DDPStrategy.backward
        run_training_batch = self.trainer.profiler.recorded_durations.get("run_training_batch", None)
        run_training_batch = run_training_batch[-1] if run_training_batch is not None else None
        optimizer_step = self.trainer.profiler.recorded_durations.get(f"[LightningModule]{self.__class__.__name__}.optimizer_step", None)
        optimizer_step = optimizer_step[-1] if optimizer_step is not None else None
        # val
        val_next = self.trainer.profiler.recorded_durations.get("[_EvaluationLoop].val_next", None)
        val_next = val_next[-1] if val_next is not None else None
        # self defined
        tokenizer_encode = self.trainer.profiler.recorded_durations.get(f"[{self.__class__.__name__}].tokenizer_encode", None)
        tokenizer_encode = tokenizer_encode[-1] if tokenizer_encode is not None else None
        tokenizer_decode = self.trainer.profiler.recorded_durations.get(f"[{self.__class__.__name__}].tokenizer_decode", None)
        tokenizer_decode = tokenizer_decode[-1] if tokenizer_decode is not None else None
        # log to tensorboard / wandb
        if train_dataloader_next is not None:
            self.log(f"{prefix}/time-train_dataloader_next", train_dataloader_next)
        if optimizer_step is not None:
            self.log(f"{prefix}/time-optimizer_step", optimizer_step)
        if tokenizer_encode is not None:
            self.log(f"{prefix}/time-tokenizer_encode", tokenizer_encode)
        if tokenizer_decode is not None:
            self.log(f"{prefix}/time-tokenizer_decode", tokenizer_decode)


    @torch.inference_mode()
    def log_videos(
        self,
        batch: Dict,
        N: int = 8,
        log_num_prompt_frames: int = 1,
        log_num_total_frames: int = 16,
        **kwargs,
    ) -> Dict:
        log = dict()

        # batch[self.video_key]: (B, T, C, H, W), -1~1
        # batch[self.label_key]: (B, T, 25)
        vid, act = self.get_input(batch)
        vid = vid[:N, :log_num_total_frames]
        act = act[:N, :log_num_total_frames]
        if not self.tokenizer_encode_after_batch_transfer:  # enter by default
            vid = self.tokenizer_encode(vid)
        recs = self.tokenizer_decode(vid)

        samples = self.generate_v_flow_shift_dpm(
            model=self.model,
            vid=vid,
            act=act,
            max_frames=self.max_frames,
            n_prompt_frames=log_num_prompt_frames,
            total_frames=log_num_total_frames,
            chunk_size=1,
            add_noise_to_ctx=True,
            use_kv_cache=False,
            desc='log_videos'
        )[:N]


        samples = self.tokenizer_decode(samples.to(vid.dtype))

        min_num_frames = min(recs.shape[1], samples.shape[1])
        log["recs_samples_v_dpm"] = torch.cat([recs[:, :min_num_frames], samples[:, :min_num_frames]], dim=4)

        # memory cleanup
        gc.collect()
        torch.cuda.empty_cache()
        return log

import os
import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from omegaconf import OmegaConf

from scipy import linalg

import torch
import torchvision
from inception import InceptionV3

import lightning.pytorch as pl
from einops import rearrange
from lightning.pytorch import seed_everything

from sgm.util import exists, instantiate_from_config, print0, get_valid_dirs
# from tvideo.mc.data.mc import MCDatasetValStd
from tvae.modules.metrics import compute_psnr, compute_ssim
from tvae.taming.lpips import LPIPS

from torchvision import transforms
from torchvision.io import write_video

from typing import Optional

import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tvideo.mc.data.re10k_adapter import RealEstate10KValidation
from tvideo.mc.data.collectfn import dict_collation_fn
from PIL import Image
import random

"""Inference and metric evaluation for the released RealEstate10K model."""

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def compute_lpips_in_chunks(perceptual_loss, dec, samples, chunk_size=8):
    """分批计算 LPIPS 以避免 OOM"""
    num_frames = dec.shape[0]
    lpips_values = []

    for i in range(0, num_frames, chunk_size):
        end_idx = min(i + chunk_size, num_frames)
        chunk_dec = dec[i:end_idx]
        chunk_samples = samples[i:end_idx]

        with torch.no_grad():
            chunk_lpips = perceptual_loss(
                chunk_dec * 2 - 1,
                chunk_samples * 2 - 1
            )
            lpips_values.append(chunk_lpips.cpu())

        # 清理显存
        torch.cuda.empty_cache()

    return torch.cat(lpips_values).mean()

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


def load_model_from_config(config, ckpt, vae=None):
    config = OmegaConf.load(config)
    config.model.params.ckpt_path = ckpt
    if vae is not None:
        config.model.params.tokenizer_config.params.ckpt_path = vae
        config.model.params.tokenizer_config.params.ckpt_path2 = vae
    config.model.params.strict_loading = False
    model = instantiate_from_config(config.model)
    return model


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/live_re10k.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/live-re10k.ckpt",
        help="path to checkpoint of model",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/real-estate-10k",
        help="path to data directory",
    )

    parser.add_argument(
        "--filelist_path",
        type=str,
        default="file_list.txt",
        help="path to filelist",
    )

    parser.add_argument(
        "--save_fd",
        type=str,
        default="outputs/evaluation",
        help="path to save samples",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch size",
    )

    parser.add_argument(
        "--num_steps",
        type=int,
        default=18,
        help="number of diffusion steps",
    )

    parser.add_argument(
        "--val_num_total_frames",
        type=int,
        default=100,
        help="number of sampled/model frames (100 with skip 2 covers the paper's 0-200 raw-frame horizon)",
    )

    parser.add_argument(
        "--val_num_prompt_frames",
        type=int,
        default=1,
        help="val_num_prompt_frames",
    )

    parser.add_argument(
        "--max_frames",
        type=int,
        default=32,
        help="max_frames",
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1,
        help="max_frames",
    )
    parser.add_argument("--vae", type=str, default="checkpoints/vae-kl16.ckpt")
    parser.add_argument("--max_videos", type=int, default=0, help="0 evaluates the full split")
    parser.add_argument(
        "--save_videos",
        action="store_true",
        help="write input/output MP4 files and comparison GIFs",
    )
    parser.add_argument(
        "--compute_fid",
        action="store_true",
        help="compute frame-level FID (stores all Inception features in memory)",
    )

    args = parser.parse_args()
    if args.batch_size != 1:
        parser.error("--batch_size must be 1 for the released Re10K evaluator")
    if args.max_videos < 0:
        parser.error("--max_videos must be non-negative")
    seed_everything(args.seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = load_model_from_config(args.config, args.ckpt, args.vae)
    #model.strict_loading=False

    model.to(device).eval()

    # dataset = MCDatasetValStd(
    #                  data_dir = args.data_dir,
    #                  filelist_path = args.filelist_path,
    #                  is_latent_input = False,
    #                  is_action_index = False,
    #                  max_frame_per_video = 64,
    #                  video_key = "mp4")

    dataset = RealEstate10KValidation(
        save_dir=args.data_dir,
        val_n_frames=args.val_num_total_frames,
        sampling_interval=2,
        video_H=256,
        video_W=256,
        use_exact_frame_span=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=False,
        collate_fn=dict_collation_fn,
    )

    perceptual_loss = LPIPS().eval()
    perceptual_loss = perceptual_loss.to(device)

    if args.save_videos:
        os.makedirs(f'{args.save_fd}/input', exist_ok=True)
        os.makedirs(f'{args.save_fd}/output', exist_ok=True)
        os.makedirs(f'{args.save_fd}/comparison', exist_ok=True)

    inception_model = None
    if args.compute_fid:
        dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        inception_model = InceptionV3([block_idx]).to(device).eval()

    psnrs, ssims, lpipss = [], [], []
    pred_xs, pred_recs = [], []

    fpss = []

    with torch.no_grad():
        for i, input in enumerate(dataloader):
            if args.max_videos and i * args.batch_size >= args.max_videos:
                break
            # video_name = input['video_name']
            # del input['video_name']
            # input = {k: v.to(device) for k, v in input.items() if k != 'video_name'}
            vid, act = model.get_input(input)
            vid = vid.to(device)
            act = act.to(device)

            vid = vid[:, :args.val_num_total_frames]
            act = act[:, :args.val_num_total_frames]

            if args.save_videos:
                vid_to_save = vid[:, args.val_num_prompt_frames:]
                vid_to_save = rearrange(vid_to_save, "b t c h w -> (b t) c h w")
            if not model.tokenizer_encode_after_batch_transfer:  # enter by default
                vid = model.tokenizer_encode(vid)

            import time
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            samples = model.generate_v_flow_shift_dpm(
            model=model.model,
            vid=vid,
            act=act,
            max_frames=args.max_frames,
            n_prompt_frames=args.val_num_prompt_frames,
            total_frames=args.val_num_total_frames,
            chunk_size=args.chunk_size,
            add_noise_to_ctx=True,
            use_kv_cache=False,
            use_half_precision=device.type == "cuda",
            desc=f'validation_step - {i}',
            denoise_steps = args.num_steps,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            generated_time_wo_cache = time.time() - start
            val_num_total_frames = vid.shape[1]
            fps_wo_cache = (val_num_total_frames - args.val_num_prompt_frames) / generated_time_wo_cache
            print0(f'[bold red]Time for inference without cache: {generated_time_wo_cache} seconds')
            print0(f'[bold red]FPS without cache: {fps_wo_cache}')

            if i != 0:
                fpss.append(fps_wo_cache)

            # remove the first ground truth frame
            min_num_frames = min(vid.shape[1], samples.shape[1])
            samples = samples[:, args.val_num_prompt_frames:min_num_frames]

            vid = vid[:, args.val_num_prompt_frames:min_num_frames]
            samples = rearrange(model.tokenizer_decode(samples.to(vid.dtype)), "b t c h w -> (b t) c h w")

            dec = rearrange(model.tokenizer_decode(vid), "b t c h w -> (b t) c h w")

            dec, samples  = map(lambda x: (x + 1) / 2, (dec.clamp(-1, 1), samples.clamp(-1, 1)))  # [-1, 1] to [0, 1]

            if args.save_videos:
                samples_to_save = rearrange(samples, "t c h w -> t h w c")
                vid_to_save = (vid_to_save + 1) / 2
                vid_to_save = rearrange(vid_to_save, "t c h w -> t h w c")

                write_video(
                    f'{args.save_fd}/output/samples_{i}.mp4',
                    (samples_to_save.cpu().numpy() * 255).astype(np.uint8),
                    fps=30,
                )
                write_video(
                    f'{args.save_fd}/input/samples_{i}.mp4',
                    (vid_to_save.cpu().numpy() * 255).astype(np.uint8),
                    fps=30,
                )
                gt_frames_for_gif = (vid_to_save.cpu().numpy() * 255).astype(np.uint8)
                generated_frames_for_gif = (samples_to_save.cpu().numpy() * 255).astype(np.uint8)
                create_comparison_gif(
                    gt_frames_for_gif,
                    generated_frames_for_gif,
                    f'{args.save_fd}/comparison/samples_{i}.gif',
                    fps=10
                )

            psnr = compute_psnr(dec, samples)
            ssim = compute_ssim(dec, samples)
            lpips = compute_lpips_in_chunks(perceptual_loss, dec, samples)

            print(f'PSNR: {psnr}, SSIM: {ssim}, LPIPS: {lpips}')

            psnrs.append(psnr.item())
            ssims.append(ssim.item())
            lpipss.append(lpips.item())

            if inception_model is not None:
                pred_x = inception_model(dec)[0]
                pred_x = pred_x.squeeze(3).squeeze(2).cpu().float().numpy()
                pred_samples = inception_model(samples)[0]
                pred_samples = pred_samples.squeeze(3).squeeze(2).cpu().float().numpy()
                pred_xs.append(pred_x)
                pred_recs.append(pred_samples)

    if args.compute_fid:
        pred_xs = np.concatenate(pred_xs, axis=0)
        pred_recs = np.concatenate(pred_recs, axis=0)
        m1, s1 = pred_xs.mean(axis=0), np.cov(pred_xs, rowvar=False)
        m2, s2 = pred_recs.mean(axis=0), np.cov(pred_recs, rowvar=False)
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        print0(f"[bold yellow]\[timg.mar.inference_evaluate][FID][/bold yellow] FID: {fid_value}")
    print0(f"[bold yellow]\[timg.mar.inference_evaluate][PSNR][/bold yellow] PSNR: {np.mean(psnrs)}")
    print0(f"[bold yellow]\[timg.mar.inference_evaluate][SSIM][/bold yellow] SSIM: {np.mean(ssims)}")
    print0(f"[bold yellow]\[timg.mar.inference_evaluate][LPIPS][/bold yellow] LPIPS: {np.mean(lpipss)}")
    if fpss:
        print0(f"[bold yellow]\[timg.mar.inference_evaluate][FPS][/bold yellow] FPS: {np.mean(fpss)}")
    else:
        print0("[bold yellow]FPS unavailable: only the warm-up sample was evaluated.[/bold yellow]")





if __name__ == "__main__":
    main()

<div align="center">

# LIVE: Long-horizon Interactive Video World Modeling

Official PyTorch release for **LIVE**, accepted at **ICML 2026**.

Junchao Huang, Ziyang Ye, Xinting Hu, Tianyu He, Guiyu Zhang, Shaoshuai Shi,
Jiang Bian, and Li Jiang.

[![arXiv](https://img.shields.io/badge/arXiv-2602.03747-b31b1b.svg)](https://arxiv.org/abs/2602.03747)
[![GitHub](https://img.shields.io/badge/GitHub-Code-blue.svg)](https://github.com/Junchao-cs/LIVE)
[![Project](https://img.shields.io/badge/Project-Page-blue.svg)](https://junchao-cs.github.io/LIVE-demo/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Checkpoints-yellow.svg)](https://huggingface.co/junchaoh-cs/LIVE-Re10K)

</div>

## News

- July 2026: LIVE is accepted at ICML 2026; RealEstate10K code and checkpoints are released.

> This release covers the RealEstate10K training, inference, and evaluation
> pipeline used by the main paper experiments. Minecraft and UE Engine assets,
> recipes, and checkpoints are not included in this release.

## Overview

Autoregressive video world models accumulate errors because inference conditions
on generated frames while conventional training conditions on ground truth.
LIVE post-trains an autoregressive diffusion model with a cycle-consistency
objective:

1. Start from `p` ground-truth prompt frames and generate a frozen forward
   rollout.
2. Reverse the rollout and camera conditions, then inject independent
   per-frame noise into the rollout context.
3. Recover the original prompt frames with a frame-level diffusion loss.
4. Progressively reduce `p` from `32` to `8` to `2` across the first three
   epochs, exposing the model to increasingly long generated context.

## Installation

The released environment uses Python 3.10, PyTorch 2.6.0, CUDA 12.4, and
Lightning 2.5.2.

```bash
conda env create -f environment.yaml
conda activate live
```

Alternatively, create a Python 3.10 environment and run
`pip install -r requirements.txt`.

The default attention backend is PyTorch SDPA. `xformers`, FlashAttention, and
Liger are not required by the released configuration.

Evaluation initializes LPIPS. Optional frame-level FID additionally initializes
Inception; public pretrained weights may be downloaded on first use, so prepare
the corresponding caches before running in an offline job. LPIPS first checks
`checkpoints/lpips/vgg.pth`; when absent, the current implementation downloads
its linear weights to `../ckpt/lpips/vgg.pth`. Torchvision's VGG weights use the
standard PyTorch cache.

## Checkpoints

Download all files into the paths expected by the configs:

```bash
huggingface-cli download junchaoh-cs/LIVE-Re10K --local-dir checkpoints
```

| Local file | Role | Recorded step |
|---|---|---:|
| [`checkpoints/live-re10k.ckpt`](https://huggingface.co/junchaoh-cs/LIVE-Re10K/blob/main/live-re10k.ckpt) | Final LIVE RealEstate10K checkpoint | 20,500 post-training steps |
| [`checkpoints/nfd-df-re10k.ckpt`](https://huggingface.co/junchaoh-cs/LIVE-Re10K/blob/main/nfd-df-re10k.ckpt) | Converged NFD-DF initializer for LIVE | 275,000 steps |
| [`checkpoints/nfd-tf-re10k.ckpt`](https://huggingface.co/junchaoh-cs/LIVE-Re10K/blob/main/nfd-tf-re10k.ckpt) | NFD Teacher Forcing baseline | 200,000 steps |
| [`checkpoints/vae-kl16.ckpt`](https://huggingface.co/junchaoh-cs/LIVE-Re10K/blob/main/vae-kl16.ckpt) | Shared 16x spatial KL-VAE | N/A |

<details>
<summary>Checkpoint SHA-256 and byte sizes</summary>

```text
f048e6077f51e615912ea57ee9992aeb1aa5d9cb23704f1b47887574ceb451fc  10344600162  nfd-df-re10k.ckpt
df89f9c4c4a925ca610e75d51b1d152e725cf62a169e09548cef5222e1331c57  10361387844  nfd-tf-re10k.ckpt
aa33f088f763bd1fab4909ffe0a61abe9ada5cc235c999e07bb370feefd8ba16  10436804929  live-re10k.ckpt
34ce001bcfffb7af67ec8af1e683a30d7bd45760855ddc7deedc1330f2cfd38f    265900046  vae-kl16.ckpt
```

</details>

These are full Lightning checkpoints, not model-only exports. The approximately
10 GB model checkpoints retain optimizer state, trainer state, and the original
run hyperparameters so training can be resumed. The VAE remains a separate
required file even though the Lightning checkpoints contain tokenizer-related
state.

## Data

We use the DFoT-prepared RealEstate10K data. The released loader reads only the
video and per-video pose trees; `captions/` and `metadata/` from a DFoT data
release may remain present but are not consumed here.

```text
data/
├── real-estate-10k/
│   ├── training_256/
│   │   └── part_XXXXX/*.mp4
│   ├── training_poses/
│   │   └── part_XXXXX/*.pt
│   ├── validation_256/
│   │   └── part_XXXXX/*.mp4
│   └── validation_poses/
│       └── part_XXXXX/*.pt
└── real-estate-10k-mini/
    ├── training_256/part_XXXXX/*.mp4
    ├── training_poses/part_XXXXX/*.pt
    ├── validation_256/part_XXXXX/*.mp4
    └── validation_poses/part_XXXXX/*.pt
```

Every video must have a `.pt` pose file with the same relative subdirectory and
stem. For example:

```text
training_256/part_00000/0000cc6d8b108390.mp4
training_poses/part_00000/0000cc6d8b108390.pt
```

The recorded training run used the full root for training and the mini root for
periodic validation. The mini split is a sanity-check subset and must not be
used to report the paper's complete-test-set metrics. Symlinks are sufficient:

```bash
mkdir -p data
ln -s /path/to/real-estate-10k data/real-estate-10k
ln -s /path/to/real-estate-10k-mini data/real-estate-10k-mini
```

If a separate mini split is unavailable, override the validation root with
`--data.params.validation.params.save_dir=/path/to/real-estate-10k`.

## Training

LIVE is post-trained from the released NFD-DF initializer; it is not trained
from scratch. The default config is the supported four-GPU recipe. If you
created the `data/` symlinks above, start training with:

```bash
bash train.sh
```

Alternatively, pass the dataset roots directly without creating symlinks. The
first path is the full RealEstate10K training root; the second is the mini
validation root:

```bash
bash train.sh \
  --data.params.train.params.save_dir=/path/to/real-estate-10k \
  --data.params.validation.params.save_dir=/path/to/real-estate-10k-mini
```

The four-GPU recipe uses per-device batch 8 with two-step gradient
accumulation, preserving an effective global batch of `4 x 8 x 2 = 64`. It
reproduces the recorded four-GPU recipe's `8e-5` learning rate.

The paper's reported run used 32 NVIDIA H100 GPUs, per-device batch 2, global
batch 64, and learning rate `4e-5`. For a four-node, eight-GPU-per-node launch,
merge the distributed override after the base config:

```bash
torchrun \
  --nnodes=4 \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  --nproc_per_node=8 \
  main.py \
  -b configs/live_re10k.yaml configs/live_re10k_32gpu.yaml \
  --logdir logs/live-re10k
```

The four-GPU recipe is provided for accessibility and preserves the effective
batch size; it is not the hardware configuration used for the reported paper
numbers. The 32-GPU override changes the learning rate to the paper run's
`4e-5`. Training logs use TensorBoard by default; add `--wandb` if Weights &
Biases logging is preferred. Logger selection does not alter optimization.

Resume a full Lightning checkpoint with the current config using:

```bash
bash train.sh --resume_from_checkpoint /path/to/checkpoint.ckpt
```

This restores the model, optimizer, scheduler, epoch, and global step. When the
new run uses a different log directory, Lightning starts fresh top-k checkpoint
bookkeeping in that directory.

To resume a run created by this release together with the configs saved in its
log directory, use `bash train.sh --resume logs/live-re10k/<run-name>` instead.
Pre-release internal log directories and direct Lightning
`LIVE.load_from_checkpoint(...)` construction are not supported; use the
current release config and `--resume_from_checkpoint` as shown above.

## Inference and Evaluation

Generate one 100-sampled-frame example and save the ground truth and prediction.
RealEstate10K is sampled with frame skip 2, so this is the paper's 0-200
raw-frame horizon:

```bash
bash infer.sh \
  --data_dir /path/to/real-estate-10k \
  --val_num_total_frames 100
```

Evaluate the complete validation tree with 18-step flow DPM sampling and a
32-frame sliding context:

```bash
python inference_evaluate.py \
  --ckpt checkpoints/live-re10k.ckpt \
  --vae checkpoints/vae-kl16.ckpt \
  --data_dir /path/to/real-estate-10k \
  --val_num_total_frames 100
```

PSNR, SSIM, and LPIPS are printed after evaluation. The direct evaluation
command does not write videos; add `--save_videos` to export MP4/GIF files to
`outputs/evaluation`. Use `--max_videos N` for a smoke run. Optional
`--compute_fid` retains all Inception features in memory, is not part of the
reported RealEstate10K protocol, and is not recommended for the complete
validation tree. The loader scans MP4 frame counts at startup, so initialization
on the full dataset can take time. The standalone evaluator uses the same
18-step generation path and frame selection as the recorded distributed
evaluation, without requiring a multi-node Lightning launch.

### Paper Metric Implementation

The quantitative metrics reported in the paper were computed with the external
[common_metrics_on_video_quality](https://github.com/CIntellifusion/common_metrics_on_video_quality)
toolkit:

```bash
git clone git@github.com:CIntellifusion/common_metrics_on_video_quality.git
```

For exact paper-metric reproduction, generate predictions and ground-truth
videos on the complete evaluation split with the protocol above, then evaluate
them with that toolkit following its README. The PSNR, SSIM, LPIPS, and optional
FID calculations built into `inference_evaluate.py` are convenient local
diagnostics; they are not the metric implementation used to produce the
paper's reported numbers.

## Reproducibility Notes

- Architecture: 774M DiT with Plucker camera conditioning.
- Resolution: 256 x 256; frame skip: 2; context window: 32 frames.
- Latent space: shared KL-VAE with 16x spatial downsampling.
- Sampler: 18-step flow DPM solver.
- LIVE initialization: converged NFD-DF checkpoint at step 275k.
- LIVE post-training: approximately 20k steps; released checkpoint at 20.5k.
- Final implementation: the first half of the cycle-training sequence keeps
  uniform timesteps and the second half uses logit-normal timesteps; both
  training rollout and validation use noisy context.

The paper describes the base checkpoint as trained for 200k+ steps until
convergence; the selected released initializer is the later 275k checkpoint.
LIVE's bounded-error language describes the cycle objective's training effect
and empirical behavior, not a formal guarantee for arbitrary rollout length.

## Repository Layout

```text
configs/                  Re10K four-GPU and paper-scale configs
sgm/                      Minimal shared instantiation/distribution utilities
tvae/                     KL-VAE implementation used by the release
datasets/datasets_short_inf/  Re10K video and pose loader
tvideo/mc/data/               Re10K adapter and Lightning data module
tvideo/mc/models/          LIVE wrapper and checkpoint-compatible core
tvideo/mc/modules/         DiT, attention masks, and flow diffusion modules
main.py                    Training and resume entry point
inference_evaluate.py      Inference, video export, and metric evaluation
```

## Citation

```bibtex
@misc{huang2026livelonghorizoninteractivevideo,
  title={LIVE: Long-horizon Interactive Video World Modeling},
  author={Junchao Huang and Ziyang Ye and Xinting Hu and Tianyu He and Guiyu Zhang and Shaoshuai Shi and Jiang Bian and Li Jiang},
  year={2026},
  eprint={2602.03747},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2602.03747},
}
```

## Acknowledgements and License

LIVE builds on NFD, Stability AI Generative Models, Latent Diffusion, NVIDIA
diffusion utilities, OpenAI diffusion implementations, LPIPS, and pytorch-fid.
See `NOTICE` for attribution details.

The repository is released under the Apache License 2.0. Third-party components
remain subject to their original licenses.

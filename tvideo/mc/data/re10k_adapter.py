import torch
import numpy as np
from omegaconf import DictConfig
from datasets.datasets_short_inf.video.realestate10k_video_dataset import RealEstate10KDataset

class RealEstate10KAdapter(RealEstate10KDataset):
    """
    RealEstate10K dataset adapter for the batch format expected by LIVE.
    """
    def __init__(self,
                 save_dir: str,
                 n_frames: int = 16,
                 video_H: int = 224,
                 video_W: int = 384,
                 val_n_frames: int = None,
                 sampling_interval: int = 1,
                 validation_multiplier: int = 1,
                 use_exact_frame_span: bool = False,
                 split: str = "training",
                 **kwargs):

        cfg = DictConfig({
            'save_dir': save_dir,
            'n_frames': n_frames,
            'video_H': video_H,
            'video_W': video_W,
            'val_n_frames': val_n_frames if val_n_frames is not None else n_frames,
            'sampling_interval': sampling_interval,
            'validation_multiplier': validation_multiplier,
            'use_exact_frame_span': use_exact_frame_span,
        })

        super().__init__(cfg, split)

    def __getitem__(self, idx):
        """
        返回nfd期望的格式：包含video和action的字典
        """
        video, actions = super().__getitem__(idx)

        return {
            'npy': video,
            'json': actions,
        }

class RealEstate10KTrain(RealEstate10KAdapter):
    def __init__(self, **kwargs):
        super().__init__(split="training", **kwargs)

class RealEstate10KValidation(RealEstate10KAdapter):
    def __init__(self, **kwargs):
        super().__init__(split="validation", **kwargs)

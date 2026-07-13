import os
import cv2
import io
import tarfile
import numpy as np
import torch
import json
from typing import Sequence, Mapping
from omegaconf import DictConfig
import random

import torchvision.transforms as transforms
from sgm.util import instantiate_from_config

from .base_video_dataset import BaseVideoDataset

# Dataset class
class RealEstate10KDataset(BaseVideoDataset):
    """
    RealEstate10K video dataset for training and validation.

    Args:
        cfg (DictConfig): Configuration object.
        split (str): Dataset split ("training" or "validation").
    """
    def __init__(self, cfg: DictConfig, split: str = "training"):
        if split == "test":
            split = "validation"
        super().__init__(cfg, split)


        if split == "training":
            self.n_frames = cfg.n_frames
        else:
            self.n_frames = getattr(cfg, 'val_n_frames', cfg.n_frames)
        self.training_dropout = 0.1
        self.transform = self._build_transforms(cfg)
        self.current_epoch = 0

    def set_current_epoch(self, epoch):
            self.current_epoch = epoch

    def _augment(self, video: torch.Tensor, camera_params: torch.Tensor):
            """
            video: (T, C, H, W)
            camera_params: (T, 16)
            """
            # Data augmentation during training
            # 1. Horizontal Flip (50%)
            if random.random() < 0.5:
                video = torch.flip(video, [-1])
                camera_params = camera_params.clone()
                camera_params[:, [5, 6, 7, 8, 12]] *= -1

            # 2. Back-and-Forth (10%)
            if random.random() < 0.1:
                v_forward = video[::2]
                v_backward = torch.flip(video[1::2], [0]) # [0] 是 T 维度
                video = torch.cat([v_forward, v_backward], dim=0).contiguous()

                p_forward = camera_params[::2]
                p_backward = torch.flip(camera_params[1::2], [0])
                camera_params = torch.cat([p_forward, p_backward], dim=0).contiguous()

            # 3. Reverse (50%)
            if random.random() < 0.5:
                video = torch.flip(video, [0]).contiguous()
                camera_params = torch.flip(camera_params, [0]).contiguous()

            return video, camera_params

    def _build_transforms(self, cfg):
        transform_list = []
        video_h = getattr(cfg, "video_H", 224)
        video_w = getattr(cfg, "video_W", 384)
        transform_list.append(transforms.Resize((video_h, video_w), antialias=True))
        transform_list.append(transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]))
        return transforms.Compose(transform_list)

    def get_data_paths(self, split: str):
        """
        Retrieve all video file paths for the given split.

        Args:
            split (str): Dataset split ("training" or "validation").

        Returns:
            List[Path]: List of video file paths.
        """
        if split == "training":
            data_dir = self.save_dir / "training_256"
            self.pose_dir = self.save_dir / "training_poses"
        else:
            data_dir = self.save_dir / "validation_256"
            self.pose_dir = self.save_dir / "validation_poses"

        paths = sorted(list(data_dir.glob("**/*.mp4")), key=lambda x: x.name)

        return paths

    def download_dataset(self):
        pass

    def __getitem__(self, idx: int):
        """
        Retrieve a single data sample by index.

        Args:
            idx (int): Index of the data sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]: Video, actions, poses, and timestamps.
        """
        max_retries = 1000
        for _ in range(max_retries):
            try:
                return self.load_data(idx)
            except Exception as e:
                print(f"Retrying due to error: {e}")
                idx = (idx + 1) % len(self)

    def load_data(self, idx):
        idx = self.idx_remap[idx]
        file_idx, _ = self.split_idx(idx)
        video_path = self.data_paths[file_idx]
        video_length = self.video_lengths[file_idx]

        frame_skip = self.sampling_interval
        min_frames_required = self.min_frames_required
        max_start_frame = video_length - min_frames_required

        if max_start_frame < 0:
            raise ValueError(
                f"Video too short for skip={frame_skip}. "
                f"Length={video_length}, Required={min_frames_required}, Path={video_path}"
            )

        if self.split == 'training':
            frame_idx = random.randint(0, max_start_frame)
        else:
            frame_idx = 0

        frame_indices = [frame_idx + i * frame_skip for i in range(self.context_length)]

        cap = cv2.VideoCapture(str(video_path))
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()
            if not success:
                raise ValueError(f"Failed to read frame {idx} from {video_path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        video = np.stack(frames)

        video_relative_path = video_path.relative_to(video_path.parent.parent)
        pose_path = self.pose_dir / video_relative_path.parent / f"{video_path.stem}.pt"
        # pose_path = self.pose_dir / f"{video_path.stem}.pt"
        if not pose_path.exists():
                    raise FileNotFoundError(f"Pose not found: {pose_path}")
        camera_params_pool = torch.load(pose_path, map_location='cpu')
        camera_params_pool = torch.cat([camera_params_pool[:, :4], camera_params_pool[:, 6:]], dim=-1).float()
        camera_params = camera_params_pool[frame_indices]

        video = torch.from_numpy(video / 255.0).float().permute(0, 3, 1, 2).contiguous()
        video = self.transform(video)

        if self.split == 'training':
            video, camera_params = self._augment(video, camera_params)

        return (
            video,
            camera_params,
        )

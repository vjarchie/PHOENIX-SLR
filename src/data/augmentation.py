# -*- coding: utf-8 -*-
"""
Video Augmentation for Sign Language Recognition.

Note: NO horizontal flip - it changes sign meaning!
"""

import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from typing import List, Tuple


class VideoAugmentation:
    """
    Video augmentation pipeline for sign language.
    
    Applies consistent transforms across all frames.
    """
    
    def __init__(
        self,
        random_crop_scale: Tuple[float, float] = (0.8, 1.0),
        random_rotation: float = 10.0,  # degrees
        color_jitter: bool = True,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        gaussian_noise_std: float = 0.02,
        temporal_mask_prob: float = 0.2,
        temporal_mask_ratio: float = 0.1,
        training: bool = True
    ):
        self.random_crop_scale = random_crop_scale
        self.random_rotation = random_rotation
        self.color_jitter = color_jitter
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.gaussian_noise_std = gaussian_noise_std
        self.temporal_mask_prob = temporal_mask_prob
        self.temporal_mask_ratio = temporal_mask_ratio
        self.training = training
    
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to video frames.
        
        Args:
            frames: Tensor of shape (T, C, H, W) or (T, H, W, C)
        
        Returns:
            Augmented frames (T, C, H, W)
        """
        if not self.training:
            return frames
        
        # Ensure (T, C, H, W) format
        if frames.dim() == 4 and frames.shape[-1] == 3:
            frames = frames.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
        
        T, C, H, W = frames.shape
        
        # Sample augmentation parameters (consistent across frames)
        # Random crop
        crop_scale = random.uniform(*self.random_crop_scale)
        crop_h = int(H * crop_scale)
        crop_w = int(W * crop_scale)
        top = random.randint(0, H - crop_h)
        left = random.randint(0, W - crop_w)
        
        # Random rotation
        angle = random.uniform(-self.random_rotation, self.random_rotation)
        
        # Color jitter parameters
        brightness_factor = 1.0 + random.uniform(-self.brightness, self.brightness)
        contrast_factor = 1.0 + random.uniform(-self.contrast, self.contrast)
        saturation_factor = 1.0 + random.uniform(-self.saturation, self.saturation)
        
        # Apply transforms to each frame
        augmented_frames = []
        for t in range(T):
            frame = frames[t]  # (C, H, W)
            
            # Crop
            frame = TF.crop(frame, top, left, crop_h, crop_w)
            frame = TF.resize(frame, [H, W])
            
            # Rotation
            if abs(angle) > 0.1:
                frame = TF.rotate(frame, angle)
            
            # Color jitter
            if self.color_jitter:
                frame = TF.adjust_brightness(frame, brightness_factor)
                frame = TF.adjust_contrast(frame, contrast_factor)
                frame = TF.adjust_saturation(frame, saturation_factor)
            
            augmented_frames.append(frame)
        
        frames = torch.stack(augmented_frames)  # (T, C, H, W)
        
        # Gaussian noise
        if self.gaussian_noise_std > 0:
            noise = torch.randn_like(frames) * self.gaussian_noise_std
            frames = frames + noise
            frames = torch.clamp(frames, 0, 1)
        
        # Temporal masking (SpecAugment-style)
        if random.random() < self.temporal_mask_prob:
            mask_len = max(1, int(T * self.temporal_mask_ratio))
            mask_start = random.randint(0, T - mask_len)
            frames[mask_start:mask_start + mask_len] = 0
        
        return frames


class TemporalSampling:
    """
    Temporal sampling/resampling for videos.
    
    Supports:
    - Uniform sampling to fixed length
    - Random speed perturbation
    """
    
    def __init__(
        self,
        target_frames: int = 64,
        speed_perturb: bool = True,
        speed_range: Tuple[float, float] = (0.8, 1.2),
        training: bool = True
    ):
        self.target_frames = target_frames
        self.speed_perturb = speed_perturb
        self.speed_range = speed_range
        self.training = training
    
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Sample frames to target length.
        
        Args:
            frames: Tensor of shape (T, ...)
        
        Returns:
            Sampled frames of shape (target_frames, ...)
        """
        T = frames.shape[0]
        
        if self.training and self.speed_perturb:
            # Random speed perturbation
            speed = random.uniform(*self.speed_range)
            effective_frames = int(self.target_frames * speed)
        else:
            effective_frames = self.target_frames
        
        if T == effective_frames:
            indices = torch.arange(T)
        elif T > effective_frames:
            # Subsample
            indices = torch.linspace(0, T - 1, effective_frames).long()
        else:
            # Repeat frames
            indices = torch.linspace(0, T - 1, effective_frames).long()
        
        sampled = frames[indices]
        
        # Ensure output is exactly target_frames
        if sampled.shape[0] != self.target_frames:
            if sampled.shape[0] > self.target_frames:
                indices = torch.linspace(0, sampled.shape[0] - 1, self.target_frames).long()
                sampled = sampled[indices]
            else:
                # Pad with last frame
                pad_size = self.target_frames - sampled.shape[0]
                padding = sampled[-1:].repeat(pad_size, *([1] * (sampled.dim() - 1)))
                sampled = torch.cat([sampled, padding], dim=0)
        
        return sampled


def get_train_transforms(target_frames: int = 64):
    """Get training augmentation pipeline."""
    return {
        'temporal': TemporalSampling(
            target_frames=target_frames,
            speed_perturb=True,
            speed_range=(0.8, 1.2),
            training=True
        ),
        'spatial': VideoAugmentation(
            random_crop_scale=(0.85, 1.0),
            random_rotation=8.0,
            color_jitter=True,
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            gaussian_noise_std=0.01,
            temporal_mask_prob=0.15,
            temporal_mask_ratio=0.08,
            training=True
        )
    }


def get_val_transforms(target_frames: int = 64):
    """Get validation transforms (no augmentation)."""
    return {
        'temporal': TemporalSampling(
            target_frames=target_frames,
            speed_perturb=False,
            training=False
        ),
        'spatial': VideoAugmentation(training=False)
    }


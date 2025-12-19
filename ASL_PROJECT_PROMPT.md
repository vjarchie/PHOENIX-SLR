# Prompt for ASL Project Enhancement

> **Copy this entire document and paste it as a prompt in your ASL project repository**

---

## Context

I'm working on an **ASL (American Sign Language) Recognition** project that currently uses a **TCN (Temporal Convolutional Network)** architecture. The model is experiencing **CTC collapse** (outputting mostly blank tokens, resulting in ~100% WER).

I have a sister project (PHOENIX-SLR) where I successfully solved this problem using a **Hybrid CTC + Attention** approach. I want to apply the same solution here.

---

---

## Your Task

Create a **new submodule** called `enhanced/` with the following improvements, **without modifying any existing files**:

### 1. New Directory Structure

```
enhanced/
├── models/
│   ├── __init__.py
│   ├── hybrid_model.py      # Hybrid CTC + Attention model
│   ├── i3d_backbone.py      # I3D feature extractor
│   └── gloss_decoder.py     # Attention decoder
├── data/
│   ├── __init__.py
│   ├── augmented_dataset.py # Dataset with augmentation
│   └── augmentations.py     # Augmentation transforms
├── scripts/
│   ├── extract_i3d_features.py  # Extract I3D features from videos
│   ├── train_hybrid.py          # Training script for hybrid model
│   └── evaluate_hybrid.py       # Evaluation script
├── configs/
│   └── hybrid_config.yaml   # Configuration file
└── README.md                # Documentation
```

### 2. Hybrid CTC + Attention Model

Implement `HybridCTCAttentionModel` with:

```python
Architecture:
┌─────────────────────────────────────────────────────────────┐
│                    I3D Feature Extractor                     │
│              (Pretrained on Kinetics-400)                    │
│                    Output: (B, T, 1024)                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               Transformer Encoder (6 layers)                 │
│                    8 heads, d_model=512                      │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
       ┌──────────────┐              ┌──────────────┐
       │   CTC Head   │              │  Attention   │
       │  (weight=0.3)│              │   Decoder    │
       │              │              │  (weight=0.7)│
       └──────────────┘              └──────────────┘
                              │
                              ▼
                    Joint Loss Function:
           L = 0.3 × CTC_Loss + 0.7 × CE_Loss
```

Key components:
- **GlossDecoder**: 3-layer Transformer decoder with cross-attention
- **Joint loss**: Prevents CTC collapse by providing stable CE gradients
- **Special tokens**: `<pad>=0, <unk>=1, <blank>=2, <sos>=3, <eos>=4`

### 3. I3D Feature Extraction (CRITICAL FOR PERFORMANCE)

I3D (Inflated 3D ConvNet) pretrained on Kinetics-400 provides **significantly better** visual features than training from scratch.

#### Why I3D?

| Feature Extractor | Pretrained On | Feature Dim | Expected WER Improvement |
|-------------------|---------------|-------------|--------------------------|
| ResNet-18 (scratch) | ImageNet (images) | 512 | Baseline |
| **I3D** | Kinetics-400 (videos) | 1024 | **10-15% better** |
| S3D | HowTo100M | 1024 | 10-15% better |
| VideoMAE | Kinetics | 768 | 12-18% better |

#### Installation

```bash
# Option 1: video_features library (RECOMMENDED)
pip install video_features

# Option 2: Clone repository
git clone https://github.com/v-iashin/video_features.git
cd video_features
pip install -r requirements.txt
```

#### Full Implementation: `extract_i3d_features.py`

```python
#!/usr/bin/env python3
"""
Extract I3D features from ASL videos.

Usage:
    python enhanced/scripts/extract_i3d_features.py \
        --video_dir data/asl_videos/ \
        --output_dir data/i3d_features/ \
        --model i3d \
        --batch_size 16 \
        --device cuda

Output: .npy files with shape (T, 1024) for each video
"""

import os
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Option 1: Using video_features library
try:
    from video_features.models.i3d.extract_i3d import ExtractI3D
    USE_VIDEO_FEATURES = True
except ImportError:
    USE_VIDEO_FEATURES = False
    print("video_features not installed, using manual extraction")


class I3DFeatureExtractor:
    """
    Extract I3D features from video files.
    
    I3D (Inflated 3D ConvNet) is pretrained on Kinetics-400
    and provides 1024-dimensional features per temporal segment.
    """
    
    def __init__(
        self,
        model_name: str = 'i3d',
        device: str = 'cuda',
        extraction_fps: int = 25,
        stack_size: int = 64,  # Number of frames per clip
        step_size: int = 32,   # Stride between clips
    ):
        self.device = device
        self.extraction_fps = extraction_fps
        self.stack_size = stack_size
        self.step_size = step_size
        
        if USE_VIDEO_FEATURES:
            # Use video_features library
            self.extractor = ExtractI3D(
                device=device,
                extraction_fps=extraction_fps,
                stack_size=stack_size,
                step_size=step_size
            )
        else:
            # Manual loading using torchvision or pytorchvideo
            self._load_model_manual()
    
    def _load_model_manual(self):
        """Load I3D model manually if video_features not available."""
        try:
            # Try pytorchvideo
            from pytorchvideo.models.hub import i3d_r50
            self.model = i3d_r50(pretrained=True)
            self.model = self.model.to(self.device)
            self.model.eval()
            print("Loaded I3D from pytorchvideo")
        except ImportError:
            # Fallback to torchvision r3d
            from torchvision.models.video import r3d_18, R3D_18_Weights
            self.model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
            # Remove classification head, keep features
            self.model.fc = torch.nn.Identity()
            self.model = self.model.to(self.device)
            self.model.eval()
            print("Loaded R3D-18 from torchvision (I3D alternative)")
    
    def extract(self, video_path: str) -> np.ndarray:
        """
        Extract features from a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            features: (T, 1024) numpy array
        """
        if USE_VIDEO_FEATURES:
            features = self.extractor.extract(video_path)
            return features.cpu().numpy()
        else:
            return self._extract_manual(video_path)
    
    def _extract_manual(self, video_path: str) -> np.ndarray:
        """Manual feature extraction."""
        import cv2
        
        # Read video frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames in {video_path}")
        
        # Convert to tensor: (T, H, W, C) -> (C, T, H, W)
        frames = np.stack(frames)
        frames = frames.astype(np.float32) / 255.0
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0)
        frames = frames.to(self.device)
        
        # Extract features in chunks
        all_features = []
        with torch.no_grad():
            for i in range(0, frames.shape[2], self.step_size):
                end = min(i + self.stack_size, frames.shape[2])
                if end - i < 8:  # Skip very short clips
                    continue
                clip = frames[:, :, i:end, :, :]
                
                # Pad if needed
                if clip.shape[2] < self.stack_size:
                    pad = torch.zeros(1, 3, self.stack_size - clip.shape[2], 224, 224)
                    clip = torch.cat([clip, pad.to(self.device)], dim=2)
                
                feat = self.model(clip)
                all_features.append(feat.cpu().numpy())
        
        if len(all_features) == 0:
            # If video too short, process as single clip
            clip = frames
            if clip.shape[2] < self.stack_size:
                pad = torch.zeros(1, 3, self.stack_size - clip.shape[2], 224, 224)
                clip = torch.cat([clip, pad.to(self.device)], dim=2)
            with torch.no_grad():
                feat = self.model(clip)
            all_features.append(feat.cpu().numpy())
        
        features = np.concatenate(all_features, axis=0)
        return features  # (T, 1024) or (T, 512) depending on model


def extract_all_features(
    video_dir: str,
    output_dir: str,
    model_name: str = 'i3d',
    device: str = 'cuda',
    batch_size: int = 16
):
    """Extract features from all videos in a directory."""
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f'**/*{ext}'))
    
    print(f"Found {len(video_files)} videos")
    
    # Initialize extractor
    extractor = I3DFeatureExtractor(model_name=model_name, device=device)
    
    # Extract features
    for video_path in tqdm(video_files, desc="Extracting features"):
        output_path = output_dir / f"{video_path.stem}.npy"
        
        # Skip if already extracted
        if output_path.exists():
            continue
        
        try:
            features = extractor.extract(str(video_path))
            np.save(output_path, features)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            continue
    
    print(f"Features saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract I3D features from videos")
    parser.add_argument('--video_dir', type=str, required=True, help='Directory with videos')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for features')
    parser.add_argument('--model', type=str, default='i3d', choices=['i3d', 's3d', 'r3d'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=16)
    
    args = parser.parse_args()
    
    extract_all_features(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size
    )
```

#### I3D Backbone Wrapper: `i3d_backbone.py`

```python
"""
I3D Backbone for loading pre-extracted features or extracting on-the-fly.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class I3DBackbone(nn.Module):
    """
    Wrapper for I3D features.
    
    Can either:
    1. Load pre-extracted features from .npy files
    2. Extract features on-the-fly (slower, for inference)
    """
    
    def __init__(
        self,
        features_dir: str = None,
        output_dim: int = 512,
        extract_online: bool = False,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.features_dir = Path(features_dir) if features_dir else None
        self.extract_online = extract_online
        self.i3d_dim = 1024  # I3D outputs 1024-dim features
        
        # Project I3D features to model dimension
        self.projection = nn.Sequential(
            nn.Linear(self.i3d_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(0.1)
        )
        
        if extract_online:
            from .extract_i3d_features import I3DFeatureExtractor
            self.extractor = I3DFeatureExtractor(device=device)
    
    def load_features(self, video_id: str) -> torch.Tensor:
        """Load pre-extracted features."""
        feat_path = self.features_dir / f"{video_id}.npy"
        if not feat_path.exists():
            raise FileNotFoundError(f"Features not found: {feat_path}")
        
        features = np.load(feat_path)
        return torch.from_numpy(features).float()
    
    def forward(self, x: torch.Tensor, video_ids: list = None) -> torch.Tensor:
        """
        Args:
            x: Pre-loaded features (B, T, 1024) or video paths
            video_ids: List of video IDs for loading features
            
        Returns:
            features: (B, T, output_dim)
        """
        if video_ids is not None and self.features_dir is not None:
            # Load from files
            batch_features = []
            for vid in video_ids:
                feat = self.load_features(vid)
                batch_features.append(feat)
            
            # Pad to same length
            max_len = max(f.shape[0] for f in batch_features)
            padded = []
            for f in batch_features:
                if f.shape[0] < max_len:
                    pad = torch.zeros(max_len - f.shape[0], f.shape[1])
                    f = torch.cat([f, pad], dim=0)
                padded.append(f)
            
            x = torch.stack(padded).to(next(self.projection.parameters()).device)
        
        # Project to output dimension
        x = self.projection(x)
        
        return x
```

### 4. Data Augmentation (CRITICAL FOR GENERALIZATION)

Data augmentation is essential to prevent overfitting and improve generalization.
Sign language has unique augmentation requirements - be careful not to flip horizontally
as it can change sign meaning!

#### Expected Impact

| Augmentation Level | WER Improvement |
|--------------------|-----------------|
| None | Baseline |
| Basic (spatial only) | 3-5% better |
| **Full (spatial + temporal)** | **8-12% better** |
| Advanced (+ MixUp, CutMix) | 10-15% better |

#### Full Implementation: `augmentations.py`

```python
"""
Data Augmentation for Sign Language Video Recognition.

IMPORTANT: Sign language is sensitive to hand shape and movement direction.
- DO NOT flip horizontally (left/right hand matters!)
- BE CAREFUL with rotation (can distort signs)
- Temporal augmentations are very effective
"""

import numpy as np
import torch
import torch.nn.functional as F
import random
from typing import Dict, List, Tuple, Optional
import cv2


# ============================================================================
# SPATIAL AUGMENTATIONS (Applied per-frame)
# ============================================================================

class RandomRotation:
    """Rotate frames by small angle."""
    
    def __init__(self, degrees: float = 10, p: float = 0.5):
        self.degrees = degrees
        self.p = p
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """
        Args:
            frames: (T, H, W, C) numpy array
        Returns:
            rotated frames
        """
        if random.random() > self.p:
            return frames
        
        angle = random.uniform(-self.degrees, self.degrees)
        T, H, W, C = frames.shape
        center = (W // 2, H // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        rotated = np.stack([
            cv2.warpAffine(frames[t], M, (W, H))
            for t in range(T)
        ])
        return rotated


class RandomScale:
    """Scale frames by random factor."""
    
    def __init__(self, scale_range: Tuple[float, float] = (0.9, 1.1), p: float = 0.5):
        self.scale_range = scale_range
        self.p = p
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return frames
        
        scale = random.uniform(*self.scale_range)
        T, H, W, C = frames.shape
        new_H, new_W = int(H * scale), int(W * scale)
        
        # Scale and crop/pad back to original size
        scaled = np.stack([
            cv2.resize(frames[t], (new_W, new_H))
            for t in range(T)
        ])
        
        # Crop or pad to original size
        if scale > 1:
            # Crop center
            start_h = (new_H - H) // 2
            start_w = (new_W - W) // 2
            scaled = scaled[:, start_h:start_h+H, start_w:start_w+W, :]
        else:
            # Pad
            pad_h = (H - new_H) // 2
            pad_w = (W - new_W) // 2
            padded = np.zeros((T, H, W, C), dtype=scaled.dtype)
            padded[:, pad_h:pad_h+new_H, pad_w:pad_w+new_W, :] = scaled
            scaled = padded
        
        return scaled


class ColorJitter:
    """Randomly adjust brightness, contrast, saturation."""
    
    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.1,
        p: float = 0.5
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.p = p
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return frames
        
        frames = frames.astype(np.float32)
        
        # Brightness
        if self.brightness > 0:
            delta = random.uniform(-self.brightness, self.brightness)
            frames = frames + delta
        
        # Contrast
        if self.contrast > 0:
            factor = random.uniform(1 - self.contrast, 1 + self.contrast)
            mean = frames.mean()
            frames = (frames - mean) * factor + mean
        
        # Saturation (convert to HSV)
        if self.saturation > 0:
            factor = random.uniform(1 - self.saturation, 1 + self.saturation)
            for t in range(frames.shape[0]):
                hsv = cv2.cvtColor((frames[t] * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
                frames[t] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255
        
        return np.clip(frames, 0, 1)


class GaussianNoise:
    """Add Gaussian noise to frames."""
    
    def __init__(self, std: float = 0.05, p: float = 0.3):
        self.std = std
        self.p = p
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return frames
        
        noise = np.random.normal(0, self.std, frames.shape).astype(np.float32)
        return np.clip(frames + noise, 0, 1)


class GaussianBlur:
    """Apply Gaussian blur to frames."""
    
    def __init__(self, kernel_size: int = 5, p: float = 0.3):
        self.kernel_size = kernel_size
        self.p = p
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return frames
        
        return np.stack([
            cv2.GaussianBlur(frames[t], (self.kernel_size, self.kernel_size), 0)
            for t in range(frames.shape[0])
        ])


# ============================================================================
# TEMPORAL AUGMENTATIONS (Critical for sign language!)
# ============================================================================

class TemporalCrop:
    """Randomly crop a temporal segment."""
    
    def __init__(self, min_ratio: float = 0.8, p: float = 0.5):
        self.min_ratio = min_ratio
        self.p = p
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return frames
        
        T = frames.shape[0]
        new_T = int(T * random.uniform(self.min_ratio, 1.0))
        start = random.randint(0, T - new_T)
        
        return frames[start:start+new_T]


class SpeedPerturbation:
    """Change playback speed by resampling frames."""
    
    def __init__(self, rates: List[float] = [0.9, 1.0, 1.1], p: float = 0.5):
        self.rates = rates
        self.p = p
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return frames
        
        rate = random.choice(self.rates)
        T = frames.shape[0]
        new_T = int(T / rate)
        
        if new_T == T:
            return frames
        
        # Resample using linear interpolation
        indices = np.linspace(0, T - 1, new_T).astype(np.float32)
        new_frames = []
        
        for idx in indices:
            low = int(idx)
            high = min(low + 1, T - 1)
            alpha = idx - low
            
            frame = (1 - alpha) * frames[low] + alpha * frames[high]
            new_frames.append(frame)
        
        return np.stack(new_frames)


class TemporalJitter:
    """Randomly drop or duplicate frames."""
    
    def __init__(self, drop_prob: float = 0.1, dup_prob: float = 0.1, p: float = 0.5):
        self.drop_prob = drop_prob
        self.dup_prob = dup_prob
        self.p = p
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return frames
        
        new_frames = []
        for t in range(frames.shape[0]):
            if random.random() < self.drop_prob:
                continue  # Drop frame
            
            new_frames.append(frames[t])
            
            if random.random() < self.dup_prob:
                new_frames.append(frames[t])  # Duplicate frame
        
        if len(new_frames) == 0:
            return frames  # Don't drop all frames
        
        return np.stack(new_frames)


class FrameShift:
    """Shift all frames by random offset (circular)."""
    
    def __init__(self, max_shift: int = 5, p: float = 0.5):
        self.max_shift = max_shift
        self.p = p
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return frames
        
        shift = random.randint(-self.max_shift, self.max_shift)
        return np.roll(frames, shift, axis=0)


# ============================================================================
# ADVANCED AUGMENTATIONS
# ============================================================================

class MixUp:
    """
    Mix two samples (frames and labels).
    Requires dataset-level implementation.
    """
    
    def __init__(self, alpha: float = 0.2, p: float = 0.3):
        self.alpha = alpha
        self.p = p
    
    def __call__(
        self,
        frames1: np.ndarray,
        frames2: np.ndarray,
        labels1: List[int],
        labels2: List[int]
    ) -> Tuple[np.ndarray, List[int], float]:
        """
        Returns mixed frames and lambda for loss computation.
        """
        if random.random() > self.p:
            return frames1, labels1, 1.0
        
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Align frame lengths
        T1, T2 = frames1.shape[0], frames2.shape[0]
        if T1 != T2:
            # Resample frames2 to match frames1
            indices = np.linspace(0, T2 - 1, T1).astype(int)
            frames2 = frames2[indices]
        
        mixed = lam * frames1 + (1 - lam) * frames2
        
        return mixed, labels1, lam  # Use labels1 with weight lam


class TemporalMask:
    """Mask random temporal segments (like SpecAugment for audio)."""
    
    def __init__(
        self,
        num_masks: int = 2,
        max_mask_len: int = 10,
        p: float = 0.5
    ):
        self.num_masks = num_masks
        self.max_mask_len = max_mask_len
        self.p = p
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return frames
        
        frames = frames.copy()
        T = frames.shape[0]
        
        for _ in range(self.num_masks):
            mask_len = random.randint(1, min(self.max_mask_len, T // 4))
            start = random.randint(0, T - mask_len)
            frames[start:start+mask_len] = 0  # Zero out frames
        
        return frames


class SpatialMask:
    """Mask random spatial regions (cutout)."""
    
    def __init__(
        self,
        num_masks: int = 2,
        max_size: float = 0.2,  # Fraction of image size
        p: float = 0.5
    ):
        self.num_masks = num_masks
        self.max_size = max_size
        self.p = p
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return frames
        
        frames = frames.copy()
        T, H, W, C = frames.shape
        
        for _ in range(self.num_masks):
            mask_h = int(H * random.uniform(0.1, self.max_size))
            mask_w = int(W * random.uniform(0.1, self.max_size))
            
            y = random.randint(0, H - mask_h)
            x = random.randint(0, W - mask_w)
            
            frames[:, y:y+mask_h, x:x+mask_w, :] = 0
        
        return frames


# ============================================================================
# COMPOSED AUGMENTATION PIPELINE
# ============================================================================

class SignLanguageAugmentation:
    """
    Complete augmentation pipeline for sign language video data.
    
    Usage:
        augmenter = SignLanguageAugmentation(mode='train')
        augmented_frames = augmenter(frames)
    """
    
    def __init__(self, mode: str = 'train', intensity: str = 'medium'):
        """
        Args:
            mode: 'train', 'val', or 'test'
            intensity: 'light', 'medium', or 'heavy'
        """
        self.mode = mode
        self.intensity = intensity
        self.transforms = self._build_transforms()
    
    def _build_transforms(self) -> List:
        if self.mode != 'train':
            return []  # No augmentation for val/test
        
        if self.intensity == 'light':
            return [
                ColorJitter(brightness=0.1, contrast=0.1, p=0.3),
                GaussianNoise(std=0.02, p=0.2),
            ]
        
        elif self.intensity == 'medium':
            return [
                # Spatial
                RandomRotation(degrees=10, p=0.5),
                RandomScale(scale_range=(0.95, 1.05), p=0.3),
                ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.5),
                GaussianNoise(std=0.03, p=0.3),
                
                # Temporal (most important!)
                SpeedPerturbation(rates=[0.9, 0.95, 1.0, 1.05, 1.1], p=0.5),
                TemporalCrop(min_ratio=0.85, p=0.3),
                TemporalJitter(drop_prob=0.05, dup_prob=0.05, p=0.3),
            ]
        
        else:  # heavy
            return [
                # Spatial
                RandomRotation(degrees=15, p=0.6),
                RandomScale(scale_range=(0.9, 1.1), p=0.5),
                ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, p=0.6),
                GaussianNoise(std=0.05, p=0.4),
                GaussianBlur(kernel_size=3, p=0.2),
                SpatialMask(num_masks=1, max_size=0.15, p=0.3),
                
                # Temporal
                SpeedPerturbation(rates=[0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15], p=0.6),
                TemporalCrop(min_ratio=0.8, p=0.4),
                TemporalJitter(drop_prob=0.08, dup_prob=0.08, p=0.4),
                TemporalMask(num_masks=1, max_mask_len=5, p=0.2),
            ]
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """
        Apply augmentations to frames.
        
        Args:
            frames: (T, H, W, C) numpy array, values in [0, 1]
            
        Returns:
            augmented frames
        """
        for transform in self.transforms:
            frames = transform(frames)
        
        return frames


# ============================================================================
# FEATURE-LEVEL AUGMENTATIONS (For I3D features)
# ============================================================================

class FeatureDropout:
    """Dropout on feature dimensions."""
    
    def __init__(self, p: float = 0.1):
        self.p = p
    
    def __call__(self, features: np.ndarray) -> np.ndarray:
        if random.random() > 0.5:
            return features
        
        mask = np.random.binomial(1, 1 - self.p, features.shape[-1])
        return features * mask / (1 - self.p)


class FeatureNoise:
    """Add noise to features."""
    
    def __init__(self, std: float = 0.1, p: float = 0.5):
        self.std = std
        self.p = p
    
    def __call__(self, features: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return features
        
        noise = np.random.normal(0, self.std, features.shape)
        return features + noise


class TemporalFeatureMask:
    """Mask temporal positions in features."""
    
    def __init__(self, num_masks: int = 2, max_len: int = 5, p: float = 0.5):
        self.num_masks = num_masks
        self.max_len = max_len
        self.p = p
    
    def __call__(self, features: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return features
        
        features = features.copy()
        T = features.shape[0]
        
        for _ in range(self.num_masks):
            mask_len = random.randint(1, min(self.max_len, T // 4))
            start = random.randint(0, max(0, T - mask_len))
            features[start:start+mask_len] = 0
        
        return features


class FeatureAugmentation:
    """Augmentation pipeline for pre-extracted features."""
    
    def __init__(self, mode: str = 'train'):
        self.mode = mode
        if mode == 'train':
            self.transforms = [
                FeatureDropout(p=0.1),
                FeatureNoise(std=0.05, p=0.3),
                TemporalFeatureMask(num_masks=1, max_len=3, p=0.3),
            ]
        else:
            self.transforms = []
    
    def __call__(self, features: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            features = transform(features)
        return features
```

#### Usage in Dataset

```python
from enhanced.data.augmentations import SignLanguageAugmentation, FeatureAugmentation

class AugmentedDataset(Dataset):
    def __init__(self, base_dataset, mode='train', use_i3d=True):
        self.base = base_dataset
        self.use_i3d = use_i3d
        
        # Choose augmentation based on input type
        if use_i3d:
            self.augmenter = FeatureAugmentation(mode=mode)
        else:
            self.augmenter = SignLanguageAugmentation(mode=mode, intensity='medium')
    
    def __getitem__(self, idx):
        sample = self.base[idx]
        
        if self.use_i3d:
            features = sample['features']
            features = self.augmenter(features.numpy())
            sample['features'] = torch.from_numpy(features)
        else:
            frames = sample['frames']
            frames = self.augmenter(frames.numpy())
            sample['frames'] = torch.from_numpy(frames)
        
        return sample
```

### 5. Training Script

Create `train_hybrid.py` with these features:

```python
# Key training parameters
config = {
    'model': {
        'type': 'hybrid',
        'd_model': 512,
        'nhead': 8,
        'num_encoder_layers': 6,
        'num_decoder_layers': 3,
        'input_dim': 1024,  # I3D features
        'ctc_weight': 0.3,
        'ce_weight': 0.7,
    },
    'training': {
        'epochs': 100,
        'batch_size': 8,
        'lr': 0.0001,
        'warmup_epochs': 5,
        'scheduler': 'cosine',
        'gradient_clip': 5.0,
    },
    'data': {
        'use_i3d_features': True,
        'augmentation': True,
        'max_frames': 150,
    }
}

# Must include:
# 1. Warmup + Cosine LR scheduler
# 2. Gradient clipping (max_norm=5.0)
# 3. Blank ratio monitoring (CTC collapse indicator)
# 4. Joint loss computation
# 5. Checkpoint saving (best.pth, latest.pth)
# 6. Metrics logging to JSONL for dashboard
```

### 6. Dataset Modifications

Create `augmented_dataset.py` that wraps existing dataset:

```python
class AugmentedASLDataset(Dataset):
    """
    Wrapper around existing ASL dataset with:
    1. I3D feature loading (instead of raw frames)
    2. Data augmentation
    3. Decoder tokens (<sos>, <eos>)
    """
    
    def __init__(
        self,
        base_dataset,  # Your existing ASLDataset
        i3d_features_dir: str,
        augment: bool = True,
        use_i3d: bool = True
    ):
        self.base = base_dataset
        self.i3d_dir = i3d_features_dir
        self.augment = augment
        self.use_i3d = use_i3d
        
        # Add special tokens to vocab
        self._extend_vocab()
    
    def _extend_vocab(self):
        # Add <sos> and <eos> if not present
        if '<sos>' not in self.base.vocab:
            self.base.vocab['<sos>'] = len(self.base.vocab)
        if '<eos>' not in self.base.vocab:
            self.base.vocab['<eos>'] = len(self.base.vocab)
    
    def __getitem__(self, idx):
        sample = self.base[idx]
        
        if self.use_i3d:
            # Load pre-extracted I3D features
            features = np.load(f"{self.i3d_dir}/{sample['id']}.npy")
            sample['features'] = torch.tensor(features)
        
        if self.augment:
            sample = self.apply_augmentations(sample)
        
        # Add decoder tokens
        sample['decoder_input'] = [SOS_IDX] + sample['gloss_ids']
        sample['decoder_target'] = sample['gloss_ids'] + [EOS_IDX]
        
        return sample
```

### 7. Collate Function

Update collate function for hybrid training:

```python
def hybrid_collate_fn(batch):
    """
    Returns:
    - features: (B, T, D) - I3D features
    - ctc_targets: (B, L) - CTC targets (no special tokens)
    - decoder_input: (B, L+1) - <sos> + targets
    - decoder_target: (B, L+1) - targets + <eos>
    - Various masks and lengths
    """
    # Sort by length, pad, create masks
    # Similar to PHOENIX implementation
```

---

## Key Learnings from PHOENIX Project

### Why TCN + CTC Failed

| Issue | Impact |
|-------|--------|
| CTC assumes output independence | Can't model ASL phrase structure |
| TCN lacks global context | Limited receptive field |
| No output supervision | Only alignment loss |
| Blank token dominance | Easy local minimum |

### Why Hybrid Works

1. **Attention decoder** provides stable gradients even when CTC collapses
2. **Cross-entropy loss** gives direct supervision for output tokens
3. **Shared encoder** benefits from both signals
4. **Joint training** balances alignment (CTC) and generation (CE)

### Critical Implementation Details

```python
# 1. Loss weighting - CE should dominate early
ctc_weight = 0.3
ce_weight = 0.7

# 2. Warmup is essential
warmup_epochs = 5
lr_schedule = warmup + cosine_annealing

# 3. Monitor blank ratio
if blank_ratio > 0.95:
    print("WARNING: CTC collapse detected!")

# 4. Gradient clipping prevents explosions
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

# 5. Use pretrained backbone
# I3D pretrained on Kinetics >> training from scratch
```

---

## Expected Results

| Approach | Expected WER |
|----------|--------------|
| Current TCN + CTC | 100% (collapsed) |
| Hybrid (no I3D) | ~50-60% |
| Hybrid + I3D | ~35-45% |
| Hybrid + I3D + Augmentation | ~25-35% |

---

## Files to Create

1. `enhanced/models/hybrid_model.py` - Main model
2. `enhanced/models/gloss_decoder.py` - Attention decoder
3. `enhanced/models/i3d_backbone.py` - I3D feature extractor wrapper
4. `enhanced/data/augmented_dataset.py` - Dataset with augmentations
5. `enhanced/data/augmentations.py` - Augmentation transforms
6. `enhanced/scripts/extract_i3d_features.py` - Feature extraction
7. `enhanced/scripts/train_hybrid.py` - Training script
8. `enhanced/scripts/evaluate_hybrid.py` - Evaluation script
9. `enhanced/configs/hybrid_config.yaml` - Configuration
10. `enhanced/README.md` - Documentation

---

## Do NOT Modify

- Any files in `src/`
- `train.py`
- `evaluate.py`
- Existing model implementations

The enhanced approach should be completely self-contained in the `enhanced/` submodule.

---

## Reference Implementation

See the PHOENIX-SLR project at `D:\PHOENIX-SLR` for working implementation of:
- `HybridCTCAttentionModel` in `src/models/transformer.py`
- `GlossDecoder` in `src/models/transformer.py`
- Collate function in `src/data/phoenix_dataset.py`
- Training loop in `train_hybrid.py`

---

## Questions to Answer

1. Read the existing codebase first - understand the current TCN implementation
2. What is the vocabulary size? What special tokens exist?
3. What is the input format (raw frames or features)?
4. What is the current dataset structure?
5. Are there existing augmentations?

Then proceed with implementation, starting with the model architecture.


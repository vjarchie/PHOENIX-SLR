"""
PHOENIX-2014 Dataset loader adapted for CorrNet+.
Outputs video tensors in (C, T, H, W) format expected by CorrNet+.
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random


class PhoenixDatasetCorrNet(Dataset):
    """
    PHOENIX-2014 Dataset for CorrNet+.
    
    Key differences from standard loader:
    1. Returns video as (C, T, H, W) instead of (T, C, H, W)
    2. Includes temporal augmentation (random sampling, speed perturbation)
    3. Returns sequence length for variable-length batch handling
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        feature_type: str = 'fullFrame-210x260px',
        max_frames: int = 300,
        img_size: Tuple[int, int] = (224, 224),
        transform=None,
        vocab: Dict[str, int] = None,
        temporal_augment: bool = True
    ):
        """
        Args:
            data_dir: Path to phoenix2014-release directory
            split: 'train', 'dev', or 'test'
            feature_type: Type of features to load
            max_frames: Maximum number of frames
            img_size: (height, width) for resizing
            transform: Optional spatial transforms
            vocab: Gloss vocabulary mapping
            temporal_augment: Enable temporal augmentation for training
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.feature_type = feature_type
        self.max_frames = max_frames
        self.img_size = img_size
        self.transform = transform
        self.temporal_augment = temporal_augment and (split == 'train')
        
        self.samples = self._load_annotations()
        
        if vocab is None:
            self.vocab = self._build_vocab()
        else:
            self.vocab = vocab
        
        self.idx2gloss = {v: k for k, v in self.vocab.items()}
        
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        print(f"PhoenixDatasetCorrNet loaded:")
        print(f"  Split: {split}")
        print(f"  Samples: {len(self.samples)}")
        print(f"  Vocabulary: {len(self.vocab)} glosses")
        print(f"  Temporal augment: {self.temporal_augment}")
    
    def _load_annotations(self) -> List[Dict]:
        """Load annotations from corpus files."""
        samples = []
        
        anno_dir = self.data_dir / "phoenix-2014-multisigner" / "annotations" / "manual"
        corpus_file = anno_dir / f"{self.split}.corpus.csv"
        
        if not corpus_file.exists():
            corpus_file = anno_dir / f"PHOENIX-2014-T.{self.split}.corpus.csv"
        
        if not corpus_file.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_file}")
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines[1:]:
            parts = line.strip().split('|')
            if len(parts) >= 4:
                sample = {
                    'id': parts[0],
                    'folder': parts[1],
                    'signer': parts[2],
                    'annotation': parts[3].split()
                }
                samples.append(sample)
        
        return samples
    
    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary from training annotations."""
        vocab = {'<pad>': 0, '<sos>': 1, '<blank>': 2}
        
        for sample in self.samples:
            for gloss in sample['annotation']:
                if gloss not in vocab:
                    vocab[gloss] = len(vocab)
        
        return vocab
    
    def _load_video(self, folder: str) -> np.ndarray:
        """Load video frames from folder."""
        folder_path = folder.replace('/*.png', '').replace('\\*.png', '')
        
        features_dir = self.data_dir / "phoenix-2014-multisigner" / "features" / self.feature_type / self.split / folder_path
        
        if not features_dir.exists():
            raise FileNotFoundError(f"Frame directory not found: {features_dir}")
        
        frame_files = sorted(features_dir.glob("*.png"))
        
        if len(frame_files) > self.max_frames:
            indices = np.linspace(0, len(frame_files) - 1, self.max_frames, dtype=int)
            frame_files = [frame_files[i] for i in indices]
        
        frames = []
        for frame_path in frame_files:
            img = cv2.imread(str(frame_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)
                frames.append(img)
        
        if len(frames) == 0:
            raise ValueError(f"No valid frames found in {features_dir}")
        
        return np.stack(frames, axis=0)
    
    def _temporal_augment(self, frames: np.ndarray) -> np.ndarray:
        """Apply temporal augmentation (random temporal scaling)."""
        if not self.temporal_augment:
            return frames
        
        T = len(frames)
        
        scale = random.uniform(0.8, 1.2)
        new_T = max(1, int(T * scale))
        
        if new_T == T:
            return frames
        
        indices = np.linspace(0, T - 1, new_T).astype(int)
        return frames[indices]
    
    def _normalize(self, frames: np.ndarray) -> torch.Tensor:
        """Normalize frames to ImageNet statistics."""
        frames = frames.astype(np.float32) / 255.0
        frames = (frames - self.mean) / self.std
        
        frames = torch.from_numpy(frames).float()
        frames = frames.permute(3, 0, 1, 2)
        
        return frames
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            dict with:
                - video: (C, T, H, W) tensor
                - label: (L,) gloss indices
                - video_len: original sequence length
                - label_len: label length
                - id: sample id
        """
        sample = self.samples[idx]
        
        frames = self._load_video(sample['folder'])
        frames = self._temporal_augment(frames)
        
        if self.transform is not None:
            frames = np.stack([self.transform(f) for f in frames])
        
        video_len = len(frames)
        video = self._normalize(frames)
        
        label = [self.vocab.get(g, self.vocab['<blank>']) for g in sample['annotation']]
        label = torch.tensor(label, dtype=torch.long)
        
        return {
            'video': video,
            'label': label,
            'video_len': video_len,
            'label_len': len(label),
            'id': sample['id'],
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for variable-length videos.
    
    Returns:
        dict with:
            - videos: (B, C, T_max, H, W) padded tensor
            - labels: (B, L_max) padded labels
            - video_lens: (B,) original sequence lengths
            - label_lens: (B,) label lengths
            - ids: list of sample ids
    """
    videos = [item['video'] for item in batch]
    labels = [item['label'] for item in batch]
    video_lens = torch.tensor([item['video_len'] for item in batch])
    label_lens = torch.tensor([item['label_len'] for item in batch])
    ids = [item['id'] for item in batch]
    
    max_video_len = max(v.shape[1] for v in videos)
    max_label_len = max(len(l) for l in labels)
    
    C, _, H, W = videos[0].shape
    B = len(videos)
    
    padded_videos = torch.zeros(B, C, max_video_len, H, W)
    padded_labels = torch.zeros(B, max_label_len, dtype=torch.long)
    
    for i, (video, label) in enumerate(zip(videos, labels)):
        T = video.shape[1]
        padded_videos[i, :, :T, :, :] = video
        padded_labels[i, :len(label)] = label
    
    return {
        'videos': padded_videos,
        'labels': padded_labels,
        'video_lens': video_lens,
        'label_lens': label_lens,
        'ids': ids,
    }

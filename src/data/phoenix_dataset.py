# -*- coding: utf-8 -*-
"""
RWTH-PHOENIX-Weather 2014 Dataset Loader

Handles loading of video frames and gloss annotations.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET


class PhoenixDataset(Dataset):
    """
    RWTH-PHOENIX-Weather 2014 Dataset.
    
    Loads video frames and corresponding gloss annotations.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',  # 'train', 'dev', 'test'
        feature_type: str = 'fullFrame-210x260px',  # or 'trackedRightHand-92x132px'
        max_frames: int = 300,
        transform=None,
        load_video: bool = True,
        vocab: Dict[str, int] = None
    ):
        """
        Args:
            data_dir: Path to phoenix2014-release directory
            split: 'train', 'dev', or 'test'
            feature_type: Type of features to load
            max_frames: Maximum number of frames to load
            transform: Optional transform for video frames
            load_video: Whether to load video frames (False for pre-extracted features)
            vocab: Gloss vocabulary mapping (word -> index)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.feature_type = feature_type
        self.max_frames = max_frames
        self.transform = transform
        self.load_video = load_video
        
        # Load annotations
        self.samples = self._load_annotations()
        
        # Build or use provided vocabulary
        if vocab is None:
            self.vocab = self._build_vocab()
        else:
            self.vocab = vocab
        
        self.idx2gloss = {v: k for k, v in self.vocab.items()}
        
        print(f"PhoenixDataset loaded:")
        print(f"  Split: {split}")
        print(f"  Samples: {len(self.samples)}")
        print(f"  Vocabulary: {len(self.vocab)} glosses")
    
    def _load_annotations(self) -> List[Dict]:
        """Load annotations from corpus files."""
        samples = []
        
        # Path to annotations
        anno_dir = self.data_dir / "phoenix-2014-multisigner" / "annotations" / "manual"
        corpus_file = anno_dir / f"{self.split}.corpus.csv"
        
        if not corpus_file.exists():
            # Try alternative path
            corpus_file = anno_dir / f"PHOENIX-2014-T.{self.split}.corpus.csv"
        
        if not corpus_file.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_file}")
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Parse CSV (format: id|folder|signer|annotation)
        for line in lines[1:]:  # Skip header
            parts = line.strip().split('|')
            if len(parts) >= 4:
                sample = {
                    'id': parts[0],
                    'folder': parts[1],
                    'signer': parts[2],
                    'annotation': parts[3].split()  # Gloss sequence
                }
                samples.append(sample)
        
        return samples
    
    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary from annotations."""
        vocab = {
            '<pad>': 0,
            '<unk>': 1,
            '<blank>': 2,   # CTC blank
            '<sos>': 3,     # Start of sequence (for decoder)
            '<eos>': 4      # End of sequence (for decoder)
        }
        
        for sample in self.samples:
            for gloss in sample['annotation']:
                if gloss not in vocab:
                    vocab[gloss] = len(vocab)
        
        return vocab
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Load video frames
        if self.load_video:
            frames = self._load_frames(sample)
        else:
            # Load pre-extracted features
            frames = self._load_features(sample)
        
        # Encode gloss sequence
        glosses = sample['annotation']
        gloss_ids = [self.vocab.get(g, self.vocab['<unk>']) for g in glosses]
        
        # Handle both tensor and numpy array from _load_frames
        if isinstance(frames, torch.Tensor):
            frames_tensor = frames.float()
        else:
            frames_tensor = torch.tensor(frames, dtype=torch.float32)
        
        return {
            'id': sample['id'],
            'frames': frames_tensor,
            'gloss_ids': torch.tensor(gloss_ids, dtype=torch.long),
            'glosses': glosses,
            'length': frames_tensor.shape[0]
        }
    
    def _load_frames(self, sample: Dict) -> np.ndarray:
        """Load video frames as numpy array with uniform temporal subsampling."""
        # The folder field contains path like "folder_name/1/*.png"
        # We need to strip the glob pattern to get the actual directory
        folder_path = sample['folder'].replace('/*.png', '').replace('\\*.png', '')
        
        frame_dir = (
            self.data_dir / "phoenix-2014-multisigner" / 
            "features" / self.feature_type / self.split / folder_path
        )
        
        if not frame_dir.exists():
            raise FileNotFoundError(f"Frame directory not found: {frame_dir}")
        
        # Get all frame files
        all_frame_files = sorted(frame_dir.glob("*.png"))
        total_frames = len(all_frame_files)
        
        # Uniform temporal subsampling if we have more frames than max_frames
        if total_frames > self.max_frames:
            # Sample frames uniformly across the video
            indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int)
            frame_files = [all_frame_files[i] for i in indices]
        else:
            frame_files = all_frame_files
        
        frames = []
        for f in frame_files:
            img = cv2.imread(str(f))
            if img is None:
                continue  # Skip if frame can't be read
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Normalize to [0, 1]
            img = img.astype(np.float32) / 255.0
            frames.append(img)
        
        if len(frames) == 0:
            raise ValueError(f"No valid frames found in {frame_dir}")
        
        # Stack to tensor: (T, H, W, C)
        frames = np.stack(frames)
        frames = torch.from_numpy(frames).float()
        
        # Apply video-level transforms if provided
        if self.transform:
            # Transform expects dict with 'temporal' and 'spatial' keys
            if isinstance(self.transform, dict):
                # Apply temporal sampling first
                if 'temporal' in self.transform:
                    frames = self.transform['temporal'](frames)
                # Convert to (T, C, H, W) for spatial transforms
                if frames.dim() == 4 and frames.shape[-1] == 3:
                    frames = frames.permute(0, 3, 1, 2)
                # Apply spatial augmentation
                if 'spatial' in self.transform:
                    frames = self.transform['spatial'](frames)
            else:
                # Legacy: apply transform directly
                frames = self.transform(frames)
        else:
            # Default: just permute to (T, C, H, W)
            if frames.dim() == 4 and frames.shape[-1] == 3:
                frames = frames.permute(0, 3, 1, 2)
        
        return frames
    
    def _load_features(self, sample: Dict) -> np.ndarray:
        """Load pre-extracted features."""
        feat_path = (
            self.data_dir / "phoenix-2014-multisigner" / 
            "features" / f"{sample['folder']}.npy"
        )
        
        if feat_path.exists():
            features = np.load(feat_path)
            return features[:self.max_frames]
        
        # Fallback to loading frames
        return self._load_frames(sample)


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for variable length sequences.
    
    Returns both CTC targets and decoder targets for hybrid training.
    
    CTC targets: Original gloss IDs (no special tokens)
    Decoder input: <sos> + gloss IDs (teacher forcing input)
    Decoder target: gloss IDs + <eos> (prediction target)
    """
    # Special token indices
    SOS_IDX = 3
    EOS_IDX = 4
    PAD_IDX = 0
    
    # Sort by length (descending)
    batch = sorted(batch, key=lambda x: x['length'], reverse=True)
    
    # Get max lengths
    max_frame_len = max(x['frames'].shape[0] for x in batch)
    max_gloss_len = max(len(x['gloss_ids']) for x in batch)
    
    # Pad frames
    frames = []
    frame_lengths = []
    src_padding_mask = []
    for x in batch:
        f = x['frames']
        seq_len = f.shape[0]
        pad_len = max_frame_len - seq_len
        
        # Padding mask: True where padded
        mask = torch.zeros(max_frame_len, dtype=torch.bool)
        mask[seq_len:] = True
        src_padding_mask.append(mask)
        
        if pad_len > 0:
            pad = torch.zeros(pad_len, *f.shape[1:])
            f = torch.cat([f, pad], dim=0)
        frames.append(f)
        frame_lengths.append(x['length'])
    
    # CTC targets (original, no special tokens)
    ctc_targets = []
    gloss_lengths = []
    for x in batch:
        g = x['gloss_ids']
        pad_len = max_gloss_len - len(g)
        if pad_len > 0:
            g = torch.cat([g, torch.zeros(pad_len, dtype=torch.long)])
        ctc_targets.append(g)
        gloss_lengths.append(len(x['gloss_ids']))
    
    # Decoder input: <sos> + gloss_ids (for teacher forcing)
    # Decoder target: gloss_ids + <eos> (what we predict)
    max_dec_len = max_gloss_len + 1  # +1 for <sos> or <eos>
    decoder_input = []
    decoder_target = []
    tgt_padding_mask = []
    
    for x in batch:
        g = x['gloss_ids']
        g_len = len(g)
        
        # Decoder input: <sos> + glosses
        dec_in = torch.cat([
            torch.tensor([SOS_IDX], dtype=torch.long),
            g
        ])
        # Pad to max_dec_len
        if len(dec_in) < max_dec_len:
            dec_in = torch.cat([dec_in, torch.zeros(max_dec_len - len(dec_in), dtype=torch.long)])
        decoder_input.append(dec_in)
        
        # Decoder target: glosses + <eos>
        dec_tgt = torch.cat([
            g,
            torch.tensor([EOS_IDX], dtype=torch.long)
        ])
        # Pad to max_dec_len
        if len(dec_tgt) < max_dec_len:
            dec_tgt = torch.cat([dec_tgt, torch.zeros(max_dec_len - len(dec_tgt), dtype=torch.long)])
        decoder_target.append(dec_tgt)
        
        # Target padding mask: True where padded
        mask = torch.zeros(max_dec_len, dtype=torch.bool)
        mask[g_len + 1:] = True  # +1 because we include <eos>
        tgt_padding_mask.append(mask)
    
    return {
        'ids': [x['id'] for x in batch],
        'frames': torch.stack(frames),
        'frame_lengths': torch.tensor(frame_lengths),
        'src_padding_mask': torch.stack(src_padding_mask),
        
        # CTC targets (no special tokens)
        'ctc_targets': torch.stack(ctc_targets),
        'gloss_lengths': torch.tensor(gloss_lengths),
        
        # Decoder targets
        'decoder_input': torch.stack(decoder_input),    # <sos> + glosses + <pad>
        'decoder_target': torch.stack(decoder_target),  # glosses + <eos> + <pad>
        'tgt_padding_mask': torch.stack(tgt_padding_mask),
        
        # Original glosses for display
        'glosses': [x['glosses'] for x in batch],
        
        # For backward compatibility
        'gloss_ids': torch.stack(ctc_targets)
    }




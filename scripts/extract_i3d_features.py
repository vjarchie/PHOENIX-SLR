"""
Extract I3D features from PHOENIX dataset videos.

Usage:
    python scripts/extract_i3d_features.py \
        --data_dir data/phoenix2014-release \
        --output_dir data/i3d_features \
        --device cuda

Requirements:
    pip install torch torchvision tqdm numpy opencv-python

Note: This script uses PyTorch's R3D-18 as a lighter alternative to I3D.
      For true I3D, install: pip install video_features
"""

import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision import transforms
from tqdm import tqdm
import csv
from PIL import Image
from typing import List, Tuple


class VideoFeatureExtractor:
    """Extract spatio-temporal features from video frames using R3D-18."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
        # Load pretrained R3D-18 (similar to I3D but lighter)
        print("Loading R3D-18 model pretrained on Kinetics-400...")
        weights = R3D_18_Weights.KINETICS400_V1
        self.model = r3d_18(weights=weights)
        
        # Remove final classification layer to get features
        self.model.fc = nn.Identity()
        self.model = self.model.to(device)
        self.model.eval()
        
        # Feature dimension after removing FC layer
        self.feature_dim = 512  # R3D-18 outputs 512-dim features
        
        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.43216, 0.394666, 0.37645],
                std=[0.22803, 0.22145, 0.216989]
            )
        ])
        
        print(f"Model loaded. Feature dimension: {self.feature_dim}")
    
    def load_frames(self, frame_dir: Path, max_frames: int = 300) -> np.ndarray:
        """Load frames from a directory."""
        frame_files = sorted(frame_dir.glob("*.png"))
        if not frame_files:
            frame_files = sorted(frame_dir.glob("*.jpg"))
        
        if not frame_files:
            return None
        
        # Subsample if too many frames
        if len(frame_files) > max_frames:
            indices = np.linspace(0, len(frame_files) - 1, max_frames, dtype=int)
            frame_files = [frame_files[i] for i in indices]
        
        frames = []
        for f in frame_files:
            img = Image.open(f).convert('RGB')
            img_tensor = self.transform(img)
            frames.append(img_tensor)
        
        return torch.stack(frames)  # (T, C, H, W)
    
    @torch.no_grad()
    def extract(self, frames: torch.Tensor, chunk_size: int = 16) -> np.ndarray:
        """
        Extract features from frames.
        
        Args:
            frames: (T, C, H, W) tensor of video frames
            chunk_size: Number of frames to process together (R3D processes 16 frames)
        
        Returns:
            features: (num_chunks, 512) numpy array
        """
        T = frames.shape[0]
        
        if T < chunk_size:
            # Pad with zeros if fewer than chunk_size frames
            padding = torch.zeros(chunk_size - T, *frames.shape[1:])
            frames = torch.cat([frames, padding], dim=0)
            T = chunk_size
        
        all_features = []
        
        # Process in chunks of chunk_size frames
        for i in range(0, T, chunk_size):
            chunk = frames[i:i + chunk_size]
            
            # Pad last chunk if needed
            if len(chunk) < chunk_size:
                padding = torch.zeros(chunk_size - len(chunk), *chunk.shape[1:])
                chunk = torch.cat([chunk, padding], dim=0)
            
            # Reshape for R3D: (N, C, T, H, W)
            chunk = chunk.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)
            chunk = chunk.to(self.device)
            
            # Extract features
            features = self.model(chunk)  # (1, 512)
            all_features.append(features.cpu().numpy())
        
        return np.concatenate(all_features, axis=0)  # (num_chunks, 512)


def parse_corpus(corpus_path: Path) -> List[Tuple[str, str, List[str]]]:
    """Parse PHOENIX corpus CSV file."""
    samples = []
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='|')
        for row in reader:
            video_id = row['id']
            folder = row['folder'].replace('/*.png', '').replace('\\', '/')
            annotation = row['annotation'].split()
            samples.append((video_id, folder, annotation))
    
    return samples


def main():
    parser = argparse.ArgumentParser(description='Extract I3D features from PHOENIX dataset')
    parser.add_argument('--data_dir', type=str, default='data/phoenix2014-release',
                        help='Path to PHOENIX dataset')
    parser.add_argument('--output_dir', type=str, default='data/i3d_features',
                        help='Output directory for features')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--max_frames', type=int, default=300,
                        help='Maximum frames per video')
    parser.add_argument('--chunk_size', type=int, default=16,
                        help='Frames per chunk for R3D')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Initialize extractor
    extractor = VideoFeatureExtractor(device=args.device)
    
    # Process each split
    splits = ['train', 'dev', 'test']
    base_path = data_dir / 'phoenix-2014-multisigner'
    
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing {split} split...")
        print('='*60)
        
        # Create output directory
        split_output = output_dir / split
        split_output.mkdir(parents=True, exist_ok=True)
        
        # Parse corpus
        corpus_path = base_path / 'annotations' / 'manual' / f'{split}.corpus.csv'
        if not corpus_path.exists():
            print(f"Corpus file not found: {corpus_path}")
            continue
        
        samples = parse_corpus(corpus_path)
        print(f"Found {len(samples)} samples")
        
        # Save annotations for later use
        annotations = {}
        
        # Process each video
        for video_id, folder, annotation in tqdm(samples, desc=f"Extracting {split}"):
            # Find frame directory (folder is like "videoname/1", need to add split prefix)
            frame_dir = base_path / 'features' / 'fullFrame-210x260px' / split / folder
            
            if not frame_dir.exists():
                print(f"Frame directory not found: {frame_dir}")
                continue
            
            # Load frames
            frames = extractor.load_frames(frame_dir, max_frames=args.max_frames)
            if frames is None:
                print(f"No frames found in: {frame_dir}")
                continue
            
            # Extract features
            features = extractor.extract(frames, chunk_size=args.chunk_size)
            
            # Save features
            feature_path = split_output / f"{video_id}.npy"
            np.save(feature_path, features)
            
            # Save annotation
            annotations[video_id] = annotation
        
        # Save annotations
        annotations_path = split_output / 'annotations.npy'
        np.save(annotations_path, annotations, allow_pickle=True)
        print(f"Saved {len(annotations)} annotations to {annotations_path}")
    
    print(f"\n{'='*60}")
    print("Feature extraction complete!")
    print(f"Features saved to: {output_dir}")
    print('='*60)


if __name__ == '__main__':
    main()


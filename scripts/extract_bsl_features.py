# -*- coding: utf-8 -*-
"""
Extract video features for Sign Language Recognition.

This script uses R3D-18 pretrained on Kinetics-400 (auto-downloads).
Optionally loads BSL-1K weights if you manually download them.

The Kinetics model provides good motion features and is a significant
improvement over frame-by-frame ImageNet features.

Usage:
    # Default: Kinetics-pretrained R3D-18 (automatic download)
    python scripts/extract_bsl_features.py --device cuda

    # With manually downloaded BSL-1K weights (better for sign language)
    python scripts/extract_bsl_features.py \
        --model_path checkpoints/pretrained/bsl1k_i3d.pth.tar \
        --device cuda

BSL-1K Manual Download Instructions:
    1. Visit: https://github.com/gulvarol/bsl1k
    2. Find the model download links in their README (Google Drive)
    3. Download the I3D checkpoint file
    4. Save to: checkpoints/pretrained/bsl1k_i3d.pth.tar
    5. Run with --model_path flag

References:
    - BSL-1K: Scaling up co-articulated sign language recognition (BMVC 2020)
    - https://github.com/gulvarol/bsl1k
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.video import r3d_18, R3D_18_Weights
from tqdm import tqdm
import csv
from PIL import Image
from typing import List, Tuple, Optional


class VideoFeatureExtractor(nn.Module):
    """
    Video feature extractor using R3D-18 with Kinetics-400 pretrained weights.
    
    Auto-downloads Kinetics weights. Optionally loads BSL-1K weights if provided.
    """
    
    def __init__(self, bsl_weights_path: Optional[Path] = None):
        super().__init__()
        
        # Load R3D-18 with Kinetics-400 pretrained weights (AUTO-DOWNLOADS!)
        print("=" * 60)
        print("Loading R3D-18 pretrained on Kinetics-400...")
        print("(This will auto-download ~120MB on first run)")
        print("=" * 60)
        
        self.model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        
        # Get feature dimension before removing FC
        self.feature_dim = self.model.fc.in_features  # 512
        
        # Remove classification head for feature extraction
        self.model.fc = nn.Identity()
        
        print(f"✓ Loaded Kinetics-pretrained R3D-18")
        print(f"  Feature dimension: {self.feature_dim}")
        
        # Optionally load BSL-1K weights if provided
        if bsl_weights_path:
            self._try_load_bsl_weights(bsl_weights_path)
    
    def _try_load_bsl_weights(self, checkpoint_path: Path):
        """Attempt to load BSL-1K pretrained weights."""
        if not checkpoint_path.exists():
            print(f"\n⚠ BSL-1K weights not found: {checkpoint_path}")
            print("  Continuing with Kinetics weights (still good for motion)")
            print("\n  To get BSL-1K weights:")
            print("  1. Visit: https://github.com/gulvarol/bsl1k")
            print("  2. Download I3D checkpoint from their Google Drive link")
            print(f"  3. Save to: {checkpoint_path}")
            return
        
        try:
            print(f"\nLoading BSL-1K weights from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Try to load compatible weights
            model_dict = self.model.state_dict()
            loaded_count = 0
            
            for k, v in state_dict.items():
                # Clean up key names (remove common prefixes)
                k_clean = k.replace('module.', '').replace('base_model.', '')
                
                if k_clean in model_dict and v.shape == model_dict[k_clean].shape:
                    model_dict[k_clean] = v
                    loaded_count += 1
            
            if loaded_count > 0:
                self.model.load_state_dict(model_dict, strict=False)
                print(f"✓ Loaded {loaded_count}/{len(model_dict)} layers from BSL-1K")
            else:
                print("⚠ BSL checkpoint format incompatible - using Kinetics weights")
                
        except Exception as e:
            print(f"⚠ Could not load BSL weights: {e}")
            print("  Using Kinetics-pretrained weights")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from video clip.
        
        Args:
            x: (B, C, T, H, W) video tensor
        
        Returns:
            (B, 512) feature tensor
        """
        return self.model(x)


class SignLanguageFeatureExtractor:
    """
    Complete feature extraction pipeline for sign language videos.
    
    Uses R3D-18 pretrained on Kinetics-400 (auto-downloads).
    Optionally uses BSL-1K weights if manually provided.
    """
    
    def __init__(self, device: str = 'cuda', bsl_weights_path: Optional[Path] = None):
        self.device = device
        
        # Initialize model
        self.model = VideoFeatureExtractor(bsl_weights_path=bsl_weights_path)
        self.model = self.model.to(device)
        self.model.eval()
        
        self.feature_dim = self.model.feature_dim
        
        # Preprocessing transforms (R3D-18 uses 112x112)
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.43216, 0.394666, 0.37645],  # Kinetics normalization
                std=[0.22803, 0.22145, 0.216989]
            )
        ])
        
        print(f"\n✓ Feature extractor ready on {device}")
    
    def load_frames(self, frame_dir: Path, max_frames: int = 300) -> Optional[torch.Tensor]:
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
            try:
                img = Image.open(f).convert('RGB')
                img_tensor = self.transform(img)
                frames.append(img_tensor)
            except Exception as e:
                print(f"Error loading frame {f}: {e}")
                continue
        
        if not frames:
            return None
        
        return torch.stack(frames)  # (T, C, H, W)
    
    @torch.no_grad()
    def extract(self, frames: torch.Tensor, chunk_size: int = 16) -> np.ndarray:
        """
        Extract features from frames.
        
        Args:
            frames: (T, C, H, W) tensor of video frames
            chunk_size: Number of frames to process together (R3D uses 16)
        
        Returns:
            features: (num_chunks, 512) numpy array
        """
        T = frames.shape[0]
        
        if T < chunk_size:
            # Pad with last frame if fewer than chunk_size frames
            padding = frames[-1:].repeat(chunk_size - T, 1, 1, 1)
            frames = torch.cat([frames, padding], dim=0)
            T = chunk_size
        
        all_features = []
        
        # Process in chunks of chunk_size frames
        for i in range(0, T, chunk_size):
            chunk = frames[i:i + chunk_size]
            
            # Pad last chunk if needed
            if len(chunk) < chunk_size:
                padding = chunk[-1:].repeat(chunk_size - len(chunk), 1, 1, 1)
                chunk = torch.cat([chunk, padding], dim=0)
            
            # Reshape for R3D: (T, C, H, W) -> (1, C, T, H, W)
            chunk = chunk.permute(1, 0, 2, 3).unsqueeze(0)
            chunk = chunk.to(self.device)
            
            # Extract features
            features = self.model(chunk)  # (1, 512)
            all_features.append(features.cpu().numpy())
        
        return np.concatenate(all_features, axis=0)


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
    parser = argparse.ArgumentParser(
        description='Extract video features for sign language recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default (Kinetics-pretrained, auto-downloads):
    python scripts/extract_bsl_features.py --device cuda

    # With BSL-1K weights (manual download required):
    python scripts/extract_bsl_features.py --model_path checkpoints/pretrained/bsl1k.pth --device cuda
        """
    )
    parser.add_argument('--data_dir', type=str, default='data/phoenix2014-release',
                        help='Path to PHOENIX dataset')
    parser.add_argument('--output_dir', type=str, default='data/video_features',
                        help='Output directory for features')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--max_frames', type=int, default=300,
                        help='Maximum frames per video')
    parser.add_argument('--chunk_size', type=int, default=16,
                        help='Frames per chunk for R3D')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to BSL-1K weights (optional, will use Kinetics if not provided)')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Initialize extractor
    bsl_path = Path(args.model_path) if args.model_path else None
    extractor = SignLanguageFeatureExtractor(
        device=args.device,
        bsl_weights_path=bsl_path
    )
    
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
        
        # Save annotations
        annotations = {}
        
        # Process each video
        for video_id, folder, annotation in tqdm(samples, desc=f"Extracting {split}"):
            frame_dir = base_path / 'features' / 'fullFrame-210x260px' / split / folder
            
            if not frame_dir.exists():
                continue
            
            # Load frames
            frames = extractor.load_frames(frame_dir, max_frames=args.max_frames)
            if frames is None:
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
        print(f"Saved {len(annotations)} samples to {split_output}")
    
    print(f"\n{'='*60}")
    print("Feature extraction complete!")
    print(f"Features saved to: {output_dir}")
    print(f"Feature dimension: {extractor.feature_dim}")
    print('='*60)


if __name__ == '__main__':
    main()

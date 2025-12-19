# -*- coding: utf-8 -*-
"""
Video File Demo for Sign Language Recognition.

Test the model on video files instead of live camera.

Usage:
    python demo_video.py --video path/to/video.mp4
    python demo_video.py --folder data/phoenix2014-release/...
"""

import sys
import argparse
from pathlib import Path
from glob import glob

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models.transformer import HybridCTCAttentionModel
from translation.gloss_to_english import translate_glosses


class SignLanguageInference:
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load checkpoint
        print(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.vocab = checkpoint['vocab']
        self.idx2gloss = {v: k for k, v in self.vocab.items()}
        
        # Special tokens
        self.sos_idx = self.vocab['<sos>']
        self.eos_idx = self.vocab['<eos>']
        self.pad_idx = self.vocab['<pad>']
        self.blank_idx = self.vocab['<blank>']
        
        # Initialize model
        self.model = HybridCTCAttentionModel(
            vocab_size=len(self.vocab),
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=3,
            dim_feedforward=2048,
            dropout=0.2,
            max_seq_len=500,
            ctc_weight=0.3,
            use_resnet=True
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded! Vocabulary: {len(self.vocab)} glosses")
    
    def preprocess_frames(self, frames, max_frames=64):
        """Preprocess frames for inference."""
        # Subsample if too many frames
        if len(frames) > max_frames:
            indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
            frames = [frames[i] for i in indices]
        
        processed = []
        for frame in frames:
            # Resize to PHOENIX dimensions
            frame = cv2.resize(frame, (210, 260))
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            processed.append(frame)
        
        # Stack and convert to tensor
        frames_np = np.stack(processed)  # (T, H, W, C)
        frames_tensor = torch.from_numpy(frames_np).float()
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # (T, C, H, W)
        frames_tensor = frames_tensor.unsqueeze(0).to(self.device)  # (1, T, C, H, W)
        
        return frames_tensor
    
    def predict(self, frames_tensor):
        """Run inference on preprocessed frames."""
        with torch.no_grad():
            pred_tokens = self.model.greedy_decode(
                frames_tensor,
                src_key_padding_mask=None,
                max_len=50,
                sos_idx=self.sos_idx,
                eos_idx=self.eos_idx
            )
            
            # Convert to glosses
            pred_seq = []
            for token_id in pred_tokens[0].tolist():
                if token_id == self.eos_idx:
                    break
                if token_id not in [self.sos_idx, self.pad_idx, self.blank_idx]:
                    if token_id in self.idx2gloss:
                        gloss = self.idx2gloss[token_id]
                        if not gloss.startswith('__') and not gloss.startswith('<'):
                            pred_seq.append(gloss)
            
            # Get English translation
            english = translate_glosses(pred_seq) if pred_seq else ""
            
            return pred_seq, english
    
    def process_video(self, video_path: str):
        """Process a video file."""
        print(f"\nProcessing: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None, None
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        if len(frames) < 5:
            print(f"Warning: Video too short ({len(frames)} frames)")
            return None, None
        
        print(f"  Loaded {len(frames)} frames")
        
        # Preprocess and predict
        frames_tensor = self.preprocess_frames(frames)
        glosses, english = self.predict(frames_tensor)
        
        return glosses, english
    
    def process_frame_folder(self, folder_path: str):
        """Process a folder of frame images."""
        print(f"\nProcessing folder: {folder_path}")
        
        # Find frame files
        frame_files = sorted(glob(f"{folder_path}/*.png")) + sorted(glob(f"{folder_path}/*.jpg"))
        
        if len(frame_files) < 5:
            print(f"Warning: Too few frames ({len(frame_files)})")
            return None, None
        
        print(f"  Found {len(frame_files)} frames")
        
        # Load frames
        frames = []
        for f in frame_files:
            img = cv2.imread(f)
            if img is not None:
                frames.append(img)
        
        # Preprocess and predict
        frames_tensor = self.preprocess_frames(frames)
        glosses, english = self.predict(frames_tensor)
        
        return glosses, english


def main():
    parser = argparse.ArgumentParser(description='Sign Language Recognition - Video Demo')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/e2e/best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to video file')
    parser.add_argument('--folder', type=str, default=None,
                        help='Path to folder of frame images')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    if args.video is None and args.folder is None:
        # Demo with sample from PHOENIX dataset
        sample_folder = "data/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/test/01April_2010_Thursday_heute_default-0/1"
        if Path(sample_folder).exists():
            args.folder = sample_folder
            print("No input specified, using sample from test set...")
        else:
            print("Usage:")
            print("  python demo_video.py --video path/to/video.mp4")
            print("  python demo_video.py --folder path/to/frames/")
            return
    
    # Initialize model
    model = SignLanguageInference(args.checkpoint, args.device)
    
    # Process input
    if args.video:
        glosses, english = model.process_video(args.video)
    else:
        glosses, english = model.process_frame_folder(args.folder)
    
    if glosses:
        print("\n" + "=" * 60)
        print("RESULT")
        print("=" * 60)
        print(f"German (DGS):  {' '.join(glosses)}")
        print(f"English:       {english}")
        print("=" * 60)
    else:
        print("No prediction generated")


if __name__ == '__main__':
    main()



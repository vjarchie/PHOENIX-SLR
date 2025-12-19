# -*- coding: utf-8 -*-
"""
Gloss-to-Video Retrieval Module

Retrieves and concatenates video clips from the dataset based on gloss sequences.
This is the retrieval-based approach for sign language production.
"""

import os
import cv2
import numpy as np
import pickle
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class GlossVideoRetriever:
    """
    Retrieves video clips for given gloss sequences.
    
    Uses the existing PHOENIX dataset to find video segments
    corresponding to each gloss and concatenates them.
    """
    
    def __init__(
        self,
        data_dir: str,
        index_path: str = None,
        frame_size: Tuple[int, int] = (260, 210),  # W, H
        fps: int = 25,
        blend_frames: int = 5  # Frames to blend between clips
    ):
        """
        Args:
            data_dir: Path to phoenix2014-release directory
            index_path: Path to pre-built gloss-video index
            frame_size: Output frame size (width, height)
            fps: Output video FPS
            blend_frames: Number of frames for crossfade blending
        """
        self.data_dir = Path(data_dir)
        self.frame_size = frame_size
        self.fps = fps
        self.blend_frames = blend_frames
        
        # Gloss to video segments mapping
        # Each entry: gloss -> [(video_path, start_frame, end_frame), ...]
        self.gloss_index: Dict[str, List[Dict]] = defaultdict(list)
        
        # Load or build index
        if index_path and os.path.exists(index_path):
            self.load_index(index_path)
        else:
            print("No index found. Call build_index() to create one.")
    
    def build_index(
        self,
        corpus_file: str,
        feature_type: str = 'fullFrame-210x260px',
        split: str = 'train',
        save_path: str = None
    ):
        """
        Build gloss-video index from PHOENIX dataset.
        
        Uses uniform segmentation to approximate gloss boundaries.
        For better results, use alignment data if available.
        
        Args:
            corpus_file: Path to corpus CSV file
            feature_type: Feature type (fullFrame or trackedRightHand)
            split: Dataset split (train/dev/test)
            save_path: Path to save the index
        """
        print(f"Building gloss-video index from {corpus_file}...")
        
        # Parse corpus file
        with open(corpus_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        samples_processed = 0
        glosses_indexed = 0
        
        for line in lines[1:]:  # Skip header
            parts = line.strip().split('|')
            if len(parts) < 4:
                continue
            
            sample_id = parts[0]
            folder = parts[1].replace('/*.png', '').replace('\\*.png', '')
            glosses = parts[3].split()
            
            # Get frame directory
            frame_dir = (
                self.data_dir / "phoenix-2014-multisigner" /
                "features" / feature_type / split / folder
            )
            
            if not frame_dir.exists():
                continue
            
            # Count frames
            frame_files = sorted(frame_dir.glob("*.png"))
            num_frames = len(frame_files)
            
            if num_frames == 0 or len(glosses) == 0:
                continue
            
            # Uniform segmentation: divide frames equally among glosses
            frames_per_gloss = num_frames / len(glosses)
            
            for i, gloss in enumerate(glosses):
                start_frame = int(i * frames_per_gloss)
                end_frame = int((i + 1) * frames_per_gloss)
                
                # Store segment info
                segment = {
                    'sample_id': sample_id,
                    'frame_dir': str(frame_dir),
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'num_frames': end_frame - start_frame
                }
                
                self.gloss_index[gloss].append(segment)
                glosses_indexed += 1
            
            samples_processed += 1
            if samples_processed % 500 == 0:
                print(f"  Processed {samples_processed} samples...")
        
        print(f"Index built: {len(self.gloss_index)} unique glosses, {glosses_indexed} segments")
        
        if save_path:
            self.save_index(save_path)
    
    def save_index(self, path: str):
        """Save index to file."""
        # Convert defaultdict to regular dict for pickling
        index_data = {
            'gloss_index': dict(self.gloss_index),
            'frame_size': self.frame_size,
            'fps': self.fps
        }
        
        with open(path, 'wb') as f:
            pickle.dump(index_data, f)
        print(f"Index saved to {path}")
    
    def load_index(self, path: str):
        """Load index from file."""
        with open(path, 'rb') as f:
            index_data = pickle.load(f)
        
        self.gloss_index = defaultdict(list, index_data['gloss_index'])
        self.frame_size = index_data.get('frame_size', self.frame_size)
        self.fps = index_data.get('fps', self.fps)
        
        print(f"Index loaded: {len(self.gloss_index)} glosses")
    
    def _load_segment_frames(self, segment: Dict) -> np.ndarray:
        """Load frames for a segment."""
        frame_dir = Path(segment['frame_dir'])
        start = segment['start_frame']
        end = segment['end_frame']
        
        frame_files = sorted(frame_dir.glob("*.png"))
        
        frames = []
        for i in range(start, min(end, len(frame_files))):
            img = cv2.imread(str(frame_files[i]))
            if img is not None:
                # Resize to target size
                img = cv2.resize(img, self.frame_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(img)
        
        if len(frames) == 0:
            # Return black frames if loading failed
            return np.zeros((5, self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
        
        return np.array(frames)
    
    def _crossfade_blend(self, clip1: np.ndarray, clip2: np.ndarray) -> np.ndarray:
        """
        Blend end of clip1 with start of clip2 using crossfade.
        
        Args:
            clip1: (T1, H, W, C) first clip
            clip2: (T2, H, W, C) second clip
            
        Returns:
            Blended clip
        """
        if len(clip1) < self.blend_frames or len(clip2) < self.blend_frames:
            # Just concatenate if clips are too short
            return np.concatenate([clip1, clip2], axis=0)
        
        # Get frames to blend
        end_frames = clip1[-self.blend_frames:]
        start_frames = clip2[:self.blend_frames]
        
        # Create blend
        blended = []
        for i in range(self.blend_frames):
            alpha = i / self.blend_frames
            blended_frame = (
                (1 - alpha) * end_frames[i] + alpha * start_frames[i]
            ).astype(np.uint8)
            blended.append(blended_frame)
        
        # Combine: clip1[:-blend] + blended + clip2[blend:]
        result = np.concatenate([
            clip1[:-self.blend_frames],
            np.array(blended),
            clip2[self.blend_frames:]
        ], axis=0)
        
        return result
    
    def retrieve(
        self,
        gloss_sequence: List[str],
        selection: str = 'random',
        blend: bool = True
    ) -> np.ndarray:
        """
        Retrieve video for a gloss sequence.
        
        Args:
            gloss_sequence: List of gloss tokens
            selection: How to select among multiple clips ('random', 'first', 'longest')
            blend: Whether to blend between clips
            
        Returns:
            Video frames as numpy array (T, H, W, C)
        """
        if not gloss_sequence:
            # Return blank video
            return np.zeros((30, self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
        
        clips = []
        missing_glosses = []
        
        for gloss in gloss_sequence:
            if gloss not in self.gloss_index or len(self.gloss_index[gloss]) == 0:
                missing_glosses.append(gloss)
                continue
            
            # Select segment
            segments = self.gloss_index[gloss]
            
            if selection == 'random':
                segment = random.choice(segments)
            elif selection == 'longest':
                segment = max(segments, key=lambda x: x['num_frames'])
            else:  # 'first'
                segment = segments[0]
            
            # Load frames
            frames = self._load_segment_frames(segment)
            clips.append(frames)
        
        if missing_glosses:
            print(f"Warning: Missing glosses in index: {missing_glosses}")
        
        if not clips:
            return np.zeros((30, self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
        
        # Concatenate clips
        if blend and len(clips) > 1:
            result = clips[0]
            for i in range(1, len(clips)):
                result = self._crossfade_blend(result, clips[i])
        else:
            result = np.concatenate(clips, axis=0)
        
        return result
    
    def retrieve_as_video(
        self,
        gloss_sequence: List[str],
        output_path: str,
        selection: str = 'random'
    ) -> str:
        """
        Retrieve and save as video file.
        
        Args:
            gloss_sequence: List of gloss tokens
            output_path: Output video path (e.g., 'output.mp4')
            selection: Clip selection method
            
        Returns:
            Path to saved video
        """
        frames = self.retrieve(gloss_sequence, selection=selection)
        
        # Write video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            self.fps,
            self.frame_size
        )
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        return output_path
    
    def get_available_glosses(self) -> List[str]:
        """Get list of all indexed glosses."""
        return list(self.gloss_index.keys())
    
    def get_gloss_stats(self, gloss: str) -> Dict:
        """Get statistics for a gloss."""
        if gloss not in self.gloss_index:
            return {'exists': False}
        
        segments = self.gloss_index[gloss]
        num_frames = [s['num_frames'] for s in segments]
        
        return {
            'exists': True,
            'num_segments': len(segments),
            'avg_frames': np.mean(num_frames),
            'min_frames': np.min(num_frames),
            'max_frames': np.max(num_frames)
        }


class AlignmentBasedRetriever(GlossVideoRetriever):
    """
    Enhanced retriever that uses alignment data for precise gloss boundaries.
    
    Uses the alignment files from PHOENIX if available.
    """
    
    def build_index_from_alignment(
        self,
        alignment_file: str,
        feature_type: str = 'fullFrame-210x260px',
        split: str = 'train',
        save_path: str = None
    ):
        """
        Build index using alignment data for precise boundaries.
        
        Args:
            alignment_file: Path to alignment file (train.alignment)
            feature_type: Feature type
            split: Dataset split
            save_path: Path to save index
        """
        print(f"Building index from alignment: {alignment_file}...")
        
        with open(alignment_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Parse alignment format
        # Format varies by dataset - adjust as needed
        for line in lines:
            # Parse alignment (implementation depends on file format)
            pass
        
        if save_path:
            self.save_index(save_path)


def build_default_index(data_dir: str, output_path: str = None) -> GlossVideoRetriever:
    """
    Convenience function to build index from PHOENIX data.
    
    Args:
        data_dir: Path to phoenix2014-release
        output_path: Where to save the index
        
    Returns:
        Configured GlossVideoRetriever
    """
    retriever = GlossVideoRetriever(data_dir)
    
    corpus_file = os.path.join(
        data_dir,
        "phoenix-2014-multisigner",
        "annotations", "manual",
        "train.corpus.csv"
    )
    
    if output_path is None:
        output_path = os.path.join(data_dir, "gloss_video_index.pkl")
    
    retriever.build_index(
        corpus_file=corpus_file,
        feature_type='fullFrame-210x260px',
        split='train',
        save_path=output_path
    )
    
    return retriever


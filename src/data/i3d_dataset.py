"""
Dataset for loading pre-extracted I3D features.

This is much faster than loading raw frames since features are pre-computed.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


class I3DFeatureDataset(Dataset):
    """
    Dataset for pre-extracted I3D/R3D features.
    
    Features are stored as .npy files: (num_chunks, feature_dim)
    Annotations are stored in annotations.npy
    """
    
    def __init__(
        self,
        features_dir: str,
        split: str = 'train',
        vocab: Optional[Dict[str, int]] = None,
        max_seq_len: int = 300
    ):
        """
        Args:
            features_dir: Directory containing extracted features
            split: 'train', 'dev', or 'test'
            vocab: Existing vocabulary (for dev/test splits)
            max_seq_len: Maximum sequence length
        """
        self.features_dir = Path(features_dir) / split
        self.split = split
        self.max_seq_len = max_seq_len
        
        # Load annotations
        annotations_path = self.features_dir / 'annotations.npy'
        self.annotations = np.load(annotations_path, allow_pickle=True).item()
        
        # Get list of feature files
        self.samples = []
        for video_id, glosses in self.annotations.items():
            feature_path = self.features_dir / f"{video_id}.npy"
            if feature_path.exists():
                self.samples.append({
                    'id': video_id,
                    'feature_path': feature_path,
                    'glosses': glosses
                })
        
        print(f"Loaded {len(self.samples)} samples from {split}")
        
        # Build or use existing vocabulary
        if vocab is None:
            self.vocab = self._build_vocab()
        else:
            self.vocab = vocab
        
        self.idx2gloss = {v: k for k, v in self.vocab.items()}
        self.feature_dim = self._get_feature_dim()
        
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Feature dimension: {self.feature_dim}")
    
    def _get_feature_dim(self) -> int:
        """Get feature dimension from first sample."""
        if len(self.samples) > 0:
            features = np.load(self.samples[0]['feature_path'])
            return features.shape[-1]
        return 512  # Default for R3D-18
    
    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary from annotations."""
        vocab = {
            '<pad>': 0,
            '<unk>': 1,
            '<blank>': 2,
            '<sos>': 3,
            '<eos>': 4
        }
        
        for sample in self.samples:
            for gloss in sample['glosses']:
                if gloss not in vocab:
                    vocab[gloss] = len(vocab)
        
        return vocab
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Load features
        features = np.load(sample['feature_path'])  # (num_chunks, feature_dim)
        
        # Truncate if needed
        if len(features) > self.max_seq_len:
            features = features[:self.max_seq_len]
        
        # Convert glosses to IDs
        glosses = sample['glosses']
        gloss_ids = [self.vocab.get(g, self.vocab['<unk>']) for g in glosses]
        
        # Add <sos> and <eos> for decoder
        decoder_input_ids = [self.vocab['<sos>']] + gloss_ids
        decoder_target_ids = gloss_ids + [self.vocab['<eos>']]
        
        return {
            'id': sample['id'],
            'features': torch.tensor(features, dtype=torch.float32),
            'gloss_ids': torch.tensor(gloss_ids, dtype=torch.long),
            'glosses': glosses,
            'length': len(features),
            'decoder_input_ids': torch.tensor(decoder_input_ids, dtype=torch.long),
            'decoder_target_ids': torch.tensor(decoder_target_ids, dtype=torch.long)
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for DataLoader."""
    # Sort by feature length (descending)
    batch = sorted(batch, key=lambda x: x['length'], reverse=True)
    
    # Get max lengths
    max_feature_len = max(x['length'] for x in batch)
    max_gloss_len = max(len(x['gloss_ids']) for x in batch)
    max_decoder_len = max(len(x['decoder_input_ids']) for x in batch)
    
    # Get feature dimension
    feature_dim = batch[0]['features'].shape[-1]
    
    # Pad features
    features = []
    feature_lengths = []
    for x in batch:
        f = x['features']
        pad_len = max_feature_len - len(f)
        if pad_len > 0:
            padding = torch.zeros(pad_len, feature_dim)
            f = torch.cat([f, padding], dim=0)
        features.append(f)
        feature_lengths.append(x['length'])
    
    # Pad gloss IDs
    gloss_ids = []
    gloss_lengths = []
    for x in batch:
        g = x['gloss_ids']
        pad_len = max_gloss_len - len(g)
        if pad_len > 0:
            g = torch.cat([g, torch.zeros(pad_len, dtype=torch.long)])
        gloss_ids.append(g)
        gloss_lengths.append(len(x['gloss_ids']))
    
    # Pad decoder inputs and targets
    decoder_input_ids = []
    decoder_target_ids = []
    decoder_lengths = []
    for x in batch:
        di = x['decoder_input_ids']
        dt = x['decoder_target_ids']
        pad_len = max_decoder_len - len(di)
        if pad_len > 0:
            di = torch.cat([di, torch.zeros(pad_len, dtype=torch.long)])
            dt = torch.cat([dt, torch.zeros(pad_len, dtype=torch.long)])
        decoder_input_ids.append(di)
        decoder_target_ids.append(dt)
        decoder_lengths.append(len(x['decoder_input_ids']))
    
    return {
        'ids': [x['id'] for x in batch],
        'features': torch.stack(features),
        'gloss_ids': torch.stack(gloss_ids),
        'feature_lengths': torch.tensor(feature_lengths),
        'gloss_lengths': torch.tensor(gloss_lengths),
        'glosses': [x['glosses'] for x in batch],
        'decoder_input_ids': torch.stack(decoder_input_ids),
        'decoder_target_ids': torch.stack(decoder_target_ids),
        'decoder_lengths': torch.tensor(decoder_lengths)
    }



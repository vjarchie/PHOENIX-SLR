# -*- coding: utf-8 -*-
"""
Data Preparation Script for Text-to-Gloss Training

Extracts text-gloss pairs from PHOENIX dataset annotations.
The PHOENIX dataset contains German glosses, so we use them as both
source (in text form) and target (as gloss sequences).

For real German text → DGS gloss, you would need:
1. German spoken language annotations (not in PHOENIX)
2. Or synthetic generation from glosses

This script creates training data using the available gloss sequences,
treating lowercase glosses as "text" and uppercase as "glosses".
"""

import os
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def load_corpus(corpus_file: str) -> List[Dict]:
    """Load annotations from corpus file."""
    samples = []
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines[1:]:  # Skip header
        parts = line.strip().split('|')
        if len(parts) >= 4:
            sample = {
                'id': parts[0],
                'folder': parts[1],
                'signer': parts[2],
                'glosses': parts[3].split()
            }
            samples.append(sample)
    
    return samples


def gloss_to_pseudo_text(glosses: List[str]) -> str:
    """
    Convert gloss sequence to pseudo-text for training.
    
    This creates a synthetic "German-like" text from glosses
    by lowercasing and adding simple markers.
    
    For real application, you'd need actual German translations.
    """
    # Convert glosses to lowercase "words"
    words = []
    for gloss in glosses:
        # Handle special markers
        if gloss.startswith('__') and gloss.endswith('__'):
            continue  # Skip markers like __ON__, __OFF__
        if gloss.startswith('loc-'):
            # Location marker - extract the location
            loc = gloss.replace('loc-', '').lower()
            words.append(f"in {loc}")
        elif gloss.startswith('cl-'):
            # Classifier - skip or convert
            continue
        elif gloss.startswith('poss-'):
            # Possessive
            words.append(gloss.replace('poss-', '').lower())
        elif '-PLUSPLUS' in gloss:
            # Plural/emphasis marker
            base = gloss.replace('-PLUSPLUS', '').lower()
            words.append(f"viel {base}")
        else:
            words.append(gloss.lower().replace('-', ' '))
    
    return ' '.join(words)


def create_training_pairs(
    samples: List[Dict],
    include_pseudo_text: bool = True
) -> List[Dict]:
    """
    Create text-gloss training pairs.
    
    Args:
        samples: List of corpus samples
        include_pseudo_text: Whether to generate pseudo-text from glosses
        
    Returns:
        List of training pairs with 'text' and 'glosses' keys
    """
    pairs = []
    
    for sample in samples:
        glosses = sample['glosses']
        
        # Filter out special tokens
        clean_glosses = [
            g for g in glosses 
            if not (g.startswith('__') and g.endswith('__'))
        ]
        
        if not clean_glosses:
            continue
        
        if include_pseudo_text:
            text = gloss_to_pseudo_text(clean_glosses)
        else:
            text = ' '.join(clean_glosses).lower()
        
        pairs.append({
            'id': sample['id'],
            'text': text,
            'glosses': clean_glosses
        })
    
    return pairs


def build_vocabularies(pairs: List[Dict]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Build text and gloss vocabularies."""
    
    # Text vocabulary
    text_vocab = {
        '<pad>': 0,
        '<unk>': 1,
        '<sos>': 2,
        '<eos>': 3
    }
    
    # Gloss vocabulary (reuse existing if possible)
    gloss_vocab = {
        '<pad>': 0,
        '<unk>': 1,
        '<blank>': 2,
        '<sos>': 3,
        '<eos>': 4
    }
    
    # Build from data
    for pair in pairs:
        # Text words
        words = pair['text'].lower().split()
        for word in words:
            if word not in text_vocab:
                text_vocab[word] = len(text_vocab)
        
        # Glosses
        for gloss in pair['glosses']:
            if gloss not in gloss_vocab:
                gloss_vocab[gloss] = len(gloss_vocab)
    
    return text_vocab, gloss_vocab


def save_data(
    pairs: List[Dict],
    text_vocab: Dict,
    gloss_vocab: Dict,
    output_dir: str,
    train_ratio: float = 0.9
):
    """Save training data and vocabularies."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Shuffle and split
    random.shuffle(pairs)
    split_idx = int(len(pairs) * train_ratio)
    
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    
    # Save pairs
    with open(output_dir / 'train_pairs.json', 'w', encoding='utf-8') as f:
        json.dump(train_pairs, f, ensure_ascii=False, indent=2)
    
    with open(output_dir / 'val_pairs.json', 'w', encoding='utf-8') as f:
        json.dump(val_pairs, f, ensure_ascii=False, indent=2)
    
    # Save vocabularies
    with open(output_dir / 'text_vocab.json', 'w', encoding='utf-8') as f:
        json.dump(text_vocab, f, ensure_ascii=False, indent=2)
    
    with open(output_dir / 'gloss_vocab.json', 'w', encoding='utf-8') as f:
        json.dump(gloss_vocab, f, ensure_ascii=False, indent=2)
    
    print(f"\nData saved to {output_dir}")
    print(f"  Training pairs: {len(train_pairs)}")
    print(f"  Validation pairs: {len(val_pairs)}")
    print(f"  Text vocabulary: {len(text_vocab)} words")
    print(f"  Gloss vocabulary: {len(gloss_vocab)} glosses")


def main():
    """Main data preparation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare Text-to-Gloss training data')
    parser.add_argument('--data-dir', type=str, default='data/phoenix2014-release',
                       help='Path to phoenix2014-release directory')
    parser.add_argument('--output-dir', type=str, default='data/text2gloss',
                       help='Output directory for processed data')
    parser.add_argument('--train-ratio', type=float, default=0.9,
                       help='Training data ratio')
    
    args = parser.parse_args()
    
    # Find corpus file
    corpus_file = os.path.join(
        args.data_dir,
        'phoenix-2014-multisigner',
        'annotations', 'manual',
        'train.corpus.csv'
    )
    
    if not os.path.exists(corpus_file):
        print(f"Error: Corpus file not found: {corpus_file}")
        return
    
    print(f"Loading corpus from {corpus_file}...")
    samples = load_corpus(corpus_file)
    print(f"Loaded {len(samples)} samples")
    
    print("\nCreating training pairs...")
    pairs = create_training_pairs(samples, include_pseudo_text=True)
    print(f"Created {len(pairs)} pairs")
    
    print("\nBuilding vocabularies...")
    text_vocab, gloss_vocab = build_vocabularies(pairs)
    
    print("\nSaving data...")
    save_data(pairs, text_vocab, gloss_vocab, args.output_dir, args.train_ratio)
    
    # Show examples
    print("\n" + "="*60)
    print("Sample pairs:")
    print("="*60)
    for i, pair in enumerate(pairs[:5]):
        print(f"\n[{i+1}] ID: {pair['id']}")
        print(f"    Text: {pair['text']}")
        print(f"    Gloss: {' '.join(pair['glosses'])}")


if __name__ == '__main__':
    main()


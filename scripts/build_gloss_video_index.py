# -*- coding: utf-8 -*-
"""
Build Gloss-Video Index

Creates an index mapping each gloss to video segments from the PHOENIX dataset.
This index is used by the retrieval-based video synthesis system.

Run this once before using the Speech-to-Sign pipeline.
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.speech_to_sign.gloss_retriever import GlossVideoRetriever


def main():
    parser = argparse.ArgumentParser(description='Build gloss-video index')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/phoenix2014-release',
        help='Path to phoenix2014-release directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='gloss_video_index.pkl',
        help='Output index file path'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'dev', 'test'],
        help='Dataset split to use'
    )
    parser.add_argument(
        '--feature-type',
        type=str,
        default='fullFrame-210x260px',
        help='Feature type (fullFrame or trackedRightHand)'
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Verify data exists
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("\nMake sure the PHOENIX dataset is extracted to this location.")
        return
    
    # Find corpus file
    corpus_file = data_dir / "phoenix-2014-multisigner" / "annotations" / "manual" / f"{args.split}.corpus.csv"
    
    if not corpus_file.exists():
        print(f"Error: Corpus file not found: {corpus_file}")
        return
    
    print("="*60)
    print("Building Gloss-Video Index")
    print("="*60)
    print(f"\nData directory: {data_dir}")
    print(f"Split: {args.split}")
    print(f"Feature type: {args.feature_type}")
    print(f"Output: {args.output}")
    print()
    
    # Create retriever and build index
    retriever = GlossVideoRetriever(str(data_dir))
    
    retriever.build_index(
        corpus_file=str(corpus_file),
        feature_type=args.feature_type,
        split=args.split,
        save_path=args.output
    )
    
    # Print statistics
    print("\n" + "="*60)
    print("Index Statistics")
    print("="*60)
    
    glosses = retriever.get_available_glosses()
    print(f"\nTotal unique glosses: {len(glosses)}")
    
    # Top glosses by segment count
    gloss_counts = [
        (g, len(retriever.gloss_index[g]))
        for g in glosses
    ]
    gloss_counts.sort(key=lambda x: -x[1])
    
    print("\nTop 20 glosses by segment count:")
    for g, c in gloss_counts[:20]:
        stats = retriever.get_gloss_stats(g)
        print(f"  {g}: {c} segments, avg {stats['avg_frames']:.1f} frames")
    
    print("\n" + "="*60)
    print(f"Index saved to: {args.output}")
    print("="*60)


if __name__ == '__main__':
    main()


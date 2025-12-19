# -*- coding: utf-8 -*-
"""
Test the trained Sign Language Recognition model.

Usage:
    python test_model.py --checkpoint checkpoints/e2e/best.pth --split test
"""

import sys
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.phoenix_dataset import PhoenixDataset, collate_fn
from data.augmentation import get_val_transforms
from models.transformer import HybridCTCAttentionModel


def compute_wer(predictions: list, targets: list) -> float:
    """Compute Word Error Rate."""
    total_errors = 0
    total_words = 0
    
    for pred, target in zip(predictions, targets):
        if isinstance(pred, str):
            pred = pred.split()
        if isinstance(target, str):
            target = target.split()
        
        m, n = len(pred), len(target)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred[i-1] == target[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        total_errors += dp[m][n]
        total_words += max(n, 1)
    
    return total_errors / max(total_words, 1)


def main(args):
    print("=" * 60)
    print("Sign Language Recognition - Model Testing")
    print("=" * 60)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    vocab = checkpoint['vocab']
    best_wer = checkpoint.get('best_wer', 'N/A')
    epoch = checkpoint.get('epoch', 'N/A')
    
    print(f"  Trained for: {epoch} epochs")
    print(f"  Best dev WER: {best_wer*100:.2f}%" if isinstance(best_wer, float) else f"  Best WER: {best_wer}")
    print(f"  Vocabulary size: {len(vocab)}")
    
    # Load dataset
    print(f"\nLoading {args.split} dataset...")
    val_transforms = get_val_transforms(target_frames=64)
    
    dataset = PhoenixDataset(
        args.data_dir,
        split=args.split,
        max_frames=64,
        transform=val_transforms,
        vocab=vocab
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = HybridCTCAttentionModel(
        vocab_size=len(vocab),
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
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Special tokens
    idx2gloss = {v: k for k, v in vocab.items()}
    sos_idx = vocab['<sos>']
    eos_idx = vocab['<eos>']
    pad_idx = vocab['<pad>']
    blank_idx = vocab['<blank>']
    
    # Evaluate
    print(f"\nEvaluating on {args.split} split ({len(dataset)} samples)...")
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            frames = batch['frames'].to(device)
            frame_lengths = batch['frame_lengths']
            
            max_frame_len = frames.size(1)
            src_key_padding_mask = torch.arange(max_frame_len, device=device).unsqueeze(0) >= frame_lengths.unsqueeze(1).to(device)
            
            # Decode using attention decoder
            pred_tokens = model.greedy_decode(
                frames,
                src_key_padding_mask=src_key_padding_mask,
                max_len=50,
                sos_idx=sos_idx,
                eos_idx=eos_idx
            )
            
            # Convert to glosses
            for i in range(pred_tokens.size(0)):
                pred_seq = []
                for token_id in pred_tokens[i].tolist():
                    if token_id == eos_idx:
                        break
                    if token_id not in [sos_idx, pad_idx, blank_idx]:
                        if token_id in idx2gloss:
                            pred_seq.append(idx2gloss[token_id])
                all_predictions.append(pred_seq)
            
            all_targets.extend(batch['glosses'])
    
    # Compute WER
    wer = compute_wer(all_predictions, all_targets)
    
    print("\n" + "=" * 60)
    print(f"RESULTS on {args.split.upper()} split")
    print("=" * 60)
    print(f"  Samples: {len(dataset)}")
    print(f"  WER: {wer*100:.2f}%")
    print("=" * 60)
    
    # Show some examples
    if args.show_examples > 0:
        print(f"\nExample predictions (first {args.show_examples}):")
        print("-" * 60)
        for i in range(min(args.show_examples, len(all_predictions))):
            print(f"\n[{i+1}] Target:     {' '.join(all_targets[i])}")
            print(f"    Prediction: {' '.join(all_predictions[i])}")
    
    return wer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Sign Language Recognition Model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/e2e/best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/phoenix2014-release',
                        help='Path to PHOENIX dataset')
    parser.add_argument('--split', type=str, default='test', choices=['dev', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--show_examples', type=int, default=5,
                        help='Number of example predictions to show')
    
    args = parser.parse_args()
    main(args)



# -*- coding: utf-8 -*-
"""
Evaluation Script for PHOENIX Sign Language Recognition

Computes Word Error Rate (WER) on dev/test sets.
"""

import sys
import json
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.transformer import SignLanguageTransformer, CNNTransformer
from data.phoenix_dataset import PhoenixDataset, collate_fn


def levenshtein_distance(pred: list, target: list) -> tuple:
    """
    Compute Levenshtein distance and return (substitutions, insertions, deletions).
    """
    m, n = len(pred), len(target)
    
    # DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i-1] == target[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # deletion
                    dp[i][j-1],      # insertion
                    dp[i-1][j-1]     # substitution
                )
    
    return dp[m][n]


def compute_wer(predictions: list, targets: list) -> dict:
    """
    Compute Word Error Rate and related metrics.
    
    Returns:
        dict with WER, total_errors, total_words, etc.
    """
    total_errors = 0
    total_words = 0
    total_correct = 0
    
    for pred, target in zip(predictions, targets):
        pred_words = pred if isinstance(pred, list) else pred.split()
        target_words = target if isinstance(target, list) else target.split()
        
        errors = levenshtein_distance(pred_words, target_words)
        total_errors += errors
        total_words += len(target_words)
        
        if pred_words == target_words:
            total_correct += 1
    
    wer = total_errors / max(total_words, 1) * 100
    accuracy = total_correct / max(len(predictions), 1) * 100
    
    return {
        'wer': wer,
        'total_errors': total_errors,
        'total_words': total_words,
        'sentence_accuracy': accuracy,
        'num_samples': len(predictions)
    }


def ctc_decode(log_probs: torch.Tensor, lengths: torch.Tensor, idx2gloss: dict, blank_idx: int = 2) -> list:
    """Greedy CTC decoding."""
    predictions = []
    batch_size = log_probs.size(1)
    
    for b in range(batch_size):
        seq_len = lengths[b].item() if lengths is not None else log_probs.size(0)
        best_path = torch.argmax(log_probs[:seq_len, b, :], dim=-1).cpu().numpy()
        
        # Collapse repeated and remove blanks
        decoded = []
        prev = -1
        blank_count = 0
        for idx in best_path:
            if idx == blank_idx:
                blank_count += 1
            elif idx != prev:
                decoded.append(idx2gloss.get(idx, '<unk>'))
            prev = idx
        
        predictions.append(decoded)
    
    # Calculate blank ratio
    blank_ratio = blank_count / len(best_path) if len(best_path) > 0 else 0
    
    return predictions


def evaluate(model, dataloader, device, idx2gloss, blank_idx=2):
    """Evaluate model and compute WER."""
    model.eval()
    all_predictions = []
    all_targets = []
    total_blank_ratio = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            frames = batch['frames'].to(device)
            frame_lengths = batch['frame_lengths']
            
            # Forward pass
            log_probs = model(frames)
            
            # Decode predictions
            predictions = ctc_decode(log_probs, frame_lengths, idx2gloss, blank_idx)
            
            # Compute blank ratio for monitoring
            batch_size = log_probs.size(1)
            for b in range(batch_size):
                seq_len = frame_lengths[b].item()
                best_path = torch.argmax(log_probs[:seq_len, b, :], dim=-1).cpu()
                blank_count = (best_path == blank_idx).sum().item()
                total_blank_ratio += blank_count / seq_len
            
            num_batches += batch_size
            
            all_predictions.extend(predictions)
            all_targets.extend(batch['glosses'])
    
    avg_blank_ratio = total_blank_ratio / max(num_batches, 1)
    
    # Compute WER
    metrics = compute_wer(all_predictions, all_targets)
    metrics['blank_ratio'] = avg_blank_ratio
    
    return metrics, all_predictions, all_targets


def main():
    parser = argparse.ArgumentParser(description="Evaluate PHOENIX SLR Model")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/phoenix2014-release')
    parser.add_argument('--split', type=str, default='dev', choices=['dev', 'test'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default=None, help='Path to save predictions')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    checkpoint_path = Path(args.checkpoint)
    checkpoint_dir = checkpoint_path.parent
    
    logger.info(f"Device: {device}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    
    # Load config and vocab
    with open(checkpoint_dir / 'config.json', 'r') as f:
        config = json.load(f)
    
    with open(checkpoint_dir / 'vocab.json', 'r') as f:
        vocab = json.load(f)
    
    idx2gloss = {v: k for k, v in vocab.items()}
    
    logger.info(f"Vocabulary size: {len(vocab)}")
    
    # Load dataset
    dataset = PhoenixDataset(
        args.data_dir,
        split=args.split,
        max_frames=300,
        load_video=True,
        vocab=vocab
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Load model
    if config.get('model_type', 'transformer') == 'cnn_transformer':
        model = CNNTransformer(
            input_dim=config['input_dim'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_encoder_layers=config['num_layers'],
            vocab_size=config['vocab_size']
        )
    else:
        model = SignLanguageTransformer(
            input_dim=config['input_dim'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_encoder_layers=config['num_layers'],
            vocab_size=config['vocab_size'],
            use_cnn_backbone=config.get('use_cnn_backbone', True),
            cnn_type=config.get('cnn_type', 'simple')
        )
    
    # Load weights
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    
    logger.info(f"Model loaded from epoch {ckpt.get('epoch', 'unknown')}")
    
    # Evaluate
    metrics, predictions, targets = evaluate(
        model, dataloader, device, idx2gloss, 
        blank_idx=vocab.get('<blank>', 2)
    )
    
    # Print results
    logger.info("="*60)
    logger.info(f"EVALUATION RESULTS ({args.split} set)")
    logger.info("="*60)
    logger.info(f"Word Error Rate (WER): {metrics['wer']:.2f}%")
    logger.info(f"Sentence Accuracy: {metrics['sentence_accuracy']:.2f}%")
    logger.info(f"Total Errors: {metrics['total_errors']}")
    logger.info(f"Total Words: {metrics['total_words']}")
    logger.info(f"Blank Ratio: {metrics['blank_ratio']*100:.1f}%")
    logger.info(f"Samples: {metrics['num_samples']}")
    logger.info("="*60)
    
    # Show sample predictions
    logger.info("\nSample Predictions:")
    for i in range(min(5, len(predictions))):
        logger.info(f"\n[{i+1}]")
        logger.info(f"  Target: {' '.join(targets[i])}")
        logger.info(f"  Pred:   {' '.join(predictions[i])}")
    
    # Save predictions if requested
    if args.output:
        output_data = {
            'metrics': metrics,
            'predictions': [' '.join(p) for p in predictions],
            'targets': [' '.join(t) for t in targets]
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"\nPredictions saved to: {args.output}")
    
    # Warning for CTC collapse
    if metrics['blank_ratio'] > 0.8:
        logger.warning("⚠️  HIGH BLANK RATIO DETECTED!")
        logger.warning("This may indicate CTC collapse. Consider:")
        logger.warning("  1. Training longer")
        logger.warning("  2. Using smaller learning rate")
        logger.warning("  3. Adding data augmentation")


if __name__ == "__main__":
    main()




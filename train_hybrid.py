# -*- coding: utf-8 -*-
"""
Hybrid CTC + Attention Training Script for PHOENIX Sign Language Recognition

Joint training with CTC and Cross-Entropy losses to prevent CTC collapse.
Reference: "Sign Language Transformers" (Camgöz et al., 2020)
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Training log file for dashboard
TRAINING_LOG_FILE = Path(__file__).parent / "training_log.jsonl"

def log_metrics(epoch, train_loss, ctc_loss, ce_loss, dev_loss, wer, blank_ratio, lr):
    """Log metrics to JSONL file for dashboard visualization."""
    # Convert any tensors to Python floats
    def to_float(x):
        if hasattr(x, 'item'):
            return x.item()
        return float(x)
    
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "epoch": epoch,
        "train_loss": to_float(train_loss),
        "ctc_loss": to_float(ctc_loss),
        "ce_loss": to_float(ce_loss),
        "dev_loss": to_float(dev_loss),
        "wer": to_float(wer),
        "blank_ratio": to_float(blank_ratio),
        "learning_rate": to_float(lr)
    }
    with open(TRAINING_LOG_FILE, "a") as f:
        f.write(json.dumps(metrics) + "\n")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.transformer import HybridCTCAttentionModel
from data.phoenix_dataset import PhoenixDataset, collate_fn


def compute_wer(predictions: list, targets: list) -> float:
    """Compute Word Error Rate."""
    total_errors = 0
    total_words = 0
    
    for pred, target in zip(predictions, targets):
        pred_words = pred if isinstance(pred, list) else pred.split()
        target_words = target if isinstance(target, list) else target.split()
        
        # Dynamic programming for edit distance
        m, n = len(pred_words), len(target_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_words[i-1] == target_words[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        total_errors += dp[m][n]
        total_words += n
    
    return total_errors / max(total_words, 1) * 100


def ctc_decode(log_probs: torch.Tensor, lengths: torch.Tensor, idx2gloss: dict, blank_idx: int = 2) -> list:
    """Greedy CTC decoding."""
    predictions = []
    batch_size = log_probs.size(1)
    
    for b in range(batch_size):
        seq_len = lengths[b].item() if lengths is not None else log_probs.size(0)
        best_path = torch.argmax(log_probs[:seq_len, b, :], dim=-1).cpu().numpy()
        
        decoded = []
        prev = -1
        for idx in best_path:
            if idx != prev and idx != blank_idx:
                decoded.append(idx2gloss.get(idx, '<unk>'))
            prev = idx
        
        predictions.append(decoded)
    
    return predictions


def compute_blank_ratio(log_probs: torch.Tensor, lengths: torch.Tensor, blank_idx: int = 2) -> float:
    """Compute the ratio of blank predictions."""
    batch_size = log_probs.size(1)
    total_blank = 0
    total_frames = 0
    
    for b in range(batch_size):
        seq_len = lengths[b].item() if lengths is not None else log_probs.size(0)
        best_path = torch.argmax(log_probs[:seq_len, b, :], dim=-1)
        blank_count = (best_path == blank_idx).sum().item()
        total_blank += blank_count
        total_frames += seq_len
    
    return total_blank / max(total_frames, 1)


def train_epoch(model, dataloader, optimizer, device, epoch, blank_idx=2, pad_idx=0):
    """Train for one epoch with hybrid CTC + Attention loss."""
    model.train()
    total_loss = 0
    total_ctc_loss = 0
    total_ce_loss = 0
    total_blank_ratio = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Move data to device
        frames = batch['frames'].to(device)
        frame_lengths = batch['frame_lengths']
        src_padding_mask = batch['src_padding_mask'].to(device)
        
        # CTC targets
        ctc_targets = batch['ctc_targets'].to(device)
        gloss_lengths = batch['gloss_lengths']
        
        # Decoder targets
        decoder_input = batch['decoder_input'].to(device)
        decoder_target = batch['decoder_target'].to(device)
        tgt_padding_mask = batch['tgt_padding_mask'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        ctc_log_probs, decoder_logits, _ = model(
            frames, 
            decoder_input,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        # Compute joint loss
        loss, ctc_loss_val, ce_loss_val = model.compute_loss(
            ctc_log_probs,
            decoder_logits,
            ctc_targets,
            decoder_target,
            frame_lengths,
            gloss_lengths,
            ctc_blank=blank_idx,
            pad_idx=pad_idx
        )
        
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_ctc_loss += ctc_loss_val
            total_ce_loss += ce_loss_val
            num_batches += 1
            
            # Monitor blank ratio
            with torch.no_grad():
                blank_ratio = compute_blank_ratio(ctc_log_probs, frame_lengths, blank_idx)
                total_blank_ratio += blank_ratio
        else:
            blank_ratio = 0
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ctc': f'{ctc_loss_val:.4f}',
            'ce': f'{ce_loss_val:.4f}',
            'blank': f'{blank_ratio*100:.1f}%'
        })
    
    n = max(num_batches, 1)
    return total_loss/n, total_ctc_loss/n, total_ce_loss/n, total_blank_ratio/n


def evaluate(model, dataloader, device, idx2gloss, blank_idx=2, pad_idx=0):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            frames = batch['frames'].to(device)
            frame_lengths = batch['frame_lengths']
            src_padding_mask = batch['src_padding_mask'].to(device)
            ctc_targets = batch['ctc_targets'].to(device)
            gloss_lengths = batch['gloss_lengths']
            decoder_input = batch['decoder_input'].to(device)
            decoder_target = batch['decoder_target'].to(device)
            tgt_padding_mask = batch['tgt_padding_mask'].to(device)
            
            # Forward
            ctc_log_probs, decoder_logits, _ = model(
                frames, decoder_input,
                src_key_padding_mask=src_padding_mask,
                tgt_key_padding_mask=tgt_padding_mask
            )
            
            # Loss
            loss, _, _ = model.compute_loss(
                ctc_log_probs, decoder_logits,
                ctc_targets, decoder_target,
                frame_lengths, gloss_lengths,
                blank_idx, pad_idx
            )
            
            if torch.isfinite(torch.tensor(loss)):
                total_loss += loss
                num_batches += 1
            
            # Decode using CTC (faster)
            predictions = ctc_decode(ctc_log_probs, frame_lengths, idx2gloss, blank_idx)
            all_predictions.extend(predictions)
            all_targets.extend(batch['glosses'])
    
    avg_loss = total_loss / max(num_batches, 1)
    wer = compute_wer(all_predictions, all_targets)
    
    return avg_loss, wer


def main():
    parser = argparse.ArgumentParser(description="Train Hybrid CTC+Attention Model")
    parser.add_argument('--data_dir', type=str, default='data/phoenix2014-release')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/hybrid')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_encoder_layers', type=int, default=6)
    parser.add_argument('--num_decoder_layers', type=int, default=3)
    parser.add_argument('--ctc_weight', type=float, default=0.3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume', action='store_true')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    checkpoint_path = Path(args.checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("HYBRID CTC + ATTENTION TRAINING")
    logger.info("="*60)
    logger.info(f"Device: {device}")
    logger.info(f"CTC weight: {args.ctc_weight}")
    logger.info(f"CE weight: {1 - args.ctc_weight}")
    
    # Clear previous training log
    if TRAINING_LOG_FILE.exists() and not args.resume:
        TRAINING_LOG_FILE.unlink()
    
    # Load dataset
    logger.info("Loading dataset...")
    train_dataset = PhoenixDataset(
        args.data_dir, 
        split='train',
        max_frames=64,
        load_video=True
    )
    
    dev_dataset = PhoenixDataset(
        args.data_dir,
        split='dev',
        max_frames=64,
        load_video=True,
        vocab=train_dataset.vocab
    )
    
    logger.info(f"Vocabulary size: {len(train_dataset.vocab)}")
    logger.info(f"Special tokens: <pad>=0, <unk>=1, <blank>=2, <sos>=3, <eos>=4")
    
    # DataLoaders
    num_workers = 2 if args.device == 'cuda' else 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    # Create hybrid model
    model = HybridCTCAttentionModel(
        vocab_size=len(train_dataset.vocab),
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        ctc_weight=args.ctc_weight,
        use_resnet=True
    )
    model = model.to(device)
    
    # Save config
    config = {
        'model_type': 'hybrid',
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_encoder_layers': args.num_encoder_layers,
        'num_decoder_layers': args.num_decoder_layers,
        'vocab_size': len(train_dataset.vocab),
        'ctc_weight': args.ctc_weight
    }
    with open(checkpoint_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save vocabulary
    with open(checkpoint_path / 'vocab.json', 'w') as f:
        json.dump(train_dataset.vocab, f, indent=2)
    
    # Optimizer with warmup
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Warmup + Cosine Annealing
    warmup_epochs = 3
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return 0.1 + 0.9 * (epoch / warmup_epochs)
        else:
            progress = (epoch - warmup_epochs) / (args.epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training state
    best_wer = float('inf')
    start_epoch = 0
    
    if args.resume and (checkpoint_path / 'latest.pth').exists():
        ckpt = torch.load(checkpoint_path / 'latest.pth', map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_wer = ckpt.get('best_wer', float('inf'))
        logger.info(f"Resumed from epoch {start_epoch}")
    
    logger.info("="*60)
    logger.info("STARTING HYBRID TRAINING")
    logger.info("="*60)
    
    blank_idx = train_dataset.vocab['<blank>']
    pad_idx = train_dataset.vocab['<pad>']
    
    for epoch in range(start_epoch, args.epochs):
        # Dynamic loss weighting (optional: more CE early, more CTC later)
        # Uncomment to use dynamic weighting:
        # progress = epoch / args.epochs
        # model.ctc_weight = 0.1 + 0.4 * progress  # 0.1 -> 0.5
        # model.ce_weight = 1.0 - model.ctc_weight
        
        # Train
        train_loss, ctc_loss, ce_loss, blank_ratio = train_epoch(
            model, train_loader, optimizer, device, epoch + 1, blank_idx, pad_idx
        )
        
        # Evaluate
        dev_loss, dev_wer = evaluate(
            model, dev_loader, device, train_dataset.idx2gloss, blank_idx, pad_idx
        )
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log
        logger.info(
            f"Epoch {epoch+1}: Loss={train_loss:.4f} (CTC={ctc_loss:.4f}, CE={ce_loss:.4f}), "
            f"Dev Loss={dev_loss:.4f}, WER={dev_wer:.2f}%, Blank={blank_ratio*100:.1f}%"
        )
        
        log_metrics(epoch+1, train_loss, ctc_loss, ce_loss, dev_loss, dev_wer, blank_ratio, current_lr)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'dev_loss': dev_loss,
            'wer': dev_wer,
            'best_wer': best_wer,
            'blank_ratio': blank_ratio
        }
        
        torch.save(checkpoint, checkpoint_path / 'latest.pth')
        
        if dev_wer < best_wer:
            best_wer = dev_wer
            checkpoint['best_wer'] = best_wer
            torch.save(checkpoint, checkpoint_path / 'best.pth')
            logger.info(f"  *** New best WER: {best_wer:.2f}% ***")
    
    logger.info("="*60)
    logger.info(f"TRAINING COMPLETE - Best WER: {best_wer:.2f}%")
    logger.info("="*60)


if __name__ == "__main__":
    main()


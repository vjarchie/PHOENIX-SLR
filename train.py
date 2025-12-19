# -*- coding: utf-8 -*-
"""
Training Script for PHOENIX Sign Language Recognition

Uses Transformer + CTC architecture.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
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


def log_metrics(epoch, train_loss, dev_loss, wer, blank_ratio, lr):
    """Log metrics to JSONL file for dashboard visualization."""
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "epoch": epoch,
        "train_loss": train_loss,
        "dev_loss": dev_loss,
        "wer": wer,
        "blank_ratio": blank_ratio,
        "learning_rate": lr
    }
    with open(TRAINING_LOG_FILE, "a") as f:
        f.write(json.dumps(metrics) + "\n")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.transformer import SignLanguageTransformer, CNNTransformer
from data.phoenix_dataset import PhoenixDataset, collate_fn


def compute_wer(predictions: list, targets: list) -> float:
    """Compute Word Error Rate."""
    total_errors = 0
    total_words = 0
    
    for pred, target in zip(predictions, targets):
        # Simple Levenshtein distance
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
    
    # log_probs: (seq, batch, vocab)
    batch_size = log_probs.size(1)
    
    for b in range(batch_size):
        seq_len = lengths[b].item() if lengths is not None else log_probs.size(0)
        
        # Get best path
        best_path = torch.argmax(log_probs[:seq_len, b, :], dim=-1).cpu().numpy()
        
        # Collapse repeated and remove blanks
        decoded = []
        prev = -1
        for idx in best_path:
            if idx != prev and idx != blank_idx:
                decoded.append(idx2gloss.get(idx, '<unk>'))
            prev = idx
        
        predictions.append(decoded)
    
    return predictions


def compute_blank_ratio(log_probs: torch.Tensor, lengths: torch.Tensor, blank_idx: int = 2) -> float:
    """Compute the ratio of blank predictions (for CTC collapse monitoring)."""
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


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, blank_idx=2):
    """Train for one epoch with CTC collapse monitoring."""
    model.train()
    total_loss = 0
    num_batches = 0
    total_blank_ratio = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        frames = batch['frames'].to(device)
        gloss_ids = batch['gloss_ids'].to(device)
        frame_lengths = batch['frame_lengths']
        gloss_lengths = batch['gloss_lengths']
        
        optimizer.zero_grad()
        
        # Forward pass
        log_probs = model(frames)  # (seq, batch, vocab)
        
        # Compute input lengths (after model processing)
        input_lengths = frame_lengths.clone()
        
        # CTC loss
        loss = criterion(
            log_probs, 
            gloss_ids, 
            input_lengths, 
            gloss_lengths
        )
        
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Monitor blank ratio (key CTC collapse indicator)
            with torch.no_grad():
                blank_ratio = compute_blank_ratio(log_probs, frame_lengths, blank_idx)
                total_blank_ratio += blank_ratio
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'blank': f'{blank_ratio*100:.1f}%' if torch.isfinite(loss) else 'N/A'
        })
    
    avg_loss = total_loss / max(num_batches, 1)
    avg_blank_ratio = total_blank_ratio / max(num_batches, 1)
    
    return avg_loss, avg_blank_ratio


def evaluate(model, dataloader, criterion, device, idx2gloss):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            frames = batch['frames'].to(device)
            gloss_ids = batch['gloss_ids'].to(device)
            frame_lengths = batch['frame_lengths']
            gloss_lengths = batch['gloss_lengths']
            
            # Forward pass
            log_probs = model(frames)
            
            # CTC loss
            loss = criterion(log_probs, gloss_ids, frame_lengths, gloss_lengths)
            
            if torch.isfinite(loss):
                total_loss += loss.item()
                num_batches += 1
            
            # Decode predictions
            predictions = ctc_decode(log_probs, frame_lengths, idx2gloss)
            all_predictions.extend(predictions)
            all_targets.extend(batch['glosses'])
    
    avg_loss = total_loss / max(num_batches, 1)
    wer = compute_wer(all_predictions, all_targets)
    
    return avg_loss, wer


def main():
    parser = argparse.ArgumentParser(description="Train PHOENIX SLR Model")
    parser.add_argument('--data_dir', type=str, default='data/phoenix2014-release')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/transformer')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_type', type=str, default='transformer', 
                        choices=['transformer', 'cnn_transformer'])
    parser.add_argument('--resume', action='store_true')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    checkpoint_path = Path(args.checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Device: {device}")
    logger.info(f"Model type: {args.model_type}")
    
    # Load dataset
    # IMPORTANT: Reduced max_frames to 64 to prevent OOM on 64GB RAM
    # Each frame is 210x260x3 floats = 655KB, so 64 frames = ~42MB per sample
    logger.info("Loading dataset...")
    train_dataset = PhoenixDataset(
        args.data_dir, 
        split='train',
        max_frames=64,  # Reduced from 128 to prevent RAM exhaustion
        load_video=True
    )
    
    dev_dataset = PhoenixDataset(
        args.data_dir,
        split='dev',
        max_frames=64,  # Reduced from 128
        load_video=True,
        vocab=train_dataset.vocab
    )
    
    # DataLoader configuration - balanced for memory efficiency
    # Reduced prefetch to prevent RAM exhaustion
    num_workers = 2 if args.device == 'cuda' else 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,  # Don't keep workers alive (saves RAM)
        prefetch_factor=2 if num_workers > 0 else None  # Limit prefetch
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
    
    # Create model
    # CNN backbone extracts 512-dim features from raw frames
    input_dim = 512  # Feature dimension after CNN backbone
    
    if args.model_type == 'transformer':
        model = SignLanguageTransformer(
            input_dim=input_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_layers,
            vocab_size=len(train_dataset.vocab),
            use_cnn_backbone=True,  # Use CNN to process raw frames
            cnn_type='resnet'       # Using ResNet-18 pretrained on ImageNet for better features
        )
    else:
        model = CNNTransformer(
            input_dim=input_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_layers,
            vocab_size=len(train_dataset.vocab)
        )
    
    model = model.to(device)
    
    # Save config
    config = {
        'model_type': args.model_type,
        'input_dim': input_dim,
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_layers': args.num_layers,
        'vocab_size': len(train_dataset.vocab),
        'use_cnn_backbone': True,
        'cnn_type': 'resnet'  # ResNet-18 pretrained on ImageNet
    }
    with open(checkpoint_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save vocabulary
    with open(checkpoint_path / 'vocab.json', 'w') as f:
        json.dump(train_dataset.vocab, f, indent=2)
    
    # Optimizer and scheduler with warmup to avoid CTC collapse
    # Higher initial LR with warmup helps model learn features before CTC takes over
    optimizer = optim.AdamW(model.parameters(), lr=args.lr * 3, weight_decay=0.01)  # 3x higher LR
    
    # Warmup + Cosine Annealing scheduler
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup from 0.1x to 1x over warmup_epochs
            return 0.1 + 0.9 * (epoch / warmup_epochs)
        else:
            # Cosine annealing after warmup
            progress = (epoch - warmup_epochs) / (args.epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    logger.info(f"Using warmup ({warmup_epochs} epochs) + cosine annealing, peak LR: {args.lr * 3}")
    
    # CTC Loss
    ctc_loss = nn.CTCLoss(blank=train_dataset.vocab['<blank>'], reduction='mean', zero_infinity=True)
    
    # Training loop
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
    logger.info("STARTING TRAINING")
    logger.info("="*60)
    
    blank_idx = train_dataset.vocab['<blank>']
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_blank_ratio = train_epoch(
            model, train_loader, optimizer, ctc_loss, device, epoch + 1, blank_idx
        )
        
        # Evaluate
        dev_loss, dev_wer = evaluate(model, dev_loader, ctc_loss, device, train_dataset.idx2gloss)
        
        scheduler.step()
        
        # Log with CTC collapse warning
        logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Dev Loss={dev_loss:.4f}, WER={dev_wer:.2f}%, Blank={train_blank_ratio*100:.1f}%")
        
        # Log metrics to file for dashboard
        current_lr = optimizer.param_groups[0]['lr']
        log_metrics(epoch + 1, train_loss, dev_loss, dev_wer, train_blank_ratio, current_lr)
        
        # Warning for potential CTC collapse
        if train_blank_ratio > 0.8:
            logger.warning("HIGH BLANK RATIO! Model may be experiencing CTC collapse.")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'dev_loss': dev_loss,
            'wer': dev_wer,
            'best_wer': best_wer,
            'blank_ratio': train_blank_ratio
        }
        
        torch.save(checkpoint, checkpoint_path / 'latest.pth')
        
        if dev_wer < best_wer:
            best_wer = dev_wer
            checkpoint['best_wer'] = best_wer
            torch.save(checkpoint, checkpoint_path / 'best.pth')
            logger.info(f"  New best WER: {best_wer:.2f}%")
    
    logger.info("="*60)
    logger.info(f"TRAINING COMPLETE - Best WER: {best_wer:.2f}%")
    logger.info("="*60)


if __name__ == "__main__":
    main()




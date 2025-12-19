# -*- coding: utf-8 -*-
"""
End-to-End Hybrid CTC + Attention Training with Data Augmentation.

This script trains the CNN backbone end-to-end with the Transformer,
using data augmentation to prevent overfitting.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.phoenix_dataset import PhoenixDataset, collate_fn
from data.augmentation import get_train_transforms, get_val_transforms
from models.transformer import HybridCTCAttentionModel


def compute_wer(predictions: list, targets: list) -> float:
    """Compute Word Error Rate."""
    total_errors = 0
    total_words = 0
    
    for pred, target in zip(predictions, targets):
        # Handle string or list inputs
        if isinstance(pred, str):
            pred = pred.split()
        if isinstance(target, str):
            target = target.split()
        
        # Levenshtein distance
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


def compute_blank_ratio(log_probs, lengths, blank_idx):
    """Compute ratio of blank predictions."""
    predictions = log_probs.argmax(dim=-1)  # (T, B)
    total_blanks = 0
    total_frames = 0
    
    for b in range(predictions.size(1)):
        seq_len = lengths[b].item() if isinstance(lengths[b], torch.Tensor) else lengths[b]
        seq = predictions[:seq_len, b]
        total_blanks += (seq == blank_idx).sum().item()
        total_frames += seq_len
    
    return total_blanks / max(total_frames, 1)


def ctc_decode(log_probs, lengths, idx2gloss, blank_idx):
    """Greedy CTC decoding."""
    predictions = log_probs.argmax(dim=-1)  # (T, B)
    batch_size = predictions.size(1)
    
    decoded = []
    for b in range(batch_size):
        seq_len = lengths[b].item() if isinstance(lengths[b], torch.Tensor) else lengths[b]
        seq = predictions[:seq_len, b].tolist()
        
        # Remove blanks and consecutive duplicates
        result = []
        prev = None
        for idx in seq:
            if idx != blank_idx and idx != prev:
                if idx in idx2gloss:
                    result.append(idx2gloss[idx])
            prev = idx
        
        decoded.append(result)
    
    return decoded


def log_metrics(log_file: Path, epoch: int, metrics: dict):
    """Log metrics to JSONL file."""
    entry = {
        'timestamp': datetime.now().isoformat(),
        'epoch': epoch,
        **{k: float(v) if isinstance(v, (int, float, torch.Tensor)) else v 
           for k, v in metrics.items()}
    }
    
    with open(log_file, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def train_epoch(model, dataloader, optimizer, ctc_criterion, ce_criterion, 
                device, epoch, blank_idx, ctc_weight, ce_weight, max_grad_norm=5.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_ctc_loss = 0
    total_ce_loss = 0
    total_blank_ratio = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        optimizer.zero_grad()
        
        # Get batch data
        frames = batch['frames'].to(device)  # (B, T, C, H, W)
        gloss_ids = batch['ctc_targets'].to(device)
        frame_lengths = batch['frame_lengths']
        gloss_lengths = batch['gloss_lengths']
        decoder_input_ids = batch['decoder_input'].to(device)
        decoder_target_ids = batch['decoder_target'].to(device)
        # Compute decoder lengths from gloss_lengths (+1 for <sos> or <eos>)
        decoder_lengths = gloss_lengths + 1
        
        batch_size = frames.size(0)
        max_frame_len = frames.size(1)
        max_dec_len = decoder_input_ids.size(1)
        
        # Create masks
        src_key_padding_mask = torch.arange(max_frame_len, device=device).unsqueeze(0) >= frame_lengths.unsqueeze(1).to(device)
        tgt_key_padding_mask = torch.arange(max_dec_len, device=device).unsqueeze(0) >= decoder_lengths.unsqueeze(1).to(device)
        
        # Forward pass (returns ctc_log_probs, decoder_logits, encoder_output)
        try:
            ctc_log_probs, decoder_output, _ = model(
                frames,
                decoder_input_ids,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
        except RuntimeError as e:
            print(f"Forward pass error: {e}")
            continue
        
        # CTC loss
        ctc_input_lengths = frame_lengths.to(device)
        ctc_target_lengths = gloss_lengths.to(device)
        
        try:
            ctc_loss = ctc_criterion(
                ctc_log_probs,
                gloss_ids,
                ctc_input_lengths,
                ctc_target_lengths
            )
        except RuntimeError:
            ctc_loss = torch.tensor(0.0, device=device)
        
        # CE loss
        ce_loss = ce_criterion(
            decoder_output.view(-1, model.vocab_size),
            decoder_target_ids.view(-1)
        )
        
        # Combined loss
        loss = ctc_weight * ctc_loss + ce_weight * ce_loss
        
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            
            total_loss += loss.item()
            total_ctc_loss += ctc_loss.item() if torch.isfinite(ctc_loss) else 0
            total_ce_loss += ce_loss.item()
            num_batches += 1
            
            with torch.no_grad():
                blank_ratio = compute_blank_ratio(ctc_log_probs, frame_lengths, blank_idx)
                total_blank_ratio += blank_ratio
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ctc': f'{ctc_loss.item():.4f}' if torch.isfinite(ctc_loss) else 'N/A',
            'ce': f'{ce_loss.item():.4f}',
            'blank': f'{blank_ratio*100:.1f}%' if torch.isfinite(loss) else 'N/A'
        })
    
    return {
        'loss': total_loss / max(num_batches, 1),
        'ctc_loss': total_ctc_loss / max(num_batches, 1),
        'ce_loss': total_ce_loss / max(num_batches, 1),
        'blank_ratio': total_blank_ratio / max(num_batches, 1)
    }


@torch.no_grad()
def evaluate(model, dataloader, ctc_criterion, ce_criterion, device, idx2gloss, blank_idx,
             sos_idx, eos_idx, pad_idx):
    """Evaluate model using decoder predictions."""
    model.eval()
    total_loss = 0
    total_ctc_loss = 0
    total_ce_loss = 0
    total_blank_ratio = 0
    num_batches = 0
    all_predictions = []
    all_targets = []
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        frames = batch['frames'].to(device)
        gloss_ids = batch['ctc_targets'].to(device)
        frame_lengths = batch['frame_lengths']
        gloss_lengths = batch['gloss_lengths']
        decoder_input_ids = batch['decoder_input'].to(device)
        decoder_target_ids = batch['decoder_target'].to(device)
        decoder_lengths = gloss_lengths + 1
        
        max_frame_len = frames.size(1)
        max_dec_len = decoder_input_ids.size(1)
        
        src_key_padding_mask = torch.arange(max_frame_len, device=device).unsqueeze(0) >= frame_lengths.unsqueeze(1).to(device)
        tgt_key_padding_mask = torch.arange(max_dec_len, device=device).unsqueeze(0) >= decoder_lengths.unsqueeze(1).to(device)
        
        ctc_log_probs, decoder_output, _ = model(
            frames,
            decoder_input_ids,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Losses
        try:
            ctc_loss = ctc_criterion(ctc_log_probs, gloss_ids, frame_lengths.to(device), gloss_lengths.to(device))
        except:
            ctc_loss = torch.tensor(0.0, device=device)
        
        ce_loss = ce_criterion(decoder_output.view(-1, model.vocab_size), decoder_target_ids.view(-1))
        
        if torch.isfinite(ctc_loss) and torch.isfinite(ce_loss):
            total_ctc_loss += ctc_loss.item()
            total_ce_loss += ce_loss.item()
            total_loss += (0.3 * ctc_loss.item() + 0.7 * ce_loss.item())
            blank_ratio = compute_blank_ratio(ctc_log_probs, frame_lengths, blank_idx)
            total_blank_ratio += blank_ratio
            num_batches += 1
        
        # Decode using attention decoder (greedy)
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
    
    wer = compute_wer(all_predictions, all_targets)
    
    return {
        'loss': total_loss / max(num_batches, 1),
        'ctc_loss': total_ctc_loss / max(num_batches, 1),
        'ce_loss': total_ce_loss / max(num_batches, 1),
        'wer': wer,
        'blank_ratio': total_blank_ratio / max(num_batches, 1)
    }


def main(args):
    print("=" * 60)
    print("End-to-End Hybrid Training with Data Augmentation")
    print("=" * 60)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Log file
    log_file = Path('training_log.jsonl')
    if log_file.exists():
        log_file.unlink()
    
    # Get augmentation transforms
    train_transforms = get_train_transforms(target_frames=args.max_frames)
    val_transforms = get_val_transforms(target_frames=args.max_frames)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = PhoenixDataset(
        args.data_dir,
        split='train',
        max_frames=args.max_frames,
        transform=train_transforms
    )
    
    dev_dataset = PhoenixDataset(
        args.data_dir,
        split='dev',
        max_frames=args.max_frames,
        transform=val_transforms,
        vocab=train_dataset.vocab
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=False
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Model parameters
    vocab_size = len(train_dataset.vocab)
    
    print(f"\nInitializing model...")
    model = HybridCTCAttentionModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        ctc_weight=args.ctc_weight,
        use_resnet=True  # End-to-end ResNet-18 backbone
    )
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss functions
    ctc_criterion = nn.CTCLoss(blank=train_dataset.vocab['<blank>'], zero_infinity=True)
    ce_criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab['<pad>'], label_smoothing=0.1)
    
    # Optimizer with different LR for CNN backbone
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'cnn' in name.lower() or 'backbone' in name.lower():
            backbone_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},  # Lower LR for pretrained backbone
        {'params': other_params, 'lr': args.lr}
    ], weight_decay=args.weight_decay)
    
    # LR scheduler with warmup
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + torch.cos(torch.tensor((epoch - warmup_epochs) / (args.epochs - warmup_epochs) * 3.14159)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    best_wer = float('inf')
    patience = 15
    no_improve_count = 0
    
    idx2gloss = train_dataset.idx2gloss
    blank_idx = train_dataset.vocab['<blank>']
    sos_idx = train_dataset.vocab['<sos>']
    eos_idx = train_dataset.vocab['<eos>']
    pad_idx = train_dataset.vocab['<pad>']
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"CTC weight: {args.ctc_weight}, CE weight: {args.ce_weight}")
    print(f"Data augmentation: ENABLED")
    print(f"Label smoothing: 0.1")
    print("=" * 60)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, ctc_criterion, ce_criterion,
            device, epoch, blank_idx, args.ctc_weight, args.ce_weight
        )
        
        # Evaluate
        dev_metrics = evaluate(
            model, dev_loader, ctc_criterion, ce_criterion,
            device, idx2gloss, blank_idx, sos_idx, eos_idx, pad_idx
        )
        
        # Update LR
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_metrics['loss']:.4f} (CTC: {train_metrics['ctc_loss']:.4f}, CE: {train_metrics['ce_loss']:.4f})")
        print(f"  Dev Loss: {dev_metrics['loss']:.4f} (CTC: {dev_metrics['ctc_loss']:.4f}, CE: {dev_metrics['ce_loss']:.4f})")
        print(f"  Dev WER: {dev_metrics['wer']*100:.2f}%")
        print(f"  Blank Ratio: {train_metrics['blank_ratio']*100:.1f}%")
        print(f"  LR: {current_lr:.6f}")
        
        # Log metrics
        log_metrics(log_file, epoch, {
            'train_loss': train_metrics['loss'],
            'train_ctc_loss': train_metrics['ctc_loss'],
            'train_ce_loss': train_metrics['ce_loss'],
            'dev_loss': dev_metrics['loss'],
            'dev_ctc_loss': dev_metrics['ctc_loss'],
            'dev_ce_loss': dev_metrics['ce_loss'],
            'dev_wer': dev_metrics['wer'],
            'blank_ratio': train_metrics['blank_ratio'],
            'learning_rate': current_lr
        })
        
        # Save best model
        if dev_metrics['wer'] < best_wer:
            best_wer = dev_metrics['wer']
            no_improve_count = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_wer': best_wer,
                'vocab': train_dataset.vocab
            }, checkpoint_dir / 'best.pth')
            print(f"  [NEW BEST] WER: {best_wer*100:.2f}%")
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break
        
        # Save latest checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_wer': best_wer,
            'vocab': train_dataset.vocab
        }, checkpoint_dir / 'latest.pth')
        
        print("-" * 60)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best WER: {best_wer*100:.2f}%")
    print(f"Model saved to: {checkpoint_dir / 'best.pth'}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='End-to-End Hybrid Training with Augmentation')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data/phoenix2014-release',
                        help='Path to PHOENIX dataset')
    parser.add_argument('--max_frames', type=int, default=64,
                        help='Maximum frames per video')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--ctc_weight', type=float, default=0.3)
    parser.add_argument('--ce_weight', type=float, default=0.7)
    
    # Model
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_encoder_layers', type=int, default=6)
    parser.add_argument('--num_decoder_layers', type=int, default=3)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.2)  # Higher dropout
    parser.add_argument('--max_seq_len', type=int, default=500)
    
    # System
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/e2e')
    
    args = parser.parse_args()
    main(args)


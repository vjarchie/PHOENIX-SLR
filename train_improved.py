# -*- coding: utf-8 -*-
"""
Improved Training Script with All Enhancements.

This script incorporates:
1. Enhanced temporal augmentation (crop-paste, dropout, multi-scale)
2. Gradient accumulation for effective batch size = 8
3. Optimized hyperparameters based on research

Usage:
    python train_improved.py --device cuda

For BSL-pretrained features:
    1. First extract features: python scripts/extract_bsl_features.py --device cuda
    2. Then train: python train_improved.py --use_bsl_features --device cuda
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
from data.augmentation import get_train_transforms, get_val_transforms, apply_augmentations
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


def compute_blank_ratio(log_probs, lengths, blank_idx):
    """Compute ratio of blank predictions."""
    predictions = log_probs.argmax(dim=-1)
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
    predictions = log_probs.argmax(dim=-1)
    batch_size = predictions.size(1)
    
    decoded = []
    for b in range(batch_size):
        seq_len = lengths[b].item() if isinstance(lengths[b], torch.Tensor) else lengths[b]
        seq = predictions[:seq_len, b].tolist()
        
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
                device, epoch, blank_idx, ctc_weight, ce_weight, 
                max_grad_norm=5.0, accumulation_steps=4, use_amp=False, scaler=None):
    """
    Train for one epoch with gradient accumulation and optional mixed precision.
    
    Args:
        use_amp: Use automatic mixed precision (bf16/fp16)
        scaler: GradScaler for AMP (only needed for fp16, not bf16)
    """
    model.train()
    total_loss = 0
    total_ctc_loss = 0
    total_ce_loss = 0
    total_blank_ratio = 0
    num_batches = 0
    
    optimizer.zero_grad()
    
    effective_batch = dataloader.batch_size * accumulation_steps
    amp_str = " [bf16]" if use_amp else ""
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} (eff.batch={effective_batch}){amp_str}")
    
    # Determine autocast dtype (bf16 preferred for RTX 30/40/50 series)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    for batch_idx, batch in enumerate(pbar):
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
        
        # Mixed precision context (bf16 preferred, saves ~50% VRAM)
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            try:
                ctc_log_probs, decoder_output, _, output_seq_len = model(
                    frames,
                    decoder_input_ids,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask
                )
            except RuntimeError as e:
                print(f"Forward pass error: {e}")
                continue
            
            # Adjust input lengths for compressed sequence (R3D backbone)
            input_seq_len = frame_lengths.max().item()
            if output_seq_len != input_seq_len:
                # Scale frame lengths proportionally
                scale = output_seq_len / input_seq_len
                ctc_input_lengths = (frame_lengths.float() * scale).ceil().long().clamp(min=1).to(device)
            else:
                ctc_input_lengths = frame_lengths.to(device)
            ctc_target_lengths = gloss_lengths.to(device)
            
            try:
                ctc_loss = ctc_criterion(ctc_log_probs.float(), gloss_ids, ctc_input_lengths, ctc_target_lengths)
            except RuntimeError:
                ctc_loss = torch.tensor(0.0, device=device)
            
            ce_loss = ce_criterion(decoder_output.view(-1, model.vocab_size), decoder_target_ids.view(-1))
            
            loss = (ctc_weight * ctc_loss + ce_weight * ce_loss) / accumulation_steps
        
        if torch.isfinite(loss):
            # Backward pass (scaler handles fp16, bf16 doesn't need it)
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_loss += loss.item() * accumulation_steps
            total_ctc_loss += ctc_loss.item() if torch.isfinite(ctc_loss) else 0
            total_ce_loss += ce_loss.item()
            num_batches += 1
            
            with torch.no_grad():
                blank_ratio = compute_blank_ratio(ctc_log_probs, frame_lengths, blank_idx)
                total_blank_ratio += blank_ratio
        
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
            optimizer.zero_grad()
        
        pbar.set_postfix({
            'loss': f'{loss.item() * accumulation_steps:.4f}',
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
    """Evaluate model."""
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
        
        ctc_log_probs, decoder_output, _, output_seq_len = model(
            frames, decoder_input_ids,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Adjust input lengths for compressed sequence (R3D backbone)
        input_seq_len = frame_lengths.max().item()
        if output_seq_len != input_seq_len:
            scale = output_seq_len / input_seq_len
            ctc_input_lengths = (frame_lengths.float() * scale).ceil().long().clamp(min=1).to(device)
        else:
            ctc_input_lengths = frame_lengths.to(device)
        
        try:
            ctc_loss = ctc_criterion(ctc_log_probs, gloss_ids, ctc_input_lengths, gloss_lengths.to(device))
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
        
        pred_tokens = model.greedy_decode(
            frames,
            src_key_padding_mask=src_key_padding_mask,
            max_len=50,
            sos_idx=sos_idx,
            eos_idx=eos_idx
        )
        
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
    print("=" * 70)
    print("IMPROVED TRAINING - Enhanced Augmentation + Gradient Accumulation")
    print("=" * 70)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Log file
    log_file = Path('training_improved_log.jsonl')
    if log_file.exists():
        log_file.unlink()
    
    # Get ENHANCED augmentation transforms
    print("\n[+] Using ENHANCED temporal augmentation:")
    print("    - Temporal crop-paste (segment swapping)")
    print("    - Temporal dropout (frame dropping)")
    print("    - Multi-scale temporal sampling")
    print("    - Speed perturbation + time masking")
    
    train_transforms = get_train_transforms(target_frames=args.max_frames, enhanced=True)
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
    
    # Model
    vocab_size = len(train_dataset.vocab)
    
    print(f"\nInitializing Hybrid CTC+Attention model with {args.backbone} backbone...")
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
        backbone_type=args.backbone,
        r3d_chunk_size=args.r3d_chunk_size
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
    
    # Optimizer with different LR for backbone
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'cnn' in name.lower() or 'backbone' in name.lower():
            backbone_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': other_params, 'lr': args.lr}
    ], weight_decay=args.weight_decay)
    
    # LR scheduler with warmup
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + torch.cos(torch.tensor((epoch - warmup_epochs) / (args.epochs - warmup_epochs) * 3.14159)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Setup mixed precision training (bf16 preferred for RTX 30/40/50 series)
    use_amp = args.use_amp and device.type == 'cuda'
    scaler = None
    amp_dtype = "disabled"
    
    if use_amp:
        if torch.cuda.is_bf16_supported():
            amp_dtype = "bfloat16"
            scaler = None  # bf16 doesn't need GradScaler
            print("[OK] Using BFloat16 mixed precision (optimal for RTX 30/40/50 series)")
        else:
            amp_dtype = "float16"
            scaler = torch.amp.GradScaler('cuda')
            print("[OK] Using Float16 mixed precision with GradScaler")
    
    # Training config summary
    effective_batch_size = args.batch_size * args.accumulation_steps
    print(f"\n{'='*70}")
    print("TRAINING CONFIGURATION (Optimized for 16GB VRAM):")
    print(f"  Batch size: {args.batch_size} x {args.accumulation_steps} = {effective_batch_size} effective")
    print(f"  Max frames: {args.max_frames}")
    print(f"  Mixed precision: {amp_dtype}")
    print(f"  CTC weight: {args.ctc_weight}, CE weight: {args.ce_weight}")
    print(f"  Enhanced temporal augmentation: ENABLED")
    print(f"  Label smoothing: 0.1")
    print(f"  Epochs: {args.epochs}")
    print(f"{'='*70}\n")
    
    # Training loop
    best_wer = float('inf')
    patience = 15
    no_improve_count = 0
    
    idx2gloss = train_dataset.idx2gloss
    blank_idx = train_dataset.vocab['<blank>']
    sos_idx = train_dataset.vocab['<sos>']
    eos_idx = train_dataset.vocab['<eos>']
    pad_idx = train_dataset.vocab['<pad>']
    
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, ctc_criterion, ce_criterion,
            device, epoch, blank_idx, args.ctc_weight, args.ce_weight,
            accumulation_steps=args.accumulation_steps,
            use_amp=use_amp,
            scaler=scaler
        )
        
        dev_metrics = evaluate(
            model, dev_loader, ctc_criterion, ce_criterion,
            device, idx2gloss, blank_idx, sos_idx, eos_idx, pad_idx
        )
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_metrics['loss']:.4f} (CTC: {train_metrics['ctc_loss']:.4f}, CE: {train_metrics['ce_loss']:.4f})")
        print(f"  Dev Loss: {dev_metrics['loss']:.4f} (CTC: {dev_metrics['ctc_loss']:.4f}, CE: {dev_metrics['ce_loss']:.4f})")
        print(f"  Dev WER: {dev_metrics['wer']*100:.2f}%")
        print(f"  Blank Ratio: {train_metrics['blank_ratio']*100:.1f}%")
        print(f"  LR: {current_lr:.6f}")
        
        log_metrics(log_file, epoch, {
            'train_loss': train_metrics['loss'],
            'train_ctc_loss': train_metrics['ctc_loss'],
            'train_ce_loss': train_metrics['ce_loss'],
            'dev_loss': dev_metrics['loss'],
            'dev_ctc_loss': dev_metrics['ctc_loss'],
            'dev_ce_loss': dev_metrics['ce_loss'],
            'dev_wer': dev_metrics['wer'],
            'blank_ratio': train_metrics['blank_ratio'],
            'learning_rate': current_lr,
            'effective_batch_size': effective_batch_size
        })
        
        if dev_metrics['wer'] < best_wer:
            best_wer = dev_metrics['wer']
            no_improve_count = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_wer': best_wer,
                'vocab': train_dataset.vocab,
                'config': {
                    'batch_size': args.batch_size,
                    'accumulation_steps': args.accumulation_steps,
                    'enhanced_augmentation': True
                }
            }, checkpoint_dir / 'best.pth')
            print(f"  [NEW BEST] WER: {best_wer*100:.2f}%")
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_wer': best_wer,
            'vocab': train_dataset.vocab
        }, checkpoint_dir / 'latest.pth')
        
        print("-" * 70)
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best WER: {best_wer*100:.2f}%")
    print(f"Model saved to: {checkpoint_dir / 'best.pth'}")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Improved Training with Enhancements')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data/phoenix2014-release')
    parser.add_argument('--max_frames', type=int, default=64,
                        help='Frames per video. 64 recommended for PHOENIX dataset.')
    
    # Training - optimized for 16GB VRAM with bf16 + 64 frames
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size per step (4 works with 64 frames + bf16 on 16GB)')
    parser.add_argument('--accumulation_steps', type=int, default=8,
                        help='Gradient accumulation (effective batch = 4 * 8 = 32)')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--ctc_weight', type=float, default=0.3)
    parser.add_argument('--ce_weight', type=float, default=0.7)
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision (bf16/fp16). Enabled by default.')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable mixed precision (use fp32)')
    
    # Model
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_encoder_layers', type=int, default=6)
    parser.add_argument('--num_decoder_layers', type=int, default=3)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--max_seq_len', type=int, default=500)
    
    # Backbone
    parser.add_argument('--backbone', type=str, default='r3d', 
                        choices=['r3d', 'resnet', 'simple'],
                        help='Video backbone: r3d (Kinetics-pretrained), resnet (ImageNet), simple')
    parser.add_argument('--r3d_chunk_size', type=int, default=8,
                        help='Chunk size for R3D-18 backbone (frames per clip)')
    
    # System
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/improved')
    
    args = parser.parse_args()
    
    # Handle --no_amp flag
    if args.no_amp:
        args.use_amp = False
    
    main(args)


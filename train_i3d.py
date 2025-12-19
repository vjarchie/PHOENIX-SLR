"""
Training script for I3D feature-based hybrid model.

Usage:
    # First extract features:
    python scripts/extract_i3d_features.py

    # Then train:
    python train_i3d.py --epochs 100 --batch_size 8 --device cuda
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.i3d_dataset import I3DFeatureDataset, collate_fn
from src.models.i3d_model import I3DHybridModel


def compute_wer(predictions: list, targets: list) -> float:
    """Compute Word Error Rate."""
    total_errors = 0
    total_words = 0
    
    for pred, target in zip(predictions, targets):
        pred_words = pred if isinstance(pred, list) else pred.split()
        target_words = target if isinstance(target, list) else target.split()
        
        # Levenshtein distance
        m, n = len(pred_words), len(target_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_words[i - 1] == target_words[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        
        total_errors += dp[m][n]
        total_words += n
    
    return total_errors / max(total_words, 1)


def ctc_decode(log_probs, lengths, idx2gloss, blank_idx=2):
    """Greedy CTC decoding."""
    predictions = []
    log_probs = log_probs.transpose(0, 1)  # (batch, seq, vocab)
    
    for i, (seq, length) in enumerate(zip(log_probs, lengths)):
        seq = seq[:length]
        indices = seq.argmax(dim=-1).tolist()
        
        # Remove blanks and consecutive duplicates
        decoded = []
        prev = None
        for idx in indices:
            if idx != blank_idx and idx != prev:
                if idx in idx2gloss:
                    gloss = idx2gloss[idx]
                    if gloss not in ['<pad>', '<unk>', '<blank>', '<sos>', '<eos>']:
                        decoded.append(gloss)
            prev = idx
        
        predictions.append(decoded)
    
    return predictions


def compute_blank_ratio(log_probs, lengths, blank_idx=2):
    """Compute ratio of blank predictions."""
    log_probs = log_probs.transpose(0, 1)  # (batch, seq, vocab)
    total_blanks = 0
    total_tokens = 0
    
    for seq, length in zip(log_probs, lengths):
        seq = seq[:length]
        predictions = seq.argmax(dim=-1)
        total_blanks += (predictions == blank_idx).sum().item()
        total_tokens += length
    
    return total_blanks / max(total_tokens, 1)


def log_metrics(log_file: Path, epoch: int, metrics: dict):
    """Log metrics to JSONL file for dashboard."""
    metrics['epoch'] = epoch
    metrics['timestamp'] = datetime.now().isoformat()
    
    # Convert tensors to floats
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            metrics[key] = value.item()
    
    with open(log_file, 'a') as f:
        f.write(json.dumps(metrics) + '\n')


def train_epoch(
    model, dataloader, optimizer, ctc_criterion, ce_criterion,
    device, epoch, blank_idx, ctc_weight, ce_weight
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_ctc_loss = 0
    total_ce_loss = 0
    total_blank_ratio = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        features = batch['features'].to(device)
        gloss_ids = batch['gloss_ids'].to(device)
        feature_lengths = batch['feature_lengths']
        gloss_lengths = batch['gloss_lengths']
        decoder_input_ids = batch['decoder_input_ids'].to(device)
        decoder_target_ids = batch['decoder_target_ids'].to(device)
        decoder_lengths = batch['decoder_lengths']
        
        optimizer.zero_grad()
        
        # Generate masks
        max_feat_len = features.size(1)
        max_dec_len = decoder_input_ids.size(1)
        
        src_key_padding_mask = torch.arange(max_feat_len, device=device).unsqueeze(0) >= feature_lengths.unsqueeze(1).to(device)
        tgt_key_padding_mask = torch.arange(max_dec_len, device=device).unsqueeze(0) >= decoder_lengths.unsqueeze(1).to(device)
        
        # Forward pass
        ctc_log_probs, decoder_output = model(
            features,
            decoder_input_ids,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # CTC loss
        ctc_loss = ctc_criterion(
            ctc_log_probs,
            gloss_ids,
            feature_lengths,
            gloss_lengths
        )
        
        # CE loss for decoder
        ce_loss = ce_criterion(
            decoder_output.view(-1, model.vocab_size),
            decoder_target_ids.view(-1)
        )
        
        # Combined loss
        loss = ctc_weight * ctc_loss + ce_weight * ce_loss
        
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_ctc_loss += ctc_loss.item()
            total_ce_loss += ce_loss.item()
            num_batches += 1
            
            with torch.no_grad():
                blank_ratio = compute_blank_ratio(ctc_log_probs, feature_lengths, blank_idx)
                total_blank_ratio += blank_ratio
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ctc': f'{ctc_loss.item():.4f}',
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
             sos_idx=3, eos_idx=4, pad_idx=0):
    """Evaluate model using decoder predictions (not CTC)."""
    model.eval()
    total_loss = 0
    total_ctc_loss = 0
    total_ce_loss = 0
    total_blank_ratio = 0
    num_batches = 0
    all_predictions = []
    all_targets = []
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        features = batch['features'].to(device)
        gloss_ids = batch['gloss_ids'].to(device)
        feature_lengths = batch['feature_lengths']
        gloss_lengths = batch['gloss_lengths']
        decoder_input_ids = batch['decoder_input_ids'].to(device)
        decoder_target_ids = batch['decoder_target_ids'].to(device)
        decoder_lengths = batch['decoder_lengths']
        
        max_feat_len = features.size(1)
        max_dec_len = decoder_input_ids.size(1)
        
        src_key_padding_mask = torch.arange(max_feat_len, device=device).unsqueeze(0) >= feature_lengths.unsqueeze(1).to(device)
        tgt_key_padding_mask = torch.arange(max_dec_len, device=device).unsqueeze(0) >= decoder_lengths.unsqueeze(1).to(device)
        
        ctc_log_probs, decoder_output = model(
            features,
            decoder_input_ids,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        ctc_loss = ctc_criterion(ctc_log_probs, gloss_ids, feature_lengths, gloss_lengths)
        ce_loss = ce_criterion(decoder_output.view(-1, model.vocab_size), decoder_target_ids.view(-1))
        
        if torch.isfinite(ctc_loss) and torch.isfinite(ce_loss):
            total_ctc_loss += ctc_loss.item()
            total_ce_loss += ce_loss.item()
            total_loss += (0.3 * ctc_loss.item() + 0.7 * ce_loss.item())
            blank_ratio = compute_blank_ratio(ctc_log_probs, feature_lengths, blank_idx)
            total_blank_ratio += blank_ratio
            num_batches += 1
        
        # Decode using DECODER (not CTC) - greedy decoding
        pred_tokens = model.greedy_decode(
            features, 
            src_key_padding_mask=src_key_padding_mask,
            max_len=50,
            sos_idx=sos_idx,
            eos_idx=eos_idx
        )
        
        # Convert predictions to glosses
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
    print("I3D Feature-based Hybrid Training")
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
    
    # Clear previous log for new training
    if log_file.exists():
        log_file.unlink()
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = I3DFeatureDataset(args.features_dir, split='train')
    dev_dataset = I3DFeatureDataset(args.features_dir, split='dev', vocab=train_dataset.vocab)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    print("\nInitializing model...")
    model = I3DHybridModel(
        input_dim=train_dataset.feature_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        vocab_size=len(train_dataset.vocab),
        max_seq_len=args.max_seq_len,
        pad_token_id=train_dataset.vocab['<pad>'],
        blank_token_id=train_dataset.vocab['<blank>']
    ).to(device)
    
    # Loss functions
    ctc_criterion = nn.CTCLoss(blank=train_dataset.vocab['<blank>'], zero_infinity=True)
    ce_criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab['<pad>'])
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # LR scheduler with warmup
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + torch.cos(torch.tensor((epoch - warmup_epochs) / (args.epochs - warmup_epochs) * 3.14159)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    best_wer = float('inf')
    idx2gloss = train_dataset.idx2gloss
    blank_idx = train_dataset.vocab['<blank>']
    sos_idx = train_dataset.vocab['<sos>']
    eos_idx = train_dataset.vocab['<eos>']
    pad_idx = train_dataset.vocab['<pad>']
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"CTC weight: {args.ctc_weight}, CE weight: {args.ce_weight}")
    print("=" * 60)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, ctc_criterion, ce_criterion,
            device, epoch, blank_idx, args.ctc_weight, args.ce_weight
        )
        
        # Evaluate using decoder predictions
        dev_metrics = evaluate(
            model, dev_loader, ctc_criterion, ce_criterion,
            device, idx2gloss, blank_idx,
            sos_idx=sos_idx, eos_idx=eos_idx, pad_idx=pad_idx
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_wer': best_wer,
                'vocab': train_dataset.vocab
            }, checkpoint_dir / 'best.pth')
            print(f"  [NEW BEST] WER: {best_wer*100:.2f}%")
        
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
    parser = argparse.ArgumentParser(description='Train I3D feature-based hybrid model')
    
    # Data
    parser.add_argument('--features_dir', type=str, default='data/i3d_features',
                        help='Directory with extracted I3D features')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--ctc_weight', type=float, default=0.3)
    parser.add_argument('--ce_weight', type=float, default=0.7)
    
    # Model
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_encoder_layers', type=int, default=6)
    parser.add_argument('--num_decoder_layers', type=int, default=3)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max_seq_len', type=int, default=500)
    
    # System
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=0)  # 0 for pre-extracted features
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/i3d')
    
    args = parser.parse_args()
    main(args)



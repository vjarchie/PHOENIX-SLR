# -*- coding: utf-8 -*-
"""
End-to-End Hybrid Training with Visual Alignment Constraint (VAC).

VAC adds alignment supervision to prevent overfitting and improve generalization.
Reference: Min et al., "Visual Alignment Constraint for CSLR" (ICCV 2021)

Components:
- SeqCTC: CTC loss on encoder output (after Transformer)
- ConvCTC: CTC loss on visual features (after CNN, before Transformer)
- Distillation: KL divergence to align the two predictions
- CE: Cross-entropy loss for attention decoder
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
from losses.vac_loss import VACHybridLoss


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
    predictions = log_probs.argmax(dim=-1)  # (T, B)
    total_blanks = 0
    total_frames = 0
    
    for b in range(predictions.size(1)):
        seq_len = lengths[b].item() if isinstance(lengths[b], torch.Tensor) else lengths[b]
        seq = predictions[:seq_len, b]
        total_blanks += (seq == blank_idx).sum().item()
        total_frames += seq_len
    
    return total_blanks / max(total_frames, 1)


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


def train_epoch(model, dataloader, optimizer, vac_loss_fn, device, epoch, blank_idx, 
                max_grad_norm=5.0, accumulation_steps=1):
    """Train for one epoch with VAC loss."""
    model.train()
    total_loss = 0
    total_seq_ctc = 0
    total_conv_ctc = 0
    total_dist = 0
    total_ce = 0
    total_blank_ratio = 0
    num_batches = 0
    
    optimizer.zero_grad()
    
    effective_batch = f"(eff. batch={dataloader.batch_size * accumulation_steps})" if accumulation_steps > 1 else ""
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} {effective_batch}")
    
    for batch_idx, batch in enumerate(pbar):
        frames = batch['frames'].to(device)
        gloss_ids = batch['ctc_targets'].to(device)
        frame_lengths = batch['frame_lengths'].to(device)
        gloss_lengths = batch['gloss_lengths'].to(device)
        decoder_input_ids = batch['decoder_input'].to(device)
        decoder_target_ids = batch['decoder_target'].to(device)
        decoder_lengths = gloss_lengths + 1
        
        batch_size = frames.size(0)
        max_frame_len = frames.size(1)
        max_dec_len = decoder_input_ids.size(1)
        
        # Create masks
        src_key_padding_mask = torch.arange(max_frame_len, device=device).unsqueeze(0) >= frame_lengths.unsqueeze(1)
        tgt_key_padding_mask = torch.arange(max_dec_len, device=device).unsqueeze(0) >= decoder_lengths.unsqueeze(1)
        
        try:
            # Forward pass WITH VAC (returns 5 values)
            seq_ctc_log_probs, decoder_output, encoder_output, output_seq_len, conv_ctc_logits = model(
                frames,
                decoder_input_ids,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                use_vac=True
            )
        except RuntimeError as e:
            print(f"Forward pass error: {e}")
            continue
        
        # Adjust input lengths for compressed sequence (if using R3D)
        ctc_input_lengths = torch.full((batch_size,), output_seq_len, device=device, dtype=torch.long)
        for i in range(batch_size):
            # Scale down frame_lengths proportionally
            ratio = output_seq_len / max_frame_len
            ctc_input_lengths[i] = max(1, int(frame_lengths[i].item() * ratio))
        
        # Get seq_ctc_logits (before log_softmax) for VAC loss
        # Note: seq_ctc_log_probs is already log_softmax'd, we need raw logits
        # Reconstruct from encoder output
        seq_ctc_logits = model.ctc_proj(encoder_output).transpose(0, 1)  # (T, B, V)
        
        try:
            # Compute VAC loss
            loss_dict = vac_loss_fn(
                seq_logits=seq_ctc_logits,
                conv_logits=conv_ctc_logits,
                decoder_logits=decoder_output,
                ctc_targets=gloss_ids,
                decoder_targets=decoder_target_ids,
                input_lengths=ctc_input_lengths,
                target_lengths=gloss_lengths
            )
            loss = loss_dict['total'] / accumulation_steps
            
        except RuntimeError as e:
            print(f"Loss computation error: {e}")
            continue
        
        if torch.isfinite(loss):
            loss.backward()
            
            total_loss += loss.item() * accumulation_steps
            total_seq_ctc += loss_dict['seq_ctc']
            total_conv_ctc += loss_dict['conv_ctc']
            total_dist += loss_dict['distillation']
            total_ce += loss_dict.get('ce', 0)
            num_batches += 1
            
            with torch.no_grad():
                blank_ratio = compute_blank_ratio(seq_ctc_log_probs, ctc_input_lengths, blank_idx)
                total_blank_ratio += blank_ratio
        
        # Update weights every accumulation_steps batches
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        
        pbar.set_postfix({
            'loss': f'{loss.item() * accumulation_steps:.4f}',
            'seq': f'{loss_dict["seq_ctc"]:.3f}',
            'conv': f'{loss_dict["conv_ctc"]:.3f}',
            'dist': f'{loss_dict["distillation"]:.3f}',
            'blank': f'{blank_ratio*100:.1f}%' if torch.isfinite(loss) else 'N/A'
        })
    
    return {
        'loss': total_loss / max(num_batches, 1),
        'seq_ctc': total_seq_ctc / max(num_batches, 1),
        'conv_ctc': total_conv_ctc / max(num_batches, 1),
        'distillation': total_dist / max(num_batches, 1),
        'ce': total_ce / max(num_batches, 1),
        'blank_ratio': total_blank_ratio / max(num_batches, 1)
    }


@torch.no_grad()
def evaluate(model, dataloader, vac_loss_fn, device, idx2gloss, blank_idx, sos_idx, eos_idx, pad_idx):
    """Evaluate model using decoder predictions."""
    model.eval()
    total_loss = 0
    total_blank_ratio = 0
    num_batches = 0
    all_predictions = []
    all_targets = []
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        frames = batch['frames'].to(device)
        gloss_ids = batch['ctc_targets'].to(device)
        frame_lengths = batch['frame_lengths'].to(device)
        gloss_lengths = batch['gloss_lengths'].to(device)
        decoder_input_ids = batch['decoder_input'].to(device)
        decoder_target_ids = batch['decoder_target'].to(device)
        decoder_lengths = gloss_lengths + 1
        
        batch_size = frames.size(0)
        max_frame_len = frames.size(1)
        max_dec_len = decoder_input_ids.size(1)
        
        src_key_padding_mask = torch.arange(max_frame_len, device=device).unsqueeze(0) >= frame_lengths.unsqueeze(1)
        tgt_key_padding_mask = torch.arange(max_dec_len, device=device).unsqueeze(0) >= decoder_lengths.unsqueeze(1)
        
        # Forward pass with VAC
        seq_ctc_log_probs, decoder_output, encoder_output, output_seq_len, conv_ctc_logits = model(
            frames,
            decoder_input_ids,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            use_vac=True
        )
        
        # Adjust input lengths
        ctc_input_lengths = torch.full((batch_size,), output_seq_len, device=device, dtype=torch.long)
        for i in range(batch_size):
            ratio = output_seq_len / max_frame_len
            ctc_input_lengths[i] = max(1, int(frame_lengths[i].item() * ratio))
        
        seq_ctc_logits = model.ctc_proj(encoder_output).transpose(0, 1)
        
        try:
            loss_dict = vac_loss_fn(
                seq_logits=seq_ctc_logits,
                conv_logits=conv_ctc_logits,
                decoder_logits=decoder_output,
                ctc_targets=gloss_ids,
                decoder_targets=decoder_target_ids,
                input_lengths=ctc_input_lengths,
                target_lengths=gloss_lengths
            )
            
            if torch.isfinite(loss_dict['total']):
                total_loss += loss_dict['total'].item() if isinstance(loss_dict['total'], torch.Tensor) else loss_dict['total']
                blank_ratio = compute_blank_ratio(seq_ctc_log_probs, ctc_input_lengths, blank_idx)
                total_blank_ratio += blank_ratio
                num_batches += 1
        except:
            pass
        
        # Decode using attention decoder
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
        'wer': wer,
        'blank_ratio': total_blank_ratio / max(num_batches, 1)
    }


def main(args):
    print("=" * 70)
    print("End-to-End Hybrid Training with Visual Alignment Constraint (VAC)")
    print("=" * 70)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = Path('training_log_vac.jsonl')
    if log_file.exists():
        log_file.unlink()
    
    # Transforms
    train_transforms = get_train_transforms(target_frames=args.max_frames)
    val_transforms = get_val_transforms(target_frames=args.max_frames)
    
    # Datasets
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
    
    vocab_size = len(train_dataset.vocab)
    blank_idx = train_dataset.vocab['<blank>']
    pad_idx = train_dataset.vocab['<pad>']
    sos_idx = train_dataset.vocab['<sos>']
    eos_idx = train_dataset.vocab['<eos>']
    
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
        ctc_weight=args.seq_ctc_weight,  # This is for model's internal compute_loss, not used with VAC
        use_resnet=True
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # VAC Loss
    print(f"\nInitializing VAC loss...")
    vac_loss_fn = VACHybridLoss(
        seq_ctc_weight=args.seq_ctc_weight,
        conv_ctc_weight=args.conv_ctc_weight,
        dist_weight=args.dist_weight,
        ce_weight=args.ce_weight,
        temperature=args.temperature,
        blank_idx=blank_idx,
        pad_idx=pad_idx,
        label_smoothing=0.1
    )
    
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
    
    # LR scheduler
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
    
    effective_batch_size = args.batch_size * args.accumulation_steps
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size} x {args.accumulation_steps} = {effective_batch_size} effective")
    print(f"\nVAC Loss Weights:")
    print(f"  SeqCTC: {args.seq_ctc_weight}")
    print(f"  ConvCTC: {args.conv_ctc_weight}")
    print(f"  Distillation: {args.dist_weight}")
    print(f"  CE: {args.ce_weight}")
    print(f"  Temperature: {args.temperature}")
    print("=" * 70)
    
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, vac_loss_fn, device, epoch, blank_idx,
            accumulation_steps=args.accumulation_steps
        )
        
        dev_metrics = evaluate(
            model, dev_loader, vac_loss_fn, device, idx2gloss, blank_idx, sos_idx, eos_idx, pad_idx
        )
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"    SeqCTC: {train_metrics['seq_ctc']:.4f}, ConvCTC: {train_metrics['conv_ctc']:.4f}")
        print(f"    Distill: {train_metrics['distillation']:.4f}, CE: {train_metrics['ce']:.4f}")
        print(f"  Dev Loss: {dev_metrics['loss']:.4f}")
        print(f"  Dev WER: {dev_metrics['wer']*100:.2f}%")
        print(f"  Blank Ratio: {train_metrics['blank_ratio']*100:.1f}%")
        print(f"  LR: {current_lr:.6f}")
        
        log_metrics(log_file, epoch, {
            'train_loss': train_metrics['loss'],
            'train_seq_ctc': train_metrics['seq_ctc'],
            'train_conv_ctc': train_metrics['conv_ctc'],
            'train_distillation': train_metrics['distillation'],
            'train_ce': train_metrics['ce'],
            'dev_loss': dev_metrics['loss'],
            'dev_wer': dev_metrics['wer'],
            'blank_ratio': train_metrics['blank_ratio'],
            'learning_rate': current_lr
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
                'vac_config': {
                    'seq_ctc_weight': args.seq_ctc_weight,
                    'conv_ctc_weight': args.conv_ctc_weight,
                    'dist_weight': args.dist_weight,
                    'ce_weight': args.ce_weight,
                    'temperature': args.temperature
                }
            }, checkpoint_dir / 'best_vac.pth')
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
        }, checkpoint_dir / 'latest_vac.pth')
        
        print("-" * 70)
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Best WER: {best_wer*100:.2f}%")
    print(f"Model saved to: {checkpoint_dir / 'best_vac.pth'}")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='End-to-End Training with VAC Loss')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='data/phoenix2014-release')
    parser.add_argument('--max_frames', type=int, default=64)
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    
    # VAC Loss weights (from VAC paper)
    parser.add_argument('--seq_ctc_weight', type=float, default=1.0,
                        help='Weight for Sequence CTC loss')
    parser.add_argument('--conv_ctc_weight', type=float, default=1.0,
                        help='Weight for Visual/Conv CTC loss')
    parser.add_argument('--dist_weight', type=float, default=25.0,
                        help='Weight for distillation loss')
    parser.add_argument('--ce_weight', type=float, default=0.7,
                        help='Weight for decoder cross-entropy loss')
    parser.add_argument('--temperature', type=float, default=8.0,
                        help='Temperature for distillation')
    
    # Model
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_encoder_layers', type=int, default=6)
    parser.add_argument('--num_decoder_layers', type=int, default=3)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--max_seq_len', type=int, default=500)
    
    # System
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/vac')
    
    args = parser.parse_args()
    main(args)

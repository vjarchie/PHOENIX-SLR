# -*- coding: utf-8 -*-
"""
Training Script for Text-to-Gloss Model

Trains a Transformer-based model to translate German text to DGS gloss sequences.
This is the first step in the Speech-to-Sign pipeline.
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List


# Import model
from src.speech_to_sign.text_to_gloss import TextToGlossModel, TextTokenizer


class Text2GlossDataset(Dataset):
    """Dataset for text-to-gloss training."""
    
    def __init__(
        self,
        pairs_file: str,
        text_vocab: Dict[str, int],
        gloss_vocab: Dict[str, int],
        max_text_len: int = 100,
        max_gloss_len: int = 50
    ):
        with open(pairs_file, 'r', encoding='utf-8') as f:
            self.pairs = json.load(f)
        
        self.text_vocab = text_vocab
        self.gloss_vocab = gloss_vocab
        self.max_text_len = max_text_len
        self.max_gloss_len = max_gloss_len
        
        # Special tokens
        self.text_pad = text_vocab.get('<pad>', 0)
        self.text_sos = text_vocab.get('<sos>', 2)
        self.text_eos = text_vocab.get('<eos>', 3)
        self.text_unk = text_vocab.get('<unk>', 1)
        
        self.gloss_pad = gloss_vocab.get('<pad>', 0)
        self.gloss_sos = gloss_vocab.get('<sos>', 3)
        self.gloss_eos = gloss_vocab.get('<eos>', 4)
        self.gloss_unk = gloss_vocab.get('<unk>', 1)
        
        print(f"Text2GlossDataset: {len(self.pairs)} pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Encode text
        text_words = pair['text'].lower().split()
        text_ids = [self.text_sos]
        for word in text_words[:self.max_text_len - 2]:
            text_ids.append(self.text_vocab.get(word, self.text_unk))
        text_ids.append(self.text_eos)
        
        # Encode glosses
        gloss_input = [self.gloss_sos]  # Decoder input starts with <sos>
        gloss_target = []  # Target ends with <eos>
        
        for gloss in pair['glosses'][:self.max_gloss_len - 1]:
            gid = self.gloss_vocab.get(gloss, self.gloss_unk)
            gloss_input.append(gid)
            gloss_target.append(gid)
        
        gloss_target.append(self.gloss_eos)
        
        return {
            'text_ids': text_ids,
            'gloss_input': gloss_input,
            'gloss_target': gloss_target,
            'text': pair['text'],
            'glosses': pair['glosses']
        }


def collate_fn(batch):
    """Custom collate function for padding."""
    # Get max lengths
    max_text_len = max(len(b['text_ids']) for b in batch)
    max_gloss_len = max(len(b['gloss_input']) for b in batch)
    
    text_ids = []
    text_padding_mask = []
    gloss_input = []
    gloss_target = []
    gloss_padding_mask = []
    
    for b in batch:
        # Pad text
        text = b['text_ids']
        text_pad_len = max_text_len - len(text)
        text_ids.append(text + [0] * text_pad_len)
        text_padding_mask.append([False] * len(text) + [True] * text_pad_len)
        
        # Pad gloss input
        gi = b['gloss_input']
        gi_pad_len = max_gloss_len - len(gi)
        gloss_input.append(gi + [0] * gi_pad_len)
        gloss_padding_mask.append([False] * len(gi) + [True] * gi_pad_len)
        
        # Pad gloss target
        gt = b['gloss_target']
        gt_pad_len = max_gloss_len - len(gt)
        gloss_target.append(gt + [0] * gt_pad_len)
    
    return {
        'text_ids': torch.tensor(text_ids, dtype=torch.long),
        'text_padding_mask': torch.tensor(text_padding_mask, dtype=torch.bool),
        'gloss_input': torch.tensor(gloss_input, dtype=torch.long),
        'gloss_target': torch.tensor(gloss_target, dtype=torch.long),
        'gloss_padding_mask': torch.tensor(gloss_padding_mask, dtype=torch.bool)
    }


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        text_ids = batch['text_ids'].to(device)
        text_mask = batch['text_padding_mask'].to(device)
        gloss_input = batch['gloss_input'].to(device)
        gloss_target = batch['gloss_target'].to(device)
        gloss_mask = batch['gloss_padding_mask'].to(device)
        
        optimizer.zero_grad()
        
        # Forward
        logits = model(text_ids, gloss_input, text_mask, gloss_mask)
        
        # Loss (ignore padding)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            gloss_target.reshape(-1),
            ignore_index=0  # Ignore padding
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def evaluate(model, dataloader, device, idx2gloss):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    correct_glosses = 0
    total_glosses = 0
    
    with torch.no_grad():
        for batch in dataloader:
            text_ids = batch['text_ids'].to(device)
            text_mask = batch['text_padding_mask'].to(device)
            gloss_input = batch['gloss_input'].to(device)
            gloss_target = batch['gloss_target'].to(device)
            gloss_mask = batch['gloss_padding_mask'].to(device)
            
            # Forward
            logits = model(text_ids, gloss_input, text_mask, gloss_mask)
            
            # Loss
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                gloss_target.reshape(-1),
                ignore_index=0
            )
            
            total_loss += loss.item()
            num_batches += 1
            
            # Accuracy (for non-padding positions)
            predictions = torch.argmax(logits, dim=-1)
            mask = ~gloss_mask
            correct_glosses += ((predictions == gloss_target) & mask).sum().item()
            total_glosses += mask.sum().item()
    
    accuracy = correct_glosses / total_glosses if total_glosses > 0 else 0
    
    return total_loss / num_batches, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train Text-to-Gloss model')
    
    # Data
    parser.add_argument('--data-dir', type=str, default='data/text2gloss',
                       help='Directory with training data')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/text2gloss',
                       help='Directory to save checkpoints')
    
    # Model
    parser.add_argument('--d-model', type=int, default=256,
                       help='Model dimension')
    parser.add_argument('--nhead', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--num-encoder-layers', type=int, default=3,
                       help='Number of encoder layers')
    parser.add_argument('--num-decoder-layers', type=int, default=3,
                       help='Number of decoder layers')
    parser.add_argument('--dim-feedforward', type=int, default=1024,
                       help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load vocabularies
    data_dir = Path(args.data_dir)
    
    with open(data_dir / 'text_vocab.json', 'r', encoding='utf-8') as f:
        text_vocab = json.load(f)
    
    with open(data_dir / 'gloss_vocab.json', 'r', encoding='utf-8') as f:
        gloss_vocab = json.load(f)
    
    idx2gloss = {v: k for k, v in gloss_vocab.items()}
    
    print(f"Text vocabulary: {len(text_vocab)} words")
    print(f"Gloss vocabulary: {len(gloss_vocab)} glosses")
    
    # Create datasets
    train_dataset = Text2GlossDataset(
        data_dir / 'train_pairs.json',
        text_vocab,
        gloss_vocab
    )
    
    val_dataset = Text2GlossDataset(
        data_dir / 'val_pairs.json',
        text_vocab,
        gloss_vocab
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    # Create model
    model = TextToGlossModel(
        text_vocab_size=len(text_vocab),
        gloss_vocab_size=len(gloss_vocab),
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    ).to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = {
        'text_vocab_size': len(text_vocab),
        'gloss_vocab_size': len(gloss_vocab),
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_encoder_layers': args.num_encoder_layers,
        'num_decoder_layers': args.num_decoder_layers,
        'dim_feedforward': args.dim_feedforward,
        'dropout': args.dropout
    }
    
    with open(checkpoint_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Training loop
    best_val_loss = float('inf')
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, device, idx2gloss)
        
        # Update scheduler
        scheduler.step()
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, checkpoint_dir / 'best.pth')
            print(f"  Saved best model (loss: {val_loss:.4f})")
        
        # Save latest
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc
        }, checkpoint_dir / 'latest.pth')
    
    print("\n" + "="*60)
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("="*60)


if __name__ == '__main__':
    main()


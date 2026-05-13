import torch
import gc
from torch.utils.data import DataLoader
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from models.transformer import HybridCTCAttentionModel
from models.losses import VACLoss
import torch.nn as nn

def test_mem_leak():
    device = torch.device('cuda')
    model = HybridCTCAttentionModel(
        vocab_size=1200,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=3,
        dim_feedforward=2048,
        dropout=0.2,
        max_seq_len=500,
        ctc_weight=0.3,
        use_resnet=True,
        backbone_type='resnet50'
    ).to(device)
    
    ctc_criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    ce_criterion = nn.CrossEntropyLoss(ignore_index=0)
    vac_criterion = VACLoss(blank_idx=0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    print("Initial VRAM:", torch.cuda.memory_allocated() / 1e6, "MB")
    
    # Fake data
    B, T, C, H, W = 4, 64, 3, 224, 224
    frames = torch.rand(B, T, C, H, W, device=device)
    decoder_input = torch.randint(0, 1200, (B, 15), device=device)
    decoder_target = torch.randint(0, 1200, (B, 15), device=device)
    gloss_ids = torch.randint(1, 1200, (B * 10,), device=device)  # 1D for CTC
    frame_lengths = torch.full((B,), T, dtype=torch.long, device=device)
    gloss_lengths = torch.full((B,), 10, dtype=torch.long, device=device)
    
    src_mask = torch.zeros((B, T), dtype=torch.bool, device=device)
    tgt_mask = torch.zeros((B, 15), dtype=torch.bool, device=device)
    
    for epoch in range(15):
        for batch in range(10): # dummy loop
            optimizer.zero_grad()
            
            ctc_log_probs, decoder_output, _, _, conv_ctc_logits = model(
                frames, decoder_input,
                src_key_padding_mask=src_mask,
                tgt_key_padding_mask=tgt_mask,
                use_vac=True
            )
            
            ctc_loss = ctc_criterion(ctc_log_probs, gloss_ids, frame_lengths, gloss_lengths)
            vac_loss = vac_criterion(conv_ctc_logits, gloss_ids, frame_lengths, gloss_lengths)
            ce_loss = ce_criterion(decoder_output.view(-1, 1200), decoder_target.view(-1))
            
            loss = ctc_loss + ce_loss + vac_loss
            loss.backward()
            optimizer.step()
            
            # evaluate simulate
            with torch.no_grad():
                model.eval()
                _ = model.greedy_decode(frames, src_key_padding_mask=src_mask, max_len=10)
                model.train()
                
        print(f"Epoch {epoch} VRAM:", torch.cuda.memory_allocated() / 1e6, "MB")

if __name__ == '__main__':
    test_mem_leak()

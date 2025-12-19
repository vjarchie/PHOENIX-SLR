# Agent Prompt for PHOENIX-SLR Project

Copy and paste the following prompt when starting a new Cursor session:

---

## PROMPT START

I'm working on a **Semester III academic project** for **Real-Time Bidirectional Continuous Sign Language Interpretation**.

### Project Context

This is a **continuation** from a previous ASL project at `D:\Semester3`. That project encountered **CTC collapse issues** (model outputting mostly blanks). After extensive troubleshooting, I decided to switch to the **RWTH-PHOENIX dataset** which has published benchmarks.

**Read these files first for full context:**
1. `README.md` - Project overview, architecture, and instructions
2. `PREVIOUS_PROJECT_CONTEXT.md` - All learnings from the failed ASL approach

### Key Technical Decisions Already Made

| Decision | Choice | Reason |
|----------|--------|--------|
| Dataset | RWTH-PHOENIX (German SL) | Has published benchmarks (26.8% WER) |
| Architecture | **Transformer + CTC** | Better than TCN for this task |
| Features | Video frames (210×260) | Dataset provides pre-extracted frames |
| Target | <30% Word Error Rate | Achievable with Transformer |

### Lessons from Previous Project (D:\Semester3)

**Critical issues to avoid:**
1. **CTC Collapse**: Monitor blank ratio during training - if >80%, model is collapsing
2. **Coordinate Mismatch**: Verify all data uses consistent coordinate system
3. **Large Vocabulary**: Start with full PHOENIX vocab (~1200 glosses), don't reduce
4. **Loss vs WER**: Track WER during training, not just loss

**What worked:**
- AdamW optimizer with warmup
- Gradient clipping (max_norm=5.0)
- Data augmentation
- OneCycleLR scheduler

### Current Status

✅ Project structure created at `D:\PHOENIX-SLR`
✅ Transformer model implemented (`src/models/transformer.py`)
✅ Dataset loader implemented (`src/data/phoenix_dataset.py`)
✅ Training script ready (`train.py`)
✅ Evaluation script ready (`evaluate.py`)

❌ Dataset not downloaded yet (53 GB)
❌ Model not trained yet

### Your Task: Execute Next Steps

Please proceed with:

1. **Download the PHOENIX dataset** (~53 GB)
   ```powershell
   python scripts/download_phoenix.py
   ```
   
2. **Verify dataset structure** after download

3. **Start training** with:
   ```powershell
   python train.py --epochs 100 --batch_size 4 --device cuda
   ```

4. **Monitor for CTC collapse** - check blank ratio in predictions

5. **Evaluate on dev set** when training completes:
   ```powershell
   python evaluate.py --checkpoint checkpoints/transformer/best.pth --split dev
   ```

### Important Files Reference

| File | Purpose |
|------|---------|
| `train.py` | Main training script |
| `evaluate.py` | WER evaluation |
| `src/models/transformer.py` | Transformer + CTC model |
| `src/data/phoenix_dataset.py` | PHOENIX dataset loader |
| `configs/transformer_ctc.yaml` | Training config |

### Dataset Info

- **URL**: https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/
- **Download**: https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014.v3.tar.gz
- **Size**: ~53 GB
- **Vocabulary**: ~1,200 glosses
- **Published SOTA**: 26.8% WER

### Success Criteria

| Metric | Target | Notes |
|--------|--------|-------|
| WER (dev) | <35% | Good progress |
| WER (test) | <30% | Competitive with published |
| Blank Ratio | <30% | Model not collapsing |
| Training Time | ~6-12h | On RTX 5070 Ti |

Please start by reading the context files, then proceed with the dataset download.

## PROMPT END

---

*This prompt provides complete context for any AI agent to continue the project.*




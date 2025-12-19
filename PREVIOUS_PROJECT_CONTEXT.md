# Previous Project Context & Learnings

## Original Academic Requirements

**Title**: An Integrated Temporal Convolutional Network (TCN) Architecture for Real-Time Bidirectional Continuous Sign Language Interpretation

**Semester**: III

**Objective**: Build a bidirectional sign language interpretation system:
1. **Sign-to-Speech**: Camera captures sign language → Model recognizes → Text/Speech output
2. **Speech-to-Sign**: Speech input → Text → Sign video synthesis

---

## Previous Project Summary (D:\Semester3)

### What Was Built

| Component | Status | Details |
|-----------|--------|---------|
| **TCN Model** | ✅ Built | 4-layer TCN with CTC loss, ~20M parameters |
| **Real-time Pipeline** | ✅ Working | MediaPipe → Adapter → Model → CTC Decode → TTS |
| **Two-Phase Training** | ✅ Completed | Phase 1 (isolated words) + Phase 2 (sentences) |
| **Data Augmentation** | ✅ Implemented | Temporal, spatial, and occlusion augmentations |
| **Training Dashboard** | ✅ Built | Streamlit-based live monitoring |

### Datasets Used

| Dataset | Type | Samples | Purpose |
|---------|------|---------|---------|
| **WLASL** | Isolated words | 11,980 | Phase 1 pretraining |
| **MS-ASL** | Isolated words | 25,513 | Phase 1 pretraining |
| **How2Sign** | Continuous sentences | 7,841 | Phase 2 finetuning |
| **OpenASL** | Continuous sentences | 33,117 | Phase 2 finetuning |

### Final Model Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **Vocabulary** | 4,000 words | After two-phase training |
| **CTC Loss** | ~3.7 | Still high, CTC collapse issue |
| **Predictions** | Mostly blanks | Model not converging well |
| **WER** | Not measured | Too many empty predictions |

---

## Key Challenges Encountered

### 1. CTC Collapse Problem (Critical)

**Issue**: Model outputs 95%+ blank tokens, predicting very few actual words.

**Cause Analysis**:
- Vocabulary too large for dataset size
- CTC loss finds local minimum where outputting blanks minimizes loss
- Insufficient training data diversity

**Attempted Solutions**:
- ✅ Reduced vocabulary (6068 → 1000 → 2000 → 4000)
- ✅ Longer training (200 → 500 → 700 epochs)
- ✅ Two-phase training approach
- ✅ Data augmentation
- ❌ None fully resolved the issue

**Lesson**: CTC models need careful vocabulary sizing and sufficient data.

### 2. Feature Format Mismatch

**Issue**: Training data (How2Sign) used OpenPose format (274 features), but MediaPipe extracts different format (1629 features).

**Solution**: Created `MediaPipeToOpenPoseAdapter`:
```
MediaPipe (1629)  →  OpenPose (274)
├── Pose: 33×3     →  Body: 25×2
├── L.Hand: 21×3   →  L.Hand: 21×2  
├── R.Hand: 21×3   →  R.Hand: 21×2
└── Face: 468×3    →  Face: 70×2
```

### 3. Coordinate Scale Mismatch

**Issue**: MediaPipe outputs normalized (0-1), but training data was in pixel coordinates (0-1280).

**Solution**: Added scaling in adapter:
```python
x_scaled = x_normalized * frame_width  # 1280
y_scaled = y_normalized * frame_height  # 720
```

### 4. Protobuf/TensorFlow Conflicts

**Issue**: MediaPipe requires protobuf<5, TensorFlow requires protobuf>=5.28

**Solution**: Uninstalled TensorFlow (not needed for PyTorch inference)

### 5. Windows-Specific Issues

| Issue | Solution |
|-------|----------|
| Unicode emojis in console | Removed emojis from print statements |
| Long path limitation | Enabled LongPathsEnabled in registry |
| GPU not detected | Reinstalled PyTorch with CUDA support |

---

## Architecture Details (Previous Project)

### TCN Model Structure

```
Input: Pose keypoints (T × 274)
↓
BatchNorm1d (274)
↓
TCN Block 1: Conv1d(274→256), BN, ReLU, Dropout(0.3)
TCN Block 2: Conv1d(256→256), BN, ReLU, Dropout(0.3)
TCN Block 3: Conv1d(256→512), BN, ReLU, Dropout(0.3)
TCN Block 4: Conv1d(512→512), BN, ReLU, Dropout(0.3)
↓
Linear(512 → vocab_size)
↓
Log Softmax
↓
CTC Loss / Greedy Decoding
```

**Parameters**: ~20 million

### Training Configuration

| Parameter | Phase 1 | Phase 2 |
|-----------|---------|---------|
| Epochs | 500 | 500 |
| Batch Size | 16 | 16 |
| Learning Rate | 0.001 | 0.0001 |
| Optimizer | AdamW | AdamW |
| Scheduler | OneCycleLR | OneCycleLR |
| Augmentation | None | Medium |

---

## Why Switching to RWTH-PHOENIX?

### Reasons for Change

| Reason | Explanation |
|--------|-------------|
| **Benchmark Available** | PHOENIX has published WER benchmarks (26.8% SOTA) |
| **Reproducible Results** | Can compare against published papers |
| **Better Data Quality** | Professional studio recording, consistent lighting |
| **Gloss Annotations** | Clean, manually verified annotations |
| **Research Standard** | Most CSLR papers use this dataset |

### Comparison

| Aspect | Previous (How2Sign/ASL) | New (PHOENIX/DGS) |
|--------|-------------------------|-------------------|
| Language | American Sign Language | German Sign Language |
| Annotations | Sentence text | Gloss sequences |
| Quality | YouTube videos | Professional recording |
| Benchmark | None established | 26.8% WER (SOTA) |
| Size | ~40K samples | ~9K samples |
| Features | Keypoints (274) | Video frames (210×260) |

### Expected Benefits

1. **Clear benchmark target**: Aim for <30% WER
2. **Validated architecture**: Papers show which models work
3. **Cleaner experiments**: Professional data quality
4. **Academic credibility**: Standard dataset for publications

---

## Lessons for New Project

### Do's ✅

1. **Start with proven architecture**: Use Transformer (SOTA on PHOENIX)
2. **Smaller vocabulary first**: Begin with full PHOENIX vocab (~1200 glosses)
3. **Monitor WER during training**: Not just loss
4. **Use published preprocessing**: Follow paper methodologies
5. **Implement early stopping**: Based on validation WER

### Don'ts ❌

1. **Don't use TCN alone**: Limited for this task (40-50% WER typical)
2. **Don't ignore blank ratio**: Early sign of CTC collapse
3. **Don't train too long without validation**: Loss can mislead
4. **Don't mix coordinate systems**: Verify all data is normalized consistently

---

## Files to Reference from Previous Project

If you need to reference code patterns:

| File | Purpose |
|------|---------|
| `D:\Semester3\src\models\tcn.py` | TCN architecture reference |
| `D:\Semester3\src\data\augmentation.py` | Data augmentation patterns |
| `D:\Semester3\train_phase2.py` | Training loop with CTC loss |
| `D:\Semester3\src\utils\text_processing.py` | Vocabulary management |
| `D:\Semester3\src\pipelines\mediapipe_to_openpose.py` | Feature adapter pattern |
| `D:\Semester3\PROJECT_SUMMARY.md` | Detailed challenge documentation |
| `D:\Semester3\TECHNICAL_DOCUMENTATION.md` | Architecture & training details |

---

## Technical Documentation Location

Full technical documentation for the previous project:
- `D:\Semester3\TECHNICAL_DOCUMENTATION.md` (657 lines)
- `D:\Semester3\TWO_PHASE_TRAINING_PLAN.md` (detailed training strategy)

---

*This document preserves context from the D:\Semester3 project for reference in the new PHOENIX-SLR project.*




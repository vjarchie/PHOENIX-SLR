# Source Files Walkthrough - PHOENIX-SLR Project

> **Guide to understanding the codebase structure and key components**

---

## 📁 Project Structure Overview

```
PHOENIX-SLR/
├── src/                    # Main source code
│   ├── models/            # Neural network architectures
│   ├── data/              # Dataset loaders and augmentation
│   ├── speech_to_sign/    # Text/Speech → Sign Video pipeline
│   ├── translation/       # Gloss → English translation
│   └── utils/             # Utility functions
├── train_hybrid.py        # Main training script
├── demo_camera.py         # Live camera demo
├── demo_video.py          # Video file demo
└── demo_speech_to_sign.py # Speech-to-sign Streamlit demo
```

---

## 🧠 1. MODEL ARCHITECTURE (`src/models/transformer.py`)

**Purpose**: Contains the core neural network architectures

### Key Classes:

#### **`HybridCTCAttentionModel`** (Lines 548-912)
**The main model** - Your best performing architecture

**Components:**
- **CNN Backbone** (ResNet-18): Extracts visual features from video frames
- **Transformer Encoder** (6 layers): Models temporal dependencies
- **CTC Head** (weight 0.3): Handles alignment
- **Attention Decoder** (3 layers, weight 0.7): Prevents CTC collapse

**Key Methods:**
- `forward()`: Forward pass with joint CTC+Attention loss
- `greedy_decode()`: Inference decoding
- `ctc_decode()`: CTC-only decoding

**Why Important**: This is your core innovation - solves CTC collapse problem

---

#### **`GlossDecoder`** (Lines 402-500)
**Transformer Decoder** for autoregressive gloss generation

**Purpose**: 
- Models output dependencies (gloss-to-gloss relationships)
- Provides stable gradients via cross-entropy loss
- Prevents CTC collapse

**Key Features:**
- Cross-attention to encoder outputs
- Masked self-attention
- Autoregressive generation

---

#### **`CNNBackbone`** (Lines 116-167)
**ResNet-18 Feature Extractor**

**Purpose**: Converts raw video frames → 512-dim feature vectors

**Features:**
- ImageNet pretrained weights
- Processes frames independently
- Output: (B, T, 512) features

---

#### **`R3D18Backbone`** (Lines 21-113)
**3D CNN for Video Processing** (Alternative to ResNet-18)

**Purpose**: Processes video chunks (8 frames) instead of individual frames

**Features:**
- Kinetics-400 pretrained
- Captures motion patterns
- Better for temporal understanding

---

## 📊 2. DATA LOADING (`src/data/phoenix_dataset.py`)

**Purpose**: Loads PHOENIX-2014 dataset with video frames and gloss annotations

### Key Class: `PhoenixDataset` (Lines 18-342)

**Responsibilities:**
- Loads video frames from disk
- Parses gloss annotations from CSV files
- Builds vocabulary from training data
- Handles frame sampling and padding

**Key Methods:**
- `_load_annotations()`: Reads corpus CSV files
- `_build_vocab()`: Creates gloss vocabulary
- `__getitem__()`: Returns video frames + gloss sequence

**Data Format:**
- Input: Video frames (T, H, W, C)
- Output: Gloss sequence (list of strings)

---

## 🎯 3. TRAINING (`train_hybrid.py`)

**Purpose**: Main training script for Hybrid CTC+Attention model

### Key Functions:

#### **`compute_wer()`** (Lines 59-95)
**Word Error Rate Calculation**
- Uses dynamic programming (edit distance)
- Measures recognition accuracy

#### **`train_epoch()`** (Lines 120-200)
**Single Epoch Training**
- Forward pass through model
- Computes joint loss (CTC + Cross-Entropy)
- Backpropagation and optimization
- Logs metrics

#### **`validate()`** (Lines 202-280)
**Validation Loop**
- Evaluates on dev set
- Computes WER
- Tracks blank ratio (collapse indicator)

#### **`main()`** (Lines 311-446)
**Training Orchestration**
- Sets up data loaders
- Initializes model
- Training loop with early stopping
- Checkpoint saving

**Key Features:**
- Joint loss: `0.3 * CTC + 0.7 * CrossEntropy`
- Learning rate scheduling (warmup + cosine)
- Gradient clipping
- Early stopping based on WER

---

## 🎬 4. DEMO FILES

### **`demo_video.py`** - Video File Processing

**Purpose**: Process pre-recorded videos or frame folders

**Key Class: `SignLanguageInference`**
- Loads trained model
- Processes video frames
- Generates predictions
- Translates to English

**Usage:**
```python
model = SignLanguageInference(checkpoint_path)
glosses, english = model.process_video(video_path)
```

---

### **`demo_camera.py`** - Live Camera Demo

**Purpose**: Real-time sign language recognition from webcam

**Key Features:**
- Frame buffer (64 frames)
- Real-time prediction
- OpenCV visualization
- Overlays predictions on video

**Controls:**
- SPACE: Capture and predict
- C: Clear buffer
- R: Record mode
- Q: Quit

---

### **`demo_speech_to_sign.py`** - Speech-to-Sign Streamlit App

**Purpose**: Interactive web interface for text/speech → sign video

**Features:**
- Text input → Sign video
- Audio upload → Sign video
- Visual video playback
- Gloss sequence display

---

## 🔄 5. SPEECH-TO-SIGN PIPELINE (`src/speech_to_sign/`)

### **`pipeline.py`** - Main Pipeline

**Class: `SpeechToSignPipeline`**

**Pipeline Flow:**
1. Speech → Text (Whisper ASR)
2. Text → Gloss (Translation)
3. Gloss → Video (Retrieval)

**Key Methods:**
- `transcribe_speech()`: ASR using Whisper
- `text_to_glosses()`: Text → Gloss translation
- `glosses_to_video()`: Video retrieval and synthesis

---

### **`gloss_retriever.py`** - Video Retrieval System

**Class: `GlossVideoRetriever`**

**Purpose**: Retrieves sign language video clips for each gloss

**Key Features:**
- Indexed database of 1,236 glosses
- Video segment retrieval
- Concatenative synthesis
- Smooth transitions between clips

**Methods:**
- `retrieve()`: Get video for gloss sequence
- `retrieve_as_video()`: Save as video file
- `build_index()`: Create index from dataset

---

### **`text_to_gloss.py`** - Text Translation

**Classes:**
- `RuleBasedTextToGloss`: Rule-based translation (weather domain)
- `TextToGlossModel`: Neural translation model (planned)

**Purpose**: Converts German text → DGS gloss sequence

---

## 🌐 6. TRANSLATION (`src/translation/gloss_to_english.py`)

**Purpose**: Translates DGS glosses to English

**Key Function: `translate_glosses()`**

**Features:**
- Dictionary-based translation
- Handles compound glosses
- Fallback for unknown glosses

**Usage:**
```python
glosses = ['MORGEN', 'REGEN', 'STARK']
english = translate_glosses(glosses)
# Returns: "Tomorrow rain strong"
```

---

## 🔧 7. DATA AUGMENTATION (`src/data/augmentation.py`)

**Purpose**: Video augmentation for training

**Key Features:**
- Spatial augmentation (crop, rotation, color jitter)
- Temporal augmentation (speed perturbation, frame dropout)
- **Critical**: No horizontal flip (changes sign meaning)

---

## 📈 8. EVALUATION (`evaluate.py`)

**Purpose**: Model evaluation and WER computation

**Features:**
- Batch evaluation
- WER calculation
- Prediction visualization
- Results export

---

## 🎛️ 9. CONFIGURATION FILES

### **`configs/transformer_ctc.yaml`**
Training hyperparameters:
- Learning rate
- Batch size
- Model dimensions
- Loss weights

---

## 🔑 Key File Relationships

```
Training Flow:
train_hybrid.py
  ↓ uses
src/models/transformer.py (HybridCTCAttentionModel)
  ↓ uses
src/data/phoenix_dataset.py (PhoenixDataset)
  ↓ loads
data/phoenix2014-release/ (Dataset)

Inference Flow:
demo_video.py / demo_camera.py
  ↓ loads
checkpoints/hybrid/best.pth
  ↓ uses
src/models/transformer.py (HybridCTCAttentionModel)
  ↓ uses
src/translation/gloss_to_english.py (translate_glosses)

Speech-to-Sign Flow:
demo_speech_to_sign.py
  ↓ uses
src/speech_to_sign/pipeline.py (SpeechToSignPipeline)
  ↓ uses
src/speech_to_sign/text_to_gloss.py
src/speech_to_sign/gloss_retriever.py
```

---

## 🎯 Most Important Files (Priority Order)

1. **`src/models/transformer.py`** - Core architecture (HybridCTC+Attention)
2. **`train_hybrid.py`** - Training script (your best results)
3. **`src/data/phoenix_dataset.py`** - Data loading
4. **`demo_video.py`** - Video inference demo
5. **`demo_camera.py`** - Live camera demo
6. **`src/speech_to_sign/pipeline.py`** - Bidirectional pipeline
7. **`src/translation/gloss_to_english.py`** - Translation module

---

## 💡 Key Design Patterns

1. **Modular Architecture**: Separate models, data, and pipelines
2. **Checkpoint Management**: Separate vocab.json from model weights
3. **Joint Loss**: Hybrid CTC+Attention prevents collapse
4. **Transfer Learning**: Pretrained ResNet-18 and R3D-18
5. **Retrieval-Based Synthesis**: Video retrieval for sign production

---

## 🐛 Common Issues & Solutions

1. **Checkpoint Loading**: Vocab in separate file (`vocab.json`)
2. **PyTorch 2.6**: Need `weights_only=False` in `torch.load()`
3. **Frame Format**: Expects (B, T, C, H, W) or (B, T, H, W, C)
4. **Vocabulary**: 1,236 glosses + special tokens (`<pad>`, `<sos>`, `<eos>`, `<blank>`)

---

*Last Updated: December 2025*





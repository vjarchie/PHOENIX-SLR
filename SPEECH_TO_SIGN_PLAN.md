# Speech-to-Sign Implementation Plan

> **Goal**: Reverse the SLR (Sign Language Recognition) pipeline to create SLP (Sign Language Production)
> 
> **Timeline**: 2 Weeks
> **Effort**: Reuse existing trained models, data, and infrastructure

---

## Executive Summary

### Current System: Sign → Text (SLR)
```
Video Frames → CNN → Transformer Encoder → CTC/Decoder → Gloss Sequence → Text
```

### Target System: Text → Sign (SLP)
```
Text/Speech → Text → Gloss Sequence → Transformer Decoder → Video Generation/Retrieval
```

---

## Architecture Comparison

| Component | SLR (Current) | SLP (Target) |
|-----------|---------------|--------------|
| Input | Video frames | Text/Speech |
| Encoder | CNN + Transformer | Text Embedding + Transformer |
| Decoder | Gloss Decoder | Video/Frame Generator |
| Output | Gloss → Text | Video frames |
| Loss | CTC + CE | Reconstruction + GAN |

---

## Recommended Approach: Hybrid Retrieval + Synthesis

Given the 2-week constraint, we'll use a **practical hybrid approach**:

### Phase 1: Text → Gloss Translation (Days 1-5)
Use the EXISTING gloss vocabulary and train a reverse model.

### Phase 2: Gloss → Video Synthesis (Days 6-12)
Two parallel tracks:
- **Track A (Safe)**: Retrieval-based - concatenate existing video clips
- **Track B (Advanced)**: Generative model using existing encoder as decoder

### Phase 3: Integration & Demo (Days 13-14)
End-to-end pipeline with UI.

---

## Week 1: Text-to-Gloss Translation

### Day 1-2: Data Preparation

The PHOENIX dataset already has aligned pairs:
```
Gloss: MORGEN REGEN STARK NORD
Text:  "Tomorrow heavy rain in the north"
```

We need to create the **reverse mapping**: Text → Gloss

#### Approach 1: Rule-Based (German Weather Domain)
Since PHOENIX is weather-specific, we can create rule-based mapping:

```python
# Simple keyword mapping
TEXT_TO_GLOSS = {
    "morgen": "MORGEN",
    "regen": "REGEN", 
    "schnee": "SCHNEE",
    "nord": "NORD",
    "süd": "SUED",
    ...
}
```

#### Approach 2: Neural Translation
Train a small seq2seq model using existing annotations.

### Day 3-5: Train Text-to-Gloss Model

**Option A: Reuse Decoder Architecture**
```python
class TextToGlossModel(nn.Module):
    def __init__(self, vocab_size, gloss_vocab_size, d_model=512):
        # Text encoder (BERT-like or simple embedding)
        self.text_encoder = TransformerEncoder(...)
        
        # Reuse our GlossDecoder but in reverse!
        self.gloss_decoder = GlossDecoder(
            vocab_size=gloss_vocab_size,
            d_model=d_model
        )
```

**Option B: Use Pre-trained German NMT**
- Use Helsinki-NLP/opus-mt-de-en as base
- Fine-tune for German text → DGS gloss

---

## Week 2: Gloss-to-Video Synthesis

### Day 6-8: Retrieval-Based System (Safe Path)

Create a **gloss-indexed video database**:

```python
class GlossVideoRetriever:
    def __init__(self, data_dir):
        self.gloss_to_videos = {}  # gloss -> list of video clips
        self._build_index()
    
    def _build_index(self):
        """Index all video segments by their gloss annotations."""
        for sample in dataset:
            glosses = sample['annotation']
            frames = sample['frames']
            
            # Use dynamic time warping or uniform split
            segments = self._segment_video(frames, glosses)
            
            for gloss, segment in zip(glosses, segments):
                if gloss not in self.gloss_to_videos:
                    self.gloss_to_videos[gloss] = []
                self.gloss_to_videos[gloss].append(segment)
    
    def retrieve(self, gloss_sequence):
        """Retrieve and concatenate video clips for gloss sequence."""
        video_clips = []
        for gloss in gloss_sequence:
            if gloss in self.gloss_to_videos:
                # Random selection or best-match
                clip = random.choice(self.gloss_to_videos[gloss])
                video_clips.append(clip)
        
        return self._smooth_concatenate(video_clips)
```

### Day 9-11: Generative Model (Advanced Path)

**Approach: Conditional Video Generation**

Use our trained encoder as a "semantic understanding" backbone:

```python
class SignVideoGenerator(nn.Module):
    def __init__(self, trained_encoder, d_model=512):
        """
        Use the TRAINED encoder from SLR as a teacher.
        
        The encoder understands sign language visual features.
        We train a decoder to generate those features from gloss.
        """
        # Freeze pretrained encoder (for feature extraction)
        self.sign_encoder = trained_encoder
        for param in self.sign_encoder.parameters():
            param.requires_grad = False
        
        # Gloss embedding
        self.gloss_embedding = nn.Embedding(gloss_vocab_size, d_model)
        
        # Temporal decoder (generates frame features)
        self.frame_decoder = nn.TransformerDecoder(...)
        
        # CNN decoder (features → pixels)
        self.pixel_decoder = CNNDecoder(d_model, output_channels=3)
    
    def forward(self, gloss_ids, num_frames):
        # Embed glosses
        gloss_features = self.gloss_embedding(gloss_ids)
        
        # Decode to frame features
        frame_features = self.frame_decoder(gloss_features, num_frames)
        
        # Decode to pixels
        video = self.pixel_decoder(frame_features)
        
        return video
```

**Training Strategy: Feature Matching**
```python
def train_generator(generator, encoder, dataloader):
    """
    Train generator to produce frames that encode to same features.
    
    Loss = ||Encoder(Generated) - Encoder(Real)||²
    """
    for batch in dataloader:
        real_frames = batch['frames']
        glosses = batch['gloss_ids']
        
        # Generate frames from glosses
        fake_frames = generator(glosses, num_frames=real_frames.shape[1])
        
        # Extract features using pretrained encoder
        with torch.no_grad():
            real_features = encoder.encode(real_frames)
        fake_features = encoder.encode(fake_frames)
        
        # Feature matching loss
        loss = F.mse_loss(fake_features, real_features)
        
        # Optional: pixel-level reconstruction
        loss += 0.1 * F.l1_loss(fake_frames, real_frames)
```

### Day 12: Skeleton-Based Generation (Alternative)

If video generation is too hard, use **pose/skeleton representation**:

```python
class SkeletonGenerator(nn.Module):
    """
    Generate 2D/3D skeleton sequences instead of pixels.
    Much easier to train, can render with avatar later.
    """
    def __init__(self, gloss_vocab_size, num_joints=25):
        self.embedding = nn.Embedding(gloss_vocab_size, 512)
        self.decoder = nn.TransformerDecoder(...)
        self.joint_predictor = nn.Linear(512, num_joints * 3)  # x, y, confidence
    
    def forward(self, gloss_ids, num_frames):
        gloss_features = self.embedding(gloss_ids)
        frame_features = self.decoder(gloss_features)
        skeletons = self.joint_predictor(frame_features)
        return skeletons.view(-1, num_frames, 25, 3)
```

---

## Day 13-14: Integration

### Full Pipeline

```python
class SpeechToSignPipeline:
    def __init__(self, asr_model, text2gloss_model, gloss2video_model):
        self.asr = asr_model  # Whisper or similar
        self.text2gloss = text2gloss_model
        self.gloss2video = gloss2video_model
    
    def __call__(self, audio_or_text):
        # Step 1: Speech to Text (if audio input)
        if isinstance(audio_or_text, np.ndarray):
            text = self.asr.transcribe(audio_or_text)
        else:
            text = audio_or_text
        
        # Step 2: Text to Gloss
        gloss_sequence = self.text2gloss.translate(text)
        
        # Step 3: Gloss to Video
        video = self.gloss2video.generate(gloss_sequence)
        
        return {
            'text': text,
            'glosses': gloss_sequence,
            'video': video
        }
```

### Web Demo (Streamlit)

```python
# demo_speech_to_sign.py
import streamlit as st

st.title("🤟 Speech to Sign Language")

input_mode = st.radio("Input Mode", ["Text", "Audio"])

if input_mode == "Text":
    text = st.text_area("Enter German text:", "Morgen gibt es Regen im Norden")
    
    if st.button("Generate Sign Language"):
        with st.spinner("Generating..."):
            result = pipeline(text)
        
        st.subheader("Gloss Sequence")
        st.write(" → ".join(result['glosses']))
        
        st.subheader("Sign Language Video")
        st.video(result['video'])

elif input_mode == "Audio":
    audio = st.audio_input("Record your speech")
    if audio:
        result = pipeline(audio)
        st.video(result['video'])
```

---

## Implementation Files to Create

```
PHOENIX-SLR/
├── src/
│   ├── speech_to_sign/
│   │   ├── __init__.py
│   │   ├── text_to_gloss.py       # Text → Gloss model
│   │   ├── gloss_retriever.py     # Retrieval-based video lookup
│   │   ├── video_generator.py     # Generative model (optional)
│   │   ├── pipeline.py            # Full pipeline
│   │   └── skeleton_generator.py  # Pose-based generation
│   └── ...
├── train_text2gloss.py            # Training script for T2G
├── train_video_generator.py       # Training script for G2V
├── demo_speech_to_sign.py         # Streamlit demo
└── SPEECH_TO_SIGN_PLAN.md         # This document
```

---

## Detailed 2-Week Schedule

### Week 1: Text → Gloss

| Day | Task | Output |
|-----|------|--------|
| 1 | Extract text-gloss pairs from PHOENIX annotations | `data/text_gloss_pairs.json` |
| 2 | Build text vocabulary, tokenizer | `src/speech_to_sign/text_tokenizer.py` |
| 3 | Implement Text-to-Gloss model | `src/speech_to_sign/text_to_gloss.py` |
| 4 | Train Text-to-Gloss model | Trained model checkpoint |
| 5 | Evaluate T2G, fix issues | Working T2G model |

### Week 2: Gloss → Video + Integration

| Day | Task | Output |
|-----|------|--------|
| 6 | Build gloss-video index from dataset | `data/gloss_video_index.pkl` |
| 7 | Implement retrieval-based synthesis | `src/speech_to_sign/gloss_retriever.py` |
| 8 | Add video smoothing/blending | Smooth video output |
| 9 | (Optional) Implement generative model | `src/speech_to_sign/video_generator.py` |
| 10 | (Optional) Train generative model | Trained generator |
| 11 | Build full pipeline | `src/speech_to_sign/pipeline.py` |
| 12 | Add speech input (Whisper) | Speech → Sign |
| 13 | Build Streamlit demo | `demo_speech_to_sign.py` |
| 14 | Testing, polish, documentation | Complete system |

---

## Reusing Existing Components

| Existing Component | Reuse In SLP |
|-------------------|--------------|
| `PhoenixDataset` | Extract text-gloss pairs, build video index |
| `vocab.json` | Gloss vocabulary (same for T2G output) |
| `GlossDecoder` | Basis for Text-to-Gloss decoder |
| `CNNBackbone` (pretrained) | Feature extractor for video generator training |
| `HybridCTCAttentionModel.encode()` | Teacher signal for generative training |
| `train_hybrid.py` | Template for training scripts |
| `dashboard.py` | Template for demo UI |

---

## Expected Results

### Minimum Viable Product (Day 10)
- Text input → Gloss sequence → Retrieved video clips
- Works for PHOENIX weather domain
- ~70% visual quality (clip concatenation visible)

### Full System (Day 14)
- Speech input → Text → Gloss → Smooth video
- Optional: Generated video (if generative model works)
- Web demo with real-time preview

---

## Technical Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| No direct text-gloss pairs | Parse annotations, reverse engineer from corpus |
| Video clip boundaries unknown | Use uniform segmentation or train aligner |
| Generated video quality | Feature matching loss + retrieval fallback |
| German text variations | Use lemmatization or subword tokenization |
| Real-time performance | Precompute embeddings, use retrieval |

---

## Quick Start Commands

```bash
# Day 1: Extract data
python scripts/prepare_text2gloss_data.py

# Day 4: Train Text-to-Gloss
python train_text2gloss.py --epochs 50 --batch_size 16

# Day 7: Build video index
python scripts/build_gloss_video_index.py

# Day 13: Run demo
streamlit run demo_speech_to_sign.py
```

---

## Alternative: Simpler Lookup-Only Approach (1 Week)

If time is very tight, implement just:

1. **Keyword extraction** from input text
2. **Gloss mapping** using a dictionary
3. **Video retrieval** using pre-indexed clips
4. **Concatenation** with simple crossfade

This gives a working demo in ~5 days but lower quality.

---

*Document created: December 17, 2025*
*Project: PHOENIX Sign Language Production (SLP)*


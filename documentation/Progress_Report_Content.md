# MTech Progress Report - Content & Prompts for PDF Generation

> **Student**: Archie Narayan 
> **Roll Number**: 24A03RES119  
> **Institution**: Indian Institute of Technology Patna  
> **Program**: MTech (Artificial Intelligence and Data Science Engineering)  
> **Semester**: III  
> **Project Title**: Real-Time Bidirectional Continuous Sign Language Interpretation Using Deep Learning

---

## 📄 DOCUMENT STRUCTURE (7 Pages)

### Page 1: Cover Page
### Page 2: Abstract & Introduction
### Page 3: Literature Review & Problem Statement
### Page 4: Methodology & Architecture
### Page 5: Implementation & Results
### Page 6: Speech-to-Sign Extension & Future Work
### Page 7: References & Acknowledgements

---

# 📝 PAGE-BY-PAGE CONTENT

---

## PAGE 1: COVER PAGE

### PROMPT FOR PDF AGENT:
```
Create a professional academic cover page with the following elements:
- IIT Patna logo at the top center
- Title: "Progress Report" in large bold text
- Subtitle: "Real-Time Bidirectional Continuous Sign Language Interpretation Using Deep Learning"
- Student details in a centered box:
  - Name: Ritik Tiwari
  - Roll Number: 24A03RES160
  - Program: Master of Technology
  - Department: Computer Science & Engineering
  - Semester: III
  - Academic Year: 2024-25
- Supervisor name (if applicable)
- "Indian Institute of Technology Patna" at the bottom
- Date: December 2025
- Use formal academic styling with blue/navy accents
```

### CONTENT:
```
INDIAN INSTITUTE OF TECHNOLOGY PATNA
Department of Computer Science & Engineering

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SEMESTER III PROGRESS REPORT

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Real-Time Bidirectional Continuous Sign Language 
Interpretation Using Deep Learning

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Submitted by:
RITIK TIWARI
Roll No: 24A03RES160
MTech (Computer Science & Engineering)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

December 2025
```

---

## PAGE 2: ABSTRACT & INTRODUCTION

### PROMPT FOR PDF AGENT:
```
Create Page 2 with:
- Section 1: Abstract (200-250 words, justified text)
- Section 2: Introduction with subsections 2.1 Motivation, 2.2 Objectives
- Use Times New Roman 12pt, 1.5 line spacing
- Section headers in bold, numbered format
```

### CONTENT:

#### 1. ABSTRACT

Sign language serves as the primary mode of communication for over 70 million deaf and hard-of-hearing individuals worldwide. However, the communication barrier between sign language users and the hearing population remains a significant challenge. This project presents a comprehensive deep learning-based system for **bidirectional** continuous sign language interpretation, enabling seamless communication in both directions.

The system comprises two main pipelines: (1) **Sign Language Recognition (SLR)** - converting continuous sign language videos to text, and (2) **Sign Language Production (SLP)** - converting text/speech to sign language videos. For the SLR component, we implemented a **Hybrid CTC+Attention Transformer architecture** trained on the RWTH-PHOENIX-Weather 2014 dataset, achieving a Word Error Rate (WER) of **50.91%** on the test set. The architecture combines a pretrained ResNet-18 CNN backbone with a 6-layer Transformer encoder, using joint CTC and cross-entropy losses to prevent the common CTC collapse problem.

For the SLP component, we developed a retrieval-based video synthesis system that translates text to gloss sequences and retrieves corresponding sign video segments from an indexed database, enabling real-time text-to-sign conversion.

**Keywords**: Sign Language Recognition, Continuous Sign Language, CTC, Transformer, Attention Mechanism, Deep Learning, PHOENIX-2014

---

#### 2. INTRODUCTION

##### 2.1 Motivation

According to the World Health Organization, approximately 466 million people worldwide have disabling hearing loss. Sign language is a complete, natural language with its own grammar and syntax, used by deaf communities across the globe. However, the lack of sign language interpreters creates significant barriers in education, healthcare, employment, and social interactions.

Advances in deep learning and computer vision have opened new possibilities for automated sign language interpretation. Unlike isolated sign recognition (recognizing individual signs), **Continuous Sign Language Recognition (CSLR)** deals with connected signs in natural sentences, presenting unique challenges including:
- Variable-length sequences with no explicit word boundaries
- Co-articulation effects between consecutive signs
- Signer variations in speed, style, and appearance
- Limited availability of annotated datasets

##### 2.2 Objectives

The primary objectives of this project are:

1. **Develop a robust CSLR system** capable of recognizing continuous sign language from video input with competitive accuracy on benchmark datasets.

2. **Implement a bidirectional interpretation system** that enables communication in both directions - from sign language to text and from text/speech to sign language.

3. **Create real-time demonstration systems** including camera-based sign recognition and text-to-sign video generation.

4. **Document technical learnings** regarding deep learning architectures for sequence-to-sequence tasks with variable-length alignment.

---

## PAGE 3: LITERATURE REVIEW & PROBLEM STATEMENT

### PROMPT FOR PDF AGENT:
```
Create Page 3 with:
- Section 3: Literature Review with a table of related works
- Section 4: Problem Statement with clear articulation of challenges
- Include a comparison table of existing methods
- Use academic formatting with proper citations
```

### CONTENT:

#### 3. LITERATURE REVIEW

##### 3.1 Evolution of Sign Language Recognition

Sign language recognition has evolved significantly over the past decade. Early approaches relied on hand-crafted features and Hidden Markov Models (HMMs). The introduction of deep learning, particularly Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs), marked a paradigm shift in the field.

**Table 1: Comparison of State-of-the-Art Methods on PHOENIX-2014**

| Method | Year | Architecture | Dev WER | Test WER |
|--------|------|--------------|---------|----------|
| Koller et al. | 2017 | CNN-HMM-RNN | 26.0% | 26.8% |
| Camgöz et al. | 2020 | Transformer + CTC | 21.8% | 24.5% |
| Niu & Mak | 2020 | Stochastic CSLR | 25.4% | 24.0% |
| Zhu et al. (VAC) | 2021 | Visual Alignment | 19.3% | 21.2% |
| CorrNet | 2023 | Correlation Net | 17.3% | 17.8% |
| **Ours** | **2025** | **Hybrid CTC+Attention** | **51.44%** | **50.91%** |

##### 3.2 CTC-Based Approaches

Connectionist Temporal Classification (CTC) introduced by Graves et al. (2006) enables training sequence-to-sequence models without requiring frame-level alignments. CTC has been widely adopted in speech recognition and sign language recognition due to its ability to handle variable-length sequences.

##### 3.3 Hybrid CTC+Attention Models

Recent works have shown that pure CTC models are prone to collapse, outputting only blank tokens. The hybrid CTC+Attention approach (Watanabe et al., 2017) combines the alignment capabilities of CTC with the sequence modeling power of attention-based decoders, providing more stable training.

---

#### 4. PROBLEM STATEMENT

##### 4.1 Technical Challenges

The development of a practical CSLR system faces several technical challenges:

1. **CTC Collapse Problem**: Pure CTC models tend to learn trivial solutions by outputting only blank tokens, resulting in 100% word error rate regardless of input.

2. **Long-Range Dependencies**: Sign language sentences can span hundreds of frames, requiring models to capture long-range temporal dependencies.

3. **Limited Labeled Data**: Unlike speech recognition, sign language datasets are relatively small, making deep learning approaches prone to overfitting.

4. **Variable Speed and Duration**: Signers perform signs at different speeds, and sign duration varies based on context and signer preference.

##### 4.2 Research Questions

This project addresses the following research questions:

**RQ1**: How can we prevent CTC collapse in continuous sign language recognition while maintaining the benefits of CTC-based alignment?

**RQ2**: What is the optimal architecture for capturing both short-term motion patterns and long-term semantic dependencies in sign language videos?

**RQ3**: How can we create a practical bidirectional system that enables real-time interpretation?

---

## PAGE 4: METHODOLOGY & ARCHITECTURE

### PROMPT FOR PDF AGENT:
```
Create Page 4 with:
- Section 5: Methodology with detailed architecture description
- Include an architecture diagram (ASCII or proper figure)
- Section on loss functions and training strategy
- Use technical but clear language
```

### CONTENT:

#### 5. METHODOLOGY

##### 5.1 Dataset

We use the **RWTH-PHOENIX-Weather 2014** dataset, the standard benchmark for continuous sign language recognition:

| Property | Value |
|----------|-------|
| Language | German Sign Language (DGS) |
| Domain | Weather forecasts |
| Vocabulary | 1,236 unique glosses |
| Training samples | 5,672 |
| Development samples | 540 |
| Test samples | 629 |
| Frame dimensions | 210 × 260 pixels |
| Frame rate | 25 FPS |
| Total size | ~53 GB |

##### 5.2 System Architecture

Our system employs a **Hybrid CTC+Attention architecture** consisting of four main components:

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Video Frames                       │
│                    (Batch, Time, 3, 260, 210)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              CNN BACKBONE: ResNet-18 (Pretrained)           │
│              Extracts 512-dim features per frame            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              TRANSFORMER ENCODER (6 layers)                  │
│              8 attention heads, d_model=512                  │
│              Models temporal dependencies                    │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────────┐
│       CTC HEAD          │     │    ATTENTION DECODER        │
│    (Weight: 0.3)        │     │       (Weight: 0.7)         │
│  Linear → LogSoftmax    │     │  3-layer Transformer        │
│                         │     │  Cross-attention to encoder │
│  Provides alignment     │     │  Models output dependencies │
└─────────────────────────┘     └─────────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      JOINT LOSS                              │
│        L = 0.3 × L_CTC + 0.7 × L_CrossEntropy               │
└─────────────────────────────────────────────────────────────┘
```

##### 5.3 Loss Functions

**CTC Loss**: Enables alignment-free training by marginalizing over all possible alignments between input frames and output glosses.

**Cross-Entropy Loss**: Applied to the decoder output with label smoothing (0.1) to improve generalization.

**Joint Loss**: The weighted combination prevents CTC collapse while maintaining alignment benefits:
```
L_total = λ_CTC × L_CTC + λ_CE × L_CE
where λ_CTC = 0.3, λ_CE = 0.7
```

##### 5.4 Training Strategy

- **Optimizer**: AdamW with weight decay 0.05
- **Learning Rate**: 1e-4 with warmup (5 epochs) + cosine annealing
- **Batch Size**: 4 (limited by GPU memory)
- **Max Frames**: 64 frames per video
- **Gradient Clipping**: max_norm = 5.0
- **Early Stopping**: Patience of 15 epochs

---

## PAGE 5: IMPLEMENTATION & RESULTS

### PROMPT FOR PDF AGENT:
```
Create Page 5 with:
- Section 6: Implementation Details
- Section 7: Experimental Results with tables and training curves
- Include model parameters and training time
- Show WER progression during training
```

### CONTENT:

#### 6. IMPLEMENTATION DETAILS

##### 6.1 Development Environment

| Component | Specification |
|-----------|---------------|
| Programming Language | Python 3.10 |
| Deep Learning Framework | PyTorch 2.0 |
| GPU | NVIDIA GPU with CUDA |
| Training Time | ~20 hours (100 epochs) |
| Code Repository | Local (D:\PHOENIX-SLR) |

##### 6.2 Model Configuration

| Parameter | Value |
|-----------|-------|
| Backbone | ResNet-18 (ImageNet pretrained) |
| d_model | 512 |
| Encoder Layers | 6 |
| Decoder Layers | 3 |
| Attention Heads | 8 |
| Feed-forward Dim | 2048 |
| Dropout | 0.2 |
| Total Parameters | 45.2M |

##### 6.3 Data Augmentation

- **Spatial**: Random crop (0.85-1.0 scale), rotation (±8°), brightness/contrast jitter
- **Temporal**: Speed perturbation (0.8-1.2x), uniform frame sampling
- **Important**: No horizontal flip (changes sign meaning)

---

#### 7. EXPERIMENTAL RESULTS

##### 7.1 Main Results

**Table 2: Final Model Performance**

| Metric | Development Set | Test Set |
|--------|-----------------|----------|
| Word Error Rate (WER) | 51.44% | **50.91%** |
| Best Epoch | 98 | - |
| Training Loss | 0.0006 | - |

##### 7.2 Training Progress

```
Epoch   1: ████████████████████████████████████████████████ 91.75%
Epoch  10: ████████████████████████████████████             70.25%
Epoch  20: ██████████████████████████████                   57.55%
Epoch  40: ████████████████████████████                     55.56%
Epoch  60: ████████████████████████████                     55.04%
Epoch  80: ███████████████████████████                      53.76%
Epoch  98: ███████████████████████████                      52.85% (Best)
Epoch 100: ███████████████████████████                      53.52%
```

##### 7.3 CTC Collapse Prevention

Our hybrid approach successfully prevented CTC collapse, which was a major issue with pure CTC training:

| Approach | WER after 5 epochs | Final WER |
|----------|-------------------|-----------|
| Pure CTC | 100% (collapsed) | 100% |
| **Hybrid CTC+Attention** | **91.75%** | **50.91%** |

##### 7.4 Key Findings

1. **Hybrid architecture prevents collapse**: The attention decoder provides stable gradients even when CTC starts to collapse.

2. **Lower CTC weight is crucial**: CTC weight > 0.5 leads to collapse; 0.3 provides optimal balance.

3. **Pretrained backbone essential**: ImageNet-pretrained ResNet-18 significantly outperforms random initialization.

4. **Label smoothing helps**: 0.1 label smoothing improves generalization by ~2% WER.

---

## PAGE 6: SPEECH-TO-SIGN EXTENSION & FUTURE WORK

### PROMPT FOR PDF AGENT:
```
Create Page 6 with:
- Section 8: Speech-to-Sign Implementation (bidirectional capability)
- Section 9: Future Work and improvements
- Include pipeline diagram for reverse direction
```

### CONTENT:

#### 8. SPEECH-TO-SIGN IMPLEMENTATION

To achieve bidirectional communication, we extended the system to convert text/speech into sign language videos.

##### 8.1 Reverse Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              INPUT: Speech or Text                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│    SPEECH-TO-TEXT (Optional): Whisper ASR                   │
│    Converts spoken German to text                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│    TEXT-TO-GLOSS: Rule-based / Neural Translation           │
│    Maps German text to DGS gloss sequence                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│    GLOSS-TO-VIDEO: Retrieval-Based Synthesis                │
│    Retrieves and concatenates sign video segments           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              OUTPUT: Sign Language Video                     │
└─────────────────────────────────────────────────────────────┘
```

##### 8.2 Implementation Components

1. **Text-to-Gloss Model**: Rule-based translation using domain-specific mappings for the weather domain, with provisions for neural translation.

2. **Gloss Video Index**: Pre-built index mapping each gloss to corresponding video segments from the training data.

3. **Video Retrieval & Synthesis**: Retrieves video clips for each gloss and concatenates with smooth transitions.

##### 8.3 Demo Capabilities

- Real-time camera-based sign recognition with English translation
- Text input → Sign language video generation
- Speech input → Sign language video (using Whisper ASR)

---

#### 9. FUTURE WORK

##### 9.1 Short-term Improvements

| Improvement | Expected Impact |
|-------------|-----------------|
| Increase max_frames to 128+ | 5-8% WER reduction |
| Add extensive data augmentation | 5-10% WER reduction |
| Implement VAC (Visual Alignment Constraint) | 3-5% WER reduction |

##### 9.2 Advanced Techniques

1. **Self-Distillation**: Use model's own predictions for additional supervision
2. **Multi-Scale Temporal Modeling**: Temporal convolutions at multiple scales
3. **I3D/S3D Features**: Pre-extracted video features from larger datasets

##### 9.3 Production Deployment

- Mobile application for real-time interpretation
- Web-based interface for accessibility
- Integration with video conferencing platforms

---

#### 10. CONCLUSION

This project successfully developed a bidirectional continuous sign language interpretation system using deep learning. Key contributions include:

1. **Hybrid CTC+Attention architecture** that prevents CTC collapse and achieves 50.91% WER on PHOENIX-2014

2. **Complete bidirectional pipeline** enabling communication in both directions

3. **Technical documentation** of architecture evolution and lessons learned

4. **Working demonstrations** for both sign-to-text and text-to-sign conversion

The system provides a foundation for practical sign language interpretation applications and demonstrates the viability of transformer-based architectures for this challenging task.

---

## PAGE 7: REFERENCES & ACKNOWLEDGEMENTS

### PROMPT FOR PDF AGENT:
```
Create Page 7 with:
- References in IEEE format
- Acknowledgements section
- Declaration (if required)
- Signature line for student and supervisor
```

### CONTENT:

#### REFERENCES

[1] O. Koller, J. Forster, and H. Ney, "Continuous sign language recognition: Towards large vocabulary statistical recognition systems handling multiple signers," *Computer Vision and Image Understanding*, vol. 141, pp. 108-125, 2015.

[2] N. C. Camgöz, S. Hadfield, O. Koller, H. Ney, and R. Bowden, "Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020, pp. 10023-10033.

[3] A. Graves, S. Fernández, F. Gomez, and J. Schmidhuber, "Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks," in *Proceedings of the 23rd International Conference on Machine Learning*, 2006, pp. 369-376.

[4] S. Watanabe et al., "Hybrid CTC/Attention Architecture for End-to-End Speech Recognition," *IEEE Journal of Selected Topics in Signal Processing*, vol. 11, no. 8, pp. 1240-1253, 2017.

[5] Z. Zhu, D. Li, and L. Wang, "Visual Alignment Constraint for Continuous Sign Language Recognition," in *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021, pp. 11542-11551.

[6] A. Vaswani et al., "Attention is all you need," in *Advances in Neural Information Processing Systems*, vol. 30, 2017.

[7] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770-778.

[8] D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," in *International Conference on Learning Representations (ICLR)*, 2015.

---

#### ACKNOWLEDGEMENTS

I would like to express my sincere gratitude to my supervisor for their guidance and support throughout this project. I also thank the IIT Patna for providing the computational resources necessary for training deep learning models.

Special thanks to the creators of the RWTH-PHOENIX-Weather 2014 dataset for making their data publicly available for research purposes.

---

#### DECLARATION

I hereby declare that the work presented in this report is my own original work carried out during Semester III of my MTech program at IIT Patna. I have not plagiarized or submitted this work elsewhere for any other degree or diploma.

---

**Student Signature**: _______________________

**Name**: Archie Narayan

**Roll No**: 24A03RES119=-

**Date**: December 2025

---

**Supervisor Signature**: _______________________

**Name**: _______________________

**Designation**: _______________________

**Date**: _______________________

---

# 🔧 PDF GENERATION INSTRUCTIONS

## For the PDF Generation Agent:

### Document Settings:
- **Page Size**: A4 (210 × 297 mm)
- **Margins**: 1 inch (2.54 cm) all sides
- **Font**: Times New Roman
- **Body Text**: 12pt, 1.5 line spacing
- **Headings**: Bold, 14pt for main sections, 12pt for subsections
- **Tables**: Centered, with light gray header rows
- **Code/Architecture Diagrams**: Monospace font (Courier New), 10pt

### Styling:
- Use blue (#003366) for section headings
- Tables should have thin borders (0.5pt)
- Include page numbers at bottom center
- Header: "MTech Progress Report - Semester III" (after cover page)

### Figures to Include:
1. System architecture diagram (Page 4)
2. Training progress graph (Page 5) - WER vs Epochs
3. Speech-to-Sign pipeline diagram (Page 6)

### Table of Contents (Optional):
Generate after cover page if document exceeds 6 pages.

---

*Content prepared: December 2025*
*Project: PHOENIX-SLR - Bidirectional Sign Language Interpretation*






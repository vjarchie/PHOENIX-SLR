# PHOENIX-SLR Viva Presentation (2-3 minutes)

## Slide 1 - Motivation and Problem

**Key points (3-4 bullets)**
- Sign language communication still faces a major accessibility gap in real-world settings.
- Continuous Sign Language Recognition (CSLR) is harder than isolated signs due to unclear sign boundaries and long temporal context.
- Project goal: build a practical pipeline for `video -> gloss text` with a bidirectional demo.
- Dataset: RWTH-PHOENIX-Weather 2014 (standard CSLR benchmark).

**Visual suggestion**
- Use `motivation_accessibility.svg`.

**Examiner may ask**
- Why PHOENIX dataset instead of collecting your own dataset?

**Safe answer**
- PHOENIX is the most established CSLR benchmark with standard splits and published baselines, so the evaluation is academically valid and comparable.

---

## Slide 2 - Main Architecture

**Key points (3-4 bullets)**
- Input video frames are encoded by a pretrained ResNet-18 visual backbone.
- A Transformer encoder captures temporal dependencies across frame features.
- Two decoding paths are used jointly: CTC head and attention decoder.
- Output is gloss sequence, optionally shown with readable English rendering in demo.

**Visual suggestion**
- Use `hybrid_architecture.svg`.

**Examiner may ask**
- What is your main architectural contribution?

**Safe answer**
- The core design choice is the hybrid CTC+Attention setup to combine alignment stability with better sequence modeling.

---

## Slide 3 - Approach and Why Hybrid

**Key points (3-4 bullets)**
- Pure CTC can become blank-heavy in difficult continuous sequences.
- Attention decoder captures token dependencies and improves sequence coherence.
- Hybrid loss balances both strengths: CTC for alignment, attention for contextual decoding.
- In this implementation, hybrid setup was more reliable than pure CTC behavior.

**Examiner may ask**
- Why not use only attention-based decoding?

**Safe answer**
- CTC provides strong monotonic alignment guidance for variable-length signing streams, which complements attention and improves stability.

---

## Slide 4 - Demo and Limitations

**Key points (3-4 bullets)**
- Sign -> Text: webcam signing decoded to glosses and English output (including auto decode mode).
- Text -> Sign: German phrase to rule-based glosses to PHOENIX clip retrieval playback.
- Practical value: demonstrates two-way communication direction.
- Limitations: test WER ~50.9%, weather-domain data, signer/style variation, retrieval-based text-to-sign is not full generative signing.

**Visual suggestion**
- Use `limitations_future.svg` (left side focus for this slide).

**Examiner may ask**
- Why is WER not close to SOTA?

**Safe answer**
- This is a resource-constrained academic implementation without large-scale pretraining and heavy optimization used in SOTA systems.

---

## Slide 5 - Conclusion and Future Scope

**Key points (3-4 bullets)**
- Implemented a complete CSLR pipeline and integrated a usable bidirectional prototype.
- Hybrid CTC+Attention improved practical robustness compared to pure CTC path.
- Demonstrated end-to-end usability, not just offline metrics.
- Future work: stronger pretraining, signer adaptation, language-model support, and neural text-to-sign generation.

**Visual suggestion**
- Use `limitations_future.svg` (right side roadmap focus for this slide).

**Examiner may ask**
- What is the most practical next step?

**Safe answer**
- Improve generalization first using stronger pretrained visual features plus signer/domain adaptation, then push WER down before deployment-level expansion.

---

## Optional opening and closing lines

**Opening (1 line)**
- "This work focuses on making continuous sign language understanding more practical using a hybrid recognition architecture and a live bidirectional demo."

**Closing (1 line)**
- "This project establishes a practical baseline for CSLR and a clear path toward real assistive communication systems."

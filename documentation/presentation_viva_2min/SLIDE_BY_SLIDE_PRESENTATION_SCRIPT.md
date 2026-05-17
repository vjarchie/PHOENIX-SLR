# Slide-by-Slide Presentation Script

Use this as your speaking script while showing `SLIDES_DOCUMENT.md` (5 slides, 2-3 minutes total).

---

## Slide 1 - Motivation and Problem (25-30 sec)

**What to say**
"Good morning. This project focuses on reducing the communication gap between sign language users and spoken-language users.  
The main challenge is that I am solving **continuous** sign language recognition, where sign boundaries are not explicit and timing varies a lot.  
So my objective was to build a practical end-to-end pipeline from video to gloss text, and then demonstrate a bidirectional prototype.  
For evaluation, I used RWTH-PHOENIX-Weather 2014 because it is a standard benchmark with established splits."

**What to point on slide**
- Accessibility gap icon/visual
- "Continuous" difficulty line
- Dataset name

**If interrupted with question**
"I chose PHOENIX for academic comparability and reproducibility, not because it is the easiest dataset."

---

## Slide 2 - Main Architecture (30-35 sec)

**What to say**
"At the model level, each frame first goes through a pretrained ResNet-18 to extract visual features.  
Then a Transformer encoder captures temporal context across the sequence.  
For decoding, I use a hybrid design with two paths: a CTC head and an attention decoder.  
This gives gloss sequence output, and for demo readability I also show a simple English rendering."

**What to point on slide**
- ResNet-18 block
- Transformer encoder block
- Split into CTC + Attention

**If interrupted with question**
"The core contribution here is not a brand-new block, but the practical hybrid combination that improved stability in my setup."

---

## Slide 3 - Approach and Why Hybrid (30-35 sec)

**What to say**
"The reason for hybrid decoding is practical.  
Pure CTC can become blank-heavy in difficult continuous sequences.  
Attention helps sequence coherence and token dependency modeling, but CTC is very useful for monotonic alignment.  
So combining them gave better behavior in my implementation than relying on only one method."

**What to point on slide**
- "Pure CTC blank-heavy" bullet
- "Hybrid loss balances both strengths" bullet

**If interrupted with question**
"In one line: CTC gives alignment stability, attention gives sequence quality."

---

## Slide 4 - Demo and Limitations (35-40 sec)

**What to say**
"I implemented a bidirectional prototype.  
In Sign-to-Text mode, webcam signing is decoded into glosses and readable text, including auto-decode for streaming-like interaction.  
In Text-to-Sign mode, German input is converted to glosses and visualized using PHOENIX clip retrieval.  
Current limitations are: test WER around 50.9%, weather-domain constraints, signer variability, and retrieval-based output instead of fully generative sign production."

**What to point on slide**
- Sign->Text line
- Text->Sign line
- Limitations bullet

**If interrupted with question**
"This is a constrained academic implementation; SOTA-level performance typically uses much larger pretraining and compute budgets."

---

## Slide 5 - Conclusion and Future Scope (25-30 sec)

**What to say**
"To conclude, this work delivers a full CSLR pipeline plus a practical bidirectional demo.  
The hybrid CTC+Attention design improved robustness compared to pure CTC behavior in this project context.  
Beyond metrics, I focused on end-to-end usability.  
The next practical step is improving generalization using stronger pretraining and signer adaptation, followed by language-model integration and neural text-to-sign generation."

**What to point on slide**
- "Complete pipeline" bullet
- "Future work" bullet

**Final closing line**
"This project establishes a practical baseline for continuous sign language understanding and a clear path toward real assistive communication systems."

---

## Optional 10-second opening before Slide 1

"I will briefly cover motivation, architecture, why hybrid decoding, demo results with limitations, and practical next steps."

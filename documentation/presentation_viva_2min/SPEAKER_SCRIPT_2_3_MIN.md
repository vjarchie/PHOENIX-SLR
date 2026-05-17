# Speaker Script (2-3 Minutes)

## Slide 1 (about 25 seconds)
Good morning. My project addresses an accessibility problem: converting continuous sign language video into usable text. Continuous sign language recognition is harder than isolated sign recognition because sign boundaries are not explicit and temporal context is long. I used the RWTH-PHOENIX-Weather 2014 dataset because it is a standard benchmark and gives academically valid comparison.

## Slide 2 (about 30 seconds)
At a high level, each frame is encoded using a pretrained ResNet-18 backbone. Then a Transformer encoder models temporal dependencies across the sequence. For decoding, I use a hybrid setup with both CTC and attention signals. This produces gloss-level predictions and supports readable output in the live demo.

## Slide 3 (about 35 seconds)
The reason for choosing hybrid CTC plus attention is practical. Pure CTC can become blank-dominant in difficult continuous sequences. CTC is still useful for alignment over variable-length inputs, while attention improves contextual decoding and token relationships. Combining both gave more reliable behavior in my implementation compared to a pure CTC path.

## Slide 4 (about 35 seconds)
I also built a bidirectional demo. In Sign-to-Text mode, webcam input is decoded to glosses and readable text, including auto decode for streaming-like usage. In Text-to-Sign mode, German input is converted to glosses and visualized using PHOENIX clip retrieval. Current limitations are test WER around 50.9 percent, domain limitation of weather data, signer variation challenges, and retrieval-style text-to-sign rather than fully generative synthesis.

## Slide 5 (about 25 seconds)
To conclude, this work delivers an end-to-end CSLR pipeline with a practical demo and clear architectural reasoning. Hybrid CTC plus attention improved robustness in this project setting. Future work includes stronger pretraining, signer adaptation, language-model integration, and more natural neural sign generation.

---

## Backup one-line closing
"This project establishes a practical and explainable baseline for continuous sign language understanding, with a clear path toward real assistive communication deployment."

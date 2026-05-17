# Viva Question Bank (50+) with Model Answers

Based on the 2-3 minute presentation on PHOENIX-SLR hybrid CTC+Attention and bidirectional demo.

## A) Motivation and Problem Definition

1) **Q:** What is the real-world problem your project is trying to solve?  
**A:** It addresses the communication gap by converting continuous sign language video into readable text, enabling assistive interaction.

2) **Q:** Why is sign language recognition important in assistive technology?  
**A:** It supports accessibility for deaf and hard-of-hearing users in education, public services, and daily communication.

3) **Q:** Why did you choose continuous sign language recognition instead of isolated signs?  
**A:** Continuous signing is closer to real communication; isolated signs are simpler but less practical.

4) **Q:** What makes continuous sign language recognition difficult?  
**A:** Sign boundaries are implicit, temporal context is long, and signer style varies significantly.

5) **Q:** How is CSLR different from standard action recognition?  
**A:** CSLR requires sequence-level linguistic decoding (glosses), not just classifying one action clip.

6) **Q:** Why is accessibility framing important in your project motivation?  
**A:** It defines the practical impact and helps prioritize usable system behavior over only benchmark numbers.

7) **Q:** Who are the direct beneficiaries of your system?  
**A:** Deaf users, interpreters, educators, and institutions requiring assistive communication tools.

8) **Q:** What is the scope boundary of your project problem statement?  
**A:** Gloss-level recognition and demo-level bidirectional interaction; not full production-grade translation.

## B) Dataset and Data Understanding

9) **Q:** Why did you choose RWTH-PHOENIX-Weather 2014?  
**A:** It is a standard CSLR benchmark with accepted splits and strong prior work for comparison.

10) **Q:** What are the train/dev/test splits in PHOENIX?  
**A:** Roughly 5.6k train, 540 dev, and 629 test samples as defined by the official benchmark split.

11) **Q:** Why is PHOENIX considered a benchmark dataset?  
**A:** It is widely used in CSLR literature, enabling reproducibility and fair baseline comparison.

12) **Q:** What are gloss annotations and why are they useful?  
**A:** Glosses are tokenized sign labels; they provide a structured intermediate representation for sequence modeling.

13) **Q:** What are the limitations of the PHOENIX dataset?  
**A:** Domain bias (weather), limited spontaneity, and constrained signer/environment diversity.

14) **Q:** How does domain specificity (weather) affect generalization?  
**A:** Models may overfit domain vocabulary/patterns and perform worse on open-domain signing.

15) **Q:** Did you perform any dataset cleaning? If yes, what?  
**A:** Mostly standardized preprocessing and pipeline checks; no heavy manual relabeling.

16) **Q:** What kind of signer variability exists in the dataset?  
**A:** Differences in speed, articulation, body pose, and signing style across speakers.

17) **Q:** How does frame rate/resolution affect your model?  
**A:** They control temporal detail and visual clarity; too low harms motion cues, too high increases compute cost.

18) **Q:** What data preprocessing steps are applied before model input?  
**A:** Frame resizing, color conversion, normalization, and temporal buffering/windowing.

## C) Architecture and Design Choices

19) **Q:** Can you explain your architecture in one minute?  
**A:** ResNet-18 extracts visual features, Transformer models temporal context, and hybrid CTC+Attention decodes gloss sequences.

20) **Q:** Why did you use ResNet-18 as visual backbone?  
**A:** It offers a good tradeoff between feature quality and compute efficiency for limited hardware.

21) **Q:** Why not use a larger backbone like ResNet-50 or ViT?  
**A:** Resource constraints and training stability considerations made ResNet-18 the practical choice.

22) **Q:** What role does the Transformer encoder play in your pipeline?  
**A:** It captures long-range temporal dependencies across frame features.

23) **Q:** Why combine CTC and attention instead of using only one?  
**A:** CTC stabilizes alignment; attention improves contextual sequence decoding.

24) **Q:** What is the intuition behind CTC in sign language decoding?  
**A:** It learns monotonic alignment between variable-length input frames and output tokens.

25) **Q:** What is the intuition behind attention decoder in your setup?  
**A:** It models token dependencies and improves output coherence beyond frame-wise alignment.

26) **Q:** How do CTC and attention complement each other?  
**A:** CTC constrains alignment; attention refines linguistic sequence quality.

27) **Q:** What are the key hyperparameters you selected and why?  
**A:** Model dimension, layer counts, dropout, and CTC weight were tuned for stability and available compute.

28) **Q:** Why did you choose your final CTC/attention balance?  
**A:** Lower CTC emphasis reduced blank collapse while keeping alignment support.

29) **Q:** How does your model handle variable-length sequences?  
**A:** CTC naturally handles variable lengths, and Transformer processes padded sequences with masking.

30) **Q:** How is positional information handled in temporal modeling?  
**A:** Through Transformer positional encoding over temporal feature tokens.

## D) Training and Optimization

31) **Q:** What loss function(s) did you use in training?  
**A:** A hybrid objective combining CTC loss and attention decoding loss.

32) **Q:** What optimizer and learning rate schedule did you use?  
**A:** Adam-family optimization with warmup and decay scheduling for stable convergence.

33) **Q:** Did you use warmup, and why?  
**A:** Yes, warmup reduced early training instability and prevented divergence.

34) **Q:** How did you prevent overfitting?  
**A:** Dropout, augmentation, validation monitoring, and checkpoint selection.

35) **Q:** Did you use data augmentation? Which ones?  
**A:** Yes, moderate visual augmentations that preserve sign semantics.

36) **Q:** Why is horizontal flip usually risky in sign language tasks?  
**A:** It can invert meaningful handedness and change semantic interpretation.

37) **Q:** What batch size did you use and why?  
**A:** A small batch size due to GPU memory limits and high video model cost.

38) **Q:** How did hardware constraints affect training choices?  
**A:** They limited model scale and batch size, so stability-oriented design was prioritized.

39) **Q:** What training instability did you observe?  
**A:** Blank-dominant predictions and unstable decoding quality in pure CTC settings.

40) **Q:** How did you detect blank collapse behavior?  
**A:** Predictions became mostly blank tokens with poor gloss output across samples.

41) **Q:** What practical changes improved training stability most?  
**A:** Hybrid decoding, careful loss weighting, and safer optimization schedule.

## E) Evaluation and Results

42) **Q:** Why did you choose WER as the primary metric?  
**A:** WER is standard for sequence recognition and captures substitution/insertion/deletion errors.

43) **Q:** How should we interpret your reported WER value?  
**A:** It indicates moderate recognition quality and a meaningful baseline under constrained resources.

44) **Q:** Why is your WER not near SOTA?  
**A:** SOTA uses larger models, more pretraining, and heavier optimization than this project setup.

45) **Q:** What does your result still demonstrate despite higher WER?  
**A:** A complete working pipeline with justified architecture and demonstrable real-time behavior.

46) **Q:** Did you evaluate qualitative outputs besides numeric metrics?  
**A:** Yes, through live demo behavior and decoded gloss inspection.

47) **Q:** What are common error types in your predictions?  
**A:** Boundary errors, missed tokens, and occasional repeated or substituted glosses.

48) **Q:** Are substitutions, insertions, or deletions more frequent?  
**A:** Deletions and substitutions are typically dominant in this setup.

49) **Q:** How robust is the model to signer style changes?  
**A:** Partially robust, but still sensitive to style and speed variation.

50) **Q:** How would you improve evaluation rigor in future work?  
**A:** Add signer-wise breakdowns, error analysis reports, and ablation studies.

## F) Demo and System Integration

51) **Q:** What are the two directions supported in your bidirectional demo?  
**A:** Sign->Text recognition and Text->Sign retrieval-based visualization.

52) **Q:** How does Sign -> Text flow work in your desktop app?  
**A:** Webcam frames are buffered, preprocessed, decoded by hybrid model, then shown as gloss and English text.

53) **Q:** How does Text -> Sign flow work in your desktop app?  
**A:** German text is mapped to glosses (rule-based), then clips are retrieved and played as sign sequence.

54) **Q:** Is the text-to-sign side generative or retrieval-based?  
**A:** Retrieval-based; it composes existing sign clips rather than generating novel signer motion.

55) **Q:** Why did you choose retrieval-based text-to-sign for demo?  
**A:** It is practical, faster to build, and robust for prototype demonstration.

56) **Q:** What are the practical limitations of retrieval-based synthesis?  
**A:** Limited smoothness, limited vocabulary flexibility, and dependence on indexed clips.

57) **Q:** How does auto/streaming decode improve usability?  
**A:** It avoids manual button dependence and updates output continuously for demo realism.

58) **Q:** Why did the manual decode mode exist initially?  
**A:** It simplified debugging and ensured deterministic checks during early validation.

59) **Q:** What causes live demo glitches in Tkinter/OpenCV style apps?  
**A:** UI-thread overload, frequent redraws, and thread-unsafe image update paths.

60) **Q:** What fixes did you apply to stabilize the demo UI?  
**A:** Main-thread rendering, fixed preview layout, throttled refresh, and safer decode scheduling.

61) **Q:** Why do you use a recent-frame decode window instead of full buffer?  
**A:** It reduces stale context and improves relevance of current signing segment.

## G) Practical Deployment and Ethics

62) **Q:** Can this system be deployed in hospitals/public offices today?  
**A:** Not yet as-is; it needs stronger accuracy, robustness, and compliance hardening.

63) **Q:** What is needed before real-world deployment?  
**A:** Better generalization, low-latency optimization, safety checks, and user-centered evaluation.

64) **Q:** What latency constraints matter in assistive use?  
**A:** Low response delay is critical; interaction should feel near real-time.

65) **Q:** What reliability threshold is acceptable for practical use?  
**A:** It depends on use case, but significantly higher than current WER for high-stakes settings.

66) **Q:** What ethical risks exist in sign language AI systems?  
**A:** Misinterpretation risk, exclusion of dialects, and overtrust in imperfect systems.

67) **Q:** How can model bias affect deaf users?  
**A:** Uneven performance across signer groups can create unfair accessibility outcomes.

68) **Q:** What privacy concerns exist with camera-based systems?  
**A:** Sensitive visual data capture requires strict consent, storage policy, and secure processing.

69) **Q:** How would you make the system more inclusive across dialects?  
**A:** Train with diverse signer/dialect data and evaluate subgroup-level performance explicitly.

## H) Comparison, Alternatives, and Future Scope

70) **Q:** Why not use pure CTC with language model rescoring only?  
**A:** It helps, but hybrid directly improves sequence modeling during learning, not only post-processing.

71) **Q:** Why not train a full encoder-decoder without CTC?  
**A:** Pure attention can be less stable in alignment-heavy continuous signing scenarios.

72) **Q:** How would a pretrained video-language model help this task?  
**A:** Better transferable visual-temporal representations can improve accuracy and robustness.

73) **Q:** Would a signer-adaptive module likely improve performance?  
**A:** Yes, adaptation can reduce style variance effects and improve personalization.

74) **Q:** How would you improve gloss-to-text quality further?  
**A:** Add a stronger language model or dedicated gloss-to-text translation model.

75) **Q:** What is your plan to move beyond weather-domain constraints?  
**A:** Expand training data to broader domains and perform domain-adaptive fine-tuning.

76) **Q:** What is the highest-impact next experiment you would run?  
**A:** Stronger pretrained backbone plus controlled ablation of hybrid loss weights.

77) **Q:** If given 3 more months, what would you prioritize?  
**A:** Generalization improvements, richer evaluation, and smoother text-to-sign generation.

78) **Q:** If given 1 GPU week, what would you prioritize?  
**A:** Targeted fine-tuning and ablations to maximize WER gain per compute hour.

## I) Thesis/Presentation-Specific Questions

79) **Q:** What is your key technical contribution in this project?  
**A:** A practical hybrid CTC+Attention CSLR setup integrated into an end-to-end system.

80) **Q:** What is your key engineering contribution in this project?  
**A:** Building and stabilizing a usable bidirectional desktop demo from model to interface.

81) **Q:** Which result are you most confident about and why?  
**A:** The full pipeline integration and consistent hybrid behavior improvements in practical decoding.

82) **Q:** Which claim in your presentation is most limited by data?  
**A:** Cross-domain generalization, because training data is weather-domain focused.

83) **Q:** What is one design decision you would change if restarting?  
**A:** I would start with stronger pretraining and systematic ablation from the beginning.

84) **Q:** What are your project assumptions that examiners should note?  
**A:** Gloss supervision quality, benchmark split validity, and constrained compute environment.

85) **Q:** What are the top 3 takeaways from your work?  
**A:** Hybrid decoding is practical, end-to-end integration matters, and usability requires robust engineering.

86) **Q:** What is your one-line conclusion for this thesis?  
**A:** This project delivers a practical CSLR baseline with clear technical rationale and a realistic path for improvement.

---

## Quick Revision Tip

Use a 3-layer answer strategy:
- **Layer 1 (10 sec):** direct answer
- **Layer 2 (20 sec):** technical reason
- **Layer 3 (optional):** practical implication

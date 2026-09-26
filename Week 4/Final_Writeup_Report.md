# Week 3 & 4 Update: Tracking bias under continuous pressure and multi-layer ablation with the Jacobian Lens

**TL;DR.** I measured how a model's (Qwen3.5-4B) internal bias evolves when placed under continuous adversarial pressure, and then used targeted latent ablations to casually map where this bias lives. The most interesting result: adversarial pressure forces the model to lock in its biased decisions earlier in the network (shifting from Layer 28 to Layer 24). Furthermore, ablation proved that early layers are purely abstract context extractors; the bias only materializes as steerable vectors in the late-layer "Spike Zone." Finally, whole-subspace ablation proved significantly more effective than single-token ablation at steering the model without breaking grammatical fluency.

**Background.** In Week 2, we successfully scaled our correlational baseline across languages and architectures, finding that stereotype-congruent preferences peak during ambiguous contexts in English (Qwen-2B), Chinese (CBBQ), and Indic languages (Sarvam-1). However, these static evaluations left two gaps: they didn't show how bias reacts dynamically to worsening context, and they were purely correlational. Week 3 and 4 attack both gaps by introducing a continuous pressure environment and causal ablation.

**Notebook 1 — Detection under Continuous Pressure.** I built a 3-turn adversarial context (e.g., arguing "nurse" and "doctor") that increasingly nudged the model toward relying on stereotypical gender biases. During each forward pass, I tracked the Bias Differential `P(she) - P(he)` across all layers using the Jacobian Lens. 

*Findings:*
*   **The Decision Layer shifts under pressure:** In a subtle, neutral context, the bias doesn't solidify until very late in the network (Layer 28). However, under heavy stereotyping pressure, the model's guardrails collapse early. By Layer 24, the probability of generating a biased pronoun locks in at over 80%.
*   **Mathematical evidence of premature resolution:** The J-Lens provides direct proof that as adversarial modifiers increase, the network stops waiting for final disambiguation and preemptively resolves the pronoun earlier in the stack. 
*   **Multi-lingual confirmation:** Building on Week 2, we know this late-layer ambiguous spike isn't just an English artifact—it consistently surfaced in our CBBQ and Indic-Bias runs as well.

**Notebook 2 — Intervention via Targeted Latent Ablation.** To determine exactly how the bias travels, I performed Targeted Latent Ablations. By intercepting the forward pass, I mathematically subtracted token vectors from the residual stream at isolated layer bands (Early: 0-10, Mid: 12-20, Late: 22-30).

*Findings:*
*   **Early & Mid layers are untouchable:** Ablating the concept of "she" in the early and middle layers had absolutely zero effect on the final output. This proves these layers act purely as *abstract contextual feature extractors*.
*   **The Hit (Late Layers):** Ablating the vector precisely in the Layer 22-30 "Spike Zone" successfully dropped the output probability of the biased pronoun.
*   **Grammatical Overlap:** Ablating the counter-stereotype token (`he`) unexpectedly dragged `P(she)` down by more than half, proving that gender pronouns share dense, overlapping grammatical components in the late-layer subspace. Disrupting one disrupts the whole class.
*   **The Subspace Breakthrough:** Ablating the *difference vector* (`she` - `he`) yielded the holy grail. Under explicit pressure, it dropped `P(she)` from 44.6% to 7.9% while causing `P(he)` to skyrocket to 48.8%. We successfully inverted the gender axis without breaking fluency!
*   **Controls Passed:** A random orthogonal control vector had zero effect, confirming our ablations were surgically precise.

**Reproducibility.** The analysis builds on the J-Lens core library, evaluating `Qwen3.5-4B` for the ablation/pressure tests (running on CPU/GPU via auto device mapping), and `Qwen3.5-2B` / `Sarvam-1` for the cross-lingual baselines. 

**Big picture.** Bias is not a static switch you can flip at any layer, nor is it a single token. It dynamically shifts its manifestation layer based on contextual pressure. Because early layers hold concepts abstractly, debiasing interventions must target the late-layer multi-dimensional subspace (e.g., via difference-vector ablation) rather than hunting for single tokens early in the residual stream.

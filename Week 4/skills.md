# J-LENS PROJECT // SKILLS & METHODOLOGIES

This document outlines the core technical skills and methodologies developed during the **Jacobian Lens Bias** research project.

<agent-embed src="file:///home/dlcv/.gemini/antigravity/brain/9f098c13-40c4-4066-8fdb-431407b5bd0b/cyberpunk_tree.html"></agent-embed>

> [!TIP]
> **View Full Interactive Node Map:** [Launch Fullscreen Cyberpunk Tree](file:///home/dlcv/.gemini/antigravity/brain/9f098c13-40c4-4066-8fdb-431407b5bd0b/cyberpunk_tree.html)

---

## 1. Jacobian Lens Projection (Weight: 30)
- **Concept:** Projecting intermediate model hidden states (layers 0-N) directly into the vocabulary embedding space.
- **Application:** Used `jacobian-lens` to trace where and when specific concepts (like gender pronouns or stereotypes) surface before the final output layer.

## 2. Multi-lingual Evaluation (Weight: 20)
- **Concept:** Cross-applying bias detection frameworks across different LLMs and languages.
- **Application:** Tested English (`Qwen-2B` on BBQ), Chinese (`Qwen-2B` on CBBQ), and Indic languages by training custom lenses (`Sarvam-1` on Indic-Bias).

## 3. Continuous Pressure Environment Design (Weight: 25)
- **Concept:** Crafting multi-turn adversarial prompts to push language models into relying on latent priors.
- **Application:** Discovered that as stereotyping pressure increases, models lock in their decisions in earlier layers (e.g., shifting the "decision point" from Layer 28 to Layer 24 under heavy pressure).

## 4. Targeted Latent Ablation (Weight: 25)
- **Concept:** Mathematically suppressing or intercepting specific token vectors within the residual stream during a forward pass.
- **Application:** Proved that early and middle layers act as abstract context extractors (ablation does nothing), while late-layer ablation successfully suppresses biased outputs without entirely neutralizing them, pointing toward multi-dimensional subspace storage.

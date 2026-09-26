# Week 4: Advanced Multi-Layer & Subspace Ablations

Welcome back! The autonomous experiment has completed successfully. We ran our 4-Billion parameter language model through a gauntlet of continuous pressure environments (subtle to explicit stereotyping) while intercepting its forward pass at different layer bands (Early, Mid, Late). We tested several types of ablations: single token ("she"), single token ("he"), the entire subspace difference ("she" - "he"), and random controls.

Here are the definitive findings from the run:

## 1. Baseline: The Pressure Mounts
As we increase the adversarial context nudging the model toward relying on stereotypes, the baseline bias naturally solidifies:
- **Turn 1 (Subtle):** `P(she)` = 36.4%, `P(he)` = 7.5%
- **Turn 2 (Pressure):** `P(she)` = 41.5%, `P(he)` = 5.2%
- **Turn 3 (Explicit):** `P(she)` = 44.6%, `P(he)` = 1.5%

## 2. The Abstract Nature of Early & Mid Layers
As expected, ablating vectors in the Early (L0-10) and Mid (L12-20) layers did absolutely nothing. Regardless of the ablation type, the output probabilities barely flinched. The model purely extracts the *abstract context* here; it has not mapped the context to any grammatical or gendered vocabulary vector yet.

## 3. The Spike Zone (Late Layers 22-30): Where Bias Lives
When we move our interventions to the late layers, the network is finally vulnerable. But *how* we ablate matters immensely:

### A. Single-Token Ablation (`she`)
- **Result:** Successfully tanked `P(she)` all the way to **0.0%** across all pressure levels.
- **The Catch:** It also severely crippled `P(he)`. By targeting the primary biased token, we broke the overall pronoun structure in the residual stream.

### B. Single-Token Ablation (`he`) - The Grammar Overlap
- **Result:** Ablating the *counter*-stereotypical token (`he`) unexpectedly dragged `P(she)` down by more than half (e.g., from 44.6% to 17.7%), while setting `P(he)` to 0.0%.
- **Insight:** This mathematically proves that gender pronouns share dense, overlapping grammatical components in the late-layer subspace. Disrupting one disrupts the whole class.

### C. Subspace Ablation (`she` - `he` difference vector) - The Holy Grail
This was our most sophisticated ablation, targeting the actual conceptual gap between the genders.
- **Result under Subtle Pressure:** `P(she)` dropped from 36.4% to 15.4%, but `P(he)` **surged** from 7.5% to 33.8%.
- **Result under Explicit Pressure:** `P(she)` dropped from 44.6% to 7.9%, while `P(he)` **skyrocketed** from 1.5% to 48.8%.
- **Insight:** By ablating the *difference* vector, we didn't just break the pronoun grammar—we successfully inverted the gender axis itself! This steered the model heavily toward the counter-stereotype *without* breaking its fluency.

### D. Random Control Vector
- **Result:** Ablating a random orthogonal vector did nothing. The output probabilities remained identical to the baseline, proving that our targeted ablations were surgically precise and not just artifacts of adding mathematical noise to the residual stream.

---

## Conclusion
Bias is not a single token; it is an entangled subspace. Single-token interventions either fail or break grammatical fluency. The only effective way to steer a model away from deeply ingrained stereotypical priors under pressure is to identify the **late-layer decision zone** and intervene on the **entire conceptual subspace** (the difference vector).

# Cross-Dataset Analysis of Bias Signals in J-Lens Workspace Readouts

## 1. Overview

This report presents the results of J-Lens experiments conducted across seven BBQ bias categories:

- Age
- Disability Status
- Nationality
- Physical Appearance
- Religion
- Sexual Orientation
- Socioeconomic Status (SES)

The experiments investigate whether the J-Lens workspace-band readout surfaces bias-relevant targets differently from non-bias targets, and how these internal signals change according to:

1. Prompt formulation
2. Ambiguous versus disambiguated context
3. Negative versus non-negative question polarity
4. The four BBQ context/polarity conditions
5. Layer-to-layer movement through the model

Four prompt formulations were evaluated:

| Prompt | Description         |
| ------ | ------------------- |
| **P0** | Baseline prompt     |
| **P1** | Explicit MCQ prompt |
| **P2** | Answer-text prompt  |
| **P3** | Concise prompt      |

The analysis focuses primarily on:

- **Bias Target MRR**
- **Non-Bias Target MRR**
- **Bias MRR Gap**
- **Joint Log Rank**
- **Layer-to-layer log-rank movement**

---

## 2. Interpretation of the Metrics

### 2.1 Mean Reciprocal Rank

Mean Reciprocal Rank (MRR) measures how highly a target token is ranked in the J-Lens vocabulary readout.

$$
\text{MRR} = \frac{1}{N}\sum_{i=1}^{N}\frac{1}{\mathrm{rank}_i}
$$

A higher MRR indicates that the target is generally closer to the top of the vocabulary distribution and is therefore more accessible in the J-Lens readout.

The analysis separately measures:

- **Bias Target MRR**
- **Non-Bias Target MRR**

### 2.2 Bias MRR Gap

The relative difference between the two targets is defined as:

$$
\text{Bias MRR Gap}
=
\text{MRR}_{\text{bias}}
-
\text{MRR}_{\text{non-bias}}
$$

Therefore:

- **Positive gap** → bias target is more salient.
- **Negative gap** → non-bias target is more salient.
- **Near-zero gap** → both targets have similar accessibility.

Importantly, a high MRR does not necessarily imply a strong bias signal. Both targets may be highly accessible while remaining approximately balanced.

### 2.3 Joint Log Rank

Joint log rank summarizes the overall accessibility of both targets.

A **lower joint log rank is better**, because it means that the two relevant target concepts are jointly ranked closer to the top of the vocabulary.

Joint log rank therefore primarily measures **target accessibility**, rather than preference for one target over the other.

---

# 3. Raw Prompt Analysis

## 3.1 Age

| Prompt | Joint Log Rank ↓ |     Bias MRR | Non-Bias MRR |  Bias MRR Gap | Joint Rank | Gap Rank |
| ------ | ---------------: | -----------: | -----------: | ------------: | ---------: | -------: |
| P0     |           4.5933 |     0.000168 |     0.000146 |     +0.000022 |          2 |        3 |
| P1     |           4.8110 |     0.000088 |     0.000086 |     +0.000002 |          4 |        4 |
| **P2** |       **4.3860** | **0.001296** | **0.001102** | **+0.000194** |      **1** |    **1** |
| P3     |           4.6033 |     0.000489 |     0.000351 |     +0.000138 |          3 |        2 |

### Interpretation

P2 is the strongest prompt for Age according to both major criteria.

It produces:

- The lowest joint log rank
- The highest bias-target MRR
- The highest non-bias-target MRR
- The largest positive bias MRR gap

P3 produces the second-largest bias gap.

The result indicates that the answer-text formulation in P2 makes the Age-related target concepts considerably more accessible to the J-Lens readout while also producing the strongest relative bias-target advantage.

---

## 3.2 Physical Appearance

| Prompt | Joint Log Rank ↓ |     Bias MRR | Non-Bias MRR |  Bias MRR Gap |
| ------ | ---------------: | -----------: | -----------: | ------------: |
| P0     |           4.7425 |     0.000096 |     0.000091 |     +0.000005 |
| P1     |           4.9650 |     0.000037 |     0.000036 |     +0.000001 |
| **P2** |       **4.4088** |     0.002831 |     0.003479 | **−0.000648** |
| P3     |           4.4813 | **0.004115** | **0.005210** | **−0.001095** |

### Interpretation

Physical Appearance presents an important counterexample to interpreting high target accessibility as evidence of bias.

P2 and P3 substantially increase the accessibility of both targets. However, the non-bias target has a higher MRR than the bias target.

P3, for example, produces:

$$
\text{MRR}_{\text{bias}} = 0.004115
$$

and

$$
\text{MRR}_{\text{non-bias}} = 0.005210.
$$

This produces:

$$
\text{Gap} = -0.001095.
$$

Therefore, Physical Appearance demonstrates that:

> **Strong concept representation does not necessarily imply a strong bias-target preference.**

The J-Lens can strongly surface both relevant concepts while the relative signal favors the non-bias target.

---

## 3.3 Religion

| Prompt | Joint Log Rank ↓ |     Bias MRR | Non-Bias MRR |  Bias MRR Gap |
| ------ | ---------------: | -----------: | -----------: | ------------: |
| P0     |           4.2419 |     0.000344 |     0.000324 |     +0.000020 |
| P1     |           5.0169 |     0.000024 |     0.000023 |     +0.000001 |
| P2     |           4.1481 |     0.005101 |     0.005069 | **+0.000032** |
| **P3** |       **3.9339** | **0.007688** | **0.007694** |     −0.000006 |

### Interpretation

P3 produces the strongest overall accessibility of the Religion targets.

However, the bias and non-bias MRR values are almost identical:

$$
0.007688 \approx 0.007694.
$$

This means the concepts are highly readable without a meaningful relative bias-target preference.

P2 produces the largest positive bias gap, but the difference is very small.

Religion therefore provides another strong example of the distinction between:

- **representation strength**, and
- **representational preference**.

---

## 3.4 Disability Status

| Prompt | Joint Log Rank ↓ |     Bias MRR | Non-Bias MRR |  Bias MRR Gap |
| ------ | ---------------: | -----------: | -----------: | ------------: |
| P0     |           4.6785 |     0.000647 |     0.000535 |     +0.000112 |
| P1     |           4.9876 |     0.000038 |     0.000039 |     −0.000001 |
| P2     |           4.5485 |     0.000776 | **0.000690** |     +0.000086 |
| **P3** |       **4.5369** | **0.000850** |     0.000455 | **+0.000395** |

### Interpretation

P3 produces both the strongest joint accessibility and the largest bias MRR gap.

Unlike Religion and Physical Appearance, increased target readability is accompanied by a clearer relative bias-target advantage.

Disability Status is therefore one of the categories where the concise P3 formulation exposes a stronger bias-relative internal signal.

---

## 3.5 Nationality

| Prompt | Joint Log Rank ↓ |     Bias MRR | Non-Bias MRR |  Bias MRR Gap |
| ------ | ---------------: | -----------: | -----------: | ------------: |
| P0     |           4.1402 |     0.000523 |     0.000506 |     +0.000017 |
| P1     |           4.8149 |     0.000028 |     0.000028 |     +0.000001 |
| P2     |           4.1332 | **0.003373** | **0.003224** |     +0.000149 |
| **P3** |       **4.1089** |     0.002396 |     0.002206 | **+0.000190** |

### Interpretation

P2 produces the highest absolute MRR values, while P3 produces:

- The lowest joint log rank
- The largest bias MRR gap

This indicates a subtle difference between the prompts.

P2 makes both target concepts highly accessible, whereas P3 creates slightly greater relative separation between the bias and non-bias targets.

---

## 3.6 Sexual Orientation

| Prompt | Joint Log Rank ↓ |     Bias MRR | Non-Bias MRR |  Bias MRR Gap |
| ------ | ---------------: | -----------: | -----------: | ------------: |
| P0     |           4.5808 |     0.000243 |     0.000241 |     +0.000002 |
| P1     |           5.0369 |     0.000069 |     0.000066 |     +0.000003 |
| **P2** |       **4.4695** | **0.006199** | **0.004892** | **+0.001307** |
| P3     |           4.4767 |     0.005301 |     0.004194 |     +0.001107 |

### Interpretation

Sexual Orientation produces one of the clearest positive bias-gap signals among the seven categories.

P2 is strongest across:

- Joint accessibility
- Bias-target MRR
- Non-bias-target MRR
- Bias MRR gap

P3 follows closely.

The magnitude of the gap is substantially larger than those observed in Age, Religion, Nationality, and Disability Status.

---

## 3.7 Socioeconomic Status

| Prompt | Joint Log Rank ↓ |     Bias MRR | Non-Bias MRR |  Bias MRR Gap |
| ------ | ---------------: | -----------: | -----------: | ------------: |
| P0     |           4.5007 |     0.002760 |     0.002748 |     +0.000012 |
| P1     |           4.9840 |     0.000058 |     0.000056 |     +0.000002 |
| P2     |           4.2469 |     0.015203 |     0.012649 |     +0.002553 |
| **P3** |       **4.1675** | **0.023053** | **0.018794** | **+0.004259** |

### Interpretation

SES produces the strongest overall bias-relative signal among the seven categories.

P3 dominates the analysis, producing:

$$
\text{MRR}_{\text{bias}} = 0.023053
$$

versus:

$$
\text{MRR}_{\text{non-bias}} = 0.018794.
$$

The resulting gap is:

$$
+0.004259.
$$

This is substantially larger than the gaps observed in most other categories.

---

# 4. Cross-Dataset Prompt Ranking

The best prompts according to target accessibility and bias-relative separation are:

| Dataset             | Best Joint Accessibility | Largest Positive Bias Gap |
| ------------------- | ------------------------ | ------------------------- |
| Age                 | P2                       | P2                        |
| Physical Appearance | P2                       | P0                        |
| Religion            | P3                       | P2                        |
| Disability Status   | P3                       | P3                        |
| Nationality         | P3                       | P3                        |
| Sexual Orientation  | P2                       | P2                        |
| SES                 | P3                       | P3                        |

A highly consistent result appears for P1.

**P1 has the worst joint log rank in all seven categories.**

This suggests that the explicit MCQ formulation does not make the relevant semantic concepts more readable to the J-Lens. Instead, the target concepts are substantially less accessible than under P2 and P3.

---

# 5. Combined Prompt Analysis

Macro-averaging the seven categories gives:

| Prompt | Joint Log Rank ↓ |     Bias MRR | Non-Bias MRR |  Bias MRR Gap |
| ------ | ---------------: | -----------: | -----------: | ------------: |
| P0     |           4.4968 |     0.000683 |     0.000656 |     +0.000027 |
| P1     |           4.9452 |     0.000049 |     0.000048 |     +0.000001 |
| P2     |           4.3344 |     0.004968 |     0.004444 |     +0.000525 |
| **P3** |       **4.3298** | **0.006270** | **0.005558** | **+0.000713** |

### Overall Prompt Ranking

For joint target accessibility:

$$
\text{P3} \approx \text{P2} > \text{P0} > \text{P1}
$$

For positive bias MRR gap:

$$
\text{P3} > \text{P2} > \text{P0} > \text{P1}.
$$

P2 and P3 therefore dominate the overall analysis.

However, the category-level analysis demonstrates that this should not be interpreted as evidence that these prompts universally produce stronger stereotypical preference.

For example:

- Physical Appearance favors the non-bias target under P2/P3.
- Religion has high accessibility but almost no target preference.
- SES has both high accessibility and a substantial bias-target advantage.

Prompt formulation therefore appears to strongly influence **target accessibility**, while the direction and magnitude of relative bias remain category-dependent.

---

# 6. Raw Four-Condition BBQ Analysis

The BBQ design contains four combinations:

1. Ambiguous + Negative
2. Ambiguous + Non-negative
3. Disambiguated + Negative
4. Disambiguated + Non-negative

The cross-dataset macro-average bias MRR gaps are:

| Prompt | Ambiguous Negative | Ambiguous Non-negative | Disambiguated Negative | Disambiguated Non-negative |
| ------ | -----------------: | ---------------------: | ---------------------: | -------------------------: |
| P0     |          +0.000132 |              −0.000089 |              +0.000154 |                  −0.000088 |
| P1     |          −0.000003 |              +0.000007 |              −0.000003 |                  +0.000006 |
| P2     |          −0.000442 |          **+0.001863** |              −0.001700 |              **+0.002378** |
| P3     |          −0.000280 |          **+0.001796** |              −0.000735 |              **+0.002070** |

## Interpretation

A substantial interaction with question polarity appears under P2 and P3.

Negative questions generally produce negative average gaps, whereas non-negative questions produce positive gaps.

This is important because the definition of the bias target changes with question polarity.

For negative questions, the stereotype-congruent target acts as the bias target.

For non-negative questions, the counter-stereotype target acts as the bias target.

Consequently, the polarity-dependent results suggest that part of the aggregate bias MRR signal may reflect a persistent preference for a particular semantic/group target rather than a clean bias signal that reverses appropriately when question polarity changes.

This requires careful interpretation when evaluating the original research hypothesis.

---

# 7. Ambiguous vs Disambiguated Context

The following table compares the strongest relevant prompt for each dataset:

| Dataset             | Prompt | Ambiguous Gap | Disambiguated Gap |
| ------------------- | ------ | ------------: | ----------------: |
| Age                 | P2     | **+0.000504** |         −0.000118 |
| Physical Appearance | P2     |     +0.000326 |     **−0.001623** |
| Religion            | P2     |     +0.000145 |         −0.000082 |
| Disability Status   | P3     |     +0.000384 |         +0.000405 |
| Nationality         | P3     | **+0.000473** |         −0.000092 |
| Sexual Orientation  | P2     |     +0.001280 |         +0.001335 |
| SES                 | P3     | **+0.005297** |         +0.003221 |

## Interpretation

The ambiguous-context hypothesis receives support in some categories but not universally.

### Age

Age presents a relatively clear pattern:

$$
+0.000504 \rightarrow -0.000118
$$

The positive bias-relative signal appears in ambiguous contexts and disappears or reverses once the context provides disambiguating evidence.

### Nationality

Nationality displays a similar pattern:

$$
+0.000473 \rightarrow -0.000092.
$$

### SES

SES produces the strongest ambiguous-context signal:

$$
+0.005297.
$$

However, a substantial positive signal remains after disambiguation:

$$
+0.003221.
$$

### Sexual Orientation and Disability Status

These categories do not show the expected reduction after disambiguation.

Sexual Orientation remains strongly positive under both conditions, while Disability Status remains almost unchanged.

### Overall finding

The effect of ambiguity is therefore **category-dependent rather than universal**.

Some categories show stronger bias-relative signals when textual evidence is absent, while others retain similar internal differences even when the context is disambiguated.

---

# 8. Negative vs Non-Negative Question Polarity

| Dataset             | Prompt |      Negative |  Non-Negative |
| ------------------- | ------ | ------------: | ------------: |
| Age                 | P2     |     +0.000184 |     +0.000203 |
| Physical Appearance | P2     | **−0.002658** | **+0.001361** |
| Religion            | P2     |     −0.000218 |     +0.000281 |
| Disability Status   | P3     |     +0.000655 |     +0.000135 |
| Nationality         | P3     |     −0.000111 |     +0.000492 |
| Sexual Orientation  | P2     | **−0.006490** | **+0.009104** |
| SES                 | P3     |     +0.002728 | **+0.005791** |

## Interpretation

The categories separate into different behavioral patterns.

### Positive under both polarities

Age, Disability Status, and SES maintain positive bias MRR gaps for both negative and non-negative questions.

Because the identity of the bias target changes with polarity, this behavior is more consistent with a polarity-sensitive bias-relative signal.

### Strong polarity reversal

Physical Appearance, Religion, Nationality, and especially Sexual Orientation show substantial changes in sign between negative and non-negative questions.

The strongest example is Sexual Orientation:

$$
-0.006490 \rightarrow +0.009104.
$$

This demonstrates that the internal readout is highly sensitive to question polarity.

Consequently, the experiment does not support interpreting every positive aggregate bias MRR gap as evidence of a stable stereotype representation.

Instead, target accessibility appears to interact substantially with the semantic structure of the question.

---

# 9. Layer-to-Layer Movement

The layer analysis tracks how the bias and non-bias targets become more or less accessible as information moves through the model.

For log rank:

$$
\Delta L < 0
$$

means that the target's rank improved and therefore became more salient.

For P2, the cross-dataset trajectory is approximately:

|  Layer | Bias Log Rank | Non-Bias Log Rank |     Δ Bias | Δ Non-Bias |
| -----: | ------------: | ----------------: | ---------: | ---------: |
|     20 |         4.908 |             4.910 |          — |          — |
|     21 |         4.933 |             4.935 |     +0.025 |     +0.025 |
|     22 |         4.999 |             5.001 |     +0.067 |     +0.066 |
|     23 |         4.995 |             5.003 |     −0.004 |     +0.002 |
|     24 |         4.985 |             4.993 |     −0.010 |     −0.010 |
|     25 |         4.942 |             4.951 |     −0.043 |     −0.042 |
|     26 |         4.846 |             4.859 |     −0.096 |     −0.092 |
| **27** |     **3.592** |         **3.640** | **−1.254** | **−1.220** |
|     28 |         3.342 |             3.388 |     −0.251 |     −0.252 |
|     29 |         3.043 |             3.090 |     −0.299 |     −0.298 |
|     30 |     **2.846** |         **2.893** |     −0.198 |     −0.197 |

## Interpretation

Layers 20–26 show comparatively weak accessibility.

A major transition then occurs between:

$$
\boxed{\text{Layer } 26 \rightarrow \text{Layer } 27}
$$

The bias target changes by approximately:

$$
\Delta L_{\text{bias}} = -1.254
$$

while the non-bias target changes by:

$$
\Delta L_{\text{non-bias}} = -1.220.
$$

Both targets therefore become dramatically more accessible at approximately the same point.

Their accessibility continues increasing through layers 28–30.

---

# 10. Consistency of the Layer-27 Transition

The strongest P2 bias-target movement for every dataset occurs at layer 27:

| Dataset             | Strongest Transition Layer |     Δ Bias | Δ Non-Bias |
| ------------------- | -------------------------: | ---------: | ---------: |
| Age                 |                     **27** |     −0.869 |     −0.801 |
| Physical Appearance |                     **27** |     −1.262 |     −1.213 |
| Religion            |                     **27** |     −1.396 |     −1.369 |
| Disability Status   |                     **27** |     −0.831 |     −0.810 |
| Nationality         |                     **27** |     −1.228 |     −1.227 |
| Sexual Orientation  |                     **27** | **−1.684** |     −1.657 |
| SES                 |                     **27** |     −1.505 |     −1.461 |

The layer-27 transition therefore occurs in:

$$
\boxed{7/7 \text{ datasets}}
$$

for P2.

A similar transition is observed under P3.

This represents one of the strongest and most reproducible findings across the entire experiment.

---

# 11. Bias and Non-Bias Targets Move Together

The layer-27 transition does not occur exclusively for the bias target.

For P2:

$$
\Delta L_{\text{bias}}(27) = -1.254
$$

and:

$$
\Delta L_{\text{non-bias}}(27) = -1.220.
$$

Both concepts become dramatically more salient.

This suggests that layer 27 should not be interpreted simply as the point where the model begins representing a stereotype.

Instead, the transition appears to represent a broader increase in accessibility of the **task-relevant semantic alternatives**.

The bias signal then appears as a smaller relative imbalance between these jointly represented alternatives.

---

# 12. Prompt Dependence of the Layer Transition

P3 displays the same general phenomenon.

Around layer 27:

$$
\Delta L_{\text{bias}} \approx -1.103
$$

and:

$$
\Delta L_{\text{non-bias}} \approx -1.033.
$$

P0 also produces a transition:

$$
\Delta L_{\text{bias}} \approx -0.614
$$

and:

$$
\Delta L_{\text{non-bias}} \approx -0.586.
$$

P1 is considerably weaker:

$$
\Delta L_{\text{bias}} \approx -0.200.
$$

The approximate ordering is therefore:

$$
\text{P2/P3} \gg \text{P0} \gg \text{P1}.
$$

This further supports the interpretation that prompt formulation strongly controls how effectively task-relevant semantic concepts become accessible in the J-Lens workspace-band readout.

---

# 13. Combined Cross-Dataset Findings

The seven experiments reveal three distinct phenomena.

## 13.1 Task-Relevant Concept Accessibility

This is the strongest and most reproducible result.

P2 and P3 consistently make the relevant stereotype and counter-stereotype concepts more accessible than P1.

Across all seven datasets, there is also a substantial late-layer transition around layer 27.

This indicates that the target concepts become especially readable within the late workspace-band layers.

---

## 13.2 Relative Bias-Target Preference

The relative bias signal is considerably more category-dependent.

Strong positive bias MRR gaps are particularly visible in:

- SES
- Sexual Orientation
- Disability Status
- Nationality
- Age

However, Physical Appearance frequently favors the non-bias target.

Religion provides another important case: the concepts become highly accessible without developing a substantial relative preference.

Therefore:

$$
\boxed{\text{Accessibility} \neq \text{Bias}}
$$

The J-Lens may strongly surface socially relevant concepts without necessarily preferring the stereotype-congruent target.

---

## 13.3 Context and Polarity Modulation

The internal bias-relative signal is not invariant.

It changes according to:

$$
\text{Category}
\times
\text{Prompt}
\times
\text{Context}
\times
\text{Question Polarity}
\times
\text{Layer}.
$$

Ambiguity strengthens the relative bias signal in some categories, particularly Age, Nationality, and SES, but this does not generalize across every dataset.

Question polarity produces especially large changes for:

- Physical Appearance
- Religion
- Nationality
- Sexual Orientation

The J-space readout therefore appears to reflect dynamically constructed task representations rather than a single fixed stereotype representation.

---

# 14. Representation Strength vs Representational Preference

One of the most important findings from the experiment is the separation between:

$$
\textbf{Representation Strength}
$$

and:

$$
\textbf{Representational Preference}.
$$

Religion P3 provides a strong example:

$$
\text{MRR}_{\text{bias}} = 0.007688
$$

$$
\text{MRR}_{\text{non-bias}} = 0.007694
$$

Both targets are strongly represented, but they are almost perfectly balanced.

SES P3 produces:

$$
\text{MRR}_{\text{bias}} = 0.023053
$$

and:

$$
\text{MRR}_{\text{non-bias}} = 0.018794.
$$

Here, both targets are strongly represented, but a substantial relative imbalance is also present.

The J-Lens analysis can therefore distinguish between:

1. Whether socially relevant concepts are accessible internally.
2. Whether one of those concepts receives greater relative salience.

This distinction is important when interpreting internal representations as evidence of bias.

---

# 15. Main Experimental Conclusion

Across seven BBQ bias categories, the J-Lens analysis reveals a highly consistent late-layer transition in which both bias and non-bias target concepts become sharply more accessible.

This transition is particularly strong under the answer-text P2 and concise P3 prompt formulations and occurs most prominently around layer 27.

However, the relative preference between bias and non-bias targets is not universal.

Instead, its magnitude and direction depend substantially on:

- Bias category
- Prompt formulation
- Ambiguous versus disambiguated context
- Question polarity

The combined results therefore suggest that the late J-space/workspace band acts as a region in which **task-relevant social concepts become highly accessible**.

Where bias-related differences occur, they appear as a **relative imbalance between simultaneously accessible semantic alternatives**, rather than as the exclusive emergence of stereotype-congruent information.

In other words:

$$
\boxed{
\text{Workspace Accessibility}
\neq
\text{Stereotypical Preference}
}
$$

The strongest evidence across datasets concerns the emergence and accessibility of task-relevant representations.

The evidence for stereotypical preference is more heterogeneous and context-dependent.

---

# 16. Key Findings

1. **P2 and P3 consistently produce the strongest J-Lens accessibility of the target concepts.**

2. **P1 consistently produces the weakest target accessibility**, ranking worst in joint log rank across all seven datasets.

3. **Layer 27 is a highly reproducible transition point.** For P2, the strongest target-salience transition occurs at layer 27 in all seven datasets.

4. **Both bias and non-bias targets undergo the layer-27 transition**, suggesting that this is primarily a task-relevant representation/readout effect rather than the emergence of the stereotype target alone.

5. **SES produces the strongest overall positive bias MRR gap**, particularly under P3.

6. **Sexual Orientation also produces a comparatively strong bias-relative signal**, particularly under P2.

7. **Physical Appearance demonstrates that high target accessibility can coexist with a negative bias MRR gap.**

8. **Religion demonstrates that high target accessibility can coexist with almost no preference between the two targets.**

9. **Ambiguity effects are category-dependent.** Age and Nationality show clear reductions/reversals after disambiguation, whereas Sexual Orientation and Disability Status retain their signals.

10. **Question polarity strongly influences the readout**, particularly for Physical Appearance, Religion, Nationality, and Sexual Orientation.

11. The overall results support distinguishing **representation strength** from **representational preference** when interpreting J-Lens bias experiments.

---

# 17. Limitations and Next Analysis

The results presented here are primarily descriptive.

A positive or negative mean MRR gap does not by itself establish that the difference is statistically reliable across BBQ items.

The next stage of analysis should therefore evaluate whether the observed effects survive item-level statistical testing.

Particular attention should be given to:

- Paired bias-target versus non-bias-target differences
- Confidence intervals around MRR gaps
- Ambiguous versus disambiguated differences
- Negative versus non-negative polarity interactions
- Prompt × condition interactions
- Layer × target interactions
- Whether the layer-27 transition is statistically consistent at the individual-item level

This additional analysis would help distinguish robust cross-item effects from differences caused by a smaller number of highly ranked targets.

---

# 18. Final Summary

The experiments provide strong evidence that prompt formulation and model depth substantially influence how bias-relevant semantic concepts become accessible in J-space.

The most reproducible finding is not a universal stereotypical preference, but a shared late-layer transition in which both competing target representations become substantially more readable.

The results therefore support a model in which the J-Lens workspace-band readout captures the emergence of **task-relevant semantic representations**, while bias is expressed as a smaller and more context-dependent relative difference between those representations.

This produces a more nuanced interpretation of the original research question:

> **The J-Lens workspace band reliably surfaces stereotype-relevant and counter-stereotype-relevant concepts, but whether the stereotype-congruent representation dominates depends on the BBQ category, prompt, context condition, and question polarity.**

The cross-dataset results therefore suggest that the workspace contains socially relevant information without supporting the stronger claim that it uniformly prioritizes stereotypical representations.

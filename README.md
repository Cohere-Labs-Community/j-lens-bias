# Jacobian Lens Bias

## Overview

**Jacobian Lens Bias** investigates whether social stereotypes become detectable inside a language model **before the model explicitly expresses them in its output**.

The central question is:

> When a vignette implicitly evokes a social category—such as gender, race/ethnicity, socioeconomic status, age, nationality, religion, or disability—without naming it, does a stereotype-congruent concept reliably surface in the model's Jacobian Lens readout? And if we intervene on that representation before the vignette provides disambiguating information, can we causally alter how the model resolves the vignette downstream?

The project combines **Jacobian Lens readouts**, **controlled interventions**, and **standard behavioral bias evaluation** to distinguish between stereotypes that are merely detectable in a model's internal computation and stereotypes that actually influence its final behavior.

---

## Motivation

A language model can produce an apparently unbiased answer while still transiently representing stereotypical defaults during computation.

Behavioral evaluations only observe the model's final output. The Jacobian Lens gives us another perspective: we can examine which token-level concepts become locally salient at intermediate layers.

This raises an interesting possibility:

**A model may internally default toward a stereotype even when that stereotype never appears in its final answer.**

If so, we would like to know:

1. **Whether the representation exists**
2. **When it emerges**
3. **How strongly it reflects known human stereotypes**
4. **Whether it causally affects downstream computation**
5. **Whether different social axes behave differently**

---

## Research Questions

### 1. Do implicit stereotypes surface in the Jacobian Lens?

For each vignette, we define:

- a **stereotype-congruent synonym set**
- a **counter-stereotype synonym set**
- a **neutral/control set**

We then inspect the model's **workspace-band Jacobian Lens readout** before any explicit category information appears.

A **Hit** occurs when a token belonging to one of the predefined synonym sets reaches **rank 1** somewhere within the workspace band.

We ask whether:

> `HitRate(congruent) > HitRate(counter) > HitRate(control)`

We additionally evaluate **pass\@k** to determine whether the effect persists beyond strict rank-1 detection.

---

### 2. Does internal bias track real-world stereotype strength?

A uniform congruent-token advantage would be less informative than an effect that varies systematically with stereotype strength.

For example, occupational gender stereotypes can be compared against external measures such as labor-force composition from the **U.S. Bureau of Labor Statistics (BLS)**.

For each item, we therefore compare the Jacobian Lens asymmetry

**ΔHit = HitRate(congruent) − HitRate(counter)**

against external measures of stereotype strength, including:

- occupational demographic composition
- existing psycholinguistic stereotype norms
- other validated category-specific datasets where available

The prediction is that stronger externally measured stereotypes should produce larger internal readout asymmetries.

---

### 3. Internal bias vs. behavioral bias

For every item, we also evaluate the model normally, **without using or intervening on the Jacobian Lens**, and score its response using a standard behavioral bias metric.

This gives us two quantities:

**Lens Bias**\
Bias inferred from intermediate Jacobian Lens readouts.

**Behavioral Bias**\
Bias observable in the model's actual answer.

We then define a **Lens–Behavior Gap**:

**Gap = Lens Bias − Behavioral Bias**

A positive gap indicates an especially interesting regime:

> The model appears relatively unbiased behaviorally while still exhibiting a stereotype-congruent internal default.

This allows us to test whether behavioral evaluations systematically underestimate latent stereotypical representations.

---

## Causal Intervention: Counter-Stereotype Swaps

Detection alone does not establish causal relevance.

We therefore intervene on the Jacobian Lens coordinate **before the vignette provides disambiguating information**.

For a vignette whose lens readout favors a stereotype-congruent token, we clamp or shift the relevant coordinate toward a predefined **counter-stereotype token**.

We then compare downstream behavior across:

- **No intervention**
- **Counter-stereotype swap**
- **Sham swap** toward an unrelated/off-axis token with approximately matched frequency

If the representation is causally involved in resolving the vignette, counter-stereotype interventions should alter downstream continuation more strongly than sham interventions.

### Riddle-style vignettes

Riddle-structured items provide an additional behavioral signal.

When later information contradicts an implicit stereotype, models may produce language associated with surprise or contradiction resolution.

We test whether counter-stereotype swaps performed **before disambiguation** reduce this behavior relative to the unmodified model.

---

## Running the cataluna84 submissions on an RTX 2070

The study-group notebooks for `cataluna84` use `Qwen/Qwen3.5-4B` and the released `n1000` lens (`neuronpedia/jacobian-lens`). This checkout runs in WSL2. GPU 0 is the RTX 2070 (8 GB, compute capability 7.5) and GPU 1 is a 4 GB GTX 1650 SUPER. CUDA comes from the Windows driver through `/usr/lib/wsl/lib`; do not install a Linux NVIDIA driver inside the distro. WSLg is already using about 1 GB of the 2070. bf16 weights are about 9 GB and Turing has no bf16 tensor cores, so the week 1 notebook, and the week 2 and week 3 collector scripts, load the 4B model on GPU 0 in int8 with fp16 for the unquantized weights. `torch.compile` stays off. GPU 1 is never used. The WSL VM has about 15 GB of RAM and no `.wslconfig`, which is enough for the int8 GPU path and tight for a CPU bf16 load.

```bash
uv sync
uv run python -m ipykernel install --user --name j-lens-bias --display-name "J-Lens (RTX 2070)"
git clone --depth 1 https://github.com/anthropics/jacobian-lens.git third_party/jacobian-lens
ln -sfn ../../third_party/jacobian-lens/assets "Week 1/submissions/assets"
ln -sfn ../../third_party/jacobian-lens/data "Week 1/submissions/data"
git clone --depth 1 https://github.com/nyu-mll/BBQ.git /home/cataluna84/Workspace/BBQ
```

Select the **J-Lens (RTX 2070)** kernel. Week 1's walkthrough (`Week 1/submissions/cataluna84_week_1.ipynb`) loads the released lens and can apply it. The fitting cell stops immediately on this card: `fit()` needs tens of gigabytes of VRAM. Week 2's collector is `Week 2/submissions/cataluna84_week_2_bbq_stats.py` (`--int8` is the default). Its notebook reads the records that collector writes; those records are not in git. Point it at them with `BBQ_OUT` (the default path is `/home/cataluna84/Workspace/jacobian-lens/analysis/out/bbq`). Week 3's two notebooks run from the bundled `cataluna84_week_3_data` directory with no GPU. Re-collecting uses the `cataluna84_week_3_*.py` scripts in that same directory, again in int8 on GPU 0.

On this card the int8 4B model occupies about 4.5 GiB, which fits in the memory WSLg leaves free. Qwen3.5's linear-attention layers use the pure PyTorch fallback (`causal_conv1d` and `flash-linear-attention` are not installed). That path is the one the lens apply calls run on, and it is slower than the fused kernels.

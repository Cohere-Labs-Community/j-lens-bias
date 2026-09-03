# J Lens Bias — Week 1

## Goal

The goal for **Week 1** is to make sure everyone gets the Jacobian Lens repo working **end-to-end**.

**No bias experiments yet.** This week is about understanding the tooling, running the existing pipeline, and extracting Jacobian Lens readouts programmatically.

Repository: [https://github.com/anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)

## Assignment

### 1. Set up the Jacobian Lens

- Use a **pre-fitted Jacobian Lens on Qwen3.6-27B**.
- Run `walkthrough.ipynb` from top to bottom.
- Confirm that you can generate a **slice-vis page that renders correctly**.

### 2. Work with `lens.apply()`

Use the repo's own `apply()` interface.

- Call `lens.apply()` with positions covering **every valid source position in the prompt**.
- Generate a full **slice-vis grid**:
  - one column per position
  - one row per layer

### 3. Run on Experiment Prompts

Use prompts from the repo's own:

`data/experiments/`

Run as many prompts as your available compute and time allow.

For **each prompt × layer × position**, extract the **top-1 token and its rank**, matching what is displayed on the slice-vis page.

### 4. Summarize Your Results

Produce a **per-layer summary table**, aggregated over:

- all positions
- all prompts in your experiment set

Feel free to include any additional visualizations or observations you find interesting.

## Submission

Submit your work as a Jupyter notebook using the following naming convention:

`username_week_1.ipynb`

Please keep your notebook reproducible and preserve the important outputs, tables, and visualizations.

## Presentation

Be ready to briefly present your results in our next session on **04/09**.

If you have questions about the assignment, implementation, or what to present, drop a message in the channel. This work is meant to be collaborative, and **there are no wrong answers!**

> **Compute note:** If the experiment is too heavy for your available compute, feel free to use a smaller Qwen model with an available pre-fitted lens.

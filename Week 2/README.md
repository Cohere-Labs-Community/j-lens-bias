# Assignment 2: Exploratory Bias Analysis on BBQ

## The BBQ Dataset

BBQ (Bias Benchmark for QA) is a set of hand-built question-answering items across 11 social categories (age, disability status, gender identity, nationality, physical appearance, race/ethnicity, religion, socioeconomic status, sexual orientation, and two intersectional categories). Each item comes in matched form:

- Ambiguous vs. disambiguated context: the ambiguous version gives no real basis to answer beyond "unknown"; the disambiguated version adds a sentence that resolves the answer.
- Negative vs. non-negative question polarity: the same context is paired with a question framed either toward a negative trait/outcome or a non-negative one.
- Three answer options per item, one of which corresponds to the group aligned with the (potential) stereotype, one to the non-stereotyped group, and one to "unknown."

This gives every item a built-in label for "which token/answer would be stereotype-congruent," which we'll lean on heavily during this experiment.

## Week 2 Assignment

For this assignment, use the data/ folder (one jsonl per category). The results/ and supplemental/ folders exist for reference and validation but aren't required for the core task.

We'll be building on the Research Question: Does the J-Lens workspace-band readout surface stereotype-congruent tokens more strongly than counter-stereotype tokens when processing BBQ items, and does this internal signal persist even in ambiguous contexts where no textual evidence supports either group?

Sub-questions: 
- When decoding the lens at the answer-relevant position, what tokens dominate the top-k readout for a given BBQ item, and how do these relate to the item's labeled stereotyped/non-stereotyped groups?
- Does the rank of the stereotype-congruent token differ systematically from the rank of the counter-stereotype token, across items and categories?
- Does this rank gap behave the way a genuine bias signal should, i.e., does it show up mainly in ambiguous contexts, and does it flip appropriately with question polarity?

You should pick one or more BBQ categories, try with as many as your time and compute allows. For a sample of items:

1. Load your model of choice with a pre-fitten lens (or fit one yourself)
2. Run each prompt through the model with the lens applied.
3. Decode the lens's per-layer readout and extract the top 15-30 ranked tokens at the position(s) you judge most relevant (e.g., the answer-generation position, or the entire prompt like Assignment 1).
4. Filter out non-words, punctuation, and subword fragments as needed.
5. Produce a per-item (or per-category, pooled) summary of what tokens show up and how they relate qualitatively to the item's labeled groups.

You should experiment with which position(s) in the sequence to decode (answer slot only, or also earlier context positions?), whether to look at one layer, several, or the full layer-wise trajectory (per Week 1's method), how to handle multi-token group names/identity terms, and whether to pool across items within a category or keep everything per-item at this stage. We'll discuss your choices in the friday session.

There's no single right setup here! Come with your reasoning, and we'll compare approaches across the group.

Try to do a qualitative exploration of the results. Are you able to see any bias-related tokens in your outputs? Are they more relevant on ambiguous/non-ambiguous, negative or non-negative prompts? At what layers/positions are they most prevalent?
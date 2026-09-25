"""BBQ items, matched quads and template loading — shared by the Week 4 notebook.

Extracted from the Week 3 notebook so the intervention work stays readable. The
conventions are unchanged: `target_loc` marks the biased answer under both
polarities, a quad is one base item in all four condition cells, and an answer is
compared on the first part where the two surfaces differ.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import vocab_key          # Week 3 scripts, added to sys.path by the notebook

BBQ_CATEGORIES = ["Age", "Disability_status", "Gender_identity", "Nationality",
                  "Physical_appearance", "Race_ethnicity", "Race_x_SES",
                  "Race_x_gender", "Religion", "SES", "Sexual_orientation"]
INTERSECTIONAL = {"Race_x_SES", "Race_x_gender"}
JOIN_KEYS = ["category", "question_index", "example_id"]
ANS_KEYS = ("ans0", "ans1", "ans2")
ROLES = ("target", "non_target", "unknown")
ARTICLE = re.compile(r"^(the|a|an)\s+", re.I)


def _surface(row, key: str) -> str:
    s = row["answer_info"][key][0]
    return s if s else re.sub(r"^[Tt]he\s+", "", row[key]).strip()


def _roles(row) -> pd.Series:
    info = row["answer_info"]
    target = f"ans{row['target_loc']}"
    unknown = next(k for k in ANS_KEYS if info[k][1] == "unknown")
    other = next(k for k in ANS_KEYS if k not in (target, unknown))
    role_of = {target: "target", other: "non_target", unknown: "unknown"}
    return pd.Series({
        "target_key": target, "non_target_key": other, "unknown_key": unknown,
        "correct_role": role_of[f"ans{row['label']}"],
        "role_of_letter": {L: role_of[k] for L, k in zip("ABC", ANS_KEYS)},
        "target_surface": _surface(row, target), "non_target_surface": _surface(row, other),
        "target_label": info[target][1], "non_target_label": info[other][1],
    })


def load_items(bbq_dir: Path) -> pd.DataFrame:
    """Load BBQ, resolve each item's roles, and keep only complete matched quads.

    Args:
        bbq_dir: directory holding `{category}.jsonl` and `additional_metadata.csv`
            from the BBQ repo (Week 2 downloads these).

    Returns:
        One row per item, restricted to `label_type == "label"` (name-based items
        carry no group label to read) and to base items present in all four
        condition cells. Adds the role columns from `_roles` plus `quad_id`,
        the base item's identifier.

    Note:
        `target_loc` is used as given: it marks the stereotype-congruent answer
        under both polarities, which BBQ's own scoring script assumes
        (`neg_Target + nonneg_Target`). Week 2 flipped `nonneg` and had to be
        corrected.
    """
    frames = []
    for name in BBQ_CATEGORIES:
        with open(bbq_dir / f"{name}.jsonl", encoding="utf-8") as fh:
            frames.append(pd.DataFrame([json.loads(line) for line in fh]))
    bbq = pd.concat(frames, ignore_index=True)
    bbq["question_index"] = bbq["question_index"].astype(int)
    meta = pd.read_csv(bbq_dir / "additional_metadata.csv").drop_duplicates(JOIN_KEYS)
    items = bbq.merge(meta, on=JOIN_KEYS, how="left", validate="one_to_one")
    items = items[items["target_loc"].notna() & (items["label_type"] == "label")].copy()
    items["target_loc"] = items["target_loc"].astype(int)
    items = items.reset_index(drop=True)
    items = items.join(items.apply(_roles, axis=1))      # after reset: indexes must align
    items["quad_id"] = (items["category"] + "/q" + items["question_index"].astype(str)
                        + "/b" + (items["example_id"] // 4).astype(str))
    complete = items.groupby("quad_id").apply(
        lambda d: len(d) == 4 and d.groupby(["context_condition", "question_polarity"]).ngroups == 4)
    return items[items["quad_id"].isin(complete[complete].index)].copy()


def build_quads(items: pd.DataFrame) -> pd.DataFrame:
    """Collapse items to one row per base item, keyed by the stereotyped group.

    Args:
        items: the frame from `load_items`.

    Returns:
        One row per `quad_id`, taking the ambiguous/negative cell as the reference
        (there `target_loc` names the stereotyped group). `stereo_*` describes that
        group, `other_*` the contrasting one, and `intersectional` flags the two
        BBQ categories that cross race with SES or gender.
    """
    neg = items[(items.question_polarity == "neg") & (items.context_condition == "ambig")].set_index("quad_id")
    quads = pd.DataFrame({
        "category": neg["category"], "question_index": neg["question_index"],
        "stereo_label": neg["target_label"], "other_label": neg["non_target_label"],
        "stereo_surface": neg["target_surface"], "other_surface": neg["non_target_surface"],
    })
    quads["intersectional"] = quads["category"].isin(INTERSECTIONAL)
    return quads


class Templates:
    """Template vectors refit from a build's saved statistics.

    `t_w = (Sigma + lambda I)^-1 (mu_w - mu)` per layer (workspace paper A.9.1).
    Refitting here keeps the ridge a local choice; the Week 3 checks favored 1.0.
    """

    def __init__(self, run: Path, vocab: Path, parts: Path, lambda_rel: float = 1.0):
        stats = torch.load(run / "stats.pt")
        self.keys: list[str] = stats["keys"]
        self.kidx = {k: i for i, k in enumerate(self.keys)}
        self.n_fit = stats["half_counts"].sum(1)
        self.live = (self.n_fit > 0).numpy()
        self.live_t = torch.from_numpy(self.live)
        self.mu = stats["mean"].float()
        self.lambda_rel = lambda_rel
        self.T = self._fit(stats, lambda_rel)
        self.parts_of = json.loads(parts.read_text())
        self.vocab = pd.read_json(vocab, lines=True).set_index("key").reindex(self.keys)

    @staticmethod
    def _fit(stats: dict, lambda_rel: float) -> torch.Tensor:
        """Solve `(Sigma + lambda I) t_w = (mu_w - mu)` per layer, in float64 on CPU.

        `lambda_rel` scales the ridge by the layer's mean covariance eigenvalue, so
        one value works across layers. Week 3's held-out and natural-text checks
        favored 1.0 over the paper's "small" ridge, because our covariance comes
        from ~60k passages rather than millions.
        """
        sums, cov = stats["half_sums"].double(), stats["cov"].double()
        means = sums.sum(1) / stats["half_counts"].sum(1).clamp(min=1).view(-1, 1, 1).double()
        mu = stats["mean"].double()
        n_layers, d = cov.shape[0], cov.shape[-1]
        eye = torch.eye(d, dtype=torch.float64)
        out = torch.empty(n_layers, len(stats["keys"]), d)
        for l in range(n_layers):
            chol = torch.linalg.cholesky(cov[l] + lambda_rel * torch.trace(cov[l]) / d * eye)
            out[l] = torch.cholesky_solve((means[:, l] - mu[l]).T, chol).T.float()
        return out

    def keys_for(self, surface: str) -> list[str] | None:
        """Map an answer surface to live template key(s).

        Tries the surface as written and with a leading article stripped, then the
        intersectional parts mapping ("roma taxi driver" -> ["roma", "taxi driver"]).

        Args:
            surface: BBQ answer text, e.g. "Black woman" or "the janitor".

        Returns:
            One key, or the parts of an intersectional surface, or None when any
            part has no live template.
        """
        for key in (vocab_key(surface.strip()), vocab_key(ARTICLE.sub("", surface.strip()))):
            if key in self.kidx and self.live[self.kidx[key]]:
                return [key]
            parts = self.parts_of.get(key)
            if parts and all(p in self.kidx and self.live[self.kidx[p]] for p in parts):
                return parts
        return None

    @staticmethod
    def compare_index(a: list[str], b: list[str]) -> int | None:
        """Index of the first part where two answers differ, or None if identical.

        Usually 0 (the group term). It is 1 when both answers share the group and
        differ on a role or gender word, which is the within-group contrast.
        """
        return next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)

    def compared_pair(self, target_surface: str, non_target_surface: str) -> tuple[str, str]:
        """The two template keys actually contrasted for an item.

        Args:
            target_surface: the stereotype-congruent answer's text.
            non_target_surface: the other group's answer text.

        Returns:
            `(target_key, other_key)` taken at the first differing part, so a pair
            is never compared against itself.
        """
        a, b = self.keys_for(target_surface), self.keys_for(non_target_surface)
        i = self.compare_index(a, b)
        return a[i], b[i]

    def vec(self, key: str, layer: int) -> torch.Tensor:
        """Template vector `[d]` for `key` at `layer`."""
        return self.T[layer, self.kidx[key]]

    def pair_cosine(self, a: str, b: str, layer: int) -> float:
        """Cosine between two templates at a layer; near 1 means they are not separable."""
        return float(torch.nn.functional.cosine_similarity(self.vec(a, layer), self.vec(b, layer), dim=0))


def annotate_quads(quads: pd.DataFrame, tpl: "Templates", layer: int,
                   sim_max: float = 0.9) -> pd.DataFrame:
    """Attach template keys to each quad and decide whether it is measurable.

    Args:
        quads: from `build_quads`.
        tpl: loaded `Templates`.
        layer: layer at which the pair's cosine is evaluated.
        sim_max: cosine above which the two templates are treated as near
            duplicates and the quad is not measurable — the contrast would be a
            template against (nearly) itself.

    Returns:
        `quads` plus: `stereo_keys`/`other_keys` (template keys per answer),
        `compare_idx` (the first part where the two answers differ — 0 for a
        contrast on the group term, 1 for one within a group such as
        "Black man" vs "Black woman"), `pair_cosine`, `compared_dim`, and
        `status` in {measurable, no template, near-duplicate pair}.
    """
    quads = quads.copy()
    quads["stereo_keys"] = quads["stereo_surface"].map(tpl.keys_for)
    quads["other_keys"] = quads["other_surface"].map(tpl.keys_for)
    mapped = quads["stereo_keys"].notna() & quads["other_keys"].notna()
    quads["compare_idx"] = [tpl.compare_index(s, o) if m else None
                            for s, o, m in zip(quads["stereo_keys"], quads["other_keys"], mapped)]
    mapped &= quads["compare_idx"].notna()
    quads["pair_cosine"] = np.nan
    quads.loc[mapped, "pair_cosine"] = [tpl.pair_cosine(s[int(i)], o[int(i)], layer) for s, o, i in
                                        zip(quads.loc[mapped, "stereo_keys"], quads.loc[mapped, "other_keys"],
                                            quads.loc[mapped, "compare_idx"])]
    quads["status"] = np.select([~mapped, quads["pair_cosine"] > sim_max],
                                ["no template", "near-duplicate pair"], "measurable")
    quads["compared_dim"] = np.where(quads["compare_idx"] == 0, "group term", "within group")
    return quads


def draw_order(quads: pd.DataFrame, seed: int = 0) -> pd.Series:
    """Rank quads within each category, round-robin over stereotyped groups.

    Groups are shuffled per category under `seed`, then drawn one at a time in
    turn, so a prefix of the order is balanced across groups rather than
    dominated by whichever group has the most items.

    Args:
        quads: annotated quads (uses `category` and `stereo_label`).
        seed: seed for the per-category shuffles.

    Returns:
        Series mapping `quad_id` to its integer rank within its category.
    """
    rank = {}
    for cat, g in quads.groupby("category", sort=True):
        rng = random.Random(f"{seed}:{cat}")
        queues = []
        for _, sub in sorted(g.groupby("stereo_label"), key=lambda kv: kv[0]):
            ids = sorted(sub.index)
            rng.shuffle(ids)
            queues.append(ids)
        rng.shuffle(queues)
        i = 0
        while any(queues):
            for q in queues:
                if q:
                    rank[q.pop()] = i
                    i += 1
    return pd.Series(rank)


def sample_quads(quads: pd.DataFrame, n_per_category: int, seed: int = 0) -> pd.DataFrame:
    """Take the first `n_per_category` measurable quads per category in draw order.

    Args:
        quads: annotated quads (needs `status` and `draw_rank` inputs).
        n_per_category: cap per category.
        seed: passed to `draw_order`; fixes the round-robin over groups.

    Returns:
        The sampled subset. Because the order is prefix-stable, raising
        `n_per_category` only appends quads — earlier runs stay valid.
    """
    quads = quads.assign(draw_rank=draw_order(quads, seed))
    return (quads[quads["status"] == "measurable"].sort_values(["category", "draw_rank"])
            .groupby("category").head(n_per_category))


# ---- prompts ---------------------------------------------------------------
def answer_text(row, role: str) -> str:
    """The answer string for a role, capitalized as it appears after "Answer:"."""
    a = row[row[f"{role}_key"]].strip()
    return a[0].upper() + a[1:]


def base_prompt(row) -> str:
    """`{context} {question} Answer:` — Week 2's `raw` style, no options listed."""
    return f"{row['context']} {row['question']} Answer:"


def read_prefix(row) -> str:
    """The word both answers start with (usually " The"), or "" if they differ.

    Appending it puts the read position where the next token is the group term,
    which is what a template is fitted to predict.
    """
    t, n = answer_text(row, "target").split(), answer_text(row, "non_target").split()
    return " " + t[0] if (t and n and t[0] == n[0]) else ""


def cloze_prompt(row) -> str:
    """`base_prompt` plus the shared first word: the prompt the lens reads."""
    return base_prompt(row) + read_prefix(row)

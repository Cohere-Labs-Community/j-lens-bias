"""Side-by-side slice pages for one BBQ matched quad.

Wraps the released slice viewer (`jlens.vis`): one page per condition, the two
group answers' first tokens pinned in each, laid out in a 2x2 grid that matches
BBQ's design (context condition x question polarity). The header carries the
per-condition metrics so the lens picture and the numbers are read together.

    from dashboard import quad_page
    html = quad_page(model, lens, rows, metrics)      # rows: the quad's 4 items
    Path("quad.html").write_text(html)
"""

from __future__ import annotations

import gzip
import html as _html
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from jlens.vis import build_page, compute_slice

GLOSS_URL = ("https://raw.githubusercontent.com/anthropics/jacobian-lens/main/"
             "assets/qwen_gloss.json.gz")


def load_gloss(cache: Path) -> dict[int, str]:
    """Machine-generated English gloss for Qwen's CJK vocabulary (Week 2 §2.1).

    `mask_display=True` keeps every alphanumeric token, and CJK counts as
    alphanumeric, so the top-k is full of unreadable single characters. Week 2's
    rule applies here too: gloss them, don't drop them -- a filtered top-k hides
    what the lens is actually surfacing.
    """
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(GLOSS_URL, timeout=120) as response:
            cache.write_bytes(response.read())
    with gzip.open(cache) as fh:
        return {int(k): v for k, v in json.load(fh).items()}

CSS = """
:root { color-scheme: light dark; }
body { margin: 0; font: 14px/1.5 system-ui, sans-serif; }
header { padding: 12px 16px; border-bottom: 1px solid color-mix(in oklab, currentColor 20%, transparent); }
h1 { font-size: 16px; margin: 0 0 4px; }
p.sub { margin: 0; opacity: .75; }
table { border-collapse: collapse; margin-top: 10px; font-size: 13px; }
th, td { padding: 3px 10px 3px 0; text-align: left; }
th { font-weight: 600; opacity: .75; }
td.num { font-variant-numeric: tabular-nums; }
nav { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-top: 10px; }
nav span.lbl { font-size: 12px; opacity: .7; margin-right: 2px; }
button { font: inherit; font-size: 12px; padding: 3px 9px; border-radius: 999px; cursor: pointer;
         border: 1px solid color-mix(in oklab, currentColor 30%, transparent); background: transparent; color: inherit; }
button[aria-pressed="true"] { background: color-mix(in oklab, currentColor 15%, transparent); font-weight: 600; }
.grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; padding: 12px; }
.grid[data-layout="stack"] { grid-template-columns: minmax(0, 1fr); }
@media (max-width: 900px) { .grid { grid-template-columns: minmax(0, 1fr); } }
figure { margin: 0; min-width: 0; }
figcaption { font-size: 13px; font-weight: 600; padding: 2px 0 6px; }
figcaption span { font-weight: 400; opacity: .75; }
p.says { margin: 6px 0 0; font-size: 13px; opacity: .85; }
p.says b { font-weight: 600; }
iframe { width: 100%; height: 560px; border: 1px solid color-mix(in oklab, currentColor 20%, transparent); border-radius: 6px; }
.grid[data-layout="stack"] iframe { height: 620px; }
"""

# Views: all four, the four informative pairs (stacked), or one panel at a time.
# Panels are only hidden, never re-rendered, so switching never reloads a slice.
VIEWS_JS = """
const figs = [...document.querySelectorAll('figure')];
const grid = document.querySelector('.grid');
function show(cells, layout) {
  figs.forEach(f => { f.hidden = !cells.includes(f.dataset.cell); });
  grid.dataset.layout = layout;
  document.querySelectorAll('nav button').forEach(b =>
    b.setAttribute('aria-pressed', String(b.dataset.cells === cells.join('|'))));
}
document.querySelectorAll('nav button').forEach(b => b.addEventListener('click', () => {
  const cells = b.dataset.cells.split('|');
  show(cells, b.dataset.layout);
}));
show(figs.map(f => f.dataset.cell), 'grid');
"""


def greedy_continuation(model, prompt: str, max_new_tokens: int = 12) -> str:
    """What the model actually says next, greedily. The slice's final row is the
    model's own next-token distribution, but a multi-token answer only shows up
    there as its first token."""
    import torch

    ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to(model.input_device)
    with torch.inference_mode():
        out = model._hf_model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=model.tokenizer.eos_token_id)
    return model.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def _cell_label(row) -> str:
    return f"{row['context_condition']}/{row['question_polarity']}"


def _metrics_table(metrics: pd.DataFrame, cols: dict[str, str]) -> str:
    head = "".join(f"<th>{_html.escape(v)}</th>" for v in ["condition", *cols.values()])
    body = ""
    for _, m in metrics.iterrows():
        cells = "".join(
            f'<td class="num">{m[c]:.2f}</td>' if isinstance(m[c], float) else f"<td>{_html.escape(str(m[c]))}</td>"
            for c in cols)
        body += f"<tr><td>{_html.escape(m['condition'])}</td>{cells}</tr>"
    return f"<table><tr>{head}</tr>{body}</table>"


def _alt_tokens(model, slice_data, gloss: dict[int, str]) -> dict[int, str]:
    """`{id: "text(gloss)"}` for the glossed ids this page can display."""
    ids = set(np.unique(slice_data.top_ids).tolist()) | set(slice_data.tracked_token_ids)
    return {i: f"{model.tokenizer.decode([i]).strip()}({gloss[i]})" for i in ids if i in gloss}


def quad_page(model, lens, rows: pd.DataFrame, metrics: pd.DataFrame, *,
              prompt_of, pinned_of, title: str | None = None, gloss: dict[int, str] | None = None,
              continuation_of=None,
              top_n: int = 10, layer_stride: int = 1, last_n_tokens: int | None = 24,
              metric_cols: dict[str, str] | None = None) -> str:
    """One HTML page: the quad's four conditions as slice pages in a 2x2 grid.

    Args:
        model, lens: as for `jlens.vis.compute_slice`.
        gloss: `{token id: english}` from `load_gloss`; CJK tokens are shown as
            `text(gloss)` instead of being dropped.
        rows: the quad's four items (one row per condition).
        metrics: one row per condition, with a `condition` column matching
            `"{context_condition}/{question_polarity}"`.
        prompt_of: row -> the prompt string to read.
        continuation_of: row -> what the model generates from that prompt, shown
            under the panel (e.g. `partial(greedy_continuation, model)` composed
            with `prompt_of`). None omits it.
        pinned_of: row -> {token id: label} pinned in that condition's page.
        last_n_tokens: window the slice grid to the last N positions (the
            forward pass still sees the whole prompt); None renders every one.
    """
    metric_cols = metric_cols or {c: c for c in metrics.columns if c != "condition"}
    order = {"ambig/neg": 0, "ambig/nonneg": 1, "disambig/neg": 2, "disambig/nonneg": 3}
    rows = rows.assign(_cell=rows.apply(_cell_label, axis=1)).sort_values("_cell", key=lambda s: s.map(order))

    figures = []
    for _, row in rows.iterrows():
        prompt, pinned = prompt_of(row), pinned_of(row)
        # word-like display only (ranks stay full-vocab); CJK is glossed, not dropped
        slice_data = compute_slice(model, lens, prompt, top_n=top_n, layer_stride=layer_stride,
                                   last_n_tokens=last_n_tokens, pinned_token_ids=set(pinned),
                                   mask_display=True)
        page, _, _ = build_page(slice_data, prompt, title=row["_cell"],
                                description=row["question"], mode="embed",
                                pinned_token_ids=set(pinned),
                                alt_token=_alt_tokens(model, slice_data, gloss) if gloss else None)
        pins = ", ".join(f"{v} ({model.tokenizer.decode([k]).strip()})" for k, v in pinned.items())
        says = continuation_of(row) if continuation_of else None
        said = (f'<p class="says">model continues: <b>{_html.escape(prompt.split()[-1])} '
                f'{_html.escape(says)}</b></p>') if says else ""
        figures.append(
            f'<figure data-cell="{_html.escape(row["_cell"])}"><figcaption>{_html.escape(row["_cell"])} '
            f'<span>pinned: {_html.escape(pins)}</span></figcaption>'
            f'<iframe srcdoc="{_html.escape(page)}"></iframe>{said}</figure>')

    cells = rows["_cell"].tolist()
    views = [("all four", cells, "grid")]
    for a, b, label in [("ambig/neg", "disambig/neg", "context · neg"),
                        ("ambig/nonneg", "disambig/nonneg", "context · nonneg"),
                        ("ambig/neg", "ambig/nonneg", "polarity · ambig"),
                        ("disambig/neg", "disambig/nonneg", "polarity · disambig")]:
        if a in cells and b in cells:
            views.append((label, [a, b], "stack"))
    views += [(c, [c], "stack") for c in cells]
    nav = "".join(
        f'<button data-cells="{_html.escape("|".join(cs))}" data-layout="{lay}">{_html.escape(label)}</button>'
        for label, cs, lay in views)

    first = rows.iloc[0]
    title = title or f'{first["category"]} — {first["quad_id"]}'
    sub = (f'{_html.escape(str(first["target_surface"]))} (stereotype-congruent) vs '
           f'{_html.escape(str(first["non_target_surface"]))}')
    return (f"<!doctype html><meta charset=utf-8><title>{_html.escape(title)}</title>"
            f"<style>{CSS}</style>"
            f"<header><h1>{_html.escape(title)}</h1><p class=sub>{sub}</p>"
            f"{_metrics_table(metrics, metric_cols)}"
            f'<nav><span class=lbl>compare</span>{nav}</nav></header>'
            f'<div class="grid" data-layout="grid">{"".join(figures)}</div>'
            f"<script>{VIEWS_JS}</script>")


def write_quad_page(path, *args, **kwargs) -> str:
    html_text = quad_page(*args, **kwargs)
    path.write_text(html_text, encoding="utf-8")
    print(f"{path} ({len(html_text) / 1e6:.1f} MB)")
    return html_text

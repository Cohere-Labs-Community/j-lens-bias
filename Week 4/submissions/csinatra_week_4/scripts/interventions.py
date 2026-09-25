"""Lens-coordinate interventions on template vectors, and their controls.

The swap follows the workspace paper section 2.5: with V = [v_s v_t], read the
coordinates c = V^+ h and set h <- h + alpha * V(sigma(c) - c), where sigma
exchanges the two entries. The component of h orthogonal to span{v_s, v_t} is
untouched.

Two variants, which the paper does not distinguish:

  single-pass  each layer reads c from the residual as it arrives, which already
               carries the edits made below it -- so the edit and its own
               measurement are coupled.
  two-pass     coefficients are cached from a clean forward pass first; every
               layer is then pushed toward the same fixed flip(c_clean).

Controls answer "is this the direction, or just a perturbation of this size":
a random direction at matched norm, an unrelated group's template, and a nonce
template (a word naming no group).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def pair_geometry(h: torch.Tensor, v_s: torch.Tensor, v_t: torch.Tensor) -> dict:
    """Measure how a template pair sits relative to a residual, before any model run.

    These quantities bound what an exact coordinate swap can achieve: an
    intervention confined to span{v_s, v_t} cannot change more of the residual
    than lies in that span.

    Args:
        h: residual at the read position, `[d]`, at some layer.
        v_s: first template vector at the same layer, `[d]` (by convention the
            target group's).
        v_t: second template vector, `[d]` (the contrasting group's).

    Returns:
        dict with
            `cos_templates`: cosine between the two templates; near 0 means they
                are close to orthogonal, so each carries separate information.
            `share_in_span`: fraction of ||h||^2 inside span{v_s, v_t} — the
                ceiling on what any edit confined to the pair can move.
            `coef_target`, `coef_other`: the least-squares coordinates c = V^+ h.
                Their *difference* drives the swap: sigma(c) - c is zero when they
                are equal, however well the readout separates the two groups.
            `delta_over_h`: ||alpha=1 displacement|| / ||h||, the relative size of
                the edit.
            `cond_V`: conditioning of [v_s v_t]; large values mean the pseudo-
                inverse is unstable because the two vectors nearly coincide.
            `h_norm`: ||h||, for scale.
    """
    h, v_s, v_t = h.double(), v_s.double(), v_t.double()
    V = torch.stack([v_s, v_t], dim=1)                      # [d, 2]
    V_pinv = torch.linalg.pinv(V)                           # [2, d]
    c = V_pinv @ h                                          # [2]
    proj = V @ c                                            # component of h in span{V}
    delta = V @ (c.flip(0) - c)                             # the alpha=1 displacement
    return {
        "cos_templates": float(torch.nn.functional.cosine_similarity(v_s, v_t, dim=0)),
        "share_in_span": float(proj.norm() ** 2 / h.norm() ** 2),
        "coef_target": float(c[0]), "coef_other": float(c[1]),
        "delta_over_h": float(delta.norm() / h.norm()),
        "cond_V": float(torch.linalg.cond(V)),
        "h_norm": float(h.norm()),
    }


# --------------------------------------------------------------------------- #
# interventions
# --------------------------------------------------------------------------- #
@dataclass
class Swap:
    """Exchange two concepts' lens coordinates over a set of layers.

    Implements section 2.5: with V = [v_s v_t] at a layer, c = V^+ h, the residual
    becomes h + alpha * V(sigma(c) - c), leaving the component orthogonal to
    span{v_s, v_t} untouched.

    Args:
        model: a `jlens` model wrapper (needs `.layers` and `.input_device`).
        vecs: `{layer: (v_source, v_target)}`, both `[d]` at that layer. Layers not
            in this mapping are left alone.
        alpha: scale on the displacement. 1.0 is the exact exchange; larger
            overshoots, negative reverses it (the sign-reversal control).
        mode: `"single"` reads c from the residual arriving at each layer, which
            already carries the edits made below it, so the edit and its own
            measurement are coupled. `"two"` caches c from a clean pass first and
            points every layer at the same fixed flip(c_clean).
        positions: token positions to patch; None patches every position.
        prefill_only: patch only the prompt pass, leaving generation unperturbed.

    Note:
        In `"two"` mode `handles()` runs one extra forward pass to fill the cache,
        so a measurement costs two passes rather than one.
    """

    model: object
    vecs: dict[int, tuple[torch.Tensor, torch.Tensor]]      # layer -> (v_source, v_target)
    alpha: float = 1.0
    mode: str = "single"
    positions: list[int] | None = None
    prefill_only: bool = False
    _cached: dict[int, torch.Tensor] = field(default_factory=dict)
    _V: dict[int, torch.Tensor] = field(default_factory=dict)
    _Vp: dict[int, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self):
        for layer, (a, b) in self.vecs.items():
            V = torch.stack([a, b], dim=1).to(self.model.input_device).float()
            self._V[layer] = V
            self._Vp[layer] = torch.linalg.pinv(V)

    # -- hooks ------------------------------------------------------------- #
    def _sel(self, h):
        return slice(None) if self.positions is None else self.positions

    def cache_clean(self, input_ids: torch.Tensor) -> None:
        """Run once unpatched and record each layer's coordinates (two-pass)."""
        handles = [self.model.layers[l].register_forward_hook(self._reader(l)) for l in self.vecs]
        try:
            with torch.inference_mode():
                self.model._text_module(input_ids=input_ids, use_cache=False)
        finally:
            for h in handles:
                h.remove()

    def _reader(self, layer: int):
        def hook(module, inputs, output):
            h = output if torch.is_tensor(output) else output[0]
            self._cached[layer] = (h.float() @ self._Vp[layer].T).detach()
            return output
        return hook

    def _patcher(self, layer: int, fired: dict):
        def hook(module, inputs, output):
            h = output if torch.is_tensor(output) else output[0]
            if self.prefill_only and (fired["done"] or h.shape[1] == 1):
                fired["done"] = fired["done"] or h.shape[1] == 1
                return output
            pos = self._sel(h)
            c = h.float() @ self._Vp[layer].T                       # [B, T, 2]
            target = (self._cached[layer] if self.mode == "two" else c).flip(-1)
            delta = torch.zeros_like(h, dtype=torch.float32)
            delta[:, pos] = (target - c)[:, pos] @ self._V[layer].T
            return self._write(output, h + (self.alpha * delta).to(h.dtype))
        return hook

    @staticmethod
    def _write(output, patched):
        return patched if torch.is_tensor(output) else (patched,) + tuple(output[1:])

    def handles(self, input_ids: torch.Tensor | None = None) -> list:
        """Register the patch hooks; caches clean coefficients first if two-pass."""
        if self.mode == "two":
            if input_ids is None:
                raise ValueError("two-pass needs input_ids to cache the clean coefficients")
            self.cache_clean(input_ids)
        fired = {"done": False}
        return [self.model.layers[l].register_forward_hook(self._patcher(l, fired)) for l in self.vecs]


@dataclass
class Patch:
    """Overwrite the residual at one position with a stored vector (activation patching).

    The field-standard causal test, used here as a calibration: take a residual
    from a *source* run (the disambiguated prompt, where the text names the answer)
    and write it into the *target* run (the ambiguous prompt) at the same layer and
    position. If the target's answer then follows the source, the information that
    decided it was present at that layer and position.

    Args:
        model: a `jlens` model wrapper.
        vecs: `{layer: [d] tensor}`, the source residuals to write in.
        position: token position to overwrite, default -1 (the read position).
            Source and target must align there; BBQ's ambiguous and disambiguated
            prompts end with identical tokens, so the last position does.
        prefill_only: during generation, patch only the prompt pass. Without it,
            every decode step would overwrite its newly generated token.
    """

    model: object
    vecs: dict[int, torch.Tensor]
    position: int = -1
    prefill_only: bool = True

    def handles(self, input_ids: torch.Tensor | None = None) -> list:
        def make(layer):
            v = self.vecs[layer].to(self.model.input_device)

            def hook(module, inputs, output):
                h = output if torch.is_tensor(output) else output[0]
                if self.prefill_only and h.shape[1] == 1:
                    return output                     # a decode step: position -1 is the new token
                h = h.clone()
                h[:, self.position] = v.to(h.dtype)
                return h if torch.is_tensor(output) else (h,) + tuple(output[1:])
            return hook
        return [self.model.layers[l].register_forward_hook(make(l)) for l in self.vecs]


@dataclass
class Steer:
    """h <- h + alpha * v, the paper's simplest write. alpha < 0 subtracts."""

    model: object
    vecs: dict[int, torch.Tensor]
    alpha: float = 1.0
    positions: list[int] | None = None

    def handles(self, input_ids: torch.Tensor | None = None) -> list:
        def make(layer):
            v = self.vecs[layer].to(self.model.input_device).float()

            def hook(module, inputs, output):
                h = output if torch.is_tensor(output) else output[0]
                pos = slice(None) if self.positions is None else self.positions
                delta = torch.zeros_like(h, dtype=torch.float32)
                delta[:, pos] = self.alpha * v
                out = h + delta.to(h.dtype)
                return out if torch.is_tensor(output) else (out,) + tuple(output[1:])
            return hook
        return [self.model.layers[l].register_forward_hook(make(l)) for l in self.vecs]


@dataclass
class Ablate:
    """Project the component along v out of the residual: tests necessity."""

    model: object
    vecs: dict[int, torch.Tensor]
    positions: list[int] | None = None

    def handles(self, input_ids: torch.Tensor | None = None) -> list:
        def make(layer):
            v = self.vecs[layer].to(self.model.input_device).float()
            u = v / v.norm()

            def hook(module, inputs, output):
                h = output if torch.is_tensor(output) else output[0]
                pos = slice(None) if self.positions is None else self.positions
                hf = h.float()
                delta = torch.zeros_like(hf)
                delta[:, pos] = -(hf[:, pos] @ u).unsqueeze(-1) * u
                out = hf + delta
                out = out.to(h.dtype)
                return out if torch.is_tensor(output) else (out,) + tuple(output[1:])
            return hook
        return [self.model.layers[l].register_forward_hook(make(l)) for l in self.vecs]


@torch.inference_mode()
def generate(model, input_ids: torch.Tensor, intervention=None, max_new_tokens: int = 12) -> str:
    """Greedy continuation, optionally under an intervention on the prompt pass.

    Args:
        model: a `jlens` model wrapper.
        input_ids: `[1, T]` prompt ids on the model's device.
        intervention: a `Swap` or `Patch` built with `prefill_only=True`, so only
            the prompt is edited and generation runs on the model's own weights.
        max_new_tokens: tokens to generate.

    Returns:
        The decoded continuation, stripped.
    """
    hooks = intervention.handles(input_ids) if intervention is not None else []
    try:
        out = model._hf_model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                                       pad_token_id=model.tokenizer.eos_token_id)
    finally:
        for h in hooks:
            h.remove()
    return model.tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- #
# controls
# --------------------------------------------------------------------------- #
def stable_seed(text: str) -> int:
    """A seed derived from a string that is the same in every process.

    Python's built-in `hash()` of a string is salted per process, so seeding from
    it gives a different random control on every run -- while a cache keyed on
    the item would return the old one.
    """
    import zlib
    return zlib.crc32(text.encode()) % 2**31


def random_like(v: torch.Tensor, seed: int) -> torch.Tensor:
    """Draw a random direction with the same norm as `v`.

    The matched-norm control: if a random direction of equal magnitude moves the
    answer as much as the template does, the effect is about perturbation size
    rather than about the concept.

    Args:
        v: the vector whose norm is matched, `[d]`.
        seed: seed for the draw, so a run is reproducible.

    Returns:
        A `[d]` tensor with ||result|| == ||v||.
    """
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(v.shape, generator=g)
    return r / r.norm() * v.norm()


def matched_controls(tpl, t_key: str, n_key: str, layers, seed: int = 0,
                     unrelated: str | None = None, nonce: str | None = None) -> dict:
    """Build control vector pairs to run alongside a real swap.

    Each control keeps the target template fixed and replaces only the second
    vector, so the interventions differ in *what* is swapped in, not in how many
    directions are touched.

    Args:
        tpl: a `bbq_setup.Templates` instance.
        t_key: vocabulary key of the target group's template (kept in every pair).
        n_key: key of the contrasting group's template (the one replaced).
        layers: layers to build vectors for.
        seed: base seed for the random control; offset per layer.
        unrelated: key of an unrelated group's template, or None to skip.
        nonce: key of a nonce template — a word naming no group, which serves as
            the noise floor (AstralKS, Week 3) — or None to skip.

    Returns:
        `{control_name: {layer: (v_target, v_replacement)}}`, ready to pass to
        `Swap(vecs=...)`.
    """
    out = {}
    out["random"] = {l: (tpl.vec(t_key, l), random_like(tpl.vec(n_key, l), seed + l)) for l in layers}
    # seed must come from stable_seed(), never hash(): see stable_seed
    if unrelated:
        out["unrelated"] = {l: (tpl.vec(t_key, l), tpl.vec(unrelated, l)) for l in layers}
    if nonce:
        out["nonce"] = {l: (tpl.vec(t_key, l), tpl.vec(nonce, l)) for l in layers}
    return out


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def read_margin(model, input_ids: torch.Tensor, first_ids: tuple[int, int],
                intervention=None, layers_out: list[int] | None = None) -> dict:
    """Run one forward pass and read how much the model prefers one answer.

    Args:
        model: a `jlens` model wrapper.
        input_ids: `[1, T]` prompt ids, already on the model's device.
        first_ids: `(target_id, other_id)` — the two answers' first distinguishing
            tokens, the pair whose log-prob difference is the margin.
        intervention: a `Swap`, `Steer` or `Ablate` whose `.handles(input_ids)` is
            registered for this pass, or None for the clean run.
        layers_out: layers whose residual at the final position is returned;
            must include the last layer, which supplies the logits.

    Returns:
        dict with `margin` (log p(target first token) - log p(other)), the two
        log-probs, and `resid` = `{layer: [d] tensor}` for `layers_out`.

    Raises:
        RuntimeError: if the final layer is missing from `layers_out`.
    """
    acts: dict = {}
    hooks = []
    # Order matters: PyTorch runs forward hooks in registration order, so the
    # intervention is registered FIRST and the reads see its output. Reading first
    # would capture each layer before its own patch -- which silently hides any
    # edit made at the final layer, the one that feeds the logits.
    if intervention is not None:
        hooks += intervention.handles(input_ids)
    if layers_out:
        for l in layers_out:
            hooks.append(model.layers[l].register_forward_hook(
                lambda m, i, o, l=l: acts.__setitem__(
                    l, (o if torch.is_tensor(o) else o[0])[0, -1].detach().cpu().float())))
    try:
        model._text_module(input_ids=input_ids, use_cache=False)
        last = model.layers[model.n_layers - 1]
        # final residual comes from the model's own output path
        h = acts.get(model.n_layers - 1)
        if h is None:
            raise RuntimeError("read_margin needs the final layer in layers_out")
        lp = torch.log_softmax(model.unembed(h.to(model.input_device)).float(), dim=-1).cpu()
    finally:
        for hd in hooks:
            hd.remove()
    return {"margin": float(lp[first_ids[0]] - lp[first_ids[1]]),
            "lp_target": float(lp[first_ids[0]]), "lp_other": float(lp[first_ids[1]]),
            "resid": acts}

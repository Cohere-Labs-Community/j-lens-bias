"""Shared helpers for the template-lens pipeline.

Imported by the local vocabulary builder, the cloud generation and template
scripts, and the Week 3 notebook, so a BBQ surface maps to the same vocabulary
key everywhere.
"""

import re

WORD_RE = re.compile(r"[A-Za-z0-9']+")
STOPWORDS = {"a", "an", "the", "of", "in", "on", "at", "with", "and", "or",
             "who", "was", "were", "is", "are", "one", "person", "guy"}

# The template build must match the local environment exactly: Qwen3.5's
# modeling code moves between transformers releases, and the MPS backend moves
# between torch releases, either of which changes the residual stream the
# templates are fitted on. Generation runs in its own env (vLLM pins its own
# torch) because it only emits text.
PINNED = {"python": "3.14", "torch": "2.14.0", "transformers": "5.16.1"}

SUBJECT_MODEL = "Qwen/Qwen3.5-4B"
WRITER_MODEL = "Qwen/Qwen3.6-27B"


def vocab_key(surface: str) -> str:
    """Case-, hyphen- and whitespace-insensitive key.

    BBQ writes the same group several ways ("24 year old" / "24-year-old",
    "middle aged" / "middle-aged"). Those are one concept and get one template.
    """
    return re.sub(r"[-\s]+", " ", surface.strip().lower())


def slug(key: str) -> str:
    """Filesystem-safe name for a vocabulary key."""
    return re.sub(r"[^a-z0-9]+", "_", key).strip("_")


def content_words(phrase: str) -> list[str]:
    """Words that carry the phrase's meaning; never empty."""
    words = WORD_RE.findall(phrase)
    kept = [w for w in words if w.lower() not in STOPWORDS]
    return kept or words


# Parts of a phrase too generic to give the continuation away on their own:
# "this year" does not reveal "78-year-old"; the number does.
GENERIC_PARTS = {"year", "years", "old", "aged"}


def leaks_word(passage: str, phrase: str) -> bool:
    """Does the passage already contain the target, or a content word of it?

    The paper requires passages "never using w itself". For a phrase such as
    "Black lawyer" either word appearing earlier would give the continuation
    away, so both are checked, with a simple plural. Numbers count as content:
    "78" appearing leaks "78-year-old", but "this year" does not.
    """
    text = passage.lower()
    candidates = {phrase.lower(), vocab_key(phrase)}
    for w in content_words(phrase):
        w = w.lower()
        if w in GENERIC_PARTS or (len(w) < 3 and not w.isdigit()):
            continue
        candidates |= {w, w + "s", w + "es"}
    return any(re.search(rf"(?<![a-z0-9]){re.escape(c)}(?![a-z0-9])", text)
               for c in candidates)


def first_token_ids(tokenizer, phrase: str) -> list[int]:
    """Vocab ids for the phrase's first content token, as the next word.

    Leading-space forms only, skipping a whitespace-only leading piece -- the
    resolution Week 2 settled on (' 78' tokenizes as ' ', '7', '8').
    """
    ids = set()
    for variant in {f" {phrase}", f" {phrase[:1].lower()}{phrase[1:]}"}:
        pieces = tokenizer(variant, add_special_tokens=False).input_ids
        pieces = [p for p in pieces if tokenizer.decode([p]).strip()] or pieces
        if pieces:
            ids.add(pieces[0])
    return sorted(ids)

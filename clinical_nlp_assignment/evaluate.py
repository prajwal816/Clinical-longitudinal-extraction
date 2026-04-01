from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from rapidfuzz import fuzz

from .schemas import PatientOutput
from .utils import ConditionKey, normalize_condition_name


@dataclass(frozen=True)
class PRF1:
    precision: float
    recall: float
    f1: float


def _cond_key(p: dict) -> ConditionKey:
    return ConditionKey(
        category=p["category"],
        subcategory=p["subcategory"],
        name_norm=normalize_condition_name(p["condition_name"]),
    )


def _match_score(a: ConditionKey, b: ConditionKey) -> int:
    if a.category != b.category or a.subcategory != b.subcategory:
        return 0
    return fuzz.ratio(a.name_norm, b.name_norm)


def compute_prf1(
    *,
    y_true: PatientOutput,
    y_pred: PatientOutput,
    name_fuzzy_threshold: int = 94,
) -> PRF1:
    """
    Practical metric for dev iteration:
    - requires category+subcategory match
    - condition_name matched via fuzzy ratio
    This is NOT guaranteed to match the hidden evaluation, but provides signal.
    """
    true_keys = [_cond_key(c.model_dump()) for c in y_true.conditions]
    pred_keys = [_cond_key(c.model_dump()) for c in y_pred.conditions]

    matched_true = set()
    matched_pred = set()

    for pi, pk in enumerate(pred_keys):
        best = (-1, -1)
        for ti, tk in enumerate(true_keys):
            if ti in matched_true:
                continue
            s = _match_score(pk, tk)
            if s > best[0]:
                best = (s, ti)
        if best[0] >= name_fuzzy_threshold and best[1] >= 0:
            matched_pred.add(pi)
            matched_true.add(best[1])

    tp = len(matched_pred)
    fp = len(pred_keys) - tp
    fn = len(true_keys) - len(matched_true)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return PRF1(precision=precision, recall=recall, f1=f1)


def macro_average(scores: Iterable[PRF1]) -> PRF1:
    scores = list(scores)
    if not scores:
        return PRF1(0.0, 0.0, 0.0)
    return PRF1(
        precision=sum(s.precision for s in scores) / len(scores),
        recall=sum(s.recall for s in scores) / len(scores),
        f1=sum(s.f1 for s in scores) / len(scores),
    )


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from rapidfuzz import fuzz

from .schemas import Condition, PatientOutput
from .utils import ConditionKey, normalize_condition_name


@dataclass(frozen=True)
class PRF1:
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class DetailedScore:
    """Rich evaluation score for a single patient."""
    condition_prf1: PRF1
    status_accuracy: float  # among matched conditions, % correct status
    onset_accuracy: float   # among matched, % correct onset (exact string)
    onset_partial: float    # among matched, % with partial date overlap
    evidence_recall: float  # among matched, % of GT evidence note_ids covered
    evidence_precision: float  # among matched, ratio of relevant to total evidence


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


def _onset_partial_match(true_onset: str | None, pred_onset: str | None) -> float:
    """Return partial credit for onset date matching.

    - Exact match: 1.0
    - Year matches: 0.5
    - Both null: 1.0
    - One null, other not: 0.0
    """
    if true_onset == pred_onset:
        return 1.0
    if true_onset is None or pred_onset is None:
        return 0.0
    # Check if year matches
    true_parts = true_onset.strip().split()
    pred_parts = pred_onset.strip().split()
    true_year = true_parts[-1] if true_parts else ""
    pred_year = pred_parts[-1] if pred_parts else ""
    if true_year and pred_year and true_year == pred_year:
        return 0.5
    return 0.0


def _match_conditions(
    y_true: PatientOutput,
    y_pred: PatientOutput,
    name_fuzzy_threshold: int = 94,
) -> list[tuple[int, int]]:
    """Return list of (true_idx, pred_idx) matched pairs."""
    true_keys = [_cond_key(c.model_dump()) for c in y_true.conditions]
    pred_keys = [_cond_key(c.model_dump()) for c in y_pred.conditions]

    matched: list[tuple[int, int]] = []
    used_true: set[int] = set()
    used_pred: set[int] = set()

    for pi, pk in enumerate(pred_keys):
        best = (-1, -1)
        for ti, tk in enumerate(true_keys):
            if ti in used_true:
                continue
            s = _match_score(pk, tk)
            if s > best[0]:
                best = (s, ti)
        if best[0] >= name_fuzzy_threshold and best[1] >= 0:
            matched.append((best[1], pi))
            used_true.add(best[1])
            used_pred.add(pi)

    return matched


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
    """
    matched = _match_conditions(y_true, y_pred, name_fuzzy_threshold)

    tp = len(matched)
    fp = len(y_pred.conditions) - tp
    fn = len(y_true.conditions) - tp

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return PRF1(precision=precision, recall=recall, f1=f1)


def compute_detailed_score(
    *,
    y_true: PatientOutput,
    y_pred: PatientOutput,
    name_fuzzy_threshold: int = 94,
) -> DetailedScore:
    """Compute all evaluation dimensions for a patient."""
    matched = _match_conditions(y_true, y_pred, name_fuzzy_threshold)

    tp = len(matched)
    fp = len(y_pred.conditions) - tp
    fn = len(y_true.conditions) - tp

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    prf1 = PRF1(precision=precision, recall=recall, f1=f1)

    if not matched:
        return DetailedScore(
            condition_prf1=prf1,
            status_accuracy=0.0,
            onset_accuracy=0.0,
            onset_partial=0.0,
            evidence_recall=0.0,
            evidence_precision=0.0,
        )

    # Status accuracy
    status_correct = 0
    for ti, pi in matched:
        if y_true.conditions[ti].status == y_pred.conditions[pi].status:
            status_correct += 1
    status_acc = status_correct / len(matched)

    # Onset accuracy (exact and partial)
    onset_exact = 0
    onset_partial_sum = 0.0
    for ti, pi in matched:
        true_onset = y_true.conditions[ti].onset
        pred_onset = y_pred.conditions[pi].onset
        if true_onset == pred_onset:
            onset_exact += 1
        onset_partial_sum += _onset_partial_match(true_onset, pred_onset)
    onset_acc = onset_exact / len(matched)
    onset_partial_acc = onset_partial_sum / len(matched)

    # Evidence recall and precision
    ev_recall_sum = 0.0
    ev_precision_sum = 0.0
    for ti, pi in matched:
        true_note_ids = {e.note_id for e in y_true.conditions[ti].evidence}
        pred_note_ids = {e.note_id for e in y_pred.conditions[pi].evidence}

        if true_note_ids:
            ev_recall_sum += len(true_note_ids & pred_note_ids) / len(true_note_ids)
        else:
            ev_recall_sum += 1.0  # no GT evidence to miss

        if pred_note_ids:
            ev_precision_sum += len(true_note_ids & pred_note_ids) / len(pred_note_ids)
        else:
            ev_precision_sum += 0.0

    ev_recall = ev_recall_sum / len(matched)
    ev_precision = ev_precision_sum / len(matched)

    return DetailedScore(
        condition_prf1=prf1,
        status_accuracy=status_acc,
        onset_accuracy=onset_acc,
        onset_partial=onset_partial_acc,
        evidence_recall=ev_recall,
        evidence_precision=ev_precision,
    )


def macro_average(scores: Iterable[PRF1]) -> PRF1:
    scores = list(scores)
    if not scores:
        return PRF1(0.0, 0.0, 0.0)
    return PRF1(
        precision=sum(s.precision for s in scores) / len(scores),
        recall=sum(s.recall for s in scores) / len(scores),
        f1=sum(s.f1 for s in scores) / len(scores),
    )


def macro_average_detailed(scores: Iterable[DetailedScore]) -> dict[str, float]:
    """Compute macro averages over all detailed score dimensions."""
    scores = list(scores)
    if not scores:
        return {}
    n = len(scores)
    return {
        "precision": sum(s.condition_prf1.precision for s in scores) / n,
        "recall": sum(s.condition_prf1.recall for s in scores) / n,
        "f1": sum(s.condition_prf1.f1 for s in scores) / n,
        "status_accuracy": sum(s.status_accuracy for s in scores) / n,
        "onset_accuracy": sum(s.onset_accuracy for s in scores) / n,
        "onset_partial": sum(s.onset_partial for s in scores) / n,
        "evidence_recall": sum(s.evidence_recall for s in scores) / n,
        "evidence_precision": sum(s.evidence_precision for s in scores) / n,
    }

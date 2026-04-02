"""
Evaluation framework for clinical condition extraction.

Computes multi-dimensional metrics matching the official evaluation criteria:
- Condition identification (P/R/F1)
- Status accuracy
- Onset/date accuracy
- Evidence quality (recall and precision)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from rapidfuzz import fuzz

from .schemas import Condition, PatientOutput
from .utils import ConditionKey, normalize_condition_name


@dataclass(frozen=True)
class PRF1:
    precision: float
    recall: float
    f1: float


@dataclass
class DetailedScore:
    """Multi-dimensional evaluation for a single patient."""
    patient_id: str = ""
    # Condition identification
    condition_precision: float = 0.0
    condition_recall: float = 0.0
    condition_f1: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    # Status accuracy (among matched conditions)
    status_correct: int = 0
    status_total: int = 0
    # Onset accuracy (among matched conditions)
    onset_exact_correct: int = 0
    onset_partial_correct: int = 0
    onset_total: int = 0
    # Evidence quality (among matched conditions)
    evidence_note_recall_sum: float = 0.0  # fraction of GT evidence note_ids covered
    evidence_note_precision_sum: float = 0.0  # fraction of predicted evidence note_ids that are in GT
    evidence_count: int = 0  # number of matched conditions with evidence comparison

    @property
    def status_accuracy(self) -> float:
        return self.status_correct / self.status_total if self.status_total else 0.0

    @property
    def onset_exact_accuracy(self) -> float:
        return self.onset_exact_correct / self.onset_total if self.onset_total else 0.0

    @property
    def onset_partial_accuracy(self) -> float:
        return self.onset_partial_correct / self.onset_total if self.onset_total else 0.0

    @property
    def evidence_note_recall(self) -> float:
        return self.evidence_note_recall_sum / self.evidence_count if self.evidence_count else 0.0

    @property
    def evidence_note_precision(self) -> float:
        return (
            self.evidence_note_precision_sum / self.evidence_count if self.evidence_count else 0.0
        )

    # Aliases used by train/eval helpers and report text.
    @property
    def evidence_recall(self) -> float:
        return self.evidence_note_recall

    @property
    def evidence_precision(self) -> float:
        return self.evidence_note_precision

    @property
    def onset_accuracy(self) -> float:
        return self.onset_exact_accuracy

    @property
    def onset_partial(self) -> float:
        return self.onset_partial_accuracy

    def summary_line(self) -> str:
        return (
            f"{self.patient_id}: "
            f"P={self.condition_precision:.3f} R={self.condition_recall:.3f} F1={self.condition_f1:.3f} | "
            f"Status={self.status_accuracy:.3f} | "
            f"Onset(exact)={self.onset_exact_accuracy:.3f} Onset(partial)={self.onset_partial_accuracy:.3f} | "
            f"EvidRecall={self.evidence_note_recall:.3f} EvidPrec={self.evidence_note_precision:.3f}"
        )


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


def _normalize_date(d: str | None) -> str | None:
    """Normalize date string for comparison."""
    if d is None:
        return None
    d = d.strip()
    if not d:
        return None
    return d


def _dates_partial_match(pred: str | None, true: str | None) -> bool:
    """Check if dates partially match (year matches at minimum)."""
    if pred is None and true is None:
        return True
    if pred is None or true is None:
        return False
    # Extract years
    pred_years = re.findall(r"\d{4}", pred)
    true_years = re.findall(r"\d{4}", true)
    if pred_years and true_years:
        return pred_years[-1] == true_years[-1]
    return pred.strip().lower() == true.strip().lower()


def compute_detailed_score(
    *,
    y_true: PatientOutput,
    y_pred: PatientOutput,
    name_fuzzy_threshold: int = 88,
) -> DetailedScore:
    """Compute multi-dimensional evaluation metrics for one patient."""
    score = DetailedScore(patient_id=y_true.patient_id)

    true_conds = [(i, _cond_key(c.model_dump()), c) for i, c in enumerate(y_true.conditions)]
    pred_conds = [(i, _cond_key(c.model_dump()), c) for i, c in enumerate(y_pred.conditions)]

    matched_true: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[tuple[Condition, Condition]] = []  # (true_cond, pred_cond)

    # Greedy matching: for each predicted, find best unmatched true
    for pi, pk, pc in pred_conds:
        best = (-1, -1, None)
        for ti, tk, tc in true_conds:
            if ti in matched_true:
                continue
            s = _match_score(pk, tk)
            if s > best[0]:
                best = (s, ti, tc)
        if best[0] >= name_fuzzy_threshold and best[1] >= 0:
            matched_pred.add(pi)
            matched_true.add(best[1])
            matches.append((best[2], pc))  # type: ignore

    tp = len(matches)
    fp = len(pred_conds) - tp
    fn = len(true_conds) - len(matched_true)

    score.tp = tp
    score.fp = fp
    score.fn = fn
    score.condition_precision = tp / (tp + fp) if (tp + fp) else 0.0
    score.condition_recall = tp / (tp + fn) if (tp + fn) else 0.0
    score.condition_f1 = (
        (2 * score.condition_precision * score.condition_recall /
         (score.condition_precision + score.condition_recall))
        if (score.condition_precision + score.condition_recall) else 0.0
    )

    # Evaluate matched conditions on status, onset, evidence
    for true_c, pred_c in matches:
        # Status
        score.status_total += 1
        if pred_c.status == true_c.status:
            score.status_correct += 1

        # Onset
        true_onset = _normalize_date(true_c.onset)
        pred_onset = _normalize_date(pred_c.onset)
        score.onset_total += 1
        if true_onset == pred_onset:
            score.onset_exact_correct += 1
            score.onset_partial_correct += 1
        elif _dates_partial_match(pred_onset, true_onset):
            score.onset_partial_correct += 1

        # Evidence note recall: fraction of GT evidence note_ids present in pred
        true_note_ids = {ev.note_id for ev in true_c.evidence}
        pred_note_ids = {ev.note_id for ev in pred_c.evidence}
        recall = len(true_note_ids & pred_note_ids) / len(true_note_ids) if true_note_ids else 0.0
        precision = (
            len(true_note_ids & pred_note_ids) / len(pred_note_ids) if pred_note_ids else 0.0
        )
        score.evidence_note_recall_sum += recall
        score.evidence_note_precision_sum += precision
        score.evidence_count += 1

    return score


def compute_prf1(
    *,
    y_true: PatientOutput,
    y_pred: PatientOutput,
    name_fuzzy_threshold: int = 88,
) -> PRF1:
    """Backward-compatible P/R/F1 computation."""
    ds = compute_detailed_score(
        y_true=y_true, y_pred=y_pred, name_fuzzy_threshold=name_fuzzy_threshold
    )
    return PRF1(precision=ds.condition_precision, recall=ds.condition_recall, f1=ds.condition_f1)


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
    """Compute macro-averaged metrics across patients."""
    items = list(scores)
    n = len(items)
    if not n:
        return {}
    return {
        "condition_precision": sum(s.condition_precision for s in items) / n,
        "condition_recall": sum(s.condition_recall for s in items) / n,
        "condition_f1": sum(s.condition_f1 for s in items) / n,
        "status_accuracy": sum(s.status_accuracy for s in items) / n,
        "onset_exact_accuracy": sum(s.onset_exact_accuracy for s in items) / n,
        "onset_partial_accuracy": sum(s.onset_partial_accuracy for s in items) / n,
        "evidence_note_recall": sum(s.evidence_note_recall for s in items) / n,
        "evidence_note_precision": sum(s.evidence_note_precision for s in items) / n,
        # Aliases expected by train.py and report text.
        "evidence_recall": sum(s.evidence_note_recall for s in items) / n,
        "evidence_precision": sum(s.evidence_note_precision for s in items) / n,
        "onset_accuracy": sum(s.onset_exact_accuracy for s in items) / n,
        "onset_partial": sum(s.onset_partial_accuracy for s in items) / n,
    }

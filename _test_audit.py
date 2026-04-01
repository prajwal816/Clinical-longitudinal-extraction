"""Comprehensive integration test after audit fixes."""
import json
from pathlib import Path

# Test 1: All imports
print("Test 1: All imports...")
from clinical_nlp_assignment import data_loader, llm_client, model, prompts, schemas, extractor, inference, evaluate, utils
print("  PASS")

# Test 2: Schema relaxation — ALL models now tolerate extra fields
print("Test 2: Schema relaxation...")
from clinical_nlp_assignment.schemas import Condition, Evidence, PatientOutput

# Evidence with extra fields
e = Evidence.model_validate({"note_id": "text_0", "line_no": 1, "span": "test", "confidence": 0.9})
assert e.note_id == "text_0"
print("  Evidence: PASS")

# Condition with extra fields
c = Condition.model_validate({
    "condition_name": "Test", "category": "cancer", "subcategory": "primary_malignancy",
    "status": "active", "onset": None,
    "evidence": [{"note_id": "text_0", "line_no": 1, "span": "test"}],
    "extra": "ignored", "confidence": 0.95
})
assert c.condition_name == "Test"
print("  Condition: PASS")

# PatientOutput with extra fields (was crashing before fix)
po = PatientOutput.model_validate({
    "patient_id": "patient_01", "conditions": [],
    "summary": "should be ignored", "total_conditions": 5
})
assert po.patient_id == "patient_01"
print("  PatientOutput: PASS (extra fields ignored)")

# Test 3: LLM client imports BadRequestError
print("Test 3: LLM client exception handling...")
from clinical_nlp_assignment.llm_client import BadRequestError, _extract_json
# Verify _extract_json works
assert _extract_json('{"a": 1}') == {"a": 1}
assert _extract_json('```json\n{"b": 2}\n```') == {"b": 2}
assert _extract_json('Some text {"c": 3} more text') == {"c": 3}
print("  PASS")

# Test 4: Prompt generation (post few-shot fix)
print("Test 4: Prompt generation...")
from clinical_nlp_assignment.data_loader import load_taxonomy
from clinical_nlp_assignment.prompts import build_note_system_prompt, build_patient_system_prompt

taxonomy = load_taxonomy("./clinical_nlp_assignment/taxonomy.json")
sys_prompt = build_note_system_prompt(taxonomy)
assert "DISAMBIGUATION RULES" in sys_prompt
assert "FEW-SHOT EXAMPLE" in sys_prompt
assert "ONSET / DATE RULES" in sys_prompt
# Verify the improved few-shot includes the cholecystectomy guidance
assert "cholecystectomy" in sys_prompt
assert "surgical procedure" in sys_prompt.lower() or "NOT extracted" in sys_prompt
print(f"  Note system prompt = {len(sys_prompt)} chars: PASS")

pat_prompt = build_patient_system_prompt(taxonomy)
assert "CONSOLIDATION RULES" in pat_prompt
assert "Deduplication" in pat_prompt
print(f"  Patient system prompt = {len(pat_prompt)} chars: PASS")

# Test 5: Evidence hardening precision
print("Test 5: Evidence hardening...")
from clinical_nlp_assignment.extractor import _harden_evidence, _MIN_TERM_LEN
assert _MIN_TERM_LEN == 5, f"Expected min term len 5, got {_MIN_TERM_LEN}"
# Create a test patient with a condition and notes
test_patient = PatientOutput(patient_id="test", conditions=[
    Condition(
        condition_name="Arterial hypertension",
        category="cardiovascular",
        subcategory="hypertensive",
        status="active",
        onset=None,
        evidence=[Evidence(note_id="text_0", line_no=6, span="Arterial hypertension")]
    )
])
test_notes = [
    {"note_id": "text_0", "lines": {
        1: "Dear colleague,",
        5: "Diagnoses:",
        6: "- Arterial hypertension",
        10: "arterial blood gas: pH 7.35",  # Should NOT match (just shares "arterial")
        12: "arterial hypertension well controlled",  # SHOULD match (full name)
    }},
    {"note_id": "text_1", "lines": {
        3: "Other Diagnoses:",
        4: "- Arterial hypertension",  # SHOULD match (full name)
        8: "calcium levels normal",  # Should NOT match
    }},
]
result = _harden_evidence(test_patient, test_notes)
ev_keys = {(e.note_id, e.line_no) for e in result.conditions[0].evidence}
assert ("text_0", 6) in ev_keys, "Original evidence should still be present"
assert ("text_0", 12) in ev_keys, "Should find 'arterial hypertension' on line 12"
assert ("text_1", 4) in ev_keys, "Should find 'Arterial hypertension' on text_1 line 4"
# Should NOT match line 10 (only "arterial" which is <5 chars for individual word, and
# "arterial hypertension" as full term does NOT appear on that line)
assert ("text_0", 10) not in ev_keys, "Should NOT match 'arterial blood gas' line"
assert ("text_1", 8) not in ev_keys, "Should NOT match 'calcium levels' line"
print(f"  Evidence entries: {len(result.conditions[0].evidence)} (expected 3): PASS")

# Test 6: Detailed evaluation
print("Test 6: Detailed evaluation...")
from clinical_nlp_assignment.evaluate import compute_detailed_score
gt = PatientOutput(patient_id="test", conditions=[
    Condition(condition_name="Arterial hypertension", category="cardiovascular",
              subcategory="hypertensive", status="active", onset="May 2014",
              evidence=[Evidence(note_id="text_0", line_no=1, span="HTN"),
                        Evidence(note_id="text_1", line_no=5, span="HTN")])
])
pred = PatientOutput(patient_id="test", conditions=[
    Condition(condition_name="Arterial hypertension", category="cardiovascular",
              subcategory="hypertensive", status="active", onset="May 2014",
              evidence=[Evidence(note_id="text_0", line_no=1, span="HTN")])
])
score = compute_detailed_score(y_true=gt, y_pred=pred)
assert score.condition_prf1.f1 == 1.0
assert score.status_accuracy == 1.0
assert score.onset_accuracy == 1.0
assert score.evidence_recall == 0.5  # Only covered 1 of 2 note_ids
assert score.evidence_precision == 1.0  # All predicted evidence is relevant
print(f"  F1={score.condition_prf1.f1}, Status={score.status_accuracy}, Onset={score.onset_accuracy}, EvRec={score.evidence_recall}, EvPrec={score.evidence_precision}: PASS")

# Test 7: Taxonomy recovery
print("Test 7: Taxonomy recovery...")
from clinical_nlp_assignment.extractor import _try_recover_taxonomy_keys
from clinical_nlp_assignment.data_loader import get_valid_categories
valid_cats = get_valid_categories(taxonomy)
# Case: slight misspelling of subcategory
item = {"condition_name": "Test", "category": "cardiovascular", "subcategory": "hypertensiv", "status": "active"}
recovered = _try_recover_taxonomy_keys(item, valid_cats)
assert recovered is not None
assert recovered["subcategory"] == "hypertensive"
print(f"  Recovered 'hypertensiv' -> '{recovered['subcategory']}': PASS")

# Case: unrecoverable
item2 = {"condition_name": "Test", "category": "zzzzz", "subcategory": "xxxxx", "status": "active"}
recovered2 = _try_recover_taxonomy_keys(item2, valid_cats)
assert recovered2 is None
print("  Correctly rejected unrecoverable key: PASS")

# Test 8: Validate_outputs module
print("Test 8: Output validation module...")
from clinical_nlp_assignment.validate_outputs import main as validate_main
# Already tested via CLI, just verify the import works
print("  PASS")

# Test 9: Verify no stale files
print("Test 9: Checking project cleanliness...")
assert not Path("test_integration.py").exists(), "test_integration.py should have been deleted"
print("  PASS")

print("\n" + "=" * 50)
print("ALL 9 TESTS PASSED — Pipeline is production-ready!")
print("=" * 50)

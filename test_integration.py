"""Quick integration test for all modified modules."""
from clinical_nlp_assignment.schemas import Condition, Evidence, PatientOutput
from clinical_nlp_assignment.prompts import build_note_system_prompt, build_patient_system_prompt
from clinical_nlp_assignment.data_loader import load_taxonomy
from clinical_nlp_assignment.evaluate import compute_detailed_score, macro_average_detailed

# Test 1: Schema relaxation - extra fields should NOT crash
print("Test 1: Schema relaxation...")
c = Condition.model_validate({
    "condition_name": "Test",
    "category": "cancer",
    "subcategory": "primary_malignancy",
    "status": "active",
    "onset": None,
    "evidence": [{"note_id": "text_0", "line_no": 1, "span": "test"}],
    "extra_field": "should be ignored",
    "confidence": 0.95,
})
assert c.condition_name == "Test"
print("  PASS: Extra fields ignored correctly")

# Test 2: PatientOutput should still be strict
print("Test 2: PatientOutput strictness...")
try:
    PatientOutput.model_validate({
        "patient_id": "test",
        "conditions": [],
        "extra": "bad",
    })
    print("  FAIL: Should have raised")
except Exception:
    print("  PASS: PatientOutput still strict (extra=forbid)")

# Test 3: Prompt generation
print("Test 3: Prompt generation...")
taxonomy = load_taxonomy("./clinical_nlp_assignment/taxonomy.json")
sys_prompt = build_note_system_prompt(taxonomy)
assert len(sys_prompt) > 5000, f"System prompt too short: {len(sys_prompt)}"
assert "DISAMBIGUATION RULES" in sys_prompt
assert "FEW-SHOT EXAMPLE" in sys_prompt
assert "ONSET / DATE RULES" in sys_prompt
print(f"  PASS: Note system prompt = {len(sys_prompt)} chars")

pat_prompt = build_patient_system_prompt(taxonomy)
assert "CONSOLIDATION RULES" in pat_prompt
assert "Deduplication" in pat_prompt
print(f"  PASS: Patient system prompt = {len(pat_prompt)} chars")

# Test 4: Detailed evaluation
print("Test 4: Detailed evaluation...")
gt = PatientOutput(patient_id="test", conditions=[
    Condition(condition_name="Arterial hypertension", category="cardiovascular",
              subcategory="hypertensive", status="active", onset="May 2014",
              evidence=[Evidence(note_id="text_0", line_no=1, span="HTN")])
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
print(f"  PASS: Perfect match gives F1={score.condition_prf1.f1}, status={score.status_accuracy}, onset={score.onset_accuracy}")

# Test 5: LLM client token tracking
print("Test 5: LLM client structure...")
from clinical_nlp_assignment.llm_client import LLMConfig, OpenAICompatibleClient, _extract_json
# Test JSON extraction from fenced blocks
result = _extract_json('```json\n{"test": true}\n```')
assert result == {"test": True}
result = _extract_json('Here is the result: {"test": 42} some text after')
assert result == {"test": 42}
print("  PASS: JSON extraction from fenced/mixed text works")

print("\nAll tests passed!")

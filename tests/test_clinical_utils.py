from clinical_utils import (
    extract_hauser_state,
    extract_moca_score,
    extract_summary,
    is_high_risk_response,
    mask_patient_id,
    normalize_whatsapp_number,
    strip_internal_tags,
)


def test_normalize_whatsapp_number_adds_prefix():
    assert normalize_whatsapp_number("+447123456789") == "whatsapp:+447123456789"


def test_mask_patient_id_masks_last_four():
    assert mask_patient_id("whatsapp:+447123456789") == "***6789"


def test_strip_internal_tags_removes_suffix_tags():
    text = "You are doing well today. [SUMMARY] Symptom: Mood, Severity: None, Score: 0, Context: Stable."
    assert strip_internal_tags(text) == "You are doing well today."


def test_extract_summary_parses_score_and_severity():
    text = "[SUMMARY] Symptom: Stiffness, Severity: High, Score: 3, Context: Sudden OFF episode."
    summary = extract_summary(text)
    assert summary is not None
    assert summary["symptom"] == "Stiffness"
    assert summary["severity"] == "High"
    assert summary["score"] == 3


def test_extract_moca_and_hauser():
    assert extract_moca_score("[MOCA] Score: 2/3, Context: Clock hands incorrect.") == 2
    assert extract_hauser_state("[HAUSER] State: dyskinesia, Context: Evening movement.") == "DYSKINESIA"


def test_high_risk_detection_for_high_summary():
    text = "[SUMMARY] Symptom: Fall, Severity: High, Score: 3, Context: Fell near stairs."
    assert is_high_risk_response(text) is True


def test_high_risk_detection_for_zero_moca():
    text = "Feedback text. [MOCA] Score: 0/3, Context: Severe cognitive decline."
    assert is_high_risk_response(text) is True


def test_non_risk_response():
    text = "[SUMMARY] Symptom: Mood, Severity: None, Score: 0, Context: Stable and positive."
    assert is_high_risk_response(text) is False

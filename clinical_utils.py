import re

SUMMARY_PATTERN = re.compile(
    r"\[SUMMARY\]\s*Symptom:\s*(?P<symptom>.*?),\s*Severity:\s*(?P<severity>None|Low|Medium|High),\s*Score:\s*(?P<score>[0-3]),\s*Context:\s*(?P<context>.*)",
    re.IGNORECASE | re.DOTALL,
)
MOCA_PATTERN = re.compile(r"\[MOCA\]\s*Score:\s*(?P<score>[0-3])/3", re.IGNORECASE)
HAUSER_PATTERN = re.compile(
    r"\[HAUSER\]\s*State:\s*(?P<state>ON|OFF|DYSKINESIA|ASLEEP)",
    re.IGNORECASE,
)


def normalize_whatsapp_number(number):
    if not number:
        return None
    cleaned = number.strip()
    if not cleaned:
        return None
    if not cleaned.startswith("whatsapp:"):
        cleaned = f"whatsapp:{cleaned}"
    return cleaned


def mask_patient_id(patient_id):
    if not patient_id:
        return "***"
    cleaned = patient_id.replace("whatsapp:", "")
    cleaned = cleaned.strip()
    if len(cleaned) < 4:
        return "***"
    return f"***{cleaned[-4:]}"


def strip_internal_tags(text):
    if not text:
        return ""
    earliest_tag_pos = len(text)
    for tag in ("[SUMMARY]", "[PROFILE]", "[HAUSER]", "[MOCA]"):
        pos = text.find(tag)
        if pos != -1 and pos < earliest_tag_pos:
            earliest_tag_pos = pos
    return text[:earliest_tag_pos].strip()


def extract_summary(text):
    if not text:
        return None
    match = SUMMARY_PATTERN.search(text)
    if not match:
        return None
    data = match.groupdict()
    data["score"] = int(data["score"])
    data["severity"] = data["severity"].title()
    return data


def extract_moca_score(text):
    if not text:
        return None
    match = MOCA_PATTERN.search(text)
    if not match:
        return None
    return int(match.group("score"))


def extract_hauser_state(text):
    if not text:
        return None
    match = HAUSER_PATTERN.search(text)
    if not match:
        return None
    return match.group("state").upper()


def is_high_risk_response(text):
    summary = extract_summary(text)
    if summary and (summary["severity"] == "High" or summary["score"] >= 3):
        return True

    moca_score = extract_moca_score(text)
    if moca_score == 0:
        return True

    return False

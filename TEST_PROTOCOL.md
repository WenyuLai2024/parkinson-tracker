# System Test Protocol (Dissertation Evidence)

## Purpose
Provide repeatable evidence that the system works end-to-end and that core evaluation outputs can be reproduced.

## Pre-conditions
- `.env` is configured with Twilio, OpenAI, and Supabase credentials.
- Dashboard auth is set:
  - `DASHBOARD_USERNAME`
  - `DASHBOARD_PASSWORD`
- Dependencies installed with `pip install -r requirements.txt`.

## Test A: Build/Runtime Sanity
1. Run:
   - `python -m compileall app.py app_ai.py dashboard.py clinical_utils.py test_ai.py`
2. Expected result:
   - No syntax errors.

## Test B: Webhook + Text Path
1. Start backend: `python app.py`
2. Send a WhatsApp text symptom message.
3. Expected result:
   - Patient receives reply.
   - Reply excludes internal tags.
   - Record is inserted into `chat_history`.

## Test C: Voice Path
1. Send a WhatsApp voice note.
2. Expected result:
   - Voice is transcribed.
   - AI response is generated from transcript.
   - Interaction is logged.

## Test D: Image Path (MoCA)
1. Send a clock-drawing image.
2. Expected result:
   - Model analyses image.
   - Response contains MoCA tagging internally.
   - Dashboard trend updates after refresh.

## Test E: Caregiver Access Boundary
1. Send caregiver query from configured caregiver number.
2. Expected result:
   - Summary includes linked-patient updates only.
   - No cross-patient leakage.
   - Identifiers are masked in summary context.

## Test F: Alert Trigger
1. Send a high-severity symptom message.
2. Expected result:
   - Structured risk logic triggers caregiver alert.
   - Alert event appears in backend logs.

## Test G: Dashboard Auth
1. Launch dashboard: `python -m streamlit run dashboard.py`
2. Expected result:
   - Login prompt appears when auth env vars are set.
   - Access granted only with valid credentials.

## Test H: Offline Evaluation Reproducibility
1. Run: `python test_ai.py`
2. Expected result:
   - `test_report_results.csv` updated.
   - `confusion_matrix_evaluation.png` generated.
   - Accuracy printed in terminal.

## Test I: Signature Validation
1. Ensure `ENABLE_TWILIO_SIGNATURE_VALIDATION=true`.
2. Send a non-Twilio POST request to `/sms` without `X-Twilio-Signature`.
3. Expected result:
   - HTTP 403 rejected.

## Test J: Scheduler Duplicate Guard
1. Start multiple worker processes (or multiple local app instances).
2. Expected result:
   - Only one process prints scheduler lock acquired and sends proactive check-ins.
   - Other workers print scheduler start skipped.

## Evidence Checklist for Dissertation Appendix
- Screenshot of backend running and receiving webhook payload.
- Screenshot of dashboard login page.
- Screenshot of patient trend view (Hauser + MoCA + symptom chart).
- Copy of evaluation accuracy output.
- Confusion matrix figure.
- Git commit hash of tested version.

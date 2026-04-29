# Parkinson Bot (Dissertation Project)

## Overview
This project is a cloud-native Parkinson's symptom tracking assistant:
- WhatsApp + Twilio for patient/caregiver interaction
- OpenAI (`gpt-4o`, Whisper, vision) for symptom extraction and conversational support
- Supabase PostgreSQL for longitudinal clinical logs
- Streamlit dashboard for clinician-facing trend analysis and report export

## Architecture
### 1) Ingestion and Conversation Layer
- `app.py`
  - Twilio webhook endpoint: `/sms`
  - Voice transcription (`whisper-1`) and image handling
  - Proactive check-in scheduler (APScheduler)
  - Emergency alert forwarding to caregiver

### 2) AI Orchestration Layer
- `app_ai.py`
  - Maintains system prompt and multi-turn context injection
  - Calls OpenAI chat model and writes structured outputs to DB
  - Supports multimodal prompt assembly for image uploads

### 3) Clinical Tag Parsing Layer
- `clinical_utils.py`
  - Centralized parsing for `[SUMMARY]`, `[HAUSER]`, `[MOCA]`
  - Shared utilities for risk detection and tag sanitization

### 4) Visualization and Reporting Layer
- `dashboard.py`
  - Pulls `chat_history` + `patient_profiles`
  - Builds Hauser/MoCA/symptom trend charts
  - Generates AI summary and PDF export

## Data Flow (High-Level)
1. Patient sends WhatsApp text/voice/image.
2. `app.py` processes media and calls `get_ai_response(...)`.
3. `app_ai.py` generates response with structured tags and logs to `chat_history`.
4. `app.py` strips internal tags before replying to patient.
5. `dashboard.py` reads DB logs and renders longitudinal trends.
6. High-risk tagged events trigger caregiver alerts.

## Required Environment Variables
- `OPENAI_API_KEY`
- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_PHONE_NUMBER`
- `TWILIO_WEBHOOK_URL` (recommended for strict signature validation)
- `SUPABASE_DB_URL`
- `CAREGIVER_PHONE_NUMBER`
- `DASHBOARD_PUBLIC_URL` (optional but recommended for caregiver alert links)
- `DASHBOARD_USERNAME` (recommended for dashboard access control)
- `DASHBOARD_PASSWORD_HASH` (recommended for dashboard access control; if set, plaintext password is ignored)

Optional runtime controls:
- `DB_CONNECT_TIMEOUT_SECONDS` (default: `10`)
- `MEDIA_DOWNLOAD_TIMEOUT_SECONDS` (default: `20`)
- `PROACTIVE_INTERVAL_MINUTES` (default: `60`)
- `PROACTIVE_CHECKIN_TIMEZONE` (default: `Europe/London`)
- `CAREGIVER_CONTEXT_LOG_LIMIT` (default: `10`)
- `ENABLE_PROACTIVE_CHECKIN` (default: `true`)
- `ENABLE_TWILIO_SIGNATURE_VALIDATION` (default: `true`)
- `DIALOGUE_CONTEXT_TURNS` (default: `3`)
- `SCHEDULER_REQUIRE_LEADER_LOCK` (default: `true`)
- `SCHEDULER_LEADER_PORT` (default: `47200`)
- `DASHBOARD_SESSION_TIMEOUT_MINUTES` (default: `60`)
- `DASHBOARD_REQUIRE_AUTH` (default: `true`; when true and auth config is missing, dashboard blocks access)
- `DASHBOARD_LOOKBACK_DAYS` (default: `365`)
- `DASHBOARD_HISTORY_LIMIT` (default: `5000`)
- `FLASK_DEBUG` (default: `false`; keep `false` in production)

## Run Locally
```bash
pip install -r requirements.txt
python app.py
python -m streamlit run dashboard.py
```

## Notes for Dissertation
- `test_ai.py` + `test_dataset.csv` provide evaluation workflow for extraction accuracy.
- `generate_ppmi_baseline.py` creates a synthetic cohort baseline for trend comparison.
- `simulate_data_cloud.py` can be destructive. Use one of:
  - `python simulate_data_cloud.py --confirm-reset` (clear and reseed)
  - `python simulate_data_cloud.py --skip-reset` (append only)
- Webhook security is enforced through Twilio request signature validation.
- Proactive scheduler has a single-process leader lock to avoid duplicate dispatch in multi-worker deployments.
- Conversation context is pulled from cloud DB, so context survives process restarts.

## Quick Validation
```bash
# 1) Basic compile check
python -m compileall app.py app_ai.py dashboard.py clinical_utils.py test_ai.py

# 2) Run backend
python app.py

# 3) Run dashboard
python -m streamlit run dashboard.py

# 4) Run extraction evaluation
python test_ai.py
```

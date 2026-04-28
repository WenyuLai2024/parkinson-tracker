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
- `SUPABASE_DB_URL`
- `CAREGIVER_PHONE_NUMBER`

Optional runtime controls:
- `DB_CONNECT_TIMEOUT_SECONDS` (default: `10`)
- `MEDIA_DOWNLOAD_TIMEOUT_SECONDS` (default: `20`)
- `PROACTIVE_INTERVAL_MINUTES` (default: `60`)
- `CAREGIVER_CONTEXT_LOG_LIMIT` (default: `10`)
- `ENABLE_PROACTIVE_CHECKIN` (default: `true`)

## Run Locally
```bash
pip install -r requirements.txt
python app.py
streamlit run dashboard.py
```

## Notes for Dissertation
- `test_ai.py` + `test_dataset.csv` provide evaluation workflow for extraction accuracy.
- `generate_ppmi_baseline.py` creates a synthetic cohort baseline for trend comparison.
- `simulate_data_cloud.py` is destructive (clears DB tables) and should only be used in test environments.

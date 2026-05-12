# Parkinson Tracker (Dissertation Project)

LLM-based conversational symptom tracking prototype for Parkinson's disease.

This repository contains an end-to-end engineering system:
- WhatsApp intake through Twilio webhooks
- Multimodal processing (text, voice, image)
- Structured clinical tag extraction with OpenAI models
- Supabase PostgreSQL persistence
- Streamlit dashboard for longitudinal trend review and reporting

## Key Capabilities
- Conversational intake from patients in a familiar channel (WhatsApp)
- Structured internal protocol: `[SUMMARY]`, `[HAUSER]`, `[MOCA]`, `[PDQ39]`, `[PROFILE]`
- Asymmetric dual-track access (patient write path + caregiver read-only summarisation path)
- High-risk alerting from structured severity thresholds
- Reproducible offline evaluation pipeline with confusion matrix outputs

## Repository Structure
- `app.py`: webhook ingestion, routing, proactive scheduler, alert dispatch
- `app_ai.py`: LLM orchestration, context handling, structured extraction
- `clinical_utils.py`: shared parsers and risk helpers
- `dashboard.py`: clinical visualisation and report export
- `test_ai.py`: automated extraction evaluation
- `test_dataset_mds04_native.csv`: primary 300-sample evaluation set
- `docs/`: deployment, architecture, testing, and security documentation
- `scripts/quick_check.ps1`: local environment and project health checks

## Quick Start (Local)
### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Configure environment
```bash
copy .env.example .env
```
Then fill required values in `.env`:
- `OPENAI_API_KEY`
- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_PHONE_NUMBER`
- `SUPABASE_DB_URL`

### 3) Run health check
```powershell
powershell -ExecutionPolicy Bypass -File scripts/quick_check.ps1
```

### 4) Run services
```bash
python app.py
python -m streamlit run dashboard.py
```

## Reproduce Evaluation Outputs
Primary dataset run:
```bash
python test_ai.py --dataset test_dataset_mds04_native.csv --output-prefix mds04_native_
```

This produces:
- `mds04_native_test_report_results.csv`
- `mds04_native_confusion_matrix_evaluation.png`

## Documentation
- Deployment: `docs/DEPLOYMENT.md`
- Architecture: `docs/ARCHITECTURE.md`
- Testing and reproducibility: `docs/TESTING.md`
- Security and privacy controls: `docs/SECURITY.md`

## Safety Boundary
This project is an engineering feasibility prototype, not a diagnostic system.
Outputs support tracking and communication, not clinical decision replacement.

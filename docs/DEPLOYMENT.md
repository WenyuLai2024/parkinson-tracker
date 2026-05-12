# Deployment Guide

This document describes practical deployment paths for the Parkinson Tracker prototype.

## 1) Local Deployment (Windows or macOS/Linux)

### Prerequisites
- Python 3.10+
- `pip`
- Valid credentials for OpenAI, Twilio, and Supabase

### Steps
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Create local environment file:
```bash
copy .env.example .env
```
3. Fill required values in `.env`:
- `OPENAI_API_KEY`
- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_PHONE_NUMBER`
- `SUPABASE_DB_URL`
4. Run health check:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/quick_check.ps1
```
5. Start backend:
```bash
python app.py
```
6. Start dashboard:
```bash
python -m streamlit run dashboard.py
```

## 2) Render Deployment (Prototype)

This repository was designed for prototype deployment on Render free tier.

### Service split
- Web service: Flask backend (`app.py`)
- Web service: Streamlit dashboard (`dashboard.py`)

### Required environment variables (Render)
- `OPENAI_API_KEY`
- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_PHONE_NUMBER`
- `SUPABASE_DB_URL`
- `TWILIO_WEBHOOK_URL` (strongly recommended for signature verification)
- `DASHBOARD_USERNAME` and `DASHBOARD_PASSWORD_HASH` (recommended)

### Operational notes
- Free-tier instances may sleep on inactivity, causing cold-start latency.
- Keepalive and scheduler lock controls are available through environment variables.

## 3) Post-Deployment Smoke Checks

1. `GET /ping` returns healthy status.
2. Twilio webhook successfully reaches `POST /sms`.
3. Dashboard auth challenge appears (if auth required).
4. New patient message appears in `chat_history`.
5. High-risk message triggers caregiver alert path.

## 4) Troubleshooting

- `403 invalid Twilio signature`: verify `TWILIO_WEBHOOK_URL` exactly matches external webhook URL.
- DB connection errors: verify `SUPABASE_DB_URL`, network policy, and timeout settings.
- Duplicate proactive sends: keep leader lock and DB execution lock enabled.
- Missing dashboard login: check `DASHBOARD_REQUIRE_AUTH`, `DASHBOARD_USERNAME`, `DASHBOARD_PASSWORD_HASH`.

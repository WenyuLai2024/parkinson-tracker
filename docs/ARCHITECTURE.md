# Architecture Overview

## System Goal
Convert daily, unstructured Parkinson symptom narratives into structured, traceable clinical signals while preserving conversational usability.

## Five-Layer Architecture

1. Access Layer
- Twilio WhatsApp webhook receives inbound messages.

2. Orchestration Layer
- Flask routing handles request validation, media branching, and role routing.

3. AI Engine Layer
- OpenAI-based inference generates natural replies plus internal structured labels.
- Primary labels: `[SUMMARY]`, `[HAUSER]`, `[MOCA]`, `[PDQ39]`, `[PROFILE]`.

4. Persistence Layer
- Supabase PostgreSQL stores longitudinal logs and profile links.
- Core tables:
- `chat_history`
- `patient_profiles`
- Operational lock table for scheduler execution coordination.

5. Presentation Layer
- Streamlit dashboard for trend charts, transcript review, and report export.

## Core Data Flow

1. Patient sends text/voice/image via WhatsApp.
2. `/sms` receives payload and validates Twilio signature (when enabled).
3. Media is processed (voice transcription, image encoding) before LLM call.
4. LLM response is post-processed:
- user-facing message has internal tags stripped
- full tagged response is stored for analytics
5. Structured severity can trigger caregiver alerting.
6. Dashboard reads DB logs and visualizes longitudinal trends.

## Access Boundary Model (Dual-Track)

- Patient write track:
- normal extraction flow
- writes new records to `chat_history`

- Caregiver read-only track:
- checks linkage in `patient_profiles`
- returns de-identified summary of linked patient only
- does not write caregiver message into patient clinical log

## Robustness Controls

- Twilio signature verification (default on)
- Timeout controls for DB and media operations
- Graceful degradation for multimodal failures
- Scheduler duplicate-send prevention with leader lock and DB lease lock
- Summary repair loop for missing `[SUMMARY]` format on clinically relevant input

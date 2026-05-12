# Security and Privacy Controls

## Security Posture
This prototype prioritizes practical safeguards for a telemedicine-adjacent engineering context.

## 1) Request Authenticity

- Twilio signature validation is supported and enabled by default.
- Incoming `/sms` requests can be rejected when signature checks fail.
- `TWILIO_WEBHOOK_URL` should match the externally reachable webhook URL exactly.

## 2) Access Boundary Separation

- Patient messages follow the write path and persist to `chat_history`.
- Caregiver messages follow a read-only summary path.
- Caregiver summary access is linkage-checked in `patient_profiles`.
- The design avoids caregiver writes polluting patient symptom logs.

## 3) Data Minimization

- User-facing responses strip internal extraction tags.
- Dashboard and caregiver summaries can use de-identified representations.
- Sensitive logging is controlled by `LOG_SENSITIVE_DATA` (default `false`).

## 4) Scheduler Safety

- Proactive messaging uses leader-lock and DB execution-lock options.
- This reduces duplicate sends in multi-instance deployment.

## 5) Operational Risk Notes

- External dependencies (OpenAI, Twilio, Supabase, network) remain availability risks.
- Free-tier hosting may introduce latency and cold starts.
- This project is an engineering prototype, not a certified medical device.

## 6) Recommended Deployment Defaults

- `ENABLE_TWILIO_SIGNATURE_VALIDATION=true`
- `DASHBOARD_REQUIRE_AUTH=true`
- `LOG_SENSITIVE_DATA=false`
- configure `DASHBOARD_PASSWORD_HASH` (preferred over plaintext password)

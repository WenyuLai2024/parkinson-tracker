import os
import re
import uuid
import datetime
from collections import defaultdict

import psycopg2
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from flask import Flask, request
from openai import OpenAI
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

from app_ai import get_ai_response
from clinical_utils import (
    is_high_risk_response,
    mask_patient_id,
    normalize_whatsapp_number,
    strip_internal_tags,
)

# =================================================================
# 1. System Configuration & API Credentials
# =================================================================
load_dotenv()
app = Flask(__name__)

TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
SUPABASE_URL = os.getenv("SUPABASE_DB_URL")
CAREGIVER_NUMBER = os.getenv("CAREGIVER_PHONE_NUMBER")

DB_CONNECT_TIMEOUT_SECONDS = int(os.getenv("DB_CONNECT_TIMEOUT_SECONDS", "10"))
MEDIA_DOWNLOAD_TIMEOUT_SECONDS = int(os.getenv("MEDIA_DOWNLOAD_TIMEOUT_SECONDS", "20"))
PROACTIVE_INTERVAL_MINUTES = int(os.getenv("PROACTIVE_INTERVAL_MINUTES", "60"))
CAREGIVER_CONTEXT_LOG_LIMIT = int(os.getenv("CAREGIVER_CONTEXT_LOG_LIMIT", "10"))
ENABLE_PROACTIVE_CHECKIN = os.getenv("ENABLE_PROACTIVE_CHECKIN", "true").lower() == "true"

# Initialize Twilio and OpenAI client instances
twilio_client = Client(TWILIO_SID, TWILIO_AUTH)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# In-memory conversation history (maintains session context per patient)
conversation_history = defaultdict(list)


# =================================================================
# 2. Utility Helpers
# =================================================================
def get_db_connection():
    return psycopg2.connect(SUPABASE_URL, connect_timeout=DB_CONNECT_TIMEOUT_SECONDS)


def is_valid_whatsapp_patient_id(identifier):
    return bool(re.fullmatch(r"whatsapp:\+\d{7,15}", identifier or ""))


def get_linked_patient_ids_for_caregiver(caregiver_sender):
    """Return patient IDs explicitly linked to this caregiver via patient_profiles."""
    raw_sender = caregiver_sender.replace("whatsapp:", "", 1)
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT patient_id
                    FROM patient_profiles
                    WHERE emergency_contact = %s OR emergency_contact = %s
                    """,
                    (caregiver_sender, raw_sender),
                )
                rows = cur.fetchall()
    except Exception as e:
        print(f"Caregiver linkage query failed: {e}")
        return []

    linked_ids = []
    for row in rows:
        normalized = normalize_whatsapp_number(row[0])
        if normalized and is_valid_whatsapp_patient_id(normalized):
            linked_ids.append(normalized)
    return sorted(set(linked_ids))


# =================================================================
# 3. Infrastructure Resilience: Anti-Sleep Mechanism
# =================================================================
@app.route("/ping", methods=["GET"])
def keep_alive():
    """
    Endpoint for external cron jobs (e.g., cron-job.org) to ping periodically.
    Prevents cloud instances (e.g., Render free tier) from entering a sleep state,
    mitigating cold-start timeout errors.
    """
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[Keep-Alive Ping] Server operational at {current_time}")
    return "Server is awake and operational.", 200


# =================================================================
# 4. Proactive Ecological Momentary Assessment (EMA) Scheduler
# =================================================================
def proactive_clinical_checkin():
    """
    Executes scheduled proactive clinical inquiries.
    Queries active patients and dispatches a standardized neurological assessment prompt.
    """
    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Executing Proactive EMA Check-in...")

    try:
        active_patient_ids = set()
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute("SELECT DISTINCT patient_id FROM patient_profiles")
                    active_patient_ids.update(row[0] for row in cur.fetchall() if row and row[0])
                except Exception as e:
                    print(f"patient_profiles query skipped: {e}")

                cur.execute("SELECT DISTINCT patient_id FROM chat_history")
                active_patient_ids.update(row[0] for row in cur.fetchall() if row and row[0])

        patients = []
        for patient_id in active_patient_ids:
            normalized = normalize_whatsapp_number(patient_id)
            if normalized and is_valid_whatsapp_patient_id(normalized):
                patients.append(normalized)
            else:
                print(f"Skipping non-patient identifier in proactive check-in: {patient_id}")

        if not patients:
            print("No active cohorts found for proactive dispatch.")
            return

        checkin_message = (
            "Hi there! It's your AI clinical assistant. "
            "Just checking in on your mobility this afternoon. "
            "How is your medication working right now? Are you feeling any stiffness (OFF) "
            "or involuntary movements (Dyskinesia)?"
        )

        for patient_phone in sorted(set(patients)):
            try:
                msg = twilio_client.messages.create(
                    from_=TWILIO_NUMBER,
                    body=checkin_message,
                    to=patient_phone,
                )
                print(f"Proactive EMA dispatched to {patient_phone} (SID: {msg.sid})")

                conversation_history[patient_phone].append(
                    {"user": "[System Proactive Trigger]", "ai": checkin_message}
                )
            except Exception as inner_e:
                print(f"Failed to dispatch proactive message to {patient_phone}: {inner_e}")

    except Exception as e:
        print(f"Database transaction error during proactive check-in: {e}")


scheduler = None
if ENABLE_PROACTIVE_CHECKIN:
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        proactive_clinical_checkin,
        "interval",
        minutes=PROACTIVE_INTERVAL_MINUTES,
        id="proactive_clinical_checkin",
        replace_existing=True,
    )
    scheduler.start()
    print(f"Scheduler started: proactive check-in every {PROACTIVE_INTERVAL_MINUTES} minutes")
else:
    print("Scheduler disabled via ENABLE_PROACTIVE_CHECKIN=false")


# =================================================================
# 5. Primary Webhook: Dual-Track Communication Routing
# =================================================================
@app.route("/sms", methods=["POST"])
def sms_reply():
    """
    Primary Twilio webhook endpoint.
    Implements a role-based pathway:
    - caregiver read-only summary
    - patient write-enabled clinical ingestion
    """
    sender_number = request.values.get("From", "")
    msg_received = request.values.get("Body", "").strip()
    num_media = int(request.values.get("NumMedia", 0))
    image_url = None

    caregiver_whatsapp = normalize_whatsapp_number(CAREGIVER_NUMBER)

    # -----------------------------------------------------------------
    # ROUTE C: Asymmetric Caregiver Gateway (Read-Only Summarization)
    # -----------------------------------------------------------------
    if caregiver_whatsapp and sender_number == caregiver_whatsapp:
        print(f"[RBAC] Caregiver authorization detected for {sender_number}. Read-only pathway.")
        try:
            linked_patient_ids = get_linked_patient_ids_for_caregiver(sender_number)
            if not linked_patient_ids:
                reply_text = "No linked patient records were found for this caregiver account."
            else:
                linked_lookup_ids = set()
                for patient_id in linked_patient_ids:
                    linked_lookup_ids.add(patient_id)
                    linked_lookup_ids.add(patient_id.replace("whatsapp:", "", 1))

                with get_db_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT patient_id, timestamp, user_message, response
                            FROM chat_history
                            WHERE patient_id = ANY(%s)
                            ORDER BY timestamp DESC
                            LIMIT %s
                            """,
                            (list(linked_lookup_ids), CAREGIVER_CONTEXT_LOG_LIMIT),
                        )
                        recent_logs = cur.fetchall()

                if not recent_logs:
                    reply_text = "There are no recent clinical updates available at this time."
                else:
                    history_context = "\n\n".join(
                        [
                            (
                                f"Patient {mask_patient_id(row[0])} at {row[1]}\n"
                                f"Patient message: {row[2]}\n"
                                f"Assistant response: {strip_internal_tags(str(row[3]))}"
                            )
                            for row in recent_logs
                        ]
                    )

                    caregiver_prompt = f"""
                    You are a compassionate medical assistant speaking to a family caregiver.
                    The caregiver is authorized to view these recent updates:
                    {history_context}

                    Write a short reassuring update in 2-4 sentences.
                    Mention major changes in mobility, mood, sleep, or cognition.
                    Do not use medical jargon and do not include internal tags.
                    """

                    summary_response = openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": caregiver_prompt}],
                        temperature=0.2,
                    )
                    reply_text = summary_response.choices[0].message.content

            resp = MessagingResponse()
            resp.message(reply_text)
            return str(resp)

        except Exception as e:
            print(f"Caregiver pathway critical error: {e}")
            resp = MessagingResponse()
            resp.message("System temporarily unavailable. Please try again later.")
            return str(resp)

    # -----------------------------------------------------------------
    # NORMAL PATIENT PATHWAY (Write-Enabled Clinical Data Ingestion)
    # -----------------------------------------------------------------
    if num_media > 0:
        media_content_type = request.values.get("MediaContentType0", "")
        media_url = request.values.get("MediaUrl0", "")

        if "audio" in media_content_type:
            print(f"Audio payload detected. Downloading from: {media_url}")
            auth = (TWILIO_SID, TWILIO_AUTH)
            try:
                response = requests.get(media_url, auth=auth, timeout=MEDIA_DOWNLOAD_TIMEOUT_SECONDS)
            except Exception as media_error:
                print(f"Audio retrieval failed: {media_error}")
                response = None

            if response is not None and response.status_code == 200:
                temp_filename = f"temp_audio_{uuid.uuid4().hex}.ogg"
                with open(temp_filename, "wb") as f:
                    f.write(response.content)
                try:
                    print("Executing audio transcription via OpenAI Whisper...")
                    with open(temp_filename, "rb") as audio_file:
                        transcript = openai_client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file,
                        )
                    msg_received = transcript.text
                    print(f"Transcription completed: {msg_received}")
                except Exception as e:
                    print(f"Whisper processing failed: {e}")
                    msg_received = "System Note: Voice payload processing failed."
                finally:
                    if os.path.exists(temp_filename):
                        os.remove(temp_filename)
            else:
                msg_received = "System Note: Failed to retrieve voice payload."

        elif "image" in media_content_type:
            image_url = media_url
            print(f"Image attachment detected: {image_url}")

    if not msg_received and not image_url:
        return str(MessagingResponse())

    print(f"\n--- Inbound Patient Telemetry from {sender_number} ---")
    print(f"[Payload]: '{msg_received}'")

    full_ai_response = get_ai_response(
        msg_received,
        conversation_history[sender_number],
        sender_number,
        image_url,
    )

    # --- Clinical Deterioration Alerting Subsystem ---
    if is_high_risk_response(full_ai_response):
        print(f"[CRITICAL EVENT] High-risk threshold breached for ({sender_number}).")

        patient_name = "Unknown Patient"
        caregiver_alert_number = caregiver_whatsapp

        try:
            raw_sender = sender_number.replace("whatsapp:", "", 1)
            with get_db_connection() as conn:
                with conn.cursor() as c:
                    c.execute(
                        """
                        SELECT name, emergency_contact
                        FROM patient_profiles
                        WHERE patient_id = %s OR patient_id = %s
                        LIMIT 1
                        """,
                        (sender_number, raw_sender),
                    )
                    result = c.fetchone()

            if result:
                if result[0]:
                    patient_name = result[0].split()[0]
                if result[1]:
                    caregiver_alert_number = normalize_whatsapp_number(result[1])
        except Exception as e:
            print(f"Demographic retrieval failed during alert sequence: {e}")

        if caregiver_alert_number:
            alert_msg = (
                "[SYSTEM EMERGENCY ALERT]\n"
                f"Warning: {patient_name} ({sender_number}) has recently reported severe symptoms "
                "(e.g., critical stiffness or severe cognitive deviation).\n"
                "Please verify the patient's immediate safety.\n"
                "Clinical Dashboard: https://parkinson-tracker-v1.streamlit.app/"
            )
            try:
                twilio_client.messages.create(
                    from_=TWILIO_NUMBER,
                    body=alert_msg,
                    to=caregiver_alert_number,
                )
                print(f"Emergency alert sent to {caregiver_alert_number}")
            except Exception as e:
                print(f"Emergency alert transmission failed: {e}")

    history_msg = msg_received if msg_received else "[Uploaded Visual Artifact]"
    conversation_history[sender_number].append({"user": history_msg, "ai": full_ai_response})

    if len(conversation_history[sender_number]) > 5:
        conversation_history[sender_number].pop(0)

    display_response = strip_internal_tags(full_ai_response)
    if not display_response:
        display_response = "Thanks, I have logged your update."

    print(f"[System TX] Replying to {sender_number}: {display_response}")

    resp = MessagingResponse()
    resp.message(display_response)
    return str(resp)


if __name__ == "__main__":
    try:
        app.run(port=5000, debug=True, use_reloader=False)
    except (KeyboardInterrupt, SystemExit):
        if scheduler:
            scheduler.shutdown()

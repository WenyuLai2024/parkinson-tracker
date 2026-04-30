import os
import re
import socket
import uuid
import datetime
from zoneinfo import ZoneInfo

import psycopg2
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from flask import Flask, abort, request
from openai import OpenAI
from twilio.request_validator import RequestValidator
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
TWILIO_WEBHOOK_URL = os.getenv("TWILIO_WEBHOOK_URL", "").strip()
SUPABASE_URL = os.getenv("SUPABASE_DB_URL")
CAREGIVER_NUMBER = os.getenv("CAREGIVER_PHONE_NUMBER")
DASHBOARD_PUBLIC_URL = os.getenv("DASHBOARD_PUBLIC_URL", "").strip()

DB_CONNECT_TIMEOUT_SECONDS = int(os.getenv("DB_CONNECT_TIMEOUT_SECONDS", "10"))
MEDIA_DOWNLOAD_TIMEOUT_SECONDS = int(os.getenv("MEDIA_DOWNLOAD_TIMEOUT_SECONDS", "20"))
PROACTIVE_INTERVAL_MINUTES = int(os.getenv("PROACTIVE_INTERVAL_MINUTES", "60"))
CAREGIVER_CONTEXT_LOG_LIMIT = int(os.getenv("CAREGIVER_CONTEXT_LOG_LIMIT", "10"))
PROACTIVE_CHECKIN_TIMEZONE = os.getenv("PROACTIVE_CHECKIN_TIMEZONE", "Europe/London")
DIALOGUE_CONTEXT_TURNS = int(os.getenv("DIALOGUE_CONTEXT_TURNS", "3"))
LOG_SENSITIVE_DATA = os.getenv("LOG_SENSITIVE_DATA", "false").lower() == "true"

ENABLE_PROACTIVE_CHECKIN = os.getenv("ENABLE_PROACTIVE_CHECKIN", "true").lower() == "true"
ENABLE_TWILIO_SIGNATURE_VALIDATION = (
    os.getenv("ENABLE_TWILIO_SIGNATURE_VALIDATION", "true").lower() == "true"
)
SCHEDULER_REQUIRE_LEADER_LOCK = os.getenv("SCHEDULER_REQUIRE_LEADER_LOCK", "true").lower() == "true"
SCHEDULER_LEADER_PORT = int(os.getenv("SCHEDULER_LEADER_PORT", "47200"))
SCHEDULER_EXECUTION_LOCK_ENABLED = os.getenv("SCHEDULER_EXECUTION_LOCK_ENABLED", "true").lower() == "true"
SCHEDULER_EXECUTION_LOCK_NAME = os.getenv("SCHEDULER_EXECUTION_LOCK_NAME", "proactive_clinical_checkin")
SCHEDULER_EXECUTION_LOCK_OWNER = os.getenv(
    "SCHEDULER_EXECUTION_LOCK_OWNER", f"{socket.gethostname()}-{os.getpid()}"
)
SCHEDULER_EXECUTION_LOCK_TTL_SECONDS = int(
    os.getenv(
        "SCHEDULER_EXECUTION_LOCK_TTL_SECONDS",
        str(max(120, PROACTIVE_INTERVAL_MINUTES * 60)),
    )
)
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"
FLASK_PORT = int(os.getenv("PORT", "5000"))

# Initialize Twilio and OpenAI client instances
twilio_client = Client(TWILIO_SID, TWILIO_AUTH)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
twilio_request_validator = RequestValidator(TWILIO_AUTH) if TWILIO_AUTH else None

scheduler = None
scheduler_lock_socket = None


# =================================================================
# 2. Utility Helpers
# =================================================================
def get_db_connection():
    return psycopg2.connect(SUPABASE_URL, connect_timeout=DB_CONNECT_TIMEOUT_SECONDS)


def is_valid_whatsapp_patient_id(identifier):
    return bool(re.fullmatch(r"whatsapp:\+\d{7,15}", identifier or ""))


def masked_phone_for_log(number):
    normalized = normalize_whatsapp_number(number or "")
    if normalized:
        return mask_patient_id(normalized)
    return "***"


def safe_text_preview_for_log(text, max_len=120):
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return "<empty>"
    if len(cleaned) <= max_len:
        return cleaned
    return f"{cleaned[:max_len]}..."


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


def get_recent_conversation_history(patient_id, max_turns=DIALOGUE_CONTEXT_TURNS):
    """Load recent user/assistant turns from DB instead of in-memory cache."""
    normalized = normalize_whatsapp_number(patient_id)
    if not normalized:
        return []

    lookup_ids = [normalized, normalized.replace("whatsapp:", "", 1)]
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT user_message, response
                    FROM chat_history
                    WHERE patient_id = ANY(%s)
                    ORDER BY timestamp DESC
                    LIMIT %s
                    """,
                    (lookup_ids, max_turns),
                )
                rows = cur.fetchall()
    except Exception as e:
        print(f"Conversation history lookup failed: {e}")
        return []

    rows.reverse()
    history = []
    for user_message, response in rows:
        history.append(
            {
                "user": str(user_message or ""),
                "ai": str(response or ""),
            }
        )
    return history


def get_checkin_time():
    """Return current datetime for proactive check-ins in configured timezone."""
    try:
        return datetime.datetime.now(ZoneInfo(PROACTIVE_CHECKIN_TIMEZONE))
    except Exception:
        print(
            f"Invalid PROACTIVE_CHECKIN_TIMEZONE '{PROACTIVE_CHECKIN_TIMEZONE}', "
            "using server local time."
        )
        return datetime.datetime.now()


def build_proactive_checkin_message(now_dt):
    hour = now_dt.hour
    if 5 <= hour < 12:
        daypart = "this morning"
    elif 12 <= hour < 17:
        daypart = "this afternoon"
    elif 17 <= hour < 22:
        daypart = "this evening"
    else:
        daypart = "today"

    return (
        "Hi there! It's your AI clinical assistant. "
        f"Just checking in on your mobility {daypart}. "
        "How is your medication working right now? Are you feeling any stiffness (OFF) "
        "or involuntary movements (Dyskinesia)?"
    )


def acquire_scheduler_leader_lock():
    """
    Ensure only one process starts APScheduler on this host.
    Uses a local TCP bind lock (single-host only).
    """
    global scheduler_lock_socket

    if not SCHEDULER_REQUIRE_LEADER_LOCK:
        return True

    lock_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        lock_socket.bind(("127.0.0.1", SCHEDULER_LEADER_PORT))
        lock_socket.listen(1)
        scheduler_lock_socket = lock_socket
        print(f"Scheduler leader lock acquired on 127.0.0.1:{SCHEDULER_LEADER_PORT}")
        return True
    except OSError:
        print(
            "Scheduler leader lock not acquired. "
            "Another worker likely owns scheduler execution."
        )
        lock_socket.close()
        return False


def release_scheduler_leader_lock():
    global scheduler_lock_socket
    if scheduler_lock_socket:
        scheduler_lock_socket.close()
        scheduler_lock_socket = None


def ensure_scheduler_execution_lock_table(cur):
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS scheduler_execution_locks (
            lock_name TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            locked_until TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )


def try_acquire_scheduler_execution_lock():
    """
    Cross-instance execution lock (DB lease lock).
    This prevents duplicate proactive dispatch when multiple app instances run.
    """
    if not SCHEDULER_EXECUTION_LOCK_ENABLED:
        return True

    lease_seconds = max(60, SCHEDULER_EXECUTION_LOCK_TTL_SECONDS)
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                ensure_scheduler_execution_lock_table(cur)
                cur.execute(
                    """
                    INSERT INTO scheduler_execution_locks (lock_name, owner_id, locked_until, updated_at)
                    VALUES (%s, %s, NOW() + (%s * INTERVAL '1 second'), NOW())
                    ON CONFLICT (lock_name) DO UPDATE
                    SET
                        owner_id = EXCLUDED.owner_id,
                        locked_until = EXCLUDED.locked_until,
                        updated_at = NOW()
                    WHERE
                        scheduler_execution_locks.locked_until < NOW()
                        OR scheduler_execution_locks.owner_id = EXCLUDED.owner_id
                    RETURNING owner_id, locked_until
                    """,
                    (
                        SCHEDULER_EXECUTION_LOCK_NAME,
                        SCHEDULER_EXECUTION_LOCK_OWNER,
                        lease_seconds,
                    ),
                )
                row = cur.fetchone()
            conn.commit()

        if row:
            print(
                "Proactive execution lock acquired "
                f"(name={SCHEDULER_EXECUTION_LOCK_NAME}, owner={SCHEDULER_EXECUTION_LOCK_OWNER})."
            )
            return True

        print(
            "Proactive execution lock is held by another instance. "
            "Skipping this dispatch cycle."
        )
        return False
    except Exception as e:
        print(f"Failed to acquire proactive execution lock: {e}")
        # Fail-safe: skip proactive send to avoid accidental duplicate notifications.
        return False


def get_signature_validation_url():
    """Build URL used for Twilio signature validation."""
    if TWILIO_WEBHOOK_URL:
        return TWILIO_WEBHOOK_URL

    scheme = request.headers.get("X-Forwarded-Proto", request.scheme)
    host = request.headers.get("X-Forwarded-Host", request.host)
    path = request.path
    query = request.query_string.decode("utf-8") if request.query_string else ""
    return f"{scheme}://{host}{path}" + (f"?{query}" if query else "")


def validate_twilio_signature_or_abort():
    if not ENABLE_TWILIO_SIGNATURE_VALIDATION:
        return

    if not twilio_request_validator:
        print("Twilio signature validation is enabled but TWILIO_AUTH_TOKEN is missing.")
        abort(500)

    twilio_signature = request.headers.get("X-Twilio-Signature", "")
    if not twilio_signature:
        print("Missing X-Twilio-Signature header.")
        abort(403)

    validation_url = get_signature_validation_url()
    try:
        valid = twilio_request_validator.validate(validation_url, request.form, twilio_signature)
    except Exception as e:
        print(f"Twilio signature validation error: {e}")
        abort(403)

    if not valid:
        print("Twilio signature mismatch. Request rejected.")
        abort(403)


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
    checkin_time = get_checkin_time()
    print(
        f"\n[{checkin_time.strftime('%Y-%m-%d %H:%M:%S')}] "
        f"Executing Proactive EMA Check-in (TZ: {PROACTIVE_CHECKIN_TIMEZONE}, hour={checkin_time.hour})..."
    )

    if not try_acquire_scheduler_execution_lock():
        return

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

        checkin_message = build_proactive_checkin_message(checkin_time)
        print(f"Proactive check-in prompt: {checkin_message}")

        for patient_phone in sorted(set(patients)):
            try:
                msg = twilio_client.messages.create(
                    from_=TWILIO_NUMBER,
                    body=checkin_message,
                    to=patient_phone,
                )
                print(
                    "Proactive EMA dispatched to "
                    f"{masked_phone_for_log(patient_phone)} (SID: {msg.sid})"
                )
            except Exception as inner_e:
                print(
                    "Failed to dispatch proactive message to "
                    f"{masked_phone_for_log(patient_phone)}: {inner_e}"
                )

    except Exception as e:
        print(f"Database transaction error during proactive check-in: {e}")


if ENABLE_PROACTIVE_CHECKIN:
    if acquire_scheduler_leader_lock():
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
        print("Scheduler start skipped in this process.")
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
    validate_twilio_signature_or_abort()

    sender_number = request.values.get("From", "")
    msg_received = request.values.get("Body", "").strip()
    num_media = int(request.values.get("NumMedia", 0))
    image_url = None

    caregiver_whatsapp = normalize_whatsapp_number(CAREGIVER_NUMBER)

    # -----------------------------------------------------------------
    # ROUTE C: Asymmetric Caregiver Gateway (Read-Only Summarization)
    # -----------------------------------------------------------------
    if caregiver_whatsapp and sender_number == caregiver_whatsapp:
        print(
            "[RBAC] Caregiver authorization detected for "
            f"{masked_phone_for_log(sender_number)}. Read-only pathway."
        )
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
            print("Audio payload detected. Downloading media from Twilio.")
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
                    if LOG_SENSITIVE_DATA:
                        print(f"Transcription completed: {msg_received}")
                    else:
                        print(
                            "Transcription completed "
                            f"(chars={len(msg_received)}, preview='{safe_text_preview_for_log(msg_received)}')."
                        )
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
            print("Image attachment detected.")

    if not msg_received and not image_url:
        return str(MessagingResponse())

    print(f"\n--- Inbound Patient Telemetry from {masked_phone_for_log(sender_number)} ---")
    if LOG_SENSITIVE_DATA:
        print(f"[Payload]: '{msg_received}'")
    else:
        print(
            "[Payload metadata] "
            f"chars={len(msg_received)}, has_image={bool(image_url)}, media_count={num_media}, "
            f"preview='{safe_text_preview_for_log(msg_received)}'"
        )

    conversation_context = get_recent_conversation_history(sender_number, max_turns=DIALOGUE_CONTEXT_TURNS)
    full_ai_response = get_ai_response(
        msg_received,
        conversation_context,
        sender_number,
        image_url,
    )

    # --- Clinical Deterioration Alerting Subsystem ---
    if is_high_risk_response(full_ai_response):
        print(
            "[CRITICAL EVENT] High-risk threshold breached for "
            f"({masked_phone_for_log(sender_number)})."
        )

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
                f"Warning: {patient_name} ({masked_phone_for_log(sender_number)}) has recently reported severe symptoms "
                "(e.g., critical stiffness or severe cognitive deviation).\n"
                "Please verify the patient's immediate safety."
            )
            if DASHBOARD_PUBLIC_URL:
                alert_msg += f"\nClinical Dashboard: {DASHBOARD_PUBLIC_URL}"
            try:
                twilio_client.messages.create(
                    from_=TWILIO_NUMBER,
                    body=alert_msg,
                    to=caregiver_alert_number,
                )
                print(f"Emergency alert sent to {masked_phone_for_log(caregiver_alert_number)}")
            except Exception as e:
                print(f"Emergency alert transmission failed: {e}")

    display_response = strip_internal_tags(full_ai_response)
    if not display_response:
        display_response = "Thanks, I have logged your update."

    if LOG_SENSITIVE_DATA:
        print(f"[System TX] Replying to {sender_number}: {display_response}")
    else:
        print(
            "[System TX] Replying to "
            f"{masked_phone_for_log(sender_number)} (chars={len(display_response)}, "
            f"preview='{safe_text_preview_for_log(display_response)}')"
        )

    resp = MessagingResponse()
    resp.message(display_response)
    return str(resp)


if __name__ == "__main__":
    try:
        app.run(port=FLASK_PORT, debug=FLASK_DEBUG, use_reloader=False)
    except (KeyboardInterrupt, SystemExit):
        if scheduler:
            scheduler.shutdown()
        release_scheduler_leader_lock()

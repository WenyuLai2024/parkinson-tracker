import os
import requests
import uuid
import psycopg2
import datetime
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client  
from collections import defaultdict
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler 
from openai import OpenAI

# Import external AI processing logic (Prompt engineering and GPT-4o calls)
from app_ai import get_ai_response

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

# Initialize Twilio and OpenAI client instances
twilio_client = Client(TWILIO_SID, TWILIO_AUTH)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# In-memory conversation history (maintains session context per patient)
conversation_history = defaultdict(list)

# =================================================================
# 2. Infrastructure Resilience: Anti-Sleep Mechanism
# =================================================================
@app.route("/ping", methods=['GET'])
def keep_alive():
    """
    Endpoint for external cron jobs (e.g., cron-job.org) to ping periodically.
    Prevents cloud instances (e.g., Render free tier) from entering a sleep state, 
    mitigating Twilio HTTP retrieval timeout errors (11200) during cold starts.
    """
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"🔄 [Keep-Alive Ping] Server operational at {current_time}")
    return "Server is awake and operational.", 200

# =================================================================
# 3. Proactive Ecological Momentary Assessment (EMA) Scheduler
# =================================================================
def proactive_clinical_checkin():
    """
    Executes scheduled proactive clinical inquiries.
    Queries the cloud database for active patients and dispatches a standardized 
    neurological assessment prompt via the Twilio API.
    """
    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] ⏰ Executing Proactive EMA Check-in...")
    
    try:
        conn = psycopg2.connect(SUPABASE_URL)
        c = conn.cursor()
        c.execute("SELECT DISTINCT patient_id FROM chat_history")
        patients = c.fetchall()
        conn.close()
        
        if not patients:
            print("No active cohorts found for proactive dispatch.")
            return

        checkin_message = (
            "Hi there! It's your AI clinical assistant. 🩺\n"
            "Just checking in on your mobility this afternoon. "
            "How is your medication working right now? Are you feeling any stiffness (OFF) "
            "or involuntary movements (Dyskinesia)?"
        )

        for p in patients:
            patient_phone = p[0].strip()
            if not patient_phone.startswith("whatsapp:"):
                patient_phone = f"whatsapp:{patient_phone}"
                
            try:
                msg = twilio_client.messages.create(
                    from_=TWILIO_NUMBER,
                    body=checkin_message,
                    to=patient_phone
                )
                print(f"✅ Proactive EMA dispatched to {patient_phone} (SID: {msg.sid})")
                
                conversation_history[patient_phone].append({
                    "user": "[System Proactive Trigger]", 
                    "ai": checkin_message
                })
            except Exception as inner_e:
                print(f"⚠️ Failed to dispatch proactive message to {patient_phone}: {inner_e}")

    except Exception as e:
        print(f"❌ Database Transaction Error during Proactive Check-in: {e}")

# Initialize and mount the background scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(proactive_clinical_checkin, 'interval', minutes=60)
scheduler.start()

# =================================================================
# 4. Primary Webhook: Dual-Track Communication Routing
# =================================================================
@app.route("/sms", methods=['POST'])
def sms_reply():
    """
    Primary Twilio webhook endpoint. 
    Implements a Role-Based Access Control (RBAC) gateway to separate 
    patient clinical data ingestion from caregiver read-only inquiries.
    """
    sender_number = request.values.get('From', '')
    msg_received = request.values.get('Body', '').strip()
    num_media = int(request.values.get('NumMedia', 0))
    image_url = None

    # Formatting caregiver number for Whatsapp protocol comparison
    caregiver_whatsapp = CAREGIVER_NUMBER
    if caregiver_whatsapp and not caregiver_whatsapp.startswith("whatsapp:"):
        caregiver_whatsapp = f"whatsapp:{caregiver_whatsapp}"

    # -----------------------------------------------------------------
    # ROUTE C: Asymmetric Caregiver Gateway (Read-Only Summarization)
    # -----------------------------------------------------------------
    if caregiver_whatsapp and sender_number == caregiver_whatsapp:
        print(f"👀 [RBAC] Caregiver authorization detected for {sender_number}. Initiating Read-Only Pathway.")
        try:
            conn = psycopg2.connect(SUPABASE_URL)
            cur = conn.cursor()
            # Retrieve the most recent clinical interactions (cross-sectional view)
            cur.execute("""
                SELECT timestamp, user_message, response 
                FROM chat_history 
                ORDER BY timestamp DESC LIMIT 5
            """)
            recent_logs = cur.fetchall()
            conn.close()

            if not recent_logs:
                reply_text = "There are no recent clinical updates available at this time."
            else:
                # Compile chronological history context for LLM summarization
                history_context = "\n".join([f"Patient: {row[1]}\nSystem: {row[2]}" for row in recent_logs])
                
                caregiver_prompt = f"""
                You are a compassionate medical assistant speaking to the family caregiver of a Parkinson's patient.
                Here are the patient's most recent interactions with our tracking system:
                {history_context}
                
                Please write a short, reassuring update (2-3 sentences max) summarizing the patient's recent condition (e.g., motor fluctuations, mood).
                DO NOT use overly technical jargon. DO NOT include internal tracking tags like [HAUSER] or [SUMMARY].
                """
                
                summary_response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": caregiver_prompt}],
                    temperature=0.3
                )
                reply_text = summary_response.choices[0].message.content
                
            # Discard execution here: return response WITHOUT database insertion to preserve data integrity
            resp = MessagingResponse()
            resp.message(reply_text)
            return str(resp)
            
        except Exception as e:
            print(f"❌ Caregiver Pathway Critical Error: {e}")
            resp = MessagingResponse()
            resp.message("System temporarily unavailable. Please try again later.")
            return str(resp)

    # -----------------------------------------------------------------
    # NORMAL PATIENT PATHWAY (Write-Enabled Clinical Data Ingestion)
    # -----------------------------------------------------------------
    # --- Multimedia Processing Subsystem (Voice to Text & Imagery) ---
    if num_media > 0:
        media_content_type = request.values.get('MediaContentType0', '')
        media_url = request.values.get('MediaUrl0', '')

        if 'audio' in media_content_type:
            print(f"🎤 Audio payload detected. Initiating download: {media_url}")
            auth = (TWILIO_SID, TWILIO_AUTH)
            response = requests.get(media_url, auth=auth)
            
            if response.status_code == 200:
                temp_filename = f"temp_audio_{uuid.uuid4().hex}.ogg"
                with open(temp_filename, "wb") as f:
                    f.write(response.content)
                try:
                    print("🎧 Executing NLP audio transcription via OpenAI Whisper...")
                    with open(temp_filename, "rb") as audio_file:
                        transcript = openai_client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file
                        )
                    msg_received = transcript.text
                    print(f"📝 Transcription verified: {msg_received}")
                except Exception as e:
                    print(f"❌ Whisper NLP processing failed: {e}")
                    msg_received = "System Note: Voice payload processing failed."
                finally:
                    if os.path.exists(temp_filename):
                        os.remove(temp_filename)
            else:
                msg_received = "System Note: Failed to retrieve voice payload."
                
        elif 'image' in media_content_type:
            image_url = media_url
            print(f"🖼️ [Visual Attachment Detected]: {image_url}")

    if not msg_received and not image_url:
        return str(MessagingResponse())

    print(f"\n--- Inbound Patient Telemetry from {sender_number} ---")
    print(f"[Payload]: '{msg_received}'")

    # Execute Core AI Diagnostics & Data Structuring
    full_ai_response = get_ai_response(msg_received, conversation_history[sender_number], sender_number, image_url)

    # --- Clinical Deterioration Alerting Subsystem ---
    if "Severity: High" in full_ai_response or "[HAUSER] OFF" in full_ai_response or "[MOCA] Score: 0/3" in full_ai_response:
        print(f"🚨 [CRITICAL EVENT] High-risk threshold breached for ({sender_number}). Triggering emergency protocols...")
        
        patient_name = "Unknown Patient"
        caregiver_alert_number = caregiver_whatsapp
        
        try:
            conn = psycopg2.connect(SUPABASE_URL)
            c = conn.cursor()
            c.execute("SELECT name, emergency_contact FROM patient_profiles WHERE patient_id = %s", (sender_number,))
            result = c.fetchone()
            
            if result:
                if result[0]:
                    patient_name = result[0].split()[0]
                if result[1]:
                    caregiver_alert_number = result[1]
                    if not caregiver_alert_number.startswith("whatsapp:"):
                        caregiver_alert_number = f"whatsapp:{caregiver_alert_number}"
            conn.close()
        except Exception as e:
            print(f"⚠️ Demographic retrieval failed during alert sequence: {e}")

        if caregiver_alert_number:
            alert_msg = (
                f"🚨 [SYSTEM EMERGENCY ALERT]\n"
                f"Warning: {patient_name} ({sender_number}) has recently reported severe symptoms "
                f"(e.g., critical stiffness or severe cognitive deviation).\n"
                f"Please verify the patient's immediate safety!\n"
                f"🔗 Clinical Dashboard: https://parkinson-tracker-v1.streamlit.app/"
            )
            try:
                twilio_client.messages.create(
                    from_=TWILIO_NUMBER,
                    body=alert_msg,
                    to=caregiver_alert_number
                )
                print(f"✅ Emergency broadcast successfully relayed to {caregiver_alert_number}")
            except Exception as e:
                print(f"❌ Emergency broadcast transmission failed: {e}")

    # --- Contextual Memory Management ---
    history_msg = msg_received if msg_received else "[Uploaded Visual Artifact]"
    conversation_history[sender_number].append({"user": history_msg, "ai": full_ai_response})
    
    # Maintain rolling context window to optimize LLM token usage
    if len(conversation_history[sender_number]) > 5:
        conversation_history[sender_number].pop(0)
    
    # UI Sanitization: Strip internal structuring tags before transmitting to patient
    display_response = full_ai_response
    for tag in ["[SUMMARY]", "[PROFILE]", "[HAUSER]", "[MOCA]"]:
        if tag in display_response:
            display_response = display_response.split(tag)[0].strip()

    print(f"[System TX] Routing standardized response to {sender_number}: {display_response}")
    
    resp = MessagingResponse()
    resp.message(display_response)
    return str(resp)

if __name__ == "__main__":
    try:
        app.run(port=5000, debug=True, use_reloader=False)
    except (KeyboardInterrupt, SystemExit):
        # Ensure graceful termination of scheduled clinical threads
        scheduler.shutdown()
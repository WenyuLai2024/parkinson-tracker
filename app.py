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

# Load environment variables from .env file
load_dotenv()
app = Flask(__name__)

# --- System Configurations & API Credentials ---
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
SUPABASE_URL = os.getenv("SUPABASE_DB_URL")

# Initialize Twilio and OpenAI clients
twilio_client = Client(TWILIO_SID, TWILIO_AUTH)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# In-memory conversation history (stores the last 5 messages per patient to maintain context)
conversation_history = defaultdict(list)

# ==========================================
# 🟢 ANTI-SLEEP / KEEP-ALIVE ENDPOINT (SOLUTION A)
# ==========================================
@app.route("/ping", methods=['GET'])
def keep_alive():
    """
    Endpoint for external cron jobs (e.g., cron-job.org) to ping every 10 minutes.
    This prevents the Render free tier instance from entering a sleep state, 
    thereby eliminating Twilio's 11200 HTTP retrieval failure (timeout) during cold starts.
    """
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"🔄 [Keep-Alive Ping Received] Server is awake at {current_time}")
    return "Server is awake and running!", 200

def proactive_clinical_checkin():
    """
    Proactive Ecological Momentary Assessment (EMA):
    Queries the cloud database for active patients and sends a scheduled clinical check-in message.
    """
    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] ⏰ Running Proactive Check-in...")
    
    try:
        # Connect to Cloud PostgreSQL (Supabase) to fetch all active patient IDs
        conn = psycopg2.connect(SUPABASE_URL)
        c = conn.cursor()
        c.execute("SELECT DISTINCT patient_id FROM chat_history")
        patients = c.fetchall()
        conn.close()
        
        if not patients:
            print("No patients found in DB to send proactive messages.")
            return

        # Define the standardized clinical prompt for the patient
        checkin_message = (
            "Hi there! It's your AI clinical assistant. 🩺\n"
            "Just checking in on your mobility this afternoon. "
            "How is your medication working right now? Are you feeling any stiffness (OFF) "
            "or involuntary movements (Dyskinesia)?"
        )

        # Dispatch messages to all active patients via Twilio API
        for p in patients:
            patient_phone = p[0].strip()
            
            # Defensive Programming: Ensure WhatsApp prefix exists
            if not patient_phone.startswith("whatsapp:"):
                patient_phone = f"whatsapp:{patient_phone}"
                
            try:
                msg = twilio_client.messages.create(
                    from_=TWILIO_NUMBER,
                    body=checkin_message,
                    to=patient_phone
                )
                print(f"✅ Proactive message sent to {patient_phone} (SID: {msg.sid})")
                
                # Record the system-initiated message in the conversation history
                conversation_history[patient_phone].append({
                    "user": "[System Proactive Trigger]", 
                    "ai": checkin_message
                })
            except Exception as inner_e:
                print(f"⚠️ Failed to send proactive message to {patient_phone}. Reason: {inner_e}")

    except Exception as e:
        print(f"❌ Critical Database Error in Proactive Check-in: {e}")

# --- Background Task Scheduler Setup ---
scheduler = BackgroundScheduler()

# Continue to run the check-in every 60 minutes
scheduler.add_job(proactive_clinical_checkin, 'interval', minutes=60)
scheduler.start()

@app.route("/sms", methods=['POST'])
def sms_reply():
    """
    Main Webhook for Twilio. 
    Processes incoming WhatsApp messages including text, image, and voice (multilingual triage).
    """
    sender_number = request.values.get('From', '')
    msg_received = request.values.get('Body', '').strip()
    
    num_media = int(request.values.get('NumMedia', 0))
    image_url = None

    # --- MULTIMEDIA HANDLING (Voice & Images) ---
    if num_media > 0:
        media_content_type = request.values.get('MediaContentType0', '')
        media_url = request.values.get('MediaUrl0', '')

        if 'audio' in media_content_type:
            print(f"🎤 Voice message received. Downloading from: {media_url}")
            auth = (TWILIO_SID, TWILIO_AUTH)
            response = requests.get(media_url, auth=auth)
            
            if response.status_code == 200:
                temp_filename = f"temp_audio_{uuid.uuid4().hex}.ogg"
                with open(temp_filename, "wb") as f:
                    f.write(response.content)
                try:
                    print("🎧 Processing audio with OpenAI Whisper...")
                    with open(temp_filename, "rb") as audio_file:
                        transcript = openai_client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file
                        )
                    msg_received = transcript.text
                    print(f"📝 Transcription result: {msg_received}")
                except Exception as e:
                    print(f"❌ Whisper transcription failed: {e}")
                    msg_received = "System Note: Voice processing failed."
                finally:
                    if os.path.exists(temp_filename):
                        os.remove(temp_filename)
            else:
                msg_received = "System Note: Failed to download voice message."
                
        elif 'image' in media_content_type:
            image_url = media_url
            print(f"🖼️ [Attachment Detected]: {image_url}")

    if not msg_received and not image_url:
        return str(MessagingResponse())

    print(f"\n--- New Incoming Message from {sender_number} ---")
    print(f"[Patient Text/Voice]: '{msg_received}'")

    # Core AI Logic
    full_ai_response = get_ai_response(msg_received, conversation_history[sender_number], sender_number, image_url)

    # --- Caregiver Alert System ---
    if "Severity: High" in full_ai_response or "[HAUSER] OFF" in full_ai_response or "[MOCA] Score: 0/3" in full_ai_response:
        print(f"🚨 [CRITICAL EVENT] High-risk symptoms or cognitive decline detected for ({sender_number})! Triggering alert...")
        
        patient_name = "Unknown Patient"
        caregiver_number = os.getenv("CAREGIVER_PHONE_NUMBER") 
        
        try:
            conn = psycopg2.connect(SUPABASE_URL)
            c = conn.cursor()
            c.execute("SELECT name, emergency_contact FROM patient_profiles WHERE patient_id = %s", (sender_number,))
            result = c.fetchone()
            
            if result:
                if result[0]:
                    patient_name = result[0].split()[0]
                if result[1]:
                    caregiver_number = result[1]
            conn.close()
        except Exception as e:
            print(f"⚠️ Database query for patient profile failed: {e}")

        if caregiver_number and not caregiver_number.startswith("whatsapp:"):
            caregiver_number = f"whatsapp:{caregiver_number}"

        if caregiver_number:
            alert_msg = (
                f"🚨 [SYSTEM EMERGENCY ALERT]\n"
                f"Warning: {patient_name} ({sender_number}) has just reported severe symptoms "
                f"(e.g., extreme stiffness or severe cognitive deviation in MoCA test).\n"
                f"Please check on the patient's safety immediately!\n"
                f"🔗 View detailed clinical data: https://parkinson-tracker-v1.streamlit.app/"
            )
            try:
                twilio_client.messages.create(
                    from_=TWILIO_NUMBER,
                    body=alert_msg,
                    to=caregiver_number
                )
                print(f"✅ Caregiver alert successfully sent to {caregiver_number} for patient {patient_name}")
            except Exception as e:
                print(f"❌ Failed to send caregiver alert: {e}")
        else:
            print("⚠️ No caregiver number configured in DB or ENV. Alert cancelled.")

    # --- Session Management & Reply Formatting ---
    history_msg = msg_received if msg_received else "[Uploaded Image]"
    conversation_history[sender_number].append({"user": history_msg, "ai": full_ai_response})
    
    if len(conversation_history[sender_number]) > 5:
        conversation_history[sender_number].pop(0)
    
    display_response = full_ai_response
    for tag in ["[SUMMARY]", "[PROFILE]", "[HAUSER]", "[MOCA]"]:
        if tag in display_response:
            display_response = display_response.split(tag)[0].strip()

    print(f"[AI] Sending reply to {sender_number}: {display_response}")
    
    resp = MessagingResponse()
    resp.message(display_response)
    return str(resp)

if __name__ == "__main__":
    try:
        app.run(port=5000, debug=True, use_reloader=False)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
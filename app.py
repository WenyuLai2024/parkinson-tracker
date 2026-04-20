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

# Load environment variables from .env file (for local testing)
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

def proactive_clinical_checkin():
    """
    Proactive Ecological Momentary Assessment (EMA):
    Queries the cloud database for active patients and sends a scheduled clinical check-in message.
    This acts as a digital health intervention to improve patient data adherence.
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
            patient_phone = p[0]
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

    except Exception as e:
        print(f"❌ Proactive Check-in Error: {e}")


# --- Background Task Scheduler Setup ---
scheduler = BackgroundScheduler()

# 💡 TEST MODE: Trigger the proactive check-in exactly 10 seconds after server startup
run_time = datetime.datetime.now() + datetime.timedelta(seconds=10)
scheduler.add_job(proactive_clinical_checkin, 'date', run_date=run_time)

# PRODUCTION MODE: Continue to run the check-in every 60 minutes thereafter
scheduler.add_job(proactive_clinical_checkin, 'interval', minutes=60)

# Start the background scheduler
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

        # 1. Handle Voice Messages via Whisper API (Multilingual Support)
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
                    # Overwrite empty text with transcribed voice text (standardised to written text)
                    msg_received = transcript.text
                    print(f"📝 Transcription result: {msg_received}")
                except Exception as e:
                    print(f"❌ Whisper transcription failed: {e}")
                    msg_received = "System Note: Voice processing failed."
                finally:
                    # Privacy preservation: delete local audio file after processing
                    if os.path.exists(temp_filename):
                        os.remove(temp_filename)
            else:
                msg_received = "System Note: Failed to download voice message."
                
        # 2. Handle Image Messages (Vision processing for MoCA Clock Drawing Test)
        elif 'image' in media_content_type:
            image_url = media_url
            print(f"🖼️ [Attachment Detected]: {image_url}")

    # Prevent processing of empty requests
    if not msg_received and not image_url:
        return str(MessagingResponse())

    print(f"\n--- New Incoming Message from {sender_number} ---")
    print(f"[Patient Text/Voice]: '{msg_received}'")

    # Core AI Logic: Pass the text (or transcribed voice) and image URL to the GPT-4o model
    full_ai_response = get_ai_response(msg_received, conversation)
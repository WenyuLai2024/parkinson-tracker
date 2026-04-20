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

# Import your external AI processing logic
from app_ai import get_ai_response

# Load environment variables
load_dotenv()
app = Flask(__name__)

# System Configurations
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
SUPABASE_URL = os.getenv("SUPABASE_DB_URL")

# Initialize Twilio and OpenAI clients
twilio_client = Client(TWILIO_SID, TWILIO_AUTH)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# In-memory conversation history (stores the last 5 messages per patient)
conversation_history = defaultdict(list)

def proactive_clinical_checkin():
    """
    Queries the cloud database for active patients and sends a proactive check-in message.
    Acts as a scheduled digital health intervention.
    """
    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] ⏰ Running Proactive Check-in...")
    
    try:
        # Connect to Cloud PostgreSQL to fetch all active patients
        conn = psycopg2.connect(SUPABASE_URL)
        c = conn.cursor()
        c.execute("SELECT DISTINCT patient_id FROM chat_history")
        patients = c.fetchall()
        conn.close()
        
        if not patients:
            print("No patients found in DB to send proactive messages.")
            return

        # Define the standardized check-in prompt
        checkin_message = (
            "Hi there! It's your AI clinical assistant. 🩺\n"
            "Just checking in on your mobility this afternoon. "
            "How is your medication working right now? Are you feeling any stiffness (OFF) "
            "or involuntary movements (Dyskinesia)?"
        )

        # Dispatch messages to all active patients
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

# Initialize the background scheduler for proactive messages (Currently set to 60 minutes)
scheduler = BackgroundScheduler()
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
                
        # 2. Handle Image Messages (Vision processing)
        elif 'image' in media_content_type:
            image_url = media_url
            print(f"🖼️ [Attachment Detected]: {image_url}")

    # Prevent processing of empty requests
    if not msg_received and not image_url:
        return str(MessagingResponse())

    print(f"\n--- New Incoming Message from {sender_number} ---")
    print(f"[Patient Text/Voice]: '{msg_received}'")

    # Core AI Logic: Pass the text (or transcribed voice) and image URL to the GPT-4o model
    full_ai_response = get_ai_response(msg_received, conversation_history[sender_number], sender_number, image_url)


    # ==========================================
    # 🚨 DIRECTION 3: Event-Driven Caregiver Alert System
    # ==========================================
    # Trigger a parallel alert if the AI detects high clinical severity or a severe OFF state
    if "Severity: High" in full_ai_response or "[HAUSER] OFF" in full_ai_response:
        print(f"🚨 [CRITICAL EVENT] High-risk symptoms detected for ({sender_number})! Triggering caregiver alert...")
        
        # Retrieve the designated caregiver's WhatsApp number from environment variables
        caregiver_number = os.getenv("CAREGIVER_PHONE_NUMBER")
        
        if caregiver_number:
            # Construct the emergency notification payload
            # Replace the Streamlit link with your actual Dashboard URL
            alert_msg = (
                f"🚨 [SYSTEM EMERGENCY ALERT]\n"
                f"Your family member ({sender_number}) has just reported severe Parkinson's symptoms (High Severity or Severe Motor Fluctuation).\n"
                f"Please check on the patient's safety immediately!\n"
                f"🔗 View detailed clinical data: https://parkinson-tracker-3idgng8wxnc9gggpcymc3e.streamlit.app/"
            )
            try:
                # Dispatch the alert via Twilio concurrently (without interrupting the patient's reply)
                twilio_client.messages.create(
                    from_=TWILIO_NUMBER,
                    body=alert_msg,
                    to=caregiver_number
                )
                print(f"✅ Caregiver alert successfully sent to {caregiver_number}")
            except Exception as e:
                print(f"❌ Failed to send caregiver alert: {e}")
        else:
            print("⚠️ CAREGIVER_PHONE_NUMBER is not configured in environment variables. Alert cancelled.")
    # ==========================================


    # --- Session Management & Reply Formatting ---
    # Update short-term conversation history for context continuity
    history_msg = msg_received if msg_received else "[Uploaded Image]"
    conversation_history[sender_number].append({"user": history_msg, "ai": full_ai_response})
    
    # Maintain a rolling window of 5 turns to prevent token overflow
    if len(conversation_history[sender_number]) > 5:
        conversation_history[sender_number].pop(0)
    
    # Information Hiding: Strip internal clinical tags before sending the empathetic response to the patient
    display_response = full_ai_response
    for tag in ["[SUMMARY]", "[PROFILE]", "[HAUSER]", "[MOCA]"]:
        if tag in display_response:
            display_response = display_response.split(tag)[0].strip()

    print(f"[AI] Sending reply to {sender_number}: {display_response}")
    
    # Dispatch the sanitized response back to the patient via WhatsApp
    resp = MessagingResponse()
    resp.message(display_response)
    return str(resp)

if __name__ == "__main__":
    try:
        # Run locally (Note: Gunicorn will bypass this block when deployed on Render)
        app.run(port=5000, debug=True, use_reloader=False)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
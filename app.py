import os
import psycopg2
import datetime
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client  
from collections import defaultdict
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler 

from app_ai import get_ai_response

load_dotenv()
app = Flask(__name__)

TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
SUPABASE_URL = os.getenv("SUPABASE_DB_URL")
twilio_client = Client(TWILIO_SID, TWILIO_AUTH)

conversation_history = defaultdict(list)

def proactive_clinical_checkin():
    """
    Queries the cloud database for active patients and sends a proactive check-in.
    """
    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] ⏰ Running Proactive Check-in...")
    
    try:
        # Connecting to Cloud PostgreSQL
        conn = psycopg2.connect(SUPABASE_URL)
        c = conn.cursor()
        c.execute("SELECT DISTINCT patient_id FROM chat_history")
        patients = c.fetchall()
        conn.close()
        
        if not patients:
            print("No patients found in DB to send proactive messages.")
            return

        checkin_message = (
            "Hi there! It's your AI clinical assistant. 🩺\n"
            "Just checking in on your mobility this afternoon. "
            "How is your medication working right now? Are you feeling any stiffness (OFF) "
            "or involuntary movements (Dyskinesia)?"
        )

        for p in patients:
            patient_phone = p[0]
            msg = twilio_client.messages.create(
                from_=TWILIO_NUMBER,
                body=checkin_message,
                to=patient_phone
            )
            print(f"✅ Proactive message sent to {patient_phone} (SID: {msg.sid})")
            
            conversation_history[patient_phone].append({
                "user": "[System Proactive Trigger]", 
                "ai": checkin_message
            })

    except Exception as e:
        print(f"❌ Proactive Check-in Error: {e}")

scheduler = BackgroundScheduler()
# For testing: triggers every 1 minute. Change to 'cron' for production.
scheduler.add_job(proactive_clinical_checkin, 'interval', minutes=60)
scheduler.start()

@app.route("/sms", methods=['POST'])
def sms_reply():
    sender_number = request.values.get('From', '')
    msg_received = request.values.get('Body', '').strip()
    
    num_media = int(request.values.get('NumMedia', 0))
    image_url = request.values.get('MediaUrl0', None) if num_media > 0 else None

    if not msg_received and not image_url:
        return str(MessagingResponse())

    print(f"\n--- New Incoming Message from {sender_number} ---")
    if image_url:
        print(f"[Attachment Detected]: {image_url}")
    print(f"[Patient Text]: '{msg_received}'")

    full_ai_response = get_ai_response(msg_received, conversation_history[sender_number], sender_number, image_url)

    history_msg = msg_received if msg_received else "[Uploaded Image]"
    conversation_history[sender_number].append({"user": history_msg, "ai": full_ai_response})
    
    if len(conversation_history[sender_number]) > 5:
        conversation_history[sender_number].pop(0)
    
    display_response = full_ai_response
    if "[SUMMARY]" in display_response:
        display_response = display_response.split("[SUMMARY]")[0].strip()
    if "[PROFILE]" in display_response:
        display_response = display_response.split("[PROFILE]")[0].strip()
    if "[HAUSER]" in display_response:
        display_response = display_response.split("[HAUSER]")[0].strip()
    if "[MOCA]" in display_response:
        display_response = display_response.split("[MOCA]")[0].strip()

    print(f"[AI] Sending reply to {sender_number}: {display_response}")
    resp = MessagingResponse()
    resp.message(display_response)
    return str(resp)

if __name__ == "__main__":
    try:
        app.run(port=5000, debug=True, use_reloader=False)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
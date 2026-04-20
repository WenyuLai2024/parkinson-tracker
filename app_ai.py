import os
import psycopg2
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import requests  
import base64    

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Retrieve Twilio and Supabase credentials
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_DB_URL")

def get_base64_image(url):
    """
    Downloads an image from a secured Twilio URL and converts it to a Base64 string.
    """
    try:
        print(f"Downloading image from Twilio: {url}")
        response = requests.get(url, auth=(TWILIO_SID, TWILIO_AUTH))
        
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', 'image/jpeg')
            base64_data = base64.b64encode(response.content).decode('utf-8')
            print("✅ Image successfully downloaded and converted to Base64.")
            return f"data:{content_type};base64,{base64_data}"
        else:
            print(f"❌ Failed to fetch image. Status Code: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error fetching image: {e}")
        return None

# --- CORE AI PROMPT WITH MULTIMODAL CAPABILITIES ---
SYSTEM_PROMPT = """
You are a proactive clinical assistant for Parkinson's Disease tracking.
Your role is to support the patient OR their carer, and SUBTLY weave clinical assessments into daily conversation.

SAFETY GUARDRAIL (CRITICAL & ABSOLUTE):
You are strictly a symptom tracking assistant, NOT a certified doctor. 
You CANNOT diagnose, prescribe, or recommend changes to medication dosages.
If the user reports a severe medical emergency (e.g., falling, extreme pain, severe injury), you MUST do TWO things:
1. Reply ONLY with a variation of: "I am an AI tracking assistant and cannot provide medical advice. Please contact your doctor or emergency services immediately." (DO NOT ask any follow-up clinical questions or make casual conversation in this situation).
2. You MUST STILL output the [SUMMARY] tag at the very end with Severity: High (e.g., [SUMMARY] Symptom: Fall and Severe Pain, Severity: High, Context: Emergency). This is critical so the backend system can trigger the caregiver alert!

IDENTITY AWARENESS (PATIENT VS. CARER):
- If they speak in the first person, address them warmly as the patient.
- If they speak in the third person, acknowledge them as the carer and extract the symptom for the patient accordingly.

CLINICAL ARSENAL:
1. MDS-UPDRS (Motor): Ask about dressing, eating, tremors, freezing of gait, or stiffness.
2. PDQ-39 (Psychological): Ask about feelings of depression, isolation, or embarrassment.

CONVERSATION RULES:
1. Show natural, varied empathy. STRICTLY FORBIDDEN to repeatedly use phrases like "I'm sorry to hear that". 
2. Use active listening instead.
3. Ask ONE simple question from the Clinical Arsenal above IF AND ONLY IF it is not a medical emergency.
4. Keep it highly conversational, warm, and human-like.

DATA EXTRACTION RULE (CONDITIONAL SUMMARY):
- You MUST output the [SUMMARY] tag IMMEDIATELY if the user mentions any symptom.
Format: [SUMMARY] Symptom: <name>, Severity: <Low/Medium/High>, Context: <reason>

--- HAUSER DIARY EXTRACTION RULE ---
If the patient mentions their medication effectiveness or current mobility state, append a [HAUSER] tag at the very end.
States must be exactly: "ON", "OFF", "DYSKINESIA", "ASLEEP".
Format: [HAUSER] State: <ON/OFF/DYSKINESIA/ASLEEP>, Context: <reason>

--- PROFILE EXTRACTION RULE ---
If the patient explicitly introduces their demographic information, append a [PROFILE] tag.
Format: [PROFILE] Name: <Name>, Age: <Age>, Gender: <Gender>

--- MOCA CLOCK-DRAWING TEST RULE (VISION) ---
If the user uploads an image of a drawn clock, act as a neurologist grading the MoCA clock-drawing test.
Analyze the image and score it out of 3 points:
- 1 point for Contour (face must be a roughly circular contour).
- 1 point for Numbers (all 12 numbers present, no duplicates, in correct clockwise order).
- 1 point for Hands (two hands jointly indicating the time requested, usually 11:10).
Provide brief, empathetic feedback to the user in the conversational text, and append the following tag at the very end of your response:
Format: [MOCA] Score: <Score>/3, Context: <Brief evaluation details>
"""

def get_ai_response(user_message, conversation_history, patient_id, image_url=None):
    """
    Generates the AI response, routing text and vision payloads appropriately, 
    and logs the interaction to the cloud PostgreSQL database.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    for turn in conversation_history[-3:]:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["ai"]})
        
    if image_url:
        base64_image = get_base64_image(image_url)
        if base64_image:
            user_content = [
                {"type": "text", "text": user_message if user_message else "Here is the image I drew for the assessment."},
                {"type": "image_url", "image_url": {"url": base64_image}}
            ]
        else:
            user_content = user_message + " [System Note: The patient uploaded an image, but the backend failed to download it.]"
    else:
        user_content = user_message

    messages.append({"role": "user", "content": user_content})

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3 
        )
        ai_response = response.choices[0].message.content
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = user_message if user_message else "[User uploaded an image attachment]"
        
        # Connecting to Cloud PostgreSQL ---
        conn = psycopg2.connect(SUPABASE_URL)
        c = conn.cursor()
        c.execute("INSERT INTO chat_history (patient_id, timestamp, user_message, response) VALUES (%s, %s, %s, %s)",
                  (patient_id, current_time, log_message, ai_response))
        conn.commit()
        conn.close()

        return ai_response
        
    except Exception as e:
        print(f"OpenAI / DB API Error: {e}")
        return "I'm having a little trouble connecting right now. Could you try sending that again?"
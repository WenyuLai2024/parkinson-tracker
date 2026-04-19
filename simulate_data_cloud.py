import os
import psycopg2
from datetime import datetime, timedelta
import random
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_DB_URL")

def generate_mock_data():
    """
    Injects high-quality mock patient data into the cloud database 
    for thesis chart demonstrations and system evaluation.
    """
    print("⏳ Connecting to the cloud database...")
    conn = psycopg2.connect(SUPABASE_URL)
    c = conn.cursor()
    
    # Clear old data (to ensure a clean demonstration environment)
    c.execute("DELETE FROM chat_history") 
    c.execute("DELETE FROM patient_profiles")
    
    now = datetime.now()
    
    # ==========================================
    # 👤 Patient 1: John Doe (Demonstrating severe motor fluctuations)
    # ==========================================
    patient_1 = "+447111111111"
    c.execute("INSERT INTO patient_profiles (patient_id, name, age, gender, years_diagnosed, current_medication) VALUES (%s, %s, %s, %s, %s, %s)", 
              (patient_1, "John Doe", "72", "Male", "5", "Levodopa/Carbidopa 250/25mg"))
    
    print("💉 Injecting 30 consecutive days of clinical data for John...")
    
    # Simulate high-frequency interactions over the past 30 days
    for i in range(30, 0, -1):
        target_date = now - timedelta(days=i)
        
        # 9:00 AM: Just took medication, good mobility state (ON)
        time_m = target_date.replace(hour=9, minute=random.randint(0, 30)).strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO chat_history (patient_id, timestamp, user_message, response) VALUES (%s, %s, %s, %s)",
                  (patient_1, time_m, "I just had my morning medication and breakfast. Walking is easy right now.", 
                   "That's great news, John! Enjoy your morning mobility.\n[HAUSER] State: ON, Context: Morning medication kicked in\n[SUMMARY] Symptom: Normal mobility, Severity: Low, Context: Post-medication"))

        # 2:00 PM: Medication wearing off, stiffness appears (OFF)
        time_a = target_date.replace(hour=14, minute=random.randint(0, 30)).strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO chat_history (patient_id, timestamp, user_message, response) VALUES (%s, %s, %s, %s)",
                  (patient_1, time_a, "My legs feel extremely heavy and stiff. I can barely get out of my chair.", 
                   "I'm sorry to hear you're feeling stiff. Please rest.\n[HAUSER] State: OFF, Context: Afternoon wear-off, heavy legs\n[SUMMARY] Symptom: Severe stiffness, Severity: High, Context: Wearing off period"))
        
        # 7:00 PM: Occasional dyskinesia or continuing ON state
        if random.random() > 0.6:
            time_e = target_date.replace(hour=19, minute=random.randint(0, 30)).strftime("%Y-%m-%d %H:%M:%S")
            c.execute("INSERT INTO chat_history (patient_id, timestamp, user_message, response) VALUES (%s, %s, %s, %s)",
                      (patient_1, time_e, "My arms are jerking around a bit on their own while watching TV.", 
                       "Thanks for reporting this involuntary movement.\n[HAUSER] State: DYSKINESIA, Context: Evening jerking movements\n[SUMMARY] Symptom: Involuntary movements, Severity: Medium, Context: Evening rest"))

        # Every 7 days, perform a cognitive clock-drawing test (MoCA showing a declining trend)
        if i % 7 == 0:
            moca_score = 3 if i > 14 else (2 if i > 7 else 1)
            time_moca = target_date.replace(hour=11, minute=0).strftime("%Y-%m-%d %H:%M:%S")
            c.execute("INSERT INTO chat_history (patient_id, timestamp, user_message, response) VALUES (%s, %s, %s, %s)",
                      (patient_1, time_moca, "[User uploaded an image attachment of a clock]", 
                       f"Thank you for the drawing. Keep practicing!\n[MOCA] Score: {moca_score}/3, Context: Routine weekly cognitive check."))

    # ==========================================
    # 👤 Patient 2: Mary Smith (Demonstrating mild symptoms and psychological issues)
    # ==========================================
    patient_2 = "+447222222222"
    c.execute("INSERT INTO patient_profiles (patient_id, name, age, gender, years_diagnosed, current_medication) VALUES (%s, %s, %s, %s, %s, %s)", 
              (patient_2, "Mary Smith", "65", "Female", "2", "Ropinirole 2mg"))
    
    print("💉 Injecting data for Mary...")
    
    mock_interactions_2 = [
        (5, "10:00", "I just feel a bit embarrassed going out because of my slight tremor.", "It's completely normal to feel that way.\n[SUMMARY] Symptom: Social anxiety due to tremor, Severity: Medium, Context: Going outside"),
        (2, "15:30", "Did some gardening today. No physical issues, just tired.", "Gardening is great exercise! Rest well.\n[SUMMARY] Symptom: General fatigue, Severity: Low, Context: Post-gardening")
    ]
    for days_ago, time_str, user_msg, ai_resp in mock_interactions_2:
        target_date = now - timedelta(days=days_ago)
        timestamp = datetime.combine(target_date.date(), datetime.strptime(time_str, "%H:%M").time()).strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO chat_history (patient_id, timestamp, user_message, response) VALUES (%s, %s, %s, %s)", 
                  (patient_2, timestamp, user_msg, ai_resp))
                  
    conn.commit()
    conn.close()
    print("✅ Success! All mock data has been successfully injected into the Supabase cloud server!")
    print("👉 Now go refresh your Streamlit webpage to see the results!")

if __name__ == "__main__":
    generate_mock_data()
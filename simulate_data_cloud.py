import argparse
import os
import random
from datetime import datetime, timedelta

import psycopg2
from dotenv import load_dotenv

print("Initializing Clinical Data Simulation Protocol...")

# =================================================================
# 1. Environment & Database Configuration
# =================================================================
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_DB_URL")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Seed synthetic Parkinson's data into cloud DB."
    )
    parser.add_argument(
        "--confirm-reset",
        action="store_true",
        help="Required to allow destructive reset of chat_history and patient_profiles.",
    )
    parser.add_argument(
        "--skip-reset",
        action="store_true",
        help="Append synthetic data without deleting existing rows.",
    )
    return parser.parse_args()


def guard_destructive_action(args):
    if args.skip_reset:
        return
    if not args.confirm_reset:
        raise SystemExit(
            "Refusing to run destructive reset. Re-run with --confirm-reset "
            "or use --skip-reset to append data."
        )


args = parse_args()
guard_destructive_action(args)

try:
    conn = psycopg2.connect(SUPABASE_URL)
    c = conn.cursor()

    # =================================================================
    # 2. Optional Database Reset (Tear-down Phase)
    # =================================================================
    if not args.skip_reset:
        print("Purging existing interaction logs from cloud database...")
        c.execute("DELETE FROM chat_history")
        c.execute("DELETE FROM patient_profiles")

    # =================================================================
    # 3. Patient Cohort Seeding
    # =================================================================
    patient_1 = "whatsapp:+447123456789"

    c.execute(
        """
        INSERT INTO patient_profiles (patient_id, name, age, gender, years_diagnosed, current_medication)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (patient_id) DO UPDATE SET
            name = EXCLUDED.name,
            age = EXCLUDED.age,
            gender = EXCLUDED.gender,
            years_diagnosed = EXCLUDED.years_diagnosed,
            current_medication = EXCLUDED.current_medication
        """,
        (patient_1, "John Doe", 68, "Male", 4, "Levodopa 100mg (3x/day)"),
    )

    # =================================================================
    # 4. Longitudinal Data Injection (360-Day Retrospective Cohort)
    # =================================================================
    print("Injecting 12 months of longitudinal clinical data for Patient 1...")

    now = datetime.now()

    for i in range(360, 0, -3):
        target_date = now - timedelta(days=i)

        time_m = target_date.replace(hour=9, minute=random.randint(0, 30)).strftime("%Y-%m-%d %H:%M:%S")
        c.execute(
            "INSERT INTO chat_history (patient_id, timestamp, user_message, response) VALUES (%s, %s, %s, %s)",
            (
                patient_1,
                time_m,
                "I just had my morning medication and breakfast. Walking is easy right now.",
                "That's great news! Enjoy your morning mobility.\n[HAUSER] State: ON, Context: Post-medication mobility\n[SUMMARY] Symptom: Normal mobility, Severity: Low, Score: 1, Context: Morning routine",
            ),
        )

        time_a = target_date.replace(hour=14, minute=random.randint(0, 30)).strftime("%Y-%m-%d %H:%M:%S")
        c.execute(
            "INSERT INTO chat_history (patient_id, timestamp, user_message, response) VALUES (%s, %s, %s, %s)",
            (
                patient_1,
                time_a,
                "My legs feel extremely heavy and stiff. I can barely get out of my chair.",
                "I'm sorry to hear you're feeling stiff. Please rest safely.\n[HAUSER] State: OFF, Context: Afternoon medication wear-off\n[SUMMARY] Symptom: Severe stiffness, Severity: High, Score: 3, Context: Wearing off period",
            ),
        )

        if random.random() > 0.7:
            time_e = target_date.replace(hour=19, minute=random.randint(0, 30)).strftime("%Y-%m-%d %H:%M:%S")
            c.execute(
                "INSERT INTO chat_history (patient_id, timestamp, user_message, response) VALUES (%s, %s, %s, %s)",
                (
                    patient_1,
                    time_e,
                    "My arms are jerking around a bit on their own while watching TV.",
                    "Thanks for reporting this involuntary movement.\n[HAUSER] State: DYSKINESIA, Context: Evening involuntary movements\n[SUMMARY] Symptom: Dyskinesia, Severity: Medium, Score: 2, Context: Evening rest",
                ),
            )

        if i % 30 == 0:
            moca_score = 3 if i > 180 else (2 if i > 90 else 1)
            time_moca = target_date.replace(hour=11, minute=0).strftime("%Y-%m-%d %H:%M:%S")
            c.execute(
                "INSERT INTO chat_history (patient_id, timestamp, user_message, response) VALUES (%s, %s, %s, %s)",
                (
                    patient_1,
                    time_moca,
                    "[User uploaded an image attachment of a clock]",
                    f"Thank you for the drawing. Keep practicing!\n[MOCA] Score: {moca_score}/3, Context: Routine monthly cognitive check.",
                ),
            )

    conn.commit()
    print("Clinical simulation completed successfully. Cloud DB updated.")

except psycopg2.Error as e:
    print(f"Database Connection Error: {e}")
finally:
    if "conn" in locals():
        conn.close()

import streamlit as st
import pandas as pd
import altair as alt
import os
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
SUPABASE_URL = os.getenv("SUPABASE_DB_URL")

st.set_page_config(page_title="Clinician Dashboard", layout="wide", page_icon="⚕️")

st.markdown(
    """
    <style>
    div[data-baseweb="select"] > div { cursor: pointer !important; }
    div[data-baseweb="select"] input { cursor: text !important; }
    </style>
    """, unsafe_allow_html=True
)

try:
    # Connecting to Cloud PostgreSQL and mapping to Pandas DataFrame
    conn = psycopg2.connect(SUPABASE_URL)
    cur = conn.cursor()
    
    # Fetch chat history
    cur.execute("SELECT * FROM chat_history")
    history_rows = cur.fetchall()
    if history_rows:
        history_cols = [desc[0] for desc in cur.description]
        raw_df = pd.DataFrame(history_rows, columns=history_cols)
    else:
        raw_df = pd.DataFrame()

    # Fetch patient profiles
    try:
        cur.execute("SELECT * FROM patient_profiles")
        profile_rows = cur.fetchall()
        if profile_rows:
            profile_cols = [desc[0] for desc in cur.description]
            profiles_df = pd.DataFrame(profile_rows, columns=profile_cols)
        else:
            profiles_df = pd.DataFrame()
    except Exception:
        profiles_df = pd.DataFrame()
        
    conn.close()
    
    if raw_df.empty:
        st.warning("No data found in the cloud database. Please initialize patient interactions via WhatsApp.")

    st.title("Parkinson's Disease Longitudinal Tracker (Cloud-Native)")

    st.sidebar.header("Global Navigation")
    
    raw_patient_list = raw_df['patient_id'].unique().tolist() if not raw_df.empty else []
    
    profile_dict = {}
    if not profiles_df.empty:
        profile_dict = pd.Series(profiles_df.name.values, index=profiles_df.patient_id).to_dict()

    def generate_display_name(phone_str):
        masked_phone = f"***{phone_str[-4:]}" if len(phone_str) >= 4 else "Unknown"
        if phone_str in profile_dict and profile_dict[phone_str] != "Unknown":
            full_name = profile_dict[phone_str]
            first_name = full_name.split()[0]
            last_initial = full_name.split()[1][0] + "." if len(full_name.split()) > 1 else ""
            return f"{first_name} {last_initial} ({masked_phone})"
        return f"Unknown Patient ({masked_phone})"
        
    display_to_raw_map = {generate_display_name(pid): pid for pid in raw_patient_list}
    display_names = list(display_to_raw_map.keys())
    
    HOME_OPTION = "Home / Overview"
    options_list = [HOME_OPTION] + display_names
    
    selected_display_name = st.sidebar.selectbox("Select Patient EHR:", options_list)
    st.sidebar.markdown("---")
    
    if selected_display_name == HOME_OPTION:
        st.sidebar.info("Data Privacy Mode: Patient identifiers are masked to comply with healthcare data protection regulations.")
        st.markdown("## 🏥 Welcome to the Clinical Decision Support System")
        st.info("👈 Please select a patient from the sidebar to view their individual Electronic Health Record (EHR).")
        st.divider()
        st.subheader("System Overview Metrics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Total Monitored Patients", value=len(raw_patient_list))
        with col2:
            st.metric(label="Total Recorded Interactions", value=len(raw_df))
            
        st.divider()
        st.markdown("""
        ### Core Capabilities
        - **Cloud-Native Infrastructure**: Data is securely stored and retrieved from PostgreSQL (Supabase).
        - **Continuous Tracking**: Integrates with WhatsApp for seamless patient logging.
        - **AI-Powered Analysis**: Utilizes LLMs to extract structured clinical symptoms.
        - **Conversational Hauser Diary**: Passively infers medication ON/OFF fluctuations.
        """)
        
    else:
        selected_patient_raw = display_to_raw_map[selected_display_name]
        df = raw_df[raw_df['patient_id'] == selected_patient_raw].copy()

        st.sidebar.subheader("📌 Page Contents")
        st.sidebar.markdown("""
        * [🟢 Hauser Diary (Motor)](#hauser)
        * [🧠 MoCA Trend (Cognitive)](#moca)
        * [📈 Symptom Logs & Trend](#symptoms)
        * [🤖 AI Summary Report](#report)
        """)
        st.sidebar.markdown("---")
        st.sidebar.caption("Data Privacy Mode Active.")

        tab1, tab2 = st.tabs(["📊 Clinical Dashboard", "💬 Raw Chat Transcript"])
        
        with tab1:
            st.markdown(f"### Clinician Portal: Electronic Health Record (EHR)")
            
            if not profiles_df.empty:
                patient_profile = profiles_df[profiles_df['patient_id'] == selected_patient_raw]
                if not patient_profile.empty:
                    profile_data = patient_profile.iloc[0]
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.caption("Name (Masked)")
                        display_name = profile_data['name'].split()[0] if profile_data['name'] != "Unknown" else "Unknown"
                        st.info(f"**{display_name} ***")
                    with col2:
                        st.caption("Demographics")
                        st.info(f"**{profile_data['age']} yrs / {profile_data['gender']}**")
                    with col3:
                        st.caption("Time since Diagnosis")
                        st.info(f"**{profile_data['years_diagnosed']} Years**")
                    with col4:
                        st.caption("Current Medication")
                        st.info(f"**{profile_data['current_medication']}**")
                    st.divider()
            
            extracted_symp = df['response'].str.extract(r"\[SUMMARY\] Symptom: (.*?), Severity: (.*?), Context: (.*)")
            extracted_symp.columns = ['Symptom', 'Severity', 'Context']
            clinical_df = pd.concat([df, extracted_symp], axis=1).dropna(subset=['Symptom']).copy()
            severity_weights = {"Low": 1, "Medium": 2, "High": 3}
            clinical_df['Severity_Score'] = clinical_df['Severity'].map(severity_weights)

            extracted_hauser = df['response'].str.extract(r"\[HAUSER\] State: (.*?), Context: (.*)")
            extracted_hauser.columns = ['State', 'Context']
            hauser_df = pd.concat([df['timestamp'], extracted_hauser], axis=1).dropna(subset=['State']).copy()
            hauser_df['timestamp'] = pd.to_datetime(hauser_df['timestamp'])

            extracted_moca = df['response'].str.extract(r"\[MOCA\] Score: (.*?)/3, Context: (.*)")
            extracted_moca.columns = ['Score', 'Context']
            moca_df = pd.concat([df['timestamp'], extracted_moca], axis=1).dropna(subset=['Score']).copy()
            moca_df['timestamp'] = pd.to_datetime(moca_df['timestamp'])
            moca_df['Score'] = pd.to_numeric(moca_df['Score']) 

            st.subheader("Conversational Hauser Diary (Motor Fluctuations)", anchor="hauser")
            if not hauser_df.empty:
                hauser_colors = alt.Scale(domain=['ON', 'OFF', 'DYSKINESIA', 'ASLEEP'], range=['#28a745', '#dc3545', '#fd7e14', '#6c757d'])
                hauser_df = hauser_df.sort_values('timestamp')
                line = alt.Chart(hauser_df).mark_line(interpolate='step-after', color='#adb5bd', strokeWidth=2, strokeDash=[5, 5]).encode(
                    x=alt.X('timestamp:T', title='Date & Time', axis=alt.Axis(format='%Y-%m-%d %H:%M', labelAngle=-45)),
                    y=alt.Y('State:N', title='Medication State', sort=['ON', 'DYSKINESIA', 'OFF', 'ASLEEP'])
                )
                points = alt.Chart(hauser_df).mark_circle(size=300, opacity=1).encode(
                    x='timestamp:T',
                    y=alt.Y('State:N', sort=['ON', 'DYSKINESIA', 'OFF', 'ASLEEP']),
                    color=alt.Color('State:N', scale=hauser_colors, legend=None),
                    tooltip=['timestamp', 'State', 'Context']
                )
                st.altair_chart((line + points).properties(height=250).interactive(), use_container_width=True)
            else:
                st.info("No Hauser Diary state fluctuations detected yet.")

            st.divider()

            st.subheader("Cognitive Assessment Trend (MoCA Clock-Drawing)", anchor="moca")
            if not moca_df.empty:
                moca_df = moca_df.sort_values('timestamp')
                moca_chart = alt.Chart(moca_df).mark_line(point=alt.OverlayMarkDef(filled=False, fill='white', size=200)).encode(
                    x=alt.X('timestamp:T', title='Date & Time', axis=alt.Axis(format='%Y-%m-%d %H:%M', labelAngle=-45)),
                    y=alt.Y('Score:Q', scale=alt.Scale(domain=[-0.2, 3.2]), axis=alt.Axis(tickCount=4), title='MoCA Score (0-3)'),
                    color=alt.value('#6f42c1'), 
                    tooltip=['timestamp', 'Score', 'Context']
                ).interactive().properties(height=250)
                st.altair_chart(moca_chart, use_container_width=True)
            else:
                st.info("No Cognitive Assessment (MoCA) data found for this patient.")
                
            st.divider()

            left_col, right_col = st.columns([1.2, 2])
            with left_col:
                st.subheader("Recent Clinical Logs", anchor="symptoms")
                if not clinical_df.empty:
                    display_df = clinical_df.sort_values(by='timestamp', ascending=False)[['timestamp', 'Symptom', 'Severity', 'Context']].copy()
                    display_df.rename(columns={'timestamp': 'Date & Time'}, inplace=True)
                    st.dataframe(display_df.head(10), use_container_width=True, hide_index=True)
                else:
                    st.info("No structured logs found for this patient.")

            with right_col:
                st.subheader("Symptom Severity Trend")
                if not clinical_df.empty:
                    clinical_df['timestamp'] = pd.to_datetime(clinical_df['timestamp'])
                    clinical_df = clinical_df.sort_values(by='timestamp', ascending=True)
                    line_chart = alt.Chart(clinical_df).mark_line(point=True).encode(
                        x=alt.X('timestamp:T', title='Date & Time', axis=alt.Axis(format='%Y-%m-%d %H:%M', labelAngle=-45)),
                        y=alt.Y('Severity_Score', scale=alt.Scale(domain=[0, 3.2]), title='Severity Score'),
                        tooltip=['timestamp', 'Symptom', 'Severity', 'Context']
                    ).interactive()
                    st.altair_chart(line_chart, use_container_width=True)
                    st.caption("Legend: 1=Low, 2=Medium, 3=High")
                else:
                    st.warning("Insufficient data to generate trend analysis.")
            
            st.markdown("---")
            st.subheader("AI Pre-Consultation Summary", anchor="report")
            
            if st.button("Generate Clinical Report"):
                if not clinical_df.empty or not hauser_df.empty or not moca_df.empty:
                    with st.spinner("Analyzing patient records via GPT-4o..."):
                        try:
                            symp_str = clinical_df[['timestamp', 'Symptom', 'Severity', 'Context']].tail(10).to_string() if not clinical_df.empty else "No symptom data."
                            hauser_str = hauser_df[['timestamp', 'State', 'Context']].tail(10).to_string() if not hauser_df.empty else "No Hauser data."
                            moca_str = moca_df[['timestamp', 'Score', 'Context']].tail(5).to_string() if not moca_df.empty else "No MoCA data."
                            
                            prompt = f"""
                            You are a professional neurologist analyzing a Parkinson's patient's continuous tracking logs.
                            Patient Identifier: {selected_display_name}
                            
                            Symptom Logs:
                            {symp_str}
                            
                            Hauser Diary (Motor Fluctuations) Logs:
                            {hauser_str}
                            
                            Cognitive Assessment (MoCA Clock-Drawing) Logs:
                            {moca_str}
                            
                            Please write a concise, professional clinical summary (max 3 paragraphs) including symptom trends, medication ON/OFF state patterns, cognitive trends, and recommendations.
                            """
                            
                            response = client.chat.completions.create(
                                model="gpt-4o",
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.2
                            )
                            st.success("Report generation complete.")
                            st.write(response.choices[0].message.content)
                        except Exception as e:
                            st.error(f"Failed to generate AI report: {e}")
                else:
                    st.warning("Please collect more logs before generating a report.")
                    
        with tab2:
            st.markdown(f"### Interaction History for {selected_display_name}")
            chat_name = profile_dict.get(selected_patient_raw, "Unknown").split()[0] + " (Patient/Carer)" if selected_patient_raw in profile_dict and profile_dict[selected_patient_raw] != "Unknown" else "Unknown Patient"
            
            for index, row in df.iterrows():
                with st.chat_message("user", avatar="👤"): 
                    st.write(f"**{chat_name}** ({row['timestamp']}): {row['user_message']}")
                with st.chat_message("assistant", avatar="🤖"):
                    ai_text = str(row['response'])
                    for tag in ["[SUMMARY]", "[PROFILE]", "[HAUSER]", "[MOCA]"]:
                        if tag in ai_text: ai_text = ai_text.split(tag)[0].strip()
                    st.write(f"**AI Assistant**: {ai_text}")
                st.divider()

except psycopg2.Error as e:
    st.error(f"Database Connection Error: {e}")
except Exception as e:
    st.error(f"System Error: {e}")
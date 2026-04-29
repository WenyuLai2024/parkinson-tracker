import streamlit as st
import pandas as pd
import altair as alt
import os
import time
import math
import hmac
import hashlib
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv
from fpdf import FPDF
from clinical_utils import strip_internal_tags

# =================================================================
# 1. SYSTEM CONFIGURATION & ENVIRONMENT
# =================================================================
# Loading environment variables for secure database and API access
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
SUPABASE_URL = os.getenv("SUPABASE_DB_URL")
DASHBOARD_USERNAME = os.getenv("DASHBOARD_USERNAME", "").strip()
DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "").strip()
DASHBOARD_PASSWORD_HASH = os.getenv("DASHBOARD_PASSWORD_HASH", "").strip().lower()
DASHBOARD_SESSION_TIMEOUT_MINUTES = int(os.getenv("DASHBOARD_SESSION_TIMEOUT_MINUTES", "60"))
DASHBOARD_LOOKBACK_DAYS = int(os.getenv("DASHBOARD_LOOKBACK_DAYS", "365"))
DASHBOARD_HISTORY_LIMIT = int(os.getenv("DASHBOARD_HISTORY_LIMIT", "5000"))
DASHBOARD_REQUIRE_AUTH = os.getenv("DASHBOARD_REQUIRE_AUTH", "true").lower() == "true"

# Define the global UI architecture of the dashboard
st.set_page_config(page_title="Clinician Dashboard", layout="wide", page_icon="C")

# Injection of custom CSS to fix dropdown UX and pointer behavior
st.markdown(
    """
    <style>
    div[data-baseweb="select"] > div { cursor: pointer !important; }
    div[data-baseweb="select"] input { cursor: text !important; }
    </style>
    """, unsafe_allow_html=True
)


def render_login_gate():
    st.title("Clinician Dashboard Login")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        if DASHBOARD_PASSWORD_HASH:
            # If hash is configured, only hash-based auth is accepted.
            computed_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
            password_ok = hmac.compare_digest(computed_hash, DASHBOARD_PASSWORD_HASH)
        else:
            # Plaintext fallback is only used when hash is not configured.
            password_ok = bool(DASHBOARD_PASSWORD) and hmac.compare_digest(password, DASHBOARD_PASSWORD)

        if username == DASHBOARD_USERNAME and password_ok:
            st.session_state.dashboard_authenticated = True
            st.session_state.dashboard_auth_expiry = time.time() + (DASHBOARD_SESSION_TIMEOUT_MINUTES * 60)
            st.success("Login successful.")
            st.rerun()
        st.error("Invalid credentials.")
    st.stop()


if "dashboard_authenticated" not in st.session_state:
    st.session_state.dashboard_authenticated = False
if "dashboard_auth_expiry" not in st.session_state:
    st.session_state.dashboard_auth_expiry = 0.0

if st.session_state.dashboard_authenticated and time.time() > st.session_state.dashboard_auth_expiry:
    st.session_state.dashboard_authenticated = False
    st.session_state.dashboard_auth_expiry = 0.0
    st.warning("Session expired. Please sign in again.")

auth_configured = bool(DASHBOARD_USERNAME and (DASHBOARD_PASSWORD or DASHBOARD_PASSWORD_HASH))
if DASHBOARD_REQUIRE_AUTH:
    if not auth_configured:
        st.error(
            "Dashboard auth is required but not configured. "
            "Set DASHBOARD_USERNAME and DASHBOARD_PASSWORD_HASH (recommended) in environment/secrets."
        )
        st.stop()
    if not st.session_state.dashboard_authenticated:
        render_login_gate()
else:
    if auth_configured and not st.session_state.dashboard_authenticated:
        render_login_gate()
    elif not auth_configured:
        st.sidebar.warning(
            "Auth disabled by DASHBOARD_REQUIRE_AUTH=false. "
            "Configure DASHBOARD_USERNAME + DASHBOARD_PASSWORD_HASH before production use."
        )

if st.session_state.dashboard_authenticated:
    # Sliding session timeout: each valid interaction extends the session window.
    st.session_state.dashboard_auth_expiry = time.time() + (DASHBOARD_SESSION_TIMEOUT_MINUTES * 60)
    if st.sidebar.button("Log out"):
        st.session_state.dashboard_authenticated = False
        st.session_state.dashboard_auth_expiry = 0.0
        st.rerun()

try:
    # =================================================================
    # 2. DATA INGESTION (CLOUD POSTGRESQL & SYNTHETIC BASELINE)
    # =================================================================
    # Establish connection to the cloud-hosted PostgreSQL instance (Supabase)
    conn = psycopg2.connect(SUPABASE_URL)
    cur = conn.cursor()
    
    # 2.1 Retrieve scoped longitudinal chat history (bounded lookback + cap)
    cutoff_time = (
        pd.Timestamp.utcnow() - pd.Timedelta(days=DASHBOARD_LOOKBACK_DAYS)
    ).to_pydatetime().replace(tzinfo=None)
    cur.execute(
        """
        SELECT *
        FROM chat_history
        WHERE timestamp >= %s
        ORDER BY timestamp DESC
        LIMIT %s
        """,
        (cutoff_time, DASHBOARD_HISTORY_LIMIT),
    )
    history_rows = cur.fetchall()
    if history_rows:
        history_cols = [desc[0] for desc in cur.description]
        raw_df = pd.DataFrame(history_rows, columns=history_cols)
    else:
        raw_df = pd.DataFrame()

    # 2.2 Retrieve patient profiles (Demographics, Medication, etc.)
    try:
        cur.execute("SELECT * FROM patient_profiles")
        profile_rows = cur.fetchall()
        if profile_rows:
            profile_cols = [desc[0] for desc in cur.description]
            profiles_df = pd.DataFrame(profile_rows, columns=profile_cols)
        else:
            profiles_df = pd.DataFrame()
    except Exception:
        # Fallback to an empty dataframe if the table structure is missing
        profiles_df = pd.DataFrame()
        
    conn.close()
    
    # 2.3 Load the PPMI Synthetic Baseline for cross-sectional clinical validation
    try:
        ppmi_df = pd.read_csv('ppmi_synthetic_baseline.csv')
    except FileNotFoundError:
        ppmi_df = pd.DataFrame()
    
    # Validation check for data presence
    if raw_df.empty:
        st.warning("No data found in the cloud database. Please initialize patient interactions via WhatsApp.")

    # --- Main Dashboard Title ---
    st.title("Parkinson's Disease Longitudinal Tracker (Cloud-Native)")

    # =================================================================
    # 3. GLOBAL NAVIGATION & IDENTITY ANONYMIZATION
    # =================================================================
    st.sidebar.header("Global Navigation")
    st.sidebar.caption(
        f"Data scope: last {DASHBOARD_LOOKBACK_DAYS} days, capped at {DASHBOARD_HISTORY_LIMIT} records."
    )
    
    # Extract unique patient identifiers for the navigation menu
    raw_patient_list = raw_df['patient_id'].unique().tolist() if not raw_df.empty else []
    
    # Create a mapping dictionary for phone numbers to patient names
    profile_dict = pd.Series(profiles_df.name.values, index=profiles_df.patient_id).to_dict() if not profiles_df.empty else {}

    def generate_display_name(phone_str):
        """
        Anonymizes patient identifiers for HIPAA/GDPR compliance.
        Example output: "John D. (***0684)"
        """
        masked_phone = f"***{phone_str[-4:]}" if len(phone_str) >= 4 else "Unknown"
        if phone_str in profile_dict and profile_dict[phone_str] != "Unknown":
            full_name = profile_dict[phone_str]
            first_name = full_name.split()[0]
            last_initial = full_name.split()[1][0] + "." if len(full_name.split()) > 1 else ""
            return f"{first_name} {last_initial} ({masked_phone})"
        return f"Unknown Patient ({masked_phone})"
        
    # Map display names back to raw IDs for backend querying
    display_to_raw_map = {generate_display_name(pid): pid for pid in raw_patient_list}
    display_names = list(display_to_raw_map.keys())
    
    HOME_OPTION = "Home / Overview"
    options_list = [HOME_OPTION] + display_names
    
    selected_display_name = st.sidebar.selectbox("Select Patient EHR:", options_list)
    st.sidebar.markdown("---")
    
    # =================================================================
    # SESSION STATE NAVIGATION GUARD (ANTI-IDM)
    # =================================================================
    # This logic detects view transitions (switching patients or navigating home)
    # and immediately nukes the report cache to prevent IDM background sniffing.
    if "current_view_id" not in st.session_state:
        st.session_state.current_view_id = HOME_OPTION
    if "ai_report" not in st.session_state:
        st.session_state.ai_report = None
    if "pdf_bytes" not in st.session_state:
        st.session_state.pdf_bytes = None
    if "pdf_ready" not in st.session_state:
        st.session_state.pdf_ready = False
    
    # If the user navigates away, wipe all clinical report session data
    if st.session_state.current_view_id != selected_display_name:
        st.session_state.ai_report = None
        st.session_state.pdf_bytes = None
        st.session_state.pdf_ready = False # Specific flag to hide download button
        st.session_state.current_view_id = selected_display_name
    
    # =================================================================
    # 4. VIEW ROUTING: HOME / SYSTEM OVERVIEW
    # =================================================================
    if selected_display_name == HOME_OPTION:
        st.sidebar.info("Data Privacy Mode: Patient identifiers are masked to comply with healthcare regulations.")
        st.markdown("## Welcome to the Clinical Decision Support System")
        st.info("Please select a patient from the sidebar to view their individual Electronic Health Record (EHR).")
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
        - **Cloud-Native Infrastructure**: Secure data storage via PostgreSQL.
        - **Continuous Tracking**: WhatsApp integration for Ecological Momentary Assessment (EMA).
        - **AI-Powered Analysis**: NLP-based extraction of structured clinical symptoms.
        - **Conversational Hauser Diary**: Passive monitoring of medication ON/OFF states.
        - **External Validation**: Benchmarking against Parkinson's Progression Markers Initiative (PPMI) data.
        - **Interoperability**: Automated AI clinical report generation for EHR integration.
        """)
        
    # =================================================================
    # 5. VIEW ROUTING: INDIVIDUAL PATIENT EHR
    # =================================================================
    else:
        # Retrieve the raw patient identifier for filtering
        selected_patient_raw = display_to_raw_map[selected_display_name]
        df = raw_df[raw_df['patient_id'] == selected_patient_raw].copy()

        # Sidebar Indexing for navigation within the EHR
        st.sidebar.subheader("Page Contents")
        st.sidebar.markdown("""
        * [Hauser Diary (Motor)](#hauser)
        * [MoCA Trend (Cognitive)](#moca)
        * [Symptom Logs & Trend](#symptoms)
        * [AI Summary & Export](#report)
        """)
        st.sidebar.markdown("---")
        st.sidebar.caption("Data Privacy Mode Active.")

        tab1, tab2 = st.tabs(["Clinical Dashboard", "Raw Chat Transcript"])
        
        with tab1:
            st.markdown(f"### Clinician Portal: Electronic Health Record (EHR)")
            
            # --- Feature: Patient Demographics Header ---
            patient_profile = pd.DataFrame()
            if not profiles_df.empty:
                patient_profile = profiles_df[profiles_df['patient_id'] == selected_patient_raw]

            if not patient_profile.empty:
                profile_data = patient_profile.iloc[0]
                display_name = profile_data.get("name", "Unknown")
                age_val = profile_data.get("age", "Unknown")
                gender_val = profile_data.get("gender", "Unknown")
                years_val = profile_data.get("years_diagnosed", "Unknown")
                med_val = profile_data.get("current_medication", "Unknown")
            else:
                display_name = "Unknown"
                age_val = "Unknown"
                gender_val = "Unknown"
                years_val = "Unknown"
                med_val = "Unknown"

            if isinstance(display_name, str) and display_name != "Unknown":
                display_name = display_name.split()[0]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.caption("Name (Masked)")
                st.info(f"**{display_name} ***")
            with col2:
                st.caption("Demographics")
                st.info(f"**{age_val} yrs / {gender_val}**" if age_val != "Unknown" else "**Unknown**")
            with col3:
                st.caption("Time since Diagnosis")
                st.info(f"**{years_val} Years**" if years_val != "Unknown" else "**Unknown**")
            with col4:
                st.caption("Current Medication")
                st.info(f"**{med_val}**")
            st.divider()
            
            # --- Feature: Regular Expression Data Extraction (MDS-UPDRS Scoring) ---
            extracted_symp = df['response'].str.extract(
                r"\[SUMMARY\]\s*Symptom:\s*(.*?),\s*Severity:\s*(.*?)(?:,\s*Score:\s*(\d+))?,\s*Context:\s*(.*)"
            )
            extracted_symp.columns = ['Symptom', 'Severity', 'Score', 'Context']
            clinical_df = pd.concat([df, extracted_symp], axis=1).dropna(subset=['Symptom']).copy()
            
            # Mapping categorical severity to numerical weights for quantitative analysis
            severity_weights = {"None": 0, "Low": 1, "Medium": 2, "High": 3}
            clinical_df['Score'] = pd.to_numeric(clinical_df['Score'], errors='coerce')
            clinical_df['Severity_Score'] = clinical_df['Score'].fillna(clinical_df['Severity'].map(severity_weights))

            # Extraction of Motor Fluctuation states (Hauser Diary)
            extracted_hauser = df['response'].str.extract(r"\[HAUSER\] State: (.*?), Context: (.*)")
            extracted_hauser.columns = ['State', 'Context']
            hauser_df = pd.concat([df['timestamp'], extracted_hauser], axis=1).dropna(subset=['State']).copy()
            hauser_df['timestamp'] = pd.to_datetime(hauser_df['timestamp'])

            # Extraction of Cognitive Assessment scores (MoCA subset)
            extracted_moca = df['response'].str.extract(r"\[MOCA\] Score: (.*?)/3, Context: (.*)")
            extracted_moca.columns = ['Score', 'Context']
            moca_df = pd.concat([df['timestamp'], extracted_moca], axis=1).dropna(subset=['Score']).copy()
            moca_df['timestamp'] = pd.to_datetime(moca_df['timestamp'])
            moca_df['Score'] = pd.to_numeric(moca_df['Score']) 

            # --- Chart 1: Conversational Hauser Diary (Motor Fluctuations) ---
            st.subheader("Conversational Hauser Diary (Motor Fluctuations)", anchor="hauser")
            if not hauser_df.empty:
                hauser_colors = alt.Scale(domain=['ON', 'OFF', 'DYSKINESIA', 'ASLEEP'], range=['#28a745', '#dc3545', '#fd7e14', '#6c757d'])
                hauser_df = hauser_df.sort_values('timestamp')
                # Visualize state transitions using a step-after interpolation
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

            # --- Chart 2: MoCA Cognitive Assessment Trend ---
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

            # --- Chart 3: Symptom Logs & External Validation (PPMI Benchmark) ---
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
                st.subheader("Symptom Severity Trend vs PPMI Cohort")
                if not clinical_df.empty:
                    clinical_df['timestamp'] = pd.to_datetime(clinical_df['timestamp'])
                    clinical_df = clinical_df.sort_values(by='timestamp', ascending=True)
                    
                    # Align the synthetic PPMI baseline to the patient's individual timeline
                    if not ppmi_df.empty:
                        first_entry = clinical_df['timestamp'].min()
                        ppmi_df['timestamp'] = ppmi_df['Month'].apply(lambda x: first_entry + pd.DateOffset(months=int(x)-1))
                        ppmi_df['Cohort_Avg'] = ppmi_df[['Tremor_Baseline', 'Sleep_Baseline', 'Mood_Baseline']].mean(axis=1)

                    # Layer 1: Patient's empirical data (Solid Blue Line)
                    line_chart = alt.Chart(clinical_df).mark_line(point=True, color='#1f77b4', strokeWidth=3).encode(
                        x=alt.X('timestamp:T', title='Date & Time', axis=alt.Axis(format='%Y-%m-%d %H:%M', labelAngle=-45)),
                        y=alt.Y('Severity_Score:Q', scale=alt.Scale(domain=[-0.2, 3.2]), title='Severity Score'),
                        tooltip=['timestamp', 'Symptom', 'Severity', 'Context']
                    )
                    
                    # Layer 2: PPMI synthetic cohort baseline (Dashed Red Line)
                    if not ppmi_df.empty:
                        ppmi_chart = alt.Chart(ppmi_df).mark_line(strokeDash=[5, 5], color='#d62728', strokeWidth=2).encode(
                            x='timestamp:T', 
                            y='Cohort_Avg:Q',
                            tooltip=[alt.Tooltip('Month', title='PPMI Month'), alt.Tooltip('Cohort_Avg', title='Cohort Average Score')]
                        )
                        final_chart = (line_chart + ppmi_chart).interactive()
                    else:
                        final_chart = line_chart.interactive()

                    st.altair_chart(final_chart, use_container_width=True)
                    st.caption("Solid Blue: Patient Data | Dashed Red: PPMI Synthetic Cohort Baseline")
                    st.markdown("*Clinical Validation Note: The red dashed line represents the progression baseline derived from the Parkinson's Progression Markers Initiative (PPMI) cohort.*")
                else:
                    st.warning("Insufficient data to generate trend analysis.")
            
            st.markdown("---")
            
            # =================================================================
            # AI CLINICAL REPORT GENERATOR & TWO-STAGE EXPORT (SNIFFING FIX)
            # =================================================================
            st.subheader("AI Pre-Consultation Summary & Export", anchor="report")
            
            # Stage 1: Explicit Text Synthesis (No binary data produced in this phase)
            if st.button("Generate Clinical Analysis"):
                if not clinical_df.empty or not hauser_df.empty:
                    with st.spinner("Executing neuro-analytical summarization via GPT-4o..."):
                        try:
                            # Aggregate data for LLM context window
                            symp_str = clinical_df[['timestamp', 'Symptom', 'Severity']].tail(10).to_string()
                            hauser_str = hauser_df[['timestamp', 'State']].tail(10).to_string()
                            
                            prompt = f"Analyze Parkinson's tracking data for {selected_display_name}. Symptoms: {symp_str}. Hauser: {hauser_str}. Write a professional 3-paragraph clinical summary."
                            
                            response = client.chat.completions.create(
                                model="gpt-4o",
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.2
                            )
                            # Cache the report text in session state
                            st.session_state.ai_report = response.choices[0].message.content
                            st.session_state.pdf_ready = False # Keep the binary hidden
                        except Exception as e:
                            st.error(f"Analysis failed: {e}")
                else:
                    st.warning("Insufficient data.")

            # Stage 2: Narrative Review and Conditional Binary Preparation
            if st.session_state.ai_report:
                st.write(st.session_state.ai_report)
                
                # We hide the download button behind an extra manual step to deceive IDM sniffers
                if not st.session_state.pdf_ready:
                    if st.button(" Prepare PDF for Export"):
                        try:
                            pdf = FPDF()
                            pdf.add_page()
                            pdf.set_font("Arial", 'B', 15)
                            pdf.cell(0, 10, 'Neurological Assessment Report', 0, 1, 'C')
                            pdf.set_font("Arial", size=11)
                            pdf.ln(5)
                            pdf.cell(0, 10, f"Patient: {selected_display_name}", 0, 1)
                            pdf.cell(0, 10, f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}", 0, 1)
                            pdf.ln(5)
                            # Handle encoding for PDF compatibility
                            safe_txt = st.session_state.ai_report.encode('latin-1', 'replace').decode('latin-1')
                            pdf.multi_cell(0, 8, safe_txt)
                            
                            st.session_state.pdf_bytes = pdf.output(dest='S').encode('latin-1')
                            st.session_state.pdf_ready = True
                            st.rerun() # Refresh once to render the actual download link
                        except Exception as e:
                            st.error(f"Binary build failed: {e}")
                
                # Final Phase: Render the actual download link only when ready
                if st.session_state.pdf_ready and st.session_state.pdf_bytes:
                    st.success("Clinical report finalized and ready for download.")
                    st.download_button(
                        label="Download Clinical PDF Report",
                        data=st.session_state.pdf_bytes,
                        file_name=f"Report_{selected_patient_raw}.pdf",
                        mime="application/pdf"
                    )
        # --- TAB 2: Raw Conversation Transcript Archive ---
        with tab2:
            st.markdown(f"### Historical Interaction Archive: {selected_display_name}")
            chat_name = profile_dict.get(selected_patient_raw, "Unknown").split()[0] + " (Patient/Carer)" if selected_patient_raw in profile_dict and profile_dict[selected_patient_raw] != "Unknown" else "Unknown Patient"

            if df.empty:
                st.info("No conversation history found for this patient yet.")
            else:
                transcript_df = df.copy()
                transcript_df["timestamp"] = pd.to_datetime(transcript_df["timestamp"], errors="coerce")
                transcript_df = transcript_df.dropna(subset=["timestamp"]).sort_values(
                    "timestamp", ascending=False
                ).reset_index(drop=True)

                if transcript_df.empty:
                    st.info("No valid timestamped messages found for this patient.")
                else:
                    min_date = transcript_df["timestamp"].min().date()
                    max_date = transcript_df["timestamp"].max().date()

                    filter_col1, filter_col2, filter_col3 = st.columns([2, 1, 1.2])
                    with filter_col1:
                        date_range = st.date_input(
                            "Filter by date range",
                            value=(min_date, max_date),
                            min_value=min_date,
                            max_value=max_date,
                            key=f"transcript_date_range_{selected_patient_raw}",
                        )
                    with filter_col2:
                        page_size = st.selectbox(
                            "Rows per page",
                            options=[10, 20, 50],
                            index=0,
                            key=f"transcript_page_size_{selected_patient_raw}",
                        )

                    if isinstance(date_range, tuple):
                        if len(date_range) == 2:
                            start_date, end_date = date_range
                        elif len(date_range) == 1:
                            start_date = end_date = date_range[0]
                        else:
                            start_date, end_date = min_date, max_date
                    else:
                        start_date = end_date = date_range

                    if start_date > end_date:
                        start_date, end_date = end_date, start_date

                    start_ts = pd.Timestamp(start_date)
                    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                    filtered_df = transcript_df[
                        (transcript_df["timestamp"] >= start_ts) & (transcript_df["timestamp"] <= end_ts)
                    ].reset_index(drop=True)

                    total_rows = len(filtered_df)
                    with filter_col3:
                        st.metric("Filtered records", total_rows)

                    if total_rows == 0:
                        st.info("No messages in the selected date range.")
                    else:
                        page_key = f"transcript_page_{selected_patient_raw}"
                        filter_sig_key = f"transcript_filter_signature_{selected_patient_raw}"
                        filter_signature = f"{start_date}_{end_date}_{page_size}"

                        if page_key not in st.session_state:
                            st.session_state[page_key] = 1
                        if filter_sig_key not in st.session_state:
                            st.session_state[filter_sig_key] = filter_signature
                        elif st.session_state[filter_sig_key] != filter_signature:
                            st.session_state[filter_sig_key] = filter_signature
                            st.session_state[page_key] = 1

                        total_pages = max(1, math.ceil(total_rows / page_size))
                        st.session_state[page_key] = min(max(1, st.session_state[page_key]), total_pages)

                        nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 2])
                        with nav_col1:
                            if st.button(
                                "Previous page",
                                key=f"transcript_prev_{selected_patient_raw}",
                                disabled=st.session_state[page_key] <= 1,
                            ):
                                st.session_state[page_key] -= 1
                                st.rerun()
                        with nav_col2:
                            if st.button(
                                "Next page",
                                key=f"transcript_next_{selected_patient_raw}",
                                disabled=st.session_state[page_key] >= total_pages,
                            ):
                                st.session_state[page_key] += 1
                                st.rerun()
                        with nav_col3:
                            selected_page = st.number_input(
                                "Page",
                                min_value=1,
                                max_value=total_pages,
                                value=int(st.session_state[page_key]),
                                step=1,
                                key=f"transcript_page_input_{selected_patient_raw}",
                            )
                            if selected_page != st.session_state[page_key]:
                                st.session_state[page_key] = int(selected_page)
                                st.rerun()

                        current_page = int(st.session_state[page_key])
                        start_idx = (current_page - 1) * page_size
                        end_idx = min(start_idx + page_size, total_rows)
                        page_df = filtered_df.iloc[start_idx:end_idx]

                        st.caption(
                            f"Showing records {start_idx + 1}-{end_idx} of {total_rows} | Page {current_page}/{total_pages}"
                        )

                        for _, row in page_df.iterrows():
                            with st.chat_message("user"):
                                st.write(f"**{chat_name}** ({row['timestamp']}): {row['user_message']}")

                            with st.chat_message("assistant"):
                                ai_text = str(row["response"])
                                # Cleaning internal tags before displaying to clinicians
                                ai_text = strip_internal_tags(ai_text)
                                st.write(f"**AI Assistant**: {ai_text}")
                            st.divider()

# Global exception handling for database or system critical errors
except psycopg2.Error as e:
    st.error(f"Database Connection Error: {e}")
except Exception as e:
    st.error(f"System Critical Error: {e}")


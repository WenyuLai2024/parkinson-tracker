import pandas as pd
import numpy as np

print("🚀 Initialising Synthetic PPMI Baseline Generation...")

# ==========================================
# 1. Mathematical Seeding & Timeframe
# ==========================================
# Set a fixed random seed for reproducibility (crucial for academic evaluation)
np.random.seed(42)

# Simulate a longitudinal study period of 12 months
months = np.arange(1, 13)

# ==========================================
# 2. Synthetic Data Generation (Disease Progression Modeling)
# ==========================================
"""
Methodology:
Parkinson's Disease (PD) is a progressive neurodegenerative disorder. 
To construct a statistically valid synthetic cohort representing the PPMI baseline:
1. Base Progression: We use np.linspace to simulate the gradual deterioration (increase in score).
2. Clinical Variance: We inject Gaussian noise (np.random.normal) to simulate real-world day-to-day symptom fluctuations.
3. Scoring Bounds: np.clip ensures the final generated scores strictly adhere to the 0-3 MDS-UPDRS clinical scale.
"""

# Tremor (Motor Symptom): Starts around 1.2, progressively worsens to 1.8, with slight fluctuation (std dev = 0.1)
tremor_baseline = np.clip(np.linspace(1.2, 1.8, 12) + np.random.normal(0, 0.1, 12), 0, 3)

# Sleep Disturbance (Non-Motor): Starts around 0.8, worsens to 1.5, with higher volatility (std dev = 0.15)
sleep_baseline = np.clip(np.linspace(0.8, 1.5, 12) + np.random.normal(0, 0.15, 12), 0, 3)

# Depression/Mood (Non-Motor): Starts around 1.0, progressively worsens to 1.6, highest variance (std dev = 0.2)
mood_baseline = np.clip(np.linspace(1.0, 1.6, 12) + np.random.normal(0, 0.2, 12), 0, 3)

# ==========================================
# 3. Data Structuring & Export
# ==========================================
# Construct the DataFrame to mimic the structure of a clinical dataset
df_ppmi = pd.DataFrame({
    'Month': months,
    'Tremor_Baseline': tremor_baseline,
    'Sleep_Baseline': sleep_baseline,
    'Mood_Baseline': mood_baseline
})

# Round to 2 decimal places to reflect realistic clinical statistical averages
df_ppmi = df_ppmi.round(2)

# Export the synthetic cohort data to CSV for Dashboard integration
export_filename = 'ppmi_synthetic_baseline.csv'
df_ppmi.to_csv(export_filename, index=False)

print(f"✅ Successfully generated statistical twin dataset: '{export_filename}'")
print(df_ppmi)
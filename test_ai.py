import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import time

# Import the core AI processing function from the existing backend
from app_ai import get_ai_response

print("🚀 Initialising Automated AI Evaluation Pipeline...")

# ==========================================
# 1. Dataset Loading
# ==========================================
try:
    # Load the evaluation dataset containing test cases and ground truth labels
    df = pd.read_csv('test_dataset.csv')
    print(f"✅ Dataset loaded successfully. Total test cases: {len(df)}")
except FileNotFoundError:
    print("❌ Error: 'test_dataset.csv' not found. Please check the directory.")
    exit()

# Initialize arrays to store raw AI outputs and parsed numerical scores
ai_predictions = []
extracted_scores = []

# ==========================================
# 2. Automated Testing Execution
# ==========================================
print("-" * 50)
for index, row in df.iterrows():
    test_id = row['Test ID']
    user_input = row['Patient Input']
    ground_truth = str(row['Ground Truth']).strip()
    
    print(f"[{index+1}/{len(df)}] Evaluating {test_id}...")
    
    # Invoke the LLM endpoint (simulating a first-time interaction with empty context)
    # Using "AUTO_TESTER" as a mock patient_id to isolate test data in the cloud database
    ai_response = get_ai_response(user_input, [], "AUTO_TESTER", None, persist_log=False)
    ai_predictions.append(ai_response)
    
    # Regex extraction: Isolate the numerical score from the structured [SUMMARY] tag
    match = re.search(r"Score: (\d+)", ai_response)
    if match:
        pred_score = match.group(1)
    else:
        pred_score = "Error" # Flag for unparseable or non-compliant AI outputs
        
    extracted_scores.append(pred_score)
    print(f"   Ground Truth: {ground_truth} | AI Prediction: {pred_score}")
    
    # Rate limiting to prevent OpenAI API 429 Too Many Requests error
    time.sleep(1)

print("-" * 50)

# ==========================================
# 3. Metrics Calculation & Data Export
# ==========================================
# Append results back to the dataframe for comprehensive reporting
df['AI Prediction Text'] = ai_predictions
df['Predicted Score'] = extracted_scores

# Filter out unparseable responses to ensure valid metric calculations
valid_df = df[df['Predicted Score'] != "Error"].copy()

# Cast scores to integers for scikit-learn metric functions
valid_df['Ground Truth'] = valid_df['Ground Truth'].astype(int)
valid_df['Predicted Score'] = valid_df['Predicted Score'].astype(int)

# Calculate overall system accuracy
accuracy = accuracy_score(valid_df['Ground Truth'], valid_df['Predicted Score'])
print(f"🏆 Evaluation Complete! System Accuracy: {accuracy * 100:.2f}%")

# Export the detailed interaction log for the Thesis Appendix
df.to_csv('test_report_results.csv', index=False)
print("📄 Detailed evaluation report saved as 'test_report_results.csv'")

# ==========================================
# 4. Visualisation: Confusion Matrix
# ==========================================
print("📊 Generating Confusion Matrix for performance analysis...")
labels = [0, 1, 2, 3] # UPDRS/MDS severity scale
cm = confusion_matrix(valid_df['Ground Truth'], valid_df['Predicted Score'], labels=labels)

# Configure plot aesthetics
plt.figure(figsize=(8, 6))
# Render heatmap using an academic blue color palette
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, annot_kws={"size": 16})

plt.title('AI Symptom Extraction Confusion Matrix', fontsize=16)
plt.ylabel('Actual Ground Truth Score (0-3)', fontsize=14)
plt.xlabel('AI Predicted Score (0-3)', fontsize=14)

# Export high-resolution figure suitable for academic publication
plt.savefig('confusion_matrix_evaluation.png', dpi=300, bbox_inches='tight')
print("📸 Evaluation chart saved successfully: 'confusion_matrix_evaluation.png'")

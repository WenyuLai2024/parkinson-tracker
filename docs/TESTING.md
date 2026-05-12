# Testing and Reproducibility

## Scope
This project uses mixed verification:
- system-level path checks (webhook, multimodal, dashboard, access boundaries)
- offline extraction evaluation on fixed datasets

## 1) Quick Integrity Checks

Run compile checks:
```bash
python -m compileall app.py app_ai.py dashboard.py clinical_utils.py test_ai.py
```

Run optional unit tests:
```bash
pytest -q
```

## 2) System Path Verification

Reference checklist: `TEST_PROTOCOL.md`.

Recommended sequence:
1. Start backend: `python app.py`
2. Send WhatsApp text message and verify DB write.
3. Send WhatsApp voice message and verify transcription path.
4. Send image (clock-drawing style) and verify vision-assisted flow.
5. Verify caregiver access boundary behavior.
6. Verify high-risk alert trigger.

## 3) Reproduce Offline Extraction Evaluation

Default run:
```bash
python test_ai.py --dataset test_dataset_mds04_native.csv --output-prefix mds04_native_
```

Expected output files:
- `mds04_native_test_report_results.csv`
- `mds04_native_confusion_matrix_evaluation.png`

Alternative legacy naming (no prefix):
- `test_report_results.csv`
- `confusion_matrix_evaluation.png`

## 4) Determinism and Drift Notes

- The dataset, parser rules, and output schema are fixed.
- LLM outputs are probabilistic; exact metrics may vary across runs.
- Keep prompt versions and parsing logic unchanged when reproducing reported numbers.

## 5) Evidence Packaging for Review

Minimum recommended artefacts:
- latest commit hash
- evaluation CSV output
- confusion matrix figure
- short run log excerpt for backend startup and scheduler status

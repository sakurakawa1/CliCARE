import os
import json
import re

# Cancer-related keywords
CANCER_KEYWORDS = [
    'cancer', 'carcinoma', 'malignant', 'tumor', 'tumour', 'neoplasm', 'sarcoma', 'lymphoma', 'leukemia', 'myeloma', 'adenocarcinoma', 'melanoma', 'glioma', 'blastoma', 'metastatic', 'metastasis', 'metastasized', 'malignancy'
]

# --- Configuration ---
INPUT_DIR = 'clinic_json'
OUTPUT_DIR = 'clinic_json_cancer'
OUTPUT_DATA_DIR = 'clinic_json_cancer_data'
MAX_DATA = 2500
MIN_RECORDS = 8
# --- End Configuration ---

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)

def contains_cancer(text):
    """Checks if a given text contains any of the cancer-related keywords."""
    text_lower = text.lower()
    for kw in CANCER_KEYWORDS:
        if kw in text_lower:
            return True
    return False

def filter_patient_file(input_path, output_path):
    """
    Filters a patient's record file for admissions related to cancer and saves the result.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    text = data.get('text', '')
    
    # Split by admission record
    records = re.split(r'The patient was recorded for the \d+ time:', text)
    headers = re.findall(r'The patient was recorded for the \d+ time:', text)
    filtered_records = []

    # Note: records[0] is usually empty due to the split pattern
    for i, rec in enumerate(records[1:], start=0):
        if not rec.strip():
            continue
        
        # Check the diagnosis section of each record
        match = re.search(r'patient diagnosis information:([^\n]*)', rec)
        if match:
            diag_text = match.group(1)
            if contains_cancer(diag_text):
                # Keep this admission record if it's cancer-related
                if i < len(headers):
                    filtered_records.append(headers[i] + rec) # Keep leading/trailing spaces from rec
                else: # Fallback, should not happen with correct regex
                    filtered_records.append(rec)

    if filtered_records:
        # Reassemble the text with only the filtered records
        new_text = '\n\n'.join(filtered_records)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({'text': new_text}, f, ensure_ascii=False, indent=4)
        return len(filtered_records)
        
    return 0

def main():
    """
    Main function to process all patient files, filter for cancer cases,
    and select a subset for the final dataset.
    """
    patient_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
    cancer_patient_counts = []
    
    print(f"Filtering {len(patient_files)} patient files for cancer-related admissions...")
    for fname in patient_files:
        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, fname)
        count = filter_patient_file(in_path, out_path)
        if count >= MIN_RECORDS:
            cancer_patient_counts.append((fname, count))
    
    print(f"Found {len(cancer_patient_counts)} patients with at least {MIN_RECORDS} cancer-related admissions.")
    
    # Select the top MAX_DATA cases with the most admission records (must be >= MIN_RECORDS)
    cancer_patient_counts.sort(key=lambda x: -x[1])
    
    print(f"Copying the top {min(MAX_DATA, len(cancer_patient_counts))} files to the final data directory...")
    for i, (fname, count) in enumerate(cancer_patient_counts[:MAX_DATA]):
        src = os.path.join(OUTPUT_DIR, fname)
        dst = os.path.join(OUTPUT_DATA_DIR, fname)
        with open(src, 'r', encoding='utf-8') as fsrc, open(dst, 'w', encoding='utf-8') as fdst:
            json.dump(json.load(fsrc), fdst, ensure_ascii=False, indent=4)
        print(f"  ({i+1}) Copied {fname} with {count} records.")

if __name__ == '__main__':
    main()
import pandas as pd
import re

def extract_cancer_diagnoses(csv_path, output_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Define cancer-related keywords
    cancer_keywords = [
        'cancer', 'tumor', 'tumour', 'carcinoma', 'sarcoma', 'leukemia', 'lymphoma',
        'melanoma', 'neoplasm', 'malignant', 'metastasis', 'metastatic',
        'cancerous', 'oncology', 'oncological'
    ]
    
    # Create a regular expression pattern
    pattern = '|'.join(cancer_keywords)
    
    # Find rows in the 'long_title' column that contain cancer keywords
    cancer_mask = df['long_title'].str.contains(pattern, case=False, na=False)
    cancer_cases = df[cancer_mask]
    
    # Extract subject_id and save to a txt file
    cancer_subject_ids = cancer_cases['subject_id'].unique()
    
    with open(output_path, 'w') as f:
        for subject_id in cancer_subject_ids:
            f.write(f"{subject_id}\n")
    
    print(f"Found {len(cancer_subject_ids)} cancer-related cases")
    print(f"Results have been saved to: {output_path}")

if __name__ == "__main__":
    # Set the input and output file paths
    input_csv = "./extract_hosp/diagnoses_icd.csv"
    output_txt = "cancer_subject_ids.txt"
    
    extract_cancer_diagnoses(input_csv, output_txt)
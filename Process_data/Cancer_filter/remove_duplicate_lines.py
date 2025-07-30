import os
import json
import re

# --- Configuration ---
INPUT_DIR = 'clinic_json_cancer_data_api'
OUTPUT_DIR = 'clinic_json_cancer_data_api_remove'
os.makedirs(OUTPUT_DIR, exist_ok=True)
# --- End Configuration ---

def remove_duplicate_phrases(text):
    """
    Removes duplicate phrases and sub-phrases from a given text.
    """
    # First, split by common delimiters
    phrases = re.split(r'[，,;；\n]', text)
    final_phrases = []
    
    for phrase in phrases:
        phrase = phrase.strip()
        if not phrase:
            continue
            
        # If this segment is long and contains the same sub-phrase multiple times, split by space and deduplicate
        if len(phrase) > 30 and ' ' in phrase:
            sub_phrases = phrase.split(' ')
            seen_sub = set()
            dedup_sub = []
            for sub in sub_phrases:
                if sub and sub not in seen_sub:
                    seen_sub.add(sub)
                    dedup_sub.append(sub)
            final_phrases.append(' '.join(dedup_sub))
        else:
            final_phrases.append(phrase)
            
    # Then, perform global deduplication on the processed phrases
    seen_globally = set()
    result = []
    for p in final_phrases:
        if p and p not in seen_globally:
            seen_globally.add(p)
            result.append(p)
            
    return ', '.join(result)

def process_file(filepath, outpath):
    """
    Processes a single JSON file to deduplicate content within each admission record.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    text = data.get('text', '')
    
    # Split by admission record
    # The pattern creates a list like: ['', header1, record1, header2, record2, ...]
    records = re.split(r'(The patient was recorded for the \d+ time:)', text)
    filtered_records = []
    
    for i in range(1, len(records), 2):
        header = records[i]
        record_content = records[i+1].strip() if i+1 < len(records) else ''
        
        if not record_content:
            continue
            
        deduplicated_record = remove_duplicate_phrases(record_content)
        filtered_records.append(header + '\n' + deduplicated_record)
        
    if filtered_records:
        new_text = '\n\n'.join(filtered_records)
        data['text'] = new_text
        with open(outpath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"[INFO] Processed and deduplicated: {os.path.basename(filepath)} -> {os.path.basename(outpath)}")
    else:
        print(f"[INFO] No valid admission records in {os.path.basename(filepath)}, skipped.")

def main():
    """
    Main function to find and process all JSON files in the input directory.
    """
    files_to_process = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
    print(f"Found {len(files_to_process)} files to process.")
    for filename in files_to_process:
        filepath = os.path.join(INPUT_DIR, filename)
        outpath = os.path.join(OUTPUT_DIR, filename)
        process_file(filepath, outpath)
    print("All files have been processed.")

if __name__ == '__main__':
    main()
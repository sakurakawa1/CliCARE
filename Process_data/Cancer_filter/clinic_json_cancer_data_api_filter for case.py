import os
import json
import re
from openai import OpenAI
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
OPENAI_API_KEY = "sk-XXX"  # Replace with your actual key
OPENAI_BASE_URL = "https://api.deepseek.com"

INPUT_DIR = 'extracted_cancer_cases'
OUTPUT_DIR = 'extracted_cancer_cases_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
# --- End Configuration ---

def gpt_filter_cancer_info(text):
    """
    Uses the GPT model to extract only cancer-related information from a medical record text.
    """
    prompt = (
        "Extract ONLY cancer-related information from this medical record. "
        "Keep: cancer diagnosis, tumor details, malignant findings, cancer medications, cancer tests, cancer surgeries, cancer treatments. "
        "Remove: non-cancer conditions, routine care, unrelated medications, general health info. "
        "If no cancer info found, OR if only a cancer diagnosis exists but there are NO related tests/treatments/medications, respond with: __NO_CANCER_RELATED_RECORD__. "
        "Only output cancer-related content. Keep the response concise. "
        "Record:\n"
        f"{text}\n"
        "Extracted cancer info:"
    )
    for attempt in range(1, 4):  # Retry up to 3 times
        print(f"[DEBUG] API call attempt {attempt}, prompt length: {len(prompt)}, prompt head: {prompt[:100].replace(chr(10), ' ')} ...")
        try:
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                temperature=0.1,
                max_tokens=40000
            )
            print("[DEBUG] API call success.")
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[DEBUG] API error on attempt {attempt}: {e}")
            time.sleep(2)
    print("[DEBUG] GPT API call failed after 3 attempts.")
    return ""  # Return empty string on failure

def process_file(input_path, output_path):
    """
    Processes a single JSON file, filters its content for cancer-related information,
    and saves the result to a new JSON file.
    """
    # If the output file already exists, skip processing.
    if os.path.exists(output_path):
        print(f"[INFO] {os.path.basename(output_path)} already exists, skipping processing.")
        return
        
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    text = data.get('text', '')
    
    # Check if it contains the admission record split marker
    if 'The patient\'s admission was recorded for the' in text:
        # Split by admission record
        # The pattern creates a list like: ['', header1, record1, header2, record2, ...]
        records = re.split(r'(The patient\'s admission was recorded for the \d+ time:)', text)
        filtered_admissions = []
        for i in range(1, len(records), 2):
            header = records[i]
            record_content = records[i+1].strip() if i+1 < len(records) else ''
            
            if not record_content:
                continue
            
            # Length limit: truncate to 62K if over 64K
            if len(record_content) > 64 * 1024:
                record_content = record_content[:62 * 1024]
            
            filtered_content = gpt_filter_cancer_info(record_content)
            
            if filtered_content and filtered_content != "__NO_CANCER_RELATED_RECORD__":
                filtered_admissions.append(header + '\n' + filtered_content)
    else:
        # Only one admission record, filter the entire text directly
        record_content = text.strip()
        if len(record_content) > 64 * 1024:
            record_content = record_content[:62 * 1024]
            
        filtered_content = gpt_filter_cancer_info(record_content)
        filtered_admissions = []
        if filtered_content and filtered_content != "__NO_CANCER_RELATED_RECORD__":
            filtered_admissions.append(filtered_content)
    
    if filtered_admissions:
        new_text = '\n\n'.join(filtered_admissions)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({'text': new_text}, f, ensure_ascii=False, indent=4)
        print(f"[SUCCESS] Processed and saved {os.path.basename(output_path)}")
    else:
        # If no cancer-related content, do not generate the file.
        if os.path.exists(output_path):
            os.remove(output_path)
        print(f"[INFO] No cancer-related content found in {os.path.basename(input_path)}, skipped.")

def find_all_json_files(root_dir):
    """New function to recursively find all json files."""
    json_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith('.json'):
                json_files.append(os.path.join(dirpath, fname))
    return json_files

def main():
    """Main function to run the batch processing."""
    files_to_process = find_all_json_files(INPUT_DIR)
    print(f"[INFO] Found {len(files_to_process)} JSON files to process.")
    
    tasks = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        for input_path in files_to_process:
            # Construct a relative path to maintain subfolder structure
            relative_path = os.path.relpath(os.path.dirname(input_path), INPUT_DIR)
            output_sub_dir = os.path.join(OUTPUT_DIR, relative_path)
            os.makedirs(output_sub_dir, exist_ok=True)
            
            output_path = os.path.join(output_sub_dir, os.path.basename(input_path))
            tasks.append(executor.submit(process_file, input_path, output_path))
            
        for future in as_completed(tasks):
            try:
                future.result()
            except Exception as e:
                print(f"[ERROR] An error occurred while processing a file: {e}")

if __name__ == '__main__':
    main()  # Start batch processing
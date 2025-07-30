import os
import json
import re
from openai import OpenAI
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
OPENAI_API_KEY = "sk-XXX"  # Replace with your actual key
OPENAI_BASE_URL = "https://api.deepseek.com"

INPUT_DIR = 'clinic_json_cancer_data'
OUTPUT_DIR = 'clinic_json_cancer_data_api'
os.makedirs(OUTPUT_DIR, exist_ok=True)
# --- End Configuration ---

def gpt_filter_cancer_info(text):
    """
    Uses a GPT model to filter a medical record for cancer-related information.
    """
    prompt = (
        "You are a medical information extraction assistant. Read the following hospitalization record and keep ONLY all information related to cancer (including diagnosis, medication, tests, surgeries, etc. related to tumor, malignant, neoplasm, carcinoma, cancer, etc.)."
        "Remove all non-cancer-related content. "
        "If there is no cancer-related information, output ONLY: __NO_CANCER_RELATED_RECORD__."
        "Do not add any explanation. "
        "The original record is:\n"
        f"{text}\n"
        "Please output the filtered result:"
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
    return ""  # Return empty on failure

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

    # Split by admission record
    # The pattern creates a list like: ['', header1, record1, header2, record2, ...]
    records = re.split(r'(The patient was recorded for the \d+ time:)', text)
    filtered_admissions = []
    
    for i in range(1, len(records), 2):
        header = records[i]
        record_content = records[i+1] if i+1 < len(records) else ''
        if record_content.strip():
            filtered_content = gpt_filter_cancer_info(record_content)
            if filtered_content and filtered_content != "__NO_CANCER_RELATED_RECORD__":
                filtered_admissions.append(header + '\n' + filtered_content)
                
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

def main():
    """
    Main function to find all JSON files and process them using a thread pool.
    """
    files_to_process = [f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
    print(f"[INFO] Found {len(files_to_process)} JSON files to process.")
    
    tasks = []
    # Adjust the number of threads according to the actual situation
    with ThreadPoolExecutor(max_workers=16) as executor:
        for filename in files_to_process:
            input_path = os.path.join(INPUT_DIR, filename)
            output_path = os.path.join(OUTPUT_DIR, filename)
            tasks.append(executor.submit(process_file, input_path, output_path))
            
        for future in as_completed(tasks):
            try:
                future.result()
            except Exception as e:
                print(f"[ERROR] An error occurred while processing a file: {e}")

if __name__ == '__main__':
    main()  # Start batch processing
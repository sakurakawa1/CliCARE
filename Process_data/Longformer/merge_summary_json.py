import os
import json

def process_json_file(input_filepath, output_filepath):
    """
    Processes a JSON file containing multiple records, merging them into a single text field
    with summaries for previous records and the full text for the latest one.
    """
    with open(input_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Get all record_x items and sort them in order
    record_items = [(k, v) for k, v in data.items() if k.startswith('record_')]
    record_items.sort(key=lambda x: int(x[0].split('_')[1]))
    records = [v for k, v in record_items]

    merged_texts = []
    for idx, record in enumerate(records):
        if idx == len(records) - 1:
            # Use original_text for the last record
            text = record.get('original_text', '')
        else:
            # Use summary for the others
            text = record.get('summary', '')
        merged_texts.append(f"The patient was recorded for the {idx+1} time:\n{text}")

    # Add separators
    if len(records) > 1:
        # If there are multiple records, add separators
        summary_records = merged_texts[:-1]  # All records except the last one
        latest_record = merged_texts[-1]     # The last record
        
        merged = "--- Previous Records Summary ---\n\n" + "\n".join(summary_records) + "\n\n--- Latest Record (Uncompressed) ---\n\n" + latest_record
    else:
        # If there is only one record, use it directly
        merged = "\n".join(merged_texts)

    # Build a new dictionary, keeping only non-record_x fields
    new_data = {k: v for k, v in data.items() if not k.startswith('record_')}
    new_data['text'] = merged

    # Output to a new file
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)

def main():
    """
    Main function to process all JSON files in the input folder and save
    the results to the output folder.
    """
    input_folder = 'Summary_data'
    output_folder = 'Summary_data_output'
    os.makedirs(output_folder, exist_ok=True)
    
    file_list = [f for f in os.listdir(input_folder) if f.endswith('.json')]
    print(f"Found {len(file_list)} files to process.")

    for filename in file_list:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        try:
            process_json_file(input_path, output_path)
            print(f"Successfully processed {filename}")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
    
    print("Processing complete.")


if __name__ == '__main__':
    main()
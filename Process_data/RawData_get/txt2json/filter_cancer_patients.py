import pandas as pd
import os
import shutil

def create_output_dirs():
    """Create output directories"""
    os.makedirs('./cancer_hosp', exist_ok=True)
    os.makedirs('./cancer_icu', exist_ok=True)

def read_cancer_subject_ids(file_path):
    """Read the list of cancer patient IDs"""
    with open(file_path, 'r') as f:
        # Ensure IDs are converted to integers
        return set(int(line.strip()) for line in f if line.strip())

def process_csv_file(input_file, output_file, cancer_subject_ids):
    """Process a single CSV file"""
    print(f"Processing file: {input_file}")
    
    try:
        # First, read the first few lines of the file to check column names
        df_sample = pd.read_csv(input_file, nrows=5, low_memory=False)
        if 'subject_id' not in df_sample.columns:
            print(f"Warning: 'subject_id' column not found in {input_file}")
            return
        
        # Read large files in chunks
        chunk_size = 5000000
        first_chunk = True
        total_rows = 0
        filtered_rows = 0
        
        # Set data types
        dtype_dict = {
            'subject_id': 'Int64',  # Use nullable integer type
            'hadm_id': 'Int64',
            'icustay_id': 'Int64' # Corrected from stay_id to icustay_id if that's the common name
        }
        
        # Adjust dtype dict to only include columns present in the sample
        actual_dtype = {k: v for k, v in dtype_dict.items() if k in df_sample.columns}
        
        for chunk in pd.read_csv(input_file, 
                                 chunksize=chunk_size, 
                                 low_memory=False,
                                 dtype=actual_dtype):
            # Ensure the subject_id column is of numeric type
            chunk['subject_id'] = pd.to_numeric(chunk['subject_id'], errors='coerce')
            
            # Filter rows that contain cancer patient IDs
            filtered_chunk = chunk[chunk['subject_id'].isin(cancer_subject_ids)]
            
            total_rows += len(chunk)
            filtered_rows += len(filtered_chunk)
            
            # Write to file
            if len(filtered_chunk) > 0:
                if first_chunk:
                    filtered_chunk.to_csv(output_file, index=False, mode='w')
                    first_chunk = False
                else:
                    filtered_chunk.to_csv(output_file, index=False, mode='a', header=False)
        
        if total_rows > 0:
            print(f"File {input_file} processing complete:")
            print(f"- Total rows: {total_rows}")
            print(f"- Filtered rows: {filtered_rows}")
            print(f"- Filtering ratio: {(filtered_rows/total_rows*100):.2f}%")
        else:
            print(f"File {input_file} is empty or could not be processed.")

    except Exception as e:
        print(f"Error processing file {input_file}: {str(e)}")

def main():
    # Create output directories
    create_output_dirs()
    
    # Read cancer patient IDs
    cancer_ids_file = 'cancer_subject_ids.txt'
    if not os.path.exists(cancer_ids_file):
        print(f"Error: Cancer IDs file not found at '{cancer_ids_file}'")
        return
        
    cancer_subject_ids = read_cancer_subject_ids(cancer_ids_file)
    print(f"Read {len(cancer_subject_ids)} cancer patient IDs")
    print("ID sample:", list(cancer_subject_ids)[:5])  # Show the first 5 IDs as an example
    
    # Process hospital data
    hosp_dir = './extract_hosp'
    print("\n--- Processing Hospital Data ---")
    if os.path.isdir(hosp_dir):
        for file in os.listdir(hosp_dir):
            if file.endswith('.csv'):
                input_path = os.path.join(hosp_dir, file)
                output_path = os.path.join('./cancer_hosp', file)
                process_csv_file(input_path, output_path, cancer_subject_ids)
    else:
        print(f"Directory not found: {hosp_dir}")

    # Process ICU data
    icu_dir = './extract_icu'
    print("\n--- Processing ICU Data ---")
    if os.path.isdir(icu_dir):
        for file in os.listdir(icu_dir):
            if file.endswith('.csv'):
                input_path = os.path.join(icu_dir, file)
                output_path = os.path.join('./cancer_icu', file)
                process_csv_file(input_path, output_path, cancer_subject_ids)
    else:
        print(f"Directory not found: {icu_dir}")

    print("\nAll files processed successfully!")

if __name__ == "__main__":
    main()
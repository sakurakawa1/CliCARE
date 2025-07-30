import pandas as pd
import os
import glob
import warnings
import gc  # Add garbage collection module

# Suppress specific pandas warnings for chained assignment
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# Define input and output directories
HOSP_DIR = './hosp'
ICU_DIR = './icu'
EXTRACT_HOSP_DIR = './extract_hosp'
EXTRACT_ICU_DIR = './extract_icu'

# Ensure output directories exist
os.makedirs(EXTRACT_HOSP_DIR, exist_ok=True)
os.makedirs(EXTRACT_ICU_DIR, exist_ok=True)

# Define "Useful Fields" for Extraction
# This dictionary maps table names to their selected columns.
# Note: d_micro is explicitly excluded as it was removed in MIMIC-IV v0.4.[5]
USEFUL_FIELDS = {
    # Patient basic information
    'patients': ['subject_id', 'gender', 'anchor_age', 'anchor_year', 'anchor_year_group', 'dod'],
    
    # Admission information
    'admissions': ['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime', 'admission_type',
                   'admission_location', 'discharge_location', 'insurance', 'race', 'hospital_expire_flag'],
    
    # Transfer information
    'transfers': ['subject_id', 'hadm_id', 'transfer_id', 'eventtype', 'careunit', 'intime', 'outtime'],
    
    # Lab results
    'labevents': ['subject_id', 'hadm_id', 'itemid', 'charttime', 'value', 'valuenum', 'valueuom', 'flag'],
    'd_labitems': ['itemid', 'label', 'fluid', 'category'],
    
    # Microbiology results
    'microbiologyevents': ['subject_id', 'hadm_id', 'micro_specimen_id', 'charttime', 'spec_type_desc',
                           'test_name', 'org_name', 'ab_name', 'interpretation'],
    
    # Medication information
    'emar': ['subject_id', 'hadm_id', 'emar_id', 'emar_seq', 'poe_id', 'pharmacy_id', 'charttime', 'medication'],
    'emar_detail': ['subject_id', 'emar_id', 'emar_seq', 'dose_given', 'dose_given_unit', 'product_description', 
                    'infusion_rate', 'infusion_rate_unit', 'route'],
    
    # Prescription information
    'prescriptions': ['subject_id', 'hadm_id', 'pharmacy_id', 'poe_id', 'starttime', 'stoptime', 'drug',
                      'formulary_drug_cd', 'dose_val_rx', 'dose_unit_rx', 'route'],
    
    # Diagnosis information
    'diagnoses_icd': ['subject_id', 'hadm_id', 'seq_num', 'icd_code', 'icd_version'],
    'd_icd_diagnoses': ['icd_code', 'icd_version', 'long_title'],
    
    # Procedure information
    'procedures_icd': ['subject_id', 'hadm_id', 'seq_num', 'chartdate', 'icd_code', 'icd_version'],
    'd_icd_procedures': ['icd_code', 'icd_version', 'long_title'],
    
    # Other important information
    'drgcodes': ['subject_id', 'hadm_id', 'drg_type', 'drg_code', 'description', 'drg_severity', 'drg_mortality'],
    'omr': ['subject_id', 'chartdate', 'seq_num', 'result_name', 'result_value'],

    # ICU module related tables
    # ICU admission information
    'icustays': ['subject_id', 'hadm_id', 'stay_id', 'first_careunit', 'last_careunit', 'intime', 'outtime', 'los'],
    
    # Vitals and monitoring data
    'chartevents': ['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'value', 'valuenum', 'valueuom'],
    'd_items': ['itemid', 'label', 'category', 'unitname', 'lownormalvalue', 'highnormalvalue'],
    
    # Medication and infusion information
    'inputevents': ['subject_id', 'hadm_id', 'stay_id', 'starttime', 'endtime', 'itemid', 
                    'amount', 'amountuom', 'rate', 'rateuom', 'patientweight'],
    'ingredientevents': ['subject_id', 'hadm_id', 'stay_id', 'starttime', 'endtime', 'itemid',
                         'amount', 'amountuom', 'rate', 'rateuom'],
    
    # Output information
    'outputevents': ['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'value', 'valueuom'],
    
    # Procedure and event information
    'procedureevents': ['subject_id', 'hadm_id', 'stay_id', 'starttime', 'endtime', 'itemid',
                        'value', 'valueuom', 'location', 'locationcategory']
}

def load_dictionary_tables(module_dir, useful_fields_map):
    """
    Loads and returns dictionary tables for efficient merging.
    """
    dict_tables = {}
    
    # Define different dictionary tables based on the module
    if 'hosp' in module_dir:
        dict_names = {
            'd_labitems': 'itemid',
            'd_icd_diagnoses': ['icd_code', 'icd_version'],
            'd_icd_procedures': ['icd_code', 'icd_version'],
            'd_hcpcs': 'hcpcs_cd'
        }
    else:  # ICU module
        dict_names = {
            'd_items': 'itemid'
        }

    print(f"Loading dictionary tables from {module_dir}...")
    for dict_name, _ in dict_names.items():
        file_path = os.path.join(module_dir, f'{dict_name}.csv.gz')
        if os.path.exists(file_path):
            try:
                df_dict = pd.read_csv(file_path, compression='gzip', low_memory=False)
                # Select only useful fields for dictionary tables
                if dict_name in useful_fields_map:
                    selected_cols = [col for col in useful_fields_map[dict_name] if col in df_dict.columns]
                    df_dict = df_dict[selected_cols]
                dict_tables[dict_name] = df_dict
                print(f"  - Loaded {dict_name}, {len(df_dict)} rows.")
            except Exception as e:
                print(f"  - Error loading {dict_name}: {e}")
        else:
            print(f"  - Warning: Dictionary table {dict_name}.csv.gz not found in {module_dir}, skipped.")
    return dict_tables

def extract_and_save_mimic_data(module_dir, output_dir, useful_fields_map):
    """
    Extracts useful information from MIMIC-IV tables, performs necessary joins,
    and saves the extracted data to the specified output directory.
    """
    print(f"\n--- Processing module: {module_dir} ---")
    print(f"Output directory: {output_dir}")

    # Load dictionary tables first
    dict_tables = load_dictionary_tables(module_dir, useful_fields_map)

    # Save dictionary tables to output_dir
    for dict_name, df_dict in dict_tables.items():
        output_file_path = os.path.join(output_dir, f'{dict_name}.csv')
        df_dict.to_csv(output_file_path, index=False)
        print(f"  - Dictionary table {dict_name} saved to {output_file_path}")
        # Clean up dictionary table data
        del df_dict
        gc.collect()

    # Process other tables
    all_gz_files = glob.glob(os.path.join(module_dir, '*.csv.gz'))
    
    # Sort files to ensure consistent processing order
    all_gz_files.sort() 

    # Print all found files
    print(f"\nFound the following files to process:")
    for file_path in all_gz_files:
        print(f"  - {os.path.basename(file_path)}")

    for file_path in all_gz_files:
        table_name = os.path.basename(file_path).replace('.csv.gz', '')
        
        # Check if the output file already exists
        output_file_path = os.path.join(output_dir, f'{table_name}.csv')
        if os.path.exists(output_file_path):
            file_size = os.path.getsize(output_file_path) / (1024 * 1024)  # Convert to MB
            print(f"\nSkipping table {table_name}: Output file already exists ({file_size:.2f} MB)")
            continue
        
        # Skip dictionary tables if already loaded and processed
        if table_name in dict_tables:
            print(f"Skipping {table_name} as it is an already processed dictionary table.")
            continue

        if table_name not in useful_fields_map:
            print(f"Skipping table {table_name}: No useful fields defined.")
            continue

        print(f"\nProcessing table: {table_name}")
        try:
            # Check file size
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
            print(f"  - File size: {file_size:.2f} MB")
            
            # Use chunking to process all files
            chunk_size = 100000  # Read 100,000 rows at a time
            print(f"  - Processing with chunking, chunk size: {chunk_size}")
            chunks = pd.read_csv(file_path, compression='gzip', 
                                 chunksize=chunk_size, low_memory=False)
            
            # Process the first chunk
            first_chunk = True
            chunk_count = 0
            total_rows = 0
            skipped_rows = 0
            
            for chunk in chunks:
                chunk_count += 1
                print(f"  - Processing chunk {chunk_count}, rows in current chunk: {len(chunk)}")
                
                # Select columns
                selected_cols = [col for col in useful_fields_map[table_name] if col in chunk.columns]
                if len(selected_cols) != len(useful_fields_map[table_name]):
                    missing_cols = set(useful_fields_map[table_name]) - set(selected_cols)
                    print(f"  - Warning: The following fields do not exist in the table: {missing_cols}")
                
                chunk = chunk[selected_cols].copy()
                
                # Check and skip rows with only identifiers
                if table_name == 'emar':
                    # For emar, check if all columns except emar_id are null
                    non_id_cols = [col for col in chunk.columns if col != 'emar_id']
                    valid_rows = chunk[non_id_cols].notna().any(axis=1)
                    skipped_rows += (~valid_rows).sum()
                    chunk = chunk[valid_rows]
                elif table_name == 'emar_detail':
                    # For emar_detail, check if all columns except emar_id are null
                    non_id_cols = [col for col in chunk.columns if col != 'emar_id']
                    valid_rows = chunk[non_id_cols].notna().any(axis=1)
                    skipped_rows += (~valid_rows).sum()
                    chunk = chunk[valid_rows]
                elif table_name == 'poe':
                    # For poe, check if all columns except poe_id are null
                    non_id_cols = [col for col in chunk.columns if col != 'poe_id']
                    valid_rows = chunk[non_id_cols].notna().any(axis=1)
                    skipped_rows += (~valid_rows).sum()
                    chunk = chunk[valid_rows]
                elif table_name == 'poe_detail':
                    # For poe_detail, check if all columns except poe_id are null
                    non_id_cols = [col for col in chunk.columns if col != 'poe_id']
                    valid_rows = chunk[non_id_cols].notna().any(axis=1)
                    skipped_rows += (~valid_rows).sum()
                    chunk = chunk[valid_rows]
                
                # If chunk is empty, skip saving
                if len(chunk) == 0:
                    print(f"  - Current chunk has no valid data, skipping save.")
                    continue
                
                # Perform merges for tables that need it
                if table_name == 'labevents' and 'd_labitems' in dict_tables:
                    chunk = chunk.merge(dict_tables['d_labitems'], on='itemid', how='left')
                elif table_name == 'chartevents' and 'd_items' in dict_tables:
                    chunk = chunk.merge(dict_tables['d_items'], on='itemid', how='left')
                elif table_name == 'diagnoses_icd' and 'd_icd_diagnoses' in dict_tables:
                    chunk = chunk.merge(dict_tables['d_icd_diagnoses'], on=['icd_code', 'icd_version'], how='left')
                elif table_name == 'procedures_icd' and 'd_icd_procedures' in dict_tables:
                    chunk = chunk.merge(dict_tables['d_icd_procedures'], on=['icd_code', 'icd_version'], how='left')
                elif table_name == 'hcpcsevents' and 'd_hcpcs' in dict_tables:
                    chunk = chunk.merge(dict_tables['d_hcpcs'], on='hcpcs_cd', how='left')
                
                # Save chunk
                if first_chunk:
                    chunk.to_csv(output_file_path, index=False, mode='w')
                    first_chunk = False
                else:
                    chunk.to_csv(output_file_path, index=False, mode='a', header=False)
                
                total_rows += len(chunk)
                print(f"  - Current chunk processing complete, valid rows: {len(chunk)}")
                
                # Clean up memory
                del chunk
                gc.collect()
            
            print(f"  - Chunk processing finished, total chunks processed: {chunk_count}")
            print(f"  - Total valid rows: {total_rows}")
            print(f"  - Skipped empty rows: {skipped_rows}")
            print(f"  - Data saved to {output_file_path}")

        except KeyError as ke:
            print(f"  - Error processing {table_name}: Missing expected column - {ke}. Skipped.")
        except Exception as e:
            print(f"  - An unexpected error occurred while processing {table_name}: {e}. Skipped.")
        
        # Force garbage collection after processing each file
        gc.collect()

    # Clean up dictionary tables
    del dict_tables
    gc.collect()

    print(f"\n--- Finished processing module: {module_dir} ---")
    print(f"All files in output directory: {output_dir}:")
    for file in os.listdir(output_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(output_dir, file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
            print(f"  - {file}: {file_size:.2f} MB")

# Execute extraction for Hosp module
extract_and_save_mimic_data(HOSP_DIR, EXTRACT_HOSP_DIR, USEFUL_FIELDS)

# Execute extraction for ICU module
extract_and_save_mimic_data(ICU_DIR, EXTRACT_ICU_DIR, USEFUL_FIELDS)

print("\n--- MIMIC-IV data extraction complete. ---")
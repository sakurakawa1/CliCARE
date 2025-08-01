# RawData_get

This section is used for the preprocessing and format conversion of raw clinical data. It contains two main submodules:

## csv2txt

-   **Function**: To organize raw clinical data from CSV format by patient and admission ID, converting it into structured TXT files for subsequent processing.
-   **Key Contents**:
    -   `run.sh`: A batch processing script that sequentially runs Python scripts for various data modules, outputs the results as TXT files, and then calls `local.py` to further organize the data by patient.
    -   `local.py`: Saves the TXT output from each module into separate files based on "patient-admission ID" and adds section labels (e.g., `====Step_xxx====`) within each file.
    -   `patient_inpatient.txt`: A mapping table of patient IDs to admission IDs.
    -   `*_CDR_*.py`: Each script is responsible for processing a specific type of raw CSV data (e.g., basic patient information, lab results, medical orders) and outputs it in a standardized text format.
-   **Processing Flow**:
    1.  Run `run.sh` to automatically process all modules in sequence.
    2.  Each module script reads the original CSV and outputs standardized text based on the admission ID to patient ID mapping.
    3.  `local.py` organizes the data into files by patient, facilitating subsequent merging and analysis.

## txt2json

-   **Function**: To further convert the TXT files generated by `csv2txt` into a structured JSON format, suitable for data analysis and machine learning.
-   **Key Contents**:
    -   `extract_useful_info.py`: Extracts useful fields from the CSV files.
    -   `extract_cancer_diagnoses.py`: Filters diagnosis information, retaining only cancer-related diagnoses.
    -   `filter_cancer_patients.py`: After filtering for cancer diagnoses, this script queries the information of patients identified with cancer.
    -   `clinic_json.py`: The main script that recursively reads the organized TXT files, parses various types of information based on section labels, and outputs them as standard JSON.
-   **Processing Flow**:
    1.  Run `clinic_json.py`, specifying the input folder path.
    2.  The script automatically parses each patient's TXT file and structures various types of information (e.g., admission, diagnosis, lab results) into JSON.
    3.  Supports features like field translation and time format processing.

---

## Recommended Usage

1.  **Prepare Raw Data**: Place all original CSV files and `patient_inpatient.txt` into the `csv2txt` directory.
2.  **Batch Conversion**: Run `csv2txt/run.sh` to generate TXT files organized by patient.
3.  **Structured Output**: Run `txt2json/clinic_json.py` to batch convert the TXT files to JSON for subsequent analysis.

---

For detailed script parameters or custom processing workflows, please refer to the comments within each script.
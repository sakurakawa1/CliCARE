# Cancer Data Processing


###  `clinic_json_cancer_filter.py`
Filters raw medical record JSON files using cancer keywords to retain only the admission records that contain cancer-related terms in their diagnosis information.

### `clinic_json_cancer_data_api_filter.py` 
Calls an AI model to further process the initially screened medical records, retaining only the information that is explicitly cancer-related.

###  `remove_duplicate_lines.py`
Addresses the issue of repetitive outputs (hallucinations) from the large language model. It deduplicates the processed medical record text by removing duplicate phrases and sentences to streamline the content.

### `extract_cancer_cases.py` 
Classifies cases from the processed medical data according to predefined cancer types and keywords. It gives priority to cases with more than 10 hospital admissions and selects representative cases for each cancer type.

###  `clinic_json_cancer_data_api_filter for case.py` 
Performs a final refinement on the selected representative cases to ensure that each case file contains only the most essential cancer-related diagnosis and treatment information.
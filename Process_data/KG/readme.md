# Knowledge Graph Processing

### `NCCN.py`
Processes NCCN (National Comprehensive Cancer Network) guideline documents. It calls the DeepSeek model to extract structured information from the text, populating a predefined JSON schema that includes cancer types, diagnostic recommendations, and staged treatment plans.

### `CSCO.py`
Processes CSCO (Chinese Society of Clinical Oncology) guideline documents. It also uses a large language model to extract and structure text content into a specific JSON format, which includes clinical recommendations, biomarker requirements, and Traditional Chinese Medicine (TCM) recommendations.

### `ESMO.py`
Processes ESMO (European Society for Medical Oncology) guideline documents. It utilizes a large language model to convert unstructured guideline text into structured JSON data according to ESMO standards, with a focus on information like staging, treatment plans, and biomarkers.

### `run_all_guideline.py`
An execution script that runs the other guideline processing scripts (NCCN.py, CSCO.py, ESMO.py) in parallel via multi-threading. It records the run results and duration for each script and reports any execution failures.

### `build_clinical_temporal_kg multi.py`
Uses a large language model to extract structured information (e.g., diagnoses, admission details, and clinical events) from clinical texts to build a clinical temporal knowledge graph in a Neo4j database. The script creates nodes and relationships for patients, admissions, and clinical events (like drugs and tests), establishing their temporal connections, and supports multi-threading for improved efficiency.

### `data_with_KG.py`
Similar in function to `data_with_KG_Align.py` but simplified. It connects to the Neo4j database to enrich JSON files by querying information only from the guideline knowledge graph (KG), without including data from alignments or the clinical temporal knowledge graph (TKG).

### `data_with_KG_Align.py`
Connects to a Neo4j graph database to enrich JSON files with information from the knowledge graph. It extracts cancer diagnoses from the JSON files, queries the KG for related examination and drug information (including aligned data from both the guideline KG and the clinical temporal TKG), and then adds this enriched information back to the JSON files.

### `entities_and_relations1.txt`
Defines the knowledge graph schema. It details the entity types (like Cancer, Patient, Treatment), their relationships (like HAS_TREATMENT, HAS_ADMISSION), and the specific properties of each entity within the graph database.

### `rename_json_files.py`
Standardizes filenames by renaming the generated JSON files.
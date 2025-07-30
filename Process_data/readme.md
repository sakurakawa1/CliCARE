# Data Processing

This section contains the data processing for the MIMIC-Cancer project. Each subdirectory handles a specific stage of the data processing workflow.

## Directory Structure

### `RawData_get`
Handles the initial data extraction and retrieval from the MIMIC database. This includes downloading raw medical records, patient demographics, and clinical notes.

### `Cancer_filter`
Filters and processes medical records to identify cancer-related cases. Uses keyword-based filtering and AI model refinement to retain only cancer-related admission records.

### `Longformer`
Processes long medical texts using the Clinical-Longformer model. Creates condensed summaries by extracting key information from medical records based on sentence importance scores.

### `KG`
Builds and manages knowledge graphs for clinical data. Includes processing of clinical guidelines (NCCN, CSCO, ESMO) and construction of temporal knowledge graphs in Neo4j database.

## Processing Workflow

1. **Raw Data Extraction** (`RawData_get/`) → Extract initial data from MIMIC
2. **Cancer Filtering** (`Cancer_filter/`) → Filter cancer-related cases
3. **Text Summarization** (`Longformer/`) → Summarize long medical texts
4. **Knowledge Graph Construction** (`KG/`) → Build clinical knowledge graphs


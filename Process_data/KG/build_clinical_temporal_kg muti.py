import os
import json
from neo4j import GraphDatabase
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import re
import requests
from openai import OpenAI
from difflib import SequenceMatcher
import threading
import concurrent.futures

class ClinicalTemporalKnowledgeGraphWithLLM:
    # Constant definitions
    MAX_RECORD_LENGTH = 64 * 1024  # 64KB, max length for a single admission record
    MIN_RECORD_LENGTH = 400   # Min length for a single admission record
    MIN_TOTAL_LENGTH = 500    # Min total length for all of a patient's records
    
    def __init__(self, uri: str, user: str, password: str, openai_api_key: str, openai_base_url: str = None, database: str = "newcancer"):
        """Initialize Neo4j connection"""
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        # Configure OpenAI client
        if openai_base_url:
            self.openai_client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
        else:
            self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Configure HTTP API
        self.http_uri = uri.replace("bolt://", "http://").replace(":7687", ":7474")
        self.auth = (user, password)

    def _execute_cypher_http(self, cypher: str, parameters: Dict = None) -> Dict:
        """Execute Cypher query using HTTP API"""
        # Neo4j 4.x version uses /db/data/transaction/commit
        url = f"{self.http_uri}/db/data/transaction/commit"
        
        # Add USE statement before the query to specify the database
        if not cypher.strip().upper().startswith('USE'):
            cypher = f"USE {self.database}\n{cypher}"
        
        payload = {
            "statements": [
                {
                    "statement": cypher,
                    "parameters": parameters or {}
                }
            ]
        }
        
        try:
            response = requests.post(url, json=payload, auth=self.auth, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"HTTP API call failed: {e}")
            return {"errors": [str(e)]}

    def close(self):
        """Close the database connection"""
        pass

    def test_database_connection(self):
        """Test the database connection"""
        print(f"Testing connection to database: {self.database}")
        try:
            # First, test the basic connection
            result = self._execute_cypher_http("RETURN 1 as test")
            if "errors" in result and result["errors"]:
                print(f"Basic connection failed: {result['errors']}")
                return False
            
            # Test connection to the specified database
            result = self._execute_cypher_http(f"USE {self.database}\nRETURN 1 as test")
            if "errors" in result and result["errors"]:
                print(f"Database {self.database} connection failed: {result['errors']}")
                return False
            
            print("Database connection successful")
            return True
        except Exception as e:
            print(f"Database connection test failed: {e}")
            return False

    def create_clinical_constraints_and_indexes(self):
        """Create constraints and indexes for clinical data related nodes"""
        print("Creating Neo4j constraints and indexes for clinical data...")
        try:
            # Execute constraint creation one by one
            print("Creating Patient constraint...")
            self._execute_cypher_http("CREATE CONSTRAINT IF NOT EXISTS ON (p:Patient) ASSERT p.id IS UNIQUE")
            
            print("Creating HospitalAdmission constraint...")
            self._execute_cypher_http("CREATE CONSTRAINT IF NOT EXISTS ON (ha:HospitalAdmission) ASSERT ha.id IS UNIQUE")
            
            print("Creating ClinicalEvent constraint...")
            self._execute_cypher_http("CREATE CONSTRAINT IF NOT EXISTS ON (ce:ClinicalEvent) ASSERT ce.id IS UNIQUE")
            
            print("Creating indexes...")
            self._execute_cypher_http("CREATE INDEX IF NOT EXISTS FOR (ha:HospitalAdmission) ON (ha.admission_time)")
            self._execute_cypher_http("CREATE INDEX IF NOT EXISTS FOR (ce:ClinicalEvent) ON (ce.name)")
            self._execute_cypher_http("CREATE INDEX IF NOT EXISTS FOR (d:Drug) ON (d.name)")
            self._execute_cypher_http("CREATE INDEX IF NOT EXISTS FOR (b:Biomarker) ON (b.name)")
            
            print("Clinical data constraints and indexes created successfully.")
        except Exception as e:
            print(f"Error creating clinical data constraints: {e}")
            # Try a simpler operation to test the connection
            try:
                result = self._execute_cypher_http("RETURN 1 as test")
                print("Connection test successful")
            except Exception as test_e:
                print(f"Connection test failed: {test_e}")

    def _call_llm_api(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Call the large language model using OpenAI API for information extraction.
        """
        try:
            print(f"Calling API, prompt length: {len(prompt)}")
            
            response = self.openai_client.chat.completions.create(
                model="deepseek-reasoner",  # or use another suitable model
                messages=[
                    {"role": "system", "content": "You are a professional medical information extraction assistant. Please extract key information from the given clinical records and return it in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                response_format={
                    'type': 'json_object'
                },
                temperature=0.1,  # Lower temperature for more stable output
                # max_tokens=3000,  # Increase token limit
                timeout=600  # Increase timeout
            )
            
            # Extract response content
            content = response.choices[0].message.content.strip()
            print(f"API response length: {len(content)}")
            print(f"First 100 characters of API response: {content[:100]}")
            
            # Try to parse JSON
            try:
                # If the response is in JSON format, parse it directly
                if content.startswith('{'):
                    result = json.loads(content)
                    print("JSON parsing successful")
                    return result
                else:
                    # If the response is not in JSON format, try to extract the JSON part
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    if json_start != -1 and json_end != 0:
                        json_str = content[json_start:json_end]
                        result = json.loads(json_str)
                        print("Successfully extracted JSON from response")
                        return result
                    else:
                        print(f"Could not extract JSON from response, full response: {content}")
                        return None
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}")
                print(f"Original response: {content}")
                return None
                
        except Exception as e:
            print(f"OpenAI API call failed: {e}")
            return None

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate the similarity between two strings.
        """
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

    def _find_most_similar_cancer(self, cancer_name: str, existing_cancers: List[str], threshold: float = 0.8) -> Optional[str]:
        """
        Find the most similar name among existing cancer names and return if similarity exceeds the threshold.
        """
        if not existing_cancers:
            return None
            
        best_match = None
        best_similarity = 0
        
        for existing_cancer in existing_cancers:
            similarity = self._calculate_similarity(cancer_name, existing_cancer)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = existing_cancer
        
        if best_similarity >= threshold:
            return best_match
        return None

    def _get_existing_cancer_names(self) -> List[str]:
        """
        Get all existing cancer names from Neo4j.
        """
        result = self._execute_cypher_http("MATCH (c:Cancer) RETURN c.name AS name")
        if "data" in result:
            # The result format for HTTP API is slightly different
            return [row['row'][0] for row in result['results'][0]['data']] if result['results'] and result['results'][0]['data'] else []
        return []

    def _normalize_cancer_name(self, cancer_name: str) -> str:
        """
        Normalize cancer names using similarity calculation.
        """
        # Basic cleaning
        cancer_name = cancer_name.strip()
        
        # Remove common modifiers and descriptive words in English
        cancer_name = re.sub(r'\b(post-operative|recurrent|metastatic|advanced|early stage|stage [IVX]+|personal history of|malignant neoplasm of|secondary malignant neoplasm of|neoplasm of uncertain behavior)\b', '', cancer_name, flags=re.IGNORECASE).strip()
        
        # Remove extra punctuation and spaces
        cancer_name = re.sub(r'[^\w\s]', ' ', cancer_name)
        cancer_name = re.sub(r'\s+', ' ', cancer_name).strip()
        
        # Get existing cancer names
        existing_cancers = self._get_existing_cancer_names()
        
        # Find the most similar cancer name
        similar_cancer = self._find_most_similar_cancer(cancer_name, existing_cancers, threshold=0.8)
        
        if similar_cancer:
            print(f"Normalizing cancer name: '{cancer_name}' -> '{similar_cancer}'")
            return similar_cancer
        
        # If no similar name is found, return the cleaned name
        return cancer_name

    def _merge_entity_with_base_kg(self, label: str, name: str, extra_fields: Dict = None):
        """Merge entity using HTTP API"""
        result = self._execute_cypher_http(f"MATCH (n:{label} {{name: $name}}) RETURN n", {"name": name})
        
        # Check if the query itself had errors
        if result.get("errors") and result["errors"]:
            print(f"Error checking entity existence for {label} '{name}': {result['errors']}")
            return

        # Check the data part of the response
        if result.get("results") and result['results'][0]['data']:
            # Node exists, update source
            self._execute_cypher_http(
                f"MATCH (n:{label} {{name: $name}}) SET n.source = CASE WHEN NOT 'ClinicalData' IN n.source THEN n.source + 'ClinicalData' ELSE n.source END",
                {"name": name}
            )
        else:
            # Node does not exist, create new node
            fields = ", ".join([f"n.{k} = ${k}" for k in (extra_fields or {})])
            set_clause = f"ON CREATE SET {fields}, n.source = ['ClinicalData']" if fields else "ON CREATE SET n.source = ['ClinicalData']"
            self._execute_cypher_http(
                f"MERGE (n:{label} {{name: $name}}) {set_clause}",
                {"name": name, **(extra_fields or {})}
            )

    def _parse_time_offset(self, admission_time_str: str, offset_str: str) -> datetime:
        """
        Parse the time offset from a medical order and calculate the absolute time.
        """
        admission_time = datetime.strptime(admission_time_str, "%Y-%m-%d %H:%M:%S")

        # Match English time units
        days_match = re.search(r'(\d+)\s*(days?|d)', offset_str, re.IGNORECASE)
        hours_match = re.search(r'(\d+)\s*(hours?|h)', offset_str, re.IGNORECASE)
        minutes_match = re.search(r'(\d+)\s*(minutes?|min)', offset_str, re.IGNORECASE)

        delta = timedelta(
            days=int(days_match.group(1)) if days_match else 0,
            hours=int(hours_match.group(1)) if hours_match else 0,
            minutes=int(minutes_match.group(1)) if minutes_match else 0
        )
        return admission_time + delta

    def _build_llm_prompt(self, record_text: str, patient_id: str, admission_idx: str) -> str:
        """
        Build the prompt to send to the large language model for information extraction.
        """
        # Simplify the prompt to make it more direct and clear
        prompt = f"""
Please extract key information from the following clinical record and return it in JSON format:

Patient ID: {patient_id}
Record Number: {admission_idx}

Clinical Record Text:
{record_text}

Please extract the following information and return it in JSON format:

{{
    "diagnosis": "Primary cancer diagnosis (extract from 'patient diagnosis information')",
    "admission_info": {{
        "admission_time": "2020-01-01 00:00:00",
        "department": "Oncology", 
        "clinical_diagnosis_summary": "Summary of cancer diagnosis and history"
    }},
    "events": [
        {{
            "type": "Medication/Procedure/Examination/Care/BiomarkerTest/OtherClinicalEvent",
            "name": "Specific event name"
        }}
    ]
}}

Notes:
1. For the input content, please only filter out cancer-related content
2. Return only JSON format, no other text
3. Use null values if information is unclear
4. Ensure correct JSON format
5. In events, only record event type and name, no need to record results
6. For diagnosis, only keep the major cancer category, e.g., "Breast Cancer" for "Malignant Breast Tumor" and "Right Breast Cancer"
7. All results should be in English
"""
        return prompt

    def safe_get(self, val, default=""):
        if isinstance(val, str) and val.strip():
            return val.strip()
        if val is None or val == "":
            return default
        return str(val)

    def process_patient_clinical_data(self, file_path: str):
        """
        Process a single patient's clinical data file to build a temporal knowledge graph.
        """
        print(f"Starting to process patient clinical data file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                clinical_raw_data = json.load(f)
        except Exception as e:
            print(f"Failed to read file {file_path}: {e}")
            return

        patient_id = os.path.basename(file_path).split('.')[0]

        try:
            # MERGE Patient node
            self._execute_cypher_http(
                "MERGE (p:Patient {id: $patient_id}) ON CREATE SET p.type = 'Patient', p.source = ['ClinicalData']",
                {"patient_id": patient_id}
            )

            text_content = clinical_raw_data.get('text', '')
            if not text_content:
                print(f"Warning: Text content for patient {patient_id} is empty")
                return
            
            # Split treatment record blocks - adapt to English data format
            split_blocks = re.split(r'The patient was recorded for the (\d+) time:\n', text_content)
            hospital_admission_blocks_parsed = []
            
            if len(split_blocks) > 1:
                for i in range(1, len(split_blocks), 2):
                    admission_idx = self.safe_get(split_blocks[i])
                    block_text = self.safe_get(split_blocks[i+1])
                    hospital_admission_blocks_parsed.append((admission_idx, block_text))
            
            # If no matching pattern is found, try splitting by paragraph
            if not hospital_admission_blocks_parsed:
                paragraphs = text_content.split('\n\n')
                for i, paragraph in enumerate(paragraphs):
                    if self.safe_get(paragraph):
                        hospital_admission_blocks_parsed.append((str(i+1), self.safe_get(paragraph)))

            # Filter record length
            filtered_blocks = []
            total_length = 0
            
            for admission_idx, block_text in hospital_admission_blocks_parsed:
                block_length = len(self.safe_get(block_text))
                total_length += block_length
                
                if block_length < self.MIN_RECORD_LENGTH:
                    print(f"Skipping admission record {admission_idx} for patient {patient_id}, length {block_length} is less than {self.MIN_RECORD_LENGTH} characters")
                    continue
                
                filtered_blocks.append((admission_idx, block_text))
            
            # Check the total length of all records
            if total_length < self.MIN_TOTAL_LENGTH:
                print(f"Skipping patient {patient_id}, total length of all admission records {total_length} is less than {self.MIN_TOTAL_LENGTH} characters")
                return
            
            print(f"Patient {patient_id} valid admission records: {len(filtered_blocks)}/{len(hospital_admission_blocks_parsed)}")
            
            previous_admission_node_id = None

            for admission_idx, admission_block_text in filtered_blocks:
                # Check and truncate oversized admission records (64K limit)
                original_length = len(admission_block_text)
                
                if original_length > self.MAX_RECORD_LENGTH:
                    print(f"Admission record {admission_idx} for patient {patient_id} has length {original_length}, exceeding {self.MAX_RECORD_LENGTH//1024}K. Truncating to {self.MAX_RECORD_LENGTH//1024}K")
                    admission_block_text = admission_block_text[:self.MAX_RECORD_LENGTH-1024]
                    print(f"Length after truncation: {len(admission_block_text)}")
                
                # Call LLM API to extract information
                prompt = self._build_llm_prompt(admission_block_text, patient_id, admission_idx)
                extracted_data = self._call_llm_api(prompt)

                if not extracted_data:
                    print(f"Skipping admission {admission_idx} for patient {patient_id} because the large model did not return valid data.")
                    continue

                # Data extracted from the large model
                llm_diagnosis = self.safe_get(extracted_data.get('diagnosis'), 'Unknown Disease')
                normalized_cancer_name = self._normalize_cancer_name(llm_diagnosis)
                
                admission_info = extracted_data.get('admission_info', {})
                admission_time_iso = self.safe_get(admission_info.get('admission_time'), '2020-01-01 00:00:00')
                department = self.safe_get(admission_info.get('department'), 'Oncology')
                clinical_diagnosis_summary = self.safe_get(admission_info.get('clinical_diagnosis_summary'), '')
                
                # Since there's no explicit time in the real data, use a default time or a relative time based on the record number
                if admission_time_iso == '2020-01-01 00:00:00':
                    try:
                        record_num = int(admission_idx)
                        relative_days = record_num - 1
                        base_time = datetime(2020, 1, 1)
                        calculated_time = base_time + timedelta(days=relative_days)
                        admission_time_iso = calculated_time.strftime('%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        pass

                # --- Process Cancer and Diagnosis ---
                self._merge_entity_with_base_kg("Cancer", normalized_cancer_name, {"type": "Cancer"})
                self._execute_cypher_http(
                    "MATCH (p:Patient {id: $patient_id}) MATCH (c:Cancer {name: $cancer_name}) MERGE (p)-[:HAS_CANCER]->(c)",
                    {"patient_id": patient_id, "cancer_name": normalized_cancer_name}
                )

                # --- Process HospitalAdmission Node ---
                admission_id_full = f"{patient_id}_Admission_{admission_idx}"
                self._execute_cypher_http(
                    """
                    MATCH (p:Patient {id: $patient_id})
                    MERGE (ha:HospitalAdmission {id: $admission_id})
                    ON CREATE SET 
                        ha.admission_time = $admission_time,
                        ha.department = $department,
                        ha.clinical_diagnosis = $clinical_diagnosis,
                        ha.type = 'HospitalAdmission',
                        ha.source = 'ClinicalData'
                    ON MATCH SET 
                        ha.admission_time = $admission_time,
                        ha.department = $department,
                        ha.clinical_diagnosis = $clinical_diagnosis
                    MERGE (p)-[:HAS_ADMISSION]->(ha)
                    """,
                    {
                        "patient_id": patient_id,
                        "admission_id": admission_id_full,
                        "admission_time": admission_time_iso,
                        "department": department,
                        "clinical_diagnosis": clinical_diagnosis_summary
                    }
                )

                # Establish temporal relationships between admission records
                if previous_admission_node_id:
                    self._execute_cypher_http(
                        """
                        MATCH (prev_ha:HospitalAdmission {id: $prev_id})
                        MATCH (curr_ha:HospitalAdmission {id: $curr_id})
                        MERGE (prev_ha)-[:PRECEDES_ADMISSION]->(curr_ha)
                        """,
                        {"prev_id": previous_admission_node_id, "curr_id": admission_id_full}
                    )
                previous_admission_node_id = admission_id_full

                # --- Process ClinicalEvents during admission ---
                for event_data in extracted_data.get('events', []):
                    event_type = self.safe_get(event_data.get('type'), 'OtherClinicalEvent')
                    item_name = self.safe_get(event_data.get('name'), 'Unknown Event')

                    event_id = f"{patient_id}_Event_{admission_idx}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{item_name.replace(' ', '')}"

                    self._execute_cypher_http(
                        """
                        MATCH (ha:HospitalAdmission {id: $admission_id})
                        MERGE (ce:ClinicalEvent {id: $event_id})
                        ON CREATE SET 
                            ce.type = $event_type,
                            ce.name = $event_name,
                            ce.source = 'ClinicalData'
                        ON MATCH SET
                            ce.type = $event_type,
                            ce.name = $event_name
                        MERGE (ha)-[:INCLUDES_EVENT]->(ce)
                        """,
                        {
                            "admission_id": admission_id_full,
                            "event_id": event_id,
                            "event_type": event_type,
                            "event_name": item_name
                        }
                    )

                    # Fuse with base KG entities
                    if event_type in ["Examination", "BiomarkerTest"]:
                        self._merge_entity_with_base_kg("Biomarker", item_name, {"type": "Biomarker"})
                        self._execute_cypher_http(
                            "MATCH (ce:ClinicalEvent {id: $event_id}) MATCH (b:Biomarker {name: $bio_name}) MERGE (ce)-[:MEASURES]->(b)",
                            {"event_id": event_id, "bio_name": item_name}
                        )
                    elif event_type == "Medication":
                        self._merge_entity_with_base_kg("Drug", item_name, {"type": "Drug"})
                        self._execute_cypher_http(
                            "MATCH (ce:ClinicalEvent {id: $event_id}) MATCH (d:Drug {name: $drug_name}) MERGE (ce)-[:IS_DRUG]->(d)",
                            {"event_id": event_id, "drug_name": item_name}
                        )
                    elif event_type == "Procedure": # Changed from Examination to Procedure
                        self._merge_entity_with_base_kg("Procedure", item_name, {"type": "Procedure"})
                        self._execute_cypher_http(
                            "MATCH (ce:ClinicalEvent {id: $event_id}) MATCH (e:Procedure {name: $proc_name}) MERGE (ce)-[:IS_PROCEDURE]->(e)",
                            {"event_id": event_id, "proc_name": item_name}
                        )

            print(f"Clinical data import for patient {patient_id} complete.")
            
        except Exception as e:
            print(f"Error processing data for patient {patient_id}: {e}")

    def build_clinical_temporal_kg(self, clinical_data_dirs: List[str]):
        """
        Main function to build the clinical temporal knowledge graph (multi-threaded, max 32 threads).
        """
        try:
            self.create_clinical_constraints_and_indexes()
            
            for clinical_data_dir in clinical_data_dirs:
                if not os.path.isdir(clinical_data_dir):
                    print(f"Error: Clinical data directory '{clinical_data_dir}' does not exist. Please check the path.")
                    continue
                    
                print(f"Processing directory: {clinical_data_dir}")
                json_files = [f for f in os.listdir(clinical_data_dir) if f.endswith(".json")]
                print(f"Found {len(json_files)} JSON files")
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
                    futures = []
                    for i, file in enumerate(json_files, 1):
                        file_path = os.path.join(clinical_data_dir, file)
                        futures.append(executor.submit(self.process_patient_clinical_data, file_path))
                    # Wait for all tasks to complete
                    concurrent.futures.wait(futures)
                print(f"All JSON files in directory {clinical_data_dir} have been processed.")
        except Exception as e:
            print(f"Error building the knowledge graph: {e}")


if __name__ == "__main__":
    # Neo4j connection configuration
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "your_password"
    OPENAI_API_KEY = "sk-XXX"  # Please replace with your OpenAI API key
    OPENAI_BASE_URL = "https://api.deepseek.com/v1"  # Optional: custom API endpoint

    # Only get all json files under the extracted_cancer_cases directory
    base_dir = "./extracted_cancer_cases"
    CLINICAL_DATA_DIRS = [base_dir] if os.path.exists(base_dir) else []
    print(f"Will process directories: {CLINICAL_DATA_DIRS}")

    clinical_kg_builder = ClinicalTemporalKnowledgeGraphWithLLM(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENAI_API_KEY, OPENAI_BASE_URL, "newcancer")


    print("Starting to build the knowledge graph...")
    print("Starting to build the clinical temporal knowledge graph (using HTTP API)...")
    
    # Test database connection
    if not clinical_kg_builder.test_database_connection():
        print("Database connection failed, please check your configuration")
        exit(1)
    
    clinical_kg_builder.build_clinical_temporal_kg(CLINICAL_DATA_DIRS)
    print("\nClinical temporal knowledge graph construction complete!")
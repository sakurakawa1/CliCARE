from openai import OpenAI
import os
import json
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"./log/NCCN/deepseek_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# DeepSeek API Configuration
# url = 'https://ark.cn-beijing.volces.com/api/v3'
# api_key = 'your_volces_api_key'
url = 'https://api.deepseek.com/v1'
api_key = 'sk-d46680ddd44d4023a62affcc8d8252a0'

client = OpenAI(
    base_url=url,
    api_key=api_key
)

# Define the target JSON structure
TARGET_SCHEMA = {
    "guideline_id": "",
    "cancer_info": {
        "cancer_name": ""
    },
    "diagnosis_recommendations": {
        "examinations": []  # Merged examination names and details
    },
    "staging_treatment_plans": [
        {
            "staging_criteria": "",
            "risk_group": "",
            "treatment_plans": [
                {
                    "treatment_line": "",
                    "patient_subgroup": "",
                    "plan_name": "",
                    "plan_details": "",
                    "nccn_evidence_category": "",
                    "nccn_recommendation_category": ""
                }
            ]
        }
    ],
    "biomarker_clinical_significance": {
        "clinical_significance": ""
    }
}

# Define QA template
QA_TEMPLATE = """Please extract the following information from the text and generate it in JSON format:
{
  "cancer_info": {
    "cancer_name": "Breast Cancer"
  },
  "diagnosis_recommendations": {
    "examinations": [
      "Chest CT plain scan + enhancement",
      "Fine needle aspiration biopsy",
      "Breast MRI examination"
    ]
  },
  "staging_treatment_plans": [
    {
      "staging_criteria": "Early stage",
      "risk_group": "Low risk",
      "treatment_plans": [
        {
          "treatment_line": "First-line treatment",
          "patient_subgroup": "Young patients",
          "plan_name": "AC-T regimen",
          "plan_details": "Doxorubicin 60mg/m2",
          "nccn_evidence_category": "Category 1",
          "nccn_recommendation_category": "Preferred regimen"
        }
      ]
    }
  ],
  "biomarker_clinical_significance": {
    "clinical_significance": "Predicts efficacy of targeted therapy"
  }
}

Please ensure:
1. All fields conform to the above JSON structure
2. The examinations in diagnosis_recommendations must list in detail all examinations mentioned in the document, including imaging, pathology, etc.
3. The staging_treatment_plans field must list in detail all mentioned fields in the document
4. biomarker_clinical_significance must include all mentioned clinical significances
5. Data must be accurate and reflect the actual content of the medical guideline
6. Please answer in English
7. Please ensure the extracted information is complete and does not omit any important clinical recommendations
8. For each treatment plan, list its applicable conditions, specific content, and evidence level in detail
9. If a field contains multiple values, please list all of them, such as diagnosis items, staging treatment plans, treatment plans, clinical significance, etc.
"""

def chunk_text(text: str, chunk_size: int = 90000) -> List[str]:
    """Split text into chunks of a specified size."""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

def summarize_text(text: str) -> str:
    """Summarize text using the DeepSeek API."""
    try:
        logging.info(f"Starting to summarize text, length: {len(text)} characters")
        start_time = time.time()
        
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": "You are a professional medical literature analysis assistant, skilled at summarizing medical guideline documents. Please analyze according to the following steps:\n1. First, identify the primary cancer type and key information in the document\n2. Extract all diagnostic recommendations and examination items\n3. Summarize staging criteria and risk group classifications\n4. Analyze treatment plans, including treatment lines, patient subgroups, and plan details\n5. Extract NCCN evidence categories and recommendation categories\n6. Summarize the clinical significance of biomarkers\nPlease ensure the summary is comprehensive and accurate, retaining all important medical information."},
                {"role": "user", "content": f"This is a part of an NCCN cancer practice guideline. Please follow the steps above to summarize the information in detail, retaining all important medical information, including disease name related information, examination items, and key content on diagnosis, treatment, staging, etc. Please answer in English.\n\n{text}"}
            ],
            temperature=0.2,
            stream=False
        )
        
        end_time = time.time()
        logging.info(f"Text summarization complete, time taken: {end_time - start_time:.2f} seconds")
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"An error occurred while summarizing text: {e}")
        return text

def process_chunk(chunk: str, is_final: bool = False) -> Dict[str, Any]:
    """Process a text chunk and extract information."""
    results = {}
    
    if not is_final:
        # If it's not the final chunk, only summarize
        logging.info(f"Processing non-final chunk, size: {len(chunk)} characters")
        summary = summarize_text(chunk)
        results["summary"] = summary
    else:
        # If it's the final chunk, perform full QA extraction
        logging.info(f"Processing final chunk, size: {len(chunk)} characters")
        try:
            logging.info("Starting information extraction")
            start_time = time.time()
            
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": "You are a professional medical literature analysis assistant, skilled at extracting structured information from medical guidelines. Please analyze according to the following steps:\n1. First, identify the guideline document ID and primary cancer type\n2. Extract all diagnostic recommendations and examination items\n3. Analyze staging criteria and risk group classifications\n4. Extract treatment plans, including treatment lines, patient subgroups, and plan details\n5. Summarize NCCN evidence categories and recommendation categories\n6. Analyze the clinical significance of biomarkers\nPlease ensure the extracted information is complete and accurate."},
                    {"role": "user", "content": f"{QA_TEMPLATE}\n\nText content:\n{chunk}"}
                ],
                temperature=0.2,
                stream=False
            )
            
            # Parse the response and extract results
            content = response.choices[0].message.content
            logging.info(f"Original content returned by API: {content}")
            
            try:
                # Try to parse JSON directly
                results = json.loads(content)
                logging.info("Successfully parsed JSON directly")
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract the JSON part
                logging.info("Direct JSON parsing failed, attempting to extract JSON part")
                import re
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    try:
                        results = json.loads(json_match.group())
                        logging.info("Successfully extracted and parsed JSON part")
                    except json.JSONDecodeError:
                        logging.error("Could not parse the extracted JSON content")
                        results = TARGET_SCHEMA.copy()
                else:
                    logging.error("JSON content not found")
                    results = TARGET_SCHEMA.copy()
            
            # Validate the result structure
            if not isinstance(results, dict):
                logging.error("Result is not in dictionary format")
                results = TARGET_SCHEMA.copy()
            
            # Ensure all required fields exist and are not empty
            for key in TARGET_SCHEMA:
                if key not in results or not results[key]:
                    logging.warning(f"Field {key} does not exist or is empty, using default value")
                    results[key] = TARGET_SCHEMA[key]
            
            end_time = time.time()
            logging.info(f"Information extraction complete, time taken: {end_time - start_time:.2f} seconds")
            logging.info(f"Extracted JSON result: {json.dumps(results, ensure_ascii=False, indent=2)}")
            
        except Exception as e:
            logging.error(f"An error occurred during processing: {e}")
            results = TARGET_SCHEMA.copy()
    
    return results

def process_document(file_path: str) -> Dict[str, Any]:
    """Process a single document."""
    try:
        logging.info(f"Starting to process document: {file_path}")
        start_time = time.time()
        
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        logging.info(f"File reading complete, total size: {len(text)} characters")
        
        # Chunking
        chunks = chunk_text(text)
        logging.info(f"Document has been split into {len(chunks)} chunks")
        
        # Process each chunk and collect summaries
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            logging.info(f"Processing chunk {i+1}/{len(chunks)}")
            summary = summarize_text(chunk)
            chunk_summaries.append(summary)
            logging.info(f"Chunk {i+1} summarization complete, length: {len(summary)} characters")
        
        # Combine all summaries
        combined_summary = "\n".join(chunk_summaries)
        logging.info(f"All chunk summaries combined, total length: {len(combined_summary)} characters")
        
        # Perform final processing on the combined summary
        logging.info("Starting final information extraction")
        try:
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": "You are a professional medical literature analysis assistant, skilled at extracting structured information from medical guidelines. Please analyze according to the following steps:\n1. First, identify the guideline document ID and primary cancer type\n2. Extract all diagnostic recommendations and examination items\n3. Analyze staging criteria and risk group classifications\n4. Extract treatment plans, including treatment lines, patient subgroups, and plan details\n5. Summarize NCCN evidence categories and recommendation categories\n6. Analyze the clinical significance of biomarkers\nPlease ensure the extracted information is complete and accurate, especially for cases with multiple stages and treatment plans."},
                    {"role": "user", "content": f"{QA_TEMPLATE}\n\nText content:\n{combined_summary}"}
                ],
                temperature=0.2,
                stream=False
            )
            
            # Parse the response and extract results
            content = response.choices[0].message.content
            try:
                results = json.loads(content)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    try:
                        results = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        logging.error("Could not parse JSON content")
                        results = TARGET_SCHEMA.copy()
                else:
                    logging.error("JSON content not found")
                    results = TARGET_SCHEMA.copy()
            
            # Validate result structure
            if not isinstance(results, dict):
                logging.error("Result is not in dictionary format")
                results = TARGET_SCHEMA.copy()
            
            # Ensure all required fields exist
            for key in TARGET_SCHEMA:
                if key not in results:
                    results[key] = TARGET_SCHEMA[key]
            
            # Set the document ID
            results["guideline_id"] = os.path.basename(file_path)
            
            end_time = time.time()
            logging.info(f"Document processing complete, total time taken: {end_time - start_time:.2f} seconds")
            return results
            
        except Exception as e:
            logging.error(f"An error occurred during final processing: {e}")
            return TARGET_SCHEMA.copy()
    
    except Exception as e:
        logging.error(f"An error occurred while processing document {file_path}: {e}")
        return TARGET_SCHEMA.copy()

def process_directory(input_dir: str, output_dir: str):
    """Process all documents in a directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")
    
    total_files = len([f for f in os.listdir(input_dir) if f.endswith('.txt')])
    processed_files = 0
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            processed_files += 1
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
            
            logging.info(f"Starting to process file {processed_files}/{total_files}: {filename}")
            start_time = time.time()
            
            try:
                # Process document
                results = process_document(input_path)
                
                # Validate results
                if not isinstance(results, dict):
                    logging.error(f"Processing result is not a valid dictionary format: {type(results)}")
                    continue
                
                # Ensure the result conforms to the target structure
                for key in TARGET_SCHEMA:
                    if key not in results:
                        logging.warning(f"Field {key} is missing in the result, using default value")
                        results[key] = TARGET_SCHEMA[key]
                
                # Save the results
                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=4)
                    logging.info(f"Results successfully saved to: {output_path}")
                    
                    # Validate the saved file
                    if os.path.exists(output_path):
                        file_size = os.path.getsize(output_path)
                        logging.info(f"Saved file size: {file_size} bytes")
                        if file_size == 0:
                            logging.error("Saved file is empty")
                    else:
                        logging.error("File save failed, file does not exist")
                        
                except Exception as e:
                    logging.error(f"An error occurred while saving results: {e}")
                    # Try saving to a backup path
                    backup_path = os.path.join(output_dir, f"backup_{os.path.splitext(filename)[0]}.json")
                    try:
                        with open(backup_path, 'w', encoding='utf-8') as f:
                            json.dump(results, f, ensure_ascii=False, indent=4)
                        logging.info(f"Results have been saved to backup path: {backup_path}")
                    except Exception as e2:
                        logging.error(f"An error also occurred when saving to the backup path: {e2}")
            
            except Exception as e:
                logging.error(f"An error occurred while processing file {filename}: {e}")
            
            end_time = time.time()
            logging.info(f"File {filename} processing complete, time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    INPUT_DIR = "./guideline/NCCNOCR"
    # INPUT_DIR = "./clinical_test/NCCN_test"
    OUTPUT_DIR = "./process_guideline/NCCN"
    
    # Ensure the output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logging.info(f"Created output directory: {OUTPUT_DIR}")
    
    logging.info("Starting to process NCCN guideline directory")
    process_directory(INPUT_DIR, OUTPUT_DIR)
    logging.info("All files have been processed")
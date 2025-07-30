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
        logging.FileHandler(f"./log/ESMO/esmo_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# DeepSeek API Configuration
# url = 'https://ark.cn-beijing.volces.com/api/v3'
# api_key = 'your_volces_api_key'
url = 'https://api.deepseek.com/v1'
api_key = 'sk-xxx'

client = OpenAI(
    base_url=url,
    api_key=api_key
)

# Define the target JSON structure
TARGET_SCHEMA = {
    "guideline_identifier": "",  # e.g., "ESMO_BreastCancer_2024_DiagnosisTreatment"
    "cancer_focus": {
        "primary_cancer": "",  # e.g., "Breast Cancer", "Non-Small Cell Lung Cancer"
        "related_syndrome_or_condition": ""  # e.g., "Hereditary Breast and Ovarian Cancer Syndrome" (if applicable)
    },
    "staging_treatment_plans": [
        {
            "staging_criteria": "",  # Describes staging criteria
            "risk_group": "",  # Risk group classification
            "treatment_plans": [
                {
                    "clinical_context": "",  # Describes the clinical context/patient subgroup for which the recommendation applies
                    "recommendation_type": "",  # e.g., "Diagnosis", "Staging", "Treatment", "Screening", "Follow-up"
                    "recommendation_content": "",  # Specific content/measures/plan of the recommendation
                    "treatment_line": "",  # Applicable only when the recommendation type is treatment
                    "biomarker_requirements": [
                        {
                            "name": "",  # e.g., "ER", "PD-L1", "BRCA1"
                            "status": "",  # e.g., "Positive", "≥50%", "Mutation"
                            "testing_guidance": ""  # Brief guidance related to testing
                        }
                    ],
                    "esmo_evidence_level": ""  # ESMO evidence level / recommendation strength
                }
            ]
        }
    ],
    "biomarker_clinical_significance": {
        "clinical_significance": ""  # e.g., "Predicts efficacy of targeted therapy X", "Early prognostic indicator", "Diagnoses subtype Y"
    }
}

# Define QA template
QA_TEMPLATE = """Please extract the following information from the text and generate it in JSON format:
{
  "guideline_identifier": "ESMO_BreastCancer_2024_DiagnosisTreatment",
  "cancer_focus": {
    "primary_cancer": "Breast Cancer",
    "related_syndrome_or_condition": "Hereditary Breast and Ovarian Cancer Syndrome"
  },
  "staging_treatment_plans": [
    {
      "staging_criteria": "Early stage",
      "risk_group": "Low risk",
      "treatment_plans": [
        {
          "clinical_context": "Early-stage ER-positive breast cancer",
          "recommendation_type": "Treatment",
          "recommendation_content": "Specific treatment plan details",
          "treatment_line": "First-line",
          "biomarker_requirements": [
            {
              "name": "ER",
              "status": "Positive",
              "testing_guidance": "Testing guidance"
            }
          ],
          "esmo_evidence_level": "Level I Recommendation"
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
2. The staging_treatment_plans field must list in detail all stages, risk groups, and treatment plans mentioned in the document, including:
   - Criteria for each stage
   - Classification of each risk group
   - Specific content of each treatment plan
   - All related biomarker requirements
   - All ESMO evidence levels
3. Data must be accurate and reflect the actual content of the medical guideline
4. If a field has no information, keep the field name but leave the value empty
5. Please answer in English
6. Please ensure the extracted information is complete and does not omit any important clinical recommendations
7. For each treatment plan, list its applicable conditions, specific content, and evidence level in detail
8. Biomarker requirements must include complete testing information, including name, status, and testing guidance
9. Fields such as staging_treatment_plans, treatment_plans, and biomarker_clinical_significance may contain multiple entries; please list all values if there are more than one
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
                {"role": "system", "content": "You are a professional medical literature analysis assistant, skilled at summarizing medical guideline documents. Please analyze according to the following steps:\n1. First, identify the primary cancer type and related syndromes in the document\n2. Extract all staging criteria and risk group classifications\n3. Summarize treatment plans, including plans for different clinical contexts\n4. Analyze biomarker requirements and testing guidance\n5. Extract ESMO evidence levels\nPlease ensure the summary is comprehensive and accurate, retaining all important medical information."},
                {"role": "user", "content": f"This is a part of an ESMO cancer practice guideline. Please follow the steps above to summarize the information in detail, retaining all important medical information, including key content on diagnosis, treatment, staging, etc. Please answer in English.\n\n{text}"}
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
                    {"role": "system", "content": "You are a professional medical literature analysis assistant, skilled at extracting structured information from medical guidelines. Please analyze according to the following steps:\n1. First, identify the guideline identifier and primary cancer type\n2. Extract all staging criteria and risk group classifications\n3. Analyze treatment plans, including clinical context, recommendation type, and content\n4. Extract biomarker requirements, including name, status, and testing guidance\n5. Summarize ESMO evidence levels\n6. Analyze the clinical significance of biomarkers\nPlease ensure the extracted information is complete and accurate, especially for cases with multiple stages and treatment plans."},
                    {"role": "user", "content": f"{QA_TEMPLATE}\n\n{combined_summary}"}
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
            results["guideline_identifier"] = os.path.basename(file_path)
            
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
            
            results = process_document(input_path)
            
            # Save the results
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
                logging.info(f"Results have been saved to: {output_path}")
            except Exception as e:
                logging.error(f"An error occurred while saving results: {e}")
            
            end_time = time.time()
            logging.info(f"File {filename} processing complete, time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    INPUT_DIR = "./guideline/ESMOOCR"
    # INPUT_DIR = "./clinical_test/ESMO_test"
    OUTPUT_DIR = "./process_guideline/ESMO"
    logging.info("Starting to process ESMO guideline directory")
    process_directory(INPUT_DIR, OUTPUT_DIR)
    logging.info("All files have been processed")
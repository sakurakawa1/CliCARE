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
        logging.FileHandler(f"./log/CSCO/csco_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# DeepSeek API Configuration
# url = 'https://ark.cn-beijing.volces.com/api/v3'
# api_key = 'your_volces_api_key'
url = 'https://api.deepseek.com/v1'
api_key = 'sk-XXX' # Replace with your actual key
client = OpenAI(
    base_url=url,
    api_key=api_key
)

# Define the target JSON structure
TARGET_SCHEMA = {
    "guideline_id": "",  # e.g., "CSCO_LungCancer_2024"
    "cancer_info": {
        "cancer_name": ""  # Primary cancer type
    },
    "clinical_recommendations": [
        {
            "clinical_context": "",  # Clinical context / patient subgroup
            "recommendation_type": "",  # Recommendation type (diagnosis, staging, treatment, etc.)
            "recommendation_content": "",  # Specific recommendation content
            "treatment_line": "",  # Treatment line (if applicable)
            "biomarker_requirements": [
                {
                    "biomarker_name": "",  # Biomarker name
                    "status": "",  # Status (mutation, positive, etc.)
                    "testing_guidance": ""  # Testing guidance
                }
            ],
            "recommendation_level": "",  # CSCO recommendation level
            "evidence_level": ""  # CSCO evidence level
        }
    ],
    "biomarker_clinical_significance": {
        "clinical_significance": ""  # Clinical significance
    },
    "tcm_recommendations": [  # TCM (Traditional Chinese Medicine) treatment recommendations
        {
            "syndrome_type": "",  # TCM syndrome type
            "tcm_treatment_plan": "",  # Specific TCM plan
            "recommendation_grade": ""  # Recommendation grade
        }
    ]
}

# Merged QA Template
QA_TEMPLATE = """Please extract the following information from the text and generate it in JSON format:
{
  "guideline_id": "CSCO_LungCancer_2024",
  "cancer_info": {
    "cancer_name": "Lung Cancer"
  },
  "clinical_recommendations": [
    {
      "clinical_context": "Early-stage non-small cell lung cancer",
      "recommendation_type": "Treatment Plan",
      "recommendation_content": "Specific treatment plan details",
      "treatment_line": "First-line",
      "biomarker_requirements": [
        {
          "biomarker_name": "EGFR",
          "status": "Negative",
          "testing_guidance": "Testing guidance"
        }
      ],
      "recommendation_level": "Level I Recommendation",
      "evidence_level": "Evidence 1A"
    }
  ],
  "biomarker_clinical_significance": {
    "clinical_significance": "Predicts efficacy of targeted therapy"
  },
  "tcm_recommendations": [
    {
      "syndrome_type": "Qi and Yin Deficiency",
      "tcm_treatment_plan": "Specific TCM plan",
      "recommendation_grade": "Level I Recommendation"
    }
  ]
}

Please ensure:
1. All fields conform to the above JSON structure
2. Fields such as clinical_recommendations, biomarker_requirements, tcm_recommendations, and biomarker_clinical_significance may contain multiple entries; please list all values if there are more than one, especially for clinical recommendations
3. Data must be accurate and reflect the actual content of the medical guideline
4. If a field has no information, keep the field name but leave the value empty
5. Please answer in English
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
                {"role": "system", "content": "You are a professional medical literature analysis assistant, skilled at summarizing medical guideline documents. Please analyze according to the following steps:\n1. First, identify the primary cancer type and key information in the document\n2. Extract all diagnosis-related recommendations and standards\n3. Summarize treatment plans, including plans for different stages and risk groups\n4. Extract biomarker-related information\n5. Summarize Traditional Chinese Medicine (TCM) treatment recommendations (if any)\nPlease ensure the summary is comprehensive and accurate, retaining all important medical information."},
                {"role": "user", "content": f"This is a part of a CSCO cancer practice guideline. Please follow the steps above to summarize the information in detail, retaining all important medical information, including key content on diagnosis, treatment, staging, etc. Please answer in English.\n\n{text}"}
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
                    {"role": "system", "content": "You are a professional medical literature analysis assistant, skilled at extracting structured information from medical guidelines. Please analyze according to the following steps:\n1. First, identify the document ID and primary cancer type\n2. Extract all clinical recommendations, including context, type, content, etc.\n3. Analyze biomarker requirements, including name, status, and testing guidance\n4. Extract recommendation levels and evidence levels\n5. Summarize the clinical significance of biomarkers\n6. Extract Traditional Chinese Medicine (TCM) treatment recommendations (if any)\nPlease ensure the extracted information is complete and accurate."},
                    {"role": "user", "content": f"{QA_TEMPLATE}\n\n{chunk}"}
                ],
                temperature=0.2,
                stream=False
            )
            
            # Parse the response and extract results
            content = response.choices[0].message.content
            try:
                # Try to parse JSON directly
                results = json.loads(content)
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract the JSON part
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
            
            # Validate the result structure
            if not isinstance(results, dict):
                logging.error("Result is not in dictionary format")
                results = TARGET_SCHEMA.copy()
            
            # Ensure all required fields are present
            for key in TARGET_SCHEMA:
                if key not in results:
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
                    {"role": "system", "content": "You are a professional medical literature analysis assistant, skilled at extracting structured information from medical guidelines. Please analyze according to the following steps:\n1. First, identify the document ID and primary cancer type\n2. Extract all clinical recommendations, including context, type, content, etc.\n3. Analyze biomarker requirements, including name, status, and testing guidance\n4. Extract recommendation levels and evidence levels\n5. Summarize the clinical significance of biomarkers\n6. Extract Traditional Chinese Medicine (TCM) treatment recommendations (if any)\nPlease ensure the extracted information is complete and accurate, especially for cases with multiple stages and treatment plans."},
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
    INPUT_DIR = "./guideline/CSCOOCR"
    # INPUT_DIR = "./clinical_test/CSCO_test"
    
    OUTPUT_DIR = "./process_guideline/CSCO"
    logging.info("Starting to process CSCO guideline directory")
    process_directory(INPUT_DIR, OUTPUT_DIR)
    logging.info("All files have been processed")
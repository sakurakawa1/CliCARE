#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cancer Case Extraction Script
Extracts different types of cancer cases using vector similarity calculation, aiming to extract a specified number of cases per type.
"""

import os
import json
import re
import logging
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import psutil  # Add memory monitoring
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CancerCaseExtractor:
    def __init__(self, data_dir: str, output_dir: str = "extracted_cancer_cases", use_clustering: bool = True):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.cases_per_type = 10
        self.use_clustering = use_clustering  # Whether to use clustering

        # Cancer type keyword mapping (English)
        self.cancer_keywords = {
            'breast': ['breast', 'mammary', 'mastectomy', 'mammogram'],
            'lung': ['lung', 'pulmonary', 'bronchial', 'bronchogenic'],
            'gastric': ['gastric', 'stomach', 'gastrectomy'],
            'liver': ['liver', 'hepatic', 'hepatocellular'],
            'colorectal': ['colorectal', 'colon', 'rectal', 'colonic', 'rectum'],
            'esophageal': ['esophageal', 'esophagus', 'esophagectomy'],
            'cervical': ['cervical', 'cervix', 'cervical cancer'],
            'ovarian': ['ovarian', 'ovary', 'oophorectomy'],
            'prostate': ['prostate', 'prostatic', 'prostatectomy'],
            'bladder': ['bladder', 'cystectomy', 'urothelial'],
            'kidney': ['kidney', 'renal', 'nephrectomy'],
            'thyroid': ['thyroid', 'thyroidectomy'],
            'lymphoma': ['lymphoma', 'lymphatic', 'hodgkin', 'non-hodgkin'],
            'leukemia': ['leukemia', 'leukemic', 'myeloid', 'lymphocytic'],
            'brain': ['brain', 'cerebral', 'intracranial', 'glioblastoma'],
            'bone': ['bone', 'osteosarcoma', 'sarcoma'],
            'skin': ['skin', 'melanoma', 'cutaneous', 'basal cell', 'squamous cell'],
            'pancreatic': ['pancreatic', 'pancreas', 'pancreatectomy'],
            'gallbladder': ['gallbladder', 'biliary', 'cholangiocarcinoma'],
            'oral': ['oral', 'tongue', 'mouth', 'oropharyngeal']
        }

        # Exclusion words (non-cancer related)
        self.exclude_keywords = [
            'benign', 'cyst', 'polyp', 'inflammation', 'infection', 'tuberculosis', 'pneumonia', 'gastritis', 'hepatitis',
            'nephritis', 'thyroiditis', 'mastitis', 'prostatitis', 'cystitis', 'cervicitis',
            'hyperplasia', 'hypertrophy', 'nodule', 'adenoma', 'fibroma', 'lipoma', 'hemangioma',
            'teratoma', 'hamartoma', 'leiomyoma', 'neurofibroma', 'osteochondroma',
            'calcification', 'stone', 'thrombus', 'embolism', 'infarction', 'ischemia', 'hypoxia',
            'edema', 'effusion', 'abscess', 'fistula', 'sinus', 'ulcer', 'erosion',
            'atrophy', 'sclerosis', 'fibrosis', 'ossification', 'necrosis', 'gangrene',
            'allergy', 'hypersensitivity', 'autoimmune', 'rheumatoid', 'gout',
            'diabetes', 'hypertension', 'heart disease', 'cerebrovascular', 'kidney disease', 'liver disease',
            'anemia', 'thrombocytopenia', 'leukopenia', 'coagulation disorder',
            'malnutrition', 'vitamin deficiency', 'electrolyte disorder', 'acid-base imbalance'
        ]

    def load_json_files(self) -> List[Dict]:
        """Load all JSON files."""
        json_files = []
        
        if not os.path.exists(self.data_dir):
            logging.error(f"Data directory does not exist: {self.data_dir}")
            return json_files
            
        try:
            for filename in os.listdir(self.data_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.data_dir, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            # Add file path information
                            data['_file_path'] = file_path
                            data['_filename'] = filename
                            json_files.append(data)
                    except json.JSONDecodeError as e:
                        logging.warning(f"JSON parsing error for {filename}: {e}")
                    except Exception as e:
                        logging.warning(f"Error reading file {filename}: {e}")
                        
            logging.info(f"Successfully loaded {len(json_files)} JSON files")
            return json_files
            
        except Exception as e:
            logging.error(f"Error reading directory: {e}")
            return json_files

    def extract_diagnosis_text(self, data: Dict) -> Optional[str]:
        """Extract diagnosis text from JSON data."""
        text = data.get('text', '')
        if not text:
            return None
            
        # Find diagnosis-related text (English patterns)
        diagnosis_patterns = [
            r'patient diagnosis information:\s*diagnosis\s*([^\n]+)',
            r'diagnosis:\s*([^\n]+)',
            r'primary diagnosis:\s*([^\n]+)',
            r'discharge diagnosis:\s*([^\n]+)',
            r'admission diagnosis:\s*([^\n]+)',
            r'pathological diagnosis:\s*([^\n]+)',
            r'final diagnosis:\s*([^\n]+)'
        ]
        
        for pattern in diagnosis_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no explicit diagnosis field is found, return the entire text
        return text

    def preprocess_text(self, text: str) -> str:
        """Preprocess text."""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters, keep English letters, numbers, and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Tokenize (split by space for English)
        words = text.split()
        
        # Filter stopwords and short words
        filtered_words = []
        for word in words:
            if len(word) > 2 and word.strip():  # English words should be at least 3 characters long
                filtered_words.append(word)
        
        return ' '.join(filtered_words)

    def classify_cancer_type(self, diagnosis_text: str) -> Optional[str]:
        """Classify cancer type."""
        if not diagnosis_text:
            return None
        
        # Check if it contains exclusion words
        diagnosis_lower = diagnosis_text.lower()
        for exclude_word in self.exclude_keywords:
            if exclude_word.lower() in diagnosis_lower:
                return None
        
        # Preprocess diagnosis text, remove status-describing words
        processed_text = self._remove_status_words(diagnosis_text)
        
        # Check cancer type
        for cancer_type, keywords in self.cancer_keywords.items():
            for keyword in keywords:
                if keyword.lower() in processed_text.lower():
                    return cancer_type
        
        return None

    def _remove_status_words(self, text: str) -> str:
        """Remove status-describing words to keep core cancer type information."""
        if not text:
            return text
        
        # List of status-describing words (English)
        status_words = [
            'preoperative', 'postoperative', 'pre-op', 'post-op', 'preoperative diagnosis', 'postoperative diagnosis',
            'pre-chemotherapy', 'post-chemotherapy', 'pre-radiation', 'post-radiation', 'pre-treatment', 'post-treatment',
            'recurrent', 'metastatic', 'advanced', 'early', 'intermediate', 'progressive',
            'primary', 'secondary', 'multiple', 'single',
            'left', 'right', 'bilateral', 'lateral',
            'upper', 'lower', 'anterior', 'posterior', 'internal', 'external',
            'invasive', 'non-invasive', 'in situ', 'aggressive',
            'well-differentiated', 'poorly differentiated', 'undifferentiated', 'high-grade', 'intermediate-grade', 'low-grade',
            't1', 't2', 't3', 't4', 'n0', 'n1', 'n2', 'n3', 'm0', 'm1',
            'stage i', 'stage ii', 'stage iii', 'stage iv', 'stage 1', 'stage 2', 'stage 3', 'stage 4',
            'suspicious', 'suspected', 'pending', 'awaiting', 'further evaluation',
            'with', 'complicated by', 'secondary to', 'concurrent', 'comorbidity'
        ]
        
        # Create a temporary text for processing
        temp_text = text.lower()
        
        # Remove status words
        for status_word in status_words:
            temp_text = temp_text.replace(status_word, ' ')
        
        # Clean up extra spaces
        temp_text = re.sub(r'\s+', ' ', temp_text).strip()
        
        return temp_text

    def calculate_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """Calculate text similarity matrix."""
        if not texts:
            return np.array([])
        
        # If there are too few texts, return directly
        if len(texts) < 2:
            return np.array([])
        
        # Filter empty texts
        valid_texts = [text for text in texts if text and text.strip()]
        if len(valid_texts) < 2:
            return np.array([])
        
        # Use TF-IDF vectorization, fix parameter settings
        try:
            vectorizer = TfidfVectorizer(
                max_features=min(1000, len(valid_texts)),  # Ensure it does not exceed the number of texts
                ngram_range=(1, 2),
                min_df=1,  # Change to 1 to avoid conflict with max_df
                max_df=0.95,  # Change to 0.95 to avoid being too strict
                stop_words=None,  # Do not use stopwords, as we have already preprocessed
                lowercase=True  # Convert English to lowercase
            )
            
            tfidf_matrix = vectorizer.fit_transform(valid_texts)
            
            # Check if the matrix is empty
            if tfidf_matrix.shape[0] < 2:
                return np.array([])
            
            similarity_matrix = cosine_similarity(tfidf_matrix)
            return similarity_matrix
            
        except ValueError as ve:
            logging.warning(f"TF-IDF vectorization parameter error: {ve}")
            # Try simpler parameters
            try:
                vectorizer = TfidfVectorizer(
                    max_features=100,
                    min_df=1,
                    max_df=1.0
                )
                tfidf_matrix = vectorizer.fit_transform(valid_texts)
                similarity_matrix = cosine_similarity(tfidf_matrix)
                return similarity_matrix
            except Exception as e2:
                logging.error(f"Simplified TF-IDF also failed: {e2}")
                return np.array([])
        except Exception as e:
            logging.error(f"Error calculating similarity matrix: {e}")
            return np.array([])

    def cluster_similar_cases(self, cases: List[Dict], similarity_threshold: float = 0.7) -> List[List[Dict]]:
        """Cluster similar cases."""
        if not cases:
            return []
        
        # If the number of cases is too large, sample first
        max_cases_for_clustering = 1000
        if len(cases) > max_cases_for_clustering:
            logging.info(f"Number of cases is too large ({len(cases)}), randomly sampling {max_cases_for_clustering} for clustering")
            import random
            random.shuffle(cases)
            cases = cases[:max_cases_for_clustering]
        
        # Extract diagnosis text
        texts = []
        valid_cases = []
        
        for case in cases:
            diagnosis_text = self.extract_diagnosis_text(case)
            if diagnosis_text:
                processed_text = self.preprocess_text(diagnosis_text)
                if processed_text:
                    texts.append(processed_text)
                    valid_cases.append(case)
        
        if not texts:
            return []
        
        # Calculate similarity matrix
        similarity_matrix = self.calculate_similarity_matrix(texts)
        
        if similarity_matrix.size == 0:
            return [[case] for case in valid_cases]
        
        # If there are too few texts, return directly
        if len(texts) < 2:
            return [[case] for case in valid_cases]
        
        # Use DBSCAN for clustering
        # Convert similarity to distance (1 - similarity), ensure no negative values
        distance_matrix = 1 - similarity_matrix
        
        # Ensure the distance matrix does not contain negative values
        distance_matrix = np.clip(distance_matrix, 0, 1)
        distance_matrix = np.nan_to_num(distance_matrix, nan=1.0)
        
        # Check if the distance matrix is valid
        if np.any(np.isnan(distance_matrix)) or np.any(np.isinf(distance_matrix)):
            logging.warning("Distance matrix contains invalid values, skipping clustering")
            return self._simple_clustering(valid_cases, texts)
        
        # Use DBSCAN for clustering
        clustering = DBSCAN(
            eps=1-similarity_threshold,
            min_samples=1,
            metric='precomputed'
        )
        
        try:
            cluster_labels = clustering.fit_predict(distance_matrix)
            
            # Group by cluster
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                clusters[label].append(valid_cases[i])
            
            return list(clusters.values())
            
        except Exception as e:
            logging.error(f"Error during clustering: {e}")
            logging.info("Using simple grouping instead of clustering")
            return self._simple_clustering(valid_cases, texts)

    def _simple_clustering(self, cases: List[Dict], texts: List[str]) -> List[List[Dict]]:
        """Simple text similarity-based clustering method."""
        if len(cases) <= 1:
            return [[case] for case in cases]
        
        clusters = []
        used_indices = set()
        
        for i, (case, text) in enumerate(zip(cases, texts)):
            if i in used_indices:
                continue
            
            # Create a new cluster
            current_cluster = [case]
            used_indices.add(i)
            
            # Find similar cases
            for j, (other_case, other_text) in enumerate(zip(cases, texts)):
                if j in used_indices or i == j:
                    continue
                
                # Simple text similarity calculation
                common_words = set(text.split()) & set(other_text.split())
                total_words = set(text.split()) | set(other_text.split())
                
                if len(total_words) > 0:
                    similarity = len(common_words) / len(total_words)
                    if similarity > 0.5:  # Similarity threshold
                        current_cluster.append(other_case)
                        used_indices.add(j)
            
            clusters.append(current_cluster)
        
        return clusters

    def process_files_incrementally(self) -> Dict[str, List[Dict]]:
        """Process files one by one to avoid memory issues."""
        cancer_groups = defaultdict(list)
        processed_count = 0
        error_count = 0
        
        if not os.path.exists(self.data_dir):
            logging.error(f"Data directory does not exist: {self.data_dir}")
            return cancer_groups
            
        try:
            # Get a list of all JSON files
            json_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
            total_files = len(json_files)
            logging.info(f"Found {total_files} JSON files, starting to process one by one...")
            
            for filename in json_files:
                file_path = os.path.join(self.data_dir, filename)
                try:
                    # Load files one by one
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Add file path information
                    data['_file_path'] = file_path
                    data['_filename'] = filename
                    
                    # Process a single file
                    diagnosis_text = self.extract_diagnosis_text(data)
                    if diagnosis_text:
                        cancer_type = self.classify_cancer_type(diagnosis_text)
                        if cancer_type:
                            cancer_groups[cancer_type].append(data)
                            # Debug info: display text before and after processing
                            processed_text = self._remove_status_words(diagnosis_text)
                            logging.debug(f"Cancer type: {cancer_type}")
                            logging.debug(f"  Original diagnosis: {diagnosis_text[:100]}...")
                            logging.debug(f"  After processing: {processed_text[:100]}...")
                            logging.debug(f"  Filename: {filename}")
                            logging.debug("-" * 50)
                    
                    processed_count += 1
                    
                    # Output progress every 100 processed files
                    if processed_count % 100 == 0:
                        # Get memory usage
                        process = psutil.Process()
                        memory_info = process.memory_info()
                        memory_mb = memory_info.rss / 1024 / 1024
                        logging.info(f"Processed {processed_count}/{total_files} files, memory usage: {memory_mb:.1f} MB")
                        
                except json.JSONDecodeError as e:
                    logging.warning(f"JSON parsing error for {filename}: {e}")
                    error_count += 1
                except Exception as e:
                    logging.warning(f"Error processing file {filename}: {e}")
                    error_count += 1
            
            logging.info(f"File processing complete!")
            logging.info(f"  Total files: {total_files}")
            logging.info(f"  Successfully processed: {processed_count}")
            logging.info(f"  Processing errors: {error_count}")
            logging.info(f"  Found cancer types: {len(cancer_groups)}")
            
            return cancer_groups
            
        except Exception as e:
            logging.error(f"Error processing files: {e}")
            return cancer_groups

    def has_more_than_10_admissions(self, data: Dict) -> bool:
        """Determine if the case has more than 10 admission records for the same patient (by counting the max X in 'The patient was recorded for the X time')."""
        text = data.get('text', '')
        matches = re.findall(r'The patient was recorded for the (\d+) time', text)
        if matches:
            max_num = max(int(x) for x in matches)
            return max_num > 10
        return False

    def extract_cancer_cases(self) -> Dict[str, List[Dict]]:
        """Extract cancer cases."""
        logging.info("Starting to extract cancer cases...")
        
        # Process files one by one
        cancer_groups = self.process_files_incrementally()
        
        logging.info(f"Found {len(cancer_groups)} cancer types")
        for cancer_type, cases in cancer_groups.items():
            logging.info(f"  {cancer_type}: {len(cases)} cases")
        
        # Select representative cases for each cancer type
        selected_cases = {}
        
        # First, filter all cases with more than 10 admission records
        def admission_count(case):
            text = case.get('text', '')
            matches = re.findall(r'The patient was recorded for the (\d+) time', text)
            if matches:
                return max(int(x) for x in matches)
            return 0
        
        # Collect all cases with more than 10 admission records, grouped by cancer type
        valid_cases_by_type = {}
        all_valid_cases = []
        
        for cancer_type, cases in cancer_groups.items():
            valid_cases = [case for case in cases if admission_count(case) > 10]
            if valid_cases:
                valid_cases_by_type[cancer_type] = valid_cases
                all_valid_cases.extend(valid_cases)
                logging.info(f"  {cancer_type} cases with more than 10 admission records: {len(valid_cases)} cases")
        
        # Sort all valid cases by admission count in descending order
        all_valid_cases_sorted = sorted(all_valid_cases, key=admission_count, reverse=True)
        
        # First, select the top N cases for each cancer type
        for cancer_type, valid_cases in valid_cases_by_type.items():
            cases_sorted = sorted(valid_cases, key=admission_count, reverse=True)
            selected = cases_sorted[:self.cases_per_type]
            selected_cases[cancer_type] = selected
            logging.info(f"  {cancer_type} finally selected {len(selected_cases[cancer_type])} cases")
        
        # Calculate the current total number of cases
        total_selected = sum(len(cases) for cases in selected_cases.values())
        logging.info(f"Current total number of cases: {total_selected}")
        
        # If the total is less than 120, supplement from other types to around 150
        if total_selected < 120:
            target_total = 150
            needed = target_total - total_selected
            logging.info(f"Total cases less than 120, need to supplement {needed} cases to reach {target_total}")
            
            # Select from all valid cases sorted by admission count, but exclude already selected cases
            selected_filepaths = set()
            for cases in selected_cases.values():
                for case in cases:
                    selected_filepaths.add(case.get('_file_path'))
            
            # Select from remaining cases
            remaining_cases = [case for case in all_valid_cases_sorted if case.get('_file_path') not in selected_filepaths]
            
            # Group remaining cases by cancer type
            remaining_by_type = defaultdict(list)
            for case in remaining_cases:
                # Reclassify cases (as they might have been processed already)
                diagnosis_text = self.extract_diagnosis_text(case)
                if diagnosis_text:
                    cancer_type = self.classify_cancer_type(diagnosis_text)
                    if cancer_type:
                        remaining_by_type[cancer_type].append(case)
            
            # Supplement types evenly using round-robin until the target number is reached
            type_order = list(remaining_by_type.keys())
            if type_order:
                logging.info(f"  Supplementing via round-robin, target to add {needed} cases")
                
                # Round-robin supplement until the target number is reached or no more cases are available
                type_index = 0
                while needed > 0:
                    cancer_type = type_order[type_index % len(type_order)]
                    available_cases = remaining_by_type[cancer_type]
                    
                    if available_cases:
                        # Take one case from the current type
                        case_to_add = available_cases.pop(0)
                        
                        if cancer_type not in selected_cases:
                            selected_cases[cancer_type] = []
                        
                        selected_cases[cancer_type].append(case_to_add)
                        needed -= 1
                        logging.info(f"  Supplemented {cancer_type}: 1 case (still need: {needed})")
                    else:
                        # If the current type has no more cases, check if other types have cases
                        if not any(remaining_by_type.values()):
                            logging.warning(f"  No more available cases, stopping supplement")
                            break
                    
                    type_index += 1
        
        final_total = sum(len(cases) for cases in selected_cases.values())
        logging.info(f"Final total number of cases: {final_total}")
        
        return selected_cases

    def save_extracted_cases(self, selected_cases: Dict[str, List[Dict]]):
        """Save the extracted cases."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Save cases for each cancer type
        for cancer_type, cases in selected_cases.items():
            cancer_dir = os.path.join(self.output_dir, cancer_type)
            if not os.path.exists(cancer_dir):
                os.makedirs(cancer_dir)
            
            for i, case in enumerate(cases):
                # Generate output filename
                original_filename = case.get('_filename', f'case_{i+1}.json')
                output_filename = f"{cancer_type}_case_{i+1:02d}.json"
                output_path = os.path.join(cancer_dir, output_filename)
                
                # Save case data
                case_data = {k: v for k, v in case.items() if not k.startswith('_')}
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(case_data, f, ensure_ascii=False, indent=2)
        
        # Generate summary report
        summary_path = os.path.join(self.output_dir, 'extraction_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("Cancer Case Extraction Summary Report\n")
            f.write("=" * 50 + "\n\n")
            
            for cancer_type, cases in selected_cases.items():
                f.write(f"{cancer_type}:\n")
                f.write(f"  Extracted cases: {len(cases)}\n")
                for i, case in enumerate(cases):
                    diagnosis_text = self.extract_diagnosis_text(case)
                    f.write(f"  Case {i+1}: {case.get('_filename', 'unknown')}\n")
                    f.write(f"    Diagnosis: {diagnosis_text[:100]}...\n")
                f.write("\n")
        
        logging.info(f"Cases have been saved to: {self.output_dir}")
        logging.info(f"Summary report: {summary_path}")

    def run(self):
        """Execute the complete extraction process."""
        logging.info("Starting cancer case extraction process...")
        
        try:
            # Extract cancer cases
            selected_cases = self.extract_cancer_cases()
            
            if not selected_cases:
                logging.warning("No qualifying cancer cases found")
                return
            
            # Save the extracted cases
            self.save_extracted_cases(selected_cases)
            
            # Output statistics
            logging.info("=" * 50)
            logging.info("Extraction complete!")
            logging.info(f"Extracted a total of {len(selected_cases)} cancer types")
            total_cases = sum(len(cases) for cases in selected_cases.values())
            logging.info(f"Total number of cases: {total_cases}")
            logging.info(f"Output directory: {self.output_dir}")
            
        except Exception as e:
            logging.error(f"An error occurred during execution: {e}")
            raise

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Cancer Case Extraction Tool')
    parser.add_argument('--data_dir', type=str, 
                        default="clinic_json_cancer",
                        help='Path to the data directory')
    parser.add_argument('--output_dir', type=str, 
                        default="extracted_cancer_cases",
                        help='Path to the output directory')
    parser.add_argument('--no_clustering', action='store_true',
                        help='Disable clustering and select cases randomly')
    parser.add_argument('--cases_per_type', type=int, default=10,
                        help='Number of cases to extract per cancer type')
    
    args = parser.parse_args()
    
    # Create the extractor
    extractor = CancerCaseExtractor(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        use_clustering=not args.no_clustering
    )
    
    # Set the number of cases
    extractor.cases_per_type = args.cases_per_type
    
    # Run the extraction
    extractor.run()

if __name__ == "__main__":
    main()
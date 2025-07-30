import json
import os
import torch
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from torch.utils.data import Dataset, DataLoader
from transformers import LongformerTokenizer, LongformerForSequenceClassification
from torch.optim import AdamW
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
from tqdm import tqdm
import argparse
import safetensors

class ClinicalRecordDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=4096, window_size=512, stride=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.window_size = window_size
        self.stride = stride
        self.records = self._load_and_process_json(json_file)
        
    def _clean_time_info(self, text):
        # English time-related regular expressions
        time_patterns = [
            r"\b\d{1,2}/\d{1,2}/\d{4}\b",  # Date format
            r"\b\d{4}-\d{1,2}-\d{1,2}\b",  # Another date format
            r"\b\d{1,2}:\d{2}(?::\d{2})?\b",  # Time format
            r"\bday[s]?\b", r"\bhour[s]?\b", r"\bminute[s]?\b",
            r"\bAM\b|\bPM\b",
            r"\bthe event type was [a-zA-Z]+\b",
            r"\bthe drg type was [A-Z]+\b",
            r"\bgroup description: [A-Z ]+\b",
        ]
        combined_pattern = "|".join(f"({pattern})" for pattern in time_patterns)
        cleaned_text = re.sub(combined_pattern, "", text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)
        return cleaned_text.strip()

    def _create_windows(self, text):
        # English sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        windows = []
        current_window = []
        current_length = 0
        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length > self.window_size:
                if current_window:
                    windows.append(" ".join(current_window))
                current_window = [sentence]
                current_length = sentence_length
            else:
                current_window.append(sentence)
                current_length += sentence_length
        if current_window:
            windows.append(" ".join(current_window))
        return windows

    def _load_and_process_json(self, json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        records = []
        text = data['text']
        # Split by admission using "The patient was recorded for the N time:" as priority delimiter (English)
        admission_records = re.split(r'The patient was recorded for the \d+ time:', text)[1:]
        if not admission_records:
            # If the pattern is not found, process the entire text as one record
            admission_records = [text]
        section_patterns = {
            'diagnosis': r'patient diagnosis information:([^\n]*)',
            'medication': r'patient medication information:([^\n]*)',
            'prescription': r'prescription information:([^\n]*)',
            'procedure': r'procedure information:([^\n]*)',
            'transfer': r'patient transfer information:([^\n]*)',
            'medication_details': r'patient medication details:([^\n]*)',
            'other': r'other information:([^\n]*)',
        }
        for record in admission_records:
            full_record = []
            for section_name, pattern in section_patterns.items():
                section_matches = re.finditer(pattern, record, re.IGNORECASE | re.DOTALL)
                for match in section_matches:
                    section_text = match.group(1).strip()
                    if section_text:
                        section_text = self._clean_time_info(section_text)
                        if section_name == 'medication':
                            section_text = self._filter_abnormal_results(section_text)
                        if section_text:
                            full_record.append(f"{section_name}: {section_text}")
            if full_record:
                full_text = "\n".join(full_record)
                windows = self._create_windows(full_text)
                for window in windows:
                    records.append({
                        'text': window,
                        'full_text': full_text
                    })
        return records

    def _filter_abnormal_results(self, text):
        # Keep only abnormal results (English)
        test_items = re.split(r',|;|\|', text)
        abnormal_items = []
        for item in test_items:
            item = item.strip()
            if not item:
                continue
            # Skip normal results
            if re.search(r'normal', item, re.IGNORECASE):
                continue
            # Abnormal keywords
            abnormal_markers = [
                'abnormal', 'high', 'low', 'positive', 'negative', 'suspicious', 'pending', 'critical', 'alert'
            ]
            if any(marker in item.lower() for marker in abnormal_markers):
                abnormal_items.append(item)
            elif re.search(r'[<>↑↓]', item):
                abnormal_items.append(item)
            elif re.search(r'(abnormal|high|low|positive|negative|suspicious|pending|critical|alert)', item, re.IGNORECASE):
                abnormal_items.append(item)
        return " | ".join(abnormal_items) if abnormal_items else ""
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        record = self.records[idx]
        text = record['text']
        
        # Encode the text
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Ensure the input tensors have the correct dimensions
        input_ids = encoded['input_ids'].squeeze(0)  # [max_length]
        attention_mask = encoded['attention_mask'].squeeze(0)  # [max_length]
        
        return {
            'text': text,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'full_text': record['full_text']
        }

class ClinicalSummarizer:
    def __init__(self, model_name="yikuan8/Clinical-Longformer", gpu_id=0):
        # Set the specified GPU
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            self.device = f'cuda:{gpu_id}'
        else:
            self.device = 'cpu'
            
        self.tokenizer = LongformerTokenizer.from_pretrained(model_name)
        self.model = LongformerForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2,
            use_safetensors=True  # Use safetensors format
        )
        self.model.to(self.device)
        self.sentence_transformer = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
    def _calculate_sentence_importance(self, sentence):
        # Ensure sentence is a string type
        if not isinstance(sentence, str):
            sentence = str(sentence)
            
        # Calculate sentence importance based on multiple features
        features = []
        
        # 1. Sentence length feature (in medical records, medium-length sentences often contain more information)
        length = len(sentence)
        length_score = 1.0 - abs(length - 150) / 300  # Assuming 150 characters is the ideal length
        features.append(length_score)
        
        # 2. Keyword feature
        medical_keywords = {
            # Cancer/diagnosis related (weight: 1.0)
            'cancer': 1.0, 'malignant neoplasm': 1.0, 'neoplasm': 1.0, 'tumor': 1.0, 'carcinoma': 1.0, 'recurrence': 1.0, 'metastasis': 1.0,
            # ... (all keywords are in English, no change needed)
            'secondary': 0.5,
        }
        keyword_score = 0
        for keyword, weight in medical_keywords.items():
            if keyword in sentence.lower():
                keyword_score += weight
        keyword_score = min(keyword_score / 7, 1.0)
        features.append(keyword_score)
        
        # 3. Position feature (with sliding windows, all sentences are considered equally important)
        position_score = 1.0
        features.append(position_score)
        
        # 4. Time information feature (sentences with specific time information are often more important)
        time_patterns = [
            r'\b\d{4}-\d{1,2}-\d{1,2}\b',
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            r'admit', r'transfer', r'day', r'hour', r'minute'
        ]
        time_score = 0
        for pattern in time_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                time_score = 0.5
                break
        features.append(time_score)
        
        # 5. Numerical information feature (sentences with specific numbers are often more important)
        number_score = 0.5 if re.search(r'\d+', sentence) else 0
        features.append(number_score)
        
        # 6. Treatment stage feature (sentences from different treatment stages have different importance)
        treatment_stages = {
            'preoperative': 0.8, 'intraoperative': 0.9, 'postoperative': 0.8,
            'chemotherapy': 0.9, 'radiotherapy': 0.9, 'follow-up': 0.7
        }
        stage_score = 0
        for stage, weight in treatment_stages.items():
            if stage in sentence.lower():
                stage_score = weight
                break
        features.append(stage_score)
        
        # Combined score (using weighted average)
        weights = [0.15, 0.35, 0.1, 0.15, 0.1, 0.15]  # Keywords are most important, followed by treatment stage and length
        importance_score = sum(score * weight for score, weight in zip(features, weights))
        
        return importance_score
        
    def train_proxy_task(self, train_dataloader, num_epochs=6):
        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        best_summaries = {}  # Store the best summary for each data point
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            
            for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
                # Get text content from the batch
                text = batch['text'][0] if isinstance(batch['text'], list) else batch['text']
                text_key = str(text)
                
                # Calculate sentence importance score
                importance_score = self._calculate_sentence_importance(text)
                
                # Convert importance score to a binary label
                label = 1 if importance_score > 0.5 else 0
                
                # Prepare model inputs
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = torch.tensor([label]).to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                
                # Generate summary
                summary = self.extract_summary(text)
                cleaned_summary = self._clean_summary_prefix(" ".join(summary) + ".")
                
                # Update best summary
                if text_key not in best_summaries or loss.item() < best_summaries[text_key]['loss']:
                    best_summaries[text_key] = {
                        'summary': cleaned_summary,
                        'loss': loss.item(),
                        'epoch': epoch + 1
                    }
                
                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_dataloader)
            print(f'Epoch {epoch + 1} average loss: {avg_loss:.4f}')
        
        return best_summaries
    
    def extract_summary(self, text, k=5):
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Encode and predict for each sentence
        self.model.eval()
        sentence_scores = []
        
        with torch.no_grad():
            for sentence in sentences:
                # Encode the sentence
                encoded = self.tokenizer(
                    sentence,
                    return_tensors='pt',
                    truncation=True,
                    max_length=4096,
                    padding='max_length'
                )
                
                # Ensure the input tensors have the correct dimensions
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # Create global attention mask
                global_attention_mask = torch.zeros_like(attention_mask)
                global_attention_mask[:, 0] = 1  # Set the first token to have global attention
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask
                )
                scores = torch.softmax(outputs.logits, dim=1)
                # Use the probability of the positive class as the importance score
                importance_score = scores[0][1].item()
                
                # Calculate the sentence's compression score
                compression_score = self._calculate_compression_score(sentence)
                
                # Final score = importance score * compression score
                final_score = importance_score * compression_score
                
                sentence_scores.append((sentence, final_score))
        
        # Sort by importance score and select top k sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Apply compression rules
        compressed_summary = self._compress_summary(sentence_scores[:k])
        
        return compressed_summary

    def _calculate_compression_score(self, sentence):
        """Calculate the sentence's compression score"""
        # 1. Remove redundant information
        redundant_patterns = [
            # ... (all patterns are in English, no change needed)
            r'no suspicious lymph node',
        ]
        original_length = len(sentence)
        compressed_sentence = sentence
        for pattern in redundant_patterns:
            compressed_sentence = re.sub(pattern, '', compressed_sentence, flags=re.IGNORECASE)
        compression_ratio = len(compressed_sentence) / original_length if original_length > 0 else 0
        # Info density: count medical keywords
        medical_info = re.findall(r'[a-zA-Z0-9]+|cancer|chemotherapy|surgery|diagnosis|drug|dose|admit|transfer|procedure|abnormal|positive|negative|high|low|critical|alert', sentence, re.IGNORECASE)
        info_density = len(medical_info) / len(sentence) if len(sentence) > 0 else 0
        
        # 3. Calculate sentence complexity
        # Simpler sentences get a higher score
        complexity_score = 1.0 - min(len(sentence.split()) / 20, 1.0)  # Assuming 20 words is the ideal length
        
        # Combined score
        compression_score = (compression_ratio * 0.4 + info_density * 0.4 + complexity_score * 0.2)
        
        return compression_score

    def _compress_summary(self, sentence_scores):
        """Compress the summary content"""
        compressed_sentences = []
        seen_info = set()  # Used to track included information
        
        for sentence, score in sentence_scores:
            # Extract key information from the sentence
            key_info = self._extract_key_info(sentence)
            
            # Check if it contains new important information
            if not any(info in seen_info for info in key_info):
                # Compress the sentence
                compressed_sentence = self._compress_sentence(sentence)
                compressed_sentences.append(compressed_sentence)
                
                # Update the set of seen information
                seen_info.update(key_info)
        
        return compressed_sentences

    def _extract_key_info(self, sentence):
        """Extract key information from the sentence"""
        # Define key information patterns
        key_patterns = [
            # ... (all patterns are in English, no change needed)
            r'\d{1,2}:\d{2}(?::\d{2})?',
        ]
        key_info = []
        for pattern in key_patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            key_info.extend(matches)
        return set(key_info)

    def _compress_sentence(self, sentence):
        """Compress a single sentence"""
        # 1. Remove redundant information
        redundant_patterns = [
             # ... (all patterns are in English, no change needed)
            r'no suspicious lymph node',
        ]
        compressed = sentence
        for pattern in redundant_patterns:
            compressed = re.sub(pattern, '', compressed)
        
        # 2. Merge duplicate information
        compressed = re.sub(r'(\w+)(?:\s+\1)+', r'\1', compressed)
        
        # 3. Simplify expressions
        replacements = {
             # ... (all patterns are in English, no change needed)
            r'cancer antigen 27-29': 'CA27-29',
        }
        for pattern, repl in replacements.items():
            compressed = re.sub(pattern, repl, compressed, flags=re.IGNORECASE)
        return compressed.strip()

    def _clean_summary_prefix(self, summary):
        """Clean useless prefixes from the summary"""
        # Define common prefix cleaning patterns
        prefix_patterns = [
            r'^[^：:]*[：:][，,]\s*',  # Match content before colon plus comma
            r'^[^：:]*[：:][。.]\s*',  # Match content before colon plus period
            r'^[^：:]*[：:][；;]\s*',  # Match content before colon plus semicolon
            r'^[^：:]*[：:]\s*',       # Match content before colon
        ]
        
        # Process each sentence
        sentences = summary.split('.')
        cleaned_sentences = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Apply all cleaning patterns
            cleaned = sentence
            for pattern in prefix_patterns:
                cleaned = re.sub(pattern, '', cleaned)
            
            if cleaned.strip():
                cleaned_sentences.append(cleaned.strip())
        
        return '.'.join(cleaned_sentences) + '.'

def main(gpu_id=0):
    # Initialize model and tokenizer
    summarizer = ClinicalSummarizer(gpu_id=gpu_id)
    
    # Create save directory
    os.makedirs("./Summary_data", exist_ok=True)
    
    # Get all JSON files
    json_files = [f for f in os.listdir("./High_Quality_data") if f.endswith('.json')]
    
    # Filter out already processed files
    processed_files = set()
    if os.path.exists("./Summary_data"):
        processed_files = {f.replace('.json', '') for f in os.listdir("./Summary_data") if f.endswith('.json')}
    
    # Only process unprocessed files
    unprocessed_files = []
    for json_file in json_files:
        base_name = os.path.splitext(json_file)[0]
        if base_name not in processed_files:
            unprocessed_files.append(json_file)
    
    total_files = len(unprocessed_files)
    print(f"Starting to process {total_files} unprocessed JSON files...")
    print(f"Skipping {len(json_files) - total_files} already processed files")
    
    if total_files == 0:
        print("All files have been processed!")
        return
    
    # Set batch size
    batch_size = 64  # Process 64 files at a time
    
    # Process files in batches
    for i in range(0, total_files, batch_size):
        batch_files = unprocessed_files[i:i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{(total_files + batch_size - 1)//batch_size}")
        
        # Store all datasets for the current batch
        batch_datasets = []
        batch_paths = []
        
        # Load all data for the current batch
        for json_file in batch_files:
            try:
                json_path = os.path.join("./High_Quality_data", json_file)
                dataset = ClinicalRecordDataset(json_path, summarizer.tokenizer)
                batch_datasets.append(dataset)
                batch_paths.append(json_path)
                print(f"Loaded file: {json_file}")
            except Exception as e:
                print(f"Error loading file {json_file}: {str(e)}")
                continue
        
        if not batch_datasets:
            continue
        
        # Combine all datasets
        combined_dataset = torch.utils.data.ConcatDataset(batch_datasets)
        dataloader = DataLoader(combined_dataset, batch_size=1, shuffle=True)
        
        # Train the model and get the best summaries
        best_summaries = summarizer.train_proxy_task(dataloader)
        
        # Process the summary for each file
        current_idx = 0
        for dataset, json_path in zip(batch_datasets, batch_paths):
            try:
                # Get the original filename (without path and extension)
                base_name = os.path.splitext(os.path.basename(json_path))[0]
                output_file = f"./Summary_data/{base_name}.json"
                
                # Store summaries for all records
                summaries = {}
                
                # Organize the best summaries
                for j in range(len(dataset)):
                    record = dataset[j]
                    text_key = str(record['text'])
                    best_info = best_summaries.get(text_key, {
                        'summary': '',
                        'loss': 999999.0,
                        'epoch': 0
                    })
                    
                    record_key = f"record_{j+1}"
                    summaries[record_key] = {
                        "summary": best_info['summary'],
                        "original_text": record['text'],
                        "best_loss": float(best_info['loss']),
                        "best_epoch": int(best_info['epoch'])
                    }
                
                # Save as a JSON file
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(summaries, f, ensure_ascii=False, indent=4)
                
                print(f"Summary for file {base_name} saved to: {output_file}")
                print(f"Processing complete, total of {len(summaries)} records. The summary with the lowest loss was saved for each record.")
                
            except Exception as e:
                print(f"Error processing file {json_path}: {str(e)}")
                continue
        
        print(f"Batch {i//batch_size + 1} processing complete")
    
    print("\nAll files processed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=1, help='GPU ID to use (default: 0)')
    args = parser.parse_args()
    main(gpu_id=args.gpu_id)
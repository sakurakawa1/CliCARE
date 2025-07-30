# Longformer Text Processing

### `extractive_summarizer.py`
Utilizes a pre-trained clinical long-text model (Clinical-Longformer) to create a condensed summary by extracting key information from medical records based on sentence importance scores.

### `merge_summary_json.py`
Merges multiple summary records into a single JSON file. In the output, all records are represented by their previously generated summaries, except for the final record, which retains its original, uncompressed text.
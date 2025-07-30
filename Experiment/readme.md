# Clinical  Generation and Evaluation 

This section outlines a comprehensive pipeline for generating, fine-tuning, and evaluating Large Language Models (LLMs) on the task of creating clinical pathways from patient health records. The workflow consists of four main stages: data generation, model fine-tuning, prediction, and comparative evaluation.

##  Overview

The process follows these sequential steps:

1.  **Patient Data (Input JSONs)**
    `->` **`QA_generation.py`** (Generates instruction-following dataset)
2.  **Fine-tuning Data (Alpaca-formatted JSON)**
    `->` **`Fine-Turning.sh`** (Fine-tunes a base model using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory))
3.  **Trained LoRA Adapter**
    `->` **`predict.sh`** (Generates predictions on an evaluation set)
4.  **Model Predictions (JSONL format)**
    `->` **`Evaluation.py`**
5.  **Evaluation Reports (TXT Summary & Detailed JSONL)**

---

## File Descriptions

### 1. `QA_generation.py` - Instruction Data Generation
Transforms patient clinical records into Q&A format training data with RAG enhancement and configurable parameters (`--use_kg` for knowledge graph, `--compress_latest` for record compression, `--extract_history` for history extraction).

### 2. `Fine-Turning.sh` - Model Fine-tuning
Performs LoRA fine-tuning on base models using LLaMA-Factory framework with configurable parameters including model path, dataset, learning rate, and training epochs.

### 3. `predict.sh` - Prediction Generation
Runs inference with fine-tuned models to generate predictions on evaluation datasets and saves results in JSONL format.

### 4. `Evaluation.py` - Model Evaluation
Uses LLM-as-a-judge method to evaluate multiple models across four dimensions (factual accuracy, completeness, clinical soundness, actionability) and generates detailed scoring reports and summary statistics.
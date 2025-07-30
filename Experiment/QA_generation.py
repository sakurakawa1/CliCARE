# -*- coding: utf-8 -*-
import os
import json
import logging
# 设置环境变量以使用镜像站点
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from typing import List, Dict, Optional, Tuple, Any
import re
import math
import sys
import time
import concurrent.futures
import argparse
from tqdm import tqdm

# --- Tiktoken (用于Token估算) ---
try:
    import tiktoken
    # 使用cl100k_base编码器，适用于较新的GPT模型
    tokenizer = tiktoken.get_encoding("cl100k_base")
    logging.info("Tiktoken tokenizer loaded (cl100k_base for estimation).")
except ImportError:
    logging.warning("Tiktoken not installed. Using character count heuristic. Please run `pip install tiktoken` for better accuracy.")
    tokenizer = None
except Exception as e:
    logging.error(f"Error loading tiktoken tokenizer: {e}")
    tokenizer = None

# --- 依赖库导入 ---
try:
    from openai import OpenAI, RateLimitError, APIError, APITimeoutError
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    if not all([OpenAI, FAISS, HuggingFaceEmbeddings, DirectoryLoader, TextLoader, RecursiveCharacterTextSplitter]):
         raise ImportError("One or more required libraries are missing.")
except ImportError as e:
    logging.error(f"Import Error: {e}. Please install all required dependencies: pip install openai langchain-community faiss-cpu sentence-transformers tiktoken tqdm")
    sys.exit(1)

# --- 配置与常量 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
API_KEY = "sk-XXX"  # 请替换为您的API密钥
BASE_URL = "https://api.deepseek.com"
CLINICAL_PATH_DIR = "/home/admin1/code/CdrDataProcess/process_clinical_records/clinical_pathways_En"
max_worker = 32

# --- 新增：通过命令行参数进行灵活配置 ---
parser = argparse.ArgumentParser(description='Clinical Pathway Processor from Patient Records')
parser.add_argument('--use_kg', action='store_true', help='Enable Knowledge Graph (KG) mode. The model will receive guideline info from the record combined with RAG context.')
parser.add_argument('--compress_latest', action='store_true', help='Enable LLM-based compression for the latest clinical record to save tokens.')
# --- 新增：控制历史记录提取的开关 ---
parser.add_argument('--extract_history', action='store_true', help='Enable LLM-based key info extraction for historical records. If disabled, raw history text will be used.')
# 新增输入输出路径参数，直接给默认值，只保留一对输入输出路径
parser.add_argument('--input_dir', type=str, default='/home/admin1/code/DataProcess/Longformer_data_KG', help='Input JSON directory')
parser.add_argument('--output_dir', type=str, default='/home/admin1/code/DataProcess/QA_Longformer_data_KG', help='Output directory')
args = parser.parse_args()

# --- 根据命令行参数动态设置路径和模式 ---
use_kg = args.use_kg
compress_latest_record = args.compress_latest
extract_history_summary = args.extract_history
logging.info(f"Knowledge Graph (KG) Mode: {'ENABLED' if use_kg else 'DISABLED'}")
logging.info(f"Latest Record Compression: {'ENABLED' if compress_latest_record else 'DISABLED'}")
logging.info(f"History Summary Extraction: {'ENABLED' if extract_history_summary else 'DISABLED'}")

# 只保留一对输入输出路径
JSON_DIR = args.input_dir
OUTPUT_DIR = args.output_dir
OUTPUT_FILENAME_SUFFIX = "_alpaca_pathway_reasoner.json"
MODEL_NAME_FOR_GENERATION = "deepseek-reasoner"
EMBEDDING_CACHE_DIR = "./embedding_cache"

# 过滤和截断的常量
MIN_RECORD_LENGTH_FOR_QUALITY = 500 # 原始文本的最小字符数，用于质量过滤

# 模型上下文和Token数配置 (以96K模型为例)
MODEL_MAX_CONTEXT = 64000  # 64k 上下文窗口
REQUESTED_OUTPUT_TOKENS = 8192  # 为模型生成回复预留的Token数
SAFETY_MARGIN = 4000  # 额外的安全边界，防止因Token估算误差导致超长
# 用于生成临床路径的输入Token阈值
INPUT_TOKEN_THRESHOLD_GENERATION = MODEL_MAX_CONTEXT - REQUESTED_OUTPUT_TOKENS - SAFETY_MARGIN
# 用于生成微调数据的输入Token阈值
FINETUNE_INPUT_TARGET_MAX_TOKENS = MODEL_MAX_CONTEXT - 2000

logging.info(f"Target Model Max Context (96K): {MODEL_MAX_CONTEXT}")
logging.info(f"Generation Input Token Threshold: {INPUT_TOKEN_THRESHOLD_GENERATION}")
logging.info(f"Fine-tuning Input Target Max Tokens: {FINETUNE_INPUT_TARGET_MAX_TOKENS}")

# --- 主处理类 ClinicalPathwayProcessorWhole ---
class ClinicalPathwayProcessorWhole:
    def __init__(self, api_key: str, base_url: str, clinical_path_dir: str, json_dir: str, use_kg: bool, compress_latest: bool, extract_history: bool, output_dir: str):
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=180.0) # 设置较长的超时时间
        self.clinical_path_dir = clinical_path_dir
        self.json_dir = json_dir
        self.output_dir = output_dir
        # 将命令行参数存为实例属性，方便各方法调用
        self.use_kg = use_kg
        self.compress_latest = compress_latest
        self.extract_history = extract_history # 新增属性
        self.vector_store = None
        self.embeddings = None
        logging.info(f"Processor initialized. Model: {MODEL_NAME_FOR_GENERATION}. Output directory: {self.output_dir}")

    def _estimate_tokens(self, text: str) -> int:
        """使用tiktoken估算文本的token数量，若失败则回退到字符数估算。"""
        if tokenizer is None:
            return math.ceil(len(text) / 1.5)  # 基于经验的回退策略
        try:
            if not isinstance(text, str): text = str(text) # 保证输入是字符串
            return len(tokenizer.encode(text))
        except Exception as e:
            logging.error(f"Token estimation error: {e}. Falling back to character count heuristic.")
            return math.ceil(len(text) / 1.5)

    def process_txt_files(self):
        """加载、分割临床路径txt文件，并构建FAISS向量库用于RAG。"""
        if not os.path.exists(self.clinical_path_dir) or not os.path.isdir(self.clinical_path_dir):
            logging.error(f"Clinical pathway directory not found: {self.clinical_path_dir}"); return
        try:
            if not os.path.exists(EMBEDDING_CACHE_DIR): os.makedirs(EMBEDDING_CACHE_DIR)
            logging.info("Loading embedding model (shibing624/text2vec-base-chinese)...")
            self.embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese", cache_folder=EMBEDDING_CACHE_DIR)
            logging.info("Embedding model loaded successfully.")

            loader = DirectoryLoader(
                self.clinical_path_dir, glob="**/*.txt", loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8', 'autodetect_encoding': True},
                show_progress=True, use_multithreading=True, silent_errors=True
            )
            logging.info("Loading documents from clinical pathway directory...")
            documents = loader.load()
            # 过滤空文档
            documents = [d for d in documents if hasattr(d, 'page_content') and d.page_content and d.page_content.strip()]
            if not documents:
                logging.warning("No valid documents found in the directory. Vector store will not be created."); self.vector_store = None; return
            logging.info(f"Loaded {len(documents)} valid documents.")

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            logging.info("Splitting documents into chunks...")
            chunks = text_splitter.split_documents(documents)
            logging.info(f"Split documents into {len(chunks)} chunks.")
            if not chunks:
                logging.warning("No chunks were generated from the documents. Vector store will not be created."); self.vector_store = None; return

            logging.info("Creating FAISS vector store from chunks...")
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            logging.info("FAISS vector store created successfully.")
        except Exception as e:
            logging.error(f"An error occurred during TXT file processing and vector store creation: {e}", exc_info=True)
            self.vector_store = None

    def _get_relevant_context(self, query: str, k: int = 2) -> str:
        """根据查询，从FAISS向量库中检索相关上下文(RAG)。"""
        if not self.vector_store:
            logging.warning("Vector store is not initialized. Cannot perform RAG search."); return "Unable to retrieve context (vector store not initialized)"
        if not query or not query.strip():
            logging.warning("RAG query is empty. Cannot perform search."); return "Unable to retrieve context (empty query)"

        max_context_len = 3000 # 限制RAG上下文的总字符数
        try:
            # fetch_k > k, 检索更多文档以提高多样性
            docs = self.vector_store.similarity_search(query, k=k, fetch_k=max(k*5, 20))
            context_with_sources = []
            total_len = 0
            for i, doc in enumerate(docs):
                source = os.path.basename(doc.metadata.get('source', f'UnknownSource_{i+1}'))
                page_content = getattr(doc, 'page_content', '')
                if page_content.strip():
                    content_to_add = f"--- Reference Document {i+1} ({source}) ---\n{page_content}\n"
                    if total_len + len(content_to_add) <= max_context_len:
                        context_with_sources.append(content_to_add)
                        total_len += len(content_to_add)
                    else: # 如果超出长度，则截断
                        remaining = max_context_len - total_len
                        if remaining > 100: # 保证有足够空间添加截断提示
                            context_with_sources.append(content_to_add[:remaining] + "...\n[Content Truncated]\n")
                        logging.warning(f"RAG context was truncated to {max_context_len} characters.")
                        break
            if not context_with_sources: return "No relevant context retrieved from the knowledge base."
            return "\n".join(context_with_sources)
        except Exception as e:
            logging.error(f"An error occurred during RAG search: {e}", exc_info=True)
            return "Internal error occurred while retrieving relevant context."

    def _call_llm_with_retry(self, messages: List[Dict[str, str]], max_tokens: int, temperature: float, task_name: str) -> str:
        """调用LLM API，并包含重试逻辑以应对网络波动和API限制。"""
        max_retries = 3; retry_delay = 5 # seconds
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(model=MODEL_NAME_FOR_GENERATION, messages=messages, temperature=temperature, max_tokens=max_tokens)
                if response.choices and response.choices[0].message and response.choices[0].message.content:
                    return response.choices[0].message.content.strip()
                else:
                    logging.warning(f"API response was empty or invalid for task '{task_name}' (Attempt {attempt+1}). Response: {response}")
                    if attempt < max_retries - 1: time.sleep(retry_delay); continue
                    else: raise ValueError(f"API response empty/invalid after {max_retries} attempts for {task_name}")
            except RateLimitError as rle:
                logging.warning(f"Rate limit exceeded for task '{task_name}' (Attempt {attempt + 1}): {rle}. Retrying after delay...")
                time.sleep(retry_delay * (attempt + 1))
            except (APIError, APITimeoutError) as ae:
                logging.warning(f"API error or timeout for task '{task_name}' (Attempt {attempt + 1}): {ae}. Retrying after delay...")
                time.sleep(retry_delay * (attempt + 1))
            except Exception as e:
                logging.error(f"An unexpected error occurred during API call for task '{task_name}' (Attempt {attempt + 1}): {e}", exc_info=True)
                if attempt < max_retries - 1: time.sleep(retry_delay)
                else: raise e # 在最后一次尝试后重新抛出异常
        raise Exception(f"API call for task '{task_name}' failed after {max_retries} attempts.")

    def _extract_key_info(self, full_history_text: str) -> str:
        """使用LLM从完整的历史病历中提取结构化的关键信息。"""
        if not full_history_text or not full_history_text.strip(): return "No relevant medical history available for extraction."
        
        # 为提取任务的Prompt本身留出足够空间
        max_extraction_input_tokens = INPUT_TOKEN_THRESHOLD_GENERATION - 1500
        history_tokens = self._estimate_tokens(full_history_text)
        text_for_extraction = full_history_text
        
        if history_tokens > max_extraction_input_tokens:
            logging.warning(f"History text for extraction is too long ({history_tokens} tokens > {max_extraction_input_tokens}). Truncating text.")
            # 按比例截断，保留文本末尾的最新部分
            cutoff_chars = int(len(full_history_text) * max_extraction_input_tokens / (history_tokens + 1e-6) * 0.9)
            text_for_extraction = full_history_text[-cutoff_chars:]

        extraction_prompt = f"""Please extract and summarize the key clinical information about cancer from the following complete or partial medical history for subsequent clinical pathway planning. Focus on:
1. **Main diagnosis and staging:** (including explicit TNM, FIGO, etc.)
2. **Key biomarkers and trends:** (e.g., HCG, CEA, etc. with values over time)
3. **Therapeutic efficacy assessment:** (e.g., CR, PR, SD, PD)
4. **Key imaging findings:** (tumor burden and changes, e.g., lung nodules, tumor size)
5. **Important comorbidities and allergies:**
6. **Major adverse reactions/toxicities:** (events leading to treatment adjustment, e.g., grade of myelosuppression)
7. **Focus on abnormal indicators, normal indicators can be omitted**
Output as a concise, clear list. If information is missing, state \"Unknown\". Output the result directly.

**Medical history to extract from:**
---
{text_for_extraction}
---

**Structured Key Information Summary:**"""
        try:
            messages = [{"role": "user", "content": extraction_prompt}]
            extracted_info = self._call_llm_with_retry(messages, max_tokens=1536, temperature=0.0, task_name="Key Info Extraction")
            if not extracted_info or len(extracted_info) < 20:
                logging.warning(f"Extracted info seems too short: '{extracted_info}'. Returning a fallback message.")
                return "Key information extraction insufficient or failed."
            return extracted_info
        except Exception as e:
            logging.error(f"Final error during key info extraction: {e}")
            return "A critical error occurred during key information extraction."

    def _summarize_text(self, text_to_summarize: str, target_token_count: int) -> str:
        """使用LLM对长文本进行摘要，主要用于压缩历史记录。"""
        if not text_to_summarize or not text_to_summarize.strip(): return "Original text is empty."

        approx_char_count = int(target_token_count * 1.3) # 估算目标字符数
        target_length_instruction = f"Please strictly control the summary length to about {approx_char_count} characters."
        
        summary_max_tokens = target_token_count + 200 # 为摘要输出留出余量
        summary_max_tokens = min(summary_max_tokens, 4090) # 限制最大输出
        
        max_summarization_input_tokens = INPUT_TOKEN_THRESHOLD_GENERATION - summary_max_tokens - 500
        input_tokens = self._estimate_tokens(text_to_summarize)
        text_for_summarization = text_to_summarize
        
        if input_tokens > max_summarization_input_tokens:
            logging.warning(f"Text for summarization is too long ({input_tokens} > {max_summarization_input_tokens}). Truncating.")
            cutoff_chars = int(len(text_to_summarize) * max_summarization_input_tokens / (input_tokens + 1e-6) * 0.9)
            text_for_summarization = text_to_summarize[:cutoff_chars]
        
        summarization_prompt = f"""Please concisely and accurately summarize the following clinical record text, retaining the core clinical findings, diagnoses, treatments, and outcomes,but only cancer information saves. {target_length_instruction}\n\n**Text to summarize:**\n---\n{text_for_summarization}\n---\n\n**Concise Summary:**"""
        try:
            messages = [{"role": "user", "content": summarization_prompt}]
            summary = self._call_llm_with_retry(messages, max_tokens=summary_max_tokens, temperature=0.2, task_name="Text Summarization")
            if not summary or len(summary) < 10:
                logging.warning(f"Summarization result is too short: '{summary}'.")
                return f"Summary insufficient. Original text snippet:\n{text_to_summarize[:500]}..."
            return summary
        except Exception as e:
            logging.error(f"Final error during text summarization: {e}")
            fallback_len = approx_char_count if approx_char_count > 0 else 500
            return f"Text summarization failed. Original text snippet:\n{text_to_summarize[:fallback_len]}..."

    def _compress_current_record(self, record: str, target_char_count: int = 5500) -> str:
        """(可选) 使用LLM压缩当前记录，以减少token占用。"""
        if not record or not record.strip(): return ""
        
        current_tokens = self._estimate_tokens(record)
        # 估算字符数是否需要压缩，给一个宽泛的检查
        if len(record) < target_char_count + 1000:
            return record
            
        compression_prompt = f"""Please losslessly compress the following current admission record, retaining all key clinical information about cancer. Compression requirements:
1. **Must retain** all important clinical findings, test results, treatments, and assessment results.
2. **Must retain** all numeric indicators (e.g., lab results, imaging measurements, etc.).
3. **Must retain** all adverse reactions and complications.
4. Redundant, polite, non-medical, and cancer-unrelated text can be removed.
5. Normal or negative results can be briefly summarized (e.g., "CBC shows no significant abnormalities").
6. Ensure the compressed content is as concise as possible without losing any information required for clinical decision-making.

**Current admission record to compress:**
---
{record}
---

**Compressed Record:**"""
        try:
            messages = [{"role": "user", "content": compression_prompt}]
            # max_tokens可以宽松一些，因为模型会尽量简短
            compressed_record = self._call_llm_with_retry(messages, max_tokens=4096, temperature=0.1, task_name="Current Record Compression")
            
            if not compressed_record or len(compressed_record) < 100:
                logging.warning("Compression result is too short, returning original record.")
                return record
            logging.info(f"Successfully compressed current record from {len(record)} to {len(compressed_record)} characters.")
            return compressed_record
        except Exception as e:
            logging.error(f"Error compressing current record, returning original: {e}")
            return record

    def _construct_pathway_prompt(
        self,
        record_content: str,
        structured_history_summary: str,
        context: str,
        guideline_info: str = ""
    ) -> str:
        """
        构建统一的、采用思维链方法的Prompt。
        该函数根据 self.use_kg 的状态，动态地构建外部知识源部分。
        """
        # --- 1. 根据 self.use_kg 开关，动态构建知识源部分 ---
        if self.use_kg:
            # KG模式：知识源是精确的指南图谱和RAG上下文的结合
            knowledge_source_title = "External Knowledge Source (Guideline Knowledge & Clinical Reference)"
            # 合并指南信息和RAG上下文
            knowledge_source_content = f"--- Guideline Recommendations ---\n{guideline_info if guideline_info.strip() else 'No'}\n\n--- Relevant Clinical Pathway Reference Content ---\n{context if context.strip() else 'No'}"
        else:
            # 无KG模式：知识源仅为RAG检索的通用参考内容
            knowledge_source_title = "External Knowledge Source (Clinical Pathway Reference)"
            knowledge_source_content = context

        knowledge_source_section = f"""
    ### {knowledge_source_title}
    ```text
    {knowledge_source_content}
    ```
    """

        # --- 2. 定义固定的患者信息部分 ---
        patient_record_section = f"""
    ### Patient Medical History Record (Fact Base)
    #### Structured Key Information Summary (Patient History Core Information)
    ```text
    {structured_history_summary}
    ```
    #### Current Visit Record (Patient Current Visit Situation)
    ```text
    {record_content}
    ```
    """

        # --- 3. 构建统一的思维链Prompt框架 ---
        final_prompt = f"""# Role and Goal
- **Role**: You are a top-notch clinical oncology expert tasked with generating a structured, evidence-based, and highly feasible clinical pathway for a specific patient. Your reasoning process must be clear and transparent, tightly integrating the patient's unique history with the standard clinical workflow defined by clinical guidelines.

# Input Information (Information)
{patient_record_section}
{knowledge_source_section}

# Step-by-Step Instructions (Chain of Thought)
Please strictly follow the following thought steps to build the clinical pathway:

**Step 1: Patient Status Assessment and Knowledge Mapping**
- Deeply analyze `Patient Medical History Record`. Map the patient's actual situation to concepts and recommendations in `External Knowledge Source`, summarizing the patient's current clinical status. Identify and list key clinical findings related to cancer (e.g., diagnosis, staging, biomarkers, recent treatment, important laboratory results).

**Step 2: Difference Analysis (Patient Reality vs. Knowledge Standard)**
- Compare the operations executed in `Patient Medical History Record` with recommendations in `External Knowledge Source`. Clearly identify and list:
    - **`Consistent Behavior`**: Which operations in the record are consistent with recommendations in `External Knowledge Source`?
    - **`Deviant Behavior`**: Which operations in the record are not consistent with or have differences from recommendations in `External Knowledge Source`?
    - **`Critical Missing Information`**: What information is required by `External Knowledge Source` but missing in `Patient Medical History Record`?

**Step 3: Generate Structured Clinical Pathway**
- Based on the evaluations and analyses from Steps 1 and 2, generate the final clinical pathway. The pathway must be forward-looking, focusing on subsequent steps and long-term planning.
- For each recommendation, it must be briefly explained **based on** the content of `External Knowledge Source`.
- If there is missing critical information, the pathway must first indicate the check required to obtain that information, then create **[Condition]** branches based on possible check results.

# Output Format (Strict Mode)
Please output in structured Markdown format. Do not use any subjective, dialogical meta-language (e.g., "Based on the provided information, I believe..."). Please present results directly and objectively.

### Clinical Pathway: [Disease Name] ([Patient Current Status])

#### 1. Clinical Assessment Summary
- **Diagnosis**: [Should be related to cancer,only one Diagnosis, e.g., rectal cancer, postoperative]
- **Guideline Mapping Staging/Current Stage**: [e.g., adjuvant chemotherapy stage, based on high-risk II stage definition in External Knowledge Source]
- **Key Findings**:
    - - [Key Finding 1 from `Patient Medical History Record`]
    - - [Key Finding 2 from `Patient Medical History Record`]
- **Consistent Behavior**:
    - - [Consistent Behavior 1 in the record]
- **Deviant Behavior**:
    - - [Deviant Behavior 1 in the record]
- **Critical Missing Information**:
    - - [Critical Information Missing in Record but Required by External Knowledge Source 1]
    - - [Critical Information Missing in Record but Required by External Knowledge Source 2]

#### 2. Clinical Pathway Recommendations
**A. Recent Actions and Assessments (Current Treatment Cycle)**
- **[Action]** [e.g., Obtain pathological report to clarify TNM staging.]
    - **[Based on]** [e.g., According to External Knowledge Source Section 3.1, TNM staging is the basis for determining whether adjuvant chemotherapy is needed and for selecting treatment options.]
- **[Decision Point]** [e.g., Based on neutrophil count in CBC.]
    - **[Condition]** If grade 3/4 neutrophil reduction: [Executed Action, e.g., Suspend chemotherapy, give G-CSF support treatment.]
    - **[Condition]** If grade 0-2 neutrophil reduction: [Executed Action, e.g., Execute this cycle chemotherapy as planned.]

**B. Subsequent Treatment Plan**
- ...

**C. Long-term Monitoring and Follow-up**
- ...
"""
        return final_prompt

    def _extract_diagnosis(self, record: str) -> Optional[str]:
        """从记录中简单提取诊断信息用于RAG查询。"""
        try:
            # 匹配 "诊断:" 或 "临床诊断:" 后面的内容
            match = re.search(r"(?:诊断|临床诊断)\s*:\s*([^\n]+)", record, re.IGNORECASE)
            if match:
                return match.group(1).strip().replace("？", "").replace("术后", "").strip()
            return None
        except Exception:
            return None

    def _split_history_and_current(self, text: str) -> Tuple[str, str, str]:
        """
        根据新的格式分割历史记录、当前记录和Guideline Recommendations。
        - `--- Latest Record (Uncompressed) ---` 分割历史与当前。
        - `=== Guideline Recommendations ===` 分割当前与指南。
        """
        history_text = ""
        current_record = ""
        guideline_info = ""

        # 首先，用最新记录的分隔符分割整个文本
        parts = text.split("--- Latest Record (Uncompressed) ---")
        if len(parts) == 2:
            # 分隔符前是历史记录
            history_text = parts[0].strip()
            # 分隔符后是最新记录和可能的指南信息
            after_latest_part = parts[1]
            
            # 再次分割，提取指南信息
            guideline_parts = after_latest_part.split("=== Guideline Recommendations ===")
            if len(guideline_parts) == 2:
                current_record = guideline_parts[0].strip()
                guideline_info = guideline_parts[1].strip()
            else:
                current_record = after_latest_part.strip()
        else:
            # 如果没有找到最新记录的分隔符，则将所有文本视为当前记录
            logging.warning("Separator '--- Latest Record (Uncompressed) ---' not found. Treating entire text as current record.")
            current_record = text.strip()
            
        return history_text, current_record, guideline_info

    def _chunk_and_summarize(self, text: str, target_token_count: int) -> str:
        """当文本过长时，将其分块、摘要、再合并，用于压缩历史记录。"""
        if not text or not text.strip(): return ""
        
        # 设定每个块的目标大小，这里设为目标总Token数的一半
        chunk_target_tokens = target_token_count // 2
        text_tokens = self._estimate_tokens(text)
        
        # 如果文本本身没有那么长，直接摘要
        if text_tokens <= chunk_target_tokens:
            return self._summarize_text(text, target_token_count)
        
        chunks = []
        current_chunk_lines = []
        current_tokens = 0
        
        for line in text.split('\n'):
            line_tokens = self._estimate_tokens(line)
            if current_tokens + line_tokens > chunk_target_tokens and current_chunk_lines:
                chunks.append('\n'.join(current_chunk_lines))
                current_chunk_lines = [line]
                current_tokens = line_tokens
            else:
                current_chunk_lines.append(line)
                current_tokens += line_tokens
        
        if current_chunk_lines:
            chunks.append('\n'.join(current_chunk_lines))
        
        summarized_chunks = []
        logging.info(f"Splitting long history into {len(chunks)} chunks for summarization.")
        for i, chunk in enumerate(chunks):
            # 对每个块进行摘要，目标Token数设为块的Token数的一半
            chunk_summary_target = self._estimate_tokens(chunk) // 2
            summary = self._summarize_text(chunk, chunk_summary_target)
            if summary and "Summary failed" not in summary and "Summary insufficient" not in summary:
                summarized_chunks.append(f"--- History Fragment {i+1} Summary ---\n{summary}")
        
        return "\n\n".join(summarized_chunks)

    def process_patient_record(self, record_text: str) -> Dict[str, Any]:
        """
        处理单个患者的完整记录文本，生成临床路径。
        这是核心的单文件处理逻辑。
        """
        try:
            # 1. 分割文本
            history_text, current_record, guideline_info = self._split_history_and_current(record_text)

            # 2. (可选) 压缩当前记录
            if self.compress_latest:
                logging.info("Compression for latest record is enabled. Compressing...")
                current_record = self._compress_current_record(current_record)
            else:
                logging.info("Compression for latest record is disabled.")

            # 3. 根据开关处理历史记录
            structured_info = ""
            if self.extract_history:
                logging.info("History summary extraction is ENABLED. Using LLM to process history.")
                structured_info = self._extract_key_info(history_text)
            else:
                logging.info("History summary extraction is DISABLED. Using raw history text as summary.")
                structured_info = history_text  # 直接使用原始历史文本

            # 4. 获取RAG上下文
            diagnosis_for_query = self._extract_diagnosis(current_record)
            query_for_context = diagnosis_for_query if diagnosis_for_query else current_record[:500]
            rag_context = self._get_relevant_context(query_for_context, k=2)
            
            # 5. 构建初始Prompt并检查Token
            initial_prompt = self._construct_pathway_prompt(
                current_record, structured_info, rag_context, guideline_info
            )
            estimated_input_tokens = self._estimate_tokens(initial_prompt)
            
            final_prompt_for_generation = initial_prompt
            # 6. 如果超长，则启动压缩/截断逻辑
            if estimated_input_tokens > INPUT_TOKEN_THRESHOLD_GENERATION:
                logging.warning(f"Initial prompt tokens ({estimated_input_tokens}) exceed threshold ({INPUT_TOKEN_THRESHOLD_GENERATION}). Compressing/truncating components...")
                
                # 为各个部分分配Token预算
                total_available_tokens = INPUT_TOKEN_THRESHOLD_GENERATION
                current_record_budget = int(total_available_tokens * 0.5)
                history_budget = int(total_available_tokens * 0.2)
                guideline_budget = int(total_available_tokens * 0.15)
                rag_budget = int(total_available_tokens * 0.15)

                # 根据开关处理超长的历史记录
                if self.extract_history:
                    # 如果启用提取，但历史仍然太长，则压缩它再重新提取
                    if self._estimate_tokens(history_text) > history_budget:
                        compressed_history = self._chunk_and_summarize(history_text, history_budget)
                        structured_info = self._extract_key_info(compressed_history)
                else:
                    # 如果禁用提取，structured_info就是原始历史，直接截断
                    if self._estimate_tokens(structured_info) > history_budget:
                        structured_info = self._truncate_text_by_token(structured_info, history_budget)

                # 截断RAG上下文
                if self._estimate_tokens(rag_context) > rag_budget:
                    rag_context = self._truncate_text_by_token(rag_context, rag_budget)
                
                # 截断指南信息
                if self._estimate_tokens(guideline_info) > guideline_budget:
                    guideline_info = self._truncate_text_by_token(guideline_info, guideline_budget)

                # 重新构建最终的Prompt
                final_prompt_for_generation = self._construct_pathway_prompt(
                    current_record, structured_info, rag_context, guideline_info
                )

            # 7. 调用LLM生成临床路径
            messages = [{"role": "user", "content": final_prompt_for_generation}]
            generated_pathway = self._call_llm_with_retry(messages, REQUESTED_OUTPUT_TOKENS, 0.1, "Pathway Generation")
            
            if not generated_pathway or "Pathway generation failed" in generated_pathway or len(generated_pathway) < 50:
                raise ValueError("Pathway generation failed or the result was invalid.")
            
            logging.info("Pathway generated successfully.")
            return {
                "status": "success",
                "structured_info": structured_info,
                "record_content_used": current_record,
                "rag_context": rag_context,
                "guideline_info": guideline_info,
                "generated_pathway": generated_pathway,
                "error": None
            }
        except Exception as e:
            logging.error(f"Critical error in process_patient_record: {e}", exc_info=True)
            return {"status": "error", "error": f"Critical error occurred in process_patient_record: {str(e)}"}

    def _truncate_text_by_token(self, text: str, max_tokens: int) -> str:
        """按token数截断文本"""
        if self._estimate_tokens(text) <= max_tokens:
            return text
        
        # 估算截断位置
        current_tokens = self._estimate_tokens(text)
        cutoff_chars = int(len(text) * max_tokens / (current_tokens + 1e-6) * 0.9)
        return text[:cutoff_chars] + "...\n[Content Truncated]"
        
    def _process_single_json(self, file_path: str):
        """处理单个JSON文件的完整流程：读取 -> 处理 -> 保存。"""
        all_records_output_list = []
        try:
            # 文件预检查
            if os.path.getsize(file_path) == 0:
                logging.warning(f"Skipping empty file (0 bytes): {os.path.basename(file_path)}")
                return
                
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    full_text = data.get("text", "")
                except json.JSONDecodeError as je:
                    logging.error(f"JSON decode error in {file_path}, reading as raw text: {je}")
                    f.seek(0)
                    full_text = f.read()

            if not full_text or len(full_text.strip()) < MIN_RECORD_LENGTH_FOR_QUALITY:
                logging.warning(f"Skipping file due to insufficient content length: {os.path.basename(file_path)}")
                return

            # 核心处理
            result = self.process_patient_record(full_text)
            
            if result["status"] == "success":
                # 构建用于微调的 Alpaca 格式数据
                knowledge_part = ""
                if self.use_kg:
                    knowledge_part = f"Guideline Recommendations:\n{result['guideline_info']}\n\n---\nRelevant Clinical Pathway Reference:\n{result['rag_context']}"
                else:
                    knowledge_part = f"Relevant Clinical Pathway Reference:\n{result['rag_context']}"
                
                input_for_tuning = f"Structured Key Information Summary:\n{result['structured_info']}\n\n---\nCurrent Visit Record:\n{result['record_content_used']}\n\n---\n{knowledge_part}"
                
                # 如果微调输入过长，进行截断
                tuning_input_tokens = self._estimate_tokens(input_for_tuning)
                if tuning_input_tokens > FINETUNE_INPUT_TARGET_MAX_TOKENS:
                    logging.warning(f"Final fine-tuning 'input' is too long ({tuning_input_tokens} tokens). Truncating...")
                    input_for_tuning = self._truncate_text_by_token(input_for_tuning, FINETUNE_INPUT_TARGET_MAX_TOKENS)
                
                alpaca_entry = {
                    "instruction": "Generate a clinical pathway based on the following patient information and relevant guidelines.",
                    "input": input_for_tuning,
                    "output": result["generated_pathway"],
                    "system": "You are a professional clinical oncology expert. You need to develop or refine the patient's subsequent clinical pathway based on the provided medical information and clinical guidelines.",
                    "history": [],
                    "metadata": {
                        "source_file": os.path.basename(file_path),
                        "kg_mode": self.use_kg,
                        "compression_used": self.compress_latest,
                        "history_extraction_used": self.extract_history
                    }
                }
                all_records_output_list.append(alpaca_entry)
            
            # 保存结果
            if all_records_output_list:
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                output_filename = base_name + OUTPUT_FILENAME_SUFFIX
                output_path = os.path.join(self.output_dir, output_filename)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(all_records_output_list, f, ensure_ascii=False, indent=2)
                logging.info(f"Successfully processed and saved to: {output_path}")

        except Exception as e:
            logging.error(f"An unhandled error occurred while processing file {os.path.basename(file_path)}: {e}", exc_info=True)

    def process_json_files(self, max_workers: Optional[int] = None):
        """使用线程池并行处理目录中的所有JSON文件，支持进度条和断点续处理。"""
        if not os.path.exists(self.json_dir) or not os.path.isdir(self.json_dir):
            logging.error(f"JSON source directory not found: {self.json_dir}"); return
        
        # 确保向量库已准备就绪
        if self.embeddings is None or self.vector_store is None:
            logging.info("Embedding/vector store not ready. Initializing it first.")
            self.process_txt_files()
            if self.embeddings is None or self.vector_store is None:
                logging.error("Failed to initialize embedding/vector store. Aborting processing.")
                return
            logging.info("Embedding/vector store initialized successfully.")

        # 断点续处理：扫描输出目录，获取已处理的文件列表
        processed_basenames = set()
        if os.path.exists(self.output_dir):
            try:
                for filename in os.listdir(self.output_dir):
                    if filename.endswith(OUTPUT_FILENAME_SUFFIX):
                        base_name = filename.replace(OUTPUT_FILENAME_SUFFIX, '')
                        processed_basenames.add(base_name)
                if processed_basenames:
                    logging.info(f"Found {len(processed_basenames)} already processed files in {self.output_dir}. They will be skipped.")
            except OSError as list_err:
                logging.error(f"Cannot list output directory {self.output_dir}: {list_err}. Proceeding without checking existing files.")
        else:
            logging.info(f"Output directory {self.output_dir} not found. Will be created.")


        # 收集所有源文件并过滤掉已处理的
        all_source_files = [os.path.join(root, file) for root, _, files in os.walk(self.json_dir) for file in files if file.endswith('.json')]
        files_to_process = [fp for fp in all_source_files if os.path.splitext(os.path.basename(fp))[0] not in processed_basenames]

        if not files_to_process:
            logging.info("All source JSON files seem to have been processed already. Nothing to do.")
            return

        logging.info(f"Found {len(all_source_files)} total source JSON files. Starting processing for {len(files_to_process)} remaining files...")

        # 设置并行工作线程数
        if max_workers is None:
            cpu_cores = os.cpu_count() or 1
            max_workers = min(max(1, cpu_cores - 2), max_worker) # 保留一些核心给系统，最多16个
            logging.info(f"Using default max_workers: {max_workers}")
        
        processed_count = 0
        error_count = 0
        start_time = time.time()

        # 使用线程池执行任务
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='RecordWorker') as executor:
            future_to_file = {executor.submit(self._process_single_json, fp): fp for fp in files_to_process}

            # 使用tqdm显示进度条
            for future in tqdm(concurrent.futures.as_completed(future_to_file),
                                total=len(files_to_process),
                                desc="Processing JSON files"):
                file_path = future_to_file[future]
                try:
                    future.result()  # 获取结果，如果线程中有异常会在这里抛出
                    processed_count += 1
                except Exception as exc:
                    logging.error(f"Worker thread for file '{os.path.basename(file_path)}' raised an exception: {exc}", exc_info=False)
                    error_count += 1

        total_time = time.time() - start_time
        logging.info(f"JSON file processing finished in {total_time:.2f} seconds.")
        logging.info(f"Files processed in this run: {processed_count}, Errors in this run: {error_count}.")
        logging.info(f"Total source files found: {len(all_source_files)}, Skipped (already processed): {len(all_source_files) - len(files_to_process)}")


# --- 主执行逻辑 ---
if __name__ == "__main__":
    # 路径和文件存在性检查
    if not os.path.exists(CLINICAL_PATH_DIR):
        logging.error(f"Clinical pathway directory not found: {CLINICAL_PATH_DIR}"); sys.exit(1)
    if not os.path.exists(JSON_DIR):
        logging.error(f"Source records directory not found: {JSON_DIR}"); sys.exit(1)

    try:
        # 确保输出目录存在
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            logging.info(f"Created output directory: {OUTPUT_DIR}")

        # 实例化并运行处理器
        processor = ClinicalPathwayProcessorWhole(
            api_key=API_KEY,
            base_url=BASE_URL,
            clinical_path_dir=CLINICAL_PATH_DIR,
            json_dir=JSON_DIR,
            use_kg=args.use_kg,
            compress_latest=args.compress_latest,
            extract_history=args.extract_history,
            output_dir=OUTPUT_DIR
        )

        # 启动处理流程
        processor.process_json_files(max_workers=16)

        logging.info("All processing finished.")
    except Exception as main_err:
        logging.critical(f"A critical error occurred in the main execution block: {main_err}", exc_info=True)
        sys.exit(1)


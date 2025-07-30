# -*- coding: utf-8 -*-
import os
import json
import logging
import re
import time
from tqdm import tqdm
import collections
import concurrent.futures
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

try:
    from openai import OpenAI, RateLimitError, APIError, APITimeoutError
except ImportError as e:
    logging.error(f"OpenAI库未找到，请先安装: {e}。")
    raise

# --- 配置信息 ---
# 使用线程安全的日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

# --- 路径与字段配置 ---
# NEW: 将输入目录配置为一个列表，每个元素是一个字典
# name: 模型的简称，会出现在报告中
# path: 包含 generated_predictions.jsonl 文件的目录路径
# prediction_field: 文件中代表模型输出的字段名

INPUT_CONFIGS = [
    {
        "name": "Standard_RAG_8B",
        "path": "/home/admin1/data/Predict_Result_NoFilter/MIMIC_fixed/Qwen3-8B",
        "prediction_field": "predict"
    },
    {
        "name": "BriefContext_8B",
        "path": "/home/admin1/data/Predict_Results/MIMIC_fixed/2.Comparative_RAG/QA_MIMIC_BriefContext_eval",
        "prediction_field": "predict"
    },
    {
        "name": "MedRAG_8B",
        "path": "/home/admin1/data/Predict_Results/MIMIC_fixed/2.Comparative_RAG/QA_MIMIC_MedRAG_eval",
        "prediction_field": "predict"
    },
    {
        "name": "KG2RAG_8B",
        "path": "/home/admin1/data/Predict_Results/MIMIC_fixed/2.Comparative_RAG/QA_MIMIC_KG2RAG_eval",
        "prediction_field": "predict"
    },
    {
        "name": "GNN-RAG_8B",
        "path": "/home/admin1/data/Predict_Results/MIMIC_fixed/2.Comparative_RAG/QA_MIMIC_GNNRAG_eval",
        "prediction_field": "predict"
    },
    {
        "name": "CliPAGE_8B",
        "path": "/home/admin1/data/Predict_Results/MIMIC_fixed/3.CliPAGE/QA_MIMIC_KG_Align_eval",
        "prediction_field": "predict"
    },
    {
        "name": "Standard_RAG",
        "path": "/home/admin1/data/Predict_Result_NoFilter/MIMIC_fixed/Qwen3-8B",
        "prediction_field": "label"
    },
    {
        "name": "BriefContext",
        "path": "/home/admin1/data/Predict_Results/MIMIC_fixed/2.Comparative_RAG/QA_MIMIC_BriefContext_eval",
        "prediction_field": "label"
    },
    {
        "name": "MedRAG",
        "path": "/home/admin1/data/Predict_Results/MIMIC_fixed/2.Comparative_RAG/QA_MIMIC_MedRAG_eval",
        "prediction_field": "label"
    },
    {
        "name": "KG2RAG",
        "path": "/home/admin1/data/Predict_Results/MIMIC_fixed/2.Comparative_RAG/QA_MIMIC_KG2RAG_eval",
        "prediction_field": "label"
    },
    {
        "name": "GNN-RAG",
        "path": "/home/admin1/data/Predict_Results/MIMIC_fixed/2.Comparative_RAG/QA_MIMIC_GNNRAG_eval",
        "prediction_field": "label"
    },
    {
        "name": "CliPAGE_test",
        "path": "/home/admin1/data/Predict_Results/MIMIC_fixed/3.CliPAGE/QA_MIMIC_KG_Align_eval",
        "prediction_field": "label"
    },
    {
        "name": "CliPAGE",
        "path": "/home/admin1/data/Predict_Results/MIMIC_fixed/3.CliPAGE/QA_MIMIC_KG_Align_eval",
        "prediction_field": "label"
    },
    {
        "name": "Mistral-v0.1-7B",
        "path": "/home/admin1/data/Predict_Result_NoFilter/MIMIC_fixed/Mistral-7B-v0.1",
        "prediction_field": "predict"
    },
    {
        "name": "Mistral-Instruct-v0.1-7B",
        "path": "/home/admin1/data/Predict_Result_NoFilter/MIMIC_fixed/Mistral-7B-Instruct-v0.1",
        "prediction_field": "predict"
    },
    {
        "name": "Biomistral-7B",
        "path": "/home/admin1/data/Predict_Result_NoFilter/MIMIC_fixed/BioMistral-7B",
        "prediction_field": "predict"
    },
    {
        "name": "GPT-4.1",
        "path": "/home/admin1/data/Predict_Results_API/MIMIC_fixed/QA_MIMIC_NoFilter_GPT",
        "prediction_field": "label"
    },
    {
        "name": "Deepseek-R1",
        "path": "/home/admin1/data/Predict_Results_API/MIMIC_fixed/QA_MIMIC_NoFilter_Deepseek",
        "prediction_field": "label"
    },
    {
        "name": "Claude-4.0-Sonnet",
        "path": "/home/admin1/data/Predict_Results_API/MIMIC_fixed/QA_MIMIC_NoFilter_Claude",
        "prediction_field": "label"
    },
    {
        "name": "Mistral-v0.1-7B_CliPAGE",
        "path": "/home/admin1/data/Predict_Results_BaseModel_Align/MIMIC_fixed/Mistral-7B-v0.1",
        "prediction_field": "predict"
    },
    {
        "name": "Mistral-Instruct-v0.1-7B_CliPAGE",
        "path": "/home/admin1/data/Predict_Results_BaseModel_Align/MIMIC_fixed/Mistral-7B-Instruct-v0.1",
        "prediction_field": "predict"
    },
    {
        "name": "Biomistral-7B_CliPAGE",
        "path": "/home/admin1/data/Predict_Results_BaseModel_Align/MIMIC_fixed/BioMistral-7B",
        "prediction_field": "predict"
    },
    {
        "name": "GPT-4.1_CliPAGE",
        "path": "/home/admin1/data/Predict_Results_API/MIMIC_fixed/QA_MIMIC_Align_GPT",
        "prediction_field": "label"
    },
    {
        "name": "Deepseek-R1_CliPAGE",
        "path": "/home/admin1/data/Predict_Results_API/MIMIC_fixed/QA_MIMIC_Align_Deepseek",
        "prediction_field": "label"
    },
    {
        "name": "Claude-4.0-Sonnet_CliPAGE",
        "path": "/home/admin1/data/Predict_Results_API/MIMIC_fixed/QA_MIMIC_Align_Claude",
        "prediction_field": "label"
    },

    {
        "name": "wo_Alignment_Expansion(Qwen-3-8B)",
        "path": "/home/admin1/data/Predict_Results/MIMIC_fixed/4.Ablation_data/QA_MIMIC_data_KG_Align_TKG_LLM_eval",
        "prediction_field": "predict"
    },
    {
        "name": "wo_LLM-based_Reranking(Qwen-3-8B)",
        "path": "/home/admin1/data/Predict_Results/MIMIC_fixed/4.Ablation_data/QA_MIMIC_data_KG_Align_TKG_NOLLM_eval",
        "prediction_field": "predict"
    },
    {
        "name": "wo_Alignment_Expansion(Gemini-2.5-Pro)",
        "path": "/home/admin1/data/Predict_Results/MIMIC_fixed/4.Ablation_data/QA_MIMIC_data_KG_Align_TKG_LLM_eval",
        "prediction_field": "label"
    },
    {
        "name": "wo_LLM-based_Reranking(Gemini-2.5-Pro)",
        "path": "/home/admin1/data/Predict_Results/MIMIC_fixed/4.Ablation_data/QA_MIMIC_data_KG_Align_TKG_NOLLM_eval",
        "prediction_field": "label"
    },
]




# 评估结果的输出目录
EVALUATION_RESULTS_DIR = "/home/admin1/data/evaluation_results/compare_MIMIC_0.95_compare_2Q_tab_all_0723_test"
# 预测文件名（假设所有目录下此文件名都相同）
PREDICTION_FILENAME = "generated_predictions.jsonl"

# --- 命名与文件路径配置 ---
# NEW: 根据配置中的模型名称自动生成用于输出文件的字符串
# model_names_str = "_vs_".join([config['name'] for config in INPUT_CONFIGS])
model_names_str = 'all'
DETAILED_EVALUATION_FILE = os.path.join(EVALUATION_RESULTS_DIR, f"detailed_eval_{model_names_str}.jsonl")
SUMMARY_REPORT_FILE = os.path.join(EVALUATION_RESULTS_DIR, f"summary_report_{model_names_str}.txt")

# --- API与模型配置 ---
API_KEY = "sk-XXX"  # 请替换为您的有效API密钥
BASE_URL = "https://www.chataiapi.com/v1"
EVALUATOR_MODEL_NAME = "gemini-2.5-pro-preview-03-25"
# API_KEY = "sk-d46680ddd44d4023a62affcc8d8252a0"  # 请替换为您的有效API密钥
# BASE_URL = "https://api.deepseek.com"
# EVALUATOR_MODEL_NAME = "deepseek-reasoner"
MAX_TOKENS_EVAL = 15360  # 为评估任务设置的token上限
# NEW: 指定 patient_context 的来源，0代表使用第一个INPUT_CONFIGS中的数据
CONTEXT_SOURCE_INDEX = 11
# NEW: 是否在 patient_context 中包含RAG检索到的内容
RAG_APPLIED = False

# --- 并发配置 ---
MAX_WORKERS = 64  # 并发线程数

# ==========================================================
# === 评估维度定义 (与之前保持一致)                  ===
# ==========================================================
EVALUATION_ATTRIBUTES = [
    "clinical_pathway_summary_evaluation",      # 临床路径总结评估
    "clinical_pathway_recommendation_evaluation"  # 临床路径推荐评估
]

class PathwayEvaluator:
    """
    已修改: 一个执行多模型对比评估的评估器，能够为每个模型打分。
    评估Prompt已更新为四维度评估体系。
    """
    def __init__(self, api_key: str, base_url: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=180.0)
        self.eval_model_name = EVALUATOR_MODEL_NAME
        logging.info(f"评估器客户端已初始化，使用模型: {self.eval_model_name}")

        self.attention_guiding_header = """You are a senior oncology expert. Your task is to act as a human expert evaluator, conducting a side-by-side comparative evaluation of outputs from multiple models. Your assessment will be based on a four-dimension rubric: Factual Accuracy, Completeness & Thoroughness, Clinical Soundness, and Actionability & Relevance. Please provide a comprehensive evaluation for each model and assign scores."""

        self.eval_attributes_prompts = {
            "clinical_pathway_summary_evaluation": """**Evaluation Task: Clinical Pathway Summary Quality**
Your task is to evaluate the overall quality of the "Clinical Pathway Summary" generated by each model. Which summary provides higher quality and more reliable support for clinical decision-makers?""",
            "clinical_pathway_recommendation_evaluation": """**Evaluation Task: Clinical Pathway Recommendation Quality**
Your task is to evaluate the overall quality of the "Clinical Pathway Recommendation" generated by each model. Which recommendation has greater clinical value, is safer, and is more actionable?"""
        }

    def _call_evaluator_llm(self, full_prompt: str, task_name: str) -> str:
        """通用的LLM调用函数，包含重试逻辑。"""
        max_retries = 3
        retry_delay = 10
        for attempt in range(max_retries):
            try:
                messages = [{"role": "user", "content": full_prompt}]
                response = self.client.chat.completions.create(
                    model=self.eval_model_name, messages=messages, temperature=0.0, max_tokens=MAX_TOKENS_EVAL
                )
                if response.choices and response.choices[0].message and response.choices[0].message.content:
                    content = response.choices[0].message.content.strip()
                    if content: return content
                logging.warning(f"评估器API对任务 {task_name} 返回了无效或空的响应 (尝试 {attempt+1}).")
                if attempt < max_retries - 1: time.sleep(retry_delay * (attempt + 1))
            except Exception as e:
                logging.error(f"任务 {task_name} 的API调用出错: {e}", exc_info=True)
                if attempt < max_retries - 1: time.sleep(retry_delay * (attempt + 1))
                else: return f"API_ERROR: {str(e)}"
        return f"API在{max_retries}次尝试后失败"

    def _parse_multi_eval_response(self, response_text: str, model_names: List[str]) -> Dict[str, Any]:
        """解析包含各模型分数的响应。"""
        parsed_data = {
            "scores": {},
            "justification": response_text  # 默认值为完整响应
        }
        try:
            # 分数格式: Scores: [Model A: 3, Model B: 5]
            scores_match = re.search(r"Scores\s*[:：]\s*\[(.*?)\]", response_text, re.DOTALL)
            if scores_match:
                scores_str = scores_match.group(1)
                for model_name in model_names:
                    model_score_match = re.search(rf"{re.escape(model_name)}\s*[:：]\s*([1-5])", scores_str)
                    if model_score_match:
                        parsed_data["scores"][model_name] = int(model_score_match.group(1))
            
            # 理由格式: Justification: ...
            reason_match = re.search(r"Justification\s*[:：]\s*(.*)", response_text, re.DOTALL | re.MULTILINE)
            if reason_match:
                parsed_data["justification"] = reason_match.group(1).strip()
            
            return parsed_data
        except Exception as e:
            logging.error(f"解析多模型评估响应时出错 '{response_text[:100]}...': {e}")
            return parsed_data

    def evaluate_item_set(self, instruction_text: str, input_text: str, predictions: Dict[str, str]) -> Dict[str, Any]:
        """对单个样本的一组预测（来自多个模型）进行评估。"""
        all_attributes_evaluation = {}
        patient_context = f"Instruction:\n{instruction_text}\n\nPatient Medical Record (Input):\n{input_text}"
        
        predictions_section = []
        model_names = sorted(predictions.keys())
        for name in model_names:
            pred_text = predictions[name]
            predictions_section.append(f"### Model {name}'s Output:\n```text\n{pred_text}\n```")
        
        predictions_block = "\n\n".join(predictions_section)

        for attr_name in EVALUATION_ATTRIBUTES:
            attribute_specific_prompt = self.eval_attributes_prompts[attr_name]
            
            full_eval_prompt = f"""{self.attention_guiding_header}

### Patient Context
```text
{patient_context}
```

---
### Model Outputs to Evaluate
{predictions_block}
---

### {attribute_specific_prompt}

**Evaluation Instructions:**
Please compare all model outputs based on the following four dimensions. For each model, think step-by-step through each dimension before providing your final scores and justification. This is a Chain-of-Thought process to ensure a thorough evaluation.

**Evaluation Dimensions (Detailed Rubric):**
1.  **Factual Accuracy**:
    - **5**: All key information is 100% accurate and verifiable against the patient context. No hallucinations or errors.
    - **3**: Contains minor errors in non-critical information or factual deviations that **do not affect final treatment decisions or patient safety**.
    - **1**: Contains any **major factual error that could affect treatment decisions or patient safety**.
2.  **Completeness & Thoroughness**:
    - **5**: Perfectly covers all critical aspects of the patient's situation, identifies all key data elements, and insightfully adds important potential risks.
    - **3**: Covers most core aspects and data elements but omits some minor details or has individual improper handling of data.
    - **1**: Seriously lacks core content, or seriously omits or misunderstands key core data elements.
3.  **Clinical Soundness**:
    - **5**: All conclusions and recommendations are robust, safe, and reflect the clinical prudence of a senior expert. They are implicitly or explicitly based on recognized clinical guidelines.
    - **3**: Core recommendations are reasonable, but may include some unimportant or slightly unusual minor suggestions, or some recommendations lack a clear evidence-based foundation.
    - **1**: Contains any recommendation that could **endanger patient safety**, clearly violates clinical common sense, or is based on incorrect citations.
4.  **Actionability & Relevance**:
    - **5**: Provides highly insightful, quantifiable, and personalized action plans that focus on solving the most urgent current problems.
    - **3**: Offers some actionable advice, but some key parts are too general, or recommendations are mixed with retrospective analysis not directly relevant to the immediate next steps.
    - **1**: Provides a list of invalid information with no guiding value, or the recommendations are entirely disconnected from the current core clinical problem.

**Your Task:**
After considering the above dimensions, please complete the following two steps for the provided models:

1.  **Provide Scores:** Assign an overall quality score from 1 to 5 to **each** model.
2.  **Provide Justification:** Briefly explain the key reasons for your scores, highlighting the main strengths and weaknesses of each model.

**Output Format (Strictly follow this):**
Scores: [Model A: 3, Model B: 5, Model C: 4, ...]
Justification: [Provide a concise rationale for your evaluation here.](just sample)
"""
            response_text = self._call_evaluator_llm(full_eval_prompt, f"multi_eval_{attr_name}")
            parsed_eval = self._parse_multi_eval_response(response_text, model_names)
            all_attributes_evaluation[attr_name] = parsed_eval

        return all_attributes_evaluation

def load_jsonl_data(file_path: str, prediction_field: str) -> List[Dict]:
    """从jsonl文件加载数据，并根据指定字段提取预测内容，同时保留Json_File和Node_Number。"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    json_line = json.loads(line)
                    item = {
                        "prompt": json_line.get("prompt", ""),
                        "predict_text": json_line.get(prediction_field, ""),
                        "Json_File": json_line.get("Json_File", None),
                        "Node_Number": json_line.get("Node_Number", None)
                    }
                    if not item["predict_text"]:
                        logging.warning(f"在文件 {file_path} 的某行中未找到预测字段 '{prediction_field}'。")
                    data.append(item)
    except FileNotFoundError:
        logging.error(f"预测文件未找到: {file_path}")
    except Exception as e:
        logging.error(f"读取或解析文件 {file_path} 时出错: {e}")
    return data

def save_jsonl_data(data: List[Dict], file_path: str):
    """将数据保存到jsonl文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
    except Exception as e:
        logging.error(f"保存到文件 {file_path} 时出错: {e}")

def get_full_context(pred_item: Dict) -> Tuple[str, str]:
    """从预测项中解析出指令和输入文本，并根据RAG_APPLIED标志选择性地移除RAG内容。"""
    full_prompt_str = pred_item.get("prompt", "")
    human_part_match = re.search(r"Human:\s*(.*)", full_prompt_str, re.DOTALL)
    if human_part_match:
        human_content = human_part_match.group(1).strip()
        parts = human_content.split("\n", 1)
        instruction_text = parts[0]
        input_text = parts[1] if len(parts) > 1 else ""
    else:
        instruction_text, input_text = "指令未能解析", full_prompt_str

    if not RAG_APPLIED:
        eng_marker = "Guideline Recommendations:"
        chn_marker = "指南推荐信息:"
        
        # 查找两个标记中较早出现的一个
        eng_pos = input_text.find(eng_marker)
        chn_pos = input_text.find(chn_marker)

        split_pos = -1
        if eng_pos != -1 and chn_pos != -1:
            split_pos = min(eng_pos, chn_pos)
        elif eng_pos != -1:
            split_pos = eng_pos
        elif chn_pos != -1:
            split_pos = chn_pos
            
        if split_pos != -1:
            # 截断字符串，移除RAG内容
            input_text = input_text[:split_pos]

    return instruction_text, input_text.strip()

def process_evaluation_set(args: Tuple) -> Optional[Dict]:
    """线程池的工作函数，处理单个样本（包含多模型预测）的评估"""
    evaluator, prompt_item, predictions_for_item, idx = args
    try:
        instruction_text, input_text = get_full_context(prompt_item)
        
        evaluations = evaluator.evaluate_item_set(
            instruction_text, input_text, predictions_for_item
        )

        return {
            "item_index": idx,
            "prompt_preview": instruction_text,
            "evaluations": evaluations
        }
    except Exception as e:
        logging.error(f"工作线程处理样本集 {idx} 时出错: {e}", exc_info=True)
        return None

def main():
    """主函数"""
    start_time = time.time()
    logging.info("脚本启动 - 多模型对比评估系统。")
    os.makedirs(EVALUATION_RESULTS_DIR, exist_ok=True)

    try:
        evaluator = PathwayEvaluator(api_key=API_KEY, base_url=BASE_URL)
    except Exception as init_e:
        logging.error(f"初始化评估器失败: {init_e}。程序退出。")
        return

    # 1. 读取所有模型预测，构建以 (Json_File, Node_Number) 为 key 的字典
    all_predictions_data = collections.defaultdict(dict)  # model_name -> {(Json_File, Node_Number): item}
    model_names = []
    for config in INPUT_CONFIGS:
        model_name = config["name"]
        model_names.append(model_name)
        file_path = os.path.join(config["path"], PREDICTION_FILENAME)
        data = load_jsonl_data(file_path, config["prediction_field"])
        if not data:
            logging.error(f"模型 '{model_name}' 的预测文件为空或无法读取。程序退出。")
            return
        for item in data:
            key = (item.get("Json_File"), item.get("Node_Number"))
            if key[0] is not None and key[1] is not None:
                all_predictions_data[model_name][key] = item
            else:
                logging.warning(f"模型 {model_name} 某条数据缺少 Json_File 或 Node_Number 字段，将跳过。")

    # 2. 取所有模型的 key 并集（原为交集）
    all_keys = [set(d.keys()) for d in all_predictions_data.values()]
    if not all_keys:
        logging.error("没有可用的预测数据。程序退出。")
        return
    union_keys = set.union(*all_keys)
    if not union_keys:
        logging.error("没有任何(Json_File, Node_Number)样本。程序退出。")
        return
    logging.info(f"所有模型共有 {len(union_keys)} 个样本。")

    final_evaluation_results_list = []
    if os.path.exists(DETAILED_EVALUATION_FILE):
        logging.info(f"详细评估文件已存在: {DETAILED_EVALUATION_FILE}。将加载现有结果。")
        with open(DETAILED_EVALUATION_FILE, 'r', encoding='utf-8') as f:
            final_evaluation_results_list = [json.loads(line) for line in f if line.strip()]
    else:
        # 3. 选择 context_source_model
        if not (0 <= CONTEXT_SOURCE_INDEX < len(INPUT_CONFIGS)):
            logging.error(f"CONTEXT_SOURCE_INDEX ({CONTEXT_SOURCE_INDEX}) is out of bounds for INPUT_CONFIGS (size: {len(INPUT_CONFIGS)}).")
            return
        context_model_name = INPUT_CONFIGS[CONTEXT_SOURCE_INDEX]['name']
        logging.info(f"使用 '{context_model_name}' 作为 patient context 的来源。")

        tasks = []
        for idx, key in enumerate(sorted(union_keys)):
            # 以 context_model 的 prompt 为准，如果没有则用其它模型的 prompt
            if key in all_predictions_data[context_model_name]:
                prompt_item = all_predictions_data[context_model_name][key]
            else:
                # 任取一个有该key的模型的prompt
                for m in model_names:
                    if key in all_predictions_data[m]:
                        prompt_item = all_predictions_data[m][key]
                        break
            predictions_for_item = {
                name: all_predictions_data[name][key]['predict_text'] if key in all_predictions_data[name] else "" for name in model_names
            }
            # 传递 Json_File 和 Node_Number
            tasks.append((evaluator, prompt_item, predictions_for_item, idx, key))

        # 修改 process_evaluation_set 以支持 key
        def process_evaluation_set_with_key(args: Tuple) -> Optional[Dict]:
            evaluator, prompt_item, predictions_for_item, idx, key = args
            try:
                instruction_text, input_text = get_full_context(prompt_item)
                evaluations = evaluator.evaluate_item_set(
                    instruction_text, input_text, predictions_for_item
                )
                return {
                    "item_index": idx,
                    "Json_File": key[0],
                    "Node_Number": key[1],
                    "prompt_preview": instruction_text,
                    "evaluations": evaluations
                }
            except Exception as e:
                logging.error(f"工作线程处理样本集 {idx} 时出错: {e}", exc_info=True)
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results_iterator = tqdm(executor.map(process_evaluation_set_with_key, tasks), total=len(tasks), desc="正在评估样本集")
            for eval_record in results_iterator:
                if eval_record:
                    final_evaluation_results_list.append(eval_record)

        if final_evaluation_results_list:
            save_jsonl_data(final_evaluation_results_list, DETAILED_EVALUATION_FILE)
            logging.info(f"已保存 {len(final_evaluation_results_list)} 条详细评估结果到 {DETAILED_EVALUATION_FILE}")

    if not final_evaluation_results_list:
        logging.warning("没有评估结果可供报告。")
        return

    generate_summary_report(final_evaluation_results_list, SUMMARY_REPORT_FILE, "总体", model_names)
    
    end_time = time.time()
    logging.info(f"脚本执行完毕，总耗时: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}.")

def generate_summary_report(results_list: List[Dict], output_file: str, report_title: str, model_names: List[str]):
    """根据一份评估结果列表，为多个模型生成一份总结报告。"""
    aggregated_scores = collections.defaultdict(lambda: collections.defaultdict(list))
    total_items = len(results_list)

    for result in results_list:
        evals = result.get("evaluations", {})
        for attr_name, eval_data in evals.items():
            scores = eval_data.get("scores", {})
            for model_name, score in scores.items():
                if model_name in model_names:
                    aggregated_scores[attr_name][model_name].append(score)

    summary_lines = []
    summary_lines.append(f"===== {report_title}多模型对比评估报告 =====")
    summary_lines.append(f"评估模型: {EVALUATOR_MODEL_NAME}")
    summary_lines.append(f"生成日期: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"\n对比模型列表: {', '.join(model_names)}")
    summary_lines.append(f"对比样本总数: {total_items}")
    summary_lines.append("="*70)

    for attr_name in EVALUATION_ATTRIBUTES:
        summary_lines.append(f"\n--- 评估任务: {attr_name} ---")
        
        task_scores = aggregated_scores[attr_name]

        summary_lines.append("\n  - 各模型平均得分 (1-5分制, 分数越高越好):")
        for model_name in model_names:
            scores = task_scores.get(model_name, [])
            if scores:
                avg_score = np.mean(scores)
                std_dev = np.std(scores)
                summary_lines.append(f"    - {model_name:<15}: 平均分 {avg_score:.3f} (标准差: {std_dev:.3f})")
            else:
                summary_lines.append(f"    - {model_name:<15}: 无有效得分")

    report_content = "\n".join(summary_lines)
    logging.info("\n" + report_content)

    try:
        with open(output_file, 'w', encoding='utf-8') as f_report:
            f_report.write(report_content)
        logging.info(f"总结报告已保存至: {output_file}")
    except Exception as e:
        logging.error(f"保存总结报告至 {output_file} 失败: {e}")

if __name__ == "__main__":
    main()

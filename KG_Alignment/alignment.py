# FILE: alignment.py
# Description: Core implementation of the alignment algorithm, including DTW and LLM reranking, using a medical BERT model

from typing import Dict, List, Tuple, Optional
import numpy as np
from thefuzz import fuzz
import json
from dtaidistance import dtw
from sentence_transformers import SentenceTransformer, util
import os
import requests
import time
from pathlib import Path
import math
from openai import OpenAI

# Import the same data structures as in alignment_v1.py
from data_structures import PatientTrajectory, GuidelinePathway, ClinicalEvent, GuidelineNode, AlignmentResult
from config import OPENAI_API_KEY, OPENAI_BASE_URL
from graph_connector import GraphConnector

# New: Support for fastdtw
try:
    from fastdtw import fastdtw
    FASTDTW_AVAILABLE = True
except ImportError:
    FASTDTW_AVAILABLE = False

# --- Initialize External Models ---
# Load a pre-trained medical BERT model for calculating semantic similarity
try:
    print("Loading English medical BERT model...")
    semantic_model = SentenceTransformer('medicalai/ClinicalBERT')
    print("ClinicalBERT model loaded successfully.")
except Exception as e:
    print(f"Error: Could not load the ClinicalBERT English medical BERT model, attempting to load a general Chinese model. {e}")
    try:
        print("Loading general Chinese sentence embedding model...")
        semantic_model = SentenceTransformer('shibing624/text2vec-base-chinese')
        print("General Chinese sentence embedding model loaded successfully.")
    except Exception as e2:
        print(f"Error: Could not load any semantic model. Semantic similarity will be unavailable. {e2}")
        semantic_model = None

# === Type Alignment Whitelist (Strictly corresponds to actual KG structure) ===
TYPE_ALIGNMENT_WHITELIST = {
    # TKG entity type : List of allowed KG entity types for alignment
    'ClinicalEvent': ['Treatment', 'Examination', 'Biomarker'],
    'Drug': ['Treatment'],
    'Biomarker': ['Biomarker', 'Examination'],
    'Procedure': ['Examination', 'Treatment'],
    'HospitalAdmission': ['ClinicalSituation'],
    'Patient': ['Patient'],  # Theoretically does not need alignment
}

# === Key Attributes Table (Covers all major entity types) ===
KEY_ATTRIBUTES = {
    'Treatment': ['name', 'line', 'recommendation_level', 'evidence_level'],
    'ClinicalSituation': ['stage', 'name', 'risk_group'],
    'Biomarker': ['name', 'guidance', 'status'],
    'Examination': ['name'],
    'Cancer': ['name'],
    'Patient': ['id'],
    'HospitalAdmission': ['clinical_diagnosis', 'admission_time', 'department'],
    'ClinicalEvent': ['name', 'type'],
    'Drug': ['name'],
}

# === Analysis of Reasonable Alignment Target Types ===
# According to entities_and_relations.txt, the main alignment targets for TKG should be:
# - ClinicalEvent -> Treatment, Examination, Biomarker (core medical actions/tests/markers in guidelines)
# - Drug -> Treatment
# - Biomarker -> Biomarker, Examination
# - Procedure -> Examination, Treatment
# - HospitalAdmission -> ClinicalSituation
# Other types like Patient, Cancer, etc., are generally not aligned at the event level.
# The code below only performs semantic similarity alignment for these target types.

# Whitelist of reasonable alignment target types
TARGET_TYPE_WHITELIST = {
    'ClinicalEvent': ['Treatment', 'Examination', 'Biomarker'],
    'Drug': ['Treatment'],
    'Biomarker': ['Biomarker', 'Examination'],
    'Procedure': ['Examination', 'Treatment'],
    'HospitalAdmission': ['ClinicalSituation'],
}

def _check_target_type(tkg_entity, kg_entity):
    """Only allow TKG entities to align with reasonable KG target types"""
    tkg_type = tkg_entity.labels[-1] if tkg_entity.labels else 'Unknown'
    kg_types = kg_entity.labels
    allowed_kg_types = TARGET_TYPE_WHITELIST.get(tkg_type, [])
    return any(kg_type in allowed_kg_types for kg_type in kg_types)

def _get_textual_representation(node) -> str:
    """Create a textual representation for a node for semantic embedding"""
    props = node.properties
    # Only concatenate fields with actual semantic meaning
    text_parts = []
    for key in ['name', 'content', 'description']:
        v = props.get(key, '')
        if v:
            text_parts.append(str(v))
    # Do not concatenate labels and type
    return ' '.join(text_parts).strip()

def _get_pathway_textual_representation(pathway: GuidelinePathway, idx: int) -> str:
    """Concatenate the text of all nodes from the start to index idx"""
    text_parts = []
    for i in range(idx + 1):
        node = pathway.steps[i]
        node_text = _get_textual_representation(node)
        if node_text:
            text_parts.append(node_text)
    return ' -> '.join(text_parts)

# === Type Whitelist Filtering ===
def _check_type_compatibility(tkg_entity: ClinicalEvent, kg_entity: GuidelineNode) -> float:
    """
    Type whitelist filtering, only structurally allowed types are considered for alignment.
    For example: ClinicalEvent can only align with Treatment/Examination/Biomarker.
    """
    tkg_type = tkg_entity.labels[-1] if tkg_entity.labels else 'Unknown'
    kg_types = kg_entity.labels
    allowed_kg_types = TYPE_ALIGNMENT_WHITELIST.get(tkg_type, [])
    if any(kg_type in allowed_kg_types for kg_type in kg_types):
        return 1.0
    return 0.0

# === Property Similarity with Priority on Key Attributes ===
def _calculate_property_similarity(tkg_entity: ClinicalEvent, kg_entity: GuidelineNode) -> float:
    kg_type = kg_entity.labels[-1] if kg_entity.labels else 'Unknown'
    key_attrs = KEY_ATTRIBUTES.get(kg_type, [])
    tkg_props = tkg_entity.properties
    kg_props = kg_entity.properties
    if not tkg_props or not kg_props or not key_attrs:
        return 0.0
    match_count = 0
    for attr in key_attrs:
        if attr in tkg_props and attr in kg_props and str(tkg_props[attr]) == str(kg_props[attr]):
            match_count += 1
    return match_count / len(key_attrs) if key_attrs else 0.0

# === Name Similarity ===
def _calculate_name_similarity(tkg_entity: ClinicalEvent, kg_entity: GuidelineNode) -> float:
    event_name = tkg_entity.properties.get('name', '')
    if tkg_entity.linked_entity and 'name' in tkg_entity.linked_entity:
        event_name = tkg_entity.linked_entity['name']
    step_name = kg_entity.properties.get('name', '')
    if not event_name or not step_name:
        return 0.0
    similarity = fuzz.token_set_ratio(event_name, step_name) / 100.0
    return similarity

# === Semantic Similarity ===
def _calculate_semantic_similarity(tkg_entity: ClinicalEvent, kg_entity: GuidelineNode, pathway: Optional[GuidelinePathway]=None, step_idx: Optional[int]=None) -> float:
    semantic_similarity = 0.0
    if semantic_model:
        try:
            event_text = _get_textual_representation(tkg_entity)
            # New: If pathway and step_idx are provided, concatenate the path text
            if pathway is not None and step_idx is not None:
                step_text = _get_pathway_textual_representation(pathway, step_idx)
            else:
                step_text = _get_textual_representation(kg_entity)
            if event_text and step_text:
                embedding1 = semantic_model.encode(event_text, convert_to_tensor=True)
                embedding2 = semantic_model.encode(step_text, convert_to_tensor=True)
                cosine_scores = util.cos_sim(embedding1, embedding2)
                semantic_similarity = cosine_scores.item()
        except Exception as e:
            semantic_similarity = 0.0
    else:
        semantic_similarity = 0.0
    return semantic_similarity

# === Combined Similarity ===
def calculate_entity_similarity(tkg_entity: ClinicalEvent, kg_entity: GuidelineNode, pathway: Optional[GuidelinePathway]=None, step_idx: Optional[int]=None) -> float:
    """
    Align using only semantic similarity.
    """
    sem_sim = _calculate_semantic_similarity(tkg_entity, kg_entity, pathway, step_idx)
    return sem_sim

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8)

def align_trajectory_to_pathways(
    trajectory: PatientTrajectory,
    pathways: List[GuidelinePathway]
) -> Tuple[float, GuidelinePathway, list, list]:
    """
    New logic: After BERT encoding, calculate semantic score and DTW score via two separate paths, then combine them with weighting.
    """
    if trajectory is None:
        print("[Debug] trajectory is None!")
        return 0.0, None, [], []
    if not hasattr(trajectory, 'flattened_events') or trajectory.flattened_events is None:
        print("[Debug] trajectory.flattened_events is None!")
        return 0.0, None, [], []
    if not isinstance(trajectory.flattened_events, list):
        print(f"[Debug] trajectory.flattened_events has an abnormal type: {type(trajectory.flattened_events)}")
        return 0.0, None, [], []
    if pathways is None:
        print("[Debug] pathways is None!")
        return 0.0, None, [], []
    if not isinstance(pathways, list):
        print(f"[Debug] pathways has an abnormal type: {type(pathways)}")
        return 0.0, None, [], []

    best_score = float('-inf')
    best_pathway = None
    best_alignment_path = None
    best_alignments = []

    s1 = trajectory.flattened_events
    for pathway in pathways:
        if pathway is None or pathway.steps is None or not pathway.steps:
            continue
        s2 = pathway.steps
        # 1. BERT encoding (in two paths)
        # Path 1: Node-level BERT encoding (single node text)
        tkg_node_embeddings = []
        for event in s1:
            text = _get_textual_representation(event)
            if semantic_model and text:
                tkg_node_embeddings.append(semantic_model.encode(text, convert_to_numpy=True))
            else:
                tkg_node_embeddings.append(np.zeros(768))
        kg_node_embeddings = []
        for step in s2:
            text = _get_textual_representation(step)
            if semantic_model and text:
                kg_node_embeddings.append(semantic_model.encode(text, convert_to_numpy=True))
            else:
                kg_node_embeddings.append(np.zeros(768))
        # Path 2: Sequence-level BERT encoding (full path vector sequence)
        # Directly use the node vector sequence for DTW
        dtw_sim = calculate_dtw_similarity(tkg_node_embeddings, kg_node_embeddings)
        # 2. Node-level alignment (each TKG node against all KG nodes)
        alignments = []
        used_kg_idx = set()
        for i, event in enumerate(s1):
            event_emb = tkg_node_embeddings[i]
            best_node_score = -float('inf')
            best_idx = -1
            for j, kg_node in enumerate(s2):
                if 'Examination' in kg_node.labels:
                    if event.properties.get('type') != 'Examination':
                        continue
                if 'Cancer' in kg_node.labels:
                    if event.labels[-1] != 'Examination':
                        continue
                else:
                    if j in used_kg_idx:
                        continue
                kg_emb = kg_node_embeddings[j]
                # Path 1: Node-level semantic score
                sem_sim = cosine_similarity(event_emb, kg_emb)
                # Path 2: Sequence-level DTW score (global)
                # dtw_sim has already been calculated for this pathway
                # 3. Weighted fusion
                final_score = 0.7 * sem_sim + 0.3 * dtw_sim
                if j % 500 == 0:
                    print(f"   Aligning to KG node {kg_node.node_id} : Semantic score={sem_sim:.3f}, DTW score={dtw_sim:.3f}, Combined score={final_score:.3f}")
                if final_score > best_node_score:
                    best_node_score = final_score
                    best_idx = j
            if best_node_score >= 0.5 and best_idx != -1:
                kg_node = s2[best_idx]
                print(f"   ==> Selected alignment: TKG node {event.node_id} -> KG node {kg_node.node_id} (Semantic score={cosine_similarity(event_emb, kg_node_embeddings[best_idx]):.3f}, DTW score={dtw_sim:.3f}, Combined score={best_node_score:.3f})")
                alignments.append((event.node_id, kg_node.node_id))
                if 'Cancer' not in kg_node.labels:
                    used_kg_idx.add(best_idx)
        # Select the pathway with the highest combined score
        if len(alignments) > 0 and best_node_score > best_score:
            best_score = best_node_score
            best_pathway = pathway
            best_alignment_path = None
            best_alignments = alignments
    return best_score, best_pathway, best_alignment_path, best_alignments

def llm_rerank_and_verify(
    patient_trajectory: PatientTrajectory,
    best_match_pathway: GuidelinePathway,
    dtw_alignment: list,
    dtw_score: float
) -> List[AlignmentResult]:
    """
    Perform automatic tiered filtering on the alignment results.
    Correction:
    1. Relax filtering conditions to also allow high type and semantic similarity alignments.
    2. Use DTW + multi-dimensional similarity directly when LLM is unavailable.
    """
    alignment_pairs = []
    for event_idx, step_idx in dtw_alignment:
        event = patient_trajectory.flattened_events[event_idx]
        step = best_match_pathway.steps[step_idx]
        similarity = calculate_entity_similarity(event, step)
        initial_score = similarity
        alignment_pairs.append(AlignmentResult(tkg_event=event, gg_node=step, initial_score=initial_score))
    alignment_pairs.sort(key=lambda x: x.initial_score, reverse=True)
    final_alignments = []
    for candidate in alignment_pairs:
        type_sim = _check_type_compatibility(candidate.tkg_event, candidate.gg_node)
        sem_sim = _calculate_semantic_similarity(candidate.tkg_event, candidate.gg_node)
        # Strategy 1: High similarity automatic alignment
        if candidate.initial_score >= 0.8:
            final_alignments.append(candidate)
            print(f"High similarity auto-alignment: Event({candidate.tkg_event.node_id}) <-> Step({candidate.gg_node.node_id}), Score: {candidate.initial_score:.3f}")
        # Strategy 2: Strong type match + high semantic similarity
        elif type_sim >= 0.8 and sem_sim >= 0.5:
            final_alignments.append(candidate)
            print(f"Strong type match + high semantic alignment: Event({candidate.tkg_event.node_id}) <-> Step({candidate.gg_node.node_id}), Score: {candidate.initial_score:.3f}")
        # Strategy 3: Also keep low-score pairs if type and semantics are reasonable
        elif candidate.initial_score >= 0.5:
            final_alignments.append(candidate)
            print(f"Low score but reasonable type and semantic alignment: Event({candidate.tkg_event.node_id}) <-> Step({candidate.gg_node.node_id}), Score: {candidate.initial_score:.3f}")
        else:
            print(f"Skipping very low similarity alignment: Event({candidate.tkg_event.node_id}) <-> Step({candidate.gg_node.node_id}), Score: {candidate.initial_score:.3f}")
    print(f"\nAlignment result statistics:")
    print(f"- Total candidates: {len(alignment_pairs)}")
    print(f"- Final alignments: {len(final_alignments)}")
    final_alignments.sort(key=lambda x: x.initial_score, reverse=True)
    return final_alignments

# New: Save alignment pairs to txt
def extract_id_from_info_for_save(info):
    # Compatible with 'TKG node: id=...' or 'KG node: id=...' or direct id
    if isinstance(info, str):
        if info.startswith('TKG node:') or info.startswith('KG node:'):
            s = info.split('id=')[1]
            return s.split(',')[0].strip()
    return str(info)

def save_alignments_to_txt(alignments, filename="alignments.txt", tkg_nodes=None, kg_nodes=None):
    # Ensure the directory exists
    import os
    dirname = os.path.dirname(filename)
    if dirname:  # Only create if the directory name is not empty
        os.makedirs(dirname, exist_ok=True)
    
    with open(filename, "w", encoding="utf-8") as f:
        for tkg_id, kg_id in alignments:
            tkg_id_str = extract_id_from_info_for_save(tkg_id)
            kg_id_str = extract_id_from_info_for_save(kg_id)
            
            # Build TKG node information
            tkg_info = f"TKG Node: id={tkg_id_str}"
            if tkg_nodes:
                # Try multiple ways to find the node
                tkg_node = None
                if tkg_id_str in tkg_nodes:
                    tkg_node = tkg_nodes[tkg_id_str]
                elif tkg_id in tkg_nodes:
                    tkg_node = tkg_nodes[tkg_id]
                elif str(tkg_id) in tkg_nodes:
                    tkg_node = tkg_nodes[str(tkg_id)]
                
                if tkg_node:
                    name = tkg_node.properties.get('name', '')
                    labels = tkg_node.labels if hasattr(tkg_node, 'labels') else []
                    # Keep all properties
                    tkg_info += f", name={name}, labels={labels}, properties={tkg_node.properties}"
            
            # Build KG node information
            kg_info = f"KG Node: id={kg_id_str}"
            if kg_nodes:
                # Try multiple ways to find the node
                kg_node = None
                if kg_id_str in kg_nodes:
                    kg_node = kg_nodes[kg_id_str]
                elif kg_id in kg_nodes:
                    kg_node = kg_nodes[kg_id]
                elif str(kg_id) in kg_nodes:
                    kg_node = kg_nodes[str(kg_id)]
                
                if kg_node:
                    name = kg_node.properties.get('name', '')
                    labels = kg_node.labels if hasattr(kg_node, 'labels') else []
                    # Keep all properties
                    kg_info += f", name={name}, labels={labels}, properties={kg_node.properties}"
            
            f.write(f"{tkg_info}\nALIGN_TO\n{kg_info}\n\n")
    
    print(f"Saved {len(alignments)} alignment relationships to {filename}")

# === LLM Re-alignment Matching Function ===

def call_llm_api_for_realignment(tkg_node_info: str, kg_node_info: str, cancer_type: str) -> Tuple[bool, float, str]:
    """
    Call the LLM API to re-evaluate a single alignment pair
    """
    try:
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        prompt = f"""
You are a medical knowledge graph alignment expert. Please evaluate whether the following two medical entities should be aligned:

**Cancer Type**: {cancer_type}

**TKG Node Information**:
{tkg_node_info}

**KG Node Information**:
{kg_node_info}

Please analyze from the following perspectives:
1. Semantic Similarity: Are the two entities similar in medical concept?
2. Type Compatibility: Do the entity types match?
3. Clinical Relevance: Are they related in the treatment workflow?
4. Property Match: Are key properties consistent?

Please return the result in JSON format:
{{
    "should_align": true/false,
    "confidence": 0.0-1.0,
}}
"""
        print("=== prompt printed, preparing to request LLM ===")
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=False,
            temperature=0.1,
            max_tokens=64000
        )
        content = response.choices[0].message.content
        # Parse JSON response
        try:
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = content[json_start:json_end]
                parsed = json.loads(json_str)
                should_align = parsed.get('should_align', False)
                confidence = float(parsed.get('confidence', 0.0))
                reason = parsed.get('reason', 'No reason provided')
                return should_align, confidence, reason
            else:
                should_align = 'true' in content.lower()
                confidence = 0.5 if should_align else 0.0
                reason = content
                return should_align, confidence, reason
        except json.JSONDecodeError:
            should_align = 'true' in content.lower()
            confidence = 0.5 if should_align else 0.0
            reason = content
            return should_align, confidence, reason
    except Exception as e:
        print(f"LLM API call exception: {e}")
        return False, 0.0, f"Call exception: {str(e)}"

def parse_alignment_file(file_path: str) -> List[Tuple[str, str]]:
    alignments = []
    current_tkg = ""
    current_kg = ""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('TKG Node:'):
                if current_tkg and current_kg and current_tkg.startswith('TKG Node:') and current_kg.startswith('KG Node:'):
                    print(f"[DEBUG] Parsed a pair: TKG={current_tkg!r}, KG={current_kg!r}")
                    alignments.append((current_tkg, current_kg))
                current_tkg = line
                current_kg = ""
            elif line.startswith('KG Node:'):
                current_kg = line
            elif line == 'ALIGN_TO':
                continue
        # Add the last pair at the end of the file
        if current_tkg and current_kg and current_tkg.startswith('TKG Node:') and current_kg.startswith('KG Node:'):
            print(f"[DEBUG] Parsed the last pair: TKG={current_tkg!r}, KG={current_kg!r}")
            alignments.append((current_tkg, current_kg))
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
    return alignments

def process_alignment_file_with_llm(file_path: str, cancer_type: str, tkg_nodes=None, kg_nodes=None, bert_alignments=None) -> List[Tuple[str, str, bool, float, str]]:
    """
    Process a single alignment file in batches using LLM (to avoid exceeding context length limits), and automatically save to the print2 directory with the same filename.
    Alignment pairs with a semantic score > 0.9 are kept directly and do not participate in LLM re-matching.
    Args:
        file_path: Input file path
        cancer_type: Cancer type
        tkg_nodes: TKG node map {id: node}
        kg_nodes: KG node map {id: node}
        bert_alignments: BERT semantic alignment results [(tkg_id, kg_id, score), ...]
    Returns:
        [(tkg_node_info, kg_node_info, should_align, confidence, reason), ...]
    """
    print(f"Processing file in chunks: {file_path}")
    alignments = parse_alignment_file(file_path)
    if not alignments:
        print(f"File {file_path} did not parse to any alignment pairs, skipping.")
        return []
    
    # Separate high-confidence alignment pairs (semantic score > 0.9)
    high_confidence_alignments = []
    llm_alignments = []
    
    # Build semantic score map
    score_map = {}
    if bert_alignments:
        for tkg_id, kg_id, score in bert_alignments:
            score_map[(tkg_id, kg_id)] = score
    
    # Separate alignment pairs based on semantic score
    for tkg_info, kg_info in alignments:
        # Extract ID
        tkg_id = extract_id_from_info_for_save(tkg_info)
        kg_id = extract_id_from_info_for_save(kg_info)
        
        # Find semantic score
        score = score_map.get((tkg_id, kg_id), 0.0)
        
        if score > 0.9:
            high_confidence_alignments.append((tkg_info, kg_info))
        else:
            llm_alignments.append((tkg_info, kg_info))
    
    print(f"Number of alignment pairs requiring LLM processing: {len(llm_alignments)}")
    if high_confidence_alignments:
        print(f"Number of high-confidence alignment pairs kept directly: {len(high_confidence_alignments)}")
    
    # Process alignment pairs requiring LLM in chunks
    all_results = []
    if llm_alignments:
        chunk_size = 100  # 100 alignment pairs per chunk, can be adjusted as needed
        total_chunks = (len(llm_alignments) + chunk_size - 1) // chunk_size
        print(f"Total of {len(llm_alignments)} alignment pairs, divided into {total_chunks} chunks for processing, with a max of {chunk_size} pairs per chunk")
        
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(llm_alignments))
            chunk_alignments = llm_alignments[start_idx:end_idx]
            
            print(f"Processing chunk {chunk_idx + 1}/{total_chunks}, containing {len(chunk_alignments)} alignment pairs...")
            
            # Construct batch prompt
            prompt = f"""
You are a medical knowledge graph alignment expert. Please batch evaluate and re-match the following alignment pairs:

Each alignment pair is formatted as follows:
TKG Node: ...
KG Node: ...

Please:
1. For each pair, determine if it should be aligned (should_align: true/false)
2. Provide a confidence score (confidence: 0.0-1.0)
3. (Important) If you think there is a better match (e.g., the TKG node should be matched to a different KG node), please replace it and re-match.
4. (Important) For re-matched pairs, try to ensure the number of alignments is not significantly different from before. The number of re-matched alignments should not be less than 80% of the original number; do not reduce it too much.
5. Do not output any extra content, please strictly output in the following JSON format.

Please return the results as a JSON array, with each element having the following structure:
{{
    "tkg": "TKG node content",
    "kg": "KG node content",
    "should_align": true/false,
    "confidence": 0.0-1.0,
}}

Here are all the alignment pairs to be evaluated:
"""
            for i, (tkg, kg) in enumerate(chunk_alignments):
                prompt += f"\nAlignment Pair {i+1}:\nTKG Node: {tkg}\nKG Node: {kg}\n"
            prompt += "\nPlease strictly follow the JSON array format specified above for the result."
            
            print(f"Chunk {chunk_idx + 1} prompt length: {len(prompt)}")
            if len(prompt) > 60000:  # Warn when approaching 64K limit
                print(f"Warning: Chunk {chunk_idx + 1} prompt length is approaching the 64K limit, further chunking may be required")
            
            try:
                client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
                response = client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    stream=False,
                    temperature=0.1,
                    max_tokens=64000
                )
                content = response.choices[0].message.content
                json_start = content.find('[')
                json_end = content.rfind(']') + 1
                if json_start != -1 and json_end != -1:
                    json_str = content[json_start:json_end]
                    try:
                        arr = json.loads(json_str)
                        chunk_results = []
                        for item in arr:
                            tkg = item.get('tkg', '')
                            kg = item.get('kg', '')
                            should_align = item.get('should_align', False)
                            confidence = item.get('confidence', 0.0)
                            try:
                                confidence = float(confidence)
                            except Exception:
                                confidence = 0.0
                            reason = item.get('reason', '')
                            chunk_results.append((tkg, kg, should_align, confidence, reason))
                        all_results.extend(chunk_results)
                        print(f"Chunk {chunk_idx + 1} processing complete, obtained {len(chunk_results)} results")
                    except Exception as e:
                        print(f"Chunk {chunk_idx + 1} JSON parsing failed: {e}, falling back to single-item processing")
                        # Fallback to single-item processing
                        for tkg_info, kg_info in chunk_alignments:
                            should_align, confidence, reason = call_llm_api_for_realignment(
                                tkg_info, kg_info, cancer_type
                            )
                            try:
                                confidence = float(confidence)
                            except Exception:
                                confidence = 0.0
                            all_results.append((tkg_info, kg_info, should_align, confidence, reason))
                            time.sleep(0.5)
                else:
                    print(f"Chunk {chunk_idx + 1} failed to parse JSON, falling back to single-item processing")
                    # Fallback to single-item processing
                    for tkg_info, kg_info in chunk_alignments:
                        should_align, confidence, reason = call_llm_api_for_realignment(
                            tkg_info, kg_info, cancer_type
                        )
                        try:
                            confidence = float(confidence)
                        except Exception:
                            confidence = 0.0
                        all_results.append((tkg_info, kg_info, should_align, confidence, reason))
                        time.sleep(0.5)
            except Exception as e:
                print(f"Chunk {chunk_idx + 1} LLM API call exception: {e}, falling back to single-item processing")
                # Fallback to single-item processing
                for tkg_info, kg_info in chunk_alignments:
                    should_align, confidence, reason = call_llm_api_for_realignment(
                        tkg_info, kg_info, cancer_type
                    )
                    try:
                        confidence = float(confidence)
                    except Exception:
                        confidence = 0.0
                    all_results.append((tkg_info, kg_info, should_align, confidence, reason))
                    time.sleep(0.5)
    
    # Merge high-confidence alignment pairs and LLM processing results
    final_results = []
    
    # Add high-confidence alignment pairs (kept directly, confidence 0.95)
    for tkg_info, kg_info in high_confidence_alignments:
        final_results.append((tkg_info, kg_info, True, 0.95, "High-confidence semantic alignment"))
    
    # Add LLM processing results
    final_results.extend(all_results)
    
    print(f"Final result statistics: {len(high_confidence_alignments)} high-confidence pairs kept, {len(all_results)} processed by LLM, total {len(final_results)}")
    
    # === Auto-save to print2 ===
    print2_dir = os.path.join(os.path.dirname(file_path), '..', 'print2')
    print2_dir = os.path.abspath(print2_dir)
    os.makedirs(print2_dir, exist_ok=True)
    filename = os.path.basename(file_path)
    output_path = os.path.join(print2_dir, filename)
    save_llm_results_to_file(final_results, output_path, filename, tkg_nodes=tkg_nodes, kg_nodes=kg_nodes)
    return final_results

def save_llm_results_to_file(results, output_path: str, original_filename: str, tkg_nodes=None, kg_nodes=None):
    """
    Save LLM re-alignment results to a file, in the same format as print1.
    """
    try:
        # Ensure the directory exists
        import os
        dirname = os.path.dirname(output_path)
        if dirname:  # Only create if the directory name is not empty
            os.makedirs(dirname, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for tkg_id, kg_id, should_align, confidence, reason in results:
                if should_align:
                    # Extract ID
                    tkg_id_str = extract_id_from_info_for_save(tkg_id)
                    kg_id_str = extract_id_from_info_for_save(kg_id)
                    
                    # Build TKG node information
                    tkg_info = f"TKG Node: id={tkg_id_str}"
                    if tkg_nodes:
                        # Try multiple ways to find the node
                        tkg_node = None
                        if tkg_id_str in tkg_nodes:
                            tkg_node = tkg_nodes[tkg_id_str]
                        elif tkg_id in tkg_nodes:
                            tkg_node = tkg_nodes[tkg_id]
                        elif str(tkg_id) in tkg_nodes:
                            tkg_node = tkg_nodes[str(tkg_id)]
                        
                        if tkg_node:
                            name = tkg_node.properties.get('name', '')
                            labels = tkg_node.labels if hasattr(tkg_node, 'labels') else []
                            # Keep all properties
                            tkg_info += f", name={name}, labels={labels}, properties={tkg_node.properties}"
                    
                    # Build KG node information
                    kg_info = f"KG Node: id={kg_id_str}"
                    if kg_nodes:
                        # Try multiple ways to find the node
                        kg_node = None
                        if kg_id_str in kg_nodes:
                            kg_node = kg_nodes[kg_id_str]
                        elif kg_id in kg_nodes:
                            kg_node = kg_nodes[kg_id]
                        elif str(kg_id) in kg_nodes:
                            kg_node = kg_nodes[str(kg_id)]
                        
                        if kg_node:
                            name = kg_node.properties.get('name', '')
                            labels = kg_node.labels if hasattr(kg_node, 'labels') else []
                            # Keep all properties
                            kg_info += f", name={name}, labels={labels}, properties={kg_node.properties}"
                    
                    f.write(f"{tkg_info}\nALIGN_TO\n{kg_info}\n\n")
        
        print(f"Results have been saved to: {output_path}")
    except Exception as e:
        print(f"Error saving result file: {e}")

def read_alignment_result_from_file(file_path: str) -> list:
    """
    Read an alignment result file, return a list of alignment pairs (can be adjusted based on file format).
    """
    results = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Simple format check, supports print2 and print1 formats
        if content.startswith('# LLM Re-alignment Results'):
            # print2 format, parse by separator
            blocks = content.split('==================================================')
            for block in blocks:
                if '**TKG Node**:' in block and '**KG Node**:' in block:
                    tkg = kg = align = conf = reason = None
                    for line in block.splitlines():
                        if line.startswith('**TKG Node**:'):
                            tkg = line.replace('**TKG Node**:', '').strip()
                        elif line.startswith('**KG Node**:'):
                            kg = line.replace('**KG Node**:', '').strip()
                        elif line.startswith('**LLM Judgment**:'):
                            align = 'Align' in line
                        elif line.startswith('**Confidence**:'):
                            conf_str = line.replace('**Confidence**:', '').strip()
                            try:
                                conf = float(conf_str)
                            except Exception:
                                conf = 0.0
                        elif line.startswith('**Reason**:'):
                            reason = line.replace('**Reason**:', '').strip()
                    if tkg and kg:
                        results.append((tkg, kg, align, conf, reason))
        else:
            # print1 format, simple parsing
            lines = content.splitlines()
            tkg = kg = None
            for line in lines:
                if line.startswith('TKG Node:'):
                    tkg = line.strip()
                elif line.startswith('KG Node:'):
                    kg = line.strip()
                elif line == 'ALIGN_TO' or not line.strip():
                    continue
                if tkg and kg:
                    results.append((tkg, kg, True, 1.0, 'print1 original alignment'))
                    tkg = kg = None
    except Exception as e:
        print(f"Failed to read alignment result file {file_path}: {e}")
    return results

def bootea_iterative_alignment(tkg_nodes, kg_nodes, initial_alignments, max_iter=5, sim_threshold=0.7):
    """
    BootEA self-extended alignment: Iteratively expand more alignment pairs using already aligned pairs.
    Optimization: Pre-encode node BERT vectors + batch cosine similarity calculation.
    Terminates early if no new alignment pairs are added for 3 consecutive rounds.
    """
    aligned_tkg = set([a[0] for a in initial_alignments])
    aligned_kg = set([a[1] for a in initial_alignments])
    alignments = set((a[0], a[1]) for a in initial_alignments)
    tkg_id2node = {n.node_id: n for n in tkg_nodes}
    kg_id2node = {n.node_id: n for n in kg_nodes}
    
    # 1. Pre-encode all nodes (using global cache)
    print("[BootEA] Pre-encoding all node vectors...")
    
    # Ensure guideline nodes are pre-encoded
    if not _kg_embeddings_cache:
        precompute_kg_embeddings(kg_nodes)
    
    # Pre-encode TKG nodes
    tkg_emb_dict = {}
    for n in tkg_nodes:
        text = _get_textual_representation(n)
        if text and semantic_model:
            try:
                tkg_emb_dict[n.node_id] = semantic_model.encode(text, convert_to_numpy=True)
            except Exception as e:
                print(f"[Warning] TKG node {n.node_id} encoding failed: {e}")
                tkg_emb_dict[n.node_id] = None
        else:
            tkg_emb_dict[n.node_id] = None
    print(f"[BootEA] Initial number of alignment pairs: {len(alignments)}")
    no_new_count = 0
    for it in range(max_iter):
        if it == 0:
            cur_threshold = 0.7
        elif it == 1:
            cur_threshold = 0.65
        elif it == 2:
            cur_threshold = 0.6
        else:
            cur_threshold = 0.55
        new_alignments = set()
        # Collect unaligned nodes
        tkg_unaligned = [n for n in tkg_nodes if n.node_id not in aligned_tkg]
        kg_unaligned = [n for n in kg_nodes if n.node_id not in aligned_kg]
        if not tkg_unaligned or not kg_unaligned:
            break
            
        # Filter valid vectors
        valid_tkg_unaligned = [n for n in tkg_unaligned if tkg_emb_dict.get(n.node_id) is not None]
        valid_kg_unaligned = [n for n in kg_unaligned if _kg_embeddings_cache.get(n.node_id) is not None]
        
        if not valid_tkg_unaligned or not valid_kg_unaligned:
            print(f"[BootEA] Round {it+1} has no valid vectors, skipping")
            continue
            
        # Build vector matrix
        tkg_vecs = np.stack([tkg_emb_dict[n.node_id] for n in valid_tkg_unaligned])
        kg_vecs = np.stack([_kg_embeddings_cache[n.node_id] for n in valid_kg_unaligned])
        # Batch cosine similarity (N, M)
        tkg_norm = np.linalg.norm(tkg_vecs, axis=1, keepdims=True)
        kg_norm = np.linalg.norm(kg_vecs, axis=1, keepdims=True)
        sim_matrix = np.dot(tkg_vecs, kg_vecs.T) / (tkg_norm * kg_norm.T + 1e-8)
        # For each TKG node, select the KG node with the max score
        for i, tkg in enumerate(valid_tkg_unaligned):
            best_j = np.argmax(sim_matrix[i])
            best_score = sim_matrix[i, best_j]
            kg = valid_kg_unaligned[best_j]
            # BootEA structural enhancement: neighbor alignment score
            tkg_neighbors = set([n.node_id for n in tkg.neighbors]) if hasattr(tkg, 'neighbors') else set()
            kg_neighbors = set([n.node_id for n in kg.neighbors]) if hasattr(kg, 'neighbors') else set()
            neighbor_score = 0
            if tkg_neighbors and kg_neighbors:
                matched = 0
                for tkg_n in tkg_neighbors:
                    for kg_n in kg_neighbors:
                        if (tkg_n, kg_n) in alignments:
                            matched += 1
                neighbor_score = matched / max(len(tkg_neighbors), 1)
            total_score = 0.7 * best_score + 0.3 * neighbor_score
            if total_score >= cur_threshold:
                new_alignments.add((tkg.node_id, kg.node_id))
        if not new_alignments:
            no_new_count += 1
            print(f"[BootEA] Round {it+1} added no new alignment pairs, {no_new_count} consecutive rounds with no new pairs.")
            if no_new_count >= 3:
                print(f"[BootEA] No new alignment pairs for 3 consecutive rounds, converging early.")
                break
            continue
        else:
            no_new_count = 0
        for tkg_id, kg_id in new_alignments:
            aligned_tkg.add(tkg_id)
            aligned_kg.add(kg_id)
        alignments.update(new_alignments)
        print(f"[BootEA] Round {it+1} expansion, new alignments: {len(new_alignments)}, cumulative alignments: {len(alignments)}")
        # Output some examples of new alignment pairs
        if len(new_alignments) > 0:
            print("[BootEA] Example of new alignment pairs:")
            for idx, (tkg_id, kg_id) in enumerate(list(new_alignments)[:5]):
                print(f"  TKG Node: {tkg_id} <-> KG Node: {kg_id}")
    print(f"[BootEA] Final cumulative number of alignment pairs: {len(alignments)}")
    return list(alignments)

# Global variable: pre-encoded guideline node vectors
_kg_embeddings_cache = {}

def precompute_kg_embeddings(kg_nodes):
    """
    Pre-compute semantic vectors for all guideline nodes to improve calculation efficiency.
    """
    global _kg_embeddings_cache
    print("[Semantic Pre-encoding] Starting to pre-compute guideline node vectors...")
    _kg_embeddings_cache.clear()
    
    for kg_node in kg_nodes:
        kg_text = _get_textual_representation(kg_node)
        if kg_text and semantic_model:
            try:
                kg_emb = semantic_model.encode(kg_text, convert_to_numpy=True)
                _kg_embeddings_cache[kg_node.node_id] = kg_emb
            except Exception as e:
                print(f"[Warning] Node {kg_node.node_id} encoding failed: {e}")
                _kg_embeddings_cache[kg_node.node_id] = None
        else:
            _kg_embeddings_cache[kg_node.node_id] = None
    
    valid_count = sum(1 for emb in _kg_embeddings_cache.values() if emb is not None)
    print(f"[Semantic Pre-encoding] Complete! Pre-encoded {len(_kg_embeddings_cache)} nodes in total, with {valid_count} valid encodings.")

def bert_semantic_align(tkg_nodes, kg_nodes, top_k=1, threshold=0.7):
    """
    Perform initial alignment using BERT semantic similarity, returning a list of candidate alignment pairs [(tkg_id, kg_id, sim_score)].
    Optimization: Use pre-encoded guideline node vectors to improve calculation efficiency.
    """
    alignments = []
    
    # Ensure guideline nodes are pre-encoded
    if not _kg_embeddings_cache:
        precompute_kg_embeddings(kg_nodes)
    
    print(f"[Semantic Alignment] Starting to align {len(tkg_nodes)} TKG nodes to {len(kg_nodes)} guideline nodes...")
    
    for i, tkg in enumerate(tkg_nodes):
        if i % 100 == 0:
            print(f"[Semantic Alignment] Progress: {i}/{len(tkg_nodes)}")
            
        best_kg = None
        best_score = -1
        tkg_text = _get_textual_representation(tkg)
        if not tkg_text or not semantic_model:
            continue
            
        # Encode TKG node
        try:
            tkg_emb = semantic_model.encode(tkg_text, convert_to_numpy=True)
        except Exception as e:
            print(f"[Warning] TKG node {tkg.node_id} encoding failed: {e}")
            continue
        
        # Calculate similarity with all pre-encoded guideline nodes
        for kg in kg_nodes:
            kg_emb = _kg_embeddings_cache.get(kg.node_id)
            if kg_emb is None:
                continue
                
            # Calculate cosine similarity
            sim = np.dot(tkg_emb, kg_emb) / (np.linalg.norm(tkg_emb) * np.linalg.norm(kg_emb) + 1e-8)
            
            if sim > best_score:
                best_score = sim
                best_kg = kg
        
        if best_kg and best_score >= threshold:
            alignments.append((tkg.node_id, best_kg.node_id, best_score))
            if len(alignments) % 10 == 0:
                print(f"[Semantic Alignment] Found {len(alignments)} alignment pairs so far, latest score: {best_score:.3f}")
    
    print(f"[Semantic Alignment] Complete! Found a total of {len(alignments)} alignment pairs")
    return alignments

def fix_llm_alignment_by_id(llm_results, print1_alignments, tkg_nodes, kg_nodes):
    """
    Correct the node information in LLM-generated alignment pairs based on the node IDs from print1.
    llm_results: [(tkg_info, kg_info, should_align, confidence, reason), ...]
    print1_alignments: [(tkg_id, kg_id), ...]
    tkg_nodes/kg_nodes: {id: node_obj}
    Returns corrected llm_results: [(tkg_id, kg_id, should_align, confidence, reason), ...]
    """
    # Build a map from id to node
    tkg_id_map = {str(tkg_id): tkg_id for tkg_id, _ in print1_alignments}
    kg_id_map = {str(kg_id): kg_id for _, kg_id in print1_alignments}
    def extract_id_from_info(info):
        # Assume format is like 'TKG Node: id=123, ...' or 'KG Node: id=456, ...'
        if 'id=' in info:
            s = info.split('id=')[1]
            id_val = s.split(',')[0].strip()
            # Remove redundant 'id=' prefix
            if id_val.startswith('id='):
                id_val = id_val[3:]
            return id_val
        return None
    fixed_results = []
    for tkg_info, kg_info, should_align, confidence, reason in llm_results:
        tkg_id = extract_id_from_info(tkg_info)
        kg_id = extract_id_from_info(kg_info)
        # Correct using the id map from print1
        if tkg_id in tkg_id_map:
            tkg_id_fixed = tkg_id_map[tkg_id]
        else:
            tkg_id_fixed = tkg_id
        if kg_id in kg_id_map:
            kg_id_fixed = kg_id_map[kg_id]
        else:
            kg_id_fixed = kg_id
        fixed_results.append((tkg_id_fixed, kg_id_fixed, should_align, confidence, reason))
    return fixed_results
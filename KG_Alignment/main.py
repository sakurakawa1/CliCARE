# FILE: main.py
# Description: The main entry point of the project, coordinating the entire graph alignment and fusion process

from tqdm import tqdm
import concurrent.futures
from pathlib import Path
import os

from graph_connector import GraphConnector
from trajectory_extraction import get_patient_trajectory, get_guideline_pathways
from alignment import (
    bert_semantic_align,
    bootea_iterative_alignment,
    save_alignments_to_txt,
    save_llm_results_to_file,
    process_alignment_file_with_llm,
    read_alignment_result_from_file,
    fix_llm_alignment_by_id,
    precompute_kg_embeddings
)
from fusion import fuse_alignments
from data_structures import AlignmentResult

def process_patient(patient_id, all_kg_nodes):
    """Processing function for a single patient, independently creates a database connector within the thread and uses pre-computed guideline node vectors"""
    connector = GraphConnector()
    try:
        print(f"\n--- Processing patient: {patient_id} ---")

        # a. Get the patient's cancer type
        cancer_type = connector.get_patient_cancer_type(patient_id)
        if not cancer_type:
            print(f"Warning: Cannot determine the cancer type for patient {patient_id}, skipping this patient.")
            return
        print(f"The cancer type for patient {patient_id} is: {cancer_type}")

        # b. Get the patient's treatment trajectory
        patient_trajectory = get_patient_trajectory(patient_id, connector)
        print(f"[Debug] patient_trajectory={patient_trajectory}")
        
        if not all_kg_nodes:
            print(f"Warning: No guideline nodes found, skipping patient {patient_id}.")
            return

        # Check key variables for None and type
        if patient_trajectory is None:
            print(f"[Debug] patient_trajectory is None, skipping!")
            return
        if not hasattr(patient_trajectory, 'flattened_events') or patient_trajectory.flattened_events is None:
            print(f"[Debug] patient_trajectory.flattened_events is None, skipping!")
            return
        if not isinstance(patient_trajectory.flattened_events, list):
            print(f"[Debug] Abnormal type for patient_trajectory.flattened_events: {type(patient_trajectory.flattened_events)}, skipping!")
            return
        if all_kg_nodes is None:
            print(f"[Debug] all_kg_nodes is None, skipping!")
            return
        if not isinstance(all_kg_nodes, list):
            print(f"[Debug] Abnormal type for all_kg_nodes: {type(all_kg_nodes)}, skipping!")
            return
        if not patient_trajectory.flattened_events or not all_kg_nodes:
            print(f"Warning: The trajectory or guideline nodes for patient {patient_id} are empty, cannot perform alignment.")
            return

        BootEA_Align_path = f"./BootEA_Align/alignments_{patient_id}.txt"
        Path("BootEA_Align").mkdir(exist_ok=True)
        if Path(BootEA_Align_path).exists():
            print(f"BootEA_Align result already exists, writing directly back to Neo4j: {BootEA_Align_path}")
            alignment_results = []
            with open(BootEA_Align_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            tkg_id = kg_id = None
            tkg_nodes = {str(node.node_id): node for node in patient_trajectory.flattened_events}
            kg_nodes = {str(node.node_id): node for node in all_kg_nodes}
            # Read the set of alignment pairs from LLM_Align
            LLM_Align_path = f"./LLM_Align/alignments_{patient_id}.txt"
            llm_pairs = set()
            if Path(LLM_Align_path).exists():
                with open(LLM_Align_path, 'r', encoding='utf-8') as f2:
                    lines2 = f2.readlines()
                tkg2 = kg2 = None
                for line in lines2:
                    if line.startswith('TKG Node: id='):
                        tkg2 = line.split('id=')[1].split(',')[0].strip()
                    if line.startswith('KG Node: id='):
                        kg2 = line.split('id=')[1].split(',')[0].strip()
                        if tkg2 is not None and kg2 is not None:
                            llm_pairs.add((tkg2, kg2))
                            tkg2 = kg2 = None
            # Process BootEA_Align
            for line in lines:
                if line.startswith('TKG Node: id='):
                    tkg_id = line.split('id=')[1].split(',')[0].strip()
                if line.startswith('KG Node: id='):
                    kg_id = line.split('id=')[1].split(',')[0].strip()
                    if tkg_id is not None and kg_id is not None:
                        tkg_node = tkg_nodes.get(tkg_id)
                        kg_node = kg_nodes.get(kg_id)
                        if tkg_node and kg_node:
                            rationale = "LLM" if (tkg_id, kg_id) in llm_pairs else "BootEA"
                            alignment_results.append(
                                AlignmentResult(
                                    tkg_event=tkg_node,
                                    gg_node=kg_node,
                                    initial_score=1.0,
                                    llm_score=1.0,
                                    llm_rationale=rationale
                                )
                            )
                        tkg_id = kg_id = None
            if alignment_results:
                print(f"Found a total of {len(alignment_results)} alignment relationships, writing back to Neo4j...")
                fuse_alignments(alignment_results, connector)
            else:
                print(f"No valid alignment relationships found in {BootEA_Align_path}.")
            return

        # 1. Initial alignment using semantic similarity
        tkg_nodes = [node for node in patient_trajectory.flattened_events]
        kg_nodes = all_kg_nodes  # Directly use the pre-computed guideline nodes
        bert_alignments = bert_semantic_align(tkg_nodes, kg_nodes, top_k=1, threshold=0.7)
        print(f"BERT initial alignment complete, {len(bert_alignments)} pairs found. Saving to Bert_Align...")
        Bert_Align_path = f"./Bert_Align/alignments_{patient_id}.txt"
        Path("Bert_Align").mkdir(exist_ok=True)
        # Build a node map to ensure complete node information is saved
        tkg_nodes_map = {str(node.node_id): node for node in tkg_nodes}
        kg_nodes_map = {str(node.node_id): node for node in kg_nodes}
        save_alignments_to_txt([(a[0], a[1]) for a in bert_alignments], filename=Bert_Align_path, tkg_nodes=tkg_nodes_map, kg_nodes=kg_nodes_map)
        
        # 2. LLM reranking (passing semantic score information from bert_alignments)
        print("Calling LLM for reranking...")
        # Pass the semantic score information from bert_alignments to the LLM processing function
        llm_results = process_alignment_file_with_llm(Bert_Align_path, cancer_type, tkg_nodes=tkg_nodes_map, kg_nodes=kg_nodes_map, bert_alignments=bert_alignments)
        print(f"LLM reranking complete, {len(llm_results)} pairs found. Saving to LLM_Align...")
        LLM_Align_path = f"./LLM_Align/alignments_{patient_id}.txt"
        Path("LLM_Align").mkdir(exist_ok=True)
        # Ensure LLM_Align also saves complete node information
        save_llm_results_to_file(llm_results, LLM_Align_path, Bert_Align_path, tkg_nodes=tkg_nodes_map, kg_nodes=kg_nodes_map)
        
        # 3. BootEA iterative expansion
        print("BootEA iterative expansion...")
        tkg_nodes_map = {str(node.node_id): node for node in tkg_nodes}
        kg_nodes_map = {str(node.node_id): node for node in kg_nodes}
        initial_alignments = [(item[0], item[1]) for item in llm_results if len(item) > 2 and item[2]]
        bootea_results = bootea_iterative_alignment(tkg_nodes, kg_nodes, initial_alignments, max_iter=6, sim_threshold=0.7)
        print(f"BootEA expansion complete, {len(bootea_results)} pairs found. Saving to BootEA_Align...")
        save_alignments_to_txt(bootea_results, filename=BootEA_Align_path, tkg_nodes=tkg_nodes_map, kg_nodes=kg_nodes_map)
        
        # 4. Write back to Neo4j
        alignment_results = []
        for tkg_id, kg_id in bootea_results:
            tkg_node = tkg_nodes_map.get(tkg_id)
            kg_node = kg_nodes_map.get(kg_id)
            if tkg_node and kg_node:
                alignment_results.append(
                    AlignmentResult(
                        tkg_event=tkg_node,
                        gg_node=kg_node,
                        initial_score=1.0,
                        llm_score=1.0,
                        llm_rationale="BootEA+LLM"
                    )
                )
        if alignment_results:
            print(f"Found a total of {len(alignment_results)} final alignment relationships, writing back to Neo4j...")
            fuse_alignments(alignment_results, connector)
        else:
            print("No valid alignment relationships found.")
            
    except Exception as e:
        print(f"An error occurred while processing patient {patient_id}: {e}")
    finally:
        connector.close()

def main():
    """Main execution function"""
    print("--- Starting the knowledge graph alignment and fusion process ---")
    
    # 1. Initialize the database connector
    connector = GraphConnector()
    if not connector.driver:
        print("Database connection failed, program terminated.")
        return

    try:
        # 2. Get all patient IDs
        patient_ids = connector.get_all_patient_ids()
        if not patient_ids:
            print("No patients found in the database, program terminated.")
            return
            
        print(f"Found a total of {len(patient_ids)} patients, starting multi-threaded processing...")
        
        # 3. Pre-fetch all guideline nodes and compute vectors (only once)
        print("Fetching all guideline nodes and pre-computing vectors...")
        all_kg_nodes = connector.get_all_guideline_nodes()
        if not all_kg_nodes:
            print("No guideline nodes found, program terminated.")
            return
        print(f"Retrieved {len(all_kg_nodes)} guideline nodes, starting to pre-compute vectors...")
        precompute_kg_embeddings(all_kg_nodes)
        print("Guideline node vector pre-computation complete!")
        
        # 4. Multi-threaded processing of patients
        max_workers = 16  # 16 threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks, passing the pre-computed guideline nodes
            futures = [executor.submit(process_patient, pid, all_kg_nodes) for pid in patient_ids]
            
            # Wait for all tasks to complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f"An error occurred in thread execution: {exc}")

    except Exception as e:
        print(f"A critical error occurred in the main process: {e}")
    finally:
        # 5. Close the database connection
        connector.close()
        print("--- Process execution finished ---")

if __name__ == "__main__":
    main()
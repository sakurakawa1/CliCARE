# FILE: trajectory_extraction.py
# Description: Implements the logic for extracting patient trajectories and guideline pathways from the graph

import networkx as nx
from typing import List

from graph_connector import GraphConnector
from data_structures import PatientTrajectory, GuidelinePathway, GuidelineNode

def get_patient_trajectory(patient_id: str, connector: GraphConnector) -> PatientTrajectory:
    """
    Builds the complete treatment trajectory for a specified patient ID.
    """
    admissions = connector.get_patient_admissions(patient_id)
    return PatientTrajectory(patient_id=patient_id, admissions=admissions)


def get_guideline_pathways(cancer_name: str, connector: GraphConnector) -> List[GuidelinePathway]:
    """
    Extracts all possible, standardized treatment pathways for a specified cancer type.
    """
    guideline_graph = connector.get_guideline_graph_for_cancer(cancer_name)
    if guideline_graph is None:
        return []
    if not hasattr(guideline_graph, 'nodes') or guideline_graph.nodes is None:
        return []

    # 2. Identify the start and end nodes of the pathways
    # Start nodes are typically 'ClinicalSituation' or 'Cancer'
    # End nodes are those with no outgoing edges (leaf nodes), usually representing the end of treatment or follow-up
    start_nodes = [
        node_id for node_id, data in guideline_graph.nodes(data=True)
        if 'Cancer' in data['data'].labels or 'ClinicalSituation' in data['data'].labels
    ]
    end_nodes = [node_id for node_id in guideline_graph.nodes() if guideline_graph.out_degree(node_id) == 0]

    if not start_nodes:
        # If there are no ClinicalSituation nodes, start from the Cancer node
        start_nodes = [node_id for node_id, data in guideline_graph.nodes(data=True) if 'Cancer' in data['data'].labels]

    if not start_nodes or not end_nodes:
        return []

    # 3. Use a graph traversal algorithm (like all_simple_paths) to find all paths
    all_raw_paths = []
    for start_node in start_nodes:
        for end_node in end_nodes:
            if nx.has_path(guideline_graph, start_node, end_node):
                paths = nx.all_simple_paths(guideline_graph, source=start_node, target=end_node)
                all_raw_paths.extend(list(paths))

    if all_raw_paths is None:
        return []
    # 4. Convert the raw paths (lists of node IDs) into a list of GuidelinePathway objects
    guideline_pathways = []
    for i, path_ids in enumerate(all_raw_paths):
        steps = [guideline_graph.nodes[node_id]['data'] for node_id in path_ids]
        pathway = GuidelinePathway(pathway_id=f"{cancer_name}_path_{i+1}", steps=steps)
        guideline_pathways.append(pathway)
        
    if guideline_pathways is None:
        return []
    return guideline_pathways if guideline_pathways is not None else []
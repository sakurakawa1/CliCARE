# FILE: graph_connector.py
# Description: Encapsulates all interactions with the Neo4j database

import networkx as nx
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
from data_structures import GuidelineNode, ClinicalEvent, HospitalAdmission

class GraphConnector:
    """Handles connection and data queries with the Neo4j database"""

    def __init__(self):
        """Initialize the database driver"""
        try:
            self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            self.driver.verify_connectivity()
            print(f"Successfully connected to Neo4j database: {NEO4J_DATABASE}")
        except Exception as e:
            print(f"Could not connect to Neo4j database: {e}")
            self.driver = None

    def close(self):
        """Close the database connection"""
        if self.driver:
            self.driver.close()
            print("Disconnected from the Neo4j database.")
            
    def get_all_patient_ids(self) -> List[str]:
        """Get all patient IDs from the TKG"""
        if not self.driver:
            return []
        with self.driver.session(database=NEO4J_DATABASE) as session:
            result = session.run("MATCH (p:Patient) RETURN p.id AS patientId")
            return [record["patientId"] for record in result]

    def get_patient_cancer_type(self, patient_id: str) -> Optional[str]:
        """Get the cancer type for a patient based on their ID"""
        if not self.driver:
            return None
        with self.driver.session(database=NEO4J_DATABASE) as session:
            cypher = """
            MATCH (p:Patient {id: $patient_id})-[:HAS_CANCER]->(c:Cancer)
            RETURN c.name AS cancer_name
            LIMIT 1
            """
            result = session.run(cypher, patient_id=patient_id).single()
            return result["cancer_name"] if result else None

    def get_patient_admissions(self, patient_id: str) -> List[HospitalAdmission]:
        """Get all hospital admission records and their included clinical events for a patient"""
        if not self.driver:
            return []
            
        cypher = """
        MATCH (p:Patient {id: $patient_id})-[:HAS_ADMISSION]->(ha:HospitalAdmission)
        // Optionally match clinical events included during the hospital stay
        OPTIONAL MATCH (ha)-[:INCLUDES_EVENT]->(ce:ClinicalEvent)
        // Further get entities (drugs or biomarkers) associated with the event
        OPTIONAL MATCH (ce)-[r:IS_DRUG|MEASURES]->(entity)
        RETURN ha, COLLECT({event: ce, linked_entity: entity}) as events
        ORDER BY ha.admission_time
        """
        with self.driver.session(database=NEO4J_DATABASE) as session:
            results = session.run(cypher, patient_id=patient_id)
            admissions = []
            for record in results:
                ha_node = record["ha"]
                
                clinical_events = []
                for event_data in record["events"]:
                    if event_data and event_data["event"]:
                        event_node = event_data["event"]
                        linked_entity_node = event_data.get("linked_entity")
                        clinical_events.append(ClinicalEvent(
                            node_id=event_node.id,
                            labels=list(event_node.labels),
                            properties=dict(event_node.items()),
                            linked_entity=dict(linked_entity_node.items()) if linked_entity_node else None
                        ))

                admissions.append(HospitalAdmission(
                    node_id=ha_node.id,
                    properties=dict(ha_node.items()),
                    events=clinical_events,
                    admission_time=ha_node.get("admission_time")
                ))
        return admissions

    def get_guideline_graph_for_cancer(self, cancer_name: str) -> nx.DiGraph:
        """
        For a specific cancer type, extract the relevant guideline subgraph from the database 
        and build a NetworkX graph object.
        This graph will be used to extract all possible treatment pathways in the next step.
        """
        print(f"[Debug] get_guideline_graph_for_cancer: cancer_name={cancer_name}")
        if not self.driver:
            print("[Debug] Neo4j driver is not connected, returning an empty DiGraph!")
            return nx.DiGraph()
        try:
            cypher = """
            MATCH (c:Cancer {name: $cancer_name})
            OPTIONAL MATCH path = (c)-[:HAS_CLINICAL_SITUATION|HAS_EXAMINATION*1..5]->(n)
            WHERE n:ClinicalSituation OR n:Examination OR n:Treatment OR n:Biomarker
            RETURN nodes(path) as nodes, relationships(path) as rels
            """
            graph = nx.DiGraph()
            with self.driver.session(database=NEO4J_DATABASE) as session:
                results = session.run(cypher, cancer_name=cancer_name)
                for record in results:
                    path_nodes = record['nodes']
                    path_rels = record['rels']
                    for i in range(len(path_rels)):
                        start_node_data = path_nodes[i]
                        end_node_data = path_nodes[i+1]
                        start_node = GuidelineNode(node_id=start_node_data.id, labels=list(start_node_data.labels), properties=dict(start_node_data.items()))
                        end_node = GuidelineNode(node_id=end_node_data.id, labels=list(end_node_data.labels), properties=dict(end_node_data.items()))
                        graph.add_node(start_node.node_id, data=start_node)
                        graph.add_node(end_node.node_id, data=end_node)
                        graph.add_edge(start_node.node_id, end_node.node_id, type=path_rels[i].type)
            print(f"[Debug] number of guideline_graph nodes: {len(graph.nodes)}")
            return graph
        except Exception as e:
            print(f"[Debug] an exception occurred in get_guideline_graph_for_cancer: {e}")
            return nx.DiGraph()

    def get_all_guideline_nodes(self) -> List[GuidelineNode]:
        """
        Get all guideline nodes, not limited by cancer type.
        Returns a list of all guideline nodes available for alignment.
        """
        if not self.driver:
            print("[Debug] Neo4j driver is not connected, returning an empty list!")
            return []
        
        try:
            cypher = """
            MATCH (n)
            WHERE n:Treatment OR n:Examination OR n:Biomarker OR n:ClinicalSituation
            RETURN n, labels(n) as labels
            """
            
            guideline_nodes = []
            with self.driver.session(database=NEO4J_DATABASE) as session:
                results = session.run(cypher)
                for record in results:
                    node_data = record["n"]
                    labels = record["labels"]
                    guideline_node = GuidelineNode(
                        node_id=node_data.id,
                        labels=labels,
                        properties=dict(node_data.items())
                    )
                    guideline_nodes.append(guideline_node)
            
            print(f"[Debug] Retrieved {len(guideline_nodes)} guideline nodes")
            return guideline_nodes
            
        except Exception as e:
            print(f"[Debug] an exception occurred in get_all_guideline_nodes: {e}")
            return []
# FILE: fusion.py
# Description: Writes high-quality alignment results back to the Neo4j database

from typing import List
from graph_connector import GraphConnector
from data_structures import AlignmentResult
from neo4j import GraphDatabase
from config import NEO4J_DATABASE

def fuse_alignments(
    high_quality_alignments: List[AlignmentResult],
    connector: GraphConnector
):
    """
    Writes high-quality alignment results directly to the Neo4j database
    (implements Cypher writing and debug output here).
    """
    if not high_quality_alignments:
        print("No high-quality alignment results to fuse.")
        return

    print(f"Starting to fuse {len(high_quality_alignments)} high-quality alignment relationships...")

    driver = connector.driver
    for alignment in high_quality_alignments:
        tkg_id = ""
        kg_name = ""
        kg_type = ""
        try:
            tkg_id = str(alignment.tkg_event.properties.get('id'))
            kg_name = str(alignment.gg_node.properties.get('name'))
            kg_type = alignment.gg_node.properties.get('type')
            cypher = """
            MATCH (tkg_event {id: $tkg_event_id})
            MATCH (gg_node {name: $kg_name, type: $kg_type})
            MERGE (gg_node)-[r:ALIGNED_TO]->(tkg_event)
            SET r.llm_score = $score, r.llm_rationale = $rationale, r.updated_at = timestamp()
            """
            with driver.session(database=NEO4J_DATABASE) as session:
                session.run(
                    cypher,
                    tkg_event_id=tkg_id,
                    kg_name=kg_name,
                    kg_type=kg_type,
                    score=alignment.llm_score if alignment.llm_score is not None else 1.0,
                    rationale=alignment.llm_rationale or "Automatic alignment"
                )
                # Debug: Query the newly written relationship and print details
                debug_cypher = """
                MATCH (gg_node)-[r:ALIGNED_TO]->(tkg_event)
                WHERE gg_node.name = $kg_name AND gg_node.type = $kg_type AND tkg_event.id = $tkg_event_id
                RETURN gg_node, r, tkg_event
                """
                result = session.run(debug_cypher, tkg_event_id=tkg_id, kg_name=kg_name, kg_type=kg_type)
                record = result.single()
                if record:
                    print("[Debug] Relationship written:")
                    print("  KG Node:", dict(record["gg_node"].items()))
                    print("  Relationship:", dict(record["r"].items()))
                    print("  TKG Node:", dict(record["tkg_event"].items()))
                else:
                    print("[Debug] Could not find the newly written alignment relationship! Please check the consistency of the node name/type and TKG id.")
        except Exception as e:
            print(f"An error occurred while fusing alignment relationship: KG({kg_name}) -> TKG({tkg_id}). Error: {e}")

    print("Fusion process complete.")
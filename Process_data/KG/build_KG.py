import os
import json
from neo4j import GraphDatabase
from typing import Dict, List, Any

class CancerKnowledgeGraph:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """Initialize Neo4j connection"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        
    def close(self):
        """Close the database connection"""
        self.driver.close()
        
    def clear_database(self):
        """Clear the database"""
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
            
    def create_constraints(self):
        """Create constraints"""
        # Removed constraint creation because different documents might contain the same cancer name
        pass
        
    def process_csco_guideline(self, file_path: str):
        """Process CSCO guideline file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Validate cancer name
        cancer_name = data["cancer_info"]["cancer_name"]
        # Safe type checking and string handling
        if isinstance(cancer_name, list):
            cancer_name = cancer_name[0] if cancer_name else ""
        elif not isinstance(cancer_name, str):
            cancer_name = str(cancer_name) if cancer_name else ""
            
        if not cancer_name or cancer_name.strip() == "":
            print(f"Warning: Cancer name in file {file_path} is empty, skipping")
            return
            
        with self.driver.session(database=self.database) as session:
            with session.begin_transaction() as tx:
                # Create or get the Cancer node
                tx.run(
                    """
                    MERGE (c:Cancer {name: $name})
                    ON CREATE SET 
                        c.source = ['CSCO'],
                        c.type = 'Cancer'
                    ON MATCH SET 
                        c.source = CASE 
                            WHEN NOT 'CSCO' IN c.source THEN c.source + 'CSCO'
                            ELSE c.source
                        END
                    """,
                    name=cancer_name
                )
                
                # Process clinical recommendations
                for rec in data["clinical_recommendations"]:
                    # Create ClinicalSituation node
                    tx.run(
                        """
                        MATCH (c:Cancer {name: $cancer_name})
                        MERGE (s:ClinicalSituation {name: $situation_name})
                        ON CREATE SET 
                            s.stage = $stage,
                            s.source = 'CSCO',
                            s.type = 'ClinicalSituation'
                        ON MATCH SET 
                            s.stage = CASE WHEN $stage IS NOT NULL THEN $stage ELSE s.stage END
                        MERGE (c)-[:HAS_CLINICAL_SITUATION]->(s)
                        """,
                        cancer_name=cancer_name,
                        situation_name=rec["clinical_context"],
                        stage=rec["clinical_context"].split("(")[0] if "(" in rec["clinical_context"] else rec["clinical_context"]
                    )
                    
                    # Create Treatment node
                    tx.run(
                        """
                        MATCH (s:ClinicalSituation {name: $situation_name})
                        MERGE (t:Treatment {name: $treatment_name})
                        ON CREATE SET 
                            t.content = $content,
                            t.line = $line,
                            t.evidence_level = $evidence_level,
                            t.recommendation_level = $recommendation_level,
                            t.source = 'CSCO',
                            t.type = 'Treatment'
                        ON MATCH SET 
                            t.content = CASE WHEN $content IS NOT NULL THEN $content ELSE t.content END,
                            t.line = CASE WHEN $line IS NOT NULL THEN $line ELSE t.line END,
                            t.evidence_level = CASE WHEN $evidence_level IS NOT NULL THEN $evidence_level ELSE t.evidence_level END,
                            t.recommendation_level = CASE WHEN $recommendation_level IS NOT NULL THEN $recommendation_level ELSE t.recommendation_level END
                        MERGE (s)-[:HAS_TREATMENT]->(t)
                        """,
                        situation_name=rec["clinical_context"],
                        treatment_name=rec["recommendation_content"],
                        content=rec["recommendation_content"],
                        line=rec.get("treatment_line", "Unknown"),
                        evidence_level=rec.get("evidence_level", "Unknown"),
                        recommendation_level=rec.get("recommendation_level", "Unknown")
                    )
                    
                    # Process biomarkers
                    for biomarker in rec.get("biomarker_requirements", []):
                        tx.run(
                            """
                            MATCH (t:Treatment {name: $treatment_name})
                            MERGE (b:Biomarker {name: $bio_name})
                            ON CREATE SET 
                                b.status = $status,
                                b.guidance = $guidance,
                                b.source = 'CSCO',
                                b.type = 'Biomarker'
                            ON MATCH SET 
                                b.status = CASE WHEN $status IS NOT NULL THEN $status ELSE b.status END,
                                b.guidance = CASE WHEN $guidance IS NOT NULL THEN $guidance ELSE b.guidance END
                            MERGE (t)-[:REQUIRES_BIOMARKER]->(b)
                            """,
                            treatment_name=rec["recommendation_content"],
                            bio_name=biomarker.get("biomarker_name", "Unknown"),
                            status=biomarker.get("status", "Unknown"),
                            guidance=biomarker.get("testing_guidance", "Unknown")
                        )
                        
    def process_esmo_guideline(self, file_path: str):
        """Process ESMO guideline file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Validate cancer name
        cancer_name = data["cancer_focus"]["primary_cancer"]
        # Safe type checking and string handling
        if isinstance(cancer_name, list):
            cancer_name = cancer_name[0] if cancer_name else ""
        elif not isinstance(cancer_name, str):
            cancer_name = str(cancer_name) if cancer_name else ""
            
        if not cancer_name or cancer_name.strip() == "":
            print(f"Warning: Cancer name in file {file_path} is empty, skipping")
            return
            
        with self.driver.session(database=self.database) as session:
            with session.begin_transaction() as tx:
                # Create or get the Cancer node
                tx.run(
                    """
                    MERGE (c:Cancer {name: $name})
                    ON CREATE SET 
                        c.source = ['ESMO'],
                        c.type = 'Cancer'
                    ON MATCH SET 
                        c.source = CASE 
                            WHEN NOT 'ESMO' IN c.source THEN c.source + 'ESMO'
                            ELSE c.source
                        END
                    """,
                    name=cancer_name
                )
                
                # Process staging treatment plans
                for stage in data["staging_treatment_plans"]:
                    for treatment_group in stage["treatment_plans"]:
                        # Create ClinicalSituation node
                        tx.run(
                            """
                            MATCH (c:Cancer {name: $cancer_name})
                            MERGE (s:ClinicalSituation {name: $situation_name})
                            ON CREATE SET 
                                s.stage = $stage,
                                s.risk_group = $risk_group,
                                s.source = 'ESMO',
                                s.type = 'ClinicalSituation'
                            ON MATCH SET 
                                s.stage = CASE WHEN $stage IS NOT NULL THEN $stage ELSE s.stage END,
                                s.risk_group = CASE WHEN $risk_group IS NOT NULL THEN $risk_group ELSE s.risk_group END
                            MERGE (c)-[:HAS_CLINICAL_SITUATION]->(s)
                            """,
                            cancer_name=cancer_name,
                            situation_name=treatment_group["clinical_context"],
                            stage=stage.get("staging_criteria", "Unknown"),
                            risk_group=stage.get("risk_group", "Unknown")
                        )
                        
                        # Create Treatment node
                        tx.run(
                            """
                            MATCH (s:ClinicalSituation {name: $situation_name})
                            MERGE (t:Treatment {name: $treatment_name})
                            ON CREATE SET 
                                t.content = $content,
                                t.line = $line,
                                t.evidence_level = $evidence_level,
                                t.source = 'ESMO',
                                t.type = 'Treatment'
                            ON MATCH SET 
                                t.content = CASE WHEN $content IS NOT NULL THEN $content ELSE t.content END,
                                t.line = CASE WHEN $line IS NOT NULL THEN $line ELSE t.line END,
                                t.evidence_level = CASE WHEN $evidence_level IS NOT NULL THEN $evidence_level ELSE t.evidence_level END
                            MERGE (s)-[:HAS_TREATMENT]->(t)
                            """,
                            situation_name=treatment_group["clinical_context"],
                            treatment_name=treatment_group["recommendation_content"],
                            content=treatment_group["recommendation_content"],
                            line=treatment_group.get("treatment_line", "Unknown"),
                            evidence_level=treatment_group.get("esmo_evidence_level", "Unknown")
                        )
                        
                        # Process biomarkers
                        for biomarker in treatment_group.get("biomarker_requirements", []):
                            tx.run(
                                """
                                MATCH (t:Treatment {name: $treatment_name})
                                MERGE (b:Biomarker {name: $bio_name})
                                ON CREATE SET 
                                    b.status = $status,
                                    b.guidance = $guidance,
                                    b.source = 'ESMO',
                                    b.type = 'Biomarker'
                                ON MATCH SET 
                                    b.status = CASE WHEN $status IS NOT NULL THEN $status ELSE b.status END,
                                    b.guidance = CASE WHEN $guidance IS NOT NULL THEN $guidance ELSE b.guidance END
                                MERGE (t)-[:REQUIRES_BIOMARKER]->(b)
                                """,
                                treatment_name=treatment_group["recommendation_content"],
                                bio_name=biomarker.get("name", "Unknown"),
                                status=biomarker.get("status", "Unknown"),
                                guidance=biomarker.get("testing_guidance", "Unknown")
                            )
                            
    def process_nccn_guideline(self, file_path: str):
        """Process NCCN guideline file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Validate cancer name
        cancer_name = data["cancer_info"]["cancer_name"]
        # Safe type checking and string handling
        if isinstance(cancer_name, list):
            cancer_name = cancer_name[0] if cancer_name else ""
        elif not isinstance(cancer_name, str):
            cancer_name = str(cancer_name) if cancer_name else ""
            
        if not cancer_name or cancer_name.strip() == "":
            print(f"Warning: Cancer name in file {file_path} is empty, skipping")
            return
            
        with self.driver.session(database=self.database) as session:
            with session.begin_transaction() as tx:
                # Create or get the Cancer node
                tx.run(
                    """
                    MERGE (c:Cancer {name: $name})
                    ON CREATE SET 
                        c.source = ['NCCN'],
                        c.type = 'Cancer'
                    ON MATCH SET 
                        c.source = CASE 
                            WHEN NOT 'NCCN' IN c.source THEN c.source + 'NCCN'
                            ELSE c.source
                        END
                    """,
                    name=cancer_name
                )
                
                # Process examination items
                if "diagnosis_recommendations" in data and "examinations" in data["diagnosis_recommendations"]:
                    for exam in data["diagnosis_recommendations"]["examinations"]:
                        tx.run(
                            """
                            MATCH (c:Cancer {name: $cancer_name})
                            MERGE (e:Examination {name: $exam_name})
                            ON CREATE SET 
                                e.source = 'NCCN',
                                e.type = 'Examination'
                            MERGE (c)-[:HAS_EXAMINATION]->(e)
                            """,
                            cancer_name=cancer_name,
                            exam_name=exam
                        )
                
                # Process staging treatment plans
                for stage in data["staging_treatment_plans"]:
                    for treatment_group in stage["treatment_plans"]:
                        # Create ClinicalSituation node
                        tx.run(
                            """
                            MATCH (c:Cancer {name: $cancer_name})
                            MERGE (s:ClinicalSituation {name: $situation_name})
                            ON CREATE SET 
                                s.stage = $stage,
                                s.risk_group = $risk_group,
                                s.source = 'NCCN',
                                s.type = 'ClinicalSituation'
                            ON MATCH SET 
                                s.stage = CASE WHEN $stage IS NOT NULL THEN $stage ELSE s.stage END,
                                s.risk_group = CASE WHEN $risk_group IS NOT NULL THEN $risk_group ELSE s.risk_group END
                            MERGE (c)-[:HAS_CLINICAL_SITUATION]->(s)
                            """,
                            cancer_name=cancer_name,
                            situation_name=stage["staging_criteria"],
                            stage=stage.get("staging_criteria", "Unknown"),
                            risk_group=stage.get("risk_group", "Unknown")
                        )
                        
                        # Create Treatment node
                        tx.run(
                            """
                            MATCH (s:ClinicalSituation {name: $situation_name})
                            MERGE (t:Treatment {name: $treatment_name})
                            ON CREATE SET 
                                t.content = $content,
                                t.line = $line,
                                t.evidence_level = $evidence_level,
                                t.recommendation_level = $recommendation_level,
                                t.source = 'NCCN',
                                t.type = 'Treatment'
                            ON MATCH SET 
                                t.content = CASE WHEN $content IS NOT NULL THEN $content ELSE t.content END,
                                t.line = CASE WHEN $line IS NOT NULL THEN $line ELSE t.line END,
                                t.evidence_level = CASE WHEN $evidence_level IS NOT NULL THEN $evidence_level ELSE t.evidence_level END,
                                t.recommendation_level = CASE WHEN $recommendation_level IS NOT NULL THEN $recommendation_level ELSE t.recommendation_level END
                            MERGE (s)-[:HAS_TREATMENT]->(t)
                            """,
                            situation_name=stage["staging_criteria"],
                            treatment_name=treatment_group.get("plan_name", treatment_group.get("recommendation_content", "Unknown")),
                            content=treatment_group.get("plan_details", treatment_group.get("recommendation_content", "Unknown")),
                            line=treatment_group.get("treatment_line", "Unknown"),
                            evidence_level=treatment_group.get("nccn_evidence_category", treatment_group.get("esmo_evidence_level", "Unknown")),
                            recommendation_level=treatment_group.get("nccn_recommendation_category", treatment_group.get("recommendation_level", "Unknown"))
                        )
                        
    def build_knowledge_graph(self):
        """Build the knowledge graph"""
        try:
            # Clear the database
            self.clear_database()
            
            # Process CSCO guidelines
            csco_dir = "./process_guideline/CSCO"
            for file in os.listdir(csco_dir):
                if file.endswith(".json"):
                    self.process_csco_guideline(os.path.join(csco_dir, file))
                    
            # Process ESMO guidelines
            esmo_dir = "./process_guideline/ESMO"
            for file in os.listdir(esmo_dir):
                if file.endswith(".json"):
                    self.process_esmo_guideline(os.path.join(esmo_dir, file))
                    
            # Process NCCN guidelines
            nccn_dir = "./process_guideline/NCCN"
            for file in os.listdir(nccn_dir):
                if file.endswith(".json"):
                    self.process_nccn_guideline(os.path.join(nccn_dir, file))
                    
        finally:
            self.close()

if __name__ == "__main__":
    # Neo4j connection configuration
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"  # Username
    NEO4J_PASSWORD = "your_password"  # Password
    DATABASE_NAME = "newcancer"  # Database name
    
    try:
        # Create a KnowledgeGraph instance
        kg = CancerKnowledgeGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, DATABASE_NAME)
        
        # Test connection and check the database
        with kg.driver.session() as session:
            # Check if the database exists
            result = session.run("SHOW DATABASES")
            databases = [record["name"] for record in result]
            print(f"Available databases: {databases}")
            
            if DATABASE_NAME not in databases:
                print(f"Database '{DATABASE_NAME}' does not exist, creating it...")
                # Create the database (requires admin privileges)
                try:
                    session.run(f"CREATE DATABASE {DATABASE_NAME}")
                    print(f"Database '{DATABASE_NAME}' created successfully!")
                except Exception as create_error:
                    print(f"Failed to create database: {create_error}")
                    print("Please create the database manually in Neo4j Desktop, or use the default database 'neo4j'")
                    DATABASE_NAME = "neo4j"
                    # Recreate the instance using the default database
                    kg = CancerKnowledgeGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, DATABASE_NAME)
            
            # Test connection to the specified database
            with kg.driver.session(database=DATABASE_NAME) as test_session:
                result = test_session.run("RETURN 1 as test")
                print(f"Neo4j connection successful! Using database: {DATABASE_NAME}")
        
        # Build the knowledge graph
        kg.build_knowledge_graph()
        
        print("Knowledge graph construction complete!")
        
    except Exception as e:
        print(f"An error occurred while connecting to the Neo4j database: {e}")
        print("Please check the following configurations:")
        print("1. Whether the Neo4j database is running")
        print("2. Whether the username and password are correct")
        print("3. Whether port 7687 is accessible")
        print("4. If using Neo4j Desktop, ensure the database is started")
        print("\nCommon solutions:")
        print("- The default username is usually 'neo4j'")
        print("- For a fresh install, the default password might be 'neo4j', which you will be required to change")
        print("- Check the database status in Neo4j Desktop")
        print("- Ensure you have privileges to create databases")
    finally:
        if 'kg' in locals():
            kg.close()
from neo4j import GraphDatabase

# Connection Information
uri = "bolt://localhost:7687"
user = "neo4j"
password = "your_password_here"  # Change to your password
database = "newcancer"  # Specify the database name

driver = GraphDatabase.driver(uri, auth=(user, password))

def get_all_labels(tx):
    """Gets all labels from the database."""
    result = tx.run("CALL db.labels()")
    return [record["label"] for record in result]

def get_all_relationship_types(tx):
    """Gets all relationship types from the database."""
    result = tx.run("CALL db.relationshipTypes()")
    return [record["relationshipType"] for record in result]

def get_entity_properties(tx):
    """Get property information for each entity (only property names, not specific values)."""
    query = """
    MATCH (n)
    RETURN DISTINCT labels(n) as labels, keys(properties(n)) as prop_names
    """
    return [record for record in tx.run(query)]

def get_kg_triples(tx):
    """Get the KG part centered on Cancer (excluding patient-related relationships)."""
    query = """
    MATCH (a)-[r]->(b)
    WHERE NOT 'Patient' IN labels(a) AND NOT 'Patient' IN labels(b)
    RETURN DISTINCT labels(a) as start_labels, type(r) as rel_type, labels(b) as end_labels
    """
    return [record for record in tx.run(query)]

def get_tkg_triples(tx):
    """Get the TKG part centered on Patients (including patient-related relationships)."""
    query = """
    MATCH (a)-[r]->(b)
    WHERE 'Patient' IN labels(a) OR 'Patient' IN labels(b) 
       OR 'HospitalAdmission' IN labels(a) OR 'HospitalAdmission' IN labels(b)
       OR 'Diagnosis' IN labels(a) OR 'Diagnosis' IN labels(b)
       OR 'Treatment' IN labels(a) OR 'Treatment' IN labels(b)
       OR 'Medication' IN labels(a) OR 'Medication' IN labels(b)
       OR 'LabResult' IN labels(a) OR 'LabResult' IN labels(b)
       OR 'Procedure' IN labels(a) OR 'Procedure' IN labels(b)
       OR 'Encounter' IN labels(a) OR 'Encounter' IN labels(b)
       OR 'Symptom' IN labels(a) OR 'Symptom' IN labels(b)
       OR 'VitalSign' IN labels(a) OR 'VitalSign' IN labels(b)
    RETURN DISTINCT labels(a) as start_labels, type(r) as rel_type, labels(b) as end_labels
    """
    return [record for record in tx.run(query)]

with driver.session(database=database) as session:
    labels = session.read_transaction(get_all_labels)
    rel_types = session.read_transaction(get_all_relationship_types)
    entity_props = session.read_transaction(get_entity_properties)
    kg_triples = session.read_transaction(get_kg_triples)
    tkg_triples = session.read_transaction(get_tkg_triples)
    
    print("All node types (labels):", labels)
    print("All relationship types:", rel_types)
    print("Entity property information:", entity_props)
    print("KG triples (centered on Cancer):", kg_triples)
    print("TKG triples (centered on Patient):", tkg_triples)

    # Save KG and TKG triples to txt
    with open("entities_and_relations.txt", "w", encoding="utf-8") as f:
        f.write("Entity Property Descriptions:\n")
        f.write("=" * 50 + "\n")
        for entity in entity_props:
            labels = entity['labels']
            prop_names = entity['prop_names']
            f.write(f"Entity Type: {labels}\n")
            f.write(f"Property Names: {prop_names}\n")
            f.write("-" * 30 + "\n")
        
        f.write("\nKG (centered on Cancer):\n")
        f.write("=" * 50 + "\n")
        for triple in kg_triples:
            start_labels = triple['start_labels']
            rel_type = triple['rel_type']
            end_labels = triple['end_labels']
            f.write(f"({start_labels}, {rel_type}, {end_labels})\n")
        
        f.write("\nTKG (centered on Patient):\n")
        f.write("=" * 50 + "\n")
        for triple in tkg_triples:
            start_labels = triple['start_labels']
            rel_type = triple['rel_type']
            end_labels = triple['end_labels']
            f.write(f"({start_labels}, {rel_type}, {end_labels})\n")
    
    print("Entity properties, KG, and TKG triples have been saved to entities_and_relations.txt")

driver.close()
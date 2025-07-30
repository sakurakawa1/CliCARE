* **main.py**
    The main entry point of the program. It orchestrates the entire knowledge graph alignment workflow, including data loading, entity and relation alignment, fusion, and evaluation. Supports various alignment strategies and parameter configurations, suitable for batch experiments and workflow automation.

* **alignment.py**
    Implements the core entity alignment algorithms. This module calls BERT for semantic similarity calculation, uses an LLM to rerank the results, and then applies BootEA for expansion. It serves as the main logic module for the knowledge graph alignment process.

* **neo4j_inspect.py**
    Provides functions for interacting with the Neo4j graph database, including data import, querying, and structure checking. This facilitates the validation of the knowledge graph before and after the alignment process.

* **data_structures.py**
    Defines the core data structures used in the knowledge graph alignment process, such as entities, relations, and alignment pairs. This facilitates data transfer and management between different modules.

* **fusion.py**
    Implements the process of writing the alignment relationships to the database.

* **graph_connector.py**
    Implements the connection and operational interface for Neo4j. It supports batch data writing, querying, and structure management.

* **trajectory_extraction.py**
    Implements the logic for extracting trajectories (such as patient visit pathways, event sequences, etc.) from the knowledge graph to prepare for subsequent operations.

* **config.py**
    Stores global configuration parameters, such as database connections, file paths, and alignment parameters, for easy and centralized management and modification.

* **repair.py**
    Since calling the large model for reranking may cause some distortion in node information, this script uses the authentic KG data to repair the nodes. It automatically detects and corrects anomalies or omissions that occur during the alignment process.

---
The alignment results from the initial semantic calculation are saved in the `Bert_Align` directory. The results after LLM reranking are in `LLM_Align`. The results after BootEA expansion are in `BootEA_Align`. The contents of `print3` represent the final alignment relationships that are written to the knowledge graph.
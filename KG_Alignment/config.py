# FILE: config.py
# Description: Contains all configuration information for the project

# --- Neo4j Database Connection Configuration ---
# Please update the following information according to your Neo4j instance
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password_here" # Please replace with your password
NEO4J_DATABASE = "database"  # Specify the name of the database to use

# --- Large Language Model (LLM) API Configuration ---
# Please provide the API key according to your LLM service provider
# If using OpenAI, fill in OPENAI_API_KEY
# If using another service compatible with the OpenAI interface, you may also need to fill in OPENAI_BASE_URL
OPENAI_API_KEY = "sk-XXX" # Please replace with your API Key
OPENAI_BASE_URL = "https://api.deepseek.com" # e.g., "https://api.openai.com/v1" or another proxy address
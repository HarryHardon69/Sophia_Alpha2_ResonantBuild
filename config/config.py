# config/config.py

"""Configuration settings for Sophia_Alpha2_ResonantBuild."""

import os

# --- Path Configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOGS_DIR = os.path.join(DATA_DIR, 'logs')
# Add other necessary paths here, e.g., for models, personas, etc.
# PERSONAS_DIR = os.path.join(DATA_DIR, 'personas')
# MEMORY_STORE_DIR = os.path.join(DATA_DIR, 'memory_store')

# --- Resource Management ---
# Placeholders for CPU/GPU allocation, memory limits, etc.
MAX_CPU_CORES = None  # None for auto, or specify an integer
USE_GPU = True       # or False, or specify GPU ID
GPU_MEMORY_LIMIT = None # In MB

# --- System Behavior Flags ---
DEBUG_MODE = True
VERBOSE_LOGGING = True
AUTO_UPDATE_KB = False # Whether the system can update its core knowledge base automatically

# --- API Keys & Endpoints ---
# Structure for API keys (actual keys should be stored in .env or similar, not here)
API_KEYS = {
    'openai': 'YOUR_OPENAI_API_KEY_HERE', # Example
    'another_service': 'YOUR_OTHER_API_KEY_HERE' # Example
}
SERVICE_ENDPOINTS = {
    'language_model_api': 'https://api.example.com/v1/language', # Example
    'knowledge_graph_api': 'https://api.example.com/v1/kg'    # Example
}

# --- Persona Configuration ---
# Basic persona settings, can be expanded significantly
DEFAULT_PERSONA = 'Sophia_Alpha2_Core'
PERSONA_CONFIG_PATH = os.path.join(DATA_DIR, 'personas', DEFAULT_PERSONA + '.json') # Example

# --- Ethics Module Configuration ---
ETHICS_MODULE_ENABLED = True
ETHICS_CONFIG_PATH = os.path.join(DATA_DIR, 'ethics_store', 'ethics_config.json') # Example
ETHICAL_FRAMEWORK_VERSION = "1.0.0"

# --- SNN Configuration (Spacetime Manifold - Brain) ---
# Placeholders for SNN specific settings
SNN_MODEL_PATH = os.path.join(BASE_DIR, 'core', 'models', 'snn_spacetime_manifold_v1.pt') # Example
SNN_NEURON_COUNT = 1000000 # Example
SNN_SYNAPTIC_DENSITY = 0.1  # Example

# --- Memory Configuration ---
MEMORY_TYPE = 'hybrid' # e.g., 'local_json', 'kg_database', 'hybrid'
KNOWLEDGE_GRAPH_PATH = os.path.join(DATA_DIR, 'memory_store', 'knowledge_graph.json') # Example
MEMORY_LOG_PATH = os.path.join(DATA_DIR, 'memory_store', 'memory_log.json') # Example

# --- Dialogue System Configuration ---
DEFAULT_LANGUAGE = 'en-US'
MAX_DIALOGUE_HISTORY = 50 # Number of turns to keep in short-term memory

# --- GUI Configuration ---
GUI_THEME = 'dark' # 'light' or 'dark'
STREAMLIT_PORT = 8501

# --- Logging Configuration ---
LOG_FILE = os.path.join(LOGS_DIR, 'sophia_alpha2.log')
LOG_LEVEL = 'INFO' # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# --- Other Configurations ---
# Add other specific module configurations as needed

if __name__ == '__main__':
    # This part is usually for testing the config values
    print(f"Base Directory: {BASE_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Log File Path: {LOG_FILE}")
    print(f"Default Persona: {DEFAULT_PERSONA}")
    print(f"Debug Mode: {DEBUG_MODE}")

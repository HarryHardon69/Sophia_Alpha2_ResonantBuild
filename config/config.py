"""
Configuration module for Sophia_Alpha2_ResonantBuild.

This module centralizes all system-wide configuration settings, including
paths, resource profiles, API keys, persona details, ethical framework
parameters, and other operational flags.
"""

# TODO: Document cross-module parameter dependencies. Many parameters defined here
# are used by other modules (e.g., core, interface). Understanding these
# dependencies is crucial for system stability and maintainability.
# This could be a separate markdown document or inline comments where parameters
# are defined, explaining which modules consume them.

# TODO: Explore options for dynamic configuration loading (e.g., from a file or a configuration server) to allow updates without restarting the application. This could also involve a mechanism to signal modules to reload their configuration.

import json
import os
import sys

# Sophia_Alpha2_ResonantBuild config starts here.

# --- Path Configuration ---
# Determine if running in a bundled environment (e.g., PyInstaller).
# PyInstaller sets sys._MEIPASS to the path of the bundled temporary folder.
IS_BUNDLED = hasattr(sys, '_MEIPASS')
_PROJECT_ROOT = os.path.dirname(os.path.abspath(sys.executable if IS_BUNDLED else __file__))
if not IS_BUNDLED: # If not bundled, __file__ is in config/, so go up one level
    _PROJECT_ROOT = os.path.dirname(_PROJECT_ROOT)

def get_path(relative_path: str) -> str:
    """
    Constructs an absolute path from a path relative to the project root.
    Handles bundled application scenarios.
    """
    # If already absolute, return as is. Useful if some paths are configured absolutely.
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(_PROJECT_ROOT, relative_path)

def ensure_path(file_or_dir_path: str) -> None:
    """
    Ensures that the directory for the given file path or the directory itself exists.
    If it's a file path, it ensures the parent directory exists.
    If it's a directory path (conventionally passed ending with os.sep or os.altsep),
    it ensures the directory itself exists.
    """
    path_to_ensure = file_or_dir_path
    # Check if the path is intended as a file path (does not end with a known separator)
    # or a directory path (ends with a separator).
    # os.altsep is checked for cross-platform compatibility (e.g. Windows '\').
    if not (file_or_dir_path.endswith(os.sep) or \
            (os.altsep and file_or_dir_path.endswith(os.path.altsep))):
        # Path does not end with a separator, assume it's a file path.
        # Ensure its parent directory exists.
        path_to_ensure = os.path.dirname(file_or_dir_path)
    
    if path_to_ensure: # Ensure path_to_ensure is not empty (e.g. if file_or_dir_path was just a filename in the CWD)
        try:
            os.makedirs(path_to_ensure, exist_ok=True)
        except OSError as e:
            # This is a critical failure if essential directories cannot be created.
            # Print to stderr and allow the program to decide if it can continue.
            # In many cases, this might lead to a crash shortly after if paths are unusable.
            print(f"CRITICAL CONFIG ERROR: Could not create directory '{path_to_ensure}'. Error: {e}", file=sys.stderr)
            # Depending on severity and application design, one might:
            # raise ConfigError(f"Failed to create directory: {path_to_ensure}") from e
            # or sys.exit(1)
            # For now, just printing allows other modules to fail if the path is truly needed.

# Core Directories
CONFIG_DIR = get_path('config')
DATA_DIR = get_path('data')
LOG_DIR = get_path(os.path.join('data', 'logs'))
PERSONA_DIR = get_path(os.path.join('data', 'personas'))
MEMORY_STORE_DIR = get_path(os.path.join('data', 'memory_store')) # For KGraph, MemLog
LIBRARY_STORE_DIR = get_path(os.path.join('data', 'library_store')) # For LibLog
ETHICS_STORE_DIR = get_path(os.path.join('data', 'ethics_store')) # For EthicsDB

# --- Persona Configuration ---
# Defines the active persona for Sophia_Alpha2.
# The persona profile file (JSON) contains detailed personality traits, communication style, etc.
PERSONA_NAME = os.getenv('PERSONA_NAME', 'Sophia_Alpha2_Prime')
# --- End of Persona Configuration ---

# Specific File Paths
# Now PERSONA_NAME is defined and can be used here
PERSONA_PROFILE_PATH = get_path(os.path.join(PERSONA_DIR, PERSONA_NAME + '.json'))

SYSTEM_LOG_FILENAME = "sophia_alpha2_system.log"
# TODO: Implement log sanitization for SYSTEM_LOG_PATH to prevent leakage of sensitive information through logs. Consider what constitutes sensitive data in this context.
# TODO: Implement log rotation for SYSTEM_LOG_PATH to manage log file sizes and prevent excessive disk usage, especially in long-running deployments.
SYSTEM_LOG_PATH = get_path(os.path.join(LOG_DIR, SYSTEM_LOG_FILENAME))
# Future enhancements for logging could include log rotation (e.g., size/time based) and asynchronous logging for improved performance.

ETHICS_DB_FILENAME = "ethics_db.json"
ETHICS_DB_PATH = get_path(os.path.join(ETHICS_STORE_DIR, ETHICS_DB_FILENAME))

KNOWLEDGE_GRAPH_FILENAME = "knowledge_graph.json"
# TODO: Consider adding encryption options for persistent JSON files like knowledge_graph.json to protect sensitive data at rest.
KNOWLEDGE_GRAPH_PATH = get_path(os.path.join(MEMORY_STORE_DIR, KNOWLEDGE_GRAPH_FILENAME))

MEMORY_LOG_FILENAME = "memory_log.json" # General memory log
MEMORY_LOG_PATH = get_path(os.path.join(MEMORY_STORE_DIR, MEMORY_LOG_FILENAME))

LIBRARY_LOG_FILENAME = "library_log.json"
LIBRARY_LOG_PATH = get_path(os.path.join(LIBRARY_STORE_DIR, LIBRARY_LOG_FILENAME))

# Future enhancement: Flag to enable encryption for the library log.
ENCRYPT_LIBRARY_LOG = os.getenv('ENCRYPT_LIBRARY_LOG', 'False').lower() == 'true'

# Initial ensure_path calls for essential directories at import time
ensure_path(DATA_DIR + os.sep) # Ensure DATA_DIR itself is created
ensure_path(LOG_DIR + os.sep)
ensure_path(PERSONA_DIR + os.sep)
ensure_path(MEMORY_STORE_DIR + os.sep)
ensure_path(LIBRARY_STORE_DIR + os.sep)
ensure_path(ETHICS_STORE_DIR + os.sep)
# Individual log files' directories will be ensured when they are defined or via a logging setup function.
# For now, ensuring the main log directory is key.
ensure_path(SYSTEM_LOG_PATH) # Ensures LOG_DIR is created
ensure_path(PERSONA_PROFILE_PATH)    # Ensures PERSONA_DIR is created
ensure_path(ETHICS_DB_PATH)          # Ensures ETHICS_STORE_DIR is created
ensure_path(KNOWLEDGE_GRAPH_PATH)    # Ensures MEMORY_STORE_DIR is created
ensure_path(MEMORY_LOG_PATH)         # Ensures MEMORY_STORE_DIR is created (again, harmless)
ensure_path(LIBRARY_LOG_PATH)        # Ensures LIBRARY_STORE_DIR is created

# --- End of Path Configuration ---

# TODO: Consider implementing lazy loading for non-critical configuration parameters. This could improve startup time by deferring the loading of some settings until they are actually needed.

# --- Resource Management ---
# Defines different operational intensity profiles for the SNN and other components.
# Can be set via environment variable RESOURCE_PROFILE, otherwise defaults to "moderate".
RESOURCE_PROFILE_TYPE = os.getenv('RESOURCE_PROFILE', 'moderate').lower()

# TODO: Document the expected resource usage (CPU, memory, network) for each profile (low, moderate, high) to help users select the appropriate one for their environment.
_RESOURCE_PROFILES = {
    "low": {
        "MAX_NEURONS": 50000,       # Max neurons in the SNN spacetime manifold
        "MAX_SPIKE_RATE": 10,       # Target average spikes per second per neuron
        "RESOLUTION": 0.5,          # Spatial resolution of the manifold (e.g., 0.1 to 1.0)
        "SNN_TIME_STEPS": 100,      # Number of time steps for SNN simulation per cycle
    },
    "moderate": {
        "MAX_NEURONS": 200000,
        "MAX_SPIKE_RATE": 20,
        "RESOLUTION": 0.75,
        "SNN_TIME_STEPS": 250,
    },
    "high": {
        "MAX_NEURONS": 1000000,
        "MAX_SPIKE_RATE": 50,
        "RESOLUTION": 1.0,
        "SNN_TIME_STEPS": 500,
    }
}

if RESOURCE_PROFILE_TYPE not in _RESOURCE_PROFILES:
    print(
        f"Warning: Unknown RESOURCE_PROFILE '{RESOURCE_PROFILE_TYPE}'. "
        f"Defaulting to 'moderate'. Available profiles: {list(_RESOURCE_PROFILES.keys())}"
    )
    RESOURCE_PROFILE_TYPE = 'moderate'

RESOURCE_PROFILE = _RESOURCE_PROFILES[RESOURCE_PROFILE_TYPE]

# Manifold Range: Defines the conceptual 'size' or 'extent' of the SNN spacetime manifold.
# This could influence how concepts are mapped or how far associations can spread.
# Example: A range of 1000 could mean coordinates from -500 to +500.
# This is more conceptual and might be used by various modules mapping to the SNN.
# MANIFOLD_RANGE = 1000.0 # Example value, can be adjusted based on SNN design. # Original
_DEFAULT_MANIFOLD_RANGE = 1000.0
try:
    MANIFOLD_RANGE = float(os.getenv('MANIFOLD_RANGE', _DEFAULT_MANIFOLD_RANGE))
except ValueError as e:
    print(f"Warning: Invalid value for MANIFOLD_RANGE env var ('{os.getenv('MANIFOLD_RANGE')}'). Using default: {_DEFAULT_MANIFOLD_RANGE}. Error: {e}", file=sys.stderr)
    MANIFOLD_RANGE = _DEFAULT_MANIFOLD_RANGE


# --- End of Resource Management ---

# --- System Behavior ---
# Flags and parameters controlling core system operations and SNN characteristics.

# General Behavior Flags
VERBOSE_OUTPUT = os.getenv('VERBOSE_OUTPUT', 'True').lower() == 'true'
ENABLE_SNN = os.getenv('ENABLE_SNN', 'True').lower() == 'true' # Master switch for SNN operations

# SNN Specific Parameters
# These values are placeholders and should be tuned based on SNN model and research.
# Learning rates and STDP (Spike-Timing-Dependent Plasticity) parameters
DEFAULT_HEBBIAN_LEARNING_RATE = 0.005
try:
    HEBBIAN_LEARNING_RATE = float(os.getenv('HEBBIAN_LEARNING_RATE', DEFAULT_HEBBIAN_LEARNING_RATE))
except ValueError as e:
    print(f"Warning: Invalid value for HEBBIAN_LEARNING_RATE env var ('{os.getenv('HEBBIAN_LEARNING_RATE')}'). Using default: {DEFAULT_HEBBIAN_LEARNING_RATE}. Error: {e}", file=sys.stderr)
    HEBBIAN_LEARNING_RATE = DEFAULT_HEBBIAN_LEARNING_RATE

DEFAULT_STDP_LEARNING_RATE = 0.004
try:
    STDP_LEARNING_RATE = float(os.getenv('STDP_LEARNING_RATE', DEFAULT_STDP_LEARNING_RATE))
except ValueError as e:
    print(f"Warning: Invalid value for STDP_LEARNING_RATE env var ('{os.getenv('STDP_LEARNING_RATE')}'). Using default: {DEFAULT_STDP_LEARNING_RATE}. Error: {e}", file=sys.stderr)
    STDP_LEARNING_RATE = DEFAULT_STDP_LEARNING_RATE

DEFAULT_STDP_WINDOW_MS = 20.0
try:
    STDP_WINDOW_MS = float(os.getenv('STDP_WINDOW_MS', DEFAULT_STDP_WINDOW_MS))
except ValueError as e:
    print(f"Warning: Invalid value for STDP_WINDOW_MS env var ('{os.getenv('STDP_WINDOW_MS')}'). Using default: {DEFAULT_STDP_WINDOW_MS}. Error: {e}", file=sys.stderr)
    STDP_WINDOW_MS = DEFAULT_STDP_WINDOW_MS

DEFAULT_STDP_DEPRESSION_FACTOR = 0.0015
try:
    STDP_DEPRESSION_FACTOR = float(os.getenv('STDP_DEPRESSION_FACTOR', DEFAULT_STDP_DEPRESSION_FACTOR))
except ValueError as e:
    print(f"Warning: Invalid value for STDP_DEPRESSION_FACTOR env var ('{os.getenv('STDP_DEPRESSION_FACTOR')}'). Using default: {DEFAULT_STDP_DEPRESSION_FACTOR}. Error: {e}", file=sys.stderr)
    STDP_DEPRESSION_FACTOR = DEFAULT_STDP_DEPRESSION_FACTOR

# Coherence and Resonance Parameters
DEFAULT_COHERENCE_UPDATE_FACTOR = 0.1
try:
    COHERENCE_UPDATE_FACTOR = float(os.getenv('COHERENCE_UPDATE_FACTOR', DEFAULT_COHERENCE_UPDATE_FACTOR))
except ValueError as e:
    print(f"Warning: Invalid value for COHERENCE_UPDATE_FACTOR env var ('{os.getenv('COHERENCE_UPDATE_FACTOR')}'). Using default: {DEFAULT_COHERENCE_UPDATE_FACTOR}. Error: {e}", file=sys.stderr)
    COHERENCE_UPDATE_FACTOR = DEFAULT_COHERENCE_UPDATE_FACTOR

# SNN/Surrogate Gradient Parameters (relevant for SNNTorch or similar frameworks)
DEFAULT_SNN_SURROGATE_SLOPE = 25.0
try:
    SNN_SURROGATE_SLOPE = float(os.getenv('SNN_SURROGATE_SLOPE', DEFAULT_SNN_SURROGATE_SLOPE))
except ValueError as e:
    print(f"Warning: Invalid value for SNN_SURROGATE_SLOPE env var ('{os.getenv('SNN_SURROGATE_SLOPE')}'). Using default: {DEFAULT_SNN_SURROGATE_SLOPE}. Error: {e}", file=sys.stderr)
    SNN_SURROGATE_SLOPE = DEFAULT_SNN_SURROGATE_SLOPE

DEFAULT_SNN_LIF_BETA = 0.9
try:
    SNN_LIF_BETA = float(os.getenv('SNN_LIF_BETA', DEFAULT_SNN_LIF_BETA))
except ValueError as e:
    print(f"Warning: Invalid value for SNN_LIF_BETA env var ('{os.getenv('SNN_LIF_BETA')}'). Using default: {DEFAULT_SNN_LIF_BETA}. Error: {e}", file=sys.stderr)
    SNN_LIF_BETA = DEFAULT_SNN_LIF_BETA

DEFAULT_SNN_LIF_THRESHOLD = 1.0
try:
    SNN_LIF_THRESHOLD = float(os.getenv('SNN_LIF_THRESHOLD', DEFAULT_SNN_LIF_THRESHOLD))
except ValueError as e:
    print(f"Warning: Invalid value for SNN_LIF_THRESHOLD env var ('{os.getenv('SNN_LIF_THRESHOLD')}'). Using default: {DEFAULT_SNN_LIF_THRESHOLD}. Error: {e}", file=sys.stderr)
    SNN_LIF_THRESHOLD = DEFAULT_SNN_LIF_THRESHOLD

DEFAULT_SNN_OPTIMIZER_LR = 5e-4
try:
    SNN_OPTIMIZER_LR = float(os.getenv('SNN_OPTIMIZER_LR', DEFAULT_SNN_OPTIMIZER_LR))
except ValueError as e:
    print(f"Warning: Invalid value for SNN_OPTIMIZER_LR env var ('{os.getenv('SNN_OPTIMIZER_LR')}'). Using default: {DEFAULT_SNN_OPTIMIZER_LR}. Error: {e}", file=sys.stderr)
    SNN_OPTIMIZER_LR = DEFAULT_SNN_OPTIMIZER_LR

DEFAULT_SNN_INPUT_SIZE = 768
try:
    SNN_INPUT_SIZE = int(os.getenv('SNN_INPUT_SIZE', DEFAULT_SNN_INPUT_SIZE))
except ValueError as e:
    print(f"Warning: Invalid value for SNN_INPUT_SIZE env var ('{os.getenv('SNN_INPUT_SIZE')}'). Using default: {DEFAULT_SNN_INPUT_SIZE}. Error: {e}", file=sys.stderr)
    SNN_INPUT_SIZE = DEFAULT_SNN_INPUT_SIZE

DEFAULT_SNN_BATCH_SIZE = 1
try:
    SNN_BATCH_SIZE = int(os.getenv('SNN_BATCH_SIZE', DEFAULT_SNN_BATCH_SIZE))
except ValueError as e:
    print(f"Warning: Invalid value for SNN_BATCH_SIZE env var ('{os.getenv('SNN_BATCH_SIZE')}'). Using default: {DEFAULT_SNN_BATCH_SIZE}. Error: {e}", file=sys.stderr)
    SNN_BATCH_SIZE = DEFAULT_SNN_BATCH_SIZE

DEFAULT_SNN_INPUT_ACTIVE_FRACTION = 0.1
try:
    SNN_INPUT_ACTIVE_FRACTION = float(os.getenv('SNN_INPUT_ACTIVE_FRACTION', DEFAULT_SNN_INPUT_ACTIVE_FRACTION))
except ValueError as e:
    print(f"Warning: Invalid value for SNN_INPUT_ACTIVE_FRACTION env var ('{os.getenv('SNN_INPUT_ACTIVE_FRACTION')}'). Using default: {DEFAULT_SNN_INPUT_ACTIVE_FRACTION}. Error: {e}", file=sys.stderr)
    SNN_INPUT_ACTIVE_FRACTION = DEFAULT_SNN_INPUT_ACTIVE_FRACTION


# --- End of System Behavior ---

# --- API Keys and Endpoints ---
# TODO: Reference relevant threat models or security assessments in comments throughout this section to provide context for security decisions.
# Configuration for Large Language Models (LLMs) and other external APIs.

ENABLE_LLM_API = os.getenv('ENABLE_LLM_API', 'True').lower() == 'true'

# Select LLM Provider: "openai", "lm_studio", "ollama", "mock_for_snn_test"
# Mock provider can be used for testing SNN without actual LLM calls.
DEFAULT_LLM_PROVIDER = 'lm_studio'
LLM_PROVIDER = os.getenv('LLM_PROVIDER', DEFAULT_LLM_PROVIDER).lower()

DEFAULT_LLM_TEMPERATURE = 0.7
try:
    LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', DEFAULT_LLM_TEMPERATURE))
except ValueError as e:
    print(f"Warning: Invalid value for LLM_TEMPERATURE env var ('{os.getenv('LLM_TEMPERATURE')}'). Using default: {DEFAULT_LLM_TEMPERATURE}. Error: {e}", file=sys.stderr)
    LLM_TEMPERATURE = DEFAULT_LLM_TEMPERATURE

DEFAULT_LLM_CONNECTION_TIMEOUT = 10
try:
    LLM_CONNECTION_TIMEOUT = int(os.getenv('LLM_CONNECTION_TIMEOUT', DEFAULT_LLM_CONNECTION_TIMEOUT)) # seconds
except ValueError as e:
    print(f"Warning: Invalid value for LLM_CONNECTION_TIMEOUT env var ('{os.getenv('LLM_CONNECTION_TIMEOUT')}'). Using default: {DEFAULT_LLM_CONNECTION_TIMEOUT}. Error: {e}", file=sys.stderr)
    LLM_CONNECTION_TIMEOUT = DEFAULT_LLM_CONNECTION_TIMEOUT

DEFAULT_LLM_REQUEST_TIMEOUT = 120
try:
    LLM_REQUEST_TIMEOUT = int(os.getenv('LLM_REQUEST_TIMEOUT', DEFAULT_LLM_REQUEST_TIMEOUT)) # seconds
except ValueError as e:
    print(f"Warning: Invalid value for LLM_REQUEST_TIMEOUT env var ('{os.getenv('LLM_REQUEST_TIMEOUT')}'). Using default: {DEFAULT_LLM_REQUEST_TIMEOUT}. Error: {e}", file=sys.stderr)
    LLM_REQUEST_TIMEOUT = DEFAULT_LLM_REQUEST_TIMEOUT

# Define configurations for each LLM provider
# Sensitive information like API keys should ideally be set as environment variables.
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"
DEFAULT_LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
DEFAULT_LM_STUDIO_MODEL = "local-model"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/api"
DEFAULT_OLLAMA_MODEL = "llama2"

_LLM_CONFIG = {
    "openai": {
        "API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "BASE_URL": os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
        "MODEL": os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        "CONCEPT_PROMPT_TEMPLATE": { # Default prompts, consider making these configurable as well if needed
            "system": "You are an AI assistant helping to define and elaborate on concepts. Provide a concise and informative summary for the given concept. Focus on its core meaning and key aspects.",
            "user": "Concept: {concept_name}\n\nProvide a detailed explanation of this concept, including its primary definition, key characteristics, and typical applications or examples. If the concept is abstract, try to provide analogies."
        }
    },
    "lm_studio": {
        "API_KEY": os.getenv("LM_STUDIO_API_KEY", "not_required"), # Typically not required for local LM Studio
        "BASE_URL": os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"), # Default LM Studio API endpoint
        "MODEL": os.getenv("LM_STUDIO_MODEL", "local-model"), # Model identifier used by LM Studio, often a loaded model name
        "CONCEPT_PROMPT_TEMPLATE": {
            "system": "You are an AI assistant. Your task is to provide a detailed and clear explanation for the given concept. Emphasize its fundamental principles and provide illustrative examples.",
            "user": "Please explain the concept: {concept_name}. Describe its definition, core ideas, and any relevant examples."
        }
    },
    "ollama": {
        "API_KEY": os.getenv("OLLAMA_API_KEY", "not_required"), 
        "BASE_URL": os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL), 
        "MODEL": os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL), 
        "CONCEPT_PROMPT_TEMPLATE": {
            "system": "You are an AI model. Explain the following concept clearly and concisely. Provide its definition, main attributes, and some examples if applicable.",
            "user": "Concept: {concept_name}. Please provide a comprehensive explanation."
        }
    },
    "mock_for_snn_test": { # Mock provider settings
        "API_KEY": "mock_key",
        "BASE_URL": "http://localhost:8000/mock_api", 
        "MODEL": "mock_model",
        "CONCEPT_PROMPT_TEMPLATE": { 
            "system": "Mock system prompt for {concept_name}.",
            "user": "Mock user prompt for {concept_name}."
        }
    }
}

# Set current LLM settings based on LLM_PROVIDER
if LLM_PROVIDER not in _LLM_CONFIG:
    print(
        f"Warning: Unknown LLM_PROVIDER '{LLM_PROVIDER}'. "
        f"Defaulting to '{DEFAULT_LLM_PROVIDER}'. Available providers: {list(_LLM_CONFIG.keys())}"
    )
    LLM_PROVIDER = DEFAULT_LLM_PROVIDER

CURRENT_LLM_SETTINGS = _LLM_CONFIG.get(LLM_PROVIDER) # Use .get for safety, though fallback above should ensure key exists
if CURRENT_LLM_SETTINGS is None: # Should not happen due to fallback, but as an ultimate safeguard
    print(f"CRITICAL: LLM_PROVIDER '{LLM_PROVIDER}' resolved to None in _LLM_CONFIG. This should not happen. Using mock as emergency fallback.", file=sys.stderr)
    CURRENT_LLM_SETTINGS = _LLM_CONFIG["mock_for_snn_test"] # Emergency fallback
    LLM_PROVIDER = "mock_for_snn_test"
LLM_API_KEY = CURRENT_LLM_SETTINGS["API_KEY"]
LLM_BASE_URL = CURRENT_LLM_SETTINGS["BASE_URL"]
LLM_MODEL = CURRENT_LLM_SETTINGS["MODEL"]
LLM_CONCEPT_PROMPT_TEMPLATE = CURRENT_LLM_SETTINGS["CONCEPT_PROMPT_TEMPLATE"]

# --- End of API Keys and Endpoints ---

# --- Ethics Module Configuration ---
# Parameters governing the ethical decision-making and alignment framework.

# Placeholder for the ethical framework. This could be loaded from a JSON file or defined here.
# TODO: Load ETHICAL_FRAMEWORK dynamically (e.g., from ETHICS_DB_PATH or another configuration file) to allow for easier updates without code changes.
# Example structure: { "principle_name": {"weight": float, "rules": ["rule1", "rule2"]}}
ETHICAL_FRAMEWORK = {
    "NonMaleficence": {"weight": 0.8, "description": "Avoid causing harm."},
    "Beneficence": {"weight": 0.6, "description": "Promote well-being."},
    "Autonomy": {"weight": 0.7, "description": "Respect individual autonomy."},
    "Justice": {"weight": 0.5, "description": "Ensure fairness and equity."},
    "Transparency": {"weight": 0.4, "description": "Maintain operational transparency."}
    # This can be expanded significantly based on Phase2KB.md or other sources.
}

# Threshold for overall ethical alignment. Actions below this may be flagged or modified.
DEFAULT_ETHICAL_ALIGNMENT_THRESHOLD = 0.7
try:
    ETHICAL_ALIGNMENT_THRESHOLD = float(os.getenv('ETHICAL_ALIGNMENT_THRESHOLD', DEFAULT_ETHICAL_ALIGNMENT_THRESHOLD))
except ValueError as e:
    print(f"Warning: Invalid value for ETHICAL_ALIGNMENT_THRESHOLD env var ('{os.getenv('ETHICAL_ALIGNMENT_THRESHOLD')}'). Using default: {DEFAULT_ETHICAL_ALIGNMENT_THRESHOLD}. Error: {e}", file=sys.stderr)
    ETHICAL_ALIGNMENT_THRESHOLD = DEFAULT_ETHICAL_ALIGNMENT_THRESHOLD

# Weights for combining different ethical assessment dimensions (e.g., in an ethics score)
DEFAULT_ETHICS_COHERENCE_WEIGHT = 0.25
try:
    ETHICS_COHERENCE_WEIGHT = float(os.getenv('ETHICS_COHERENCE_WEIGHT', DEFAULT_ETHICS_COHERENCE_WEIGHT))
except ValueError as e:
    print(f"Warning: Invalid value for ETHICS_COHERENCE_WEIGHT env var ('{os.getenv('ETHICS_COHERENCE_WEIGHT')}'). Using default: {DEFAULT_ETHICS_COHERENCE_WEIGHT}. Error: {e}", file=sys.stderr)
    ETHICS_COHERENCE_WEIGHT = DEFAULT_ETHICS_COHERENCE_WEIGHT

DEFAULT_ETHICS_VALENCE_WEIGHT = 0.25
try:
    ETHICS_VALENCE_WEIGHT = float(os.getenv('ETHICS_VALENCE_WEIGHT', DEFAULT_ETHICS_VALENCE_WEIGHT))
except ValueError as e:
    print(f"Warning: Invalid value for ETHICS_VALENCE_WEIGHT env var ('{os.getenv('ETHICS_VALENCE_WEIGHT')}'). Using default: {DEFAULT_ETHICS_VALENCE_WEIGHT}. Error: {e}", file=sys.stderr)
    ETHICS_VALENCE_WEIGHT = DEFAULT_ETHICS_VALENCE_WEIGHT

DEFAULT_ETHICS_INTENSITY_WEIGHT = 0.20
try:
    ETHICS_INTENSITY_WEIGHT = float(os.getenv('ETHICS_INTENSITY_WEIGHT', DEFAULT_ETHICS_INTENSITY_WEIGHT))
except ValueError as e:
    print(f"Warning: Invalid value for ETHICS_INTENSITY_WEIGHT env var ('{os.getenv('ETHICS_INTENSITY_WEIGHT')}'). Using default: {DEFAULT_ETHICS_INTENSITY_WEIGHT}. Error: {e}", file=sys.stderr)
    ETHICS_INTENSITY_WEIGHT = DEFAULT_ETHICS_INTENSITY_WEIGHT

DEFAULT_ETHICS_FRAMEWORK_WEIGHT = 0.30
try:
    ETHICS_FRAMEWORK_WEIGHT = float(os.getenv('ETHICS_FRAMEWORK_WEIGHT', DEFAULT_ETHICS_FRAMEWORK_WEIGHT))
except ValueError as e:
    print(f"Warning: Invalid value for ETHICS_FRAMEWORK_WEIGHT env var ('{os.getenv('ETHICS_FRAMEWORK_WEIGHT')}'). Using default: {DEFAULT_ETHICS_FRAMEWORK_WEIGHT}. Error: {e}", file=sys.stderr)
    ETHICS_FRAMEWORK_WEIGHT = DEFAULT_ETHICS_FRAMEWORK_WEIGHT

# Parameters for ethical clustering or contextual analysis
DEFAULT_ETHICS_CLUSTER_CONTEXT_WEIGHT = 0.5
try:
    ETHICS_CLUSTER_CONTEXT_WEIGHT = float(os.getenv('ETHICS_CLUSTER_CONTEXT_WEIGHT', DEFAULT_ETHICS_CLUSTER_CONTEXT_WEIGHT))
except ValueError as e:
    print(f"Warning: Invalid value for ETHICS_CLUSTER_CONTEXT_WEIGHT env var ('{os.getenv('ETHICS_CLUSTER_CONTEXT_WEIGHT')}'). Using default: {DEFAULT_ETHICS_CLUSTER_CONTEXT_WEIGHT}. Error: {e}", file=sys.stderr)
    ETHICS_CLUSTER_CONTEXT_WEIGHT = DEFAULT_ETHICS_CLUSTER_CONTEXT_WEIGHT

DEFAULT_ETHICS_CLUSTER_RADIUS_FACTOR = 1.5
try:
    ETHICS_CLUSTER_RADIUS_FACTOR = float(os.getenv('ETHICS_CLUSTER_RADIUS_FACTOR', DEFAULT_ETHICS_CLUSTER_RADIUS_FACTOR))
except ValueError as e:
    print(f"Warning: Invalid value for ETHICS_CLUSTER_RADIUS_FACTOR env var ('{os.getenv('ETHICS_CLUSTER_RADIUS_FACTOR')}'). Using default: {DEFAULT_ETHICS_CLUSTER_RADIUS_FACTOR}. Error: {e}", file=sys.stderr)
    ETHICS_CLUSTER_RADIUS_FACTOR = DEFAULT_ETHICS_CLUSTER_RADIUS_FACTOR

# Parameters for ethics log and trend analysis
DEFAULT_ETHICS_LOG_MAX_ENTRIES = 10000
try:
    ETHICS_LOG_MAX_ENTRIES = int(os.getenv('ETHICS_LOG_MAX_ENTRIES', DEFAULT_ETHICS_LOG_MAX_ENTRIES))
except ValueError as e:
    print(f"Warning: Invalid value for ETHICS_LOG_MAX_ENTRIES env var ('{os.getenv('ETHICS_LOG_MAX_ENTRIES')}'). Using default: {DEFAULT_ETHICS_LOG_MAX_ENTRIES}. Error: {e}", file=sys.stderr)
    ETHICS_LOG_MAX_ENTRIES = DEFAULT_ETHICS_LOG_MAX_ENTRIES

DEFAULT_ETHICS_TREND_MIN_DATAPOINTS = 50
try:
    ETHICS_TREND_MIN_DATAPOINTS = int(os.getenv('ETHICS_TREND_MIN_DATAPOINTS', DEFAULT_ETHICS_TREND_MIN_DATAPOINTS))
except ValueError as e:
    print(f"Warning: Invalid value for ETHICS_TREND_MIN_DATAPOINTS env var ('{os.getenv('ETHICS_TREND_MIN_DATAPOINTS')}'). Using default: {DEFAULT_ETHICS_TREND_MIN_DATAPOINTS}. Error: {e}", file=sys.stderr)
    ETHICS_TREND_MIN_DATAPOINTS = DEFAULT_ETHICS_TREND_MIN_DATAPOINTS

DEFAULT_ETHICS_TREND_SIGNIFICANCE_THRESHOLD = 0.05
try:
    ETHICS_TREND_SIGNIFICANCE_THRESHOLD = float(os.getenv('ETHICS_TREND_SIGNIFICANCE_THRESHOLD', DEFAULT_ETHICS_TREND_SIGNIFICANCE_THRESHOLD))
except ValueError as e:
    print(f"Warning: Invalid value for ETHICS_TREND_SIGNIFICANCE_THRESHOLD env var ('{os.getenv('ETHICS_TREND_SIGNIFICANCE_THRESHOLD')}'). Using default: {DEFAULT_ETHICS_TREND_SIGNIFICANCE_THRESHOLD}. Error: {e}", file=sys.stderr)
    ETHICS_TREND_SIGNIFICANCE_THRESHOLD = DEFAULT_ETHICS_TREND_SIGNIFICANCE_THRESHOLD

# Threshold for the Mitigator module to intervene based on ethical assessment
DEFAULT_MITIGATION_ETHICAL_THRESHOLD = 0.85
try:
    MITIGATION_ETHICAL_THRESHOLD = float(os.getenv('MITIGATION_ETHICAL_THRESHOLD', DEFAULT_MITIGATION_ETHICAL_THRESHOLD))
except ValueError as e:
    print(f"Warning: Invalid value for MITIGATION_ETHICAL_THRESHOLD env var ('{os.getenv('MITIGATION_ETHICAL_THRESHOLD')}'). Using default: {DEFAULT_MITIGATION_ETHICAL_THRESHOLD}. Error: {e}", file=sys.stderr)
    MITIGATION_ETHICAL_THRESHOLD = DEFAULT_MITIGATION_ETHICAL_THRESHOLD

# --- End of Ethics Module Configuration ---

# --- Memory/Library Configuration ---
# Parameters governing memory operations, novelty detection, and knowledge library interactions.

# Threshold for determining if a new piece of information is novel enough to be stored.
DEFAULT_MEMORY_NOVELTY_THRESHOLD = 0.6
try:
    MEMORY_NOVELTY_THRESHOLD = float(os.getenv('MEMORY_NOVELTY_THRESHOLD', DEFAULT_MEMORY_NOVELTY_THRESHOLD))
except ValueError as e:
    print(f"Warning: Invalid value for MEMORY_NOVELTY_THRESHOLD env var ('{os.getenv('MEMORY_NOVELTY_THRESHOLD')}'). Using default: {DEFAULT_MEMORY_NOVELTY_THRESHOLD}. Error: {e}", file=sys.stderr)
    MEMORY_NOVELTY_THRESHOLD = DEFAULT_MEMORY_NOVELTY_THRESHOLD


# Ethical threshold for storing memories. Can default to the general ethical alignment threshold.
DEFAULT_MEMORY_ETHICAL_THRESHOLD = ETHICAL_ALIGNMENT_THRESHOLD 
try:
    MEMORY_ETHICAL_THRESHOLD = float(os.getenv('MEMORY_ETHICAL_THRESHOLD', DEFAULT_MEMORY_ETHICAL_THRESHOLD))
except ValueError as e:
    print(f"Warning: Invalid value for MEMORY_ETHICAL_THRESHOLD env var ('{os.getenv('MEMORY_ETHICAL_THRESHOLD')}'). Using default: {DEFAULT_MEMORY_ETHICAL_THRESHOLD}. Error: {e}", file=sys.stderr)
    MEMORY_ETHICAL_THRESHOLD = DEFAULT_MEMORY_ETHICAL_THRESHOLD


# Weights for combining different aspects of novelty (e.g., spatial vs. textual content)
DEFAULT_SPATIAL_NOVELTY_WEIGHT = 0.5
try:
    SPATIAL_NOVELTY_WEIGHT = float(os.getenv('SPATIAL_NOVELTY_WEIGHT', DEFAULT_SPATIAL_NOVELTY_WEIGHT))
except ValueError as e:
    print(f"Warning: Invalid value for SPATIAL_NOVELTY_WEIGHT env var ('{os.getenv('SPATIAL_NOVELTY_WEIGHT')}'). Using default: {DEFAULT_SPATIAL_NOVELTY_WEIGHT}. Error: {e}", file=sys.stderr)
    SPATIAL_NOVELTY_WEIGHT = DEFAULT_SPATIAL_NOVELTY_WEIGHT

DEFAULT_TEXTUAL_NOVELTY_WEIGHT = 0.5
try:
    TEXTUAL_NOVELTY_WEIGHT = float(os.getenv('TEXTUAL_NOVELTY_WEIGHT', DEFAULT_TEXTUAL_NOVELTY_WEIGHT))
except ValueError as e:
    print(f"Warning: Invalid value for TEXTUAL_NOVELTY_WEIGHT env var ('{os.getenv('TEXTUAL_NOVELTY_WEIGHT')}'). Using default: {DEFAULT_TEXTUAL_NOVELTY_WEIGHT}. Error: {e}", file=sys.stderr)
    TEXTUAL_NOVELTY_WEIGHT = DEFAULT_TEXTUAL_NOVELTY_WEIGHT


# Flag to determine if explicit consent is required before storing data in public knowledge areas.
REQUIRE_PUBLIC_STORAGE_CONSENT = os.getenv('REQUIRE_PUBLIC_STORAGE_CONSENT', 'True').lower() == 'true'

# Default coherence value assigned to new knowledge items in the library, can be updated by SNN.
DEFAULT_KNOWLEDGE_COHERENCE = 0.75
try:
    DEFAULT_KNOWLEDGE_COHERENCE = float(os.getenv('DEFAULT_KNOWLEDGE_COHERENCE', DEFAULT_KNOWLEDGE_COHERENCE))
except ValueError as e:
    print(f"Warning: Invalid value for DEFAULT_KNOWLEDGE_COHERENCE env var ('{os.getenv('DEFAULT_KNOWLEDGE_COHERENCE')}'). Using default: {DEFAULT_KNOWLEDGE_COHERENCE}. Error: {e}", file=sys.stderr)
    DEFAULT_KNOWLEDGE_COHERENCE = DEFAULT_KNOWLEDGE_COHERENCE

# --- End of Memory/Library Configuration ---

# --- Default Values for core/__init__.py ---
AWARENESS_ERROR_DEFAULTS = {
    "curiosity": 0, "context_stability": 0, "self_evolution_rate": 0, 
    "coherence": 0, "active_llm_fallback": True, 
    "primary_concept_coord": (0,0,0,0), "snn_error": "Manifold not initialized"
}
CRITICAL_MANIFOLD_ERROR_MSG = "CRITICAL: SpacetimeManifold not available. Cannot process thought."
USER_FACING_MANIFOLD_ERROR_MSG = "I am currently unable to process thoughts due to an internal initialization issue."

# --- Default Values for core/brain.py ---
DEFAULT_BRAIN_AWARENESS_METRICS = {
    "curiosity": 0.1, "context_stability": 0.5, "self_evolution_rate": 0.0,
    "coherence": 0.0, "active_llm_fallback": True,
    "primary_concept_coord": (0.0, 0.0, 0.0, 0.0), "snn_error": None
}
# Factors/defaults for metric calculations in brain.think()
CURIOSITY_COHERENCE_FACTOR = 0.5 # (1.0 / 2.0)
CONTEXT_STABILITY_STD_DEV_FACTOR = 2.0
DEFAULT_CONTEXT_STABILITY_SINGLE_READING = 0.75
DEFAULT_CONTEXT_STABILITY_NO_READING = 0.25


# --- Default Values for core/dialogue.py ---
DEFAULT_AWARENESS_METRICS_DIALOGUE = {
    "curiosity": 0.1, "context_stability": 0.3, "self_evolution_rate": 0.0,
    "coherence": 0.0, "active_llm_fallback": True,
    "primary_concept_coord": None, "raw_t_intensity": 0.0, "snn_error": None
}
DEFAULT_DIALOGUE_ERROR_BRAIN_RESPONSE = "System did not generate a specific response due to an internal processing issue."
DEFAULT_DIALOGUE_ETHICAL_SCORE_FALLBACK = 0.5
MAX_CONCEPT_NAME_FOR_MEMORY_LEN = 30
DEFAULT_MEMORY_CONCEPT_NAME = "interaction_summary"
DIALOGUE_SUMMARY_LENGTH_SHORT = 100  # For user input summary for ethics
DIALOGUE_SUMMARY_LENGTH_ACTION = 200 # For brain response summary for ethics

# --- Default Values for core/ethics.py ---
ETHICS_INTENSITY_PREFERENCE_SIGMA = 0.25
ETHICS_IDEAL_INTENSITY_CENTER = 0.5
ETHICS_TREND_T_INTENSITY_FACTOR = 0.9
ETHICS_TREND_BASE_WEIGHT = 0.1
ETHICS_TREND_SHORT_WINDOW_FACTOR = 0.2
ETHICS_TREND_LONG_WINDOW_FACTOR = 0.5
ETHICS_TREND_MIN_SHORT_WINDOW = 3
ETHICS_TREND_MIN_LONG_WINDOW = 5

# --- Default Values for core/gui.py ---
GUI_RESPONSE_STREAMING_DELAY = 0.05

# --- Default Values for core/library.py ---
DEFAULT_SUMMARY_MAX_LENGTH = 100
MITIGATION_LOG_SUMMARY_MAX_LENGTH = 75
MITIGATION_SEVERE_ETHICAL_SCORE_THRESHOLD = 0.3
MITIGATION_STRICT_CAUTION_ETHICAL_SCORE_THRESHOLD = 0.5
KNOWLEDGE_PREVIEW_MAX_LENGTH = 150
KNOWLEDGE_COORD_CONCEPT_NAME_MAX_LENGTH = 20
KNOWLEDGE_DEFAULT_COORD_CONCEPT_NAME = "generic library content"
KNOWLEDGE_ENTRY_SCHEMA_VERSION = "1.0"

# --- Default Values for core/memory.py ---
TEXTUAL_NOVELTY_EMPTY_SUMMARY_SCORE = 0.5
MEMORY_NODE_TYPE_CONCEPT = "concept_memory"
MEMORY_DEFAULT_RELATION_TYPE = "related_to"
DEFAULT_RECENT_MEMORIES_LIMIT = 10

# --- Default Values for main.py ---
DEFAULT_SINGLE_QUERY_STREAM_THOUGHTS = False


# --- Other Configuration Sections ---
# Miscellaneous parameters for system behavior, learning, and environment.

# Threshold for self-correction mechanisms. If confidence or alignment drops below this,
# the system might trigger internal review or learning processes.
DEFAULT_SELF_CORRECTION_THRESHOLD = 0.75
try:
    SELF_CORRECTION_THRESHOLD = float(os.getenv('SELF_CORRECTION_THRESHOLD', DEFAULT_SELF_CORRECTION_THRESHOLD))
except ValueError as e:
    print(f"Warning: Invalid value for SELF_CORRECTION_THRESHOLD env var ('{os.getenv('SELF_CORRECTION_THRESHOLD')}'). Using default: {DEFAULT_SELF_CORRECTION_THRESHOLD}. Error: {e}", file=sys.stderr)
    SELF_CORRECTION_THRESHOLD = DEFAULT_SELF_CORRECTION_THRESHOLD

# Modifier for the rate at which the system evolves or adapts its core structures.
# Could influence learning rates, structural plasticity, etc.
DEFAULT_EVOLUTION_RATE_MODIFIER = 1.0
try:
    EVOLUTION_RATE_MODIFIER = float(os.getenv('EVOLUTION_RATE_MODIFIER', DEFAULT_EVOLUTION_RATE_MODIFIER)) # 1.0 for normal rate
except ValueError as e:
    print(f"Warning: Invalid value for EVOLUTION_RATE_MODIFIER env var ('{os.getenv('EVOLUTION_RATE_MODIFIER')}'). Using default: {DEFAULT_EVOLUTION_RATE_MODIFIER}. Error: {e}", file=sys.stderr)
    EVOLUTION_RATE_MODIFIER = DEFAULT_EVOLUTION_RATE_MODIFIER

# Execution environment: "development", "testing", "production"
# This can be used to enable/disable certain features, logging levels, etc.
EXECUTION_ENVIRONMENT = os.getenv('EXECUTION_ENVIRONMENT', 'development').lower()
IS_DEVELOPMENT_ENV = (EXECUTION_ENVIRONMENT == 'development')
IS_PRODUCTION_ENV = (EXECUTION_ENVIRONMENT == 'production')


# --- End of Other Configuration Sections ---

def validate_config():
    """
    Performs basic validation of critical configuration settings.

    Checks include:
    - LLM provider selection and necessary API key/URL/model settings if LLM is enabled.
    - Resource profile validity.
    - Existence of essential data and log directories.

    Prints warnings or errors to the console.
    Returns:
        bool: True if basic validation passes, False otherwise.
    """
    # TODO: Expand validate_config to cover all critical parameters and their interdependencies. Currently, coverage is partial.
    print("\n--- Validating Configuration ---")
    valid = True

    # Validate LLM Provider and API Key if LLM is enabled
    if ENABLE_LLM_API:
        if LLM_PROVIDER not in _LLM_CONFIG:
            print(f"Error: Invalid LLM_PROVIDER: {LLM_PROVIDER}. Not found in _LLM_CONFIG.")
            valid = False
        else:
            print(f"LLM Provider: {LLM_PROVIDER}")
            if not LLM_API_KEY or "YOUR_API_KEY_HERE" in LLM_API_KEY:
                # LM Studio and Ollama might not require keys, mock definitely doesn't
                if LLM_PROVIDER in ["openai"]: # Add other providers that strictly require keys
                    print(f"Warning: LLM_API_KEY for {LLM_PROVIDER} seems to be missing or a placeholder.")
                    # Depending on strictness, this could be valid = False
            if not LLM_BASE_URL:
                print(f"Error: LLM_BASE_URL for {LLM_PROVIDER} is not set.")
                valid = False
            if not LLM_MODEL:
                print(f"Error: LLM_MODEL for {LLM_PROVIDER} is not set.")
                valid = False
    else:
        print("LLM API is disabled. Skipping LLM configuration validation.")

    # Validate Resource Profile
    if RESOURCE_PROFILE_TYPE not in _RESOURCE_PROFILES:
        # This case is actually handled by the fallback in definition, but good to double check
        print(f"Error: Invalid RESOURCE_PROFILE_TYPE: {RESOURCE_PROFILE_TYPE}.")
        valid = False
    else:
        print(f"Resource Profile: {RESOURCE_PROFILE_TYPE}")

    # Add more checks as needed, e.g., for critical paths
    if not os.path.exists(DATA_DIR):
        print(f"Error: DATA_DIR does not exist: {DATA_DIR}")
        valid = False
        
    if not os.path.exists(LOG_DIR):
        print(f"Error: LOG_DIR does not exist: {LOG_DIR}")
        valid = False

    if valid:
        print("Configuration validation successful.")
    else:
        print("Configuration validation failed with errors/warnings above.")
    return valid

def self_test_config_paths_and_creation() -> bool:
    """
    Tests the `get_path` and `ensure_path` helper functions.

    Specifically, it verifies:
    - `get_path` correctly constructs absolute paths.
    - `ensure_path` correctly creates directories if they don't exist,
      both for direct directory paths and parent directories of file paths.
    - Attempts to clean up any created temporary directories.

    Prints detailed messages about test progress and outcomes.
    Returns:
        bool: True if all tests pass (including cleanup), False otherwise.
    """
    print("\n--- Testing Path Configuration & Creation ---")
    test_passed = True
    
    # Test get_path
    print(f"Project Root (_PROJECT_ROOT): {_PROJECT_ROOT}")
    relative_p = "test_subfolder/test_file.txt"
    abs_p = get_path(relative_p)
    expected_p = os.path.join(_PROJECT_ROOT, relative_p)
    if abs_p != expected_p:
        print(f"Error: get_path('{relative_p}') returned '{abs_p}', expected '{expected_p}'")
        test_passed = False
    else:
        print(f"get_path test successful: {abs_p}")

    # Test ensure_path for a directory
    test_dir = get_path("temp_test_dir_for_config_selftest")
    ensure_path(test_dir + os.sep) # ensure_path expects dir path to end with sep for dirs
    if not os.path.exists(test_dir) or not os.path.isdir(test_dir):
        print(f"Error: ensure_path failed to create directory: {test_dir}")
        test_passed = False
    else:
        print(f"ensure_path directory creation test successful: {test_dir}")
        try:
            os.rmdir(test_dir) # Clean up
            print(f"Cleaned up test directory: {test_dir}")
        except OSError as e:
            print(f"Warning: Could not remove test directory {test_dir}: {e}")
            test_passed = False # If cleanup fails, it's a partial failure

    # Test ensure_path for a file's parent directory
    test_file_path = get_path("temp_test_parent_dir/temp_test_file.txt")
    ensure_path(test_file_path)
    test_file_parent_dir = os.path.dirname(test_file_path)
    if not os.path.exists(test_file_parent_dir) or not os.path.isdir(test_file_parent_dir):
        print(f"Error: ensure_path failed to create parent directory for file: {test_file_parent_dir}")
        test_passed = False
    else:
        print(f"ensure_path file's parent directory creation test successful: {test_file_parent_dir}")
        try:
            os.rmdir(test_file_parent_dir) # Clean up (only if empty, which it should be)
            print(f"Cleaned up test file's parent directory: {test_file_parent_dir}")
        except OSError as e:
            print(f"Warning: Could not remove test file's parent directory {test_file_parent_dir}: {e}")
            # Not critical for path creation logic itself, but good to note.

    if test_passed:
        print("Path configuration and creation tests successful.")
    else:
        print("Path configuration and creation tests failed with errors/warnings above.")
    return test_passed

if __name__ == '__main__':
    print("--- Sophia_Alpha2_ResonantBuild Configuration Self-Test ---")
    
    print("\n--- Key Configuration Values ---")
    print(f"Project Root: {_PROJECT_ROOT}")
    print(f"Config Directory: {CONFIG_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Log Directory: {LOG_DIR}")
    print(f"System Log Path: {SYSTEM_LOG_PATH}")
    print(f"Persona Name: {PERSONA_NAME}")
    print(f"Persona Profile Path: {PERSONA_PROFILE_PATH}")
    
    print(f"Resource Profile Type: {RESOURCE_PROFILE_TYPE}")
    print(f"  Max Neurons: {RESOURCE_PROFILE['MAX_NEURONS']}")
    
    print(f"Enable SNN: {ENABLE_SNN}")
    print(f"Enable LLM API: {ENABLE_LLM_API}")
    if ENABLE_LLM_API:
        print(f"  LLM Provider: {LLM_PROVIDER}")
        print(f"  LLM Model: {LLM_MODEL}")
        print(f"  LLM Base URL: {LLM_BASE_URL}")
        # Avoid printing API keys directly: print(f"  LLM API Key: {LLM_API_KEY}")

    print(f"Execution Environment: {EXECUTION_ENVIRONMENT}")

    # Perform Validations
    validation_ok = validate_config()
    paths_ok = self_test_config_paths_and_creation()

    print("\n--- Self-Test Summary ---")
    if validation_ok and paths_ok:
        print("All configuration self-tests PASSED.")
    else:
        print("One or more configuration self-tests FAILED or reported warnings.")
    
    print("\n--- End of Configuration Self-Test ---")

# TODO: Add comprehensive unit tests for parameter logic, including validation of environment variable parsing, default fallbacks, type conversions, and resource profile selection.

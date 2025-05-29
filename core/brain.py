"""
Core cognitive engine for Sophia_Alpha2, housing the Spacetime Manifold (SNN).

This module is responsible for:
- Implementing the Spacetime Manifold using snnTorch.
- Bootstrapping concepts using Large Language Models (LLMs).
- Applying Hebbian/STDP learning rules.
- Generating awareness metrics based on SNN activity and coherence.
"""

import datetime
import json
import os
import re
import socket # For _try_connect_llm
import sys
import time
import traceback # For error logging

import numpy as np
import requests # Added: For LLM API calls
import snntorch as snn
import torch
import torch.nn as nn # Part of torch
from snntorch import surrogate

# Attempt to import configuration from the parent package
try:
    from .. import config
except ImportError:
    # Fallback for standalone execution or testing (assuming config.py is one level up)
    # This might be adjusted or primarily handled by how main.py sets up sys.path
    print("Brain.py: Could not import 'config' from parent package. Attempting relative import for standalone use.")
    try:
        # If core/ is in PYTHONPATH, this won't work. This assumes running from project root for standalone.
        # Or that brain.py is temporarily moved up for testing.
        # A more robust standalone approach might involve adding project_root to sys.path
        # in the __main__ block if direct execution is intended.
        import config # This will only work if config.py is directly in PYTHONPATH
        print("Brain.py: Successfully imported 'config' directly (likely for standalone testing).")
    except ImportError:
        print("Brain.py: Failed to import 'config' for standalone use. Critical error.")
        # In a real scenario, might raise or exit if config is absolutely needed at import time by other lines.
        # For now, many parts will fail if config isn't loaded.
        config = None # Placeholder if import fails, to prevent immediate crash

# Further module-level constants or setup can go here.

# --- Module-Level Logging ---
# Ensure config is loaded to get log paths and levels
LOG_LEVELS = {"debug": 10, "info": 20, "warning": 30, "error": 40, "critical": 50}

def _log_system_event(event_type: str, data: dict, level: str = "info"):
    """
    Logs a structured system event from the brain module.
    Data is serialized to JSON.
    Respects LOG_LEVEL from config.
    """
    if not config or not hasattr(config, 'SYSTEM_LOG_PATH') or not hasattr(config, 'LOG_LEVEL'):
        print(f"Warning: Config not available for _log_system_event. Event: {event_type}, Data: {data}")
        return

    numeric_level = LOG_LEVELS.get(level.lower(), LOG_LEVELS["info"])
    config_numeric_level = LOG_LEVELS.get(config.LOG_LEVEL.lower(), LOG_LEVELS["info"])

    if numeric_level < config_numeric_level:
        return # Skip logging if event level is below configured log level

    try:
        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "module": "brain",
            "event_type": event_type,
            "level": level.upper(),
            "data": data
        }
        
        # Ensure the log directory exists (config.ensure_path should handle this for SYSTEM_LOG_PATH at import)
        # config.ensure_path(config.SYSTEM_LOG_PATH) # Redundant if called in config, but safe

        with open(config.SYSTEM_LOG_PATH, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
    except Exception as e:
        # Fallback to print if logging to file fails
        print(f"Error logging system event to file: {e}", file=sys.stderr)
        print(f"Original event: {event_type}, Data: {data}", file=sys.stderr)
        # Also log the error itself using the same mechanism, but to avoid recursion, check event_type
        if event_type != "logging_error":
             _log_system_event("logging_error", {"error": str(e), "original_event": event_type}, level="error")


# --- Singleton SpacetimeManifold Instance Management ---
_shared_manifold_instance = None

def get_shared_manifold(force_recreate: bool = False):
    """
    Provides access to the singleton SpacetimeManifold instance.
    Initializes it if it doesn't exist or if force_recreate is True.

    Args:
        force_recreate (bool, optional): If True, any existing instance is discarded
                                         and a new one is created. Defaults to False.
    
    Returns:
        SpacetimeManifold | None: The singleton instance, or None if creation fails
                                  (e.g., SpacetimeManifold class not yet defined).
    """
    global _shared_manifold_instance
    if _shared_manifold_instance is None or force_recreate:
        if force_recreate and _shared_manifold_instance is not None:
            _log_system_event("manifold_recreation_forced", 
                              {"message": "Forcing recreation of SpacetimeManifold instance."})
        
        # Placeholder for SpacetimeManifold class definition
        # This will be defined in a subsequent step.
        # For now, to make this runnable, we can assign a dummy or check for its existence.
        if 'SpacetimeManifold' in globals():
            _shared_manifold_instance = SpacetimeManifold()
            _log_system_event("manifold_instance_created", 
                              {"message": "New SpacetimeManifold instance created."})
        else:
            # This case will be hit until SpacetimeManifold is defined.
            # _log_system_event might not work if config isn't fully loaded here during initial dev.
            print("Warning: SpacetimeManifold class not yet defined. Cannot create instance in get_shared_manifold.", file=sys.stderr)
            # _shared_manifold_instance remains None
            
    return _shared_manifold_instance

def reset_manifold_for_test():
    """Resets the shared manifold instance for isolated testing."""
    global _shared_manifold_instance
    if _shared_manifold_instance is not None and hasattr(_shared_manifold_instance, 'device') and _shared_manifold_instance.device.type == 'cuda':
        # Potentially clear CUDA cache if specific GPU memory issues arise during testing,
        # but usually not needed just for resetting instance logic.
        # torch.cuda.empty_cache() 
        # _log_system_event("cuda_cache_cleared_for_test_reset", {}, level="debug")
        pass # Avoid torch.cuda calls if torch not fully imported or CUDA not available during basic script load

    _shared_manifold_instance = None
    _log_system_event("shared_manifold_instance_reset_for_test", {}, level="debug")

class SpacetimeManifold:
    """
    The cognitive core of Sophia_Alpha2, representing the SNN Spacetime Manifold.
    Manages SNN operations, concept bootstrapping, learning, and awareness metrics.
    """
    def __init__(self):
        _log_system_event("manifold_initialization_start", {"message": "Initializing SpacetimeManifold..."})

        if not config:
            print("CRITICAL ERROR: config module not loaded. SpacetimeManifold cannot initialize.", file=sys.stderr)
            _log_system_event("manifold_initialization_failure", {"error": "Config module not loaded"}, level="critical")
            raise ImportError("Config module not loaded. SpacetimeManifold requires config.")

        # --- Device Configuration ---
        # Determine compute device (GPU if enabled and available, else CPU)
        # Use getattr for USE_GPU for robustness against it missing in config.
        use_gpu_flag = getattr(config, 'USE_GPU', False) # Default to False if not set
        if use_gpu_flag and torch.cuda.is_available():
            self.device = torch.device("cuda")
            _log_system_event("device_config", {"device": "cuda", "reason": "USE_GPU is True in config and CUDA available"})
        else:
            self.device = torch.device("cpu")
            reason = "USE_GPU is False in config" if not use_gpu_flag else "CUDA not available (though USE_GPU was True)"
            if not torch.cuda.is_available() and use_gpu_flag: # More specific reason
                 reason = "CUDA not available (though USE_GPU was True in config)"
            elif not use_gpu_flag:
                 reason = "USE_GPU is False or not set in config"

            _log_system_event("device_config", {"device": "cpu", "reason": reason})
        
        print(f"SpacetimeManifold using device: {self.device}") # User feedback


        # --- Load Parameters from Config ---
        self.range = float(config.MANIFOLD_RANGE)
        
        # Resource Profile parameters
        self.max_neurons = int(config.RESOURCE_PROFILE["MAX_NEURONS"])
        self.resolution = float(config.RESOURCE_PROFILE["RESOLUTION"]) # May be used for coordinate scaling or grid density
        self.snn_time_steps = int(config.RESOURCE_PROFILE["SNN_TIME_STEPS"])

        # SNN Architecture & Input
        self.snn_input_size = int(config.SNN_INPUT_SIZE) # Dimension of input vectors (e.g., LLM embeddings)
        # For now, output size of the SNN is equivalent to max_neurons (each neuron is an output feature)
        self.snn_output_size = self.max_neurons 

        # Learning Parameters
        # STDP window in seconds (config is in ms)
        self.tau_stdp_s = float(config.STDP_WINDOW_MS) / 1000.0 
        # Using HEBBIAN_LEARNING_RATE as the primary STDP learning rate as per task spec
        self.lr_stdp = float(config.HEBBIAN_LEARNING_RATE) 
        self.stdp_depression_factor = float(config.STDP_DEPRESSION_FACTOR)
        # self.stdp_dt_threshold_s = float(config.STDP_DT_THRESHOLD_MS) / 1000.0 if hasattr(config, 'STDP_DT_THRESHOLD_MS') else 0.01 # Example if needed

        # LIF Neuron Parameters
        self.lif_beta = float(config.SNN_LIF_BETA) # Decay rate
        self.lif_threshold = float(config.SNN_LIF_THRESHOLD) # Firing threshold
        self.spike_grad = surrogate.fast_sigmoid(slope=float(config.SNN_SURROGATE_SLOPE))

        # Optimizer
        self.optimizer_lr = float(config.SNN_OPTIMIZER_LR)

        # --- Initialize SNN Components ---
        # Fully connected layer: input_size -> output_size (max_neurons)
        self.fc = nn.Linear(self.snn_input_size, self.snn_output_size).to(self.device)
        
        # Leaky Integrate-and-Fire (LIF) neurons
        self.lif1 = snn.Leaky(
            beta=self.lif_beta, 
            threshold=self.lif_threshold, 
            spike_grad=self.spike_grad,
            learn_beta=True, # Allow beta to be learned
            learn_threshold=False # Threshold typically not learned with surrogate gradients easily
        ).to(self.device)

        # Initialize membrane potential and spikes
        # Batch size is assumed to be 1 (processing one concept/input at a time)
        self.batch_size = 1 
        # Initialize LIF membrane potential using snnTorch's utility for the defined LIF layer
        self.mem = self.lif1.init_leaky() 
        # Initialize spike tensor (output spikes) for the first step
        self.spk = torch.zeros(self.batch_size, self.snn_output_size, device=self.device)

        # --- Initialize Optimizer ---
        # Adam optimizer is chosen for its adaptive learning rate capabilities.
        # It will optimize parameters of the fully connected layer (self.fc) and
        # the learnable parameters of the LIF neuron layer (self.lif1), such as 'beta'.
        self.optimizer = torch.optim.Adam(
            list(self.fc.parameters()) + list(self.lif1.parameters()), # Parameters to optimize
            lr=self.optimizer_lr # Learning rate from config
        )

        # --- Manifold State Variables ---
        # self.coordinates: Stores spatial and temporal intensity information for each concept.
        # Format: {concept_name: (x_coord, y_coord, z_coord, t_intensity_coord)}
        self.coordinates = {}  # Stores {concept_name: (x, y, z, t_intensity)}
        self.coherence = 0.0   # Global coherence metric of the manifold
        self.last_avg_stdp_weight_change = 0.0 # For self_evolution_rate metric

        _log_system_event("manifold_initialization_complete", 
                          {"device": str(self.device), "max_neurons": self.max_neurons, 
                           "input_size": self.snn_input_size, "time_steps": self.snn_time_steps})

    def _mock_phi3_concept_data(self, concept_name: str) -> dict:
        """
        Provides mock data for a concept if LLM is unavailable or for testing.
        Includes summary, valence, abstraction, relevance, and intensity.
        """
        _log_system_event("mock_concept_data_access", {"concept_name": concept_name}, level="debug")
        
        # Predefined mock concepts
        # Valence: -1 (neg) to 1 (pos)
        # Abstraction: 0 (concrete) to 1 (abstract)
        # Relevance: 0 (irrelevant) to 1 (highly relevant to core themes)
        # Intensity: 0 (low) to 1 (high emotional/cognitive load)
        mock_concepts = {
            "love": {"summary": "Mock summary: A complex emotion of deep affection and connection.", "valence": 0.8, "abstraction": 0.7, "relevance": 0.9, "intensity": 0.9},
            "ethics": {"summary": "Mock summary: Moral principles governing behavior.", "valence": 0.5, "abstraction": 0.8, "relevance": 1.0, "intensity": 0.7},
            "ai": {"summary": "Mock summary: Intelligence demonstrated by machines.", "valence": 0.3, "abstraction": 0.6, "relevance": 1.0, "intensity": 0.6},
            "resonance": {"summary": "Mock summary: The quality of being deep, full, and reverberating; a shared understanding.", "valence": 0.7, "abstraction": 0.7, "relevance": 0.9, "intensity": 0.8},
            "unknown": {"summary": "Mock summary: Something not known or identified.", "valence": 0.0, "abstraction": 0.5, "relevance": 0.3, "intensity": 0.2},
            "void": {"summary": "Mock summary: A completely empty space; nothingness.", "valence": -0.5, "abstraction": 0.9, "relevance": 0.1, "intensity": 0.4},
            "default": {"summary": "Mock summary: Default concept data used as fallback.", "valence": 0.1, "abstraction": 0.4, "relevance": 0.2, "intensity": 0.3}
        }
        
        concept_key = concept_name.lower() if concept_name else "default"
        if concept_key not in mock_concepts:
            # Try partial match or use default
            for key in mock_concepts:
                if key in concept_key:
                    concept_key = key
                    break
            else: # If no partial match found
                concept_key = "default"
                
        data = mock_concepts[concept_key]
        data["concept_name"] = concept_name # Ensure original name is part of data
        
        _log_system_event("mock_concept_data_used", {"concept_name": concept_name, "data": data}, level="info")
        return data

    def _try_connect_llm(self) -> bool:
        """
        Attempts to establish a connection to the configured LLM provider's base URL.
        Returns True if successful (socket connection), False otherwise.
        """
        if not config.ENABLE_LLM_API:
            _log_system_event("llm_connection_skipped", {"reason": "LLM API disabled in config"}, level="info")
            return False

        try:
            # Parse hostname and port from LLM_BASE_URL
            # Example: "http://localhost:1234/v1" -> "localhost", 1234
            # Example: "https://api.openai.com/v1" -> "api.openai.com", 443 (default for https)
            url_parts = config.LLM_BASE_URL.split('/')
            if len(url_parts) < 3:
                _log_system_event("llm_connection_failure", {"reason": "Invalid LLM_BASE_URL format", "url": config.LLM_BASE_URL}, level="error")
                return False

            protocol = url_parts[0].replace(':', '') # http or https
            domain_port = url_parts[2].split(':')
            hostname = domain_port[0]
            
            if len(domain_port) > 1:
                port = int(domain_port[1])
            else:
                port = 443 if protocol == 'https' else 80 # Default ports

            _log_system_event("llm_connection_attempt", {"hostname": hostname, "port": port, "timeout": config.LLM_CONNECTION_TIMEOUT}, level="debug")
            
            with socket.create_connection((hostname, port), timeout=config.LLM_CONNECTION_TIMEOUT):
                _log_system_event("llm_connection_success", {"hostname": hostname, "port": port}, level="info")
                return True
        except (socket.error, ValueError, IndexError) as e:
            _log_system_event("llm_connection_failure", {"error": str(e), "url": config.LLM_BASE_URL}, level="warning")
            return False

    def bootstrap_concept_from_llm(self, concept_name: str) -> tuple:
        """
        Bootstraps a concept using the configured LLM or falls back to mock data.
        Returns a tuple: (coordinates_tuple, intensity_float, summary_string).
        Coordinates are (x, y, z, t_intensity) normalized based on self.range.
        Intensity is the 't' coordinate before normalization.
        """
        _log_system_event("concept_bootstrap_start", {"concept_name": concept_name})
        llm_data = None # Initialize to ensure it's defined in all paths

        # Attempt to connect to LLM API if enabled
        if config.ENABLE_LLM_API and self._try_connect_llm():
            headers = {"Content-Type": "application/json"}
            # Add Authorization header for OpenAI if API key is present
            if config.LLM_PROVIDER == "openai" and config.LLM_API_KEY != "YOUR_OPENAI_API_KEY_HERE_IF_NOT_SET_AS_ENV":
                headers["Authorization"] = f"Bearer {config.LLM_API_KEY}"
            
            # Construct the prompt for the LLM.
            # Uses templates from config, ensuring a core instruction is present.
            user_prompt_content = config.LLM_CONCEPT_PROMPT_TEMPLATE.get("user", "Explain the concept: {concept_name}").format(concept_name=concept_name)
            if "Explain the concept:" not in user_prompt_content: # Fallback for minimal templates
                 user_prompt_content = f"Explain the concept: {concept_name}. {user_prompt_content}"

            # Prepare payload for the LLM API call
            payload = {
                "model": config.LLM_MODEL,
                "messages": [
                    {"role": "system", "content": config.LLM_CONCEPT_PROMPT_TEMPLATE.get("system", "You are an AI assistant providing concise information about concepts.")},
                    {"role": "user", "content": user_prompt_content }
                ],
                "temperature": config.LLM_TEMPERATURE,
                # "max_tokens": ... # Consider adding max_tokens if necessary for the provider
            }
            # Request JSON response format if supported by the provider (e.g., OpenAI, LM Studio)
            if config.LLM_PROVIDER in ["openai", "lm_studio"]:
                 payload["response_format"] = {"type": "json_object"}

            # Determine the correct API endpoint URL based on the provider
            api_url = config.LLM_BASE_URL
            if config.LLM_PROVIDER == "openai":
                api_url = os.path.join(config.LLM_BASE_URL, "chat/completions")
            elif config.LLM_PROVIDER == "ollama":
                 # Ollama's /api/chat is suitable for message-based interactions.
                 # /api/generate has a different payload structure.
                 api_url = os.path.join(config.LLM_BASE_URL, "chat")

            try:
                _log_system_event("llm_api_call_start", {"url": api_url, "provider": config.LLM_PROVIDER, "model": config.LLM_MODEL}, level="debug")
                # Make the POST request to the LLM API
                response = requests.post(
                    api_url,
                    headers=headers,
                    data=json.dumps(payload), 
                    timeout=config.LLM_REQUEST_TIMEOUT
                )
                response.raise_for_status() # Raise an HTTPError for bad status codes (4xx or 5xx)
                
                raw_response_text = response.text # Store raw text for potential regex fallback
                _log_system_event("llm_api_call_success_raw", {"concept_name": concept_name, "status_code": response.status_code, "response_head": raw_response_text[:200]}, level="debug")

                # Attempt to parse the JSON response from the LLM
                try:
                    # Provider-specific parsing logic
                    if config.LLM_PROVIDER == "openai":
                        # OpenAI chat completions nest content: response.json()['choices'][0]['message']['content']
                        json_content_str = response.json()['choices'][0]['message']['content']
                        llm_data = json.loads(json_content_str)
                    elif config.LLM_PROVIDER == "lm_studio":
                        # Assume LM Studio with response_format="json_object" returns direct JSON
                        llm_data = response.json()
                    elif config.LLM_PROVIDER == "ollama":
                        # Ollama chat (non-streaming) returns JSON with content at: response.json()['message']['content']
                        # This content itself might be a JSON string.
                        message_content_str = response.json()['message']['content']
                        try:
                            llm_data = json.loads(message_content_str)
                        except json.JSONDecodeError: # If content is not a JSON string, log and prepare for regex
                            _log_system_event("llm_parsing_warning", {"concept_name": concept_name, "detail": "Ollama message content not direct JSON, attempting regex.", "content": message_content_str[:200]}, level="warning")
                            llm_data = None # Explicitly set to None to trigger regex fallback
                            # Fall through to regex if direct parse fails (llm_data is None)
                    else: # Generic attempt for other providers
                        llm_data = response.json()

                    # Validate if the parsed data has the expected structure (all required keys)
                    if llm_data is not None and (not isinstance(llm_data, dict) or not all(k in llm_data for k in ["summary", "valence", "abstraction", "relevance", "intensity"])):
                        _log_system_event("llm_parsing_warning", {"concept_name": concept_name, "detail": "Parsed JSON lacks required keys. Attempting regex on raw text.", "parsed_data_type": str(type(llm_data))}, level="warning")
                        llm_data = None # Force regex fallback if structure is incorrect

                except (json.JSONDecodeError, KeyError) as e_parse: # Handle errors during JSON parsing
                    _log_system_event("llm_parsing_error_json", {"concept_name": concept_name, "error": str(e_parse), "response_text_head": raw_response_text[:200]}, level="warning")
                    llm_data = None # Signal to try regex parsing

                # Regex fallback if JSON parsing failed or yielded invalid structure
                if llm_data is None:
                    _log_system_event("llm_parsing_attempt_regex", {"concept_name": concept_name}, level="debug")
                    # Define regex patterns to extract data fields from raw text.
                    # These patterns assume a JSON-like string within the response.
                    patterns = {
                        "summary": r'"summary":\s*"(.*?)"', # Captures text within quotes
                        "valence": r'"valence":\s*(-?\d+\.?\d*)', # Captures float/int
                        "abstraction": r'"abstraction":\s*(-?\d+\.?\d*)',
                        "relevance": r'"relevance":\s*(-?\d+\.?\d*)',
                        "intensity": r'"intensity":\s*(-?\d+\.?\d*)'
                    }
                    extracted_data = {}
                    for key, pattern in patterns.items():
                        match = re.search(pattern, raw_response_text, re.IGNORECASE | re.DOTALL)
                        if match:
                            if key == "summary":
                                extracted_data[key] = match.group(1).strip()
                            else: # For numeric fields, convert to float
                                try:
                                    extracted_data[key] = float(match.group(1))
                                except ValueError: # Handle conversion error
                                    _log_system_event("llm_regex_type_error", {"key": key, "value": match.group(1)}, level="warning")
                                    extracted_data[key] = 0.0 # Default on error
                        else: # If a key is not found by regex
                             _log_system_event("llm_regex_key_not_found", {"key": key}, level="warning")
                             if key != "summary": extracted_data[key] = 0.0 # Default for numeric
                             else: extracted_data[key] = "Summary not extracted." # Default for summary

                    # If all required keys were successfully extracted via regex
                    if all(k in extracted_data for k in ["summary", "valence", "abstraction", "relevance", "intensity"]):
                        llm_data = extracted_data
                        _log_system_event("llm_parsing_success_regex", {"concept_name": concept_name, "data_keys": list(llm_data.keys())}, level="info")
                    else: # Regex extraction also failed to get all keys
                        _log_system_event("llm_parsing_failure_regex", {"concept_name": concept_name, "extracted_keys": list(extracted_data.keys())}, level="error")
                        # llm_data remains None or partially filled
                        
            except requests.exceptions.RequestException as e_req: # Handle network/request errors
                _log_system_event("llm_api_call_failure", {"concept_name": concept_name, "error": str(e_req)}, level="error")
            except Exception as e_gen: # Catch any other unexpected errors during LLM processing
                _log_system_event("llm_processing_error_unknown", {"concept_name": concept_name, "error": str(e_gen), "trace": traceback.format_exc()}, level="critical")

        # Fallback to mock data if LLM data acquisition failed or LLM is disabled
        if not llm_data: # This check handles cases where llm_data is None or became None after failed parsing/validation
            _log_system_event("llm_data_unavailable_using_mock", {"concept_name": concept_name}, level="warning")
            llm_data = self._mock_phi3_concept_data(concept_name)
        
        # Ensure default values for all expected keys in llm_data to prevent KeyErrors later
        # This is especially important if mock data or a failed LLM response (even after regex)
        # does not guarantee all keys.
        llm_data.setdefault("summary", "No summary available.")
        llm_data.setdefault("valence", 0.0)
        llm_data.setdefault("abstraction", 0.0)
        llm_data.setdefault("relevance", 0.0)
        llm_data.setdefault("intensity", 0.0)

        # --- Calculate 4D Manifold Coordinates ---
        # Coordinates are derived from LLM/mock data (valence, abstraction, relevance, intensity).
        # These values are normalized and scaled by `self.range`.
        # x: derived from valence (-1 to 1), mapped to (-self.range/2 to +self.range/2)
        # y: derived from abstraction (0 to 1), mapped to (0 to +self.range/2)
        # z: derived from relevance (0 to 1), mapped to (0 to +self.range/2)
        # t_coord_intensity: derived from raw_intensity (0 to 1), mapped to (0 to +self.range/2)
        # raw_intensity is returned separately as 'intensity_float'.

        half_range = self.range / 2.0 # Pre-calculate for efficiency
        
        # Clip values to expected ranges before scaling
        x = np.clip(float(llm_data["valence"]), -1.0, 1.0) * half_range
        y = np.clip(float(llm_data["abstraction"]), 0.0, 1.0) * half_range 
        z = np.clip(float(llm_data["relevance"]), 0.0, 1.0) * half_range
        
        raw_intensity = np.clip(float(llm_data["intensity"]), 0.0, 1.0) # This is the (0-1) intensity
        t_coord_intensity = raw_intensity * half_range # This is the scaled 't' coordinate

        coordinates = (x, y, z, t_coord_intensity) # Store as a tuple
        self.coordinates[concept_name] = coordinates # Update manifold's dictionary
        
        _log_system_event("concept_bootstrap_complete", 
                          {"concept_name": concept_name, "coordinates": coordinates, 
                           "raw_intensity": raw_intensity, 
                           "source": "LLM" if config.ENABLE_LLM_API and not llm_data.get("summary","").startswith("Mock summary:") else "Mock"}, 
                          level="info")
        
        return coordinates, raw_intensity, str(llm_data["summary"])

    def update_stdp(self, pre_spk_flat: torch.Tensor, post_spk_flat: torch.Tensor, 
                    current_weights: torch.Tensor, 
                    concept_name: str, prev_t_intensity: float) -> tuple[torch.Tensor, float]:
        """
        Implements STDP learning rule or a Hebbian-like update.
        
        Args:
            pre_spk_flat: Flattened presynaptic spikes (from input features or previous layer).
                          Expected shape: (batch_size, num_presynaptic_neurons) which is (1, snn_input_size)
            post_spk_flat: Flattened postsynaptic spikes (from the SNN's output neurons).
                           Expected shape: (batch_size, num_postsynaptic_neurons) which is (1, snn_output_size)
            current_weights: The current weight matrix of the fc layer.
                             Expected shape: (num_postsynaptic_neurons, num_presynaptic_neurons)
                                             which is (snn_output_size, snn_input_size)
            concept_name: Name of the concept being processed, to fetch its current t_intensity.
            prev_t_intensity: t_intensity from the previous significant processing step or concept.

        Returns:
            Tuple of (updated_weights, delta_w_mean_abs).
        """
        if not config.ENABLE_SNN: # STDP only applies if SNN is active
            return current_weights.clone(), 0.0

        _log_system_event("stdp_update_start", {"concept": concept_name}, level="debug")

        # Ensure tensors are on the correct device for computation
        pre_spk_flat = pre_spk_flat.to(self.device)
        post_spk_flat = post_spk_flat.to(self.device)
        current_weights_clone = current_weights.clone().to(self.device) # Work on a clone to avoid modifying original tensor in-place prematurely

        # Get current t_intensity for the concept from stored coordinates
        current_concept_coords = self.coordinates.get(concept_name)
        if not current_concept_coords: # Should ideally not happen if concept was bootstrapped
            _log_system_event("stdp_error_no_coords", {"concept": concept_name}, level="warning")
            return current_weights_clone, 0.0 # Return original weights if no coordinates found
        
        # The 't' coordinate (index 3) is used as a proxy for the timing of the current event.
        # This was previously scaled by self.range/2. For STDP, the raw_intensity (0-1) might be
        # more suitable for calculating delta_t if prev_t_intensity is also raw.
        # Current implementation uses scaled t_coord_intensity. This needs to be consistent.
        current_t_intensity = current_concept_coords[3] 

        # --- STDP/Hebbian Logic using t_intensity as a proxy for time ---
        # delta_t = t_current_event - t_previous_event
        # Positive delta_t: Current event is "after" previous. If pre-synaptic spike (input) led to post-synaptic spike (output)
        #                   under this condition, it implies potentiation (LTP).
        # Negative delta_t: Current event is "before" previous. If post-synaptic spike occurred before pre-synaptic under this
        #                   timing, it implies depression (LTD).
        # This is a conceptual application of STDP principles where t_intensity differences model event timing.
        delta_t = current_t_intensity - prev_t_intensity 
        
        # Initialize weight change matrix
        delta_w = torch.zeros_like(current_weights_clone)

        # Hebbian term: Correlation of pre-synaptic and post-synaptic activity.
        # post_spk_flat.T (shape [snn_output_size, 1]) @ pre_spk_flat (shape [1, snn_input_size])
        # results in an outer product (shape [snn_output_size, snn_input_size]),
        # representing correlated activity between each input and output neuron.
        hebbian_term = torch.mm(post_spk_flat.T, pre_spk_flat)

        # STDP rule application based on the sign of delta_t (timing proxy)
        if delta_t > 0: # Potentiation case (LTP, pre-before-post like)
            # Timing factor: exponential decay based on delta_t and STDP window (tau_stdp_s).
            # Smaller delta_t (events closer in time in the "correct" order) leads to stronger potentiation.
            timing_factor = torch.exp(-torch.abs(delta_t) / (self.tau_stdp_s + 1e-9)) # Epsilon for stability
            delta_w = self.lr_stdp * timing_factor * hebbian_term
        elif delta_t < 0: # Depression case (LTD, post-before-pre like)
            timing_factor = torch.exp(-torch.abs(delta_t) / (self.tau_stdp_s + 1e-9))
            # LTD is scaled by stdp_depression_factor.
            delta_w = -self.lr_stdp * self.stdp_depression_factor * timing_factor * hebbian_term
        # If delta_t is zero, no weight change occurs under this specific timing rule.

        # Apply the calculated weight change
        updated_weights = current_weights_clone + delta_w
        
        # Clip weights to a reasonable range if necessary, e.g., [-1, 1] or [0, 1] if only excitatory
        # For now, no clipping.
        # updated_weights = torch.clamp(updated_weights, -1.0, 1.0) 

        delta_w_mean_abs = torch.mean(torch.abs(delta_w)).item()

        # Update global coherence
        # Coherence decreases if weights change significantly, increases if they stabilize
        # This is a simplified model of coherence.
        coherence_change = -self.lr_stdp * delta_w_mean_abs # More change = less coherence
        self.coherence += coherence_change * config.COHERENCE_UPDATE_FACTOR
        self.coherence = np.clip(self.coherence, -1.0, 1.0) # Assuming coherence is within a range

        if delta_w_mean_abs > 1e-5: # Log if change is somewhat significant
            _log_system_event("stdp_update_applied", 
                              {"concept": concept_name, "delta_w_mean_abs": delta_w_mean_abs, 
                               "delta_t_intensity_proxy": delta_t, "new_coherence": self.coherence}, 
                              level="debug")
            
        return updated_weights, delta_w_mean_abs

    def warp_manifold(self, input_text: str) -> tuple[list, str, list, list, tuple]:
        """
        Main SNN processing loop. Bootstraps a concept, runs SNN simulation with STDP learning,
        and updates manifold state.

        Args:
            input_text (str): The primary concept text to process.

        Returns:
            tuple: A tuple containing:
                - thought_steps_log (list): Chronological log of processing steps and observations.
                - final_monologue (str): A summary string of the SNN processing outcome.
                - spk_rec_list (list): List of numpy arrays, where each array holds the spike
                                       data for one time step of the SNN simulation.
                - activity_levels_list (list): List of floats representing mean SNN activity
                                               per neuron at each time step.
                - primary_concept_coord_tuple (tuple): The (x,y,z,t) coordinates of the
                                                       bootstrapped primary concept.
        """
        _log_system_event("warp_manifold_start", {"input_text": input_text})
        thought_steps_log = [] # Stores a chronological log of processing steps

        # 1. Bootstrap Primary Concept: Convert input text to manifold coordinates and properties.
        try:
            concept_coords, concept_raw_intensity, concept_summary = self.bootstrap_concept_from_llm(input_text)
            if concept_coords is None: # Should be handled by mock fallback in bootstrap if LLM fails
                _log_system_event("warp_manifold_error", {"error": "Concept bootstrapping failed critically (returned None)"}, level="critical")
                # Return empty/error state
                return [], "Error: Concept bootstrapping failed critically.", [], [], (0.0, 0.0, 0.0, 0.0)
            
            primary_concept_coord_tuple = concept_coords
            # prev_t_intensity_for_stdp: Use the raw (0-1) intensity of the newly bootstrapped concept
            # as the 'previous' time marker for the first STDP update cycle within this warp.
            # This establishes the initial temporal context for subsequent SNN activity.
            prev_t_intensity_for_stdp = concept_raw_intensity 
            thought_steps_log.append(f"Bootstrapped concept '{input_text}': Coords={concept_coords}, RawIntensity={concept_raw_intensity:.2f}, Summary: {concept_summary[:50]}...")
        except Exception as e_bootstrap: # Catch any unexpected error during bootstrapping
            _log_system_event("warp_manifold_bootstrap_exception", {"input_text": input_text, "error": str(e_bootstrap), "trace": traceback.format_exc()}, level="critical")
            return [], f"Error during concept bootstrapping for '{input_text}': {str(e_bootstrap)}", [], [], (0.0, 0.0, 0.0, 0.0)

        # 2. Initialize SNN State for the current warp cycle:
        # Reset membrane potential for LIF neurons. Spikes are recorded per step.
        self.mem = self.lif1.init_leaky() # Re-initialize membrane potential for the batch (size 1)
        
        spk_rec_list = []  # Stores spike recordings from each SNN time step
        activity_levels_list = [] # Stores mean SNN activity level at each time step
        
        # Work on a clone of the current weights for this warp cycle's STDP updates.
        # This allows accumulating changes within the cycle before applying to the model.
        current_weights = self.fc.weight.data.clone().to(self.device)
        
        total_stdp_change_accumulator = 0.0 # Accumulates magnitude of STDP changes
        num_stdp_updates = 0 # Counts how many STDP updates occurred

        # 3. SNN Simulation Loop: Iterate over configured number of time steps.
        _log_system_event("snn_simulation_loop_start", 
                          {"time_steps": self.snn_time_steps, "concept": input_text, 
                           "initial_intensity_for_stdp_context": prev_t_intensity_for_stdp}, 
                          level="debug")
                          
        for step in range(self.snn_time_steps):
            # --- a. Create Input Features Tensor ---
            # The input to the SNN is derived from the bootstrapped concept's properties.
            # Current approach: Activate a subset of input neurons proportionally to concept_raw_intensity.
            # This is a simplified representation; more complex mappings from concept features to SNN inputs are possible.
            input_features = torch.zeros(self.batch_size, self.snn_input_size, device=self.device)
            if self.snn_input_size > 0:
                # Activate a portion of input neurons (e.g., first 10% or at least one)
                num_active_inputs = min(self.snn_input_size, max(1, self.snn_input_size // 10)) 
                activation_value = concept_raw_intensity # Modulate activation by concept's raw intensity
                input_features[0, :num_active_inputs] = activation_value
            
            # --- b. SNN Forward Pass ---
            # Propagate input features through the fully connected layer and then the LIF neuron layer.
            # self.fc: [batch_size, snn_input_size] -> [batch_size, snn_output_size]
            # self.lif1: Processes weighted inputs to produce output spikes and updated membrane potentials.
            try:
                current_fc_out = self.fc(input_features) # Output of the dense layer
                spk_out, self.mem = self.lif1(current_fc_out, self.mem) # Output spikes and updated membrane
            except Exception as e_forward: # Catch errors during SNN computation
                _log_system_event("snn_forward_pass_error", {"step": step, "concept": input_text, "error": str(e_forward), "trace": traceback.format_exc()}, level="error")
                thought_steps_log.append(f"Error in SNN forward pass at step {step} for '{input_text}': {e_forward}")
                break # Exit simulation loop on error

            # Record output spikes (convert to NumPy array on CPU for storage/analysis)
            spk_rec_list.append(spk_out.clone().cpu().numpy())

            # --- c. STDP Learning ---
            # Apply STDP if there's both pre-synaptic (input_features) and post-synaptic (spk_out) activity.
            # This ensures learning occurs only when relevant signals are present.
            if torch.any(spk_out > 0) and torch.any(input_features > 0):
                updated_weights, delta_w_mean_abs = self.update_stdp(
                    pre_spk_flat=input_features,    # Current input to SNN
                    post_spk_flat=spk_out,          # Resulting output spikes
                    current_weights=current_weights,# Current weights (clone) being updated in this cycle
                    concept_name=input_text,        # Primary concept for contextual t_intensity
                    prev_t_intensity=prev_t_intensity_for_stdp # Temporal context from initial bootstrap
                )
                current_weights = updated_weights # Persist STDP changes for this warp cycle
                total_stdp_change_accumulator += delta_w_mean_abs
                num_stdp_updates +=1
            else: # No STDP update if no relevant pre/post activity
                delta_w_mean_abs = 0.0

            # --- d. Log Step Details & Collect Activity Metrics ---
            current_spike_count = spk_out.sum().item() # Total spikes in this step
            # Mean activity: proportion of neurons firing in this step
            activity_levels_list.append(current_spike_count / self.snn_output_size if self.snn_output_size > 0 else 0.0) 

            log_msg = (f"Step {step+1}/{self.snn_time_steps}: Spikes={current_spike_count}, "
                       f"STDP_dW_Mean={delta_w_mean_abs:.4e}, Coherence={self.coherence:.3f}")
            thought_steps_log.append(log_msg)
            # Log detailed SNN step info periodically (e.g., 10 times during simulation)
            if (step + 1) % max(1, self.snn_time_steps // 10) == 0:
                 _log_system_event("snn_simulation_step", {"step": step + 1, "total_steps": self.snn_time_steps, 
                                                          "concept": input_text, "spikes": current_spike_count, 
                                                          "stdp_dw_mean": delta_w_mean_abs, 
                                                          "coherence": self.coherence}, level="debug")

        # 4. After Simulation Loop - Finalize Changes:
        # Apply the accumulated weight changes from this warp cycle to the actual model's parameters.
        self.fc.weight.data = current_weights.to(self.device) 
        
        # Calculate the average STDP weight change for this cycle (if any updates occurred).
        if num_stdp_updates > 0:
            self.last_avg_stdp_weight_change = total_stdp_change_accumulator / num_stdp_updates
        else:
            self.last_avg_stdp_weight_change = 0.0 # No STDP updates means no average change
        
        _log_system_event("snn_simulation_loop_end", 
                          {"concept": input_text, 
                           "avg_stdp_weight_change": self.last_avg_stdp_weight_change,
                           "final_coherence": self.coherence}, 
                          level="info")

        # 5. Generate Monologue and Return Results:
        final_monologue = f"Warp manifold processing complete for '{input_text}'. Summary: {concept_summary} Final Coherence: {self.coherence:.3f}."
        thought_steps_log.append(f"Final SNN coherence for '{input_text}': {self.coherence:.3f}, Avg STDP dW: {self.last_avg_stdp_weight_change:.4e}")

        _log_system_event("warp_manifold_end", {"input_text": input_text, "final_coherence": self.coherence, "monologue_summary": final_monologue[:100]})
        
        return thought_steps_log, final_monologue, spk_rec_list, activity_levels_list, primary_concept_coord_tuple

    def think(self, input_text: str, stream_thought_steps: bool = False) -> tuple[list, str, dict]:
        """
        Primary interface for initiating thought processes in the SpacetimeManifold.

        Orchestrates SNN processing (if enabled via config.ENABLE_SNN) or falls back
        to LLM-based concept bootstrapping. Calculates and returns awareness metrics
        based on the processing path taken and SNN activity.

        Args:
            input_text (str): The input text, typically a concept or query to be processed.
            stream_thought_steps (bool, optional): If True and VERBOSE_OUTPUT is enabled
                                                   in config, prints thought steps to console.
                                                   Defaults to False.

        Returns:
            tuple: A tuple containing:
                - thought_steps (list): A list of strings detailing the internal processing steps.
                - response_text (str): The primary textual output or summary from the thought process.
                - awareness_metrics (dict): A dictionary of metrics including:
                    - "curiosity" (float): Calculated curiosity level.
                    - "context_stability" (float): Calculated context stability.
                    - "self_evolution_rate" (float): Rate of SNN self-evolution (STDP change).
                    - "coherence" (float): Current SNN coherence level.
                    - "active_llm_fallback" (bool): True if LLM fallback was used.
                    - "primary_concept_coord" (tuple): (x,y,z,t) coordinates of the input concept.
                    - "snn_error" (str | None): Description of SNN error if one occurred, else None.
        """
        _log_system_event("think_start", {"input_text": input_text, "stream_steps": stream_thought_steps})
        
        thought_steps = [] # Initialize list to store logs of thought process
        response_text = ""
        awareness_metrics = {
            "curiosity": 0.1, # Default: Low curiosity
            "context_stability": 0.5, # Default: Moderate stability
            "self_evolution_rate": 0.0, # Default: No evolution detected
            "coherence": 0.0, # Default: Neutral coherence
            "active_llm_fallback": True, # Assume LLM fallback is active by default
            "primary_concept_coord": (0.0, 0.0, 0.0, 0.0), # Default coordinates
            "snn_error": None # Placeholder for any SNN processing errors
        }

        snn_processed_successfully = False # Flag to track if SNN processing completed

        # Path 1: SNN Processing (if enabled in config)
        if config.ENABLE_SNN:
            try:
                _log_system_event("think_snn_path_start", {"input_text": input_text})
                # Call warp_manifold to perform SNN simulation and learning
                thought_steps, response_text, spk_rec_list, activity_levels, primary_coord = \
                    self.warp_manifold(input_text)
                
                # Update awareness metrics based on SNN processing results
                awareness_metrics["active_llm_fallback"] = False # SNN path was taken
                awareness_metrics["primary_concept_coord"] = primary_coord
                awareness_metrics["coherence"] = np.clip(self.coherence, -1.0, 1.0) # Global coherence from SNN
                # self_evolution_rate: based on average STDP weight change, scaled and clipped
                awareness_metrics["self_evolution_rate"] = np.clip(self.last_avg_stdp_weight_change * 1000, 0.0, 1.0) 

                # Calculate Curiosity metric:
                # Based on normalized mean SNN spike activity and (1 - abs(coherence)).
                # Higher activity and lower coherence (more dissonance) can imply higher curiosity.
                if spk_rec_list and any(step.size > 0 for step in spk_rec_list): # Check if there are any spikes recorded
                    # Calculate mean spikes per neuron per time step, averaged over all time steps.
                    mean_spikes_overall = np.mean([np.mean(step_spikes) for step_spikes in spk_rec_list if step_spikes.size > 0])
                    # Normalize mean_spikes_overall to a 0-1 range. Max possible is 1.0 (all neurons fire at every step).
                    normalized_mean_spikes = np.clip(mean_spikes_overall, 0.0, 1.0) 
                    # Combine normalized activity with a measure of dissonance (1 - abs(coherence)).
                    awareness_metrics["curiosity"] = np.clip((normalized_mean_spikes + (1.0 - abs(self.coherence))) / 2.0, 0.0, 1.0)
                else: # No spikes recorded, curiosity driven only by coherence/dissonance
                    awareness_metrics["curiosity"] = np.clip((1.0 - abs(self.coherence)) / 2.0, 0.0, 1.0)

                # Calculate Context Stability metric:
                # Based on the standard deviation of SNN activity levels over time.
                # Lower std_dev implies more stable activity.
                if activity_levels and len(activity_levels) > 1:
                    std_dev_activity = np.std(activity_levels)
                    # Normalize stability: 1.0 for zero std_dev, decreasing as std_dev increases.
                    # Max std_dev for 0/1 data is ~0.5. (2*std_dev) maps this to a 0-1 range for stability.
                    awareness_metrics["context_stability"] = np.clip(1.0 - (2 * std_dev_activity), 0.0, 1.0) 
                elif activity_levels and len(activity_levels) == 1: # Single activity reading
                     awareness_metrics["context_stability"] = 0.75 # Moderately stable by default
                else: # No activity levels recorded
                    awareness_metrics["context_stability"] = 0.25 # Low stability

                snn_processed_successfully = True # Mark SNN path as successful
                _log_system_event("think_snn_path_complete", {"input_text": input_text, "metrics": awareness_metrics})

            except Exception as e_snn_warp: # Handle errors during SNN processing
                _log_system_event("think_snn_path_error", 
                                  {"input_text": input_text, "error": str(e_snn_warp), "trace": traceback.format_exc()}, 
                                  level="critical")
                response_text = f"An error occurred during SNN processing for '{input_text}': {str(e_snn_warp)}. Falling back to LLM."
                thought_steps.append(response_text)
                awareness_metrics["snn_error"] = str(e_snn_warp) # Record the SNN error
                # snn_processed_successfully remains False, will trigger LLM fallback
        
        # Path 2: LLM Fallback (if SNN disabled or SNN error occurred)
        if not snn_processed_successfully:
            _log_system_event("think_llm_fallback_path_start", 
                              {"input_text": input_text, 
                               "reason": "SNN disabled in config" if not config.ENABLE_SNN else "SNN processing error"}, 
                              level="info")
            try:
                concept_coords, concept_raw_intensity, concept_summary = \
                    self.bootstrap_concept_from_llm(input_text)
                
                response_text = f"LLM Fallback: Concept '{input_text}' - Summary: {concept_summary}"
                thought_steps = [f"SNN processing was skipped or failed. Used LLM for concept '{input_text}'.",
                                 f"LLM Summary: {concept_summary}"]
                
                awareness_metrics["active_llm_fallback"] = True
                awareness_metrics["primary_concept_coord"] = concept_coords if concept_coords else (0,0,0,0)
                # Default/neutral metrics for LLM fallback
                awareness_metrics["curiosity"] = 0.1 
                awareness_metrics["context_stability"] = 0.5 
                awareness_metrics["self_evolution_rate"] = 0.0
                awareness_metrics["coherence"] = 0.0 # No SNN processing to establish coherence
            except Exception as e_bootstrap_fallback:
                _log_system_event("think_llm_fallback_bootstrap_error", 
                                  {"input_text": input_text, "error": str(e_bootstrap_fallback), "trace": traceback.format_exc()}, 
                                  level="critical")
                response_text = f"A critical error occurred during LLM fallback processing: {str(e_bootstrap_fallback)}"
                thought_steps.append(response_text)
                # Metrics remain at their initial defaults indicating critical failure

        if stream_thought_steps and config.VERBOSE_OUTPUT:
            print("\n--- Sophia_Alpha2 Thought Steps ---")
            for step_info in thought_steps:
                print(step_info)
            print("--- End of Thought Steps ---\n")

        _log_system_event("think_end", {"input_text": input_text, "response_summary": response_text[:100], "metrics": awareness_metrics})
        return thought_steps, response_text, awareness_metrics

if __name__ == '__main__':
    # --- Test Utilities ---
    class TempConfigOverride:
        # ... (previously defined TempConfigOverride class code) ...
        # Ensure this class definition is present here.
        # For brevity in this prompt, assuming it's correctly defined from Step 8.
        def __init__(self, temp_configs):
            self.temp_configs = temp_configs
            self.original_values = {}
        def __enter__(self):
            if not config: raise ImportError("Config module not loaded")
            for key, value in self.temp_configs.items():
                self.original_values[key] = getattr(config, key, None)
                setattr(config, key, value)
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            if not config: return
            for key, original_value in self.original_values.items():
                if hasattr(config, key) and original_value is None and key not in self.temp_configs:
                    pass # Attribute was likely created by the test, not originally present
                elif original_value is not None:
                    setattr(config, key, original_value)
                elif hasattr(config, key): # Attribute was created by test, and no original value to restore
                    delattr(config, key)


    # --- Original __main__ preamble ---
    print("core/brain.py loaded as main.")
    if config:
        print(f"Config module successfully imported. Project root from config: {getattr(config, '_PROJECT_ROOT', 'N/A')}")
        # Ensure essential config paths are created for tests if not already
        config.ensure_path(config.LOG_DIR + os.sep)
        config.ensure_path(config.DATA_DIR + os.sep)
    else:
        print("CRITICAL: Config module failed to import. Brain module tests cannot run without config.")
        sys.exit(1) # Exit if config is not available for tests
        
    print(f"PyTorch version: {torch.__version__}")
    print(f"snnTorch version: {snn.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Device for tests (from config): {config.USE_GPU and torch.cuda.is_available()}")


    # --- Test Function Definitions ---
    def run_test(test_func, *args):
        test_name = test_func.__name__
        print(f"--- Running Test: {test_name} ---")
        try:
            reset_manifold_for_test() # Ensure clean state
            result = test_func(*args)
            if result:
                print(f"PASS: {test_name}")
            else:
                print(f"FAIL: {test_name}")
            return result
        except Exception as e:
            print(f"ERROR in {test_name}: {e}")
            traceback.print_exc()
            return False

    def test_snn_processing_and_stpd():
        print("Testing SNN processing path (SNN enabled, LLM mock for speed)...")
        # Ensure LLM is mocked to avoid actual API calls and ensure predictability
        # Also ensure SNN is enabled.
        with TempConfigOverride({"ENABLE_SNN": True, "LLM_PROVIDER": "mock_for_snn_test", "VERBOSE_OUTPUT": False}):
            manifold = get_shared_manifold(force_recreate=True) # Get a fresh instance with current config
            if not manifold: return False
            
            initial_weight_mean = torch.mean(manifold.fc.weight.data).item()
            _, _, metrics = manifold.think("test concept for SNN and STDP")
            
            if metrics["active_llm_fallback"]:
                print("FAIL: SNN path not taken, LLM fallback was active.")
                return False
            
            final_weight_mean = torch.mean(manifold.fc.weight.data).item()
            # STDP should cause some weight change, but could be zero if no spikes or specific conditions
            # self_evolution_rate is based on last_avg_stdp_weight_change
            if metrics["self_evolution_rate"] == 0.0 and initial_weight_mean == final_weight_mean :
                 print("INFO: No STDP weight change detected or self_evolution_rate is 0. This might be ok if no spikes occurred.")
                 # This isn't a strict fail, but good to note. For a robust test, ensure input causes spikes.
            else:
                 print(f"INFO: STDP applied. Initial weight mean: {initial_weight_mean:.6f}, Final: {final_weight_mean:.6f}, Evolution Rate: {metrics['self_evolution_rate']:.4f}")

            expected_keys = ["curiosity", "context_stability", "self_evolution_rate", "coherence", "active_llm_fallback", "primary_concept_coord", "snn_error"]
            if not all(key in metrics for key in expected_keys):
                print(f"FAIL: Awareness metrics missing keys. Found: {metrics.keys()}")
                return False
            print(f"Awareness metrics (SNN path): {metrics}")
        return True

    def test_llm_fallback_snn_disabled():
        print("Testing LLM fallback (SNN disabled, LLM mock)...")
        with TempConfigOverride({"ENABLE_SNN": False, "LLM_PROVIDER": "mock_for_snn_test", "VERBOSE_OUTPUT": False}):
            manifold = get_shared_manifold(force_recreate=True)
            if not manifold: return False
            
            _, _, metrics = manifold.think("test concept for LLM fallback")
            if not metrics["active_llm_fallback"]:
                print("FAIL: LLM fallback was not active when SNN disabled.")
                return False
            if metrics["snn_error"] is not None:
                print(f"FAIL: SNN error reported during LLM fallback: {metrics['snn_error']}")
                return False
            print(f"Awareness metrics (LLM fallback / SNN disabled): {metrics}")
        return True

    def test_llm_fallback_snn_error():
        print("Testing LLM fallback due to simulated SNN error (LLM mock)...")
        # To simulate an error, we can temporarily break something warp_manifold relies on
        # For example, by setting snn_time_steps to zero or an invalid value if not caught by __init__
        # A more direct way: modify warp_manifold to raise an error under a test condition,
        # or temporarily override a method it calls to raise an error.
        # For now, this test is conceptual as direct error injection is hard from here without changing main code.
        # Let's assume a config change could induce an error (e.g., invalid SNN_INPUT_SIZE if not validated in init)
        # This test will be similar to SNN disabled for now, as the fallback logic is the same.
        with TempConfigOverride({"ENABLE_SNN": True, "LLM_PROVIDER": "mock_for_snn_test", 
                                 "SNN_TIME_STEPS": 0, # This might cause an issue or just run 0 steps
                                 "VERBOSE_OUTPUT": False}): 
            manifold = get_shared_manifold(force_recreate=True) # Recreate with potentially problematic config
            if not manifold: return False

            _, _, metrics = manifold.think("test concept for SNN error fallback")
            if not metrics["active_llm_fallback"]:
                print("FAIL: LLM fallback was not active after potential SNN error.")
                return False
            # We expect snn_error to be populated IF an error was actually caught by think()
            # If SNN_TIME_STEPS=0 just means no SNN processing but no error, snn_error might be None
            if metrics["snn_error"] is None and manifold.snn_time_steps == 0:
                 print("INFO: SNN error was None; SNN_TIME_STEPS=0 might have resulted in no SNN loop rather than an error.")
            elif metrics["snn_error"]:
                 print(f"INFO: SNN error correctly reported: {metrics['snn_error']}")

            print(f"Awareness metrics (LLM fallback / SNN error sim): {metrics}")
        return True


    def test_empty_input():
        print("Testing empty input string...")
        with TempConfigOverride({"ENABLE_SNN": True, "LLM_PROVIDER": "mock_for_snn_test", "VERBOSE_OUTPUT": False}):
            manifold = get_shared_manifold(force_recreate=True)
            if not manifold: return False
            
            # Mock concept for "default" or empty should be hit by bootstrap_concept_from_llm
            _, response, metrics = manifold.think("") 
            if not response:
                print("FAIL: No response for empty input.")
                return False
            if metrics["primary_concept_coord"] == (0,0,0,0) and "default" not in response.lower() and "unknown" not in response.lower():
                # (0,0,0,0) might be valid if mock for "" is all zeros. Check response instead.
                print(f"FAIL: Empty input did not seem to resolve to a default/unknown concept. Response: {response}")
                return False
            print(f"Response for empty input: {response[:100]}...")
        return True

    def test_awareness_metrics_structure():
        print("Testing structure of awareness metrics dictionary...")
        # Test with SNN enabled first
        with TempConfigOverride({"ENABLE_SNN": True, "LLM_PROVIDER": "mock_for_snn_test", "VERBOSE_OUTPUT": False}):
            manifold = get_shared_manifold(force_recreate=True)
            if not manifold: return False
            _, _, metrics_snn = manifold.think("test concept for metrics structure")
        
        expected_keys = ["curiosity", "context_stability", "self_evolution_rate", "coherence", "active_llm_fallback", "primary_concept_coord", "snn_error"]
        if not all(key in metrics_snn for key in expected_keys):
            print(f"FAIL (SNN path): Awareness metrics missing keys. Found: {metrics_snn.keys()}, Expected: {expected_keys}")
            return False
        print(f"Metrics keys (SNN path): OK ({list(metrics_snn.keys())})")

        # Test with SNN disabled (LLM fallback)
        with TempConfigOverride({"ENABLE_SNN": False, "LLM_PROVIDER": "mock_for_snn_test", "VERBOSE_OUTPUT": False}):
            manifold = get_shared_manifold(force_recreate=True)
            if not manifold: return False
            _, _, metrics_llm = manifold.think("test concept for metrics structure fallback")

        if not all(key in metrics_llm for key in expected_keys):
            print(f"FAIL (LLM fallback path): Awareness metrics missing keys. Found: {metrics_llm.keys()}, Expected: {expected_keys}")
            return False
        print(f"Metrics keys (LLM fallback path): OK ({list(metrics_llm.keys())})")
        return True

    # --- Main Test Execution Logic ---
    print("\n--- Starting SpacetimeManifold Self-Tests ---")
    tests_to_run = [
        test_snn_processing_and_stpd,
        test_llm_fallback_snn_disabled,
        test_llm_fallback_snn_error, # This test is more of an observation currently
        test_empty_input,
        test_awareness_metrics_structure,
    ]
    
    results = []
    for test_fn in tests_to_run:
        results.append(run_test(test_fn))
        # Small delay or separator for readability if running many tests quickly
        time.sleep(0.1) 

    print("\n--- SpacetimeManifold Self-Test Summary ---")
    passed_count = sum(1 for r in results if r)
    total_count = len(results)
    print(f"Tests Passed: {passed_count}/{total_count}")

    if passed_count == total_count:
        print("All tests PASSED successfully!")
        # sys.exit(0) # Not exiting here to allow further interaction if needed when run as script
    else:
        print("One or more tests FAILED. Please review logs above.")
        sys.exit(1) # Exit with error code if any test failed

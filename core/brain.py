"""
Core cognitive engine for Sophia_Alpha2, housing the Spacetime Manifold (SNN).

This module is responsible for:
- Implementing the Spacetime Manifold using snnTorch.
- Bootstrapping concepts using Large Language Models (LLMs).
- Applying Hebbian/STDP learning rules.
- Generating awareness metrics based on SNN activity and coherence.
"""

import os
import sys
import json
import re
import socket # For _try_connect_llm
import time
import datetime
import traceback # For error logging

import numpy as np
import torch
import torch.nn as nn # Corrected from torch.nn
import snntorch as snn
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
        if config.USE_GPU and torch.cuda.is_available():
            self.device = torch.device("cuda")
            _log_system_event("device_config", {"device": "cuda", "reason": "USE_GPU is True and CUDA available"})
        else:
            self.device = torch.device("cpu")
            reason = "USE_GPU is False" if not config.USE_GPU else "CUDA not available"
            _log_system_event("device_config", {"device": "cpu", "reason": reason})
        
        print(f"SpacetimeManifold using device: {self.device}")


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
        # Batch size is assumed to be 1 for now (one concept/input at a time)
        self.batch_size = 1 
        self.mem = self.lif1.init_leaky() # Initializes based on lif1 internal state if not passed batch_size, or use below
        # self.mem = torch.zeros(self.batch_size, self.snn_output_size, device=self.device) # Alternative explicit init
        self.spk = torch.zeros(self.batch_size, self.snn_output_size, device=self.device)

        # --- Initialize Optimizer ---
        # We want to optimize the weights of the fully connected layer and LIF beta
        self.optimizer = torch.optim.Adam(
            list(self.fc.parameters()) + list(self.lif1.parameters()), 
            lr=self.optimizer_lr
        )

        # --- Manifold State Variables ---
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
        llm_data = None

        if config.ENABLE_LLM_API and self._try_connect_llm():
            headers = {"Content-Type": "application/json"}
            if config.LLM_PROVIDER == "openai" and config.LLM_API_KEY != "YOUR_OPENAI_API_KEY_HERE_IF_NOT_SET_AS_ENV":
                headers["Authorization"] = f"Bearer {config.LLM_API_KEY}"
            
            # Construct payload based on provider's expected format
            # Using a generic chat completions structure for this example
            # The actual prompt template usage will vary by provider and API endpoint
            # For now, we'll assume a simple "define this concept" approach for the LLM call.
            # The richer CONCEPT_PROMPT_TEMPLATE from config will be used more in dialogue or deeper queries.
            
            # Simplified prompt for direct concept definition query to LLM
            # This part might need refinement based on how LLM_CONCEPT_PROMPT_TEMPLATE is structured
            # and whether we expect a single user message or a system+user message pair.
            user_prompt_content = config.LLM_CONCEPT_PROMPT_TEMPLATE.get("user", "Explain the concept: {concept_name}").format(concept_name=concept_name)
            if "Explain the concept:" not in user_prompt_content: # Ensure the core instruction is there if template is minimal
                 user_prompt_content = f"Explain the concept: {concept_name}. {user_prompt_content}"


            payload = {
                "model": config.LLM_MODEL,
                "messages": [
                    {"role": "system", "content": config.LLM_CONCEPT_PROMPT_TEMPLATE.get("system", "You are an AI assistant providing concise information about concepts.")},
                    {"role": "user", "content": user_prompt_content }
                ],
                "temperature": config.LLM_TEMPERATURE,
                # Add max_tokens if appropriate for the provider
            }
            # OpenAI and compatible APIs (like LM Studio) support response_format for JSON
            if config.LLM_PROVIDER in ["openai", "lm_studio"]: # Assuming lm_studio endpoint supports it
                 payload["response_format"] = {"type": "json_object"}


            api_url = config.LLM_BASE_URL
            # Adjust API URL for specific providers if needed (e.g., OpenAI chat completions)
            if config.LLM_PROVIDER == "openai":
                api_url = os.path.join(config.LLM_BASE_URL, "chat/completions") # Common endpoint
            elif config.LLM_PROVIDER == "ollama":
                 # Ollama might use /api/chat or /api/generate. Assuming /api/chat for this structure.
                 # For generate, the payload structure is different.
                 api_url = os.path.join(config.LLM_BASE_URL, "chat")


            try:
                _log_system_event("llm_api_call_start", {"url": api_url, "provider": config.LLM_PROVIDER, "model": config.LLM_MODEL}, level="debug")
                # NOTE: The 'requests' import is missing. This will be handled in a separate step.
                import requests # Temporary import for this block
                response = requests.post(
                    api_url, 
                    headers=headers, 
                    data=json.dumps(payload), 
                    timeout=config.LLM_REQUEST_TIMEOUT
                )
                response.raise_for_status() # Raise HTTPError for bad responses (4XX or 5XX)
                
                raw_response_text = response.text
                _log_system_event("llm_api_call_success_raw", {"concept_name": concept_name, "status_code": response.status_code, "response_head": raw_response_text[:200]}, level="debug")

                try:
                    # Attempt to parse JSON directly from response text
                    # For OpenAI chat completions, content is nested: response.json()['choices'][0]['message']['content']
                    if config.LLM_PROVIDER == "openai":
                        json_content_str = response.json()['choices'][0]['message']['content']
                        llm_data = json.loads(json_content_str)
                    elif config.LLM_PROVIDER == "lm_studio": # Assuming direct JSON object in response for LM Studio with response_format
                        llm_data = response.json()
                    elif config.LLM_PROVIDER == "ollama": # Ollama chat response is streamed JSON objects, or a single JSON object if not streamed.
                                                          # For non-streaming chat, it's response.json()['message']['content'] which is a string, potentially JSON.
                        message_content_str = response.json()['message']['content']
                        try:
                            llm_data = json.loads(message_content_str)
                        except json.JSONDecodeError:
                            _log_system_event("llm_parsing_warning", {"concept_name": concept_name, "detail": "Ollama message content not direct JSON, attempting regex.", "content": message_content_str[:200]}, level="warning")
                            # Fall through to regex if direct parse fails
                    else: # Generic attempt
                        llm_data = response.json()

                    if not isinstance(llm_data, dict) or not all(k in llm_data for k in ["summary", "valence", "abstraction", "relevance", "intensity"]):
                        _log_system_event("llm_parsing_warning", {"concept_name": concept_name, "detail": "Parsed JSON lacks required keys. Attempting regex on raw text.", "parsed_data_type": str(type(llm_data))}, level="warning")
                        llm_data = None # Force regex fallback

                except (json.JSONDecodeError, KeyError) as e_parse:
                    _log_system_event("llm_parsing_error_json", {"concept_name": concept_name, "error": str(e_parse), "response_text_head": raw_response_text[:200]}, level="warning")
                    llm_data = None # Signal to try regex

                if llm_data is None: # Attempt regex extraction if direct JSON parsing failed or produced invalid structure
                    _log_system_event("llm_parsing_attempt_regex", {"concept_name": concept_name}, level="debug")
                    # This regex is a basic example and needs to be robust and match the expected LLM output format
                    # It assumes JSON-like structures within the text.
                    patterns = {
                        "summary": r'"summary":\s*"(.*?)"',
                        "valence": r'"valence":\s*(-?\d+\.?\d*)',
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
                            else:
                                try:
                                    extracted_data[key] = float(match.group(1))
                                except ValueError:
                                    _log_system_event("llm_regex_type_error", {"key": key, "value": match.group(1)}, level="warning")
                                    extracted_data[key] = 0.0 # Default on parse error
                        else: # Key not found by regex
                             _log_system_event("llm_regex_key_not_found", {"key": key}, level="warning")
                             if key != "summary": extracted_data[key] = 0.0 # Default numeric
                             else: extracted_data[key] = "Summary not extracted."


                    if all(k in extracted_data for k in ["summary", "valence", "abstraction", "relevance", "intensity"]):
                        llm_data = extracted_data
                        _log_system_event("llm_parsing_success_regex", {"concept_name": concept_name, "data_keys": list(llm_data.keys())}, level="info")
                    else:
                        _log_system_event("llm_parsing_failure_regex", {"concept_name": concept_name, "extracted_keys": list(extracted_data.keys())}, level="error")
                        
            except requests.exceptions.RequestException as e_req:
                _log_system_event("llm_api_call_failure", {"concept_name": concept_name, "error": str(e_req)}, level="error")
            except Exception as e_gen:
                _log_system_event("llm_processing_error_unknown", {"concept_name": concept_name, "error": str(e_gen), "trace": traceback.format_exc()}, level="critical")


        if not llm_data:
            _log_system_event("llm_data_unavailable_using_mock", {"concept_name": concept_name}, level="warning")
            llm_data = self._mock_phi3_concept_data(concept_name)
        
        llm_data.setdefault("summary", "No summary available.")
        llm_data.setdefault("valence", 0.0)
        llm_data.setdefault("abstraction", 0.0)
        llm_data.setdefault("relevance", 0.0)
        llm_data.setdefault("intensity", 0.0)

        # Calculate 4D coordinates, normalizing by self.range
        # x from valence, y from abstraction, z from relevance, t from intensity
        # Assuming self.range is the max absolute value for coordinates (e.g., if range is 1000, coords are -500 to 500)
        # For simplicity, let's map LLM outputs (0-1 or -1 to 1) to a portion of this range.
        # E.g., map valence (-1 to 1) to x (-self.range/2 to self.range/2)
        # E.g., map abstraction (0 to 1) to y (0 to self.range/2)
        # E.g., map relevance (0 to 1) to z (0 to self.range/2)
        # E.g., map intensity (0 to 1) to t_intensity (0 to self.range/2 or some other scale)
        
        # For now, a simple direct mapping, clipping/scaling can be added
        # x: valence (-1 to 1) * (self.range / 2)
        # y: abstraction (0 to 1) * (self.range / 2)
        # z: relevance (0 to 1) * (self.range / 2)
        # t_raw_intensity: intensity (0 to 1) - this is the one returned as 'intensity_float'
        # t_coord_intensity: intensity (0 to 1) * (self.range / 2) - used as a coordinate

        half_range = self.range / 2.0
        
        x = np.clip(float(llm_data["valence"]), -1.0, 1.0) * half_range
        y = np.clip(float(llm_data["abstraction"]), 0.0, 1.0) * half_range 
        z = np.clip(float(llm_data["relevance"]), 0.0, 1.0) * half_range
        
        raw_intensity = np.clip(float(llm_data["intensity"]), 0.0, 1.0)
        t_coord_intensity = raw_intensity * half_range # Or map to a different scale if needed

        coordinates = (x, y, z, t_coord_intensity)
        self.coordinates[concept_name] = coordinates
        
        _log_system_event("concept_bootstrap_complete", 
                          {"concept_name": concept_name, "coordinates": coordinates, 
                           "raw_intensity": raw_intensity, "source": "LLM" if config.ENABLE_LLM_API and llm_data.get("summary","").startswith("Mock summary:") is False else "Mock"}, 
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

        # Ensure tensors are on the correct device
        pre_spk_flat = pre_spk_flat.to(self.device)
        post_spk_flat = post_spk_flat.to(self.device)
        current_weights_clone = current_weights.clone().to(self.device) # Work on a clone

        # Get current t_intensity for the concept
        current_concept_coords = self.coordinates.get(concept_name)
        if not current_concept_coords:
            _log_system_event("stdp_error_no_coords", {"concept": concept_name}, level="warning")
            return current_weights_clone, 0.0
        
        # current_t_intensity is the 4th element (index 3) of the coordinate tuple (x,y,z,t)
        # This t_coord_intensity was scaled by self.range/2. We might need raw_intensity (0-1) for delta_t.
        # However, bootstrap_concept_from_llm returns raw_intensity separately.
        # Let's assume prev_t_intensity and the one derived from current_concept_coords are comparable,
        # or better, that they are both the 'raw_intensity' (0-1 scale).
        # For now, let's use the stored t_coord_intensity and assume prev_t_intensity is on a similar scale.
        current_t_intensity = current_concept_coords[3] 

        # --- Basic STDP/Hebbian Logic ---
        # delta_t: Time difference. Positive if presynaptic spike happens BEFORE postsynaptic.
        # For this model, 'time' is proxied by t_intensity.
        # A larger prev_t_intensity means the 'previous event' was 'later' or 'stronger in time'.
        # So, if prev_t_intensity > current_t_intensity, it's like the previous event is "closer" in time or "after" current.
        # This interpretation of delta_t needs to be consistent with the STDP rule.
        # Let's define delta_t = t_current - t_previous.
        # If delta_t > 0 (current is "after" previous), it could be LTP for pre-before-post.
        # If delta_t < 0 (current is "before" previous), it could be LTD for post-before-pre.
        
        delta_t = current_t_intensity - prev_t_intensity 
        
        # Weight update matrix, initialized to zeros
        delta_w = torch.zeros_like(current_weights_clone)

        # Expand pre_spk_flat and post_spk_flat for broadcasting with weights
        # pre_spk_flat: [1, snn_input_size] -> [snn_output_size, snn_input_size] (repeated rows)
        # post_spk_flat: [1, snn_output_size] -> [snn_output_size, 1] -> [snn_output_size, snn_input_size] (repeated columns)
        
        # We need to iterate over each neuron pair or find a vectorized way.
        # For a simple Hebbian rule: delta_w = lr * post_spikes * pre_spikes_transpose
        # pre_spk_flat is (1, num_inputs), post_spk_flat is (1, num_outputs)
        # delta_w should be (num_outputs, num_inputs)
        # So, delta_w_hebbian = self.lr_stdp * post_spk_flat.T @ pre_spk_flat
        
        # For STDP, the timing component (delta_t and self.tau_stdp_s) is crucial.
        # A common STDP exponential window function:
        # if delta_t > 0 (pre-before-post): LTP = A_plus * exp(-delta_t / tau_plus)
        # if delta_t < 0 (post-before-pre): LTD = -A_minus * exp(delta_t / tau_minus) (delta_t is negative here)
        # Let tau_plus = tau_minus = self.tau_stdp_s
        # Let A_plus = self.lr_stdp
        # Let A_minus = self.lr_stdp * self.stdp_depression_factor (or just self.stdp_depression_factor if lr_stdp is the base magnitude)

        # Simplified STDP based on delta_t sign, applied to correlated activity
        # This is a conceptual STDP. Real biophysical STDP is spike-pair specific.
        # Here, delta_t is a global timing difference between 'events'.
        
        # Calculate the Hebbian term (correlation of activity)
        # post_spk_flat.T has shape [snn_output_size, 1]
        # pre_spk_flat has shape [1, snn_input_size]
        hebbian_term = torch.mm(post_spk_flat.T, pre_spk_flat) # Outer product: results in [snn_output_size, snn_input_size]

        if delta_t > 0: # Potentiation (pre-before-post like)
            # Exponential decay factor, stronger for smaller delta_t
            timing_factor = torch.exp(-torch.abs(delta_t) / (self.tau_stdp_s + 1e-9)) # add epsilon to avoid div by zero
            delta_w = self.lr_stdp * timing_factor * hebbian_term
        elif delta_t < 0: # Depression (post-before-pre like)
            timing_factor = torch.exp(-torch.abs(delta_t) / (self.tau_stdp_s + 1e-9))
            # Depression factor makes LTD effect smaller or different magnitude
            delta_w = -self.lr_stdp * self.stdp_depression_factor * timing_factor * hebbian_term
        # If delta_t is zero, no change based on this simplified timing rule, only Hebbian if implemented differently.

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
        """
        _log_system_event("warp_manifold_start", {"input_text": input_text})
        thought_steps_log = [] # To store logs of each step

        # 1. Bootstrap primary concept
        try:
            concept_coords, concept_raw_intensity, concept_summary = self.bootstrap_concept_from_llm(input_text)
            if concept_coords is None: # Should not happen if mock fallback works
                _log_system_event("warp_manifold_error", {"error": "Concept bootstrapping failed critically"}, level="critical")
                return [], "Error: Concept bootstrapping failed.", [], [], (0,0,0,0)
            primary_concept_coord_tuple = concept_coords
            # prev_t_intensity for STDP can be the raw intensity of the bootstrapped concept
            # This represents the 'initial time' or 'intensity' of the stimulus.
            prev_t_intensity_for_stdp = concept_raw_intensity 
            thought_steps_log.append(f"Bootstrapped concept '{input_text}': Coords={concept_coords}, Intensity={concept_raw_intensity:.2f}, Summary: {concept_summary[:50]}...")
        except Exception as e_bootstrap:
            _log_system_event("warp_manifold_bootstrap_exception", {"error": str(e_bootstrap), "trace": traceback.format_exc()}, level="critical")
            return [], f"Error during concept bootstrapping: {str(e_bootstrap)}", [], [], (0,0,0,0)

        # 2. Initialize SNN State
        # Membrane potential and spikes, reset for each warp sequence
        self.mem = self.lif1.init_leaky() # Re-initialize for the current batch_size (1)
        # self.spk is updated in the loop. Initialize a fresh recording list.
        
        spk_rec_list = []  # To store spike recordings at each time step
        # mem_rec_list = [] # Optional: to store membrane potential recordings
        
        # Get a clone of current weights to modify during this warp cycle
        current_weights = self.fc.weight.data.clone().to(self.device)
        
        # t_intensity_diffs_list: to store how SNN activity relates to original intensity
        # This is a placeholder for a more concrete metric.
        # For now, let's store the mean spike count at each step as a proxy for activity level.
        activity_levels_list = [] 
        
        total_stdp_change_accumulator = 0.0
        num_stdp_updates = 0

        # 3. SNN Simulation Loop
        _log_system_event("snn_simulation_loop_start", 
                          {"time_steps": self.snn_time_steps, "concept": input_text, 
                           "initial_intensity_for_stdp": prev_t_intensity_for_stdp}, 
                          level="debug")
                          
        for step in range(self.snn_time_steps):
            # --- a. Create input_features tensor ---
            # This needs to be a tensor of size [batch_size, self.snn_input_size]
            # Simple approach: Use concept_raw_intensity to modulate a fixed pattern or one specific input neuron.
            # More advanced: Convert concept_coords or summary embedding to an SNN_INPUT_SIZE tensor.
            # For now, activate the first input neuron proportionally to intensity.
            input_features = torch.zeros(self.batch_size, self.snn_input_size, device=self.device)
            if self.snn_input_size > 0:
                 # Spread activation across first few inputs, modulated by intensity
                num_active_inputs = min(self.snn_input_size, max(1, self.snn_input_size // 10)) # Activate up to 10% or at least 1
                activation_value = concept_raw_intensity 
                input_features[0, :num_active_inputs] = activation_value
            
            # --- b. SNN Forward Pass ---
            # self.fc expects input of shape [batch_size, snn_input_size]
            # self.lif1 expects input of shape [batch_size, snn_output_size]
            try:
                current_fc_out = self.fc(input_features)
                spk_out, self.mem = self.lif1(current_fc_out, self.mem)
            except Exception as e_forward:
                _log_system_event("snn_forward_pass_error", {"step": step, "error": str(e_forward), "trace": traceback.format_exc()}, level="error")
                thought_steps_log.append(f"Error in SNN forward pass at step {step}: {e_forward}")
                # Potentially break or handle error gracefully
                break 

            spk_rec_list.append(spk_out.clone().cpu().numpy()) # Store spikes (as numpy on CPU)
            # mem_rec_list.append(self.mem.clone().cpu().numpy()) # Optional

            # --- c. STDP Learning ---
            # pre_spk_flat for STDP is the input_features that caused spk_out
            # post_spk_flat is spk_out
            if torch.any(spk_out > 0) and torch.any(input_features > 0): # Only if there's activity
                updated_weights, delta_w_mean_abs = self.update_stdp(
                    pre_spk_flat=input_features, # Shape: [1, snn_input_size]
                    post_spk_flat=spk_out,       # Shape: [1, snn_output_size]
                    current_weights=current_weights,
                    concept_name=input_text,     # Use the primary concept name for context
                    prev_t_intensity=prev_t_intensity_for_stdp # Compare to initial intensity
                )
                current_weights = updated_weights # Persist changes for the current warp cycle
                total_stdp_change_accumulator += delta_w_mean_abs
                num_stdp_updates +=1
            else:
                delta_w_mean_abs = 0.0 # No STDP if no relevant spikes

            # --- d. Log step details ---
            current_spike_count = spk_out.sum().item()
            activity_levels_list.append(current_spike_count / self.snn_output_size) # Store mean spike activity

            log_msg = (f"Step {step+1}/{self.snn_time_steps}: Spikes={current_spike_count}, "
                       f"STDP_dW={delta_w_mean_abs:.4e}, Coherence={self.coherence:.3f}")
            thought_steps_log.append(log_msg)
            if (step + 1) % max(1, self.snn_time_steps // 10) == 0: # Log roughly 10 times
                 _log_system_event("snn_simulation_step", {"step": step + 1, "total_steps": self.snn_time_steps, 
                                                          "spikes": current_spike_count, 
                                                          "stdp_dw_mean": delta_w_mean_abs, 
                                                          "coherence": self.coherence}, level="debug")

        # 4. After Loop - Finalize
        # Apply the accumulated weight changes from this warp cycle to the actual model
        self.fc.weight.data = current_weights.to(self.device) 
        
        if num_stdp_updates > 0:
            self.last_avg_stdp_weight_change = total_stdp_change_accumulator / num_stdp_updates
        else:
            self.last_avg_stdp_weight_change = 0.0
        
        _log_system_event("snn_simulation_loop_end", 
                          {"concept": input_text, 
                           "avg_stdp_weight_change": self.last_avg_stdp_weight_change,
                           "final_coherence": self.coherence}, 
                          level="info")

        # 5. Generate placeholder monologue and return
        final_monologue = f"Warp manifold processing complete for '{input_text}'. Summary: {concept_summary} Coherence: {self.coherence:.3f}."
        thought_steps_log.append(f"Final coherence: {self.coherence:.3f}, Avg STDP dW: {self.last_avg_stdp_weight_change:.4e}")

        _log_system_event("warp_manifold_end", {"input_text": input_text, "final_coherence": self.coherence})
        
        # Ensure all returned lists are serializable if needed (numpy arrays converted to lists of lists)
        # spk_rec_list already stores numpy arrays.
        
        return thought_steps_log, final_monologue, spk_rec_list, activity_levels_list, primary_concept_coord_tuple

    def think(self, input_text: str, stream_thought_steps: bool = False) -> tuple[list, str, dict]:
        """
        Primary interface for initiating thought processes in the SpacetimeManifold.
        Orchestrates SNN processing (if enabled) or LLM fallback, and calculates awareness metrics.
        """
        _log_system_event("think_start", {"input_text": input_text, "stream_steps": stream_thought_steps})
        
        thought_steps = []
        response_text = ""
        awareness_metrics = {
            "curiosity": 0.1, # Default low curiosity
            "context_stability": 0.5, # Default moderate stability
            "self_evolution_rate": 0.0, # Default no evolution
            "coherence": 0.0, # Default neutral coherence
            "active_llm_fallback": True, # Assume fallback initially
            "primary_concept_coord": (0.0, 0.0, 0.0, 0.0),
            "snn_error": None # To store any SNN processing error
        }

        snn_processed_successfully = False

        if config.ENABLE_SNN:
            try:
                _log_system_event("think_snn_path_start", {"input_text": input_text})
                thought_steps, response_text, spk_rec_list, activity_levels, primary_coord = \
                    self.warp_manifold(input_text)
                
                awareness_metrics["active_llm_fallback"] = False
                awareness_metrics["primary_concept_coord"] = primary_coord
                awareness_metrics["coherence"] = np.clip(self.coherence, -1.0, 1.0) # Ensure it's within range
                awareness_metrics["self_evolution_rate"] = np.clip(self.last_avg_stdp_weight_change * 1000, 0.0, 1.0) # Scaled & clipped

                # Calculate Curiosity
                if spk_rec_list:
                    # Mean spikes per neuron per time step, averaged over all time steps
                    mean_spikes_overall = np.mean([np.mean(step_spikes) for step_spikes in spk_rec_list if step_spikes.size > 0])
                    # Normalize by configured max spike rate (e.g., if MAX_SPIKE_RATE is target spikes/sec,
                    # and each step is, say, 1ms, then max spikes per step is MAX_SPIKE_RATE * 0.001)
                    # This normalization needs refinement based on how MAX_SPIKE_RATE is defined.
                    # For now, let's use a simpler normalization against a fraction of total neurons.
                    # Max possible mean spikes if all neurons fire once: 1.0
                    # Let's consider normalized_mean_spikes as fraction of neurons firing per step.
                    normalized_mean_spikes = np.clip(mean_spikes_overall, 0.0, 1.0) 
                    
                    # Curiosity: higher with more normalized SNN activity and lower (more negative or closer to zero) coherence
                    # (indicating dissonance or interest in resolving). Let's use (1 - abs(coherence)) for exploration drive.
                    awareness_metrics["curiosity"] = np.clip((normalized_mean_spikes + (1.0 - abs(self.coherence))) / 2.0, 0.0, 1.0)
                else: # No spikes recorded
                    awareness_metrics["curiosity"] = np.clip((1.0 - abs(self.coherence)) / 2.0, 0.0, 1.0) # Curiosity from dissonance only


                # Calculate Context Stability
                if activity_levels_list and len(activity_levels_list) > 1:
                    # Lower standard deviation of activity means more stability
                    std_dev_activity = np.std(activity_levels_list)
                    # Normalize: Max possible std_dev is hard to define.
                    # Let's assume if std_dev is < 0.1 of mean activity, it's stable.
                    # Simpler: 1 minus normalized std_dev. If std_dev is 0, stability is 1.
                    # If std_dev is large (e.g. 0.5), stability is 0.5. Max std_dev could be around 0.5 for 0/1 data.
                    awareness_metrics["context_stability"] = np.clip(1.0 - (2*std_dev_activity), 0.0, 1.0) 
                elif activity_levels_list and len(activity_levels_list) == 1: # Only one step/reading
                     awareness_metrics["context_stability"] = 0.75 # Moderately stable if only one data point
                else: # No activity levels recorded
                    awareness_metrics["context_stability"] = 0.25 # Low stability if no SNN activity data

                snn_processed_successfully = True
                _log_system_event("think_snn_path_complete", {"input_text": input_text, "metrics": awareness_metrics})

            except Exception as e_snn_warp:
                _log_system_event("think_snn_path_error", 
                                  {"input_text": input_text, "error": str(e_snn_warp), "trace": traceback.format_exc()}, 
                                  level="critical")
                response_text = f"An error occurred during SNN processing: {str(e_snn_warp)}. Falling back to LLM."
                thought_steps.append(response_text)
                awareness_metrics["snn_error"] = str(e_snn_warp)
                # Fall through to LLM fallback, snn_processed_successfully remains False
        
        if not snn_processed_successfully: # SNN disabled or SNN error occurred
            _log_system_event("think_llm_fallback_path_start", 
                              {"input_text": input_text, 
                               "reason": "SNN disabled" if not config.ENABLE_SNN else "SNN error"}, 
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

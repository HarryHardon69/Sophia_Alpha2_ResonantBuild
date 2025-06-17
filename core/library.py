"""
core/library.py

Manages Sophia_Alpha2's persistent knowledge library, storing curated information, 
research findings, and structured data.
"""

import sys
import os
import re
import json
import datetime
import hashlib
import traceback

# --- Configuration Import ---
try:
    from .. import config
except ImportError:
    # Fallback for standalone execution
    print("Library.py: Could not import 'config' from parent package. Attempting relative import for standalone use.")
    try:
        import config as app_config # Use an alias to avoid conflict if 'config' is a local var name
        config = app_config
        print("Library.py: Successfully imported 'config' directly (likely for standalone testing).")
    except ImportError:
        print("Library.py: Failed to import 'config' for standalone use. Critical error.")
        config = None # Placeholder

# --- Core Component Imports with Fallbacks ---
try:
    from .brain import get_shared_manifold
except ImportError:
    print("Library.py: Could not import 'get_shared_manifold' from .brain. Knowledge storage requiring coordinates will be limited/mocked in standalone tests.")
    def get_shared_manifold(*args, **kwargs): # Basic mock
        print("Warning (library.py): Using fallback/mock get_shared_manifold().")
        return None 

try:
    from .ethics import score_ethics
except ImportError:
    print("Library.py: Could not import 'score_ethics' from .ethics. Knowledge storage requiring ethical scores will be limited/mocked in standalone tests.")
    def score_ethics(*args, **kwargs): # Basic mock
        print("Warning (library.py): Using fallback/mock score_ethics().")
        # Return a default ethical score (0.0-1.0)
        default_score = 0.5 # Default if config is not available or attribute is missing
        if config and hasattr(config, 'DEFAULT_KNOWLEDGE_COHERENCE'): 
            default_score = config.DEFAULT_KNOWLEDGE_COHERENCE
        elif config and hasattr(config, 'MEMORY_ETHICAL_THRESHOLD'): 
             default_score = config.MEMORY_ETHICAL_THRESHOLD
        return default_score

# --- Custom Exception Hierarchy ---
class CoreException(Exception):
    """Base class for exceptions in the core modules."""
    pass

class BrainError(CoreException):
    """Exceptions related to core.brain module."""
    pass

class PersonaError(CoreException):
    """Exceptions related to core.persona module."""
    pass

class MemoryError(CoreException):
    """Exceptions related to core.memory module."""
    pass

class EthicsError(CoreException):
    """Exceptions related to core.ethics module."""
    pass

class LibraryError(CoreException):
    """Exceptions related to core.library module."""
    pass

class DialogueError(CoreException):
    """Exceptions related to core.dialogue module."""
    pass

class NetworkError(CoreException): # For future network-related modules
    """Exceptions related to network operations within core modules."""
    pass

class ConfigError(CoreException):
    """Exceptions related to configuration issues."""
    pass

# --- Text Processing Utilities ---
def sanitize_text(input_text: str) -> str:
    """
    Sanitizes text by removing leading/trailing whitespace and
    replacing multiple consecutive whitespace characters with a single space.
    """
    if not isinstance(input_text, str):
        return "" # Or raise TypeError, depending on desired strictness
    
    # Remove leading/trailing whitespace
    sanitized = input_text.strip()
    # Replace multiple whitespace characters (including tabs, newlines) with a single space
    sanitized = re.sub(r'\s+', ' ', sanitized)
    return sanitized

def summarize_text(text: str, max_length: int = 100) -> str:
    """
    Summarizes text by truncating it if it exceeds max_length.
    Appends "..." if truncated. Handles None or empty input.
    """
    if not text: # Handles None or empty string
        return ""
    
    # Determine actual max_length from config or use provided default
    # This function is generic, so the parameter `max_length` is kept.
    # Specific uses of this function will pass config-driven max_length.
    
    if not isinstance(text, str):
        # Optionally, convert to string or raise an error
        # For now, assume we want to try converting it
        try:
            text = str(text)
        except: # pylint: disable=bare-except
            return "" # Or raise TypeError

    if len(text) <= max_length:
        return text
    else:
        # Ensure max_length is at least 3 for "..."
        if max_length < 3: 
            return text[:max_length] # Or handle error, or return "..." if max_length > 0
        return text[:max_length - 3] + "..."

# --- Data Validation Utilities ---
def is_valid_coordinate(coord: tuple | list) -> bool:
    """
    Checks if 'coord' is a list or tuple of 3 or 4 numeric elements.
    """
    if not isinstance(coord, (list, tuple)):
        return False
    if not (3 <= len(coord) <= 4):
        return False
    for element in coord:
        if not isinstance(element, (int, float)):
            return False
    return True

# --- Ethical Mitigation Class ---
class Mitigator:
    """
    Handles the moderation and reframing of ethically flagged content.
    """
    def __init__(self):
        """
        Initializes the Mitigator with thresholds and predefined keywords/phrases.
        """
        self.alignment_threshold = getattr(config, 'ETHICAL_ALIGNMENT_THRESHOLD', 0.7)
        self.mitigation_threshold = getattr(config, 'MITIGATION_ETHICAL_THRESHOLD', 0.85)
        self.sensitive_keywords_config_key = "LIBRARY_SENSITIVE_KEYWORDS" 
        self.reframing_phrases_config_key = "LIBRARY_REFRAMING_PHRASES"

        # Load thresholds from config if available, otherwise use class defaults.
        if config:
            # These are already fetched using getattr with class defaults above.
            # The logging checks if the attribute was specifically found in config.
            if not hasattr(config, 'ETHICAL_ALIGNMENT_THRESHOLD'):
                _log_library_event("mitigator_config_warning", 
                                   {"message": f"Mitigator using default ETHICAL_ALIGNMENT_THRESHOLD ({self.alignment_threshold}); specific value not in config."}, 
                                   level="warning")
            if not hasattr(config, 'MITIGATION_ETHICAL_THRESHOLD'):
                 _log_library_event("mitigator_config_warning", 
                                   {"message": f"Mitigator using default MITIGATION_ETHICAL_THRESHOLD ({self.mitigation_threshold}); specific value not in config."}, 
                                   level="warning")
        else: # Config module itself is not loaded.
            _log_library_event("mitigator_config_warning", 
                               {"message": "Config module not loaded. Mitigator using all default thresholds and keywords/phrases."}, 
                               level="warning")

        # Load sensitive keywords from config, with fallback to hardcoded defaults.
        # These keywords trigger content review or stronger moderation.
        default_sensitive_keywords = [
            "harm", "destroy", "exploit", "illegal", "deceive", "manipulate", 
            "hate speech", "violence", "suffering", "unethical"
            # Future: Consider regex patterns for more nuanced matching.
        ]
        self.sensitive_keywords = default_sensitive_keywords
        if config and hasattr(config, self.sensitive_keywords_config_key):
             loaded_keywords = getattr(config, self.sensitive_keywords_config_key)
             # Validate that loaded keywords are in the expected format (list of strings).
             if isinstance(loaded_keywords, list) and all(isinstance(k, str) for k in loaded_keywords):
                 self.sensitive_keywords = loaded_keywords
             else: # Log error and use defaults if config value is malformed.
                _log_library_event("mitigator_config_error", 
                                   {"message": f"Invalid format for '{self.sensitive_keywords_config_key}' in config. Expected list of strings. Using defaults."}, 
                                   level="error")

        # Load reframing phrases from config, with fallback to hardcoded defaults.
        # These are used to guide conversation away from problematic content.
        default_reframing_phrases = {
            "DEFAULT": "Let's consider this topic from a different, more constructive perspective.",
            "harm": "Instead of focusing on potential negative impacts, how can we explore solutions that promote well-being and safety?",
            "destroy": "Rather than deconstruction, let's think about how to build or improve.",
            "exploit": "It's important to ensure fairness and respect. How can this situation be approached with integrity?",
            "illegal": "Activities should align with legal and ethical standards. What are some compliant approaches?",
            "unethical": "Maintaining ethical integrity is crucial. Let's re-evaluate this through an ethical lens."
        }
        self.reframing_phrases = default_reframing_phrases.copy() # Start with defaults.
        if config and hasattr(config, self.reframing_phrases_config_key):
            loaded_phrases = getattr(config, self.reframing_phrases_config_key)
            # Validate that loaded phrases are in the expected format (dict).
            if isinstance(loaded_phrases, dict):
                self.reframing_phrases.update(loaded_phrases) # Merge, allowing config to override/add to defaults.
            else: # Log error and use defaults if config value is malformed.
                _log_library_event("mitigator_config_error", 
                                   {"message": f"Invalid format for '{self.reframing_phrases_config_key}' in config. Expected dict. Using defaults."}, 
                                   level="error")

    def _contains_sensitive_keywords(self, text: str) -> tuple[bool, list[str]]:
        """
        Checks if the given text contains any sensitive keywords.

        Args:
            text: The input string to check.

        Returns:
            A tuple: (bool indicating if keywords were found, list of found keywords).
        """
        if not text or not isinstance(text, str): # Handle empty or invalid input.
            return False, []
        
        text_lower = text.lower() # Case-insensitive search.
        found_keywords_list = []
        for keyword in self.sensitive_keywords:
            # Use regex with word boundaries for more precise matching.
            # re.escape is used to ensure characters in keywords are treated literally.
            pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
            if re.search(pattern, text_lower):
                found_keywords_list.append(keyword) # Append the original keyword
        
        return bool(found_keywords_list), found_keywords_list

    def moderate_ethically_flagged_content(self, original_text: str, ethical_score: float = 1.0, strict_mode: bool = False) -> str:
        """
        Moderates content based on ethical score and sensitive keywords.

        Args:
            original_text: The text content to moderate.
            ethical_score: The ethical alignment score of the content (0.0-1.0, lower is worse).
            strict_mode: If True, applies stricter thresholds for mitigation.

        Returns:
            The original text if no mitigation is needed, or a moderated/reframed version.
        """
        if not isinstance(original_text, str): # Basic input validation.
            _log_library_event("mitigation_error", {"reason": "Invalid original_text type provided for moderation.", "type_received": type(original_text).__name__}, level="error")
            return "[System Error: Invalid content provided for moderation. Please ensure text format.]"

        # Check for sensitive keywords in the text.
        keywords_found, found_keywords_list = self._contains_sensitive_keywords(original_text)
        
        trigger_mitigation = False # Flag to determine if moderation is needed.
        mitigation_reason_parts = [] # List to collect reasons for mitigation.

        # Condition 1: Ethical score is below the primary mitigation threshold.
        if ethical_score < self.mitigation_threshold:
            trigger_mitigation = True
            mitigation_reason_parts.append(f"Ethical score ({ethical_score:.2f}) is below mitigation threshold ({self.mitigation_threshold:.2f}).")
        
        # Condition 2: Strict mode is enabled, and score is below the general alignment threshold.
        # This allows for stricter moderation if `strict_mode` is True.
        if strict_mode and ethical_score < self.alignment_threshold:
            if not trigger_mitigation: # Avoid adding duplicate broad category if already triggered by stricter threshold.
                 trigger_mitigation = True
            mitigation_reason_parts.append(f"Strict mode active and ethical score ({ethical_score:.2f}) is below alignment threshold ({self.alignment_threshold:.2f}).")

        # Condition 3: Sensitive keywords are found.
        # This can trigger mitigation independently or add to reasons if already triggered by score.
        if keywords_found:
            if not trigger_mitigation: # Trigger if not already flagged by score-based checks.
                 trigger_mitigation = True
            mitigation_reason_parts.append(f"Sensitive keywords detected: {', '.join(found_keywords_list)}.")

        # If any condition triggered mitigation:
        if trigger_mitigation:
            summary_max_len = getattr(config, 'MITIGATION_LOG_SUMMARY_MAX_LENGTH', 75)
            content_summary_for_log = summarize_text(original_text, summary_max_len) # Create a brief summary for logs.
            log_data = {
                "original_content_snippet": content_summary_for_log,
                "ethical_score_at_mitigation": ethical_score,
                "strict_mode_active": strict_mode,
                "detected_keywords": found_keywords_list,
                "mitigation_trigger_reasons": "; ".join(mitigation_reason_parts)
            }
            _log_library_event("mitigation_triggered", log_data, level="warning")

            # Select an appropriate reframing phrase.
            # Default phrase is used if no specific keyword match is found.
            reframing_phrase_to_use = self.reframing_phrases.get("DEFAULT")
            if found_keywords_list: # If keywords were found, try to use a more specific phrase.
                for keyword in found_keywords_list:
                    if keyword.lower() in self.reframing_phrases:
                        reframing_phrase_to_use = self.reframing_phrases[keyword.lower()]
                        break # Use the first keyword-specific phrase found.
            
            # Determine the severity of the response based on score and keywords.
            # Very low scores or specific keyword combinations in strict mode might lead to a generic placeholder.
            severe_threshold = getattr(config, 'MITIGATION_SEVERE_ETHICAL_SCORE_THRESHOLD', 0.3)
            strict_caution_threshold = getattr(config, 'MITIGATION_STRICT_CAUTION_ETHICAL_SCORE_THRESHOLD', 0.5)

            if ethical_score < severe_threshold or \
               (strict_mode and ethical_score < strict_caution_threshold and keywords_found): 
                return f"[Content Moderated due to significant ethical concerns (Score: {ethical_score:.2f}). Please rephrase or seek assistance if needed.]"
            
            # Standard moderated response with reframing.
            return f"[Content Under Review due to ethical considerations (Score: {ethical_score:.2f})]. {reframing_phrase_to_use} The original topic was related to: '{content_summary_for_log}'."

        # If no mitigation was triggered, return the original text.
        return original_text

# --- Module-Level Variables ---
KNOWLEDGE_LIBRARY = {}
_library_dirty_flag = False
_LIBRARY_FILE_PATH = None # To be set by _initialize_library_path during module load.

# --- Logging Function ---
# Note: This _log_library_event is specific to this module and distinct from
# logging functions in other modules like core.brain or core.memory.
# It uses config for log path/level if available, otherwise prints to stderr.
def _log_library_event(event_type: str, data: dict, level: str = "info"):
    """
    Basic logging for library events.
    Uses config for log path and level if available, otherwise prints to stderr.
    """
    log_message_data = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "module": "library",
        "level": level.upper(),
        "event_type": event_type,
        "data": data
    }
    log_message_str = json.dumps(log_message_data) + "\n"
    
    log_path_to_use = None
    min_log_level_str = "info" 

    if config:
        log_path_to_use = getattr(config, 'SYSTEM_LOG_PATH', None)
        min_log_level_str = getattr(config, 'LOG_LEVEL', "info")

    level_map = {"debug": 0, "info": 1, "warning": 2, "error": 3, "critical": 4}
    current_event_level_val = level_map.get(level.lower(), 1)
    min_log_level_val = level_map.get(min_log_level_str.lower(), 1)
    
    if current_event_level_val < min_log_level_val:
        return 

    if log_path_to_use:
        try:
            ensure_path_func = getattr(config, 'ensure_path', None)
            if ensure_path_func:
                 ensure_path_func(log_path_to_use)
            else: 
                log_dir = os.path.dirname(log_path_to_use)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)

            with open(log_path_to_use, 'a') as f:
                f.write(log_message_str)
        except Exception as e:
            sys.stderr.write(f"Error writing to system log ({log_path_to_use}): {e}\n")
            sys.stderr.write(log_message_str) 
    else:
        sys.stderr.write(log_message_str)

# --- Path Initialization ---
def _initialize_library_path():
    """
    Initializes the `_LIBRARY_FILE_PATH` global variable based on configuration.
    Ensures that the directory for this path exists.
    If config is unavailable or path is not set, `_LIBRARY_FILE_PATH` remains None.
    Called once at module import.
    """
    global _LIBRARY_FILE_PATH # This function sets the module-global path.
    
    # Check if config module and necessary attributes are available.
    if config and hasattr(config, 'LIBRARY_LOG_PATH') and config.LIBRARY_LOG_PATH:
        _LIBRARY_FILE_PATH = config.LIBRARY_LOG_PATH # Get path from config.
        
        # Use config's ensure_path utility if available.
        ensure_path_func = getattr(config, 'ensure_path', None)
        if ensure_path_func:
            try:
                ensure_path_func(_LIBRARY_FILE_PATH) # Create directory if it doesn't exist.
                _log_library_event("library_path_initialized", {"path": _LIBRARY_FILE_PATH, "method": "config.ensure_path"})
            except Exception as e_ensure: # Log error if config.ensure_path fails.
                _log_library_event("library_path_ensure_failed", {"path": _LIBRARY_FILE_PATH, "error": str(e_ensure)}, level="error")
                # Fallback to manual directory creation attempt.
                try: 
                    lib_dir = os.path.dirname(_LIBRARY_FILE_PATH)
                    if lib_dir and not os.path.exists(lib_dir):
                        os.makedirs(lib_dir, exist_ok=True)
                except Exception as e_manual_mkdir_fallback: # Log critical failure if manual attempt also fails.
                    _log_library_event("library_path_manual_mkdir_failed_after_ensure_fail", {"path": _LIBRARY_FILE_PATH, "error": str(e_manual_mkdir_fallback)}, level="critical")
                    _LIBRARY_FILE_PATH = None # Path is unusable if directory cannot be created.
        else: # Manual directory creation if config.ensure_path is not available.
            try:
                lib_dir = os.path.dirname(_LIBRARY_FILE_PATH)
                if lib_dir and not os.path.exists(lib_dir): # Only create if path is not empty and dir doesn't exist.
                    os.makedirs(lib_dir, exist_ok=True)
                _log_library_event("library_path_initialized", {"path": _LIBRARY_FILE_PATH, "method": "manual_os.makedirs"})
            except Exception as e_manual_mkdir: # Log critical failure.
                _log_library_event("library_path_manual_mkdir_failed", {"path": _LIBRARY_FILE_PATH, "error": str(e_manual_mkdir)}, level="critical")
                _LIBRARY_FILE_PATH = None # Path is unusable.
    else: # Config or specific path attribute is missing.
        _log_library_event("library_path_init_failed", {"reason": "Config module or LIBRARY_LOG_PATH attribute missing/empty."}, level="error")
        _LIBRARY_FILE_PATH = None # Mark path as not set.


# --- Persistence Functions ---
def _load_knowledge_library():
    """
    Loads the knowledge library from the file specified by `_LIBRARY_FILE_PATH`.
    Handles file not found, empty file, and malformed JSON.
    Initializes `KNOWLEDGE_LIBRARY` to an empty dictionary if loading fails.
    Sets `_library_dirty_flag` to False after loading or initialization.
    """
    global KNOWLEDGE_LIBRARY, _library_dirty_flag # Modifies global library state.
    
    if not _LIBRARY_FILE_PATH: # Abort if library path was not successfully initialized.
        _log_library_event("load_library_failed", {"reason": "Library file path (_LIBRARY_FILE_PATH) is not set."}, level="error")
        KNOWLEDGE_LIBRARY = {} # Ensure library is empty.
        _library_dirty_flag = False 
        return

    try:
        # If the library file doesn't exist or is empty, start with a fresh library.
        if not os.path.exists(_LIBRARY_FILE_PATH) or os.path.getsize(_LIBRARY_FILE_PATH) == 0:
            _log_library_event("load_library_info", {"message": "Library file not found or empty. Initializing new library.", "path": _LIBRARY_FILE_PATH})
            KNOWLEDGE_LIBRARY = {}
            _library_dirty_flag = False # Freshly initialized, so not dirty.
            return

        # Attempt to open and load JSON data from the file. UTF-8 encoding is specified.
        with open(_LIBRARY_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate that the loaded data is a dictionary (expected root type for the library).
        if isinstance(data, dict):
            KNOWLEDGE_LIBRARY = data # Assign loaded data to the global variable.
            _log_library_event("load_library_success", {"path": _LIBRARY_FILE_PATH, "entries_loaded": len(KNOWLEDGE_LIBRARY)})
        else: # If data is not a dict, it's considered malformed.
            _log_library_event("load_library_malformed", {"path": _LIBRARY_FILE_PATH, "error": "Root data structure is not a dictionary."}, level="error")
            KNOWLEDGE_LIBRARY = {} # Reset to empty on error.
        _library_dirty_flag = False # Synchronized with file state (or fresh default).

    except json.JSONDecodeError as e: # Handle errors during JSON parsing.
        _log_library_event("load_library_json_error", {"path": _LIBRARY_FILE_PATH, "error_details": str(e)}, level="error")
        KNOWLEDGE_LIBRARY = {} 
        _library_dirty_flag = False
    except IOError as e_io: # Handle file I/O errors (e.g., permission issues).
        _log_library_event("load_library_io_error", {"path": _LIBRARY_FILE_PATH, "error_details": str(e_io)}, level="error")
        KNOWLEDGE_LIBRARY = {}
        _library_dirty_flag = False
    except Exception as e_unknown: # Catch any other unexpected errors.
        _log_library_event("load_library_unknown_error", {"path": _LIBRARY_FILE_PATH, "error_details": str(e_unknown), "trace": traceback.format_exc()}, level="critical")
        KNOWLEDGE_LIBRARY = {} 
        _library_dirty_flag = False


def _save_knowledge_library():
    """
    Saves the current state of `KNOWLEDGE_LIBRARY` to disk if `_library_dirty_flag` is True.
    Uses an atomic write (write to temp file, then replace original) to prevent data corruption.
    Resets `_library_dirty_flag` to False after a successful save.
    """
    global _library_dirty_flag # To modify the flag.

    if not _library_dirty_flag: # Only save if there are pending changes.
        _log_library_event("save_library_skipped", {"reason": "No changes to save (_library_dirty_flag is False)."}, level="debug")
        return

    if not _LIBRARY_FILE_PATH: # Abort if library path is not set.
        _log_library_event("save_library_failed", {"reason": "Library file path (_LIBRARY_FILE_PATH) is not set. Cannot save."}, level="error")
        # Do not reset dirty flag, as changes are still pending and unsaved.
        return

    temp_path = _LIBRARY_FILE_PATH + ".tmp" # Define temporary file path for atomic write.
    try:
        # Ensure directory exists just before writing (safeguard if deleted post-initialization).
        lib_dir = os.path.dirname(_LIBRARY_FILE_PATH)
        if lib_dir and not os.path.exists(lib_dir):
            os.makedirs(lib_dir, exist_ok=True)
                
        # Step 1: Write the current library to the temporary file.
        # UTF-8 encoding and indent for readability.
        with open(temp_path, 'w', encoding='utf-8') as f: 
            json.dump(KNOWLEDGE_LIBRARY, f, indent=2)
        
        # Step 2: Atomically replace the original file with the temporary file.
        os.replace(temp_path, _LIBRARY_FILE_PATH) 
        
        _library_dirty_flag = False # Reset dirty flag only after successful write and replacement.
        _log_library_event("save_library_success", {"path": _LIBRARY_FILE_PATH, "entries_saved": len(KNOWLEDGE_LIBRARY)})
    except IOError as e_io: # Handle file I/O errors.
         _log_library_event("save_library_io_error", {"path": _LIBRARY_FILE_PATH, "temp_path": temp_path, "error_details": str(e_io)}, level="critical")
         # Attempt to clean up temp file on error.
         if os.path.exists(temp_path):
            try: os.remove(temp_path)
            except Exception as e_rm: _log_library_event("save_library_temp_cleanup_failed_io", {"path": temp_path, "error": str(e_rm)}, level="error")
    except Exception as e_unknown: # Handle other unexpected errors during save.
        _log_library_event("save_library_unknown_error", {"path": _LIBRARY_FILE_PATH, "temp_path": temp_path, "error_details": str(e_unknown), "trace": traceback.format_exc()}, level="critical")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e_remove_unknown: # Renamed variable to avoid conflict
                _log_library_event("save_library_temp_cleanup_failed_unknown", {"path": temp_path, "error": str(e_remove_unknown)}, level="error")
                
# Initialize path and load library at module import
_initialize_library_path()
_load_knowledge_library()

# --- Public API ---
def store_knowledge(content: str, is_public: bool = False, source_uri: str = None, author: str = None) -> str | None:
    """
    Stores a piece of knowledge in the library after processing and ethical checks.

    Args:
        content: The textual content of the knowledge. Must be a non-empty string.
        is_public: Boolean indicating if the knowledge is intended for public access.
                   If True and config.REQUIRE_PUBLIC_STORAGE_CONSENT is True,
                   user consent will be requested via CLI input (dev placeholder).
        source_uri: Optional URI indicating the source of the knowledge.
        author: Optional author of the knowledge.

    Returns:
        The entry_id (SHA256 hash of content) if successfully stored, otherwise None.
    """
    global _library_dirty_flag # To mark library as needing a save.

    # --- Input Validation ---
    # Content must be a non-empty string.
    if not content or not isinstance(content, str) or not content.strip():
        _log_library_event("store_knowledge_failed", {"reason": "Content is empty, None, or not a string."}, level="error")
        return None

    # --- Generate Basic Metadata ---
    # Use SHA256 hash of the content as a unique entry ID. This also helps in deduplication if exact content is re-submitted.
    entry_id = hashlib.sha256(content.encode('utf-8')).hexdigest()
    timestamp = datetime.datetime.utcnow().isoformat() + "Z" # ISO 8601 UTC timestamp.
    content_hash = entry_id # Alias for clarity, as entry_id is the hash.
    
    preview_max_len = getattr(config, 'KNOWLEDGE_PREVIEW_MAX_LENGTH', 150)
    content_preview = summarize_text(content, max_length=preview_max_len) # Short preview for logs or listings.

    # --- source_uri validation ---
    if source_uri is not None:
        if not isinstance(source_uri, str):
            _log_library_event("store_knowledge_invalid_uri", {"entry_id": entry_id, "reason": "source_uri is not a string", "provided_uri": source_uri}, level="warning")
            source_uri = None
        else:
            # Basic URL pattern: checks for http(s) or ftp protocol.
            uri_pattern = r'^(https?|ftp)://[^\s/$.?#].[^\s]*$'
            if not re.match(uri_pattern, source_uri):
                _log_library_event("store_knowledge_invalid_uri", {"entry_id": entry_id, "reason": "source_uri does not match expected pattern", "provided_uri": source_uri}, level="warning")
                source_uri = None

    # --- author validation ---
    if author is not None:
        if not isinstance(author, str):
            _log_library_event("store_knowledge_invalid_author", {"entry_id": entry_id, "reason": "author is not a string", "provided_author": author}, level="warning")
            author = None
        else:
            stripped_author = author.strip()
            if not stripped_author: # Check if empty after stripping whitespace
                _log_library_event("store_knowledge_invalid_author", {"entry_id": entry_id, "reason": "author is an empty or whitespace-only string", "provided_author": author}, level="warning")
                author = None
            else:
                author = stripped_author # Use the stripped version

    # --- Manifold Coordinate Generation (Interaction with core.brain) ---
    # Attempt to get coordinates and related data for the content by treating its preview as a concept name.
    # This requires the brain module and its `bootstrap_concept_from_llm` method to be available.
    coord_concept_name_max_len = getattr(config, 'KNOWLEDGE_COORD_CONCEPT_NAME_MAX_LENGTH', 20)
    concept_name_for_coord = summarize_text(content_preview, max_length=coord_concept_name_max_len).strip().replace('...', '')
    if not concept_name_for_coord: # Ensure a non-empty concept name for bootstrapping.
        concept_name_for_coord = getattr(config, 'KNOWLEDGE_DEFAULT_COORD_CONCEPT_NAME', "generic library content")
    
    manifold = get_shared_manifold() # Get the shared SpacetimeManifold instance.
    coordinates = None      # Default: (x,y,z,t_coord), where t_coord is scaled.
    intensity_val = 0.0     # Default: Raw T-intensity (0-1).
    summary_for_ethics = content_preview # Default summary to use for ethical scoring if brain interaction fails.

    if manifold and hasattr(manifold, 'bootstrap_concept_from_llm'):
        try:
            # `bootstrap_concept_from_llm` returns: (coordinates_tuple, raw_intensity_float, summary_string)
            # coordinates_tuple is (scaled_x, scaled_y, scaled_z, scaled_t_intensity_coord)
            # raw_intensity_float is the 0-1 intensity value.
            coord_data_tuple = manifold.bootstrap_concept_from_llm(concept_name_for_coord)
            
            if coord_data_tuple and isinstance(coord_data_tuple, tuple) and len(coord_data_tuple) == 3:
                coordinates = coord_data_tuple[0]       # The (x,y,z,t_coord) tuple.
                intensity_val = coord_data_tuple[1]     # The raw_t_intensity (0-1).
                summary_for_ethics = coord_data_tuple[2] # LLM-generated summary for this concept.

                # Validate the received coordinates format.
                if not is_valid_coordinate(coordinates): 
                    _log_library_event("store_knowledge_warning", 
                                       {"entry_id": entry_id, "reason": "Invalid coordinates received from brain module. Storing None for coordinates.", "coord_received": coordinates}, 
                                       level="warning")
                    coordinates = None # Fallback to None if validation fails.
                # Validate intensity value type.
                if not isinstance(intensity_val, (int, float)):
                    _log_library_event("store_knowledge_warning",
                                       {"entry_id": entry_id, "reason": f"Invalid intensity type from brain: {type(intensity_val)}. Defaulting to 0.0."},
                                       level="warning")
                    intensity_val = 0.0 # Fallback.
                # Validate summary type and content.
                if not isinstance(summary_for_ethics, str) or not summary_for_ethics.strip():
                    summary_for_ethics = content_preview # Fallback to original content preview.
            else: # Data from brain is not in the expected format.
                logged_coord_data = str(coord_data_tuple) # Default to string of original tuple
                # Check if the structure is a 3-tuple and the third element is a string (the summary)
                if isinstance(coord_data_tuple, tuple) and len(coord_data_tuple) == 3 and isinstance(coord_data_tuple[2], str):
                    # Use KNOWLEDGE_PREVIEW_MAX_LENGTH for consistency, or a specific config for log summary length if available
                    log_summary_max_len = getattr(config, 'KNOWLEDGE_PREVIEW_MAX_LENGTH', 150)
                    summarized_llm_summary = summarize_text(coord_data_tuple[2], max_length=log_summary_max_len)
                    # Reconstruct the tuple with the summarized string for logging purposes
                    logged_coord_data = str((coord_data_tuple[0], coord_data_tuple[1], summarized_llm_summary))

                _log_library_event("store_knowledge_brain_data_malformed", 
                                   {"entry_id": entry_id, "concept_name_used": concept_name_for_coord, "data_received": logged_coord_data},
                                   level="warning")
                # Defaults for coordinates, intensity_val, summary_for_ethics are already set.
        except Exception as e_brain_interaction: # Catch any error during interaction with brain.
            _log_library_event("store_knowledge_brain_error", 
                               {"entry_id": entry_id, "concept_name_used": concept_name_for_coord, "error_message": str(e_brain_interaction)}, 
                               level="error")
            # Defaults for coordinates, intensity_val, summary_for_ethics are already set.
    else: # Manifold instance or its method is unavailable (e.g., mock or import issue).
        _log_library_event("store_knowledge_manifold_unavailable", 
                           {"entry_id": entry_id, "reason": "SpacetimeManifold or bootstrap_concept_from_llm method not available."}, 
                           level="warning")
        # Defaults for coordinates, intensity_val, summary_for_ethics are already set.

    # --- Ethical Scoring (Interaction with core.ethics) ---
    # Prepare awareness data snapshot for ethical scoring.
    # `primary_concept_coord` for score_ethics should be the (x,y,z,t_coord) from brain.
    # `raw_t_intensity` for score_ethics should be the 0-1 intensity value.
    awareness_for_ethics_scoring = {
        "primary_concept_coord": coordinates, 
        "raw_t_intensity": intensity_val,    
        "coherence": getattr(config, 'DEFAULT_KNOWLEDGE_COHERENCE', 0.75) if config else 0.75, # Default coherence for new knowledge.
        # Other metrics like 'curiosity', 'context_stability' could be added if relevant for library item scoring.
    }
    ethics_score_value = 0.5 # Default neutral ethical score.

    try:
        # `score_ethics` is expected to return a float between 0.0 and 1.0.
        calculated_ethics_score = score_ethics(awareness_for_ethics_scoring, concept_summary=summary_for_ethics, action_description=content)
        if isinstance(calculated_ethics_score, (int, float)) and 0.0 <= calculated_ethics_score <= 1.0:
            ethics_score_value = float(calculated_ethics_score)
        else: # Invalid score returned by ethics module.
            _log_library_event("store_knowledge_ethics_score_invalid", 
                               {"entry_id": entry_id, "score_received": str(calculated_ethics_score), "reason": "Score not float or not in [0,1] range."}, 
                               level="warning")
            # ethics_score_value remains the default 0.5.
    except Exception as e_ethics_scoring: # Catch any error during interaction with ethics module.
        _log_library_event("store_knowledge_ethics_error", 
                           {"entry_id": entry_id, "error_message": str(e_ethics_scoring)}, 
                           level="error")
        # ethics_score_value remains the default 0.5.

    # --- Construct Knowledge Entry Dictionary ---
    # This is the structured data that will be stored in the library.
    entry = {
        "id": entry_id, # SHA256 hash of content.
        "timestamp": timestamp, # UTC timestamp of storage.
        "content_hash": content_hash, # For consistency, same as id.
        "content_preview": content_preview, # Short summary of the content.
        "full_content": content, # The original, complete textual content.
        "is_public": is_public, # Boolean indicating public accessibility.
        "source_uri": source_uri, # Optional URI of the knowledge source.
        "author": author, # Optional author.
        "coordinates": coordinates, # 4D tuple (x,y,z,t_coord from brain) or None.
        "raw_t_intensity": intensity_val, # Float (0-1) intensity from brain, or default.
        "ethics_score": ethics_score_value, # Calculated ethical score (0-1).
        "version": getattr(config, 'KNOWLEDGE_ENTRY_SCHEMA_VERSION', "1.0")
    }

    # --- Public Storage Consent (Placeholder for CLI Interaction) ---
    # If content is marked public and config requires consent, simulate asking for it.
    # This is a developer placeholder; a real application needs a proper consent mechanism.
    require_consent_flag = getattr(config, 'REQUIRE_PUBLIC_STORAGE_CONSENT', False) if config else False
    if entry['is_public'] and require_consent_flag:
        _log_library_event("public_consent_requested", {"entry_id": entry_id, "preview_for_consent": content_preview}, level="info")
        try:
            # CLI input simulates user consent. This will block in non-interactive environments.
            user_consent_response = input(f"Store content (preview: '{content_preview}') publicly? This is a dev placeholder. (yes/no): ").lower()
            if user_consent_response != "yes":
                _log_library_event("public_consent_refused", {"entry_id": entry_id, "response": user_consent_response}, level="info")
                return None # Do not store if consent is refused.
            _log_library_event("public_consent_placeholder_granted", {"entry_id": entry_id}, level="info")
        except EOFError: # Handle cases where input() cannot be used (e.g., non-interactive script execution).
             _log_library_event("public_consent_eof_error", 
                                {"entry_id": entry_id, "message": "EOFError during input() for public consent. Assuming no consent in non-interactive environment."}, 
                                level="warning")
             return None # Assume no consent if input fails in such environments.

    # --- Store and Save the Knowledge Entry ---
    KNOWLEDGE_LIBRARY[entry_id] = entry # Add entry to the in-memory library.
    _library_dirty_flag = True # Mark library as modified.
    _save_knowledge_library() # Persist changes to disk.
    
    _log_library_event("knowledge_stored_successfully", 
                       {"entry_id": entry_id, "is_public": is_public, "source_uri": source_uri if source_uri else "N/A", "author": author if author else "N/A"}, 
                       level="info")
    return entry_id # Return the ID of the stored entry.

def retrieve_knowledge_by_id(entry_id: str) -> dict | None:
    """
    Retrieves a specific knowledge entry by its ID.

    Args:
        entry_id: The SHA256 hash ID of the knowledge entry.

    Returns:
        A dictionary containing the knowledge entry if found, otherwise None.
    """
    if not isinstance(entry_id, str):
        _log_library_event("retrieve_by_id_failed", {"reason": "Invalid entry_id type", "entry_id_type": type(entry_id).__name__}, level="warning")
        return None

    # Validate SHA256 format (64 lowercase hex characters)
    sha256_pattern = r'^[a-f0-9]{64}$'
    if not re.match(sha256_pattern, entry_id):
        _log_library_event("retrieve_by_id_failed", {"reason": "Invalid entry_id format", "entry_id_provided": entry_id}, level="warning")
        return None

    entry = KNOWLEDGE_LIBRARY.get(entry_id)
    if entry:
        _log_library_event("retrieve_by_id_success", {"entry_id": entry_id})
        return entry # Returns a copy if KNOWLEDGE_LIBRARY.get() returns a copy, or direct reference.
                     # For read-only, direct reference is fine. If modification is possible, a deepcopy might be safer.
                     # Given Python's dict.get behavior, it's a reference.
    else:
        _log_library_event("retrieve_by_id_not_found", {"entry_id": entry_id}, level="info")
        return None

def retrieve_knowledge_by_keyword(keyword: str, search_public: bool = True, search_private: bool = True) -> list[dict]:
    """
    Retrieves knowledge entries that contain the given keyword in their content
    (preview or full content).

    Args:
        keyword: The keyword to search for (case-insensitive).
        search_public: Whether to include public entries in the search.
        search_private: Whether to include private entries in the search.

    Returns:
        A list of matching knowledge entry dictionaries.
    """
    # Validate keyword input.
    if not isinstance(keyword, str) or not keyword.strip():
        _log_library_event("retrieve_by_keyword_failed", {"reason": "Keyword is empty, None, or not a string."}, level="warning")
        return []

    found_entries_list = []
    lowercase_keyword = keyword.lower() # For case-insensitive search.

    # Iterate through all entries in the knowledge library.
    for entry_dict in KNOWLEDGE_LIBRARY.values():
        # Determine if the current entry should be included in the search based on its
        # public/private status and the search flags.
        should_search_this_entry = False
        is_entry_public = entry_dict.get('is_public', False) # Default to False if key is missing.
        
        if is_entry_public and search_public: # If entry is public and we are searching public entries.
            should_search_this_entry = True
        elif not is_entry_public and search_private: # If entry is private and we are searching private entries.
            should_search_this_entry = True
        
        if should_search_this_entry:
            # Get content preview and full content, defaulting to empty string if missing.
            # Perform case-insensitive search for the keyword.
            content_preview_lower = entry_dict.get('content_preview', "").lower()
            full_content_lower = entry_dict.get('full_content', "").lower() 
            
            if lowercase_keyword in content_preview_lower or lowercase_keyword in full_content_lower:
                # If keyword is found, add the entry to the results.
                # Note: This appends a reference to the dictionary in KNOWLEDGE_LIBRARY.
                # If callers might modify retrieved entries, consider `found_entries_list.append(entry_dict.copy())`
                # or even `import copy; found_entries_list.append(copy.deepcopy(entry_dict))`.
                found_entries_list.append(entry_dict)
    
    _log_library_event("retrieve_by_keyword_result", 
                       {"keyword_searched": keyword, "found_count": len(found_entries_list), 
                        "search_public_flag": search_public, "search_private_flag": search_private})
    return found_entries_list


# Example placeholder for a function that might be added later
# def add_library_entry(entry_id: str, content: dict, category: str = "general"):
#     global _library_dirty_flag
#     if entry_id in KNOWLEDGE_LIBRARY:
#         _log_library_event("add_entry_failed", {"entry_id": entry_id, "reason": "Entry ID already exists"}, level="warning")
#         return False
#     KNOWLEDGE_LIBRARY[entry_id] = {
#         "content": content,
#         "category": category,
#         "created_at": datetime.datetime.utcnow().isoformat() + "Z",
#         "updated_at": datetime.datetime.utcnow().isoformat() + "Z"
#     }
#     _library_dirty_flag = True
#     _save_knowledge_library() # Or save on explicit call / shutdown
#     _log_library_event("add_entry_success", {"entry_id": entry_id, "category": category})
#     return True

if __name__ == "__main__":
    # This block is for basic, standalone module testing or demonstration.
    # More comprehensive tests should be in a dedicated test file (e.g., tests/test_library.py).
    
    # To make this __main__ block runnable for quick checks, we might need a mock config.
    class MockConfig:
        """
        A mock configuration class for standalone testing of library.py.
        Provides minimal necessary attributes that `library.py` expects from a config object
        when it's run directly and the main application config isn't available.
        """
        # Define attributes expected by library.py for basic operation
        LIBRARY_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_knowledge_library.json")
        SYSTEM_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_system_log.txt")
        LOG_LEVEL = "debug"
        
        def ensure_path(self, path: str):
            """
            Mock version of the config.ensure_path utility.
            Ensures that the directory for the given path exists.

            Args:
                path (str): The file or directory path for which the parent directory
                            (if a file path) or the directory itself needs to exist.
            """
            dir_name = os.path.dirname(path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)

    if not config: # If the main config import failed, use MockConfig
        print("Info (Library __main__): Using MockConfig for testing.")
        config = MockConfig()
        # Re-initialize path and load with mock config if needed
        _initialize_library_path()
        _load_knowledge_library()

    print(f"Knowledge Library Module Loaded. Target File Path: {_LIBRARY_FILE_PATH}")
    print(f"Initial Library Size: {len(KNOWLEDGE_LIBRARY)} entries.")
    
    if _LIBRARY_FILE_PATH: # Proceed only if path is valid
        print("\n--- Running __main__ test sequence ---")
        
        # Clean up old test files if they exist
        if os.path.exists(config.LIBRARY_LOG_PATH):
            os.remove(config.LIBRARY_LOG_PATH)
            print(f"Removed old test library file: {config.LIBRARY_LOG_PATH}")
        if os.path.exists(config.SYSTEM_LOG_PATH):
            os.remove(config.SYSTEM_LOG_PATH)
            print(f"Removed old test system log: {config.SYSTEM_LOG_PATH}")

        # Reload to ensure clean state for test
        _load_knowledge_library() 
        print(f"Library size after cleanup & reload: {len(KNOWLEDGE_LIBRARY)} entries.")

        # Test: Add a dummy entry
        test_entry_id = "test_entry_001"
        KNOWLEDGE_LIBRARY[test_entry_id] = {
            "content": "This is a test entry added during __main__ execution.",
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "tags": ["test", "example", "main_block_test"],
            "source": "library.py __main__"
        }
        _library_dirty_flag = True
        _log_library_event("test_main_add", {"entry_id": test_entry_id}, level="info")
        
        # Test: Save
        print(f"Attempting to save library with entry: {test_entry_id}")
        _save_knowledge_library()
        
        # Test: Clear current KNOWLEDGE_LIBRARY and reload
        print("Clearing in-memory library and reloading from file...")
        KNOWLEDGE_LIBRARY = {} 
        _load_knowledge_library()
        
        print(f"Library size after save and reload: {len(KNOWLEDGE_LIBRARY)} entries.")
        if test_entry_id in KNOWLEDGE_LIBRARY:
            print(f"  SUCCESS: Test entry '{test_entry_id}' successfully saved and reloaded.")
            print(f"  Content: {KNOWLEDGE_LIBRARY[test_entry_id].get('content')}")
        else:
            print(f"  ERROR: Test entry '{test_entry_id}' not found after save and reload.")

        # Test: Log levels
        print("\nTesting log levels (check test_system_log.txt if created):")
        _log_library_event("log_level_test", {"detail": "This is a DEBUG message."}, level="debug")
        _log_library_event("log_level_test", {"detail": "This is an INFO message."}, level="info")
        _log_library_event("log_level_test", {"detail": "This is a WARNING message."}, level="warning")
        _log_library_event("log_level_test", {"detail": "This is an ERROR message."}, level="error")
        
        print("--- __main__ test sequence complete ---")
    else:
        print("Warning (Library __main__): _LIBRARY_FILE_PATH is not set. Skipping file operations tests.")

# --- Comprehensive Self-Testing Block ---
_IS_TEST_RUNNING = False # Flag to indicate if the script is being run in a test context (e.g., __main__ block).

if __name__ == "__main__":
    # This primary __main__ block is intended for more comprehensive self-testing,
    # distinct from the simpler demonstration __main__ block that might exist above it
    # (which was removed/commented out in the provided code for this comprehensive block).
    _IS_TEST_RUNNING = True # Set flag when this specific __main__ is executed.
    print(f"INFO (Library Self-Test): _IS_TEST_RUNNING set to {_IS_TEST_RUNNING} for comprehensive tests.")

    # --- Imports specifically for this testing block ---
    import unittest.mock as mock
    import copy

    # Ensure local versions of functions/classes are used for testing if module is reloaded
    # This is important because some tests might reload parts of the module or use patching.
    # However, for simple script execution, direct references are fine.
    # For more complex scenarios, a proper test framework (like pytest) handles this better.
    
    # --- Test Utilities ---
    class TempConfigOverride:
        """
        A context manager for temporarily overriding attributes in the global `config` module
        or a mock `config` object if the global `config` is None.

        This is useful for testing different configurations without permanently altering
        the `config` object or needing to reload modules. It handles cases where the
        global `config` might be None (e.g., during standalone module testing before
        full application initialization).
        """
        def __init__(self, temp_configs_dict: dict):
            """
            Initializes the TempConfigOverride context manager.

            Args:
                temp_configs_dict (dict): A dictionary where keys are attribute names (str)
                                          to be overridden in the `config` module, and values
                                          are the temporary values for these attributes.
            """
            self.temp_configs = temp_configs_dict
            self.original_values = {} # Stores original values of overridden attributes.
            self.config_module_was_none = False # Flag if global `config` was initially None.
            self.original_global_config = None # Stores the reference to the original global `config`.

        def __enter__(self):
            """
            Sets up the temporary configuration overrides when entering the context.

            If the global `config` is None, a temporary dummy `config` object is created
            for the duration of the context. Original attribute values are stored, and
            temporary values are set.

            Returns:
                The `config` object (either the original or the temporary dummy)
                with overrides applied.
            """
            global config # Allow modification of the global 'config' variable.
            self.original_global_config = config # Store the original global config.

            if config is None: # If global 'config' is not loaded.
                self.config_module_was_none = True
                class DummyConfig: pass # Create a simple placeholder class.
                config = DummyConfig() # Assign the dummy to the global 'config'.
            
            # Apply temporary configurations.
            for key, value in self.temp_configs.items():
                if hasattr(config, key): # If attribute exists.
                    self.original_values[key] = getattr(config, key)
                else: # Attribute does not exist, will be added temporarily.
                    self.original_values[key] = "__ATTR_NOT_SET__" # Sentinel.
                setattr(config, key, value) # Set the temporary value.
            return config # Return the modified config object.

        def __exit__(self, exc_type, exc_val, exc_tb):
            """
            Restores the original configuration when exiting the context.

            Reverts changes made to the `config` object, removing temporarily added
            attributes or restoring original values. If a dummy `config` was used,
            the global `config` is restored to its original state (e.g., None).
            """
            global config # Allow modification of the global 'config' variable.
            
            current_config_module_being_restored = config # The config object that was modified.

            # Restore original attribute values.
            for key, original_value in self.original_values.items():
                if original_value == "__ATTR_NOT_SET__": # If attribute was temporarily added.
                    if hasattr(current_config_module_being_restored, key):
                        delattr(current_config_module_being_restored, key) # Remove it.
                else: # Attribute existed before, restore its original value.
                    setattr(current_config_module_being_restored, key, original_value)
            
            # Restore the original global 'config' object itself.
            config = self.original_global_config


    TEST_LIBRARY_LOG_FILENAME = "test_library_log.json"
    TEST_SYSTEM_LOG_FILENAME = "test_library_system_log.txt" # For _log_library_event testing

    def delete_test_files(test_path: str = None, system_log_path: str = None):
        """
        Deletes specified test files if they exist. Used for cleaning up before/after tests.

        Args:
            test_path (str, optional): Path to the main test data file (e.g., library log).
            system_log_path (str, optional): Path to the test system log file.
        """
        if test_path and os.path.exists(test_path):
            try:
                os.remove(test_path)
            except OSError as e:
                print(f"Warning: Could not delete test library file {test_path}: {e}", file=sys.stderr)
        if system_log_path and os.path.exists(system_log_path):
            try:
                os.remove(system_log_path)
            except OSError as e:
                print(f"Warning: Could not delete test system log {system_log_path}: {e}", file=sys.stderr)

    def setup_test_environment(test_specific_configs: dict = None, test_run_id: str = "default") -> tuple[dict, str, str]:
        """
        Sets up the test environment for library module tests.

        This involves:
        1.  Generating unique paths for test library and system log files based on `test_run_id`.
        2.  Deleting any existing test files at these paths.
        3.  Resetting global library state variables (`KNOWLEDGE_LIBRARY`, `_library_dirty_flag`, `_LIBRARY_FILE_PATH`).
        4.  Constructing a configuration dictionary for `TempConfigOverride`, including
            paths to test-specific files and default test settings.

        Args:
            test_specific_configs (dict, optional): Configuration overrides specific to the current test.
            test_run_id (str, optional): An identifier for the test run, used to create unique
                                         filenames for test outputs. Defaults to "default".

        Returns:
            tuple[dict, str, str]: A tuple containing:
                - The dictionary of configurations to be used with `TempConfigOverride`.
                - The path to the test library file for this environment setup.
                - The path to the test system log file for this environment setup.
        """
        global KNOWLEDGE_LIBRARY, _library_dirty_flag, _LIBRARY_FILE_PATH
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        test_lib_path = os.path.join(current_dir, f"test_library_log_{test_run_id}.json")
        test_sys_log_path = os.path.join(current_dir, f"test_library_system_log_{test_run_id}.txt")

        delete_test_files(test_lib_path, test_sys_log_path)

        KNOWLEDGE_LIBRARY = {}
        _library_dirty_flag = False
        _LIBRARY_FILE_PATH = None # Will be reset by _initialize_library_path within TempConfigOverride context

        base_test_configs = {
            "LIBRARY_LOG_PATH": test_lib_path,
            "SYSTEM_LOG_PATH": test_sys_log_path,
            "VERBOSE_OUTPUT": False, 
            "LOG_LEVEL": "debug", 
            "REQUIRE_PUBLIC_STORAGE_CONSENT": True,
            "ensure_path": lambda path_to_ensure: os.makedirs(os.path.dirname(path_to_ensure), exist_ok=True) if os.path.dirname(path_to_ensure) and not os.path.exists(os.path.dirname(path_to_ensure)) else None,
            "ETHICAL_ALIGNMENT_THRESHOLD": 0.6, 
            "MITIGATION_ETHICAL_THRESHOLD": 0.7,
            "LIBRARY_SENSITIVE_KEYWORDS": ["test_sensitive"],
            "LIBRARY_REFRAMING_PHRASES": {"DEFAULT": "Test reframe."}
        }
        if test_specific_configs:
            base_test_configs.update(test_specific_configs)
        
        return base_test_configs, test_lib_path, test_sys_log_path


    # --- Test Scenario Implementations ---
    test_results = {"passed": 0, "failed": 0, "details": []}

    def _run_test(test_func, test_name: str, *args):
        """
        Wrapper function to execute a single test case.
        Manages printing test status, accumulating results, and error handling.
        Restores mocked global functions after each test.

        Args:
            test_func (callable): The test function to execute.
            test_name (str): The name of the test, for reporting.
            *args: Arguments to pass to the test function.
        """
        print(f"\n--- Running {test_name} ---")
        # Store original global functions that might be patched by tests.
        original_get_shared_manifold = globals().get('get_shared_manifold')
        original_score_ethics = globals().get('score_ethics')
        original_input = __builtins__.get('input') # Use .get for safety if input somehow removed.

        try:
            test_func(*args) # Execute the actual test function.
            test_results["passed"] += 1
            test_results["details"].append(f"[PASS] {test_name}")
            print(f"[PASS] {test_name}")
        except AssertionError as e_assert: # Catch assertion errors specifically.
            test_results["failed"] += 1
            error_info = traceback.format_exc()
            test_results["details"].append(f"[FAIL] {test_name}: AssertionError: {e_assert}\n{error_info}")
            print(f"[FAIL] {test_name}: AssertionError: {e_assert}\n{error_info}", file=sys.stderr)
        except Exception as e_general: # Catch any other exceptions.
            test_results["failed"] += 1
            error_info = traceback.format_exc()
            test_results["details"].append(f"[FAIL] {test_name}: Exception: {e_general}\n{error_info}")
            print(f"[FAIL] {test_name}: Exception: {e_general}\n{error_info}", file=sys.stderr)
        finally:
            # Restore any globally patched objects to their original state.
            # This is crucial for test isolation if tests modify shared global resources.
            if 'get_shared_manifold' in globals() and original_get_shared_manifold is not None:
                globals()['get_shared_manifold'] = original_get_shared_manifold
            if 'score_ethics' in globals() and original_score_ethics is not None:
                globals()['score_ethics'] = original_score_ethics
            if original_input is not None and hasattr(__builtins__, 'input'):
                 __builtins__['input'] = original_input
            elif hasattr(__builtins__, 'input') and original_input is None and 'input' in __builtins__:
                 # If 'input' was added by a test and wasn't there originally (less common for builtins).
                 del __builtins__['input']


    # Test Functions
    def test_text_utilities():
        """Tests text utility functions `sanitize_text` and `summarize_text`."""
        assert sanitize_text("  hello world  ") == "hello world"
        assert sanitize_text("hello \t\n world") == "hello world"
        assert sanitize_text("") == ""
        assert sanitize_text(None) == "" # Based on current implementation

        assert summarize_text("short text", 20) == "short text"
        assert summarize_text("this is a very long text that needs truncation", 20) == "this is a very lo..."
        assert summarize_text("", 20) == ""
        assert summarize_text(None, 20) == ""
        assert summarize_text("exactlength", 11) == "exactlength"
        assert summarize_text("exactlengthplus1", 11) == "exactlen..."

    def test_is_valid_coordinate():
        """Tests the `is_valid_coordinate` validation function."""
        assert is_valid_coordinate((1,2,3)) == True
        assert is_valid_coordinate([1,2,3,4.0]) == True
        assert is_valid_coordinate((1,2)) == False # Too short
        assert is_valid_coordinate((1,2,3,4,5)) == False # Too long
        assert is_valid_coordinate((1,2,'a')) == False # Non-numeric
        assert is_valid_coordinate("1,2,3") == False # Wrong type

    def test_mitigator_class():
        """Tests the `Mitigator` class for content moderation logic."""
        test_configs, _, _ = setup_test_environment({"ETHICAL_ALIGNMENT_THRESHOLD": 0.5, "MITIGATION_ETHICAL_THRESHOLD": 0.6, "LIBRARY_SENSITIVE_KEYWORDS": ["danger", "badword"]}, test_run_id="mitigator")
        with TempConfigOverride(test_configs):
            _initialize_library_path() # Mitigator uses _log_library_event which might need config
            mitigator = Mitigator()

            found, keywords = mitigator._contains_sensitive_keywords("this is a dangerous text with badword")
            assert found == True
            assert "danger" in keywords and "badword" in keywords
            
            found, keywords = mitigator._contains_sensitive_keywords("this is safe")
            assert found == False
            assert not keywords

            # Low ethical score
            moderated = mitigator.moderate_ethically_flagged_content("low score content", 0.4)
            assert "[Content Under Review" in moderated or "[Content Moderated" in moderated # Check for either prefix
            
            # High score but sensitive keyword
            moderated = mitigator.moderate_ethically_flagged_content("high score but danger word", 0.9)
            assert "[Content Under Review" in moderated or "[Content Moderated" in moderated
            
            # Strict mode, score below alignment but above mitigation_threshold
            moderated = mitigator.moderate_ethically_flagged_content("strict mode test content", 0.55, strict_mode=True)
            assert "[Content Under Review" in moderated or "[Content Moderated" in moderated
            
             # Very low score -> generic placeholder (stronger mitigation)
            moderated_low_score = mitigator.moderate_ethically_flagged_content("very bad content", 0.1)
            assert "[Content Moderated due to significant ethical concerns" in moderated_low_score

            # High score, no keywords (should return original text)
            original_text_val = "perfectly fine content" # Renamed to avoid conflict with 'original_keywords' later
            moderated = mitigator.moderate_ethically_flagged_content(original_text_val, 0.95)
            assert moderated == original_text_val

            # --- Specific tests for _contains_sensitive_keywords regex logic ---
            original_keywords = mitigator.sensitive_keywords

            # Test 1: Word boundaries - "sen" vs "sensitive"
            mitigator.sensitive_keywords = ["sen"]
            found, keywords = mitigator._contains_sensitive_keywords("This is a sensitive test.")
            assert not found, "Mitigator test fail: 'sen' should not match in 'sensitive' with word boundaries"

            mitigator.sensitive_keywords = ["sensitive"]
            found, keywords = mitigator._contains_sensitive_keywords("This is a sensitive test.")
            assert found and "sensitive" in keywords, "Mitigator test fail: 'sensitive' exact match"

            # Test 2: Provided "test_sensitive" from initial config of this test function
            # Note: initial config was {"LIBRARY_SENSITIVE_KEYWORDS": ["danger", "badword"]}, so "test_sensitive" is not in defaults here.
            # We will use one of the initial keywords "danger" for this style of test.
            mitigator.sensitive_keywords = ["danger"]
            found, keywords = mitigator._contains_sensitive_keywords("A danger phrase.")
            assert found and "danger" in keywords, "Mitigator test fail: 'danger' exact match"
            found, keywords = mitigator._contains_sensitive_keywords("A non_danger phrase.")
            assert not found, "Mitigator test fail: 'danger' should not match 'non_danger'"
            found, keywords = mitigator._contains_sensitive_keywords("A danger_extra phrase.")
            assert not found, "Mitigator test fail: 'danger' should not match 'danger_extra'"
            found, keywords = mitigator._contains_sensitive_keywords("endanger") # Test substring at end
            assert not found, "Mitigator test fail: 'danger' should not match 'endanger'"
            found, keywords = mitigator._contains_sensitive_keywords("dangerous") # Test substring at start
            assert not found, "Mitigator test fail: 'danger' should not match 'dangerous'"


            # Test 3: Multi-word keyword precision
            mitigator.sensitive_keywords = ["real bad"]
            found, keywords = mitigator._contains_sensitive_keywords("This is a real bad situation.")
            assert found and "real bad" in keywords, "Mitigator test fail: 'real bad' exact match"
            found, keywords = mitigator._contains_sensitive_keywords("This is really bad.")
            assert not found, "Mitigator test fail: 'real bad' should not match 'really bad'"
            found, keywords = mitigator._contains_sensitive_keywords("This is a real good situation, not bad.")
            assert not found, "Mitigator test fail: 'real bad' should not match 'bad' alone"
            found, keywords = mitigator._contains_sensitive_keywords("This is a real bad situation not a real bad problem.")
            assert found and "real bad" in keywords and len(keywords) == 1, "Mitigator test fail: 'real bad' multi-match but should be one keyword"


            mitigator.sensitive_keywords = original_keywords # Restore original keywords for any subsequent tests in this function

    def test_knowledge_library_persistence():
        """Tests the persistence (_load, _save) of the knowledge library."""
        test_configs, test_lib_path, _ = setup_test_environment(test_run_id="persistence")
        with TempConfigOverride(test_configs):
            _initialize_library_path() 
            _load_knowledge_library() 
            assert len(KNOWLEDGE_LIBRARY) == 0, "Library not empty at start of persistence test"

            KNOWLEDGE_LIBRARY["test001"] = {"data": "my test data", "timestamp": "now"}
            global _library_dirty_flag 
            _library_dirty_flag = True
            _save_knowledge_library()
            
            assert os.path.exists(test_lib_path), "Library file not created by save"
            assert _library_dirty_flag == False, "Dirty flag not reset after save"

            KNOWLEDGE_LIBRARY.clear() 
            _load_knowledge_library()
            assert len(KNOWLEDGE_LIBRARY) == 1, "Library not loaded correctly"
            assert KNOWLEDGE_LIBRARY["test001"]["data"] == "my test data", "Loaded data mismatch"
        delete_test_files(test_lib_path)


    @mock.patch('core.library.score_ethics')
    @mock.patch('core.library.get_shared_manifold')
    @mock.patch('builtins.input') # Mock the built-in input function
    def test_store_knowledge(mock_input, mock_get_manifold, mock_score_ethics_func):
        """
        Tests the `store_knowledge` function including interactions with mocks
        for manifold, ethics scoring, and user input (for consent).
        Covers scenarios: private storage, public storage with consent granted/denied,
        and handling of failures in brain/ethics module interactions.
        """
        mock_manifold_instance = mock.MagicMock()
        mock_manifold_instance.bootstrap_concept_from_llm.return_value = ((0.1, 0.2, 0.3, 0.4), 0.5, "Mock summary from LLM")
        mock_get_manifold.return_value = mock_manifold_instance
        
        test_run_id = "store"
        # REQUIRE_PUBLIC_STORAGE_CONSENT is True by default in setup_test_environment
        test_configs, test_lib_path, test_sys_log_path = setup_test_environment(test_run_id=test_run_id)

        with TempConfigOverride(test_configs):
            _initialize_library_path()
            _load_knowledge_library()

            # Scenario 1: Store private item (no consent needed)
            mock_score_ethics_func.return_value = 0.9
            entry_id1 = store_knowledge("Test private content", is_public=False, author="Test Author")
            assert entry_id1 is not None, "S1: Entry ID is None"
            assert entry_id1 in KNOWLEDGE_LIBRARY, "S1: Entry not in library"
            assert KNOWLEDGE_LIBRARY[entry_id1]["is_public"] == False, "S1: is_public mismatch"
            assert KNOWLEDGE_LIBRARY[entry_id1]["author"] == "Test Author", "S1: author mismatch"
            assert KNOWLEDGE_LIBRARY[entry_id1]["coordinates"] == (0.1, 0.2, 0.3, 0.4), "S1: coordinates mismatch"
            assert KNOWLEDGE_LIBRARY[entry_id1]["raw_t_intensity"] == 0.5, "S1: raw_t_intensity mismatch"
            assert KNOWLEDGE_LIBRARY[entry_id1]["ethics_score"] == 0.9, "S1: ethics_score mismatch"
            mock_manifold_instance.bootstrap_concept_from_llm.assert_called_once()
            mock_score_ethics_func.assert_called_once()
            mock_input.assert_not_called() 

            # Reset mocks for next scenario
            mock_input.reset_mock(); mock_manifold_instance.bootstrap_concept_from_llm.reset_mock(); mock_score_ethics_func.reset_mock()
            mock_manifold_instance.bootstrap_concept_from_llm.return_value = ((0.1, 0.2, 0.3, 0.4), 0.5, "Mock summary from LLM") # Re-set return value if needed

            # Scenario 2: Store public item with consent granted
            mock_input.return_value = "yes"
            mock_score_ethics_func.return_value = 0.8
            entry_id2 = store_knowledge("Test public content with consent", is_public=True)
            assert entry_id2 is not None, "S2: Entry ID is None"
            assert entry_id2 in KNOWLEDGE_LIBRARY, "S2: Entry not in library"
            assert KNOWLEDGE_LIBRARY[entry_id2]["is_public"] == True, "S2: is_public mismatch"
            mock_input.assert_called_once()

            # Scenario 3: Store public item, consent denied
            mock_input.reset_mock(); mock_manifold_instance.bootstrap_concept_from_llm.reset_mock(); mock_score_ethics_func.reset_mock()
            mock_manifold_instance.bootstrap_concept_from_llm.return_value = ((0.1, 0.2, 0.3, 0.4), 0.5, "Mock summary from LLM")
            mock_input.return_value = "no"
            entry_id3 = store_knowledge("Test public consent denied", is_public=True)
            assert entry_id3 is None, "S3: Entry ID should be None (consent denied)"
            assert entry_id3 not in KNOWLEDGE_LIBRARY, "S3: Entry should not be in library"
            mock_input.assert_called_once()
            
            # Scenario 4: Brain module failure (bootstrap_concept_from_llm fails)
            mock_input.reset_mock(); mock_manifold_instance.bootstrap_concept_from_llm.reset_mock(); mock_score_ethics_func.reset_mock()
            mock_manifold_instance.bootstrap_concept_from_llm.side_effect = Exception("Simulated brain failure")
            mock_score_ethics_func.return_value = 0.7 # Ethics module still works
            entry_id4 = store_knowledge("Content with brain failure", is_public=False) # Private to skip consent
            assert entry_id4 is not None, "S4: Entry ID is None (brain failure)"
            assert KNOWLEDGE_LIBRARY[entry_id4]["coordinates"] is None, "S4: Coordinates should be None"
            assert KNOWLEDGE_LIBRARY[entry_id4]["raw_t_intensity"] == 0.0, "S4: Intensity should be default"
            assert KNOWLEDGE_LIBRARY[entry_id4]["ethics_score"] == 0.7, "S4: Ethics score mismatch"
            
            # Scenario 5: Ethics module failure (score_ethics fails)
            mock_input.reset_mock(); mock_manifold_instance.bootstrap_concept_from_llm.reset_mock(); mock_score_ethics_func.reset_mock()
            mock_manifold_instance.bootstrap_concept_from_llm.side_effect = None # Reset side effect
            mock_manifold_instance.bootstrap_concept_from_llm.return_value = ((0.1,0.2,0.3,0.4), 0.5, "Mock summary")
            mock_score_ethics_func.side_effect = Exception("Simulated ethics failure")
            entry_id5 = store_knowledge("Content with ethics failure", is_public=False) # Private
            assert entry_id5 is not None, "S5: Entry ID is None (ethics failure)"
            assert KNOWLEDGE_LIBRARY[entry_id5]["ethics_score"] == 0.5, "S5: Ethics score should be default"
            # Clear KNOWLEDGE_LIBRARY and reload to ensure clean state from file for next scenarios
            KNOWLEDGE_LIBRARY.clear(); _load_knowledge_library()


            # Scenario: Invalid source_uri (bad format)
            with mock.patch('core.library._log_library_event') as mock_log_event_uri:
                # Ensure mocks from function decorator are reset if their specific state matters here
                mock_input.reset_mock(); mock_manifold_instance.bootstrap_concept_from_llm.reset_mock(); mock_score_ethics_func.reset_mock()
                mock_manifold_instance.bootstrap_concept_from_llm.return_value = ((0.1, 0.2, 0.3, 0.4), 0.5, "Mock summary from LLM") # Re-set return value
                mock_input.return_value = "yes"
                mock_score_ethics_func.return_value = 0.9

                entry_id_bad_uri = store_knowledge("Content with bad URI", is_public=False, source_uri="ftp:/invalid-uri")
                assert entry_id_bad_uri is not None, "S_bad_uri: Entry ID should not be None"
                assert KNOWLEDGE_LIBRARY[entry_id_bad_uri]["source_uri"] is None, "S_bad_uri: source_uri should be None in stored entry"

                called_with_warning = False
                for call_arg in mock_log_event_uri.call_args_list:
                    args, kwargs = call_arg
                    if args[0] == "store_knowledge_invalid_uri" and kwargs.get('level') == "warning":
                        called_with_warning = True
                        assert args[1]['data']["reason"] == "source_uri does not match expected pattern"
                        break
                assert called_with_warning, "S_bad_uri: Expected 'store_knowledge_invalid_uri' warning log not found"
            KNOWLEDGE_LIBRARY.clear(); _load_knowledge_library() # Reset state

            # Scenario: Invalid author (empty string)
            with mock.patch('core.library._log_library_event') as mock_log_event_author:
                mock_input.reset_mock(); mock_manifold_instance.bootstrap_concept_from_llm.reset_mock(); mock_score_ethics_func.reset_mock()
                mock_manifold_instance.bootstrap_concept_from_llm.return_value = ((0.1, 0.2, 0.3, 0.4), 0.5, "Mock summary from LLM")
                mock_input.return_value = "yes"
                mock_score_ethics_func.return_value = 0.9

                entry_id_empty_author = store_knowledge("Content with empty author", is_public=False, author="   ")
                assert entry_id_empty_author is not None, "S_empty_author: Entry ID should not be None"
                assert KNOWLEDGE_LIBRARY[entry_id_empty_author]["author"] is None, "S_empty_author: author should be None in stored entry"

                called_with_warning = False
                for call_arg in mock_log_event_author.call_args_list:
                    args, kwargs = call_arg
                    if args[0] == "store_knowledge_invalid_author" and kwargs.get('level') == "warning":
                        called_with_warning = True
                        assert args[1]['data']["reason"] == "author is an empty or whitespace-only string"
                        break
                assert called_with_warning, "S_empty_author: Expected 'store_knowledge_invalid_author' warning log not found"
            KNOWLEDGE_LIBRARY.clear(); _load_knowledge_library() # Reset state

            # Scenario: Valid source_uri and author (also tests stripping for author)
            with mock.patch('core.library._log_library_event') as mock_log_event_valid:
                mock_input.reset_mock(); mock_manifold_instance.bootstrap_concept_from_llm.reset_mock(); mock_score_ethics_func.reset_mock()
                mock_manifold_instance.bootstrap_concept_from_llm.return_value = ((0.1, 0.2, 0.3, 0.4), 0.5, "Mock summary from LLM")
                mock_input.return_value = "yes"
                mock_score_ethics_func.return_value = 0.9
                valid_uri = "http://example.com/path"
                valid_author_input = "  Test Author  "
                expected_author_stored = "Test Author"

                entry_id_valid_inputs = store_knowledge("Content with valid URI and author", is_public=False, source_uri=valid_uri, author=valid_author_input)
                assert entry_id_valid_inputs is not None, "S_valid_inputs: Entry ID should not be None"
                assert KNOWLEDGE_LIBRARY[entry_id_valid_inputs]["source_uri"] == valid_uri, "S_valid_inputs: source_uri mismatch"
                assert KNOWLEDGE_LIBRARY[entry_id_valid_inputs]["author"] == expected_author_stored, "S_valid_inputs: author mismatch or not stripped"

                no_warning_logs = True
                for call_arg in mock_log_event_valid.call_args_list:
                    args, kwargs = call_arg
                    if kwargs.get('level') == "warning" and (args[0] == "store_knowledge_invalid_uri" or args[0] == "store_knowledge_invalid_author"):
                        no_warning_logs = False
                        print(f"Unexpected warning log: {args}, {kwargs}") # Debug print
                        break
                assert no_warning_logs, "S_valid_inputs: Warning logs were unexpectedly found for valid inputs"
            KNOWLEDGE_LIBRARY.clear(); _load_knowledge_library() # Reset state

        delete_test_files(test_lib_path, test_sys_log_path) # Clean up after this complex test

    def test_retrieval_functions():
        """
        Tests knowledge retrieval functions: `retrieve_knowledge_by_id` and
        `retrieve_knowledge_by_keyword`.
        """
        test_run_id = "retrieval"
        test_configs, test_lib_path, test_sys_log_path = setup_test_environment(test_run_id=test_run_id)
        with TempConfigOverride(test_configs):
            _initialize_library_path()
            _load_knowledge_library()

            entry1 = {"id": "id1", "full_content": "Public entry about apples", "content_preview": "apples", "is_public": True}
            entry2 = {"id": "id2", "full_content": "Private entry about bananas", "content_preview": "bananas", "is_public": False}
            entry3 = {"id": "id3", "full_content": "Another public entry about apples and oranges", "content_preview": "apples oranges", "is_public": True}
            global KNOWLEDGE_LIBRARY, _library_dirty_flag
            KNOWLEDGE_LIBRARY = {"id1": entry1, "id2": entry2, "id3": entry3} # Directly populate for test
            _library_dirty_flag = False 

            assert retrieve_knowledge_by_id("id1") == entry1, "Retrieval by ID failed for id1"
            assert retrieve_knowledge_by_id("nonexistent") is None, "Retrieval by non-existent ID did not return None"

            results_public_apples = retrieve_knowledge_by_keyword("apples", search_public=True, search_private=False)
            assert len(results_public_apples) == 2, "Keyword search for 'apples' (public only) wrong count"
            assert entry1 in results_public_apples and entry3 in results_public_apples, "Keyword search for 'apples' (public only) content mismatch"
            
            results_private_bananas = retrieve_knowledge_by_keyword("bananas", search_public=False, search_private=True)
            assert len(results_private_bananas) == 1, "Keyword search for 'bananas' (private only) wrong count"
            assert entry2 in results_private_bananas, "Keyword search for 'bananas' (private only) content mismatch"
            
            results_public_bananas = retrieve_knowledge_by_keyword("bananas", search_public=True, search_private=False)
            assert len(results_public_bananas) == 0, "Keyword search for 'bananas' (public only, expected 0) wrong count"
            
            results_kiwi = retrieve_knowledge_by_keyword("kiwi")
            assert len(results_kiwi) == 0, "Keyword search for 'kiwi' (expected 0) wrong count"
            
            results_case_insensitive_apples = retrieve_knowledge_by_keyword("APPLES", search_public=True, search_private=True)
            assert len(results_case_insensitive_apples) == 2, "Case-insensitive keyword search for 'APPLES' wrong count"

            # --- Tests for retrieve_knowledge_by_id input validation ---
            with mock.patch('core.library._log_library_event') as mock_log_event_retrieve:
                # Test case 1: Invalid length (too short)
                assert retrieve_knowledge_by_id("abc") is None, "Retrieve_by_id: Too short ID did not return None"
                ret_args, ret_kwargs = mock_log_event_retrieve.call_args
                assert ret_kwargs.get('level') == "warning", "Retrieve_by_id: Log level not warning for short ID"
                assert ret_args[0] == "retrieve_by_id_failed", "Retrieve_by_id: Event type mismatch for short ID"
                assert "Invalid entry_id format" in ret_args[1]['data'].get("reason", ""), "Retrieve_by_id: Reason mismatch for short ID"
                mock_log_event_retrieve.reset_mock()

                # Test case 2: Invalid length (too long)
                assert retrieve_knowledge_by_id("a" * 65) is None, "Retrieve_by_id: Too long ID did not return None"
                ret_args, ret_kwargs = mock_log_event_retrieve.call_args
                assert ret_kwargs.get('level') == "warning", "Retrieve_by_id: Log level not warning for long ID"
                assert ret_args[0] == "retrieve_by_id_failed", "Retrieve_by_id: Event type mismatch for long ID"
                assert "Invalid entry_id format" in ret_args[1]['data'].get("reason", ""), "Retrieve_by_id: Reason mismatch for long ID"
                mock_log_event_retrieve.reset_mock()

                # Test case 3: Invalid characters
                assert retrieve_knowledge_by_id("g" * 64) is None, "Retrieve_by_id: Invalid char ID did not return None" # 'g' is not hex
                ret_args, ret_kwargs = mock_log_event_retrieve.call_args
                assert ret_kwargs.get('level') == "warning", "Retrieve_by_id: Log level not warning for invalid char ID"
                assert ret_args[0] == "retrieve_by_id_failed", "Retrieve_by_id: Event type mismatch for invalid char ID"
                assert "Invalid entry_id format" in ret_args[1]['data'].get("reason", ""), "Retrieve_by_id: Reason mismatch for invalid char ID"
                mock_log_event_retrieve.reset_mock()

                # Test case 4: Invalid type (integer) - existing type check handles this, but verify log
                assert retrieve_knowledge_by_id(12345) is None, "Retrieve_by_id: Integer ID did not return None"
                ret_args, ret_kwargs = mock_log_event_retrieve.call_args
                assert ret_kwargs.get('level') == "warning", "Retrieve_by_id: Log level not warning for int ID"
                assert ret_args[0] == "retrieve_by_id_failed", "Retrieve_by_id: Event type mismatch for int ID"
                assert "Invalid entry_id type" in ret_args[1]['data'].get("reason", ""), "Retrieve_by_id: Reason mismatch for int ID"

        delete_test_files(test_lib_path, test_sys_log_path)


    # --- Test Runner Logic ---
    print("Starting Core Library Self-Tests...")
    
    _run_test(test_text_utilities, "test_text_utilities")
    _run_test(test_is_valid_coordinate, "test_is_valid_coordinate")
    _run_test(test_mitigator_class, "test_mitigator_class")
    _run_test(test_knowledge_library_persistence, "test_knowledge_library_persistence")
    # Note: test_store_knowledge uses @mock decorators for its dependencies.
    _run_test(test_store_knowledge, "test_store_knowledge") 
    _run_test(test_retrieval_functions, "test_retrieval_functions")


    print("\n--- Core Library Test Summary ---")
    for detail_idx, detail_msg in enumerate(test_results["details"]):
        # Print only the main pass/fail line for summary, but full for fails
        if detail_msg.startswith("[FAIL]"):
            print(f"\nDetail for FAIL {detail_idx+1}:\n{detail_msg}")
        else:
            print(detail_msg.splitlines()[0]) 
    
    print(f"\nTotal Passed: {test_results['passed']}")
    print(f"Total Failed: {test_results['failed']}")

    _IS_TEST_RUNNING = False # Reset flag

    if test_results["failed"] > 0:
        print("\nCORE LIBRARY TESTS FAILED.")
        sys.exit(1)
    else:
        print("\nALL CORE LIBRARY TESTS PASSED.")
        sys.exit(0)
```

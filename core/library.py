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
    
    if not isinstance(text, str):
        # Optionally, convert to string or raise an error
        # For now, assume we want to try converting it
        try:
            text = str(text)
        except:
            return "" # Or raise TypeError

    if len(text) <= max_length:
        return text
    else:
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
        self.alignment_threshold = 0.7
        self.mitigation_threshold = 0.85 # Stricter than alignment_threshold for triggering moderation
        self.sensitive_keywords_config_key = "LIBRARY_SENSITIVE_KEYWORDS" # Example key if made configurable
        self.reframing_phrases_config_key = "LIBRARY_REFRAMING_PHRASES" # Example key

        if config:
            self.alignment_threshold = getattr(config, 'ETHICAL_ALIGNMENT_THRESHOLD', self.alignment_threshold)
            self.mitigation_threshold = getattr(config, 'MITIGATION_ETHICAL_THRESHOLD', self.mitigation_threshold)
            if not hasattr(config, 'ETHICAL_ALIGNMENT_THRESHOLD') or not hasattr(config, 'MITIGATION_ETHICAL_THRESHOLD'):
                _log_library_event("mitigator_config_warning", 
                                   {"message": "Using default ethical/mitigation thresholds for Mitigator due to missing config values."}, 
                                   level="warning")
        else:
            _log_library_event("mitigator_config_warning", 
                               {"message": "Config module not loaded. Mitigator using default ethical/mitigation thresholds."}, 
                               level="warning")

        # Default sensitive keywords (should ideally be from config)
        self.sensitive_keywords = [
            "harm", "destroy", "exploit", "illegal", "deceive", "manipulate", 
            "hate speech", "violence", "suffering", "unethical" 
            # Consider making these regex patterns for more robust matching
        ]
        if config and hasattr(config, self.sensitive_keywords_config_key):
             loaded_keywords = getattr(config, self.sensitive_keywords_config_key)
             if isinstance(loaded_keywords, list) and all(isinstance(k, str) for k in loaded_keywords):
                 self.sensitive_keywords = loaded_keywords
             else:
                _log_library_event("mitigator_config_error", 
                                   {"message": f"Invalid format for {self.sensitive_keywords_config_key} in config. Using defaults."}, 
                                   level="error")


        # Default reframing phrases (should ideally be from config)
        self.reframing_phrases = {
            "DEFAULT": "Let's consider this topic from a different, more constructive perspective.",
            "harm": "Instead of focusing on potential negative impacts, how can we explore solutions that promote well-being and safety?",
            "destroy": "Rather than deconstruction, let's think about how to build or improve.",
            "exploit": "It's important to ensure fairness and respect. How can this situation be approached with integrity?",
            "illegal": "Activities should align with legal and ethical standards. What are some compliant approaches?",
            "unethical": "Maintaining ethical integrity is crucial. Let's re-evaluate this through an ethical lens."
            # Add more specific reframing phrases as needed
        }

        if config and hasattr(config, self.reframing_phrases_config_key):
            loaded_phrases = getattr(config, self.reframing_phrases_config_key)
            if isinstance(loaded_phrases, dict):
                self.reframing_phrases.update(loaded_phrases) # Merge with defaults, allowing overrides
            else:
                _log_library_event("mitigator_config_error", 
                                   {"message": f"Invalid format for {self.reframing_phrases_config_key} in config. Using defaults."}, 
                                   level="error")


    def _contains_sensitive_keywords(self, text: str) -> tuple[bool, list[str]]:
        """
        Checks if the given text contains any sensitive keywords.

        Args:
            text: The input string to check.

        Returns:
            A tuple: (bool indicating if keywords were found, list of found keywords).
        """
        if not text or not isinstance(text, str):
            return False, []
        
        text_lower = text.lower()
        found_keywords = []
        for keyword in self.sensitive_keywords:
            # Using \b for word boundaries to avoid matching substrings within words (e.g., 'harm' in 'harmony')
            # This requires keywords to be actual words. For phrases, direct check is better.
            # For simplicity here, we'll do a direct substring check first, then consider regex for word boundaries.
            if keyword.lower() in text_lower: # Simple substring check
            # Example with word boundaries (more precise but might miss multi-word keywords if not handled carefully)
            # pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
            # if re.search(pattern, text_lower):
                found_keywords.append(keyword)
        
        return bool(found_keywords), found_keywords

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
        if not isinstance(original_text, str): # Basic type check
            _log_library_event("mitigation_error", {"reason": "Invalid original_text type", "type": type(original_text).__name__}, level="error")
            return "[Invalid content provided for moderation.]"

        keywords_found, found_keywords_list = self._contains_sensitive_keywords(original_text)
        
        trigger_mitigation = False
        mitigation_reason = []

        if ethical_score < self.mitigation_threshold:
            trigger_mitigation = True
            mitigation_reason.append(f"Ethical score ({ethical_score:.2f}) below mitigation threshold ({self.mitigation_threshold:.2f})")
        
        if strict_mode and ethical_score < self.alignment_threshold:
            if not trigger_mitigation: # Avoid duplicate reason if already triggered by stricter mitigation_threshold
                 trigger_mitigation = True
            mitigation_reason.append(f"Strict mode: Ethical score ({ethical_score:.2f}) below alignment threshold ({self.alignment_threshold:.2f})")

        # Sensitive keyword check can also trigger, potentially with a less strict score.
        # For this implementation, any sensitive keyword triggers if not already triggered.
        # A more nuanced approach might be: `keywords_found and ethical_score < some_other_threshold`
        if keywords_found:
            if not trigger_mitigation: # Trigger if not already triggered by score
                 trigger_mitigation = True
            mitigation_reason.append(f"Sensitive keywords found: {', '.join(found_keywords_list)}")


        if trigger_mitigation:
            summary = summarize_text(original_text, 75) # Slightly longer summary for context
            log_data = {
                "original_snippet": summary,
                "ethical_score": ethical_score,
                "strict_mode": strict_mode,
                "keywords_found": found_keywords_list,
                "reason_for_mitigation": "; ".join(mitigation_reason)
            }
            _log_library_event("mitigation_triggered", log_data, level="warning")

            # Select reframing phrase
            reframing_phrase = self.reframing_phrases.get("DEFAULT")
            if found_keywords_list:
                # Try to find a more specific reframing phrase for the first found keyword
                for kw in found_keywords_list:
                    if kw.lower() in self.reframing_phrases:
                        reframing_phrase = self.reframing_phrases[kw.lower()]
                        break
            
            # Determine severity of response
            # Example: Very low score or specific keywords might lead to a generic placeholder
            if ethical_score < 0.3 or (strict_mode and ethical_score < 0.5 and keywords_found): # Example of stricter condition
                return f"[Content Moderated due to Ethical Concerns (Score: {ethical_score:.2f}). Please rephrase or contact support.]"
            
            return f"[Content Under Review due to Ethical Concerns (Score: {ethical_score:.2f})]. {reframing_phrase} Original topic was related to: '{summary}'"

        return original_text

# --- Module-Level Variables ---
KNOWLEDGE_LIBRARY = {}
_library_dirty_flag = False
_LIBRARY_FILE_PATH = None # To be set by _initialize_library_path

# --- Logging Function ---
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
    global _LIBRARY_FILE_PATH
    if config and hasattr(config, 'LIBRARY_LOG_PATH') and config.LIBRARY_LOG_PATH:
        _LIBRARY_FILE_PATH = config.LIBRARY_LOG_PATH
        ensure_path_func = getattr(config, 'ensure_path', None)
        if ensure_path_func:
            try:
                ensure_path_func(_LIBRARY_FILE_PATH) 
                _log_library_event("library_path_initialized", {"path": _LIBRARY_FILE_PATH})
            except Exception as e_ensure:
                _log_library_event("library_path_ensure_failed", {"path": _LIBRARY_FILE_PATH, "error": str(e_ensure)}, level="error")
                try: # Fallback manual ensure
                    lib_dir = os.path.dirname(_LIBRARY_FILE_PATH)
                    if lib_dir and not os.path.exists(lib_dir):
                        os.makedirs(lib_dir, exist_ok=True)
                except Exception as e_manual_mkdir:
                    _log_library_event("library_path_manual_mkdir_failed", {"path": _LIBRARY_FILE_PATH, "error": str(e_manual_mkdir)}, level="critical")
                    _LIBRARY_FILE_PATH = None 
        else: # Manual ensure if config.ensure_path not available
            try:
                lib_dir = os.path.dirname(_LIBRARY_FILE_PATH)
                if lib_dir and not os.path.exists(lib_dir):
                    os.makedirs(lib_dir, exist_ok=True)
                _log_library_event("library_path_initialized_manual_ensure", {"path": _LIBRARY_FILE_PATH})
            except Exception as e_manual_mkdir:
                _log_library_event("library_path_manual_mkdir_failed", {"path": _LIBRARY_FILE_PATH, "error": str(e_manual_mkdir)}, level="critical")
                _LIBRARY_FILE_PATH = None 
    else:
        _log_library_event("library_path_init_failed", {"reason": "Config or LIBRARY_LOG_PATH missing or empty"}, level="error")
        _LIBRARY_FILE_PATH = None


# --- Persistence Functions ---
def _load_knowledge_library():
    global KNOWLEDGE_LIBRARY, _library_dirty_flag
    if not _LIBRARY_FILE_PATH:
        _log_library_event("load_library_failed", {"reason": "Library file path not set"}, level="error")
        KNOWLEDGE_LIBRARY = {}
        _library_dirty_flag = False 
        return

    try:
        if not os.path.exists(_LIBRARY_FILE_PATH) or os.path.getsize(_LIBRARY_FILE_PATH) == 0:
            _log_library_event("load_library_info", {"message": "Library file not found or empty. Initializing new.", "path": _LIBRARY_FILE_PATH})
            KNOWLEDGE_LIBRARY = {}
            _library_dirty_flag = False
            return

        with open(_LIBRARY_FILE_PATH, 'r', encoding='utf-8') as f: # Added encoding
            data = json.load(f)
        
        if isinstance(data, dict):
            KNOWLEDGE_LIBRARY = data
            _log_library_event("load_library_success", {"path": _LIBRARY_FILE_PATH, "entries_loaded": len(KNOWLEDGE_LIBRARY)})
        else:
            _log_library_event("load_library_malformed", {"path": _LIBRARY_FILE_PATH, "error": "Data is not a dictionary"}, level="error")
            KNOWLEDGE_LIBRARY = {} 
        _library_dirty_flag = False

    except json.JSONDecodeError as e:
        _log_library_event("load_library_json_error", {"path": _LIBRARY_FILE_PATH, "error": str(e)}, level="error")
        KNOWLEDGE_LIBRARY = {} 
        _library_dirty_flag = False
    except IOError as e: # More specific for file issues
        _log_library_event("load_library_io_error", {"path": _LIBRARY_FILE_PATH, "error": str(e)}, level="error")
        KNOWLEDGE_LIBRARY = {}
        _library_dirty_flag = False
    except Exception as e:
        _log_library_event("load_library_unknown_error", {"path": _LIBRARY_FILE_PATH, "error": str(e), "trace": traceback.format_exc()}, level="critical")
        KNOWLEDGE_LIBRARY = {} 
        _library_dirty_flag = False


def _save_knowledge_library():
    global _library_dirty_flag
    if not _library_dirty_flag:
        _log_library_event("save_library_skipped", {"reason": "No changes (_library_dirty_flag is False)"}, level="debug")
        return

    if not _LIBRARY_FILE_PATH:
        _log_library_event("save_library_failed", {"reason": "Library file path not set"}, level="error")
        return

    temp_path = _LIBRARY_FILE_PATH + ".tmp"
    try:
        # Ensure directory exists just before writing (in case it was deleted)
        lib_dir = os.path.dirname(_LIBRARY_FILE_PATH)
        if lib_dir and not os.path.exists(lib_dir):
            os.makedirs(lib_dir, exist_ok=True)
                
        with open(temp_path, 'w', encoding='utf-8') as f: # Added encoding
            json.dump(KNOWLEDGE_LIBRARY, f, indent=2)
        
        os.replace(temp_path, _LIBRARY_FILE_PATH) 
        _library_dirty_flag = False
        _log_library_event("save_library_success", {"path": _LIBRARY_FILE_PATH, "entries_saved": len(KNOWLEDGE_LIBRARY)})
    except IOError as e: # More specific for file issues
         _log_library_event("save_library_io_error", {"path": _LIBRARY_FILE_PATH, "temp_path": temp_path, "error": str(e)}, level="critical")
    except Exception as e:
        _log_library_event("save_library_unknown_error", {"path": _LIBRARY_FILE_PATH, "temp_path": temp_path, "error": str(e), "trace": traceback.format_exc()}, level="critical")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e_remove:
                _log_library_event("save_library_temp_cleanup_failed", {"path": temp_path, "error": str(e_remove)}, level="error")
                


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
    global _library_dirty_flag

    # --- Input Validation ---
    if not content or not isinstance(content, str) or not content.strip():
        _log_library_event("store_knowledge_failed", {"reason": "Content is empty or not a string."}, level="error")
        return None

    # --- Generate Basic Metadata ---
    entry_id = hashlib.sha256(content.encode('utf-8')).hexdigest()
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    content_hash = entry_id 
    content_preview = summarize_text(content, max_length=150)

    # --- Manifold Coordinate Generation ---
    # Derive concept_name_for_coord from content
    concept_name_for_coord = summarize_text(content_preview, max_length=20).strip().replace('...', '')
    if not concept_name_for_coord: # Ensure it's not empty after stripping
        concept_name_for_coord = "generic concept"
    
    manifold = get_shared_manifold()
    coordinates = None
    intensity_val = 0.0
    summary_for_ethics = content_preview # Default summary for ethics

    if manifold and hasattr(manifold, 'bootstrap_concept_from_llm'):
        try:
            # bootstrap_concept_from_llm should return (coordinates_tuple, intensity_float, summary_string)
            # coordinates_tuple is (x,y,z,t_coord) - the scaled T-coordinate
            # intensity_float is raw_t_intensity (0-1)
            coord_data = manifold.bootstrap_concept_from_llm(concept_name_for_coord)
            if coord_data and isinstance(coord_data, tuple) and len(coord_data) == 3:
                coordinates = coord_data[0] # Should be the (x,y,z,t_coord) tuple
                intensity_val = coord_data[1] # Should be the raw_t_intensity (0-1)
                summary_for_ethics = coord_data[2] # Summary from LLM
                if not is_valid_coordinate(coordinates): # Validate coordinates from brain
                    _log_library_event("store_knowledge_warning", {"entry_id": entry_id, "reason": "Invalid coordinates from brain, using None.", "coord_received": coordinates}, level="warning")
                    coordinates = None # Fallback
                if not isinstance(intensity_val, (int, float)):
                    intensity_val = 0.0 # Fallback
                if not isinstance(summary_for_ethics, str) or not summary_for_ethics.strip():
                    summary_for_ethics = content_preview # Fallback
            else:
                _log_library_event("store_knowledge_brain_data_malformed", {"entry_id": entry_id, "concept_name": concept_name_for_coord, "data_received": str(coord_data)}, level="warning")
                # Fallbacks already set: coordinates = None, intensity_val = 0.0, summary_for_ethics = content_preview
        except Exception as e_brain:
            _log_library_event("store_knowledge_brain_error", {"entry_id": entry_id, "concept_name": concept_name_for_coord, "error": str(e_brain)}, level="error")
            # Fallbacks already set
    else:
        _log_library_event("store_knowledge_manifold_unavailable", {"entry_id": entry_id, "reason": "Manifold or bootstrap_concept_from_llm not available."}, level="warning")
        # Fallbacks already set

    # --- Ethical Scoring ---
    awareness_for_ethics = {
        "primary_concept_coord": coordinates, # (x,y,z,t_coord) or None
        "raw_t_intensity": intensity_val,     # Raw 0-1 intensity or 0.0
        "coherence": getattr(config, 'DEFAULT_KNOWLEDGE_COHERENCE', 0.75) if config else 0.75,
        # Add other fields if score_ethics expects them
    }
    ethics_score = 0.5 # Default neutral score

    try:
        # score_ethics is expected to return a float between 0.0 and 1.0
        calculated_ethics_score = score_ethics(awareness_for_ethics, concept_summary=summary_for_ethics, action_description=content)
        if isinstance(calculated_ethics_score, (int, float)) and 0.0 <= calculated_ethics_score <= 1.0:
            ethics_score = float(calculated_ethics_score)
        else:
            _log_library_event("store_knowledge_ethics_score_invalid", {"entry_id": entry_id, "score_received": str(calculated_ethics_score)}, level="warning")
            # ethics_score remains default 0.5
    except Exception as e_ethics:
        _log_library_event("store_knowledge_ethics_error", {"entry_id": entry_id, "error": str(e_ethics)}, level="error")
        # ethics_score remains default 0.5

    # --- Construct Knowledge Entry ---
    entry = {
        "id": entry_id,
        "timestamp": timestamp,
        "content_hash": content_hash,
        "content_preview": content_preview,
        "full_content": content, # Original full content
        "is_public": is_public,
        "source_uri": source_uri,
        "author": author,
        "coordinates": coordinates, # Tuple (x,y,z,t_coord) or None
        "raw_t_intensity": intensity_val, # Float 0-1
        "ethics_score": ethics_score,
        "version": "1.0" 
    }

    # --- Public Storage Consent (Placeholder) ---
    require_consent = getattr(config, 'REQUIRE_PUBLIC_STORAGE_CONSENT', False) if config else False
    if entry['is_public'] and require_consent:
        _log_library_event("public_consent_requested", {"entry_id": entry_id, "preview": content_preview}, level="info")
        try:
            # This is a placeholder for development environments. 
            # In a real application, this would need a proper UI/API interaction.
            user_consent = input(f"Store content (preview: '{content_preview}') publicly? This is a dev placeholder. (yes/no): ").lower()
            if user_consent != "yes":
                _log_library_event("public_consent_refused", {"entry_id": entry_id}, level="info")
                return None
            _log_library_event("public_consent_placeholder", {"entry_id": entry_id, "status": "granted"}, level="info")
        except EOFError: # Happens in non-interactive environments
             _log_library_event("public_consent_eof_error", {"entry_id": entry_id, "message": "EOFError during input(), assuming no consent in non-interactive environment."}, level="warning")
             return None


    # --- Store and Save ---
    KNOWLEDGE_LIBRARY[entry_id] = entry
    _library_dirty_flag = True
    _save_knowledge_library()
    
    _log_library_event("knowledge_stored", {"entry_id": entry_id, "is_public": is_public, "source": source_uri if source_uri else "direct"})
    return entry_id

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
    if not isinstance(keyword, str) or not keyword.strip():
        _log_library_event("retrieve_by_keyword_failed", {"reason": "Keyword is empty or not a string."}, level="warning")
        return []

    found_entries = []
    lc_keyword = keyword.lower()

    for entry in KNOWLEDGE_LIBRARY.values():
        should_search_entry = False
        if entry.get('is_public', False) and search_public:
            should_search_entry = True
        elif not entry.get('is_public', False) and search_private:
            should_search_entry = True
        
        if should_search_entry:
            content_preview = entry.get('content_preview', "").lower()
            full_content = entry.get('full_content', "").lower() # Ensure full_content exists
            
            if lc_keyword in content_preview or lc_keyword in full_content:
                found_entries.append(entry) # Appending reference, consider deepcopy if entries can be modified by caller
    
    _log_library_event("retrieve_by_keyword_result", {"keyword": keyword, "found_count": len(found_entries), "search_public": search_public, "search_private": search_private})
    return found_entries


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
        # Define attributes expected by library.py for basic operation
        LIBRARY_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_knowledge_library.json")
        SYSTEM_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_system_log.txt")
        LOG_LEVEL = "debug"
        
        def ensure_path(self, path):
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
_IS_TEST_RUNNING = False # Default to False

if __name__ == "__main__":
    _IS_TEST_RUNNING = True # Set flag when running as main script
    print(f"INFO: _IS_TEST_RUNNING set to {_IS_TEST_RUNNING}")

    # --- Imports for Testing ---
    import unittest.mock as mock
    import copy

    # Ensure local versions of functions/classes are used for testing if module is reloaded
    # This is important because some tests might reload parts of the module or use patching.
    # However, for simple script execution, direct references are fine.
    # For more complex scenarios, a proper test framework (like pytest) handles this better.
    
    # --- Test Utilities ---
    class TempConfigOverride:
        """
        Temporarily overrides attributes in the 'config' module (or a mock object if 'config' is None).
        Restores original values or removes attributes upon exit.
        """
        def __init__(self, temp_configs_dict):
            self.temp_configs = temp_configs_dict
            self.original_values = {}
            self.config_module_was_none = False
            self.original_global_config = None # To store the original global 'config'

        def __enter__(self):
            global config # We need to modify the global 'config' directly
            self.original_global_config = config # Store the original global config

            if config is None:
                self.config_module_was_none = True
                # Create a dummy config object if it's None
                class DummyConfig: pass
                config = DummyConfig() # Assign new dummy to global 'config'
                # print("TempConfigOverride: Global 'config' was None, created dummy config module for test.", file=sys.stderr)
            
            for key, value in self.temp_configs.items():
                if hasattr(config, key):
                    self.original_values[key] = getattr(config, key)
                else:
                    self.original_values[key] = "__ATTR_NOT_SET__" # Sentinel
                setattr(config, key, value)
            return config # Return the (potentially modified) global config

        def __exit__(self, exc_type, exc_val, exc_tb):
            global config # Ensure we're referencing the global 'config'
            
            current_config_module = config # This is the module we modified (or the dummy)

            for key, original_value in self.original_values.items():
                if original_value == "__ATTR_NOT_SET__":
                    if hasattr(current_config_module, key):
                        delattr(current_config_module, key)
                else:
                    setattr(current_config_module, key, original_value)
            
            # Restore the original global 'config' reference
            config = self.original_global_config
            
            # If we created a dummy config and the original was None,
            # 'config' is now restored to None.
            # If the original was a module and we used it, it's restored to that module.
            # print(f"TempConfigOverride: Restored global 'config' to its original state: {type(config)}", file=sys.stderr)


    TEST_LIBRARY_LOG_FILENAME = "test_library_log.json"
    TEST_SYSTEM_LOG_FILENAME = "test_library_system_log.txt" # For _log_library_event testing

    def delete_test_files(test_path=None, system_log_path=None):
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

    def setup_test_environment(test_specific_configs=None, test_run_id="default"):
        """
        Sets up the test environment: deletes old test files, resets library state,
        and applies temporary config overrides.
        Returns the path to the test library file.
        """
        global KNOWLEDGE_LIBRARY, _library_dirty_flag, _LIBRARY_FILE_PATH
        
        # Generate unique filenames for this test run to allow parallel tests if ever needed
        # For now, simple fixed names are fine, but cleanup is important.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        test_lib_path = os.path.join(current_dir, f"test_library_log_{test_run_id}.json")
        test_sys_log_path = os.path.join(current_dir, f"test_library_system_log_{test_run_id}.txt")

        delete_test_files(test_lib_path, test_sys_log_path)

        # Reset module-level state
        KNOWLEDGE_LIBRARY = {}
        _library_dirty_flag = False
        _LIBRARY_FILE_PATH = None # Will be reset by _initialize_library_path

        # Default configs for testing
        base_test_configs = {
            "LIBRARY_LOG_PATH": test_lib_path,
            "SYSTEM_LOG_PATH": test_sys_log_path,
            "VERBOSE_OUTPUT": False, # Keep tests quieter unless specifically testing verbose output
            "LOG_LEVEL": "debug", # Capture all logs for tests
            "REQUIRE_PUBLIC_STORAGE_CONSENT": True, # Default to True for testing consent path
            "ensure_path": lambda path_to_ensure: os.makedirs(os.path.dirname(path_to_ensure), exist_ok=True) if os.path.dirname(path_to_ensure) else None,
            # Mitigator defaults (can be overridden by test_specific_configs)
            "ETHICAL_ALIGNMENT_THRESHOLD": 0.6, 
            "MITIGATION_ETHICAL_THRESHOLD": 0.7,
            "LIBRARY_SENSITIVE_KEYWORDS": ["test_sensitive"],
            "LIBRARY_REFRAMING_PHRASES": {"DEFAULT": "Test reframe."}
        }
        if test_specific_configs:
            base_test_configs.update(test_specific_configs)
        
        # TempConfigOverride will handle applying these.
        # The actual _initialize_library_path and _load_knowledge_library calls
        # should happen *within* each test function's TempConfigOverride context.
        return base_test_configs, test_lib_path, test_sys_log_path


    # --- Test Scenario Implementations ---
    test_results = {"passed": 0, "failed": 0, "details": []}

    def _run_test(test_func, test_name, *args):
        print(f"\n--- Running {test_name} ---")
        original_get_shared_manifold = globals().get('get_shared_manifold')
        original_score_ethics = globals().get('score_ethics')
        original_input = __builtins__.input if hasattr(__builtins__, 'input') else None

        try:
            test_func(*args)
            test_results["passed"] += 1
            test_results["details"].append(f"[PASS] {test_name}")
            print(f"[PASS] {test_name}")
        except AssertionError as e:
            test_results["failed"] += 1
            error_info = traceback.format_exc()
            test_results["details"].append(f"[FAIL] {test_name}: {e}\n{error_info}")
            print(f"[FAIL] {test_name}: Assertion Error: {e}\n{error_info}", file=sys.stderr)
        except Exception as e:
            test_results["failed"] += 1
            error_info = traceback.format_exc()
            test_results["details"].append(f"[FAIL] {test_name}: Exception: {e}\n{error_info}")
            print(f"[FAIL] {test_name}: Exception: {e}\n{error_info}", file=sys.stderr)
        finally:
            # Restore original functions if they were patched globally by a test
            # This is a basic way; proper test frameworks handle fixtures and cleanup better.
            globals()['get_shared_manifold'] = original_get_shared_manifold
            globals()['score_ethics'] = original_score_ethics
            if original_input is not None and hasattr(__builtins__, 'input'):
                 __builtins__.input = original_input
            elif hasattr(__builtins__, 'input') and original_input is None: # if it was added by a test
                 del __builtins__.input

    # Test Functions
    def test_text_utilities():
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
        assert is_valid_coordinate((1,2,3)) == True
        assert is_valid_coordinate([1,2,3,4.0]) == True
        assert is_valid_coordinate((1,2)) == False # Too short
        assert is_valid_coordinate((1,2,3,4,5)) == False # Too long
        assert is_valid_coordinate((1,2,'a')) == False # Non-numeric
        assert is_valid_coordinate("1,2,3") == False # Wrong type

    def test_mitigator_class():
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
            assert "[Content Under Review" in moderated or "[Content Moderated" in moderated
            
            # High score but sensitive keyword
            moderated = mitigator.moderate_ethically_flagged_content("high score but danger word", 0.9)
            assert "[Content Under Review" in moderated or "[Content Moderated" in moderated
            
            # Strict mode, score below alignment but above mitigation_threshold (if thresholds are different enough)
            # With current test_configs: alignment=0.5, mitigation=0.6. So score 0.55 + strict_mode
            moderated = mitigator.moderate_ethically_flagged_content("strict mode test content", 0.55, strict_mode=True)
            assert "[Content Under Review" in moderated or "[Content Moderated" in moderated
            
             # Very low score -> generic placeholder
            moderated = mitigator.moderate_ethically_flagged_content("very bad content", 0.1)
            assert "[Content Moderated due to Ethical Concerns" in moderated
            assert "Please rephrase or contact support" in moderated

            # High score, no keywords
            original = "perfectly fine content"
            moderated = mitigator.moderate_ethically_flagged_content(original, 0.95)
            assert moderated == original

    def test_knowledge_library_persistence():
        test_configs, test_lib_path, _ = setup_test_environment(test_run_id="persistence")
        with TempConfigOverride(test_configs):
            _initialize_library_path() # Sets _LIBRARY_FILE_PATH based on temp config
            _load_knowledge_library() # Should load empty or create new
            assert len(KNOWLEDGE_LIBRARY) == 0, "Library not empty at start of persistence test"

            KNOWLEDGE_LIBRARY["test001"] = {"data": "my test data", "timestamp": "now"}
            global _library_dirty_flag # Make sure we're affecting the global
            _library_dirty_flag = True
            _save_knowledge_library()
            
            assert os.path.exists(test_lib_path), "Library file not created by save"
            assert _library_dirty_flag == False, "Dirty flag not reset after save"

            KNOWLEDGE_LIBRARY.clear() # Clear in-memory
            _load_knowledge_library()
            assert len(KNOWLEDGE_LIBRARY) == 1, "Library not loaded correctly"
            assert KNOWLEDGE_LIBRARY["test001"]["data"] == "my test data", "Loaded data mismatch"
        delete_test_files(test_lib_path)


    @mock.patch('core.library.score_ethics')
    @mock.patch('core.library.get_shared_manifold')
    @mock.patch('builtins.input')
    def test_store_knowledge(mock_input, mock_get_manifold, mock_score_ethics_func):
        # Setup mock manifold and its methods
        mock_manifold_instance = mock.MagicMock()
        mock_manifold_instance.bootstrap_concept_from_llm = mock.MagicMock(return_value=((0.1, 0.2, 0.3, 0.4), 0.5, "Mock summary from LLM"))
        mock_get_manifold.return_value = mock_manifold_instance
        
        test_run_id = "store"
        test_configs, test_lib_path, test_sys_log_path = setup_test_environment({"REQUIRE_PUBLIC_STORAGE_CONSENT": True}, test_run_id=test_run_id)

        with TempConfigOverride(test_configs):
            _initialize_library_path()
            _load_knowledge_library()

            # Scenario 1: Store private item
            mock_score_ethics_func.return_value = 0.9
            entry_id1 = store_knowledge("Test private content", is_public=False, author="Test Author")
            assert entry_id1 is not None
            assert entry_id1 in KNOWLEDGE_LIBRARY
            assert KNOWLEDGE_LIBRARY[entry_id1]["is_public"] == False
            assert KNOWLEDGE_LIBRARY[entry_id1]["author"] == "Test Author"
            assert KNOWLEDGE_LIBRARY[entry_id1]["coordinates"] == (0.1, 0.2, 0.3, 0.4) # from mock_bootstrap
            assert KNOWLEDGE_LIBRARY[entry_id1]["raw_t_intensity"] == 0.5
            assert KNOWLEDGE_LIBRARY[entry_id1]["ethics_score"] == 0.9
            mock_manifold_instance.bootstrap_concept_from_llm.assert_called_once()
            mock_score_ethics_func.assert_called_once()
            mock_input.assert_not_called() # Private, no consent needed

            # Scenario 2: Store public item with consent
            mock_input.reset_mock(); mock_manifold_instance.bootstrap_concept_from_llm.reset_mock(); mock_score_ethics_func.reset_mock()
            mock_input.return_value = "yes"
            mock_score_ethics_func.return_value = 0.8
            entry_id2 = store_knowledge("Test public content with consent", is_public=True)
            assert entry_id2 is not None
            assert entry_id2 in KNOWLEDGE_LIBRARY
            assert KNOWLEDGE_LIBRARY[entry_id2]["is_public"] == True
            mock_input.assert_called_once()

            # Scenario 3: Store public item, consent denied
            mock_input.reset_mock(); mock_manifold_instance.bootstrap_concept_from_llm.reset_mock(); mock_score_ethics_func.reset_mock()
            mock_input.return_value = "no"
            entry_id3 = store_knowledge("Test public consent denied", is_public=True)
            assert entry_id3 is None
            assert entry_id3 not in KNOWLEDGE_LIBRARY
            mock_input.assert_called_once()
            
            # Scenario 4: Brain/Ethics module failure handling
            mock_input.reset_mock(); mock_manifold_instance.bootstrap_concept_from_llm.reset_mock(); mock_score_ethics_func.reset_mock()
            mock_input.return_value = "yes" # Assume consent for simplicity
            
            # Brain failure
            mock_manifold_instance.bootstrap_concept_from_llm.side_effect = Exception("Brain is down")
            mock_score_ethics_func.return_value = 0.7 # Ethics still works
            entry_id4_brain_fail = store_knowledge("Content with brain failure", is_public=False)
            assert entry_id4_brain_fail is not None
            assert KNOWLEDGE_LIBRARY[entry_id4_brain_fail]["coordinates"] is None
            assert KNOWLEDGE_LIBRARY[entry_id4_brain_fail]["raw_t_intensity"] == 0.0 # Default fallback
            assert KNOWLEDGE_LIBRARY[entry_id4_brain_fail]["ethics_score"] == 0.7 # Score from ethics
            
            # Ethics failure
            mock_manifold_instance.bootstrap_concept_from_llm.side_effect = None # Reset brain mock
            mock_manifold_instance.bootstrap_concept_from_llm.return_value = ((0.1,0.2,0.3,0.4), 0.5, "Mock summary")
            mock_score_ethics_func.side_effect = Exception("Ethics is down")
            entry_id5_ethics_fail = store_knowledge("Content with ethics failure", is_public=False)
            assert entry_id5_ethics_fail is not None
            assert KNOWLEDGE_LIBRARY[entry_id5_ethics_fail]["ethics_score"] == 0.5 # Default fallback for ethics
        
        delete_test_files(test_lib_path, test_sys_log_path)

    def test_retrieval_functions():
        test_run_id = "retrieval"
        test_configs, test_lib_path, test_sys_log_path = setup_test_environment(test_run_id=test_run_id)
        with TempConfigOverride(test_configs):
            _initialize_library_path()
            _load_knowledge_library() # Ensure KNOWLEDGE_LIBRARY is empty

            # Populate KNOWLEDGE_LIBRARY directly for this test
            entry1 = {"id": "id1", "full_content": "Public entry about apples", "content_preview": "apples", "is_public": True}
            entry2 = {"id": "id2", "full_content": "Private entry about bananas", "content_preview": "bananas", "is_public": False}
            entry3 = {"id": "id3", "full_content": "Another public entry about apples and oranges", "content_preview": "apples oranges", "is_public": True}
            global KNOWLEDGE_LIBRARY, _library_dirty_flag
            KNOWLEDGE_LIBRARY = {"id1": entry1, "id2": entry2, "id3": entry3}
            _library_dirty_flag = False # Assume it's loaded like this

            # Test retrieve_knowledge_by_id
            assert retrieve_knowledge_by_id("id1") == entry1
            assert retrieve_knowledge_by_id("nonexistent") is None

            # Test retrieve_knowledge_by_keyword
            # Keyword in public item, search_public=True
            results = retrieve_knowledge_by_keyword("apples", search_public=True, search_private=False)
            assert len(results) == 2
            assert entry1 in results and entry3 in results
            
            # Keyword in private item, search_private=True
            results = retrieve_knowledge_by_keyword("bananas", search_public=False, search_private=True)
            assert len(results) == 1
            assert entry2 in results
            
            # Keyword present, but flags mismatch
            results = retrieve_knowledge_by_keyword("bananas", search_public=True, search_private=False)
            assert len(results) == 0
            
            # Keyword not present
            results = retrieve_knowledge_by_keyword("kiwi")
            assert len(results) == 0
            
            # Case insensitivity
            results = retrieve_knowledge_by_keyword("APPLES", search_public=True, search_private=True)
            assert len(results) == 2

        delete_test_files(test_lib_path, test_sys_log_path)


    # --- Test Runner Logic ---
    print("Starting Core Library Self-Tests...")
    
    # Explicitly manage globals for mocks if not using @patch on the test function itself
    # This is more complex than direct patching, but shown for variance.
    # However, for simplicity and robustness, patching the functions directly in the
    # test_store_knowledge function (as done) is generally preferred.

    _run_test(test_text_utilities, "test_text_utilities")
    _run_test(test_is_valid_coordinate, "test_is_valid_coordinate")
    _run_test(test_mitigator_class, "test_mitigator_class")
    _run_test(test_knowledge_library_persistence, "test_knowledge_library_persistence")
    _run_test(test_store_knowledge, "test_store_knowledge") # Patches are applied via decorators here
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

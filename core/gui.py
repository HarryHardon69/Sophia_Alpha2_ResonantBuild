"""
core/gui.py

Provides a Streamlit-based Graphical User Interface for interacting with
Sophia_Alpha2.
"""

__version__ = "0.1.0"

import json
import os
import sys
import time # Used for response streaming simulation
import traceback
import html # For escaping HTML characters
import logging # For structured logging
from cryptography.fernet import Fernet, InvalidToken
import json # For serializing data before encryption
import zlib # For compressing data before encryption
import functools # For lru_cache
# datetime is not directly used but useful for timestamped logging if added later.
# import datetime 

import streamlit as st

# --- Logger Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr  # Output to stderr
)
logger = logging.getLogger(__name__)

# --- Encryption Setup ---
# IMPORTANT: This key should be stored securely, e.g., in config or environment variables, not hardcoded.
# For demonstration purposes, we generate and use a hardcoded key.
# In a real scenario, you might load this from config.SESSION_ENCRYPTION_KEY
_ENCRYPTION_KEY = Fernet.generate_key()
_FERNET_INSTANCE = Fernet(_ENCRYPTION_KEY)

def encrypt_data(data):
    """Encrypts data after serializing to JSON and compressing if appropriate."""
    if isinstance(data, (list, dict)): # Compress list or dict data
        serialized_data = json.dumps(data).encode('utf-8')
        compressed_data = zlib.compress(serialized_data)
        payload = compressed_data
        # logger.debug(f"Compressed data from {len(serialized_data)} to {len(compressed_data)}")
    elif isinstance(data, str):
        payload = data.encode('utf-8') # Encrypt strings directly
    elif isinstance(data, bytes):
        payload = data # Assume bytes are already formatted (e.g. already compressed)
    else: # For other types, serialize to JSON then compress (though primarily targeting list/dict)
        serialized_data = json.dumps(data).encode('utf-8')
        payload = zlib.compress(serialized_data)
        # logger.debug(f"Serialized and compressed other data type to {len(payload)}")

    return _FERNET_INSTANCE.encrypt(payload)

def decrypt_data(token):
    """Decrypts data, decompresses, and deserializes from JSON if needed."""
    try:
        decrypted_payload = _FERNET_INSTANCE.decrypt(token)

        # Attempt decompression; if it fails, assume it was not compressed (e.g. simple string)
        try:
            decompressed_bytes = zlib.decompress(decrypted_payload)
        except zlib.error: # Not compressed or corrupted
            # logger.debug("Data was not zlib compressed or is corrupted, trying direct decode.")
            decompressed_bytes = decrypted_payload # Treat as if it was not compressed

        # Attempt to deserialize JSON; if it fails, return as string
        try:
            return json.loads(decompressed_bytes.decode('utf-8'))
        except json.JSONDecodeError:
            # If JSON decoding fails, it might be a simple string that was encrypted
            # or binary data that was not meant to be JSON.
            return decompressed_bytes.decode('utf-8', errors='replace') # Return as string, replace errors
    except InvalidToken:
        logger.error("Failed to decrypt data: Invalid token or key.")
        return None
    except Exception as e:
        logger.exception(f"An unexpected error occurred during decryption: {e}")
        return None


# --- Configuration Import ---
try:
    # Attempt to get key from config if available
    if config and hasattr(config, 'SESSION_ENCRYPTION_KEY'):
        loaded_key = getattr(config, 'SESSION_ENCRYPTION_KEY')
        if isinstance(loaded_key, str): # Assuming key is stored as string
            _ENCRYPTION_KEY = loaded_key.encode('utf-8')
        else: # Assuming key is stored as bytes
            _ENCRYPTION_KEY = loaded_key
        _FERNET_INSTANCE = Fernet(_ENCRYPTION_KEY)
        logger.info("Successfully loaded SESSION_ENCRYPTION_KEY from config.")
    else:
        logger.warning("SESSION_ENCRYPTION_KEY not found in config. Using a generated key for this session. THIS IS NOT SECURE FOR PRODUCTION.")

    from .. import config
except ImportError:
    logger.warning("Could not import 'config' from parent package. Attempting for standalone use.")
    try:
        import config as app_config # Use an alias
        config = app_config
        logger.info("Successfully imported 'config' directly.")
    except ImportError:
        logger.error("Failed to import 'config'. Some features may be limited or use defaults.")
        config = None # Placeholder

# --- Core dialogue module imports ---
_DIALOGUE_AVAILABLE = False
_PERSONA_AVAILABLE = False 
_GET_DIALOGUE_PERSONA_AVAILABLE = False 

# Attempt to import actual dialogue generation
try:
    from ..dialogue import generate_response as original_generate_response
    from ..persona import Persona 
    from ..dialogue import get_dialogue_persona
    _DIALOGUE_AVAILABLE = True 
    _GET_DIALOGUE_PERSONA_AVAILABLE = True 
    _PERSONA_AVAILABLE = True 
    if not getattr(sys, '_IS_TEST_RUNNING', False):
        logger.info("Successfully imported core.dialogue and core.persona components.")

    # --- Response Caching ---
    # IMPORTANT: This cache is based on input only. If persona state significantly alters
    # responses for the same input, this cache might return stale data.
    # Consider a more sophisticated cache invalidation strategy if this becomes an issue.
    @functools.lru_cache(maxsize=32) # Cache up to 32 recent responses
    def _cached_generate_response(user_input_tuple_key, stream_thought_steps=False):
        # user_input_tuple_key is expected to be a tuple, e.g. (user_input_string,)
        # This is to make it hashable for lru_cache.
        user_input = user_input_tuple_key[0]
        logger.info(f"Cache miss for: {user_input[:50]}...") # Log cache misses for observation
        return original_generate_response(user_input, stream_thought_steps=stream_thought_steps)

    generate_response = _cached_generate_response

except ImportError as e_dialogue_import:
    if not getattr(sys, '_IS_TEST_RUNNING', False):
        logger.error(f"Error importing from core.dialogue or core.persona: {e_dialogue_import}. GUI functionality will be significantly limited.")
    
    # Fallback generate_response if imports fail
    def generate_response(user_input_tuple_key, stream_thought_steps=False): # Matches cached signature
        user_input = user_input_tuple_key[0] if isinstance(user_input_tuple_key, tuple) else user_input_tuple_key
        return f"Error: Dialogue module not available. Cannot process '{user_input}'.", ["Dialogue module unavailable."], {"error": "Dialogue module unavailable"}
    
    class MockPersonaSingleton:
        name = "ErrorPersona"
        mode = "error"
        traits = ["Unavailable"]
        awareness = {"status": "Persona module unavailable"}
        def get_intro(self): return "Persona module is unavailable."
        def _initialize_default_state_and_save(self): pass 

    def get_dialogue_persona():
        return MockPersonaSingleton()
    
    if not _PERSONA_AVAILABLE: 
        class Persona: 
            name = "MockErrorPersona"
            def __init__(self, *args, **kwargs):
                self.name = "MockErrorPersona"
                self.mode = "error_mock_class"
                self.traits = ["Unavailable_mock_class"]
                self.awareness = {"status": "Persona class unavailable_mock_class"}
            def get_intro(self): return "Persona class (mock) is unavailable."
            def update_awareness(self, *args, **kwargs): pass
            def save_state(self, *args, **kwargs): pass
            def load_state(self, *args, **kwargs): pass
            def _initialize_default_state_and_save(self): pass

# --- Streamlit Page Configuration ---
try:
    persona_name_for_title = getattr(config, 'PERSONA_NAME', 'Sophia_Alpha2') if config else 'Sophia_Alpha2'
    st.set_page_config(
        page_title=f"{persona_name_for_title} GUI",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception as e_st_config:
    if not getattr(sys, '_IS_TEST_RUNNING', False): 
        logger.error(f"Streamlit page config failed. Are you running this as a Streamlit app? Error: {e_st_config}")

# --- Session State Initialization ---
def initialize_session_state():
    # Initialize non-encrypted parts first or ensure they are handled correctly
    if 'persona_instance' not in st.session_state: # Not encrypting persona object directly
        st.session_state.persona_instance = None
    if 'stream_thoughts_gui' not in st.session_state: # Basic boolean, no need to encrypt
        default_stream_thoughts = getattr(config, 'VERBOSE_OUTPUT', False) if config else False
        st.session_state.stream_thoughts_gui = default_stream_thoughts
    if 'error_message' not in st.session_state: # Error messages for UI, not encrypted
        st.session_state.error_message = None

    # Initialize encrypted session state variables
    encrypted_fields = {
        'dialogue_history': [],
        'last_thought_steps': [],
        'last_awareness_metrics': {}
    }

    # Placeholder for loading history from memory
    logger.info("Placeholder: Add call to core.memory.load_dialogue_history(...) here and populate/replace encrypted_dialogue_history if available.")
    # Example:
    # loaded_history = core.memory.load_dialogue_history()
    # if loaded_history is not None:
    #     encrypted_fields['dialogue_history'] = loaded_history # Assuming it's already in the correct list-of-dicts format

    for key, default_value in encrypted_fields.items():
        encrypted_key_name = f"encrypted_{key}"
        if encrypted_key_name not in st.session_state:
            # Encrypt and store the default value (or loaded value for dialogue_history)
            if key == 'dialogue_history' and 'loaded_history' in locals() and loaded_history is not None:
                 st.session_state[encrypted_key_name] = encrypt_data(loaded_history)
                 logger.info(f"Initialized and encrypted '{key}' from memory into session state.")
            else:
                st.session_state[encrypted_key_name] = encrypt_data(default_value)
                logger.debug(f"Initialized and encrypted '{key}' with default value in session state.")

    # Attempt to initialize or retrieve the Persona instance if it's not already in session state.
    if st.session_state.persona_instance is None:
        if _GET_DIALOGUE_PERSONA_AVAILABLE and _PERSONA_AVAILABLE: # Check if dependencies are met.
            try:
                # get_dialogue_persona() should handle its own singleton logic.
                st.session_state.persona_instance = get_dialogue_persona()
                if st.session_state.persona_instance is None: # If get_dialogue_persona() fails.
                    st.session_state.error_message = "Critical: Failed to initialize Persona instance via get_dialogue_persona()."
                    if not getattr(sys, '_IS_TEST_RUNNING', False): logger.error(f"Initialize: {st.session_state.error_message}")
                else: # Successfully got persona instance.
                    if not getattr(sys, '_IS_TEST_RUNNING', False): logger.info(f"Initialize: Persona instance '{st.session_state.persona_instance.name}' loaded/retrieved into session state.")
            except Exception as e_get_persona_init: # Catch any unexpected error.
                st.session_state.error_message = f"Critical error during get_dialogue_persona(): {str(e_get_persona_init)}"
                if not getattr(sys, '_IS_TEST_RUNNING', False):
                    logger.exception(f"Initialize: {st.session_state.error_message}") # Logs with exception info
        elif not _GET_DIALOGUE_PERSONA_AVAILABLE or not _PERSONA_AVAILABLE : # If core dependencies are missing.
            st.session_state.error_message = "Core dialogue.py or persona.py components are not available. GUI functionality will be severely limited."
            if not getattr(sys, '_IS_TEST_RUNNING', False): logger.error(f"Initialize: {st.session_state.error_message}")
            # Use the mock persona defined at the top of the file if Persona class itself was not available.
            if st.session_state.persona_instance is None:
                 st.session_state.persona_instance = get_dialogue_persona() # This will return the MockPersonaSingleton

# Call initialization at script load (Streamlit re-runs the script on interaction).
initialize_session_state()

# --- Constants ---
MAX_HISTORY_TURNS = 50 # Max 50 user messages and 50 assistant responses

# --- Authentication ---
# Mock credentials (replace with st.secrets in a real environment)
MOCK_USERNAME = "admin"
MOCK_PASSWORD = "password123" # In a real app, use st.secrets

def check_login():
    """Checks if the user is authenticated, displays login form if not."""
    if st.session_state.get('authenticated'):
        return True

    st.title("Login Required")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            # Use st.secrets for production
            # app_username = st.secrets.get("APP_USERNAME", MOCK_USERNAME)
            # app_password = st.secrets.get("APP_PASSWORD", MOCK_PASSWORD)
            app_username = MOCK_USERNAME
            app_password = MOCK_PASSWORD

            if username == app_username and password == app_password:
                st.session_state['authenticated'] = True
                st.rerun() # Rerun to reflect authenticated state
            else:
                st.error("Incorrect username or password.")
    return False

# --- GUI Rendering Logic ---
def render_main_interface():
    """
    Renders the main Streamlit interface, including sidebar and chat elements.
    
    This function orchestrates the display of persona information, awareness metrics,
    dialogue history, user input handling, response generation, and thought streaming.
    It relies on `st.session_state` for maintaining state across interactions.
    """
    if not check_login():
        return # Stop rendering if not authenticated

    # --- Initial Error Display ---
    # Display any critical error message that might have occurred during startup/initialization.
    # `startup_error_displayed` flag ensures it's shown only once per session if the error persists.
    if st.session_state.error_message and 'startup_error_displayed' not in st.session_state:
        st.error(st.session_state.error_message)
        st.session_state.startup_error_displayed = True
    
    # --- Persona Instance Handling ---
    # Retrieve the persona instance from session state.
    persona = st.session_state.persona_instance # Persona object itself is not encrypted
    if persona is None: # If persona is still None after initialization attempts.
        if not st.session_state.error_message: # Show a generic error if no specific one was set.
            st.error("Fatal Error: Persona instance is unavailable. The GUI cannot operate.")
        return # Halt further rendering if persona is essential and missing.

    # --- Initial Message / Persona Introduction ---
    dialogue_history = decrypt_data(st.session_state.encrypted_dialogue_history)
    if dialogue_history is None: dialogue_history = [] # Handle decryption failure

    if not dialogue_history and hasattr(persona, 'get_intro'):
        try:
            intro_message = persona.get_intro()
            dialogue_history.append({"role": "assistant", "content": intro_message})
            st.session_state.encrypted_dialogue_history = encrypt_data(dialogue_history)
        except Exception as e_intro_render: # Handle errors getting intro.
            error_intro_msg = f"Error displaying persona introduction: {str(e_intro_render)}"
            # This append is to a local var, ensure it's saved if needed, or log appropriately
            logger.exception("Error displaying persona introduction:")
            # Potentially add this error to a non-encrypted UI error display if critical

    # --- Sidebar Rendering ---
    # The sidebar displays persona details, awareness metrics, and control elements.
    st.sidebar.title("Controls & Awareness Monitor")
    if persona: # Check if persona object exists and is not None.
        st.sidebar.subheader(f"Persona: {getattr(persona, 'name', 'N/A')}")
        st.sidebar.caption(f"Mode: {getattr(persona, 'mode', 'N/A')}")
        traits_list_sidebar = getattr(persona, 'traits', [])
        if isinstance(traits_list_sidebar, list): st.sidebar.caption(f"Traits: {', '.join(traits_list_sidebar)}")
        else: st.sidebar.caption(f"Traits (raw): {str(traits_list_sidebar)}") # Fallback for unexpected type.
        st.sidebar.markdown("---")
        
        # Display overall awareness metrics from the Persona instance.
        st.sidebar.subheader("Current Overall Awareness")
        awareness_data_sidebar = getattr(persona, 'awareness', {})
        if isinstance(awareness_data_sidebar, dict):
            for key, value in awareness_data_sidebar.items():
                display_key_sidebar = key.replace('_', ' ').title() # Format key for display.
                # Format values appropriately (float, list/tuple, or string).
                if isinstance(value, float): st.sidebar.text(f"{display_key_sidebar}: {value:.2f}")
                elif isinstance(value, (list, tuple)): st.sidebar.text(f"{display_key_sidebar}: {str(value)}")
                else: st.sidebar.text(f"{display_key_sidebar}: {value}")
        else: st.sidebar.text("Awareness data is not in the expected dictionary format.")
    else: st.sidebar.warning("Persona instance not loaded. Awareness data unavailable.")
    st.sidebar.markdown("---")

    # Display snapshot of awareness metrics from the last interaction.
    last_awareness_metrics = decrypt_data(st.session_state.encrypted_last_awareness_metrics)
    if last_awareness_metrics is None: last_awareness_metrics = {} # Handle decryption failure

    if last_awareness_metrics:
        st.sidebar.subheader("Last Interaction Snapshot")
        metrics_snapshot = last_awareness_metrics # Use decrypted data

        def get_metric_value(metric_name, default=0.0, is_numeric=True):
            val = metrics_snapshot.get(metric_name, default)
            if is_numeric:
                if not isinstance(val, (int, float)):
                    logger.warning(f"Metric '{metric_name}' is not numeric: {val}. Defaulting to {default}.")
                    return default
                return float(val)
            if not isinstance(val, str): # For non-numeric, expecting string typically
                 # For things like active_llm_fallback (bool) or primary_concept_coord (list/tuple)
                if isinstance(val, (bool, list, tuple)):
                    return str(val) # Convert common non-string types to string for display
                logger.warning(f"Metric '{metric_name}' is not a string: {val}. Defaulting to {str(default)}.")
                return str(default)
            return val

        curiosity_metric = get_metric_value('curiosity')
        coherence_metric = get_metric_value('coherence')
        context_stability_metric = get_metric_value('context_stability')
        self_evolution_rate_metric = get_metric_value('self_evolution_rate')

        col1_sidebar, col2_sidebar = st.sidebar.columns(2)
        col1_sidebar.metric("Curiosity", f"{curiosity_metric:.2f}")
        col2_sidebar.metric("Coherence", f"{coherence_metric:.2f}")
        col1_sidebar.metric("Context Stability", f"{context_stability_metric:.2f}")
        col2_sidebar.metric("Self Evolution", f"{self_evolution_rate_metric:.2f}")

        llm_fallback_used = get_metric_value('active_llm_fallback', 'N/A', is_numeric=False)
        st.sidebar.text(f"LLM Fallback Used: {llm_fallback_used}")

        pcc_display_sidebar = get_metric_value('primary_concept_coord', 'N/A', is_numeric=False)
        st.sidebar.text(f"Primary Concept Coords: {pcc_display_sidebar}")
    st.sidebar.markdown("---")

    # Settings controls in the sidebar.
    st.sidebar.subheader("Settings & Actions")
    st.checkbox("Show Thought Stream Expander", key="stream_thoughts_gui", help="Toggle visibility of Sophia's detailed thought process for the last response.") # Not encrypted
    
    if st.sidebar.button("Clear Dialogue History"):
        # Encrypt and store empty values for the cleared fields
        st.session_state.encrypted_dialogue_history = encrypt_data([])
        st.session_state.encrypted_last_thought_steps = encrypt_data([])
        st.session_state.encrypted_last_awareness_metrics = encrypt_data({})

        # Re-add persona intro if available
        current_dialogue_history = [] # Start fresh
        if persona and hasattr(persona, 'get_intro'):
            try:
                current_dialogue_history.append({"role": "assistant", "content": persona.get_intro()})
                st.session_state.encrypted_dialogue_history = encrypt_data(current_dialogue_history)
            except Exception as e_reinit_intro:
                # Log error, potentially add error message to UI if critical
                logger.error(f"Error re-initializing intro after clearing history: {e_reinit_intro}")
                current_dialogue_history.append({"role": "assistant", "content": f"Error re-initializing intro: {e_reinit_intro}"})
                st.session_state.encrypted_dialogue_history = encrypt_data(current_dialogue_history) # Save error message too
        st.rerun()

    if st.sidebar.button("Reset Persona State"):
        if persona and hasattr(persona, '_initialize_default_state_and_save'):
            try:
                persona._initialize_default_state_and_save() # Persona state reset
                # Clear encrypted session state related to dialogue and previous interaction
                st.session_state.encrypted_dialogue_history = encrypt_data([])
                st.session_state.encrypted_last_thought_steps = encrypt_data([])
                st.session_state.encrypted_last_awareness_metrics = encrypt_data({})

                # Re-add persona intro
                current_dialogue_history_reset = []
                if hasattr(persona, 'get_intro'):
                    current_dialogue_history_reset.append({"role": "assistant", "content": persona.get_intro()})
                    st.session_state.encrypted_dialogue_history = encrypt_data(current_dialogue_history_reset)

                st.sidebar.success("Persona state has been reset to defaults.")
            except Exception as e_reset_persona:
                logger.error(f"Error resetting persona state: {e_reset_persona}")
                st.sidebar.error(f"Error resetting persona state: {e_reset_persona}")
        else: st.sidebar.warning("Persona instance unavailable or does not support state reset.")
        st.rerun()

    # --- Main Chat Interface Rendering ---
    persona_display_name = getattr(persona, 'name', 'Sophia') if persona else 'Sophia_Unavailable'
    st.title(f"Chat with {persona_display_name}")

    # Decrypt dialogue history for display
    current_dialogue_history_display = decrypt_data(st.session_state.encrypted_dialogue_history)
    if current_dialogue_history_display is None: current_dialogue_history_display = [] # Handle decryption failure

    for message_item in current_dialogue_history_display:
        role_for_msg, avatar_char_for_msg = ("assistant", "🧠") if message_item.get("role") == "assistant" else ("user", "👤")
        message_sender_name = persona_display_name if role_for_msg == "assistant" else "You"
        with st.chat_message(name=message_sender_name, avatar=avatar_char_for_msg):
            st.markdown(message_item["content"])

    # Expander for displaying Sophia's thought process if toggled.
    # Decrypt last_thought_steps for display
    last_thought_steps_display = decrypt_data(st.session_state.encrypted_last_thought_steps)
    if last_thought_steps_display is None: last_thought_steps_display = [] # Handle decryption failure

    if st.session_state.get("stream_thoughts_gui", False) and last_thought_steps_display:
        with st.expander("Sophia's Thought Stream (Last Response)", expanded=False):
            thought_steps_formatted_text = "".join([f"- {step}\n" for step in last_thought_steps_display])
            st.text_area("Detailed Thoughts:", value=thought_steps_formatted_text, height=200, disabled=True, key="thought_stream_display_area")

    # Chat input field for user query.
    user_query_input = st.chat_input(f"Ask {persona_display_name}...")
    if user_query_input:
        sanitized_query = html.escape(user_query_input)

        # --- GUI Command Handling ---
        if sanitized_query.startswith("/"):
            command_parts = sanitized_query.split()
            command = command_parts[0].lower()
            # args = command_parts[1:] # Future use for commands with arguments

            if command == "/clear_metrics":
                st.session_state.encrypted_last_awareness_metrics = encrypt_data({})
                st.toast("Awareness metrics cleared.", icon="🧹")
                logger.info("GUI command: /clear_metrics executed.")
                # No rerun needed if toast is sufficient, otherwise st.rerun()
            elif command == "/help_gui":
                help_text = """
                **Available GUI Commands:**
                - `/clear_metrics`: Clears the 'Last Interaction Snapshot' metrics display.
                - `/help_gui`: Displays this help message.
                """
                # Add help text to dialogue history as an assistant message
                dialogue_history_for_help = decrypt_data(st.session_state.encrypted_dialogue_history)
                if dialogue_history_for_help is None: dialogue_history_for_help = []
                dialogue_history_for_help.append({"role": "assistant", "content": help_text})
                st.session_state.encrypted_dialogue_history = encrypt_data(dialogue_history_for_help)
                logger.info("GUI command: /help_gui executed.")
                st.rerun() # Rerun to display the help message in chat
            else:
                # Unknown command, treat as regular input for now or show error
                # For now, let's add it to history as a user message and let assistant handle it or show "unknown command"
                dialogue_history_for_unknown_command = decrypt_data(st.session_state.encrypted_dialogue_history)
                if dialogue_history_for_unknown_command is None: dialogue_history_for_unknown_command = []
                unknown_cmd_message = f"Unknown command: {command}. Treating as regular input."
                # dialogue_history_for_unknown_command.append({"role": "assistant", "content": unknown_cmd_message}) # Option 1: Assistant message
                # st.toast(unknown_cmd_message, icon="❓") # Option 2: Toast
                # Option 3: Just process as normal input (current behavior if no special handling)
                logger.warning(f"Unknown GUI command: {command}. Treating as regular input.")
                # Fall through to regular message processing if we want to send it to the assistant.
                # For now, let's just add user's original command to history and proceed.
                dialogue_history_for_update = decrypt_data(st.session_state.encrypted_dialogue_history)
                if dialogue_history_for_update is None: dialogue_history_for_update = []
                dialogue_history_for_update.append({"role": "user", "content": sanitized_query})
                st.session_state.encrypted_dialogue_history = encrypt_data(dialogue_history_for_update)
                # Placeholder for saving user input to memory
                logger.info(f"Placeholder: Add call to core.memory.save_dialogue_history_event('user', '{sanitized_query}') here.")
                # Then proceed to generate response for the unknown command as if it's normal text
                process_as_regular_input = True
        else: # Not a command, process as regular input
            process_as_regular_input = True
            dialogue_history_for_update = decrypt_data(st.session_state.encrypted_dialogue_history)
            if dialogue_history_for_update is None: dialogue_history_for_update = []
            dialogue_history_for_update.append({"role": "user", "content": sanitized_query})
            st.session_state.encrypted_dialogue_history = encrypt_data(dialogue_history_for_update)
            # Placeholder for saving user input to memory
            logger.info(f"Placeholder: Add call to core.memory.save_dialogue_history_event('user', '{sanitized_query}') here.")

        if process_as_regular_input:
            with st.chat_message(name=persona_display_name, avatar="🧠"):
                response_display_placeholder = st.empty()
                response_display_placeholder.markdown("Thinking...")
                try:
                    if not _DIALOGUE_AVAILABLE:
                        sophia_response_text, temp_thought_steps, temp_awareness_metrics = "Error: Dialogue module not available. Cannot process input.", ["Dialogue module unavailable."], {"error": "Dialogue module unavailable"}
                    else:
                        # Use tuple as key for caching
                        sophia_response_text, temp_thought_steps, temp_awareness_metrics = generate_response((sanitized_query,), stream_thought_steps=False)

                    st.session_state.encrypted_last_thought_steps = encrypt_data(temp_thought_steps)
                    st.session_state.encrypted_last_awareness_metrics = encrypt_data(temp_awareness_metrics)

                full_response_for_streaming = ""
                response_words = sophia_response_text.split()
                    if not response_words: # Handle empty response string.
                        response_display_placeholder.markdown(sophia_response_text.strip())
                    else:
                        # Validate streaming_delay
                        streaming_delay_val = getattr(config, 'GUI_RESPONSE_STREAMING_DELAY', 0.05) if config else 0.05
                        if not isinstance(streaming_delay_val, (float, int)) or streaming_delay_val < 0:
                            logger.warning(f"Invalid GUI_RESPONSE_STREAMING_DELAY '{streaming_delay_val}'. Defaulting to 0.05.")
                            streaming_delay_val = 0.05

                        for word_chunk in response_words:
                            full_response_for_streaming += word_chunk + " "
                            response_display_placeholder.markdown(full_response_for_streaming + "▌")
                            time.sleep(streaming_delay_val)
                        response_display_placeholder.markdown(full_response_for_streaming.strip())

                    dialogue_history_for_response_update = decrypt_data(st.session_state.encrypted_dialogue_history)
                    if dialogue_history_for_response_update is None: dialogue_history_for_response_update = []
                    dialogue_history_for_response_update.append({"role": "assistant", "content": sophia_response_text.strip()})

                    # --- History Capping ---
                    if len(dialogue_history_for_response_update) > MAX_HISTORY_TURNS * 2: # Each turn has 2 messages
                        # Preserve intro message if it's the first one and from assistant
                        intro_message_present = False
                        if dialogue_history_for_response_update and dialogue_history_for_response_update[0].get("role") == "assistant":
                            # This is a simple check; could be more robust to identify a true intro
                            intro_message_present = True

                        num_messages_to_remove = len(dialogue_history_for_response_update) - (MAX_HISTORY_TURNS * 2)
                        if intro_message_present:
                            # Keep the first message (intro), remove from the second message onwards
                            del dialogue_history_for_response_update[1 : num_messages_to_remove + 1]
                        else:
                            del dialogue_history_for_response_update[0 : num_messages_to_remove]
                        logger.info(f"Capped dialogue history to {MAX_HISTORY_TURNS * 2} messages (approx {MAX_HISTORY_TURNS} turns).")

                    st.session_state.encrypted_dialogue_history = encrypt_data(dialogue_history_for_response_update)
                    # Placeholder for saving assistant response to memory
                    logger.info(f"Placeholder: Add call to core.memory.save_dialogue_history_event('assistant', '{sophia_response_text.strip()}') here.")

                except Exception as e_generate_response_gui:
                    error_text_gui = f"Error during response generation: {str(e_generate_response_gui)}"
                    response_display_placeholder.error(error_text_gui)
                    dialogue_history_for_error_update = decrypt_data(st.session_state.encrypted_dialogue_history)
                    if dialogue_history_for_error_update is None: dialogue_history_for_error_update = []
                    dialogue_history_for_error_update.append({"role": "assistant", "content": error_text_gui}) # Add error as assistant message
                    # History capping applies here too
                    if len(dialogue_history_for_error_update) > MAX_HISTORY_TURNS * 2:
                        intro_message_present_err = False
                        if dialogue_history_for_error_update and dialogue_history_for_error_update[0].get("role") == "assistant":
                            intro_message_present_err = True
                        num_messages_to_remove_err = len(dialogue_history_for_error_update) - (MAX_HISTORY_TURNS * 2)
                        if intro_message_present_err:
                             del dialogue_history_for_error_update[1 : num_messages_to_remove_err + 1]
                        else:
                            del dialogue_history_for_error_update[0 : num_messages_to_remove_err]
                    st.session_state.encrypted_dialogue_history = encrypt_data(dialogue_history_for_error_update)
                    # Placeholder for saving error event to memory
                    logger.info(f"Placeholder: Add call to core.memory.save_dialogue_history_event('assistant_error', '{error_text_gui}') here.")
                    if not getattr(sys, '_IS_TEST_RUNNING', False): logger.exception("Error during response generation:")
            st.rerun() # Rerun to update the chat display immediately.

# --- GUI Entry Point ---
def start_gui():
    """
    Initializes and renders the Streamlit Graphical User Interface (GUI).
    
    This function is the main entry point for launching the GUI. It ensures
    that the session state is initialized (though `initialize_session_state`
    is called at module import/load time by Streamlit) and then calls
    `render_main_interface` to draw the UI components.
    """
    # initialize_session_state() is called when the script is first run by Streamlit.
    # Subsequent interactions re-run the script but session state persists.
    # No need to call it again here unless a forced re-initialization is desired under specific conditions.
    
    if config and getattr(config, 'VERBOSE_OUTPUT', False) and not getattr(sys, '_IS_TEST_RUNNING', False):
        logger.info("Starting Streamlit interface via start_gui().")
    
    render_main_interface()

# --- Main Execution Block ---
if __name__ == "__main__":
    if not getattr(sys, '_IS_TEST_RUNNING', False): # Avoid printing this during unit tests
        logger.info("Running in direct mode. For full Streamlit app execution, use 'streamlit run core/gui.py'")
    
    start_gui()
    
    # Display any error messages that might have been set during startup or session init
    # This check is useful if start_gui() or initialize_session_state() sets an error
    # that needs to be displayed prominently if the rest of the UI fails to render.
    # However, render_main_interface() already has logic to display st.session_state.error_message.
    # This can be a final catch-all.
    if 'error_message' in st.session_state and st.session_state.error_message and st.session_state.get('authenticated'):
        # Only show startup error if authenticated, to avoid overlap with login UI
        if 'startup_error_displayed' not in st.session_state: # Avoid duplicate display if already shown
            st.error(f"Startup Error: {st.session_state.error_message}") # This is a UI error, keep st.error
            st.session_state.startup_error_displayed = True


    if not getattr(sys, '_IS_TEST_RUNNING', False):
        logger.info("GUI.py execution finished. If the Streamlit server is running, the app should be visible in your browser.")
```

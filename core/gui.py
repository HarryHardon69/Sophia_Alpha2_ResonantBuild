"""
core/gui.py

Provides a Streamlit-based Graphical User Interface for interacting with
Sophia_Alpha2.
"""

import json
import os
import sys
import time # Used for response streaming simulation
import traceback
# datetime is not directly used but useful for timestamped logging if added later.
# import datetime 

import streamlit as st

# --- Configuration Import ---
try:
    from .. import config
except ImportError:
    print("GUI.py: Could not import 'config' from parent package. Attempting for standalone use.", file=sys.stderr)
    try:
        import config as app_config # Use an alias
        config = app_config
        print("GUI.py: Successfully imported 'config' directly.", file=sys.stderr)
    except ImportError:
        print("GUI.py: Failed to import 'config'. Some features may be limited or use defaults.", file=sys.stderr)
        config = None # Placeholder

# --- Core dialogue module imports ---
_DIALOGUE_AVAILABLE = False
_PERSONA_AVAILABLE = False 
_GET_DIALOGUE_PERSONA_AVAILABLE = False 

try:
    from ..dialogue import generate_response, get_dialogue_persona
    from ..persona import Persona 
    _DIALOGUE_AVAILABLE = True 
    _GET_DIALOGUE_PERSONA_AVAILABLE = True 
    _PERSONA_AVAILABLE = True 
    if not getattr(sys, '_IS_TEST_RUNNING', False):
        print("GUI.py: Successfully imported core.dialogue and core.persona components.")
except ImportError as e_dialogue_import:
    if not getattr(sys, '_IS_TEST_RUNNING', False):
        print(f"GUI.py: Error importing from core.dialogue or core.persona: {e_dialogue_import}. GUI functionality will be significantly limited.", file=sys.stderr)
    
    def generate_response(user_input, stream_thought_steps=False):
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
        print(f"GUI.py: Streamlit page config failed. Are you running this as a Streamlit app? Error: {e_st_config}", file=sys.stderr)

# --- Session State Initialization ---
def initialize_session_state():
    if 'dialogue_history' not in st.session_state:
        st.session_state.dialogue_history = []
    if 'persona_instance' not in st.session_state:
        st.session_state.persona_instance = None 
    if 'stream_thoughts_gui' not in st.session_state:
        default_stream_thoughts = getattr(config, 'VERBOSE_OUTPUT', False) if config else False
        st.session_state.stream_thoughts_gui = default_stream_thoughts
    if 'last_thought_steps' not in st.session_state:
        st.session_state.last_thought_steps = []
    if 'last_awareness_metrics' not in st.session_state:
        st.session_state.last_awareness_metrics = {}
    if 'error_message' not in st.session_state:
        st.session_state.error_message = None # Stores any critical error message for display.

    # Attempt to initialize or retrieve the Persona instance if it's not already in session state.
    if st.session_state.persona_instance is None:
        if _GET_DIALOGUE_PERSONA_AVAILABLE and _PERSONA_AVAILABLE: # Check if dependencies are met.
            try:
                # get_dialogue_persona() should handle its own singleton logic.
                st.session_state.persona_instance = get_dialogue_persona()
                if st.session_state.persona_instance is None: # If get_dialogue_persona() fails.
                    st.session_state.error_message = "Critical: Failed to initialize Persona instance via get_dialogue_persona()."
                    if not getattr(sys, '_IS_TEST_RUNNING', False): print(f"GUI.py Initialize: {st.session_state.error_message}", file=sys.stderr)
                else: # Successfully got persona instance.
                    if not getattr(sys, '_IS_TEST_RUNNING', False): print(f"GUI.py Initialize: Persona instance '{st.session_state.persona_instance.name}' loaded/retrieved into session state.")
            except Exception as e_get_persona_init: # Catch any unexpected error.
                st.session_state.error_message = f"Critical error during get_dialogue_persona(): {str(e_get_persona_init)}"
                if not getattr(sys, '_IS_TEST_RUNNING', False):
                    print(f"GUI.py Initialize: {st.session_state.error_message}", file=sys.stderr)
                    traceback.print_exc() # Log full traceback for debugging.
        elif not _GET_DIALOGUE_PERSONA_AVAILABLE or not _PERSONA_AVAILABLE : # If core dependencies are missing.
            st.session_state.error_message = "Core dialogue.py or persona.py components are not available. GUI functionality will be severely limited."
            if not getattr(sys, '_IS_TEST_RUNNING', False): print(f"GUI.py Initialize: {st.session_state.error_message}", file=sys.stderr)
            # Use the mock persona defined at the top of the file if Persona class itself was not available.
            if st.session_state.persona_instance is None:
                 st.session_state.persona_instance = get_dialogue_persona() # This will return the MockPersonaSingleton

# Call initialization at script load (Streamlit re-runs the script on interaction).
initialize_session_state()

# --- GUI Rendering Logic ---
def render_main_interface():
    """
    Renders the main Streamlit interface, including sidebar and chat elements.
    
    This function orchestrates the display of persona information, awareness metrics,
    dialogue history, user input handling, response generation, and thought streaming.
    It relies on `st.session_state` for maintaining state across interactions.
    """
    # --- Initial Error Display ---
    # Display any critical error message that might have occurred during startup/initialization.
    # `startup_error_displayed` flag ensures it's shown only once per session if the error persists.
    if st.session_state.error_message and 'startup_error_displayed' not in st.session_state:
        st.error(st.session_state.error_message)
        st.session_state.startup_error_displayed = True 
    
    # --- Persona Instance Handling ---
    # Retrieve the persona instance from session state.
    persona = st.session_state.persona_instance
    if persona is None: # If persona is still None after initialization attempts.
        if not st.session_state.error_message: # Show a generic error if no specific one was set.
            st.error("Fatal Error: Persona instance is unavailable. The GUI cannot operate.")
        return # Halt further rendering if persona is essential and missing.

    # --- Initial Message / Persona Introduction ---
    # If dialogue history is empty, display the persona's introduction message.
    if not st.session_state.dialogue_history and hasattr(persona, 'get_intro'):
        try:
            intro_message = persona.get_intro()
            st.session_state.dialogue_history.append({"role": "assistant", "content": intro_message})
        except Exception as e_intro_render: # Handle errors getting intro.
            error_intro_msg = f"Error displaying persona introduction: {str(e_intro_render)}"
            st.session_state.dialogue_history.append({"role": "assistant", "content": error_intro_msg})
            if not getattr(sys, '_IS_TEST_RUNNING', False): traceback.print_exc() # Log for debugging.

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
    if st.session_state.last_awareness_metrics:
        st.sidebar.subheader("Last Interaction Snapshot")
        metrics_snapshot = st.session_state.last_awareness_metrics
        curiosity_metric, coherence_metric = metrics_snapshot.get('curiosity', 0.0), metrics_snapshot.get('coherence', 0.0)
        context_stability_metric, self_evolution_rate_metric = metrics_snapshot.get('context_stability', 0.0), metrics_snapshot.get('self_evolution_rate', 0.0)
        col1_sidebar, col2_sidebar = st.sidebar.columns(2) # Use columns for compact display.
        col1_sidebar.metric("Curiosity", f"{curiosity_metric:.2f}")
        col2_sidebar.metric("Coherence", f"{coherence_metric:.2f}")
        col1_sidebar.metric("Context Stability", f"{context_stability_metric:.2f}")
        col2_sidebar.metric("Self Evolution", f"{self_evolution_rate_metric:.2f}")
        st.sidebar.text(f"LLM Fallback Used: {metrics_snapshot.get('active_llm_fallback', 'N/A')}")
        pcc_display_sidebar = str(metrics_snapshot.get('primary_concept_coord', 'N/A'))
        st.sidebar.text(f"Primary Concept Coords: {pcc_display_sidebar}")
    st.sidebar.markdown("---")

    # Settings controls in the sidebar.
    st.sidebar.subheader("Settings & Actions")
    st.checkbox("Show Thought Stream Expander", key="stream_thoughts_gui", help="Toggle visibility of Sophia's detailed thought process for the last response.")
    
    if st.sidebar.button("Clear Dialogue History"):
        st.session_state.dialogue_history = [] # Clear history.
        st.session_state.last_thought_steps = [] # Clear last thoughts.
        st.session_state.last_awareness_metrics = {} # Clear last metrics.
        # Re-add persona intro if available.
        if persona and hasattr(persona, 'get_intro'):
            try: st.session_state.dialogue_history.append({"role": "assistant", "content": persona.get_intro()})
            except Exception as e_reinit_intro: st.session_state.dialogue_history.append({"role": "assistant", "content": f"Error re-initializing intro: {e_reinit_intro}"})
        st.rerun() # Rerun Streamlit script to reflect changes.

    if st.sidebar.button("Reset Persona State"):
        if persona and hasattr(persona, '_initialize_default_state_and_save'):
            try:
                persona._initialize_default_state_and_save() # Call persona's reset method.
                # Clear session state related to dialogue and previous interaction.
                st.session_state.dialogue_history, st.session_state.last_thought_steps, st.session_state.last_awareness_metrics = [], [], {}
                if hasattr(persona, 'get_intro'): st.session_state.dialogue_history.append({"role": "assistant", "content": persona.get_intro()})
                st.sidebar.success("Persona state has been reset to defaults.")
            except Exception as e_reset_persona: st.sidebar.error(f"Error resetting persona state: {e_reset_persona}")
        else: st.sidebar.warning("Persona instance unavailable or does not support state reset.")
        st.rerun()

    # --- Main Chat Interface Rendering ---
    persona_display_name = getattr(persona, 'name', 'Sophia') if persona else 'Sophia_Unavailable'
    st.title(f"Chat with {persona_display_name}")

    # Display chat messages from dialogue history.
    for message_item in st.session_state.dialogue_history:
        role_for_msg, avatar_char_for_msg = ("assistant", "🧠") if message_item.get("role") == "assistant" else ("user", "👤")
        message_sender_name = persona_display_name if role_for_msg == "assistant" else "You"
        with st.chat_message(name=message_sender_name, avatar=avatar_char_for_msg):
            st.markdown(message_item["content"]) # Render message content as Markdown.

    # Expander for displaying Sophia's thought process if toggled.
    if st.session_state.get("stream_thoughts_gui", False) and st.session_state.get("last_thought_steps"):
        with st.expander("Sophia's Thought Stream (Last Response)", expanded=False):
            thought_steps_formatted_text = "".join([f"- {step}\n" for step in st.session_state.last_thought_steps])
            st.text_area("Detailed Thoughts:", value=thought_steps_formatted_text, height=200, disabled=True, key="thought_stream_display_area")

    # Chat input field for user query.
    user_query_input = st.chat_input(f"Ask {persona_display_name}...")
    if user_query_input: # If user enters a query.
        st.session_state.dialogue_history.append({"role": "user", "content": user_query_input}) # Add user query to history.
        
        # Display "Thinking..." placeholder and then generate/stream response.
        with st.chat_message(name=persona_display_name, avatar="🧠"):
            response_display_placeholder = st.empty() # Placeholder for streaming response.
            response_display_placeholder.markdown("Thinking...") 
            try:
                # Call dialogue module's generate_response function.
                if not _DIALOGUE_AVAILABLE: # Fallback if dialogue module is not available.
                    sophia_response_text, temp_thought_steps, temp_awareness_metrics = "Error: Dialogue module not available. Cannot process input.", ["Dialogue module unavailable."], {"error": "Dialogue module unavailable"}
                else:
                    # stream_thought_steps=False here as detailed thoughts are shown in expander, not inline with response.
                    sophia_response_text, temp_thought_steps, temp_awareness_metrics = generate_response(user_query_input, stream_thought_steps=False)
                
                # Update session state with the latest thoughts and metrics.
                st.session_state.last_thought_steps = temp_thought_steps
                st.session_state.last_awareness_metrics = temp_awareness_metrics
                
                # Simulate streaming of Sophia's response word by word.
                full_response_for_streaming = ""
                response_words = sophia_response_text.split()
                if not response_words: # Handle empty response string.
                    response_display_placeholder.markdown(sophia_response_text.strip())
                else:
                    for word_chunk in response_words:
                        full_response_for_streaming += word_chunk + " "
                        response_display_placeholder.markdown(full_response_for_streaming + "▌") # Use a cursor-like char.
                        time.sleep(0.05) # Short delay to simulate typing.
                    response_display_placeholder.markdown(full_response_for_streaming.strip()) # Final response without cursor.
                
                # Add final assistant response to dialogue history.
                st.session_state.dialogue_history.append({"role": "assistant", "content": sophia_response_text.strip()})
            except Exception as e_generate_response_gui: # Handle errors during response generation.
                error_text_gui = f"Error during response generation: {str(e_generate_response_gui)}"
                response_display_placeholder.error(error_text_gui)
                st.session_state.dialogue_history.append({"role": "assistant", "content": error_text_gui})
                if not getattr(sys, '_IS_TEST_RUNNING', False): traceback.print_exc() # Log full error for debugging.
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
        print("GUI: Starting Streamlit interface via start_gui().")
    
    render_main_interface()

# --- Main Execution Block ---
if __name__ == "__main__":
    if not getattr(sys, '_IS_TEST_RUNNING', False): # Avoid printing this during unit tests
        print("GUI.py: Running in direct mode. For full Streamlit app execution, use 'streamlit run core/gui.py'")
    
    start_gui()
    
    # Display any error messages that might have been set during startup or session init
    # This check is useful if start_gui() or initialize_session_state() sets an error
    # that needs to be displayed prominently if the rest of the UI fails to render.
    # However, render_main_interface() already has logic to display st.session_state.error_message.
    # This can be a final catch-all.
    if 'error_message' in st.session_state and st.session_state.error_message:
        if 'startup_error_displayed' not in st.session_state: # Avoid duplicate display if already shown
            st.error(f"Startup Error: {st.session_state.error_message}")

    if not getattr(sys, '_IS_TEST_RUNNING', False):
        print("\nGUI.py execution finished. If the Streamlit server is running, the app should be visible in your browser.")
```

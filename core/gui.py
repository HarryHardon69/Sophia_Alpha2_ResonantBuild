"""
core/gui.py

Provides a Streamlit-based Graphical User Interface for interacting with
Sophia_Alpha2.
"""

import streamlit as st
import sys
import os
import time
import json
import traceback

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
        st.session_state.error_message = None

    if st.session_state.persona_instance is None:
        if _GET_DIALOGUE_PERSONA_AVAILABLE and _PERSONA_AVAILABLE: 
            try:
                st.session_state.persona_instance = get_dialogue_persona()
                if st.session_state.persona_instance is None:
                    st.session_state.error_message = "Critical: Failed to initialize Persona instance from dialogue module."
                    if not getattr(sys, '_IS_TEST_RUNNING', False): print(f"GUI.py: {st.session_state.error_message}", file=sys.stderr)
                else:
                    if not getattr(sys, '_IS_TEST_RUNNING', False): print(f"GUI.py: Persona instance '{st.session_state.persona_instance.name}' loaded into session state.")
            except Exception as e_get_persona:
                st.session_state.error_message = f"Critical error getting persona: {str(e_get_persona)}"
                if not getattr(sys, '_IS_TEST_RUNNING', False):
                    print(f"GUI.py: {st.session_state.error_message}", file=sys.stderr)
                    traceback.print_exc()
        elif not _GET_DIALOGUE_PERSONA_AVAILABLE or not _PERSONA_AVAILABLE : 
            st.session_state.error_message = "Core dialogue or persona modules are not available. GUI cannot function fully."
            if not getattr(sys, '_IS_TEST_RUNNING', False): print(f"GUI.py: {st.session_state.error_message}", file=sys.stderr)
            if st.session_state.persona_instance is None:
                 st.session_state.persona_instance = get_dialogue_persona() 

initialize_session_state()

# --- GUI Rendering Logic ---
def render_main_interface():
    # Initial Error Checks
    if st.session_state.error_message and 'startup_error_displayed' not in st.session_state:
        st.error(st.session_state.error_message)
        st.session_state.startup_error_displayed = True # Ensure it's shown only once if it persists
    
    persona = st.session_state.persona_instance
    if persona is None:
        # Error message for persona None is already handled by initialize_session_state if it's a startup issue
        # This is an additional safeguard or for cases where it becomes None later (though less likely with current flow)
        if not st.session_state.error_message: # If no specific startup error, show a generic one
            st.error("Persona instance is not available. Cannot render full UI.")
        return

    # Initial Message Logic
    if not st.session_state.dialogue_history and hasattr(persona, 'get_intro'):
        try:
            intro_message = persona.get_intro()
            st.session_state.dialogue_history.append({"role": "assistant", "content": intro_message})
        except Exception as e_intro:
            error_intro = f"Error getting persona intro: {str(e_intro)}"
            st.session_state.dialogue_history.append({"role": "assistant", "content": error_intro})
            if not getattr(sys, '_IS_TEST_RUNNING', False): traceback.print_exc()

    # --- Sidebar Implementation ---
    st.sidebar.title("Controls & Awareness")
    if persona:
        st.sidebar.subheader(f"Persona: {getattr(persona, 'name', 'N/A')}")
        st.sidebar.caption(f"Mode: {getattr(persona, 'mode', 'N/A')}")
        traits_list = getattr(persona, 'traits', [])
        if isinstance(traits_list, list): st.sidebar.caption(f"Traits: {', '.join(traits_list)}")
        else: st.sidebar.caption(f"Traits: {str(traits_list)}")
        st.sidebar.markdown("---")
        st.sidebar.subheader("Overall Awareness")
        awareness_data = getattr(persona, 'awareness', {})
        if isinstance(awareness_data, dict):
            for key, value in awareness_data.items():
                display_key = key.replace('_', ' ').title()
                if isinstance(value, float): st.sidebar.text(f"{display_key}: {value:.2f}")
                elif isinstance(value, (list, tuple)): st.sidebar.text(f"{display_key}: {str(value)}")
                else: st.sidebar.text(f"{display_key}: {value}")
        else: st.sidebar.text("Awareness data is not in the expected format.")
    else: st.sidebar.warning("Persona instance not loaded. Awareness data unavailable.")
    st.sidebar.markdown("---")
    if st.session_state.last_awareness_metrics:
        st.sidebar.subheader("Last Interaction Snapshot")
        metrics = st.session_state.last_awareness_metrics
        curiosity, coherence = metrics.get('curiosity', 0.0), metrics.get('coherence', 0.0)
        context_stability, self_evolution_rate = metrics.get('context_stability', 0.0), metrics.get('self_evolution_rate', 0.0)
        col1, col2 = st.sidebar.columns(2)
        col1.metric("Curiosity", f"{curiosity:.2f}"); col2.metric("Coherence", f"{coherence:.2f}")
        col1.metric("Context Stability", f"{context_stability:.2f}"); col2.metric("Self Evolution", f"{self_evolution_rate:.2f}")
        st.sidebar.text(f"LLM Fallback: {metrics.get('active_llm_fallback', 'N/A')}")
        pcc_display = str(metrics.get('primary_concept_coord', 'N/A'))
        st.sidebar.text(f"Primary Concept: {pcc_display}")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Settings")
    st.checkbox("Show Thought Stream Expander", key="stream_thoughts_gui", help="Toggle visibility of Sophia's thought process for the last response.")
    if st.sidebar.button("Clear Dialogue History"):
        st.session_state.dialogue_history = []
        st.session_state.last_thought_steps = []
        st.session_state.last_awareness_metrics = {}
        if persona and hasattr(persona, 'get_intro'):
            try: st.session_state.dialogue_history.append({"role": "assistant", "content": persona.get_intro()})
            except Exception as e: st.session_state.dialogue_history.append({"role": "assistant", "content": f"Error re-initializing: {e}"})
        st.rerun()
    if st.sidebar.button("Reset Persona State"):
        if persona and hasattr(persona, '_initialize_default_state_and_save'):
            try:
                persona._initialize_default_state_and_save()
                st.session_state.dialogue_history, st.session_state.last_thought_steps, st.session_state.last_awareness_metrics = [], [], {}
                if hasattr(persona, 'get_intro'): st.session_state.dialogue_history.append({"role": "assistant", "content": persona.get_intro()})
                st.sidebar.success("Persona state reset to defaults.")
            except Exception as e: st.sidebar.error(f"Error resetting persona: {e}")
        else: st.sidebar.warning("Persona instance not available or does not support reset.")
        st.rerun()

    # --- Main Chat Interface ---
    persona_name = getattr(persona, 'name', 'Sophia') if persona else 'Sophia'
    st.title(f"Chat with {persona_name}")
    for message in st.session_state.dialogue_history:
        role, avatar_char = ("assistant", "🧠") if message.get("role") == "assistant" else ("user", "👤")
        message_name = persona_name if role == "assistant" else "You"
        with st.chat_message(name=message_name, avatar=avatar_char): st.markdown(message["content"])
    if st.session_state.get("stream_thoughts_gui", False) and st.session_state.get("last_thought_steps"):
        with st.expander("Sophia's Thought Stream (Last Response)", expanded=False):
            thought_text = "".join([f"- {step}\n" for step in st.session_state.last_thought_steps])
            st.text_area("", thought_text, height=200, disabled=True, key="thought_stream_display")
    user_query = st.chat_input(f"Ask {persona_name}...")
    if user_query:
        st.session_state.dialogue_history.append({"role": "user", "content": user_query})
        with st.chat_message(name=persona_name, avatar="🧠"):
            response_placeholder = st.empty()
            response_placeholder.markdown("Thinking...") 
            try:
                if not _DIALOGUE_AVAILABLE:
                    sophia_response_text, temp_thought_steps, temp_awareness_metrics = "Error: Dialogue module not available.", ["Dialogue module unavailable."], {"error": "Dialogue module unavailable"}
                else:
                    sophia_response_text, temp_thought_steps, temp_awareness_metrics = generate_response(user_query, stream_thought_steps=False)
                st.session_state.last_thought_steps, st.session_state.last_awareness_metrics = temp_thought_steps, temp_awareness_metrics
                full_response_streamed = ""
                words = sophia_response_text.split()
                if not words: response_placeholder.markdown(sophia_response_text.strip())
                else:
                    for chunk in words:
                        full_response_streamed += chunk + " "
                        response_placeholder.markdown(full_response_streamed + "▌")
                        time.sleep(0.05) 
                    response_placeholder.markdown(full_response_streamed.strip())
                st.session_state.dialogue_history.append({"role": "assistant", "content": sophia_response_text.strip()})
            except Exception as e_gen_resp:
                error_text = f"Error generating response: {str(e_gen_resp)}"
                response_placeholder.error(error_text)
                st.session_state.dialogue_history.append({"role": "assistant", "content": error_text})
                if not getattr(sys, '_IS_TEST_RUNNING', False): traceback.print_exc()
        st.rerun()

# --- GUI Entry Point ---
def start_gui():
    """
    Initializes and renders the Streamlit GUI.
    This function serves as the main entry point for launching the GUI.
    """
    # initialize_session_state() is already called globally when the script is imported/run.
    # So, it doesn't need to be called again here unless there's a specific reason for re-initialization
    # in a multi-page app context, which is not the case here.
    
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

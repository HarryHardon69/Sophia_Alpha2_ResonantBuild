# Sophia_Alpha2_ResonantBuild

## Vision
(Placeholder for Vision Statement - To be populated based on Phase2KB.md)

## Key Features
(Placeholder for Key Features - To be populated based on Phase2KB.md)
*   Modular core components (configurable, cognitive processing, memory, persona, knowledge library, dialogue management).
*   SNN-based cognitive architecture with LLM bootstrapping.
*   Persistent knowledge graph with novelty and ethical filtering.
*   Persistent Knowledge Library for curated information with ethical scoring.
*   Ethical Mitigation (`Mitigator` class) for content moderation.
*   Dialogue Management / Interaction Orchestration via CLI.
*   Streamlit-based GUI for interactive chat and system monitoring.

## System Architecture
(Placeholder for System Architecture Overview - To be populated based on Phase2KB.md and further design)

The `core.brain` module is a central component, housing the `SpacetimeManifold` class. This class implements a Spiking Neural Network (SNN) for cognitive processing, bootstraps concepts using Large Language Models (LLMs), incorporates STDP/Hebbian learning, and generates awareness metrics.

The `core.memory` module manages Sophia_Alpha2's knowledge graph, stored in `data/memory_store/knowledge_graph.json`. It handles storing new concepts based on novelty and ethical alignment, calculates concept novelty, and provides various functions for retrieving memories.

The `core.persona` module manages Sophia_Alpha2's identity, traits, operational mode, and evolving awareness state. Handles the persistence of this state to `persona_profile.json` (path typically defined in `config.PERSONA_PROFILE_PATH`).

The `core.library` module provides shared utility functions, ethical mitigation (`Mitigator` class), and manages the curated knowledge base (`library_log.json`), including ingestion with coordinate assignment and ethical scoring.

The `core.dialogue` module is the central interaction handler for Sophia_Alpha2. It orchestrates calls to the brain, persona, ethics, memory, and library modules to generate responses and manage the dialogue flow. Includes the main CLI loop for user interaction.

The `core.gui` module provides a Streamlit-based Graphical User Interface (GUI) for interacting with Sophia_Alpha2, visualizing thought processes, and displaying awareness metrics.

All core modules are designed to be extensively configurable via `config/config.py`.

## Directory Structure
```
Sophia_Alpha2_ResonantBuild/
├── .gitignore
├── README.md
├── config/
│   ├── __init__.py
│   └── config.py      # System-wide configuration settings
├── core/
│   ├── __init__.py    # Core package initializer, exposes key functions
│   ├── brain.py       # Cognitive core: SpacetimeManifold (SNN), LLM bootstrap, learning, awareness
│   ├── dialogue.py    # Orchestrates dialogue flow, CLI, and calls to other core modules
│   ├── ethics.py      # (Stub) For ethical framework and decision-making
│   ├── gui.py         # Streamlit-based GUI for interaction and visualization
│   ├── library.py     # Manages curated knowledge base, utilities, and ethical mitigation
│   ├── memory.py      # Manages the knowledge graph (storing, retrieving, novelty calculation)
│   └── persona.py     # Manages Sophia_Alpha2's identity, traits, mode, and awareness state
├── data/
│   ├── ethics_store/
│   │   └── .gitkeep
│   ├── library_store/ # Contains library_log.json (curated knowledge)
│   │   └── .gitkeep
│   ├── logs/
│   │   └── .gitkeep
│   ├── memory_store/  # Contains knowledge_graph.json
│   │   └── .gitkeep
│   ├── personas/
│   │   └── .gitkeep
│   ├── private/
│   │   └── .gitkeep
│   └── public/
│       └── .gitkeep
├── docs/
│   ├── Phase2KB.md      # Project knowledge base and evolving specifications
│   └── documentation.md # Detailed technical documentation
├── interface/
│   └── __init__.py
├── main.py
├── requirements.txt
└── tests/
    ├── __init__.py
    └── .gitkeep
```

## Getting Started
(Placeholder for Getting Started Instructions - To be populated later)
1.  Prerequisites
2.  Installation
3.  Running the Application:
    *   **CLI Mode**: (Instructions to be added, likely via `python main.py --cli` or `python -m core.dialogue`)
    *   **GUI Mode**: To start the Streamlit GUI, navigate to the project root directory and run:
        ```bash
        streamlit run core/gui.py
        ```
        (Alternatively, if `main.py` is updated to support it: `python main.py --gui`)


## Roadmap
(Placeholder for Project Roadmap - To be populated later)
*   Phase 1: Initial Scaffolding (Complete)
*   Phase 2: Core module implementation (config.py complete, brain.py complete, memory.py complete, persona.py updated, library.py implemented, dialogue.py implemented, gui.py implemented)
*   Phase 3: Implementation of ethics module.
*   Phase 4: ...

## Contributing
(Placeholder for Contribution Guidelines - To be populated later)

## License
(Placeholder for License Information - To be determined and added later)

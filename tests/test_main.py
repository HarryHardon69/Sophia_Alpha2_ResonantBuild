"""
Unit tests for main.py functionality.
"""

import unittest
from unittest import mock
import argparse
import sys

# Ensure the main module can be imported.
# This might require adjusting sys.path if tests are run from a different working directory
# or if the project structure isn't standard. For now, assume direct import works
# if tests are run from the project root (e.g., python -m unittest tests.test_main).
try:
    from .. import main as main_module  # If main.py is one level up from tests/
except ImportError:
    # Fallback if running tests/test_main.py directly or tests/ is not a package
    # This assumes main.py is in the parent directory of tests/
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import main as main_module


class TestMainArgumentParser(unittest.TestCase):
    """Tests for the argument parsing setup in main.py."""

    def test_interface_cli(self):
        """Test parsing of --interface cli."""
        args = main_module.parser.parse_args(['--interface', 'cli'])
        self.assertEqual(args.interface, 'cli')

    def test_interface_gui(self):
        """Test parsing of --interface gui."""
        args = main_module.parser.parse_args(['--interface', 'gui'])
        self.assertEqual(args.interface, 'gui')

    def test_verbose_short(self):
        """Test parsing of -v for verbose."""
        args = main_module.parser.parse_args(['-v'])
        self.assertTrue(args.verbose)

    def test_verbose_long(self):
        """Test parsing of --verbose for verbose."""
        args = main_module.parser.parse_args(['--verbose'])
        self.assertTrue(args.verbose)

    def test_query(self):
        """Test parsing of --query with a value."""
        test_query_string = "Hello, Sophia."
        args = main_module.parser.parse_args(['--query', test_query_string])
        self.assertEqual(args.query, test_query_string)

    def test_all_args(self):
        """Test parsing with a combination of arguments."""
        test_query_string = "Test query"
        args = main_module.parser.parse_args([
            '--interface', 'cli',
            '--query', test_query_string,
            '--verbose'
        ])
        self.assertEqual(args.interface, 'cli')
        self.assertEqual(args.query, test_query_string)
        self.assertTrue(args.verbose)

    def test_no_args(self):
        """Test parsing with no arguments (defaults)."""
        args = main_module.parser.parse_args([])
        self.assertIsNone(args.interface) # Default is None, determined later
        self.assertIsNone(args.query)
        self.assertFalse(args.verbose)


class TestMainInterfaceSelection(unittest.TestCase):
    """Tests for the interface selection logic in main_logic."""

    def setUp(self):
        """Patch 'config' and 'logger' for main_logic, and core modules."""
        # Patch config for all tests in this class
        self.mock_config_patch = mock.patch('main.config')
        self.mock_config = self.mock_config_patch.start()
        self.addCleanup(self.mock_config_patch.stop)

        # Patch logger to suppress log output during tests
        self.mock_logger_patch = mock.patch('main.logger')
        self.mock_logger = self.mock_logger_patch.start()
        self.addCleanup(self.mock_logger_patch.stop)

        # Patch core modules that main_logic tries to import
        self.mock_core_dialogue_patch = mock.patch('main.core_dialogue')
        self.mock_core_dialogue = self.mock_core_dialogue_patch.start()
        self.addCleanup(self.mock_core_dialogue_patch.stop)

        self.mock_core_gui_patch = mock.patch('main.core_gui')
        self.mock_core_gui = self.mock_core_gui_patch.start()
        self.addCleanup(self.mock_core_gui_patch.stop)

        self.mock_core_brain_patch = mock.patch('main.core_brain')
        self.mock_core_brain = self.mock_core_brain_patch.start()
        self.addCleanup(self.mock_core_brain_patch.stop)

        # Mock core package itself for version checking
        self.mock_core_patch = mock.patch('main.core')
        self.mock_core = self.mock_core_patch.start()
        self.addCleanup(self.mock_core_patch.stop)


    def test_cli_interface_arg(self):
        """Test effective_interface is 'cli' when --interface cli is passed."""
        self.mock_config.VERBOSE_OUTPUT = False # Keep test output clean
        self.mock_config.ENABLE_GUI = True # Does not matter for this test
        args = argparse.Namespace(interface='cli', query=None, verbose=False)

        # We are testing logic within main_logic, so we need to mock functions it calls
        # that are not under test or are external dependencies.
        with mock.patch('main.core_dialogue.dialogue_loop') as mock_loop:
            main_module.main_logic(args)
            # The primary check is that dialogue_loop (CLI path) is called.
            # And gui.start_gui is not.
            mock_loop.assert_called_once()
            self.mock_core_gui.start_gui.assert_not_called()

    def test_gui_interface_arg_and_config_true(self):
        """Test effective_interface is 'gui' if --interface gui and config.ENABLE_GUI is True."""
        self.mock_config.VERBOSE_OUTPUT = False
        self.mock_config.ENABLE_GUI = True
        args = argparse.Namespace(interface='gui', query=None, verbose=False)

        with mock.patch('main.core_gui.start_gui') as mock_start_gui:
            main_module.main_logic(args)
            mock_start_gui.assert_called_once()
            self.mock_core_dialogue.dialogue_loop.assert_not_called()

    def test_gui_interface_arg_but_config_false(self):
        """Test effective_interface is 'cli' (fallback) if --interface gui but config.ENABLE_GUI is False."""
        self.mock_config.VERBOSE_OUTPUT = False
        self.mock_config.ENABLE_GUI = False # GUI disabled in config
        args = argparse.Namespace(interface='gui', query=None, verbose=False)

        with mock.patch('main.core_dialogue.dialogue_loop') as mock_loop:
            main_module.main_logic(args)
            mock_loop.assert_called_once() # Should fall back to CLI
            self.mock_core_gui.start_gui.assert_not_called()

    def test_query_forces_cli_mode(self):
        """Test effective_interface is 'cli' if --query is provided, regardless of other settings."""
        self.mock_config.VERBOSE_OUTPUT = False
        self.mock_config.ENABLE_GUI = True # GUI enabled in config
        self.mock_config.MAX_QUERY_LENGTH = 1024 # For query validation
        self.mock_config.ENABLE_CLI_STREAMING = False # Non-streaming for this test
        self.mock_config.DEFAULT_SINGLE_QUERY_STREAM_THOUGHTS = False

        args = argparse.Namespace(interface='gui', query="test query", verbose=False)

        # Mock generate_response as it's called for queries
        self.mock_core_dialogue.generate_response.return_value = ("Response", ["step"], {})

        main_module.main_logic(args)
        self.mock_core_dialogue.generate_response.assert_called_once_with("test query", stream_thought_steps=False)
        self.mock_core_gui.start_gui.assert_not_called() # GUI should not be called
        self.mock_core_dialogue.dialogue_loop.assert_not_called() # Interactive loop should not run

    def test_default_interface_gui_enabled(self):
        """Test default to GUI if no interface arg and config.ENABLE_GUI is True."""
        self.mock_config.VERBOSE_OUTPUT = False
        self.mock_config.ENABLE_GUI = True
        args = argparse.Namespace(interface=None, query=None, verbose=False)

        with mock.patch('main.core_gui.start_gui') as mock_start_gui:
            main_module.main_logic(args)
            mock_start_gui.assert_called_once()

    def test_default_interface_gui_disabled(self):
        """Test default to CLI if no interface arg and config.ENABLE_GUI is False."""
        self.mock_config.VERBOSE_OUTPUT = False
        self.mock_config.ENABLE_GUI = False
        args = argparse.Namespace(interface=None, query=None, verbose=False)

        with mock.patch('main.core_dialogue.dialogue_loop') as mock_loop:
            main_module.main_logic(args)
            mock_loop.assert_called_once()


if __name__ == '__main__':
    unittest.main()

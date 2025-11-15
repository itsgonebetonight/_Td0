"""Unit tests for py4at_app main module."""
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add the current directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from demo_wrapper import script_path, validate_py4at_repository, terminate_processes


class TestConfig(unittest.TestCase):
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        self.assertEqual(config.py4at_repo_name, 'py4at')
        self.assertEqual(config.server_script_path, 'ch07/TickServer.py')
        self.assertEqual(config.strategy_script_path, 'ch07/OnlineAlgorithm.py')
        self.assertEqual(config.server_startup_delay, 1.0)
        self.assertEqual(config.process_monitor_interval, 0.5)
        self.assertEqual(config.process_termination_timeout, 5)
        self.assertEqual(config.default_log_level, 'INFO')
    
    @patch.dict('os.environ', {'PY4AT_SERVER_DELAY': '2.5', 'PY4AT_LOG_LEVEL': 'DEBUG'})
    def test_env_overrides(self):
        """Test environment variable overrides."""
        config = Config()
        self.assertEqual(config.server_startup_delay, 2.5)
        self.assertEqual(config.default_log_level, 'DEBUG')


class TestMainFunctions(unittest.TestCase):
    """Test main module functions."""
    
    def test_script_path(self):
        """Test script path construction (platform independent)."""
        root = Path('/test/root')
        result = script_path(root, 'ch07', 'TickServer.py', as_str=False)
        # Compare by suffix so test is stable across platforms (Windows vs POSIX)
        self.assertTrue(result.as_posix().endswith('test/root/ch07/TickServer.py'))
    
    @patch('demo_wrapper.logger')
    @patch('demo_wrapper.sys.exit')
    def test_validate_py4at_repository_missing_dir(self, mock_exit, mock_logger):
        """Test validation when py4at directory doesn't exist."""
        non_existent_path = Path('/non/existent/path')
        validate_py4at_repository(non_existent_path)
        mock_logger.error.assert_called_once()
        mock_exit.assert_called_once_with(1)
    
    @patch('demo_wrapper.logger')
    @patch('demo_wrapper.sys.exit')
    def test_validate_py4at_repository_missing_script(self, mock_exit, mock_logger):
        """Test validation when script files don't exist."""
        import tempfile
        from pathlib import Path as _P

        with tempfile.TemporaryDirectory() as td:
            tmp_path = _P(td)
            # Create a mock py4at directory but without the required scripts
            py4at_root = tmp_path / 'py4at'
            py4at_root.mkdir()

            validate_py4at_repository(py4at_root)
            mock_logger.error.assert_called()
            mock_exit.assert_called_once_with(1)
    
    @patch('subprocess.Popen')
    @patch('demo_wrapper.logger')
    def test_terminate_processes(self, mock_logger, mock_popen):
        """Test process termination."""
        # Create mock process that's still running
        mock_proc = Mock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345
        mock_proc.wait.return_value = None
        
        terminate_processes([mock_proc])
        
        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once_with(timeout=5)
        mock_logger.info.assert_called_with('Terminating process 12345')


class TestIntegration(unittest.TestCase):
    """Integration tests for the main application."""
    
    @patch('demo_wrapper.validate_py4at_repository')
    @patch('demo_wrapper.run_process')
    @patch('demo_wrapper.monitor_processes')
    @patch('demo_wrapper.setup_signal_handlers')
    def test_main_server_mode(self, mock_signals, mock_monitor, mock_run, mock_validate):
        """Test main function in server mode."""
        import tempfile
        from pathlib import Path as _P

        # Setup mocks
        with tempfile.TemporaryDirectory() as td:
            tmp_path = _P(td)
            mock_validate.return_value = (
                tmp_path / 'TickServer.py',
                tmp_path / 'OnlineAlgorithm.py'
            )
        mock_proc = Mock()
        mock_run.return_value = mock_proc
        
        # Mock argument parsing
        with patch('demo_wrapper.argparse.ArgumentParser.parse_args') as mock_parse:
            mock_args = Mock()
            mock_args.mode = 'server'
            mock_args.verbose = False
            mock_parse.return_value = mock_args
            
            # Import and run main from demo_wrapper
            from demo_wrapper import main
            
            # Mock the path operations
            with patch('demo_wrapper.Path') as mock_path:
                mock_path.return_value.parent.parent = tmp_path
                with patch('demo_wrapper.config.get_py4at_root') as mock_get_root:
                    mock_get_root.return_value = tmp_path

                    try:
                        main()
                    except SystemExit:
                        pass  # Expected due to mock setup
            
            # Verify server was started
            mock_run.assert_called_once()
            mock_monitor.assert_called_once()


if __name__ == '__main__':
    unittest.main()

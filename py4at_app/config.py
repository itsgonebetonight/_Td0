"""Configuration management for py4at_app."""
import os
from pathlib import Path
from typing import Dict, Any

class Config:
    """Configuration settings for the py4at application."""
    
    def __init__(self):
        # Default paths
        self.py4at_repo_name = 'py4at'
        self.server_script_path = 'ch07/TickServer.py'
        self.strategy_script_path = 'ch07/OnlineAlgorithm.py'
        
        # Process settings
        self.server_startup_delay = 1.0  # seconds
        self.process_monitor_interval = 0.5  # seconds
        self.process_termination_timeout = 5  # seconds
        
        # Logging settings
        self.default_log_level = 'INFO'
        self.log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Load environment overrides
        self._load_env_overrides()
    
    def _load_env_overrides(self) -> None:
        """Load configuration from environment variables."""
        env_mappings = {
            'PY4AT_REPO_PATH': 'py4at_repo_path',
            'PY4AT_SERVER_DELAY': 'server_startup_delay',
            'PY4AT_MONITOR_INTERVAL': 'process_monitor_interval',
            'PY4AT_TERMINATION_TIMEOUT': 'process_termination_timeout',
            'PY4AT_LOG_LEVEL': 'default_log_level',
        }
        
        for env_var, attr_name in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if attr_name in ['server_startup_delay', 'process_monitor_interval', 'process_termination_timeout']:
                    setattr(self, attr_name, float(value))
                else:
                    setattr(self, attr_name, value)
    
    def get_py4at_root(self, workspace_root: Path) -> Path:
        """Get the py4at repository path."""
        # Check for custom path from environment
        custom_path = getattr(self, 'py4at_repo_path', None)
        if custom_path:
            return Path(custom_path)
        
        # Default to sibling directory
        return workspace_root / self.py4at_repo_name
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not attr.startswith('_') and not callable(getattr(self, attr))
        }

# Global configuration instance
config = Config()

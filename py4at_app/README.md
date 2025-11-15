# py4at Local Demo Wrapper

An improved CLI wrapper that runs the TickServer and OnlineAlgorithm from the `py4at` repo
as subprocesses with enhanced logging, error handling, and process management.

## Features

- **Enhanced Logging**: Structured logging with configurable levels and timestamps
- **Graceful Process Management**: Proper cleanup with signal handling and timeouts
- **Configuration Management**: Environment-based configuration overrides
- **Type Safety**: Full type hints for better code maintainability
- **Error Handling**: Comprehensive error handling with informative messages
- **Process Monitoring**: Real-time process output monitoring and status tracking

## Quick Start (PowerShell)

```powershell
cd <workspace-root>\py4at_app

# Install dependencies
pip install -r requirements.txt

# Run server only
python main.py server

# Run strategy only (connects to server at tcp://0.0.0.0:5555)
python main.py strategy

# Run both (server then strategy)
python main.py both

# Enable verbose logging
python main.py both --verbose
```

## Configuration

The application supports configuration via environment variables:

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `PY4AT_REPO_PATH` | Custom path to py4at repository | `../py4at` |
| `PY4AT_SERVER_DELAY` | Server startup delay in seconds | `1.0` |
| `PY4AT_MONITOR_INTERVAL` | Process monitoring interval in seconds | `0.5` |
| `PY4AT_TERMINATION_TIMEOUT` | Process termination timeout in seconds | `5` |
| `PY4AT_LOG_LEVEL` | Default logging level | `INFO` |

Example with custom configuration:
```powershell
$env:PY4AT_SERVER_DELAY = "2.0"
$env:PY4AT_LOG_LEVEL = "DEBUG"
python main.py both
```

## Testing

Run the test suite:
```powershell
python -m pytest test_main.py -v

# Run with coverage
python -m pytest test_main.py --cov=. --cov-report=html
```

## Requirements

- Python 3.7+
- The `py4at` repository as a sibling directory at `../py4at`
- Dependencies listed in `requirements.txt`

## Architecture

The wrapper consists of:

- `main.py`: Main CLI application with process management
- `config.py`: Configuration management with environment overrides
- `test_main.py`: Unit tests for core functionality

## Notes

- This wrapper expects the `py4at` repository to be present as a sibling
  directory at `../py4at` (unless overridden via `PY4AT_REPO_PATH`)
- It runs the original scripts without modification for quick demos
- All subprocess output is logged with process ID for debugging
- Processes are terminated gracefully on Ctrl-C or system signals

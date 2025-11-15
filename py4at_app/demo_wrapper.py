#!/usr/bin/env python3
"""
Tiny CLI wrapper to run TickServer and OnlineAlgorithm from the py4at repo.

This runs upstream scripts as subprocesses so we don't have to modify
the original repository files. Kept as a demo wrapper separate from the
main application CLI.
"""
import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

from config import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.default_log_level),
    format=config.log_format,
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def script_path(py4at_root: Path, *parts: str, as_str: bool = True) -> Union[str, Path]:
    """Construct absolute path to script within py4at repository.

    Returns a string by default for backward compatibility. Set `as_str=False`
    to receive a `pathlib.Path` object for internal checks.
    """
    p = (py4at_root / Path(*parts)).absolute()
    return str(p) if as_str else p


def run_process(cmd: List[str], env: Optional[dict] = None) -> subprocess.Popen:
    """Start a subprocess with proper error handling and logging."""
    try:
        logger.info(f'Starting: {" ".join(cmd)}')
        return subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
    except FileNotFoundError as e:
        logger.error(f'Command not found: {cmd[0]}')
        raise
    except subprocess.SubprocessError as e:
        logger.error(f'Failed to start process: {e}')
        raise


def validate_py4at_repository(py4at_root: Path) -> Tuple[Path, Path]:
    """Validate py4at repository exists and return script paths."""
    if not py4at_root.is_dir():
        logger.error(f'Cannot find py4at repository at {py4at_root}')
        sys.exit(1)
        return

    server_script = script_path(py4at_root, *config.server_script_path.split('/'), as_str=False)
    strategy_script = script_path(py4at_root, *config.strategy_script_path.split('/'), as_str=False)

    if not server_script.exists():
        logger.error(f'Server script not found: {server_script}')
        sys.exit(1)
        return

    if not strategy_script.exists():
        logger.error(f'Strategy script not found: {strategy_script}')
        sys.exit(1)
        return

    return server_script, strategy_script


def setup_signal_handlers(processes: List[subprocess.Popen]) -> None:
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f'Received signal {signum} — terminating subprocesses...')
        terminate_processes(processes)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def terminate_processes(processes: List[subprocess.Popen]) -> None:
    """Gracefully terminate all processes."""
    for proc in processes:
        if proc and proc.poll() is None:
            try:
                logger.info(f'Terminating process {proc.pid}')
                proc.terminate()
                # Give process time to terminate gracefully
                proc.wait(timeout=config.process_termination_timeout)
            except subprocess.TimeoutExpired:
                logger.warning(f'Process {proc.pid} did not terminate, killing...')
                proc.kill()
            except Exception as e:
                logger.error(f'Error terminating process {proc.pid}: {e}')


def monitor_processes(processes: List[subprocess.Popen]) -> None:
    """Monitor running processes and log their output."""
    active_processes = [p for p in processes if p is not None]

    while active_processes:
        time.sleep(config.process_monitor_interval)

        # Check for process output
        for proc in active_processes[:]:
            if proc.poll() is not None:
                logger.info(f'Process {proc.pid} exited with code {proc.returncode}')
                active_processes.remove(proc)
            else:
                # Log any available output
                try:
                    while True:
                        line = proc.stdout.readline()
                        if not line:
                            break
                        logger.info(f'[PID {proc.pid}] {line.strip()}')
                except Exception as e:
                    logger.debug(f'Error reading output from process {proc.pid}: {e}')


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run py4at demo server/strategy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s server    Run only the tick server
  %(prog)s strategy  Run only the online algorithm
  %(prog)s both      Run server then strategy
        """
    )
    parser.add_argument(
        'mode',
        choices=['server', 'strategy', 'both'],
        help='what to run (server, strategy, or both)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='enable verbose logging'
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Assume this app folder is a sibling of the cloned `py4at` repo
    here = Path(__file__).parent.absolute()
    workspace_root = here.parent
    py4at_root = config.get_py4at_root(workspace_root)

    logger.info(f'Workspace root: {workspace_root}')
    logger.info(f'py4at repository: {py4at_root}')

    server_script, strategy_script = validate_py4at_repository(py4at_root)
    python = sys.executable or 'python'

    processes: List[subprocess.Popen] = []

    try:
        if args.mode in ('server', 'both'):
            server_proc = run_process([python, '-u', str(server_script)])
            processes.append(server_proc)
            # Give server time to bind the socket
            logger.info('Waiting for server to start...')
            time.sleep(config.server_startup_delay)

        if args.mode in ('strategy', 'both'):
            strat_proc = run_process([python, '-u', str(strategy_script)])
            processes.append(strat_proc)

        # Setup signal handlers for graceful shutdown
        setup_signal_handlers(processes)

        # Monitor processes
        monitor_processes(processes)

    except KeyboardInterrupt:
        logger.info('Keyboard interrupt received — terminating subprocesses...')
    except Exception as e:
        logger.error(f'Unexpected error: {e}')
        sys.exit(1)
    finally:
        terminate_processes(processes)
        logger.info('Cleanup completed')


if __name__ == '__main__':
    main()

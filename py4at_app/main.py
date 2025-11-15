#!/usr/bin/env python3
"""
Tiny CLI wrapper to run TickServer and OnlineAlgorithm from the py4at repo.

Usage:
  python main.py server    # start the tick server
  python main.py strategy  # start the online algorithm (client)
  python main.py both      # start server then strategy (useful for local demo)

This wrapper runs the upstream scripts as subprocesses so we don't have to
modify the original repository files.
"""
import argparse
import os
import subprocess
import sys
import time


def script_path(py4at_root, *parts):
    return os.path.abspath(os.path.join(py4at_root, *parts))


def run_process(cmd, env=None):
    print('Starting:', ' '.join(cmd))
    return subprocess.Popen(cmd, env=env)


def main():
    parser = argparse.ArgumentParser(description='Run py4at demo server/strategy')
    parser.add_argument('mode', choices=['server', 'strategy', 'both'], help='what to run')
    args = parser.parse_args()

    # Assume this app folder is a sibling of the cloned `py4at` repo
    here = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(here, '..'))
    py4at_root = os.path.join(workspace_root, 'py4at')

    if not os.path.isdir(py4at_root):
        print('Error: cannot find py4at repository at', py4at_root)
        sys.exit(1)

    server_script = script_path(py4at_root, 'ch07', 'TickServer.py')
    strategy_script = script_path(py4at_root, 'ch07', 'OnlineAlgorithm.py')

    python = sys.executable or 'python'

    server_proc = None
    strat_proc = None

    try:
        if args.mode in ('server', 'both'):
            server_proc = run_process([python, '-u', server_script])
            # Give server some time to bind the socket
            time.sleep(1.0)

        if args.mode in ('strategy', 'both'):
            strat_proc = run_process([python, '-u', strategy_script])

        # Wait for processes. If both are running, wait until one finishes or Ctrl-C.
        procs = [p for p in (server_proc, strat_proc) if p is not None]
        while True:
            time.sleep(0.5)
            # Poll children and break if any exited
            for p in list(procs):
                if p.poll() is not None:
                    print('Process exited with code', p.returncode)
                    procs.remove(p)
            if not procs:
                break

    except KeyboardInterrupt:
        print('\nKeyboard interrupt received — terminating subprocesses...')
    finally:
        for p in (strat_proc, server_proc):
            if p is not None and p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass


if __name__ == '__main__':
    main()

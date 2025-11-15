# py4at local demo wrapper

This tiny wrapper runs the TickServer and OnlineAlgorithm from the `py4at` repo
as subprocesses so you can demo a local tick generator and a simple online
strategy.

Quick start (PowerShell):

```powershell
cd <workspace-root>\py4at_app
# install dependencies (use conda or pip)
pip install -r requirements.txt

# run server only
python main.py server

# run strategy only (connects to server at tcp://0.0.0.0:5555)
python main.py strategy

# run both (server then strategy)
python main.py both
```

Notes:
- This wrapper expects the `py4at` repository to be present as a sibling
  directory at `../py4at`.
- It runs the original scripts without modification; use it for quick demos.

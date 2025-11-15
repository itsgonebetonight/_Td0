"""Package entrypoint for `py4at_app`.

This entrypoint delegates to the canonical application CLI located in the
nested `__Td0/py4at_app/main.py` if present. If the canonical CLI is not
found, it falls back to the demo wrapper `demo_wrapper.py`.
"""
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
# Look for nested canonical app at ../__Td0/py4at_app/main.py
canonical = (HERE.parent / '__Td0' / 'py4at_app' / 'main.py')
if canonical.exists():
    runpy.run_path(str(canonical), run_name='__main__')
else:
    # Fallback to demo wrapper
    try:
        from . import demo_wrapper
        demo_wrapper.main()
    except Exception:
        print('No application entrypoint found and demo wrapper failed.', file=sys.stderr)
        raise

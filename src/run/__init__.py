from .run import run as default_run
from .run_v2 import run as run_v2

REGISTRY = {}
REGISTRY["default"] = default_run
REGISTRY["v2"] = run_v2


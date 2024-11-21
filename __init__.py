import os
import subprocess
import sys

from .nodes import api  # noqa
from .nodes.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

print("Comfy-Pack => Installing Python dependencies")
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
    stdout=subprocess.DEVNULL,
    cwd=os.path.dirname(__file__),
)

import pathlib
import sys

SRC_DIR = pathlib.Path(__file__).parent.parent / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pathlib
import os


CPACK_HOME = (
    pathlib.Path.home() / ".comfypack"
    if not os.environ.get("CPACK_HOME", "")
    else pathlib.Path(os.environ.get("CPACK_HOME", ""))
)
if not CPACK_HOME.exists():
    CPACK_HOME.mkdir()

MODEL_DIR = CPACK_HOME / "models"
WORKSPACE_DIR = CPACK_HOME / "workspace"
SHA_CACHE_FILE = CPACK_HOME / "sha_cache.json"
MODEL_SOURCE_CACHE_FILE = CPACK_HOME / "model_source_cache.json"

COMFYUI_REPO = "https://github.com/comfyanonymous/ComfyUI.git"
COMFY_PACK_REPO = "https://github.com/bentoml/comfy-pack.git"
COMFYUI_MANAGER_REPO = "https://github.com/ltdrdata/ComfyUI-Manager.git"

STRICT_MODE = os.environ.get("CPACK_STRICT_MODE", "0") in ["1", "true", "True"]

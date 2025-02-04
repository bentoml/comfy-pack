#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

virtualenv = os.environ.get("VIRTUAL_ENV")
if virtualenv and "--reload" not in sys.argv:
    print("Re-executing in virtualenv:", virtualenv)
    venv_python = os.path.join(virtualenv, "bin/python3")
    os.execl(venv_python, venv_python, *sys.argv, "--reload")

SNAPSHOT = """\
{snapshot}
"""


def _get_workspace() -> tuple[Path, dict]:
    import hashlib
    import json

    bentoml_home = os.getenv(
        "BENTOML_HOME", os.path.join(os.path.expanduser("~"), "bentoml")
    )

    checksum = hashlib.md5(SNAPSHOT.strip().encode("utf8")).hexdigest()
    wp = Path(bentoml_home) / "run" / "comfy_workspace" / checksum
    wp.parent.mkdir(parents=True, exist_ok=True)
    return wp, json.loads(SNAPSHOT)


def prepare_comfy_workspace():
    import shutil

    from comfy_pack.package import install_comfyui, install_custom_modules

    verbose = 2
    comfy_workspace, snapshot = _get_workspace()

    if not comfy_workspace.joinpath(".DONE").exists():
        if comfy_workspace.exists():
            print("Removing existing workspace")
            shutil.rmtree(comfy_workspace, ignore_errors=True)
        install_comfyui(snapshot, comfy_workspace, verbose=verbose)

        install_custom_modules(snapshot, comfy_workspace, verbose=verbose)
        comfy_workspace.joinpath(".DONE").touch()
        subprocess.run(
            ["chown", "-R", "bentoml:bentoml", str(comfy_workspace)], check=True
        )


if __name__ == "__main__":
    prepare_comfy_workspace()

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
    os.execl(venv_python, venv_python, *sys.argv[1:], "--reload")

# The script path is ./env/docker/setup_script
SRC_DIR = Path(__file__).parent.parent.parent / "src"
INPUT_DIR = SRC_DIR / "input"
sys.path.append(str(SRC_DIR))


def _get_workspace() -> tuple[Path, dict]:
    import hashlib
    import json

    from bentoml._internal.configuration.containers import BentoMLContainer

    snapshot = SRC_DIR / "snapshot.json"
    checksum = hashlib.md5(snapshot.read_bytes()).hexdigest()
    wp = (
        Path(BentoMLContainer.bentoml_home.get()) / "run" / "comfy_workspace" / checksum
    )
    wp.parent.mkdir(parents=True, exist_ok=True)
    return wp, json.loads(snapshot.read_text())


def prepare_comfy_workspace():
    import shutil
    from typing import cast

    import bentoml
    from bentoml.models import HuggingFaceModel

    from comfy_pack.package import install_comfyui, install_custom_modules

    verbose = int("BENTOML_DEBUG" in os.environ)
    comfy_workspace, snapshot = _get_workspace()
    service = bentoml.load(str(SRC_DIR.parent))

    if not comfy_workspace.joinpath(".DONE").exists():
        if comfy_workspace.exists():
            print("Removing existing workspace")
            shutil.rmtree(comfy_workspace, ignore_errors=True)
        install_comfyui(snapshot, comfy_workspace, verbose=verbose)

        for model in snapshot["models"]:
            if model.get("disabled", False):
                continue
            model_path = comfy_workspace / cast(str, model["filename"])
            if model_tag := model.get("model_tag"):
                model_path.parent.mkdir(parents=True, exist_ok=True)
                bento_model = bentoml.models.get(model_tag)
                model_file = bento_model.path_of("model.bin")
                print(f"Copying {model_file} to {model_path}")
                model_path.symlink_to(model_file)
            elif (source := model["source"]).get("source") == "huggingface":
                matched = next(
                    (
                        m
                        for m in service.models
                        if isinstance(m, HuggingFaceModel)
                        and m.model_id.lower() == source["repo"].lower()
                        and source["commit"].lower() == m.revision.lower()
                    ),
                    None,
                )
                if matched is not None:
                    model_file = os.path.join(matched.resolve(), source["path"])
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    print(f"Copying {model_file} to {model_path}")
                    model_path.symlink_to(model_file)
            else:
                print(
                    f"WARN: Unrecognized model source: {source}, the model may be missing"
                )

        for f in INPUT_DIR.glob("*"):
            if f.is_file():
                shutil.copy(f, comfy_workspace / "input" / f.name)
            elif f.is_dir():
                shutil.copytree(f, comfy_workspace / "input" / f.name)

        install_custom_modules(snapshot, comfy_workspace, verbose=verbose)
        comfy_workspace.joinpath(".DONE").touch()
        subprocess.run(
            ["chown", "-R", "bentoml:bentoml", str(comfy_workspace)], check=True
        )


if __name__ == "__main__":
    prepare_comfy_workspace()

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

COMFYUI_REPO = "https://github.com/comfyanonymous/ComfyUI.git"


def _clone_commit(url: str, commit: str, dir: Path, verbose: int = 0):
    stdout = None if verbose > 0 else subprocess.DEVNULL
    stderr = None if verbose > 1 else subprocess.DEVNULL
    subprocess.check_call(
        ["git", "clone", "--filter=blob:none", url, dir],
        stdout=stdout,
        stderr=stderr,
    )
    subprocess.check_call(
        ["git", "fetch", "-q", url, commit],
        cwd=dir,
        stdout=stdout,
        stderr=stderr,
    )
    subprocess.check_call(
        ["git", "checkout", "FETCH_HEAD"],
        cwd=dir,
        stdout=stdout,
        stderr=stderr,
    )


def install_comfyui(snapshot, workspace: Path, verbose: int = 0):
    print("Installing ComfyUI")
    comfyui_commit = snapshot["comfyui"]
    _clone_commit(COMFYUI_REPO, comfyui_commit, workspace, verbose=verbose)


def install_custom_modules(snapshot, workspace: Path, verbose: int = 0):
    print("Installing custom nodes")
    for module in snapshot["custom_nodes"]:
        url = module["url"]
        directory = url.split("/")[-1].split(".")[0]
        module_dir = workspace / "custom_nodes" / directory

        commit_hash = module["commit_hash"]
        _clone_commit(url, commit_hash, module_dir, verbose=verbose)


def install_dependencies(
    snapshot: dict,
    req_file: str,
    workspace: Path,
    verbose: int = 0,
):
    if verbose > 0:
        print("Installing Python dependencies")
    python_version = snapshot["python"]
    stdout = None if verbose > 0 else subprocess.DEVNULL
    stderr = None if verbose > 1 else subprocess.DEVNULL

    subprocess.check_call(
        ["uv", "python", "install", python_version],
        cwd=workspace,
        stdout=stdout,
        stderr=stderr,
    )
    venv = workspace / ".venv"
    if (venv / "DONE").exists():
        return
    venv_py = (
        venv / "Scripts" / "python.exe" if os.name == "nt" else venv / "bin" / "python"
    )
    subprocess.check_call(
        [
            "uv",
            "venv",
            "--python",
            python_version,
            venv,
        ],
        stdout=stdout,
        stderr=stderr,
    )
    subprocess.check_call(
        [
            "uv",
            "pip",
            "install",
            "-p",
            str(venv_py),
            "pip",
        ],
        stdout=stdout,
        stderr=stderr,
    )
    subprocess.check_call(
        [
            "uv",
            "pip",
            "install",
            "-p",
            str(venv_py),
            "-r",
            req_file,
            "--no-deps",
        ],
        stdout=stdout,
        stderr=stderr,
    )
    with open(venv / "DONE", "w") as f:
        f.write("DONE")
    return venv_py


def install(cpack: str | Path, workspace: str | Path = "workspace", verbose: int = 0):
    workspace = Path(workspace)
    with tempfile.TemporaryDirectory() as temp_dir:
        pack_dir = Path(temp_dir) / ".cpack"
        shutil.unpack_archive(cpack, pack_dir)
        snapshot = json.loads((pack_dir / "snapshot.json").read_text())
        req_txt_file = pack_dir / "requirements.txt"

        install_comfyui(snapshot, workspace, verbose=verbose)
        install_custom_modules(snapshot, workspace, verbose=verbose)
        install_dependencies(snapshot, str(req_txt_file), workspace, verbose=verbose)

        for f in (pack_dir / "input").glob("*"):
            shutil.copy(f, workspace / "input" / f.name)

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

from .const import COMFYUI_REPO, MODEL_DIR, STRICT_MODE
from .hash import get_sha256
from .utils import get_self_git_commit

if TYPE_CHECKING:
    import bentoml

COMFY_PACK_DIR = Path(__file__).parent


def _clone_commit(url: str, commit: str, dir: Path, verbose: int = 0):
    stdout = None if verbose > 0 else subprocess.DEVNULL
    stderr = None if verbose > 1 else subprocess.DEVNULL
    subprocess.check_call(
        ["git", "clone", "--recurse-submodules", "--filter=blob:none", url, dir],
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
    subprocess.check_call(
        ["git", "submodule", "update", "--init", "--recursive"],
        cwd=dir,
        stdout=stdout,
        stderr=stderr,
    )


def install_comfyui(snapshot, workspace: Path, verbose: int = 0):
    print("Installing ComfyUI")
    comfyui_commit = snapshot["comfyui"]
    if workspace.exists():
        if workspace.joinpath(".DONE").exists():
            commit = (workspace / ".DONE").read_text()
            if commit.strip() == comfyui_commit:
                print("ComfyUI is already installed")
                return
        shutil.rmtree(workspace)
    _clone_commit(COMFYUI_REPO, comfyui_commit, workspace, verbose=verbose)
    with open(workspace / ".DONE", "w") as f:
        f.write(comfyui_commit)


def install_custom_modules(snapshot, workspace: Path, verbose: int = 0):
    print("Installing custom nodes")
    for module in snapshot["custom_nodes"]:
        url = module["url"]
        if not url.strip():
            print(f"Skipping invalid custom node: {module}")
            continue
        directory = url.split("/")[-1].split(".")[0]
        module_dir = workspace / "custom_nodes" / directory

        if module_dir.exists():
            if module_dir.joinpath(".DONE").exists():
                commit = (module_dir / ".DONE").read_text()
                if commit.strip() == module["commit_hash"]:
                    print(f"{directory} is already installed")
                    continue
            shutil.rmtree(module_dir)

        commit_hash = module["commit_hash"]
        _clone_commit(url, commit_hash, module_dir, verbose=verbose)

        if module_dir.joinpath("install.py").exists():
            env = os.environ.copy()
            venv = workspace / ".venv"
            if venv.exists():
                python = (
                    venv / "Scripts" / "python.exe"
                    if os.name == "nt"
                    else venv / "bin" / "python"
                )
                if "PATH" in env:
                    env["PATH"] = f"{str(python.parent)}:{env['PATH']}"
                else:
                    env["PATH"] = str(python.parent)
                env["VIRTUAL_ENV"] = str(venv)
            else:
                python = Path(sys.executable)

            if verbose > 0:
                print(f"Installing {directory} custom node")
                print(f"$ {python.absolute()} install.py")
            subprocess.check_call(
                [str(python.absolute()), "install.py"],
                cwd=module_dir,
                stdout=subprocess.DEVNULL if verbose == 0 else None,
            )

        with open(module_dir / ".DONE", "w") as f:
            f.write(commit_hash)


def install_dependencies(
    snapshot: dict,
    req_file: str,
    workspace: Path,
    verbose: int = 0,
):
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
    if verbose > 0:
        print(f"Installing dependencies from {req_file}")
    install_cmd = [
        "uv",
        "pip",
        "install",
        "-p",
        str(venv_py),
        "-r",
        req_file,
        "--no-deps",
    ]
    if STRICT_MODE:
        pass
    else:
        install_cmd.extend(["--index-strategy", "unsafe-best-match"])
    subprocess.check_call(
        install_cmd,
        stdout=stdout,
        stderr=stderr,
    )
    with open(venv / "DONE", "w") as f:
        f.write("DONE")
    return venv_py


def get_search_url(sha: str) -> str:
    """Generate custom search URLs for model on HuggingFace and CivitAI"""
    base_url = "https://duckduckgo.com"
    sha = sha.upper()
    hf_query = f"{sha} OR {sha[:10]}"
    hf_query = urllib.parse.quote(hf_query)
    return f"{base_url}?q={hf_query}"


def download_file(url: str, dest_path: Path, progress_callback=None):
    """Download file with progress tracking"""
    if subprocess.call(["curl", "--version"], stdout=subprocess.DEVNULL) == 0:
        subprocess.check_call(
            ["curl", "-L", url, "-o", str(dest_path)],
        )
        return True
    try:
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get("content-length", 0))
            block_size = 8192
            downloaded = 0

            with open(dest_path, "wb") as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    f.write(buffer)
                    if progress_callback:
                        progress = (
                            (downloaded / total_size) * 100 if total_size > 0 else 0
                        )
                        progress_callback(progress)
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def show_progress(filename: str):
    """Progress callback function"""

    def callback(progress):
        print(f"\rDownloading {filename}: {progress:.1f}%", end="")

    return callback


def create_model_symlink(global_path: Path, sha: str, target_path: Path, filename: str):
    """Create symlink from global storage to workspace"""
    source = global_path / sha
    target = target_path / filename

    if target.exists():
        if target.is_symlink():
            target.unlink()
        else:
            raise RuntimeError(f"File {target} already exists and is not a symlink")

    target.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(source, target)


def retrieve_models(
    snapshot: dict,
    workspace: Path,
    download: bool = True,
    all_models: bool = False,
    verbose: int = 0,
):
    """Retrieve models from user downloads"""
    print("Retrieving models")
    models = snapshot.get("models", [])
    if not models:
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for model in models:
        sha = model["sha256"]
        filename = model["filename"]
        disabled = model.get("disabled", False)
        if (workspace / filename).exists():
            if not (MODEL_DIR / sha).exists() and (workspace / filename).is_file():
                shutil.move(workspace / filename, MODEL_DIR / sha)
                create_model_symlink(MODEL_DIR, sha, workspace, filename)
            continue

        if (MODEL_DIR / sha).exists():
            print(f"Model {filename} already exists in cache")
            create_model_symlink(MODEL_DIR, sha, workspace, filename)
            continue

        if disabled and not all_models:
            continue

        if not download:
            continue

        print(f"\nModel {filename} is never downloaded before")
        if source := model.get("source"):
            url = source["download_url"]
            target_path = MODEL_DIR / sha
            download_thread = threading.Thread(
                target=download_file,
                args=(url, target_path, show_progress(filename)),
            )
            download_thread.start()
            download_thread.join()

            if not target_path.exists():
                print("\nDownload failed!")
                continue

            print("\nDownload completed! Verifying SHA256...")
            terget_sha = get_sha256(str(target_path))
            if terget_sha != sha:
                print("SHA256 verification failed! File may be corrupted or incorrect.")
                target_path.unlink()
            else:
                continue

        search_url = get_search_url(sha)
        print(f"Search URL: {search_url}")
        print(f"Path: {workspace / filename}")

        while True:
            path = input("Enter path to downloaded file (or 'skip' to skip): ")
            if path.lower() == "skip":
                break

            try:
                # Check if input is a URL
                if path.startswith(("http://", "https://")):
                    url = path
                    target_path = MODEL_DIR / sha

                    # Start download in a separate thread
                    download_thread = threading.Thread(
                        target=download_file,
                        args=(url, target_path, show_progress(filename)),
                    )
                    download_thread.start()
                    download_thread.join()

                    if not target_path.exists():
                        print("\nDownload failed!")
                        continue

                    print("\nDownload completed! Verifying SHA256...")
                    terget_sha = get_sha256(str(target_path))
                    if terget_sha != sha:
                        print(
                            "SHA256 verification failed! File may be corrupted or incorrect."
                        )
                        target_path.unlink()
                        continue

                    print("SHA256 verification successful!")
                else:
                    # Handle local file
                    downloaded_path = Path(path)
                    if not downloaded_path.exists():
                        print("File does not exist!")
                        continue

                    # Verify SHA256 before copying
                    print("Verifying SHA256...")
                    target_sha = get_sha256(str(downloaded_path))
                    if target_sha != sha:
                        print(
                            f"Downloaded file SHA256 does not match expected SHA256: {target_sha} != {sha}"
                        )
                        continue

                    print("SHA256 verification successful!")
                    # Copy to global storage
                    shutil.copy2(downloaded_path, MODEL_DIR / sha)

                # Create symlink
                create_model_symlink(MODEL_DIR, sha, workspace, filename)
                print(f"Model {filename} installed successfully")
                break
            except Exception as e:
                print(f"Error processing file: {e}")
                continue


def install(
    cpack: str | Path,
    workspace: str | Path = "workspace",
    preheat: bool = True,
    all_models: bool = False,
    verbose: int = 0,
):
    workspace = Path(workspace)
    print(f"Installing package {cpack} to {workspace} (verbose={verbose})")
    with tempfile.TemporaryDirectory() as temp_dir:
        pack_dir = Path(temp_dir) / ".cpack"
        shutil.unpack_archive(cpack, pack_dir)
        snapshot = json.loads((pack_dir / "snapshot.json").read_text())
        req_txt_file = pack_dir / "requirements.txt"

        install_comfyui(snapshot, workspace, verbose=verbose)
        install_dependencies(snapshot, str(req_txt_file), workspace, verbose=verbose)

        for f in (pack_dir / "input").glob("*"):
            if f.is_file():
                shutil.copy(f, workspace / "input" / f.name)
            elif f.is_dir():
                shutil.copytree(f, workspace / "input" / f.name, dirs_exist_ok=True)

        retrieve_models(
            snapshot,
            workspace,
            verbose=verbose,
            download=False,
        )

        install_custom_modules(snapshot, workspace, verbose=verbose)
        if preheat:
            from .run import ComfyUIServer

            with ComfyUIServer(
                str(workspace),
                verbose=verbose,
                venv=str(workspace / ".venv"),
            ) as _:
                pass

        retrieve_models(snapshot, workspace, verbose=verbose, all_models=all_models)


required_files = ["snapshot.json", "requirements.txt"]


def build_bento(
    bento_name: str,
    source_dir: Path,
    *,
    version: str | None = None,
    system_packages: list[str] | None = None,
    include_default_system_packages: bool = True,
) -> bentoml.Bento:
    import bentoml

    for f in required_files:
        if not (source_dir / f).exists():
            raise FileNotFoundError(f"Not a valid comfy-pack package: missing `{f}`")

    if include_default_system_packages:
        system_packages = [
            "git",
            "libglib2.0-0",
            "libsm6",
            "libxrender1",
            "libxext6",
            "ffmpeg",
            "libstdc++-12-dev",
            *(system_packages or []),
        ]
    else:
        system_packages = system_packages or []

    shutil.copy2(Path(__file__).with_name("service.py"), source_dir / "service.py")
    with Path(__file__).with_name("setup_workspace.py").open() as f:
        (source_dir / "setup_workspace.py").write_text(
            f.read().format(snapshot=snapshot_text)
        )
    snapshot_text = (source_dir / "snapshot.json").read_text()
    snapshot = json.loads(snapshot_text)
    return bentoml.build(
        "service:ComfyService",
        name=bento_name,
        version=version,
        build_ctx=str(source_dir),
        labels={"comfy-pack-version": get_self_git_commit() or "unknown"},
        models=[
            m["model_tag"]
            for m in snapshot["models"]
            if "model_tag" in m and not m.get("disabled", False)
        ],
        docker={
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "system_packages": system_packages,
            "setup_script": source_dir.joinpath("setup_workspace.py").as_posix(),
        },
        python={"requirements_txt": "requirements.txt", "lock_packages": True},
    )

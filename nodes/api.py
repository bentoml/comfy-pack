import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
from pathlib import Path

import folder_paths
from aiohttp import web
from server import PromptServer

TEMP_FOLDER = Path(__file__).parent.parent / "temp"


async def _write_requirements(zf: zipfile.ZipFile) -> None:
    print("Package => Writing requirements.txt")
    with zf.open("requirements.txt", "w") as f:
        proc = await asyncio.subprocess.create_subprocess_exec(
            sys.executable,
            "-m",
            "pip",
            "freeze",
            "--exclude-editable",
            stdout=subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        f.write(stdout)


async def _write_snapshot(zf: zipfile.ZipFile) -> None:
    proc = await asyncio.subprocess.create_subprocess_exec(
        "git", "rev-parse", "HEAD", stdout=subprocess.PIPE, cwd=folder_paths.base_path
    )
    stdout, _ = await proc.communicate()
    with zf.open("snapshot.json", "w") as f:
        data = {
            "comfyui": stdout.decode().strip(),
            "models": await _get_models(),
            "custom_nodes": await _get_custom_nodes(),
        }
        f.write(json.dumps(data, indent=2).encode())


async def _get_models() -> list:
    print("Package => Writing models")
    proc = await asyncio.subprocess.create_subprocess_exec(
        "git",
        "ls-files",
        "--others",
        folder_paths.models_dir,
        stdout=subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    models = []
    for line in stdout.decode().splitlines():
        if os.path.basename(line).startswith("."):
            continue
        filename = os.path.abspath(line)
        relpath = os.path.relpath(filename, folder_paths.base_path)
        with open(filename, "rb") as model:
            models.append(
                {
                    "filename": relpath,
                    "sha256": hashlib.sha256(model.read()).hexdigest(),
                }
            )
    return models


async def _get_custom_nodes() -> list:
    print("Package => Writing custom nodes")
    custom_nodes = os.path.join(folder_paths.base_path, "custom_nodes")
    coros = []

    async def get_node_info(subdir: Path) -> dict:
        proc = await asyncio.subprocess.create_subprocess_exec(
            "git",
            "config",
            "--get",
            "remote.origin.url",
            cwd=subdir,
            stdout=subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        url = stdout.decode().strip()

        proc = await asyncio.subprocess.create_subprocess_exec(
            "git",
            "rev-parse",
            "HEAD",
            cwd=subdir,
            stdout=subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        commit_hash = stdout.decode().strip()
        return {
            "url": url,
            "commit_hash": commit_hash,
            "disabled": subdir.name.endswith(".disabled"),
        }

    for subdir in Path(custom_nodes).iterdir():
        if not subdir.is_dir() or not subdir.joinpath(".git").exists():
            continue
        coros.append(get_node_info(subdir))

    return await asyncio.gather(*coros)


async def _write_workflow(zf: zipfile.ZipFile, data: dict) -> None:
    print("Package => Writing workflow")
    with zf.open("workflow_api.json", "w") as f:
        f.write(json.dumps(data["workflow_api"], indent=2).encode())
    with zf.open("workflow.json", "w") as f:
        f.write(json.dumps(data["workflow"], indent=2).encode())


async def _write_inputs(zf: zipfile.ZipFile) -> None:
    print("Package => Writing inputs")
    input_dir = folder_paths.get_input_directory()

    for root_path, _, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root_path, file)
            relpath = os.path.relpath(file_path, input_dir)
            with zf.open(f"inputs/{relpath}", "w") as f:
                with open(file_path, "rb") as input_file:
                    shutil.copyfileobj(input_file, f)


@PromptServer.instance.routes.post("/bentoml/pack")
async def pack_workspace(request):
    data = await request.json()
    TEMP_FOLDER.mkdir(exist_ok=True)
    older_than_1h = time.time() - 60 * 60
    for file in TEMP_FOLDER.iterdir():
        if file.is_file() and file.stat().st_ctime < older_than_1h:
            file.unlink()

    zip_filename = f"{uuid.uuid4()}.zip"

    with zipfile.ZipFile(TEMP_FOLDER / zip_filename, "w") as zf:
        await _write_requirements(zf)
        await _write_snapshot(zf)
        await _write_workflow(zf, data)
        await _write_inputs(zf)

    return web.json_response({"download_url": f"/bentoml/download/{zip_filename}"})


@PromptServer.instance.routes.get("/bentoml/download/{zip_filename}")
async def download_workspace(request):
    zip_filename = request.match_info["zip_filename"]
    return web.FileResponse(TEMP_FOLDER / zip_filename)

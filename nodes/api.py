from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Union

import folder_paths
from aiohttp import web
from server import PromptServer

ZPath = Union[Path, zipfile.Path]
TEMP_FOLDER = Path(__file__).parent.parent / "temp"
COMFY_PACK_DIR = Path(__file__).parent.parent / "src/comfy_pack"
EXCLUDE_PACKAGES = ["bentoml", "onnxruntime"]  # TODO: standardize this


async def _write_requirements(path: ZPath, extras: list[str] | None = None) -> None:
    print("Package => Writing requirements.txt")
    with path.joinpath("requirements.txt").open("w") as f:
        proc = await asyncio.subprocess.create_subprocess_exec(
            sys.executable,
            "-m",
            "pip",
            "freeze",
            "--exclude-editable",
            *[f"--exclude={p}" for p in EXCLUDE_PACKAGES],
            stdout=subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        f.write(stdout.decode().rstrip("\n") + "\n")
        if extras:
            f.write("\n".join(extras) + "\n")


async def _write_snapshot(path: ZPath, data: dict, models: list | None = None) -> None:
    proc = await asyncio.subprocess.create_subprocess_exec(
        "git", "rev-parse", "HEAD", stdout=subprocess.PIPE, cwd=folder_paths.base_path
    )
    stdout, _ = await proc.communicate()
    if models is None:
        models = await _get_models(data)
    with path.joinpath("snapshot.json").open("w") as f:
        data = {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "comfyui": stdout.decode().strip(),
            "models": models,
            "custom_nodes": await _get_custom_nodes(),
        }
        f.write(json.dumps(data, indent=2))


def _get_file_info_key(file_path: Path) -> tuple:
    # use mtime, size to determine if a file has changed
    return (file_path.stat().st_mtime, file_path.stat().st_size)


def _get_hash(file_path: Path) -> str:
    TEMP_FOLDER.mkdir(exist_ok=True)
    cpack_hash_cache_file = TEMP_FOLDER / "model_sha_cache.json"
    try:
        with cpack_hash_cache_file.open("r") as f:
            cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}

    mtime, size = _get_file_info_key(file_path)
    if file_path.absolute().as_posix() in cache:
        if (
            cache[file_path.absolute().as_posix()]["mtime"] == mtime
            and cache[file_path.absolute().as_posix()]["size"] == size
        ):
            return cache[file_path.absolute().as_posix()]["sha256"]

    with cpack_hash_cache_file.open("w") as f:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as file:
            for chunk in iter(lambda: file.read(4096), b""):
                hasher.update(chunk)
        sha = hasher.hexdigest()
        cache[file_path.absolute().as_posix()] = {
            "sha256": sha,
            "mtime": mtime,
            "size": size,
        }
        json.dump(cache, f)
        return sha


async def _get_models(data: dict, store_models: bool = False) -> list:
    print("Package => Writing models")
    proc = await asyncio.subprocess.create_subprocess_exec(
        "git",
        "ls-files",
        "--others",
        folder_paths.models_dir,
        stdout=subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()

    used_inputs = set()
    for node in data["workflow_api"].values():
        for _, v in node["inputs"].items():
            if isinstance(v, str):
                used_inputs.add(v)

    models = []
    for line in stdout.decode().splitlines():
        if os.path.basename(line).startswith("."):
            continue
        filename = os.path.abspath(line)
        relpath = os.path.relpath(filename, folder_paths.base_path)

        relpath_path = Path(relpath)

        model_data = {
            "filename": relpath,
            "sha256": _get_hash(Path(filename)),
            "explicit": relpath_path.name in used_inputs,
            "size": os.path.getsize(filename),
        }
        if store_models:
            import bentoml

            model_tag = f'cpack-model:{model_data["sha256"][:16]}'
            try:
                model = bentoml.models.get(model_tag)
            except bentoml.exceptions.NotFound:
                with bentoml.models.create(
                    model_tag,
                    labels={"filename": filename},
                ) as model:
                    shutil.copy(filename, model.path_of("model.bin"))
            model_data["model_tag"] = model_tag
        models.append(model_data)
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


async def _write_workflow(path: ZPath, data: dict) -> None:
    print("Package => Writing workflow")
    with path.joinpath("workflow_api.json").open("w") as f:
        f.write(json.dumps(data["workflow_api"], indent=2))
    with path.joinpath("workflow.json").open("w") as f:
        f.write(json.dumps(data["workflow"], indent=2))


async def _write_inputs(path: ZPath, data: dict) -> None:
    print("Package => Writing inputs")
    if isinstance(path, Path):
        path.joinpath("input").mkdir(exist_ok=True)

    input_dir = folder_paths.get_input_directory()

    used_inputs = set()
    for node in data["workflow_api"].values():
        for _, v in node["inputs"].items():
            if isinstance(v, str):
                used_inputs.add(v)

    src_root = Path(input_dir).absolute()
    for src in src_root.glob("**/*"):
        rel = src.relative_to(src_root)
        if src.is_dir():
            if isinstance(path, Path):
                path.joinpath("input").joinpath(rel).mkdir(parents=True, exist_ok=True)
        if src.is_file():
            with path.joinpath("input").joinpath(rel).open("wb") as f:
                with open(src, "rb") as input_file:
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
        path = zipfile.Path(zf)
        await _write_requirements(path)
        await _write_snapshot(path, data)
        await _write_workflow(path, data)
        await _write_inputs(path, data)

    return web.json_response({"download_url": f"/bentoml/download/{zip_filename}"})


@PromptServer.instance.routes.get("/bentoml/download/{zip_filename}")
async def download_workspace(request):
    zip_filename = request.match_info["zip_filename"]
    return web.FileResponse(TEMP_FOLDER / zip_filename)


@PromptServer.instance.routes.post("/bentoml/build")
async def build_bento(request):
    """Request body: {
        workflow_api: dict,
        workflow: dict,
        bento_name: str,
        push?: bool,
        api_key?: str,
        endpoint?: str,
        system_packages?: list[str]
    }"""
    import bentoml

    data = await request.json()

    with tempfile.TemporaryDirectory(suffix="-bento", prefix="comfy-pack-") as temp_dir:
        temp_dir_path = Path(temp_dir)
        # copy comfy_pack source code into the bento
        shutil.copytree(COMFY_PACK_DIR, temp_dir_path / COMFY_PACK_DIR.name)
        models = await _get_models(data, store_models=True)

        await _write_requirements(temp_dir_path, ["comfy-cli", "fastapi"])
        await _write_snapshot(temp_dir_path, data, models)
        await _write_workflow(temp_dir_path, data)
        await _write_inputs(temp_dir_path, data)
        shutil.copy(
            Path(__file__).with_name("service.py"),
            temp_dir_path / "service.py",
        )

        # create a bento
        try:
            bento = bentoml.build(
                "service:ComfyService",
                name=data["bento_name"],
                build_ctx=temp_dir,
                models=[m["model_tag"] for m in models if "model_tag" in m],
                docker={
                    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
                    "system_packages": [
                        "git",
                        "libglib2.0-0",
                        "libsm6",
                        "libxrender1",
                        "libxext6",
                        "ffmpeg",
                        "libstdc++-12-dev",
                        *data.get("system_packages", []),
                    ],
                },
                python={"requirements_txt": "requirements.txt", "lock_packages": True},
            )
        except bentoml.exceptions.BentoMLException as e:
            return web.json_response(
                {
                    "result": "error",
                    "error": f"Build failed: {e.__class__.__name__}: {e}",
                },
            )

    if data.get("push", False):
        credentials = {}
        if api_key := data.get("api_key"):
            credentials["api_key"] = api_key
        if endpoint := data.get("endpoint"):
            credentials["endpoint"] = endpoint
        client = bentoml.cloud.BentoCloudClient(**credentials)
        try:
            client.bento.push(bento)
        except bentoml.exceptions.BentoMLException as e:
            return web.json_response(
                {
                    "result": "error",
                    "error": f"Push failed: {e.__class__.__name__}: {e}",
                }
            )

    return web.json_response({"result": "success", "bento": str(bento.tag)})

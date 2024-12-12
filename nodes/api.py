from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Union

import folder_paths
from aiohttp import web
from server import PromptServer

from comfy_pack.hash import ModelHashes
from comfy_pack.model_helper import ModelEntry, get_model_entry

ZPath = Union[Path, zipfile.Path]
TEMP_FOLDER = Path(__file__).parent.parent / "temp"
COMFY_PACK_DIR = Path(__file__).parent.parent / "src" / "comfy_pack"
EXCLUDE_PACKAGES = ["bentoml", "onnxruntime"]  # TODO: standardize this


async def _write_requirements(path: ZPath, extras: list[str] | None = None) -> None:
    print("Package => Writing requirements.txt")
    with path.joinpath("requirements.txt").open("w") as f:
        proc = await asyncio.subprocess.create_subprocess_exec(
            sys.executable,
            "-m",
            "pip",
            "list",
            "--format",
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
        print("Package => Writing models")
        models = await _get_models()
    with path.joinpath("snapshot.json").open("w") as f:
        data = {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "comfyui": stdout.decode().strip(),
            "models": models,
            "custom_nodes": await _get_custom_nodes(),
        }
        f.write(json.dumps(data, indent=2))


def _is_port_in_use(port: int | str, host="localhost"):
    if isinstance(port, str):
        port = int(port)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((host, port))
            return True
        except ConnectionRefusedError:
            return False
        except Exception:
            return True


def _is_file_refered(file_path: Path, workflow_api: dict) -> bool:
    """ """
    used_inputs = set()
    for node in workflow_api.values():
        for _, v in node["inputs"].items():
            if isinstance(v, str):
                used_inputs.add(v)
    all_inputs = "\n".join(used_inputs)
    file_path = file_path.absolute().relative_to(folder_paths.base_path)
    relpath = Path(*file_path.parts[2:])
    return str(relpath) in all_inputs


async def _get_models(
    store_models: bool = False,
    workflow_api: dict | None = None,
    model_filter: set[str] | None = None,
    ensure_source: bool = True,
) -> list[ModelEntry]:
    proc = await asyncio.subprocess.create_subprocess_exec(
        "git",
        "ls-files",
        "--others",
        folder_paths.models_dir,
        stdout=subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()

    model_filenames = [
        os.path.abspath(line)
        for line in stdout.decode().splitlines()
        if not os.path.basename(line).startswith(".")
    ]
    model_hashes = ModelHashes()
    await model_hashes.load()
    with ThreadPoolExecutor() as executor:
        models = await asyncio.gather(
            *(
                get_model_entry(
                    f,
                    executor,
                    model_hashes,
                    store_models=store_models,
                    ensure_source=ensure_source,
                )
                for f in model_filenames
            )
        )
    # save the hashes to cache
    await model_hashes.save()
    for model in models:
        model["disabled"] = (
            model_filter is not None and model["filename"] not in model_filter
        )
        if workflow_api:
            model["refered"] = _is_file_refered(Path(model["filename"]), workflow_api)
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
        await _prepare_bento_project(
            path,
            data,
        )

    return web.json_response({"download_url": f"/bentoml/download/{zip_filename}"})


class DevServer:
    TIMEOUT = 3600 * 24
    proc: Union[None, subprocess.Popen] = None
    watch_dog_task: asyncio.Task | None = None
    last_feed = 0
    run_dir: Path | None = None
    port = 0

    @classmethod
    def start(cls, workflow_api: dict, port: int = 3000):
        cls.stop()

        cls.port = port
        # prepare a temporary directory
        cls.run_dir = Path(tempfile.mkdtemp(suffix="-bento", prefix="comfy-pack-"))
        with cls.run_dir.joinpath("workflow_api.json").open("w") as f:
            f.write(json.dumps(workflow_api, indent=2))
        shutil.copy(
            Path(__file__).with_name("service.py"),
            cls.run_dir / "service.py",
        )
        shutil.copytree(COMFY_PACK_DIR, cls.run_dir / COMFY_PACK_DIR.name)

        # find a free port
        self_port = 8188
        for i, arg in enumerate(sys.argv):
            if arg == "--port" or arg == "-p":
                self_port = int(sys.argv[i + 1])
                break

        print(f"Starting dev server at port {port}, comfyui at port {self_port}")
        cls.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "bentoml",
                "serve",
                "service:ComfyService",
                "--port",
                str(port),
            ],
            cwd=str(cls.run_dir.absolute()),
            env={
                **os.environ,
                "COMFYUI_SERVER": f"localhost:{self_port}",
            },
        )
        cls.watch_dog_task = asyncio.create_task(cls.watch_dog())
        cls.last_feed = time.time()

    @classmethod
    async def watch_dog(cls):
        while True:
            await asyncio.sleep(0.1)
            if cls.last_feed + cls.TIMEOUT < time.time():
                cls.stop()
                break

    @classmethod
    def feed_watch_dog(cls):
        if cls.proc:
            if cls.proc.poll() is None:
                cls.last_feed = time.time()
                return True
            else:
                cls.stop()
                return False
        return False

    @classmethod
    def stop(cls):
        if cls.proc:
            cls.proc.terminate()
            cls.proc.wait()
            cls.proc = None
            time.sleep(1)
            print("Dev server stopped")
        if cls.watch_dog_task:
            cls.watch_dog_task.cancel()
            cls.watch_dog_task = None
            cls.last_feed = 0
        if cls.run_dir:
            shutil.rmtree(cls.run_dir)
            cls.run_dir = None


def _parse_workflow(workflow: dict) -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = {}
    outputs = {}
    dep_map = {}

    for id, node in workflow.items():
        for input_name, v in node["inputs"].items():
            if isinstance(v, list) and len(v) == 2:  # is a link
                dep_map[tuple(v)] = node, input_name

    for id, node in workflow.items():
        node["id"] = id
        if node["class_type"].startswith("CPackInput"):
            if not node.get("inputs"):
                continue
            inputs[id] = node
        elif node["class_type"].startswith("CPackOutput"):
            if not node.get("inputs"):
                continue
            outputs[id] = node

    return inputs, outputs


def _validate_workflow(data: dict):
    workflow = data.get("workflow_api", {})
    if not workflow:
        return web.json_response(
            {
                "result": "error",
                "error": "empty workflow",
            },
        )
    input_spec, output_spec = _parse_workflow(workflow)
    if not input_spec:
        return web.json_response(
            {
                "result": "error",
                "error": "At least one ComfyPack input node is required",
            },
        )
    if not output_spec:
        return web.json_response(
            {
                "result": "error",
                "error": "At least one ComfyPack output node is required",
            },
        )


@PromptServer.instance.routes.post("/bentoml/serve")
async def serve(request):
    data = await request.json()

    if (error := _validate_workflow(data)) is not None:
        return error

    DevServer.stop()

    if _is_port_in_use(data.get("port", 3000), host=data.get("host", "localhost")):
        return web.json_response(
            {
                "result": "error",
                "error": "Port is already in use",
            },
        )
    try:
        DevServer.start(workflow_api=data["workflow_api"], port=data.get("port", 3000))
        return web.json_response(
            {
                "result": "success",
                "url": f"http://{data.get('host', 'localhost')}:{data.get('port', 3000)}",
            },
        )
    except Exception as e:
        return web.json_response(
            {
                "result": "error",
                "error": f"Build failed: {e.__class__.__name__}: {e}",
            },
        )


@PromptServer.instance.routes.post("/bentoml/serve/heartbeat")
async def heartbeat(_):
    running = DevServer.feed_watch_dog()

    if running:
        return web.json_response({"ready": True})
    else:
        return web.json_response({"error": "Server is not running"})


@PromptServer.instance.routes.post("/bentoml/serve/terminate")
async def terminate(_):
    DevServer.stop()
    return web.json_response({"result": "success"})


@PromptServer.instance.routes.get("/bentoml/download/{zip_filename}")
async def download_workspace(request):
    zip_filename = request.match_info["zip_filename"]
    return web.FileResponse(TEMP_FOLDER / zip_filename)


async def _prepare_bento_project(
    working_dir: ZPath,
    data: dict,
    store_models: bool = False,
):
    model_filter = set(data.get("models", []))
    models = await _get_models(
        store_models=store_models,
        model_filter=model_filter,
    )

    await _write_requirements(working_dir, ["comfy-cli", "fastapi"])
    await _write_snapshot(working_dir, data, models)
    await _write_workflow(working_dir, data)
    await _write_inputs(working_dir, data)
    with working_dir.joinpath("service.py").open("w") as f:
        f.write(Path(__file__).with_name("service.py").read_text())

    # Copy comfy_pack directory
    if isinstance(working_dir, Path):
        shutil.copytree(COMFY_PACK_DIR, working_dir / COMFY_PACK_DIR.name)
    else:  # zipfile.Path
        for src in COMFY_PACK_DIR.rglob("*"):
            if src.is_file():
                rel_path = src.relative_to(COMFY_PACK_DIR.parent)
                with working_dir.joinpath(rel_path).open("wb") as f:
                    f.write(src.read_bytes())
    models = [m for m in models if not m["disabled"]]  # filter out disabled models
    return models


@PromptServer.instance.routes.post("/bentoml/model/query")
async def get_models(request):
    data = await request.json()
    models = await _get_models(
        workflow_api=data.get("workflow_api"),
        ensure_source=False,
    )
    return web.json_response({"models": models})


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

    if (error := _validate_workflow(data)) is not None:
        return error

    with tempfile.TemporaryDirectory(suffix="-bento", prefix="comfy-pack-") as temp_dir:
        temp_dir_path = Path(temp_dir)
        models = await _prepare_bento_project(temp_dir_path, data, store_models=True)

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

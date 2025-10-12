from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import bentoml
import fastapi
from bentoml.models import HuggingFaceModel

import comfy_pack
import comfy_pack.run

REQUEST_TIMEOUT = 3600
BASE_DIR = Path(__file__).parent
COPY_THRESHOLD = 10 * 1024 * 1024
INPUT_DIR = BASE_DIR / "input"
logger = logging.getLogger("bentoml.service")


EXISTING_COMFYUI_SERVER = os.environ.get("COMFYUI_SERVER")

with BASE_DIR.joinpath("workflow_api.json").open() as f:
    workflow = json.load(f)

InputModel = comfy_pack.generate_input_model(workflow)
app = fastapi.FastAPI()


@lru_cache
def _get_workspace() -> Path:
    import hashlib

    from bentoml._internal.configuration.containers import BentoMLContainer

    snapshot = BASE_DIR / "snapshot.json"
    checksum = hashlib.md5(snapshot.read_bytes().strip()).hexdigest()
    wp = (
        Path(BentoMLContainer.bentoml_home.get()) / "run" / "comfy_workspace" / checksum
    )
    wp.parent.mkdir(parents=True, exist_ok=True)
    return wp


@app.get("/workflow.json")
def workflow_json():
    return workflow


def _watch_server(server: comfy_pack.run.ComfyUIServer):
    while True:
        time.sleep(1)
        if not server.is_running():
            if server.server_proc is not None:
                logger.warning(
                    "Server exited with code %s", server.server_proc.returncode
                )
                os.kill(os.getpid(), signal.SIGTERM)
            break


if not EXISTING_COMFYUI_SERVER:
    # register models
    with BASE_DIR.joinpath("snapshot.json").open("rb") as f:
        snapshot = json.load(f)
else:
    snapshot = {}


@bentoml.asgi_app(app, path="/comfy")
@bentoml.service(traffic={"timeout": REQUEST_TIMEOUT * 2}, resources={"gpu": 1})
class ComfyService:
    def __init__(self):
        logger = logging.getLogger("comfy_pack")
        logger.setLevel(logging.INFO)
        if not EXISTING_COMFYUI_SERVER:
            self.server = comfy_pack.run.ComfyUIServer(
                str(_get_workspace()),
                str(INPUT_DIR),
                verbose=int("BENTOML_DEBUG" in os.environ),
            )
            self.server.start()
            logger.info(
                "ComfyUI Server started at %s:%s", self.server.host, self.server.port
            )
            self.host = self.server.host
            self.port = self.server.port
            self.watch_thread = threading.Thread(
                target=_watch_server,
                args=(self.server,),
                daemon=True,
            )
            self.watch_thread.start()
            logger.info("Watch thread started")
        else:
            logger.info("Attaching to ComfyUI server: %s", EXISTING_COMFYUI_SERVER)
            if ":" in EXISTING_COMFYUI_SERVER:
                self.host, port = EXISTING_COMFYUI_SERVER.split(":")
                self.port = int(port)
            else:
                self.host = EXISTING_COMFYUI_SERVER
                self.port = 80

    @bentoml.api(input_spec=InputModel)
    def generate(
        self,
        *,
        ctx: bentoml.Context,
        **kwargs: Any,
    ) -> Path:
        verbose = int("BENTOML_DEBUG" in os.environ)
        workspace_path = (
            self.server.workspace
            if hasattr(self, "server")
            else os.environ.get("COMFYUI_PATH", ".")
        )
        ret = comfy_pack.run_workflow(
            self.host,
            self.port,
            workflow,
            output_dir=ctx.temp_dir,
            timeout=REQUEST_TIMEOUT,
            verbose=verbose,
            workspace=workspace_path,
            **kwargs,
        )
        if isinstance(ret, list):
            ret = ret[-1]
        return ret

    @bentoml.on_deployment
    @staticmethod
    def prepare_models():
        if EXISTING_COMFYUI_SERVER:
            return
        comfy_workspace = _get_workspace()
        if not comfy_workspace.joinpath(".DONE").exists():
            raise RuntimeError("ComfyUI workspace is not ready")
        for model in snapshot["models"]:
            if model.get("disabled", False):
                continue
            model_path = comfy_workspace / cast(str, model["filename"])
            if model_path.exists():
                continue
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
                        for m in ComfyService.models
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


if False and not EXISTING_COMFYUI_SERVER:
    for model in snapshot["models"]:
        if model.get("disabled"):
            continue
        source = model["source"]
        if source.get("source") != "huggingface" or source["repo"].startswith(
            "datasets/"
        ):
            continue
        ComfyService.models.append(HuggingFaceModel(source["repo"], source["commit"]))

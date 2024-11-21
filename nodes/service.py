from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, cast

import bentoml
import fastapi

import comfy_pack
import comfy_pack.run

REQUEST_TIMEOUT = 360
BASE_DIR = Path(__file__).parent
WORKFLOW_FILE = BASE_DIR / "workflow_api.json"
COPY_THRESHOLD = 10 * 1024 * 1024
INPUT_DIR = BASE_DIR / "input"
comfy_workspace = os.path.join(os.getcwd(), "comfy_workspace")
logger = logging.getLogger("bentoml.service")

with open(WORKFLOW_FILE, "r") as f:
    workflow = json.load(f)

InputModel = comfy_pack.generate_input_model(workflow)
app = fastapi.FastAPI()


@app.get("/workflow.json")
def workflow_json():
    return workflow


@bentoml.mount_asgi_app(app, path="/comfy")
@bentoml.service(traffic={"timeout": REQUEST_TIMEOUT * 2})
class ComfyService:
    def __init__(self):
        self.comfy_proc = comfy_pack.run.WorkflowRunner(
            comfy_workspace,
            str(INPUT_DIR),
        )
        self.comfy_proc.start()

    @bentoml.api(input_spec=InputModel)
    def generate(
        self,
        *,
        ctx: bentoml.Context,
        **kwargs: Any,
    ) -> Path:
        verbose = int("BENTOML_DEBUG" in os.environ)
        ret = self.comfy_proc.run_workflow(
            workflow,
            output_dir=ctx.temp_dir,
            timeout=REQUEST_TIMEOUT,
            verbose=verbose,
            **kwargs,
        )
        if isinstance(ret, list):
            ret = ret[-1]
        return ret

    @bentoml.on_shutdown
    def on_shutdown(self):
        self.comfy_proc.stop()

    @bentoml.on_deployment
    @staticmethod
    def prepare_comfy_workspace():
        from comfy_pack.package import install_comfyui, install_custom_modules

        verbose = int("BENTOML_DEBUG" in os.environ)

        with BASE_DIR.joinpath("snapshot.json").open("rb") as f:
            snapshot = json.load(f)

        install_comfyui(snapshot, comfy_workspace, verbose=verbose)
        install_custom_modules(snapshot, comfy_workspace, verbose=verbose)
        for model in snapshot["models"]:
            model_tag = model.get("model_tag")
            if not model_tag:
                logger.warning(
                    "Model %s is not in model store, the workflow may not work",
                    model["filename"],
                )
                continue
            model_path = Path(comfy_workspace) / cast(str, model["filename"])
            model_path.parent.mkdir(parents=True, exist_ok=True)
            bento_model = bentoml.models.get(model_tag)
            model_file = bento_model.path_of(model_path.name)
            logger.info("Copying %s to %s", model_file, model_path)
            model_path.symlink_to(model_file)

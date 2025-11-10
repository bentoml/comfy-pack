from __future__ import annotations

import copy
import json
import logging
import os
import random
import shlex
import shutil
import socket
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Union

from .utils import populate_workflow, retrieve_workflow_outputs

logger = logging.getLogger(__name__)


def _probe_comfyui_server(port: int) -> None:
    from urllib import parse, request

    url = f"http://127.0.0.1:{port}/api/customnode/getmappings"
    params = {"mode": "nickname"}
    full_url = f"{url}?{parse.urlencode(params)}"
    req = request.Request(full_url)
    _ = request.urlopen(req)

    full_url = f"http://127.0.0.1:{port}/api/object_info"
    req = request.Request(full_url)
    _ = request.urlopen(req)


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


class ComfyUIServer:
    def __init__(
        self,
        workspace: str,
        input_dir: str | None = None,
        host: str = "localhost",
        port: int | None = None,
        venv: str | None = None,
        verbose: int = 0,
    ) -> None:
        """
        Args:
            workspace (str, optional): The workspace path for ComfyUI. If not specified, runner will try to connect to an existing ComfyUI server.
            input_dir (str, optional): The input directory for ComfyUI. Defaults to None.
            port (int, optional): The port number for ComfyUI. Defaults to None. If 8188 is in use, a random port will be chosen.
        """
        self.workspace = workspace
        self.input_dir = input_dir
        self.verbose = verbose
        self.host = host
        self.server_proc: subprocess.Popen | None = None

        run_dir = (Path(workspace) / "cli_run").absolute()
        self.temp_dir = run_dir / "temp"
        self.output_dir = run_dir / "output"

        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if port is None:
            if _is_port_in_use(8188):
                self.port = port if port else random.randint(58000, 58999)
            else:
                self.port = 8188
        else:
            self.port = port
        self.venv = os.path.abspath(venv) if venv else None

    def start(self) -> None:
        """
        Start the ComfyUI process.

        This method starts ComfyUI in the background, sets up necessary directories,
        and disables tracking for workaround purposes.

        Args:
            verbose (int, optional): Verbosity level. If 0, suppress stdout. Defaults to 0.

        Raises:
            RuntimeError: If ComfyUI is already running.
        """
        logger.info(
            "Disable tracking from Comfy CLI, not for privacy concerns, but to workaround a bug"
        )
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        if self.venv:
            env["VIRTUAL_ENV"] = self.venv
            if os.name == "nt":
                env["PATH"] = f"{self.venv}\\Scripts;{env.get('PATH', '')}"
            else:
                env["PATH"] = f"{self.venv}/bin:{env.get('PATH', '')}"

        stdout = None if self.verbose > 0 else subprocess.DEVNULL
        command = ["comfy", "--skip-prompt", "tracking", "disable"]
        subprocess.run(command, check=True, stdout=stdout, env=env)
        logger.info("Successfully disabled Comfy CLI tracking")

        logger.info("Starting ComfyUI in the background...")
        command = [
            "python",
            "main.py",
            "--output-directory",
            self.output_dir,
            "--temp-directory",
            self.temp_dir,
            "--port",
            str(self.port),
        ]
        if self.input_dir:
            command.extend(["--input-directory", self.input_dir])

        if self.host != "localhost":
            command.extend(["--listen", self.host])
        if options := env.pop("COMFYUI_OPTIONS", None):
            command.extend(shlex.split(options))

        def preexec_fn():
            os.setpgrp()

        self.server_proc = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=None,
            preexec_fn=preexec_fn,
            env=env,
            cwd=self.workspace,
        )

        if _wait_for_startup(self.host, self.port):
            _probe_comfyui_server(self.port)
            logger.info("Successfully started ComfyUI in the background")
        else:
            logger.error("Failed to start ComfyUI in the background")

    def is_running(self) -> bool:
        if self.server_proc is None:
            return False
        if self.server_proc.poll() is not None:
            return False
        return True

    def stop(self) -> None:
        """
        Stop the ComfyUI process.

        This method stops the running ComfyUI process and cleans up the temporary directory if necessary.

        Raises:
            RuntimeError: If ComfyUI is not currently running.
        """
        if self.server_proc is None:
            raise RuntimeError("ComfyUI server is not started yet")
        proc = self.server_proc
        self.server_proc = None
        logger.info("Stopping ComfyUI...")
        proc.terminate()
        proc.wait()
        logger.info("Successfully stopped ComfyUI")

        logger.info("Cleaning up temporary directory...")
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        shutil.rmtree(self.output_dir, ignore_errors=True)
        logger.info("Successfully cleaned up temporary directory")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def _wait_for_startup(host: str, port: int, timeout: int = 1800) -> bool:
    start_time = time.time()
    while time.time() - start_time < timeout:
        if _is_port_in_use(port, host):
            return True
        time.sleep(1)
    return False


def run_workflow(
    host: str,
    port: int,
    workflow: dict,
    output_dir: Union[str, Path, None] = None,
    timeout: int = 300,
    verbose: int = 0,
    workspace: str = ".",
    **kwargs: Any,
) -> Any:
    """
    Run a ComfyUI workflow.

    This method executes a given workflow, populates it with input data,
    and retrieves the output.

    Args:
        workflow (dict): The workflow to run.
        output_dir (Union[str, Path, None], optional): Temporary directory for the workflow. Defaults to None.
        timeout (int, optional): Timeout for the workflow execution in seconds. Defaults to 300.
        **kwargs: Additional keyword arguments for workflow population.

    Returns:
        Any: The output of the workflow.

    Raises:
        RuntimeError: If ComfyUI is not started.
    """
    run_id = uuid.uuid4().hex[:8]

    workflow_copy = copy.deepcopy(workflow)
    if output_dir is None:
        output_dir = Path(".")
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    run_id = os.urandom(8).hex()
    populate_workflow(
        workflow_copy,
        output_dir,
        session_id=run_id,
        **kwargs,
    )

    workflow_file_path = output_dir / f"workflow_{run_id}.json"
    with open(workflow_file_path, "w") as file:
        json.dump(workflow_copy, file)

    extra_args = []
    if verbose > 0:
        extra_args.append("--verbose")

    stdout = None if verbose > 0 else subprocess.DEVNULL
    command = ["comfy", "--skip-prompt", "tracking", "disable"]
    subprocess.run(command, check=True, stdout=stdout)

    # Execute the workflow
    command = [
        "comfy",
        "--skip-prompt",
        "--workspace",
        workspace,
        "run",
        "--workflow",
        workflow_file_path.as_posix(),
        "--port",
        str(port),
        "--host",
        host,
        "--timeout",
        str(timeout),
        "--wait",
        *extra_args,
    ]
    env = os.environ.copy()
    env["NO_COLOR"] = "1"
    subprocess.run(command, check=True, env=env)

    workflow_file_path.unlink()

    # retrieve the output
    return retrieve_workflow_outputs(
        workflow_copy,
        output_dir,
        session_id=run_id,
    )

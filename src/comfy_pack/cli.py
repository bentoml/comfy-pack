import functools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import click

from .const import COMFY_PACK_REPO, COMFYUI_MANAGER_REPO, COMFYUI_REPO, WORKSPACE_DIR
from .hash import get_sha256
from .utils import get_self_git_commit


def _ensure_uv() -> None:
    """Ensure uv is installed, raise error if not."""
    try:
        subprocess.run(
            ["uv", "--version"],
            check=True,
            capture_output=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        raise RuntimeError(
            "uv is not installed. Please install it first:\n"
            "curl -LsSf https://astral.sh/uv/install.sh | sh"
        )


@click.group()
@click.version_option()
def main():
    """comfy-pack CLI"""
    pass


@main.command(
    name="init",
    help="Install latest ComfyUI and comfy-pack custom nodes and create a virtual environment",
)
@click.option(
    "--dir",
    "-d",
    default="ComfyUI",
    help="Target directory to install ComfyUI",
    type=click.Path(file_okay=False),
)
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Increase verbosity level",
)
def init(dir: str, verbose: int):
    import os

    import rich

    # Check if directory path is valid
    try:
        install_dir = Path(dir).absolute()
        if install_dir.exists() and not install_dir.is_dir():
            rich.print(f"[red]Error: {dir} exists but is not a directory[/red]")
            return 1

        # Check if directory is empty or contains ComfyUI
        if install_dir.exists():
            contents = list(install_dir.iterdir())
            if contents and not (install_dir / ".git").exists():
                rich.print(
                    f"[red]Error: Directory {dir} is not empty and doesn't appear to be a ComfyUI installation[/red]"
                )
                return 1
    except Exception as e:
        rich.print(f"[red]Error: Invalid directory path - {str(e)}[/red]")
        return 1

    # Check git installation
    try:
        subprocess.run(
            ["git", "--version"],
            check=True,
            capture_output=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        rich.print("[red]Error: git is not installed or not in PATH[/red]")
        return 1

    # Check if we have write permissions
    try:
        if not install_dir.exists():
            install_dir.mkdir(parents=True)
        test_file = install_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
    except (OSError, PermissionError) as e:
        rich.print(f"[red]Error: No write permission in {dir} - {str(e)}[/red]")
        return 1

    # Check if Python version is compatible
    if sys.version_info < (3, 8):
        rich.print("[red]Error: Python 3.8 or higher is required[/red]")
        return 1

    # Check if uv is installed
    try:
        _ensure_uv()
    except RuntimeError as e:
        rich.print(f"[red]Error: {str(e)}[/red]")
        return 1

    # Check if enough disk space is available (rough estimate: 2GB)
    try:
        free_space = shutil.disk_usage(install_dir).free
        if free_space < 2 * 1024 * 1024 * 1024:  # 2GB in bytes
            rich.print(
                "[yellow]Warning: Less than 2GB free disk space available[/yellow]"
            )
    except Exception as e:
        rich.print(
            f"[yellow]Warning: Could not check free disk space - {str(e)}[/yellow]"
        )

    # Clone ComfyUI if not exists
    if not (install_dir / ".git").exists():
        rich.print("[green]Cloning ComfyUI...[/green]")
        subprocess.run(
            [
                "git",
                "clone",
                COMFYUI_REPO,
                str(install_dir),
            ],
            check=True,
        )

    # Update ComfyUI
    rich.print("[green]Updating ComfyUI...[/green]")
    subprocess.run(
        ["git", "pull"],
        cwd=install_dir,
        check=True,
    )

    # Create and activate venv
    venv_dir = install_dir / ".venv"
    rich.print("[green]Creating virtual environment with uv...[/green]")
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    subprocess.run(
        ["uv", "venv", str(venv_dir)],
        check=True,
    )

    # Get python path for future use
    if sys.platform == "win32":
        python = str(venv_dir / "Scripts" / "python.exe")

    else:
        python = str(venv_dir / "bin" / "python")

    # Install requirements with uv
    rich.print("[green]Installing ComfyUI requirements with uv...[/green]")
    subprocess.run(
        ["uv", "pip", "install", "pip", "--upgrade"],
        env={
            "VIRTUAL_ENV": str(venv_dir),
            "PATH": str(venv_dir / "bin") + os.pathsep + os.environ["PATH"],
        },
        check=True,
    )
    subprocess.run(
        ["uv", "pip", "install", "-r", str(install_dir / "requirements.txt")],
        env={
            "VIRTUAL_ENV": str(venv_dir),
            "PATH": str(venv_dir / "bin") + os.pathsep + os.environ["PATH"],
        },
        check=True,
    )

    # Install comfy-pack as custom node
    rich.print("[green]Installing comfy-pack custom nodes...[/green]")
    custom_nodes_dir = install_dir / "custom_nodes"
    custom_nodes_dir.mkdir(exist_ok=True)

    comfyui_manager_dir = custom_nodes_dir / "ComfyUI-Manager"
    if not (comfyui_manager_dir / ".git").exists():
        # Clone ComfyUI-Manager
        subprocess.run(
            ["git", "clone", COMFYUI_MANAGER_REPO, str(comfyui_manager_dir)],
            check=True,
        )

    comfy_pack_dir = custom_nodes_dir / "comfy-pack"
    if not (comfy_pack_dir / ".git").exists():
        # Clone comfy-pack
        subprocess.run(
            ["git", "clone", COMFY_PACK_REPO, str(comfy_pack_dir)],
            check=True,
        )

    # Update comfy-pack
    subprocess.run(
        ["git", "pull"],
        cwd=comfy_pack_dir,
        check=True,
    )

    # Install comfy-pack requirements
    if (comfy_pack_dir / "requirements.txt").exists():
        subprocess.run(
            [
                python,
                "-m",
                "pip",
                "install",
                "-r",
                str(comfy_pack_dir / "requirements.txt"),
            ],
            check=True,
        )

    version = get_self_git_commit() or "unknown"
    rich.print(
        f"\n[green]✓ Installation completed! (comfy-pack version: {version})[/green]"
    )
    rich.print(f"ComfyUI directory: {install_dir}")

    rich.print(
        "\n[green]Next steps:[/green]\n"
        f"1. cd {dir}\n"
        "2. source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate\n"
        "3. python main.py"
    )


@main.command(
    name="unpack",
    help="Restore the ComfyUI workspace to specified directory",
)
@click.argument("cpack", type=click.Path(exists=True))
@click.option(
    "--dir",
    "-d",
    default="ComfyUI",
    help="target directory to restore the ComfyUI project",
    type=click.Path(file_okay=False),
)
@click.option(
    "--include-disabled-models",
    default=False,
    type=click.BOOL,
    is_flag=True,
)
@click.option(
    "--no-models",
    default=False,
    type=click.BOOL,
    is_flag=True,
    help="Do not install models",
)
@click.option(
    "--no-venv",
    is_flag=True,
    help="Do not create a virtual environment for ComfyUI",
    default=False,
)
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Increase verbosity level (use multiple times for more verbosity)",
)
@click.option(
    "--preheat",
    is_flag=True,
    help="Preheat the workspace after unpacking",
    default=False,
)
def unpack_cmd(
    cpack: str,
    dir: str,
    include_disabled_models: bool,
    no_models: bool,
    no_venv: bool,
    verbose: int,
    preheat: bool,
):
    import rich

    from .package import install

    install(
        cpack,
        dir,
        verbose=verbose,
        all_models=include_disabled_models,
        prepare_models=not no_models,
        no_venv=no_venv,
        preheat=preheat,
    )
    rich.print("\n[green]✓ ComfyUI Workspace is restored at:[/green]")
    rich.print(os.path.abspath(dir))
    steps = [f"Change directory to the restored workspace: `cd {dir}`"]
    if not no_venv:
        steps.append(
            "Source the virtual environment by running `source .venv/bin/activate`"
        )
    steps.append("Run the ComfyUI project by running `python main.py`")

    rich.print(f"\n[green]⏭️ Next steps: [/green]\n1. {steps[0]}\n2. {steps[1]}")
    if len(steps) > 2:
        rich.print(f"3. {steps[2]}")


def _print_schema(schema, verbose: int = 0):
    import rich
    from rich.table import Table

    table = Table(title="")

    # Add columns
    table.add_column("Input", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Required", style="yellow")
    table.add_column("Default", style="blue")
    table.add_column("Range", style="magenta")

    # Get required fields
    required = schema.get("required", [])

    # Add rows
    for field, info in schema["properties"].items():
        range_str = ""
        if "minimum" in info or "maximum" in info:
            min_val = info.get("minimum", "")
            max_val = info.get("maximum", "")
            range_str = f"{min_val} to {max_val}"

        table.add_row(
            field,
            info.get("format", "") or info.get("type", ""),
            "✓" if field in required else "",
            str(info.get("default", "")),
            range_str,
        )

    rich.print(table)


@functools.lru_cache
def _get_cache_workspace(cpack: str):
    sha = get_sha256(cpack)
    return WORKSPACE_DIR / sha[0:8]


@main.command(
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    },
    help="Run a ComfyUI package with the given inputs",
    add_help_option=False,
)
@click.argument("cpack", type=click.Path(exists=True, dir_okay=False))
@click.option("--output-dir", "-o", type=click.Path(), default=".")
@click.option("--help", "-h", is_flag=True, help="Show this message and input schema")
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Increase verbosity level (use multiple times for more verbosity)",
)
@click.pass_context
def run(ctx, cpack: str, output_dir: str, help: bool, verbose: int):
    import rich
    from pydantic import ValidationError

    from .utils import generate_input_model

    inputs = dict(
        zip([k.lstrip("-").replace("-", "_") for k in ctx.args[::2]], ctx.args[1::2])
    )

    with zipfile.ZipFile(cpack) as z:
        workflow = json.loads(z.read("workflow_api.json"))

    input_model = generate_input_model(workflow)

    # If help is requested, show command help and input schema
    if help:
        rich.print(
            'Usage: comfy-pack run [OPTIONS] CPACK --input1 "value1" --input2 "value2" ...'
        )
        rich.print("Run a ComfyUI package with the given inputs:")
        _print_schema(input_model.model_json_schema(), verbose)
        return 0

    try:
        validated_data = input_model(**inputs)
        rich.print("[green]✓ Input is valid![/green]")
        for field, value in validated_data.model_dump().items():
            rich.print(f"{field}: {value}")
    except ValidationError as e:
        rich.print("[red]✗ Validation failed![/red]")
        for error in e.errors():
            rich.print(f"- {error['loc'][0]}: {error['msg']}")

        rich.print("\n[yellow]Expected inputs:[/yellow]")
        _print_schema(input_model.model_json_schema(), verbose)
        return 1

    from .package import install

    workspace = _get_cache_workspace(cpack)
    if not (workspace / "DONE").exists():
        rich.print("\n[green]✓ Restoring ComfyUI Workspace...[/green]")
        if workspace.exists():
            shutil.rmtree(workspace)
        install(cpack, workspace, verbose=verbose)
        with open(workspace / "DONE", "w") as f:
            f.write("DONE")
    rich.print("\n[green]✓ ComfyUI Workspace is restored![/green]")
    rich.print(f"{workspace}")

    from .run import ComfyUIServer, run_workflow

    with ComfyUIServer(str(workspace.absolute()), verbose=verbose) as server:
        rich.print("\n[green]✓ ComfyUI is launched in the background![/green]")
        results = run_workflow(
            server.host,
            server.port,
            workflow,
            Path(output_dir).absolute(),
            verbose=verbose,
            workspace=server.workspace,
            **validated_data.model_dump(),
        )
        rich.print("\n[green]✓ Workflow is executed successfully![/green]")
        if results:
            rich.print("\n[green]✓ Retrieved outputs:[/green]")
        if isinstance(results, dict):
            for field, value in results.items():
                rich.print(f"{field}: {value}")
        elif isinstance(results, list):
            for i, value in enumerate(results):
                rich.print(f"{i}: {value}")
        else:
            rich.print(results)


@main.command(name="build-bento")
@click.argument("source")
@click.option("--name", help="Name of the bento service")
@click.option("--version", help="Version of the bento service")
def bento_cmd(source: str, name: str | None, version: str | None):
    """Build a bento from the source, which can be either a .cpack.zip file or a bento tag."""
    import bentoml
    from bentoml.bentos import BentoBuildConfig

    from .package import build_bento

    with tempfile.TemporaryDirectory() as temp_dir:
        if source.endswith(".cpack.zip"):
            name = name or os.path.basename(source).replace(".cpack.zip", "")
            shutil.unpack_archive(source, temp_dir)
            system_packages = None
            include_default_system_packages = True
        else:
            existing_bento = bentoml.get(source)
            name = name or existing_bento.tag.name
            shutil.copytree(existing_bento.path_of("src"), temp_dir, dirs_exist_ok=True)
            build_config = BentoBuildConfig.from_bento_dir(
                existing_bento.path_of("src")
            )
            requirements_txt = Path(temp_dir) / "requirements.txt"
            if (
                requirements_txt.exists()
                and "comfy-pack" not in requirements_txt.read_text()
            ):
                with open(requirements_txt, "a") as f:
                    f.write("\ncomfy-pack")
            system_packages = build_config.docker.system_packages
            include_default_system_packages = False

        build_bento(
            name,
            Path(temp_dir),
            version=version,
            system_packages=system_packages,
            include_default_system_packages=include_default_system_packages,
        )


def setup_cloud_client(
    ctx: click.Context, param: click.Parameter, value: str | None
) -> str | None:
    from bentoml._internal.configuration.containers import BentoMLContainer

    if value:
        BentoMLContainer.cloud_context.set(value)
        os.environ["BENTOML_CLOUD_CONTEXT"] = value
    return value


@main.command()
@click.argument("bento")
@click.option(
    "-w",
    "--workspace",
    type=click.Path(file_okay=False, path_type=Path),
    default="workspace",
    help="Workspace directory, defaults to './workspace'.",
)
@click.option("-v", "--verbose", count=True, help="Increase verbosity level")
@click.option(
    "--context",
    help="BentoCloud context name.",
    expose_value=False,
    callback=setup_cloud_client,
)
def unpack_bento(bento: str, workspace: Path, verbose: int):
    """Restore the ComfyUI workspace from a given bento."""
    import bentoml

    from .package import install

    try:
        bento_obj = bentoml.get(bento)
    except bentoml.exceptions.NotFound:
        click.echo(
            f"Bento {bento} not found in the local repository, trying to pull from BentoCloud",
            err=True,
        )
        bentoml.pull(bento)
        bento_obj = bentoml.get(bento)

    install(bento_obj.path_of("src"), workspace, verbose=verbose, prepare_models=False)

    if os.name == "nt":
        exe = "Scripts/python.exe"
    else:
        exe = "bin/python"
    click.echo(
        f"Workspace is ready at {workspace}\n"
        f"You can start ComfyUI by running `cd {workspace} && .venv/{exe} main.py`",
        color="green",
    )

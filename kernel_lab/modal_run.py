#!/usr/bin/env python3
"""Run a cleanCuda experiment on a selectable Modal GPU."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import shlex
import subprocess

import modal


LAB_DIR = Path(__file__).resolve().parent
REPO_DIR = LAB_DIR.parent
REMOTE_REPO = Path("/root/cleanCuda")
REMOTE_EXPERIMENTS = Path("/tmp/experiments")
CACHE_DIR = Path("/cache/binaries")

cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .add_local_dir(
        REPO_DIR,
        remote_path=str(REMOTE_REPO),
        ignore=[".git", ".kernel_lab", "venv", "__pycache__", "*.ipynb"],
    )
)

cache = modal.Volume.from_name("cleancuda-kernel-lab-cache", create_if_missing=True)
app = modal.App("cleancuda-kernel-lab", image=cuda_image)


def normalize_arch(value: str) -> str:
    lowered = value.lower().strip()
    for prefix in ("sm_", "compute_"):
        if lowered.startswith(prefix):
            lowered = lowered[len(prefix) :]
    if not lowered.isdigit():
        raise ValueError(f"invalid architecture {value!r}; expected auto, 90, sm_90, sm_100, sm_120, etc.")
    return f"sm_{lowered}"


def source_payload(source: Path) -> tuple[str, dict[str, bytes]]:
    source = source.expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".cu":
        raise ValueError(f"expected an existing .cu source file: {source}")
    root = source.parent
    files: dict[str, bytes] = {}
    for pattern in ("*.cu", "*.cuh", "*.h", "*.hpp"):
        for file in root.rglob(pattern):
            relative = file.relative_to(root).as_posix()
            files[relative] = file.read_bytes()
    return source.relative_to(root).as_posix(), files


@app.function(volumes={"/cache": cache}, timeout=30 * 60)
def compile_and_run(
    main_source: str,
    files: dict[str, bytes],
    arch: str,
    program_args: list[str],
    defines: list[str],
    nvcc_flags: list[str],
    debug: bool,
) -> dict[str, object]:
    payload_hash = hashlib.sha256()
    for relative, contents in sorted(files.items()):
        payload_hash.update(relative.encode())
        payload_hash.update(contents)
    experiment_dir = REMOTE_EXPERIMENTS / payload_hash.hexdigest()[:16]
    experiment_dir.mkdir(parents=True, exist_ok=True)
    for relative, contents in files.items():
        safe_path = PurePosixPath(relative)
        if safe_path.is_absolute() or ".." in safe_path.parts:
            raise ValueError(f"unsafe experiment path: {relative}")
        destination = experiment_dir / safe_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents)

    if arch == "auto":
        capability = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], text=True
        ).strip().splitlines()[0]
        selected_arch = normalize_arch(capability.replace(".", ""))
    else:
        selected_arch = normalize_arch(arch)

    gpu_name = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip().splitlines()[0]
    source = experiment_dir / main_source
    compile_command = [
        "nvcc",
        str(source),
        "-std=c++20",
        f"-arch={selected_arch}",
        "-lineinfo",
        "-O0" if debug else "-O3",
        "-I",
        str(REMOTE_REPO / "kernel_lab/include"),
        "-I",
        str(REMOTE_REPO),
        "-I",
        str(experiment_dir),
    ]
    if debug:
        compile_command.append("-G")
    compile_command.extend(f"-D{definition}" for definition in defines)
    compile_command.extend(nvcc_flags)

    identity = payload_hash.copy()
    identity.update("\0".join(compile_command).encode())
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    binary = CACHE_DIR / f"{source.stem}-{identity.hexdigest()[:16]}"
    compile_command.extend(["-o", str(binary)])

    compiled = False
    compile_output = ""
    if not binary.is_file():
        completed = subprocess.run(compile_command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        compile_output = completed.stdout
        if completed.returncode != 0:
            return {
                "returncode": completed.returncode,
                "gpu": gpu_name,
                "arch": selected_arch,
                "compile_command": shlex.join(compile_command),
                "compile_output": compile_output,
                "run_command": "",
                "stdout": "",
                "stderr": "",
                "cached": False,
            }
        compiled = True
        cache.commit()

    run_command = [str(binary), *program_args]
    completed = subprocess.run(run_command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {
        "returncode": completed.returncode,
        "gpu": gpu_name,
        "arch": selected_arch,
        "compile_command": shlex.join(compile_command),
        "compile_output": compile_output,
        "run_command": shlex.join(run_command),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "cached": not compiled,
    }


@app.local_entrypoint()
def main(
    source: str,
    gpu: str = "L40S",
    arch: str = "auto",
    args: str = "",
    define: str = "",
    nvcc_flags: str = "",
    debug: bool = False,
    verbose: bool = False,
) -> None:
    main_source, files = source_payload(Path(source))
    result = compile_and_run.with_options(gpu=gpu).remote(
        main_source,
        files,
        arch,
        shlex.split(args),
        shlex.split(define),
        shlex.split(nvcc_flags),
        debug,
    )
    print(f"gpu    : {result['gpu']} ({gpu})")
    print(f"arch   : {result['arch']}")
    print(f"compile: {'cached' if result['cached'] else 'built'}")
    if verbose or result["returncode"] != 0:
        print(f"nvcc   : {result['compile_command']}")
        if result["run_command"]:
            print(f"run    : {result['run_command']}")
    if result["compile_output"]:
        print(result["compile_output"], end="")
    if result["stdout"]:
        print(result["stdout"], end="")
    if result["stderr"]:
        print(result["stderr"], end="", file=__import__("sys").stderr)
    if result["returncode"] != 0:
        raise SystemExit(result["returncode"])

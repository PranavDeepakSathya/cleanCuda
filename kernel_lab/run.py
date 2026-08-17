#!/usr/bin/env python3
"""Compile and run a standalone CUDA experiment with cleanCuda's kernel lab."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


LAB_DIR = Path(__file__).resolve().parent
REPO_DIR = LAB_DIR.parent
DEFAULT_BUILD_DIR = REPO_DIR / ".kernel_lab" / "build"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Compile and run any standalone .cu file as a cleanCuda experiment.",
        epilog="Program arguments go after --, for example: run kernel.cu --arch sm_120 -- 4096",
    )
    result.add_argument("source", type=Path, help="path to a standalone CUDA source file")
    result.add_argument("--gpu", default="0", help="GPU index/UUID exposed to the experiment (default: 0)")
    result.add_argument(
        "--arch",
        default="auto",
        help="CUDA architecture: auto, 120, sm_120, compute_120, etc. (default: auto)",
    )
    result.add_argument("--nvcc", default=os.environ.get("NVCC", "nvcc"), help="nvcc executable")
    result.add_argument("--std", default="c++20", help="C++ language standard (default: c++20)")
    result.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR, help="binary cache directory")
    result.add_argument("-I", "--include", action="append", type=Path, default=[], help="additional include path")
    result.add_argument("-D", "--define", action="append", default=[], help="preprocessor definition")
    result.add_argument("--nvcc-flag", action="append", default=[], help="additional single nvcc flag")
    result.add_argument("--debug", action="store_true", help="compile with -O0 -G instead of -O3")
    result.add_argument("--force", action="store_true", help="ignore the cached binary")
    result.add_argument("--compile-only", action="store_true", help="compile but do not run")
    result.add_argument("--dry-run", action="store_true", help="print commands without compiling or running")
    result.add_argument("-v", "--verbose", action="store_true", help="print the compile and run commands")
    return result


def split_program_args(arguments: list[str]) -> tuple[list[str], list[str]]:
    if "--" not in arguments:
        return arguments, []
    separator = arguments.index("--")
    return arguments[:separator], arguments[separator + 1 :]


def normalize_arch(value: str) -> str:
    lowered = value.lower().strip()
    for prefix in ("sm_", "compute_"):
        if lowered.startswith(prefix):
            lowered = lowered[len(prefix) :]
    if not lowered.isdigit():
        raise ValueError(f"invalid architecture {value!r}; expected values such as 90, sm_90, or sm_120")
    return f"sm_{lowered}"


def detect_arch(gpu: str) -> str:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        raise RuntimeError("cannot auto-detect architecture: nvidia-smi is not available; pass --arch sm_XX")
    command = [nvidia_smi, "-i", gpu, "--query-gpu=compute_cap", "--format=csv,noheader"]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    capability = completed.stdout.strip().splitlines()[0].replace(".", "")
    return normalize_arch(capability)


def quote(command: list[str]) -> str:
    return shlex.join(command)


def main(raw_arguments: list[str] | None = None) -> int:
    cli_arguments, program_arguments = split_program_args(list(sys.argv[1:] if raw_arguments is None else raw_arguments))
    args = parser().parse_args(cli_arguments)
    source = args.source.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"source does not exist: {source}")
    if source.suffix.lower() != ".cu":
        raise SystemExit(f"expected a .cu source file: {source}")

    if args.arch == "auto":
        if args.dry_run and not shutil.which("nvidia-smi"):
            raise SystemExit("dry-run still needs an explicit --arch when nvidia-smi is unavailable")
        arch = detect_arch(args.gpu)
    else:
        try:
            arch = normalize_arch(args.arch)
        except ValueError as error:
            raise SystemExit(str(error)) from error

    nvcc = shutil.which(args.nvcc) or args.nvcc
    include_paths = [LAB_DIR / "include", REPO_DIR, source.parent, *args.include]
    compile_command = [
        nvcc,
        str(source),
        f"-std={args.std}",
        f"-arch={arch}",
        "-lineinfo",
        "-O0" if args.debug else "-O3",
    ]
    if args.debug:
        compile_command.append("-G")
    for include in include_paths:
        compile_command.extend(["-I", str(include.expanduser().resolve())])
    compile_command.extend(f"-D{definition}" for definition in args.define)
    compile_command.extend(args.nvcc_flag)

    identity = hashlib.sha256()
    identity.update(source.read_bytes())
    dependency_roots = [LAB_DIR / "include", source.parent]
    for root in dependency_roots:
        for pattern in ("*.cuh", "*.h", "*.hpp"):
            for header in sorted(root.rglob(pattern)):
                identity.update(str(header.resolve()).encode())
                identity.update(header.read_bytes())
    identity.update("\0".join(compile_command).encode())
    binary_name = f"{source.stem}-{identity.hexdigest()[:12]}"
    build_dir = args.build_dir.expanduser().resolve()
    binary = build_dir / binary_name
    compile_command.extend(["-o", str(binary)])

    run_environment = os.environ.copy()
    run_environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    run_command = [str(binary), *program_arguments]

    print(f"source : {source}")
    print(f"gpu    : {args.gpu}")
    print(f"arch   : {arch}")
    if args.verbose or args.dry_run:
        print(f"compile: {quote(compile_command)}")
        if not args.compile_only:
            print(f"run    : CUDA_VISIBLE_DEVICES={shlex.quote(args.gpu)} {quote(run_command)}")
    if args.dry_run:
        return 0

    if not shutil.which(args.nvcc) and not Path(args.nvcc).is_file():
        raise SystemExit(f"nvcc is not available: {args.nvcc}")
    build_dir.mkdir(parents=True, exist_ok=True)
    if args.force or not binary.is_file():
        print(f"building {binary.name}")
        subprocess.run(compile_command, check=True)
    else:
        print(f"cached  : {binary}")
    if args.compile_only:
        return 0
    print(f"running : {binary.name}\n")
    return subprocess.run(run_command, env=run_environment).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error

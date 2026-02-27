#!/usr/bin/env python
"""Create a runnable Ollama model from a merged Pranav profile model."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


SYSTEM_PROMPT = """You are Pranav's personal robotics and engineering assistant.
- Prioritize practical, step-by-step help for FTC/FRC, sim racing wheel builds, CAD, and embedded systems.
- Keep answers concise, specific, and implementation-first.
- When details are missing, ask short clarifying questions before giving final wiring or code changes.
- Favor safe hardware advice and verify assumptions.
"""


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(">", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def has_safetensors(model_dir: Path) -> bool:
    return any(model_dir.glob("*.safetensors"))


def detect_input_dir(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not has_safetensors(p):
            raise FileNotFoundError(
                f"No .safetensors files found in --input-dir: {p}"
            )
        return p

    candidates = [
        Path("outputs/pranav_gguf"),
        Path("outputs/pranav_8gb/pranav_merged_16bit"),
        Path("outputs/pranav_8gb/pranav_merged"),
    ]
    for c in candidates:
        if has_safetensors(c):
            return c
    raise FileNotFoundError(
        "Could not find a merged model folder. Expected .safetensors in one of:\n"
        "- outputs/pranav_gguf\n"
        "- outputs/pranav_8gb/pranav_merged_16bit\n"
        "- outputs/pranav_8gb/pranav_merged"
    )


def ensure_llama_cpp(llama_cpp_dir: Path) -> Path:
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if convert_script.exists():
        return convert_script

    git_bin = shutil.which("git")
    if git_bin is None:
        raise RuntimeError("git is required to clone llama.cpp, but git was not found.")

    llama_cpp_dir.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            git_bin,
            "clone",
            "--depth",
            "1",
            "https://github.com/ggml-org/llama.cpp",
            str(llama_cpp_dir),
        ]
    )
    if not convert_script.exists():
        raise FileNotFoundError(f"Missing converter after clone: {convert_script}")
    return convert_script


def write_modelfile(modelfile: Path, gguf_name: str) -> None:
    modelfile.write_text(
        "\n".join(
            [
                f"FROM ./{gguf_name}",
                "",
                "PARAMETER temperature 0.6",
                "PARAMETER top_p 0.9",
                "PARAMETER repeat_penalty 1.1",
                "PARAMETER num_ctx 8192",
                "",
                f'SYSTEM """{SYSTEM_PROMPT}"""',
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert merged HF model to GGUF and create an Ollama model."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Merged Hugging Face model directory containing .safetensors files.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="pranav-assistant",
        help="Target Ollama model name.",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default="q4_K_M",
        help="Quantization passed to `ollama create --quantize`.",
    )
    parser.add_argument(
        "--llama-cpp-dir",
        type=Path,
        default=Path("runtime/llama.cpp"),
        help="Local llama.cpp checkout path.",
    )
    parser.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help="Python executable to run convert_hf_to_gguf.py.",
    )
    parser.add_argument(
        "--force-convert",
        action="store_true",
        help="Rebuild GGUF even if already present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if shutil.which("ollama") is None:
        raise RuntimeError("ollama CLI not found in PATH.")

    model_dir = detect_input_dir(args.input_dir)
    out_gguf = model_dir / "pranav-assistant-f16.gguf"
    modelfile = model_dir / "Modelfile"

    if args.force_convert or not out_gguf.exists():
        convert_script = ensure_llama_cpp(args.llama_cpp_dir)
        run(
            [
                args.python_exe,
                str(convert_script),
                str(model_dir),
                "--outfile",
                str(out_gguf),
                "--outtype",
                "f16",
            ]
        )
    else:
        print(f"Using existing GGUF: {out_gguf}")

    write_modelfile(modelfile, out_gguf.name)

    run(
        [
            "ollama",
            "create",
            args.model_name,
            "-f",
            str(modelfile),
            "--quantize",
            args.quantize,
        ]
    )

    print("")
    print("Model is ready in Ollama.")
    print(f"Run: ollama run {args.model_name}")


if __name__ == "__main__":
    main()

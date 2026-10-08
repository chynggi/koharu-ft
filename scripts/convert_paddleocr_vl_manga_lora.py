"""Convert the PaddleOCR-VL-1.6 manga LoRA into the GGUF files Koharu reads.

`sorryhyun/paddleocr-vl-1.6-manga-lora` publishes two safetensors files:

* `adapter_model.safetensors` — a peft LoRA over the language model
  (18 layers x q/k/v/o/gate/up/down, r16 alpha32). `convert_lora_to_gguf.py`
  consumes it as-is: `TextModel.filter_tensors` drops the Transformers
  `language_model.` segment that this repository's torch adapter also strips.
* `tower.safetensors` — **not a LoRA**. A full fine-tune of the NaViT vision
  tower and projector. llama.cpp LoRA adapters only cover the LM, so the tower
  has to be baked into a replacement mmproj instead.

The tower conversion is where this silently goes wrong. `convert_hf_to_gguf.py
--mmproj` loads tensors through `PaddleOCRVisionModel.filter_tensors`, which
keeps only names containing `vision_model` or `mlp_AR`. Transformers names the
projector `model.projector.*`, which matches neither and is dropped without a
warning. The rename below maps the tower onto the checkpoint names the
converter expects. A successful conversion produces a mmproj whose tensor set
is identical to the stock `PaddleOCR-VL-1.6-GGUF-mmproj.gguf` (437 vision +
6 projector = 443 = the tensor count of `tower.safetensors`).

Usage:
    python scripts/convert_paddleocr_vl_manga_lora.py --output-dir <dir> \\
        [--llama-cpp <path>] [--keep-intermediate]

Requires `safetensors`, `torch` (CPU is enough), and `huggingface_hub`, plus a
llama.cpp checkout whose `convert_hf_to_gguf.py` registers the `paddleocr`
architecture (merged in llama.cpp PR #18825). Check out the release Koharu
pins in crates/koharu-runtime/src/runtime/packages/llama.rs before converting.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

ADAPTER = "sorryhyun/paddleocr-vl-1.6-manga-lora"
ADAPTER_REVISION = "26292839d1469c14212a12a1e01b5b1fe01bff15"
BASE = "PaddlePaddle/PaddleOCR-VL-1.6"
BASE_REVISION = "66317acc4c9fc17bd154591ce650735cd2855f3e"

TOWER_WEIGHTS = "tower.safetensors"
ADAPTER_WEIGHTS = "adapter_model.safetensors"

# Transformers attribute path -> checkpoint path. Matches
# crates/koharu-ml/src/paddle_ocr_vl/adapter.rs::tower_target.
TOWER_PREFIXES = (
    ("model.visual.", "visual."),
    ("model.projector.", "mlp_AR."),
)

# The stock mmproj and the replacement must agree on the tensor set. The base
# vision model ships 449 tensors; the converter drops `head` (11) and
# `packing_position_embedding` (1), and the tower contributes 6 projector
# tensors, so 443 remain.
EXPECTED_MMREF_TENSORS = 443


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--llama-cpp",
        type=Path,
        default=Path("llama.cpp"),
        help="llama.cpp checkout containing convert_hf_to_gguf.py and "
        "convert_lora_to_gguf.py",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="keep the merged HF directory the mmproj is built from",
    )
    return parser.parse_args()


def tower_target(key: str) -> str:
    for source, target in TOWER_PREFIXES:
        if key.startswith(source):
            return target + key[len(source) :]
    raise SystemExit(f"tower weight is outside the vision tower and projector: {key}")


def merge_tower(base_dir: Path, output_dir: Path) -> Path:
    """Copy the base checkpoint and overwrite the vision tower weights."""
    merged = output_dir / "merged-hf"
    if merged.exists():
        shutil.rmtree(merged)
    shutil.copytree(base_dir, merged)

    tower_path = snapshot_download(
        ADAPTER, revision=ADAPTER_REVISION, allow_patterns=[TOWER_WEIGHTS]
    )
    tower = load_file(Path(tower_path) / TOWER_WEIGHTS)

    renamed = {tower_target(name): value for name, value in tower.items()}
    for path in merged.glob("*.safetensors"):
        weights = load_file(path)
        overlapped = set(weights) & set(renamed)
        if not overlapped:
            continue
        weights.update({name: renamed[name] for name in overlapped})
        save_file(weights, path)
        print(f"  replaced {len(overlapped)} tower tensors in {path.name}")

    missing = set(renamed) - set().union(
        *(set(load_file(path)) for path in merged.glob("*.safetensors"))
    )
    if missing:
        raise SystemExit(
            f"{len(missing)} tower tensors matched nothing in the base checkpoint, "
            f"e.g. {sorted(missing)[:3]}"
        )
    return merged


def convert_mmproj(llama_cpp: Path, merged: Path, output: Path) -> None:
    command = [
        sys.executable,
        str(llama_cpp / "convert_hf_to_gguf.py"),
        str(merged),
        "--mmproj",
        "--outtype",
        "bf16",
        "--outfile",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr[-4000:])
        raise SystemExit(f"convert_hf_to_gguf.py --mmproj failed with {result.returncode}")
    print(f"  converted -> {output.name} ({output.stat().st_size / 1e9:.2f} GB)")


def convert_lora(llama_cpp: Path, base_dir: Path, output: Path) -> None:
    adapter_dir = Path(
        snapshot_download(
            ADAPTER,
            revision=ADAPTER_REVISION,
            allow_patterns=[ADAPTER_WEIGHTS, "adapter_config.json"],
        )
    )
    command = [
        sys.executable,
        str(llama_cpp / "convert_lora_to_gguf.py"),
        str(adapter_dir),
        "--base",
        str(base_dir),
        "--outtype",
        "f32",
        "--outfile",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr[-4000:])
        raise SystemExit(f"convert_lora_to_gguf.py failed with {result.returncode}")
    print(f"  converted -> {output.name} ({output.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    arguments = parse_arguments()
    output_dir = arguments.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("downloading the base checkpoint")
    base_dir = Path(
        snapshot_download(BASE, revision=BASE_REVISION, allow_patterns=["*.json", "*.safetensors"])
    )

    print("merging the fine-tuned vision tower")
    merged = merge_tower(base_dir, output_dir)

    print("converting the replacement mmproj")
    mmproj = output_dir / "manga-mmproj.gguf"
    convert_mmproj(arguments.llama_cpp, merged, mmproj)

    print("converting the LoRA adapter")
    lora = output_dir / "manga-lora.gguf"
    convert_lora(arguments.llama_cpp, base_dir, lora)

    if not arguments.keep_intermediate:
        shutil.rmtree(merged)

    print("\ndone. upload both files to a dedicated `-GGUF` repository")
    print("(e.g. Essid/paddleocr-vl-1.6-manga-lora-GGUF), then pin that repo in")
    print("crates/koharu-ml/src/paddle_ocr_vl_quantized/mod.rs. Local paths work too:")
    print(f"  lora      {lora}")
    print(f"  projector {mmproj}")


if __name__ == "__main__":
    main()

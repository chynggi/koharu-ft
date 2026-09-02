"""Convert PowerPaint V1 into the files Koharu's PowerPaint inpainter reads.

PowerPaint V1 is stock SD1.5-inpainting plus thirty extra token embedding rows:
every other tensor in its text encoder is bit-identical to the base model. So
the task prompts are pure textual inversion, and the conversion is a split
rather than a translation — the rows come out as standalone embeddings and the
rest of the model converts as an ordinary SD1.5 inpainting checkpoint.

Two details of stable-diffusion.cpp decide the tensor names used below.
`convert_tensors_name()` calls `get_sd_version()` first, and that detection only
recognises a UNet under `unet.down_blocks.` or `model.diffusion_model.
input_blocks.`, alongside a token embedding under one of a handful of exact
names. Routing the UNet through `--diffusion-model` prefixes it with
`model.diffusion_model.`, which matches neither pattern, so SD1 name conversion
is skipped and the resulting file has no detectable version. Merging the UNet
under `unet.` and the text encoder under `cond_stage_model.transformer.` — the
prefix the conditioner later looks the tensors up by — satisfies detection and
conversion at once.

Usage:
    python scripts/convert_powerpaint.py --output-dir <dir> [--sd-cli <path>]

Requires `safetensors`, `numpy`, and `huggingface_hub`, plus the `sd-cli`
binary from a stable-diffusion.cpp release matching the one Koharu pins in
crates/koharu-runtime/src/runtime/packages/diffusion.rs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file, save_file

POWERPAINT = "Sanster/PowerPaint-V1-stable-diffusion-inpainting"
BASE = "stable-diffusion-v1-5/stable-diffusion-inpainting"
BASE_VOCAB = 49408

UNET = "unet/diffusion_pytorch_model.fp16.safetensors"
VAE = "vae/diffusion_pytorch_model.fp16.safetensors"
TEXT_ENCODER = "text_encoder/model.fp16.safetensors"
TOKEN_EMBEDDING = "text_model.embeddings.token_embedding.weight"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sd-cli",
        type=Path,
        default=Path("sd-cli"),
        help="stable-diffusion.cpp CLI used for the GGUF conversion",
    )
    parser.add_argument(
        "--type",
        default="f16",
        help="output tensor type passed to sd-cli (default: f16)",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="keep the ~2 GB merged safetensors the GGUF is built from",
    )
    return parser.parse_args()


def extract_task_embeddings(output_dir: Path) -> None:
    """Split the thirty PowerPaint rows into one embedding file per task."""
    added = json.load(open(hf_hub_download(POWERPAINT, "tokenizer/added_tokens.json")))
    table = load_file(hf_hub_download(POWERPAINT, "text_encoder/model.safetensors"))[
        TOKEN_EMBEDDING
    ]
    if min(added.values()) != BASE_VOCAB:
        raise SystemExit(
            f"expected added tokens to start at {BASE_VOCAB}, got {min(added.values())}"
        )

    groups: dict[str, list[tuple[int, int]]] = {}
    for token, token_id in added.items():
        prefix, index = token.rsplit("_", 1)
        groups.setdefault(prefix, []).append((int(index), token_id))

    embeddings = output_dir / "embeddings"
    embeddings.mkdir(parents=True, exist_ok=True)
    for prefix, entries in sorted(groups.items()):
        entries.sort()
        rows = table[[token_id for _, token_id in entries]].astype(np.float32)
        save_file({"emb_params": rows}, embeddings / f"{prefix}.safetensors")
        print(f"  {prefix:8s} {rows.shape} -> embeddings/{prefix}.safetensors")


def merge_main_model(output_dir: Path) -> Path:
    """Write one file whose tensor names stable-diffusion.cpp can detect."""
    unet = load_file(hf_hub_download(POWERPAINT, UNET))
    clip = load_file(hf_hub_download(BASE, TEXT_ENCODER))

    in_channels = unet["conv_in.weight"].shape[1]
    if in_channels != 9:
        raise SystemExit(f"expected a 9-channel inpainting UNet, got {in_channels}")

    merged = {f"unet.{name}": value for name, value in unet.items()}
    merged.update(
        {f"cond_stage_model.transformer.{name}": value for name, value in clip.items()}
    )

    path = output_dir / "powerpaint-main.safetensors"
    save_file(merged, path)
    print(f"  merged {len(merged)} tensors -> {path.name}")
    return path


def convert_to_gguf(sd_cli: Path, main: Path, output: Path, tensor_type: str) -> None:
    command = [
        str(sd_cli),
        "-M", "convert",
        "--convert-name",
        "-m", str(main),
        "--vae", hf_hub_download(POWERPAINT, VAE),
        "--type", tensor_type,
        "-o", str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr[-4000:])
        raise SystemExit(f"sd-cli convert failed with code {result.returncode}")
    print(f"  converted -> {output.name} ({output.stat().st_size / 1e9:.2f} GB)")


def main() -> None:
    arguments = parse_arguments()
    output_dir = arguments.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("extracting task embeddings")
    extract_task_embeddings(output_dir)

    print("merging the main model")
    main_model = merge_main_model(output_dir)

    print("converting to GGUF")
    gguf = output_dir / "powerpaint-v1.gguf"
    convert_to_gguf(arguments.sd_cli, main_model, gguf, arguments.type)

    if not arguments.keep_intermediate:
        main_model.unlink()

    print(f"\ndone. point Koharu's PowerPaint model at {gguf}")
    print(f"and its embeddings directory at {output_dir / 'embeddings'}")


if __name__ == "__main__":
    main()

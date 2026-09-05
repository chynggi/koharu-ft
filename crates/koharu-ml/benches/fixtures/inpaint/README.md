# LaMa 4K benchmark fixtures

- `image_4k.jpg` is the 3840-pixel Wikimedia rendition of *Landscape*,
  attributed to Kanō Motonobu and contributed by the Metropolitan Museum of
  Art. Wikimedia Commons publishes the file under CC0 1.0.
  Source: <https://commons.wikimedia.org/wiki/File:MET_LC-51_95_002.jpg>
- `mask_4k.png` is a nearest-neighbor resize to 3840×2074 of `mask_1.png` from
  the Apache-2.0-licensed `enesmsahin/simple-lama-inpainting` test fixtures.
  Source: <https://github.com/enesmsahin/simple-lama-inpainting/blob/main/tests/data/mask_1.png>

SHA-256:

- `image_4k.jpg`: `1b6c0a50f8a4a5101d745bda7ee311abb3f4ee011433c71f83e71a9f718eec7a`
- `mask_4k.png`: `53f772430d81b5ded3f0b243641b5595d921c37da3fd8f026036090a98d5bb57`

## Measured: LaMa vs MI-GAN vs Manga inpainter (2026-08-22)

All benchmarks ran on the same CPU device (`koharu_ml::Device::default()`)
with the pinned builtin checkpoints, on:

- CPU: Intel Core i5-10400
- GPU: NVIDIA GeForce RTX 3060 (present but not used by these runs)
- libtorch 2.12.1 (CPU build), criterion, 10 samples

| Model | Input | Median |
|---|---|---|
| LaMa (safetensors) | 3840×2074 + mask | 11.919 s |
| MI-GAN (TorchScript) | 3840×2074 + mask | 821.78 ms |
| Manga inpainter (TorchScript ×2) | 3840×2074 + mask | 65.232 s |

MI-GAN was ~14.5× faster than LaMa on this fixture on CPU, confirming the
"fast and light erase-only model" claim. The speedup is structural: MI-GAN
only ever infers 512×512 crops, while LaMa pads and infers at full modulo-8
resolution.

The Manga inpainter is the slowest of the three: both its line model and its
inpaintor run over the full padded resolution, with no crop or resize escape
hatch on this fixture. Pick it for manga line-art quality, not speed.

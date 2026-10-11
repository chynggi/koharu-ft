# 12GB 타깃 인페인팅 모델 조사

작성일: 2026-10-11

`specs/2026-10-11-flux2-klein-variant-design.md`의 열린 항목이던 "FLUX.2 klein의
역할을 하면서 GGUF로 12GB VRAM에 확실히 fit하는 모델 + quant 선정" 조사의 기록이다.
이 문서는 결정의 근거를 보존하기 위한 것이며, 구현 계획의 대상이 아니다.

**필터 기준:** 12GB에 여유 있게 fit(경계형 제외), 품질을 해치는 저비트 회피.
CPU offload는 성능상 배제(가중치 전부 GPU 상주 기준).

## 1. stable-diffusion.cpp의 최신 지원

출처: [README](https://github.com/leejet/stable-diffusion.cpp),
[docs/flux2.md](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/flux2.md),
[docs/qwen_image_2.1.md](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/qwen_image_2.1.md),
[docs/z_image.md](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/z_image.md)

| 날짜 | 추가 지원 |
|---|---|
| 2026/09/20 | Qwen-Image-2.1 Day-0 지원 |
| 2026/08/20 | LTX-2.5 |
| 2026/08/04 | MiniMax-H3 (영상) |
| 2026/06~05 | Krea2, Ideogram4, PiD, Lens, LTX-2.3 |
| 2026/01/18 | FLUX.2-klein |
| 2025/12/01 | Z-Image |
| 2025/11/30 | FLUX.2-dev |
| 2025/10 | Qwen-Image / Qwen-Image-Edit (2509) |

핵심 확인:

- **FLUX.2 klein 9B 변환기 지원이 공식 문서화되어 있다.** `docs/flux2.md`에
  "Flux.2 klein 9B / base 9B" 항목과 `leejet/FLUX.2-klein-9B-GGUF` + Qwen3-8B
  텍스트 인코더 + `-r` 편집 예제가 있다.
- GGUF 로드는 모든 아키텍처에서 지원된다 (ckpt / safetensors / GGUF).
- Qwen-Image-2.1은 생성과 편집을 모두 지원하지만, 편집 시 mmproj(비전 타워,
  1.16GB)가 추가로 필요하다.
- 지시문 기반 편집(`-r` + 프롬프트)은 공식 문서화되어 있다. 순수 마스크 고정
  인페인팅(마스크 밖 픽셀 보존) 경로의 지원 범위는 문서상 명확하지 않다.

## 2. 후보별 구성과 VRAM

파일 크기는 Hugging Face API 실측값. "총 VRAM 추정"은 가중치 합 + 활성화/버퍼
(1024px 기준 1.5–3GB) **추정치**다.

| 모델 | 변환기 quant (크기) | 텍스트 인코더 (크기) | VAE (크기) | 가중치 합 | 총 VRAM 추정 | 12GB 적합 | 라이선스 |
|---|---|---|---|---|---|---|---|
| **FLUX.2-klein-4B (현행)** | Q4_K_M 2.60GB | Qwen3-4B Q4_K_M 2.50GB | small-decoder 0.25GB | 5.35GB | ~7–8GB | ✅ 여유 있음 | **Apache 2.0** |
| FLUX.2-klein-4B (상향) | Q6_K 3.41GB | Qwen3-4B Q5_K_M 2.89GB | 동일 | 6.55GB | ~8.5–9.5GB | ✅ 여유 있음 | Apache 2.0 |
| FLUX.2-klein-4B (최고) | Q8_0 4.30GB | Qwen3-4B Q8_0 4.28GB | 동일 | 8.83GB | ~10.5–11GB | ⚠️ 여유 축소 | Apache 2.0 |
| **FLUX.2-klein-9B** | Q4_K_M 5.91GB | Qwen3-8B Q4_K_M 5.03GB | 0.25–0.68GB | ~11.2GB | **~13GB+** | ❌ 초과 | **FLUX 비상용** |
| FLUX.2-klein-9B (저quant) | Q3_K_M 4.77GB | Qwen3-8B Q3_K_M 4.12GB | 동일 | ~9.1GB | ~11GB (경계) | ⚠️ 품질 미달 | FLUX 비상용 |
| **Qwen-Image-2.1** | Q4_K_M 4.20GB | Qwen3-VL-8B UD-Q4_K_XL 5.15GB (편집 시 mmproj +1.16GB) | vae 0.68GB | 9.91GB (편집 11.07GB) | ~11–12GB+ | ⚠️ 경계 (1024px·배치1 한정) | **Qwen Research (연구 전용)** |
| Qwen-Image-2.1 (상향) | Q5_K_M 5.39GB | 동일 | 동일 | 11.22GB | ~13GB+ | ❌ 초과 | 연구 전용 |
| Z-Image-Turbo | Q6_K 5.26GB | Qwen3-4B-Instruct Q4_K_M ~2.5GB | FLUX.1-schnell ae ~0.33GB | ~8.1GB | ~10GB | ✅ | Apache 2.0 |
| Qwen-Image (20B) | Q4_K_M 13.07GB | Qwen2.5-VL-7B Q4_K_M ~4.4GB | vae ~0.3GB | 17GB+ | ❌ | ❌ (Q2_K에서도 초과) | — |
| SDXL 인페인팅 | Q4_K_M ~3.5GB (추정) | CLIP-L + OpenCLIP ~1.4GB | 0.3GB | ~5GB | ~6–7GB | ✅ | OpenRAIL++-M |

출처: [unsloth/FLUX.2-klein-4B-GGUF](https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF),
[unsloth/FLUX.2-klein-9B-GGUF](https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF),
[unsloth/Qwen-Image-2.1-GGUF](https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF),
[unsloth/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen3-VL-8B-Instruct-GGUF),
[leejet/Z-Image-Turbo-GGUF](https://huggingface.co/leejet/Z-Image-Turbo-GGUF),
[QuantStack/Qwen-Image-GGUF](https://huggingface.co/QuantStack/Qwen-Image-GGUF),
[black-forest-labs/FLUX.2-small-decoder](https://huggingface.co/black-forest-labs/FLUX.2-small-decoder)

**전제 정정:** "9B Q4는 약 16GB"라는 통념은 부정확하다. 그 수치는 BF16 변환기
(18.16GB)나 FP8 전체 파이프라인(~15GB)에서 나온 값이고, **Q4_K_M 변환기 실측은
5.91GB**다. 결론은 달라지지 않는다 — 텍스트 인코더(Qwen3-8B)와 VAE까지 더한
가중치가 ~11.2GB여서 활성화/KV 포함 시 12GB를 초과한다.

**커뮤니티 실측:** Qwen-Image-2.1은 Q4 GGUF 스택 9.91GB로 12GB 카드에서
1024px·배치1 동작이 확인된다. Unsloth 문서도 12–16GB VRAM에 GGUF Q4_K_M,
1024×1024, batch 1을 권장한다. 다만 참조 이미지 2장 편집 시 피크가
21.3→25.6GB로 상승하는 등 **편집은 생성보다 메모리를 더 쓴다**.

## 3. 저비트 품질 손실

Unsloth 공식 LPIPS 측정 (Qwen-Image-2.1-Turbo, bf16 대비, 낮을수록 양호)
— [출처](https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF):

| quant | LPIPS |
|---|---|
| Q2_K | 0.374 |
| Q3_K_M | 0.261 |
| **Q4_K_M** | **0.153** |
| Q5_K_M | 0.074 |
| Q6_K | 0.055 |
| Q8_0 | 0.036 |

- 2–3bit는 8스텝에서 눈에 띄게 이탈한다는 것이 배포자의 권고다.
- 커뮤니티 비교에서도 **Q4_K_M 아래로 내려가면 품질이 눈에 띄게 떨어지며**,
  특히 "사람 지우기"(마스크 기반 제거)에서 저하가 명확하다. Q4_0은 디테일 손실
  보고가 있다. → **Q4_K_M이 최저 합리선.**
- **혼합 포맷 주의:** GGUF 변환기 + 비-GGUF 텍스트 인코더 조합에서 아티팩트가
  보고된다 ([ComfyUI #9481](https://github.com/comfyanonymous/ComfyUI/issues/9481)).
  변환기와 텍스트 인코더를 모두 GGUF로 통일하는 것이 안전하다.

## 4. 라이선스

| 모델 | 라이선스 | 상용 |
|---|---|---|
| FLUX.2-klein-4B / base-4B | Apache 2.0 | ✅ |
| FLUX.2-klein-9B / base-9B | FLUX Non-Commercial | ❌ |
| Qwen-Image-2.1 / Turbo | Qwen Research License | ❌ (별도 유료 라이선스 필요) |
| Z-Image / Turbo | Apache 2.0 | ✅ |
| SDXL 인페인팅 파생 | OpenRAIL++-M | ✅ (사용 제한 조항) |

Qwen-Image-2.1은 출력물의 상업적 이용은 가능하다는 커뮤니티 해석이 있으나,
라이선스 원문은 "별도 상용 라이선스 없이 Materials를 상업적 목적으로 사용 불가"로
명시한다. **앱에 모델을 탑재·배포하는 것은 상업적 사용에 해당할 가능성이 높다.**

## 5. 결론

1. **12GB 챔피언은 현행 FLUX.2-klein-4B 스택이다.** 총 ~7–8GB로 여유가 크고
   Apache 2.0이다. 여유 예산 안에서 품질을 올리려면 변환기 Q6_K + 텍스트 인코더
   Q5_K_M(~8.5–9.5GB)이 유효한 상향선이다. Q4_K_M은 저비트 기준선이므로 유지해도
   품질 기준을 위반하지 않는다.
2. **Qwen-Image-2.1은 기각한다.** 12GB에서 경계형(편집 시나리오에서 초과 가능)이고,
   연구 전용 라이선스가 상용 배포를 막는다.
3. **FLUX.2 klein 9B는 12GB 타깃에 부적합하다.** 가중치 ~11.2GB로 초과하고,
   품질 기준(Q4_K_M 이상)을 지키면서 낮출 여지가 없으며, 비상용 라이선스다.
   (16GB+ 사용자용 선택지로는 유효하다.)
4. Z-Image-Turbo는 T2I 전용이라 텍스트 제거·배경 재구성 역할을 대체할 수 없다.
   Qwen-Image 20B는 애초에 초과, SDXL 인페인팅은 품질이 구세대다.

## 확인 필요

1. sd.cpp에서 FLUX.2-klein 대상 **순수 마스크 고정 인페인팅**(마스크 밖 픽셀 보존)
   경로의 정확한 지원 범위 — 현행 4B가 쓰는 경로의 문서 대조.
2. Qwen-Image-2.1 출력물 상업적 이용 범위의 라이선스 원문 해석(법무 검토).
3. FLUX.2 klein 4B GGUF 스택의 12GB 카드 실측 피크 VRAM.

# Adapter validation record

Test host: Apple Silicon M5 Pro, 48 GB unified memory, Metal. This is an
engineering smoke test, not a generative-quality benchmark or a hardware minimum.

## Real adapter checks

| Base package | Adapter | Generation |
| --- | --- | --- |
| ACE-Step 1.5 Turbo BF16 | [ACE-Step Chinese New Year LoRA](https://huggingface.co/ACE-Step/ACE-Step-v1.5-chinese-new-year-LoRA/tree/cb829a12775740c830a6d49795f16913065dc492), PEFT rank 64, alpha 128 | 8 s, stereo 48 kHz, 4 diffusion steps, seed 1234 |
| Stable Audio 3 Medium Q8_0 | [worstplayer Goa SA3M](https://huggingface.co/worstplayer/SA3M_AS15XL_oldschool_goa_trance_LoRA/tree/ea8a0695dbdf93a0e2bc8d4ced2b83fa9dba5dc4), native rank 16, alpha 16 | 4 s, stereo 44.1 kHz, 4 diffusion steps, seed 1234 |

Base packages came from `audio-cpp/audio.cpp-gguf` revision
`1a73167f64a9bf0d280f2223a8cbbccb8273ef05`. Adapter files and model weights are
not included in this repository or any app distribution. The ACE example has
research/noncommercial restrictions in its model card; downloading it for this
check is not a license recommendation for shipping it.

CLI tests passed for absent adapter, strength 0, 0.5, and 1. For each family,
zero-strength WAV bytes exactly matched the absent-adapter WAV. Half/full
strength each produced different output. All files had the requested sample
count and channel count. Measured whole-process wall times were approximately
20–22 seconds for the ACE-Step cases and 3–6 seconds for Stable Audio; these are
short fixed-step smoke runs, not production quality/latency promises.

The `tests/lora/c_api_smoke.py` integration check exercises the same public API
used by a native plugin worker, including two sessions on one loaded model.
Reports in `tests/lora/evidence/` record raw float-audio hashes, duration, peak,
and elapsed time for baseline, enabled, reused, base-after-enabled, and disabled
runs. Exact hashes are only assertions within this host/backend; other GPU
backends are not required to generate identical audio.

The native Goa adapter's card names **Medium Base**, while the available GGUF
used for this test is **Medium**. Its successful run establishes tensor routing
and adapter application, not training-checkpoint identity or listening quality.
A second Stable Audio integration run uses the [Maqam PEFT adapter](https://huggingface.co/motiftechnologies/stable-audio-3-maqam-lora/tree/3e1d9aa6fcb72a619b4ced00a240c5039f76daf0),
whose config identifies the Medium checkpoint. All five C API checks also pass
for that adapter (rank 64, alpha 128); see `stable-audio-peft-metal.json`.

A deliberate Medium-on-Small attempt fails with a tensor shape mismatch before
inference, rather than applying a partial adapter.

## Coverage boundaries

- LoRA/DoRA math, legacy DoRA magnitude layout, malformed-file rejection,
  mapping/shape checks, and quantized backend uploads have synthetic unit coverage.
- Real downloaded adapters above are **ordinary LoRA**; no real trained DoRA
  listening test has been completed.
- Stable Audio Small Music/SFX use the same loader and exact-shape checks; this
  record does not establish compatibility with every online adapter. The real
  Stable Audio adapter used here is specifically for Medium.
- CUDA inference and Windows DAW/plugin integration are not validated by these
  Mac tests. The fork has CPU adapter-test CI for Linux, macOS, and Windows.
- LoRA-XS, BoRA, LoKR, and per-step strength schedules are
  explicitly unsupported in this first implementation.

## Ordered stacking

The native adapter unit test additionally covers two overlapping LoRA entries
with independent strengths, exact empty/all-zero bypass, repeated reads, base
isolation, stack limits/malformed fields, and mixed LoRA/DoRA stacks in both
orders against independently calculated expected weights. These tests pass on
Metal-capable macOS locally; real stacked inference evidence is tracked by the
VST.cpp integration tests. This does not establish trained DoRA audio quality.

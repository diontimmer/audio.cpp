# Stable Audio 3 and ACE-Step LoRA adapters

This fork adds native, session-scoped LoRA loading for Stable Audio 3 and
ACE-Step 1.5. Python and model conversion are not required at inference time.
Base model files are never modified. Up to eight adapters can be selected per session.

## Supported formats

| Family | Adapter | Scope |
| --- | --- | --- |
| ACE-Step 1.5 | PEFT directory containing `adapter_model.safetensors` and `adapter_config.json`, or the SafeTensors file with its sibling config | DiT decoder; bare `layers.*` PEFT targets map to `decoder.layers.*` |
| Stable Audio 3 | Native SafeTensors with JSON `lora_config` metadata | DiT Linear/flattened Conv1d weights and seconds-total conditioner |
| Stable Audio 3 | PEFT directory/file plus sibling config | Same targets as the native format |

Standard LoRA and PEFT RSLoRA scaling are supported. DoRA supports PEFT
per-row magnitudes and Stable Audio native `dora-rows`, `dora-cols`, and legacy
`dora` with singleton-axis magnitude detection. Native alpha/r and PEFT
alpha/r (or alpha/sqrt(r)) are multiplied by the user's strength. For DoRA,
strength interpolates between the original weight and the **fully normalized**
merged weight; it does not scale the low-rank update before normalization.

Unsupported configurations fail with an error: LoRA-XS, BoRA, LoKR,
step-dependent schedules, transpose/fan-in-fan-out PEFT layers, trained biases,
per-module rank/alpha patterns, replicated layers, saved full modules, and
non-SafeTensors checkpoints. ACE-Step encoder/planner/VAE adapters and Stable
Audio Foundation/Open adapters are outside this initial implementation.

Adapters must be trained for the selected base checkpoint. Names, shapes,
ranks, parameter types, and finite values are checked before creating a session;
unknown tensors are errors, not silently skipped. **Matching tensor dimensions
do not prove matching training provenance.** In particular, choose Small vs
Medium, Music vs SFX, and ACE-Step Turbo vs Base/XL deliberately. An online
adapter with no alpha/config metadata needs its original configuration from the
author; this loader does not guess its scale.

## Usage

```sh
audiocpp_cli --task gen --family ace_step --model models/Ace-Step1.5 \
  --backend metal --text "instrumental folk music" --duration-seconds 8 \
  --session-option ace_step.lora=/path/to/adapter_directory \
  --session-option ace_step.lora_strength=0.7 --out adapted.wav

audiocpp_cli --task gen --family stable_audio --model models/stable-audio-3-medium \
  --backend metal --text "instrumental goa trance" --duration-seconds 8 \
  --session-option stable_audio.lora=/path/to/style.safetensors \
  --session-option stable_audio.lora_strength=0.7 --out adapted.wav
```

Strength defaults to 1, accepts finite values in [0,10], and is independent of
alpha/r. Zero validates the adapter but returns the original tensor source:
there is no merge or dequantization/requantization when disabled. Providing a
strength without an adapter is an error. Changing adapters/strength requires a
new session, which may reuse the same loaded base model. Adapted weights belong
to that session; other sessions retain their own weights.

C API callers use `audiocpp_options_create/set` to populate the legacy keys or the stack key,
then pass the options to `audiocpp_session_create`. No ABI extension is required.
Never perform model/session creation, adapter reads, merges, or destruction on
an audio callback. Applications should snapshot adapter bytes and config before
queuing a job, and include their content hashes and strength in session identity.

Weights merge in F32 when first read/uploaded, then use the requested backend
storage type. Quantized base packages are supported through existing tensor
conversion; this is not numerically equivalent to merging the original
unquantized weights and then quantizing. Strength zero bypasses this issue.
DoRA temporarily needs the original and merged tensor plus norm scratch space;
this is allocated per adapted weight, not as a second complete base model.

## Ordered stacks

Pass `stable_audio.adapters` or `ace_step.adapters` as a JSON array:

```json
[{"path":"/adapters/texture.safetensors","strength":0.6},
 {"path":"/adapters/attack.safetensors","strength":0.3}]
```

The list replaces the legacy `.lora` / `.lora_strength` keys; combining the two
forms is an error. At most eight entries are accepted. Strength defaults to 1.
An empty list returns the base source. Every entry, including zero-strength
entries, must be valid. Zero entries return the preceding source unchanged.

Apply entries from top to bottom. Plain LoRA adds each scaled delta to the
preceding result. DoRA computes its direction and magnitude normalization using
the preceding result as its base, then interpolates by its strength. Therefore
mixed/DoRA stacks are order dependent. This is explicitly sequential merge
semantics, not a claim of equivalence with simultaneous PEFT DoRA composition.
Changing order requires a new session. Intermediate overlays use F32 without
repeated backend quantization; the final upload chooses the storage type.
Memory for adapter factors grows with the stack; merge work is performed off the
audio thread. One bad entry fails session creation; no partial stack is used.

## Validation

The ordinary `lora_adapter_test` and `lora_tensor_source_test` require no model
weights. They cover independent merge arithmetic, alpha/r and RSLoRA, DoRA
row/column normalization and interpolation, legacy magnitudes, target mapping,
Conv1d flattening, unchanged base/unadapted tensors, exact zero bypass, repeated
reads, malformed/missing/unsupported configuration, rank/shape mismatch,
non-finite adapter values, and F32/F16/BF16/Q8_0/Q4_0 backend uploads.

An opt-in public-C-API integration check exercises two sessions sharing one
model, adapter effects, session reuse, base isolation, zero strength, and finite
non-silent output with the requested duration:

```sh
python tests/lora/c_api_smoke.py \
  --library build/bin/libaudiocpp.dylib --family stable_audio \
  --model /path/to/matching/model.gguf --adapter /path/to/style.safetensors \
  --backend metal --duration 4 --report smoke-report.json
```

Real-model test evidence and limitations are recorded in
[lora-validation.md](lora-validation.md). Tests do not redistribute adapters or
base weights. Their licenses remain separate from this source code.

Format references: [Stable Audio's native loader](https://github.com/Stability-AI/stable-audio-3/blob/main/stable_audio_3/models/lora/loader.py),
[Stable Audio's merge reference](https://github.com/Stability-AI/stable-audio-3/blob/main/optimized/tflite/scripts/lora_core.py),
[ACE-Step training config](https://github.com/ace-step/ACE-Step-1.5/blob/main/acestep/training/configs.py).

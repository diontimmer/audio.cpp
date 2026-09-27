#pragma once

#include "engine/framework/assets/tensor_source.h"

namespace engine::assets {

struct LoraAdapterOptions {
    float strength = 1.0F;
    bool native_parametrizations = false;
    // Exact candidate prefixes, including the empty prefix when appropriate.
    std::vector<std::string> target_prefixes;
    std::vector<std::string> allowed_target_prefixes;
    std::string log_prefix = "lora";
};

// PEFT directory/file plus adapter_config.json, or SA3 native SafeTensors with
// embedded lora_config. One immutable adapter per overlay; no pickle execution.
// Unsupported tensors/configuration fail instead of partially applying a file.
std::shared_ptr<const TensorSource> make_lora_adapter_source(
    std::shared_ptr<const TensorSource> base,
    const std::filesystem::path & path,
    const LoraAdapterOptions & options);

// Ordered JSON list of {"path": string, "strength": number}, maximum eight.
// LoRA deltas accumulate; DoRA normalizes the result of preceding entries.
// Intermediate overlays remain F32; backend conversion occurs only at upload.
std::shared_ptr<const TensorSource> make_lora_adapter_stack_source(
    std::shared_ptr<const TensorSource> base, const std::string & json,
    const LoraAdapterOptions & options);

} // namespace engine::assets

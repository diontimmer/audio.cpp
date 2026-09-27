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
// embedded lora_config. One immutable adapter per source; no pickle execution.
// Unsupported tensors/configuration fail instead of partially applying a file.
std::shared_ptr<const TensorSource> make_lora_adapter_source(
    std::shared_ptr<const TensorSource> base,
    const std::filesystem::path & path,
    const LoraAdapterOptions & options);

} // namespace engine::assets

#include "engine/framework/assets/lora_adapter.h"

#include "engine/framework/assets/lora_tensor_source.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/io/json.h"
#include "engine/framework/io/safetensors.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <limits>
#include <set>
#include <stdexcept>

namespace engine::assets {
namespace {
namespace json = engine::io::json;
constexpr const char * native_marker = ".parametrizations.weight.0.";
using Parts = std::map<std::string, std::string>;

bool starts(const std::string & s, const std::string & prefix) { return s.rfind(prefix, 0) == 0; }
bool ends(const std::string & s, const std::string & suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

float number(const json::Value & config, const char * key, float fallback) {
    const auto * v = config.find(key);
    const float value = v ? v->as_f32() : fallback;
    if (!std::isfinite(value)) throw std::runtime_error(std::string("LoRA non-finite ") + key);
    return value;
}

std::string string_value(const json::Value & config, const char * key, const char * fallback) {
    const auto * value = config.find(key);
    return value ? value->as_string() : fallback;
}

bool flag(const json::Value & config, const char * key) {
    const auto * v = config.find(key);
    return v && !v->is_null() && v->as_bool();
}

void require_empty(const json::Value & config, const char * key) {
    const auto * v = config.find(key);
    if (!v || v->is_null() || (v->is_object() && v->as_object().empty()) ||
        (v->is_array() && v->as_array().empty())) return;
    throw std::runtime_error(std::string("Unsupported LoRA config: ") + key);
}

std::string target_name(const TensorSource & base, const std::string & module, const LoraAdapterOptions & options) {
    std::set<std::string> found;
    for (const auto & prefix : options.target_prefixes) {
        const auto name = prefix + module + ".weight";
        if (base.has_tensor(name) && std::any_of(options.allowed_target_prefixes.begin(),
            options.allowed_target_prefixes.end(), [&](const auto & allowed) { return starts(name, allowed); }))
            found.insert(name);
    }
    if (found.size() != 1)
        throw std::runtime_error("LoRA target has no unique supported base weight: " + module);
    return *found.begin();
}
} // namespace

std::shared_ptr<const TensorSource> make_lora_adapter_source(
    std::shared_ptr<const TensorSource> base, const std::filesystem::path & path,
    const LoraAdapterOptions & options) {
    if (!base) throw std::runtime_error("LoRA requires a base tensor source");
    if (!std::isfinite(options.strength) || options.strength < 0 || options.strength > 10)
        throw std::runtime_error("LoRA strength must be finite and between 0 and 10");
    const auto weights = std::filesystem::is_directory(path) ? path / "adapter_model.safetensors" : path;
    if (weights.extension() != ".safetensors")
        throw std::runtime_error("LoRA weights must be SafeTensors (.safetensors)");
    const auto index = io::load_safetensors_index(weights);
    const auto adapter = open_tensor_source(weights);
    const bool native = std::any_of(index.tensors.begin(), index.tensors.end(), [](const auto & item) {
        return item.first.find(native_marker) != std::string::npos;
    });
    json::Value config;
    if (native) {
        if (!options.native_parametrizations) throw std::runtime_error("Native parametrization adapters are not supported for this model");
        const auto it = index.metadata.find("lora_config");
        if (it == index.metadata.end()) throw std::runtime_error("Native LoRA is missing lora_config metadata");
        config = json::parse(it->second);
    } else {
        config = json::parse_file(weights.parent_path() / "adapter_config.json");
        if (json::require_string(config, "peft_type") != "LORA")
            throw std::runtime_error("Only PEFT LORA adapters are supported");
        if (flag(config, "fan_in_fan_out") || flag(config, "lora_bias") || flag(config, "use_qalora") ||
            flag(config, "use_bdlora") || string_value(config, "bias", "none") != "none")
            throw std::runtime_error("Unsupported PEFT transpose, bias, QALoRA or BDLoRA configuration");
        for (const char * key : {"rank_pattern", "alpha_pattern", "modules_to_save", "layer_replication",
             "trainable_token_indices", "target_parameters", "alora_invocation_tokens", "arrow_config"})
            require_empty(config, key);
    }
    if (!config.is_object()) throw std::runtime_error("LoRA config must be an object");
    const auto rank_number = config.require(native ? "rank" : "r").as_number();
    if (!std::isfinite(rank_number) || rank_number < 1 || rank_number > std::numeric_limits<int>::max() ||
        std::floor(rank_number) != rank_number)
        throw std::runtime_error("LoRA rank must be a positive integer");
    const auto rank = static_cast<int64_t>(rank_number);
    if (rank <= 0) throw std::runtime_error("LoRA rank must be positive");
    const float alpha = number(config, native ? "alpha" : "lora_alpha", static_cast<float>(rank));
    const float scale = alpha / ((!native && flag(config, "use_rslora")) ? std::sqrt(static_cast<float>(rank)) : rank);
    std::string type = native ? string_value(config, "adapter_type", "lora") :
        (flag(config, "use_dora") ? "dora-rows" : "lora");
    if (type == "dora") {
        type = "dora-rows";
        std::string detected;
        for (const auto & [name, info] : index.tensors) {
            if (!ends(name, ".magnitude") || info.shape.size() != 2) continue;
            const std::string axis = info.shape[0] == 1 ? "dora-cols" : info.shape[1] == 1 ? "dora-rows" : "";
            if (axis.empty() || (!detected.empty() && detected != axis))
                throw std::runtime_error("Ambiguous legacy DoRA magnitude axes");
            detected = axis;
        }
        if (!detected.empty()) type = detected;
    }
    if (type != "lora" && type != "dora-rows" && type != "dora-cols")
        throw std::runtime_error("Unsupported adapter type: " + type + " (supported: LoRA, DoRA rows/columns)");

    std::map<std::string, Parts> layers;
    for (const auto & [name, info] : index.tensors) {
        if (info.dtype != "F32" && info.dtype != "F16" && info.dtype != "BF16")
            throw std::runtime_error("LoRA tensors must use F32, F16 or BF16: " + name);
        std::string module, part;
        if (native) {
            const auto pos = name.find(native_marker);
            if (pos != std::string::npos) {
                module = name.substr(0, pos);
                part = name.substr(pos + std::string(native_marker).size());
            }
        } else {
            const std::string prefix = "base_model.model.";
            const std::string key = starts(name, prefix) ? name.substr(prefix.size()) : name;
            for (const auto & suffix : {std::string(".lora_A.weight"), std::string(".lora_B.weight"),
                                      std::string(".lora_magnitude_vector.weight")}) {
                if (ends(key, suffix)) {
                    module = key.substr(0, key.size() - suffix.size());
                    part = suffix == ".lora_A.weight" ? "lora_A" : suffix == ".lora_B.weight" ? "lora_B" : "magnitude";
                    break;
                }
            }
        }
        if (module.empty() || (part != "lora_A" && part != "lora_B" && part != "magnitude"))
            throw std::runtime_error("Unsupported LoRA tensor: " + name);
        if (!layers[module].emplace(part, name).second)
            throw std::runtime_error("Duplicate LoRA tensor for " + module);
    }
    if (layers.empty()) throw std::runtime_error("LoRA adapter contains no tensors");
    std::unordered_map<std::string, LoraTensorDelta> deltas;
    for (const auto & [module, parts] : layers) {
        if (!parts.count("lora_A") || !parts.count("lora_B"))
            throw std::runtime_error("LoRA module is missing A/B pair: " + module);
        const auto name = target_name(*base, module, options);
        auto delta = load_lora_tensor_delta(*base, *adapter, name, parts.at("lora_A"), parts.at("lora_B"), scale);
        if (delta.r != rank) throw std::runtime_error("LoRA tensor rank disagrees with config: " + module);
        if (type == "lora") {
            if (parts.count("magnitude")) throw std::runtime_error("LoRA config has unexpected DoRA magnitude: " + module);
            delta.scale *= options.strength;
        } else {
            if (!parts.count("magnitude")) throw std::runtime_error("DoRA module is missing magnitude: " + module);
            delta.normalization = type == "dora-cols" ? LoraNormalization::Columns : LoraNormalization::Rows;
            const auto length = delta.normalization == LoraNormalization::Rows ? delta.out : delta.in;
            const auto shape = adapter->require_metadata(parts.at("magnitude")).shape;
            if (shape != std::vector<int64_t>{length} &&
                !(native && (shape == std::vector<int64_t>{1, length} || shape == std::vector<int64_t>{length, 1})))
                throw std::runtime_error("DoRA magnitude shape mismatch: " + module);
            delta.magnitude = adapter->require_f32(parts.at("magnitude"));
            delta.strength = options.strength;
            delta.norm_epsilon = native ? 1.0e-12F : 0.0F;
        }
        if (!deltas.emplace(name, std::move(delta)).second)
            throw std::runtime_error("Multiple adapter modules resolve to the same weight: " + name);
    }
    // Validate even zero-strength adapters, but return the exact original source
    // so disabled adapters never dequantize/requantize or alter base weights.
    const auto result = make_lora_tensor_source(base, std::move(deltas), {}, options.log_prefix, false);
    debug::log_message(debug::LogLevel::Info, options.log_prefix,
        "validated " + std::to_string(layers.size()) + " " + type + " modules, strength=" + std::to_string(options.strength));
    return options.strength == 0.0F ? base : result;
}
std::shared_ptr<const TensorSource> make_lora_adapter_stack_source(
    std::shared_ptr<const TensorSource> base, const std::string & text,
    const LoraAdapterOptions & options) {
    if (!base) throw std::runtime_error("Adapter stack requires a base tensor source");
    if (text.size() > 65536) throw std::runtime_error("Adapter stack JSON is too large");
    const auto entries = json::parse(text);
    if (!entries.is_array() || entries.as_array().size() > 8)
        throw std::runtime_error("Adapter stack must be an array with at most eight entries");
    auto result = base;
    for (const auto & entry : entries.as_array()) {
        if (!entry.is_object()) throw std::runtime_error("Adapter entry must be an object");
        for (const auto & field : entry.as_object())
            if (field.first != "path" && field.first != "strength")
                throw std::runtime_error("Unknown adapter field: " + field.first);
        const auto path = json::require_string(entry, "path");
        if (path.empty()) throw std::runtime_error("Adapter path must not be empty");
        auto current = options;
        current.strength = number(entry, "strength", 1.0F);
        result = make_lora_adapter_source(result, std::filesystem::u8path(path), current);
    }
    return result;
}
} // namespace engine::assets

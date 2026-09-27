#include "engine/framework/assets/lora_adapter.h"
#include "engine/framework/io/safetensors.h"
#include "test_assert.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>

namespace {
using namespace engine;
using test::require;

std::vector<unsigned char> bytes(const std::vector<float> & v) {
    std::vector<unsigned char> out(v.size() * sizeof(float));
    std::memcpy(out.data(), v.data(), out.size());
    return out;
}
io::SafeTensorWriteEntry tensor(std::string name, std::vector<int64_t> shape, std::vector<float> values) {
    return {std::move(name), "F32", std::move(shape), bytes(values)};
}
void write(const std::filesystem::path & path, const std::vector<io::SafeTensorWriteEntry> & entries,
           const std::string & config = "") {
    auto data = io::encode_safetensors(entries, config.empty() ? std::vector<std::pair<std::string,std::string>>{} :
        std::vector<std::pair<std::string,std::string>>{{"lora_config", config}});
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
}
void config(const std::filesystem::path & root, const std::string & value) {
    std::ofstream(root / "adapter_config.json") << value;
}
template<class F> void rejects(F fn, const std::string & fragment) {
    try { fn(); } catch (const std::runtime_error & error) {
        require(std::string(error.what()).find(fragment) != std::string::npos,
            "unexpected failure: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("invalid adapter accepted: " + fragment);
}
void near(const std::vector<float> & a, const std::vector<float> & b) {
    require(a.size() == b.size(), "result size mismatch");
    for (size_t i=0;i<a.size();++i)
        require(std::isfinite(a[i]) && std::abs(a[i]-b[i]) < 1e-5F, "merge arithmetic mismatch");
}
void run(const std::filesystem::path & root) {
    const std::string base_name = "decoder.layers.0.self_attn.q_proj.weight";
    const std::string module = "base_model.model.layers.0.self_attn.q_proj";
    const std::vector<float> w{1,2,3,4,5,6}, a{1,2,3,4,5,6}, b{1,2,3,4};
    write(root/"base.safetensors", {tensor(base_name,{2,3},w), tensor("untouched",{2,3},w),
        tensor("model.model.transformer.proj.weight",{2,3},w), tensor("model.model.preprocess_conv.weight",{2,3,1},w),
        tensor("conditioner.conditioners.seconds_total.embedder.embedding.1.weight",{2,3},w)});
    const auto base = assets::open_tensor_source(root/"base.safetensors");
    assets::LoraAdapterOptions peft;
    peft.target_prefixes = {"", "decoder."}; peft.allowed_target_prefixes={"decoder."}; peft.strength=.5F;
    auto entries = std::vector<io::SafeTensorWriteEntry>{tensor(module+".lora_A.weight",{2,3},a), tensor(module+".lora_B.weight",{2,2},b)};
    write(root/"adapter_model.safetensors", entries);
    const std::string plain=R"({"peft_type":"LORA","r":2,"lora_alpha":4})";
    config(root, plain);
    auto overlay = assets::make_lora_adapter_source(base,root,peft);
    const std::vector<float> expected{10,14,18,23,31,39};
    near(overlay->require_f32(base_name),expected);
    near(overlay->require_f32(base_name),expected); // repeated reads must not accumulate
    near(base->require_f32(base_name),w);
    require(overlay->require_tensor_data("untouched").bytes==base->require_tensor_data("untouched").bytes,"unadapted weight changed");
    peft.strength=0;
    require(assets::make_lora_adapter_source(base,root,peft).get()==base.get(),"zero strength must bypass conversion");
    peft.strength=.5F;
    config(root,R"({"peft_type":"LORA","r":2,"lora_alpha":4,"use_rslora":true})");
    auto rs=assets::make_lora_adapter_source(base,root,peft)->require_f32(base_name);
    for(size_t i=0;i<w.size();++i) require(std::abs(rs[i]-(w[i]+(expected[i]-w[i])*std::sqrt(2.f)))<1e-5F,"RSLoRA scale incorrect");
    config(root,R"({"peft_type":"LORA","r":2,"lora_alpha":4,"use_dora":true})");
    entries.push_back(tensor(module+".lora_magnitude_vector.weight",{2},{2,3}));
    write(root/"adapter_model.safetensors",entries);
    auto dora=assets::make_lora_adapter_source(base,root,peft)->require_f32(base_name);
    std::vector<float> dora_expected(w.size());
    for(size_t o=0;o<2;++o) {
        double sum=0;
        for(size_t i=0;i<3;++i) { const double v=w[o*3+i]+2*(expected[o*3+i]-w[o*3+i]); sum+=v*v; }
        for(size_t i=0;i<3;++i) { const auto j=o*3+i; dora_expected[j]=w[j]+.5F*((w[j]+2*(expected[j]-w[j]))*(o+2)/std::sqrt(sum)-w[j]); }
    }
    near(dora,dora_expected);
    entries.pop_back();write(root/"adapter_model.safetensors",entries);
    rejects([&]{assets::make_lora_adapter_source(base,root,peft);},"missing magnitude");
    config(root,R"({"peft_type":"LORA","r":2,"rank_pattern":{"x":1}})");
    rejects([&]{assets::make_lora_adapter_source(base,root,peft);},"rank_pattern");
    config(root,R"({"peft_type":"LORA","r":2,"fan_in_fan_out":true})");
    rejects([&]{assets::make_lora_adapter_source(base,root,peft);},"transpose");
    config(root,R"({"peft_type":"LORA","r":3})");
    rejects([&]{assets::make_lora_adapter_source(base,root,peft);},"rank disagrees");
    config(root,R"({"peft_type":"LORA","r":2.5})");
    rejects([&]{assets::make_lora_adapter_source(base,root,peft);},"positive integer");
    config(root,plain);
    peft.strength=std::numeric_limits<float>::quiet_NaN();
    rejects([&]{assets::make_lora_adapter_source(base,root,peft);},"strength"); peft.strength=1;
    auto broken=entries;broken.pop_back();write(root/"adapter_model.safetensors",broken);
    rejects([&]{assets::make_lora_adapter_source(base,root,peft);},"missing A/B");
    broken=entries;broken[0].data=bytes({std::numeric_limits<float>::infinity(),2,3,4,5,6});write(root/"adapter_model.safetensors",broken);
    rejects([&]{assets::make_lora_adapter_source(base,root,peft);},"non-finite");
    broken=entries;broken.push_back(tensor("something.unknown",{1},{1}));write(root/"adapter_model.safetensors",broken);
    rejects([&]{assets::make_lora_adapter_source(base,root,peft);},"Unsupported LoRA tensor");
    write(root/"adapter_model.safetensors",entries);
    auto bad_target=peft;bad_target.target_prefixes={"other."};
    rejects([&]{assets::make_lora_adapter_source(base,root,bad_target);},"no unique");
    std::filesystem::remove(root/"adapter_config.json");
    rejects([&]{assets::make_lora_adapter_source(base,root,peft);},"adapter_config.json");

    assets::LoraAdapterOptions native;
    native.native_parametrizations=true; native.target_prefixes={"","model.","model.model.","conditioner."};
    native.allowed_target_prefixes={"model.model.","conditioner.conditioners.seconds_total."};
    for(const std::string target : {"model.transformer.proj","preprocess_conv","conditioners.seconds_total.embedder.embedding.1"}) {
        const auto key=target+".parametrizations.weight.0.";
        write(root/"native.safetensors",{tensor(key+"lora_A",{2,3},a),tensor(key+"lora_B",{2,2},b)},
              R"({"rank":2,"alpha":2,"adapter_type":"lora"})");
        const auto adapted=assets::make_lora_adapter_source(base,root/"native.safetensors",native);
        const auto name=target=="model.transformer.proj"?"model.model.transformer.proj.weight":target=="preprocess_conv"?
            "model.model.preprocess_conv.weight":"conditioner.conditioners.seconds_total.embedder.embedding.1.weight";
        near(adapted->require_f32(name),expected);
    }
    const auto key=std::string("model.transformer.proj.parametrizations.weight.0.");
    write(root/"native.safetensors",{tensor(key+"lora_A",{2,3},a),tensor(key+"lora_B",{2,2},b),tensor(key+"magnitude",{3},{1,2,3})},
        R"({"rank":2,"alpha":2,"adapter_type":"dora-cols"})");
    auto cols=assets::make_lora_adapter_source(base,root/"native.safetensors",native)->require_f32("model.model.transformer.proj.weight");
    for(size_t i=0;i<3;++i) require(std::abs(std::hypot(cols[i],cols[3+i])-(i+1))<1e-5F,"DoRA column magnitude incorrect");
    rejects([&]{assets::make_lora_adapter_source(base,root/"native.safetensors",peft);},"Native parametrization");
    write(root/"native.safetensors",{tensor(key+"lora_A",{2,3},a),tensor(key+"lora_B",{2,2},b),tensor(key+"magnitude",{1,3},{1,2,3})},
        R"({"rank":2,"alpha":2,"adapter_type":"dora"})");
    near(assets::make_lora_adapter_source(base,root/"native.safetensors",native)->require_f32("model.model.transformer.proj.weight"),cols);
    write(root/"native.safetensors",{tensor(key+"lora_A",{2,4},{1,2,3,4,5,6,7,8}),tensor(key+"lora_B",{2,2},b)},
        R"({"rank":2,"alpha":2,"adapter_type":"lora"})");
    rejects([&]{assets::make_lora_adapter_source(base,root/"native.safetensors",native);},"shape mismatch");

    write(root/"native.safetensors",{tensor(key+"lora_A",{2,3},a),tensor(key+"lora_B",{2,2},b)},
        R"({"rank":2,"alpha":2,"adapter_type":"bora"})");
    rejects([&]{assets::make_lora_adapter_source(base,root/"native.safetensors",native);},"Unsupported adapter type");
    write(root/"native.safetensors",{tensor(key+"lora_A",{2,3},a),tensor(key+"lora_B",{2,2},b)});
    rejects([&]{assets::make_lora_adapter_source(base,root/"native.safetensors",native);},"missing lora_config");
}
} // namespace
int main() {
    const auto root=std::filesystem::temp_directory_path()/ ("audiocpp_lora_adapter_"+
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    try {run(root);std::filesystem::remove_all(root);std::cout<<"lora_adapter_test passed\n";return 0;}
    catch(const std::exception & e){std::filesystem::remove_all(root);std::cerr<<e.what()<<'\n';return 1;}
}

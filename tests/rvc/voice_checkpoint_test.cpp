// Builds tiny torch-zip RVC checkpoints in the layouts written by classic RVC and by Applio, and checks that the
// native loader accepts them without Python. Weights are zeros; only metadata parsing and tensor layout matter.
#include "engine/models/rvc/assets.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool ok, const std::string & message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

class Pickle {
public:
    std::vector<uint8_t> bytes;

    Pickle & op(uint8_t value) {
        bytes.push_back(value);
        return *this;
    }
    Pickle & raw(const std::string & text) {
        bytes.insert(bytes.end(), text.begin(), text.end());
        return *this;
    }
    Pickle & u16(uint16_t value) {
        return op(value & 0xff).op(value >> 8);
    }
    Pickle & u32(uint32_t value) {
        return u16(value & 0xffff).u16(value >> 16);
    }
    Pickle & str(const std::string & text) {
        return op('X').u32(static_cast<uint32_t>(text.size())).raw(text);
    }
    Pickle & integer(uint32_t value) {
        if (value < 0x100) {
            return op('K').op(static_cast<uint8_t>(value));
        }
        if (value < 0x10000) {
            return op('M').u16(static_cast<uint16_t>(value));
        }
        return op('J').u32(value);
    }
    Pickle & binfloat(double value) {
        uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        op('G');
        for (int shift = 56; shift >= 0; shift -= 8) {
            op(static_cast<uint8_t>(bits >> shift));
        }
        return *this;
    }
    Pickle & global(const std::string & module, const std::string & name) {
        return op('c').raw(module + "\n" + name + "\n");
    }
    Pickle & int_tuple(const std::vector<uint32_t> & values) {
        op('(');
        for (const auto value : values) {
            integer(value);
        }
        return op('t');
    }
    Pickle & half_tensor(const std::vector<uint32_t> & shape, uint32_t numel) {
        std::vector<uint32_t> strides(shape.size(), 1);
        for (size_t i = shape.size(); i-- > 1;) {
            strides[i - 1] = strides[i] * shape[i];
        }
        global("torch._utils", "_rebuild_tensor_v2").op('(');
        op('(').str("storage").global("torch", "HalfStorage").str("0").str("cpu").integer(numel).op('t').op('Q');
        integer(0).int_tuple(shape).int_tuple(strides).op(0x89);
        return op('t').op('R');
    }
};

struct Tensor {
    const char * name;
    std::vector<uint32_t> shape;
};

const std::vector<Tensor> kTensors = {
    {"enc_p.emb_phone.weight", {192, 768}},
    {"enc_p.emb_pitch.weight", {256, 192}},
    {"emb_g.weight", {1, 256}},
    {"dec.ups.0.weight_v", {512, 256, 16}},
    {"dec.ups.1.weight_v", {256, 128, 16}},
    {"dec.ups.2.weight_v", {128, 64, 4}},
    {"dec.ups.3.weight_v", {64, 32, 4}},
};
constexpr uint32_t kStorageElements = 512 * 256 * 16;

enum class Flavor { Classic, Applio, ApplioRefineGan };

std::vector<uint8_t> checkpoint_pickle(Flavor flavor) {
    const bool applio = flavor != Flavor::Classic;
    Pickle p;
    p.op(0x80).op(applio ? 4 : 2);
    if (applio) {
        p.op(0x95).u32(0).u32(0);  // FRAME length is not validated by the loader.
    }
    p.op('}').op('(');
    p.str("weight").global("collections", "OrderedDict").op(')').op('R').op('(');
    for (const auto & tensor : kTensors) {
        p.str(tensor.name).half_tensor(tensor.shape, kStorageElements);
    }
    p.op('u');
    // RVC v2 40k config: spec_channels ... p_dropout ... sr. Applio writes some entries as floats.
    p.str("config").op(']').op('(');
    p.integer(1025).integer(32).integer(192).integer(192).integer(768).integer(2).integer(6).integer(3);
    if (applio) {
        p.binfloat(0.0);
    } else {
        p.integer(0);
    }
    p.integer(109).integer(40000).op('e');
    if (applio) {
        p.str("f0").op(0x88);  // Python True
        p.str("sr").integer(40000).op(0x94);
        p.str("version").op(0x8c).op(2).raw("v2").op(0x94);
        p.str("author").op('N');
        p.str("speakers_id").integer(0);
        p.str("dataset_length").binfloat(2799.5);
        p.str("model_hash").op('C').op(4).raw("\x01\x02\x03\x04");
        p.str("embedder_model").str("contentvec");
        p.str("vocoder").str(flavor == Flavor::ApplioRefineGan ? "RefineGAN" : "HiFi-GAN");
    } else {
        p.str("f0").integer(1);
        p.str("sr").str("40k");
        p.str("version").str("v2");
        p.str("info").str("300epoch");
    }
    p.op('u').op('.');
    return p.bytes;
}

void write_stored_zip(const std::filesystem::path & path, const std::vector<std::pair<std::string, std::vector<uint8_t>>> & files) {
    std::vector<uint8_t> out;
    std::vector<uint8_t> central;
    auto put16 = [](std::vector<uint8_t> & v, uint16_t x) { v.push_back(x & 0xff); v.push_back(x >> 8); };
    auto put32 = [&](std::vector<uint8_t> & v, uint32_t x) { put16(v, x & 0xffff); put16(v, x >> 16); };
    for (const auto & [name, data] : files) {
        const auto offset = static_cast<uint32_t>(out.size());
        const auto size = static_cast<uint32_t>(data.size());
        put32(out, 0x04034b50);
        put16(out, 20); put16(out, 0); put16(out, 0); put16(out, 0); put16(out, 0);
        put32(out, 0); put32(out, size); put32(out, size);
        put16(out, static_cast<uint16_t>(name.size())); put16(out, 0);
        out.insert(out.end(), name.begin(), name.end());
        out.insert(out.end(), data.begin(), data.end());
        put32(central, 0x02014b50);
        put16(central, 20); put16(central, 20); put16(central, 0); put16(central, 0); put16(central, 0); put16(central, 0);
        put32(central, 0); put32(central, size); put32(central, size);
        put16(central, static_cast<uint16_t>(name.size())); put16(central, 0); put16(central, 0);
        put16(central, 0); put16(central, 0); put32(central, 0); put32(central, offset);
        central.insert(central.end(), name.begin(), name.end());
    }
    const auto central_offset = static_cast<uint32_t>(out.size());
    out.insert(out.end(), central.begin(), central.end());
    put32(out, 0x06054b50);
    put16(out, 0); put16(out, 0);
    put16(out, static_cast<uint16_t>(files.size())); put16(out, static_cast<uint16_t>(files.size()));
    put32(out, static_cast<uint32_t>(central.size())); put32(out, central_offset); put16(out, 0);
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char *>(out.data()), static_cast<std::streamsize>(out.size()));
    require(static_cast<bool>(stream), "could not write fixture " + path.string());
}

std::filesystem::path write_checkpoint(const std::filesystem::path & dir, const std::string & stem, Flavor flavor) {
    const auto path = dir / (stem + ".pth");
    write_stored_zip(path, {
        {stem + "/data.pkl", checkpoint_pickle(flavor)},
        {stem + "/data/0", std::vector<uint8_t>(static_cast<size_t>(kStorageElements) * 2, 0)},
    });
    return path;
}

void check_loads(const std::filesystem::path & path) {
    const auto voice = engine::models::rvc::load_rvc_voice_model(path);
    require(voice.version == "v2", "version should be v2: " + path.filename().string());
    require(voice.sample_rate == 40000, "sample rate should be 40000: " + path.filename().string());
    require(voice.has_f0, "pitch guidance should come from enc_p.emb_pitch.weight: " + path.filename().string());
    require(voice.speaker_count == 1, "speaker count should come from emb_g.weight: " + path.filename().string());
    require(voice.synthesizer_layout.upsample_kernel_sizes[0] == 16, "decoder layout should use weight_v kernels");
}

}  // namespace

int main(int argc, char ** argv) {
    const auto dir = std::filesystem::temp_directory_path() / "audiocpp_rvc_voice_checkpoint_test";
    std::filesystem::create_directories(dir);
    try {
        // Optional real checkpoints to probe, e.g. a raw Applio export. Two paths also compare every tensor, which
        // shows a raw export and its converted copy load to identical weights.
        std::vector<engine::models::rvc::RvcVoiceModel> probed;
        for (int i = 1; i < argc; ++i) {
            probed.push_back(engine::models::rvc::load_rvc_voice_model(argv[i]));
            const auto & voice = probed.back();
            std::cout << argv[i] << ": " << voice.version << " " << voice.sample_rate << " Hz, f0=" << voice.has_f0
                      << ", speakers=" << voice.speaker_count << ", tensors=" << voice.checkpoint->tensors().size()
                      << "\n";
        }
        if (probed.size() == 2) {
            const auto & a = *probed[0].checkpoint;
            const auto & b = *probed[1].checkpoint;
            require(a.tensors().size() == b.tensors().size(), "probed checkpoints have different tensor sets");
            for (const auto & meta : a.tensors()) {
                require(b.has_tensor(meta.name) && a.require_tensor_data(meta.name).bytes ==
                        b.require_tensor_data(meta.name).bytes, "probed checkpoints differ at " + meta.name);
            }
            std::cout << "probed checkpoints have identical tensors\n";
        }
        check_loads(write_checkpoint(dir, "classic_v2", Flavor::Classic));
        check_loads(write_checkpoint(dir, "applio_v2", Flavor::Applio));
        bool rejected = false;
        try {
            engine::models::rvc::load_rvc_voice_model(write_checkpoint(dir, "applio_refinegan", Flavor::ApplioRefineGan));
        } catch (const std::exception & error) {
            rejected = std::string(error.what()).find("only HiFi-GAN vocoder") != std::string::npos;
        }
        require(rejected, "RefineGAN checkpoints should be rejected with a vocoder error");
    } catch (const std::exception & error) {
        std::cerr << "rvc_voice_checkpoint_test failed: " << error.what() << "\n";
        std::filesystem::remove_all(dir);
        return 1;
    }
    std::filesystem::remove_all(dir);
    std::cout << "rvc_voice_checkpoint_test passed\n";
    return 0;
}

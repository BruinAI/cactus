#include <cstdlib>
#include <filesystem>
#include <iostream>

#include "../cactus/models/model.h"

using namespace cactus::engine;

int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    fs::path asset = fs::path(__FILE__).parent_path() / "assets" / "test_monkey.png";
    if (!fs::exists(asset)) {
        std::cerr << "Asset not found: " << asset << std::endl;
        return 1;
    }

    fs::path ir_path;
    if (const char* env = std::getenv("CACTUS_GRAPH_IR")) {
        ir_path = fs::path(env);
    } else if (argc > 1) {
        ir_path = fs::path(argv[1]);
    } else {
        std::cerr << "Need path to binary IR (set CACTUS_GRAPH_IR or pass it as the first argument)" << std::endl;
        return 1;
    }

    if (!fs::exists(ir_path)) {
        std::cerr << "Binary IR not found: " << ir_path << std::endl;
        return 1;
    }

    try {
        OnnxModel model(ir_path.string(), asset.string());
        auto output = model.run();
        if (output.empty()) {
            std::cerr << "Model produced no output" << std::endl;
            return 1;
        }
        std::cout << "Model ran successfully; output length=" << output.size() << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Inference failed: " << ex.what() << std::endl;
        return 1;
    }
}


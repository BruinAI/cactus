#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include "../cactus/engine/engine.h"
#include "../cactus/models/model.h"

using namespace cactus::engine;

// Note: This test uses the Siglip2VisionModel which now uses Lfm2VlPreprocessor

void print_tensor_stats(const std::vector<float>& data, const std::string& name) {
    if (data.empty()) {
        std::cout << name << ": empty tensor" << std::endl;
        return;
    }
    
    float min_val = data[0];
    float max_val = data[0];
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    for (float val : data) {
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
        sum += val;
        sum_sq += val * val;
    }
    
    float mean = sum / data.size();
    float variance = (sum_sq / data.size()) - (mean * mean);
    float std_dev = std::sqrt(variance);
    
    std::cout << name << " statistics:" << std::endl;
    std::cout << "  Shape: (" << data.size() << ",)" << std::endl;
    std::cout << "  Min: " << min_val << std::endl;
    std::cout << "  Max: " << max_val << std::endl;
    std::cout << "  Mean: " << mean << std::endl;
    std::cout << "  Std: " << std_dev << std::endl;
    std::cout << "  First 10 values: ";
    for (size_t i = 0; i < std::min(size_t(10), data.size()); ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_folder> <image_path>" << std::endl;
        std::cerr << "Example: " << argv[0] << " ./models/siglip2 ./tests/istock.jpg" << std::endl;
        return 1;
    }

    std::string model_folder = argv[1];
    std::string image_path = argv[2];

    std::cout << "=== SigLip2 Vision Model Test ===" << std::endl;
    std::cout << "Model folder: " << model_folder << std::endl;
    std::cout << "Image path: " << image_path << std::endl << std::endl;

    // Create and initialize model
    std::cout << "Initializing model..." << std::endl;
    Config config;
    if (!config.from_json(model_folder + "/config.json")) {
        std::cerr << "Failed to load config from " << model_folder << "/config.json" << std::endl;
        return 1;
    }

    Siglip2VisionModel model(config);
    if (!model.init(model_folder, 0)) {
        std::cerr << "Failed to initialize model" << std::endl;
        return 1;
    }
    std::cout << "Model initialized successfully" << std::endl << std::endl;

    // Get image features
    std::cout << "Processing image..." << std::endl;
    auto features = model.get_image_features(image_path);
    
    std::cout << std::endl << "=== Results ===" << std::endl;
    print_tensor_stats(features, "Image features");

    // Save to file if requested
    if (argc > 3) {
        std::string output_path = argv[3];
        std::ofstream out(output_path);
        if (out.is_open()) {
            out << std::fixed << std::setprecision(6);
            for (size_t i = 0; i < features.size(); ++i) {
                out << features[i];
                if ((i + 1) % 10 == 0) out << std::endl;
                else out << " ";
            }
            out.close();
            std::cout << "Features saved to: " << output_path << std::endl;
        }
    }

    std::cout << std::endl << "Test completed successfully!" << std::endl;
    return 0;
}


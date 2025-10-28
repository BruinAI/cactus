#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include "../cactus/engine/engine.h"
#include "../cactus/models/model.h"

using namespace cactus::engine;

void print_tensor_stats(const std::vector<float>& data, const std::string& name) {
    if (data.empty()) {
        std::cout << name << ": EMPTY" << std::endl;
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
    float std = std::sqrt(std::max(0.0f, variance));
    
    std::cout << name << ":" << std::endl;
    std::cout << "  Shape: (" << data.size() << ",)" << std::endl;
    std::cout << "  Min: " << min_val << std::endl;
    std::cout << "  Max: " << max_val << std::endl;
    std::cout << "  Mean: " << mean << std::endl;
    std::cout << "  Std: " << std << std::endl;
    std::cout << "  First 10 values: ";
    for (size_t i = 0; i < std::min(size_t(10), data.size()); ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}

void save_features_to_file(const std::string& output_path, 
                           const std::vector<float>& features,
                           const Lfm2VlPreprocessor::PreprocessedImage& preprocessed) {
    std::ofstream out(output_path);
    if (!out.is_open()) {
        std::cerr << "Failed to open output file: " << output_path << std::endl;
        return;
    }

    out << std::fixed << std::setprecision(6);
    
    // Write metadata
    out << "=== PREPROCESSED IMAGE METADATA ===" << std::endl;
    out << "num_patches_height: " << preprocessed.num_patches_height << std::endl;
    out << "num_patches_width: " << preprocessed.num_patches_width << std::endl;
    out << "actual_num_patches: " << preprocessed.actual_num_patches << std::endl;
    out << "image_rows: " << preprocessed.image_rows << std::endl;
    out << "image_cols: " << preprocessed.image_cols << std::endl;
    out << "image_height: " << preprocessed.image_height << std::endl;
    out << "image_width: " << preprocessed.image_width << std::endl;
    out << "tokens_per_tile: " << preprocessed.tokens_per_tile << std::endl;
    out << "thumbnail_tokens: " << preprocessed.thumbnail_tokens << std::endl;
    out << std::endl;

    // Feature statistics
    out << "=== VISION FEATURES ===" << std::endl;
    out << "Total features: " << features.size() << std::endl;
    
    float min_val = features[0];
    float max_val = features[0];
    float sum = 0.0f;
    
    for (float val : features) {
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
        sum += val;
    }
    
    out << "Min: " << min_val << std::endl;
    out << "Max: " << max_val << std::endl;
    out << "Mean: " << (sum / features.size()) << std::endl;
    out << std::endl;

    // Write first 100 features for comparison
    out << "=== FIRST 100 FEATURES ===" << std::endl;
    for (size_t i = 0; i < std::min(size_t(100), features.size()); ++i) {
        out << features[i];
        if ((i + 1) % 10 == 0) out << std::endl;
        else out << " ";
    }
    out << std::endl;

    out.close();
    std::cout << "Features saved to: " << output_path << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_dir> <image_path> [output_path]" << std::endl;
        std::cerr << "Example: " << argv[0] << " weights/lfm2-vl-350m-fp16 test_image.png output.txt" << std::endl;
        return 1;
    }

    std::string model_dir = argv[1];
    std::string image_path = argv[2];
    std::string output_path = argc > 3 ? argv[3] : "vision_features.txt";

    std::cout << "=== SigLip2 Vision Model Test ===" << std::endl;
    std::cout << "Model directory: " << model_dir << std::endl;
    std::cout << "Image: " << image_path << std::endl;
    std::cout << std::endl;

    try {
        // Load config
        std::cout << "Loading model config..." << std::endl;
        Config config;
        if (!config.from_json(model_dir + "/config.txt")) {
            std::cerr << "Failed to load config from " << model_dir << "/config.txt" << std::endl;
            return 1;
        }
        
        std::cout << "Config loaded:" << std::endl;
        std::cout << "  Vision hidden size: " << config.vision_hidden_dim << std::endl;
        std::cout << "  Vision num layers: " << config.vision_num_layers << std::endl;
        std::cout << "  Vision patch size: " << config.vision_patch_size << std::endl;
        std::cout << "  Vision embed dim: " << config.vision_embed_dim << std::endl;
        std::cout << "  Downsample factor: " << config.downsample_factor << std::endl;
        std::cout << "  Tile size: " << config.tile_size << std::endl;
        std::cout << std::endl;

        // Initialize vision model
        std::cout << "Initializing Siglip2VisionModel..." << std::endl;
        Siglip2VisionModel model(config);
        
        std::cout << "Calling model.init()..." << std::endl;
        if (!model.init(model_dir, 0)) {
            std::cerr << "Failed to initialize model" << std::endl;
            return 1;
        }
        std::cout << "Model initialized successfully!" << std::endl;
        std::cout << std::endl;

        // Preprocess image
        std::cout << "Preprocessing image..." << std::endl;
        auto preprocessed = model.get_preprocessor().preprocess_from_file(image_path);
        
        std::cout << "Preprocessing complete:" << std::endl;
        std::cout << "  Patches (H x W): " << preprocessed.num_patches_height 
                  << " x " << preprocessed.num_patches_width << std::endl;
        std::cout << "  Actual patches: " << preprocessed.actual_num_patches << std::endl;
        std::cout << "  Image grid (rows x cols): " << preprocessed.image_rows 
                  << " x " << preprocessed.image_cols << std::endl;
        std::cout << "  Tokens per tile: " << preprocessed.tokens_per_tile << std::endl;
        std::cout << "  Thumbnail tokens: " << preprocessed.thumbnail_tokens << std::endl;
        std::cout << std::endl;

        // Run vision encoder
        std::cout << "Running vision encoder forward pass..." << std::endl;
        auto features = model.get_image_features(preprocessed);
        
        std::cout << "Vision encoding complete!" << std::endl;
        std::cout << std::endl;

        // Print statistics
        std::cout << "=== Vision Features Statistics ===" << std::endl;
        print_tensor_stats(features, "Vision Features");
        std::cout << std::endl;

        // Expected output shape: (num_patches, embed_dim)
        if (features.size() % config.vision_embed_dim == 0) {
            size_t num_tokens = features.size() / config.vision_embed_dim;
            std::cout << "Output shape: (" << num_tokens << ", " << config.vision_embed_dim << ")" << std::endl;
        } else {
            std::cout << "Warning: Feature size doesn't divide evenly by embed_dim" << std::endl;
        }
        std::cout << std::endl;

        // Save to file
        std::cout << "Saving features to file..." << std::endl;
        save_features_to_file(output_path, features, preprocessed);
        std::cout << std::endl;

        std::cout << "✓ Success!" << std::endl;
        std::cout << "\nThe vision encoder processed the image successfully." << std::endl;
        std::cout << "Output features can be used as input to the language model." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}


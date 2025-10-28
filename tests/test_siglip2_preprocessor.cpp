#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include "../cactus/engine/engine.h"

using namespace cactus::engine;

void save_preprocessed_output(const std::string& output_path, 
                               const Lfm2VlPreprocessor::PreprocessedImage& result) {
    std::ofstream out(output_path);
    if (!out.is_open()) {
        std::cerr << "Failed to open output file: " << output_path << std::endl;
        return;
    }

    out << std::fixed << std::setprecision(6);
    
    // Write metadata
    int total_tokens = result.actual_num_patches;
    out << "=== METADATA ===" << std::endl;
    out << "grid_rows: " << result.image_rows << std::endl;
    out << "grid_cols: " << result.image_cols << std::endl;
    out << "num_tiles: " << (result.image_rows * result.image_cols) << std::endl;
    out << "tokens_per_tile: " << result.tokens_per_tile << std::endl;
    out << "thumbnail_tokens: " << result.thumbnail_tokens << std::endl;
    out << "total_tokens: " << total_tokens << std::endl;
    out << "pixel_values_shape: (" << total_tokens << ", 768)" << std::endl;
    out << std::endl;

    // Write pixel values statistics for first 5 tokens
    out << "=== FIRST 5 TOKENS STATISTICS ===" << std::endl;
    int patch_dim = 768;  // 16*16*3
    for (int token_idx = 0; token_idx < std::min(5, total_tokens); ++token_idx) {
        out << "Token " << token_idx << ":" << std::endl;
        
        // Calculate statistics
        float min_val = 1e9, max_val = -1e9, sum = 0.0f;
        int start_idx = token_idx * patch_dim;
        int end_idx = start_idx + patch_dim;
        
        for (int i = start_idx; i < end_idx; ++i) {
            float val = result.pixel_values[i];
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
            sum += val;
        }
        float mean = sum / patch_dim;
        
        out << "  Min: " << min_val << std::endl;
        out << "  Max: " << max_val << std::endl;
        out << "  Mean: " << mean << std::endl;
        
        // Print first 10 values
        out << "  First 10 values: ";
        for (int i = 0; i < std::min(10, patch_dim); ++i) {
            out << result.pixel_values[start_idx + i] << " ";
        }
        out << std::endl;
    }
    out << std::endl;

    // Write complete pixel values for first token (for exact comparison)
    out << "=== FIRST TOKEN COMPLETE VALUES ===" << std::endl;
    if (total_tokens > 0) {
        for (int i = 0; i < patch_dim; ++i) {
            out << result.pixel_values[i];
            if ((i + 1) % 8 == 0) out << std::endl;
            else out << " ";
        }
    }
    out << std::endl;

    // Write global statistics
    out << "=== GLOBAL STATISTICS ===" << std::endl;
    float global_min = 1e9, global_max = -1e9, global_sum = 0.0f;
    
    for (size_t i = 0; i < result.pixel_values.size(); ++i) {
        float val = result.pixel_values[i];
        global_min = std::min(global_min, val);
        global_max = std::max(global_max, val);
        global_sum += val;
    }
    
    out << "Total tokens: " << total_tokens << std::endl;
    out << "Global min: " << global_min << std::endl;
    out << "Global max: " << global_max << std::endl;
    out << "Global mean: " << (global_sum / result.pixel_values.size()) << std::endl;

    out.close();
    std::cout << "Output saved to: " << output_path << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path> [output_path]" << std::endl;
        std::cerr << "Example: " << argv[0] << " test_image.png output_cpp.txt" << std::endl;
        return 1;
    }

    std::string image_path = argv[1];
    std::string output_path = argc > 2 ? argv[2] : "lfm2vl_output_cpp.txt";

    std::cout << "=== Lfm2VL Preprocessor Test ===" << std::endl;
    std::cout << "Image: " << image_path << std::endl;
    std::cout << std::endl;

    // Create preprocessor with default Lfm2VL config
    Lfm2VlPreprocessor::Config config;
    config.patch_size = 16;
    config.downsample_factor = 2;
    config.min_tiles = 2;
    config.max_tiles = 10;
    config.use_thumbnail = false;
    config.min_image_tokens = 64;
    config.max_image_tokens = 256;
    config.tile_size = 512;
    config.max_pixels_tolerance = 2.0f;
    config.do_image_splitting = true;
    config.do_resize = true;
    config.do_rescale = true;
    config.do_normalize = true;
    config.do_convert_rgb = true;
    config.rescale_factor = 1.0f / 255.0f;
    config.image_mean[0] = 0.5f;
    config.image_mean[1] = 0.5f;
    config.image_mean[2] = 0.5f;
    config.image_std[0] = 0.5f;
    config.image_std[1] = 0.5f;
    config.image_std[2] = 0.5f;

    Lfm2VlPreprocessor preprocessor(config);

    // Load and preprocess image
    std::cout << "Loading image..." << std::endl;
    auto result = preprocessor.preprocess_from_file(image_path);

    // Print results
    int total_tokens = result.actual_num_patches;
    std::cout << "\n=== Preprocessing Results ===" << std::endl;
    std::cout << "Grid layout: " << result.image_rows << " x " << result.image_cols << std::endl;
    std::cout << "Number of tiles: " << (result.image_rows * result.image_cols) << std::endl;
    std::cout << "Tokens per tile: " << result.tokens_per_tile << std::endl;
    std::cout << "Thumbnail tokens: " << result.thumbnail_tokens << std::endl;
    std::cout << "Total tokens: " << total_tokens << std::endl;
    std::cout << "Pixel values size: " << result.pixel_values.size() << std::endl;
    std::cout << "Pixel values per token: " << (result.pixel_values.size() / total_tokens) << std::endl;

    // Calculate some statistics
    float min_val = 1e9, max_val = -1e9;
    for (float val : result.pixel_values) {
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
    }
    std::cout << "\nPixel value range: [" << min_val << ", " << max_val << "]" << std::endl;

    // Save detailed output
    std::cout << "\nSaving detailed output..." << std::endl;
    save_preprocessed_output(output_path, result);

    std::cout << "\n✓ Success!" << std::endl;

    return 0;
}


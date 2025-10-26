#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include "../cactus/engine/engine.h"

using namespace cactus::engine;

void save_preprocessed_output(const std::string& output_path, 
                               const SigLip2Preprocessor::PreprocessedImage& result) {
    std::ofstream out(output_path);
    if (!out.is_open()) {
        std::cerr << "Failed to open output file: " << output_path << std::endl;
        return;
    }

    out << std::fixed << std::setprecision(6);
    
    // Write metadata
    out << "=== METADATA ===" << std::endl;
    out << "num_patches_height: " << result.num_patches_height << std::endl;
    out << "num_patches_width: " << result.num_patches_width << std::endl;
    out << "actual_num_patches: " << result.actual_num_patches << std::endl;
    out << "pixel_values_shape: (" << result.pixel_attention_mask.size() 
        << ", " << (result.pixel_values.size() / result.pixel_attention_mask.size()) << ")" << std::endl;
    out << std::endl;

    // Write attention mask
    out << "=== ATTENTION MASK ===" << std::endl;
    for (size_t i = 0; i < result.pixel_attention_mask.size(); ++i) {
        out << result.pixel_attention_mask[i];
        if ((i + 1) % 16 == 0) out << std::endl;
        else out << " ";
    }
    out << std::endl << std::endl;

    // Write pixel values statistics for first 5 patches
    out << "=== FIRST 5 PATCHES STATISTICS ===" << std::endl;
    int patch_size = result.pixel_values.size() / result.pixel_attention_mask.size();
    for (int patch_idx = 0; patch_idx < std::min(5, (int)result.pixel_attention_mask.size()); ++patch_idx) {
        if (result.pixel_attention_mask[patch_idx] == 0) continue;
        
        out << "Patch " << patch_idx << ":" << std::endl;
        
        // Calculate statistics
        float min_val = 1e9, max_val = -1e9, sum = 0.0f;
        int start_idx = patch_idx * patch_size;
        int end_idx = start_idx + patch_size;
        
        for (int i = start_idx; i < end_idx; ++i) {
            float val = result.pixel_values[i];
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
            sum += val;
        }
        float mean = sum / patch_size;
        
        out << "  Min: " << min_val << std::endl;
        out << "  Max: " << max_val << std::endl;
        out << "  Mean: " << mean << std::endl;
        
        // Print first 10 values
        out << "  First 10 values: ";
        for (int i = 0; i < std::min(10, patch_size); ++i) {
            out << result.pixel_values[start_idx + i] << " ";
        }
        out << std::endl;
    }
    out << std::endl;

    // Write complete pixel values for first patch (for exact comparison)
    out << "=== FIRST PATCH COMPLETE VALUES ===" << std::endl;
    if (result.pixel_attention_mask[0] == 1) {
        for (int i = 0; i < std::min(patch_size, 768); ++i) {  // 16*16*3 = 768
            out << result.pixel_values[i];
            if ((i + 1) % 8 == 0) out << std::endl;
            else out << " ";
        }
    }
    out << std::endl;

    // Write global statistics
    out << "=== GLOBAL STATISTICS ===" << std::endl;
    float global_min = 1e9, global_max = -1e9, global_sum = 0.0f;
    int valid_patches = 0;
    
    for (size_t patch_idx = 0; patch_idx < result.pixel_attention_mask.size(); ++patch_idx) {
        if (result.pixel_attention_mask[patch_idx] == 0) continue;
        valid_patches++;
        
        int start_idx = patch_idx * patch_size;
        int end_idx = start_idx + patch_size;
        
        for (int i = start_idx; i < end_idx; ++i) {
            float val = result.pixel_values[i];
            global_min = std::min(global_min, val);
            global_max = std::max(global_max, val);
            global_sum += val;
        }
    }
    
    out << "Valid patches: " << valid_patches << std::endl;
    out << "Global min: " << global_min << std::endl;
    out << "Global max: " << global_max << std::endl;
    out << "Global mean: " << (global_sum / (valid_patches * patch_size)) << std::endl;

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
    std::string output_path = argc > 2 ? argv[2] : "siglip2_output_cpp.txt";

    std::cout << "=== SigLip2 Preprocessor Test ===" << std::endl;
    std::cout << "Image: " << image_path << std::endl;
    std::cout << std::endl;

    try {
        // Create preprocessor with default SigLip2 config
        SigLip2Preprocessor::Config config;
        config.patch_size = 16;
        config.max_num_patches = 256;
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

        SigLip2Preprocessor preprocessor(config);

        // Load and preprocess image
        std::cout << "Loading image..." << std::endl;
        auto result = preprocessor.preprocess_from_file(image_path);

        // Print results
        std::cout << "\n=== Preprocessing Results ===" << std::endl;
        std::cout << "Number of patches (height x width): " 
                  << result.num_patches_height << " x " << result.num_patches_width << std::endl;
        std::cout << "Actual number of patches: " << result.actual_num_patches << std::endl;
        std::cout << "Padded to: " << result.pixel_attention_mask.size() << " patches" << std::endl;
        std::cout << "Pixel values size: " << result.pixel_values.size() << std::endl;
        std::cout << "Pixel values per patch: " << (result.pixel_values.size() / result.pixel_attention_mask.size()) << std::endl;

        // Calculate some statistics
        float min_val = 1e9, max_val = -1e9;
        for (float val : result.pixel_values) {
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
        std::cout << "\nPixel value range: [" << min_val << ", " << max_val << "]" << std::endl;

        // Count valid patches
        int valid_patches = 0;
        for (int mask : result.pixel_attention_mask) {
            if (mask == 1) valid_patches++;
        }
        std::cout << "Valid patches (mask=1): " << valid_patches << std::endl;

        // Save detailed output
        std::cout << "\nSaving detailed output..." << std::endl;
        save_preprocessed_output(output_path, result);

        std::cout << "\n✓ Success!" << std::endl;
        std::cout << "\nNext steps:" << std::endl;
        std::cout << "1. Run the Python script with the same image" << std::endl;
        std::cout << "2. Compare the outputs to verify correctness" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}


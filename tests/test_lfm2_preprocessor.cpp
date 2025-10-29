#include "../cactus/engine/engine.h"
#include <iostream>
#include <iomanip>
#include <string>

using namespace cactus::engine;

void print_first_values(const std::vector<float>& data, int count, const std::string& label) {
    std::cout << label << ": [";
    for (int i = 0; i < std::min(count, (int)data.size()); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(6) << data[i];
    }
    std::cout << "]" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }

    std::string image_path = argv[1];
    std::cout << "Testing LFM2-VL Preprocessor" << std::endl;
    std::cout << "================================" << std::endl;
    std::cout << "Image: " << image_path << std::endl << std::endl;

    // Create preprocessor with HF-matching defaults
    // All defaults already match HuggingFace Lfm2VlImageProcessorFast
    Lfm2VlPreprocessor::Config config;
    // No need to set anything - defaults are correct!
    
    Lfm2VlPreprocessor preprocessor(config);

    try {
        auto result = preprocessor.preprocess_from_file(image_path);

        // Calculate number of tiles (including thumbnail if present)
        int num_regular_tiles = result.image_rows * result.image_cols;
        int num_total_tiles = num_regular_tiles;
        if (config.use_thumbnail && num_regular_tiles > 1 && result.thumbnail_tokens > 0) {
            num_total_tiles += 1;
        }

        std::cout << "Output Structure (matching HuggingFace format):" << std::endl;
        std::cout << "================================================" << std::endl;
        std::cout << "Grid: " << result.image_rows << " x " << result.image_cols << " = " 
                  << num_regular_tiles << " regular tiles" << std::endl;
        if (num_total_tiles > num_regular_tiles) {
            std::cout << "Thumbnail: present (tile " << num_regular_tiles << ")" << std::endl;
        }
        std::cout << "Total tiles: " << num_total_tiles << std::endl << std::endl;

        // Calculate patch dimensions
        int patch_dim = config.patch_size * config.patch_size * 3; // 768
        
        std::cout << "Tensor Shapes:" << std::endl;
        std::cout << "  pixel_values: (1, " << num_total_tiles << ", " 
                  << config.max_num_patches << ", " << patch_dim << ")" << std::endl;
        std::cout << "  spatial_shapes: (1, " << num_total_tiles << ", 2)" << std::endl;
        std::cout << "  pixel_attention_mask: (1, " << num_total_tiles << ", " 
                  << config.max_num_patches << ")" << std::endl << std::endl;

        std::cout << "Shapes for gb->input():" << std::endl;
        std::cout << "  pixel_values_shape: {";
        for (size_t i = 0; i < result.pixel_values_shape.size(); ++i) {
            std::cout << result.pixel_values_shape[i];
            if (i < result.pixel_values_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "}" << std::endl;
        
        std::cout << "  pixel_attention_mask_shape: {";
        for (size_t i = 0; i < result.pixel_attention_mask_shape.size(); ++i) {
            std::cout << result.pixel_attention_mask_shape[i];
            if (i < result.pixel_attention_mask_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "}" << std::endl;
        
        std::cout << "  spatial_shapes_shape: {";
        for (size_t i = 0; i < result.spatial_shapes_shape.size(); ++i) {
            std::cout << result.spatial_shapes_shape[i];
            if (i < result.spatial_shapes_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "}" << std::endl << std::endl;

        // Print spatial_shapes
        std::cout << "spatial_shapes (in patches):" << std::endl;
        std::cout << "[" << std::endl;
        for (int i = 0; i < num_regular_tiles; ++i) {
            std::cout << "  [" << result.num_patches_height << ", " << result.num_patches_width << "]";
            if (i < num_total_tiles - 1) std::cout << ",";
            std::cout << "  # tile " << i << std::endl;
        }
        if (num_total_tiles > num_regular_tiles) {
            // Thumbnail spatial shape
            int thumb_h_patches = result.image_height / config.patch_size;
            int thumb_w_patches = result.image_width / config.patch_size;
            std::cout << "  [" << thumb_h_patches << ", " << thumb_w_patches << "]";
            std::cout << "  # thumbnail" << std::endl;
        }
        std::cout << "]" << std::endl << std::endl;

        // Print first values from each tile
        std::cout << "First 10 values from each tile:" << std::endl;
        std::cout << "================================" << std::endl;
        
        int values_per_patch = patch_dim;
        int patches_per_tile = config.max_num_patches;
        int values_per_tile = patches_per_tile * values_per_patch;

        for (int tile = 0; tile < num_total_tiles; ++tile) {
            int start_idx = tile * values_per_tile;
            std::vector<float> tile_first_10;
            for (int i = 0; i < 10 && (start_idx + i) < result.pixel_values.size(); ++i) {
                tile_first_10.push_back(result.pixel_values[start_idx + i]);
            }
            
            if (tile < num_regular_tiles) {
                print_first_values(tile_first_10, 10, "Tile " + std::to_string(tile));
            } else {
                print_first_values(tile_first_10, 10, "Thumbnail");
            }
        }

        std::cout << std::endl;

        // Print attention mask summary
        std::cout << "Attention Mask Summary:" << std::endl;
        std::cout << "=======================" << std::endl;
        for (int tile = 0; tile < num_total_tiles; ++tile) {
            int start_idx = tile * patches_per_tile;
            int active_patches = 0;
            for (int i = 0; i < patches_per_tile && (start_idx + i) < result.pixel_attention_mask.size(); ++i) {
                if (result.pixel_attention_mask[start_idx + i] == 1) {
                    active_patches++;
                }
            }
            if (tile < num_regular_tiles) {
                std::cout << "Tile " << tile << ": " << active_patches << " / " 
                          << patches_per_tile << " patches active" << std::endl;
            } else {
                std::cout << "Thumbnail: " << active_patches << " / " 
                          << patches_per_tile << " patches active" << std::endl;
            }
        }

        std::cout << std::endl << "Success!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}


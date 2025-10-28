#include "engine.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>

// Include stb_image headers (implementation is in cactus_ffi.cpp)
#include "../ffi/stb_image.h"
#include "../ffi/stb_image_resize2.h"

namespace cactus {
namespace engine {

// PreprocessedImage destructor
Lfm2VlPreprocessor::PreprocessedImage::~PreprocessedImage() {
    pixel_values.clear();
    pixel_attention_mask.clear();
}

// Constructor with config
Lfm2VlPreprocessor::Lfm2VlPreprocessor(const Config& config)
    : config_(config) {}

// Default constructor
Lfm2VlPreprocessor::Lfm2VlPreprocessor() : config_() {}

// Destructor
Lfm2VlPreprocessor::~Lfm2VlPreprocessor() = default;

int Lfm2VlPreprocessor::round_by_factor(int number, int factor) {
    return ((number + factor / 2) / factor) * factor;
}

// helpers
static inline int tokens_from_size(int h, int w, int patch, int dwn) {
    // patches per dim = h/patch, w/patch (divisible due to snap)
    const int ph = h / patch;
    const int pw = w / patch;
    // downsampled tokens per dim: ceil(ph/dwn), ceil(pw/dwn)
    const int th = (ph + dwn - 1) / dwn;
    const int tw = (pw + dwn - 1) / dwn;
    return th * tw;
}


std::pair<int,int> Lfm2VlPreprocessor::smart_resize(int height, int width) {
    const int patch = config_.patch_size;
    const int mult  = config_.patch_size * config_.downsample_factor;
    const int64_t min_pixels = (int64_t)config_.min_image_tokens * patch * patch * config_.downsample_factor * config_.downsample_factor;
    const int64_t max_pixels = (int64_t)config_.max_image_tokens * patch * patch * config_.downsample_factor * config_.downsample_factor;

    int h_bar = std::max(patch, round_by_factor(height, patch));
    int w_bar = std::max(patch, round_by_factor(width,  patch));
    printf("[smart_resize] in=%dx%d  rounded=%dx%d  min_px=%lld max_px=%lld\n",
           width, height, w_bar, h_bar, (long long)min_pixels, (long long)max_pixels);

    if ((int64_t)h_bar * w_bar > max_pixels) {
        float beta = std::sqrt((float)height * (float)width / (float)max_pixels);
        h_bar = std::max(patch, (int)std::floor(height / beta / patch) * patch);
        w_bar = std::max(patch, (int)std::floor(width  / beta / patch) * patch);
        printf("[smart_resize] scale-down beta=%.4f -> %dx%d\n", beta, w_bar, h_bar);
    } else if ((int64_t)h_bar * w_bar < min_pixels) {
        float beta = std::sqrt((float)min_pixels / ((float)height * (float)width));
        h_bar = (int)std::ceil(height * beta / patch) * patch;
        w_bar = (int)std::ceil(width  * beta / patch) * patch;
        printf("[smart_resize] scale-up   beta=%.4f -> %dx%d\n", beta, w_bar, h_bar);
    }

    int new_h = std::max(mult, round_by_factor(h_bar, mult));
    int new_w = std::max(mult, round_by_factor(w_bar, mult));
    printf("[smart_resize] snap_to_%d -> %dx%d\n", mult, new_w, new_h);
    return {new_w, new_h};
}


bool Lfm2VlPreprocessor::is_image_too_large(int height, int width) {
    const int patch = config_.patch_size;               // 16
    const int dwn   = config_.downsample_factor;        // 2
    const int h_bar = std::max(patch, round_by_factor(height, patch));
    const int w_bar = std::max(patch, round_by_factor(width,  patch));

    const int64_t lhs = (int64_t)h_bar * (int64_t)w_bar;
    const int64_t max_pixels = (int64_t)config_.max_image_tokens * patch * patch * dwn * dwn;
    const double thresh = (double)max_pixels * config_.max_pixels_tolerance;

    printf("[too_large] raw=%dx%d  h_bar=%d w_bar=%d  lhs=%lld  max_pixels=%lld  tol=%.3f  thresh=%.1f  -> %d\n",
           width, height, h_bar, w_bar, (long long)lhs, (long long)max_pixels, config_.max_pixels_tolerance, thresh,
           (int)(lhs > thresh));

    return lhs > thresh;
}


std::pair<int, int> Lfm2VlPreprocessor::find_closest_aspect_ratio(float aspect_ratio, int width, int height) {
    float best_ratio_diff = std::numeric_limits<float>::infinity();
    std::pair<int, int> best_ratio = {1, 1};
    int area = width * height;

    // Generate target ratios
    std::vector<std::pair<int, int>> target_ratios;
    for (int n = config_.min_tiles; n <= config_.max_tiles; ++n) {
        for (int w = 1; w <= n; ++w) {
            for (int h = 1; h <= n; ++h) {
                int total_tiles = w * h;
                if (total_tiles >= config_.min_tiles && total_tiles <= config_.max_tiles) {
                    target_ratios.push_back({w, h});
                }
            }
        }
    }

    // Remove duplicates and sort
    std::sort(target_ratios.begin(), target_ratios.end());
    target_ratios.erase(std::unique(target_ratios.begin(), target_ratios.end()), target_ratios.end());

    for (const auto& ratio : target_ratios) {
        float target_aspect_ratio = static_cast<float>(ratio.first) / ratio.second;
        float ratio_diff = std::abs(aspect_ratio - target_aspect_ratio);

        if (ratio_diff < best_ratio_diff) {
            best_ratio_diff = ratio_diff;
            best_ratio = ratio;
        } else if (ratio_diff == best_ratio_diff) {
            int target_area = config_.tile_size * config_.tile_size * ratio.first * ratio.second;
            if (area > 0.5f * target_area) {
                best_ratio = ratio;
            }
        }
    }

    printf("[grid_search] best gw=%d gh=%d  best_ratio=%.6f\n",
        best_ratio.first, best_ratio.second, (float)best_ratio.first / best_ratio.second);
 
    return best_ratio;
}

std::pair<int,int> Lfm2VlPreprocessor::get_grid_layout(int height, int width) {
    float aspect_ratio = (float)width / (float)height;
    printf("[grid] want aspect=%.6f  min_tiles=%d max_tiles=%d  tile=%d\n",
           aspect_ratio, config_.min_tiles, config_.max_tiles, config_.tile_size);

    auto [grid_width, grid_height] = find_closest_aspect_ratio(aspect_ratio, width, height);
    printf("[grid] chosen gw=%d gh=%d  -> target=%dx%d\n",
           grid_width, grid_height, grid_width * config_.tile_size, grid_height * config_.tile_size);

    int target_width  = config_.tile_size * grid_width;
    int target_height = config_.tile_size * grid_height;
    return {target_width, target_height};
}


Lfm2VlPreprocessor::PreprocessedImage Lfm2VlPreprocessor::preprocess_from_file(const std::string& image_path) {
    int width, height, channels;
    unsigned char* img_data = stbi_load(image_path.c_str(), &width, &height, &channels, 0);
    
    if (!img_data) {
        throw std::runtime_error("Failed to load image: " + image_path + " - " + std::string(stbi_failure_reason()));
    }

    PreprocessedImage result;
    result = preprocess_from_memory(img_data, width, height, channels);
    stbi_image_free(img_data);
    return result;
}

Lfm2VlPreprocessor::PreprocessedImage Lfm2VlPreprocessor::preprocess_from_memory(
    const unsigned char* img_data, int width, int height, int channels) {
    
    if (!img_data) {
        throw std::runtime_error("Invalid image data pointer");
    }

    // Convert to RGB if needed
    std::vector<unsigned char> rgb_data;
    if (config_.do_convert_rgb && channels != 3) {
        rgb_data = convert_to_rgb(img_data, width, height, channels);
        img_data = rgb_data.data();
        channels = 3;
    }

    if (channels != 3) {
        throw std::runtime_error("Image must have 3 channels (RGB)");
    }

    PreprocessedImage result;

    // Smart resize to get target dimensions
    // Smart resize first (snap to multiples of 32)
    auto [new_width, new_height] = smart_resize(height, width);
    result.image_width  = new_width;
    result.image_height = new_height;

    const int patch = config_.patch_size;
    const int dwn   = config_.downsample_factor;

    auto tokens_from_size = [](int h, int w, int patch, int dwn) {
        const int ph = h / patch, pw = w / patch;
        const int th = (ph + dwn - 1) / dwn;
        const int tw = (pw + dwn - 1) / dwn;
        return th * tw;
    };

    const int single_tokens = tokens_from_size(new_height, new_width, patch, dwn);
    const bool split_by_pixels = config_.do_image_splitting && is_image_too_large(height, width);
    const bool split_by_budget = config_.do_image_splitting &&
                                (single_tokens >= config_.max_image_tokens) &&
                                (single_tokens <= config_.max_num_patches);
    const bool do_split = split_by_pixels || split_by_budget;

    printf("[decision] raw=%dx%d new=%dx%d single_tokens=%d  per_tile=%d  total_cap=%d  split_pixels=%d split_budget=%d  -> do_split=%d\n",
        width, height, new_width, new_height, single_tokens,
        config_.max_image_tokens, config_.max_num_patches,
        (int)split_by_pixels, (int)split_by_budget, (int)do_split);


    
    std::vector<std::vector<float>> all_tile_patches;
    
    if (do_split) {
        // Get grid layout for splitting
        auto [grid_target_width, grid_target_height] = get_grid_layout(height, width);
        int grid_cols = grid_target_width / config_.tile_size;
        int grid_rows = grid_target_height / config_.tile_size;
        
        result.image_rows = grid_rows;
        result.image_cols = grid_cols;
        
        // Resize to grid dimensions
        std::vector<unsigned char> resized_data = resize_image(img_data, width, height, 
                                                               grid_target_width, grid_target_height, channels);
        
        // Split into tiles and process each one
        for (int row = 0; row < grid_rows; ++row) {
            for (int col = 0; col < grid_cols; ++col) {
                // Extract tile
                std::vector<unsigned char> tile_data(config_.tile_size * config_.tile_size * channels);
                for (int y = 0; y < config_.tile_size; ++y) {
                    for (int x = 0; x < config_.tile_size; ++x) {
                        int src_y = row * config_.tile_size + y;
                        int src_x = col * config_.tile_size + x;
                        int src_idx = (src_y * grid_target_width + src_x) * channels;
                        int dst_idx = (y * config_.tile_size + x) * channels;
                        for (int c = 0; c < channels; ++c) {
                            tile_data[dst_idx + c] = resized_data[src_idx + c];
                        }
                    }
                }
                
                // Normalize and patchify tile
                std::vector<float> tile_normalized = normalize_image(tile_data.data(), config_.tile_size, 
                                                                     config_.tile_size, channels);
                auto tile_patches = convert_image_to_patches(tile_normalized, config_.tile_size, 
                                                            config_.tile_size, channels, config_.patch_size);
                
                // Add tile patches to collection
                all_tile_patches.insert(all_tile_patches.end(), tile_patches.begin(), tile_patches.end());
            }
        }
        
        // Optionally add thumbnail after all tiles
        if (config_.use_thumbnail && grid_rows * grid_cols > 1) {
            std::vector<unsigned char> thumbnail_data = resize_image(img_data, width, height, 
                                                                    new_width, new_height, channels);
            std::vector<float> thumbnail_normalized = normalize_image(thumbnail_data.data(), 
                                                                      new_width, new_height, channels);
            auto thumbnail_patches = convert_image_to_patches(thumbnail_normalized, new_width, 
                                                             new_height, channels, config_.patch_size);
            all_tile_patches.insert(all_tile_patches.end(), thumbnail_patches.begin(), thumbnail_patches.end());
        }
    } else {
        // No splitting - single tile
        result.image_rows = 1;
        result.image_cols = 1;
        
        std::vector<unsigned char> resized_data = resize_image(img_data, width, height, 
                                                               new_width, new_height, channels);
        std::vector<float> normalized_data = normalize_image(resized_data.data(), new_width, new_height, channels);
        all_tile_patches = convert_image_to_patches(normalized_data, new_width, new_height, channels, config_.patch_size);
    }

    // Calculate tokens per tile (after downsampling), path-aware
    if (do_split) {
        // each tile is 512x512
        const int patches_per_tile_side = config_.tile_size / config_.patch_size;  // 512/16 = 32
        const int tokens_side = (patches_per_tile_side + dwn - 1) / dwn;           // ceil(32/2) = 16
        result.tokens_per_tile = tokens_side * tokens_side;                        // 256
    } else {
        // single-view path: tokens come from the smart-resized dimensions
        result.tokens_per_tile = tokens_from_size(result.image_height, result.image_width, patch, dwn);
    }

    // Calculate thumbnail tokens if enabled
    result.thumbnail_tokens = 0;
    if (config_.use_thumbnail && do_split && result.image_rows * result.image_cols > 1) {
        int thumb_h_patches = new_height / config_.patch_size;
        int thumb_w_patches = new_width / config_.patch_size;
        int thumb_h_tokens = static_cast<int>(std::ceil(static_cast<float>(thumb_h_patches) / config_.downsample_factor));
        int thumb_w_tokens = static_cast<int>(std::ceil(static_cast<float>(thumb_w_patches) / config_.downsample_factor));
        result.thumbnail_tokens = thumb_h_tokens * thumb_w_tokens;
    }
    
    // Total sequence length
    int tile_tokens = result.image_rows * result.image_cols * result.tokens_per_tile;
    int seq_len = tile_tokens + result.thumbnail_tokens;
    
    // Store metadata before padding
    int saved_rows = result.image_rows;
    int saved_cols = result.image_cols;
    int saved_width = result.image_width;
    int saved_height = result.image_height;
    int saved_tokens_per_tile = result.tokens_per_tile;
    int saved_thumbnail_tokens = result.thumbnail_tokens;
    
    // Pad to sequence length
    result = pad_patches(all_tile_patches, config_.tile_size, config_.tile_size, config_.patch_size, seq_len);
    
    // Restore metadata
    result.image_rows = saved_rows;
    result.image_cols = saved_cols;
    result.image_width = saved_width;
    result.image_height = saved_height;
    result.tokens_per_tile = saved_tokens_per_tile;
    result.thumbnail_tokens = saved_thumbnail_tokens;
    result.num_patches_height = config_.tile_size / config_.patch_size;
    result.num_patches_width = config_.tile_size / config_.patch_size;

    printf("[tokens] grid=%dx%d  per_tile=%d  thumb=%d  -> total=%d\n",
        result.image_cols, result.image_rows, result.tokens_per_tile, result.thumbnail_tokens,
        result.image_cols * result.image_rows * result.tokens_per_tile + result.thumbnail_tokens);
 

    return result;
}

std::vector<unsigned char> Lfm2VlPreprocessor::convert_to_rgb(
    const unsigned char* img_data, int width, int height, int channels) {
    
    std::vector<unsigned char> rgb_data(width * height * 3);
    
    if (channels == 1) {
        for (int i = 0; i < width * height; ++i) {
            rgb_data[i * 3 + 0] = img_data[i];
            rgb_data[i * 3 + 1] = img_data[i];
            rgb_data[i * 3 + 2] = img_data[i];
        }
    } else if (channels == 4) {
        for (int i = 0; i < width * height; ++i) {
            rgb_data[i * 3 + 0] = img_data[i * 4 + 0];
            rgb_data[i * 3 + 1] = img_data[i * 4 + 1];
            rgb_data[i * 3 + 2] = img_data[i * 4 + 2];
        }
    } else if (channels == 2) {
        for (int i = 0; i < width * height; ++i) {
            rgb_data[i * 3 + 0] = img_data[i * 2 + 0];
            rgb_data[i * 3 + 1] = img_data[i * 2 + 0];
            rgb_data[i * 3 + 2] = img_data[i * 2 + 0];
        }
    } else {
        throw std::runtime_error("Unsupported number of channels: " + std::to_string(channels));
    }
    
    return rgb_data;
}

std::vector<unsigned char> Lfm2VlPreprocessor::resize_image(
    const unsigned char* img_data, int src_width, int src_height,
    int dst_width, int dst_height, int channels) {
    
    std::vector<unsigned char> resized_data(dst_width * dst_height * channels);
    
    stbir_pixel_layout layout = (channels == 1) ? STBIR_1CHANNEL : 
                                (channels == 3) ? STBIR_RGB : STBIR_RGBA;
    
    unsigned char* result = stbir_resize_uint8_linear(
        img_data, src_width, src_height, 0,
        resized_data.data(), dst_width, dst_height, 0,
        layout
    );

    if (!result) {
        throw std::runtime_error("Failed to resize image");
    }

    return resized_data;
}

std::vector<float> Lfm2VlPreprocessor::normalize_image(
    const unsigned char* img_data, int width, int height, int channels) {
    
    size_t total_pixels = width * height * channels;
    std::vector<float> normalized(total_pixels);

    for (size_t i = 0; i < width * height; ++i) {
        for (int c = 0; c < channels; ++c) {
            size_t idx = i * channels + c;
            float pixel = static_cast<float>(img_data[idx]);
            
            if (config_.do_rescale) {
                pixel *= config_.rescale_factor;
            }
            
            if (config_.do_normalize) {
                pixel = (pixel - config_.image_mean[c]) / config_.image_std[c];
            }
            
            normalized[idx] = pixel;
        }
    }

    return normalized;
}

std::vector<std::vector<float>> Lfm2VlPreprocessor::convert_image_to_patches(
    const std::vector<float>& image, int width, int height, int channels, int patch_size) {
    
    int num_patches_height = height / patch_size;
    int num_patches_width = width / patch_size;
    int num_patches = num_patches_height * num_patches_width;
    int patch_elements = patch_size * patch_size * channels;

    std::vector<std::vector<float>> patches(num_patches, std::vector<float>(patch_elements));

    for (int ph = 0; ph < num_patches_height; ++ph) {
        for (int pw = 0; pw < num_patches_width; ++pw) {
            int patch_idx = ph * num_patches_width + pw;
            
            for (int y = 0; y < patch_size; ++y) {
                for (int x = 0; x < patch_size; ++x) {
                    int img_y = ph * patch_size + y;
                    int img_x = pw * patch_size + x;
                    int img_idx = (img_y * width + img_x) * channels;
                    int patch_offset = (y * patch_size + x) * channels;
                    
                    for (int c = 0; c < channels; ++c) {
                        patches[patch_idx][patch_offset + c] = image[img_idx + c];
                    }
                }
            }
        }
    }

    return patches;
}

Lfm2VlPreprocessor::PreprocessedImage Lfm2VlPreprocessor::pad_patches(
    const std::vector<std::vector<float>>& patches,
    int width, int height, int patch_size, int max_num_patches) {
    
    PreprocessedImage result;
    
    int actual_num_patches = patches.size();
    int patch_elements = patches.empty() ? 0 : patches[0].size();
    
    result.num_patches_height = height / patch_size;
    result.num_patches_width = width / patch_size;
    result.actual_num_patches = actual_num_patches;

    // Initialize with zeros (padding)
    result.pixel_values.resize(max_num_patches * patch_elements, 0.0f);
    result.pixel_attention_mask.resize(max_num_patches, 0);

    // Copy actual patches
    for (int i = 0; i < actual_num_patches && i < max_num_patches; ++i) {
        std::copy(patches[i].begin(), patches[i].end(), 
                 result.pixel_values.begin() + i * patch_elements);
        result.pixel_attention_mask[i] = 1;
    }

    return result;
}

} // namespace engine
} // namespace cactus


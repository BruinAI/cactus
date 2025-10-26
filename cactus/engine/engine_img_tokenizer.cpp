#include "engine.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>

// Include stb_image headers
#define STB_IMAGE_IMPLEMENTATION
#include "../ffi/stb_image_impl.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../ffi/stb_image_resize_impl.h"

namespace cactus {
namespace engine {

// PreprocessedImage destructor
SigLip2Preprocessor::PreprocessedImage::~PreprocessedImage() {
    pixel_values.clear();
    pixel_attention_mask.clear();
}

// Constructor with config
SigLip2Preprocessor::SigLip2Preprocessor(const Config& config)
    : config_(config) {}

// Default constructor
SigLip2Preprocessor::SigLip2Preprocessor() : config_() {}

// Destructor
SigLip2Preprocessor::~SigLip2Preprocessor() = default;

/**
 * Load and preprocess an image from a file path
 */
SigLip2Preprocessor::PreprocessedImage SigLip2Preprocessor::preprocess_from_file(const std::string& image_path) {
    int width, height, channels;
    unsigned char* img_data = stbi_load(image_path.c_str(), &width, &height, &channels, 0);
    
    if (!img_data) {
        throw std::runtime_error("Failed to load image: " + image_path + " - " + std::string(stbi_failure_reason()));
    }

    PreprocessedImage result;
    try {
        result = preprocess_from_memory(img_data, width, height, channels);
    } catch (...) {
        stbi_image_free(img_data);
        throw;
    }
    
    stbi_image_free(img_data);
    return result;
}

/**
 * Preprocess an image already loaded in memory
 */
SigLip2Preprocessor::PreprocessedImage SigLip2Preprocessor::preprocess_from_memory(
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

    // Calculate target size based on max_num_patches
    int target_height = height;
    int target_width = width;
    
    if (config_.do_resize) {
        auto [new_height, new_width] = get_image_size_for_max_num_patches(
            height, width, config_.patch_size, config_.max_num_patches, config_.binary_search_eps
        );
        target_height = new_height;
        target_width = new_width;
    }

    // Resize image if necessary
    std::vector<unsigned char> resized_data;
    if (target_height != height || target_width != width) {
        resized_data = resize_image(img_data, width, height, target_width, target_height, channels);
        img_data = resized_data.data();
        width = target_width;
        height = target_height;
    }

    // Convert to float, rescale and normalize
    std::vector<float> normalized_data = normalize_image(img_data, width, height, channels);

    // Convert to patches
    auto patches = convert_image_to_patches(normalized_data, width, height, channels, config_.patch_size);

    // Pad patches to max_num_patches
    auto result = pad_patches(patches, width, height, config_.patch_size, config_.max_num_patches);

    return result;
}

/**
 * Convert image to RGB (handles grayscale and RGBA)
 */
std::vector<unsigned char> SigLip2Preprocessor::convert_to_rgb(
    const unsigned char* img_data, int width, int height, int channels) {
    
    std::vector<unsigned char> rgb_data(width * height * 3);
    
    if (channels == 1) {
        // Grayscale to RGB: replicate the single channel
        for (int i = 0; i < width * height; ++i) {
            rgb_data[i * 3 + 0] = img_data[i];
            rgb_data[i * 3 + 1] = img_data[i];
            rgb_data[i * 3 + 2] = img_data[i];
        }
    } else if (channels == 4) {
        // RGBA to RGB: drop alpha channel
        for (int i = 0; i < width * height; ++i) {
            rgb_data[i * 3 + 0] = img_data[i * 4 + 0];
            rgb_data[i * 3 + 1] = img_data[i * 4 + 1];
            rgb_data[i * 3 + 2] = img_data[i * 4 + 2];
        }
    } else if (channels == 2) {
        // Grayscale + Alpha to RGB: use grayscale channel
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

/**
 * Get scaled image size that respects a given scale factor
 */
int SigLip2Preprocessor::get_scaled_image_size(float scale, int size, int patch_size) {
    float scaled_size = size * scale;
    // Make divisible by patch_size
    int result = static_cast<int>(std::ceil(scaled_size / patch_size)) * patch_size;
    // Ensure at least 1 patch
    result = std::max(patch_size, result);
    return result;
}

/**
 * Binary search to find optimal image dimensions for max_num_patches
 * Matches the Python implementation exactly
 */
std::pair<int, int> SigLip2Preprocessor::get_image_size_for_max_num_patches(
    int image_height, int image_width, int patch_size, int max_num_patches, float eps) {
    
    float scale_min = eps / 10.0f;
    float scale_max = 100.0f;

    // Binary search for optimal scale
    while ((scale_max - scale_min) >= eps) {
        float scale = (scale_min + scale_max) / 2.0f;
        
        int target_height = get_scaled_image_size(scale, image_height, patch_size);
        int target_width = get_scaled_image_size(scale, image_width, patch_size);
        
        int num_patches = (target_height / patch_size) * (target_width / patch_size);

        if (num_patches <= max_num_patches) {
            scale_min = scale;
        } else {
            scale_max = scale;
        }
    }

    float final_scale = scale_min;
    int target_height = get_scaled_image_size(final_scale, image_height, patch_size);
    int target_width = get_scaled_image_size(final_scale, image_width, patch_size);

    return {target_height, target_width};
}

/**
 * Resize image using bilinear interpolation (matching PIL's BILINEAR)
 */
std::vector<unsigned char> SigLip2Preprocessor::resize_image(
    const unsigned char* img_data, int src_width, int src_height,
    int dst_width, int dst_height, int channels) {
    
    std::vector<unsigned char> resized_data(dst_width * dst_height * channels);
    
    // Use stb_image_resize with linear interpolation (matches PIL BILINEAR)
    stbir_pixel_layout layout = (channels == 1) ? STBIR_1CHANNEL : 
                                (channels == 3) ? STBIR_RGB : STBIR_RGBA;
    
    unsigned char* result = stbir_resize_uint8_linear(
        img_data, src_width, src_height, 0,  // 0 = tightly packed
        resized_data.data(), dst_width, dst_height, 0,
        layout
    );

    if (!result) {
        throw std::runtime_error("Failed to resize image");
    }

    return resized_data;
}

/**
 * Normalize image: rescale to [0,1] and apply mean/std normalization
 * Formula: output = (input * rescale_factor - mean) / std
 */
std::vector<float> SigLip2Preprocessor::normalize_image(
    const unsigned char* img_data, int width, int height, int channels) {
    
    size_t total_pixels = width * height * channels;
    std::vector<float> normalized(total_pixels);

    for (size_t i = 0; i < width * height; ++i) {
        for (int c = 0; c < channels; ++c) {
            size_t idx = i * channels + c;
            float pixel = static_cast<float>(img_data[idx]);
            
            // Apply rescaling
            if (config_.do_rescale) {
                pixel *= config_.rescale_factor;
            }
            
            // Apply normalization
            if (config_.do_normalize) {
                pixel = (pixel - config_.image_mean[c]) / config_.image_std[c];
            }
            
            normalized[idx] = pixel;
        }
    }

    return normalized;
}

/**
 * Convert 3D image (height, width, channels) to 2D patches
 * Output shape: (num_patches, patch_size * patch_size * channels)
 * 
 * This matches the Python implementation:
 * 1. Reshape to (num_patches_h, patch_size, num_patches_w, patch_size, channels)
 * 2. Transpose to (num_patches_h, num_patches_w, patch_size, patch_size, channels)
 * 3. Flatten to (num_patches_h * num_patches_w, patch_size * patch_size * channels)
 */
std::vector<std::vector<float>> SigLip2Preprocessor::convert_image_to_patches(
    const std::vector<float>& image, int width, int height, int channels, int patch_size) {
    
    int num_patches_height = height / patch_size;
    int num_patches_width = width / patch_size;
    int num_patches = num_patches_height * num_patches_width;
    int patch_elements = patch_size * patch_size * channels;

    std::vector<std::vector<float>> patches(num_patches, std::vector<float>(patch_elements));

    // Extract patches
    for (int ph = 0; ph < num_patches_height; ++ph) {
        for (int pw = 0; pw < num_patches_width; ++pw) {
            int patch_idx = ph * num_patches_width + pw;
            
            // Extract patch_size x patch_size region
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

/**
 * Pad patches array to max_num_patches and create attention mask
 * Mask is 1 for real patches, 0 for padding
 */
SigLip2Preprocessor::PreprocessedImage SigLip2Preprocessor::pad_patches(
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

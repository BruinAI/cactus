#include "../cactus/engine/engine.h"
#include "../cactus/models/model.h"
#include "../cactus/graph/graph.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

void print_usage() {
    std::cerr << "Usage: siglip2_debug_runner --model <model_dir> --image <image_path>\n"
              << "       [--dump-output <file>] [--dump-layers <file>]\n";
}

void print_tensor_stats(const std::vector<float>& data, const std::string& name, size_t limit = 16) {
    if (data.empty()) {
        std::cout << name << ": EMPTY" << std::endl;
        return;
    }
    
    float min_val = data[0];
    float max_val = data[0];
    double sum = 0.0;
    double sum_sq = 0.0;
    
    for (float val : data) {
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
        sum += val;
        sum_sq += static_cast<double>(val) * static_cast<double>(val);
    }
    
    double mean = sum / static_cast<double>(data.size());
    double variance = std::max(0.0, (sum_sq / static_cast<double>(data.size())) - mean * mean);
    double stddev = std::sqrt(variance);
    
    std::cout << name << ":" << std::endl;
    std::cout << "  Min=" << std::setprecision(6) << min_val
              << " Max=" << max_val
              << " Mean=" << mean
              << " Std=" << stddev << std::endl;
    
    std::cout << "  First " << std::min(limit, data.size()) << " values:";
    for (size_t i = 0; i < std::min(limit, data.size()); ++i) {
        std::cout << ' ' << std::setprecision(5) << data[i];
    }
    if (data.size() > limit) {
        std::cout << " ...";
    }
    std::cout << std::endl;
}

void dump_node_output(CactusGraph* gb, size_t node_id, 
                      const std::string& name, std::ostream& out) {
    const auto& buf = gb->get_output_buffer(node_id);
    void* output_ptr = gb->get_output(node_id);
    if (!output_ptr) {
        out << name << ": NO OUTPUT" << std::endl;
        return;
    }
    
    size_t total_size = 1;
    for (auto dim : buf.shape) {
        total_size *= dim;
    }
    
    std::vector<float> data(total_size);
    if (buf.precision == Precision::FP32) {
        float* ptr = static_cast<float*>(output_ptr);
        std::copy(ptr, ptr + total_size, data.begin());
    } else if (buf.precision == Precision::FP16) {
        __fp16* ptr = static_cast<__fp16*>(output_ptr);
        for (size_t i = 0; i < total_size; ++i) {
            data[i] = static_cast<float>(ptr[i]);
        }
    } else if (buf.precision == Precision::INT8) {
        int8_t* ptr = static_cast<int8_t*>(output_ptr);
        float scale = buf.quantization_scale;
        for (size_t i = 0; i < total_size; ++i) {
            data[i] = ptr[i] * scale;
        }
    }
    
    out << name << " shape=[";
    for (size_t i = 0; i < buf.shape.size(); ++i) {
        out << buf.shape[i];
        if (i + 1 < buf.shape.size()) out << ",";
    }
    out << "]" << std::endl;
    
    float min_val = data[0];
    float max_val = data[0];
    double sum = 0.0;
    
    for (float val : data) {
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
        sum += val;
    }
    
    out << "  Min=" << min_val << " Max=" << max_val 
        << " Mean=" << (sum / data.size()) << std::endl;
    
    // Print first 32 values (position 0)
    out << "  First 32 values (pos 0):";
    for (size_t i = 0; i < std::min(size_t(32), data.size()); ++i) {
        if (i % 8 == 0) out << "\n    ";
        out << std::setw(10) << std::setprecision(5) << data[i];
    }
    out << std::endl;
    
    // If we have a multi-dimensional tensor with a sequence dimension, 
    // also print values from other sequence positions
    if (buf.shape.size() >= 2 && buf.shape[0] > 1) {
        size_t seq_len = buf.shape[0];
        size_t stride = 1;
        for (size_t i = 1; i < buf.shape.size(); ++i) {
            stride *= buf.shape[i];
        }
        
        // Print values from positions 1, 2, and last position
        std::vector<size_t> positions_to_check;
        if (seq_len > 1) positions_to_check.push_back(1);
        if (seq_len > 2) positions_to_check.push_back(2);
        if (seq_len > 10) positions_to_check.push_back(seq_len - 1);  // last position
        
        for (size_t pos : positions_to_check) {
            if (pos < seq_len) {
                out << "  Values at position " << pos << ":";
                size_t start_idx = pos * stride;
                for (size_t i = 0; i < std::min(size_t(24), stride); ++i) {
                    if (i % 8 == 0) out << "\n    ";
                    if (start_idx + i < data.size()) {
                        out << std::setw(10) << std::setprecision(5) << data[start_idx + i];
                    }
                }
                out << std::endl;
            }
        }
    }
}

} // namespace

using namespace cactus::engine;

int main(int argc, char** argv) {
    std::string model_dir;
    std::string image_path;
    std::string dump_output_path;
    std::string dump_layers_path;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--model" && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (arg == "--image" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (arg == "--dump-output" && i + 1 < argc) {
            dump_output_path = argv[++i];
        } else if (arg == "--dump-layers" && i + 1 < argc) {
            dump_layers_path = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage();
            return 1;
        }
    }
    
    if (model_dir.empty() || image_path.empty()) {
        std::cerr << "Error: --model and --image are required." << std::endl;
        print_usage();
        return 1;
    }
    
    std::cout << "=== Siglip2 Vision Debug Runner ===" << std::endl;
    std::cout << "Model: " << model_dir << std::endl;
    std::cout << "Image: " << image_path << std::endl;
    std::cout << std::endl;
    
    Config config;
    if (!config.from_json(model_dir + "/config.txt")) {
        std::cerr << "Failed to load config from " << model_dir << "/config.txt" << std::endl;
        return 1;
    }
    
    std::cout << "Config:" << std::endl;
    std::cout << "  vision_embed_dim: " << config.vision_embed_dim << std::endl;
    std::cout << "  vision_num_layers: " << config.vision_num_layers << std::endl;
    std::cout << "  vision_patch_size: " << config.vision_patch_size << std::endl;
    std::cout << "  vision_attention_heads: " << config.vision_attention_heads << std::endl;
    std::cout << std::endl;
    
    Siglip2VisionModel model(config);
    
    if (!model.init(model_dir, 0)) {
        std::cerr << "Failed to initialize model" << std::endl;
        return 1;
    }
    
    std::cout << "Model initialized successfully" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Preprocessing image..." << std::endl;
    auto preprocessed = model.get_preprocessor().preprocess_from_file(image_path);
    
    std::cout << "Image preprocessing complete:" << std::endl;
    std::cout << "  Grid: " << preprocessed.image_rows << "x" << preprocessed.image_cols << std::endl;
    std::cout << "  Tokens per tile: " << preprocessed.tokens_per_tile << std::endl;
    std::cout << "  Thumbnail tokens: " << preprocessed.thumbnail_tokens << std::endl;
    int total_tokens = preprocessed.tokens_per_tile * preprocessed.image_rows * preprocessed.image_cols 
                      + preprocessed.thumbnail_tokens;
    std::cout << "  Total vision tokens: " << total_tokens << std::endl;
    std::cout << std::endl;
    
    std::cout << "Running vision forward pass..." << std::endl;
    size_t output_node = model.forward_vision(preprocessed);
    
    auto* gb = static_cast<CactusGraph*>(model.graph_handle_);
    gb->execute();
    
    std::cout << "Forward pass complete" << std::endl;
    std::cout << std::endl;
    
    const auto& debug_nodes = model.get_debug_nodes();
    std::cout << "Captured " << debug_nodes.size() << " debug nodes" << std::endl;
    std::cout << std::endl;
    
    const auto& output_buf = gb->get_output_buffer(output_node);
    size_t total_output_size = 1;
    for (auto dim : output_buf.shape) {
        total_output_size *= dim;
    }
    
    std::vector<float> final_features(total_output_size);
    void* output_ptr = gb->get_output(output_node);
    
    if (output_buf.precision == Precision::FP32) {
        float* ptr = static_cast<float*>(output_ptr);
        std::copy(ptr, ptr + total_output_size, final_features.begin());
    } else if (output_buf.precision == Precision::FP16) {
        __fp16* ptr = static_cast<__fp16*>(output_ptr);
        for (size_t i = 0; i < total_output_size; ++i) {
            final_features[i] = static_cast<float>(ptr[i]);
        }
    }
    
    std::cout << "=== Final Vision Output ===" << std::endl;
    std::cout << "Shape: [";
    for (size_t i = 0; i < output_buf.shape.size(); ++i) {
        std::cout << output_buf.shape[i];
        if (i + 1 < output_buf.shape.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    print_tensor_stats(final_features, "Final Features");
    std::cout << std::endl;
    
    if (!dump_output_path.empty()) {
        std::ofstream out(dump_output_path);
        if (out.is_open()) {
            out << std::setprecision(10);
            for (float val : final_features) {
                out << val << '\n';
            }
            std::cout << "Wrote final features to: " << dump_output_path << std::endl;
        } else {
            std::cerr << "Failed to open output file: " << dump_output_path << std::endl;
        }
    }
    
    if (!dump_layers_path.empty()) {
        std::ofstream layer_out(dump_layers_path);
        if (layer_out.is_open()) {
            layer_out << "=== Debug Layer Outputs ===" << std::endl;
            layer_out << "Total debug nodes: " << debug_nodes.size() << std::endl;
            layer_out << std::endl;
            
            for (const auto& debug_node : debug_nodes) {
                layer_out << "----------------------------------------" << std::endl;
                layer_out << "Layer " << debug_node.layer_idx << ": " << debug_node.name << std::endl;
                dump_node_output(gb, debug_node.node_id, debug_node.name, layer_out);
                layer_out << std::endl;
            }
            
            std::cout << "Wrote layer debug outputs to: " << dump_layers_path << std::endl;
        } else {
            std::cerr << "Failed to open layer output file: " << dump_layers_path << std::endl;
        }
    }
    
    std::cout << std::endl;
    std::cout << "Debug run complete!" << std::endl;
    
    return 0;
}


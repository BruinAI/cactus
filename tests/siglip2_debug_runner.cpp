#include "../cactus/engine/engine.h"
#include "../cactus/models/model.h"
#include "../cactus/graph/graph.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <system_error>
#include <string>
#include <vector>

namespace {

void print_usage() {
    std::cerr << "Usage: siglip2_debug_runner --model <model_dir> --image <image_path>\n"
              << "       [--dump-output <file>] [--dump-layers <file>] [--dump-patch-embeds <file>]\n"
              << "       [--dump-node <name:path>] [--dump-node <layer:name:path>] (may repeat)\n"
              << "       [--dump-layer0-suite <dir>] [--per-tile-ablation]\n";
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

std::vector<float> extract_node_data(CactusGraph* gb, size_t node_id) {
    const auto& buf = gb->get_output_buffer(node_id);
    void* output_ptr = gb->get_output(node_id);
    if (!output_ptr) {
        return {};
    }

    size_t total_size = 1;
    for (auto dim : buf.shape) {
        total_size *= dim;
    }

    std::vector<float> data(total_size);
    if (buf.precision == Precision::FP32) {
        const float* ptr = static_cast<const float*>(output_ptr);
        std::copy(ptr, ptr + total_size, data.begin());
    } else if (buf.precision == Precision::FP16) {
        const __fp16* ptr = static_cast<const __fp16*>(output_ptr);
        for (size_t i = 0; i < total_size; ++i) {
            data[i] = static_cast<float>(ptr[i]);
        }
    } else if (buf.precision == Precision::INT8) {
        const int8_t* ptr = static_cast<const int8_t*>(output_ptr);
        float scale = buf.quantization_scale;
        for (size_t i = 0; i < total_size; ++i) {
            data[i] = ptr[i] * scale;
        }
    }

    return data;
}

bool write_tensor_binary(const std::string& path, const std::vector<float>& data,
                         const std::vector<size_t>& shape) {
    if (data.empty()) {
        return false;
    }

    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        return false;
    }

    const char magic[8] = {'S','2','V','I','S','P','E','M'};
    out.write(magic, sizeof(magic));
    uint32_t version = 1;
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));

    uint32_t rank = static_cast<uint32_t>(shape.size());
    out.write(reinterpret_cast<const char*>(&rank), sizeof(rank));
    for (size_t dim : shape) {
        uint32_t dim32 = static_cast<uint32_t>(dim);
        out.write(reinterpret_cast<const char*>(&dim32), sizeof(dim32));
    }

    out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    return true;
}

} // namespace

using namespace cactus::engine;

int main(int argc, char** argv) {
    std::string model_dir;
    std::string image_path;
    std::string dump_output_path;
    std::string dump_layers_path;
    std::string dump_patch_embeds_path;
    std::string dump_layer0_suite_dir;
    bool use_tile_ablation = false;
    struct NodeDumpRequest {
        int layer_idx;
        std::string name;
        std::string path;
    };
    std::vector<NodeDumpRequest> node_dump_requests;
    
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
        } else if (arg == "--dump-patch-embeds" && i + 1 < argc) {
            dump_patch_embeds_path = argv[++i];
        } else if (arg == "--dump-node" && i + 1 < argc) {
            std::string spec = argv[++i];
            NodeDumpRequest request;
            request.layer_idx = -1;

            size_t first_colon = spec.find(':');
            if (first_colon == std::string::npos) {
                std::cerr << "Invalid --dump-node spec: " << spec << std::endl;
                return 1;
            }

            size_t second_colon = spec.find(':', first_colon + 1);
            if (second_colon == std::string::npos) {
                request.name = spec.substr(0, first_colon);
                request.path = spec.substr(first_colon + 1);
            } else {
                std::string layer_str = spec.substr(0, first_colon);
                try {
                    request.layer_idx = std::stoi(layer_str);
                } catch (const std::exception&) {
                    std::cerr << "Invalid layer index in --dump-node spec: " << spec << std::endl;
                    return 1;
                }
                request.name = spec.substr(first_colon + 1, second_colon - first_colon - 1);
                request.path = spec.substr(second_colon + 1);
            }

            if (request.name.empty() || request.path.empty()) {
                std::cerr << "Invalid --dump-node spec: " << spec << std::endl;
                return 1;
            }
            node_dump_requests.push_back(std::move(request));
        } else if (arg == "--dump-layer0-suite" && i + 1 < argc) {
            dump_layer0_suite_dir = argv[++i];
        } else if (arg == "--per-tile-ablation") {
            use_tile_ablation = true;
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
    
    std::unique_ptr<Siglip2VisionModel> model;
    if (use_tile_ablation) {
        std::cout << "Per-tile ablation: enabled (tiles processed independently)" << std::endl;
        model = std::make_unique<Siglip2VisionModelTileAblation>(config);
    } else {
        std::cout << "Per-tile ablation: disabled" << std::endl;
        model = std::make_unique<Siglip2VisionModel>(config);
    }

    if (!model->init(model_dir, 0)) {
        std::cerr << "Failed to initialize model" << std::endl;
        return 1;
    }
    
    std::cout << "Model initialized successfully" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Preprocessing image..." << std::endl;
    auto preprocessed = model->get_preprocessor().preprocess_from_file(image_path);
    
    std::cout << "Image preprocessing complete:" << std::endl;
    std::cout << "  Grid: " << preprocessed.image_rows << "x" << preprocessed.image_cols << std::endl;
    std::cout << "  Tokens per tile: " << preprocessed.tokens_per_tile << std::endl;
    std::cout << "  Thumbnail tokens: " << preprocessed.thumbnail_tokens << std::endl;
    int total_tokens = preprocessed.tokens_per_tile * preprocessed.image_rows * preprocessed.image_cols 
                      + preprocessed.thumbnail_tokens;
    std::cout << "  Total vision tokens: " << total_tokens << std::endl;
    std::cout << std::endl;
    
    std::cout << "Running vision forward pass..." << std::endl;
    size_t output_node = model->forward_vision(preprocessed);
    
    auto* gb = static_cast<CactusGraph*>(model->graph_handle_);
    gb->execute();
    
    std::cout << "Forward pass complete" << std::endl;
    std::cout << std::endl;
    
    const auto& debug_nodes = model->get_debug_nodes();
    std::cout << "Captured " << debug_nodes.size() << " debug nodes" << std::endl;
    std::cout << std::endl;

    auto find_debug_node = [&](const std::string& target_name, int target_layer) -> const Model::DebugNode* {
        auto it = std::find_if(debug_nodes.begin(), debug_nodes.end(), [&](const Model::DebugNode& node) {
            bool name_match = node.name == target_name;
            bool layer_match = target_layer < 0 || static_cast<int>(node.layer_idx) == target_layer;
            return name_match && layer_match;
        });
        if (it == debug_nodes.end()) {
            return nullptr;
        }
        return &(*it);
    };

    auto collect_debug_nodes = [&](const std::string& target_name, int target_layer) {
        std::vector<const Model::DebugNode*> matches;
        for (const auto& node : debug_nodes) {
            bool name_match = node.name == target_name;
            bool layer_match = target_layer < 0 || static_cast<int>(node.layer_idx) == target_layer;
            if (name_match && layer_match) {
                matches.push_back(&node);
            }
        }
        return matches;
    };
    
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
    
    if (!dump_patch_embeds_path.empty()) {
        auto it = std::find_if(debug_nodes.begin(), debug_nodes.end(), [](const Model::DebugNode& node) {
            return node.name == "vision_added_patch_embeds";
        });

        if (it != debug_nodes.end()) {
            auto data = extract_node_data(gb, it->node_id);
            const auto& buf = gb->get_output_buffer(it->node_id);
            bool ok = write_tensor_binary(dump_patch_embeds_path, data, buf.shape);
            if (ok) {
                std::cout << "Wrote patch embeddings to: " << dump_patch_embeds_path << std::endl;
            } else {
                std::cerr << "Failed to write patch embeddings to: " << dump_patch_embeds_path << std::endl;
            }
        } else {
            std::cerr << "Patch embeddings node not found in debug nodes." << std::endl;
        }
    }

    for (const auto& request : node_dump_requests) {
        auto it = std::find_if(debug_nodes.begin(), debug_nodes.end(), [&](const Model::DebugNode& node) {
            bool name_match = node.name == request.name;
            bool layer_match = (request.layer_idx < 0) || (static_cast<int>(node.layer_idx) == request.layer_idx);
            return name_match && layer_match;
        });

        if (it == debug_nodes.end()) {
            std::cerr << "Debug node not found for dump: name='" << request.name << "'";
            if (request.layer_idx >= 0) {
                std::cerr << " layer=" << request.layer_idx;
            }
            std::cerr << std::endl;
            continue;
        }

        auto data = extract_node_data(gb, it->node_id);
        const auto& buf = gb->get_output_buffer(it->node_id);
        bool ok = write_tensor_binary(request.path, data, buf.shape);
        if (ok) {
            std::cout << "Wrote node dump ('" << request.name << "') to: " << request.path << std::endl;
        } else {
            std::cerr << "Failed to write node dump to: " << request.path << std::endl;
        }
    }

    if (!dump_layer0_suite_dir.empty()) {
        std::filesystem::path suite_path(dump_layer0_suite_dir);
        std::error_code ec;
        std::filesystem::create_directories(suite_path, ec);
        if (ec) {
            std::cerr << "Failed to create dump directory '" << dump_layer0_suite_dir << "': "
                      << ec.message() << std::endl;
        } else {
            struct LayerDumpSpec {
                const char* debug_name;
                int layer_idx;
                const char* filename;
            };

            const LayerDumpSpec specs[] = {
                {"vision_all_pos_embeddings", 0, "layer0_position_embedding.bin"},
                {"vision_attn_k", 0, "layer0_self_attn_k_proj.bin"},
                {"vision_attn_v", 0, "layer0_self_attn_v_proj.bin"},
                {"vision_attn_output", 0, "layer0_self_attn_out_proj.bin"},
                {"vision_mlp_fc2", 0, "layer0_mlp.bin"},
            };

            for (const auto& spec : specs) {
                auto nodes = collect_debug_nodes(spec.debug_name, spec.layer_idx);
                if (nodes.empty()) {
                    std::cerr << "Layer0 suite: debug node '" << spec.debug_name << "' (layer="
                              << spec.layer_idx << ") not found" << std::endl;
                    continue;
                }

                std::vector<float> combined_data;
                std::vector<size_t> combined_shape;
                bool shape_error = false;

                for (const auto* node : nodes) {
                    auto data = extract_node_data(gb, node->node_id);
                    const auto& buf = gb->get_output_buffer(node->node_id);

                    if (data.empty()) {
                        continue;
                    }

                    if (combined_shape.empty()) {
                        combined_shape = buf.shape;
                        if (!combined_shape.empty()) {
                            combined_shape[0] = 0;
                        }
                    } else {
                        if (combined_shape.size() != buf.shape.size()) {
                            shape_error = true;
                            break;
                        }
                        for (size_t dim = 1; dim < buf.shape.size(); ++dim) {
                            if (combined_shape[dim] != buf.shape[dim]) {
                                shape_error = true;
                                break;
                            }
                        }
                        if (shape_error) {
                            break;
                        }
                    }

                    combined_data.insert(combined_data.end(), data.begin(), data.end());
                    if (!combined_shape.empty()) {
                        combined_shape[0] += buf.shape.empty() ? 0 : buf.shape[0];
                    }
                }

                if (shape_error || combined_data.empty()) {
                    std::cerr << "Layer0 suite: incompatible shapes for debug node '" << spec.debug_name
                              << "' when combining tile outputs" << std::endl;
                    continue;
                }

                std::filesystem::path out_path = suite_path / spec.filename;
                bool ok = write_tensor_binary(out_path.string(), combined_data,
                                              combined_shape.empty() ? std::vector<size_t>{combined_data.size()} : combined_shape);
                if (ok) {
                    std::cout << "Wrote layer0 suite dump ('" << spec.debug_name << "') to: "
                              << out_path << std::endl;
                } else {
                    std::cerr << "Failed to write layer0 suite dump to: " << out_path << std::endl;
                }
            }
        }

        if (!final_features.empty()) {
            std::filesystem::path final_path = suite_path / "final_hidden_states.bin";
            bool ok = write_tensor_binary(final_path.string(), final_features, output_buf.shape);
            if (ok) {
                std::cout << "Wrote final hidden states to: " << final_path << std::endl;
            } else {
                std::cerr << "Failed to write final hidden states to: " << final_path << std::endl;
            }
        } else {
            std::cerr << "Final features buffer was empty; skipping final hidden state dump" << std::endl;
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


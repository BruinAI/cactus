#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <unordered_map>
#include <vector>

#include "../cactus/engine/engine.h"
#define private public
#define protected public
#include "../cactus/models/model.h"
#undef private
#undef protected
#include "../cactus/graph/graph.h"

using namespace cactus::engine;

namespace {

struct NamedDebugNode {
    std::string origin;
    uint32_t layer_idx;
    std::string name;
    size_t node_id;
};

std::vector<float> extract_node_data(CactusGraph* gb, size_t node_id) {
    try {
        const auto& buf = gb->get_output_buffer(node_id);
        void* output_ptr = gb->get_output(node_id);
        if (!output_ptr) {
            return {};
        }

        size_t total_size = 1;
        for (auto dim : buf.shape) {
            total_size *= dim;
        }

        if (total_size == 0) {
            return {};
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
                data[i] = static_cast<float>(ptr[i]) * scale;
            }
        } else {
            return {};
        }

        return data;
    } catch (const std::out_of_range&) {
        return {};
    } catch (const std::exception&) {
        return {};
    }
}

void print_tensor_summary(CactusGraph* gb, size_t node_id, const std::string& name) {
    const BufferDesc* buf_ptr = nullptr;
    try {
        buf_ptr = &gb->get_output_buffer(node_id);
    } catch (const std::out_of_range&) {
        std::cout << name << ": NOT AVAILABLE" << std::endl;
        return;
    } catch (const std::exception& ex) {
        std::cout << name << ": ERROR - " << ex.what() << std::endl;
        return;
    }

    auto data = extract_node_data(gb, node_id);
    if (data.empty()) {
        std::cout << name << ": NO DATA" << std::endl;
        return;
    }

    const auto& buf = *buf_ptr;
    std::cout << name << " shape=[";
    for (size_t i = 0; i < buf.shape.size(); ++i) {
        std::cout << buf.shape[i];
        if (i + 1 < buf.shape.size()) {
            std::cout << ",";
        }
    }
    std::cout << "]" << std::endl;

    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    double sum = 0.0;
    double sum_sq = 0.0;
    for (float v : data) {
        min_val = std::min(min_val, v);
        max_val = std::max(max_val, v);
        sum += v;
        sum_sq += static_cast<double>(v) * static_cast<double>(v);
    }

    double mean = sum / static_cast<double>(data.size());
    double variance = std::max(0.0, (sum_sq / static_cast<double>(data.size())) - mean * mean);
    double stddev = std::sqrt(variance);

    std::cout << "  Min=" << std::setprecision(6) << min_val
              << " Max=" << max_val
              << " Mean=" << mean
              << " Std=" << stddev << std::endl;

    size_t preview_count = std::min<size_t>(16, data.size());
    std::cout << "  First " << preview_count << " values:";
    for (size_t i = 0; i < preview_count; ++i) {
        if (i % 8 == 0) {
            std::cout << "\n    ";
        }
        std::cout << std::setw(10) << std::setprecision(5) << data[i];
    }
    if (data.size() > preview_count) {
        std::cout << " ...";
    }
    std::cout << std::endl;
}

size_t infer_token_dim(const std::vector<size_t>& shape);

bool has_multiple_tokens(CactusGraph* gb, size_t node_id) {
    try {
        const auto& buf = gb->get_output_buffer(node_id);
        if (buf.shape.empty()) {
            return false;
        }

        size_t token_dim = infer_token_dim(buf.shape);
        if (token_dim >= buf.shape.size()) {
            return false;
        }

        return buf.shape[token_dim] > 1;
    } catch (const std::exception&) {
        return false;
    }
}

size_t infer_token_dim(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        return 0;
    }

    if (shape.size() >= 2 && shape[0] == 1 && shape[1] > 1) {
        return 1;
    }

    return 0;
}

std::string shape_to_string(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        return "[]";
    }

    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        oss << shape[i];
        if (i + 1 < shape.size()) {
            oss << ",";
        }
    }
    oss << "]";
    return oss.str();
}

std::vector<float> extract_final_token_slice(const std::vector<float>& data, const std::vector<size_t>& shape) {
    if (data.empty()) {
        return {};
    }

    if (shape.empty()) {
        return data;
    }

    size_t expected_size = 1;
    for (size_t dim : shape) {
        expected_size *= dim;
    }

    if (expected_size != data.size()) {
        return data;
    }

    size_t token_dim = infer_token_dim(shape);
    size_t token_count = shape[token_dim];
    if (token_count == 0) {
        return {};
    }

    size_t block_size = 1;
    for (size_t i = token_dim + 1; i < shape.size(); ++i) {
        block_size *= shape[i];
    }

    if (block_size == 0) {
        block_size = 1;
    }

    size_t groups = 1;
    for (size_t i = 0; i < token_dim; ++i) {
        groups *= shape[i];
    }

    if (groups == 0) {
        groups = 1;
    }

    size_t group_stride = token_count * block_size;
    size_t offset = (groups - 1) * group_stride + (token_count - 1) * block_size;

    if (offset >= data.size()) {
        if (block_size >= data.size()) {
            return data;
        }
        offset = data.size() - block_size;
    }

    size_t length = block_size;
    if (offset + length > data.size()) {
        length = data.size() - offset;
    }

    std::vector<float> slice(length);
    std::copy_n(data.begin() + offset, length, slice.begin());
    return slice;
}

void print_final_token_output(CactusGraph* gb, size_t node_id, const std::string& name) {
    const BufferDesc* buf_ptr = nullptr;
    try {
        buf_ptr = &gb->get_output_buffer(node_id);
    } catch (const std::out_of_range&) {
        std::cout << name << ": NOT AVAILABLE" << std::endl;
        return;
    } catch (const std::exception& ex) {
        std::cout << name << ": ERROR - " << ex.what() << std::endl;
        return;
    }

    auto data = extract_node_data(gb, node_id);
    if (data.empty()) {
        std::cout << name << ": NO DATA" << std::endl;
        return;
    }

    const auto& shape = buf_ptr->shape;

    auto final_token = extract_final_token_slice(data, shape);
    if (final_token.empty()) {
        std::cout << name << " shape=" << shape_to_string(shape)
                  << "\n  Final token: NO DATA" << std::endl;
        return;
    }

    size_t token_dim = infer_token_dim(shape);
    size_t token_count = shape.empty() ? 1 : shape[token_dim];

    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    double sum = 0.0;
    double sum_sq = 0.0;
    for (float v : final_token) {
        min_val = std::min(min_val, v);
        max_val = std::max(max_val, v);
        sum += v;
        sum_sq += static_cast<double>(v) * static_cast<double>(v);
    }

    double mean = sum / static_cast<double>(final_token.size());
    double variance = std::max(0.0, (sum_sq / static_cast<double>(final_token.size())) - mean * mean);
    double stddev = std::sqrt(variance);

    std::cout << name << " shape=" << shape_to_string(shape) << std::endl;
    std::cout << "  Tokens=" << token_count
              << " FinalTokenLength=" << final_token.size() << std::endl;
    std::cout << "  FinalToken Min=" << std::setprecision(6) << min_val
              << " Max=" << max_val
              << " Mean=" << mean
              << " Std=" << stddev << std::endl;

    size_t preview_count = std::min<size_t>(16, final_token.size());
    std::cout << "  FinalToken Preview (first " << preview_count << "):";
    for (size_t i = 0; i < preview_count; ++i) {
        if (i % 8 == 0) {
            std::cout << "\n    ";
        }
        std::cout << std::setw(10) << std::setprecision(5) << final_token[i];
    }
    if (final_token.size() > preview_count) {
        std::cout << " ...";
    }
    std::cout << std::endl;
}

void dump_debug_node(CactusGraph* gb, const NamedDebugNode& node) {
    std::ostringstream label;
    if (!node.origin.empty()) {
        label << node.origin << " ";
    }
    label << "Layer " << node.layer_idx << " - " << node.name;

    std::string base_label = label.str();
    print_tensor_summary(gb, node.node_id, base_label);
    if (has_multiple_tokens(gb, node.node_id)) {
        print_final_token_output(gb, node.node_id, base_label + " (final token view)");
    }
}

std::vector<NamedDebugNode> collect_debug_nodes(const Lfm2VlModel& model) {
    std::vector<NamedDebugNode> merged;
    std::set<std::tuple<std::string, uint32_t, std::string>> seen;

    auto append_nodes = [&](const std::string& origin, const std::vector<Model::DebugNode>& nodes) {
        for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
            NamedDebugNode entry{origin, it->layer_idx, it->name, it->node_id};
            if (!seen.insert({entry.origin, entry.layer_idx, entry.name}).second) {
                continue;
            }
            merged.push_back(std::move(entry));
        }
    };

    append_nodes("VLM", model.get_debug_nodes());
    append_nodes("Language", model.language_model_.get_debug_nodes());
    append_nodes("Vision", model.vision_tower_.get_debug_nodes());

    std::sort(merged.begin(), merged.end(), [](const NamedDebugNode& a, const NamedDebugNode& b) {
        if (a.origin != b.origin) {
            return a.origin < b.origin;
        }
        if (a.layer_idx != b.layer_idx) {
            return a.layer_idx < b.layer_idx;
        }
        return a.name < b.name;
    });

    return merged;
}

std::string sanitize_token_text(const std::string& text) {
    std::string sanitized;
    sanitized.reserve(text.size());
    for (char c : text) {
        if (c == '\n') {
            sanitized += "\\n";
        } else if (c == '\r') {
            sanitized += "\\r";
        } else {
            sanitized += c;
        }
    }
    return sanitized;
}

std::filesystem::path resolve_image_path(
    const std::vector<std::filesystem::path>& candidates,
    const std::vector<std::filesystem::path>& search_roots) {
    std::error_code ec;
    for (const auto& candidate : candidates) {
        if (candidate.empty()) {
            continue;
        }

        // Try as provided (relative to CWD)
        if (std::filesystem::exists(candidate, ec)) {
            return std::filesystem::weakly_canonical(candidate, ec);
        }

        // Try relative to each provided root
        for (const auto& root : search_roots) {
            if (root.empty()) {
                continue;
            }

            std::filesystem::path attempt = candidate;
            if (attempt.is_relative()) {
                attempt = root / attempt;
            }

            if (std::filesystem::exists(attempt, ec)) {
                return std::filesystem::weakly_canonical(attempt, ec);
            }
        }
    }

    return {};
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <model_folder> [image_path] [prompt]\n";
    std::cout << "  model_folder : Path to LFM2-VL weights (e.g. ../../weights/lfm2-vl-350m-fp16)\n";
    std::cout << "  image_path   : Optional path to image (defaults to tests/istockphoto-184978580-2048x2048.jpg)\n";
    std::cout << "  prompt       : Optional prompt string (defaults to \"Describe this image\")\n";
    std::cout << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::filesystem::path model_folder = std::filesystem::absolute(argv[1]);
    std::error_code ec;
    if (!std::filesystem::exists(model_folder, ec)) {
        std::cerr << "Model folder does not exist: " << model_folder << std::endl;
        return 1;
    }

    const std::string default_prompt = "Describe this image.";

    std::filesystem::path executable_path;
    try {
        executable_path = std::filesystem::canonical(argv[0], ec).parent_path();
    } catch (...) {
        executable_path = std::filesystem::current_path();
    }

    std::vector<std::filesystem::path> search_roots = {
        std::filesystem::current_path(),
        executable_path,
    };

    if (!executable_path.empty()) {
        auto parent = executable_path.parent_path();
        if (!parent.empty() && parent != executable_path) {
            search_roots.push_back(parent);
            auto grandparent = parent.parent_path();
            if (!grandparent.empty() && grandparent != parent) {
                search_roots.push_back(grandparent);
            }
        }
    }

    std::vector<std::filesystem::path> candidate_images;
    if (argc >= 3) {
        candidate_images.emplace_back(argv[2]);
    }

    candidate_images.emplace_back("tests/istockphoto-184978580-2048x2048.jpg");
    candidate_images.emplace_back("../tests/istockphoto-184978580-2048x2048.jpg");
    candidate_images.emplace_back("../../tests/istockphoto-184978580-2048x2048.jpg");
    candidate_images.emplace_back("istockphoto-184978580-2048x2048.jpg");

    std::filesystem::path image_path = resolve_image_path(candidate_images, search_roots);
    if (image_path.empty()) {
        std::cerr << "Failed to locate test image. Provide path explicitly as the second argument." << std::endl;
        return 1;
    }

    std::string prompt = default_prompt;
    if (argc >= 4) {
        prompt = argv[3];
    }

    std::cout << "=== LFM2-VL Model Integration Test ===" << std::endl;
    std::cout << "Model folder : " << model_folder << std::endl;
    std::cout << "Image path   : " << image_path << std::endl;
    std::cout << "Prompt       : " << prompt << std::endl << std::endl;

    Config config;
    std::filesystem::path config_path = model_folder / "config.txt";
    if (!config.from_json(config_path.string())) {
        std::cerr << "Failed to load config from " << config_path << std::endl;
        return 1;
    }

    Lfm2VlModel model(config);
    const size_t context_size = 4096;

    std::cout << "Initializing LFM2-VL model..." << std::endl;
    if (!model.init(model_folder.string(), context_size)) {
        std::cerr << "Model initialization failed." << std::endl;
        return 1;
    }
    std::cout << "Model initialized successfully." << std::endl << std::endl;

    Tokenizer* tokenizer = model.get_tokenizer();
    if (!tokenizer) {
        std::cerr << "Tokenizer not available after initialization." << std::endl;
        return 1;
    }

    std::vector<ChatMessage> messages;
    messages.push_back({"user", image_path.string(), "image"});
    messages.push_back({"user", prompt, "text"});

    std::cout << "Formatting chat prompt using tokenizer template..." << std::endl;
    auto formatted_prompt = tokenizer->format_chat_prompt(messages, true);
    std::cout << "Formatted prompt:\n" << formatted_prompt << std::endl << std::endl;

    auto tokens = tokenizer->apply_chat_template(messages, true);
    if (tokens.empty()) {
        std::cerr << "Tokenization failed: no tokens produced." << std::endl;
        return 1;
    }

    std::cout << "Token count: " << tokens.size() << std::endl;
    std::cout << "Tokens: ";
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::cout << tokens[i];
        if (i + 1 < tokens.size()) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl << std::endl;

    std::vector<std::string> image_paths = {image_path.string()};

    std::vector<uint32_t> generation_context = tokens;
    std::vector<uint32_t> generated_tokens;
    generated_tokens.reserve(64);

    std::string decoded_base = tokenizer->decode(generation_context);
    std::string decoded_so_far = decoded_base;

    const size_t max_new_tokens = 64;
    const float temperature = -1.0f;  // Use model defaults
    const float top_p = -1.0f;
    const size_t top_k = 0;

    std::cout << "Starting multimodal generation (up to " << max_new_tokens << " tokens)..." << std::endl;

    for (size_t step = 0; step < max_new_tokens; ++step) {
        uint32_t next_token = model.generate_with_images(generation_context, image_paths, temperature, top_p, top_k);
        generation_context.push_back(next_token);
        generated_tokens.push_back(next_token);

        std::string decoded_full = tokenizer->decode(generation_context);
        std::string diff;
        if (decoded_full.size() >= decoded_so_far.size()) {
            diff = decoded_full.substr(decoded_so_far.size());
        }
        decoded_so_far = decoded_full;

        std::cout << "  [" << std::setw(2) << step + 1 << "] token_id=" << next_token;
        if (!diff.empty()) {
            std::cout << " text=\"" << sanitize_token_text(diff) << "\"";
        }
        std::cout << std::endl;

        if (next_token == config.eos_token_id) {
            std::cout << "Encountered EOS token, stopping generation." << std::endl;
            break;
        }
    }

    std::cout << std::endl;
    std::string final_output = tokenizer->decode(generation_context);
    std::cout << "Final decoded output (including prompt markup):\n" << final_output << std::endl;

    CactusGraph* gb = static_cast<CactusGraph*>(model.graph_handle_);
    std::cout << std::endl << "=== Debug: Final Token Outputs ===" << std::endl;
    if (gb) {
        auto debug_nodes = collect_debug_nodes(model);
        if (debug_nodes.empty()) {
            std::cout << "No debug nodes were captured." << std::endl;
        } else {
                for (const auto& node : debug_nodes) {
                    dump_debug_node(gb, node);
                }

                std::map<std::string, std::map<uint32_t, std::set<std::string>>> summary;
                for (const auto& node : debug_nodes) {
                    summary[node.origin][node.layer_idx].insert(node.name);
                }

                if (!summary.empty()) {
                    std::cout << std::endl << "Summary of captured debug nodes:" << std::endl;
                    for (const auto& origin_entry : summary) {
                        std::cout << "  [" << origin_entry.first << "]" << std::endl;
                        for (const auto& layer_entry : origin_entry.second) {
                            const auto& names = layer_entry.second;
                            std::cout << "    Layer " << layer_entry.first << " -> " << names.size() << " nodes";
                            size_t preview_count = 0;
                            for (const auto& name : names) {
                                if (preview_count == 0) {
                                    std::cout << " (";
                                } else if (preview_count >= 5) {
                                    std::cout << ", ...";
                                    break;
                                } else {
                                    std::cout << ", ";
                                }
                                std::cout << name;
                                ++preview_count;
                            }
                            if (!names.empty()) {
                                std::cout << ")";
                            }
                            std::cout << std::endl;
                        }
                    }
                }
        }
    } else {
        std::cout << "Graph handle unavailable; unable to dump final token outputs." << std::endl;
    }

    std::cout << std::endl << "Test completed successfully!" << std::endl;
    return 0;
}

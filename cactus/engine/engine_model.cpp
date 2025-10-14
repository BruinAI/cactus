#include "engine.h"
#include "../models/model.h"
#include "../graph/graph.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <dirent.h>
#include <algorithm>
#include <set>
#include <sstream>
#include <filesystem>
#include <limits>
#include <cctype>
#include <system_error>

namespace cactus {
namespace engine {


Model::Model()
    : tokenizer_(nullptr),
      graph_handle_(nullptr),
      initialized_(false),
      attention_scale_(0.0f),
      output_weight_node_id_(0) {
        initialize_debug_options_from_env();
}

Model::Model(const Config& config)
    : config_(config),
      tokenizer_(nullptr),
      graph_handle_(nullptr),
      initialized_(false),
      attention_scale_(0.0f),
      output_weight_node_id_(0) {
        initialize_debug_options_from_env();
}

Model::~Model() {
    if (graph_handle_) {
        delete static_cast<CactusGraph*>(graph_handle_);
    }
}

bool Model::init(const std::string& model_folder, size_t context_size, const std::string& system_prompt) {
    if (initialized_) {
        return true;
    }
    
    model_folder_path_ = model_folder;
    std::string config_path = model_folder + "/config.txt";
    
    if (!config_.from_json(config_path)) {
        return false;
    }
    
    std::string vocab_file = model_folder + "/vocab.txt";
    std::string merges_file = model_folder + "/merges.txt";
    std::string tokenizer_config_file = model_folder + "/tokenizer_config.txt";
    
    std::ifstream merges_check(merges_file);
    bool has_merges = false;
    if (merges_check.is_open()) {
        std::string line;
        int line_count = 0;
        while (std::getline(merges_check, line) && line_count < 10) {
            if (!line.empty() && line[0] != '#') {
                has_merges = true;
                break;
            }
            line_count++;
        }
        merges_check.close();
    }
    
    if (has_merges) {
        tokenizer_ = std::make_unique<BPETokenizer>();
    } else {
        tokenizer_ = std::make_unique<SPTokenizer>();
    }
    
    if (!tokenizer_->load_vocabulary_with_config(vocab_file, merges_file, tokenizer_config_file)) {
        return false;
    }
    
    auto* gb = new CactusGraph();
    graph_handle_ = gb;
    
    embedding_file_path_ = model_folder + "/token_embeddings.weights";

    load_weights_to_graph(gb);
    
    if (config_.model_type == Config::ModelType::GEMMA) {
        attention_scale_ = 1.0f / std::sqrt(256.0f); 
    } else {
        attention_scale_ = 1.0f / std::sqrt(static_cast<float>(config_.attention_head_dim));
    }
    
    Precision cache_precision;
    std::string precision_name;
    switch (config_.precision) {
        case Config::Precision::INT8:
            cache_precision = Precision::INT8;
            precision_name = "INT8";
            break;
        case Config::Precision::FP16:
            cache_precision = Precision::FP16;
            precision_name = "FP16";
            break;
        case Config::Precision::FP32:
            cache_precision = Precision::FP32;
            precision_name = "FP32";
            break;
    }
    kv_cache_.init(config_.num_layers, context_size, config_.attention_kv_heads, config_.attention_head_dim, cache_precision);
    
    size_t window_size = std::min(context_size, size_t(1024));
    size_t sink_size = 4;
    const char* env_window = std::getenv("CACTUS_KV_WINDOW_SIZE");
    const char* env_sink = std::getenv("CACTUS_KV_SINK_SIZE");
    if (env_window) {
        window_size = std::stoul(env_window);
    }
    if (env_sink) {
        sink_size = std::stoul(env_sink);
    }
    kv_cache_.set_window_size(window_size, sink_size);
    cache_k_output_nodes_.resize(config_.num_layers);
    cache_v_output_nodes_.resize(config_.num_layers);
    
    initialized_ = true;
    
    std::string warmup_text = system_prompt.empty() ? "Henry" : system_prompt;
    auto warmup_tokens = tokenizer_->encode(warmup_text);
    suspend_debug_capture();
    try {
        forward(warmup_tokens);
    } catch (...) {
        resume_debug_capture();
        throw;
    }
    debug_captures_.clear();
    resume_debug_capture();
    kv_cache_.reset();
    return true;
}


uint32_t Model::generate(const std::vector<uint32_t>& tokens, float temperature, float top_p,
                        size_t top_k, const std::string& profile_file) {
                            
    if (temperature < 0) {
        temperature = config_.default_temperature;
    }
    if (top_p < 0) {
        top_p = config_.default_top_p;
    }
    if (top_k == 0) {
        top_k = config_.default_top_k;
    }

    auto final_hidden = forward(tokens, true);

    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    auto backend = config_.default_backend == Config::Backend::CPU
        ? ComputeBackend::CPU
        : ComputeBackend::NPU;

    auto logits_node_id = gb->matmul(final_hidden, output_weight_node_id_, true, backend);
    auto sampled_token_id = gb->sample(logits_node_id, temperature, top_p, top_k);
    
    if (!profile_file.empty()) {
        gb->execute(profile_file);
    } else {
        gb->execute();
    }

    post_execute_updates(gb, tokens.size());
    flush_debug_nodes(gb);
    update_kv_cache(gb, tokens.size());
    
    auto* output_ptr = gb->get_output(sampled_token_id);
    return *static_cast<uint32_t*>(output_ptr);
}

void Model::update_kv_cache(CactusGraph* gb, size_t seq_len) {
    kv_cache_.update_from_graph(gb, cache_k_output_nodes_, cache_v_output_nodes_, 
                               seq_len, config_.num_layers, config_.attention_kv_heads, 
                               config_.attention_head_dim);
}


std::vector<float> Model::get_embeddings(const std::vector<uint32_t>& tokens, bool pooled) {
    auto final_hidden = forward(tokens);
    
    auto* gb = static_cast<CactusGraph*>(graph_handle_);
    auto* output_ptr = gb->get_output(final_hidden);
    const auto& output_buffer = gb->get_output_buffer(final_hidden);
    
    std::vector<float> embeddings;
    
    if (pooled) {
        auto pooled_hidden = gb->mean(final_hidden, 0);
        gb->execute();
        post_execute_updates(gb, tokens.size());
        flush_debug_nodes(gb);
        
        auto* pooled_ptr = gb->get_output(pooled_hidden);
        const auto& pooled_buffer = gb->get_output_buffer(pooled_hidden);
        
        size_t hidden_dim = pooled_buffer.total_size;
        embeddings.resize(hidden_dim);
        
        if (pooled_buffer.precision == Precision::FP32) {
            float* pooled_data = static_cast<float*>(pooled_ptr);
            std::copy(pooled_data, pooled_data + hidden_dim, embeddings.begin());
        } else if (pooled_buffer.precision == Precision::FP16) {
            __fp16* pooled_data = static_cast<__fp16*>(pooled_ptr);
            Quantization::fp16_to_fp32(pooled_data, embeddings.data(), hidden_dim);
        } else if (pooled_buffer.precision == Precision::INT8) {
            int8_t* pooled_data = static_cast<int8_t*>(pooled_ptr);
            float scale = pooled_buffer.quantization_scale;
            Quantization::int8_to_fp32(pooled_data, embeddings.data(), hidden_dim, scale);
        }
    } else {
        gb->execute();
        post_execute_updates(gb, tokens.size());
        flush_debug_nodes(gb);
        
        size_t total_size = output_buffer.total_size;
        embeddings.resize(total_size);
        
        if (output_buffer.precision == Precision::FP32) {
            float* hidden_states = static_cast<float*>(output_ptr);
            std::copy(hidden_states, hidden_states + total_size, embeddings.begin());
        } else if (output_buffer.precision == Precision::FP16) {
            __fp16* hidden_states = static_cast<__fp16*>(output_ptr);
            for (size_t i = 0; i < total_size; i++) {
                embeddings[i] = static_cast<float>(hidden_states[i]);
            }
        } else if (output_buffer.precision == Precision::INT8) {
            int8_t* hidden_states = static_cast<int8_t*>(output_ptr);
            float scale = output_buffer.quantization_scale;
            for (size_t i = 0; i < total_size; i++) {
                embeddings[i] = hidden_states[i] * scale;
            }
        }
    }
    
    kv_cache_.reset();
    
    return embeddings;
}

std::vector<float> Model::debug_forward(const std::vector<uint32_t>& tokens, bool use_cache,
                                        const std::string& profile_file) {
    if (!initialized_) {
        throw std::runtime_error("Model not initialized - call init() before debug_forward()");
    }

    auto final_hidden = forward(tokens, use_cache);
    auto* gb = static_cast<CactusGraph*>(graph_handle_);

    if (!profile_file.empty()) {
        gb->execute(profile_file);
    } else {
        gb->execute();
    }

    post_execute_updates(gb, tokens.size());
    flush_debug_nodes(gb);

    const auto& buffer = gb->get_output_buffer(final_hidden);
    void* data = gb->get_output(final_hidden);
    auto result = extract_tensor_as_fp32(buffer, data);

    if (use_cache) {
        update_kv_cache(gb, tokens.size());
    } else {
        kv_cache_.reset();
    }

    return result;
}

void Model::suspend_debug_capture() {
    debug_suspend_depth_++;
    debug_capture_suspended_ = true;
}

void Model::resume_debug_capture() {
    if (debug_suspend_depth_ > 0) {
        debug_suspend_depth_--;
    }
    if (debug_suspend_depth_ == 0) {
        debug_capture_suspended_ = false;
    }
}

void Model::initialize_debug_options_from_env() {
    debug_options_ = DebugOptions{};

    const char* enable_env = std::getenv("CACTUS_DEBUG_ENABLE");
    const char* stdout_env = std::getenv("CACTUS_DEBUG_STDOUT");
    const char* dir_env = std::getenv("CACTUS_DEBUG_DIR");
    const char* layers_env = std::getenv("CACTUS_DEBUG_LAYERS");
    const char* max_env = std::getenv("CACTUS_DEBUG_MAX_PRINT");

    auto is_truthy = [](const char* value) {
        if (!value) {
            return false;
        }
        std::string s(value);
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        return !(s.empty() || s == "0" || s == "false" || s == "no" || s == "off");
    };

    if (is_truthy(stdout_env)) {
        debug_options_.dump_stdout = true;
    }

    if (dir_env && *dir_env) {
        debug_options_.dump_to_files = true;
        debug_options_.dump_dir = dir_env;
    }

    if (max_env && *max_env) {
        try {
            debug_options_.max_print_values = std::max<size_t>(1, std::stoul(std::string(max_env)));
        } catch (...) {
            // ignore invalid values
        }
    }

    bool explicitly_enabled = is_truthy(enable_env);
    debug_options_.enabled = explicitly_enabled || debug_options_.dump_stdout || debug_options_.dump_to_files;

    debug_options_.include_all_layers = true;
    if (layers_env && *layers_env) {
        std::string layers_spec = layers_env;
        if (!(layers_spec == "*" || layers_spec == "all" || layers_spec == "ALL")) {
            debug_options_.include_all_layers = false;
            std::stringstream ss(layers_spec);
            std::string token;
            while (std::getline(ss, token, ',')) {
                if (token.empty()) continue;
                token.erase(0, token.find_first_not_of(" \t"));
                token.erase(token.find_last_not_of(" \t") + 1);
                if (token.empty()) continue;
                try {
                    debug_options_.layer_filter.insert(static_cast<uint32_t>(std::stoul(token)));
                } catch (...) {
                    // ignore invalid entries
                }
            }
            if (debug_options_.layer_filter.empty()) {
                debug_options_.include_all_layers = true;
            }
        }
    }

    if (debug_options_.dump_to_files && !debug_options_.dump_dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(debug_options_.dump_dir, ec);
        if (ec) {
            std::cerr << "[Cactus] Failed to create debug dump directory '" << debug_options_.dump_dir
                      << "': " << ec.message() << std::endl;
            debug_options_.dump_to_files = false;
        }
    }
}

bool Model::should_capture_layer(uint32_t layer_idx) const {
    if (!debug_options_.enabled || debug_capture_suspended_) {
        return false;
    }
    if (layer_idx == std::numeric_limits<uint32_t>::max()) {
        return true;
    }
    if (debug_options_.include_all_layers) {
        return true;
    }
    return debug_options_.layer_filter.find(layer_idx) != debug_options_.layer_filter.end();
}

void Model::capture_debug_node(uint32_t layer_idx, const std::string& tag, size_t node_id) const {
    if (!should_capture_layer(layer_idx)) {
        return;
    }
    debug_captures_.push_back({layer_idx, tag, node_id});
}

std::string Model::debug_layer_label(uint32_t layer_idx) const {
    if (layer_idx == std::numeric_limits<uint32_t>::max()) {
        return "global";
    }
    std::ostringstream ss;
    ss << "layer_" << layer_idx;
    return ss.str();
}

std::vector<float> Model::extract_tensor_as_fp32(const BufferDesc& buffer, const void* data) const {
    size_t count = buffer.total_size;
    std::vector<float> result(count, 0.0f);

    switch (buffer.precision) {
        case Precision::FP32: {
            const float* ptr = static_cast<const float*>(data);
            std::copy(ptr, ptr + count, result.begin());
            break;
        }
        case Precision::FP16: {
            const __fp16* ptr = static_cast<const __fp16*>(data);
            Quantization::fp16_to_fp32(ptr, result.data(), count);
            break;
        }
        case Precision::INT8: {
            const int8_t* ptr = static_cast<const int8_t*>(data);
            Quantization::int8_to_fp32(ptr, result.data(), count, buffer.quantization_scale);
            break;
        }
    }

    return result;
}

void Model::flush_debug_nodes(CactusGraph* gb) {
    if (!debug_options_.enabled) {
        debug_captures_.clear();
        return;
    }

    if (debug_captures_.empty()) {
        return;
    }

    size_t flush_index = debug_flush_counter_++;

    for (const auto& capture : debug_captures_) {
        void* raw_data = gb->get_output(capture.node_id);
        if (!raw_data) {
            continue;
        }

        const auto& buffer = gb->get_output_buffer(capture.node_id);
        auto values = extract_tensor_as_fp32(buffer, raw_data);

        std::ostringstream shape_ss;
        shape_ss << "[";
        for (size_t i = 0; i < buffer.shape.size(); ++i) {
            shape_ss << buffer.shape[i];
            if (i + 1 < buffer.shape.size()) {
                shape_ss << ", ";
            }
        }
        shape_ss << "]";

        if (debug_options_.dump_stdout) {
            size_t preview_count = std::min(debug_options_.max_print_values, values.size());
            std::cout << "[Cactus][" << debug_layer_label(capture.layer_idx) << "] "
                      << capture.tag << " shape=" << shape_ss.str() << " showing "
                      << preview_count << "/" << values.size() << " values: ";
            for (size_t i = 0; i < preview_count; ++i) {
                std::cout << values[i];
                if (i + 1 < preview_count) {
                    std::cout << ", ";
                }
            }
            if (preview_count < values.size()) {
                std::cout << " ...";
            }
            std::cout << std::endl;
        }

        if (debug_options_.dump_to_files) {
            std::string layer_label = debug_layer_label(capture.layer_idx);
            std::string sanitized_tag = capture.tag;
            for (char& ch : sanitized_tag) {
                if (!std::isalnum(static_cast<unsigned char>(ch))) {
                    ch = '_';
                }
            }

            std::ostringstream filename;
            filename << debug_options_.dump_dir << "/" << layer_label << "_" << sanitized_tag
                     << "_iter" << std::setw(4) << std::setfill('0') << flush_index << ".txt";

            std::ofstream ofs(filename.str());
            if (!ofs) {
                std::cerr << "[Cactus] Failed to open debug dump file: " << filename.str() << std::endl;
            } else {
                ofs << "# layer=" << layer_label << " tag=" << capture.tag
                    << " shape=" << shape_ss.str() << " precision=";
                switch (buffer.precision) {
                    case Precision::FP32: ofs << "FP32"; break;
                    case Precision::FP16: ofs << "FP16"; break;
                    case Precision::INT8: ofs << "INT8"; break;
                }
                if (buffer.precision == Precision::INT8) {
                    ofs << " scale=" << buffer.quantization_scale;
                }
                ofs << "\n";

                ofs << std::setprecision(10);
                for (size_t i = 0; i < values.size(); ++i) {
                    ofs << values[i] << '\n';
                }
            }
        }
    }

    debug_captures_.clear();
}


bool Config::from_json(const std::string& config_path) {
    std::ifstream file(config_path);
    if (!file) {
        return false;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;
        
        std::string key = line.substr(0, eq_pos);
        std::string value = line.substr(eq_pos + 1);
        
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);
        
        if (key == "vocab_size") vocab_size = std::stoul(value);
        else if (key == "bos_token_id") bos_token_id = std::stoul(value);
        else if (key == "eos_token_id") eos_token_id = std::stoul(value);
        else if (key == "num_layers") num_layers = std::stoul(value);
        else if (key == "hidden_dim") hidden_dim = std::stoul(value);
        else if (key == "ffn_intermediate_dim") ffn_intermediate_dim = std::stoul(value);
        else if (key == "attention_heads") attention_heads = std::stoul(value);
        else if (key == "attention_kv_heads") attention_kv_heads = std::stoul(value);
        else if (key == "attention_head_dim") attention_head_dim = std::stoul(value);
        else if (key == "layer_norm_eps") layer_norm_eps = std::stof(value);
        else if (key == "rope_theta") rope_theta = std::stof(value);
        else if (key == "tie_word_embeddings") tie_word_embeddings = (value == "true" || value == "1");
        else if (key == "precision") {
            if (value == "INT8") precision = Precision::INT8;
            else if (value == "FP16") precision = Precision::FP16;
            else precision = Precision::FP32;
        }
        else if (key == "model_type") {
            if (value == "gemma" || value == "GEMMA") model_type = ModelType::GEMMA;
            else if (value == "lfm2" || value == "LFM2") model_type = ModelType::LFM2;
            else model_type = ModelType::QWEN;
        }
        else if (key == "conv_L_cache") conv_L_cache = static_cast<size_t>(std::stoul(value));
        else if (key == "layer_types") {
            layer_types.clear();
            std::stringstream ss(value);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) {
                    item.erase(0, item.find_first_not_of(" \t"));
                    item.erase(item.find_last_not_of(" \t") + 1);
                    if (!item.empty()) layer_types.push_back(item);
                }
            }
        }
    }

    if (model_type == ModelType::GEMMA) {
        default_temperature = 1.0f;
        default_top_p = 0.95f;
        default_top_k = 64;
    } else if (model_type == ModelType::QWEN) {
        default_temperature = 0.6f;
        default_top_p = 0.95f;
        default_top_k = 20;
    }

    return true;
}

std::string Config::to_json() const {
    return "{}";
}

std::unique_ptr<Model> create_model(const std::string& model_folder) {
    Config config;
    std::string config_path = model_folder + "/config.txt";

    if (!config.from_json(config_path)) {
        return nullptr;
    }

    switch (config.model_type) {
        case Config::ModelType::QWEN:
            return std::make_unique<QwenModel>(config);
        case Config::ModelType::GEMMA:
            return std::make_unique<GemmaModel>(config);
        case Config::ModelType::LFM2:
            return std::make_unique<LFM2Model>(config);
        default:
            return std::make_unique<QwenModel>(config);
    }
}

}
}
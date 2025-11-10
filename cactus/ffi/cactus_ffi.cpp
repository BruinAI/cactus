#include "cactus_ffi.h"
#include "ffi_utils.h"
#include "../engine/engine.h"
#include <memory>
#include <string>
#include <thread>
#include <chrono>
#include <iostream>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <cmath>
#include <algorithm>

// STB image library - define implementation flags before including
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_BMP
#define STBI_NO_PSD
#define STBI_NO_HDR
#define STBI_NO_PIC
#define STBI_NO_PNM
#define STBI_NO_TGA
#include "stb_image_impl.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize_impl.h"

using namespace cactus::engine;
using namespace cactus::ffi;

// Import types from ffi_utils
using cactus::ffi::ToolFunction;

struct CactusModelHandle {
    std::unique_ptr<Model> model;
    std::atomic<bool> should_stop;
    std::vector<uint32_t> processed_tokens; 

    CactusModelHandle() : should_stop(false) {}
};

static std::vector<ChatMessage> parse_messages_json(const std::string& json, std::vector<std::string>& out_image_paths, std::vector<size_t>& out_image_placeholder_indices) {
    std::vector<ChatMessage> messages;
    out_image_paths.clear();
    out_image_placeholder_indices.clear();

    size_t pos = json.find('[');
    if (pos == std::string::npos) {
        throw std::runtime_error("Invalid JSON: expected array");
    }

    pos = json.find('{', pos);
    while (pos != std::string::npos) {
        ChatMessage msg;

        size_t role_pos = json.find("\"role\"", pos);
        if (role_pos == std::string::npos) break;

        size_t role_start = json.find('\"', role_pos + 6) + 1;
        size_t role_end = json.find('\"', role_start);
        msg.role = json.substr(role_start, role_end - role_start);

        size_t content_pos = json.find("\"content\"", role_end);
        if (content_pos == std::string::npos) break;

        size_t colon_pos = json.find(':', content_pos);
        if (colon_pos == std::string::npos) break;

        size_t after_colon = json.find_first_not_of(" \t\n\r", colon_pos + 1);
        if (after_colon == std::string::npos) break;

        if (json[after_colon] == '"') {
            size_t content_start = json.find('\"', content_pos + 9) + 1;
            size_t content_end = content_start;

            while (content_end < json.length()) {
                content_end = json.find('\"', content_end);
                if (content_end == std::string::npos) break;
                if (json[content_end - 1] != '\\') break;
                content_end++;
            }

            msg.content = json.substr(content_start, content_end - content_start);

            size_t escape_pos = 0;
            while ((escape_pos = msg.content.find("\\n", escape_pos)) != std::string::npos) {
                msg.content.replace(escape_pos, 2, "\n");
                escape_pos += 1;
            }
            escape_pos = 0;
            while ((escape_pos = msg.content.find("\\\"", escape_pos)) != std::string::npos) {
                msg.content.replace(escape_pos, 2, "\"");
                escape_pos += 1;
            }

            messages.push_back(msg);

            pos = json.find('{', content_pos);
            continue;
        }

        if (json[after_colon] == '[') {
            size_t arr_start = after_colon;
            int bracket_count = 1;
            size_t idx = arr_start + 1;

            while (idx < json.size() && bracket_count > 0) {
                if (json[idx] == '{') {
                    size_t obj_start = idx;
                    int obj_brace = 1;
                    size_t j = obj_start + 1;
                    while (j < json.size() && obj_brace > 0) {
                        if (json[j] == '{') obj_brace++;
                        else if (json[j] == '}') obj_brace--;
                        j++;
                    }
                    size_t obj_end = j;
                    std::string obj = json.substr(obj_start, obj_end - obj_start);

                    size_t type_pos = obj.find("\"type\"");
                    if (type_pos != std::string::npos) {
                        size_t type_colon = obj.find(':', type_pos);
                        size_t type_quote = obj.find('"', type_colon);
                        size_t type_quote_end = obj.find('"', type_quote + 1);
                        std::string type = obj.substr(type_quote + 1, type_quote_end - type_quote - 1);

                        if (type == "text") {
                            size_t text_pos = obj.find("\"text\"");
                            if (text_pos != std::string::npos) {
                                size_t tcol = obj.find(':', text_pos);
                                size_t tquote = obj.find('"', tcol);
                                size_t tquote_end = obj.find('"', tquote + 1);
                                std::string text = obj.substr(tquote + 1, tquote_end - tquote - 1);
                                
                                ChatMessage text_msg;
                                text_msg.role = msg.role;
                                text_msg.content = text;
                                text_msg.type = "text";
                                messages.push_back(text_msg);
                            }
                        } else if (type == "image") {
                            size_t path_pos = obj.find("\"path\"");
                            if (path_pos != std::string::npos) {
                                size_t pcol = obj.find(':', path_pos);
                                size_t pquote = obj.find('"', pcol);
                                size_t pquote_end = obj.find('"', pquote + 1);
                                std::string relpath = obj.substr(pquote + 1, pquote_end - pquote - 1);
                                
                                std::string abs_path;
                                try {
                                    std::filesystem::path p(relpath);
                                    abs_path = std::filesystem::absolute(p).string();
                                } catch (...) {
                                    abs_path = relpath;
                                }
                                
                                ChatMessage img_msg;
                                img_msg.role = msg.role;
                                img_msg.content = abs_path;
                                img_msg.type = "image";
                                messages.push_back(img_msg);
                                
                                out_image_paths.push_back(abs_path);
                            }
                        }
                    }

                    idx = obj_end;
                    continue;
                } else if (json[idx] == '[') {
                    bracket_count++;
                } else if (json[idx] == ']') {
                    bracket_count--;
                    if (bracket_count == 0) break;
                }
                idx++;
            }

            pos = json.find('{', idx);
            continue;
        }

        pos = json.find('{', after_colon);
    }

    return messages;
}

static bool matches_stop_sequence(const std::vector<uint32_t>& generated_tokens,
                                   const std::vector<std::vector<uint32_t>>& stop_sequences) {
    for (const auto& stop_seq : stop_sequences) {
        if (stop_seq.empty()) continue;

        if (generated_tokens.size() >= stop_seq.size()) {
            if (std::equal(stop_seq.rbegin(), stop_seq.rend(), generated_tokens.rbegin())) {
                return true;
            }
        }
    }
    return false;
}

extern "C" {

static std::string last_error_message;

const char* cactus_get_last_error() {
    return last_error_message.c_str();
}

cactus_model_t cactus_init(const char* model_path, size_t context_size) {
    try {
        auto* handle = new CactusModelHandle();
        handle->model = create_model(model_path);

        if (!handle->model) {
            last_error_message = "Failed to create model from: " + std::string(model_path);
            delete handle;
            return nullptr;
        }

        if (!handle->model->init(model_path, context_size)) {
            last_error_message = "Failed to initialize model from: " + std::string(model_path);
            delete handle;
            return nullptr;
        }
        
        return handle;
    } catch (const std::exception& e) {
        last_error_message = std::string(e.what());
        return nullptr;
    } catch (...) {
        last_error_message = "Unknown error during model initialization";
        return nullptr;
    }
}

int cactus_complete(
    cactus_model_t model,
    const char* messages_json,
    char* response_buffer,
    size_t buffer_size,
    const char* options_json,
    const char* tools_json,
    cactus_token_callback callback,
    void* user_data
) {
    if (!model) {
        std::string error_msg = last_error_message.empty() ? 
            "Model not initialized. Check model path and files." : last_error_message;
        handle_error_response(error_msg, response_buffer, buffer_size);
        return -1;
    }
    
    if (!messages_json || !response_buffer || buffer_size == 0) {
        handle_error_response("Invalid parameters", response_buffer, buffer_size);
        return -1;
    }
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        auto* handle = static_cast<CactusModelHandle*>(model);
        auto* tokenizer = handle->model->get_tokenizer();
        handle->should_stop = false;
        
        handle->model->reset_cache();
        
        std::vector<std::string> image_paths;
        std::vector<size_t> image_placeholder_indices;
        auto messages = parse_messages_json(messages_json, image_paths, image_placeholder_indices);
        
        if (messages.empty()) {
            handle_error_response("No messages provided", response_buffer, buffer_size);
            return -1;
        }
        
        float temperature, top_p;
        size_t top_k, max_tokens;
        std::vector<std::string> stop_sequences;
        parse_options_json(options_json ? options_json : "", 
                          temperature, top_p, top_k, max_tokens, stop_sequences);
        
        std::vector<ToolFunction> tools;
        if (tools_json && strlen(tools_json) > 0) {
            tools = parse_tools_json(tools_json);
        }
        
        std::string formatted_tools = format_tools_for_prompt(tools);
        std::string full_prompt = tokenizer->format_chat_prompt(messages, true, formatted_tools);

        if (full_prompt.find("ERROR:") == 0) {
            handle_error_response(full_prompt.substr(6), response_buffer, buffer_size);
            return -1;
        }

        // Use incremental processing approach from main
        std::vector<uint32_t> current_prompt_tokens = tokenizer->encode(full_prompt);
        
        std::vector<uint32_t> tokens_to_process;
        bool is_prefix = (current_prompt_tokens.size() >= handle->processed_tokens.size()) &&
                         std::equal(handle->processed_tokens.begin(), handle->processed_tokens.end(), current_prompt_tokens.begin());

        if (handle->processed_tokens.empty() || !is_prefix) {
            handle->model->reset_cache();
            tokens_to_process = current_prompt_tokens;
        } else {
            tokens_to_process.assign(current_prompt_tokens.begin() + handle->processed_tokens.size(), current_prompt_tokens.end());
        }
        
        size_t prompt_tokens = tokens_to_process.size();

        std::vector<std::vector<uint32_t>> stop_token_sequences;
        stop_token_sequences.push_back({tokenizer->get_eos_token()});
        for (const auto& stop_seq : stop_sequences) {
            stop_token_sequences.push_back(tokenizer->encode(stop_seq));
        }

        std::vector<uint32_t> generated_tokens;
        double time_to_first_token = 0.0;

        // Generate first token - use image support from HEAD if images are present
        uint32_t next_token;
        if (tokens_to_process.empty()) {
            if (handle->processed_tokens.empty()) {
                 handle_error_response("Cannot generate from empty prompt", response_buffer, buffer_size);
                 return -1;
            }
            std::vector<uint32_t> last_token_vec = { handle->processed_tokens.back() };
            next_token = handle->model->generate(last_token_vec, temperature, top_p, top_k);
        } else {
            if (!image_paths.empty()) {
                next_token = handle->model->generate_with_images(tokens_to_process, image_paths, temperature, top_p, top_k, "profile.txt");
            } else {
                next_token = handle->model->generate(tokens_to_process, temperature, top_p, top_k, "profile.txt");
            }
        }
        
        handle->processed_tokens = current_prompt_tokens;

        auto token_end = std::chrono::high_resolution_clock::now();
        time_to_first_token = std::chrono::duration_cast<std::chrono::microseconds>(token_end - start_time).count() / 1000.0;

        generated_tokens.push_back(next_token);
        handle->processed_tokens.push_back(next_token);

        if (!matches_stop_sequence(generated_tokens, stop_token_sequences)) {
            if (callback) {
                std::string new_text = tokenizer->decode({next_token});
                callback(new_text.c_str(), next_token, user_data);
            }

            for (size_t i = 1; i < max_tokens; i++) {
                if (handle->should_stop) break;

                next_token = handle->model->generate({next_token}, temperature, top_p, top_k);
                generated_tokens.push_back(next_token);
                handle->processed_tokens.push_back(next_token);

                if (matches_stop_sequence(generated_tokens, stop_token_sequences)) break;

                if (callback) {
                    std::string new_text = tokenizer->decode({next_token});
                    callback(new_text.c_str(), next_token, user_data);
                }
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double total_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
        
        size_t completion_tokens = generated_tokens.size();
        double decode_time_ms = total_time_ms - time_to_first_token;
        double tokens_per_second = completion_tokens > 1 ? ((completion_tokens - 1) * 1000.0) / decode_time_ms : 0.0;
        
        std::string response_text = tokenizer->decode(generated_tokens);
        
        std::string regular_response;
        std::vector<std::string> function_calls;
        parse_function_calls_from_response(response_text, regular_response, function_calls);
        
        std::string result = construct_response_json(regular_response, function_calls, time_to_first_token,
                                                     total_time_ms, tokens_per_second, prompt_tokens,
                                                     completion_tokens);
        
        if (result.length() >= buffer_size) {
            handle_error_response("Response buffer too small", response_buffer, buffer_size);
            return -1;
        }
        
        std::strcpy(response_buffer, result.c_str());
        
        return static_cast<int>(result.length());
        
    } catch (const std::exception& e) {
        handle_error_response(e.what(), response_buffer, buffer_size);
        return -1;
    }
}

void cactus_destroy(cactus_model_t model) {
    if (model) {
        delete static_cast<CactusModelHandle*>(model);
    }
}

void cactus_reset(cactus_model_t model) {
    if (!model) return;
    
    auto* handle = static_cast<CactusModelHandle*>(model);
    handle->model->reset_cache();
    handle->processed_tokens.clear();
}

void cactus_stop(cactus_model_t model) {
    if (!model) return;
    auto* handle = static_cast<CactusModelHandle*>(model);
    handle->should_stop = true;
}

int cactus_embed(
    cactus_model_t model,
    const char* text,
    float* embeddings_buffer,
    size_t buffer_size,
    size_t* embedding_dim
) {
    if (!model) return -1;
    if (!text || !embeddings_buffer || buffer_size == 0) return -1;
    
    try {
        auto* handle = static_cast<CactusModelHandle*>(model);
        auto* tokenizer = handle->model->get_tokenizer();
        
        std::vector<uint32_t> tokens = tokenizer->encode(text);
        if (tokens.empty()) return -1;
        
        std::vector<float> embeddings = handle->model->get_embeddings(tokens, true);
        if (embeddings.size() * sizeof(float) > buffer_size) return -2; 
        
        std::memcpy(embeddings_buffer, embeddings.data(), embeddings.size() * sizeof(float));
        if (embedding_dim) {
            *embedding_dim = embeddings.size();
        }
        
        return static_cast<int>(embeddings.size());
        
    } catch (const std::exception& e) {
        last_error_message = std::string(e.what());
        return -1;
    } catch (...) {
        last_error_message = "Unknown error during embedding generation";
        return -1;
    }
}

}
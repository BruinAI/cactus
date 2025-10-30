#include "cactus_ffi.h"
#include "../engine/engine.h"
#include <memory>
#include <string>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <thread>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include "stb_image_impl.h"
#include "stb_image_resize_impl.h"

using namespace cactus::engine;


struct CactusModel {
    std::unique_ptr<Model> model;
    std::string model_path;
    std::unordered_map<std::string, std::vector<uint32_t>> stop_sequence_cache;
    
    std::atomic<bool> should_stop;
    
    CactusModel() : should_stop(false) {}
};

static std::pair<int,int> resize_preserve_aspect(int width, int height, int longest_edge) {
    if (longest_edge <= 0) return {width, height};
    if (width >= height) {
        int nw = longest_edge;
        int nh = static_cast<int>(std::round(nw * (double)height / (double)width));
        if (nh % 2 != 0) nh++;
        return {nw, nh};
    } else {
        int nh = longest_edge;
        int nw = static_cast<int>(std::round(nh * (double)width / (double)height));
        if (nw % 2 != 0) nw++;
        return {nw, nh};
    }
}

static std::pair<int,int> resize_for_vision_encoder(int width, int height, int vision_encoder_max_size, int patch_size, int pixel_shuffle_factor) {
    if (vision_encoder_max_size <= 0) return {width, height};

    int max_dim = std::max(width, height);

    int target_size = ((max_dim + vision_encoder_max_size - 1) / vision_encoder_max_size) * vision_encoder_max_size;

    int alignment = patch_size * pixel_shuffle_factor;
    if (target_size % alignment != 0) {
        target_size = ((target_size + alignment - 1) / alignment) * alignment;
    }

    return {target_size, target_size};
}

static void convert_to_chw_float_and_normalize(unsigned char* pixels, int w, int h, int c, ImageBatch &out, float rescale, const std::array<float,3>& mean, const std::array<float,3>& std) {
    (void)c;
    out.width = static_cast<uint32_t>(w);
    out.height = static_cast<uint32_t>(h);
    out.channels = 3;
    out.data.assign((size_t)3 * w * h, 0.0f);
    out.pixel_mask.assign((size_t)w * h, 1);

    size_t plane = (size_t)w * h;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int idx = (y * w + x) * 3;
            float r = pixels[idx + 0] * rescale;
            float g = pixels[idx + 1] * rescale;
            float b = pixels[idx + 2] * rescale;
            out.data[0 * plane + y * w + x] = (r - mean[0]) / std[0];
            out.data[1 * plane + y * w + x] = (g - mean[1]) / std[1];
            out.data[2 * plane + y * w + x] = (b - mean[2]) / std[2];
        }
    }
}

static std::vector<std::vector<unsigned char>> split_image_tiles(unsigned char* pixels, int w, int h, int c, int max_size, int &out_tile_w, int &out_tile_h, int &out_grid_rows, int &out_grid_cols) {
    std::vector<std::vector<unsigned char>> tiles;
    if (w <= max_size && h <= max_size) {
        out_tile_w = w; out_tile_h = h;
        out_grid_rows = 1; out_grid_cols = 1;
        tiles.emplace_back(pixels, pixels + w*h*c);
        return tiles;
    }
    int num_splits_h = (h + max_size - 1) / max_size;
    int num_splits_w = (w + max_size - 1) / max_size;
    out_grid_rows = num_splits_h;
    out_grid_cols = num_splits_w;
    int optimal_h = (h + num_splits_h - 1) / num_splits_h;
    int optimal_w = (w + num_splits_w - 1) / num_splits_w;
    for (int r = 0; r < num_splits_h; ++r) {
        for (int cidx = 0; cidx < num_splits_w; ++cidx) {
            int sx = cidx * optimal_w;
            int sy = r * optimal_h;
            int ex = std::min(sx + optimal_w, w);
            int ey = std::min(sy + optimal_h, h);
            int tw = ex - sx;
            int th = ey - sy;
            std::vector<unsigned char> tile((size_t)tw*th*3);
            for (int yy = 0; yy < th; ++yy) {
                for (int xx = 0; xx < tw; ++xx) {
                    int src_idx = ((sy + yy) * w + (sx + xx)) * 3;
                    int dst_idx = (yy * tw + xx) * 3;
                    tile[dst_idx+0] = pixels[src_idx+0];
                    tile[dst_idx+1] = pixels[src_idx+1];
                    tile[dst_idx+2] = pixels[src_idx+2];
                }
            }
            tiles.push_back(std::move(tile));
        }
    }
    out_tile_w = max_size; out_tile_h = max_size;
    return tiles;
}

static std::string resolve_image_path(const std::string &attempt) {
    try {
        std::filesystem::path p(attempt);
        if (std::filesystem::exists(p)) return p.string();

        std::string base = p.filename().string();
        auto cwd = std::filesystem::current_path();

        std::filesystem::path cur = cwd;
        for (int i = 0; i < 6; ++i) {
            std::filesystem::path cand1 = cur / "tests" / "assets" / base;
            if (std::filesystem::exists(cand1)) return cand1.string();
            std::filesystem::path cand2 = cur / "assets" / base;
            if (std::filesystem::exists(cand2)) return cand2.string();
            cur = cur.parent_path();
            if (cur == cur.root_path()) break;
        }

        // As a last resort, try the basename in the original attempt's parent directory
        std::filesystem::path parent = p.parent_path();
        std::filesystem::path alt = parent / base;
        if (std::filesystem::exists(alt)) return alt.string();
    } catch (...) {
        // ignore filesystem errors and fall through
    }
    return attempt;
}


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
            std::string assembled;

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
                                if (!assembled.empty()) assembled += " ";
                                assembled += text;
                            }
                        } else if (type == "image") {
                            size_t path_pos = obj.find("\"path\"");
                            if (path_pos != std::string::npos) {
                                size_t pcol = obj.find(':', path_pos);
                                size_t pquote = obj.find('"', pcol);
                                size_t pquote_end = obj.find('"', pquote + 1);
                                std::string relpath = obj.substr(pquote + 1, pquote_end - pquote - 1);
                                try {
                                    std::filesystem::path p(relpath);
                                    std::filesystem::path ap = std::filesystem::absolute(p);
                                    out_image_paths.push_back(ap.string());
                                    out_image_placeholder_indices.push_back(assembled.length());
                                    if (!assembled.empty()) assembled += " ";
                                    assembled += std::string("<image>");
                                } catch (...) {
                                    out_image_placeholder_indices.push_back(assembled.length());
                                    if (!assembled.empty()) assembled += " ";
                                    assembled += std::string("<image>");
                                }
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

            msg.content = assembled;
            messages.push_back(msg);
            pos = json.find('{', idx);
            continue;
        }

        pos = json.find('{', after_colon);
    }

    return messages;
}

struct ToolFunction {
    std::string name;
    std::string description;
    std::unordered_map<std::string, std::string> parameters;
};

static std::vector<ToolFunction> parse_tools_json(const std::string& json) {
    std::vector<ToolFunction> tools;
    
    if (json.empty()) return tools;
    
    size_t pos = json.find('[');
    if (pos == std::string::npos) return tools;
    
    pos = json.find("\"function\"", pos);
    while (pos != std::string::npos) {
        ToolFunction tool;
        
        size_t name_pos = json.find("\"name\"", pos);
        if (name_pos != std::string::npos) {
            size_t name_start = json.find('\"', name_pos + 6) + 1;
            size_t name_end = json.find('\"', name_start);
            tool.name = json.substr(name_start, name_end - name_start);
        }
        
        size_t desc_pos = json.find("\"description\"", pos);
        if (desc_pos != std::string::npos) {
            size_t desc_start = json.find('\"', desc_pos + 13) + 1;
            size_t desc_end = json.find('\"', desc_start);
            tool.description = json.substr(desc_start, desc_end - desc_start);
        }
        
        size_t params_pos = json.find("\"parameters\"", pos);
        if (params_pos != std::string::npos) {
            size_t params_start = json.find('{', params_pos);
            if (params_start != std::string::npos) {
                int brace_count = 1;
                size_t params_end = params_start + 1;
                while (params_end < json.length() && brace_count > 0) {
                    if (json[params_end] == '{') brace_count++;
                    else if (json[params_end] == '}') brace_count--;
                    params_end++;
                }
                tool.parameters["schema"] = json.substr(params_start, params_end - params_start);
            }
        }
        
        tools.push_back(tool);
        
        pos = json.find("\"function\"", name_pos);
    }
    
    return tools;
}

static void parse_options_json(const std::string& json, 
                               float& temperature, float& top_p, 
                               size_t& top_k, size_t& max_tokens,
                               std::vector<std::string>& stop_sequences) {
    temperature = 0.7f;
    top_p = 0.95f;
    top_k = 20;
    max_tokens = 100;
    stop_sequences.clear();
    
    if (json.empty()) return;
    
    size_t pos = json.find("\"temperature\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        temperature = std::stof(json.substr(pos));
    }
    
    pos = json.find("\"top_p\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        top_p = std::stof(json.substr(pos));
    }
    
    pos = json.find("\"top_k\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        top_k = std::stoul(json.substr(pos));
    }
    
    pos = json.find("\"max_tokens\"");
    if (pos != std::string::npos) {
        pos = json.find(':', pos) + 1;
        max_tokens = std::stoul(json.substr(pos));
    }
    
    pos = json.find("\"stop_sequences\"");
    if (pos != std::string::npos) {
        pos = json.find('[', pos);
        if (pos != std::string::npos) {
            size_t end_pos = json.find(']', pos);
            size_t seq_pos = json.find('\"', pos);
            
            while (seq_pos != std::string::npos && seq_pos < end_pos) {
                size_t seq_start = seq_pos + 1;
                size_t seq_end = json.find('\"', seq_start);
                if (seq_end != std::string::npos) {
                    stop_sequences.push_back(json.substr(seq_start, seq_end - seq_start));
                }
                seq_pos = json.find('\"', seq_end + 1);
            }
        }
    }
}


static std::unordered_set<uint32_t> get_stop_tokens(CactusModel* wrapper, const std::vector<std::string>& stop_sequences) {
    std::unordered_set<uint32_t> stop_tokens;
    auto* tokenizer = wrapper->model->get_tokenizer();
    
    stop_tokens.insert(tokenizer->get_eos_token());
    
    for (const auto& stop_seq : stop_sequences) {
        auto cache_it = wrapper->stop_sequence_cache.find(stop_seq);
        if (cache_it != wrapper->stop_sequence_cache.end()) {
            
            for (uint32_t token : cache_it->second) {
                stop_tokens.insert(token);
            }
        } else {
            
            auto tokens = tokenizer->encode(stop_seq);
            wrapper->stop_sequence_cache[stop_seq] = tokens;
            for (uint32_t token : tokens) {
                stop_tokens.insert(token);
            }
        }
    }
    
    return stop_tokens;
}

extern "C" {

static std::string last_error_message;

const char* cactus_get_last_error() {
    return last_error_message.c_str();
}

cactus_model_t cactus_init(const char* model_path, size_t context_size) {
    try {
        auto* wrapper = new CactusModel();
        wrapper->model = create_model(model_path);
        wrapper->model_path = model_path;

        if (!wrapper->model) {
            last_error_message = "Failed to create model from: " + std::string(model_path);
            delete wrapper;
            return nullptr;
        }

        if (!wrapper->model->init(model_path, context_size)) {
            last_error_message = "Failed to initialize model from: " + std::string(model_path);
            delete wrapper;
            return nullptr;
        }
        
        auto* tokenizer = wrapper->model->get_tokenizer();
        std::vector<std::string> common_stops = {
            "\n\n", "###", "Human:", "Assistant:", "<|end|>", "<|endoftext|>", 
            "\n---", "User:", "AI:", "</s>", "<s>", "\n\nHuman:", "\n\nAssistant:"
        };
        for (const auto& stop_seq : common_stops) {
            wrapper->stop_sequence_cache[stop_seq] = tokenizer->encode(stop_seq);
        }
        
        return wrapper;
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
        
        for (auto& c : error_msg) {
            if (c == '"') c = '\'';
            if (c == '\n') c = ' ';
        }
        std::string error_json = "{\"success\":false,\"error\":\"" + error_msg + "\"}";
        if (error_json.length() < buffer_size) {
            std::strcpy(response_buffer, error_json.c_str());
        }
        return -1;
    }
    
    if (!messages_json || !response_buffer || buffer_size == 0) {
        std::string error_json = "{\"success\":false,\"error\":\"Invalid parameters\"}";
        if (error_json.length() < buffer_size) {
            std::strcpy(response_buffer, error_json.c_str());
        }
        return -1;
    }
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        auto* wrapper = static_cast<CactusModel*>(model);
        auto* tokenizer = wrapper->model->get_tokenizer();
        wrapper->should_stop = false;
        
        wrapper->model->reset_cache();
        
        
    std::vector<std::string> image_paths;
    std::vector<size_t> image_placeholder_indices;
    auto messages = parse_messages_json(messages_json, image_paths, image_placeholder_indices);
    std::vector<ImageBatch> preprocessed_images;
    std::vector<size_t> tiles_per_image;
    std::vector<std::pair<int,int>> grid_dims_per_image;
    if (!image_paths.empty()) {
            const auto& cfg = wrapper->model->get_config();
            int global_size = static_cast<int>(cfg.global_image_size);
            int max_tile_size = static_cast<int>(cfg.max_tile_size);
            int vision_size = static_cast<int>(cfg.vision_image_size);
            int patch_size = static_cast<int>(cfg.vision_patch_size);
            int shuffle_factor = static_cast<int>(cfg.pixel_shuffle_factor);
            float rescale = cfg.rescale_factor;
            std::array<float,3> mean = {cfg.image_mean, cfg.image_mean, cfg.image_mean};
            std::array<float,3> stdv = {cfg.image_std, cfg.image_std, cfg.image_std};

            for (const auto& path : image_paths) {
                int w=0,h=0,c=0;
                std::string resolved = resolve_image_path(path);
                unsigned char* data = stbi_load(resolved.c_str(), &w, &h, &c, 3);
                if (!data) {
                    tiles_per_image.push_back(0);
                    grid_dims_per_image.push_back({0, 0});
                    continue;
                }

                auto [global_w, global_h] = resize_preserve_aspect(w, h, global_size);
                std::vector<unsigned char> global_img;
                if (global_w != w || global_h != h) {
                    global_img.resize((size_t)global_w * global_h * 3);
                    unsigned char *gptr = stbir_resize_uint8_linear(data, w, h, 0, global_img.data(), global_w, global_h, 0, STBIR_RGB);
                    if (!gptr) {
                        global_img.assign(data, data + w * h * 3);
                        global_w = w; global_h = h;
                    }
                } else {
                    global_img.assign(data, data + w * h * 3);
                }

                int tile_w=0, tile_h=0, grid_rows=0, grid_cols=0;
                auto tiles = split_image_tiles(global_img.data(), global_w, global_h, 3, max_tile_size, tile_w, tile_h, grid_rows, grid_cols);
                tiles_per_image.push_back(tiles.size());
                grid_dims_per_image.push_back({grid_rows, grid_cols});

                for (auto &tile : tiles) {
                    int in_w = tile_w;
                    int in_h = tile_h;
                    int out_w = in_w;
                    int out_h = in_h;
                    std::tie(out_w, out_h) = resize_for_vision_encoder(in_w, in_h, vision_size, patch_size, shuffle_factor);

                    (void)in_w; (void)in_h; (void)out_w; (void)out_h; (void)vision_size; (void)patch_size; (void)shuffle_factor;

                    std::vector<unsigned char> resized_pixels;
                    if (out_w != in_w || out_h != in_h) {
                        resized_pixels.resize((size_t)out_w * out_h * 3);
                        unsigned char *res_ptr = stbir_resize_uint8_linear(tile.data(), in_w, in_h, 0, resized_pixels.data(), out_w, out_h, 0, STBIR_RGB);
                        if (!res_ptr) {
                            resized_pixels.assign(tile.begin(), tile.end());
                            out_w = in_w; out_h = in_h;
                        }
                    } else {
                        resized_pixels.assign(tile.begin(), tile.end());
                    }

                    ImageBatch ib;
                    convert_to_chw_float_and_normalize(resized_pixels.data(), out_w, out_h, 3, ib, rescale, mean, stdv);
                    preprocessed_images.push_back(std::move(ib));
                }

                stbi_image_free(data);
            }
        }
        if (messages.empty()) {
            std::string error_json = "{\"success\":false,\"error\":\"No messages provided\"}";
            std::strcpy(response_buffer, error_json.c_str());
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
        
        
        std::string full_prompt;
        if (!tools.empty()) {
            std::string formatted_tools_json;
            for (size_t i = 0; i < tools.size(); i++) {
                if (i > 0) formatted_tools_json += ",\n";
                formatted_tools_json += "  {\n";
                formatted_tools_json += "    \"type\": \"function\",\n";
                formatted_tools_json += "    \"function\": {\n";
                formatted_tools_json += "      \"name\": \"" + tools[i].name + "\",\n";
                formatted_tools_json += "      \"description\": \"" + tools[i].description + "\"";
                if (tools[i].parameters.find("schema") != tools[i].parameters.end()) {
                    formatted_tools_json += ",\n      \"parameters\": " + tools[i].parameters.at("schema");
                }
                formatted_tools_json += "\n    }\n  }";
            }
            
            full_prompt = tokenizer->format_chat_prompt(messages, true, formatted_tools_json);
        } else {
            full_prompt = tokenizer->format_chat_prompt(messages, true);
        }

        if (full_prompt.find("ERROR:") == 0) {
            std::string error_msg = full_prompt.substr(6);
            std::string error_json = "{\"success\":false,\"error\":\"" + error_msg + "\"}";
            std::strcpy(response_buffer, error_json.c_str());
            return -1;
        }

        if (!preprocessed_images.empty() && wrapper->model->get_config().model_type == cactus::engine::Config::ModelType::SMOLVLM) {
            uint32_t image_seq_len = wrapper->model->get_config().image_seq_len;

            std::string expanded_prompt;
            size_t last_pos = 0;
            size_t img_idx = 0;
            size_t search_pos = 0;

            while (search_pos < full_prompt.length() && img_idx < tiles_per_image.size()) {
                size_t img_pos = full_prompt.find("<image>", search_pos);
                if (img_pos == std::string::npos) break;

                expanded_prompt += full_prompt.substr(last_pos, img_pos - last_pos);

                size_t num_tiles = tiles_per_image[img_idx];
                auto [grid_rows, grid_cols] = grid_dims_per_image[img_idx];

                for (size_t tile = 0; tile < num_tiles; ++tile) {
                    int row = static_cast<int>(tile / grid_cols);
                    int col = static_cast<int>(tile % grid_cols);

                    expanded_prompt += "<fake_token_around_image>";

                    if (num_tiles > 1) {
                        expanded_prompt += "<row_" + std::to_string(row + 1) + "_col_" + std::to_string(col + 1) + ">";
                    }

                    if (tile == 0) {
                        expanded_prompt += "<global-img>";
                    }

                    for (uint32_t i = 0; i < image_seq_len; ++i) {
                        expanded_prompt += "<image>";
                    }
                    expanded_prompt += "<fake_token_around_image>";
                }

                img_idx++;
                search_pos = img_pos + 7;
                last_pos = search_pos;
            }

            expanded_prompt += full_prompt.substr(last_pos);
            full_prompt = expanded_prompt;
        }

        std::vector<uint32_t> tokens_to_process = tokenizer->encode(full_prompt);
        size_t prompt_tokens = tokens_to_process.size();
        
        
        std::unordered_set<uint32_t> stop_tokens = get_stop_tokens(wrapper, stop_sequences);
        
        std::vector<uint32_t> generated_tokens;
        double time_to_first_token = 0.0;
        std::string decoded_so_far;  


        uint32_t next_token;
        bool use_images = !preprocessed_images.empty() && wrapper->model->get_config().model_type == cactus::engine::Config::ModelType::SMOLVLM;
        (void)use_images; (void)preprocessed_images;
        if (tokens_to_process.empty()) {
            if (use_images) next_token = wrapper->model->generate_with_images({}, preprocessed_images, temperature, top_p, top_k);
            else next_token = wrapper->model->generate({}, temperature, top_p, top_k);
        } else {
            if (use_images) next_token = wrapper->model->generate_with_images(tokens_to_process, preprocessed_images, temperature, top_p, top_k, "profile.txt");
            else next_token = wrapper->model->generate(tokens_to_process, temperature, top_p, top_k, "profile.txt");
        }

        auto token_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(token_end - start_time);
        time_to_first_token = duration.count() / 1000.0;

        if (stop_tokens.count(next_token)) {
            generated_tokens.push_back(next_token);
        } else {
            generated_tokens.push_back(next_token);

            if (callback) {
                std::string full_decoded = tokenizer->decode(generated_tokens);
                std::string new_text = full_decoded.substr(decoded_so_far.length());
                decoded_so_far = full_decoded;
                callback(new_text.c_str(), next_token, user_data);
            }

            for (size_t i = 1; i < max_tokens; i++) {
                if (wrapper->should_stop) {
                    break;
                }

                std::vector<uint32_t> single_token = {next_token};
                    if (use_images) next_token = wrapper->model->generate_with_images(single_token, preprocessed_images, temperature, top_p, top_k);
                    else next_token = wrapper->model->generate(single_token, temperature, top_p, top_k);

                if (stop_tokens.count(next_token)) {
                    break;
                }

                generated_tokens.push_back(next_token);

                if (callback) {
                    std::string full_decoded = tokenizer->decode(generated_tokens);
                    std::string new_text = full_decoded.substr(decoded_so_far.length());
                    decoded_so_far = full_decoded;
                    callback(new_text.c_str(), next_token, user_data);
                }
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double total_time_ms = total_duration.count() / 1000.0;
        
        size_t completion_tokens = generated_tokens.size();
        double decode_time_ms = total_time_ms - time_to_first_token;
        double tokens_per_second = completion_tokens > 1 ? 
            ((completion_tokens - 1) * 1000.0) / decode_time_ms : 0.0;
        
        std::string response_text = tokenizer->decode(generated_tokens);

        std::string regular_response = response_text;
        std::vector<std::string> function_calls;

        size_t search_pos = 0;
        while ((search_pos = response_text.find("\"function_call\"", search_pos)) != std::string::npos) {
            
            size_t colon_pos = response_text.find(':', search_pos);
            if (colon_pos == std::string::npos) break;

            size_t json_start = response_text.find('{', colon_pos);
            if (json_start == std::string::npos) break;

            int brace_count = 1;
            size_t json_end = json_start + 1;
            while (json_end < response_text.length() && brace_count > 0) {
                if (response_text[json_end] == '{') brace_count++;
                else if (response_text[json_end] == '}') brace_count--;
                json_end++;
            }

            if (brace_count == 0) {
                std::string function_call = response_text.substr(json_start, json_end - json_start);
                function_calls.push_back(function_call);

                if (regular_response.find(function_call) != std::string::npos) {
                    size_t outer_start = response_text.rfind('{', search_pos);
                    if (outer_start != std::string::npos) {
                        int outer_brace_count = 1;
                        size_t outer_end = outer_start + 1;
                        while (outer_end < response_text.length() && outer_brace_count > 0) {
                            if (response_text[outer_end] == '{') outer_brace_count++;
                            else if (response_text[outer_end] == '}') outer_brace_count--;
                            outer_end++;
                        }
                        if (outer_brace_count == 0 && outer_end <= regular_response.length()) {
                            regular_response = response_text.substr(0, outer_start);
                            while (!regular_response.empty() && (regular_response.back() == ' ' || regular_response.back() == '\n' || regular_response.back() == '\r' || regular_response.back() == '\t')) {
                                regular_response.pop_back();
                            }
                        }
                    }
                }
            }

            search_pos = json_end;
        }
        
        std::ostringstream json_response;
        json_response << "{";
        json_response << "\"success\":true,";
        json_response << "\"response\":\"";
        for (char c : regular_response) {
            if (c == '"') json_response << "\\\"";
            else if (c == '\n') json_response << "\\n";
            else if (c == '\r') json_response << "\\r";
            else if (c == '\t') json_response << "\\t";
            else if (c == '\\') json_response << "\\\\";
            else json_response << c;
        }
        json_response << "\",";
        if (!function_calls.empty()) {
            json_response << "\"function_calls\":[";
            for (size_t i = 0; i < function_calls.size(); ++i) {
                if (i > 0) json_response << ",";
                json_response << function_calls[i];
            }
            json_response << "],";
        }
        json_response << "\"time_to_first_token_ms\":" << std::fixed << std::setprecision(2) << time_to_first_token << ",";
        json_response << "\"total_time_ms\":" << std::fixed << std::setprecision(2) << total_time_ms << ",";
        json_response << "\"tokens_per_second\":" << std::fixed << std::setprecision(2) << tokens_per_second << ",";
        json_response << "\"prefill_tokens\":" << prompt_tokens << ",";
        json_response << "\"decode_tokens\":" << completion_tokens << ",";
        json_response << "\"total_tokens\":" << (prompt_tokens + completion_tokens);
        json_response << "}";
        
        std::string result = json_response.str();
        if (result.length() >= buffer_size) {
            std::string error_json = "{\"success\":false,\"error\":\"Response buffer too small\"}";
            std::strcpy(response_buffer, error_json.c_str());
            return -1;
        }
        
        std::strcpy(response_buffer, result.c_str());
        
        return static_cast<int>(result.length());
        
    } catch (const std::exception& e) {
        std::string error_msg = e.what();
        for (auto& c : error_msg) {
            if (c == '"') c = '\'';
            if (c == '\n') c = ' ';
        }
        std::string error_json = "{\"success\":false,\"error\":\"" + error_msg + "\"}";
        if (error_json.length() < buffer_size) {
            std::strcpy(response_buffer, error_json.c_str());
        }
        return -1;
    }
}

void cactus_destroy(cactus_model_t model) {
    if (model) {
        delete static_cast<CactusModel*>(model);
    }
}

void cactus_reset(cactus_model_t model) {
    if (!model) return;
    
    auto* wrapper = static_cast<CactusModel*>(model);
    wrapper->model->reset_cache();
}

void cactus_stop(cactus_model_t model) {
    if (!model) return;
    auto* wrapper = static_cast<CactusModel*>(model);
    wrapper->should_stop = true;
}

int cactus_embed(
    cactus_model_t model,
    const char* text,
    float* embeddings_buffer,
    size_t buffer_size,
    size_t* embedding_dim
) {
    if (!model) {
        return -1;
    }
    
    if (!text || !embeddings_buffer || buffer_size == 0) {
        return -1;
    }
    
    try {
        auto* wrapper = static_cast<CactusModel*>(model);
        auto* tokenizer = wrapper->model->get_tokenizer();
        
        std::vector<uint32_t> tokens = tokenizer->encode(text);
        
        if (tokens.empty()) {
            return -1;
        }
        
        std::vector<float> embeddings = wrapper->model->get_embeddings(tokens, true);
        
        if (embeddings.size() * sizeof(float) > buffer_size) {
            return -2; 
        }
        
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
#include "../engine/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

void print_usage() {
    std::cerr << "Usage: cactus_debug_runner --model <model_dir> [--prompt <text> | --tokens <id,id,...>]\n"
              << "       [--context <size>] [--cache] [--profile <profile_file>]\n"
              << "       [--dump-final <file>] [--print-tokens]\n"
              << "       [--generate <count>] [--temperature <value>] [--top-p <value>]\n"
              << "       [--top-k <value>] [--stop-at-eos]\n"
              << "       [--dump-embeddings <file>] [--print-embeddings]\n";
}

std::vector<uint32_t> parse_tokens(const std::string& token_str) {
    std::vector<uint32_t> tokens;
    std::stringstream ss(token_str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        item.erase(0, item.find_first_not_of(" \t"));
        item.erase(item.find_last_not_of(" \t") + 1);
        if (item.empty()) {
            continue;
        }
        try {
            tokens.push_back(static_cast<uint32_t>(std::stoul(item)));
        } catch (...) {
            throw std::runtime_error("Invalid token id in --tokens list: " + item);
        }
    }
    return tokens;
}

std::string sanitize_token_text(const std::string& text) {
    std::ostringstream oss;
    for (unsigned char ch : text) {
        switch (ch) {
            case '\\': oss << "\\\\"; break;
            case '"': oss << "\\\""; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default:
                if (ch < 0x20) {
                    oss << "\\x" << std::uppercase << std::hex << std::setw(2) << std::setfill('0')
                        << static_cast<int>(ch) << std::nouppercase << std::dec << std::setfill(' ');
                } else {
                    oss << static_cast<char>(ch);
                }
                break;
        }
    }
    return oss.str();
}

std::vector<std::string> compute_token_texts(cactus::engine::Tokenizer* tokenizer,
                                             const std::vector<uint32_t>& sequence) {
    std::vector<std::string> texts(sequence.size());
    if (!tokenizer || sequence.empty()) {
        return texts;
    }

    std::vector<uint32_t> prefix;
    prefix.reserve(sequence.size());
    std::string previous_decoded;

    for (size_t i = 0; i < sequence.size(); ++i) {
        prefix.push_back(sequence[i]);
        std::string decoded = tokenizer->decode(prefix);
        if (decoded.size() >= previous_decoded.size()) {
            texts[i] = decoded.substr(previous_decoded.size());
        } else {
            texts[i].clear();
        }
        previous_decoded = std::move(decoded);
    }

    return texts;
}

} // namespace

int main(int argc, char** argv) {
    std::unordered_map<std::string, std::string> options;
    std::unordered_set<std::string> flags;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.rfind("--", 0) == 0) {
            if (i + 1 < argc) {
                std::string next(argv[i + 1]);
                if (next.rfind("--", 0) != 0) {
                    options[arg] = next;
                    ++i;
                    continue;
                }
            }
            flags.insert(arg);
        } else {
            std::cerr << "Unrecognized positional argument: " << arg << std::endl;
            print_usage();
            return 1;
        }
    }

    auto model_it = options.find("--model");
    if (model_it == options.end()) {
        std::cerr << "Error: --model <model_dir> is required." << std::endl;
        print_usage();
        return 1;
    }

    size_t context_size = 2048;
    if (auto ctx_it = options.find("--context"); ctx_it != options.end()) {
        try {
            context_size = std::stoul(ctx_it->second);
        } catch (...) {
            std::cerr << "Invalid --context value: " << ctx_it->second << std::endl;
            return 1;
        }
    }

    bool use_cache = flags.find("--cache") != flags.end();
    bool print_tokens = flags.find("--print-tokens") != flags.end();
    bool stop_at_eos = flags.find("--stop-at-eos") != flags.end();
    bool print_embeddings = flags.find("--print-embeddings") != flags.end();

    std::string profile_file;
    if (auto profile_it = options.find("--profile"); profile_it != options.end()) {
        profile_file = profile_it->second;
    }

    std::string dump_final_path;
    if (auto dump_it = options.find("--dump-final"); dump_it != options.end()) {
        dump_final_path = dump_it->second;
    }

    std::string dump_embeddings_path;
    if (auto embed_it = options.find("--dump-embeddings"); embed_it != options.end()) {
        dump_embeddings_path = embed_it->second;
    }

    size_t max_new_tokens = 0;
    if (auto gen_it = options.find("--generate"); gen_it != options.end()) {
        try {
            max_new_tokens = std::stoul(gen_it->second);
        } catch (...) {
            std::cerr << "Invalid --generate value: " << gen_it->second << std::endl;
            return 1;
        }
    }

    float temperature = -1.0f;
    if (auto temp_it = options.find("--temperature"); temp_it != options.end()) {
        try {
            temperature = std::stof(temp_it->second);
        } catch (...) {
            std::cerr << "Invalid --temperature value: " << temp_it->second << std::endl;
            return 1;
        }
    }

    float top_p = -1.0f;
    if (auto top_p_it = options.find("--top-p"); top_p_it != options.end()) {
        try {
            top_p = std::stof(top_p_it->second);
        } catch (...) {
            std::cerr << "Invalid --top-p value: " << top_p_it->second << std::endl;
            return 1;
        }
    }

    size_t top_k = 0;
    if (auto top_k_it = options.find("--top-k"); top_k_it != options.end()) {
        try {
            top_k = std::stoul(top_k_it->second);
        } catch (...) {
            std::cerr << "Invalid --top-k value: " << top_k_it->second << std::endl;
            return 1;
        }
    }

    try {
        auto model = cactus::engine::create_model(model_it->second);
        if (!model) {
            std::cerr << "Failed to create model from directory: " << model_it->second << std::endl;
            return 1;
        }

        if (!model->init(model_it->second, context_size)) {
            std::cerr << "Model initialization failed for directory: " << model_it->second << std::endl;
            return 1;
        }

        auto* tokenizer = model->get_tokenizer();

        std::vector<uint32_t> tokens;
        if (auto tokens_it = options.find("--tokens"); tokens_it != options.end()) {
            tokens = parse_tokens(tokens_it->second);
        }

        if (tokens.empty()) {
            auto prompt_it = options.find("--prompt");
            if (prompt_it == options.end()) {
                std::cerr << "Error: provide either --prompt or --tokens." << std::endl;
                print_usage();
                return 1;
            }
            if (!tokenizer) {
                std::cerr << "Tokenizer not available after model initialization." << std::endl;
                return 1;
            }
            tokens = tokenizer->encode(prompt_it->second);
        }

        if (tokens.empty()) {
            std::cerr << "No input tokens derived from prompt or token list." << std::endl;
            return 1;
        }

        if (print_tokens) {
            std::cout << "Tokens (" << tokens.size() << "): ";
            for (size_t i = 0; i < tokens.size(); ++i) {
                std::cout << tokens[i];
                if (i + 1 < tokens.size()) {
                    std::cout << ", ";
                }
            }
            std::cout << std::endl;
        }

        auto final_hidden = model->debug_forward(tokens, use_cache, profile_file);
        size_t hidden_dim = tokens.empty() ? 0 : final_hidden.size() / tokens.size();

        std::cout << "Final hidden tensor has " << final_hidden.size() << " values"
                  << " (seq_len=" << tokens.size() << ", hidden_dim=" << hidden_dim << ")" << std::endl;

        if (hidden_dim == 0 || tokens.empty()) {
            std::cout << "No hidden state information available for per-token summary." << std::endl;
        } else {
            auto token_texts = compute_token_texts(tokenizer, tokens);

            struct TokenStats {
                float min;
                float max;
                float mean;
                float stddev;
            };

            auto compute_stats = [](const float* data, size_t length) -> TokenStats {
                if (length == 0) {
                    return {0.0f, 0.0f, 0.0f, 0.0f};
                }

                float min_val = data[0];
                float max_val = data[0];
                double sum = 0.0;
                double sum_sq = 0.0;

                for (size_t i = 0; i < length; ++i) {
                    float value = data[i];
                    min_val = std::min(min_val, value);
                    max_val = std::max(max_val, value);
                    sum += value;
                    sum_sq += static_cast<double>(value) * static_cast<double>(value);
                }

                double mean = sum / static_cast<double>(length);
                double variance = std::max(0.0, (sum_sq / static_cast<double>(length)) - mean * mean);
                double stddev = std::sqrt(variance);

                return {min_val, max_val, static_cast<float>(mean), static_cast<float>(stddev)};
            };

            size_t tokens_to_show = std::min<size_t>(tokens.size(), 8);
            std::cout << "Per-token hidden summaries (showing " << tokens_to_show
                      << " of " << tokens.size() << " prefilling token(s)):" << std::endl;

            size_t preview_values = std::min<size_t>(8, hidden_dim);
            for (size_t token_index = 0; token_index < tokens_to_show; ++token_index) {
                const float* token_ptr = final_hidden.data() + token_index * hidden_dim;
                TokenStats stats = compute_stats(token_ptr, hidden_dim);

                std::cout << "  token " << token_index << " id=" << tokens[token_index];
                if (token_index < token_texts.size()) {
                    const std::string& piece = token_texts[token_index];
                    if (!piece.empty()) {
                        std::cout << " text=\"" << sanitize_token_text(piece) << "\"";
                    }
                }
                std::cout << " | min=" << std::setprecision(5) << stats.min
                          << " max=" << stats.max
                          << " mean=" << stats.mean
                          << " std=" << stats.stddev << std::endl;

                std::cout << "    values:";
                for (size_t d = 0; d < preview_values; ++d) {
                    std::cout << ' ' << std::setprecision(5) << token_ptr[d];
                }
                if (preview_values < hidden_dim) {
                    std::cout << " ...";
                }
                std::cout << std::endl;
            }
        }

        if (!dump_final_path.empty()) {
            std::ofstream ofs(dump_final_path);
            if (!ofs) {
                std::cerr << "Failed to open --dump-final path for writing: " << dump_final_path << std::endl;
            } else {
                ofs << std::setprecision(10);
                for (float value : final_hidden) {
                    ofs << value << '\n';
                }
                std::cout << "Wrote final hidden tensor to " << dump_final_path << std::endl;
            }
        }

        std::cout << "Layer dumps respect CACTUS_DEBUG_ENABLE, CACTUS_DEBUG_STDOUT, CACTUS_DEBUG_DIR, and related env vars." << std::endl;

        std::vector<uint32_t> generation_context = tokens;
        std::vector<uint32_t> generated_tokens;

        if (max_new_tokens > 0) {
            const auto& config = model->get_config();

            std::cout << "Starting autoregressive generation of " << max_new_tokens << " token(s)." << std::endl;

            model->reset_cache();

            generation_context.reserve(tokens.size() + max_new_tokens);
            generated_tokens.reserve(max_new_tokens);

            std::string decoded_so_far;
            if (tokenizer) {
                decoded_so_far = tokenizer->decode(tokens);
            }

            auto log_token = [&](size_t index) {
                uint32_t token_id = generation_context.back();
                std::cout << "  [gen " << index << "] token_id=" << token_id;
                if (tokenizer) {
                    std::string decoded_full = tokenizer->decode(generation_context);
                    std::string diff;
                    if (decoded_full.size() >= decoded_so_far.size()) {
                        diff = decoded_full.substr(decoded_so_far.size());
                    }
                    if (!diff.empty()) {
                        std::cout << " text=\"" << sanitize_token_text(diff) << "\"";
                    }
                    decoded_so_far = std::move(decoded_full);
                }
                std::cout << std::endl;
            };

            auto generate_step = [&](const std::vector<uint32_t>& input, bool allow_profile) {
                if (allow_profile && !profile_file.empty()) {
                    return model->generate(input, temperature, top_p, top_k, profile_file);
                }
                return model->generate(input, temperature, top_p, top_k);
            };

            uint32_t next_token = generate_step(generation_context, !profile_file.empty());

            generated_tokens.push_back(next_token);
            generation_context.push_back(next_token);
            log_token(0);

            bool reached_eos = stop_at_eos && next_token == config.eos_token_id;

            for (size_t i = 1; i < max_new_tokens && !reached_eos; ++i) {
                std::vector<uint32_t> input_token = {generated_tokens.back()};
                uint32_t candidate = generate_step(input_token, false);

                generated_tokens.push_back(candidate);
                generation_context.push_back(candidate);
                log_token(i);

                if (stop_at_eos && candidate == config.eos_token_id) {
                    reached_eos = true;
                }
            }

            if (reached_eos) {
                std::cout << "Hit EOS token; stopping generation." << std::endl;
            }

            if (!generated_tokens.empty()) {
                std::cout << "Generated tokens: ";
                for (size_t i = 0; i < generated_tokens.size(); ++i) {
                    std::cout << generated_tokens[i];
                    if (i + 1 < generated_tokens.size()) {
                        std::cout << ", ";
                    }
                }
                std::cout << std::endl;

                if (tokenizer) {
                    std::string generated_text = tokenizer->decode(generated_tokens);
                    std::cout << "Generated text: " << generated_text << std::endl;
                }
            } else {
                std::cout << "No tokens generated." << std::endl;
            }
        }

        bool want_embeddings = print_embeddings || !dump_embeddings_path.empty();
        if (want_embeddings) {
            if (generation_context.empty()) {
                std::cerr << "No tokens available for embedding dump." << std::endl;
            } else {
                model->reset_cache();
                auto embeddings = model->get_embeddings(generation_context, /*pooled=*/false);

                size_t total_tokens = generation_context.size();
                size_t embedding_dim = total_tokens == 0 ? 0 : embeddings.size() / total_tokens;

                if (embedding_dim * total_tokens != embeddings.size() || embedding_dim == 0) {
                    std::cerr << "Unexpected embedding size " << embeddings.size() << " for " << total_tokens
                              << " tokens." << std::endl;
                } else {
                    auto token_texts = compute_token_texts(tokenizer, generation_context);

                    std::ostringstream embed_ss;
                    embed_ss << std::setprecision(10);
                    embed_ss << "=== Token Embeddings (hidden_dim=" << embedding_dim << ") ===" << std::endl;

                    for (size_t i = 0; i < total_tokens; ++i) {
                        embed_ss << "token " << i << " [" << (i < tokens.size() ? "prompt" : "generated") << "] id="
                                 << generation_context[i];
                        if (tokenizer && i < token_texts.size()) {
                            std::string sanitized = sanitize_token_text(token_texts[i]);
                            if (!sanitized.empty()) {
                                embed_ss << " text=\"" << sanitized << "\"";
                            }
                        }
                        embed_ss << std::endl;

                        size_t offset = i * embedding_dim;
                        for (size_t d = 0; d < embedding_dim; ++d) {
                            embed_ss << embeddings[offset + d];
                            if (d + 1 < embedding_dim) {
                                embed_ss << ' ';
                            }
                        }
                        embed_ss << std::endl;
                    }

                    std::string embed_output = embed_ss.str();

                    if (print_embeddings) {
                        std::cout << embed_output;
                    }

                    if (!dump_embeddings_path.empty()) {
                        std::ofstream embed_file(dump_embeddings_path);
                        if (!embed_file) {
                            std::cerr << "Failed to open --dump-embeddings path for writing: "
                                      << dump_embeddings_path << std::endl;
                        } else {
                            embed_file << embed_output;
                            std::cout << "Wrote token embeddings to " << dump_embeddings_path << std::endl;
                        }
                    }
                }
            }
        }
    } catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
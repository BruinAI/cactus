#include "engine/engine.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

void print_usage() {
    std::cerr << "Usage: cactus_debug_runner --model <model_dir> [--prompt <text> | --tokens <id,id,...>]\n"
              << "       [--context <size>] [--cache] [--profile <profile_file>]\n"
              << "       [--dump-final <file>] [--print-tokens]\n"
              << "       [--generate <count>] [--temperature <value>] [--top-p <value>]\n"
              << "       [--top-k <value>] [--stop-at-eos]\n";
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

    std::string profile_file;
    if (auto profile_it = options.find("--profile"); profile_it != options.end()) {
        profile_file = profile_it->second;
    }

    std::string dump_final_path;
    if (auto dump_it = options.find("--dump-final"); dump_it != options.end()) {
        dump_final_path = dump_it->second;
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
            auto* tokenizer = model->get_tokenizer();
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

        size_t preview = std::min<size_t>(16, final_hidden.size());
        std::cout << "Preview: ";
        for (size_t i = 0; i < preview; ++i) {
            std::cout << final_hidden[i];
            if (i + 1 < preview) {
                std::cout << ", ";
            }
        }
        if (preview < final_hidden.size()) {
            std::cout << " ...";
        }
        std::cout << std::endl;

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

        if (max_new_tokens > 0) {
            auto* tokenizer = model->get_tokenizer();
            const auto& config = model->get_config();

            std::cout << "Starting autoregressive generation of " << max_new_tokens << " token(s)." << std::endl;

            model->reset_cache();

            std::vector<uint32_t> generation_context = tokens;
            std::vector<uint32_t> generated_tokens;
            generated_tokens.reserve(max_new_tokens);

            auto log_token = [&](size_t index, uint32_t token_id) {
                std::cout << "  [gen " << index << "] token_id=" << token_id;
                if (tokenizer) {
                    std::string piece = tokenizer->decode({token_id});
                    if (!piece.empty()) {
                        std::cout << " text=\"" << piece << "\"";
                    }
                }
                std::cout << std::endl;
            };

            auto generate_step = [&](const std::vector<uint32_t>& input, bool allow_profile) {
                if (allow_profile && !profile_file.empty()) {
                    return model->generate(input, temperature, top_p, top_k, profile_file);
                }
                return model->generate(input, temperature, top_p, top_k);
            };

            bool profile_used = false;

            if (max_new_tokens > 0) {
                std::vector<uint32_t> first_input = generation_context.empty() ? std::vector<uint32_t>{} : generation_context;
                uint32_t next_token = generate_step(first_input, !profile_used);
                profile_used = true;

                generated_tokens.push_back(next_token);
                generation_context.push_back(next_token);
                log_token(0, next_token);

                if (stop_at_eos && next_token == config.eos_token_id) {
                    std::cout << "Hit EOS token; stopping generation." << std::endl;
                } else {
                    for (size_t i = 1; i < max_new_tokens; ++i) {
                        std::vector<uint32_t> input_token = {generated_tokens.back()};
                        uint32_t candidate = generate_step(input_token, false);
                        generated_tokens.push_back(candidate);
                        generation_context.push_back(candidate);
                        log_token(i, candidate);

                        if (stop_at_eos && candidate == config.eos_token_id) {
                            std::cout << "Hit EOS token; stopping generation." << std::endl;
                            break;
                        }
                    }
                }
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
    } catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}

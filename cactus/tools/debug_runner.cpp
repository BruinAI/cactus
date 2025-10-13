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
              << "       [--dump-final <file>] [--print-tokens]\n";
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

    std::string profile_file;
    if (auto profile_it = options.find("--profile"); profile_it != options.end()) {
        profile_file = profile_it->second;
    }

    std::string dump_final_path;
    if (auto dump_it = options.find("--dump-final"); dump_it != options.end()) {
        dump_final_path = dump_it->second;
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
    } catch (const std::exception& ex) {
        std::cerr << "Fatal error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}

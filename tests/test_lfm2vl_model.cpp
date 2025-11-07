#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "../cactus/engine/engine.h"
#include "../cactus/models/model.h"

using namespace cactus::engine;

namespace {

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

    const std::string default_prompt = "Describe this image";

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
    messages.push_back({"system", "You are a helpful assistant.", "text"});
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

    std::cout << std::endl << "Test completed successfully!" << std::endl;
    return 0;
}

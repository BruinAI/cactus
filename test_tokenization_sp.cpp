#include "cactus/cactus.h"
#include <iostream>
#include <vector>
#include <filesystem>

using namespace cactus::engine;

std::filesystem::path find_project_root() {
    namespace fs = std::filesystem;
    fs::path current = fs::current_path();
    for (int depth = 0; depth < 6; ++depth) {
        if (fs::exists(current / "weights") && fs::exists(current / "cactus")) {
            return current;
        }
        if (!current.has_parent_path()) {
            break;
        }
        current = current.parent_path();
    }
    throw std::runtime_error("Unable to locate project root");
}

int main() {
    auto project_root = find_project_root();
    auto weights_dir = project_root / "weights" / "nomic-embed-text-v2-moe";

    SPTokenizer tokenizer;
    if (!tokenizer.load_vocabulary_with_config(
            (weights_dir / "vocab.txt").string(),
            (weights_dir / "merges.txt").string(),
            (weights_dir / "tokenizer_config.txt").string())) {
        std::cerr << "Failed to load tokenizer" << std::endl;
        return 1;
    }

    const std::string prompt = "Cactus activation inspection test.";
    auto tokens = tokenizer.encode(prompt);
    
    std::cout << "Tokenized: \"" << prompt << "\"" << std::endl;
    std::cout << "Number of tokens: " << tokens.size() << std::endl;
    std::cout << "Token IDs: [";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << tokens[i];
    }
    std::cout << "]" << std::endl;
    
    std::cout << "\nToken strings:" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(tokens.size(), 15); ++i) {
        std::cout << "  [" << i << "] ID=" << tokens[i] << ": \"" 
                  << tokenizer.decode({tokens[i]}) << "\"" << std::endl;
    }
    
    std::cout << "\nSpecial tokens:" << std::endl;
    std::cout << "  BOS: " << tokenizer.get_bos_token() << std::endl;
    std::cout << "  EOS: " << tokenizer.get_eos_token() << std::endl;
    std::cout << "  UNK: " << tokenizer.get_unk_token() << std::endl;
    
    std::cout << "\nExpected tokens: [0, 2041, 24392, 34704, 1363, 134071, 1830, 3034, 5, 2]" << std::endl;
    std::cout << "Expected strings: ['<s>', '▁Ca', 'ctus', '▁activa', 'tion', '▁inspect', 'ion', '▁test', '.', '</s>']" << std::endl;
    
    return 0;
}


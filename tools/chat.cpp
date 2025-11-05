#include "../cactus/ffi/cactus_ffi.h"
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <cstring>

void print_token(const char* token, uint32_t token_id, void* user_data) {
    std::cout << token << std::flush;
}

std::string escape_json(const std::string& s) {
    std::ostringstream o;
    for (char c : s) {
        switch (c) {
            case '"': o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\n': o << "\\n"; break;
            case '\r': o << "\\r"; break;
            case '\t': o << "\\t"; break;
            default: o << c; break;
        }
    }
    return o.str();
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>\n";
        std::cerr << "Example: " << argv[0] << " weights/gemma3-270m\n";
        return 1;
    }

    const char* model_path = argv[1];

    std::cout << "Loading model from " << model_path << "...\n";
    cactus_model_t model = cactus_init(model_path, 4096);

    if (!model) {
        std::cerr << "Failed to initialize model\n";
        return 1;
    }

    std::cout << "Model loaded successfully!\n\n";

    std::vector<std::string> history;

    while (true) {
        std::cout << "You: ";
        std::string user_input;
        std::getline(std::cin, user_input);

        if (user_input.empty()) continue;
        if (user_input == "quit" || user_input == "exit") break;
        if (user_input == "reset") {
            history.clear();
            cactus_reset(model);
            std::cout << "Conversation reset.\n\n";
            continue;
        }

        history.push_back(user_input);

        std::ostringstream messages_json;
        messages_json << "[";
        for (size_t i = 0; i < history.size(); i++) {
            if (i > 0) messages_json << ",";
            if (i % 2 == 0) {
                messages_json << "{\"role\":\"user\",\"content\":\""
                             << escape_json(history[i]) << "\"}";
            } else {
                messages_json << "{\"role\":\"assistant\",\"content\":\""
                             << escape_json(history[i]) << "\"}";
            }
        }
        messages_json << "]";

        const char* options = "{\"temperature\":0.7,\"top_p\":0.95,\"top_k\":40,\"max_tokens\":512,\"stop_sequences\":[\"<|im_end|>\",\"<end_of_turn>\"]}";

        char response_buffer[32768];

        std::cout << "Assistant: ";
        int result = cactus_complete(
            model,
            messages_json.str().c_str(),
            response_buffer,
            sizeof(response_buffer),
            options,
            nullptr,
            print_token,
            nullptr
        );
        std::cout << "\n\n";

        if (result < 0) {
            std::cerr << "Error: " << response_buffer << "\n\n";
            history.pop_back();
            continue;
        }

        std::string json_str(response_buffer);
        size_t response_start = json_str.find("\"response\":\"");
        if (response_start != std::string::npos) {
            response_start += 12;
            size_t response_end = json_str.find("\"", response_start);
            while (response_end != std::string::npos &&
                   json_str[response_end - 1] == '\\') {
                response_end = json_str.find("\"", response_end + 1);
            }
            if (response_end != std::string::npos) {
                std::string response = json_str.substr(response_start,
                                                       response_end - response_start);
                history.push_back(response);
            }
        }
    }

    std::cout << "Goodbye!\n";
    cactus_destroy(model);
    return 0;
}

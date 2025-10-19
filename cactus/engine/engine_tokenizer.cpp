#include "engine.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>

namespace cactus {
namespace engine {

void Tokenizer::detect_model_type(const std::string& config_path) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        model_type_ = ModelType::UNKNOWN;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        size_t pos = line.find("model_type");
        if (pos != std::string::npos) {
            std::transform(line.begin(), line.end(), line.begin(), ::tolower);

            if (line.find("qwen") != std::string::npos) {
                model_type_ = ModelType::QWEN;
                break;
            } else if (line.find("gemma") != std::string::npos) {
                model_type_ = ModelType::GEMMA;
                break;
            } else if (line.find("smol") != std::string::npos) {
                model_type_ = ModelType::SMOL;
                break;
            } else if (line.find("bert") != std::string::npos) {
                model_type_ = ModelType::BERT;
                break;
            }else if (line.find("llama") != std::string::npos) {
                model_type_ = ModelType::LLAMA;
                break;
            }
        }
    }
    file.close();
}

std::vector<uint32_t> Tokenizer::apply_chat_template(const std::vector<ChatMessage>& messages, bool add_generation_prompt) const {
    std::string formatted_prompt = format_chat_prompt(messages, add_generation_prompt);
    return encode(formatted_prompt);
}

std::string Tokenizer::format_chat_prompt(const std::vector<ChatMessage>& messages, bool add_generation_prompt, const std::string& tools_json) const {
    switch (model_type_) {
        case ModelType::QWEN:
            return format_qwen_style(messages, add_generation_prompt, tools_json);
        case ModelType::GEMMA:
            return format_gemma_style(messages, add_generation_prompt, tools_json);
        case ModelType::SMOL:
            return format_smol_style(messages, add_generation_prompt, tools_json);
        case ModelType::LLAMA:
            return format_llama_style(messages, add_generation_prompt, tools_json);
        default:
            return format_qwen_style(messages, add_generation_prompt, tools_json);
    }
}

std::string Tokenizer::format_qwen_style(const std::vector<ChatMessage>& messages, bool add_generation_prompt, const std::string& tools_json) const {
    std::string result;

    if (!tools_json.empty()) {
        result += "<|im_start|>system\n";

        bool has_system_msg = false;
        for (const auto& msg : messages) {
            if (msg.role == "system") {
                result += msg.content;
                result += "\n\n";
                has_system_msg = true;
                break;
            }
        }

        result += "You can call any of the following tools to satisfy the user's requests: [\n";
        result += tools_json;
        result += "\n]\n";
        result += "Example tool call syntax:\n";
        result += "{\n";
        result += "  \"tool_calls\": [\n";
        result += "    {\n";
        result += "      \"name\": \"tool_name\",\n";
        result += "      \"arguments\": {\n";
        result += "        \"arg1\": \"some_value\"\n";
        result += "      },\n";
        result += "      \"id\": \"call_1___\"\n";
        result += "    }\n";
        result += "  ]\n";
        result += "}";
        result += "<|im_end|>\n";

        for (const auto& msg : messages) {
            if (msg.role == "system" && has_system_msg) {
                continue; 
            } else if (msg.role == "user") {
                result += "<|im_start|>user\n" + msg.content + "<|im_end|>\n";
            } else if (msg.role == "assistant") {
                result += "<|im_start|>assistant\n" + msg.content + "<|im_end|>\n";
            }
        }
    } else {
        for (const auto& msg : messages) {
            if (msg.role == "system") {
                result += "<|im_start|>system\n" + msg.content + "<|im_end|>\n";
            } else if (msg.role == "user") {
                result += "<|im_start|>user\n" + msg.content + "<|im_end|>\n";
            } else if (msg.role == "assistant") {
                result += "<|im_start|>assistant\n" + msg.content + "<|im_end|>\n";
            }
        }
    }

    if (add_generation_prompt) {
        result += "<|im_start|>assistant\n";
    }

    return result;
}

std::string Tokenizer::format_llama_style(const std::vector<ChatMessage>& messages, bool add_generation_prompt, const std::string& tools_json) const {
    if (!tools_json.empty()) {
        return "ERROR: Tool calls are not supported for Llama models";
    }
    
    std::stringstream ss;
    bool system_handled = false;

    ss << "<|begin_of_text|>";

    for (const auto& message : messages) {
        if (!system_handled && message.role == "system") {
            ss << "<|start_header_id|>system<|end_header_id|>\n";
            ss << message.content << "\n";
            ss << "<|eot_id|>";
            system_handled = true;
            continue;
        }

        ss << "<|start_header_id|>" << message.role << "<|end_header_id|>\n";
        ss << message.content << "\n";
        ss << "<|eot_id|>";
    }

    if (add_generation_prompt) {
        ss << "<|start_header_id|>assistant<|end_header_id|>\n";
    }

    return ss.str();
}

std::string Tokenizer::format_gemma_style(const std::vector<ChatMessage>& messages, bool add_generation_prompt, const std::string& tools_json) const {

    if (!tools_json.empty()) {
        return "ERROR: Tool calls are not supported for Gemma models";
    }

    std::string result;

    result = "<bos>";

    std::string first_user_prefix = "";
    size_t start_idx = 0;

    if (!messages.empty() && messages[0].role == "system") {
        first_user_prefix = messages[0].content + "\n\n";
        start_idx = 1;
    }

    bool first_user = true;
    for (size_t i = start_idx; i < messages.size(); i++) {
        const auto& msg = messages[i];

        if (msg.role == "user") {
            result += "<start_of_turn>user";
            result += "\n";
            if (first_user && !first_user_prefix.empty()) {
                result += first_user_prefix;
                first_user = false;
            }
            result += msg.content;
            result += "<end_of_turn>";
            result += "\n";
        } else if (msg.role == "assistant") {
            result += "<start_of_turn>model";
            result += "\n";
            result += msg.content;
            result += "<end_of_turn>";
            result += "\n";
        }
    }

    if (add_generation_prompt) {
        result += "<start_of_turn>model";
        result += "\n";
    }

    return result;
}

std::string Tokenizer::format_smol_style(const std::vector<ChatMessage>& messages, bool add_generation_prompt, const std::string& tools_json) const {
    if (!tools_json.empty()) {
        return "ERROR: Tool calls are currently not supported for Smol models";
    }

    // if first message isn't system, add one
    std::string result;

    if (!messages.empty() && messages.front().role != "system") {
        result += "<|im_start|>system\n";
        result += "You are a helpful AI assistant named SmolLM, trained by Hugging Face";
        result += "<|im_end|>\n";
    }

    for (const auto& msg : messages) {
        result += "<|im_start|>";
        result += msg.role;
        result += "\n";
        result += msg.content;
        result += "<|im_end|>\n";
    }

    if (add_generation_prompt) {
        result += "<|im_start|>assistant\n";
    }

    return result;
}

} // namespace engine
} // namespace cactus
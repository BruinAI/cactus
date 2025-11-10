#include "engine.h"
#include <fstream>
#include <sstream>
#include <algorithm>

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
            } else if(line.find("lfm2") != std::string::npos) {
                model_type_ = ModelType::LFM2;
                break;
            } else if (line.find("smol") != std::string::npos) {
                model_type_ = ModelType::SMOL;
                break;
            } else if (line.find("bert") != std::string::npos) {
                model_type_ = ModelType::BERT;
                break;
            } else {
                model_type_ = ModelType::UNKNOWN;
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
        case ModelType::LFM2:
            return format_lfm2_style(messages, add_generation_prompt, tools_json);
        case ModelType::SMOL:
            return format_smol_style(messages, add_generation_prompt, tools_json);
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

        result += "You have access to the following tools:\n";
        result += "[\n";
        result += tools_json;
        result += "\n]\n\n";
        result += "When you need to call a tool, respond with a JSON object in this exact format:\n";
        result += "{\"function_call\": {\"name\": \"function_name\", \"arguments\": {\"arg1\": \"value1\"}}}";
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
        if (!tools_json.empty()) {
            result += "<|im_start|>assistant\n</think>\n\n";
        } else {
            result += "<|im_start|>assistant\n";
        }
    }

    return result;
}

std::string Tokenizer::format_lfm2_style(const std::vector<ChatMessage>& messages,
                                         bool add_generation_prompt,
                                         const std::string& tools_json) const
{
    std::string result = "<|startoftext|>";

    std::string sys_content;
    bool has_system_msg = false;
    for (const auto& msg : messages) {
        if (msg.role == "system") {
            sys_content = msg.content;
            has_system_msg = true;
            break;
        }
    }

    if (!tools_json.empty()) {
        if (!sys_content.empty()) {
            sys_content += "\n";
        }
        sys_content += "List of tools: <|tool_list_start|>[";
        if (!tools_json.empty()) {
            sys_content += "\n";
            sys_content += tools_json;
            sys_content += "\n";
        }
        sys_content += "]<|tool_list_end|>";
        sys_content += "\n\nWhen you need to call a tool, respond with a JSON object in this exact format:\n";
        sys_content += "{\"function_call\": {\"name\": \"function_name\", \"arguments\": {\"arg1\": \"value1\"}}}";
    }

    if (!sys_content.empty()) {
        result += "<|im_start|>system\n";
        result += sys_content;
        result += "<|im_end|>\n";
    }

    for (const auto& msg : messages) {
        if (msg.role == "system" && has_system_msg) {
            has_system_msg = false;
            continue;
        }
        result += "<|im_start|>" + msg.role + "\n";
        if (msg.role == "tool") {
            result += "<|tool_response_start|>";
            result += msg.content;
            result += "<|tool_response_end|>";
        } else {
            result += msg.content;
        }
        result += "<|im_end|>\n";
    }

    if (add_generation_prompt) {
        result += "<|im_start|>assistant\n";
    }

    return result;
}


std::string Tokenizer::format_gemma_style(const std::vector<ChatMessage>& messages, bool add_generation_prompt, const std::string& tools_json) const {
    std::string result;

    result = "<bos>";

    std::string first_user_prefix = "";
    size_t start_idx = 0;

    // Handle system message and tool definitions using BFCL format
    if (!tools_json.empty()) {
        // If there's a system message, prepend it to the BFCL prompt
        if (!messages.empty() && messages[0].role == "system") {
            first_user_prefix = messages[0].content + "\n\n";
            start_idx = 1;
        }

        // Extract BFCL-style function definitions from OpenAI-wrapped format
        // Input: {"type": "function", "function": {...}}
        // Output: {...}
        std::string bfcl_tools = "";
        size_t search_pos = 0;
        bool first_func = true;

        while (search_pos < tools_json.length()) {
            size_t func_pos = tools_json.find("\"function\"", search_pos);
            if (func_pos == std::string::npos) break;

            size_t colon_pos = tools_json.find(':', func_pos);
            if (colon_pos == std::string::npos) break;

            size_t brace_start = tools_json.find('{', colon_pos);
            if (brace_start == std::string::npos) break;

            // Find matching closing brace
            int brace_count = 1;
            size_t brace_end = brace_start + 1;
            while (brace_end < tools_json.length() && brace_count > 0) {
                if (tools_json[brace_end] == '{') brace_count++;
                else if (tools_json[brace_end] == '}') brace_count--;
                brace_end++;
            }

            if (brace_count == 0) {
                if (!first_func) bfcl_tools += ",\n";
                bfcl_tools += "  " + tools_json.substr(brace_start, brace_end - brace_start);
                first_func = false;
            }

            search_pos = brace_end;
        }

        // Add BFCL system prompt for Gemma 3 tool use
        first_user_prefix += "You are a helpful assistant with access to functions. Use them when needed.\n\n";
        first_user_prefix += "Available functions:\n";
        first_user_prefix += "[\n";
        first_user_prefix += bfcl_tools;
        first_user_prefix += "\n]\n\n";
        first_user_prefix += "To call a function, respond with this exact format:\n";
        first_user_prefix += "[function_name(param='value')]\n\n";
        first_user_prefix += "Rules:\n";
        first_user_prefix += "1. Function name must match exactly from list above\n";
        first_user_prefix += "2. Functions must be called within [] brackets\n";
        first_user_prefix += "3. Include () with parameters inside\n";
        first_user_prefix += "4. Use param='value' format (single quotes for strings)\n";
        first_user_prefix += "5. For numbers use param=123 (no quotes)\n";
        first_user_prefix += "6. To call multiple functions, list them within: [func1(x='a'), func2(y='b')]\n\n";
        first_user_prefix += "If no function is needed, respond normally.\n\n";
    } else if (!messages.empty() && messages[0].role == "system") {
        // No tools, but there is a system message
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
        } else if (msg.role == "tool") {
            // BFCL format for tool responses (multi-turn conversations)
            result += "<start_of_turn>user\n";
            result += "[Role: tool]\n";
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
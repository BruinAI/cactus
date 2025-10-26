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
            } else if (line.find("smolvlm") != std::string::npos) {
                model_type_ = ModelType::SMOLVLM;
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

void Tokenizer::load_special_tokens(const std::string& added_tokens_path) {
    std::ifstream file(added_tokens_path);
    if (!file.is_open()) {
        return;
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    auto find_token_id = [&content](const std::string& token_name) -> uint32_t {
        std::string search = "\"" + token_name + "\":";
        size_t pos = content.find(search);
        if (pos == std::string::npos) return 0;
        
        pos = content.find(':', pos) + 1;
        while (pos < content.length() && (content[pos] == ' ' || content[pos] == '\t')) pos++;
        
        size_t end_pos = pos;
        while (end_pos < content.length() && (std::isdigit(content[end_pos]) || content[end_pos] == '-')) end_pos++;
        
        if (end_pos > pos) {
            return static_cast<uint32_t>(std::stoul(content.substr(pos, end_pos - pos)));
        }
        return 0;
    };

    uint32_t image_token = find_token_id("<image>");
    if (image_token != 0) image_token_id_ = image_token;
    
    uint32_t fake_token = find_token_id("<fake_token_around_image>");
    if (fake_token != 0) fake_token_id_ = fake_token;
    
    uint32_t global_token = find_token_id("<global-img>");
    if (global_token != 0) global_img_token_id_ = global_token;
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
        case ModelType::SMOLVLM:
            return format_smolvlm_style(messages, add_generation_prompt, tools_json);
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
        result += "<|im_start|>assistant\n";
    }

    return result;
}

std::string Tokenizer::format_lfm2_style(const std::vector<ChatMessage>& messages,
                                         bool add_generation_prompt,
                                         const std::string& tools_json) const
{
    if (!tools_json.empty()) {
        return "ERROR: Tool calls are not supported for LFM2 models";
    }

    std::string result = "<|startoftext|>";

    for (const auto& msg : messages) {
        result += "<|im_start|>" + msg.role + "\n";
        result += msg.content;
        result += "<|im_end|>\n";
    }

    if (add_generation_prompt) {
        result += "<|im_start|>assistant\n";
    }

    return result;
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

std::string Tokenizer::format_smolvlm_style(const std::vector<ChatMessage>& messages, bool add_generation_prompt, const std::string& tools_json) const {
    if (!tools_json.empty()) {
        return "ERROR: Tool calls are currently not supported for SmolVLM models";
    }

    // if first message isn't system, add one
    std::string result;

    if (!messages.empty() && messages.front().role != "system") {
        result += "System: You are a helpful AI assistant named SmolVLM, trained by Hugging Face<end_of_utterance>\n";
    }

    for (const auto& msg : messages) {
        std::string role = msg.role;
        if (!role.empty()) {
            role[0] = static_cast<char>(::toupper(role[0]));
            for (size_t i = 1; i < role.size(); ++i) role[i] = static_cast<char>(::tolower(role[i]));
        }

        std::string content = msg.content;
        size_t first_non_ws = content.find_first_not_of(" \t\n\r");
        if (first_non_ws != std::string::npos) content = content.substr(first_non_ws);

        bool starts_with_image = false;
        const std::string image_marker = "<image>";
        if (content.size() >= image_marker.size() && content.compare(0, image_marker.size(), image_marker) == 0) {
            starts_with_image = true;
        }

        result += role;
        result += (starts_with_image ? ":" : ": ");

        result += msg.content;
        result += "<end_of_utterance>\n";
    }

    if (add_generation_prompt) {
        result += "Assistant:";
    }

    return result;
}


} // namespace engine
} // namespace cactus
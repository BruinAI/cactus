#include "test_utils.h"
#include <filesystem>
#include <fstream>
#include <iostream>

const char* g_vlm_model_path = "../../weights/lfm2-vl-350m-fp16";

struct StreamCapture {
    std::vector<std::string> tokens;
    std::vector<uint32_t> token_ids;
    int token_count = 0;
};

static void vlm_stream_cb(const char* token, uint32_t token_id, void* user_data) {
    auto* d = static_cast<StreamCapture*>(user_data);
    d->tokens.emplace_back(token);
    d->token_ids.push_back(token_id);
    d->token_count++;
    std::cout << token << std::flush;
}

bool test_vlm_pipeline() {
    std::string model_path_str(g_vlm_model_path);
    // Skip if model folder not present
    if (!std::filesystem::exists(model_path_str)) {
        std::cout << "Skipping VLM pipeline test: weights folder not found: " << model_path_str << std::endl;
        return true;
    }

    // Quick check for vision weights file presence
    std::filesystem::path vision_check = std::filesystem::path(model_path_str) / "vision_patch_embedding.weights";
    if (!std::filesystem::exists(vision_check)) {
        std::cout << "Skipping VLM pipeline test: vision weights not present in model folder: " << vision_check << std::endl;
        return true;
    }

    cactus_model_t model = cactus_init(g_vlm_model_path, 2048);

    // Build message payload with an image and a follow-up text instruction
    std::filesystem::path rel_img_path = std::filesystem::path("assets/test_image2.png");
    std::filesystem::path abs_img_path = std::filesystem::absolute(rel_img_path);
    std::string img_path_str = abs_img_path.string();

    std::string messages_json = "[";
    messages_json += "{\"role\": \"user\", \"content\": [";
    messages_json += "{\"type\": \"image\", \"path\": \"" + img_path_str + "\"},";
    messages_json += "{\"type\": \"text\", \"text\": \"Describe the main objects in this image in one sentence.\"}";
    messages_json += "]}";
    messages_json += "]";

    char response[8192];
    const char* options = R"({"max_tokens":128,"stop_sequences": ["<|image_end|>"]})";

    StreamCapture capture;

    std::cout << "\n=== VLM pipeline test ===" << std::endl;
    int res = cactus_complete(model, messages_json.c_str(), response, sizeof(response), options, nullptr, vlm_stream_cb, &capture);

    std::cout << "\n=== End of VLM test ===\n" << std::endl;
    std::cout << "Response JSON: " << response << std::endl;

    cactus_destroy(model);

    // Validate that we received some tokens and the response JSON indicates success
    if (res <= 0) return false;
    std::string resp_str(response);
    if (resp_str.find("\"success\":true") == std::string::npos) return false;
    if (capture.token_count == 0) return false;

    return true;
}

int main() {
    TestUtils::TestRunner runner("VLM Pipeline");
    runner.run_test("vlm_pipeline", test_vlm_pipeline());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}

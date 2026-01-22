#include "test_utils.h"
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <algorithm>
#include <cctype>
#include <cstdio>

using namespace EngineTestUtils;

const char* g_transcribe_model_path = std::getenv("CACTUS_TEST_TRANSCRIBE_MODEL");
const char* g_assets_path = std::getenv("CACTUS_TEST_ASSETS");

static const char* get_transcribe_prompt() {
    if (g_transcribe_model_path) {
        std::string path = g_transcribe_model_path;
        std::transform(path.begin(), path.end(), path.begin(), [](unsigned char c){ return std::tolower(c); });
        if (path.find("whisper") != std::string::npos) {
            return "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>";
        }
    }
    return "";
}

const char* g_whisper_prompt = get_transcribe_prompt();

static bool test_transcription_profile() {
    if (!g_transcribe_model_path) {
        std::cerr << "CACTUS_TEST_TRANSCRIBE_MODEL not set\n";
        return false;
    }

    cactus_model_t model = cactus_init(g_transcribe_model_path, nullptr);
    if (!model) {
        std::cerr << "Failed to initialize model\n";
        return false;
    }

    char response[1 << 15] = {0};
    
    // Enable profiling via environment variable
    const char* profile_file = "transcription_profile.json";
    setenv("CACTUS_PROFILE", profile_file, 1);
    std::remove(profile_file);

    const char* options_json = R"({
        "max_tokens": 100
    })";

    std::string audio_path = std::string(g_assets_path ? g_assets_path : ".") + "/test.wav";
    
    std::cout << "Running transcription with profiling...\n";
    std::cout << "Model: " << g_transcribe_model_path << "\n";
    std::cout << "Audio: " << audio_path << "\n";
    std::cout << "Profile output: transcription_profile.json\n\n";

    int rc = cactus_transcribe(model, audio_path.c_str(), g_whisper_prompt,
                               response, sizeof(response), options_json,
                               nullptr, nullptr, nullptr, 0);

    if (rc <= 0) {
        std::cerr << "Transcription failed\n";
        cactus_destroy(model);
        return false;
    }

    std::cout << "Transcript: " << response << "\n";
    std::cout << "Profiling completed.\n";

    cactus_destroy(model);
    return true;
}

int main() {
    if (test_transcription_profile()) {
        std::cout << "Test passed\n";
        return 0;
    } else {
        std::cout << "Test failed\n";
        return 1;
    }
}

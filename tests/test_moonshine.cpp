#include "test_utils.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>

using namespace EngineTestUtils;

const char* g_moonshine_model_path = std::getenv("CACTUS_TEST_MOONSHINE_MODEL");
const char* g_assets_path = std::getenv("CACTUS_TEST_ASSETS");
const char* g_whisper_prompt = "";

template<typename Predicate>
bool run_moonshine_test(const char* title, const char* options_json, Predicate check) {
    if (!g_moonshine_model_path) {
        std::cout << "⊘ SKIP │ " << std::left << std::setw(25) << title
                  << " │ CACTUS_TEST_MOONSHINE_MODEL not set\n";
        return true;
    }

    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║" << std::setw(42) << std::left << std::string("          ") + title << "║\n"
              << "╚══════════════════════════════════════════╝\n";

    cactus_model_t model = cactus_init(g_moonshine_model_path, nullptr);
    if (!model) {
        std::cerr << "[✗] Failed to initialize Moonshine model\n";
        return false;
    }

    char response[1 << 15] = {0};
    StreamingData stream;
    stream.model = model;

    std::string audio_path = std::string(g_assets_path) + "/test.wav";
    std::cout << "Transcript: ";
    int rc = cactus_transcribe(model, audio_path.c_str(), g_whisper_prompt,
                               response, sizeof(response), options_json,
                               stream_callback, &stream, nullptr, 0);

    std::cout << "\n\n[Results]\n";
    if (rc <= 0) {
        std::cerr << "failed\n";
        cactus_destroy(model);
        return false;
    }

    Metrics m;
    m.parse(response);
    m.print_json();

    bool ok = check(rc, m);
    cactus_destroy(model);
    return ok;
}

static bool test_moonshine_transcription() {
    return run_moonshine_test("MOONSHINE TRANSCRIPTION", R"({"max_tokens": 100})",
        [](int rc, const Metrics& m) { return rc > 0 && m.completion_tokens >= 5; });
}

int main() {
    TestUtils::TestRunner runner("Moonshine Engine Tests");
    runner.run_test("moonshine_transcription", test_moonshine_transcription());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}

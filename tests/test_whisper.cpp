#include "test_utils.h"
#include "../cactus/ffi/cactus_ffi.h"
#include "../cactus/ffi/ffi_utils.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>

// ---------------------------
// Config
// ---------------------------
static const char* kModelPath = "../../weights/whisper-medium16";
static std::string tokenizerPathStr = std::string(kModelPath) + "/tokenizer.json";
const char* tokenizerPath = tokenizerPathStr.c_str();
static const char* kMelFile   = "/Users/parkiratsandhu/Documents/programming_projects/cactus_bruinai/cactus/tests/whisper_tests/mel.npy";
static const char* kTokFile   = "/Users/parkiratsandhu/Documents/programming_projects/cactus_bruinai/cactus/tests/whisper_tests/decoder_input_tokens.npy";

// ---------------------------
// Streaming collector
// ---------------------------
struct StreamingData {
    std::vector<std::string> tokens;
    std::vector<uint32_t> token_ids;
    int token_count = 0;
    cactus_model_t model = nullptr;
    int stop_at = -1;
};

static void whisper_stream_callback(const char* token, uint32_t token_id, void* user_data) {
    auto* data = static_cast<StreamingData*>(user_data);
    data->tokens.push_back(token);
    data->token_ids.push_back(token_id);
    data->token_count++;

    // If text is empty (special token), print token id
    if (token && *token)
        std::cout << token << std::flush;
    else
        std::cout << "<" << token_id << ">" << std::flush;

    if (data->stop_at > 0 && data->token_count >= data->stop_at) {
        std::cout << "\n\n[→ Stopping at token #" << data->stop_at << "]" << std::endl;
        cactus_stop(data->model);
    }
}

// ---------------------------
// JSON field extractors (robust)
// ---------------------------
static std::string extract_json_string_field(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return {};
    size_t q1 = json.find('"', pos + pattern.size());
    if (q1 == std::string::npos) return {};
    size_t q2 = json.find('"', q1 + 1);
    if (q2 == std::string::npos) return {};
    return json.substr(q1 + 1, q2 - q1 - 1);
}

static double extract_json_number_field(const std::string& json, const std::string& key, double def = 0.0) {
    std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return def;
    size_t start = pos + pattern.size();
    while (start < json.size() && (json[start] == ' ' || json[start] == '\t')) ++start;
    size_t end = start;
    while (end < json.size() && std::string(",}] \t\n\r").find(json[end]) == std::string::npos) ++end;
    try { return std::stod(json.substr(start, end - start)); }
    catch (...) { return def; }
}

// ---------------------------
// Core test runner
// ---------------------------
template<typename Predicate>
bool run_whisper_test(const char* title,
                      float temperature,
                      float top_p,
                      size_t top_k,
                      size_t max_tokens,
                      bool use_streaming,
                      int stop_at,
                      Predicate check)
{
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║" << std::setw(42) << std::left << std::string("          ") + title << "║\n"
              << "╚══════════════════════════════════════════╝\n";

    cactus_model_t model = cactus_init(kModelPath, 2048, tokenizerPath);
    if (!model) {
        std::cerr << "[✗] Failed to initialize Whisper model\n";
        return false;
    }

    char response[1 << 15] = {0};

    StreamingData stream;
    stream.model   = model;
    stream.stop_at = stop_at;

    std::cout << "Transcript (streamed if enabled): ";
    int rc = cactus_test_whisper_from_files_json(
        model, kMelFile, kTokFile,
        response, sizeof(response),
        temperature, top_p, top_k, max_tokens,
        use_streaming ? whisper_stream_callback : nullptr,
        use_streaming ? (void*)&stream : nullptr
    );

    std::cout << "\n\n[Results]\n";
    if (rc <= 0) {
        std::cerr << "failed\n";
        cactus_destroy(model);
        return false;
    }

    std::string json = response;

    // Extract metrics (robust to field renames)
    double ttft_ms   = extract_json_number_field(json, "time_to_first_token_ms");
    double tps       = extract_json_number_field(json, "tokens_per_second");
    double total_ms  = extract_json_number_field(json, "total_time_ms");

    double n_prompt  = extract_json_number_field(json, "prompt_tokens",
                        extract_json_number_field(json, "prefill_tokens", 0.0));
    double n_comp    = extract_json_number_field(json, "completion_tokens",
                        extract_json_number_field(json, "decode_tokens", 0.0));

    std::string text = extract_json_string_field(json, "text");
    if (text.empty()) text = extract_json_string_field(json, "response");

    std::cout << "├─ Time to first token: " << std::fixed << std::setprecision(2) << ttft_ms << " ms\n"
              << "├─ Tokens per second:  " << tps << "\n"
              << "├─ Total time:         " << total_ms << " ms\n"
              << "├─ Prompt tokens:      " << n_prompt << "\n"
              << "├─ Completion tokens:  " << n_comp << "\n"
              << "├─ JSON size:          " << std::strlen(response) << " bytes\n"
              << "└─ Transcript (first 300 chars): "
              << (text.size() > 300 ? text.substr(0, 300) + "..." : text) << "\n";

    bool ok = check(rc, text, ttft_ms, tps, n_comp, use_streaming ? stream.token_count : (int)n_comp);
    std::cout << "Status: " << (ok ? "PASSED ✓" : "FAILED ✗") << "\n";

    cactus_destroy(model);
    return ok;
}

// ---------------------------
// Individual tests (robust predicates)
// ---------------------------
static bool test_whisper_prefill_only() {
    return run_whisper_test(
        "WHISPER PREFILL ONLY",
        0.0f, 1.0f, 0, 1,  // <= only 1 token decode, no continuation
        false,             // no streaming
        -1,
        [](int rc,
           const std::string& /*text*/,
           double /*ttft*/,
           double /*tps*/,
           double n_comp,
           int /*streamed_tokens*/) {
            // success if prefill ran and produced logits, even if no tokens were sampled
            return rc > 0 && n_comp >= 0.0;
        }
    );
}


static bool test_whisper_autoregressive_longer() {
    return run_whisper_test(
        "WHISPER AUTOREGRESSIVE (100 TOKENS)",
        0.0f, 0.0f, 0, 100, false, -1,
        [](int rc,
           const std::string& /*text*/,
           double /*ttft*/,
           double /*tps*/,
           double n_comp,
           int /*streamed_tokens*/) {
            return rc > 0 && n_comp >= 8.0;
        }
    );
}

static bool test_whisper_streaming_early_stop() {
    return run_whisper_test(
        "WHISPER STREAMING (EARLY STOP @ 20)",
        0.0f, 1.0f, 0, 128, true, 20,
        [](int rc,
           const std::string& /*text*/,
           double /*ttft*/,
           double /*tps*/,
           double /*n_comp*/,
           int streamed_tokens) {
            return rc > 0 && streamed_tokens >= 20;
        }
    );
}

// ---------------------------
// Test runner main
// ---------------------------
int main() {
    TestUtils::TestRunner runner("Whisper Tests");
    // runner.run_test("whisper_prefill_basic",      test_whisper_prefill_only());
    runner.run_test("whisper_autoregressive_100",  test_whisper_autoregressive_longer());
    // runner.run_test("whisper_streaming_stop20",   test_whisper_streaming_early_stop());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}

#include "../cactus/ffi/cactus_ffi.h"
#include <iostream>
#include <iomanip>
#include <cstring>
#include <vector>
#include <string>

struct TestCase {
    std::string name;
    std::string tools;
    std::string query;
    std::string expected_function;
};

void run_quick_test(cactus_model_t model) {
    std::cout << "Quick Smoke Test\n";
    std::cout << "================\n\n";

    const char* tools = R"([{
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state"
                    }
                },
                "required": ["location"]
            }
        }
    }])";

    const char* messages = R"([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather like in San Francisco?"}
    ])";

    const char* options = R"({"max_tokens": 100, "temperature": 0.1})";

    char response[4096];

    std::cout << "User query: What's the weather like in San Francisco?\n";
    std::cout << "Expected: [get_weather(location='San Francisco')]\n\n";
    std::cout << "Generating...\n";

    int result = cactus_complete(model, messages, response, sizeof(response),
                                 options, tools, nullptr, nullptr);

    if (result < 0) {
        std::cout << "✗ Generation failed\n\n";
        return;
    }

    std::cout << "\nResponse:\n";
    std::cout << "----------------------------------------\n";
    std::cout << response << "\n";
    std::cout << "----------------------------------------\n\n";

    bool has_function_calls = strstr(response, "\"function_calls\"") != nullptr;
    bool has_weather = strstr(response, "get_weather") != nullptr;
    bool has_location = strstr(response, "location") != nullptr;

    if (has_function_calls) {
        std::cout << "✓ Function calls detected\n";
        if (has_weather) std::cout << "✓ get_weather function found\n";
        if (has_location) std::cout << "✓ location parameter found\n";
    } else {
        std::cout << "✗ No function_calls in response\n";
    }

    std::cout << "\n";
}

void run_comprehensive_tests(cactus_model_t model) {
    std::cout << "Comprehensive Test Suite\n";
    std::cout << "========================\n\n";

    std::vector<TestCase> tests = {
        {
            "Weather",
            R"([{"function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"location":{"type":"string"}}}}}])",
            "What is the weather in Paris?",
            "get_weather"
        },
        {
            "Calculator",
            R"([{"function":{"name":"calculate","description":"Do math","parameters":{"type":"object","properties":{"expression":{"type":"string"}}}}}])",
            "Calculate 42 times 7",
            "calculate"
        },
        {
            "Email",
            R"([{"function":{"name":"send_email","description":"Send email","parameters":{"type":"object","properties":{"to":{"type":"string"},"subject":{"type":"string"}}}}}])",
            "Email bob@test.com about the meeting",
            "send_email"
        },
        {
            "Time",
            R"([{"function":{"name":"get_time","description":"Get time","parameters":{"type":"object","properties":{"timezone":{"type":"string"}}}}}])",
            "What time is it in Berlin?",
            "get_time"
        },
        {
            "Search",
            R"([{"function":{"name":"search","description":"Search","parameters":{"type":"object","properties":{"query":{"type":"string"}}}}}])",
            "Search for machine learning tutorials",
            "search"
        },
        {
            "Translation",
            R"([{"function":{"name":"translate","description":"Translate text","parameters":{"type":"object","properties":{"text":{"type":"string"},"target_language":{"type":"string"}}}}}])",
            "Translate hello to Spanish",
            "translate"
        },
        {
            "Reminder",
            R"([{"function":{"name":"set_reminder","description":"Set reminder","parameters":{"type":"object","properties":{"message":{"type":"string"},"time":{"type":"string"}}}}}])",
            "Remind me to call John at 3pm",
            "set_reminder"
        },
        {
            "FileRead",
            R"([{"function":{"name":"read_file","description":"Read file","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}])",
            "Read config.json",
            "read_file"
        }
    };

    const char* options = R"({"max_tokens": 100, "temperature": 0.1})";

    int passed = 0;
    std::vector<std::string> failed;

    for (size_t i = 0; i < tests.size(); i++) {
        const auto& t = tests[i];
        std::cout << "[" << (i+1) << "/" << tests.size() << "] " << t.name << ": \"" << t.query << "\"\n";

        std::string msg = "[{\"role\":\"user\",\"content\":\"" + t.query + "\"}]";
        char resp[4096];

        int res = cactus_complete(model, msg.c_str(), resp, sizeof(resp), options, t.tools.c_str(), nullptr, nullptr);

        bool ok = (res >= 0) &&
                  (strstr(resp, "\"function_calls\"") != nullptr) &&
                  (strstr(resp, t.expected_function.c_str()) != nullptr);

        if (ok) {
            std::cout << "  ✓ PASS\n";
            passed++;
        } else {
            std::cout << "  ✗ FAIL";
            if (res < 0) std::cout << " (generation error)";
            else if (strstr(resp, "\"function_calls\"") == nullptr) std::cout << " (no function_calls)";
            else std::cout << " (wrong function)";
            std::cout << "\n";
            failed.push_back(t.name);
        }
    }

    std::cout << "\n=======================================================\n";
    std::cout << "Results: " << passed << "/" << tests.size() << " passed";
    float pct = (float)passed / tests.size() * 100.0f;
    std::cout << " (" << std::fixed << std::setprecision(1) << pct << "%)\n";

    if (!failed.empty()) {
        std::cout << "\nFailed: ";
        for (size_t i = 0; i < failed.size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << failed[i];
        }
        std::cout << "\n";
    }
}

int main(int argc, char* argv[]) {
    std::cout << "Gemma BFCL Function Calling Test Suite\n";
    std::cout << "=======================================\n\n";

    const char* model_path = (argc > 1) ? argv[1] : "../../weights/gemma3-1b";
    cactus_model_t model = cactus_init(model_path, 2048);

    if (!model) {
        std::cerr << "Failed to initialize model: " << model_path << "\n";
        return 1;
    }

    std::cout << "✓ Model initialized: " << model_path << "\n\n";

    // Run quick smoke test first
    run_quick_test(model);

    // Then run comprehensive suite
    run_comprehensive_tests(model);

    cactus_destroy(model);
    return 0;
}

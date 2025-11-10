#include "../cactus/ffi/ffi_utils.h"
#include <iostream>
#include <cassert>

using namespace cactus::ffi;

void test_named_parameters() {
    std::cout << "Named Parameters Tests\n";
    std::cout << "======================\n\n";

    // Test 1: Single function call
    {
        std::cout << "Test 1: Single function call\n";
        std::string response = "[get_weather(location='San Francisco', unit='celsius')]";
        std::string regular_response;
        std::vector<std::string> function_calls;

        parse_function_calls_from_response(response, regular_response, function_calls, FunctionCallFormat::BFCL);

        std::cout << "Input: " << response << "\n";
        std::cout << "Function calls found: " << function_calls.size() << "\n";
        if (!function_calls.empty()) {
            std::cout << "Parsed: " << function_calls[0] << "\n";
        }
        std::cout << "Regular response: '" << regular_response << "'\n";

        assert(function_calls.size() == 1);
        assert(regular_response.empty());
        std::cout << "✓ PASSED\n\n";
    }

    // Test 2: Multiple function calls
    {
        std::cout << "Test 2: Multiple function calls\n";
        std::string response = "[get_weather(location='NYC'), get_time(timezone='EST')]";
        std::string regular_response;
        std::vector<std::string> function_calls;

        parse_function_calls_from_response(response, regular_response, function_calls, FunctionCallFormat::BFCL);

        std::cout << "Input: " << response << "\n";
        std::cout << "Function calls found: " << function_calls.size() << "\n";
        for (size_t i = 0; i < function_calls.size(); i++) {
            std::cout << "  [" << i << "] " << function_calls[i] << "\n";
        }

        assert(function_calls.size() == 2);
        std::cout << "✓ PASSED\n\n";
    }

    // Test 3: Numeric parameters
    {
        std::cout << "Test 3: Numeric parameters\n";
        std::string response = "[calculate_area(width=10, height=5)]";
        std::string regular_response;
        std::vector<std::string> function_calls;

        parse_function_calls_from_response(response, regular_response, function_calls, FunctionCallFormat::BFCL);

        std::cout << "Input: " << response << "\n";
        std::cout << "Parsed: " << function_calls[0] << "\n";

        assert(function_calls.size() == 1);
        assert(function_calls[0].find("\"width\":10") != std::string::npos);
        assert(function_calls[0].find("\"height\":5") != std::string::npos);
        std::cout << "✓ PASSED\n\n";
    }

    // Test 4: No function call (regular text)
    {
        std::cout << "Test 4: No function call (regular text)\n";
        std::string response = "The weather in San Francisco is sunny.";
        std::string regular_response;
        std::vector<std::string> function_calls;

        parse_function_calls_from_response(response, regular_response, function_calls, FunctionCallFormat::BFCL);

        std::cout << "Input: " << response << "\n";
        std::cout << "Function calls found: " << function_calls.size() << "\n";
        std::cout << "Regular response: '" << regular_response << "'\n";

        assert(function_calls.empty());
        assert(regular_response == response);
        std::cout << "✓ PASSED\n\n";
    }
}

void test_positional_arguments() {
    std::cout << "Positional Arguments Tests\n";
    std::cout << "===========================\n\n";

    // Test 1: Positional argument with single quotes
    {
        std::cout << "Test 1: Positional with single quotes\n";
        std::string response = "[search('machine learning')]";
        std::string regular_response;
        std::vector<std::string> function_calls;

        parse_function_calls_from_response(response, regular_response, function_calls, FunctionCallFormat::BFCL);

        std::cout << "Input: " << response << "\n";
        std::cout << "Parsed: " << function_calls[0] << "\n";

        assert(function_calls.size() == 1);
        assert(function_calls[0].find("\"name\":\"search\"") != std::string::npos);
        assert(function_calls[0].find("\"arg0\":'machine learning'") != std::string::npos);
        std::cout << "✓ PASSED\n\n";
    }

    // Test 2: Positional argument with double quotes
    {
        std::cout << "Test 2: Positional with double quotes\n";
        std::string response = "[search(\"machine learning\")]";
        std::string regular_response;
        std::vector<std::string> function_calls;

        parse_function_calls_from_response(response, regular_response, function_calls, FunctionCallFormat::BFCL);

        std::cout << "Input: " << response << "\n";
        std::cout << "Parsed: " << function_calls[0] << "\n";

        assert(function_calls.size() == 1);
        assert(function_calls[0].find("\"name\":\"search\"") != std::string::npos);
        assert(function_calls[0].find("\"arg0\":\"machine learning\"") != std::string::npos);
        std::cout << "✓ PASSED\n\n";
    }

    // Test 3: Multiple positional arguments
    {
        std::cout << "Test 3: Multiple positional arguments\n";
        std::string response = "[add(5, 10)]";
        std::string regular_response;
        std::vector<std::string> function_calls;

        parse_function_calls_from_response(response, regular_response, function_calls, FunctionCallFormat::BFCL);

        std::cout << "Input: " << response << "\n";
        std::cout << "Parsed: " << function_calls[0] << "\n";

        assert(function_calls.size() == 1);
        assert(function_calls[0].find("\"name\":\"add\"") != std::string::npos);
        assert(function_calls[0].find("\"arg0\":5") != std::string::npos);
        assert(function_calls[0].find("\"arg1\":10") != std::string::npos);
        std::cout << "✓ PASSED\n\n";
    }

    // Test 4: Named parameters (backward compatibility)
    {
        std::cout << "Test 4: Named parameters (backward compatibility)\n";
        std::string response = "[get_weather(location='NYC', unit='celsius')]";
        std::string regular_response;
        std::vector<std::string> function_calls;

        parse_function_calls_from_response(response, regular_response, function_calls, FunctionCallFormat::BFCL);

        std::cout << "Input: " << response << "\n";
        std::cout << "Parsed: " << function_calls[0] << "\n";

        assert(function_calls.size() == 1);
        assert(function_calls[0].find("\"location\":'NYC'") != std::string::npos);
        assert(function_calls[0].find("\"unit\":'celsius'") != std::string::npos);
        std::cout << "✓ PASSED\n\n";
    }
}

void test_qwen_format() {
    std::cout << "Qwen Format Tests\n";
    std::cout << "==================\n\n";

    // Test: Qwen JSON format
    {
        std::cout << "Test 1: Qwen JSON format\n";
        std::string response = R"({"function_call": {"name": "get_weather", "arguments": {"location": "Paris"}}})";
        std::string regular_response;
        std::vector<std::string> function_calls;

        parse_function_calls_from_response(response, regular_response, function_calls, FunctionCallFormat::QWEN);

        std::cout << "Input: " << response << "\n";
        std::cout << "Function calls found: " << function_calls.size() << "\n";
        if (!function_calls.empty()) {
            std::cout << "Parsed: " << function_calls[0] << "\n";
        }

        assert(function_calls.size() == 1);
        std::cout << "✓ PASSED\n\n";
    }
}

int main() {
    std::cout << "BFCL Parser Test Suite\n";
    std::cout << "======================\n\n";

    try {
        test_named_parameters();
        test_positional_arguments();
        test_qwen_format();

        std::cout << "========================================\n";
        std::cout << "All parser tests passed! ✓ (9/9)\n";
        std::cout << "========================================\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << "\n";
        return 1;
    }
}

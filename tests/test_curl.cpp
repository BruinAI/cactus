#include "test_utils.h"

#include <iostream>
#include <string>

#if __has_include(<curl/curl.h>)
#include <curl/curl.h>
#define CACTUS_TEST_HAS_CURL 1
#else
#define CACTUS_TEST_HAS_CURL 0
#endif

bool test_curl_version_info() {
#if !CACTUS_TEST_HAS_CURL
    return false;
#else
    curl_version_info_data* info = curl_version_info(CURLVERSION_NOW);
    if (!info) return false;
    if (!info->version || std::string(info->version).empty()) return false;
    if (!info->host || std::string(info->host).empty()) return false;
    return true;
#endif
}

bool test_curl_easy_init() {
#if !CACTUS_TEST_HAS_CURL
    return false;
#else
    CURL* handle = curl_easy_init();
    if (!handle) return false;
    curl_easy_cleanup(handle);
    return true;
#endif
}

bool test_curl_url_api() {
#if !CACTUS_TEST_HAS_CURL
    return false;
#else
    CURL* handle = curl_easy_init();
    if (!handle) return false;

    bool ok = true;
    ok = ok && (curl_easy_setopt(handle, CURLOPT_URL, "https://example.com/api/v1/ping?x=1") == CURLE_OK);
    ok = ok && (curl_easy_setopt(handle, CURLOPT_NOBODY, 1L) == CURLE_OK);
    ok = ok && (curl_easy_setopt(handle, CURLOPT_TIMEOUT_MS, 200L) == CURLE_OK);
    ok = ok && (curl_easy_setopt(handle, CURLOPT_CONNECTTIMEOUT_MS, 200L) == CURLE_OK);

    curl_easy_cleanup(handle);
    return ok;
#endif
}

int main() {
    TestUtils::TestRunner runner("Curl Tests");

#if !CACTUS_TEST_HAS_CURL
    runner.log_skip("curl_headers", "curl/curl.h not available in include path");
    runner.print_summary();
    return 0;
#else
    CURLcode init_rc = curl_global_init(CURL_GLOBAL_DEFAULT);
    if (init_rc != CURLE_OK) {
        runner.run_test("global_init", false);
        runner.print_summary();
        return 1;
    }

    runner.run_test("version_info", test_curl_version_info());
    runner.run_test("easy_init", test_curl_easy_init());
    runner.run_test("url_api", test_curl_url_api());

    curl_global_cleanup();
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
#endif
}

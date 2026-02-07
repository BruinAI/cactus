#include "telemetry/telemetry.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <cstdlib>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <curl/curl.h>
#include <dirent.h>
#include <functional>

namespace cactus {
namespace telemetry {

enum EventType { INIT = 0, COMPLETION = 1, EMBEDDING = 2, TRANSCRIPTION = 3 };

struct Event {
    EventType type;
    char model[128];
    bool success;
    double ttft_ms;
    double tps;
    double response_time_ms;
    int tokens;
    char message[256];
    std::chrono::system_clock::time_point timestamp;
};
static std::atomic<bool> enabled{false};
static std::atomic<int> inference_active{0};
static std::atomic<bool> shutdown_called{false};
static std::atomic<bool> atexit_registered{false};
static std::atomic<bool> cloud_disabled{false};
static const char* supabase_url = "https://ivzeouvbwsnepwojsjya.supabase.co";
static const char* supabase_key = "sb_publishable_hgsggXPMkAsEuyhQE_bwuQ_ouLN3rcc";
static std::string device_id;
static std::string project_id;
static std::string cloud_key;
static std::string device_model;
static std::string device_os;
static std::string device_os_version;
static std::string device_brand;
static std::atomic<bool> ids_ready{false};

// Forward declarations for helpers used before definition
static std::string event_type_to_string(EventType t);
static bool event_type_from_string(const std::string& s, EventType& t);
static bool extract_string_field(const std::string& line, const std::string& key, std::string& out);
static bool extract_bool_field(const std::string& line, const std::string& key, bool& out);
static bool extract_double_field(const std::string& line, const std::string& key, double& out);
static bool extract_int_field(const std::string& line, const std::string& key, int& out);

static std::string get_telemetry_dir() {
    const char* home = getenv("HOME");
    if (!home) home = "/tmp";
    std::string dir = std::string(home) + "/Library/Caches/cactus/telemetry";
    mkdir((std::string(home) + "/Library").c_str(), 0755);
    mkdir((std::string(home) + "/Library/Caches").c_str(), 0755);
    mkdir((std::string(home) + "/Library/Caches/cactus").c_str(), 0755);
    mkdir(dir.c_str(), 0755);
    return dir;
}

static std::string scoped_file_name(const std::string& prefix, const std::string& scope) {
    std::hash<std::string> hasher;
    size_t h = hasher(scope);
    std::ostringstream oss;
    oss << prefix << std::hex << h;
    return oss.str();
}

static std::string load_or_create_id(const std::string& file) {
    std::ifstream in(file);
    if (in.is_open()) {
        std::string line;
        if (std::getline(in, line) && !line.empty()) {
            return line;
        }
    }
    std::string id = new_uuid();
    std::ofstream out(file, std::ios::trunc);
    if (out.is_open()) {
        out << id;
    }
    return id;
}

static Event make_event(EventType type, const char* model, bool success, double ttft_ms, double tps, double response_time_ms, int tokens, const char* message) {
    Event e;
    e.type = type;
    e.success = success;
    e.ttft_ms = ttft_ms;
    e.tps = tps;
    e.response_time_ms = response_time_ms;
    e.tokens = tokens;
    e.timestamp = std::chrono::system_clock::now();
    std::memset(e.model, 0, sizeof(e.model));
    std::memset(e.message, 0, sizeof(e.message));
    if (model) std::strncpy(e.model, model, sizeof(e.model)-1);
    if (message) std::strncpy(e.message, message, sizeof(e.message)-1);
    return e;
}

static bool parse_event_line(const std::string& line, Event& out) {
    std::string type_str;
    if (!extract_string_field(line, "event_type", type_str)) return false;
    EventType et;
    if (!event_type_from_string(type_str, et)) return false;

    std::string model;
    extract_string_field(line, "model", model);
    bool success = false;
    extract_bool_field(line, "success", success);
    double ttft = 0.0;
    extract_double_field(line, "ttft", ttft);
    double tps = 0.0;
    extract_double_field(line, "tps", tps);
    double response_time = 0.0;
    extract_double_field(line, "response_time", response_time);
    int tokens = 0;
    extract_int_field(line, "tokens", tokens);
    std::string message;
    extract_string_field(line, "message", message);

    out = make_event(et,
                     model.empty() ? nullptr : model.c_str(),
                     success,
                     ttft,
                     tps,
                     response_time,
                     tokens,
                     message.empty() ? nullptr : message.c_str());
    return true;
}

static std::string event_type_to_string(EventType t) {
    switch (t) {
        case INIT: return "init";
        case COMPLETION: return "completion";
        case EMBEDDING: return "embedding";
        case TRANSCRIPTION: return "transcription";
        default: return "unknown";
    }
}

static bool event_type_from_string(const std::string& s, EventType& t) {
    if (s == "init") { t = INIT; return true; }
    if (s == "completion") { t = COMPLETION; return true; }
    if (s == "embedding") { t = EMBEDDING; return true; }
    if (s == "transcription") { t = TRANSCRIPTION; return true; }
    return false;
}

static bool extract_string_field(const std::string& line, const std::string& key, std::string& out) {
    std::string needle = "\"" + key + "\":";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) return false;
    pos += needle.size();
    while (pos < line.size() && line[pos] == ' ') pos++;
    if (pos >= line.size()) return false;
    if (line[pos] == '"') {
        pos++;
        size_t end = line.find('"', pos);
        if (end == std::string::npos) return false;
        out = line.substr(pos, end - pos);
        return true;
    }
    size_t end = line.find_first_of(",}", pos);
    if (end == std::string::npos) end = line.size();
    out = line.substr(pos, end - pos);
    return true;
}

static bool extract_bool_field(const std::string& line, const std::string& key, bool& out) {
    std::string raw;
    if (!extract_string_field(line, key, raw)) return false;
    if (raw == "true") { out = true; return true; }
    if (raw == "false") { out = false; return true; }
    return false;
}

static bool extract_double_field(const std::string& line, const std::string& key, double& out) {
    std::string raw;
    if (!extract_string_field(line, key, raw)) return false;
    try {
        out = std::stod(raw);
        return true;
    } catch (...) {
        return false;
    }
}

static bool extract_int_field(const std::string& line, const std::string& key, int& out) {
    std::string raw;
    if (!extract_string_field(line, key, raw)) return false;
    try {
        out = std::stoi(raw);
        return true;
    } catch (...) {
        return false;
    }
}

static std::string new_uuid() {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    uint64_t a = rng();
    uint64_t b = rng();
    a = (a & 0xffffffffffff0fffULL) | 0x0000000000004000ULL;
    b = (b & 0x3fffffffffffffffULL) | 0x8000000000000000ULL;
    std::ostringstream oss;
    oss << std::hex;
    oss << ((a >> 32) & 0xffffffffULL);
    oss << "-" << ((a >> 16) & 0xffffULL);
    oss << "-" << (a & 0xffffULL);
    oss << "-" << ((b >> 48) & 0xffffULL);
    oss << "-" << (b & 0xffffffffffffULL);
    return oss.str();
}

static bool is_valid_uuid(const std::string& s) {
    if (s.size() != 36) return false;
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (i == 8 || i == 13 || i == 18 || i == 23) {
            if (c != '-') return false;
        } else {
            bool hex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
            if (!hex) return false;
        }
    }
    return true;
}

static void collect_device_info() {
    struct utsname u;
    if (uname(&u) == 0) {
        device_os = u.sysname;
        device_os_version = u.release;
        device_model = u.machine;
        device_brand = "apple";
    }
}

static void ensure_device_row(CURL* curl) {
    if (device_id.empty()) device_id = new_uuid();
    if (device_os.empty()) collect_device_info();
    std::string url = std::string(supabase_url) + "/rest/v1/devices";
    std::ostringstream payload;
    payload << "[{";
    payload << "\"id\":\"" << device_id << "\"";
    payload << ",\"device_id\":\"" << device_id << "\"";
    if (!device_model.empty()) payload << ",\"model\":\"" << device_model << "\"";
    if (!device_os.empty()) payload << ",\"os\":\"" << device_os << "\"";
    if (!device_os_version.empty()) payload << ",\"os_version\":\"" << device_os_version << "\"";
    if (!device_brand.empty()) payload << ",\"brand\":\"" << device_brand << "\"";
    payload << "}]";
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    std::string apikey_hdr = std::string("apikey: ") + supabase_key;
    std::string auth_hdr = std::string("Authorization: Bearer ") + supabase_key;
    headers = curl_slist_append(headers, apikey_hdr.c_str());
    headers = curl_slist_append(headers, auth_hdr.c_str());
    headers = curl_slist_append(headers, "Prefer: resolution=merge-duplicates");
    headers = curl_slist_append(headers, "Content-Profile: cactus");
    headers = curl_slist_append(headers, "Accept-Profile: cactus");
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    std::string body = payload.str();
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, body.size());
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);
    curl_easy_perform(curl);
    if (headers) curl_slist_free_all(headers);
}

static bool send_payload(CURL* curl, const std::string& url, const std::string& body) {
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    std::string apikey_hdr = std::string("apikey: ") + supabase_key;
    std::string auth_hdr = std::string("Authorization: Bearer ") + supabase_key;
    headers = curl_slist_append(headers, apikey_hdr.c_str());
    headers = curl_slist_append(headers, auth_hdr.c_str());
    headers = curl_slist_append(headers, "Prefer: return=minimal");
    headers = curl_slist_append(headers, "Content-Profile: cactus");
    headers = curl_slist_append(headers, "Accept-Profile: cactus");
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, body.size());
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);
    CURLcode res = curl_easy_perform(curl);
    long code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
    if (headers) curl_slist_free_all(headers);
    return (res == CURLE_OK) && (code >= 200 && code < 300);
}

static bool send_batch_to_cloud(const std::vector<Event>& local) {
    if (!enabled.load()) return false;
    if (local.empty()) return true;
    if (cloud_disabled.load()) return false;
    if (device_id.empty()) device_id = new_uuid();
    CURL* curl = curl_easy_init();
    if (!curl) return false;
    ensure_device_row(curl);
    std::string url = std::string(supabase_url) + "/rest/v1/logs";
    std::ostringstream payload;
    payload << "[";
    for (size_t i = 0; i < local.size(); ++i) {
        const Event& e = local[i];
        payload << "{";
        payload << "\"event_type\":\"" << event_type_to_string(e.type) << "\",";
        payload << "\"model\":\"" << e.model << "\",";
        payload << "\"success\":" << (e.success ? "true" : "false") << ",";
        payload << "\"ttft\":" << e.ttft_ms << ",";
        payload << "\"tps\":" << e.tps << ",";
        payload << "\"response_time\":" << e.response_time_ms << ",";
        payload << "\"tokens\":" << e.tokens << ",";
        if (!project_id.empty()) {
            payload << "\"project_id\":\"" << project_id << "\",";
        }
        if (!cloud_key.empty()) {
            payload << "\"cloud_key\":\"" << cloud_key << "\",";
        }
        payload << "\"framework\":\"cpp\",";
        payload << "\"device_id\":\"" << device_id << "\"";
        if (e.message[0] != '\0') {
            payload << ",\"message\":\"" << e.message << "\"";
        }
        payload << "}";
        if (i + 1 < local.size()) payload << ",";
    }
    payload << "]";
    bool ok = send_payload(curl, url, payload.str());
    curl_easy_cleanup(curl);
    return ok;
}

static void write_events_to_cache(const std::vector<Event>& local) {
    std::string dir = get_telemetry_dir();
    for (const auto &e : local) {
        std::ostringstream oss;
        oss << "{\"event_type\":\"" << event_type_to_string(e.type) << "\",";
        oss << "\"model\":\"" << e.model << "\",";
        oss << "\"success\":" << (e.success ? "true" : "false") << ",";
        oss << "\"ttft\":" << e.ttft_ms << ",";
        oss << "\"tps\":" << e.tps << ",";
        oss << "\"response_time\":" << e.response_time_ms << ",";
        oss << "\"tokens\":" << e.tokens;
        if (e.message[0] != '\0') {
            oss << ",\"message\":\"" << e.message << "\"";
        }
        oss << "}";
        std::string file = dir + "/" + event_type_to_string(e.type) + ".log";
        std::ofstream out(file, std::ios::app);
        if (out.is_open()) {
            out << oss.str() << "\n";
            out.close();
        }
    }
}
static std::vector<Event> load_cached_events() {
    std::vector<Event> events;
    std::string dir = get_telemetry_dir();
    DIR* d = opendir(dir.c_str());
    if (!d) return events;
    struct dirent* ent;
    while ((ent = readdir(d)) != nullptr) {
        std::string name = ent->d_name;
        if (name.size() < 5 || name.substr(name.size() - 4) != ".log") continue;
        std::ifstream in(dir + "/" + name);
        std::string line;
        while (std::getline(in, line)) {
            Event e;
            if (parse_event_line(line, e)) {
                events.push_back(e);
            }
        }
    }
    closedir(d);
    return events;
}

static void clear_cache_files() {
    std::string dir = get_telemetry_dir();
    DIR* d = opendir(dir.c_str());
    if (!d) return;
    struct dirent* ent;
    while ((ent = readdir(d)) != nullptr) {
        std::string name = ent->d_name;
        if (name.size() < 5 || name.substr(name.size() - 4) != ".log") continue;
        std::remove((dir + "/" + name).c_str());
    }
    closedir(d);
}

static void flush_logs_with_event(const Event* latest) {
    std::vector<Event> events = load_cached_events();
    if (latest) events.push_back(*latest);
    if (events.empty()) return;
    bool cloud_ok = send_batch_to_cloud(events);
    if (cloud_ok) {
        clear_cache_files();
    } else if (latest) {
        write_events_to_cache({*latest});
    }
}

void init(const char* project_id_param, const char* project_scope, const char* cloud_key_param) {
    std::string scope = project_scope ? project_scope : "default";
    if (const char* env = std::getenv("CACTUS_NO_CLOUD_TELE")) {
        if (env[0] != '\0' && !(env[0] == '0' && env[1] == '\0')) {
            cloud_disabled.store(true);
        }
    }
    const char* env_project = std::getenv("CACTUS_PROJECT_ID");
    if (project_id_param && *project_id_param) {
        project_id = project_id_param;
    } else if (env_project && *env_project) {
        project_id = env_project;
    } else {
        std::string dir = get_telemetry_dir();
        std::string file = dir + "/" + scoped_file_name("project_", scope);
        project_id = load_or_create_id(file);
    }

    const char* env_cloud = std::getenv("CACTUS_CLOUD_KEY");
    if (cloud_key_param && *cloud_key_param) {
        cloud_key = cloud_key_param;
    } else if (env_cloud && *env_cloud) {
        cloud_key = env_cloud;
    }

    std::string dir = get_telemetry_dir();
    std::string device_file = dir + "/device_id";
    device_id = load_or_create_id(device_file);
    collect_device_info();

    curl_global_init(CURL_GLOBAL_DEFAULT);
    if (!atexit_registered.exchange(true)) {
        std::atexit([](){ shutdown(); });
    }
    ids_ready.store(true);
    setEnabled(true);
}

void setEnabled(bool en) {
    enabled.store(en);
}

void setCloudDisabled(bool disabled) {
    cloud_disabled.store(disabled);
}

void recordInit(const char* model, bool success, const char* message) {
    if (!enabled.load() || !ids_ready.load() || cloud_disabled.load()) return;
    Event e = make_event(INIT, model, success, 0.0, 0.0, 0.0, 0, message);
    if (success) {
        write_events_to_cache({e});
    } else {
        flush_logs_with_event(&e);
    }
}

void recordCompletion(const char* model, bool success, double ttft_ms, double tps, double response_time_ms, int tokens, const char* message) {
    if (!enabled.load() || !ids_ready.load() || cloud_disabled.load()) return;
    Event e = make_event(COMPLETION, model, success, ttft_ms, tps, response_time_ms, tokens, message);
    flush_logs_with_event(&e);
}

void recordEmbedding(const char* model, bool success, const char* message) {
    if (!enabled.load() || !ids_ready.load() || cloud_disabled.load()) return;
    Event e = make_event(EMBEDDING, model, success, 0.0, 0.0, 0.0, 0, message);
    flush_logs_with_event(&e);
}

void recordTranscription(const char* model, bool success, double ttft_ms, double tps, double response_time_ms, int tokens, const char* message) {
    if (!enabled.load() || !ids_ready.load() || cloud_disabled.load()) return;
    Event e = make_event(TRANSCRIPTION, model, success, ttft_ms, tps, response_time_ms, tokens, message);
    flush_logs_with_event(&e);
}

void markInference(bool active) {
    if (active) inference_active.fetch_add(1);
    else {
        int v = inference_active.load();
        if (v > 0) inference_active.fetch_sub(1);
    }
}

void flush() {
    flush_logs_with_event(nullptr);
}

void shutdown() {
    if (!shutdown_called.exchange(true)) {
        flush();
    }
    curl_global_cleanup();
}

} // namespace telemetry
} // namespace cactus

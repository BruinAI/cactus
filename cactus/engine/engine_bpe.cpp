#include "engine.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <cwctype>
#include <locale>
#include <stdexcept>
#include <set>

namespace cactus {
namespace engine {

namespace {

struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object };
    using Array = std::vector<JsonValue>;
    using Object = std::unordered_map<std::string, JsonValue>;

    Type type = Type::Null;
    double number = 0.0;
    bool boolean = false;
    std::string string;
    Array array;
    Object object;

    bool is_null() const { return type == Type::Null; }
    bool is_bool() const { return type == Type::Bool; }
    bool is_number() const { return type == Type::Number; }
    bool is_string() const { return type == Type::String; }
    bool is_array() const { return type == Type::Array; }
    bool is_object() const { return type == Type::Object; }

    const Array* get_array_ptr() const { return is_array() ? &array : nullptr; }
    const Object* get_object_ptr() const { return is_object() ? &object : nullptr; }

    const JsonValue* find(const std::string& key) const {
        if (!is_object()) return nullptr;
        auto it = object.find(key);
        if (it == object.end()) return nullptr;
        return &it->second;
    }

    double get_number(double default_value = 0.0) const { return is_number() ? number : default_value; }
    bool get_bool(bool default_value = false) const { return is_bool() ? boolean : default_value; }
    std::string get_string(const std::string& default_value = "") const { return is_string() ? string : default_value; }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& text) : text_(text) {}

    JsonValue parse() {
        skip_ws();
        JsonValue value = parse_value();
        skip_ws();
        return value;
    }

private:
    const std::string& text_;
    size_t pos_ = 0;

    bool eof() const { return pos_ >= text_.size(); }

    char peek() const {
        if (eof()) throw std::runtime_error("Unexpected end of JSON input");
        return text_[pos_];
    }

    char get() {
        if (eof()) throw std::runtime_error("Unexpected end of JSON input");
        return text_[pos_++];
    }

    void skip_ws() {
        while (!eof()) {
            char c = text_[pos_];
            if (c == ' ' || c == '\n' || c == '\t' || c == '\r') {
                ++pos_;
            } else {
                break;
            }
        }
    }

    void expect(char expected) {
        char c = get();
        if (c != expected) {
            throw std::runtime_error("JSON parse error: expected different character");
        }
    }

    void append_utf8(std::string& out, uint32_t codepoint) {
        if (codepoint <= 0x7F) {
            out.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7FF) {
            out.push_back(static_cast<char>(0xC0 | ((codepoint >> 6) & 0x1F)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        } else if (codepoint <= 0xFFFF) {
            out.push_back(static_cast<char>(0xE0 | ((codepoint >> 12) & 0x0F)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | ((codepoint >> 18) & 0x07)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        }
    }

    uint32_t parse_unicode_escape() {
        uint32_t code = 0;
        for (int i = 0; i < 4; ++i) {
            char c = get();
            code <<= 4;
            if (c >= '0' && c <= '9') code |= static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') code |= static_cast<uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') code |= static_cast<uint32_t>(c - 'A' + 10);
            else throw std::runtime_error("Invalid unicode escape in JSON string");
        }

        if (code >= 0xD800 && code <= 0xDBFF) {
            // High surrogate, expect a following low surrogate
            if (pos_ + 2 <= text_.size() && text_[pos_] == '\\' && text_[pos_ + 1] == 'u') {
                pos_ += 2;
                uint32_t low = parse_unicode_escape();
                if (low >= 0xDC00 && low <= 0xDFFF) {
                    uint32_t high = code;
                    code = 0x10000 + (((high - 0xD800) << 10) | (low - 0xDC00));
                } else {
                    throw std::runtime_error("Invalid surrogate pair in JSON string");
                }
            } else {
                throw std::runtime_error("Expected low surrogate after high surrogate in JSON string");
            }
        }

        return code;
    }

    JsonValue parse_string() {
        expect('"');
        std::string out;
        while (true) {
            char c = get();
            if (c == '"') break;
            if (c == '\\') {
                char esc = get();
                switch (esc) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        uint32_t code = parse_unicode_escape();
                        append_utf8(out, code);
                        break;
                    }
                    default:
                        throw std::runtime_error("Unknown escape sequence in JSON string");
                }
            } else {
                out.push_back(c);
            }
        }
        JsonValue value;
        value.type = JsonValue::Type::String;
        value.string = std::move(out);
        return value;
    }

    JsonValue parse_number() {
        size_t start = pos_;
        if (!eof() && (text_[pos_] == '-' || text_[pos_] == '+')) ++pos_;
        while (!eof() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
        if (!eof() && text_[pos_] == '.') {
            ++pos_;
            while (!eof() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
        }
        if (!eof() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            ++pos_;
            if (!eof() && (text_[pos_] == '-' || text_[pos_] == '+')) ++pos_;
            while (!eof() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
        }
        std::string token = text_.substr(start, pos_ - start);
        JsonValue value;
        value.type = JsonValue::Type::Number;
        value.number = std::stod(token);
        return value;
    }

    JsonValue parse_true() {
        if (text_.substr(pos_, 4) != "true") throw std::runtime_error("Invalid JSON literal");
        pos_ += 4;
        JsonValue value;
        value.type = JsonValue::Type::Bool;
        value.boolean = true;
        return value;
    }

    JsonValue parse_false() {
        if (text_.substr(pos_, 5) != "false") throw std::runtime_error("Invalid JSON literal");
        pos_ += 5;
        JsonValue value;
        value.type = JsonValue::Type::Bool;
        value.boolean = false;
        return value;
    }

    JsonValue parse_null() {
        if (text_.substr(pos_, 4) != "null") throw std::runtime_error("Invalid JSON literal");
        pos_ += 4;
        JsonValue value;
        value.type = JsonValue::Type::Null;
        return value;
    }

    JsonValue parse_array() {
        expect('[');
        JsonValue value;
        value.type = JsonValue::Type::Array;
        skip_ws();
        if (!eof() && text_[pos_] == ']') {
            ++pos_;
            return value;
        }
        while (true) {
            skip_ws();
            value.array.emplace_back(parse_value());
            skip_ws();
            if (!eof() && text_[pos_] == ',') {
                ++pos_;
                continue;
            }
            expect(']');
            break;
        }
        return value;
    }

    JsonValue parse_object() {
        expect('{');
        JsonValue value;
        value.type = JsonValue::Type::Object;
        skip_ws();
        if (!eof() && text_[pos_] == '}') {
            ++pos_;
            return value;
        }
        while (true) {
            skip_ws();
            JsonValue key = parse_string();
            skip_ws();
            expect(':');
            skip_ws();
            JsonValue val = parse_value();
            value.object.emplace(std::move(key.string), std::move(val));
            skip_ws();
            if (!eof() && text_[pos_] == ',') {
                ++pos_;
                continue;
            }
            expect('}');
            break;
        }
        return value;
    }

    JsonValue parse_value() {
        if (eof()) throw std::runtime_error("Unexpected end of JSON input");
        char c = peek();
        if (c == '"') return parse_string();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == 't') return parse_true();
        if (c == 'f') return parse_false();
        if (c == 'n') return parse_null();
        return parse_number();
    }
};

struct CodepointSpan {
    uint32_t cp;
    size_t start;
    size_t end;
};

std::vector<CodepointSpan> decode_utf8_with_spans(const std::string& text) {
    std::vector<CodepointSpan> spans;
    size_t i = 0;
    while (i < text.size()) {
        unsigned char byte = static_cast<unsigned char>(text[i]);
        size_t length = 1;
        uint32_t codepoint = 0xFFFD;

        if ((byte & 0x80) == 0) {
            codepoint = byte;
            length = 1;
        } else if ((byte & 0xE0) == 0xC0 && i + 1 < text.size()) {
            codepoint = ((byte & 0x1F) << 6) |
                        (static_cast<unsigned char>(text[i + 1]) & 0x3F);
            length = 2;
        } else if ((byte & 0xF0) == 0xE0 && i + 2 < text.size()) {
            codepoint = ((byte & 0x0F) << 12) |
                        ((static_cast<unsigned char>(text[i + 1]) & 0x3F) << 6) |
                        (static_cast<unsigned char>(text[i + 2]) & 0x3F);
            length = 3;
        } else if ((byte & 0xF8) == 0xF0 && i + 3 < text.size()) {
            codepoint = ((byte & 0x07) << 18) |
                        ((static_cast<unsigned char>(text[i + 1]) & 0x3F) << 12) |
                        ((static_cast<unsigned char>(text[i + 2]) & 0x3F) << 6) |
                        (static_cast<unsigned char>(text[i + 3]) & 0x3F);
            length = 4;
        } else {
            length = 1;
            codepoint = 0xFFFD;
        }

        spans.push_back({codepoint, i, std::min(text.size(), i + length)});
        i += length;
    }
    return spans;
}

} // namespace

BPETokenizer::BPETokenizer()
    : vocab_size_(0), unk_token_id_(0), bos_token_id_(1), eos_token_id_(2),
      vocab_mmap_ptr_(nullptr), vocab_mmap_size_(0),
      merges_mmap_ptr_(nullptr), merges_mmap_size_(0) {
    has_chat_template_ = false;
}

BPETokenizer::~BPETokenizer() {
    cleanup_mmap();
}

void BPETokenizer::cleanup_mmap() {
    if (vocab_mmap_ptr_ && vocab_mmap_ptr_ != MAP_FAILED) {
        munmap(vocab_mmap_ptr_, vocab_mmap_size_);
        vocab_mmap_ptr_ = nullptr;
    }
    if (merges_mmap_ptr_ && merges_mmap_ptr_ != MAP_FAILED) {
        munmap(merges_mmap_ptr_, merges_mmap_size_);
        merges_mmap_ptr_ = nullptr;
    }
}

bool BPETokenizer::load_vocabulary_mmap(const std::string& vocab_file, const std::string& merges_file) {
    int vocab_fd = open(vocab_file.c_str(), O_RDONLY);
    if (vocab_fd == -1) return false;

    struct stat vocab_stat;
    if (fstat(vocab_fd, &vocab_stat) == -1) {
        close(vocab_fd);
        return false;
    }

    vocab_mmap_size_ = vocab_stat.st_size;
    vocab_mmap_ptr_ = mmap(nullptr, vocab_mmap_size_, PROT_READ, MAP_PRIVATE, vocab_fd, 0);
    close(vocab_fd);

    if (vocab_mmap_ptr_ == MAP_FAILED) return false;

    std::string vocab_content(static_cast<char*>(vocab_mmap_ptr_), vocab_mmap_size_);
    std::istringstream vocab_stream(vocab_content);

    std::string line;
    uint32_t id = 0;
    token_to_id_.clear();
    id_to_token_.clear();

    while (std::getline(vocab_stream, line)) {
        if (line.empty()) continue;
        token_to_id_[line] = id;
        id_to_token_.push_back(line);
        id++;
    }
    vocab_size_ = id;

    int merges_fd = open(merges_file.c_str(), O_RDONLY);
    if (merges_fd == -1) return false;

    struct stat merges_stat;
    if (fstat(merges_fd, &merges_stat) == -1) {
        close(merges_fd);
        return false;
    }

    merges_mmap_size_ = merges_stat.st_size;
    merges_mmap_ptr_ = mmap(nullptr, merges_mmap_size_, PROT_READ, MAP_PRIVATE, merges_fd, 0);
    close(merges_fd);

    if (merges_mmap_ptr_ == MAP_FAILED) return false;

    std::string merges_content(static_cast<char*>(merges_mmap_ptr_), merges_mmap_size_);
    std::istringstream merges_stream(merges_content);

    merge_rules_.clear();
    uint32_t priority = 0;

    while (std::getline(merges_stream, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string first, second;
        if (iss >> first >> second) {
            std::string merged = first + second;
            merge_rules_.emplace_back(first, second, merged, priority);

            std::string key = first + "\x00" + second;
            auto it = merge_map_.find(key);
            if (it == merge_map_.end() || priority < it->second) {
                merge_map_[key] = priority;
            }
            priority++;
        }
    }

    std::sort(merge_rules_.begin(), merge_rules_.end(),
              [](const MergeRule& a, const MergeRule& b) {
                  return a.priority < b.priority;
              });

    return true;

        bool is_letter_cp(uint32_t cp) {
            return std::iswalpha(static_cast<wint_t>(cp)) != 0;
        }

        bool is_digit_cp(uint32_t cp) {
            return std::iswdigit(static_cast<wint_t>(cp)) != 0;
        }

        bool is_whitespace_cp(uint32_t cp) {
            return std::iswspace(static_cast<wint_t>(cp)) != 0;
        }

        bool is_other_cp(uint32_t cp) {
            return !is_whitespace_cp(cp) && !is_letter_cp(cp) && !is_digit_cp(cp);
        }

        bool match_apostrophe_sequence(const std::vector<CodepointSpan>& spans, size_t index, size_t& length) {
            static const uint32_t patterns[][3] = {
                {0x27, 0x73, 0}, // 's
                {0x27, 0x74, 0}, // 't
                {0x27, 0x72, 0x65}, // 're
                {0x27, 0x76, 0x65}, // 've
                {0x27, 0x6d, 0}, // 'm
                {0x27, 0x6c, 0x6c}, // 'll
                {0x27, 0x64, 0}  // 'd
            };

            static const size_t pattern_lengths[] = {2, 2, 3, 3, 2, 3, 2};

            for (size_t p = 0; p < std::size(pattern_lengths); ++p) {
                size_t len = pattern_lengths[p];
                if (index + len > spans.size()) continue;
                bool matches = true;
                for (size_t i = 0; i < len; ++i) {
                    if (spans[index + i].cp != patterns[p][i]) {
                        matches = false;
                        break;
                    }
                }
                if (matches) {
                    length = len;
                    return true;
                }
            }
            return false;
        }

        bool match_letter_sequence(const std::vector<CodepointSpan>& spans, size_t index, size_t& length) {
            size_t token_start = index;
            size_t pos = index;

            if (spans[pos].cp == 0x20 && pos + 1 < spans.size() && is_letter_cp(spans[pos + 1].cp)) {
                pos += 1;
            }

            if (!is_letter_cp(spans[pos].cp)) {
                return false;
            }

            size_t end = pos + 1;
            while (end < spans.size() && is_letter_cp(spans[end].cp)) {
                ++end;
            }

            length = end - token_start;
            return true;
        }

        bool match_digit_sequence(const std::vector<CodepointSpan>& spans, size_t index, size_t& length) {
            size_t token_start = index;
            size_t pos = index;

            if (spans[pos].cp == 0x20 && pos + 1 < spans.size() && is_digit_cp(spans[pos + 1].cp)) {
                pos += 1;
            }

            if (!is_digit_cp(spans[pos].cp)) {
                return false;
            }

            size_t end = pos + 1;
            while (end < spans.size() && is_digit_cp(spans[end].cp)) {
                ++end;
            }

            length = end - token_start;
            return true;
        }

        bool match_other_sequence(const std::vector<CodepointSpan>& spans, size_t index, size_t& length) {
            size_t token_start = index;
            size_t pos = index;

            if (spans[pos].cp == 0x20 && pos + 1 < spans.size() && is_other_cp(spans[pos + 1].cp)) {
                pos += 1;
            }

            if (!is_other_cp(spans[pos].cp)) {
                return false;
            }

            size_t end = pos + 1;
            while (end < spans.size() && is_other_cp(spans[end].cp)) {
                ++end;
            }

            length = end - token_start;
            return true;
        }

        bool match_whitespace_sequence(const std::vector<CodepointSpan>& spans, size_t index, size_t& length) {
            if (!is_whitespace_cp(spans[index].cp)) {
                return false;
            }

            size_t end = index + 1;
            while (end < spans.size() && is_whitespace_cp(spans[end].cp)) {
                ++end;
            }

            length = end - index;
            return true;
        }

}

bool BPETokenizer::load_vocabulary_with_config(const std::string& vocab_file, const std::string& merges_file, const std::string& config_file) {
    if (!load_vocabulary_mmap(vocab_file, merges_file)) {
        return false;
    }

    std::ifstream config_stream(config_file);
    if (!config_stream.is_open()) {
        return true;
    }

    std::string line;
    while (std::getline(config_stream, line)) {
        if (line.empty() || line[0] == '#') continue;

        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;

        std::string key = line.substr(0, eq_pos);
        std::string value = line.substr(eq_pos + 1);

        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);

        if (key == "eos_token_id") {
            eos_token_id_ = std::stoul(value);
        } else if (key == "pad_token_id") {
            if (unk_token_id_ == 0) {
                unk_token_id_ = std::stoul(value);
            }
        } else if (key == "unk_token_id" && value != "null") {
            unk_token_id_ = std::stoul(value);
        } else if (key == "bos_token_id" && value != "null") {
            bos_token_id_ = std::stoul(value);
        } else if (key == "vocab_size") {
            if (std::stoul(value) != vocab_size_) {
            }
        }
    }

    std::string special_tokens_path = config_file.substr(0, config_file.find_last_of("/\\")) + "/special_tokens.json";
    load_special_tokens(special_tokens_path);

    std::string template_path = config_file.substr(0, config_file.find_last_of("/\\")) + "/chat_template.jinja2";
    load_chat_template(template_path);

    std::string config_path = config_file.substr(0, config_file.find_last_of("/\\")) + "/config.txt";
    detect_model_type(config_path);

    return true;
}

void BPETokenizer::load_special_tokens(const std::string& config_file) {
    std::ifstream file(config_file);
    if (!file.is_open()) {
        return;
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    size_t pos = content.find("\"special_tokens\"");
    if (pos == std::string::npos) return;

    pos = content.find("{", pos);
    if (pos == std::string::npos) return;

    size_t end_pos = content.find("}", pos);
    if (end_pos == std::string::npos) return;

    std::string special_tokens_section = content.substr(pos + 1, end_pos - pos - 1);

    std::istringstream iss(special_tokens_section);
    std::string line;

    while (std::getline(iss, line)) {
        size_t colon_pos = line.find(":");
        if (colon_pos == std::string::npos) continue;

        std::string id_part = line.substr(0, colon_pos);
        std::string token_part = line.substr(colon_pos + 1);

        size_t id_start = id_part.find("\"");
        size_t id_end = id_part.find("\"", id_start + 1);
        if (id_start == std::string::npos || id_end == std::string::npos) continue;

        std::string id_str = id_part.substr(id_start + 1, id_end - id_start - 1);
        uint32_t token_id = std::stoul(id_str);

        size_t token_start = token_part.find("\"");
        size_t token_end = token_part.rfind("\"");
        if (token_start == std::string::npos || token_end == std::string::npos || token_start >= token_end) continue;

        std::string token_content = token_part.substr(token_start + 1, token_end - token_start - 1);

        special_tokens_[token_content] = token_id;
    }

}

std::vector<std::string> BPETokenizer::split_with_special_tokens(const std::string& text) const {
    std::vector<std::string> result;

    size_t start = 0;
    while (start < text.size()) {
        size_t best_match_pos = text.size();
        size_t best_match_len = 0;
        std::string best_special_token;

        for (const auto& [special_token, token_id] : special_tokens_) {
            size_t pos = text.find(special_token, start);
            if (pos != std::string::npos && pos < best_match_pos) {
                best_match_pos = pos;
                best_match_len = special_token.length();
                best_special_token = special_token;
            }
        }

        if (best_match_pos < text.size()) {
            if (best_match_pos > start) {
                std::string before = text.substr(start, best_match_pos - start);
                result.push_back(before);
            }

            result.push_back(best_special_token);
            start = best_match_pos + best_match_len;
        } else {
            if (start < text.size()) {
                result.push_back(text.substr(start));
            }
            break;
        }
    }

    return result;
}

void BPETokenizer::init_byte_mappings() const {
    if (!byte_to_unicode_.empty()) return;

    std::vector<int> bytes;

    for (int i = 33; i <= 126; ++i) {
        bytes.push_back(i);
    }


    for (int i = 161; i <= 255; ++i) {
        bytes.push_back(i);
    }

    std::vector<int> remaining_bytes;
    for (int i = 0; i <= 32; ++i) remaining_bytes.push_back(i);
    remaining_bytes.push_back(127);
    for (int i = 128; i <= 160; ++i) remaining_bytes.push_back(i);

    int unicode_start = 256;
    for (int byte : remaining_bytes) {
        bytes.push_back(byte);
    }

    for (size_t i = 0; i < bytes.size(); ++i) {
        uint8_t byte = static_cast<uint8_t>(bytes[i]);

        if (byte >= 33 && byte <= 126) {
            std::string unicode_char(1, static_cast<char>(byte));
            byte_to_unicode_[byte] = unicode_char;
            unicode_to_byte_[unicode_char] = byte;
        } else if (byte >= 161) {
            std::string unicode_char;
            unicode_char += static_cast<char>(0xC0 | (byte >> 6));
            unicode_char += static_cast<char>(0x80 | (byte & 0x3F));
            byte_to_unicode_[byte] = unicode_char;
            unicode_to_byte_[unicode_char] = byte;
        } else {
            int unicode_point = unicode_start++;
            std::string unicode_char;
            if (unicode_point < 0x800) {
                unicode_char += static_cast<char>(0xC0 | (unicode_point >> 6));
                unicode_char += static_cast<char>(0x80 | (unicode_point & 0x3F));
            } else {
                unicode_char += static_cast<char>(0xE0 | (unicode_point >> 12));
                unicode_char += static_cast<char>(0x80 | ((unicode_point >> 6) & 0x3F));
                unicode_char += static_cast<char>(0x80 | (unicode_point & 0x3F));
            }
            byte_to_unicode_[byte] = unicode_char;
            unicode_to_byte_[unicode_char] = byte;
        }
    }
}

std::string BPETokenizer::bytes_to_unicode(const std::string& text) const {
    init_byte_mappings();

    std::string result;
    for (uint8_t byte : text) {
        result += byte_to_unicode_.at(byte);
    }
    return result;
}

std::string BPETokenizer::unicode_to_bytes(const std::string& text) const {
    init_byte_mappings();

    std::string result;
    size_t i = 0;
    while (i < text.length()) {
        std::string unicode_char;

        if ((text[i] & 0x80) == 0) {
            unicode_char = text.substr(i, 1);
            i += 1;
        } else if ((text[i] & 0xE0) == 0xC0) {
            unicode_char = text.substr(i, 2);
            i += 2;
        } else if ((text[i] & 0xF0) == 0xE0) {
            unicode_char = text.substr(i, 3);
            i += 3;
        } else {
            unicode_char = text.substr(i, 1);
            i += 1;
        }

        auto it = unicode_to_byte_.find(unicode_char);
        if (it != unicode_to_byte_.end()) {
            result += static_cast<char>(it->second);
        } else {
            result += '?';
        }
    }
    return result;
}

std::vector<std::string> BPETokenizer::byte_level_split(const std::string& text) const {
    std::string unicode_text = bytes_to_unicode(text);

    std::vector<std::string> chars;
    size_t i = 0;
    while (i < unicode_text.length()) {
        size_t char_len = 1;

        if ((unicode_text[i] & 0x80) == 0) {
            char_len = 1;
        } else if ((unicode_text[i] & 0xE0) == 0xC0) {
            char_len = 2;
        } else if ((unicode_text[i] & 0xF0) == 0xE0) {
            char_len = 3;
        } else if ((unicode_text[i] & 0xF8) == 0xF0) {
            char_len = 4;
        }

        if (i + char_len <= unicode_text.length()) {
            chars.push_back(unicode_text.substr(i, char_len));
        }
        i += char_len;
    }

    return chars;
}


std::pair<int, uint32_t> BPETokenizer::find_best_merge_fast(const std::vector<std::string>& tokens) const {
    int best_pos = -1;
    uint32_t best_priority = UINT32_MAX;

    for (size_t i = 0; i < tokens.size() - 1; ++i) {
        std::string key = tokens[i] + "\x00" + tokens[i + 1];
        auto it = merge_map_.find(key);
        if (it != merge_map_.end()) {
            if (it->second < best_priority) {
                best_priority = it->second;
                best_pos = static_cast<int>(i);
            }
        }
    }

    return {best_pos, best_priority};
}

std::vector<std::string> BPETokenizer::apply_bpe(const std::vector<std::string>& tokens) const {
    if (tokens.size() <= 1) return tokens;

    std::vector<std::string> current_tokens = tokens;


    while (true) {
        auto [merge_pos, priority] = find_best_merge_fast(current_tokens);
        if (merge_pos == -1) break;


        std::vector<std::string> new_tokens;
        new_tokens.reserve(current_tokens.size() - 1);

        for (int i = 0; i < static_cast<int>(current_tokens.size()); ++i) {
            if (i == merge_pos) {
                std::string merged = current_tokens[i] + current_tokens[i + 1];
                new_tokens.push_back(merged);
                i++;
            } else {
                new_tokens.push_back(current_tokens[i]);
            }
        }
        current_tokens = std::move(new_tokens);
    }

    return current_tokens;
}

std::vector<uint32_t> BPETokenizer::encode(const std::string& text) const {
    if (text.empty()) return {};


    auto text_segments = split_with_special_tokens(text);


    std::vector<uint32_t> token_ids;

    for (const auto& segment : text_segments) {
        auto special_it = special_tokens_.find(segment);
        if (special_it != special_tokens_.end()) {
            token_ids.push_back(special_it->second);
        } else {
            auto chars = byte_level_split(segment);
            auto bpe_tokens = apply_bpe(chars);


            for (const auto& token : bpe_tokens) {
                auto it = token_to_id_.find(token);
                if (it != token_to_id_.end()) {
                    token_ids.push_back(it->second);
                } else {
                    token_ids.push_back(unk_token_id_);
                }
            }
        }
    }

    return token_ids;
}

std::string BPETokenizer::decode(const std::vector<uint32_t>& tokens) const {
    std::string unicode_result;
    for (uint32_t token_id : tokens) {
        if (token_id < id_to_token_.size()) {
            unicode_result += id_to_token_[token_id];
        }
    }

    std::string result = unicode_to_bytes(unicode_result);

    return result;
}

void BPETokenizer::load_chat_template(const std::string& template_file) {
    std::ifstream file(template_file);
    if (!file.is_open()) {
        has_chat_template_ = false;
        return;
    }

    chat_template_ = std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    has_chat_template_ = !chat_template_.empty();
}

}
}
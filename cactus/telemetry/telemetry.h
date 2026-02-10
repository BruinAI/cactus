#pragma once
#include <string>
#include <cstdint>

namespace cactus {
namespace telemetry {

void init(const char* project_id = nullptr, const char* project_scope = nullptr, const char* cloud_key = nullptr);
void setEnabled(bool enabled);
void setCloudDisabled(bool disabled);
void recordInit(const char* model, bool success, double response_time_ms, const char* message);
void recordCompletion(const char* model, bool success, double ttft_ms, double tps, double response_time_ms, int tokens, const char* message);
void recordEmbedding(const char* model, bool success, const char* message);
void recordTranscription(const char* model, bool success, double ttft_ms, double tps, double response_time_ms, int tokens, const char* message);
void recordStreamTranscription(const char* model, bool success, double ttft_ms, double tps, double response_time_ms, int tokens, const char* message);
void setStreamMode(bool in_stream);
void markInference(bool active);
void flush();
void shutdown();

} // namespace telemetry
} // namespace cactus

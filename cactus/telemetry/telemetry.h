#pragma once
#include <string>
#include <cstdint>

namespace cactus {
namespace telemetry {

void init(const char* token);
void setEnabled(bool enabled);
void setCloudDisabled(bool disabled);
void recordInit(const char* model, bool success, const char* message);
void recordCompletion(const char* model, bool success, double ttft_ms, double tps, double response_time_ms, int tokens, const char* message);
void recordEmbedding(const char* model, bool success, const char* message);
void recordTranscription(const char* model, bool success, double ttft_ms, double tps, double response_time_ms, int tokens, const char* message);
void markInference(bool active);
void flush();
void shutdown();

} // namespace telemetry
} // namespace cactus

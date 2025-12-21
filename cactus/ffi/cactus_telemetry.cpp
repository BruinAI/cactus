#include "cactus_telemetry.h"

extern "C" {

void cactus_set_telemetry_token(const char* token) {
    auto& telemetry = cactus::ffi::CactusTelemetry::getInstance();
    if (token && token[0] != '\0') {
        telemetry.setTelemetryToken(token);
        telemetry.setEnabled(true);
    } else {
        telemetry.setEnabled(false);
    }
}

void cactus_set_pro_key(const char* pro_key) {
    if (pro_key) {
        cactus::ffi::DeviceManager::setProKey(pro_key);
    }
}

}

#include "npu.h"
#include <dlfcn.h>
#include <cstdlib>
#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif
#if defined(__APPLE__) && (TARGET_OS_IPHONE)
    const char* lib_name = "@rpath/cactus_util.framework/cactus_util";
#else
    const char* lib_name = nullptr;
#endif

namespace cactus {
namespace npu {

typedef std::unique_ptr<NPUEncoder> (*create_encoder_fn)();
typedef bool (*is_npu_available_fn)();
typedef std::unique_ptr<NPUPrefill> (*create_prefill_fn)();

static void* g_cactus_util_handle = nullptr;

template<typename T>
T get_runtime_symbol(const char* symbol_name) {
    if (!lib_name) {
        return nullptr;
    }
    void* handle = dlopen(lib_name, RTLD_NOW | RTLD_GLOBAL);
    if (handle) {
        g_cactus_util_handle = handle;
    }
    if (g_cactus_util_handle) {
        void* sym = dlsym(g_cactus_util_handle, symbol_name);
        if (sym) {
            return reinterpret_cast<T>(sym);
        }
    }
    return nullptr;
}

__attribute__((weak, visibility("default")))
std::unique_ptr<NPUEncoder> create_encoder() {
    auto strong_fn = get_runtime_symbol<create_encoder_fn>("_ZN6cactus3npu14create_encoderEv");
    if (strong_fn) {
        return strong_fn();
    }
    return nullptr;
}

__attribute__((weak, visibility("default")))
std::unique_ptr<NPUPrefill> create_prefill() {
    auto strong_fn = get_runtime_symbol<create_prefill_fn>("_ZN6cactus3npu14create_prefillEv");
    if (strong_fn) {
        return strong_fn();
    }
    return nullptr;
}

__attribute__((weak, visibility("default")))
bool is_npu_available() {
    auto strong_fn = get_runtime_symbol<is_npu_available_fn>("_ZN6cactus3npu16is_npu_availableEv");
    if (strong_fn) {
        return strong_fn();
    }
    return false;
}

} // namespace npu
} // namespace cactus
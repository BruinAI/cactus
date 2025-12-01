cmake_minimum_required(VERSION 3.10)
project(CactusTools LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -arch arm64 -pthread -Wall -Wextra -pedantic -O3")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Wall -Wextra -pedantic -O3")
endif()

set(CACTUS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../cactus)
set(CACTUS_BUILD_DIR ${CACTUS_ROOT}/build)

find_library(CACTUS_LIB 
    NAMES cactus
    PATHS ${CACTUS_BUILD_DIR} ${CACTUS_BUILD_DIR}/lib
    REQUIRED
)

set(MMAN_LIB "")
if(NOT APPLE)
    find_library(MMAN_LIB mman)
    if(MMAN_LIB)
        message(STATUS "Found libmman: ${MMAN_LIB}")
    else()
        message(STATUS "libmman not found; assuming mmap symbols are provided by the C runtime")
    endif()
endif()

# Benchmark tool
add_executable(benchmark_onnx benchmark_onnx.cpp)
target_link_libraries(benchmark_onnx PRIVATE ${CACTUS_LIB})

if(MMAN_LIB)
    target_link_libraries(benchmark_onnx PRIVATE ${MMAN_LIB})
endif()

target_include_directories(benchmark_onnx PRIVATE ${CACTUS_ROOT})

message(STATUS "Added tool: benchmark_onnx")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})


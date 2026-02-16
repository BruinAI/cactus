cmake_minimum_required(VERSION 3.10)
project(CactusAndroidTests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -Wall -Wextra -pedantic -O3")

set(CACTUS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../cactus)
set(ANDROID_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../android/build)
set(CACTUS_LIB ${ANDROID_BUILD_DIR}/libcactus_static.a)
set(CACTUS_CURL_ROOT "$ENV{CACTUS_CURL_ROOT}")
set(CACTUS_CURL_INCLUDE_DIR "${CACTUS_CURL_ROOT}/include")
set(CACTUS_CURL_LIBRARY "${CACTUS_CURL_ROOT}/android/arm64-v8a/libcurl.a")

if(NOT EXISTS ${CACTUS_LIB})
    message(FATAL_ERROR "Cactus library not found at: ${CACTUS_LIB}")
endif()

message(STATUS "Using Cactus library: ${CACTUS_LIB}")

set(TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
file(GLOB TEST_SOURCES "${TESTS_DIR}/test*.cpp")
list(REMOVE_ITEM TEST_SOURCES "${TESTS_DIR}/test_utils.cpp")

foreach(TEST_FILE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    add_executable(${TEST_NAME}
        ${TEST_FILE}
        ${TESTS_DIR}/test_utils.cpp
    )
    target_link_libraries(${TEST_NAME} PRIVATE ${CACTUS_LIB})
    if(CACTUS_CURL_ROOT AND EXISTS "${CACTUS_CURL_INCLUDE_DIR}/curl/curl.h" AND EXISTS "${CACTUS_CURL_LIBRARY}")
        target_include_directories(${TEST_NAME} PRIVATE "${CACTUS_CURL_INCLUDE_DIR}")
        target_link_libraries(${TEST_NAME} PRIVATE "${CACTUS_CURL_LIBRARY}")
        message(STATUS "Test ${TEST_NAME} will link vendored libcurl: ${CACTUS_CURL_LIBRARY}")
    endif()
    target_include_directories(${TEST_NAME} PRIVATE ${CACTUS_ROOT})
    message(STATUS "Added Android test: ${TEST_NAME}")
endforeach()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

cmake_minimum_required(VERSION 3.15)
project(cactus_dwconv_test LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cactus_dwconv_test test_depthwise_causal.cpp)

# If your function lives in a separate .cpp/.h file, add it here instead of in the test:
# target_sources(cactus_dwconv_test PRIVATE ../../src/cactus_conv1d.cpp)
# target_include_directories(cactus_dwconv_test PRIVATE ../../src)

# Convenience target to run the test against fixtures
add_custom_target(run
  COMMAND cactus_dwconv_test ${CMAKE_CURRENT_SOURCE_DIR}/fixtures
  DEPENDS cactus_dwconv_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  VERBATIM
)

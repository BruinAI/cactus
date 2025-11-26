cmake_minimum_required(VERSION 3.15)
project(cactus_slice_test LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(cactus_slice_test test_slice.cpp)

add_custom_target(run
  COMMAND cactus_slice_test ${CMAKE_CURRENT_SOURCE_DIR}/slice_fixtures_multi
  DEPENDS cactus_slice_test
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  VERBATIM
)

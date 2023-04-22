cmake_minimum_required(VERSION 3.18.0)

project(deplex LANGUAGES CXX DESCRIPTION "Library for Depth Plane Extraction")

if (WIN32)
    set(BUILD_SHARED_LIBS OFF)
else()
    option(BUILD_SHARED_LIBS "Build shared library" ON)
endif()

option(BUILD_TESTS "Build C++ unit-tests" ON)
option(BUILD_BENCHMARK "Build algorithm benchmark runner" OFF)
option(BUILD_EXAMPLES "Build C++ examples of deplex usage" ON)
option(BUILD_PYTHON "Build Python bindings" OFF)
option(DEBUG_DEPLEX "Additional verbosity and results by stage" OFF)
option(DEBUG_BENCHMARK "Disable optimizations, enable MSan and ASan" OFF)

set(DEPLEX_LIB_DIR ${CMAKE_BINARY_DIR}/lib)

add_subdirectory(external)
add_subdirectory(cpp)
if (${BUILD_EXAMPLES})
    add_subdirectory(examples)
endif()
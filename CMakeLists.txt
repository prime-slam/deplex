cmake_minimum_required(VERSION 3.18.0)

project(deplex LANGUAGES CXX DESCRIPTION "Library for Depth Plane Extraction")

option(BUILD_EXAMPLES "Build C++ examples of deplex usage" ON)
option(BUILD_PYTHON "Build Python bindings" ON)

set(DEPLEX_LIB_DIR ${CMAKE_BINARY_DIR}/lib)

add_subdirectory(3rd_party)
add_subdirectory(cpp)
if (${BUILD_EXAMPLES})
    add_subdirectory(examples)
endif()
cmake_minimum_required(VERSION 3.18.0)

project(cape LANGUAGES CXX DESCRIPTION "CAPE refactor")

option(BUILD_EXAMPLES "Build C++ examples of CAPE usage" ON)

add_subdirectory(project)
if (${BUILD_EXAMPLES})
    add_subdirectory(examples)
endif()
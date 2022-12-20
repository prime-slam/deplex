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

set(DEPLEX_LIB_DIR ${CMAKE_BINARY_DIR}/lib)

if (${BUILD_PYTHON})
    find_package(Python3 3.6
            COMPONENTS Interpreter Development)
    if (Python3_FOUND)
        # Setup PYTHON_EXECUTABLE for 3rdparty modules
        set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE} CACHE STRING
                "Deprecated path to the Python executable (for 3rdparty only)" FORCE)
        message("Found Python3 executable -- ${Python3_EXECUTABLE}")
    endif()
endif()

add_subdirectory(external)
add_subdirectory(cpp)
if (${BUILD_EXAMPLES})
    add_subdirectory(examples)
endif()
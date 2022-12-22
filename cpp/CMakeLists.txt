add_subdirectory(deplex)

if (${BUILD_TESTS})
    add_subdirectory(tests)
endif()

if (${BUILD_BENCHMARK})
    add_subdirectory(benchmark)
endif()

if (${BUILD_PYTHON})
    add_subdirectory(pybind)
endif()
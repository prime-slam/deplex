if (${BUILD_TESTS})
    add_subdirectory(googletest)
endif()

if (${BUILD_PYTHON})
    add_subdirectory(pybind11)
endif()
add_subdirectory(deplex)

if (${BUILD_PYTHON})
    add_subdirectory(pybind)
endif()
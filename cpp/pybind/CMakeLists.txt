if (NOT Python3_EXECUTABLE)
    message(FATAL_ERROR "Python 3 not found in top level file")
endif()

pybind11_add_module(pybind)

#####################################
# Set deplex python-package related variables
#####################################
set(PYTHON_PACKAGE_SRC_DIR ${PROJECT_SOURCE_DIR}/python)
set(PYTHON_PACKAGE_DST_DIR ${CMAKE_BINARY_DIR}/python_package)
set(PYTHON_BINARY_DIR ${DEPLEX_LIB_DIR}/pybind)
set(PYTHON_WHEEL_DST_DIR ${PYTHON_PACKAGE_DST_DIR}/pip-package)

#####################################
# Define sources of pybind target
#####################################
add_subdirectory(plane_extraction)

target_sources(pybind PRIVATE
    deplex_pybind.cpp
)

#####################################
# Configure target properties
#####################################
target_include_directories(pybind PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
target_link_libraries(pybind PRIVATE
    deplex)

set_target_properties(pybind PROPERTIES
        FOLDER "${PYTHON_BINARY_DIR}"
        LIBRARY_OUTPUT_DIRECTORY "${PYTHON_BINARY_DIR}"
)

#####################################
# Create Python package
#####################################
add_custom_target(python-package
        COMMAND ${CMAKE_COMMAND}
        -DPYTHON_PACKAGE_SRC_DIR=${PYTHON_PACKAGE_SRC_DIR}
        -DPYTHON_PACKAGE_DST_DIR=${PYTHON_PACKAGE_DST_DIR}
        -DPYTHON_BINARY_DIR=${PYTHON_BINARY_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/make_python_package.cmake)

add_custom_target(build-wheel
        COMMAND ${Python3_EXECUTABLE} setup.py bdist_wheel --dist-dir pip-package
        COMMAND echo "pip wheel created at ${PYTHON_WHEEL_DST_DIR}"
        WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}
        DEPENDS python-package
)

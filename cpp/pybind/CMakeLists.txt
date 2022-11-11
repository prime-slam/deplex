pybind11_add_module(pybind)
add_subdirectory(plane_extraction)

target_sources(pybind PRIVATE
    deplex_pybind.cpp
)

target_include_directories(pybind PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
)

target_link_libraries(pybind PRIVATE
    deplex)
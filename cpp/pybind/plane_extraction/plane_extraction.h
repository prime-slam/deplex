#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

namespace deplex{
void pybind_plane_extraction(py::module& m);
void pybind_config(py::module& m);
void pybind_read_image(py::module& m);
void pybind_extractor(py::module& m);
}
#pragma once
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace deplex{
void pybind_plane_extraction(py::module& m);
void pybind_config(py::module& m);
void pybind_extractor(py::module& m);
}
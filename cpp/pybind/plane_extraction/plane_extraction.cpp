#include "plane_extraction.h"
#include <deplex/plane_extraction.h>

namespace deplex {
void pybind_plane_extraction(py::module& m) {
  py::module m_plane_extraction = m.def_submodule(
      "plane_extraction", "Module with Plane Extraction algorithm");

  pybind_config(m_plane_extraction);
  pybind_extractor(m_plane_extraction);
}

void pybind_config(py::module& m) {
  py::class_<deplex::config::Config>(m, "Config")
      .def(py::init<std::string>(), py::arg("path"));
}

void pybind_extractor(py::module& m) {
  py::class_<deplex::PlaneExtractor>(m, "PlaneExtractor")
      .def(py::init<int, int, deplex::config::Config>(),
           py::arg("image_height"), py::arg("image_width"),
           py::arg("config") = deplex::PlaneExtractor::kDefaultConfig)
      .def("process", &PlaneExtractor::process, py::arg("pcd_array"));
}
}  // namespace deplex
/**
 * Copyright 2022 prime-slam
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "plane_extraction.h"

#include <deplex/plane_extractor.h>

namespace deplex {
void pybind_plane_extraction(py::module& m) {
  py::module m_plane_extraction = m.def_submodule("plane_extraction", "Module with Plane Extraction algorithm");

  pybind_config(m_plane_extraction);
  pybind_extractor(m_plane_extraction);
}

void pybind_config(py::module& m) {
  py::class_<config::Config>(m, "Config").def(py::init<std::string>(), py::arg("path"));
}

void pybind_extractor(py::module& m) {
  py::class_<PlaneExtractor>(m, "PlaneExtractor")
      .def(py::init<int, int, config::Config>(), py::arg("image_height"), py::arg("image_width"),
           py::arg("config") = config::Config())
      .def("process", &PlaneExtractor::process, py::arg("pcd_array"));
}
}  // namespace deplex
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
#include "utils.h"

#include <pybind11/eigen.h>

#include <deplex/utils/utils.h>

namespace deplex {
void pybind_utils(py::module& m) {
  py::module m_utils = m.def_submodule("utils", "Plane Extraction utilities");

  pybind_image(m_utils);
}

void pybind_image(py::module& m) {
  py::class_<utils::Image>(m, "Image")
      .def(py::init<std::string>(), py::arg("image_path"))
      .def_property_readonly("height", &utils::Image::getHeight)
      .def_property_readonly("width", &utils::Image::getWidth)
      .def("transform_to_pcd", &utils::Image::toPointCloud, py::arg("intrinsics"));
}
}  // namespace deplex
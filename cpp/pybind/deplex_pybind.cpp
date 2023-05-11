/**
 * Copyright (c) 2022, Arthur Saliou, Anastasiia Kornilova
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
#include "pybind/plane_extraction/plane_extraction.h"
#include "pybind/utils/utils.h"

namespace deplex {
PYBIND11_MODULE(pybind, m) {
  m.doc() = "This is pybind module";
  pybind_plane_extraction(m);
  pybind_utils(m);
}

}  // namespace deplex
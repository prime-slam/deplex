#include <pybind11/pybind11.h>
#include "pybind/plane_extraction/plane_extraction.h"

namespace deplex {
PYBIND11_MODULE(pybind, m) {
  m.doc() = "This is pybind module";
  pybind_plane_extraction(m);
}

}  // namespace deplex
#include <pybind11/pybind11.h>

#include <deplex/deplex.h>

namespace deplex {
PYBIND11_MODULE(pybind, m) { m.doc() = "This is pybind module"; }
}  // namespace deplex
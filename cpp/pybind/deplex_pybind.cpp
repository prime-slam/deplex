#include <pybind11/pybind11.h>


namespace deplex {
PYBIND11_MODULE(pybind, m) { m.doc() = "This is pybind module"; }
}  // namespace deplex
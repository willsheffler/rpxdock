/*cppimport
<%
cfg['include_dirs'] = ['../..','../extern']
cfg['compiler_args'] = ['-std=c++17', '-w', '-Ofast']
cfg['dependencies'] = ['_orientations.hpp']

setup_pybind11(cfg)
%>
*/

#include "sicdock/sampling/_orientations.hpp"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <Eigen/Geometry>

namespace sicdock {
namespace sampling {
namespace orientations {

namespace py = pybind11;

PYBIND11_MODULE(_orientations, m) {
  m.def("read_karney_orientations", &read_karney_orientations, R"pbdoc(
        docstring in sampling/orientations.pybind.cpp
    )pbdoc",
        py::call_guard<py::gil_scoped_release>());
}
}  // namespace orientations
}  // namespace sampling
}  // namespace sicdock
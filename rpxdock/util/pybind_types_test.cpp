/*/*cppimport
<%


cfg['include_dirs'] = ['../..','../extern']
cfg['compiler_args'] = ['-std=c++17', '-w']
cfg['dependencies'] = ['pybind_types.hpp']

cfg['parallel'] = False


setup_pybind11(cfg)
%>
*/
/** \file */

#include "rpxdock/util/pybind_types.hpp"

namespace py = pybind11;

namespace rpxdock {
namespace util {

template <typename F>
py::array_t<F> test_xform_round_trip(py::array_t<F> a) {
  MapVxX3<F> e = xform_py_to_eigen(a);
  if (e.size() > 7) {
    e[3].translation()[1] += 1;
    e[7].linear()(1, 2) = 9;
  } else {
    e[0].translation()[1] += 1;
    e[0].linear()(1, 2) = 9;
  }
  return xform_eigen_to_py(e);
}
PYBIND11_MODULE(pybind_types_test, m) {
  m.def("test_xform_round_trip", &test_xform_round_trip<double>);
}

}  // namespace util
}  // namespace rpxdock
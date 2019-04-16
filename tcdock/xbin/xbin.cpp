/*cppimport
<%
cfg['include_dirs'] = ['../..','../extern']
cfg['compiler_args'] = ['-std=c++17', '-w', '-Ofast']
cfg['dependencies'] = ['../geom/bcc.hpp','../util/assertions.hpp',
'../util/global_rng.hpp', 'xbin.hpp', '../util/numeric.hpp']

setup_pybind11(cfg)
%>
*/

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "iostream"

#include "tcdock/util/Timer.hpp"
#include "tcdock/util/assertions.hpp"
#include "tcdock/util/global_rng.hpp"
#include "tcdock/util/numeric.hpp"
#include "tcdock/util/types.hpp"
#include "tcdock/xbin/xbin.hpp"

using namespace pybind11::literals;
using namespace Eigen;
using namespace tcdock;
using namespace util;
using namespace geom;

namespace py = pybind11;

namespace tcdock {
namespace xbin {

using Key = XformHash_bt24_BCC6<X3f>::Key;

template <typename F>
py::array_t<Key> _get_keys(XformHash_bt24_BCC6<X3<F>> const &binner,
                           py::array_t<F> xforms) {
  py::buffer_info inp = xforms.request();
  if (inp.ndim != 3 || inp.shape[1] != 4 || inp.shape[2] != 4)
    throw std::runtime_error("Shape must be (N, 4, 4)");
  F *ptr = (F *)inp.ptr;

  py::array_t<Key> aout(inp.shape[0]);
  py::buffer_info out = aout.request();
  Key *outptr = (Key *)out.ptr;

  for (int i = 0; i < inp.shape[0]; ++i) {
    X3<F> x(Eigen::Map<M4<F>>(ptr + 16 * i));
    Key k = binner.get_key(x);
    outptr[i] = k;
  }
  return aout;
}

template <typename F>
py::array_t<F> _get_centers(XformHash_bt24_BCC6<X3<F>> const &binner,
                            py::array_t<Key> keys) {
  py::buffer_info inp = keys.request();
  if (inp.ndim != 1) throw std::runtime_error("Shape must be (N,)");
  Key *ptr = (Key *)inp.ptr;

  std::vector<int> shape{inp.shape[0], 4, 4};
  py::array_t<F> aout(shape);
  py::buffer_info out = aout.request();
  Key *outptr = (Key *)out.ptr;

  for (int i = 0; i < inp.shape[0]; ++i) {
    X3<F> *xp = (X3<F> *)(outptr + 16 * i);
    *xp = binner.get_center(ptr[i]);
  }

  return aout;
}

PYBIND11_MODULE(xbin, m) {
  py::class_<XformHash_bt24_BCC6<X3d>>(m, "_XBin_double")
      .def(py::init<double, double, double>());
  py::class_<XformHash_bt24_BCC6<X3f>>(m, "_XBin_float")
      .def(py::init<float, float, float>());

  m.def("get_keys_double", &_get_keys<double>);
  m.def("get_centers_double", &_get_centers<double>);
}

}  // namespace xbin
}  // namespace tcdock
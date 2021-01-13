/*/*cppimport
<%


cfg['include_dirs'] = ['../..', '../extern']
cfg['compiler_args'] = ['-std=c++17', '-w', '-Ofast']
cfg['dependencies'] = ['../extern/miniball/Seb.h',
'../extern/miniball/Seb-inl.h', '../util/Timer.hpp']

cfg['parallel'] = False


setup_pybind11(cfg)
%>
*/
/** \file */

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "rpxdock/util/numeric.hpp"
#include "rpxdock/util/pybind_types.hpp"

namespace py = pybind11;
using namespace py::literals;
using namespace Eigen;

#include <iostream>

using std::cout;
using std::endl;

namespace rpxdock {
namespace geom {
namespace xform_dist {

using namespace util;

template <typename F>
py::tuple xform_dist2_split(py::array_t<F> a, py::array_t<F> b, F lever) {
  auto xa = xform_py_to_eigen(a);
  auto xb = xform_py_to_eigen(b);
  auto outcart = std::make_unique<Mx<F>>();
  auto outori = std::make_unique<Mx<F>>();
  {
    py::gil_scoped_release release;
    outcart->resize(xa.size(), xb.size());
    outori->resize(xa.size(), xb.size());
    for (int i = 0; i < xa.size(); ++i) {
      for (int j = 0; j < xb.size(); ++j) {
        F d2cart = (xa[i].translation() - xb[j].translation()).squaredNorm();
        Quaternion<F> q(xa[i].linear().transpose() * xb[j].linear());
        F d2quat = sqr(1 - q.w()) + sqr(q.x()) + sqr(q.y()) + sqr(q.z());
        F d2ori = d2quat * 4;
        (*outcart)(i, j) = d2cart;
        (*outori)(i, j) = d2ori * lever * lever;
      }
    }
  }
  return py::make_tuple(*outcart, *outori);
}

PYBIND11_MODULE(xform_dist, m) {
  m.def("xform_dist2_split", &xform_dist2_split<float>,
        "dist and ang between xforms", "x"_a, "y"_a, "lever"_a = 1.0);
  m.def("xform_dist2_split", &xform_dist2_split<double>);
}

}  // namespace xform_dist
}  // namespace geom
}  // namespace rpxdock
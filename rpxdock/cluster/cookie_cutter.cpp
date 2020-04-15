/*/*cppimport
<%


cfg['include_dirs'] = ['../..','../extern']
cfg['compiler_args'] = ['-std=c++17', '-w', '-Ofast']
cfg['dependencies'] = ['../util/types.hpp']

cfg['parallel'] = False


setup_pybind11(cfg)
%>
*/
/** \file */

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>
#include <iostream>

namespace py = pybind11;

namespace rpxdock {
/**
\namespace rpxdock::cluster
\brief namespace for clustering stuff
*/
namespace cluster {
/**
\namespace rpxdock::cluster::cookie_cutter
\brief namespace for cookie_cutter style clustering
*/
namespace cookie_cutter {

using namespace Eigen;

template <typename F>
using Vx = Matrix<F, 1, Dynamic>;
template <typename F>
using RowMatrixX = Matrix<F, Dynamic, Dynamic, RowMajor>;

template <typename F>
Vx<int> cookie_cutter(Ref<RowMatrixX<F>> pts, F thresh) {
  py::gil_scoped_release release;
  using Keeper = std::pair<Vx<F>, int>;
  std::vector<Keeper> keep;
  for (int i = 0; i < pts.rows(); ++i) {
    bool seenit = false;
    for (auto& keeper : keep) {
      Vx<F> delta = pts.row(i) - keeper.first;
      F d2 = delta.squaredNorm();
      if (d2 <= thresh * thresh) {
        seenit = true;
        break;
      }
    }
    if (!seenit) keep.push_back(Keeper(pts.row(i), i));
  }
  Vx<int> out(keep.size());
  for (int i = 0; i < keep.size(); ++i) {
    out[i] = keep[i].second;
  }
  return out;
}

PYBIND11_MODULE(cookie_cutter, m) {
  m.def("cookie_cutter", &cookie_cutter<double>);
  m.def("cookie_cutter", &cookie_cutter<float>);
}

}  // namespace cookie_cutter
}  // namespace cluster
}  // namespace rpxdock
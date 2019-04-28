/*cppimport
<%
cfg['include_dirs'] = ['../..','../extern']
cfg['compiler_args'] = ['-std=c++17', '-w', '-Ofast']
cfg['dependencies'] = ['../util/types.hpp']

setup_pybind11(cfg)
%>
*/

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <iostream>

namespace py = pybind11;

namespace sicdock {
namespace cluster {
namespace cookie_cutter {

using namespace Eigen;

template <typename F>
using VectorX = Matrix<F, 1, Dynamic>;
template <typename F>
using RowMatrixX = Matrix<F, Dynamic, Dynamic, RowMajor>;

template <typename F>
py::array_t<int> cookie_cutter(Ref<RowMatrixX<F>> pts, F thresh) {
  // std::cout << "cookie_cutter " << pts.rows() << " " << pts.cols() << " "
  // << thresh << std::endl;

  using Keeper = std::pair<VectorX<F>, int>;
  std::vector<Keeper> keep;

  for (int i = 0; i < pts.rows(); ++i) {
    bool seenit = false;
    for (auto& keeper : keep) {
      VectorX<F> delta = pts.row(i) - keeper.first;
      F d2 = delta.squaredNorm();
      if (d2 <= thresh * thresh) {
        seenit = true;
        break;
      }
    }
    if (!seenit) keep.push_back(Keeper(pts.row(i), i));
  }
  // std::cout << "keep " << keep.size() << std::endl;

  // RowMatrixX<F> out(keep.size(), pts.cols());
  // for (int i = 0; i < keep.size(); ++i) {
  // out.row(i) = keep[i].first;
  // }
  py::array_t<int> out(keep.size());
  int* ptr = (int*)out.request().ptr;
  for (int i = 0; i < keep.size(); ++i) {
    ptr[i] = keep[i].second;
  }
  return out;
}

PYBIND11_MODULE(cookie_cutter, m) {
  m.def("cookie_cutter", &cookie_cutter<double>);
}

}  // namespace cookie_cutter
}  // namespace cluster
}  // namespace sicdock
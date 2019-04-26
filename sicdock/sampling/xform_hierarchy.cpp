/*cppimport
<%
cfg['include_dirs'] = ['../..', '../extern']
cfg['compiler_args'] = ['-std=c++17', '-w']
cfg['dependencies'] = ['../util/dilated_int.hpp', '../util/numeric.hpp',
'xform_hierarchy.hpp']

setup_pybind11(cfg)
%>
*/

#include "sicdock/sampling/xform_hierarchy.hpp"
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>

namespace py = pybind11;
using namespace py::literals;

namespace sicdock {
namespace sampling {

using namespace util;
using namespace Eigen;

double dist(X3d x, X3d y) {
  double lever = 0.5;
  auto t2 = (x.translation() - y.translation()).squaredNorm();
  auto m1 = x.linear(), m2 = y.linear();
  auto a = Eigen::AngleAxisd(m1.transpose() * m2).angle();
  // std::cout << a << std::endl;
  // auto r = x.linear()-
  // return (x.matrix() - y.matrix()).norm();
  return std::sqrt(t2 + a * a * lever * lever);
}

bool TEST_xform_hier_simple() {
  V3d lb(0, 0, 0);
  V3d ub(100, 100, 100);
  V3<uint64_t> bs(1, 1, 1);
  XformHier<> xh(lb, ub, bs, 999.0);
  std::cout << xh.ori_ncell_ << std::endl;
  X3d xi, xj;

  int resl = 0;
  // random sampling?

  Eigen::MatrixXd dmat(xh.size(resl), xh.size(resl));

  for (int i = 0; i < xh.size(resl); ++i) {
    bool ivalid = xh.get_value(resl, i, xi);
    for (int j = 0; j < xh.size(resl); ++j) {
      bool jvalid = xh.get_value(resl, j, xj);
      if (ivalid && jvalid && abs(i - j) > 0) {
        dmat(i, j) = dist(xi, xj);
      } else {
        dmat(i, j) = 9e9 * (i + 1) + 9e18 * (j + 1);
      }
    }
  }

  std::cout << "mindis " << dmat.minCoeff() << std::endl;
  // std::cout << valid << " " << x.translation().transpose() << std::endl;
  // std::cout << "   " << x.linear().row(0) << std::endl;
  // std::cout << "   " << x.linear().row(1) << std::endl;

  return true;
}

template <typename F, typename I>
py::tuple get_xforms(XformHier<F, I> xh, int resl,
                     Ref<Matrix<I, Dynamic, 1>> idx) {
  std::vector<size_t> xshape{idx.size(), 4, 4};
  py::array_t<I> iout(idx.size());
  py::array_t<F> xout(xshape);
  I* iptr = (I*)iout.request().ptr;
  X3<F>* xptr = (X3<F>*)xout.request().ptr;
  size_t nout = 0;
  for (size_t i = 0; i < idx.size(); ++i) {
    iptr[nout] = idx[i];
    bool valid = xh.get_value(resl, iptr[nout], xptr[nout]);
    if (valid) ++nout;
  }
  py::tuple out(2);
  out[0] = iout[py::slice(0, nout, 1)];
  out[1] = xout[py::slice(0, nout, 1)];
  return out;
}

struct ScoreIndex {
  double score;
  uint64_t index;
};

template <typename F, typename I>
py::tuple expand_top_N_impl(XformHier<F, I> xh, int N, int resl, int Nsi,
                            std::pair<double, uint64_t>* siptr) {
  N = std::min<int>(Nsi, N);
  std::vector<size_t> xshape{N * 64, 4, 4};
  py::array_t<F> xout(xshape);
  X3<F>* xptr = (X3<F>*)xout.request().ptr;
  py::array_t<I> iout(N * 64);
  I* iptr = (I*)iout.request().ptr;

  std::nth_element(siptr, siptr + N, siptr + Nsi,
                   std::greater<std::pair<double, uint64_t>>());

  size_t nout = 0;
  for (size_t i = 0; i < N; ++i) {
    I highbits = siptr[i].second << 6;
    for (I lowbits = 0; lowbits < 64; ++lowbits) {
      iptr[nout] = highbits | lowbits;
      bool valid = xh.get_value(resl + 1, iptr[nout], xptr[nout]);
      if (valid) ++nout;
    }
  }
  py::tuple out(2);
  out[0] = iout[py::slice(0, nout, 1)];
  out[1] = xout[py::slice(0, nout, 1)];
  return out;
}  // namespace sampling

template <typename F, typename I>
py::tuple expand_top_N(XformHier<F, I> xh, int N, int resl,
                       py::array_t<ScoreIndex> score_idx) {
  std::pair<double, uint64_t>* siptr =
      (std::pair<double, uint64_t>*)score_idx.request().ptr;
  return expand_top_N_impl(xh, N, resl, score_idx.size(), siptr);
}

template <typename F, typename I>
py::tuple expand_top_N(XformHier<F, I> xh, int N, int resl,
                       py::array_t<F> score, py::array_t<I> index) {
  F* sptr = (F*)score.request().ptr;
  I* iptr = (I*)index.request().ptr;
  std::vector<std::pair<double, uint64_t>> si(score.size());
  for (size_t i = 0; i < score.size(); ++i) {
    si[i].first = sptr[i];
    si[i].second = iptr[i];
  }
  std::pair<double, uint64_t>* siptr = &si[0];
  return expand_top_N_impl(xh, N, resl, si.size(), siptr);
}

template <typename F, typename I>
void bind_xh(auto m, std::string name) {
  py::class_<XformHier<F, I>>(m, name.c_str())
      .def(py::init<V3<F>, V3<F>, V3<I>, F>(), "lb"_a, "ub"_a, "bs"_a,
           "ori_resl"_a)
      .def("size", &XformHier<F, I>::size)
      .def("get_xforms", &get_xforms<F, I>)
      .def("expand_top_N",
           (py::tuple(*)(XformHier<F, I>, int, int, py::array_t<ScoreIndex>)) &
               expand_top_N<F, I>)
      .def("expand_top_N", (py::tuple(*)(XformHier<F, I>, int, int,
                                         py::array_t<F>, py::array_t<I>)) &
                               expand_top_N<F, I>)

      /**/;
}

PYBIND11_MODULE(xform_hierarchy, m) {
  // bind_xh<float>(m, "WelzlBVH_float");
  bind_xh<double, uint64_t>(m, "XformHier");

  m.def("TEST_xform_hier_simple", &TEST_xform_hier_simple);

  PYBIND11_NUMPY_DTYPE(ScoreIndex, score, index);
}

}  // namespace sampling
}  // namespace sicdock
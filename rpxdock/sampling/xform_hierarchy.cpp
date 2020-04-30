/*/*cppimport
<%


cfg['include_dirs'] = ['../..', '../extern']
cfg['compiler_args'] = ['-std=c++17', '-w', '-Ofast']
cfg['dependencies'] = ['../util/dilated_int.hpp', '../util/numeric.hpp',
'xform_hierarchy.hpp']

cfg['parallel'] = False


setup_pybind11(cfg)
%>
*/
/** \file */

#include "rpxdock/sampling/xform_hierarchy.hpp"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <iostream>

#include "rpxdock/util/pybind_types.hpp"

namespace py = pybind11;
using namespace py::literals;

namespace rpxdock {
namespace sampling {

using namespace util;
using namespace Eigen;

using std::cout;
using std::endl;

template <int N, typename F, typename I>
py::tuple get_trans(CartHier<N, F, I> ch, int resl, Vx<I> idx) {
  std::vector<size_t> xshape{idx.size(), N};
  py::array_t<bool> iout(idx.size());
  py::array_t<F> xout(xshape);
  bool* iptr = (bool*)iout.request().ptr;
  Matrix<F, N, 1>* tptr = (Matrix<F, N, 1>*)xout.request().ptr;
  size_t nout = 0;
  for (size_t i = 0; i < idx.size(); ++i) {
    iptr[i] = ch.get_value(resl, idx[i], tptr[nout]);
    if (iptr[i]) ++nout;
  }
  py::tuple out(2);
  out[0] = iout;
  out[1] = xout[py::slice(0, nout, 1)];
  return out;
}
template <typename Hier, typename F, typename I, typename X = M3<F>>
py::tuple get_ori(Hier const& h, int resl, Vx<I> idx) {
  std::vector<size_t> orishape{idx.size(), dim_of<X>(), dim_of<X>()};
  py::array_t<bool> iout(idx.size());
  py::array_t<F> oriout(orishape);
  bool* iptr = (bool*)iout.request().ptr;
  X* oriptr = (X*)oriout.request().ptr;
  size_t nout = 0;
  for (size_t i = 0; i < idx.size(); ++i) {
    M3<F> rot;
    iptr[i] = h.get_value(resl, idx[i], rot);
    oriptr[nout] = X::Identity();
    set_rot(oriptr[nout], rot);
    if (iptr[i]) ++nout;
  }
  return py::make_tuple(iout, oriout[py::slice(0, nout, 1)]);
}

template <typename H, typename F, typename I>
py::tuple get_xforms(H xh, int resl, Vx<I> idx) {
  std::vector<size_t> xshape{idx.size(), 4, 4};
  py::array_t<bool> iout(idx.size());
  py::array_t<F> xout(xshape);
  bool* iptr = (bool*)iout.request().ptr;
  X3<F>* xptr = (X3<F>*)xout.request().ptr;
  size_t nout = 0;
  {
    py::gil_scoped_release release;
    for (size_t i = 0; i < idx.size(); ++i) {
      iptr[i] = xh.get_value(resl, idx[i], xptr[nout]);
      if (iptr[i]) ++nout;
    }
  }
  return py::make_tuple(iout, xout[py::slice(0, nout, 1)]);
}

struct ScoreIndex {
  double score;
  uint64_t index;
};

template <typename H, typename F, typename I>
py::tuple expand_top_N_impl(H xh, int N, int resl,
                            std::vector<std::pair<double, I>> si) {
  N = std::min<int>(si.size(), N);
  std::vector<size_t> xshape{N * 64, 4, 4};
  py::array_t<F> xout(xshape);
  X3<F>* xptr = (X3<F>*)xout.request().ptr;
  py::array_t<I> iout(N * 64);
  I* iptr = (I*)iout.request().ptr;
  for (auto [s, i] : si)
    if (i >= xh.size(resl))
      throw std::range_error("expand_top_N_impl: index out of bounds");

  size_t nout = 0;
  {
    py::gil_scoped_release release;

    std::nth_element(si.begin(), si.begin() + N, si.end(),
                     std::greater<std::pair<double, I>>());
    for (size_t i = 0; i < N; ++i) {
      I parent = si[i].second;
      I beg = xh.child_of_begin(parent);
      I end = xh.child_of_end(parent);
      for (I idx = beg; idx < end; ++idx) {
        iptr[nout] = idx;
        bool valid = xh.get_value(resl + 1, iptr[nout], xptr[nout]);
        if (valid) ++nout;
      }
    }
  }

  return py::make_tuple(iout[py::slice(0, nout, 1)],
                        xout[py::slice(0, nout, 1)]);
}

template <typename H, typename F, typename I>
py::tuple expand_top_N_pairs(H xh, int N, int resl,
                             py::array_t<ScoreIndex> score_idx,
                             double null_val) {
  // sketchy... should be checking strides etc
  std::pair<double, I>* siptr = (std::pair<double, I>*)score_idx.request().ptr;
  std::vector<std::pair<double, I>> si;
  {
    py::gil_scoped_release release;
    for (size_t i = 0; i < score_idx.shape()[0]; ++i)
      if (siptr[i].first != null_val) si.push_back(siptr[i]);
  }
  return expand_top_N_impl<H, F, I>(xh, N, resl, si);
}

template <typename H, typename F, typename I>
py::tuple expand_top_N_separate(H xh, int N, int resl, Vx<double> score,
                                Vx<I> index, double null_val) {
  std::vector<std::pair<double, I>> si;
  {
    py::gil_scoped_release release;
    for (size_t i = 0; i < score.size(); ++i)
      if (score[i] != null_val)
        si.push_back(std::make_pair(score[i], index[i]));
    std::pair<double, I>* siptr = &si[0];
  }
  return expand_top_N_impl<H, F, I>(xh, N, resl, si);
}

template <int N, typename F, typename I>
void bind_CartHier(auto m, std::string name) {
  using Fn = Matrix<F, N, 1>;
  using In = Matrix<I, N, 1>;
  using THIS = CartHier<N, F, I>;
  py::class_<THIS>(m, name.c_str())
      .def(py::init<Fn, Fn, In>(), "lb"_a, "ub"_a, "bs"_a)
      .def("size", &THIS::size)
      .def_readonly("ncell", &THIS::cart_ncell_)
      .def("get_trans", &get_trans<N, F, I>)
      .def("get_xforms", &get_xforms<THIS, F, I>)
      .def("sanity_check", &THIS::sanity_check)
      .def_property_readonly("dim", &THIS::dim)
      /**/;
}

template <typename F, typename I>
RotHier<F, I> RotHier_nside(F lb, F ub, I nside, V3<F> axis) {
  return RotHier<F, I>(lb, ub, nside, axis);
}
template <typename F, typename I>
void bind_RotHier(auto m, std::string kind) {
  using THIS = RotHier<F, I>;
  py::class_<THIS>(m, ("RotHier_" + kind).c_str())
      .def(py::init<F, F, F, V3<F>>(), "lb"_a, "ub"_a, "resl"_a,
           "axis"_a = V3<F>(0, 0, 1))
      .def("size", &THIS::size)
      .def_readonly("lb", &THIS::rot_lb_)
      .def_readonly("ub", &THIS::rot_ub_)
      .def_readonly("rot_cell_width", &THIS::rot_cell_width_)
      .def_readonly("rot_ncell", &THIS::rot_ncell_)
      .def_readonly("axis", &THIS::axis_)
      .def_readonly("ncell", &THIS::rot_ncell_)
      .def_property_readonly("dim", &THIS::dim)
      .def("get_ori", &get_ori<THIS, F, I, M3<F>>, "resl"_a, "idx"_a)
      .def("get_xforms", &get_ori<THIS, F, I, X3<F>>, "resl"_a, "idx"_a)
      /**/;
  m.def(("create_RotHier_nside_" + kind).c_str(), &RotHier_nside<F, I>, "lb"_a,
        "ub"_a, "ncell"_a, "axis"_a = V3<F>(0, 0, 1));
}

template <typename F, typename I>
OriHier<F, I> OriHier_nside(I nside) {
  return OriHier<F, I>(nside);
}
template <typename F, typename I>
void bind_OriHier(auto m, std::string kind) {
  using THIS = OriHier<F, I>;
  py::class_<THIS>(m, ("OriHier_" + kind).c_str())
      .def(py::init<F>(), "ori_resl"_a)
      .def("size", &THIS::size)
      .def_readonly("ncell", &THIS::ori_ncell_)
      .def_readonly("ori_nside", &THIS::onside_)
      .def("get_ori", &get_ori<THIS, F, I, M3<F>>, "resl"_a, "idx"_a)
      .def("get_ori", &get_ori<THIS, F, I, X3<F>>, "resl"_a, "idx"_a)

      /**/;
  m.def(("create_OriHier_nside_" + kind).c_str(), &OriHier_nside<F, I>,
        "nside"_a);
}

template <typename F, typename I>
XformHier<F, I> XformHier_nside(V3<F> lb, V3<F> ub, V3<I> ncart, I nside) {
  return XformHier<F, I>(lb, ub, ncart, nside);
}
template <typename F, typename I>
void bind_XformHier(auto m, std::string kind) {
  using THIS = XformHier<F, I>;
  py::class_<THIS>(m, ("XformHier_" + kind).c_str())
      .def(py::init<V3<F>, V3<F>, V3<I>, F>(), "lb"_a, "ub"_a, "bs"_a,
           "ori_resl"_a)
      .def("size", &THIS::size)
      .def_readonly("ori_nside", &THIS::onside_)
      .def_readonly("ori_resl", &THIS::ori_resl_)
      .def_readonly("cart_lb", &THIS::cart_lb_)
      .def_readonly("cart_ub", &THIS::cart_ub_)
      .def_readonly("cart_bs", &THIS::cart_bs_)
      .def_readonly("cart_cell_width", &THIS::cart_cell_width_)
      .def_readonly("cart_ncell", &THIS::cart_ncell_)
      .def_readonly("ori_ncell", &THIS::ori_ncell_)
      .def_readonly("ncell", &THIS::ncell_)
      .def_property_readonly("dim", &THIS::dim)
      .def("sanity_check", &THIS::sanity_check)
      .def("cell_index_of", py::vectorize(&THIS::cell_index_of))
      .def("hier_index_of", py::vectorize(&THIS::hier_index_of))
      .def("parent_of", py::vectorize(&THIS::parent_of))
      .def("child_of_begin", py::vectorize(&THIS::child_of_begin))
      .def("child_of_end", py::vectorize(&THIS::child_of_end))
      .def("get_xforms", &get_xforms<THIS, F, I>, "iresl"_a, "idx"_a)
      .def("expand_top_N", expand_top_N_pairs<THIS, F, I>, "nkeep"_a, "resl"_a,
           "score_idx"_a, "null_val"_a = 0)
      .def("expand_top_N", expand_top_N_separate<THIS, F, I>, "nkeep"_a,
           "resl"_a, "score"_a, "index"_a, "null_val"_a = 0)

      /**/;
  m.def(("create_XformHier_nside_" + kind).c_str(), &XformHier_nside<F, I>,
        "lb"_a, "ub"_a, "bs"_a, "nside"_a);
}

template <typename F, typename I>
OriCart1Hier<F, I> OriCart1Hier_nside(V1<F> lb, V1<F> ub, V1<I> ncart,
                                      I nside) {
  return OriCart1Hier<F, I>(lb, ub, ncart, nside);
}

template <typename F, typename I>
py::tuple RC1H_get_state(RotCart1Hier<F, I> const& h) {
  return py::make_tuple(h.rot_lb_, h.rot_ub_, h.rot_cell_width_, h.rot_ncell_,
                        h.axis_, h.cart_lb_[0], h.cart_ub_[0], h.cart_bs_[0],
                        h.cart_cell_width_[0], h.cart_bs_pref_prod_[0],
                        h.cart_ncell_, h.ncell_);
}
template <typename F, typename I>
auto RC1H_set_state(py::tuple state) {
  auto h = std::make_unique<RotCart1Hier<F, I>>(0, 0, 0, 0, 0, 0);  // dummy
  h->rot_lb_ = py::cast<F>(state[0]);
  h->rot_ub_ = py::cast<F>(state[1]);
  h->rot_cell_width_ = py::cast<F>(state[2]);
  h->rot_ncell_ = py::cast<I>(state[3]);
  h->axis_ = py::cast<V3<F>>(state[4]);
  h->cart_lb_[0] = py::cast<F>(state[5]);
  h->cart_ub_[0] = py::cast<F>(state[6]);
  h->cart_bs_[0] = py::cast<I>(state[7]);
  h->cart_cell_width_[0] = py::cast<F>(state[8]);
  h->cart_bs_pref_prod_[0] = py::cast<I>(state[9]);
  h->cart_ncell_ = py::cast<I>(state[10]);
  h->ncell_ = py::cast<I>(state[11]);
  return h;
}

template <typename F, typename I>
void bind_RotCart1Hier(auto m, std::string name) {
  using THIS = RotCart1Hier<F, I>;
  py::class_<THIS>(m, name.c_str())
      .def(py::init<F, F, I, F, F, I, V3<F>>(), "cartlb"_a, "cartub"_a,
           "cartnc"_a, "rotlb"_a, "rotub"_a, "rotnc"_a,
           "axis"_a = V3<F>(0, 0, 1))
      .def("size", &THIS::size)
      .def_readonly("rot_ncell", &THIS::rot_ncell_)
      .def_readonly("cart_lb", &THIS::cart_lb_)
      .def_readonly("cart_ub", &THIS::cart_ub_)
      .def_readonly("cart_ncell", &THIS::cart_ncell_)
      .def_readonly("cart_bs_", &THIS::cart_bs_)
      .def_readonly("cart_cell_width_", &THIS::cart_cell_width_)
      .def_readonly("cart_bs_pref_prod_", &THIS::cart_bs_pref_prod_)
      .def_readonly("rot_lb", &THIS::rot_lb_)
      .def_readonly("rot_ub", &THIS::rot_ub_)
      .def_readonly("rot_ncell", &THIS::rot_ncell_)
      .def_readonly("rot_cell_width", &THIS::rot_cell_width_)
      .def_readonly("axis", &THIS::axis_)
      .def_readonly("ncell", &THIS::ncell_)
      .def_property_readonly("dim", &THIS::dim)
      .def("sanity_check", &THIS::sanity_check)
      .def("cell_index_of", py::vectorize(&THIS::cell_index_of))
      .def("hier_index_of", py::vectorize(&THIS::hier_index_of))
      .def("parent_of", py::vectorize(&THIS::parent_of))
      .def("child_of_begin", py::vectorize(&THIS::child_of_begin))
      .def("child_of_end", py::vectorize(&THIS::child_of_end))
      .def("get_xforms", &get_xforms<THIS, F, I>, "iresl"_a, "idx"_a)
      .def("expand_top_N", expand_top_N_pairs<THIS, F, I>, "nkeep"_a, "resl"_a,
           "score_idx"_a, "null_val"_a = 0)
      .def("expand_top_N", expand_top_N_separate<THIS, F, I>, "nkeep"_a,
           "resl"_a, "score"_a, "index"_a, "null_val"_a = 0)
      .def(py::pickle([](const THIS& h) { return RC1H_get_state<F, I>(h); },
                      [](py::tuple t) { return RC1H_set_state<F, I>(t); }))

      /**/;
}

template <typename F, typename I>
void bind_OriCart1Hier(auto m, std::string name) {
  using THIS = OriCart1Hier<F, I>;
  py::class_<THIS>(m, name.c_str())
      .def(py::init<V1<F>, V1<F>, V1<I>, F>(), "lb"_a, "ub"_a, "bs"_a,
           "ori_resl"_a)
      .def("size", &THIS::size)
      .def_readonly("ori_nside", &THIS::onside_)
      .def_readonly("ori_resl", &THIS::ori_resl_)
      .def_readonly("cart_lb", &THIS::cart_lb_)
      .def_readonly("cart_ub", &THIS::cart_ub_)
      .def_readonly("cart_bs", &THIS::cart_bs_)
      .def_readonly("cart_cell_width", &THIS::cart_cell_width_)
      .def_readonly("cart_ncell", &THIS::cart_ncell_)
      .def_readonly("ori_ncell", &THIS::ori_ncell_)
      .def_readonly("ncell", &THIS::ncell_)
      .def_property_readonly("dim", &THIS::dim)
      .def("sanity_check", &THIS::sanity_check)
      .def("cell_index_of", py::vectorize(&THIS::cell_index_of))
      .def("hier_index_of", py::vectorize(&THIS::hier_index_of))
      .def("parent_of", py::vectorize(&THIS::parent_of))
      .def("child_of_begin", py::vectorize(&THIS::child_of_begin))
      .def("child_of_end", py::vectorize(&THIS::child_of_end))
      .def("get_xforms", &get_xforms<THIS, F, I>, "iresl"_a, "idx"_a)
      .def("expand_top_N", expand_top_N_pairs<THIS, F, I>, "nkeep"_a, "resl"_a,
           "score_idx"_a, "null_val"_a = 0)
      .def("expand_top_N", expand_top_N_separate<THIS, F, I>, "nkeep"_a,
           "resl"_a, "score"_a, "index"_a, "null_val"_a = 0)

      /**/;
}

template <typename I, int DIM>
Matrix<I, Dynamic, DIM + 1, RowMajor> zorder2coeffs(Vx<I> idx, I resl) {
  Matrix<I, Dynamic, DIM + 1, RowMajor> out(idx.size(), DIM + 1);
  for (size_t i = 0; i < idx.size(); ++i) {
    out(i, 0) = idx[i] >> (DIM * resl);
    I hier_index = idx[i] & (((I)1 << (DIM * resl)) - 1);
    for (size_t j = 0; j < DIM; ++j) {
      out(i, j + 1) = util::undilate<DIM>(hier_index >> j);
    }
  }
  return out;
}
template <typename I, int DIM>
Vx<I> coeffs2zorder(Ref<Matrix<I, Dynamic, DIM + 1, RowMajor>> idx, I resl) {
  Vx<I> out(idx.rows());
  for (size_t i = 0; i < idx.rows(); ++i) {
    I cell_index = idx(i, 0);
    I index = 0;
    for (size_t j = 0; j < DIM; ++j)
      index |= util::dilate<DIM>(idx(i, j + 1)) << j;
    index = index | (cell_index << (DIM * resl));
    out[i] = index;
  }
  return out;
}

template <typename I>
Vx<I> pack_zorder(I resl, Mx<I> idx) {
  Vx<I> out(idx.cols());
  int dim = idx.rows();
  for (size_t i = 0; i < idx.cols(); ++i) {
    I index = 0;
    for (size_t j = 0; j < dim; ++j) index |= util::dilate(dim, idx(j, i)) << j;
    out[i] = index;
  }
  return out;
}
template <typename I>
Mx<I> unpack_zorder(int dim, I resl, Vx<I> idx) {
  Mx<I> out(dim, idx.size());
  for (size_t i = 0; i < idx.size(); ++i) {
    I hier_index = idx[i] & (((I)1 << (dim * resl)) - 1);
    for (size_t j = 0; j < dim; ++j) {
      out(j, i) = util::undilate(dim, hier_index >> j);
    }
  }
  return out;
}

template <typename F, typename I>
struct DummyHier {
  I dim_, ncell_;
  DummyHier(int dim, int ncell) : dim_(dim), ncell_(ncell) {}
  I size(I resl) const { return ncell_ * ((I)1) << (dim_ * resl); }
  I child_of_begin(I index) const { return index << dim_; }
  I child_of_end(I index) const { return (index + 1) << dim_; }
  bool get_value(I, I, X3<F>&) const { return true; }
};

template <typename F, typename I>
void bind_DummyHier(auto m, std::string name) {
  using THIS = DummyHier<F, I>;
  py::class_<THIS>(m, name.c_str())
      .def(py::init<uint64_t, uint64_t>(), "dim"_a, "ncell"_a)
      .def("size", &THIS::size)
      .def("child_of_begin", &THIS::child_of_begin)
      .def("child_of_end", &THIS::child_of_end)
      .def("expand_top_N", expand_top_N_separate<THIS, F, I>, "nkeep"_a,
           "resl"_a, "score"_a, "index"_a, "null_val"_a = 0)
      .def(py::pickle(
          [](const THIS& h) { return py::make_tuple(h.dim_, h.ncell_); },
          [](py::tuple t) {
            return std::make_unique<DummyHier<F, I>>(py::cast<I>(t[0]),
                                                     py::cast<I>(t[1]));
          }))
      /**/;
}

PYBIND11_MODULE(xform_hierarchy, m) {
  bind_DummyHier<float, uint64_t>(m, "DummyHier_f4");
  bind_DummyHier<double, uint64_t>(m, "DummyHier_f8");

  bind_CartHier<1, double, uint64_t>(m, "CartHier1D_f8");
  bind_CartHier<2, double, uint64_t>(m, "CartHier2D_f8");
  bind_CartHier<3, double, uint64_t>(m, "CartHier3D_f8");
  bind_CartHier<4, double, uint64_t>(m, "CartHier4D_f8");
  bind_CartHier<5, double, uint64_t>(m, "CartHier5D_f8");
  bind_CartHier<6, double, uint64_t>(m, "CartHier6D_f8");
  bind_CartHier<1, float, uint64_t>(m, "CartHier1D_f4");
  bind_CartHier<2, float, uint64_t>(m, "CartHier2D_f4");
  bind_CartHier<3, float, uint64_t>(m, "CartHier3D_f4");
  bind_CartHier<4, float, uint64_t>(m, "CartHier4D_f4");
  bind_CartHier<5, float, uint64_t>(m, "CartHier5D_f4");
  bind_CartHier<6, float, uint64_t>(m, "CartHier6D_f4");

  bind_OriHier<double, uint64_t>(m, "f8");
  bind_OriHier<float, uint64_t>(m, "f4");

  bind_RotHier<double, uint64_t>(m, "f8");
  bind_RotHier<float, uint64_t>(m, "f4");

  bind_RotCart1Hier<double, uint64_t>(m, "RotCart1Hier_f8");
  bind_RotCart1Hier<float, uint64_t>(m, "RotCart1Hier_f4");

  bind_XformHier<double, uint64_t>(m, "f8");
  bind_XformHier<float, uint64_t>(m, "f4");

  m.def("zorder3coeffs", &zorder2coeffs<uint64_t, 3>);
  m.def("coeffs3zorder", &coeffs2zorder<uint64_t, 3>);
  m.def("zorder6coeffs", &zorder2coeffs<uint64_t, 6>);
  m.def("coeffs6zorder", &coeffs2zorder<uint64_t, 6>);

  m.def("unpack_zorder", &unpack_zorder<uint64_t>, "dim"_a, "resl"_a,
        "indices"_a);
  m.def("pack_zorder", &pack_zorder<uint64_t>, "resl"_a, "indices"_a);

  bind_OriCart1Hier<float, uint64_t>(m, "OriCart1Hier_f4");
  m.def("create_OriCart1Hier_4f_nside", &OriCart1Hier_nside<float, uint64_t>,
        "lb"_a, "ub"_a, "bs"_a, "nside"_a);

  PYBIND11_NUMPY_DTYPE(ScoreIndex, score, index);
}

}  // namespace sampling
}  // namespace rpxdock
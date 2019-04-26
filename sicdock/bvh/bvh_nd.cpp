/*cppimport
<%
cfg['include_dirs'] = ['../..','../extern']
cfg['compiler_args'] = ['-std=c++17', '-w', '-O1']
cfg['dependencies'] = ['../geom/primitive.hpp','../util/assertions.hpp',
'../util/global_rng.hpp', 'bvh.hpp', 'bvh_algo.hpp', '../util/numeric.hpp']

setup_pybind11(cfg)
%>
*/

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "iostream"

#include "miniball/Seb.h"
#include "sicdock/bvh/bvh.hpp"
#include "sicdock/util/Timer.hpp"
#include "sicdock/util/assertions.hpp"
#include "sicdock/util/global_rng.hpp"
#include "sicdock/util/numeric.hpp"
#include "sicdock/util/types.hpp"

using namespace pybind11::literals;
using namespace Eigen;
using namespace sicdock;
using namespace util;
using namespace geom;

namespace py = pybind11;

template <typename F, int DIM>
struct SphereND {
  using This = SphereND<F, DIM>;
  using Vn = Matrix<F, DIM, 1>;

  Vn cen;
  F rad = 0;
  int lb = 0, ub = 0;
  SphereND() { cen.fill(0); }
  SphereND(Vn c) : cen(c) {}
  SphereND(Vn c, F r) : cen(c), rad(r) {}
  This merged(This that) const {
    if (this->contains(that)) return *this;
    if (that.contains(*this)) return that;
    F d = rad + that.rad + (cen - that.cen).norm();
    // std::cout << d << std::endl;
    auto dir = (that.cen - cen).normalized();
    auto c = cen + dir * (d / 2 - this->rad);
    auto out = This(c, d / 2 + epsilon2<F>() / 2.0);
    out.lb = std::min(this->lb, that.lb);
    out.ub = std::max(this->ub, that.ub);
    return out;
  }
  // Distance from p to boundary of the Sphere
  F signdis(Vn pt) const { return (cen - pt).norm() - rad; }
  F signdis2(Vn pt) const {  // NOT square of signdis!
    return (cen - pt).squaredNorm() - rad * rad;
  }
  F signdis(This s) const { return (cen - s.cen).norm() - rad - s.rad; }
  bool intersect(This that) const {
    F rtot = rad + that.rad;
    return (cen - that.cen).squaredNorm() <= rtot;
  }
  bool contact(This that, F contact_dis) const {
    F rtot = rad + that.rad + contact_dis;
    return (cen - that.cen).squaredNorm() <= rtot * rtot;
  }
  bool contains(Vn pt) const { return (cen - pt).squaredNorm() < rad * rad; }
  bool contains(This that) const {
    auto d = (cen - that.cen).norm();
    return d + that.rad <= rad;
  }
  bool operator==(This that) const {
    return cen.isApprox(that.cen) && fabs(rad - that.rad) < epsilon2<F>();
  }
};
template <class F, int DIM>
std::ostream& operator<<(std::ostream& out, SphereND<F, DIM> const& s) {
  out << "SphereND r = " << s.rad << ", c = ";
  for (int i = 0; i < DIM; ++i) out << " " << s.cen[i];
  return out;
}
//////////////////////////////////////////

struct BVHPointAccessor {
  RefRowMajorXd data;
  BVHPointAccessor(RefRowMajorXd d) : data(d) {}
  double* operator[](size_t i) const { return (double*)data.row(i).data(); }
  double* operator[](size_t i) { return data.row(i).data(); }
  size_t size() const { return data.rows(); }
};

template <typename F, int DIM>
struct BoundingSphereND {
  static SphereND<F, DIM> bound(auto pts) {
    using Vn = Matrix<F, DIM, 1>;
    using Miniball =
        Seb::Smallest_enclosing_ball<F, decltype(pts[0]), decltype(pts)>;
    Miniball mb(DIM, pts);
    SphereND<F, DIM> out;
    out.rad = mb.radius();
    auto cen_it = mb.center_begin();
    for (int i = 0; i < DIM; ++i) out.cen[i] = cen_it[i];
    return out;
  }
};

namespace Eigen {

template <class _Pt>
struct PtIdxND {
  using Pt = _Pt;
  PtIdxND() : pos(0), idx(0) {}
  PtIdxND(Pt v, int i = 0) : pos(v), idx(i) {}
  Pt pos;
  int idx;
};
template <class Pt>
std::ostream& operator<<(std::ostream& out, PtIdxND<Pt> const& pi) {
  out << pi.idx << " " << pi.pos.transpose();
  return out;
}

template <typename F, int N>
auto bounding_vol(PtIdxND<Matrix<F, N, 1>> v) {
  auto s = SphereND<F, N>(v.pos);
  s.lb = s.ub = v.idx;
  return s;
}

}  // namespace Eigen

namespace sicdock {
namespace bvh {

template <typename F, int DIM>
SphereBVH<F, PtIdxND<Matrix<F, DIM, 1>>, DIM, SphereND<F, DIM>,
          BoundingSphereND<F, DIM>>
create_bvh_nd(RefRowMajorXd pts) {
  using Pt = Matrix<F, DIM, 1>;
  using Pi = PtIdxND<Pt>;
  using Sph = SphereND<F, DIM>;
  using BVH = SphereBVH<F, Pi, DIM, Sph, BoundingSphereND<F, DIM>>;
  std::vector<Pi> objs;
  for (int i = 0; i < pts.rows(); ++i) {
    Pi pi;
    pi.idx = i;
    for (int j = 0; j < DIM; ++j) pi.pos[j] = pts(i, j);
    objs.push_back(pi);
  }
  return BVH(objs.begin(), objs.end());
}

template <typename BVH>
Matrix<typename BVH::F, Dynamic, BVH::DIM> bvh_obj_centers(BVH& b) {
  int n = b.objs.size();
  Matrix<typename BVH::F, Dynamic, BVH::DIM> out(n, BVH::DIM);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < BVH::DIM; ++j) out(i, j) = b.objs[i].pos[j];
  return out;
}
template <typename BVH>
Matrix<typename BVH::F, BVH::DIM, 1> bvh_obj_com(BVH& b) {
  typename BVH::Object::Pt com;
  com.fill(0);
  int n = b.objs.size();
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < BVH::DIM; ++j) com[j] += b.objs[i].pos[j];
  com /= n;
  return com;
}

template <typename F, int DIM>
void bind_bvh_ND(auto m, std::string name) {
  using Pt = Matrix<F, DIM, 1>;
  using Pi = PtIdxND<Pt>;
  using Sph = SphereND<F, DIM>;
  using BVH = SphereBVH<F, Pi, DIM, Sph, BoundingSphereND<F, DIM>>;

  py::class_<BVH>(m, name.c_str())
      .def("__len__", [](BVH& b) { return b.objs.size(); })
      .def("radius", [](BVH& b) { return b.vols[b.getRootIndex()].rad; })
      .def("center", [](BVH& b) { return b.vols[b.getRootIndex()].cen; })
      .def("centers", &bvh_obj_centers<BVH>)
      .def("com", &bvh_obj_com<BVH>)
      /**/;
}

PYBIND11_MODULE(bvh_nd, m) {
  bind_bvh_ND<double, 7>(m, "SphereBVH7");
  m.def("create_bvh7", &create_bvh_nd<double, 7>);
  /**/
}
}  // namespace bvh
}  // namespace sicdock
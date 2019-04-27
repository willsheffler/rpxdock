/*cppimport
<%
cfg['include_dirs'] = ['../..','../extern']
cfg['compiler_args'] = ['-std=c++17', '-w', '-Ofast']
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
#include "sicdock/geom/primitive.hpp"
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

template <typename F, int DIM>
using BVH = SphereBVH<F, PtIdxND<Matrix<F, DIM, 1>>, DIM, SphereND<F, DIM>,
                      BoundingSphereND<F, DIM>>;

template <typename F, int DIM>
struct BVIsectND {
  using Sph = SphereND<F, DIM>;
  using Obj = PtIdxND<Matrix<F, DIM, 1>>;
  BVIsectND(F r) : rad(r), rad2(r * r) {}
  bool intersectVolumeVolume(Sph s1, Sph s2) { return s1.signdis(s2) < rad; }
  bool intersectVolumeObject(Sph s1, Obj o2) {
    return s1.signdis(o2.pos) < rad;
  }
  bool intersectObjectVolume(Obj o1, Sph s2) {
    return s2.signdis(o1.pos) < rad;
  }
  bool intersectObjectObject(Obj o1, Obj o2) {
    bool isect = (o1.pos - o2.pos).squaredNorm() < rad2;
    result |= isect;
    return isect;
  }
  F rad = 0, rad2 = 0;
  bool result = false;
};
template <typename F, int DIM>
bool bvh_isect(BVH<F, DIM>& bvh1, BVH<F, DIM>& bvh2, F thresh) {
  BVIsectND<F, DIM> query(thresh);
  sicdock::bvh::BVIntersect(bvh1, bvh2, query);
  return query.result;
}
template <typename F, int DIM>
bool naive_isect(BVH<F, DIM>& bvh1, BVH<F, DIM>& bvh2, F thresh) {
  F dist2 = thresh * thresh;
  for (auto o1 : bvh1.objs) {
    for (auto o2 : bvh2.objs) {
      auto d2 = (o1.pos - o2.pos).squaredNorm();
      if (d2 < dist2) return true;
    }
  }
  return false;
}

template <typename F>
Matrix<F, 9, 1> reinterp_mul(M3<F> rot, Matrix<F, 9, 1> v9) {
  M3<F>* cen = (M3<F>*)(&v9);
  M3<F> newcen = rot * *cen;
  return *((Matrix<F, 9, 1>*)newcen.data());
}
template <typename F>
SphereND<F, 9> reinterp_mul(M3<F> rot, SphereND<F, 9> sph) {
  SphereND<F, 9> out;
  out.rad = sph.rad;
  out.cen = reinterp_mul(rot, sph.cen);
  return out;
}
template <typename F>
Matrix<F, 12, 1> reinterp_mul(X3C<F> rot, Matrix<F, 12, 1> v12) {
  X3C<F>* cen = (X3C<F>*)(&v12);
  X3C<F> newcen = rot * *cen;
  return *((Matrix<F, 12, 1>*)newcen.data());
}
template <typename F>
SphereND<F, 12> reinterp_mul(X3C<F> rot, SphereND<F, 12> sph) {
  SphereND<F, 12> out;
  out.rad = sph.rad;
  out.cen = reinterp_mul(rot, sph.cen);
  return out;
}

template <typename F, int DIM, typename X>
struct BVHIsectNDXform {
  using Sph = SphereND<F, DIM>;
  using Obj = PtIdxND<Matrix<F, DIM, 1>>;
  BVHIsectNDXform(F r, X x = X::Identity()) : rad(r), rad2(r * r), bXa(x) {}
  bool intersectVolumeVolume(Sph s1, Sph s2) {
    return s1.signdis(reinterp_mul(bXa, s2)) < rad;
  }
  bool intersectVolumeObject(Sph s1, Obj o2) {
    return s1.signdis(reinterp_mul(bXa, o2.pos)) < rad;
  }
  bool intersectObjectVolume(Obj o1, Sph s2) {
    return (reinterp_mul(bXa, s2)).signdis(o1.pos) < rad;
  }
  bool intersectObjectObject(Obj o1, Obj o2) {
    bool isect = (o1.pos - reinterp_mul(bXa, o2.pos)).squaredNorm() < rad2;
    result |= isect;
    return isect;
  }
  F rad = 0, rad2 = 0;
  bool result = false;
  X bXa = X::Identity();
};

template <typename F>
bool bvh_isect_ori(BVH<F, 9>& bvh1, BVH<F, 9>& bvh2, M3<F> pos1, M3<F> pos2,
                   F mindist) {
  M3<F> pos = pos1.inverse() * pos2;
  BVHIsectNDXform<F, 9, M3<F>> query(mindist, pos);
  sicdock::bvh::BVIntersect(bvh1, bvh2, query);
  return query.result;
}
template <typename F>
bool naive_isect_ori(BVH<F, 9>& bvh1, BVH<F, 9>& bvh2, M3<F> pos1, M3<F> pos2,
                     F mindist) {
  M3<F> pos = pos1.inverse() * pos2;
  F dist2 = mindist * mindist;
  for (auto o1 : bvh1.objs) {
    for (auto o2 : bvh2.objs) {
      auto d2 = (o1.pos - reinterp_mul(pos, o2.pos)).squaredNorm();
      if (d2 < dist2) return true;
    }
  }
  return false;
}
template <typename F>
bool bvh_isect_xform(BVH<F, 12>& bvh1, BVH<F, 12>& bvh2, M4<F> pos1, M4<F> pos2,
                     F mindist) {
  X3C<F> x1(pos1), x2(pos2);
  X3C<F> x = x1.inverse() * x2;
  BVHIsectNDXform<F, 12, X3C<F>> query(mindist, x);
  sicdock::bvh::BVIntersect(bvh1, bvh2, query);
  return query.result;
}
template <typename F>
bool naive_isect_xform(BVH<F, 12>& bvh1, BVH<F, 12>& bvh2, M4<F> pos1,
                       M4<F> pos2, F mindist) {
  X3C<F> x1(pos1), x2(pos2);
  X3C<F> x = x1.inverse() * x2;
  F dist2 = mindist * mindist;
  for (auto o1 : bvh1.objs) {
    for (auto o2 : bvh2.objs) {
      auto d2 = (o1.pos - reinterp_mul(x, o2.pos)).squaredNorm();
      if (d2 < dist2) return true;
    }
  }
  return false;
}

template <typename F, int DIM>
BVH<F, DIM> create_bvh_nd(RefRowMajorXd pts) {
  using Pt = Matrix<F, DIM, 1>;
  using Pi = PtIdxND<Pt>;
  using BVH = BVH<F, DIM>;
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
  using BVH = BVH<F, DIM>;
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
  m.def("bvh_isect7", &bvh_isect<double, 7>);
  m.def("naive_isect7", &naive_isect<double, 7>);

  bind_bvh_ND<double, 9>(m, "SphereBVH9");
  m.def("create_bvh9", &create_bvh_nd<double, 9>);
  m.def("bvh_isect_ori", &bvh_isect_ori<double>);
  m.def("naive_isect_ori", &naive_isect_ori<double>);

  bind_bvh_ND<double, 12>(m, "SphereBVH12");
  m.def("create_bvh12", &create_bvh_nd<double, 12>);
  m.def("bvh_isect_xform", &bvh_isect_xform<double>);
  m.def("naive_isect_xform", &naive_isect_xform<double>);
}
}  // namespace bvh
}  // namespace sicdock
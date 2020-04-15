// /*
// <%
// cfg['include_dirs'] = ['../..', '../extern']
// cfg['compiler_args'] = ['-std=c++17', '-w', '-Ofast']
// cfg['dependencies'] = ['../geom/primitive.hpp','../util/assertions.hpp',
// '../util/global_rng.hpp', 'bvh.hpp', 'bvh_algo.hpp', '../util/numeric.hpp']
//
// cfg['parallel'] = False

setup_pybind11(cfg)
    // %>
    // */

/*

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "iostream"
#include "miniball/Seb.h"
#include "rpxdock/bvh/bvh.hpp"
#include "rpxdock/geom/primitive.hpp"
#include "rpxdock/util/Timer.hpp"
#include "rpxdock/util/assertions.hpp"
#include "rpxdock/util/global_rng.hpp"
#include "rpxdock/util/numeric.hpp"
#include "rpxdock/util/types.hpp"

using namespace pybind11::literals;
using namespace Eigen;
using namespace rpxdock;
using namespace util;
using namespace geom;

namespace py = pybind11;

namespace rpxdock {
namespace bvh {

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
  rpxdock::bvh::BVIntersect(bvh1, bvh2, query);
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
  rpxdock::bvh::BVIntersect(bvh1, bvh2, query);
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
struct BVMinOneND {
  using Scalar = F;
  using Sph = SphereND<F, DIM>;
  using Obj = PtIdxND<Matrix<F, DIM, 1>>;
  Matrix<F, DIM, 1> pt;
  BVMinOneND(Matrix<F, DIM, 1> p) : pt(p) {}
  F minimumOnVolume(Sph s) { return s.signdis(pt); }
  F minimumOnObject(Obj o) { return (o.pos - pt).norm(); }
};
template <typename F, int DIM>
F bvh_min_one(BVH<F, DIM>& bvh, Matrix<F, DIM, 1> pt) {
  BVMinOneND<F, DIM> query(pt);
  return rpxdock::bvh::BVMinimize(bvh, query);
}
template <typename F>
F bvh_min_one_ori(BVH<F, 9>& bvh, M3<F> m) {
  Matrix<F, 9, 1> pt = *((Matrix<F, 9, 1>*)(&m));
  BVMinOneND<F, 9> query(pt);
  return rpxdock::bvh::BVMinimize(bvh, query);
}
template <typename F, int DIM>
F bvh_min_one_quatplus(BVH<F, DIM>& bvh, Matrix<F, DIM, 1> q) {
  BVMinOneND<F, DIM> query(q);
  return rpxdock::bvh::BVMinimize(bvh, query);
}
template <typename F>
F bvh_min_one_quat(BVH<F, 4>& bvh, M3<F> rot) {
  Quaternion<F> q(rot);
  Matrix<F, 4, 1> pt = *((Matrix<F, 4, 1>*)(&q));
  BVMinOneND<F, 4> query(pt);
  return rpxdock::bvh::BVMinimize(bvh, query);
}
template <typename F>
py::tuple naive_min_one_quat(BVH<F, 4>& bvh, M3<F> rot) {
  Quaternion<F> q(rot);
  Matrix<F, 4, 1> pt = *((Matrix<F, 4, 1>*)(&q));
  F mn2 = 9e9;
  int idx = -1;
  for (int i = 0; i < bvh.objs.size(); ++i) {
    F d2 = (bvh.objs[i].pos - pt).squaredNorm();
    if (d2 < mn2) {
      mn2 = d2;
      idx = i;
    }
  }
  return py::make_tuple(std::sqrt(mn2), idx);
}

template <typename F, int DIM>
BVH<F, DIM> create_bvh_quatplus(RefMxd pts) {
  using Pt = Matrix<F, DIM, 1>;
  using Pi = PtIdxND<Pt>;
  using BVH = BVH<F, DIM>;
  std::vector<Pi> objs;
  for (int i = 0; i < pts.rows(); ++i) {
    Pi pi;
    pi.idx = i;
    for (int j = 0; j < DIM; ++j) pi.pos[j] = pts(i, j);
    objs.push_back(pi);
    for (int j = 0; j < 4; ++j) pi.pos[j] = -pi.pos[j];
    objs.push_back(pi);
  }
  return BVH(objs.begin(), objs.end());
}

template <typename F>
BVH<F, 4> create_bvh_quat(RefMxd pts) {
  if (pts.cols() == 4) {
    return create_bvh_quatplus<F, 4>(pts);
  } else if (pts.cols() != 9) {
    throw std::runtime_error("quat coords shape must be (N,9) or (N,4)");
  }
  using Pt = Matrix<F, 4, 1>;
  using Pi = PtIdxND<Pt>;
  using BVH = BVH<F, 4>;
  std::vector<Pi> objs;
  for (int i = 0; i < pts.rows(); ++i) {
    Map<M3<F>> rot(pts.row(i).data());
    Quaternion<F> q(rot);
    Pi pi;
    pi.idx = i;
    pi.pos = *((Pt*)(&q));
    objs.push_back(pi);
    pi.pos = -pi.pos;
    objs.push_back(pi);
  }
  return BVH(objs.begin(), objs.end());
}

}  // namespace bvh
}  // namespace rpxdock

* /

/*/*cppimport
<%


cfg['include_dirs'] = ['../..','../extern']
cfg['compiler_args'] = ['-std=c++17', '-w', '-Ofast']
cfg['dependencies'] = ['../geom/primitive.hpp','../util/assertions.hpp',
'../util/global_rng.hpp', 'bvh.hpp', 'bvh_algo.hpp', '../util/numeric.hpp']

cfg['parallel'] = False


setup_pybind11(cfg)
%>
*/
/** \file */

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

namespace rpxdock {
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
struct BVBVIsectND {
  using Sph = SphereND<F, DIM>;
  using Obj = PtIdxND<Matrix<F, DIM, 1>>;
  BVBVIsectND(F r) : rad(r), rad2(r * r) {}
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
bool bvh_bvh_isect(BVH<F, DIM>& bvh1, BVH<F, DIM>& bvh2, F thresh) {
  py::gil_scoped_release release;
  BVBVIsectND<F, DIM> query(thresh);
  rpxdock::bvh::BVIntersect(bvh1, bvh2, query);
  return query.result;
}
template <typename F, int DIM>
bool bvh_bvh_isect_naive(BVH<F, DIM>& bvh1, BVH<F, DIM>& bvh2, F thresh) {
  F dist2 = thresh * thresh;
  for (auto o1 : bvh1.objs) {
    for (auto o2 : bvh2.objs) {
      auto d2 = (o1.pos - o2.pos).squaredNorm();
      if (d2 < dist2) return true;
    }
  }
  return false;
}

template <typename F, int DIM>
struct BVMinDistND {
  using Scalar = F;
  using Sph = SphereND<F, DIM>;
  using Obj = PtIdxND<Matrix<F, DIM, 1>>;
  F mn = 9e9;
  int imin = -1;
  Matrix<F, DIM, 1> pt;
  BVMinDistND(Matrix<F, DIM, 1> p) : pt(p) {}
  F minimumOnVolume(Sph s) { return s.signdis(pt); }
  F minimumOnObject(Obj o) {
    F d = (o.pos - pt).norm();
    if (d < mn) {
      imin = o.idx;
      mn = d;
    }
    return d;
  }
};
template <typename F, int DIM>
py::tuple bvh_mindist(BVH<F, DIM>& bvh, RefMxd pts) {
  auto outm = std::make_unique<Vxd>();
  auto outi = std::make_unique<Vxi>();

  {
    py::gil_scoped_release release;
    outm->resize(pts.rows());
    outi->resize(pts.rows());
    // double ncmp = 0;
    for (int i = 0; i < pts.rows(); ++i) {
      BVMinDistND<F, DIM> query(pts.row(i));
      (*outm)[i] = rpxdock::bvh::BVMinimize(bvh, query);
      (*outi)[i] = query.imin;
      // ncmp += query.ncmp;
    }
    // std::cout << "bvh ncmp " << ncmp / pts.rows() << std::endl;
  }
  return py::make_tuple(*outm, *outi);
}
template <typename F, int DIM>
py::tuple bvh_mindist_naive(BVH<F, DIM>& bvh, RefMxd pts) {
  Vxd outm(pts.rows());
  Vxi outi(pts.rows());
  // int64_t ncmp = 0;
  for (int i = 0; i < pts.rows(); ++i) {
    Matrix<F, DIM, 1> pt = pts.row(i);
    F mn2 = 9e9;
    for (int j = 0; j < bvh.objs.size(); ++j) {
      // ++ncmp;
      F d2 = (bvh.objs[j].pos - pt).squaredNorm();
      if (d2 < mn2) {
        mn2 = d2;
        outi[i] = bvh.objs[j].idx;
      }
    }
    outm[i] = std::sqrt(mn2);
  }
  // std::cout << "naive ncmp " << ncmp / pts.rows() << std::endl;
  return py::make_tuple(outm, outi);
}

template <typename F, int DIM>
struct BVIsectND {
  using Sph = SphereND<F, DIM>;
  using Obj = PtIdxND<Matrix<F, DIM, 1>>;
  Matrix<F, DIM, 1> pt;
  F mindist = 0, mindist2 = 0;
  int i_isect = -1;
  BVIsectND(Matrix<F, DIM, 1> p, F r) : pt(p), mindist(r), mindist2(r * r) {}
  bool intersectVolume(Sph s) { return s.signdis(pt) < mindist; }
  bool intersectObject(Obj o) {
    bool isect = (o.pos - pt).squaredNorm() < mindist2;
    if (isect) {
      i_isect = o.idx;
      return true;
    }
    return false;
  }
};
template <typename F, int DIM>
Vxi bvh_isect(BVH<F, DIM>& bvh, RefMxd pts, F mindist) {
  py::gil_scoped_release release;
  Vxi out(pts.rows());
  for (int i = 0; i < pts.rows(); ++i) {
    BVIsectND<F, DIM> query(pts.row(i), mindist);
    rpxdock::bvh::BVIntersect(bvh, query);
    out[i] = query.i_isect;
  }
  return out;
}
template <typename F, int DIM>
Vxi bvh_isect_naive(BVH<F, DIM>& bvh, RefMxd pts, F mindist) {
  Vxi out(pts.rows());
  out.fill(-1);
  for (int i = 0; i < pts.rows(); ++i) {
    Matrix<F, DIM, 1> pt = pts.row(i);
    for (int j = 0; j < bvh.objs.size(); ++j) {
      F d2 = (bvh.objs[j].pos - pt).squaredNorm();
      if (d2 < mindist * mindist) {
        out[i] = j;
        break;
      }
    }
  }
  return out;
}

template <typename F, int DIM>
BVH<F, DIM> create_bvh_nd(RefMxd pts) {
  if (pts.cols() != DIM)
    throw std::runtime_error("input must be shape (N,DIM)");
  py::gil_scoped_release release;
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
template <typename F, int DIM>
BVH<F, DIM> create_bvh_quatplus(RefMxd pts) {
  if (pts.cols() != DIM)
    throw std::runtime_error("quat input must be shape (N,4)");
  py::gil_scoped_release release;
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

template <typename BVH>
Matrix<typename BVH::F, Dynamic, BVH::DIM> bvh_obj_centers(BVH& b) {
  py::gil_scoped_release release;
  int n = b.objs.size();
  Matrix<typename BVH::F, Dynamic, BVH::DIM> out(n, BVH::DIM);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < BVH::DIM; ++j) out(i, j) = b.objs[i].pos[j];
  return out;
}
template <typename BVH>
Matrix<typename BVH::F, BVH::DIM, 1> bvh_obj_com(BVH& b) {
  py::gil_scoped_release release;
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
  bind_bvh_ND<double, 7>(m, "SphereBVH7D");
  m.def("create_bvh7d", &create_bvh_nd<double, 7>);
  m.def("bvh_bvh_isect7d", &bvh_bvh_isect<double, 7>);
  m.def("bvh_bvh_isect7d_naive", &bvh_bvh_isect_naive<double, 7>);
  m.def("bvh_isect7d", &bvh_isect<double, 7>);
  m.def("bvh_isect7d_naive", &bvh_isect_naive<double, 7>);
  m.def("bvh_mindist7d", &bvh_mindist<double, 7>);
  m.def("bvh_mindist7d_naive", &bvh_mindist_naive<double, 7>);
  m.def("create_bvh_xform", &create_bvh_quatplus<double, 7>);

  bind_bvh_ND<double, 4>(m, "SphereBVH4D");
  m.def("create_bvh4d", &create_bvh_nd<double, 4>);
  m.def("create_bvh_quat", &create_bvh_quatplus<double, 4>);
  m.def("bvh_mindist4d", &bvh_mindist<double, 4>);
  m.def("bvh_mindist4d_naive", &bvh_mindist_naive<double, 4>);
}
}  // namespace bvh
}  // namespace rpxdock
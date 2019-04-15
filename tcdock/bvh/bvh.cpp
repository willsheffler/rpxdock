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

#include "tcdock/bvh/bvh.hpp"
#include "tcdock/util/Timer.hpp"
#include "tcdock/util/assertions.hpp"
#include "tcdock/util/global_rng.hpp"
#include "tcdock/util/numeric.hpp"
#include "tcdock/util/types.hpp"

using namespace pybind11::literals;
using namespace Eigen;
using namespace tcdock;
using namespace util;
using namespace geom;
using namespace bvh;

namespace py = pybind11;

namespace Eigen {

template <class F>
struct PtIdx {
  PtIdx() : pos(0), idx(0) {}
  PtIdx(V3<F> v, int i = 0) : pos(v), idx(i) {}
  V3<F> pos;
  int idx;
};

auto bounding_vol(V3<float> v) { return Sphere<float>(v); }
auto bounding_vol(V3<double> v) { return Sphere<double>(v); }
auto bounding_vol(PtIdx<float> v) { return Sphere<float>(v.pos); }
auto bounding_vol(PtIdx<double> v) { return Sphere<double>(v.pos); }
}  // namespace Eigen

template <typename F>
using BVH = tcdock::bvh::WelzlBVH<F, PtIdx<F>>;
using BVHf = BVH<float>;
using BVHd = BVH<double>;

namespace tcdock {
template <typename F>
BVH<F> bvh_create(py::array_t<F> xyz) {
  py::buffer_info buf = xyz.request();
  if (buf.ndim != 2 or buf.shape[1] != 3)
    throw std::runtime_error("Shape must be (N, 3)");
  F *ptr = (F *)buf.ptr;

  typedef std::vector<PtIdx<F>, aligned_allocator<PtIdx<F>>> Holder;
  Holder holder;
  for (int i = 0; i < buf.shape[0]; ++i) {
    F x = ptr[3 * i + 0];
    F y = ptr[3 * i + 1];
    F z = ptr[3 * i + 2];
    // std::cout << "add point " << x << " " << y << " " << z << std::endl;
    holder.push_back(PtIdx<F>(V3<F>(x, y, z), i));
  }
  return BVH<F>(holder.begin(), holder.end());
}

template <typename F>
struct BVHMinDistQuery {
  using Scalar = F;
  using Xform = X3<F>;
  int idx1 = -1, idx2 = -1;
  Xform bXa = Xform::Identity();
  F minval = 9e9;
  BVHMinDistQuery(Xform x = Xform::Identity()) : bXa(x) {}
  F minimumOnVolumeVolume(Sphere<F> r1, Sphere<F> r2) {
    return r1.signdis(bXa * r2);
  }
  F minimumOnVolumeObject(Sphere<F> r, PtIdx<F> v) {
    return r.signdis(bXa * v.pos);
  }
  F minimumOnObjectVolume(PtIdx<F> v, Sphere<F> r) {
    return (bXa * r).signdis(v.pos);
  }
  F minimumOnObjectObject(PtIdx<F> a, PtIdx<F> b) {
    F v = (a.pos - bXa * b.pos).norm();
    if (v < minval) {
      // std::cout << v << a.pos.transpose() << " " << b.pos.transpose()
      // << std::endl;
      minval = v;
      idx1 = a.idx;
      idx2 = b.idx;
    }
    return v;
  }
};

template <typename F>
py::tuple bvh_min_dist_fixed(BVH<F> &bvh1, BVH<F> &bvh2) {
  BVHMinDistQuery<F> minimizer;
  auto result = tcdock::bvh::BVMinimize(bvh1, bvh2, minimizer);
  return py::make_tuple(result, minimizer.idx1, minimizer.idx2);
}
template <typename F>
py::tuple bvh_min_dist(BVH<F> &bvh1, BVH<F> &bvh2, M4<F> pos1, M4<F> pos2) {
  X3<F> x1(pos1), x2(pos2);
  BVHMinDistQuery<F> minimizer(x1.inverse() * x2);
  auto result = tcdock::bvh::BVMinimize(bvh1, bvh2, minimizer);
  return py::make_tuple(result, minimizer.idx1, minimizer.idx2);
}
template <typename F>
F naive_min_dist_fixed(BVH<F> &bvh1, BVH<F> &bvh2) {
  F mind2 = 9e9;
  for (auto o1 : bvh1.objs) {
    for (auto o2 : bvh2.objs) {
      mind2 = std::min<F>(mind2, (o1.pos - o2.pos).squaredNorm());
    }
  }
  return std::sqrt(mind2);
}
template <typename F>
F naive_min_dist(BVH<F> &bvh1, BVH<F> &bvh2, M4<F> pos1, M4<F> pos2) {
  X3<F> x1(pos1), x2(pos2);
  X3<F> pos = x1.inverse() * x2;
  F mind2 = 9e9;
  for (auto o1 : bvh1.objs) {
    for (auto o2 : bvh2.objs) {
      mind2 = std::min<F>(mind2, (o1.pos - pos * o2.pos).squaredNorm());
    }
  }
  return std::sqrt(mind2);
}

template <typename F>
struct BVHIsectQuery {
  using Scalar = F;
  using Xform = X3<F>;
  BVHIsectQuery(F r, Xform x = Xform::Identity())
      : rad(r), rad2(r * r), bXa(x) {}
  bool intersectVolumeVolume(Sphere<F> r1, Sphere<F> r2) {
    return r1.signdis(bXa * r2) < rad;
  }
  bool intersectVolumeObject(Sphere<F> r, PtIdx<F> v) {
    return r.signdis(bXa * v.pos) < rad;
  }
  bool intersectObjectVolume(PtIdx<F> v, Sphere<F> r) {
    return (bXa * r).signdis(v.pos) < rad;
  }
  bool intersectObjectObject(PtIdx<F> v1, PtIdx<F> v2) {
    bool isect = (v1.pos - bXa * v2.pos).squaredNorm() < rad2;
    // bool isect = (v1.pos - bXa * v2.pos).norm() < rad;
    result |= isect;
    return isect;
  }
  F rad = 0, rad2 = 0;
  bool result = false;
  Xform bXa = Xform::Identity();
};
template <typename F>
bool bvh_isect_fixed(BVH<F> &bvh1, BVH<F> &bvh2, F thresh) {
  BVHIsectQuery<F> query(thresh);
  tcdock::bvh::BVIntersect(bvh1, bvh2, query);
  return query.result;
}
template <typename F>
bool naive_isect_fixed(BVH<F> &bvh1, BVH<F> &bvh2, F thresh) {
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
bool bvh_isect(BVH<F> &bvh1, BVH<F> &bvh2, M4<F> pos1, M4<F> pos2, F mindist) {
  X3<F> x1(pos1), x2(pos2);
  BVHIsectQuery<F> query(mindist, x1.inverse() * x2);
  tcdock::bvh::BVIntersect(bvh1, bvh2, query);
  return query.result;
}
template <typename F>
bool naive_isect(BVH<F> &bvh1, BVH<F> &bvh2, M4<F> pos1, M4<F> pos2,
                 F mindist) {
  X3<F> x1(pos1), x2(pos2);
  X3<F> pos = x1.inverse() * x2;
  F dist2 = mindist * mindist;

  // bounding sphere check
  auto vol1 = bvh1.getVolume(bvh1.getRootIndex());
  auto vol2 = bvh1.getVolume(bvh2.getRootIndex());
  vol1.cen = x1 * vol1.cen;
  vol2.cen = x2 * vol2.cen;
  if (!vol1.contact(vol2, mindist)) return false;

  // all pairs
  for (auto o1 : bvh1.objs) {
    for (auto o2 : bvh2.objs) {
      auto d2 = (o1.pos - pos * o2.pos).squaredNorm();
      if (d2 < dist2) return true;
    }
  }
  return false;
}

template <typename F>
struct BVMinAxis {
  using Scalar = F;
  using Xform = X3<F>;
  Xform bXa = Xform::Identity();
  F minval = 9e9;
  F rad;
  V3<F> dirn;
  BVMinAxis(V3<F> d, Xform x, F r) : dirn(d), bXa(x), rad(r) {
    assert(r > 0);
    dirn.normalize();
  }
  F minimumOnVolumeVolume(Sphere<F> r1, Sphere<F> r2) {
    return get_slide(r1.cen, bXa * r2.cen, r1.rad + rad, r2.rad + rad);
  }
  F minimumOnVolumeObject(Sphere<F> r, PtIdx<F> v) {
    return get_slide(r.cen, bXa * v.pos, r.rad + rad, rad);
  }
  F minimumOnObjectVolume(PtIdx<F> v, Sphere<F> r) {
    return get_slide(v.pos, bXa * r.cen, rad, r.rad + rad);
  }
  F minimumOnObjectObject(PtIdx<F> a, PtIdx<F> b) {
    // std::cout << "Obj Obj" << std::endl;
    return get_slide(a.pos, bXa * b.pos, rad, rad);
  }
  F get_slide(V3<F> cen1, V3<F> cen2, F rad1, F rad2) {
    V3<F> hypot_start = cen2 - cen1;
    F d_parallel_start = hypot_start.dot(dirn);
    F d_perpendicular_sq = hypot_start.squaredNorm() - square(d_parallel_start);
    F d_hypot_stop_sq = square(rad1 + rad2);
    if (d_hypot_stop_sq < d_perpendicular_sq) return 9e9;  // miss
    F d_parallel_stop = std::sqrt(d_hypot_stop_sq - d_perpendicular_sq);
    F moveby = d_parallel_start - d_parallel_stop;
    // V3<F> cnew = cen1 + moveby * dirn;
    // F dnew = (cnew - cen2).norm();
    return moveby;
  }
};

template <typename F>
F bvh_slide(BVH<F> &bvh1, BVH<F> &bvh2, M4<F> pos1, M4<F> pos2, F rad,
            V3<F> dirn) {
  X3<F> x1(pos1), x2(pos2);
  X3<F> x1inv = x1.inverse();
  X3<F> pos = x1inv * x2;
  V3<F> local_dir = x1inv.rotation() * dirn;
  BVMinAxis<F> query(local_dir, pos, rad);
  F result = tcdock::bvh::BVMinimize(bvh1, bvh2, query);
  return result;
}

template <typename F>
struct PPCollect {
  using Scalar = F;
  using Xform = X3<F>;
  PPCollect(F r, Xform x = Xform::Identity()) : rad(r), bXa(x) {}
  bool intersectVolumeVolume(Sphere<F> r1, Sphere<F> r2) {
    return r1.signdis(bXa * r2) < rad;
  }
  bool intersectVolumeObject(Sphere<F> r, PtIdx<F> v) {
    return r.signdis(bXa * v.pos) < rad;
  }
  bool intersectObjectVolume(PtIdx<F> v, Sphere<F> r) {
    return (bXa * r).signdis(v.pos) < rad;
  }
  bool intersectObjectObject(PtIdx<F> v1, PtIdx<F> v2) {
    bool isect = (v1.pos - bXa * v2.pos).norm() < rad;
    return false;
  }
  F rad = 0.0;
  bool result = false;
  Xform bXa = Xform::Identity();
};

PYBIND11_MODULE(bvh, m) {
  py::class_<BVHf>(m, "WelzlBVH_float");
  py::class_<BVHd>(m, "WelzlBVH_double");

  m.def("bvh_create", &bvh_create<double>);
  m.def("bvh_create_32bit", &bvh_create<float>);

  m.def("bvh_min_dist", &bvh_min_dist<double>, "min pair distance", "bvh1"_a,
        "bvh2"_a, "pos1"_a, "pos2"_a);
  m.def("bvh_min_dist_32bit", &bvh_min_dist<float>, "intersction test",
        "bvh1"_a, "bvh2"_a, "pos1"_a, "pos2"_a);
  m.def("bvh_min_dist_fixed", &bvh_min_dist_fixed<double>);
  m.def("naive_min_dist", &naive_min_dist<double>);
  m.def("naive_min_dist_fixed", &naive_min_dist_fixed<double>);

  m.def("bvh_isect", &bvh_isect<double>, "intersction test", "bvh1"_a, "bvh2"_a,
        "pos1"_a, "pos2"_a, "mindist"_a);
  m.def("bvh_isect_32bit", &bvh_isect<float>, "intersction test", "bvh1"_a,
        "bvh2"_a, "pos1"_a, "pos2"_a, "mindist"_a);
  m.def("bvh_isect_fixed", &bvh_isect_fixed<double>);
  m.def("naive_isect", &naive_isect<double>);
  m.def("naive_isect_fixed", &naive_isect_fixed<double>);

  m.def("bvh_slide", &bvh_slide<double>, "slide into contact", "bvh1"_a,
        "bvh2"_a, "pos1"_a, "pos2"_a, "rad"_a, "dirn"_a);

  m.def("bvh_slide_32bit", &bvh_slide<float>, "slide into contact", "bvh1"_a,
        "bvh2"_a, "pos1"_a, "pos2"_a, "rad"_a, "dirn"_a);
}

}  // namespace tcdock
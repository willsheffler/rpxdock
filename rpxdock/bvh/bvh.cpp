/*cppimport
<%
cfg['include_dirs'] = ['../..','../extern']
cfg['compiler_args'] = ['-std=c++17', '-w', '-Ofast']
cfg['dependencies'] = ['../geom/primitive.hpp','../util/assertions.hpp',
'../util/global_rng.hpp', 'bvh.hpp', 'bvh_algo.hpp', '../util/numeric.hpp',
'../util/pybind_types.hpp']

cfg['parallel'] = False
setup_pybind11(cfg)
%>
*/

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "iostream"

#include "rpxdock/bvh/bvh.hpp"
#include "rpxdock/util/Timer.hpp"
#include "rpxdock/util/assertions.hpp"
#include "rpxdock/util/global_rng.hpp"
#include "rpxdock/util/numeric.hpp"
#include "rpxdock/util/pybind_types.hpp"
#include "rpxdock/util/types.hpp"

using namespace pybind11::literals;
using namespace Eigen;
using namespace rpxdock;
using namespace util;
using namespace geom;

namespace py = pybind11;

namespace Eigen {

template <class F>
struct PtIdx {
  PtIdx() : pos(0), idx(0) {}
  PtIdx(V3<F> v, int i = 0) : pos(v), idx(i) {}
  V3<F> pos;
  int idx;
};
template <typename F>
auto bounding_vol(V3<F> v) {
  return Sphere<F>(v);
}
template <typename F>
auto bounding_vol(PtIdx<F> v) {
  auto s = Sphere<F>(v.pos);
  s.lb = s.ub = v.idx;
  return s;
}
}  // namespace Eigen

template <typename F>
using BVH = rpxdock::bvh::SphereBVH<F, PtIdx<F>>;
using BVHf = BVH<float>;
using BVHd = BVH<double>;

namespace rpxdock {
namespace bvh {

using namespace rpxdock::util;

template <typename F>
BVH<F> bvh_create(py::array_t<F> coords, py::array_t<bool> which) {
  py::buffer_info buf = coords.request();
  if (buf.ndim != 2 || buf.shape[1] != 3)
    throw std::runtime_error("'coords' shape must be (N, 3)");
  // std::cout << "strides " << sizeof(F) << " " << buf.strides[0] << " "
  // << buf.strides[1] << std::endl;
  F *ptr = (F *)buf.ptr;

  py::buffer_info bufw = which.request();
  if (bufw.ndim != 1 || (bufw.size > 0 && bufw.size != buf.shape[0]))
    throw std::runtime_error("'which' shape must be (N,) matching coord shape");
  bool *ptrw = (bool *)bufw.ptr;

  py::gil_scoped_release release;

  typedef std::vector<PtIdx<F>, aligned_allocator<PtIdx<F>>> Objs;
  Objs holder;

  int stride0 = buf.strides[0] / sizeof(F);
  int stride1 = buf.strides[1] / sizeof(F);

  for (int i = 0; i < buf.shape[0]; ++i) {
    // std::cout << "mask " << i << " " << ptrw[i] << std::endl;
    if (bufw.size > 0 && !ptrw[i]) continue;
    F x = ptr[stride0 * i + 0 * stride1];
    F y = ptr[stride0 * i + 1 * stride1];
    F z = ptr[stride0 * i + 2 * stride1];
    // std::cout << "add point " << i << " " << x << " " << y << " " << z
    // << std::endl;
    holder.push_back(PtIdx<F>(V3<F>(x, y, z), i));
  }
  return BVH<F>(holder.begin(), holder.end());
}

template <typename F>
struct BVHMinDistOne {
  using Scalar = F;
  int idx = -1;
  F minval = 9e9;
  V3<F> pt;
  BVHMinDistOne(V3<F> p) : pt(p) {}
  F minimumOnVolume(Sphere<F> r) { return r.signdis(pt); }
  F minimumOnObject(PtIdx<F> a) {
    F v = (a.pos - pt).norm();
    if (v < minval) {
      minval = v;
      idx = a.idx;
    }
    return v;
  }
};
template <typename F>
py::tuple bvh_min_dist_one(BVH<F> &bvh, V3<F> pt) {
  int idx;
  F result;
  {
    py::gil_scoped_release release;
    BVHMinDistOne<F> minimizer(pt);
    result = rpxdock::bvh::BVMinimize(bvh, minimizer);
    idx = minimizer.idx;
  }
  return py::make_tuple(result, idx);
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
  auto result = rpxdock::bvh::BVMinimize(bvh1, bvh2, minimizer);
  return py::make_tuple(result, minimizer.idx1, minimizer.idx2);
}
template <typename F>
py::tuple bvh_min_dist(BVH<F> &bvh1, BVH<F> &bvh2, M4<F> pos1, M4<F> pos2) {
  int idx1, idx2;
  F result;
  {
    py::gil_scoped_release release;
    X3<F> x1(pos1), x2(pos2);
    BVHMinDistQuery<F> minimizer(x1.inverse() * x2);
    result = rpxdock::bvh::BVMinimize(bvh1, bvh2, minimizer);
    idx1 = minimizer.idx1;
    idx2 = minimizer.idx2;
  }
  return py::make_tuple(result, idx1, idx2);
}
template <typename F>
py::tuple bvh_min_dist_vec(BVH<F> &bvh1, BVH<F> &bvh2, py::array_t<F> pos1,
                           py::array_t<F> pos2) {
  auto x1 = xform_py_to_eigen(pos1);
  auto x2 = xform_py_to_eigen(pos2);
  if (x1.size() != x2.size())
    throw std::runtime_error("pos1 and pos2 must have same length");
  auto mindis = std::make_unique<Vx<F>>();
  auto idx1 = std::make_unique<Vx<int>>();
  auto idx2 = std::make_unique<Vx<int>>();
  {
    py::gil_scoped_release release;
    mindis->resize(x1.size());
    idx1->resize(x1.size());
    idx2->resize(x1.size());
    for (size_t i = 0; i < x1.size(); ++i) {
      BVHMinDistQuery<F> minimizer(x1[i].inverse() * x2[i]);
      (*mindis)[i] = rpxdock::bvh::BVMinimize(bvh1, bvh2, minimizer);
      (*idx1)[i] = minimizer.idx1;
      (*idx2)[i] = minimizer.idx2;
    }
  }
  return py::make_tuple(*mindis, *idx1, *idx2);
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
  rpxdock::bvh::BVIntersect(bvh1, bvh2, query);
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
  py::gil_scoped_release release;
  X3<F> x1(pos1), x2(pos2);
  BVHIsectQuery<F> query(mindist, x1.inverse() * x2);
  rpxdock::bvh::BVIntersect(bvh1, bvh2, query);
  return query.result;
}
template <typename F>
Vx<bool> bvh_isect_vec(BVH<F> &bvh1, BVH<F> &bvh2, py::array_t<F> pos1,
                       py::array_t<F> pos2, F mindist) {
  auto x1 = xform_py_to_eigen(pos1);
  auto x2 = xform_py_to_eigen(pos2);
  if (x1.size() != x2.size() && x1.size() != 1 && x2.size() != 1)
    throw std::runtime_error("pos1 and pos2 must have same length");
  py::gil_scoped_release release;
  size_t n = std::max(x1.size(), x2.size());
  Vx<bool> out(n);
  for (size_t i = 0; i < n; ++i) {
    size_t i1 = x1.size() == 1 ? 0 : i;
    size_t i2 = x2.size() == 1 ? 0 : i;
    BVHIsectQuery<F> query(mindist, x1[i1].inverse() * x2[i2]);
    rpxdock::bvh::BVIntersect(bvh1, bvh2, query);
    out[i] = query.result;
  }
  return out;
}
template <typename F>
bool naive_isect(BVH<F> &bvh1, BVH<F> &bvh2, M4<F> pos1, M4<F> pos2,
                 F mindist) {
  X3<F> x1(pos1), x2(pos2);
  X3<F> pos = x1.inverse() * x2;
  F dist2 = mindist * mindist;

  // bounding sphere check
  auto vol1 = bvh1.getVolume(bvh1.getRootIndex());
  auto vol2 = bvh2.getVolume(bvh2.getRootIndex());
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
struct BVHIsectRange {
  using Scalar = F;
  using Xform = X3<F>;
  F rad = 0, rad2 = 0;
  int lb = 0, ub, mid;
  int minrange = 0, max_lb = -1, min_ub = 0;
  Xform bXa = Xform::Identity();
  BVHIsectRange(F r, Xform x, int _ub, int _mid = -1, int maxtrim = -1,
                int maxtrim_lb = -1, int maxtrim_ub = -1)
      : rad(r), rad2(r * r), bXa(x), mid(_mid), ub(_ub) {
    if (mid < 0) mid = ub / 2;
    max_lb = ub;

    if (maxtrim_lb >= 0 && maxtrim_ub >= 0) {
      maxtrim = maxtrim == -1 ? maxtrim_lb + maxtrim_ub : maxtrim;
      max_lb = maxtrim_lb;
      min_ub = ub - maxtrim_ub;
      mid = (max_lb + min_ub) / 2;
    } else if (maxtrim_ub >= 0) {
      maxtrim = maxtrim == -1 ? maxtrim_ub : maxtrim;
      min_ub = ub - maxtrim_ub;
      mid = 0;
    } else if (maxtrim_lb >= 0) {
      maxtrim = maxtrim == -1 ? maxtrim_lb : maxtrim;
      max_lb = maxtrim_lb;
      mid = ub;
    } else if (maxtrim < 0) {
      maxtrim = ub + 1;
    }
    minrange = ub - maxtrim;
  }
  bool intersectVolumeVolume(Sphere<F> r1, Sphere<F> r2) {
    if (r1.lb > ub || r1.ub < lb) return false;
    return r1.signdis(bXa * r2) < rad;
  }
  bool intersectVolumeObject(Sphere<F> r, PtIdx<F> v) {
    if (r.lb > ub || r.ub < lb) return false;
    return r.signdis(bXa * v.pos) < rad;
  }
  bool intersectObjectVolume(PtIdx<F> v, Sphere<F> r) {
    if (v.idx > ub || v.idx < lb) return false;
    return (bXa * r).signdis(v.pos) < rad;
  }
  bool intersectObjectObject(PtIdx<F> v1, PtIdx<F> v2) {
    bool isect = (v1.pos - bXa * v2.pos).squaredNorm() < rad2;
    if (isect) {
      if (v1.idx < mid)
        lb = std::max(v1.idx + 1, lb);
      else
        ub = std::min(v1.idx - 1, ub);
      bool ok = (ub >= min_ub) && (lb <= max_lb) && ((ub - lb) >= minrange);
      if (!ok) {
        lb = -1;
        ub = -1;
        return true;
      }
    }
    return false;
  }
};
template <typename F>
py::tuple isect_range_single(BVH<F> &bvh1, BVH<F> &bvh2, M4<F> pos1, M4<F> pos2,
                             F mindist, int maxtrim = -1, int maxtrim_lb = -1,
                             int maxtrim_ub = -1) {
  int lb, ub;
  {
    py::gil_scoped_release release;
    X3<F> x1(pos1), x2(pos2);
    BVHIsectRange<F> query(mindist, x1.inverse() * x2,
                           (int)bvh1.objs.size() - 1, -1, maxtrim, maxtrim_lb,
                           maxtrim_ub);
    rpxdock::bvh::BVIntersect(bvh1, bvh2, query);
    lb = query.lb;
    ub = query.ub;
  }
  return py::make_tuple(lb, ub);
}
template <typename F>
py::tuple isect_range(BVH<F> &bvh1, BVH<F> &bvh2, py::array_t<F> pos1,
                      py::array_t<F> pos2, F mindist, int maxtrim = -1,
                      int maxtrim_lb = -1, int maxtrim_ub = -1) {
  auto x1 = xform_py_to_eigen(pos1);
  auto x2 = xform_py_to_eigen(pos2);
  if (x1.size() != x2.size() && x1.size() != 1 && x2.size() != 1)
    throw std::runtime_error("pos1 and pos2 must have same length");
  auto lb = std::make_unique<Vx<int>>();
  auto ub = std::make_unique<Vx<int>>();
  {
    py::gil_scoped_release release;
    size_t n = std::max(x1.size(), x2.size());
    lb->resize(n);
    ub->resize(n);
    for (size_t i = 0; i < n; ++i) {
      size_t i1 = x1.size() == 1 ? 0 : i;
      size_t i2 = x2.size() == 1 ? 0 : i;
      BVHIsectRange<F> query(mindist, x1[i1].inverse() * x2[i2],
                             (int)bvh1.objs.size() - 1, -1, maxtrim, maxtrim_lb,
                             maxtrim_ub);
      rpxdock::bvh::BVIntersect(bvh1, bvh2, query);
      (*lb)[i] = query.lb;
      (*ub)[i] = query.ub;
    }
  }
  return py::make_tuple(*lb, *ub);
}
template <typename F>
py::tuple naive_isect_range(BVH<F> &bvh1, BVH<F> &bvh2, M4<F> pos1, M4<F> pos2,
                            F mindist) {
  X3<F> x1(pos1), x2(pos2);
  X3<F> pos = x1.inverse() * x2;
  F dist2 = mindist * mindist;
  int lb = 0, ub = (int)bvh1.objs.size() - 1, mid = ub / 2;

  for (auto o1 : bvh1.objs) {
    for (auto o2 : bvh2.objs) {
      // query.intersectObjectObject(o1, o2);

      auto d2 = (o1.pos - pos * o2.pos).squaredNorm();
      if (d2 < dist2) {
        if (o1.idx < mid)
          lb = std::max(o1.idx + 1, lb);
        else
          ub = std::min(o1.idx - 1, ub);
      }
    }
  }
  // return py::make_tuple(query.lb, query.ub);
  return py::make_tuple(lb, ub);
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
    if (dirn.norm() < 0.0001)
      throw std::runtime_error("Slide direction must not be 0");
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
  py::gil_scoped_release release;
  X3<F> x1(pos1), x2(pos2);
  X3<F> x1inv = x1.inverse();
  X3<F> pos = x1inv * x2;
  V3<F> local_dir = x1inv.rotation() * dirn;
  BVMinAxis<F> query(local_dir, pos, rad);
  F result = rpxdock::bvh::BVMinimize(bvh1, bvh2, query);
  return result;
}
template <typename F>
Vx<F> bvh_slide_vec(BVH<F> &bvh1, BVH<F> &bvh2, py::array_t<F> pos1,
                    py::array_t<F> pos2, F rad, V3<F> dirn) {
  auto x1 = xform_py_to_eigen(pos1);
  auto x2 = xform_py_to_eigen(pos2);
  if (x1.size() != x2.size())
    throw std::runtime_error("pos1 and pos2 must have same len");
  py::gil_scoped_release release;
  Vx<F> slides(x1.size());
  for (size_t i = 0; i < x1.size(); ++i) {
    X3<F> x1inv = x1[i].inverse();
    X3<F> pos = x1inv * x2[i];
    V3<F> local_dir = x1inv.rotation() * dirn;
    BVMinAxis<F> query(local_dir, pos, rad);
    slides[i] = rpxdock::bvh::BVMinimize(bvh1, bvh2, query);
  }
  return slides;
}

template <typename F>
struct BVHCountPairs {
  using Scalar = F;
  using Xform = X3<F>;
  F mindis = 0.0, mindis2 = 0.0;
  Xform bXa = Xform::Identity();
  int nout = 0;
  BVHCountPairs(F mind, Xform x) : mindis(mind), bXa(x), mindis2(mind * mind) {}
  bool intersectVolumeVolume(Sphere<F> v1, Sphere<F> v2) {
    return v1.signdis(bXa * v2) < mindis;
  }
  bool intersectVolumeObject(Sphere<F> v, PtIdx<F> o) {
    return v.signdis(bXa * o.pos) < mindis;
  }
  bool intersectObjectVolume(PtIdx<F> o, Sphere<F> v) {
    return (bXa * v).signdis(o.pos) < mindis;
  }
  bool intersectObjectObject(PtIdx<F> o1, PtIdx<F> o2) {
    bool isect = (o1.pos - bXa * o2.pos).squaredNorm() < mindis2;
    if (isect) nout++;
    return false;
  }
};

template <typename F>
int bvh_count_pairs(BVH<F> &bvh1, BVH<F> &bvh2, M4<F> pos1, M4<F> pos2,
                    F maxdist) {
  py::gil_scoped_release release;
  X3<F> x1(pos1), x2(pos2);
  X3<F> pos = x1.inverse() * x2;
  BVHCountPairs<F> query(maxdist, pos);
  rpxdock::bvh::BVIntersect(bvh1, bvh2, query);
  return query.nout;
}
template <typename F>
Vx<int> bvh_count_pairs_vec(BVH<F> &bvh1, BVH<F> &bvh2, py::array_t<F> pos1,
                            py::array_t<F> pos2, F maxdist) {
  auto x1 = xform_py_to_eigen(pos1);
  auto x2 = xform_py_to_eigen(pos2);
  if (x1.size() != x2.size() && x1.size() != 1 && x2.size() != 1)
    throw std::runtime_error("pos1 and pos2 must have same length");
  py::gil_scoped_release release;
  size_t n = std::max(x1.size(), x2.size());
  Vx<int> npair(n);
  for (size_t i = 0; i < x1.size(); ++i) {
    size_t i1 = x1.size() == 1 ? 0 : i;
    size_t i2 = x2.size() == 1 ? 0 : i;
    X3<F> pos = x1[i1].inverse() * x2[i2];
    BVHCountPairs<F> query(maxdist, pos);
    rpxdock::bvh::BVIntersect(bvh1, bvh2, query);
    npair[i] = query.nout;
  }
  return npair;
}
template <typename F>
struct BVHCollectPairs {
  using Scalar = F;
  using Xform = X3<F>;
  F mindis = 0.0, mindis2 = 0.0;
  Xform bXa = Xform::Identity();
  int32_t *out;
  int nout = 0, maxout = -1;
  bool overflow = false;
  BVHCollectPairs(F r, Xform x, int32_t *ptr, int mx)
      : mindis(r), bXa(x), out(ptr), maxout(mx), mindis2(r * r) {}
  bool intersectVolumeVolume(Sphere<F> v1, Sphere<F> v2) {
    return v1.signdis(bXa * v2) < mindis;
  }
  bool intersectVolumeObject(Sphere<F> v, PtIdx<F> o) {
    return v.signdis(bXa * o.pos) < mindis;
  }
  bool intersectObjectVolume(PtIdx<F> o, Sphere<F> v) {
    return (bXa * v).signdis(o.pos) < mindis;
  }
  bool intersectObjectObject(PtIdx<F> o1, PtIdx<F> o2) {
    bool isect = (o1.pos - bXa * o2.pos).squaredNorm() < mindis2;
    if (isect) {
      if (nout < maxout) {
        // std::cout << "BVH " << nout << " " << o1.idx << " " << o2.idx
        // << std::endl;
        out[2 * nout + 0] = o1.idx;
        out[2 * nout + 1] = o2.idx;
        ++nout;
      } else {
        overflow = true;
      }
    }
    return false;
  }
};

template <typename F>
py::tuple bvh_collect_pairs(BVH<F> &bvh1, BVH<F> &bvh2, M4<F> pos1, M4<F> pos2,
                            F maxdist, py::array_t<int32_t> out) {
  X3<F> x1(pos1), x2(pos2);
  X3<F> pos = x1.inverse() * x2;

  py::buffer_info buf = out.request();
  int nbuf = buf.shape[0];
  if (buf.ndim != 2 || buf.shape[1] != 2)
    throw std::runtime_error("Shape must be (N, 2)");
  if (buf.strides[0] != 2 * sizeof(int32_t))
    throw std::runtime_error("out stride is not 2F");
  int32_t *ptr = (int32_t *)buf.ptr;
  int nout;
  bool overflow;
  {
    py::gil_scoped_release release;
    BVHCollectPairs<F> query(maxdist, pos, ptr, nbuf);
    rpxdock::bvh::BVIntersect(bvh1, bvh2, query);
    nout = query.nout;
    overflow = query.overflow;
  }
  return py::make_tuple(out[py::slice(0, nout, 1)], overflow);
}

template <typename F>
int naive_collect_pairs(BVH<F> &bvh1, BVH<F> &bvh2, M4<F> pos1, M4<F> pos2,
                        F maxdist, py::array_t<int32_t> out) {
  X3<F> x1(pos1), x2(pos2);
  X3<F> pos = x1.inverse() * x2;
  F dist2 = maxdist * maxdist;

  py::buffer_info buf = out.request();
  int nbuf = buf.shape[0], nout = 0;
  if (buf.ndim != 2 || buf.shape[1] != 2)
    throw std::runtime_error("Shape must be (N, 2)");
  int32_t *ptr = (int32_t *)buf.ptr;

  // std::cout << "foo" << bvh1.getRootIndex() << " " << bvh1.objs.size()
  // << std::endl;
  // std::cout << "foo" << bvh2.getRootIndex() << " " << bvh2.objs.size()
  // << std::endl;

  // bounding sphere check
  int ridx1 = bvh1.getRootIndex(), ridx2 = bvh2.getRootIndex();
  if (ridx1 >= 0 && ridx2 >= 0) {
    auto vol1 = bvh1.getVolume(ridx1);
    auto vol2 = bvh1.getVolume(ridx2);
    vol1.cen = x1 * vol1.cen;
    vol2.cen = x2 * vol2.cen;
    if (!vol1.contact(vol2, maxdist)) return 0;
  }

  for (auto o1 : bvh1.objs) {
    for (auto o2 : bvh2.objs) {
      auto d2 = (o1.pos - pos * o2.pos).squaredNorm();
      if (d2 < dist2) {
        assert(nout < nbuf);
        // std::cout << "NAI " << nout << " " << o1.idx << " " << o2.idx
        // << std::endl;

        ptr[2 * nout + 0] = o1.idx;
        ptr[2 * nout + 1] = o2.idx;
        ++nout;
      }
    }
  }

  return nout;
}

template <typename F>
struct BVHCollectPairsVec {
  using Scalar = F;
  using Xform = X3<F>;
  F maxdis = 0.0, maxdis2 = 0.0;
  Xform bXa = Xform::Identity();
  std::vector<int32_t> &out;
  BVHCollectPairsVec(F r, Xform x, std::vector<int32_t> &o)
      : maxdis(r), bXa(x), out(o), maxdis2(r * r) {}
  bool intersectVolumeVolume(Sphere<F> v1, Sphere<F> v2) {
    return v1.signdis(bXa * v2) < maxdis;
  }
  bool intersectVolumeObject(Sphere<F> v, PtIdx<F> o) {
    return v.signdis(bXa * o.pos) < maxdis;
  }
  bool intersectObjectVolume(PtIdx<F> o, Sphere<F> v) {
    return (bXa * v).signdis(o.pos) < maxdis;
  }
  bool intersectObjectObject(PtIdx<F> o1, PtIdx<F> o2) {
    bool isect = (o1.pos - bXa * o2.pos).squaredNorm() < maxdis2;
    if (isect) {
      out.push_back(o1.idx);
      out.push_back(o2.idx);
    }
    return false;
  }
};

template <typename F, typename XF>
py::tuple bvh_collect_pairs_vec(BVH<F> &bvh1, BVH<F> &bvh2,
                                py::array_t<XF> pos1, py::array_t<XF> pos2,
                                F maxdist) {
  auto x1 = xform_py_to_eigen(pos1);
  auto x2 = xform_py_to_eigen(pos2);
  if (x1.size() != x2.size() && x1.size() != 1 && x2.size() != 1)
    throw std::runtime_error("pos1 and pos2 must have same len");

  auto lbub = std::make_unique<Matrix<int, Dynamic, 2, RowMajor>>();
  auto out = std::make_unique<Mx<int32_t>>();
  {
    py::gil_scoped_release release;
    size_t n = std::max(x1.size(), x2.size());
    lbub->resize(n, 2);
    std::vector<int32_t> pairs;
    pairs.reserve(10 * n);
    for (size_t i = 0; i < n; ++i) {
      size_t i1 = x1.size() == 1 ? 0 : i;
      size_t i2 = x2.size() == 1 ? 0 : i;
      X3<F> pos = (x1[i1].inverse() * x2[i2]).template cast<F>();
      BVHCollectPairsVec<F> query(maxdist, pos, pairs);
      (*lbub)(i, 0) = pairs.size() / 2;
      rpxdock::bvh::BVIntersect(bvh1, bvh2, query);
      (*lbub)(i, 1) = pairs.size() / 2;
    }
    out->resize(pairs.size() / 2, 2);
    for (size_t i = 0; i < pairs.size() / 2; ++i) {
      (*out)(i, 0) = pairs[2 * i + 0];
      (*out)(i, 1) = pairs[2 * i + 1];
    }
  }
  return py::make_tuple(*out, *lbub);
}

template <typename F>
struct BVHCollectPairsRangeVec {
  using Scalar = F;
  using Xform = X3<F>;
  F d = 0.0, d2 = 0.0;
  Xform bXa = Xform::Identity();
  int lb1, ub1, lb2, ub2;
  std::vector<int32_t> &out;
  BVHCollectPairsRangeVec(F r, Xform x, int l1, int u1, int l2, int u2,
                          std::vector<int32_t> &o)
      : d(r), bXa(x), lb1(l1), ub1(u1), lb2(l2), ub2(u2), out(o), d2(r * r) {}
  bool intersectVolumeVolume(Sphere<F> v1, Sphere<F> v2) {
    if (v1.ub < lb1 || v1.lb > ub1 || v2.ub < lb2 || v2.lb > ub2) return false;
    return v1.signdis(bXa * v2) < d;
  }
  bool intersectVolumeObject(Sphere<F> v, PtIdx<F> o) {
    if (v.ub < lb1 || v.lb > ub1 || o.idx < lb2 || o.idx > ub2) return false;
    return v.signdis(bXa * o.pos) < d;
  }
  bool intersectObjectVolume(PtIdx<F> o, Sphere<F> v) {
    if (o.idx < lb1 || o.idx > ub1 || v.ub < lb2 || v.lb > ub2) return false;
    return (bXa * v).signdis(o.pos) < d;
  }
  bool intersectObjectObject(PtIdx<F> o1, PtIdx<F> o2) {
    if (o1.idx < lb1 || o1.idx > ub1 || o2.idx < lb2 || o2.idx > ub2)
      return false;
    bool isect = (o1.pos - bXa * o2.pos).squaredNorm() < d2;
    if (isect) {
      out.push_back(o1.idx);
      out.push_back(o2.idx);
    }
    return false;
  }
};

template <typename F, typename XF>
py::tuple bvh_collect_pairs_range_vec(BVH<F> &bvh1, BVH<F> &bvh2,
                                      py::array_t<XF> pos1,
                                      py::array_t<XF> pos2, F maxdist,
                                      Vx<int> lb1, Vx<int> ub1, Vx<int> lb2,
                                      Vx<int> ub2) {
  auto x1 = xform_py_to_eigen(pos1);
  auto x2 = xform_py_to_eigen(pos2);
  if (x1.size() != x2.size() && x1.size() != 1 && x2.size() != 1)
    throw std::runtime_error("pos1/pos2 must be broadcastable");
  if (x1.size() != lb1.size() && x1.size() != 1 && lb1.size() != 1)
    throw std::runtime_error("pos1/lb1 must be broadcastable");
  if (x2.size() != lb2.size() && x2.size() != 1 && lb2.size() != 1)
    throw std::runtime_error("pos2/lb2 must be broadcastable");
  if (lb1.size() != lb2.size() && lb1.size() != 1 && lb2.size() != 1)
    throw std::runtime_error("lb1/lb2 must be broadcastable");
  if (lb1.size() != ub1.size() && lb1.size() != 1 && ub1.size() != 1)
    throw std::runtime_error("lb1/ub1 must be broadcastable");
  if (lb2.size() != ub2.size() && lb2.size() != 1 && ub2.size() != 1)
    throw std::runtime_error("lb2/ub2 must be broadcastable");

  auto lbub = std::make_unique<Matrix<int, Dynamic, 2, RowMajor>>();
  auto out = std::make_unique<Mx<int32_t>>();
  {
    py::gil_scoped_release release;
    size_t n = std::max(std::max(std::max(x1.size(), x2.size()),
                                 std::max(lb1.size(), ub1.size())),
                        std::max(lb2.size(), ub2.size()));
    size_t n0 = std::min(std::min(std::min(x1.size(), x2.size()),
                                  std::min(lb1.size(), ub1.size())),
                         std::min(lb2.size(), ub2.size()));
    n = n0 ? n : 0;
    lbub->resize(n, 2);
    std::vector<int32_t> pairs;
    pairs.reserve(10 * n);
    for (size_t i = 0; i < n; ++i) {
      size_t ix1 = x1.size() == 1 ? 0 : i;
      size_t ix2 = x2.size() == 1 ? 0 : i;
      int l1 = lb1.size() == 1 ? lb1[0] : lb1[i];
      int l2 = lb2.size() == 1 ? lb2[0] : lb2[i];
      int u1 = ub1.size() == 1 ? ub1[0] : ub1[i];
      int u2 = ub2.size() == 1 ? ub2[0] : ub2[i];
      X3<F> pos = (x1[ix1].inverse() * x2[ix2]).template cast<F>();
      BVHCollectPairsRangeVec<F> query(maxdist, pos, l1, u1, l2, u2, pairs);
      (*lbub)(i, 0) = pairs.size() / 2;
      rpxdock::bvh::BVIntersect(bvh1, bvh2, query);
      (*lbub)(i, 1) = pairs.size() / 2;
    }
    out->resize(pairs.size() / 2, 2);
    for (size_t i = 0; i < pairs.size() / 2; ++i) {
      (*out)(i, 0) = pairs[2 * i + 0];
      (*out)(i, 1) = pairs[2 * i + 1];
    }
  }
  return py::make_tuple(*out, *lbub);
}

template <typename F>
int bvh_print(BVH<F> &bvh) {
  for (auto o : bvh.objs) {
    py::print("BVH PT ", o.idx, o.pos.transpose());
  }
}

template <typename F>
py::array_t<F> bvh_obj_centers(BVH<F> &b) {
  int n = b.objs.size();
  auto shape = std::vector<int>{n, 4};
  py::array_t<F> out(shape);
  py::buffer_info buf = out.request();
  F *ptr = (F *)buf.ptr;
  for (int i = 0; i < n; ++i) {
    ptr[4 * i + 0] = b.objs[i].pos[0];
    ptr[4 * i + 1] = b.objs[i].pos[1];
    ptr[4 * i + 2] = b.objs[i].pos[2];
    ptr[4 * i + 3] = 1;
  }
  return out;
}
template <typename F>
V4<F> bvh_obj_com(BVH<F> &b) {
  py::gil_scoped_release release;
  int n = b.objs.size();
  V4<F> com(0, 0, 0, 1);
  for (int i = 0; i < n; ++i) {
    com[0] += b.objs[i].pos[0];
    com[1] += b.objs[i].pos[1];
    com[2] += b.objs[i].pos[2];
  }
  com /= n;
  return com;
}

template <typename F>
py::tuple BVH_get_state(BVH<F> const &bvh) {
  Vx<int> child(bvh.child.size());
  for (int i = 0; i < bvh.child.size(); ++i) child[i] = bvh.child[i];
  Mx<F> sph(bvh.vols.size(), 4);
  for (int i = 0; i < bvh.vols.size(); ++i) {
    for (int j = 0; j < 3; ++j) sph(i, j) = bvh.vols[i].cen[j];
    sph(i, 3) = bvh.vols[i].rad;
  }
  Mx<int> lbub(bvh.vols.size(), 2);
  for (int i = 0; i < bvh.vols.size(); ++i) {
    lbub(i, 0) = bvh.vols[i].lb;
    lbub(i, 1) = bvh.vols[i].ub;
  }
  Mx<F> pos(bvh.objs.size(), 3);
  for (int i = 0; i < bvh.objs.size(); ++i) {
    for (int j = 0; j < 3; ++j) pos(i, j) = bvh.objs[i].pos[j];
  }
  Vx<int> idx(bvh.objs.size());
  for (int i = 0; i < bvh.objs.size(); ++i) idx[i] = bvh.objs[i].idx;
  return py::make_tuple(child, sph, lbub, pos, idx);
}
template <typename F>
std::unique_ptr<BVH<F>> bvh_set_state(py::tuple state) {
  auto bvh = std::make_unique<BVH<F>>();
  auto child = state[0].cast<Vx<int>>();
  auto sph = state[1].cast<Mx<F>>();
  auto lbub = state[2].cast<Mx<int>>();
  auto pos = state[3].cast<Mx<F>>();
  auto idx = state[4].cast<Vx<int>>();

  for (int i = 0; i < child.size(); ++i) {
    bvh->child.push_back(child[i]);
  }
  for (int i = 0; i < sph.rows(); ++i) {
    Sphere<F> sphere;
    for (int j = 0; j < 3; ++j) sphere.cen[j] = sph(i, j);
    sphere.rad = sph(i, 3);
    sphere.lb = lbub(i, 0);
    sphere.ub = lbub(i, 1);
    bvh->vols.push_back(sphere);
  }
  for (int i = 0; i < idx.size(); ++i) {
    PtIdx<F> pt;
    for (int j = 0; j < 3; ++j) pt.pos[j] = pos(i, j);
    pt.idx = idx[i];
    bvh->objs.push_back(pt);
  }
  return bvh;
}

template <typename F>
void bind_bvh(auto m, std::string name) {
  py::class_<BVH<F>>(m, name.c_str())
      .def("__len__", [](BVH<F> &b) { return b.objs.size(); })
      .def("radius", [](BVH<F> &b) { return b.vols[b.getRootIndex()].rad; })
      .def("center", [](BVH<F> &b) { return b.vols[b.getRootIndex()].cen; })
      .def("centers", &bvh_obj_centers<F>)
      .def("com", &bvh_obj_com<F>)
      .def(py::pickle([](const BVH<F> &bvh) { return BVH_get_state<F>(bvh); },
                      [](py::tuple t) { return bvh_set_state<F>(t); }))
      /**/;
}

PYBIND11_MODULE(bvh, m) {
  // bind_bvh<float>(m, "SphereBVH_float");
  bind_bvh<double>(m, "SphereBVH_double");

  m.def("bvh_create", &bvh_create<double>, "make_bvh", "coords"_a,
        "which"_a = py::array_t<bool>());
  // m.def("bvh_create_32bit", &bvh_create<float>, "make_bvh", "coords"_a,
  // "which"_a = py::array_t<bool>());

  m.def("bvh_min_dist", &bvh_min_dist<double>, "min pair distance", "bvh1"_a,
        "bvh2"_a, "pos1"_a, "pos2"_a);
  m.def("bvh_min_dist_vec", &bvh_min_dist_vec<double>, "min pair distance",
        "bvh1"_a, "bvh2"_a, "pos1"_a, "pos2"_a);
  // m.def("bvh_min_dist_32bit", &bvh_min_dist<float>, "intersction test",
  // "bvh1"_a, "bvh2"_a, "pos1"_a, "pos2"_a);
  m.def("bvh_min_dist_fixed", &bvh_min_dist_fixed<double>);
  m.def("naive_min_dist", &naive_min_dist<double>);
  m.def("naive_min_dist_fixed", &naive_min_dist_fixed<double>);

  m.def("bvh_isect", &bvh_isect<double>, "intersction test", "bvh1"_a, "bvh2"_a,
        "pos1"_a, "pos2"_a, "mindist"_a);
  m.def("bvh_isect_vec", &bvh_isect_vec<double>, "intersction test", "bvh1"_a,
        "bvh2"_a, "pos1"_a, "pos2"_a, "mindist"_a);
  // m.def("bvh_isect_32bit", &bvh_isect<float>, "intersction test", "bvh1"_a,
  // "bvh2"_a, "pos1"_a, "pos2"_a, "mindist"_a);
  m.def("bvh_isect_fixed", &bvh_isect_fixed<double>);
  m.def("naive_isect", &naive_isect<double>);
  m.def("naive_isect_fixed", &naive_isect_fixed<double>);

  m.def("isect_range_single", &isect_range_single<double>, "intersction test",
        "bvh1"_a, "bvh2"_a, "pos1"_a, "pos2"_a, "mindist"_a, "maxtrim"_a = -1,
        "maxtrim_lb"_a = -1, "maxtrim_ub"_a = -1);
  m.def("isect_range", &isect_range<double>, "intersction test", "bvh1"_a,
        "bvh2"_a, "pos1"_a, "pos2"_a, "mindist"_a, "maxtrim"_a = -1,
        "maxtrim_lb"_a = -1, "maxtrim_ub"_a = -1);
  m.def("naive_isect_range", &naive_isect_range<double>, "intersction test",
        "bvh1"_a, "bvh2"_a, "pos1"_a, "pos2"_a, "mindist"_a);

  m.def("bvh_slide", &bvh_slide<double>, "slide into contact", "bvh1"_a,
        "bvh2"_a, "pos1"_a, "pos2"_a, "rad"_a, "dirn"_a);
  m.def("bvh_slide_vec", &bvh_slide_vec<double>, "slide into contact", "bvh1"_a,
        "bvh2"_a, "pos1"_a, "pos2"_a, "rad"_a, "dirn"_a);

  // m.def("bvh_slide_32bit", &bvh_slide<float>, "slide into contact",
  // "bvh1"_a, "bvh2"_a, "pos1"_a, "pos2"_a, "rad"_a, "dirn"_a);

  m.def("bvh_collect_pairs", &bvh_collect_pairs<double>);
  m.def("bvh_collect_pairs_vec", &bvh_collect_pairs_vec<double, float>);
  m.def("bvh_collect_pairs_vec", &bvh_collect_pairs_vec<double, double>);
  m.def("naive_collect_pairs", &naive_collect_pairs<double>);
  m.def("bvh_count_pairs", &bvh_count_pairs<double>);
  m.def("bvh_count_pairs_vec", &bvh_count_pairs_vec<double>);

  m.def("bvh_print", &bvh_print<double>);

  m.def("bvh_min_dist_one", &bvh_min_dist_one<double>);

  Vx<int> lb0(1), ub0(1);
  lb0[0] = NL<int>::min();
  ub0[0] = NL<int>::max();
  m.def("bvh_collect_pairs_range_vec",
        &bvh_collect_pairs_range_vec<double, float>, "bvh1"_a, "bvh2"_a,
        "pos1"_a, "pos2"_a, "maxdist"_a, "lb1"_a = lb0, "ub1"_a = ub0,
        "lb2"_a = lb0, "ub2"_a = ub0);
  m.def("bvh_collect_pairs_range_vec",
        &bvh_collect_pairs_range_vec<double, double>, "bvh1"_a, "bvh2"_a,
        "pos1"_a, "pos2"_a, "maxdist"_a, "lb1"_a = lb0, "ub1"_a = ub0,
        "lb2"_a = lb0, "ub2"_a = ub0);
}

}  // namespace bvh
}  // namespace rpxdock
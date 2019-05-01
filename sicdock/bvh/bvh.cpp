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
using BVH = sicdock::bvh::SphereBVH<F, PtIdx<F>>;
using BVHf = BVH<float>;
using BVHd = BVH<double>;

namespace sicdock {
namespace bvh {

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
  BVHMinDistOne<F> minimizer(pt);
  auto result = sicdock::bvh::BVMinimize(bvh, minimizer);
  return py::make_tuple(result, minimizer.idx);
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
  auto result = sicdock::bvh::BVMinimize(bvh1, bvh2, minimizer);
  return py::make_tuple(result, minimizer.idx1, minimizer.idx2);
}
template <typename F>
py::tuple bvh_min_dist(BVH<F> &bvh1, BVH<F> &bvh2, M4<F> pos1, M4<F> pos2) {
  X3<F> x1(pos1), x2(pos2);
  BVHMinDistQuery<F> minimizer(x1.inverse() * x2);
  auto result = sicdock::bvh::BVMinimize(bvh1, bvh2, minimizer);
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
  sicdock::bvh::BVIntersect(bvh1, bvh2, query);
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
  sicdock::bvh::BVIntersect(bvh1, bvh2, query);
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
  BVHIsectRange(F r, Xform x, int _ub)
      : rad(r), rad2(r * r), bXa(x), mid(_ub / 2), ub(_ub) {}
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
    if (isect)
      if (v1.idx < mid)
        lb = std::max(v1.idx, lb);
      else
        ub = std::min(v1.idx, ub);
    return false;
  }
  F rad = 0, rad2 = 0;
  int lb = 0, ub, mid;
  Xform bXa = Xform::Identity();
};
template <typename F>
py::tuple bvh_isect_range(BVH<F> &bvh1, BVH<F> &bvh2, M4<F> pos1, M4<F> pos2,
                          F mindist) {
  X3<F> x1(pos1), x2(pos2);
  BVHIsectRange<F> query(mindist, x1.inverse() * x2, bvh1.objs.size());
  sicdock::bvh::BVIntersect(bvh1, bvh2, query);
  return py::make_tuple(query.lb, query.ub);
}
template <typename F>
py::tuple naive_isect_range(BVH<F> &bvh1, BVH<F> &bvh2, M4<F> pos1, M4<F> pos2,
                            F mindist) {
  X3<F> x1(pos1), x2(pos2);
  X3<F> pos = x1.inverse() * x2;
  F dist2 = mindist * mindist;
  int lb = 0, ub = bvh1.objs.size(), mid = ub / 2;
  for (auto o1 : bvh1.objs) {
    for (auto o2 : bvh2.objs) {
      auto d2 = (o1.pos - pos * o2.pos).squaredNorm();
      if (d2 < dist2) {
        if (o1.idx < mid)
          lb = std::max(o1.idx, lb);
        else
          ub = std::min(o1.idx, ub);
      }
    }
  }
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
  X3<F> x1(pos1), x2(pos2);
  X3<F> x1inv = x1.inverse();
  X3<F> pos = x1inv * x2;
  V3<F> local_dir = x1inv.rotation() * dirn;
  BVMinAxis<F> query(local_dir, pos, rad);
  F result = sicdock::bvh::BVMinimize(bvh1, bvh2, query);
  return result;
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
  X3<F> x1(pos1), x2(pos2);
  X3<F> pos = x1.inverse() * x2;
  BVHCountPairs<F> query(maxdist, pos);
  sicdock::bvh::BVIntersect(bvh1, bvh2, query);
  return query.nout;
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
int bvh_collect_pairs(BVH<F> &bvh1, BVH<F> &bvh2, M4<F> pos1, M4<F> pos2,
                      F maxdist, py::array_t<int32_t> out) {
  X3<F> x1(pos1), x2(pos2);
  X3<F> pos = x1.inverse() * x2;

  py::buffer_info buf = out.request();
  int nbuf = buf.shape[0], nout = 0;
  if (buf.ndim != 2 || buf.shape[1] != 2)
    throw std::runtime_error("Shape must be (N, 2)");
  if (buf.strides[0] != 2 * sizeof(int32_t))
    throw std::runtime_error("out stride is not 2F");
  int32_t *ptr = (int32_t *)buf.ptr;

  BVHCollectPairs<F> query(maxdist, pos, ptr, buf.shape[0]);
  sicdock::bvh::BVIntersect(bvh1, bvh2, query);
  if (query.overflow) return -1;
  return query.nout;
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
int bvh_print(BVH<F> &bvh) {
  for (auto o : bvh.objs) {
    std::cout << "BVH PT " << o.idx << " " << o.pos.transpose() << std::endl;
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
  VectorX<int> child(bvh.child.size());
  for (int i = 0; i < bvh.child.size(); ++i) child[i] = bvh.child[i];
  RowMajorX<F> sph(bvh.vols.size(), 4);
  for (int i = 0; i < bvh.vols.size(); ++i) {
    for (int j = 0; j < 3; ++j) sph(i, j) = bvh.vols[i].cen[j];
    sph(i, 3) = bvh.vols[i].rad;
  }
  RowMajorX<int> lbub(bvh.vols.size(), 2);
  for (int i = 0; i < bvh.vols.size(); ++i) {
    lbub(i, 0) = bvh.vols[i].lb;
    lbub(i, 1) = bvh.vols[i].ub;
  }
  RowMajorX<F> pos(bvh.objs.size(), 3);
  for (int i = 0; i < bvh.objs.size(); ++i) {
    for (int j = 0; j < 3; ++j) pos(i, j) = bvh.objs[i].pos[j];
  }
  VectorX<int> idx(bvh.objs.size());
  for (int i = 0; i < bvh.objs.size(); ++i) idx[i] = bvh.objs[i].idx;

  return py::make_tuple(child, sph, lbub, pos, idx);
}
template <typename F>
std::unique_ptr<BVH<F>> bvh_set_state(py::tuple state) {
  auto bvh = std::make_unique<BVH<F>>();
  auto child = state[0].cast<VectorX<int>>();
  auto sph = state[1].cast<RowMajorX<F>>();
  auto lbub = state[2].cast<RowMajorX<int>>();
  auto pos = state[3].cast<RowMajorX<F>>();
  auto idx = state[4].cast<VectorX<int>>();

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
  // m.def("bvh_min_dist_32bit", &bvh_min_dist<float>, "intersction test",
  // "bvh1"_a, "bvh2"_a, "pos1"_a, "pos2"_a);
  m.def("bvh_min_dist_fixed", &bvh_min_dist_fixed<double>);
  m.def("naive_min_dist", &naive_min_dist<double>);
  m.def("naive_min_dist_fixed", &naive_min_dist_fixed<double>);

  m.def("bvh_isect", &bvh_isect<double>, "intersction test", "bvh1"_a, "bvh2"_a,
        "pos1"_a, "pos2"_a, "mindist"_a);
  // m.def("bvh_isect_32bit", &bvh_isect<float>, "intersction test", "bvh1"_a,
  // "bvh2"_a, "pos1"_a, "pos2"_a, "mindist"_a);
  m.def("bvh_isect_fixed", &bvh_isect_fixed<double>);
  m.def("naive_isect", &naive_isect<double>);
  m.def("naive_isect_fixed", &naive_isect_fixed<double>);

  m.def("bvh_isect_range", &bvh_isect_range<double>, "intersction test",
        "bvh1"_a, "bvh2"_a, "pos1"_a, "pos2"_a, "mindist"_a);
  m.def("naive_isect_range", &naive_isect_range<double>);

  m.def("bvh_slide", &bvh_slide<double>, "slide into contact", "bvh1"_a,
        "bvh2"_a, "pos1"_a, "pos2"_a, "rad"_a, "dirn"_a);

  // m.def("bvh_slide_32bit", &bvh_slide<float>, "slide into contact",
  // "bvh1"_a, "bvh2"_a, "pos1"_a, "pos2"_a, "rad"_a, "dirn"_a);

  m.def("bvh_collect_pairs", &bvh_collect_pairs<double>);
  m.def("bvh_count_pairs", &bvh_count_pairs<double>);
  m.def("naive_collect_pairs", &naive_collect_pairs<double>);

  m.def("bvh_print", &bvh_print<double>);

  m.def("bvh_min_dist_one", &bvh_min_dist_one<double>);
}

}  // namespace bvh
}  // namespace sicdock
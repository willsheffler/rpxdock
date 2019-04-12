// based on Eigen unsupported bvh KDTree

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <algorithm>
#include <queue>

#include "bvh_algo.hpp"

#include "rif/geom/primitive.hpp"

namespace rif {
namespace geom {

// internal pair class for the BVH--used instead of std::pair because of
// alignment
template <class F>
struct V3intPair {
  using first_type = V3<F>;
  using secont_type = int;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW_IF_VECTORIZABLE_FIXED_SIZE(F, 3)
  V3intPair(const V3<F> &v, int i) : first(v), second(i) {}
  V3<F> first;
  int second;
};

// these templates help the tree initializer get the bounding spheres either in
// from a provided iterator range or using bounding_vol in a unified way
template <typename Objs, typename Vols, typename Iter>
struct get_bvols_helper {
  void operator()(const Objs &objs, Iter sphbeg, Iter sphend, Vols &out) {
    out.insert(out.end(), sphbeg, sphend);
    eigen_assert(out.size() == objs.size());
  }
};

template <typename Objs, typename Vols>
struct get_bvols_helper<Objs, Vols, int> {
  void operator()(const Objs &objs, int, int, Vols &out) {
    out.reserve(objs.size());
    for (int i = 0; i < (int)objs.size(); ++i)
      out.push_back(bounding_vol(objs[i]));
  }
};

template <class RAiter>
struct P1Range {
  using value_type = typename RAiter::value_type::first_type;
  RAiter a, b;
  auto const &operator[](size_t i) const { return (a + i)->first; }
  auto &operator[](size_t i) { return (a + i)->first; }
  size_t size() const { return b - a; }
};
template <class RAiter>
auto p1range(RAiter a, RAiter b) {
  P1Range<RAiter> r;
  r.a = a;
  r.b = b;
  return r;
}

template <typename _Scalar, typename _Object>
class WelzlBVH {
 public:
  static int const Dim = 3;
  typedef _Object Object;
  typedef std::vector<Object, Eigen::aligned_allocator<Object> > Objs;
  typedef _Scalar Scalar;
  // typedef Eigen::AlignedBox<Scalar, Dim> Volume;
  typedef Sphere<Scalar> Volume;
  typedef std::vector<Volume, Eigen::aligned_allocator<Volume> > Vols;
  typedef int Index;
  typedef const int *VolumeIterator;  // the iterators are just pointers into
                                      // the tree's vectors
  typedef const Object *ObjectIterator;

  std::vector<int> child;  // child of x are child[2x] and
                           // child[2x+1], indices bigger than
                           // vols.size() index into objs.
  Vols vols;
  Objs objs;

  WelzlBVH() {}

  template <typename Iter>
  WelzlBVH(Iter begin, Iter end) {
    init(begin, end, 0, 0);
  }  // int is recognized by init as not being an iterator type

  template <typename OIter, typename BIter>
  WelzlBVH(OIter begin, OIter end, BIter sphbeg, BIter sphend) {
    init(begin, end, sphbeg, sphend);
  }

  /** Given an iterator range over \a Object references, constructs the BVH,
   * overwriting whatever is in there currently.
    * Requires that bounding_vol(Object) return a Volume. */
  template <typename Iter>
  void init(Iter begin, Iter end) {
    init(begin, end, 0, 0);
  }

  /** Given an iterator range over \a Object references and an iterator range
   * over their bounding vols,
    * constructs the BVH, overwriting whatever is in there currently. */
  template <typename OIter, typename BIter>
  void init(OIter begin, OIter end, BIter sphbeg, BIter sphend) {
    objs.clear();
    vols.clear();
    child.clear();

    objs.insert(objs.end(), begin, end);
    int n = static_cast<int>(objs.size());

    // if we have at most one object, we don't need any internal nodes
    if (n < 2) return;

    Vols ovol;
    VIPairs ocen;

    // compute the bounding vols depending on BIter type
    get_bvols_helper<Objs, Vols, BIter>()(objs, sphbeg, sphend, ovol);

    ocen.reserve(n);
    vols.reserve(n - 1);
    child.reserve(2 * n - 2);

    for (int i = 0; i < n; ++i) ocen.push_back(VIPair(ovol[i].center, i));

    // the recursive part of the algorithm
    build(ocen, 0, n, ovol, 0);

    Objs tmp(n);
    tmp.swap(objs);
    for (int i = 0; i < n; ++i) objs[i] = tmp[ocen[i].second];
  }

  /** \returns the index of the root of the hierarchy */
  inline Index getRootIndex() const { return (int)vols.size() - 1; }

  /** Given an \a index of a node, on exit, \a vbeg and \a vend range
   * over the indices of the volume children of the node
    * and \a obeg and \a oend range over the object children of the node
   */
  EIGEN_STRONG_INLINE
  void getChildren(Index index, VolumeIterator &vbeg, VolumeIterator &vend,
                   ObjectIterator &obeg, ObjectIterator &oend) const {
    // inlining this function should open lots of optimization opportunities to
    // the compiler
    if (index < 0) {
      vbeg = vend;
      if (!objs.empty()) obeg = &(objs[0]);
      oend = obeg + objs.size();  // output all objs--necessary
                                  // when the tree has only one
                                  // object
      return;
    }

    int nvol = static_cast<int>(vols.size());

    int idx = index * 2;
    if (child[idx + 1] < nvol) {  // second index is always bigger
      vbeg = &(child[idx]);
      vend = vbeg + 2;
      obeg = oend;
    } else if (child[idx] >= nvol) {  // if both child are objs
      vbeg = vend;
      obeg = &(objs[child[idx] - nvol]);
      oend = obeg + 2;
    } else {  // if the first child is a volume and the second is an object
      vbeg = &(child[idx]);
      vend = vbeg + 1;
      obeg = &(objs[child[idx + 1] - nvol]);
      oend = obeg + 1;
    }
    // std::cout << " gc" << vend - vbeg;
  }

  inline const Volume &getVolume(Index index) const { return vols[index]; }

 private:
  typedef V3intPair<Scalar> VIPair;
  typedef std::vector<VIPair, Eigen::aligned_allocator<VIPair> > VIPairs;
  typedef Eigen::Matrix<Scalar, Dim, 1> VectorType;

  // template <class Oiter>
  // void get_subtree_leaves(Index idx, Oiter out) {
  //   VolumeIterator vbeg = nullptr, vend = nullptr;
  //   ObjectIterator obeg = nullptr, oend = nullptr;
  //   getChildren(idx, vbeg, vend, obeg, oend);
  //   // std::cout << "gco " << idx << " range: " << vend - vbeg << " vals";
  //   // for (auto i = vbeg; i != vend; ++i) std::cout << " " << *i;
  //   // std::cout << std::endl;
  //   for (auto i = vbeg; i != vend; ++i) get_subtree_leaves(*i, out);
  //   for (auto i = obeg; i != oend; ++i) ++out = *i;
  // }

  // // todo: more efficient to build welzl sets in coodinated way?
  // void welzlize_vols(VIPairs const &ocen) {
  //   std::vector<Object> subleaf;
  //   for (int i = 0; i < vols.size(); ++i) {
  //     subleaf.clear();
  //     get_subtree_leaves(i, std::back_inserter(subleaf));
  //     if (subleaf.size() > 2) {
  //       auto welzl = welzl_bounding_sphere(subleaf);
  //       // few pathological cases w/n=3
  //       if (welzl.radius < vols[i].radius) vols[i] = welzl;
  //     }
  //   }
  // }

  struct AxisComparator {
    int dim;
    AxisComparator(int inDim) : dim(inDim) {}
    inline bool operator()(const VIPair &v1, const VIPair &v2) const {
      return v1.first[dim] < v2.first[dim];
    }
  };
  struct DotComparator {
    V3<Scalar> normal;
    DotComparator(V3<Scalar> n) : normal(n) {}
    template <class Pair>
    DotComparator(Pair p) : normal(p.second - p.first) {}
    inline bool operator()(const VIPair &v1, const VIPair &v2) const {
      return v1.first.dot(normal) < v2.first.dot(normal);
    }
  };

  // Build the part of the tree between objs[from] and objs[to] (not
  // including objs[to]). This routine partitions the ocen in [from, to) along
  // the dimension dim, recursively constructs the two halves, and adds their
  // parent node.  TODO: a cache-friendlier layout
  void build(VIPairs &ocen, int from, int to, Vols const &ovol, int dim) {
    eigen_assert(to - from > 1);
    if (to - from == 2) {
      auto merge = ovol[ocen[from].second].merged(ovol[ocen[from + 1].second]);
      vols.push_back(merge);
      child.push_back(from + (int)objs.size() - 1);
      child.push_back(from + (int)objs.size());
    } else if (to - from == 3) {
      int mid = from + 2;
      auto subtree_objs = p1range(ocen.begin() + from, ocen.begin() + to);
      nth_element(ocen.begin() + from, ocen.begin() + mid, ocen.begin() + to,
                  DotComparator(most_separated_points_on_AABB(subtree_objs)));
      // AxisComparator(dim));
      build(ocen, from, mid, ovol, (dim + 1) % Dim);
      int idx1 = (int)vols.size() - 1;
      auto merge = vols[idx1].merged(ovol[ocen[mid].second]);
      auto welzl = welzl_bounding_sphere(subtree_objs);
      vols.push_back(welzl.radius < merge.radius ? welzl : merge);
      child.push_back(idx1);
      child.push_back(mid + (int)objs.size() - 1);
    } else {
      int mid = from + (to - from) / 2;
      auto subtree_objs = p1range(ocen.begin() + from, ocen.begin() + to);
      nth_element(ocen.begin() + from, ocen.begin() + mid, ocen.begin() + to,
                  DotComparator(most_separated_points_on_AABB(subtree_objs)));
      // AxisComparator(dim));
      build(ocen, from, mid, ovol, (dim + 1) % Dim);
      int idx1 = (int)vols.size() - 1;
      build(ocen, mid, to, ovol, (dim + 1) % Dim);
      int idx2 = (int)vols.size() - 1;
      auto merge = vols[idx1].merged(vols[idx2]);
      auto welzl = welzl_bounding_sphere(subtree_objs);
      vols.push_back(welzl.radius < merge.radius ? welzl : merge);
      child.push_back(idx1);
      child.push_back(idx2);
    }
  }
};
}
}

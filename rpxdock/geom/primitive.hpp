#pragma once
/** \file */

#include <iostream>
#include <vector>

#include "rpxdock/util/types.hpp"

namespace rpxdock {
namespace geom {

using namespace util;

struct Empty {};

template <class F>
class Sphere {
  using Scalar = F;
  using Vec3 = V3<F>;
  using Mat3 = M3<F>;

 public:
  Vec3 cen;
  F rad;
  int lb = 2000000000;
  int ub = -2000000000;

  Sphere() : cen(0, 0, 0), rad(1) {}
  Sphere(Vec3 c, F r) : cen(c), rad(r) {}
  Sphere(Vec3 O) : cen(O), rad(epsilon2<F>()) {}
  Sphere(Vec3 O, Vec3 A) {
    Vec3 a = A - O;
    Vec3 o = 0.5 * a;
    rad = o.norm() + epsilon2<F>();
    cen = O + o;
  }
  Sphere(Vec3 O, Vec3 A, Vec3 B) {
    Vec3 a = A - O, b = B - O;
    F det_2 = 2.0 * ((a.cross(b)).dot(a.cross(b)));
    Vec3 o = (b.dot(b) * ((a.cross(b)).cross(a)) +
              a.dot(a) * (b.cross(a.cross(b)))) /
             det_2;
    rad = o.norm() + epsilon2<F>();
    cen = O + o;
  }
  Sphere(Vec3 O, Vec3 A, Vec3 B, Vec3 C) {
    Vec3 a = A - O, b = B - O, c = C - O;
    Mat3 cols;
    cols.col(0) = a;
    cols.col(1) = b;
    cols.col(2) = c;
    F det_2 = 2.0 * Mat3(cols).determinant();
    Vec3 o = (c.dot(c) * a.cross(b) + b.dot(b) * c.cross(a) +
              a.dot(a) * b.cross(c)) /
             det_2;
    rad = o.norm() + epsilon2<F>();
    cen = O + o;
  }

  Sphere<F> merged(Sphere<F> that) const {
    if (this->contains(that)) return *this;
    if (that.contains(*this)) return that;
    F d = rad + that.rad + (cen - that.cen).norm();
    // std::cout << d << std::endl;
    auto dir = (that.cen - cen).normalized();
    auto c = cen + dir * (d / 2 - this->rad);
    auto out = Sphere<F>(c, d / 2 + epsilon2<F>() / 2.0);
    out.lb = std::min(this->lb, that.lb);
    out.ub = std::max(this->ub, that.ub);
    return out;
  }

  // Distance from p to boundary of the Sphere
  F signdis(Vec3 P) const { return (cen - P).norm() - rad; }
  F signdis2(Vec3 P) const {  // NOT square of signdis!
    return (cen - P).squaredNorm() - rad * rad;
  }
  F signdis(Sphere<F> s) const { return (cen - s.cen).norm() - rad - s.rad; }
  bool intersect(Sphere that) const {
    F rtot = rad + that.rad;
    return (cen - that.cen).squaredNorm() <= rtot;
  }
  bool contact(Sphere that, F contact_dis) const {
    F rtot = rad + that.rad + contact_dis;
    return (cen - that.cen).squaredNorm() <= rtot * rtot;
  }
  bool contains(Vec3 pt) const { return (cen - pt).squaredNorm() < rad * rad; }
  bool contains(Sphere<F> that) const {
    auto d = (cen - that.cen).norm();
    return d + that.rad <= rad;
  }
  bool operator==(Sphere<F> that) const {
    return cen.isApprox(that.cen) && fabs(rad - that.rad) < epsilon2<F>();
  }
};

typedef Sphere<float> Spheref;
typedef Sphere<double> Sphered;

template <class F>
Sphere<F> operator*(X3<F> x, Sphere<F> s) {
  return Sphere<F>(x * s.cen, s.rad);
}

template <class Scalar>
std::ostream& operator<<(std::ostream& out, Sphere<Scalar> const& s) {
  out << "Sphere( " << s.cen.transpose() << ", " << s.rad << ")";
  return out;
}

template <class Ary>
auto welzl_bounding_sphere_impl(Ary const& points, size_t index,
                                std::vector<typename Ary::value_type>& sos,
                                size_t numsos) noexcept {
  using Pt = typename Ary::value_type;
  using Scalar = typename Pt::Scalar;
  using Sph = Sphere<Scalar>;
  // if no input points, the recursion has bottomed out. Now compute an
  // exact sphere based on points in set of support (zero through four points)
  if (index == 0) {
    switch (numsos) {
      case 0:
        return Sph(Pt(0, 0, 0), 0);
      case 1:
        return Sph(sos[0]);
      case 2:
        return Sph(sos[0], sos[1]);
      case 3:
        return Sph(sos[0], sos[1], sos[2]);
      case 4:
        return Sph(sos[0], sos[1], sos[2], sos[3]);
    }
  }
  // Pick a point at "random" (here just the last point of the input set)
  --index;
  // Recursively compute the smallest bounding sphere of the remaining points
  Sph smallestSphere =
      welzl_bounding_sphere_impl(points, index, sos, numsos);  // (*)
  // If the selected point lies inside this sphere, it is indeed the smallest
  if (smallestSphere.contains(points[index])) return smallestSphere;
  // Otherwise, update set of support to additionally contain the new point
  // assert(numsos < 4);
  if (numsos == 4) {
    // oops, numerical errors.... go ahead with what we have
    return smallestSphere;
  }
  sos[numsos] = points[index];
  // Recursively compute the smallest sphere of remaining points with new s.o.s.
  return welzl_bounding_sphere_impl(points, index, sos, numsos + 1);
}

template <class Ary, class Sph, bool range>
struct UpdateBounds {
  static void update_bounds(Ary const& points, Sph& sph) {}
};

template <class Ary, class Sph>
struct UpdateBounds<Ary, Sph, false> {
  static void update_bounds(Ary const& points, Sph& sph) {}
};
template <class Ary, class Sph>
struct UpdateBounds<Ary, Sph, true> {
  static void update_bounds(Ary const& points, Sph& sph) {
    sph.lb = 2000000000;
    sph.ub = -2000000000;
    for (size_t i = 0; i < points.size(); ++i) {
      sph.lb = std::min(sph.lb, points.get_index(i));
      sph.ub = std::max(sph.ub, points.get_index(i));
    }
  }
};

template <bool range = false, class Ary>
auto welzl_bounding_sphere(Ary const& points) noexcept {
  using Pt = typename Ary::value_type;
  using Sph = Sphere<typename Pt::Scalar>;
  std::vector<Pt> sos(4);
  Sph bound = welzl_bounding_sphere_impl(points, points.size(), sos, 0);
  UpdateBounds<Ary, Sph, range>::update_bounds(points, bound);
  // if (bound.lb < 0 || bound.lb > bound.ub || bound.ub > 2000)
  // std::cout << bound.lb << " " << bound.ub << std::endl;
  return bound;
}

template <bool range = false, class Ary>
auto central_bounding_sphere(Ary const& points) noexcept {
  using Pt = typename Ary::value_type;
  using Scalar = typename Pt::Scalar;
  using Sph = Sphere<Scalar>;
  Scalar const eps = epsilon2<Scalar>();
  Pt cen;
  Scalar rad = -1;
  if (points.size() > 0) {
    cen = Pt(0, 0, 0);
    for (size_t i = 0; i < points.size(); i++) cen += points[i];
    cen /= (Scalar)points.size();

    for (size_t i = 0; i < points.size(); i++) {
      Scalar d2 = (points[i] - cen).squaredNorm();
      if (d2 > rad) rad = d2;
    }
    rad = sqrt(rad) + eps;
  }
  Sph bound(cen, rad);
  UpdateBounds<Ary, Sph, range>::update_bounds(points, bound);
  return bound;
}

/**
 * @brief      Compute indices to the two most separated points of the (up to)
 * six points defining the AABB encompassing the point set. Return these as min
 * and max.
 */
template <class Ary>
auto most_separated_points_on_AABB(Ary const& pt) {
  using Pt = typename Ary::value_type;
  using Scalar = typename Pt::Scalar;
  // First find most extreme points along principal axes
  size_t mnx = 0, mxx = 0, mny = 0, mxy = 0, mnz = 0, mxz = 0;
  for (size_t i = 1; i < pt.size(); i++) {
    if (pt[i][0] < pt[mnx][0]) mnx = i;
    if (pt[i][0] > pt[mxx][0]) mxx = i;
    if (pt[i][1] < pt[mny][1]) mny = i;
    if (pt[i][1] > pt[mxy][1]) mxy = i;
    if (pt[i][2] < pt[mnz][2]) mnz = i;
    if (pt[i][2] > pt[mxz][2]) mxz = i;
  }
  // Compute the squared distances for the three pairs of points
  Scalar dist2x = (pt[mxx] - pt[mnx]).dot(pt[mxx] - pt[mnx]);
  Scalar dist2y = (pt[mxy] - pt[mny]).dot(pt[mxy] - pt[mny]);
  Scalar dist2z = (pt[mxz] - pt[mnz]).dot(pt[mxz] - pt[mnz]);
  // Pick the pair (mn,mx) of points most distant
  auto mn = mnx;
  auto mx = mxx;
  if (dist2y > dist2x && dist2y > dist2z) {
    mx = mxy;
    mn = mny;
  }
  if (dist2z > dist2x && dist2z > dist2y) {
    mx = mxz;
    mn = mnz;
  }
  return std::make_pair(pt[mn], pt[mx]);
}

template <typename F, int DIM>
struct SphereND {
  using This = SphereND<F, DIM>;
  using Vn = Eigen::Matrix<F, DIM, 1>;

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

}  // namespace geom
}  // namespace rpxdock

#pragma once

#include <Eigen/Geometry>
#include <iostream>
#include <vector>

namespace tcdock {
namespace geom {

template <class F>
using V3 = Eigen::Matrix<F, 3, 1>;  // todo: should use aligned vector3?
template <class F>
using M3 = Eigen::Matrix<F, 3, 3, Eigen::RowMajor>;  // to match numpy (??)
template <class F>
using X3 = Eigen::Transform<F, 3, Eigen::Affine, Eigen::RowMajor>;

template <class F>
F epsilon2() {
  return std::sqrt(std::numeric_limits<F>::epsilon());
}

template <class Ary>
auto welzl_bounding_sphere(Ary const& points) noexcept;

template <class F>
class Sphere {
  using Scalar = F;
  using Vec3 = V3<F>;
  using Mat3 = M3<F>;

 public:
  Vec3 center;
  F radius;

  Sphere() : center(0, 0, 0), radius(1) {}
  Sphere(Vec3 c, F r) : center(c), radius(r) {}
  Sphere(Vec3 O) : center(O), radius(epsilon2<F>()) {}
  Sphere(Vec3 O, Vec3 A) {
    Vec3 a = A - O;
    Vec3 o = 0.5 * a;
    radius = o.norm() + epsilon2<F>();
    center = O + o;
  }
  Sphere(Vec3 O, Vec3 A, Vec3 B) {
    Vec3 a = A - O, b = B - O;
    F det_2 = 2.0 * ((a.cross(b)).dot(a.cross(b)));
    Vec3 o = (b.dot(b) * ((a.cross(b)).cross(a)) +
              a.dot(a) * (b.cross(a.cross(b)))) /
             det_2;
    radius = o.norm() + epsilon2<F>();
    center = O + o;
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
    radius = o.norm() + epsilon2<F>();
    center = O + o;
  }

  Sphere<F> merged(Sphere<F> that) const {
    if (this->contains(that)) return *this;
    if (that.contains(*this)) return that;
    F d = radius + that.radius + (center - that.center).norm();
    // std::cout << d << std::endl;
    auto dir = (that.center - center).normalized();
    auto c = center + dir * (d / 2 - this->radius);
    return Sphere<F>(c, d / 2 + epsilon2<F>() / 2.0);
  }

  // Distance from p to boundary of the Sphere
  F signdis(Vec3 P) const { return (center - P).norm() - radius; }
  F signdis2(Vec3 P) const {  // NOT square of signdis!
    return (center - P).squaredNorm() - radius * radius;
  }
  F signdis(Sphere<F> s) const {
    return (center - s.center).norm() - radius - s.radius;
  }
  bool intersect(Sphere that) const {
    F rtot = radius + that.radius;
    return (center - that.center).squaredNorm() <= rtot;
  }
  bool contact(Sphere that, F contact_dis) const {
    F rtot = radius + that.radius + contact_dis;
    return (center - that.center).squaredNorm() <= rtot * rtot;
  }
  bool contains(Vec3 pt) const {
    return (center - pt).squaredNorm() < radius * radius;
  }
  bool contains(Sphere<F> that) const {
    auto d = (center - that.center).norm();
    return d + that.radius <= radius;
  }
  bool operator==(Sphere<F> that) const {
    return center.isApprox(that.center) &&
           fabs(radius - that.radius) < epsilon2<F>();
  }
};

typedef Sphere<float> Spheref;
typedef Sphere<double> Sphered;

template <class F>
Sphere<F> operator*(X3<F> x, Sphere<F> s) {
  return Sphere<F>(x * s.center, s.radius);
}

template <class Scalar>
std::ostream& operator<<(std::ostream& out, Sphere<Scalar> const& s) {
  out << "Sphere( " << s.center.transpose() << ", " << s.radius << ")";
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
  assert(numsos < 4);
  sos[numsos] = points[index];
  // Recursively compute the smallest sphere of remaining points with new s.o.s.
  return welzl_bounding_sphere_impl(points, index, sos, numsos + 1);
}

template <class Ary>
auto welzl_bounding_sphere(Ary const& points) noexcept {
  using Pt = typename Ary::value_type;
  // using Scalar = typename Pt::Scalar;
  // using Sph = Sphere<Scalar>;
  std::vector<Pt> sos(4);
  return welzl_bounding_sphere_impl(points, points.size(), sos, 0);
}

template <class Ary>
auto central_bounding_sphere(Ary const& points) noexcept {
  using Pt = typename Ary::value_type;
  using Scalar = typename Pt::Scalar;
  using Sph = Sphere<Scalar>;
  Scalar const eps = epsilon2<Scalar>();
  Pt center;
  Scalar radius = -1;
  if (points.size() > 0) {
    center = Pt(0, 0, 0);
    for (size_t i = 0; i < points.size(); i++) center += points[i];
    center /= (Scalar)points.size();

    for (size_t i = 0; i < points.size(); i++) {
      Scalar d2 = (points[i] - center).squaredNorm();
      if (d2 > radius) radius = d2;
    }
    radius = sqrt(radius) + eps;
  }
  return Sph(center, radius);
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
}  // namespace geom
}  // namespace tcdock

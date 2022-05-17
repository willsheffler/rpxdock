#pragma once

#include <algorithm>
#include <boost/assert.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_arithmetic.hpp>
#include <boost/utility/enable_if.hpp>
#include <cmath>
#include <limits>

#include "rpxdock/pack/types.hh"

#ifdef CEREAL
// #include <boost/serialization/access.hpp>
#include <cereal/access.hpp>
#endif

namespace rpxdock {
namespace pack {

// TODO: finish this and replace Eigen dependency in NEST
namespace impl {
struct NoInit {};
}  // namespace impl

///@brief minimal fixed size array with element-wise operations
///@note used this instead of Eigen Array in NEST to speed compilation by
/// 20%-50%
template <int N, class F = double, bool init0 = false>
struct SimpleArray {
  typedef SimpleArray<N, F> THIS;
  typedef F value_type;
  typedef F Scalar;
  typedef F* iterator;
  typedef F const* const_iterator;
  typedef F& reference;
  typedef F const& const_reference;
  typedef size_t difference_type;
  typedef size_t size_type;
  typedef std::numeric_limits<F> NL;
  F D[N];
  template <class A>
  SimpleArray(A const& a,
              typename boost::disable_if<boost::is_arithmetic<A> >::type* = 0) {
    for (int i = 0; i < N; ++i) D[i] = a[i];
  }
  template <int N2>
  SimpleArray(SimpleArray<N2, F> const& a) {
    BOOST_STATIC_ASSERT((N2 == N));
    for (int i = 0; i < N; ++i) D[i] = a[i];
  }
  template <int N2>
  SimpleArray(SimpleArray<N2, F> const& a, int ofst) {
    for (int i = 0; i < N; ++i) D[i] = a[i + ofst];
  }
  // SimpleArray() { for(size_t i = 0; i < N; ++i) D[i]=0; }
  SimpleArray() {
    if (init0) fill(0);
  }
  // explicit SimpleArray(F const* fp){ for(size_t i = 0; i < N; ++i) D[i] =
  // fp[i]; }
  SimpleArray(F a) { fill(a); }
  SimpleArray(F a, F b) {
    BOOST_STATIC_ASSERT((N == 2));
    D[0] = a;
    D[1] = b;
  }
  SimpleArray(F a, F b, F c) {
    BOOST_STATIC_ASSERT((N == 3));
    D[0] = a;
    D[1] = b;
    D[2] = c;
  }
  SimpleArray(F a, F b, F c, F d) {
    BOOST_STATIC_ASSERT((N == 4));
    D[0] = a;
    D[1] = b;
    D[2] = c;
    D[3] = d;
  }
  SimpleArray(F a, F b, F c, F d, F e) {
    BOOST_STATIC_ASSERT((N == 5));
    D[0] = a;
    D[1] = b;
    D[2] = c;
    D[3] = d;
    D[4] = e;
  }
  SimpleArray(F a, F b, F c, F d, F e, F f) {
    BOOST_STATIC_ASSERT((N == 6));
    D[0] = a;
    D[1] = b;
    D[2] = c;
    D[3] = d;
    D[4] = e;
    D[5] = f;
  }
  SimpleArray(F a, F b, F c, F d, F e, F f, F g) {
    BOOST_STATIC_ASSERT((N == 7));
    D[0] = a;
    D[1] = b;
    D[2] = c;
    D[3] = d;
    D[4] = e;
    D[5] = f;
    D[6] = g;
  }
  SimpleArray(F a, F b, F c, F d, F e, F f, F g, F h) {
    BOOST_STATIC_ASSERT((N == 8));
    D[0] = a;
    D[1] = b;
    D[2] = c;
    D[3] = d;
    D[4] = e;
    D[5] = f;
    D[6] = g;
    D[7] = h;
  }
  SimpleArray(F a, F b, F c, F d, F e, F f, F g, F h, F i) {
    BOOST_STATIC_ASSERT((N == 9));
    D[0] = a;
    D[1] = b;
    D[2] = c;
    D[3] = d;
    D[4] = e;
    D[5] = f;
    D[6] = g;
    D[7] = h;
    D[8] = i;
  }
  SimpleArray(F a, F b, F c, F d, F e, F f, F g, F h, F i, F j) {
    BOOST_STATIC_ASSERT((N == 10));
    D[0] = a;
    D[1] = b;
    D[2] = c;
    D[3] = d;
    D[4] = e;
    D[5] = f;
    D[6] = g;
    D[7] = h;
    D[8] = i;
    D[9] = j;
  }
  F& operator[](size_t i) { return D[i]; }
  F const& operator[](size_t i) const { return D[i]; }
  F& at(size_t i) {
    BOOST_VERIFY(i < N);
    return D[i];
  }
  F const& at(size_t i) const {
    BOOST_VERIFY(i < N);
    return D[i];
  }
  template <class OF>
  SimpleArray<N, OF> cast() const {
    SimpleArray<N, OF> r;
    for (int i = 0; i < N; ++i) r[i] = (*this)[i];
    return r;
  }
  THIS maxCoeff() const {
    F r = NL::min();
    for (int i = 0; i < N; ++i) r = std::max(r, D[i]);
    return r;
  }
  THIS minCoeff() const {
    F r = NL::max();
    for (int i = 0; i < N; ++i) r = std::min(r, D[i]);
    return r;
  }
  template <class I>
  THIS maxCoeff(I* arg) const {
    F r = NL::min();
    for (I i = 0; i < N; ++i) {
      *arg = r < D[i] ? i : *arg;
      r = r < D[i] ? D[i] : r;
    };
    return r;
  }
  template <class I>
  THIS minCoeff(I* arg) const {
    F r = NL::max();
    for (I i = 0; i < N; ++i) {
      *arg = r > D[i] ? i : *arg;
      r = r > D[i] ? D[i] : r;
    };
    return r;
  }
  THIS max(F b) const {
    THIS r(*this);
    for (int i = 0; i < N; ++i) r[i] = std::max(r[i], b);
    return r;
  }
  THIS min(F b) const {
    THIS r(*this);
    for (int i = 0; i < N; ++i) r[i] = std::min(r[i], b);
    return r;
  }
  THIS max(THIS const& b) const {
    THIS r(*this);
    for (int i = 0; i < N; ++i) r[i] = std::max(r[i], b[i]);
    return r;
  }
  THIS min(THIS const& b) const {
    THIS r(*this);
    for (int i = 0; i < N; ++i) r[i] = std::min(r[i], b[i]);
    return r;
  }
  F prod() const {
    F p = 1;
    for (int i = 0; i < N; ++i) p *= D[i];
    return p;
  }
  F sum() const {
    F p = 0;
    for (int i = 0; i < N; ++i) p += D[i];
    return p;
  }
  F prod(size_t l) const {
    F p = 1;
    for (int i = 0; i < l; ++i) p *= D[i];
    return p;
  }
  F sum(size_t l) const {
    F p = 0;
    for (int i = 0; i < l; ++i) p += D[i];
    return p;
  }
  bool operator==(THIS const& o) const {
    bool r = true;
    for (int i = 0; i < N; ++i) r &= D[i] == o.D[i];
    return r;
  }
  bool operator!=(THIS const& o) const { return !(*this == o); }
  F squaredNorm() const {
    F n = 0;
    for (int i = 0; i < N; ++i) n += D[i] * D[i];
    return n;
  }
  F norm() const { return std::sqrt(squaredNorm()); }
  void normalize() { *this /= norm(); }
  void fill(F v) {
    for (int i = 0; i < N; ++i) D[i] = v;
  }
  iterator begin() { return &D[0]; }
  iterator end() { return &D[N]; }
  const_iterator begin() const { return &D[0]; }
  const_iterator end() const { return &D[N]; }
  bool empty() const { return false; }
  size_type size() const { return N; }
  void swap(THIS& o) {
    for (int i = 0; i < N; ++i) std::swap(D[i], o.D[i]);
  }
// friend class boost::serialization::access;
#ifdef CEREAL
  friend class cereal::access;
#endif
  template <class Archive>
  void serialize(Archive& ar, const unsigned int) {
    for (size_t i = 0; i < N; ++i) ar& D[i];
  }
  THIS operator-() const {
    THIS r;
    for (size_t i = 0; i < N; ++i) r[i] = -D[i];
    return r;
  }
  SimpleArray<N, F> sign() const {
    SimpleArray<N, F> r;
    for (int i = 0; i < N; ++i) r[i] = D[i] > 0 ? 1.0 : -1.0;
    return r;
  }
  template <class F2>
  SimpleArray<N, F>& operator*=(F2 const& o) {
    for (int i = 0; i < N; ++i) D[i] *= o;
    return *this;
  }
  template <class F2>
  SimpleArray<N, F>& operator/=(F2 const& o) {
    for (int i = 0; i < N; ++i) D[i] /= o;
    return *this;
  }
  template <class F2>
  SimpleArray<N, F>& operator+=(F2 const& o) {
    for (int i = 0; i < N; ++i) D[i] += o;
    return *this;
  }
  template <class F2>
  SimpleArray<N, F>& operator-=(F2 const& o) {
    for (int i = 0; i < N; ++i) D[i] -= o;
    return *this;
  }
  SimpleArray<N, F>& operator*=(THIS const& o) {
    for (int i = 0; i < N; ++i) D[i] *= o[i];
    return *this;
  }
  SimpleArray<N, F>& operator/=(THIS const& o) {
    for (int i = 0; i < N; ++i) D[i] /= o[i];
    return *this;
  }
  SimpleArray<N, F>& operator+=(THIS const& o) {
    for (int i = 0; i < N; ++i) D[i] += o[i];
    return *this;
  }
  SimpleArray<N, F>& operator-=(THIS const& o) {
    for (int i = 0; i < N; ++i) D[i] -= o[i];
    return *this;
  }
};
template <int N, class F>
std::ostream& operator<<(std::ostream& out, SimpleArray<N, F> const& a) {
  for (int i = 0; i < N; ++i) out << a[i] << " ";
  return out;
}

template <int N, class F>
SimpleArray<N, F> operator+(SimpleArray<N, F> const& a,
                            SimpleArray<N, F> const& b) {
  SimpleArray<N, F> r(a);
  for (int i = 0; i < N; ++i) r[i] += b[i];
  return r;
}
template <int N, class F>
SimpleArray<N, F> operator-(SimpleArray<N, F> const& a,
                            SimpleArray<N, F> const& b) {
  SimpleArray<N, F> r(a);
  for (int i = 0; i < N; ++i) r[i] -= b[i];
  return r;
}
template <int N, class F>
SimpleArray<N, F> operator*(SimpleArray<N, F> const& a,
                            SimpleArray<N, F> const& b) {
  SimpleArray<N, F> r(a);
  for (int i = 0; i < N; ++i) r[i] *= b[i];
  return r;
}
template <int N, class F>
SimpleArray<N, F> operator/(SimpleArray<N, F> const& a,
                            SimpleArray<N, F> const& b) {
  SimpleArray<N, F> r(a);
  for (int i = 0; i < N; ++i) r[i] /= b[i];
  return r;
}
template <int N, class F, class F2>
SimpleArray<N, F> operator+(SimpleArray<N, F> const& a, F2 const& b) {
  SimpleArray<N, F> r(a);
  for (int i = 0; i < N; ++i) r[i] += b;
  return r;
}
template <int N, class F, class F2>
SimpleArray<N, F> operator-(SimpleArray<N, F> const& a, F2 const& b) {
  SimpleArray<N, F> r(a);
  for (int i = 0; i < N; ++i) r[i] -= b;
  return r;
}
template <int N, class F, class F2>
SimpleArray<N, F> operator*(SimpleArray<N, F> const& a, F2 const& b) {
  SimpleArray<N, F> r(a);
  for (int i = 0; i < N; ++i) r[i] *= b;
  return r;
}
template <int N, class F, class F2>
SimpleArray<N, F> operator/(SimpleArray<N, F> const& a, F2 const& b) {
  SimpleArray<N, F> r(a);
  for (int i = 0; i < N; ++i) r[i] /= b;
  return r;
}
template <int N, class F, class F2>
SimpleArray<N, F> operator+(F2 const& b, SimpleArray<N, F> const& a) {
  SimpleArray<N, F> r(a);
  for (int i = 0; i < N; ++i) r[i] += b;
  return r;
}
template <int N, class F, class F2>
SimpleArray<N, F> operator-(F2 const& b, SimpleArray<N, F> const& a) {
  SimpleArray<N, F> r(b);
  for (int i = 0; i < N; ++i) r[i] -= a[i];
  return r;
}
template <int N, class F, class F2>
SimpleArray<N, F> operator*(F2 const& b, SimpleArray<N, F> const& a) {
  SimpleArray<N, F> r(a);
  for (int i = 0; i < N; ++i) r[i] *= b;
  return r;
}
template <int N, class F, class F2>
SimpleArray<N, F> operator/(F2 const& b, SimpleArray<N, F> const& a) {
  SimpleArray<N, F> r(b);
  for (int i = 0; i < N; ++i) r[i] /= a[i];
  return r;
}

template <int N, class F>
SimpleArray<N, F> operator%(SimpleArray<N, F> const& a,
                            SimpleArray<N, F> const& b) {
  SimpleArray<N, F> r(a);
  for (int i = 0; i < N; ++i) r[i] %= b[i];
  return r;
}

// this uint64_t assumption seems a little sketchy....
template <int N, class F, class F2>
SimpleArray<N, uint64_t> operator<(SimpleArray<N, F> const& a, F2 const& b) {
  SimpleArray<N, uint64_t> r;
  for (int i = 0; i < N; ++i) r[i] = a[i] < b;
  return r;
}
template <int N, class F, class F2>
SimpleArray<N, uint64_t> operator<(F2 const& a, SimpleArray<N, F> const& b) {
  SimpleArray<N, uint64_t> r;
  for (int i = 0; i < N; ++i) r[i] = a < b[i];
  return r;
}

template <int N, class F>
SimpleArray<N, uint64_t> operator<(SimpleArray<N, F> const& a,
                                   SimpleArray<N, F> const& b) {
  SimpleArray<N, uint64_t> r;
  for (int i = 0; i < N; ++i) r[i] = a[i] < b[i];
  return r;
}

}  // namespace pack
}  // namespace rpxdock

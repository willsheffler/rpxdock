#pragma once
/** \file */

#include <sstream>
#include <string>

namespace rpxdock {
namespace util {

template <class T>
std::string str(T const &t) {
  std::ostringstream oss;
  oss << t;
  return oss.str();
}

template <class T>
static std::string short_str() {
  return "?";
}
template <>
std::string short_str<float>() {
  return "f4";
}
template <>
std::string short_str<double>() {
  return "f8";
}
template <>
std::string short_str<int32_t>() {
  return "i4";
}
template <>
std::string short_str<int64_t>() {
  return "i8";
}
template <>
std::string short_str<uint32_t>() {
  return "u4";
}
template <>
std::string short_str<uint64_t>() {
  return "u8";
}

template <class T>
static std::string cpp_repr() {
  return "?";
}
template <>
std::string cpp_repr<float>() {
  return "float";
}
template <>
std::string cpp_repr<double>() {
  return "double";
}
template <>
std::string cpp_repr<int32_t>() {
  return "int32_t";
}
template <>
std::string cpp_repr<int64_t>() {
  return "int64_t";
}
template <>
std::string cpp_repr<uint32_t>() {
  return "uint32_t";
}
template <>
std::string cpp_repr<uint64_t>() {
  return "uint64_t";
}
}  // namespace util
}  // namespace rpxdock

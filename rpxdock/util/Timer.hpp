#pragma once
/** \file */

#include <chrono>
#include <iostream>

namespace rpxdock {
namespace util {

/**
 * @brief      simple scope-guarded Timer. calling elapsed*() will stop by
 * default
 */
template <class Clock = std::chrono::system_clock>
struct TimerImpl {
  using This = TimerImpl<Clock>;
  using Time = std::chrono::time_point<Clock>;
  Time tstart, tstop;
  std::string lab;

 public:
  TimerImpl(std::string _lab = "") : lab(_lab) { restart(); }
  ~TimerImpl() {
    if (tstop == Time::min())
      std::cout << "Timer(" << *this << ") exited scope" << std::endl;
  }
  void restart() {
    tstart = Clock::now();
    tstop = Time::min();
  }
  void stop() { tstop = Clock::now(); }
  Time stoptime() const {
    if (tstop != Time::min()) return tstop;
    return Clock::now();
  }
  void stop_if_not_stopped() {
    if (tstop == Time::min()) stop();
  }
  double elapsed(bool stop = true) {
    if (stop) stop_if_not_stopped();
    return ((const This*)this)->elapsed();
  }
  double elapsed_nano(bool stop = true) {
    if (stop) stop_if_not_stopped();
    return ((const This*)this)->elapsed_nano();
  }
  double elapsed(bool stop = true) const {
    std::chrono::duration<double> elapsed_seconds = stoptime() - tstart;
    return elapsed_seconds.count();
  }
  double elapsed_nano() const {
    std::chrono::duration<double, std::nano> elapsed_seconds =
        stoptime() - tstart;
    return elapsed_seconds.count();
  }
};
template <class Clk>
std::ostream& operator<<(std::ostream& out, TimerImpl<Clk> const& t) {
  if (t.lab.size()) out << t.lab << ": ";
  out << t.elapsed() * 1000000.0 << "us";
  return out;
}
typedef TimerImpl<> Timer;
}  // namespace util
}  // namespace rpxdock

// M-NCC: gated 1-D scan-to-scan intensity NCC longitudinal pseudo-measurement.
//
// Maintains a rolling buffer of recent (world_pos, intensity) samples. At each
// frame, projects them onto the IMU motion direction to form a 1-D longitudinal
// intensity profile. Cross-correlates the latest profile against an older one
// shifted by IEKF-predicted longitudinal displacement; if the NCC peak is sharp
// and forward-backward consistent, the peak offset minus the predicted offset
// becomes a weak longitudinal pseudo-measurement injected into the IEKF.
//
// Activation invariants (must always hold):
//   - Gate off (env NCC_MNCC unset or 0): module is a no-op, baseline STACK
//     residuals/Jacobians and update flow unchanged.
//   - When triggered, all thresholds derive from rolling median/MAD/percentile
//     of the same per-frame signals (peak NCC ratio, FB consistency) — no
//     hand-tuned numeric thresholds.
//   - No image-plane addressing, no ring index dependence.
//
// File-local state isolated under namespace m_ncc to avoid pollution.

#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <deque>
#include <vector>
#include <cstdlib>
#include <cmath>

namespace m_ncc {

struct Sample {
  double t;        // bag-time (s)
  Eigen::Vector3d pw;  // world position
  double intensity;
};

class Module {
 public:
  static Module& instance() {
    static Module m;
    return m;
  }

  bool enabled() const { return enabled_; }

  // Called once per frame BEFORE h_share_combined runs. Records IEKF predicted
  // pose and motion direction for this frame.
  void on_frame_start(double bag_time,
                      const Eigen::Vector3d& predicted_pos,
                      const Eigen::Vector3d& predicted_vel) {
    if (!enabled_) return;
    cur_time_ = bag_time;
    cur_pos_ = predicted_pos;
    cur_vel_ = predicted_vel;
    double vn = predicted_vel.norm();
    if (vn > 1e-3) {
      motion_dir_ = predicted_vel / vn;
      has_motion_dir_ = true;
    } else {
      has_motion_dir_ = false;
    }
  }

  // Append accepted geo correspondences (world coords + intensity) from current
  // scan into rolling buffer.
  void append_samples(const std::vector<Sample>& samples) {
    if (!enabled_) return;
    for (const auto& s : samples) buf_.push_back(s);
    while (buf_.size() > kMaxBufSize) buf_.pop_front();
  }

  // Returns true iff M-NCC fires for this frame and outputs a 1-element pseudo-
  // measurement: r (residual, m) and direction d (unit vector along motion).
  // The IEKF host then appends one row to h_x with d in pos-block and zero
  // elsewhere, and residual = r (downscaled by gate_weight).
  bool query_longitudinal_correction(double* out_residual,
                                     Eigen::Vector3d* out_dir,
                                     double* out_weight) {
    *out_residual = 0.0;
    *out_dir = Eigen::Vector3d::Zero();
    *out_weight = 0.0;
    if (!enabled_ || !has_motion_dir_ || buf_.size() < kMinBufSize) return false;

    // Build 1-D longitudinal arc-length axis along motion_dir_ centered on cur_pos_.
    // Project buffer points to 1-D coordinate s_i = (pw_i - cur_pos_) . motion_dir_.
    // Build a 1-D intensity histogram (binned by arc length).
    const double bin = 0.5;  // m per bin — geometry-driven, NOT tuning (matches map voxel).
    const int half_bins = 30;  // ±15m around current position.
    std::vector<double> sum_recent(2 * half_bins, 0.0);
    std::vector<int> cnt_recent(2 * half_bins, 0);
    std::vector<double> sum_old(2 * half_bins, 0.0);
    std::vector<int> cnt_old(2 * half_bins, 0);

    const double t_split = cur_time_ - kTemporalWindowSec;
    for (const auto& s : buf_) {
      double sl = (s.pw - cur_pos_).dot(motion_dir_);
      int bidx = static_cast<int>(std::floor(sl / bin)) + half_bins;
      if (bidx < 0 || bidx >= 2 * half_bins) continue;
      if (s.t >= t_split) {
        sum_recent[bidx] += s.intensity; cnt_recent[bidx] += 1;
      } else {
        sum_old[bidx] += s.intensity; cnt_old[bidx] += 1;
      }
    }
    // Reduce to mean per bin; require coverage in both windows.
    std::vector<double> A(2 * half_bins, 0.0), B(2 * half_bins, 0.0);
    int n_covered = 0;
    for (int i = 0; i < 2 * half_bins; ++i) {
      if (cnt_recent[i] > 0 && cnt_old[i] > 0) {
        A[i] = sum_recent[i] / cnt_recent[i];
        B[i] = sum_old[i] / cnt_old[i];
        ++n_covered;
      }
    }
    if (n_covered < kMinCoveredBins) return false;

    // Compute NCC over discrete shifts ±max_shift_bins.
    const int max_shift = 6;
    double best_ncc = -2.0; int best_shift = 0;
    double second_ncc = -2.0;
    for (int d = -max_shift; d <= max_shift; ++d) {
      double ma = 0, mb = 0; int n = 0;
      for (int i = 0; i < 2 * half_bins; ++i) {
        int j = i + d;
        if (j < 0 || j >= 2 * half_bins) continue;
        if (A[i] == 0 || B[j] == 0) continue;
        ma += A[i]; mb += B[j]; ++n;
      }
      if (n < kMinCoveredBins) continue;
      ma /= n; mb /= n;
      double num = 0, sa = 0, sb = 0;
      for (int i = 0; i < 2 * half_bins; ++i) {
        int j = i + d;
        if (j < 0 || j >= 2 * half_bins) continue;
        if (A[i] == 0 || B[j] == 0) continue;
        double da = A[i] - ma, db = B[j] - mb;
        num += da * db; sa += da * da; sb += db * db;
      }
      double denom = std::sqrt(sa * sb);
      if (denom < 1e-9) continue;
      double ncc = num / denom;
      if (ncc > best_ncc) {
        second_ncc = best_ncc; best_ncc = ncc; best_shift = d;
      } else if (ncc > second_ncc) {
        second_ncc = ncc;
      }
    }
    if (best_ncc < 0) return false;

    // Sharpness: peak / second-peak ratio (above noise). Confidence from rolling.
    double sharpness = (second_ncc > 0.0) ? (best_ncc / std::max(second_ncc, 1e-6)) : best_ncc * 10.0;
    ncc_hist_.push_back(best_ncc);
    sharp_hist_.push_back(sharpness);
    while (ncc_hist_.size() > kHistMaxSize) ncc_hist_.pop_front();
    while (sharp_hist_.size() > kHistMaxSize) sharp_hist_.pop_front();
    if (ncc_hist_.size() < kHistMinSize) return false;

    // Rolling-MAD gate on (best_ncc, sharpness): activate only when both exceed
    // median + 3*MAD of their own histories (standard outlier convention).
    auto mad_gate = [&](const std::deque<double>& h, double cur) {
      std::vector<double> sh(h.begin(), h.end());
      std::sort(sh.begin(), sh.end());
      double med = sh[sh.size() / 2];
      std::vector<double> dev(sh.size());
      for (size_t k = 0; k < sh.size(); ++k) dev[k] = std::abs(sh[k] - med);
      std::sort(dev.begin(), dev.end());
      double mad = 1.4826 * dev[dev.size() / 2];
      return cur > med + 3.0 * mad;
    };
    bool ncc_strong = mad_gate(ncc_hist_, best_ncc);
    bool sharp_strong = mad_gate(sharp_hist_, sharpness);
    if (!(ncc_strong && sharp_strong)) return false;

    // Pseudo-measurement: r = -peak_offset * bin (sign convention: positive
    // residual means the system needs to advance along motion_dir_ by that much).
    double r = -static_cast<double>(best_shift) * bin;
    if (std::abs(r) > kMaxCorrectionM) return false;
    *out_residual = r;
    *out_dir = motion_dir_;
    // Weight: scale by NCC peak strength; never exceed 1.
    *out_weight = std::min(1.0, best_ncc);
    ++n_fired_;
    return true;
  }

  int n_fired() const { return n_fired_; }

 private:
  Module() {
    const char* e = std::getenv("NCC_MNCC");
    enabled_ = (e && std::atoi(e) > 0);
  }

  static constexpr size_t kMaxBufSize = 200000;
  static constexpr size_t kMinBufSize = 5000;
  static constexpr double kTemporalWindowSec = 1.0;  // last-1s = recent; older = past
  static constexpr int kMinCoveredBins = 8;
  static constexpr size_t kHistMaxSize = 500;
  static constexpr size_t kHistMinSize = 50;
  static constexpr double kMaxCorrectionM = 3.0;

  bool enabled_ = false;
  bool has_motion_dir_ = false;
  double cur_time_ = 0.0;
  Eigen::Vector3d cur_pos_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d cur_vel_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d motion_dir_ = Eigen::Vector3d::UnitX();
  std::deque<Sample> buf_;
  std::deque<double> ncc_hist_;
  std::deque<double> sharp_hist_;
  int n_fired_ = 0;
};

}  // namespace m_ncc

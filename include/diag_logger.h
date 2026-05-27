// W1 + Stage 2 instrumentation for cp3 S-A/T-B design.
// Gated by env NCC_DIAG=1; output to NCC_DIAG_OUT (default /tmp/cp3_diag.csv).
// Stage 1: dof split, ||h|| norms, FIM 12 eigenvalues, weak dir
// Stage 2 (NCC_DIAG2=1):
//   - residual percentiles (p25/p50/p75/p95/p99) of |r_geo|
//   - NIS-like statistic: median |r_geo|, MAD, sum(r^2)/(N*median^2)
//   - state pos (x,y,z), vel (x,y,z) for travel-direction alignment
//   - weak-eigenvector dot vel (longitudinal alignment cosine, abs)

#pragma once
#include <Eigen/Dense>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cmath>
#include <algorithm>
#include <vector>

namespace diag {

inline int enabled() {
    static int e = -1;
    if (e < 0) {
        const char* en = std::getenv("NCC_DIAG");
        e = (en && std::string(en) == "1") ? 1 : 0;
    }
    return e;
}

inline int stage2_enabled() {
    static int e = -1;
    if (e < 0) {
        const char* en = std::getenv("NCC_DIAG2");
        e = (en && std::string(en) == "1") ? 1 : 0;
    }
    return e;
}

inline int stage3_enabled() {
    static int e = -1;
    if (e < 0) {
        const char* en = std::getenv("NCC_DIAG3");
        e = (en && std::string(en) == "1") ? 1 : 0;
    }
    return e;
}

inline std::FILE* file() {
    static std::FILE* f = nullptr;
    static bool init = false;
    if (!init) {
        init = true;
        if (!enabled()) return nullptr;
        const char* path = std::getenv("NCC_DIAG_OUT");
        std::string p = path ? path : "/tmp/cp3_diag.csv";
        f = std::fopen(p.c_str(), "w");
        if (f) {
            std::fprintf(f, "frame,n_geo,n_photo,n_total,h_geo_norm,h_photo_norm,hx_norm,"
                            "eig0,eig1,eig2,eig3,eig4,eig5,eig6,eig7,eig8,eig9,eig10,eig11,"
                            "cond,weak_dir_x,weak_dir_y,weak_dir_z,"
                            "weak_rotx,weak_roty,weak_rotz");
            if (stage2_enabled()) {
                std::fprintf(f, ",r_p25,r_p50,r_p75,r_p95,r_p99,r_mad,nis_norm,"
                                "px,py,pz,vx,vy,vz,vnorm,weak_long_cos");
            }
            if (stage3_enabled()) {
                std::fprintf(f, ",bax,bay,baz,bgx,bgy,bgz,ba_norm,bg_norm,"
                                "proj_info_long,proj_info_total,proj_info_frac");
            }
            std::fprintf(f, "\n");
            std::fflush(f);
        }
    }
    return f;
}

// Templated state argument so we don't have to include use_ikfom here.
// State must expose .pos (3-vec) and .vel (3-vec).
template <typename StateT>
void log_frame_s2(int n_photo, int n_geo,
                  const Eigen::VectorXd& h_photo, const Eigen::VectorXd& h_geo,
                  const Eigen::MatrixXd& h_x_combined,
                  const StateT& s) {
    std::FILE* f = file();
    if (!f) return;
    static int frame = 0;
    int n_total = h_x_combined.rows();
    double h_geo_n = h_geo.size() > 0 ? h_geo.norm() : 0.0;
    double h_photo_n = h_photo.size() > 0 ? h_photo.norm() : 0.0;
    double hx_n = n_total > 0 ? h_x_combined.norm() : 0.0;

    Eigen::Matrix<double, 12, 12> fim = Eigen::Matrix<double, 12, 12>::Zero();
    if (n_total > 0 && h_x_combined.cols() >= 12) {
        Eigen::MatrixXd hx12 = h_x_combined.leftCols(12);
        fim = hx12.transpose() * hx12;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 12, 12>> es(fim);
    Eigen::Matrix<double, 12, 1> ev = es.eigenvalues();
    double cond = (ev(11) > 1e-30 && ev(0) > 0) ? (ev(11) / std::max(ev(0), 1e-30)) : 0.0;

    // Pick smallest nonzero eigenvalue's eigenvector (geo H has zero columns for unobserved dims).
    int weak_idx = 0;
    for (int k = 0; k < 12; ++k) {
        if (ev(k) > 1e-3) { weak_idx = k; break; }
    }
    Eigen::Matrix<double, 12, 1> weak = es.eigenvectors().col(weak_idx);

    std::fprintf(f, "%d,%d,%d,%d,%.6g,%.6g,%.6g",
                 frame, n_geo, n_photo, n_total, h_geo_n, h_photo_n, hx_n);
    for (int k = 0; k < 12; ++k) std::fprintf(f, ",%.6g", ev(k));
    std::fprintf(f, ",%.6g", cond);
    std::fprintf(f, ",%.6g,%.6g,%.6g,%.6g,%.6g,%.6g",
                 weak(0), weak(1), weak(2), weak(3), weak(4), weak(5));

    if (stage2_enabled()) {
        int N = (int)h_geo.size();
        double p25 = 0, p50 = 0, p75 = 0, p95 = 0, p99 = 0, mad = 0, nis_norm = 0;
        if (N > 0) {
            std::vector<double> ar(N);
            for (int i = 0; i < N; ++i) ar[i] = std::abs((double)h_geo(i));
            std::sort(ar.begin(), ar.end());
            auto pct = [&](double q) -> double {
                int idx = (int)(q * (N - 1));
                return ar[std::max(0, std::min(N - 1, idx))];
            };
            p25 = pct(0.25);
            p50 = pct(0.50);
            p75 = pct(0.75);
            p95 = pct(0.95);
            p99 = pct(0.99);
            mad = 1.4826 * p50;
            double sumsq = 0;
            for (double v : ar) sumsq += v * v;
            double sigma2 = std::max(mad * mad, 1e-12);
            nis_norm = sumsq / (N * sigma2);
        }
        double px = (double)s.pos(0);
        double py = (double)s.pos(1);
        double pz = (double)s.pos(2);
        double vx = (double)s.vel(0);
        double vy = (double)s.vel(1);
        double vz = (double)s.vel(2);
        double vnorm = std::sqrt(vx * vx + vy * vy + vz * vz);
        double weak_pos_norm = std::sqrt(weak(0) * weak(0) + weak(1) * weak(1) + weak(2) * weak(2));
        double long_cos = 0.0;
        if (vnorm > 1e-3 && weak_pos_norm > 1e-6) {
            long_cos = std::abs((weak(0) * vx + weak(1) * vy + weak(2) * vz) / (vnorm * weak_pos_norm));
        }
        std::fprintf(f, ",%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g",
                     p25, p50, p75, p95, p99, mad, nis_norm,
                     px, py, pz, vx, vy, vz, vnorm, long_cos);
    }

    if (stage3_enabled()) {
        // IMU bias (state.ba, state.bg) — for H4 detection (bias drift)
        double bax = (double)s.ba(0), bay = (double)s.ba(1), baz = (double)s.ba(2);
        double bgx = (double)s.bg(0), bgy = (double)s.bg(1), bgz = (double)s.bg(2);
        double ba_norm = std::sqrt(bax*bax + bay*bay + baz*baz);
        double bg_norm = std::sqrt(bgx*bgx + bgy*bgy + bgz*bgz);
        // Projected information on weak direction (pos block 0-2 of weak eigenvector)
        // proj_info_long = sum_i (J_t_i . e_weak_pos)^2
        // proj_info_total = sum_i ||J_t_i||^2
        double proj_long = 0, proj_total = 0;
        if (n_total > 0 && h_x_combined.cols() >= 3) {
            Eigen::Vector3d ew(weak(0), weak(1), weak(2));
            double ewn = ew.norm();
            if (ewn > 1e-9) ew /= ewn;
            for (int i = 0; i < n_total; ++i) {
                double jx = h_x_combined(i, 0), jy = h_x_combined(i, 1), jz = h_x_combined(i, 2);
                double dot = jx * ew(0) + jy * ew(1) + jz * ew(2);
                proj_long += dot * dot;
                proj_total += jx*jx + jy*jy + jz*jz;
            }
        }
        double proj_frac = (proj_total > 1e-12) ? (proj_long / proj_total) : 0.0;
        std::fprintf(f, ",%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g",
                     bax, bay, baz, bgx, bgy, bgz, ba_norm, bg_norm,
                     proj_long, proj_total, proj_frac);
    }

    std::fprintf(f, "\n");
    std::fflush(f);
    ++frame;
}

inline void log_frame(int n_photo, int n_geo,
                      const Eigen::VectorXd& h_photo, const Eigen::VectorXd& h_geo,
                      const Eigen::MatrixXd& h_x_combined) {
    struct Dummy {
        Eigen::Vector3d pos = Eigen::Vector3d::Zero();
        Eigen::Vector3d vel = Eigen::Vector3d::Zero();
        Eigen::Vector3d ba  = Eigen::Vector3d::Zero();
        Eigen::Vector3d bg  = Eigen::Vector3d::Zero();
    };
    log_frame_s2(n_photo, n_geo, h_photo, h_geo, h_x_combined, Dummy{});
}

}  // namespace diag

// RFB: Residual-Feedback Block deactivation (Plan B / paper v3 §7)
// Per module (photo, geo, te), maintain rolling window of last-iteration residual MAD.
// At start of each new frame, if a module's last-frame residual_MAD exceeded its
// rolling-window p95 (Tukey 3*MAD outlier), suppress that module for the new frame.
// 1-frame delay, no IEKF state snapshot. Activated by env NCC_RFB=1.
// All thresholds are data-derived (window median + MAD); no magic numbers.

#pragma once
#include <Eigen/Dense>
#include <cstdlib>
#include <cstdio>
#include <deque>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

namespace rfb {

inline int enabled() {
    static int e = -1;
    if (e < 0) {
        const char* en = std::getenv("NCC_RFB");
        e = (en && std::string(en) == "1") ? 1 : 0;
    }
    return e;
}

inline std::FILE* log_file() {
    static std::FILE* f = nullptr;
    static bool init = false;
    if (!init) {
        init = true;
        if (!enabled()) return nullptr;
        const char* path = std::getenv("NCC_RFB_LOG");
        std::string p = path ? path : "/tmp/rfb_decisions.csv";
        f = std::fopen(p.c_str(), "w");
        if (f) {
            std::fprintf(f, "frame,iter,n_photo,n_geo,n_te,mad_photo,mad_geo,mad_te,"
                            "wmed_photo,wmed_geo,wmed_te,sup_photo,sup_geo,sup_te\n");
            std::fflush(f);
        }
    }
    return f;
}

inline int window_size() {
    static int w = -1;
    if (w < 0) {
        const char* e = std::getenv("NCC_RFB_WIN");
        w = e ? std::max(20, std::atoi(e)) : 100;
    }
    return w;
}

inline double mad_k() {
    static double k = -1;
    if (k < 0) {
        const char* e = std::getenv("NCC_RFB_K");
        k = e ? std::atof(e) : 3.0;
    }
    return k;
}

struct State {
    std::deque<double> hist_photo, hist_geo, hist_te;
    double last_mad_photo = 0, last_mad_geo = 0, last_mad_te = 0;
    int last_frame = -1;
    int iter_in_frame = 0;
    bool sup_photo_next = false, sup_geo_next = false, sup_te_next = false;
    bool sup_photo_curr = false, sup_geo_curr = false, sup_te_curr = false;
};

inline State& state() { static State s; return s; }

inline double mad_of(const Eigen::VectorXd& v) {
    if (v.size() == 0) return 0;
    std::vector<double> a(v.size());
    for (int i = 0; i < v.size(); ++i) a[i] = std::abs(v(i));
    size_t mid = a.size() / 2;
    std::nth_element(a.begin(), a.begin() + mid, a.end());
    double med = a[mid];
    std::vector<double> dev(a.size());
    for (size_t i = 0; i < a.size(); ++i) dev[i] = std::abs(a[i] - med);
    std::nth_element(dev.begin(), dev.begin() + mid, dev.end());
    return dev[mid];
}

inline double window_median(const std::deque<double>& d) {
    if (d.empty()) return 0;
    std::vector<double> a(d.begin(), d.end());
    size_t mid = a.size() / 2;
    std::nth_element(a.begin(), a.begin() + mid, a.end());
    return a[mid];
}

inline double window_mad(const std::deque<double>& d, double med) {
    if (d.empty()) return 0;
    std::vector<double> dev(d.size());
    size_t i = 0; for (auto v : d) { dev[i++] = std::abs(v - med); }
    size_t mid = dev.size() / 2;
    std::nth_element(dev.begin(), dev.begin() + mid, dev.end());
    return dev[mid];
}

// Called from laserMapping at start-of-frame detection (g_te_frame changed).
// Promotes "next frame" suppression flags to "current frame".
inline void on_new_frame(int frame_id) {
    auto& s = state();
    if (s.last_frame != frame_id) {
        // Push previous frame's last-iter residual MAD to histories (if observed).
        auto push = [](std::deque<double>& d, double v) {
            if (v <= 0) return;
            d.push_back(v);
            while ((int)d.size() > window_size()) d.pop_front();
        };
        push(s.hist_photo, s.last_mad_photo);
        push(s.hist_geo,   s.last_mad_geo);
        push(s.hist_te,    s.last_mad_te);

        // RFB v2: rank-based (scale-invariant). Suppress if last MAD is in top-5% of window.
        // Replaces MAD-of-MAD outlier rule that was confounded by per-module residual scale.
        auto decide = [](double last, const std::deque<double>& d) {
            if ((int)d.size() < 20) return false;
            if (last <= 0) return false;
            int gt = 0;
            for (double v : d) if (last > v) ++gt;
            double rank_frac = double(gt) / double(d.size());
            return rank_frac >= 0.95;
        };
        s.sup_photo_curr = decide(s.last_mad_photo, s.hist_photo);
        s.sup_geo_curr   = decide(s.last_mad_geo,   s.hist_geo);
        s.sup_te_curr    = decide(s.last_mad_te,    s.hist_te);
        s.last_frame = frame_id;
        s.iter_in_frame = 0;
    }
}

// Called at end-of-iteration to record residual MAD per module.
inline void on_iter(int frame_id, int n_photo, int n_geo, int n_te,
                    const Eigen::VectorXd& h_photo,
                    const Eigen::VectorXd& h_geo,
                    const Eigen::VectorXd& h_te) {
    auto& s = state();
    double m_p = (n_photo>0) ? mad_of(h_photo) : 0;
    double m_g = (n_geo  >0) ? mad_of(h_geo)   : 0;
    double m_t = (n_te   >0) ? mad_of(h_te)    : 0;
    s.last_mad_photo = m_p;
    s.last_mad_geo   = m_g;
    s.last_mad_te    = m_t;
    s.iter_in_frame += 1;
    if (auto* f = log_file()) {
        std::fprintf(f, "%d,%d,%d,%d,%d,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%d,%d,%d\n",
                     frame_id, s.iter_in_frame, n_photo, n_geo, n_te,
                     m_p, m_g, m_t,
                     window_median(s.hist_photo),
                     window_median(s.hist_geo),
                     window_median(s.hist_te),
                     s.sup_photo_curr?1:0, s.sup_geo_curr?1:0, s.sup_te_curr?1:0);
        std::fflush(f);
    }
}

inline bool suppress_photo() { return enabled() && state().sup_photo_curr; }
inline bool suppress_geo()   { return enabled() && state().sup_geo_curr;   }
inline bool suppress_te()    { return enabled() && state().sup_te_curr;    }

} // namespace rfb

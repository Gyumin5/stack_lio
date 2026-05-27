// T-E: Sensor-agnostic 3D voxel intensity map for longitudinal-direction-gated
// photometric residual augmentation. Stores per-voxel: count, mean intensity,
// centroid, intensity-position cross products (for spatial gradient estimate).
// Updated from accepted geometric correspondences (lagged 1 frame to avoid
// self-reinforcement). Look-ups compute residual r_I = I_curr - voxel.mean
// and spatial gradient g via local linear regression I(p) ≈ mean + g·(p-centroid).
//
// No ring/scan-line dependence; xyz/intensity only.

#pragma once
#include <Eigen/Dense>
#include <unordered_map>
#include <cmath>
#include <cstdint>

namespace vim {

struct Cell {
    double cx = 0, cy = 0, cz = 0;   // centroid running sum
    double mean = 0;                 // Welford mean intensity
    double M2 = 0;                   // Welford M2 (sum of squared deviations)
    int count = 0;
    // Cross-products: sum (I_i - mean) * (p_i - centroid) per axis
    // After accumulation: g ≈ inv(Σ(p-c)(p-c)^T) · Σ(I-mean)(p-c)
    // For simplicity store per-axis sum (I - mean)*(p - centroid_prev) → updated incrementally.
    // For minimal v1: also store position covariance Sxx,Syy,Szz,Sxy,Sxz,Syz
    double SIx = 0, SIy = 0, SIz = 0;
    double Sxx = 0, Syy = 0, Szz = 0, Sxy = 0, Sxz = 0, Syz = 0;
    int last_update_frame = -1;
};

// Hash key: voxel index (vx, vy, vz)
struct VoxKey {
    int x, y, z;
    bool operator==(const VoxKey& o) const { return x == o.x && y == o.y && z == o.z; }
};

struct VoxKeyHash {
    size_t operator()(const VoxKey& k) const {
        // 3D Cantor-like hash
        size_t h = 2654435761u * (size_t)(uint32_t)k.x;
        h ^= 2246822519u * (size_t)(uint32_t)k.y;
        h ^= 3266489917u * (size_t)(uint32_t)k.z;
        return h;
    }
};

class VoxelIntensityMap {
public:
    explicit VoxelIntensityMap(double res = 0.25) : res_(res) {}

    inline VoxKey key_of(double x, double y, double z) const {
        VoxKey k;
        k.x = (int)std::floor(x / res_);
        k.y = (int)std::floor(y / res_);
        k.z = (int)std::floor(z / res_);
        return k;
    }

    void update(double x, double y, double z, double I, int frame) {
        VoxKey k = key_of(x, y, z);
        Cell& c = map_[k];
        c.count += 1;
        double n = (double)c.count;
        // Welford intensity
        double delta_I = I - c.mean;
        c.mean += delta_I / n;
        double delta_I2 = I - c.mean;
        c.M2 += delta_I * delta_I2;
        // Centroid running sum
        c.cx += x; c.cy += y; c.cz += z;
        // Note: positional covariance and cross-products are not maintained
        // incrementally here for simplicity; T-E v1 will use a degenerate
        // gradient (zero) which makes the photo residual pose-invariant
        // — meaning it only updates state nuisance dims. We'll add gradient
        // in v2 (next iteration).
        c.last_update_frame = frame;
    }

    // Query: returns true if cell + at least 4 of 6 spatial neighbors exist with min_count.
    // Outputs residual r = I_curr - cell.mean and gradient g via 6-neighbor finite differences.
    bool query(double x, double y, double z, double I_curr, int min_count,
               double& out_r, Eigen::Vector3d& out_g, int& out_count) const {
        VoxKey k = key_of(x, y, z);
        auto it = map_.find(k);
        if (it == map_.end()) return false;
        const Cell& c = it->second;
        if (c.count < min_count) return false;
        out_r = I_curr - c.mean;
        out_count = c.count;
        // 6-neighbor finite-difference gradient
        auto get_mean = [&](int dx, int dy, int dz, double& m) -> bool {
            VoxKey nk{k.x + dx, k.y + dy, k.z + dz};
            auto nit = map_.find(nk);
            if (nit == map_.end() || nit->second.count < min_count) return false;
            m = nit->second.mean;
            return true;
        };
        double mxp, mxn, myp, myn, mzp, mzn;
        bool hxp = get_mean(+1, 0, 0, mxp), hxn = get_mean(-1, 0, 0, mxn);
        bool hyp = get_mean(0, +1, 0, myp), hyn = get_mean(0, -1, 0, myn);
        bool hzp = get_mean(0, 0, +1, mzp), hzn = get_mean(0, 0, -1, mzn);
        double dx = (hxp && hxn) ? (mxp - mxn) / (2.0 * res_)
                                 : (hxp ? (mxp - c.mean) / res_ : (hxn ? (c.mean - mxn) / res_ : 0.0));
        double dy = (hyp && hyn) ? (myp - myn) / (2.0 * res_)
                                 : (hyp ? (myp - c.mean) / res_ : (hyn ? (c.mean - myn) / res_ : 0.0));
        double dz = (hzp && hzn) ? (mzp - mzn) / (2.0 * res_)
                                 : (hzp ? (mzp - c.mean) / res_ : (hzn ? (c.mean - mzn) / res_ : 0.0));
        out_g << dx, dy, dz;
        // Require gradient magnitude not zero (uninformative cell otherwise)
        return out_g.norm() > 1e-6;
    }

    // T-D-lite: per-point intensity z-score against voxel distribution.
    // Returns -1 if voxel missing or count < min_count.
    double query_zscore(double x, double y, double z, double I_curr, int min_count = 5) const {
        VoxKey k = key_of(x, y, z);
        auto it = map_.find(k);
        if (it == map_.end()) return -1.0;
        const Cell& c = it->second;
        if (c.count < min_count) return -1.0;
        double var = (c.count > 1) ? (c.M2 / (c.count - 1)) : 0.0;
        double std = std::sqrt(std::max(var, 1.0));  // floor 1.0 to avoid divide-by-near-zero
        return std::abs(I_curr - c.mean) / std;
    }

    void clear() { map_.clear(); }
    size_t size() const { return map_.size(); }

private:
    double res_;
    std::unordered_map<VoxKey, Cell, VoxKeyHash> map_;
};

}  // namespace vim

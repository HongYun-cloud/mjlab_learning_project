// TAMOLS C++ NLP core
//
// Step 1:
//   - 变量布局 (spline_coeffs | desired_footholds | slack)
//   - 6D 相位样条 (0/1 阶)
//   - 初始位姿/速度约束 + 相邻相位 C0/C1 连接约束
//   - horizon 内不触地腿的 desired 钉扎
//   - 局部高程图 h/h_s1/h_s2 双线性查询
//
// Step 2:
//   - GIAC 17b/17c/17d/17e/17f
//   - 摩擦锥 5 / V-A-4
//   - 凸性 30
//   - 可达 31
//
// 构建: bash build_core.sh

#include <cppad/cppad.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <atomic>
#include <cstddef>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

constexpr int BASE_DIMS = 6;
constexpr int NUM_LEGS = 4;
constexpr double GRAVITY_Z = -9.81;

using AD = CppAD::AD<double>;
using Vec3 = std::array<AD, 3>;
using Mat3 = std::array<Vec3, 3>;

inline int coeff_index(int phase, int dim, int k, int num_coeffs) {
    return (phase * BASE_DIMS + dim) * num_coeffs + k;
}

inline int desired_index(int coeff_count, int leg, int dim) {
    return coeff_count + leg * 3 + dim;
}

inline int slack_index(int coeff_count, int phase) {
    return coeff_count + NUM_LEGS * 3 + phase;
}

template <typename T>
T spline_value(
    const std::vector<T>& coeffs,
    int phase,
    int dim,
    int num_coeffs,
    double tau,
    int order
) {
    T value = T(0.0);
    if (order == 0) {
        value = coeffs[coeff_index(phase, dim, num_coeffs - 1, num_coeffs)];
        for (int k = num_coeffs - 2; k >= 0; --k) {
            value = value * T(tau) + coeffs[coeff_index(phase, dim, k, num_coeffs)];
        }
        return value;
    }
    if (order == 1) {
        value = T(double(num_coeffs - 1)) *
                coeffs[coeff_index(phase, dim, num_coeffs - 1, num_coeffs)];
        for (int k = num_coeffs - 2; k >= 1; --k) {
            value = value * T(tau) +
                    T(double(k)) * coeffs[coeff_index(phase, dim, k, num_coeffs)];
        }
        return value;
    }
    // order == 2: sum_{k>=2} k(k-1) a_k tau^(k-2)
    for (int k = num_coeffs - 1; k >= 2; --k) {
        value = value * T(tau) +
                T(double(k * (k - 1))) *
                    coeffs[coeff_index(phase, dim, k, num_coeffs)];
    }
    return value;
}

inline std::array<AD, 6> spline_pose(
    const std::vector<AD>& ax,
    int phase,
    int num_coeffs,
    double tau,
    int order
) {
    std::array<AD, 6> pose;
    for (int d = 0; d < BASE_DIMS; ++d) {
        pose[d] = spline_value(ax, phase, d, num_coeffs, tau, order);
    }
    return pose;
}

inline Mat3 rotation_matrix_zyx(const AD& roll, const AD& pitch, const AD& yaw) {
    const AD sr = CppAD::sin(roll), cr = CppAD::cos(roll);
    const AD sp = CppAD::sin(pitch), cp = CppAD::cos(pitch);
    const AD sy = CppAD::sin(yaw), cy = CppAD::cos(yaw);
    Mat3 R;
    R[0] = Vec3{cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr};
    R[1] = Vec3{sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr};
    R[2] = Vec3{-sp, cp * sr, cp * cr};
    return R;
}

inline Vec3 matvec(const Mat3& R, const Vec3& v) {
    return {
        R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
        R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
        R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
    };
}

inline Vec3 sub3(const Vec3& a, const Vec3& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

inline Vec3 cross3(const Vec3& a, const Vec3& b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

inline AD dot3(const Vec3& a, const Vec3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline AD det3(const Vec3& a, const Vec3& b, const Vec3& c) {
    return a[0] * (b[1] * c[2] - b[2] * c[1])
         + a[1] * (b[2] * c[0] - b[0] * c[2])
         + a[2] * (b[0] * c[1] - b[1] * c[0]);
}

inline Vec3 body_omega_zyx(const std::array<AD, 6>& euler, const std::array<AD, 6>& rates) {
    const AD phi = euler[3], theta = euler[4];
    const AD sp = CppAD::sin(phi), cp = CppAD::cos(phi);
    const AD st = CppAD::sin(theta), ct = CppAD::cos(theta);
    return {
        rates[3] - rates[5] * st,
        rates[4] * cp + rates[5] * sp * ct,
        -rates[4] * sp + rates[5] * cp * ct,
    };
}

inline Vec3 body_omega_rate_zyx(
    const std::array<AD, 6>& euler,
    const std::array<AD, 6>& rates,
    const std::array<AD, 6>& accels
) {
    const AD phi = euler[3], theta = euler[4];
    const AD sp = CppAD::sin(phi), cp = CppAD::cos(phi);
    const AD st = CppAD::sin(theta), ct = CppAD::cos(theta);
    const AD pd = rates[3], td = rates[4], yd = rates[5];
    const AD pdd = accels[3], tdd = accels[4], ydd = accels[5];
    return {
        pdd - ydd * st - yd * td * ct,
        tdd * cp - td * pd * sp + ydd * sp * ct + yd * pd * cp * ct - yd * td * sp * st,
        -tdd * sp - td * pd * cp + ydd * cp * ct - yd * pd * sp * ct - yd * td * cp * st,
    };
}

inline Mat3 inertia_matrix(const double* I) {
    Mat3 M;
    M[0] = Vec3{AD(I[0]), AD(I[1]), AD(I[2])};
    M[1] = Vec3{AD(I[3]), AD(I[4]), AD(I[5])};
    M[2] = Vec3{AD(I[6]), AD(I[7]), AD(I[8])};
    return M;
}

inline Vec3 angular_momentum_rate(
    const Mat3& R,
    const Mat3& inertia,
    const Vec3& omega,
    const Vec3& omega_dot
) {
    const Vec3 iw = matvec(inertia, omega);
    const Vec3 iwd = matvec(inertia, omega_dot);
    const Vec3 inner = cross3(omega, iw);
    return matvec(R, {inner[0] + iwd[0], inner[1] + iwd[1], inner[2] + iwd[2]});
}


// 五点一阶差分 (与 map_derivatives._diff1_axis 同公式).
void diff1_axis_x(const double* v, int ny, int nx, double dx, double* out) {
    const double denom = 12.0 * dx;
    for (int j = 0; j < ny; ++j) {
        const double* r = v + j * nx;
        double* o = out + j * nx;
        for (int i = 0; i < nx; ++i) {
            if (!std::isfinite(r[i])) { o[i] = std::nan(""); continue; }
            if (i >= 2 && i < nx - 2) {
                o[i] = (r[i - 2] - 8.0 * r[i - 1] + 8.0 * r[i + 1] - r[i + 2]) / denom;
            } else if (i == 0) {
                o[i] = (-25.0 * r[0] + 48.0 * r[1] - 36.0 * r[2] + 16.0 * r[3] - 3.0 * r[4]) / denom;
            } else if (i == 1) {
                o[i] = (-3.0 * r[0] - 10.0 * r[1] + 18.0 * r[2] - 6.0 * r[3] + r[4]) / denom;
            } else if (i == nx - 2) {
                o[i] = (-r[nx - 5] + 6.0 * r[nx - 4] - 18.0 * r[nx - 3] + 10.0 * r[nx - 2] + 3.0 * r[nx - 1]) / denom;
            } else {
                o[i] = (3.0 * r[nx - 5] - 16.0 * r[nx - 4] + 36.0 * r[nx - 3] - 48.0 * r[nx - 2] + 25.0 * r[nx - 1]) / denom;
            }
        }
    }
}

void diff1_axis_y(const double* v, int ny, int nx, double dy, double* out) {
    const double denom = 12.0 * dy;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int idx = j * nx + i;
            if (!std::isfinite(v[idx])) { out[idx] = std::nan(""); continue; }
            if (j >= 2 && j < ny - 2) {
                out[idx] = (v[(j - 2) * nx + i] - 8.0 * v[(j - 1) * nx + i]
                            + 8.0 * v[(j + 1) * nx + i] - v[(j + 2) * nx + i]) / denom;
            } else if (j == 0) {
                out[idx] = (-25.0 * v[i] + 48.0 * v[nx + i] - 36.0 * v[2 * nx + i]
                            + 16.0 * v[3 * nx + i] - 3.0 * v[4 * nx + i]) / denom;
            } else if (j == 1) {
                out[idx] = (-3.0 * v[i] - 10.0 * v[nx + i] + 18.0 * v[2 * nx + i]
                            - 6.0 * v[3 * nx + i] + v[4 * nx + i]) / denom;
            } else if (j == ny - 2) {
                out[idx] = (-v[(ny - 5) * nx + i] + 6.0 * v[(ny - 4) * nx + i]
                            - 18.0 * v[(ny - 3) * nx + i] + 10.0 * v[(ny - 2) * nx + i]
                            + 3.0 * v[(ny - 1) * nx + i]) / denom;
            } else {
                out[idx] = (3.0 * v[(ny - 5) * nx + i] - 16.0 * v[(ny - 4) * nx + i]
                            + 36.0 * v[(ny - 3) * nx + i] - 48.0 * v[(ny - 2) * nx + i]
                            + 25.0 * v[(ny - 1) * nx + i]) / denom;
            }
        }
    }
}

// Bilinear map query in map frame. Grid layout: [Ny, Nx] row-major.
double bilinear(
    const double* grid, int ny, int nx,
    double x_map, double y_map,
    double dx, double dy
) {
    double gx = x_map / dx;
    double gy = y_map / dy;
    gx = std::min(std::max(gx, 0.0), double(nx - 1));
    gy = std::min(std::max(gy, 0.0), double(ny - 1));
    int i0 = std::min(std::max(int(std::floor(gx)), 0), nx - 1);
    int j0 = std::min(std::max(int(std::floor(gy)), 0), ny - 1);
    int i1 = std::min(i0 + 1, nx - 1);
    int j1 = std::min(j0 + 1, ny - 1);
    double fx = gx - double(i0);
    double fy = gy - double(j0);
    double v00 = grid[j0 * nx + i0];
    double v10 = grid[j0 * nx + i1];
    double v01 = grid[j1 * nx + i0];
    double v11 = grid[j1 * nx + i1];
    double v0 = v00 * (1.0 - fx) + v10 * fx;
    double v1 = v01 * (1.0 - fx) + v11 * fx;
    return v0 * (1.0 - fy) + v1 * fy;
}

inline std::vector<int> ordered_stance(
    const int* stance_mask,
    int phase,
    bool winding_cw
) {
    static const int ccw[4] = {0, 2, 3, 1};
    std::vector<int> ordered;
    for (int idx = 0; idx < 4; ++idx) {
        int leg = ccw[idx];
        if (stance_mask[phase * NUM_LEGS + leg]) ordered.push_back(leg);
    }
    if (winding_cw) std::reverse(ordered.begin(), ordered.end());
    return ordered;
}

inline std::vector<int> stance_legs(const int* stance_mask, int phase) {
    std::vector<int> legs;
    for (int leg = 0; leg < NUM_LEGS; ++leg) {
        if (stance_mask[phase * NUM_LEGS + leg]) legs.push_back(leg);
    }
    return legs;
}

inline Vec3 foot_at(
    const std::vector<AD>& ax,
    int coeff_count,
    const int* stance_mask,
    const int* measured_mask,
    const double* measured_feet,
    int phase,
    int leg
) {
    if (measured_mask[phase * NUM_LEGS + leg]) {
        return {
            AD(measured_feet[3 * leg + 0]),
            AD(measured_feet[3 * leg + 1]),
            AD(measured_feet[3 * leg + 2]),
        };
    }
    return {
        ax[desired_index(coeff_count, leg, 0)],
        ax[desired_index(coeff_count, leg, 1)],
        ax[desired_index(coeff_count, leg, 2)],
    };
}


inline AD max_ad(const AD& a, double b) {
    return CppAD::CondExpGt(a, AD(b), a, AD(b));
}

double map_query(
    const double* grid, int ny, int nx,
    double dx, double dy,
    double origin_x, double origin_y, double yaw,
    double x, double y
) {
    const double c = std::cos(yaw);
    const double s = std::sin(yaw);
    const double dxw = x - origin_x;
    const double dyw = y - origin_y;
    const double xm = c * dxw + s * dyw;
    const double ym = -s * dxw + c * dyw;
    return bilinear(grid, ny, nx, xm, ym, dx, dy);
}


class MapAtomic : public CppAD::atomic_base<double> {
public:
    MapAtomic(
        const std::string& name,
        const double* grid, int ny, int nx,
        double dx, double dy,
        double origin_x, double origin_y, double yaw
    ) : CppAD::atomic_base<double>(
            name, CppAD::atomic_base<double>::set_sparsity_enum
        ),
        grid_(grid), ny_(ny), nx_(nx), dx_(dx), dy_(dy),
        origin_x_(origin_x), origin_y_(origin_y), yaw_(yaw) {}

    AD op(const AD& x, const AD& y) {
        CppAD::vector<AD> ax(2), ay(1);
        ax[0] = x;
        ax[1] = y;
        (*this)(ax, ay);
        return ay[0];
    }

    bool forward(
        size_t p, size_t q,
        const CppAD::vector<bool>& vx,
        CppAD::vector<bool>& vy,
        const CppAD::vector<double>& tx,
        CppAD::vector<double>& ty
    ) override {
        if (p > 1) return false;
        const size_t n_order = q + 1;
        if (tx.size() != 2 * n_order || ty.size() != n_order) return false;
        if (vx.size() != 0) {
            if (vy.size() != 1) return false;
            vy[0] = vx[0] || vx[1];
        }
        double v = 0.0, gx = 0.0, gy = 0.0;
        eval(tx[0], tx[1], v, gx, gy);
        ty[0] = v;
        if (q >= 1) {
            ty[1] = gx * tx[2] + gy * tx[3];
        }
        return true;
    }

    bool for_sparse_jac(
        size_t q,
        const CppAD::vector<std::set<size_t>>& r,
        CppAD::vector<std::set<size_t>>& s,
        const CppAD::vector<double>& x
    ) override {
        (void)q; (void)x;
        if (r.size() != 2 || s.size() != 1) return false;
        s[0] = CppAD::set_union(r[0], r[1]);
        return true;
    }

    bool reverse(
        size_t q,
        const CppAD::vector<double>& tx,
        const CppAD::vector<double>& ty,
        CppAD::vector<double>& px,
        const CppAD::vector<double>& py
    ) override {
        (void)q; (void)ty;
        if (tx.size() < 2 || px.size() != 2 || py.size() != 1) return false;
        double v = 0.0, gx = 0.0, gy = 0.0;
        eval(tx[0], tx[1], v, gx, gy);
        px[0] += gx * py[0];
        px[1] += gy * py[0];
        return true;
    }

private:
    void eval(double x, double y, double& v, double& gx, double& gy) const {
        const double c = std::cos(yaw_);
        const double s = std::sin(yaw_);
        const double xm = c * (x - origin_x_) + s * (y - origin_y_);
        const double ym = -s * (x - origin_x_) + c * (y - origin_y_);
        const double gxf = xm / dx_;
        const double gyf = ym / dy_;
        const double gxc = std::min(std::max(gxf, 0.0), double(nx_ - 1));
        const double gyc = std::min(std::max(gyf, 0.0), double(ny_ - 1));
        const int i0 = std::min(std::max(int(std::floor(gxc)), 0), nx_ - 1);
        const int j0 = std::min(std::max(int(std::floor(gyc)), 0), ny_ - 1);
        const int i1 = std::min(i0 + 1, nx_ - 1);
        const int j1 = std::min(j0 + 1, ny_ - 1);
        const double fx = gxc - double(i0);
        const double fy = gyc - double(j0);
        const double v00 = grid_[j0 * nx_ + i0];
        const double v10 = grid_[j0 * nx_ + i1];
        const double v01 = grid_[j1 * nx_ + i0];
        const double v11 = grid_[j1 * nx_ + i1];
        const double v0 = v00 * (1.0 - fx) + v10 * fx;
        const double v1 = v01 * (1.0 - fx) + v11 * fx;
        v = v0 * (1.0 - fy) + v1 * fy;

        const double dvdx_map =
            ((v10 - v00) * (1.0 - fy) + (v11 - v01) * fy) / dx_;
        const double dvdy_map =
            ((v01 - v00) * (1.0 - fx) + (v11 - v10) * fx) / dy_;
        const bool clamp_x = (gxf < 0.0 || gxf > double(nx_ - 1));
        const bool clamp_y = (gyf < 0.0 || gyf > double(ny_ - 1));
        const double dmx = clamp_x ? 0.0 : dvdx_map;
        const double dmy = clamp_y ? 0.0 : dvdy_map;
        gx = dmx * c - dmy * s;
        gy = dmx * s + dmy * c;
    }

    const double* grid_;
    int ny_, nx_;
    double dx_, dy_, origin_x_, origin_y_, yaw_;
};

struct MapAtomicEntry {
    const double* grid;
    int ny, nx;
    double dx, dy, origin_x, origin_y, yaw;
    std::shared_ptr<MapAtomic> atomic;
};

thread_local std::vector<MapAtomicEntry> g_map_atomic_cache;
static std::atomic<int> g_map_atomic_counter{0};

AD map_query_ad(
    const double* grid, int ny, int nx,
    double dx, double dy,
    double origin_x, double origin_y, double yaw,
    const AD& x, const AD& y
) {
    for (const auto& entry : g_map_atomic_cache) {
        if (entry.grid == grid && entry.ny == ny && entry.nx == nx &&
            entry.dx == dx && entry.dy == dy &&
            entry.origin_x == origin_x && entry.origin_y == origin_y &&
            entry.yaw == yaw) {
            return entry.atomic->op(x, y);
        }
    }
    auto atomic = std::make_shared<MapAtomic>(
        "map_atomic_" + std::to_string(g_map_atomic_counter++),
        grid, ny, nx, dx, dy, origin_x, origin_y, yaw
    );
    g_map_atomic_cache.push_back(
        MapAtomicEntry{grid, ny, nx, dx, dy, origin_x, origin_y, yaw, atomic}
    );
    return atomic->op(x, y);
}

}  // namespace

extern "C" int tamols_step1_eval(
    int num_phases,
    int num_coeffs,
    const double* x,
    const double* base_pose0,
    const double* base_twist0,
    const double* durations,
    const int* pin_mask,
    const double* measured_feet,
    double* residuals,
    double* jacobian,
    int* m_out
) {
    if (num_phases < 1 || num_coeffs < 2) return -1;
    const int coeff_count = num_phases * BASE_DIMS * num_coeffs;
    const int n = coeff_count + NUM_LEGS * 3 + num_phases;
    int num_pins = 0;
    for (int leg = 0; leg < NUM_LEGS; ++leg) {
        if (pin_mask[leg]) num_pins += 1;
    }
    const int m = 12 + 3 * num_pins + 12 * (num_phases - 1);

    std::vector<AD> ax(n);
    for (int i = 0; i < n; ++i) ax[i] = x[i];

    CppAD::Independent(ax);
    std::vector<AD> ay;
    ay.reserve(m);

    for (int d = 0; d < BASE_DIMS; ++d) {
        ay.push_back(
            spline_value(ax, 0, d, num_coeffs, 0.0, 0) - AD(base_pose0[d])
        );
    }
    for (int d = 0; d < BASE_DIMS; ++d) {
        ay.push_back(
            spline_value(ax, 0, d, num_coeffs, 0.0, 1) - AD(base_twist0[d])
        );
    }
    for (int leg = 0; leg < NUM_LEGS; ++leg) {
        if (!pin_mask[leg]) continue;
        for (int d = 0; d < 3; ++d) {
            ay.push_back(
                ax[desired_index(coeff_count, leg, d)] -
                AD(measured_feet[leg * 3 + d])
            );
        }
    }
    for (int p = 0; p + 1 < num_phases; ++p) {
        const double T = durations[p];
        for (int d = 0; d < BASE_DIMS; ++d) {
            ay.push_back(
                spline_value(ax, p, d, num_coeffs, T, 0) -
                spline_value(ax, p + 1, d, num_coeffs, 0.0, 0)
            );
        }
        for (int d = 0; d < BASE_DIMS; ++d) {
            ay.push_back(
                spline_value(ax, p, d, num_coeffs, T, 1) -
                spline_value(ax, p + 1, d, num_coeffs, 0.0, 1)
            );
        }
    }

    CppAD::ADFun<double> fun(ax, ay);
    std::vector<double> xv(x, x + n);
    std::vector<double> y = fun.Forward(0, xv);
    if (residuals != nullptr) {
        for (int i = 0; i < m; ++i) residuals[i] = y[i];
    }
    if (jacobian != nullptr) {
        std::vector<double> J = fun.Jacobian(xv);
        for (int i = 0; i < m * n; ++i) jacobian[i] = J[i];
    }
    *m_out = m;
    return 0;
}

extern "C" int tamols_step2_eval(
    int num_phases,
    int num_coeffs,
    const double* x,
    const double* durations,
    const int* stance_mask,
    const int* measured_mask,
    const double* measured_feet,
    const int* pin_mask,
    const double* base_pose0,
    const double* base_twist0,
    double mass,
    const double* inertia9,
    const double* hip_offsets12,
    double l_min,
    double l_max,
    double mu,
    int num_samples,
    int giac_pairs_mode,
    int winding,
    int friction_mode,
    int friction_form,
    int include_push,
    int inertial_sign,
    int convexity_fourth,
    int use_slack,
    double* residuals,
    double* jacobian,
    int* m_out
) {
    if (num_phases < 1 || num_coeffs < 2 || num_samples < 1) return -1;
    const int coeff_count = num_phases * BASE_DIMS * num_coeffs;
    const int n = coeff_count + NUM_LEGS * 3 + num_phases;

    std::vector<AD> ax(n);
    for (int i = 0; i < n; ++i) ax[i] = x[i];

    CppAD::Independent(ax);
    std::vector<AD> ay;

    // ---- V-A-1 初始位姿/速度 + 不触地腿钉扎 + 连接 ----
    for (int d = 0; d < BASE_DIMS; ++d) {
        ay.push_back(
            spline_value(ax, 0, d, num_coeffs, 0.0, 0) - AD(base_pose0[d])
        );
    }
    for (int d = 0; d < BASE_DIMS; ++d) {
        ay.push_back(
            spline_value(ax, 0, d, num_coeffs, 0.0, 1) - AD(base_twist0[d])
        );
    }
    for (int leg = 0; leg < NUM_LEGS; ++leg) {
        if (!pin_mask[leg]) continue;
        for (int d = 0; d < 3; ++d) {
            ay.push_back(
                ax[desired_index(coeff_count, leg, d)] -
                AD(measured_feet[leg * 3 + d])
            );
        }
    }
    for (int p = 0; p + 1 < num_phases; ++p) {
        const double T = durations[p];
        for (int d = 0; d < BASE_DIMS; ++d) {
            ay.push_back(
                spline_value(ax, p, d, num_coeffs, T, 0) -
                spline_value(ax, p + 1, d, num_coeffs, 0.0, 0)
            );
        }
        for (int d = 0; d < BASE_DIMS; ++d) {
            ay.push_back(
                spline_value(ax, p, d, num_coeffs, T, 1) -
                spline_value(ax, p + 1, d, num_coeffs, 0.0, 1)
            );
        }
    }

    const Mat3 inertia = inertia_matrix(inertia9);
    const bool cw = winding != 0;

    // ---- 式 (29): slack <= 0 ----
    if (use_slack) {
        for (int p = 0; p < num_phases; ++p) {
            ay.push_back(ax[slack_index(coeff_count, p)]);
        }
    }

    // ---- 17b-17f + 摩擦 ----
    for (int p = 0; p < num_phases; ++p) {
        const double T = durations[p];
        for (int s = 0; s < num_samples; ++s) {
            const double tau = T * double(s) / double(num_samples);
            const auto pose = spline_pose(ax, p, num_coeffs, tau, 0);
            const auto acc = spline_pose(ax, p, num_coeffs, tau, 2);
            const auto rate = spline_pose(ax, p, num_coeffs, tau, 1);
            const Vec3 pB = {pose[0], pose[1], pose[2]};
            const Vec3 pdd = {acc[0], acc[1], acc[2]};
            Vec3 aB;
            if (inertial_sign == 0) {
                aB = {-pdd[0], -pdd[1], AD(GRAVITY_Z) - pdd[2]};
            } else {
                aB = {pdd[0], pdd[1], pdd[2] - AD(GRAVITY_Z)};
            }
            const Mat3 R = rotation_matrix_zyx(pose[3], pose[4], pose[5]);
            const Vec3 omega = body_omega_zyx(pose, rate);
            const Vec3 omega_dot = body_omega_rate_zyx(pose, rate, acc);
            const Vec3 Ldot = angular_momentum_rate(R, inertia, omega, omega_dot);

            const std::vector<int> ordered = ordered_stance(stance_mask, p, cw);
            const int nfeet = int(ordered.size());
            if (nfeet >= 3) {
                std::vector<std::pair<int, int>> pairs;
                if (giac_pairs_mode == 0) {
                    for (int i = 0; i < nfeet; ++i) {
                        pairs.emplace_back(i, (i + 1) % nfeet);
                    }
                } else {
                    for (int i = 0; i < nfeet; ++i) {
                        for (int j = i + 1; j < nfeet; ++j) {
                            pairs.emplace_back(i, j);
                        }
                    }
                }
                for (auto [i, j] : pairs) {
                    const Vec3 pi = foot_at(
                        ax, coeff_count, stance_mask, measured_mask,
                        measured_feet, p, ordered[i]
                    );
                    const Vec3 pj = foot_at(
                        ax, coeff_count, stance_mask, measured_mask,
                        measured_feet, p, ordered[j]
                    );
                    const Vec3 pij = sub3(pj, pi);
                    AD res = AD(mass) * det3(pij, sub3(pB, pi), aB) - dot3(pij, Ldot);
                    if (use_slack) res = res - ax[slack_index(coeff_count, p)];
                    ay.push_back(res);
                }
            } else if (nfeet == 2) {
                const Vec3 p1 = foot_at(
                    ax, coeff_count, stance_mask, measured_mask,
                    measured_feet, p, ordered[0]
                );
                const Vec3 p2 = foot_at(
                    ax, coeff_count, stance_mask, measured_mask,
                    measured_feet, p, ordered[1]
                );
                const Vec3 p12 = sub3(p2, p1);
                // 17c equality
                ay.push_back(
                    AD(mass) * det3(p12, sub3(pB, p1), aB) - dot3(p12, Ldot)
                );
                // 17d inequality: -det(ez, p12, M1)
                const Vec3 M1 = {
                    cross3(sub3(pB, p1), aB)[0] - Ldot[0] / AD(mass),
                    cross3(sub3(pB, p1), aB)[1] - Ldot[1] / AD(mass),
                    cross3(sub3(pB, p1), aB)[2] - Ldot[2] / AD(mass),
                };
                AD res = -det3(Vec3{AD(0.0), AD(0.0), AD(1.0)}, p12, M1);
                if (use_slack) res = res - ax[slack_index(coeff_count, p)];
                ay.push_back(res);
            } else if (nfeet == 1) {
                const Vec3 pi = foot_at(
                    ax, coeff_count, stance_mask, measured_mask,
                    measured_feet, p, ordered[0]
                );
                const Vec3 m = cross3(sub3(pB, pi), aB);
                for (int d = 0; d < 3; ++d) {
                    ay.push_back(AD(mass) * m[d] - Ldot[d]);
                }
            } else {
                for (int d = 0; d < 3; ++d) ay.push_back(aB[d]);
                for (int d = 0; d < 3; ++d) ay.push_back(Ldot[d]);
            }

            // ---- 摩擦 5 / V-A-4 ----
            if (nfeet > 0 && friction_mode != 0) {
                if (friction_mode == 2) {
                    // force_balance_derived, squared
                    ay.push_back(
                        aB[0] * aB[0] + aB[1] * aB[1] -
                        (AD(mu) * (-aB[2])) * (AD(mu) * (-aB[2]))
                    );
                    ay.push_back(aB[2]);
                } else if (friction_form == 0) {
                    // paper_literal norm: ||aB_xy|| - mu aB_z
                    ay.push_back(
                        CppAD::sqrt(aB[0] * aB[0] + aB[1] * aB[1]) -
                        AD(mu) * aB[2]
                    );
                } else {
                    // paper_literal squared + aB_z >= 0
                    ay.push_back(
                        aB[0] * aB[0] + aB[1] * aB[1] -
                        (AD(mu) * aB[2]) * (AD(mu) * aB[2])
                    );
                    ay.push_back(-aB[2]);
                }
                if (include_push) {
                    ay.push_back(AD(GRAVITY_Z) - pdd[2]);
                }
            }
        }
    }

    // ---- 30 凸性 + 31 可达 ----
    for (int p = 0; p < num_phases; ++p) {
        const std::vector<int> stance = stance_legs(stance_mask, p);
        if (stance.size() == 4) {
            const std::vector<int> ordered = ordered_stance(stance_mask, p, false);
            const Vec3 p1 = foot_at(
                ax, coeff_count, stance_mask, measured_mask,
                measured_feet, p, ordered[0]
            );
            const Vec3 p2 = foot_at(
                ax, coeff_count, stance_mask, measured_mask,
                measured_feet, p, ordered[1]
            );
            const Vec3 p3 = foot_at(
                ax, coeff_count, stance_mask, measured_mask,
                measured_feet, p, ordered[2]
            );
            const Vec3 p4 = foot_at(
                ax, coeff_count, stance_mask, measured_mask,
                measured_feet, p, ordered[3]
            );
            const auto crossz = [](const Vec3& a, const Vec3& b) {
                return a[0] * b[1] - a[1] * b[0];
            };
            const Vec3 p12 = sub3(p2, p1), p13 = sub3(p3, p1), p14 = sub3(p4, p1);
            const Vec3 p23 = sub3(p3, p2), p24 = sub3(p4, p2);
            ay.push_back(crossz(p13, p12));
            ay.push_back(crossz(p14, p13));
            ay.push_back(crossz(p24, p23));
            if (convexity_fourth == 0) {
                ay.push_back(crossz(sub3(p1, p2), p24));
            } else {
                // p20_literal: p0 = world origin
                const Vec3 p20 = {-p2[0], -p2[1], -p2[2]};
                ay.push_back(crossz(p20, p24));
            }
        }

        const double T = durations[p];
        for (int s = 0; s < num_samples; ++s) {
            const double tau = T * double(s) / double(num_samples);
            const auto pose = spline_pose(ax, p, num_coeffs, tau, 0);
            const Vec3 pB = {pose[0], pose[1], pose[2]};
            const Mat3 R = rotation_matrix_zyx(pose[3], pose[4], pose[5]);
            for (int leg : stance) {
                const Vec3 foot = foot_at(
                    ax, coeff_count, stance_mask, measured_mask,
                    measured_feet, p, leg
                );
                const Vec3 r = {
                    AD(hip_offsets12[3 * leg + 0]),
                    AD(hip_offsets12[3 * leg + 1]),
                    AD(hip_offsets12[3 * leg + 2]),
                };
                const Vec3 hip = matvec(R, r);
                const Vec3 d = {
                    pB[0] + hip[0] - foot[0],
                    pB[1] + hip[1] - foot[1],
                    pB[2] + hip[2] - foot[2],
                };
                const AD d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                ay.push_back(AD(l_min * l_min) - d2);
                ay.push_back(d2 - AD(l_max * l_max));
            }
        }
    }

    const int m = int(ay.size());
    CppAD::ADFun<double> fun(ax, ay);
    std::vector<double> xv(x, x + n);
    std::vector<double> y = fun.Forward(0, xv);
    if (residuals != nullptr) {
        for (int i = 0; i < m; ++i) residuals[i] = y[i];
    }
    if (jacobian != nullptr) {
        std::vector<double> J = fun.Jacobian(xv);
        for (int i = 0; i < m * n; ++i) jacobian[i] = J[i];
    }
    *m_out = m;
    return 0;
}



extern "C" int tamols_step3_eval(
    int num_phases,
    int num_coeffs,
    const double* x,
    const double* durations,
    const int* stance_mask,
    const int* td_legs,
    int num_td,
    const double* previous_footholds,
    int has_prev,
    const double* inertia9,
    const double* hip_offsets12,
    double h_des,
    double eps_min,
    int num_samples,
    double ref_vx,
    double ref_vy,
    double ref_wz,
    int map_ny,
    int map_nx,
    const double* map_h,
    const double* map_hs1,
    const double* map_hs2,
    const double* grad_hx,
    const double* grad_hy,
    const double* grad_s1x,
    const double* grad_s1y,
    double map_dx,
    double map_dy,
    double origin_x,
    double origin_y,
    double yaw,
    double w_vb1,
    double w_vb2,
    double w_vb3,
    double w_vb4,
    double w_vb5,
    double w_vb6,
    double w_vb7,
    double w_vb8,
    double w_slack,
    int use_slack,
    double* record_residuals,
    double* record_weights,
    int* record_count,
    double* total_out,
    double* total_grad
) {
    if (num_phases < 1 || num_coeffs < 2) return -1;
    const int coeff_count = num_phases * BASE_DIMS * num_coeffs;
    const int n = coeff_count + NUM_LEGS * 3 + num_phases;

    std::vector<AD> ax(n);
    for (int i = 0; i < n; ++i) ax[i] = x[i];

    CppAD::Independent(ax);
    std::vector<AD> records;
    std::vector<double> weights;
    auto add_record = [&](const AD& residual, double weight) {
        records.push_back(residual);
        weights.push_back(weight);
    };

    const Mat3 inertia = inertia_matrix(inertia9);
    const auto foot_var = [&](int leg) {
        return Vec3{
            ax[desired_index(coeff_count, leg, 0)],
            ax[desired_index(coeff_count, leg, 1)],
            ax[desired_index(coeff_count, leg, 2)],
        };
    };

    // ---- V-B1 落脚点贴地 ----
    for (int k = 0; k < num_td; ++k) {
        const int leg = td_legs[k];
        const Vec3 p = foot_var(leg);
        const AD h = map_query_ad(
            map_h, map_ny, map_nx, map_dx, map_dy,
            origin_x, origin_y, yaw, p[0], p[1]
        );
        add_record((h - p[2]) * (h - p[2]), w_vb1);
    }

    // ---- V-B2 脚间碰撞 barrier ----
    for (int a = 0; a < num_td; ++a) {
        for (int b = a + 1; b < num_td; ++b) {
            const int i = td_legs[a], j = td_legs[b];
            const Vec3 pi = foot_var(i), pj = foot_var(j);
            const AD z = (pi[0] - pj[0]) * (pi[0] - pj[0])
                       + (pi[1] - pj[1]) * (pi[1] - pj[1]);
            const AD gap = max_ad(AD(eps_min) - z, 0.0);
            add_record(gap * gap, w_vb2);
        }
    }

    // ---- V-B5 边缘避让 ----
    for (int k = 0; k < num_td; ++k) {
        const int leg = td_legs[k];
        const Vec3 p = foot_var(leg);
        const AD gx = map_query_ad(
            grad_hx, map_ny, map_nx, map_dx, map_dy,
            origin_x, origin_y, yaw, p[0], p[1]
        );
        const AD gy = map_query_ad(
            grad_hy, map_ny, map_nx, map_dx, map_dy,
            origin_x, origin_y, yaw, p[0], p[1]
        );
        const AD sx = map_query_ad(
            grad_s1x, map_ny, map_nx, map_dx, map_dy,
            origin_x, origin_y, yaw, p[0], p[1]
        );
        const AD sy = map_query_ad(
            grad_s1y, map_ny, map_nx, map_dx, map_dy,
            origin_x, origin_y, yaw, p[0], p[1]
        );
        add_record(gx * gx + gy * gy + sx * sx + sy * sy, w_vb5);
    }

    // ---- V-B3 名义腿长 (midstance) ----
    // 与 midstance_from_contacts 相同的连续支撑段逻辑.
    std::vector<double> starts(num_phases, 0.0);
    for (int p = 1; p < num_phases; ++p) starts[p] = starts[p - 1] + durations[p - 1];
    for (int k = 0; k < num_td; ++k) {
        const int leg = td_legs[k];
        int run_start = -1, run_end = -1;
        for (int p = 0; p < num_phases; ++p) {
            const bool inside = stance_mask[p * NUM_LEGS + leg] != 0;
            if (inside) {
                if (run_start < 0) run_start = p;
                run_end = p;
            } else if (run_start >= 0) {
                break;
            }
        }
        if (run_start < 0) continue;
        const double t_start = starts[run_start];
        const double t_end = starts[run_end] + durations[run_end];
        const double t_mid = 0.5 * (t_start + t_end);
        int phase = run_end;
        for (int cand = run_start; cand <= run_end; ++cand) {
            if (t_mid < starts[cand] + durations[cand]) {
                phase = cand;
                break;
            }
        }
        const double tau = t_mid - starts[phase];
        const auto pose = spline_pose(ax, phase, num_coeffs, tau, 0);
        const Vec3 pB = {pose[0], pose[1], pose[2]};
        const Mat3 R = rotation_matrix_zyx(pose[3], pose[4], pose[5]);
        const Vec3 r = {
            AD(hip_offsets12[3 * leg + 0]),
            AD(hip_offsets12[3 * leg + 1]),
            AD(hip_offsets12[3 * leg + 2]),
        };
        const Vec3 hip = matvec(R, r);
        const Vec3 p = foot_var(leg);
        const Vec3 diff = {
            pB[0] + hip[0] - p[0],
            pB[1] + hip[1] - p[1],
            pB[2] + hip[2] - p[2] - AD(h_des),
        };
        add_record(dot3(diff, diff), w_vb3);
    }

    // ---- V-B4 基座姿态/高度对齐 ----
    for (int p = 0; p < num_phases; ++p) {
        const double T = durations[p];
        const double weight = w_vb4 * T / double(num_samples);
        for (int s = 0; s < num_samples; ++s) {
            const double tau = T * double(s) / double(num_samples);
            const auto pose = spline_pose(ax, p, num_coeffs, tau, 0);
            const Vec3 pB = {pose[0], pose[1], pose[2]};
            const Mat3 R = rotation_matrix_zyx(pose[3], pose[4], pose[5]);
            const AD hs2 = map_query_ad(
                map_hs2, map_ny, map_nx, map_dx, map_dy,
                origin_x, origin_y, yaw, pB[0], pB[1]
            );
            for (int leg = 0; leg < NUM_LEGS; ++leg) {
                const Vec3 r = {
                    AD(hip_offsets12[3 * leg + 0]),
                    AD(hip_offsets12[3 * leg + 1]),
                    AD(hip_offsets12[3 * leg + 2]),
                };
                const Vec3 hip = matvec(R, r);
                const AD height = pB[2] + hip[2] - AD(h_des) - AD(hs2);
                add_record(height * height, weight);
            }
        }
    }

    // ---- V-B6 上一轮解 ----
    if (has_prev) {
        for (int k = 0; k < num_td; ++k) {
            const int leg = td_legs[k];
            const Vec3 p = foot_var(leg);
            const Vec3 prev = {
                AD(previous_footholds[3 * leg + 0]),
                AD(previous_footholds[3 * leg + 1]),
                AD(previous_footholds[3 * leg + 2]),
            };
            const Vec3 d = sub3(p, prev);
            add_record(dot3(d, d), w_vb6);
        }
    }

    // ---- V-B7 速度跟踪 ----
    for (int p = 0; p < num_phases; ++p) {
        const double T = durations[p];
        const double w7 = w_vb7 * T / double(num_samples);
        for (int s = 0; s < num_samples; ++s) {
            const double tau = T * double(s) / double(num_samples);
            const auto pose = spline_pose(ax, p, num_coeffs, tau, 0);
            const auto rate = spline_pose(ax, p, num_coeffs, tau, 1);
            const Mat3 R = rotation_matrix_zyx(pose[3], pose[4], pose[5]);
            const Vec3 v_w = {rate[0], rate[1], rate[2]};
            const Vec3 v_b = {
                R[0][0] * v_w[0] + R[1][0] * v_w[1] + R[2][0] * v_w[2],
                R[0][1] * v_w[0] + R[1][1] * v_w[1] + R[2][1] * v_w[2],
                R[0][2] * v_w[0] + R[1][2] * v_w[1] + R[2][2] * v_w[2],
            };
            const Vec3 omega = body_omega_zyx(pose, rate);
            const Vec3 dv = {v_b[0] - AD(ref_vx), v_b[1] - AD(ref_vy), v_b[2]};
            const Vec3 dw = {omega[0], omega[1], omega[2] - AD(ref_wz)};
            const Vec3 idw = matvec(inertia, dw);
            add_record(dot3(dv, dv) + dot3(idw, idw), w7);
        }
    }

    // ---- V-B8 平滑性 ----
    for (int p = 0; p < num_phases; ++p) {
        const double T = durations[p];
        const double w8 = w_vb8 * T / double(num_samples);
        for (int s = 0; s < num_samples; ++s) {
            const double tau = T * double(s) / double(num_samples);
            const auto pose = spline_pose(ax, p, num_coeffs, tau, 0);
            const auto rate = spline_pose(ax, p, num_coeffs, tau, 1);
            const auto acc = spline_pose(ax, p, num_coeffs, tau, 2);
            const Mat3 R = rotation_matrix_zyx(pose[3], pose[4], pose[5]);
            const Vec3 omega = body_omega_zyx(pose, rate);
            const Vec3 omega_dot = body_omega_rate_zyx(pose, rate, acc);
            const Vec3 Ldot = angular_momentum_rate(R, inertia, omega, omega_dot);
            add_record(dot3(Ldot, Ldot), w8);
        }
    }

    // ---- 式 (29) slack ----
    if (use_slack) {
        for (int p = 0; p < num_phases; ++p) {
            add_record(ax[slack_index(coeff_count, p)], w_slack);
        }
    }

    AD total = AD(0.0);
    for (size_t i = 0; i < records.size(); ++i) {
        total = total + AD(weights[i]) * records[i];
    }
    std::vector<AD> outputs = records;
    outputs.push_back(total);
    CppAD::ADFun<double> fun(ax, outputs);
    std::vector<double> xv(x, x + n);
    std::vector<double> y = fun.Forward(0, xv);
    const int rcount = int(records.size());
    if (record_residuals != nullptr) {
        for (int i = 0; i < rcount; ++i) record_residuals[i] = y[i];
    }
    if (record_weights != nullptr) {
        for (int i = 0; i < rcount; ++i) record_weights[i] = weights[i];
    }
    if (total_out != nullptr) *total_out = y[rcount];
    if (total_grad != nullptr) {
        std::vector<double> J = fun.Jacobian(xv);
        for (int i = 0; i < n; ++i) total_grad[i] = J[rcount * n + i];
    }
    *record_count = rcount;
    return 0;
}

extern "C" int tamols_map_gradients(
    int ny, int nx, const double* h,
    double dx, double dy,
    double* grad_x, double* grad_y
) {
    if (ny < 5 || nx < 5) return -1;
    diff1_axis_x(h, ny, nx, dx, grad_x);
    diff1_axis_y(h, ny, nx, dy, grad_y);
    return 0;
}

extern "C" int tamols_map_eval(
    int ny,
    int nx,
    const double* h,
    const double* hs1,
    const double* hs2,
    double dx,
    double dy,
    double origin_x,
    double origin_y,
    double yaw,
    int npoints,
    const double* px,
    const double* py,
    double* h_out,
    double* hs1_out,
    double* hs2_out
) {
    if (ny < 2 || nx < 2 || npoints < 0) return -1;
    const double c = std::cos(yaw);
    const double s = std::sin(yaw);
    for (int i = 0; i < npoints; ++i) {
        const double dxw = px[i] - origin_x;
        const double dyw = py[i] - origin_y;
        const double xm = c * dxw + s * dyw;
        const double ym = -s * dxw + c * dyw;
        h_out[i] = bilinear(h, ny, nx, xm, ym, dx, dy);
        hs1_out[i] = bilinear(hs1, ny, nx, xm, ym, dx, dy);
        hs2_out[i] = bilinear(hs2, ny, nx, xm, ym, dx, dy);
    }
    return 0;
}

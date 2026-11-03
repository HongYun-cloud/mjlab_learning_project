// TAMOLS C++ 求解器 (Step 4)
//
// 复用 tamols_core.cpp 的布局/样条/地形/CppAD 辅助函数.
// 本文件提供:
//   1. tamols_solver_eval: f/g/h/roots + gf/Jg/Jh/Jr + H
//   2. tamols_gn_sqp_solve: C++ GN-SQP + QuadProg++
//
// 构建时只编译本文件 (它 #include tamols_core.cpp)，避免重复符号.

#include "tamols_core.cpp"

#include <cstring>
#include <memory>
#include <string>

extern "C" int qp_solve_cpp(
    int n, int p, int m,
    const double* H, const double* g0,
    const double* CE, const double* ce0,
    const double* CI, const double* ci0,
    double* x_out, double* cost_out
);

extern "C" {

struct TamolsSolverProblem {
    int num_phases;
    int num_coeffs;
    int num_samples;
    int num_td;
    int use_slack;
    int giac_pairs_mode;
    int winding;
    int friction_mode;
    int friction_form;
    int include_push;
    int inertial_sign;
    int convexity_fourth;
    int has_prev;
    int include_dynamics;
    int include_kinematic;
    double h_des;
    double eps_min;
    double mu;
    double l_min;
    double l_max;
    double mass;
    double ref_vx;
    double ref_vy;
    double ref_wz;
    int map_ny;
    int map_nx;
    double map_dx;
    double map_dy;
    double origin_x;
    double origin_y;
    double yaw;
    double weights[9];

    const double* durations;
    const int* stance_mask;
    const int* measured_mask;
    const int* pin_mask;
    const int* td_legs;
    const double* measured_feet;
    const double* base_pose0;
    const double* base_twist0;
    const double* inertia9;
    const double* hip_offsets12;
    const double* previous_footholds;
    const double* map_h;
    const double* map_hs1;
    const double* map_hs2;
    const double* grad_hx;
    const double* grad_hy;
    const double* grad_s1x;
    const double* grad_s1y;
};

struct SolverEvalResult {
    int n = 0;
    int m_ineq = 0;
    int m_eq = 0;
    int n_roots = 0;
    double f = 0.0;
    std::vector<double> g;
    std::vector<double> h;
    std::vector<double> gf;
    std::vector<double> Jg;  // m_ineq x n
    std::vector<double> Jh;  // m_eq x n
    std::vector<double> H;   // n x n
    std::shared_ptr<CppAD::ADFun<double>> fun;
    std::shared_ptr<CppAD::vector<std::set<size_t>>> sparsity;
};

static void eval_internal(
    const TamolsSolverProblem& p,
    const double* x,
    SolverEvalResult& out,
    double hessian_reg
) {
    const int P = p.num_phases;
    const int K = p.num_coeffs;
    const int coeff_count = P * BASE_DIMS * K;
    const int n = coeff_count + NUM_LEGS * 3 + P;

    std::vector<AD> ax(n);
    for (int i = 0; i < n; ++i) ax[i] = x[i];

    CppAD::Independent(ax);
    std::vector<AD> g_expr;
    std::vector<AD> h_expr;
    std::vector<AD> roots_expr;
    std::vector<AD> records;
    std::vector<double> record_weights;

    const Mat3 inertia = inertia_matrix(p.inertia9);

    auto add_g = [&](const AD& expr) { g_expr.push_back(expr); };
    auto add_h = [&](const AD& expr) { h_expr.push_back(expr); };
    auto add_record = [&](const AD& residual, double weight,
                          const std::vector<AD>& roots) {
        records.push_back(residual);
        record_weights.push_back(weight);
        const double scale = std::sqrt(2.0 * weight);
        for (const AD& r : roots) roots_expr.push_back(AD(scale) * r);
    };
    const auto foot_var = [&](int leg) {
        return Vec3{
            ax[desired_index(coeff_count, leg, 0)],
            ax[desired_index(coeff_count, leg, 1)],
            ax[desired_index(coeff_count, leg, 2)],
        };
    };

    // ================= 约束 =================
    // V-A-1 初始位姿/速度
    for (int d = 0; d < BASE_DIMS; ++d) {
        add_h(spline_value(ax, 0, d, K, 0.0, 0) - AD(p.base_pose0[d]));
    }
    for (int d = 0; d < BASE_DIMS; ++d) {
        add_h(spline_value(ax, 0, d, K, 0.0, 1) - AD(p.base_twist0[d]));
    }
    // 不触地腿钉扎
    for (int leg = 0; leg < NUM_LEGS; ++leg) {
        if (!p.pin_mask[leg]) continue;
        for (int d = 0; d < 3; ++d) {
            add_h(ax[desired_index(coeff_count, leg, d)] -
                  AD(p.measured_feet[3 * leg + d]));
        }
    }
    // 相位连接 C0/C1
    for (int ph = 0; ph + 1 < P; ++ph) {
        const double T = p.durations[ph];
        for (int d = 0; d < BASE_DIMS; ++d) {
            add_h(spline_value(ax, ph, d, K, T, 0) -
                  spline_value(ax, ph + 1, d, K, 0.0, 0));
        }
        for (int d = 0; d < BASE_DIMS; ++d) {
            add_h(spline_value(ax, ph, d, K, T, 1) -
                  spline_value(ax, ph + 1, d, K, 0.0, 1));
        }
    }
    // slack <= 0
    if (p.use_slack) {
        for (int ph = 0; ph < P; ++ph) {
            add_g(ax[slack_index(coeff_count, ph)]);
        }
    }

    const bool cw = p.winding != 0;
    // 17b-17f + 摩擦
    if (p.include_dynamics) {
    for (int ph = 0; ph < P; ++ph) {
        const double T = p.durations[ph];
        for (int s = 0; s < p.num_samples; ++s) {
            const double tau = T * double(s) / double(p.num_samples);
            const auto pose = spline_pose(ax, ph, K, tau, 0);
            const auto rate = spline_pose(ax, ph, K, tau, 1);
            const auto acc = spline_pose(ax, ph, K, tau, 2);
            const Vec3 pB = {pose[0], pose[1], pose[2]};
            const Vec3 pdd = {acc[0], acc[1], acc[2]};
            Vec3 aB;
            if (p.inertial_sign == 0) {
                aB = {-pdd[0], -pdd[1], AD(GRAVITY_Z) - pdd[2]};
            } else {
                aB = {pdd[0], pdd[1], pdd[2] - AD(GRAVITY_Z)};
            }
            const Mat3 R = rotation_matrix_zyx(pose[3], pose[4], pose[5]);
            const Vec3 omega = body_omega_zyx(pose, rate);
            const Vec3 omega_dot = body_omega_rate_zyx(pose, rate, acc);
            const Vec3 Ldot = angular_momentum_rate(R, inertia, omega, omega_dot);

            const std::vector<int> ordered = ordered_stance(p.stance_mask, ph, cw);
            const int nfeet = int(ordered.size());
            if (nfeet >= 3) {
                std::vector<std::pair<int, int>> pairs;
                if (p.giac_pairs_mode == 0) {
                    for (int i = 0; i < nfeet; ++i) pairs.emplace_back(i, (i + 1) % nfeet);
                } else {
                    for (int i = 0; i < nfeet; ++i)
                        for (int j = i + 1; j < nfeet; ++j) pairs.emplace_back(i, j);
                }
                for (auto [i, j] : pairs) {
                    const Vec3 pi = foot_at(ax, coeff_count, p.stance_mask,
                                            p.measured_mask, p.measured_feet, ph, ordered[i]);
                    const Vec3 pj = foot_at(ax, coeff_count, p.stance_mask,
                                            p.measured_mask, p.measured_feet, ph, ordered[j]);
                    const Vec3 pij = sub3(pj, pi);
                    AD res = AD(p.mass) * det3(pij, sub3(pB, pi), aB) - dot3(pij, Ldot);
                    if (p.use_slack) res = res - ax[slack_index(coeff_count, ph)];
                    add_g(res);
                }
            } else if (nfeet == 2) {
                const Vec3 p1 = foot_at(ax, coeff_count, p.stance_mask,
                                        p.measured_mask, p.measured_feet, ph, ordered[0]);
                const Vec3 p2 = foot_at(ax, coeff_count, p.stance_mask,
                                        p.measured_mask, p.measured_feet, ph, ordered[1]);
                const Vec3 p12 = sub3(p2, p1);
                add_h(AD(p.mass) * det3(p12, sub3(pB, p1), aB) - dot3(p12, Ldot));
                const Vec3 m = cross3(sub3(pB, p1), aB);
                const Vec3 M1 = {
                    m[0] - Ldot[0] / AD(p.mass),
                    m[1] - Ldot[1] / AD(p.mass),
                    m[2] - Ldot[2] / AD(p.mass),
                };
                AD res = -det3(Vec3{AD(0.0), AD(0.0), AD(1.0)}, p12, M1);
                if (p.use_slack) res = res - ax[slack_index(coeff_count, ph)];
                add_g(res);
            } else if (nfeet == 1) {
                const Vec3 pi = foot_at(ax, coeff_count, p.stance_mask,
                                        p.measured_mask, p.measured_feet, ph, ordered[0]);
                const Vec3 m = cross3(sub3(pB, pi), aB);
                for (int d = 0; d < 3; ++d) add_h(AD(p.mass) * m[d] - Ldot[d]);
            } else {
                for (int d = 0; d < 3; ++d) add_h(aB[d]);
                for (int d = 0; d < 3; ++d) add_h(Ldot[d]);
            }

            if (nfeet > 0 && p.friction_mode != 0) {
                if (p.friction_mode == 2) {
                    add_g(aB[0] * aB[0] + aB[1] * aB[1] -
                          (AD(p.mu) * (-aB[2])) * (AD(p.mu) * (-aB[2])));
                    add_g(aB[2]);
                } else if (p.friction_form == 0) {
                    add_g(CppAD::sqrt(aB[0] * aB[0] + aB[1] * aB[1]) - AD(p.mu) * aB[2]);
                } else {
                    add_g(aB[0] * aB[0] + aB[1] * aB[1] -
                          (AD(p.mu) * aB[2]) * (AD(p.mu) * aB[2]));
                    add_g(-aB[2]);
                }
                if (p.include_push) add_g(AD(GRAVITY_Z) - pdd[2]);
            }
        }
    }

    // 30 凸性 + 31 可达
    for (int ph = 0; ph < P; ++ph) {
        const std::vector<int> stance = stance_legs(p.stance_mask, ph);
        if (stance.size() == 4) {
            const std::vector<int> ordered = ordered_stance(p.stance_mask, ph, false);
            const Vec3 p1 = foot_at(ax, coeff_count, p.stance_mask, p.measured_mask,
                                    p.measured_feet, ph, ordered[0]);
            const Vec3 p2 = foot_at(ax, coeff_count, p.stance_mask, p.measured_mask,
                                    p.measured_feet, ph, ordered[1]);
            const Vec3 p3 = foot_at(ax, coeff_count, p.stance_mask, p.measured_mask,
                                    p.measured_feet, ph, ordered[2]);
            const Vec3 p4 = foot_at(ax, coeff_count, p.stance_mask, p.measured_mask,
                                    p.measured_feet, ph, ordered[3]);
            const auto crossz = [](const Vec3& a, const Vec3& b) {
                return a[0] * b[1] - a[1] * b[0];
            };
            const Vec3 p12 = sub3(p2, p1), p13 = sub3(p3, p1), p14 = sub3(p4, p1);
            const Vec3 p23 = sub3(p3, p2), p24 = sub3(p4, p2);
            add_g(crossz(p13, p12));
            add_g(crossz(p14, p13));
            add_g(crossz(p24, p23));
            if (p.convexity_fourth == 0) {
                add_g(crossz(sub3(p1, p2), p24));
            } else {
                const Vec3 p20 = {-p2[0], -p2[1], -p2[2]};
                add_g(crossz(p20, p24));
            }
        }
        const double T = p.durations[ph];
        for (int s = 0; s < p.num_samples; ++s) {
            const double tau = T * double(s) / double(p.num_samples);
            const auto pose = spline_pose(ax, ph, K, tau, 0);
            const Vec3 pB = {pose[0], pose[1], pose[2]};
            const Mat3 R = rotation_matrix_zyx(pose[3], pose[4], pose[5]);
            for (int leg : stance) {
                const Vec3 foot = foot_at(ax, coeff_count, p.stance_mask, p.measured_mask,
                                          p.measured_feet, ph, leg);
                const Vec3 r = {
                    AD(p.hip_offsets12[3 * leg + 0]),
                    AD(p.hip_offsets12[3 * leg + 1]),
                    AD(p.hip_offsets12[3 * leg + 2]),
                };
                const Vec3 hip = matvec(R, r);
                const Vec3 d = {
                    pB[0] + hip[0] - foot[0],
                    pB[1] + hip[1] - foot[1],
                    pB[2] + hip[2] - foot[2],
                };
                const AD d2 = dot3(d, d);
                add_g(AD(p.l_min * p.l_min) - d2);
                add_g(d2 - AD(p.l_max * p.l_max));
            }
        }
    }

    }  // include_kinematic

    // ================= 目标 =================
    const double w1 = p.weights[0], w2 = p.weights[1], w3 = p.weights[2];
    const double w4 = p.weights[3], w5 = p.weights[4], w6 = p.weights[5];
    const double w7 = p.weights[6], w8 = p.weights[7], ws = p.weights[8];
    (void)w1; (void)w2; (void)w3; (void)w4; (void)w5; (void)w6; (void)w7; (void)w8;

    // V-B1
    for (int k = 0; k < p.num_td; ++k) {
        const int leg = p.td_legs[k];
        const Vec3 q = foot_var(leg);
        const AD hmap = map_query_ad(p.map_h, p.map_ny, p.map_nx, p.map_dx, p.map_dy,
                                     p.origin_x, p.origin_y, p.yaw, q[0], q[1]);
        const AD r = hmap - q[2];
        add_record(r * r, p.weights[0], {r});
    }
    // V-B2
    for (int a = 0; a < p.num_td; ++a) {
        for (int b = a + 1; b < p.num_td; ++b) {
            const Vec3 pi = foot_var(p.td_legs[a]);
            const Vec3 pj = foot_var(p.td_legs[b]);
            const AD z = (pi[0] - pj[0]) * (pi[0] - pj[0]) +
                         (pi[1] - pj[1]) * (pi[1] - pj[1]);
            const AD gap = max_ad(AD(p.eps_min) - z, 0.0);
            add_record(gap * gap, p.weights[1], {gap});
        }
    }
    // V-B5
    for (int k = 0; k < p.num_td; ++k) {
        const int leg = p.td_legs[k];
        const Vec3 q = foot_var(leg);
        const AD gx = map_query_ad(p.grad_hx, p.map_ny, p.map_nx, p.map_dx, p.map_dy,
                                   p.origin_x, p.origin_y, p.yaw, q[0], q[1]);
        const AD gy = map_query_ad(p.grad_hy, p.map_ny, p.map_nx, p.map_dx, p.map_dy,
                                   p.origin_x, p.origin_y, p.yaw, q[0], q[1]);
        const AD sx = map_query_ad(p.grad_s1x, p.map_ny, p.map_nx, p.map_dx, p.map_dy,
                                   p.origin_x, p.origin_y, p.yaw, q[0], q[1]);
        const AD sy = map_query_ad(p.grad_s1y, p.map_ny, p.map_nx, p.map_dx, p.map_dy,
                                   p.origin_x, p.origin_y, p.yaw, q[0], q[1]);
        add_record(gx * gx + gy * gy + sx * sx + sy * sy, p.weights[4], {gx, gy, sx, sy});
    }
    // V-B3
    std::vector<double> starts(P, 0.0);
    for (int ph = 1; ph < P; ++ph) starts[ph] = starts[ph - 1] + p.durations[ph - 1];
    for (int k = 0; k < p.num_td; ++k) {
        const int leg = p.td_legs[k];
        int run_start = -1, run_end = -1;
        for (int ph = 0; ph < P; ++ph) {
            if (p.stance_mask[ph * NUM_LEGS + leg]) {
                if (run_start < 0) run_start = ph;
                run_end = ph;
            } else if (run_start >= 0) break;
        }
        if (run_start < 0) continue;
        const double t_start = starts[run_start];
        const double t_end = starts[run_end] + p.durations[run_end];
        const double t_mid = 0.5 * (t_start + t_end);
        int phase = run_end;
        for (int cand = run_start; cand <= run_end; ++cand) {
            if (t_mid < starts[cand] + p.durations[cand]) { phase = cand; break; }
        }
        const double tau = t_mid - starts[phase];
        const auto pose = spline_pose(ax, phase, K, tau, 0);
        const Vec3 pB = {pose[0], pose[1], pose[2]};
        const Mat3 R = rotation_matrix_zyx(pose[3], pose[4], pose[5]);
        const Vec3 r = {
            AD(p.hip_offsets12[3 * leg + 0]),
            AD(p.hip_offsets12[3 * leg + 1]),
            AD(p.hip_offsets12[3 * leg + 2]),
        };
        const Vec3 hip = matvec(R, r);
        const Vec3 q = foot_var(leg);
        const Vec3 diff = {
            pB[0] + hip[0] - q[0],
            pB[1] + hip[1] - q[1],
            pB[2] + hip[2] - q[2] - AD(p.h_des),
        };
        add_record(dot3(diff, diff), p.weights[2], {diff[0], diff[1], diff[2]});
    }
    // V-B4
    for (int ph = 0; ph < P; ++ph) {
        const double T = p.durations[ph];
        const double weight = p.weights[3] * T / double(p.num_samples);
        for (int s = 0; s < p.num_samples; ++s) {
            const double tau = T * double(s) / double(p.num_samples);
            const auto pose = spline_pose(ax, ph, K, tau, 0);
            const Vec3 pB = {pose[0], pose[1], pose[2]};
            const Mat3 R = rotation_matrix_zyx(pose[3], pose[4], pose[5]);
            const AD hs2 = map_query_ad(p.map_hs2, p.map_ny, p.map_nx, p.map_dx, p.map_dy,
                                         p.origin_x, p.origin_y, p.yaw, pB[0], pB[1]);
            for (int leg = 0; leg < NUM_LEGS; ++leg) {
                const Vec3 r = {
                    AD(p.hip_offsets12[3 * leg + 0]),
                    AD(p.hip_offsets12[3 * leg + 1]),
                    AD(p.hip_offsets12[3 * leg + 2]),
                };
                const Vec3 hip = matvec(R, r);
                const AD root = pB[2] + hip[2] - AD(p.h_des) - hs2;
                add_record(root * root, weight, {root});
            }
        }
    }
    // V-B6
    if (p.has_prev) {
        for (int k = 0; k < p.num_td; ++k) {
            const int leg = p.td_legs[k];
            const Vec3 q = foot_var(leg);
            const Vec3 prev = {
                AD(p.previous_footholds[3 * leg + 0]),
                AD(p.previous_footholds[3 * leg + 1]),
                AD(p.previous_footholds[3 * leg + 2]),
            };
            const Vec3 d = sub3(q, prev);
            add_record(dot3(d, d), p.weights[5], {d[0], d[1], d[2]});
        }
    }
    // V-B7
    for (int ph = 0; ph < P; ++ph) {
        const double T = p.durations[ph];
        const double weight = p.weights[6] * T / double(p.num_samples);
        for (int s = 0; s < p.num_samples; ++s) {
            const double tau = T * double(s) / double(p.num_samples);
            const auto pose = spline_pose(ax, ph, K, tau, 0);
            const auto rate = spline_pose(ax, ph, K, tau, 1);
            const Mat3 R = rotation_matrix_zyx(pose[3], pose[4], pose[5]);
            const Vec3 v_w = {rate[0], rate[1], rate[2]};
            const Vec3 v_b = {
                R[0][0] * v_w[0] + R[1][0] * v_w[1] + R[2][0] * v_w[2],
                R[0][1] * v_w[0] + R[1][1] * v_w[1] + R[2][1] * v_w[2],
                R[0][2] * v_w[0] + R[1][2] * v_w[1] + R[2][2] * v_w[2],
            };
            const Vec3 omega = body_omega_zyx(pose, rate);
            const Vec3 dv = {v_b[0] - AD(p.ref_vx), v_b[1] - AD(p.ref_vy), v_b[2]};
            const Vec3 dw = {omega[0], omega[1], omega[2] - AD(p.ref_wz)};
            const Vec3 idw = matvec(inertia, dw);
            const AD res = dot3(dv, dv) + dot3(idw, idw);
            add_record(res, weight, {dv[0], dv[1], dv[2], idw[0], idw[1], idw[2]});
        }
    }
    // V-B8
    for (int ph = 0; ph < P; ++ph) {
        const double T = p.durations[ph];
        const double weight = p.weights[7] * T / double(p.num_samples);
        for (int s = 0; s < p.num_samples; ++s) {
            const double tau = T * double(s) / double(p.num_samples);
            const auto pose = spline_pose(ax, ph, K, tau, 0);
            const auto rate = spline_pose(ax, ph, K, tau, 1);
            const auto acc = spline_pose(ax, ph, K, tau, 2);
            const Mat3 R = rotation_matrix_zyx(pose[3], pose[4], pose[5]);
            const Vec3 omega = body_omega_zyx(pose, rate);
            const Vec3 omega_dot = body_omega_rate_zyx(pose, rate, acc);
            const Vec3 Ldot = angular_momentum_rate(R, inertia, omega, omega_dot);
            add_record(dot3(Ldot, Ldot), weight, {Ldot[0], Ldot[1], Ldot[2]});
        }
    }
    // slack objective
    if (p.use_slack) {
        for (int ph = 0; ph < P; ++ph) {
            add_record(ax[slack_index(coeff_count, ph)], p.weights[8], {});
        }
    }

    AD total = AD(0.0);
    for (size_t i = 0; i < records.size(); ++i) {
        total = total + AD(record_weights[i]) * records[i];
    }
    std::vector<AD> outputs;
    outputs.push_back(total);
    outputs.insert(outputs.end(), g_expr.begin(), g_expr.end());
    outputs.insert(outputs.end(), h_expr.begin(), h_expr.end());
    outputs.insert(outputs.end(), roots_expr.begin(), roots_expr.end());

    auto fun_ptr = std::make_shared<CppAD::ADFun<double>>(ax, outputs);
    CppAD::ADFun<double>& fun = *fun_ptr;
    out.fun = fun_ptr;
    std::vector<double> xv(x, x + n);
    std::vector<double> y = fun.Forward(0, xv);
    // 稀疏 Jacobian: 先算一次列稀疏模式, 之后每轮复用.
    auto sparsity = std::make_shared<CppAD::vector<std::set<size_t>>>(int(outputs.size()));
    CppAD::vector<std::set<size_t>> r(n);
    for (int i = 0; i < n; ++i) r[i].insert(size_t(i));
    *sparsity = fun.ForSparseJac(size_t(n), r);
    out.sparsity = sparsity;
    std::vector<double> J = fun.SparseJacobian(xv, *sparsity);

    out.n = n;
    out.m_ineq = int(g_expr.size());
    out.m_eq = int(h_expr.size());
    out.n_roots = int(roots_expr.size());
    out.f = y[0];
    out.g.resize(out.m_ineq);
    out.h.resize(out.m_eq);
    for (int i = 0; i < out.m_ineq; ++i) out.g[i] = y[1 + i];
    for (int i = 0; i < out.m_eq; ++i) out.h[i] = y[1 + out.m_ineq + i];
    out.gf.resize(n);
    for (int j = 0; j < n; ++j) out.gf[j] = J[j];
    out.Jg.assign(out.m_ineq * n, 0.0);
    for (int i = 0; i < out.m_ineq; ++i)
        for (int j = 0; j < n; ++j)
            out.Jg[i * n + j] = J[(1 + i) * n + j];
    out.Jh.assign(out.m_eq * n, 0.0);
    for (int i = 0; i < out.m_eq; ++i)
        for (int j = 0; j < n; ++j)
            out.Jh[i * n + j] = J[(1 + out.m_ineq + i) * n + j];
    // H = Jr^T Jr + reg I
    out.H.assign(n * n, 0.0);
    const int root_offset = 1 + out.m_ineq + out.m_eq;
    for (int r = 0; r < out.n_roots; ++r) {
        for (int i = 0; i < n; ++i) {
            const double vi = J[(root_offset + r) * n + i];
            for (int j = 0; j < n; ++j) {
                out.H[i * n + j] += vi * J[(root_offset + r) * n + j];
            }
        }
    }
    for (int i = 0; i < n; ++i) out.H[i * n + i] += hessian_reg;
}


static void eval_from_fun(
    const SolverEvalResult& base,
    const double* x,
    SolverEvalResult& out,
    double hessian_reg
) {
    out.n = base.n;
    out.m_ineq = base.m_ineq;
    out.m_eq = base.m_eq;
    out.n_roots = base.n_roots;
    out.fun = base.fun;
    out.sparsity = base.sparsity;
    if (!base.fun) return;
    const int n = base.n;
    std::vector<double> xv(x, x + n);
    std::vector<double> y = base.fun->Forward(0, xv);
    std::vector<double> J = base.sparsity
        ? base.fun->SparseJacobian(xv, *base.sparsity)
        : base.fun->Jacobian(xv);

    out.f = y[0];
    out.g.resize(out.m_ineq);
    out.h.resize(out.m_eq);
    for (int i = 0; i < out.m_ineq; ++i) out.g[i] = y[1 + i];
    for (int i = 0; i < out.m_eq; ++i) out.h[i] = y[1 + out.m_ineq + i];
    out.gf.resize(n);
    for (int j = 0; j < n; ++j) out.gf[j] = J[j];
    out.Jg.assign(out.m_ineq * n, 0.0);
    for (int i = 0; i < out.m_ineq; ++i)
        for (int j = 0; j < n; ++j)
            out.Jg[i * n + j] = J[(1 + i) * n + j];
    out.Jh.assign(out.m_eq * n, 0.0);
    for (int i = 0; i < out.m_eq; ++i)
        for (int j = 0; j < n; ++j)
            out.Jh[i * n + j] = J[(1 + out.m_ineq + i) * n + j];
    out.H.assign(n * n, 0.0);
    const int root_offset = 1 + out.m_ineq + out.m_eq;
    for (int r = 0; r < out.n_roots; ++r) {
        for (int i = 0; i < n; ++i) {
            const double vi = J[(root_offset + r) * n + i];
            for (int j = 0; j < n; ++j) {
                out.H[i * n + j] += vi * J[(root_offset + r) * n + j];
            }
        }
    }
    for (int i = 0; i < n; ++i) out.H[i * n + i] += hessian_reg;
}

int tamols_solver_eval(
    const TamolsSolverProblem* p,
    const double* x,
    double* f_out,
    double* g_out,
    double* h_out,
    double* gf_out,
    double* Jg_out,
    double* Jh_out,
    double* H_out,
    int* n_out,
    int* mg_out,
    int* mh_out,
    double hessian_reg
) {
    if (p == nullptr) return -1;
    SolverEvalResult res;
    eval_internal(*p, x, res, hessian_reg);
    if (f_out) *f_out = res.f;
    if (g_out) std::memcpy(g_out, res.g.data(), res.g.size() * sizeof(double));
    if (h_out) std::memcpy(h_out, res.h.data(), res.h.size() * sizeof(double));
    if (gf_out) std::memcpy(gf_out, res.gf.data(), res.gf.size() * sizeof(double));
    if (Jg_out) std::memcpy(Jg_out, res.Jg.data(), res.Jg.size() * sizeof(double));
    if (Jh_out) std::memcpy(Jh_out, res.Jh.data(), res.Jh.size() * sizeof(double));
    if (H_out) std::memcpy(H_out, res.H.data(), res.H.size() * sizeof(double));
    if (n_out) *n_out = res.n;
    if (mg_out) *mg_out = res.m_ineq;
    if (mh_out) *mh_out = res.m_eq;
    return 0;
}


static void eval_value_from_fun(
    const SolverEvalResult& base,
    const double* x,
    double& f,
    std::vector<double>& g,
    std::vector<double>& h
) {
    if (!base.fun) return;
    std::vector<double> xv(x, x + base.n);
    std::vector<double> y = base.fun->Forward(0, xv);
    f = y[0];
    g.resize(base.m_ineq);
    h.resize(base.m_eq);
    for (int i = 0; i < base.m_ineq; ++i) g[i] = y[1 + i];
    for (int i = 0; i < base.m_eq; ++i) h[i] = y[1 + base.m_ineq + i];
}

int tamols_gn_sqp_solve(
    const TamolsSolverProblem* p,
    const double* x0,
    int max_iter,
    double tol_step,
    double tol_viol,
    double tol_stationary,
    double qp_lin_tol,
    double merit_rho,
    double armijo_eta,
    double t_min,
    double hessian_reg,
    double* x_out,
    double* f_out,
    int* iterations_out,
    double* max_viol_out,
    char* status_out,
    int status_len
) {
    if (p == nullptr) return -1;
    SolverEvalResult ev;
    eval_internal(*p, x0, ev, hessian_reg);
    const int n = ev.n;
    std::vector<double> x(x0, x0 + n);
    double f = ev.f;
    double prev_f = f;
    std::string status = "GN-SQP: max_iter";
    int iterations = 0;

    auto violation = [](const std::vector<double>& g, const std::vector<double>& h) {
        double v = 0.0;
        for (double gi : g) if (gi > 0.0) v += gi;
        for (double hi : h) v += std::abs(hi);
        return v;
    };

    for (int it = 0; it < max_iter; ++it) {
        iterations = it + 1;
        // H = Jr^T Jr + reg I already in ev.H
        const int mg = ev.m_ineq, mh = ev.m_eq;
        // QP: min 0.5 d'H d + gf'd, -Jg d - g >= 0, Jh d + h = 0
        std::vector<double> CE = ev.Jh;
        std::vector<double> ce0 = ev.h;
        std::vector<double> CI(mg * n, 0.0), ci0(mg, 0.0);
        for (int i = 0; i < mg; ++i) {
            for (int j = 0; j < n; ++j) CI[i * n + j] = -ev.Jg[i * n + j];
            ci0[i] = -ev.g[i];
        }

        std::vector<double> d(n, 0.0);
        int rc = 0;
        // 与 Python 版一致: QP 数值失败时先加大 LM 阻尼重试.
        for (double damp : {0.0, 1e-6, 1e-4, 1e-2}) {
            std::vector<double> H = ev.H;
            std::vector<double> gf = ev.gf;
            if (damp > 0.0) {
                for (int i = 0; i < n; ++i) H[i * n + i] += damp;
            }
            std::vector<double> D(n, 1.0);
            double max_abs_diag = 1.0;
            for (int i = 0; i < n; ++i) max_abs_diag = std::max(max_abs_diag, std::abs(H[i * n + i]));
            for (int i = 0; i < n; ++i) {
                double diag = H[i * n + i];
                if (!(diag > 0.0)) diag = max_abs_diag;
                D[i] = 1.0 / std::sqrt(diag);
            }
            for (int i = 0; i < n; ++i) {
                gf[i] *= D[i];
                for (int j = 0; j < n; ++j) H[i * n + j] *= D[i] * D[j];
                for (int r = 0; r < mh; ++r) CE[r * n + i] *= D[i];
                for (int r = 0; r < mg; ++r) CI[r * n + i] *= D[i];
            }
            std::vector<double> z(n, 0.0), cost(1, 0.0);
            rc = qp_solve_cpp(n, mh, mg, H.data(), gf.data(),
                              CE.data(), ce0.data(), CI.data(), ci0.data(),
                              z.data(), cost.data());
            if (rc == 0) {
                for (int i = 0; i < n; ++i) d[i] = D[i] * z[i];
                break;
            }
        }
        if (rc != 0) {
            status = (rc == 1) ? "GN-SQP: qp_infeasible" : "GN-SQP: qp_failed";
            break;
        }

        // 线性化约束违反
        double lin_viol = 0.0;
        for (int i = 0; i < mg; ++i) {
            double v = 0.0;
            for (int j = 0; j < n; ++j) v += ev.Jg[i * n + j] * d[j];
            v += ev.g[i];
            if (v > lin_viol) lin_viol = v;
        }
        for (int i = 0; i < mh; ++i) {
            double v = 0.0;
            for (int j = 0; j < n; ++j) v += ev.Jh[i * n + j] * d[j];
            v += ev.h[i];
            lin_viol = std::max(lin_viol, std::abs(v));
        }
        if (lin_viol > qp_lin_tol) {
            status = "GN-SQP: qp_lin_violation";
            break;
        }

        double step_inf = 0.0;
        for (double di : d) step_inf = std::max(step_inf, std::abs(di));
        const double viol_now = violation(ev.g, ev.h);
        if (step_inf < tol_step) {
            status = (viol_now < tol_viol) ? "GN-SQP: converged" : "GN-SQP: stationary_infeasible";
            break;
        }
        double d0 = 0.0;
        for (int i = 0; i < n; ++i) d0 += ev.gf[i] * d[i];
        d0 -= merit_rho * viol_now;
        const double phi0 = f + merit_rho * viol_now;
        double t = 1.0;
        bool accepted = false;
        std::vector<double> trial = x;
        double f_trial = f;
        std::vector<double> g_trial, h_trial;
        while (t >= t_min) {
            for (int i = 0; i < n; ++i) trial[i] = x[i] + t * d[i];
            eval_value_from_fun(ev, trial.data(), f_trial, g_trial, h_trial);
            const double phi = f_trial + merit_rho * violation(g_trial, h_trial);
            if (phi <= phi0 + armijo_eta * t * d0) {
                accepted = true;
                break;
            }
            t *= 0.5;
        }
        if (!accepted) {
            if (viol_now < tol_viol && std::abs(f - prev_f) <= tol_stationary * std::max(1.0, std::abs(f))) {
                status = "GN-SQP: converged";
            } else {
                status = "GN-SQP: line_search_stalled";
            }
            break;
        }
        prev_f = f;
        x.swap(trial);
        f = f_trial;
        ev.f = f_trial;
        ev.g = g_trial;
        ev.h = h_trial;
        if (violation(ev.g, ev.h) < tol_viol &&
            (step_inf * t < tol_step || d0 >= -tol_step)) {
            status = "GN-SQP: converged";
            break;
        }
        // atomic 地图插值允许同一个 ADFun 在任意 x 上精确复用.
        SolverEvalResult next;
        eval_from_fun(ev, x.data(), next, hessian_reg);
        ev = next;
    }
    if (status == "GN-SQP: converged" && violation(ev.g, ev.h) > tol_viol) {
        status = "GN-SQP: converged_with_violation";
    }
    std::memcpy(x_out, x.data(), n * sizeof(double));
    if (f_out) *f_out = f;
    if (iterations_out) *iterations_out = iterations;
    if (max_viol_out) *max_viol_out = violation(ev.g, ev.h);
    if (status_out && status_len > 0) {
        std::snprintf(status_out, status_len, "%s", status.c_str());
    }
    return 0;
}

}  // extern "C"

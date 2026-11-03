// QuadProg++ (C++, Goldfarb-Idnani) 的 extern "C" shim.
//
// 构建 (见 build_shim.sh):
//   g++ -O2 -fPIC -shared -DMATRIX_DIM=1024 -I<QuadProg++/src> \
//       quadprog_shim.cc <QuadProg++/src>/QuadProg++.cc <QuadProg++/src>/Array.cc \
//       -o libquadprog_shim.so
//
// 约定 (全部行优先 C 序):
//   min 0.5 d'H d + g0'd
//   s.t.  CE_row_i d + ce0_i = 0     (CE: p x n)
//         CI_row_k d + ci0_k >= 0    (CI: m x n)
// 与 QuadProg++ 内部约定 (CE^T x + ce0 = 0, 列=约束) 的转置在本文件完成.
// 返回: 0 成功; 1 不可行 (solve_quadprog 返回 infinity); 2 C++ 异常.

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include "QuadProg++.hh"

extern "C" int qp_solve_cpp(int n, int p, int m,
                            const double* H, const double* g0,
                            const double* CE, const double* ce0,
                            const double* CI, const double* ci0,
                            double* x_out, double* cost_out) {
  using quadprogpp::Matrix;
  using quadprogpp::Vector;
  using quadprogpp::solve_quadprog;
  try {
    Matrix<double> G(n, n), CEm(n, p), CIm(n, m);
    Vector<double> g0v(n), ce0v(p), ci0v(m), x(n);
    for (int i = 0; i < n; ++i) {
      g0v[i] = g0[i];
      for (int j = 0; j < n; ++j) G[i][j] = H[i * n + j];
    }
    for (int j = 0; j < p; ++j) {
      ce0v[j] = ce0[j];
      for (int i = 0; i < n; ++i) CEm[i][j] = CE[j * n + i];
    }
    for (int k = 0; k < m; ++k) {
      ci0v[k] = ci0[k];
      for (int i = 0; i < n; ++i) CIm[i][k] = CI[k * n + i];
    }
    double cost = 0.0;
    const bool timing = std::getenv("TAMOLS_QP_TIMING") != nullptr;
    auto t0 = std::chrono::steady_clock::now();
    if (timing) {
      std::fprintf(stderr, "[QP] start n=%d p=%d m=%d\n", n, p, m);
      std::fflush(stderr);
    }
    cost = solve_quadprog(G, g0v, CEm, ce0v, CIm, ci0v, x);
    if (timing) {
      double ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
      std::fprintf(stderr, "[QP] done n=%d p=%d m=%d %.3f ms cost=%g\n",
                   n, p, m, ms, cost);
      std::fflush(stderr);
    }
    if (!std::isfinite(cost)) return 1;
    for (int i = 0; i < n; ++i) x_out[i] = x[i];
    if (cost_out != nullptr) *cost_out = cost;
    return 0;
  } catch (...) {
    return 2;
  }
}

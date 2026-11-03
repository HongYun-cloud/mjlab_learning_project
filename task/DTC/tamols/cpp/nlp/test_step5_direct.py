"""Step 5: C++ 直接问题组装与既有 Step 4 打包逐项对拍.

不依赖 ``IntegratedNLP`` 的 CasADi 表达式: 从纯数组/状态构造
``TamolsSolverProblem``, 与 ``test_step4`` 中独立维护的 C 结构体比较
f/g/h/gf/Jg/Jh/H. 通过后再允许 DTC 运行时切换到 direct builder.
"""

from __future__ import annotations

import numpy as np

from myrobot_mjlab.task.DTC.tamols.cpp_solver import (  # noqa: E402
    eval_problem,
    make_problem_from_data,
)
from myrobot_mjlab.task.DTC.tamols.cpp.nlp.test_step4 import (  # noqa: E402
    cpp_eval,
    setup,
)


def _build_direct(nlp):
    """按 test_step4 的独立打包方式构造 direct problem (不使用 make_problem)."""
    P = nlp.layout.num_phases
    stance = np.zeros(P * 4, dtype=np.int32)
    measured = np.zeros(P * 4, dtype=np.int32)
    for ph in range(P):
        for leg in range(4):
            stance[ph * 4 + leg] = int(nlp.schedules[leg].in_stance(ph))
            measured[ph * 4 + leg] = int(
                nlp.schedules[leg].foot_is_measured(ph)
            )
    pin = np.array(
        [0 if nlp.schedules[leg].touchdown_leg else 1 for leg in range(4)],
        dtype=np.int32,
    )
    measured_feet = np.zeros(12)
    for leg in range(4):
        if leg in nlp.params.measured_feet:
            measured_feet[3 * leg : 3 * leg + 3] = nlp.params.measured_feet[leg]

    return make_problem_from_data(
        durations=nlp.durations,
        schedules=nlp.schedules,
        maps=nlp.maps,
        params=nlp.params,
        assumptions=nlp.assumptions,
        td_legs=nlp.touchdown_legs(),
        ref_vx=0.25,
        ref_vy=0.0,
        ref_wz=0.0,
        previous_footholds=nlp.params.previous_footholds,
    )


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0:
        return 0.0
    scale = max(1.0, float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    return float(np.max(np.abs(a - b)) / scale)


def main() -> int:
    nlp, x0, _blocks, step4_problem, _keep = setup()
    direct = _build_direct(nlp)
    direct.struct.num_phases = nlp.layout.num_phases
    direct.struct.num_coeffs = nlp.layout.num_coeffs

    rng = np.random.default_rng(0)
    samples = [x0]
    for scale in (1e-3, 1e-2):
        samples.append(x0 + scale * rng.standard_normal(x0.shape))

    worst = 0.0
    for k, x in enumerate(samples):
        f4, g4, h4, gf4, Jg4, Jh4, H4, n4 = cpp_eval(step4_problem, x)
        ref = eval_problem(direct, x)
        assert ref["n"] == n4, (ref["n"], n4)
        errs = {
            "f": _rel_err(f4, ref["f"]),
            "g": _rel_err(g4, ref["g"]),
            "h": _rel_err(h4, ref["h"]),
            "gf": _rel_err(gf4, ref["gf"]),
            "Jg": _rel_err(Jg4, ref["Jg"].reshape(-1)),
            "Jh": _rel_err(Jh4, ref["Jh"].reshape(-1)),
            "H": _rel_err(H4, ref["H"].reshape(-1)),
        }
        worst = max(worst, *errs.values())
        print(
            f"sample {k}: "
            + " ".join(f"{name}={val:.3e}" for name, val in errs.items())
        )

    # 解也应对齐 (同一初值/参数, 两种独立打包).
    x4, f4, it4, st4, v4 = _solve_step4(step4_problem, x0)
    xd, fd, itd, std, vd = direct.solve(
        x0, max_iter=5, tol_step=1e-6, tol_viol=1e-4,
        tol_stationary=1e-4, qp_lin_tol=1e-4,
    )
    dx = float(np.max(np.abs(x4 - xd)))
    print(
        f"solve: step4={st4} it={it4} f={f4:.9g} | "
        f"direct={std} it={itd} f={fd:.9g} | max|x|={dx:.3e}"
    )
    if worst > 1e-10 or dx > 1e-6:
        print(f"STEP 5: FAIL (worst eval={worst:.3e}, dx={dx:.3e})")
        return 1
    print(f"STEP 5: PASS (worst eval={worst:.3e}, dx={dx:.3e})")
    return 0


def _solve_step4(problem, x0):
    import myrobot_mjlab.task.DTC.tamols.cpp.nlp.test_step4 as s4

    return s4.cpp_solve(problem, x0, max_iter=5)


if __name__ == "__main__":
    raise SystemExit(main())

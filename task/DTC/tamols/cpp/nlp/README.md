# TAMOLS C++ NLP port (B)

目标：用 C++/Eigen + CppAD + QuadProg++ 复现论文求解器栈，逐阶段与现有
Python/CasADi 实现对齐。

## 当前进度

### Step 1 — 已完成并通过验证

- 变量布局：`spline_coeffs (P,6,5) | desired_footholds (4,3) | slack (P,)`
- 6D 相位样条（位置/一阶导）
- 初始位姿/速度约束
- 相邻相位 C0/C1 连接约束
- horizon 内不触地腿的 `desired_footholds` 钉扎
- 局部高程图 `h / h_s1 / h_s2` 双线性查询（含 origin/yaw/clamp）

验证脚本：`test_step1.py`

```text
P=3, K=5, n=105
max |r_cpp - r_py| = 0.000e+00
max |J_cpp - J_py| = 1.679e-10   # Python 中心差分 vs CppAD
max |h_cpp - h_py|   = 0.000e+00
max |s1_cpp - s1_py| = 0.000e+00
max |s2_cpp - s2_py| = 0.000e+00
STEP 1: PASS
```

C++ 残差+Jacobian 耗时（含 CppAD `ADFun` 构造）：

```text
0.329 ms/call (42x105)
```

### Step 2 — 已完成并通过验证

- GIAC 17b / 17c / 17d / 17e / 17f
- 摩擦锥 5 / V-A-4（paper_literal norm/squared、force_balance_derived、
  enabled/disabled）
- 式 (30) 足迹凸性（p21_engineering / p20_literal）
- 式 (31) 腿长可达
- GIAC slack（式 29）
- GIAC cyclic_edges / all_unordered
- winding ccw / cw

验证脚本：`test_step2.py`，对 6 组假设逐项对拍：

```text
[PASS] contact_normal + paper friction: m=265, max|r|=7.11e-15, max|J|=1.19e-08
[PASS] paper_literal + paper friction: m=265, max|r|=7.11e-15, max|J|=1.19e-08
[PASS] contact_normal + derived friction: m=265, max|r|=7.11e-15, max|J|=1.19e-08
[PASS] friction disabled: m=229, max|r|=7.11e-15, max|J|=3.11e-09
[PASS] all_unordered GIAC pairs: m=277, max|r|=7.11e-15, max|J|=1.19e-08
[PASS] convexity p20 literal: m=265, max|r|=7.11e-15, max|J|=1.19e-08
STEP 2: PASS
```

C++ 残差+Jacobian 耗时（265×105，含 CppAD `ADFun` 构造）：

```text
11.458 ms/call
```

### Step 3 — 已完成并通过验证

- V-B1 落脚点贴地
- V-B2 脚间碰撞 barrier
- V-B3 名义腿长（midstance）
- V-B4 基座姿态/高度对齐
- V-B5 地形边缘避让（h / h_s1 梯度）
- V-B6 上一轮解
- V-B7 速度跟踪
- V-B8 平滑性（L_dot）
- 式 (29) slack 目标

验证脚本：`test_step3.py`：

```text
P=3, K=5, n=105, records=120
max |record residual| = 6.661e-16
max |record weight|   = 0.000e+00
|total_cpp - total_py| = 2.220e-16
max |grad_cpp - grad_py| = 2.314e-10
STEP 3: PASS
```

C++ records + total gradient 耗时（120 条记录，含 CppAD `ADFun` 构造）：

```text
16.231 ms/call
```

### Step 4 — 已完成并通过验证

- 统一求值：`f / g / h / roots` 与 `gf / Jg / Jh / Jr / H`
- C++ GN-SQP 主循环
- QuadProg++（现有 C++ shim）解 QP
- 变量缩放、LM 阻尼重试、Armijo 线搜索、merit function
- 与 Python `solve_gn_sqp` 对比

验证脚本：`test_step4.py`：

```text
n=105, mg=223, mh=42
|f_cpp - f_py| = 2.220e-16
max |g| = 7.105e-15
max |h| = 0.000e+00
max |gf| = 1.110e-16
max |Jg| = 2.842e-14
max |Jh| = 0.000e+00
max |H|  = 1.776e-15
py status=GN-SQP: converged, it=13, f=-0.031408028, 499.8 ms
cpp status=GN-SQP: converged, it=13, f=-0.031408028
max |x_cpp - x_py| = 5.052e-08
STEP 4: PASS
```

### Step 5 — 已完成接入

- `cpp_solver.py` 用 ctypes 打包 `IntegratedNLP` 参数/地图
- `NLPFootholdPlanner` 新增 `nlp_solver_backend="cpp"`（默认）
- C++ 解 → `unpack_x` → 现有 `export_trajectory` / `TAMOLSPlan`
- 真实 DTC 状态下逐步对拍 Python：

```text
[NLP] compare cpp=GN-SQP: max_iter py=GN-SQP: max_iter
      max|x_cpp-x_py|=1.979e-07
      f_cpp=9.43673 f_py=9.43673
```

- 首次 reset 的 NLP 问题可能不可行（max_viol ~50），自动回退 grid 做初始化；
  之后的 replan 使用 C++ NLP。

## 构建与验证

```bash
bash task/DTC/tamols/cpp/nlp/build_core.sh
uv run python task/DTC/tamols/cpp/nlp/test_step1.py
uv run python task/DTC/tamols/cpp/nlp/test_step2.py
uv run python task/DTC/tamols/cpp/nlp/test_step3.py
uv run python task/DTC/tamols/cpp/nlp/test_step4.py
uv run python task/DTC/tamols/cpp/nlp/test_step5_direct.py
```

`build_core.sh` 会优先使用系统 CppAD；没有时自动下载 `libcppad-dev`
并解压到 `third_party/`（不修改系统目录）。

## 后续步骤（每步完成即验证）

### Step 6 — 异步求解完成

- `NLPFootholdPlanner` 内置单 worker 异步队列
- 主循环不等待 NLP；求解期间继续使用上一份 NLP 解或 grid
- 首次 reset 不投递 NLP，先让 grid 完成初始化
- 队列只保留最新任务；reset 时作废旧 generation
- **热启动**：C++ `x0` 优先使用上一轮 `spline_coeffs / desired_footholds /
  slack`，而不是每次从解析初值重新开始（论文 V-C/D）
- 新结果由主线程导出为 `TAMOLSPlan`，再写回命令项
- 实测真实 DTC 单环境：
  - 热启动后收敛案例约 **0.56~0.73 s / 35~43 iter**
  - 未热启动时同状态约 60 iter / 1.0 s
  - 主循环不阻塞
  - NLP 解持续生效（foothold 与最新 NLP 解差 < 1e-6）
  - 失败/未收敛时保留上一份 NLP 解，不覆盖为 grid

开关：

```bash
DTC_NLP_ASYNC=0      # 关闭异步（同步阻塞）
DTC_NLP_BACKEND=cpp  # cpp / python
DTC_NLP_COMPARE=1    # 调试：C++ vs Python 逐步对拍
```

### Step 7 — 论文三层结构接入

- Layer 1：`h_s2` 虚拟地面简化 TO（只保留初始/连接 + 目标，关闭
  GIAC/可达/凸性），worker 内 12 iter；
- Layer 2：真实地图 0.4 m / 2 cm batch search（遮挡、h_s1 梯度/正曲率、
  h_s2 垂直距离）；
- 截断：只解一次 Layer 1，再用 Layer 2 细化结果热启动真实 NLP；
- 异步 worker 内顺序执行 Layer 1 → Layer 2 → 真实 NLP，避免 CppAD
  tape 跨线程。

实测（2.4 m map, 1 env）：

```text
layer1: 12 iter, max_iter, viol ~1e-12, accepted
layer2: 4~15 ms (NumPy)
real NLP: 21~60 iter, 0.5~1.3 s
async: 主循环不阻塞，失败保留上一份 NLP 解
```

### Step 8 — CppAD atomic + 稀疏 Jacobian + ADFun 复用

- 地图双线性查询改为 CppAD `atomic_base`，支持任意 x 的 forward/reverse；
- 同一个 `ADFun` 在整个 SQP 迭代中复用，不再每轮重建；
- 使用 `ForSparseJac` + `SparseJacobian`，只算非零项；
- 数值结果与旧实现一致。

合成 `crawl_flat`：

```text
C++ GN-SQP: 447 ms -> 108 ms  (13 iter)
单次 f/g/h/Jacobian: 66 ms -> 22 ms
```

真实 DTC（异步后台）：

```text
20 iter: 约 115 ms
60 iter: 约 310~390 ms
Layer 2: 2~5 ms
主线程构建/导出: 仍约 0.47 s
```

### Step 9 — 主线程构建移入 worker + Layer 1 全约束 + GPU Layer 2

- 真实 `IntegratedNLP`/`nlp_v`/`CPPProblem` 均在 worker 构建；
- 主线程只打包地图/参数并提交，replan 从 ~0.47 s 降到 **7~23 ms**；
- Layer 1 打开完整 GIAC/可达，max_iter=20；
- Layer 2 改为 torch 批量（4 条腿一次计算）；
- Layer 1 成功时真实 NLP 常只需 **2~20 iter**。

实测：

```text
主线程 replan:       7~23 ms
后台完整求解:        0.7~1.1 s（含 Python/CasADi 构建）
真实 NLP:            2~20 iter（Layer 1 成功）
Layer 2:             4~8 ms
```

### Step 10 — 去掉 Python/CasADi 问题组装（direct builder）

- 新增 `make_problem_from_data` / `make_problem_direct`：
  直接从 `durations` / `schedules` / `TerrainMaps` / `NLPParameters`
  构造 `TamolsSolverProblem`，默认路径不再创建 `IntegratedNLP`；
- 真实 NLP 与 Layer 1 都在 worker 内用 direct builder 组装；
- 下游导出改用轻量 `PlanContext`（durations/schedules/layout），
  `export_trajectory` / `plan_from_report` 不再依赖 CasADi 对象；
- `IntegratedNLP` 仅保留给显式 `nlp_solver_backend="python"`。

对拍（Step 5）：

```text
sample 0~2: f/g/h/gf/Jg/Jh/H 误差全部 0.000e+00
solve: step4=GN-SQP it=5 f=-0.0314080134
       direct=GN-SQP it=5 f=-0.0314080134
       max|x| = 0.000e+00
STEP 5: PASS
```

真实 DTC 异步（2.4 m map, 1 env, 250 步）：

```text
主线程 replan:       6 / 6 / 7 ms    (min / median / max)
后台完整求解:        86 / 281 / 381 ms
  - Layer 1 20 iter（完整 GIAC/可达）
  - 真实 NLP: 1 / 45 / 60 iter
  - Layer 2: 约 2 ms
```

说明：

- 真实 NLP 首轮预算为 `nlp_real_first_iter=20`。初始化好时可以 3 次
  收敛；困难状态 45~60 次。之前把首轮压到 5 次虽然更快，但零速度
  站立时会接受未收敛的落脚点，造成 xy 抖动，已回退；
- Layer 1 只跑 5/10 iter 时约束违反可达 `1e0~1e3`，真实 NLP 会失败，
  因此保留 20 iter（数值正确优先）；
- Layer 2 候选点约 5.5k，GPU kernel 固定开销反而更慢；按
  `nlp_layer2_gpu_min_points` 自适应切回 CPU，约 2 ms。

### Step 11 — 每个触地事件求解一次

- 新增 `replan_on_any_touchdown`：任意一条腿触地即触发重规划；
- DTC 默认开启，等价于论文“at each leg touch-down”求解；
- trot 中两条对角腿同时触地，因此实际是每个对角对事件；
- 求解仍走异步 worker，队列只保留最新任务；单次求解跟不上时会丢弃
  中间事件，但不会阻塞主循环。

## 当前性能与后续优化

- 合成 `crawl_flat`：C++ GN-SQP 约 108 ms / 13 iter
- 真实 DTC 异步：后台约 0.09~0.38 s，主线程约 6~7 ms
- 主要剩余开销：Layer 1 20 iter、困难状态真实 NLP 60 iter
- 后续：CppADCodeGen 生成导数代码、Layer 1 更好的简化模型

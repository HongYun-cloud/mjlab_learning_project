# Student 多教师蒸馏实现说明

> 本文按“从 CLI 到梯度更新”的完整链路，说明 `task/student` 下多教师蒸馏的实现。
>
> 结论先行：
>
> 当前实现是**“按地形注册教师、每个 student 任务同一时间只加载一个教师”**的在线蒸馏框架，不是“一个环境里同时跑多个教师、按 env 路由”的真正多教师。
>
> 蒸馏算法是 rsl_rl 的 `Distillation`：student 自己跑环境，teacher 对 student 当前看到的观测给动作标签，student 用 MSE 去拟合 teacher 的确定性动作。
>
> 所谓“多教师”，目前只体现在 `teachers.py` 这个 `terrain_key -> checkpoint` 注册表，以及每个地形注册一个独立 student task。

---

## 1. 代码地图

| 文件 | 作用 |
|---|---|
| `task/student/config/__init__.py` | 注册 8 个 student task，绑定 `MultiTeacherDistillationRunner` |
| `task/student/config/rl_cfg.py` | student/teacher 模型配置、Distillation 算法配置、`obs_groups`、教师注册表字段 |
| `task/student/config/teachers.py` | `terrain_key -> teacher PPO checkpoint` 注册表 |
| `task/student/config/multi_teacher.py` | 自定义 runner：选教师、加载教师、save/load 适配 |
| `task/student/config/env_cfg.py` | 8 个地形任务的环境配置，设置地形、奖励、终止、课程、`terrain_type` |
| `task/student/tracking_env_cfg.py` | 基础环境：actor/critic 观测组、动作、命令、奖励、终止等 |
| `task/teacher/tracking_env_cfg.py` | 教师基础环境：`actor_terms = critic_terms`，教师 actor 就是特权观测 |
| `task/teacher/config/rl_cfg.py` | 教师 PPO 配置：512-256-128、Gaussian、obs normalization |
| `train/train_student.sh` | 学生训练入口，`-e` 选地形，`-t` 指定教师 checkpoint |
| `play/play_student.sh` | 学生回放入口，`-c` 指定 student checkpoint |
| `.venv/.../rsl_rl/algorithms/distillation.py` | 蒸馏算法本体 |
| `.venv/.../rsl_rl/runners/distillation_runner.py` | 蒸馏 runner，负责检查 teacher 是否加载 |
| `.venv/.../rsl_rl/models/mlp_model.py` | student/teacher 共用的 MLP 模型 |

注意：`task/student/config/runner.py` 是旧的参考副本，没有被 import；`task/student/config/symmetry.py` 也没有被 student 链路引用。真正生效的是 `multi_teacher.py`。

---

## 2. 任务注册链路：从 `uv run train` 到 8 个 student task

### 2.1 entry point 自动加载

`pyproject.toml`：

```toml
[project.entry-points."mjlab.tasks"]
myrobot_mjlab = "myrobot_mjlab"
```

`mjlab/__init__.py` 在 import 时会执行：

```python
_import_registered_packages()
  -> entry_points().select(group="mjlab.tasks")
  -> entry_point.load()  # 加载 myrobot_mjlab
```

`myrobot_mjlab/__init__.py` 再依次 import：

```python
from myrobot_mjlab.task.velocity import config
from myrobot_mjlab.task.track import config
from myrobot_mjlab.task.teacher import config as teacher_config
from myrobot_mjlab.task.student import config as student_config
...
```

所以 `import mjlab.tasks` 时，`task/student/config/__init__.py` 就会被执行，任务注册完成。

### 2.2 `task/student/config/__init__.py`

核心是 `_TERRAINS` 和循环：

```python
_TERRAINS = {
    "Flat": my_go2_flat_env_cfg,
    "StairUp": my_go2_stair_up_env_cfg,
    ...
}

for _name, _env_cfg in _TERRAINS.items():
    register_mjlab_task(
        task_id=f"Mjlab-Tracking-{_name}-Student-Unitree-Go2",
        env_cfg=_env_cfg(),
        play_env_cfg=_env_cfg(play=True),
        rl_cfg=my_go2_multi_teacher_distill_cfg(
            terrain_key=_name,
            teachers=TEACHER_CHECKPOINTS,
        ),
        runner_cls=MultiTeacherDistillationRunner,
    )
```

这里发生了三件事：

1. **每个地形一个 task_id**  
   例如 `Mjlab-Tracking-Flat-Student-Unitree-Go2`。

2. **每个 task 都绑定同一个 runner 类**  
   `runner_cls=MultiTeacherDistillationRunner`，`train.py` 会通过 `load_runner_cls(task_id)` 拿到它。

3. **每个 task 的 `terrain_key` 不同**  
   `my_go2_multi_teacher_distill_cfg(terrain_key=_name)` 把这个 key 写进 RL 配置，后面 runner 靠它从注册表里找教师。

注册表本身是 `task/student/config/teachers.py`：

```python
TEACHER_CHECKPOINTS: dict[str, str] = {
    "Flat": "",
    "StairUp": "",
    ...
}
```

现在全是空字符串，所以默认 `teacher_checkpoint=None`。训练时必须通过 `-t` 或 `--agent.teacher-checkpoint` 覆盖，否则后面会报错。

---

## 3. 配置链路：`MultiTeacherDistillationRunnerCfg`

### 3.1 配置类

`task/student/config/rl_cfg.py` 定义了两个 dataclass。

**算法配置** `MultiTeacherDistillationAlgorithmCfg`：

```python
class_name = "Distillation"
num_learning_epochs = 1
gradient_length = 15
learning_rate = 1e-3
max_grad_norm = 1.0
loss_type = "mse"
optimizer = "adam"
```

这些字段最终会原样传给 `rsl_rl.algorithms.Distillation`。

**Runner 配置** `MultiTeacherDistillationRunnerCfg` 继承 `RslRlBaseRunnerCfg`，额外定义：

```python
class_name = "MultiTeacherDistillationRunner"

student: RslRlModelCfg = 512-256-128, elu, obs_normalization=True,
    GaussianDistribution(init_std=1.0, std_type="scalar")

teacher: RslRlModelCfg = 同样 512-256-128, elu, obs_normalization=True,
    GaussianDistribution(...)

obs_groups = {
    "student": ("actor",),
    "teacher": ("critic",),
}

teachers: dict[str, str] = TEACHER_CHECKPOINTS
terrain_key: str = ""
teacher_checkpoint: str | None = None
```

最关键的字段是 `obs_groups`：

- `student` 模型读环境里的 `actor` 组 → 可部署观测。
- `teacher` 模型读环境里的 `critic` 组 → 特权观测。

这就是“特权蒸馏”的核心：教师看学生看不到的信息，学生只学可部署输入到教师动作的映射。

### 3.2 工厂函数

```python
def my_go2_multi_teacher_distill_cfg(terrain_key, teachers=None, teacher_checkpoint=None):
    cfg = MultiTeacherDistillationRunnerCfg()
    cfg.terrain_key = terrain_key
    cfg.teachers = dict(teachers or TEACHER_CHECKPOINTS)
    if teacher_checkpoint is None:
        teacher_checkpoint = cfg.teachers.get(terrain_key) or None
    cfg.teacher_checkpoint = teacher_checkpoint
    return cfg
```

逻辑：

1. 新建配置；
2. 写入当前地形的 `terrain_key`；
3. 复制教师注册表；
4. 如果注册表里当前地形有路径，就写入 `teacher_checkpoint`；空字符串会变成 `None`。

### 3.3 CLI 怎么传进来

`train.py` 里：

```python
agent_cfg = asdict(cfg.agent)
...
runner = runner_cls(env, agent_cfg, str(log_dir), device)
```

`asdict` 会把 dataclass 转成嵌套 dict，包括：

```python
{
  "terrain_key": "Flat",
  "teachers": {"Flat": "...", ...},
  "teacher_checkpoint": "...",
  "student": {...},
  "teacher": {...},
  "algorithm": {...},
  "obs_groups": {"student": ("actor",), "teacher": ("critic",)},
  ...
}
```

`tyro` 会把 `teacher_checkpoint` 映射成命令行参数：

```bash
--agent.teacher-checkpoint /path/to/teacher.pt
```

`train_student.sh` 就是把这个参数拼进去：

```bash
TEACHER_ARGS=(--agent.teacher-checkpoint "$TEACHER_CHECKPOINT")
```

所以教师来源的优先级是：

```text
--agent.teacher-checkpoint
    > teachers.py 里 terrain_key 对应的路径
    > None（打印警告，训练时抛错）
```

---

## 4. 环境与观测链路

### 4.1 基础环境 `make_velocity_env_cfg`

`task/student/tracking_env_cfg.py` 里最重要的是两个观测字典：

```python
actor_terms = {
    "base_ang_vel",
    "direction_goal",
    "head_error",
    "target_distance",
    "time_left",
    "terrain_type",        # 9 维 one-hot
    "height_command",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "height_scan",         # 1.6 x 1.0, res 0.1 -> 187 维
}

critic_terms = {
    **actor_terms,
    "base_lin_vel",        # 3
    "base_height",         # 1
    "direction_real",      # 2
    "Friction_coefficient",# 1
    "joint_pos",           # 覆盖成无噪声版本，不增加维度
    "foot_air_time",       # 4
    "foot_contact",        # 4
    "foot_contact_forces", # 12
}
```

当前 9 类地形 one-hot 下：

```text
student actor  = 244 维
teacher critic = 271 维
动作维度       = 12
```

老版本 5 类地形 one-hot 是：

```text
actor  = 240
critic = 267
```

所以旧 checkpoint 和新环境不兼容。

### 4.2 观测组

```python
observations = {
    "actor": ObservationGroupCfg(
        terms=actor_terms,
        concatenate_terms=True,
        enable_corruption=True,   # 训练时加噪声
    ),
    "critic": ObservationGroupCfg(
        terms=critic_terms,
        concatenate_terms=True,
        enable_corruption=False,  # 特权观测不加噪声
    ),
    "rnd_state": ObservationGroupCfg(...),  # 蒸馏不用
}
```

- `actor` 组是学生可部署观测，训练时开启噪声。
- `critic` 组是教师特权观测，关闭噪声。
- `rnd_state` 是给 PPO 的 RND 扩展用的；`Distillation` 明确不支持 RND，所以这个组在蒸馏里是多余的，但环境仍然会计算它，`RolloutStorage` 也会给它分配空间。

### 4.3 地形 one-hot

`actor_terms` 和 `critic_terms` 里都有：

```python
"terrain_type": ObservationTermCfg(
    func=terrain_type,
    params={"terrain_type": None},
)
```

每个任务在 `env_cfg.py` 里通过 `_set_terrain_type` 显式覆盖：

```python
def _set_terrain_type(cfg, value):
    for group_name in ("actor", "critic"):
        group = cfg.observations.get(group_name)
        if group is not None and "terrain_type" in group.terms:
            group.terms["terrain_type"].params["terrain_type"] = value
```

对应关系：

| task 后缀 | `terrain_type` 字符串 | one-hot 下标 |
|---|---|---|
| `Flat` | `flat` | 0 |
| `StairUp` | `stair_up` | 1 |
| `StairDown` | `stair_down` | 2 |
| `TireArray` | `tire_array` | 3 |
| `Platform` | `platform` | 4 |
| `PlumBlossom` | `plum` | 6 |
| `Stones2Rows` | `stone_2rows` | 7 |
| `LowClearanceBar` | `low_clearance_bar` | 8 |

教师 checkpoint 必须和当前任务的这个 one-hot 一致。用 Flat 的教师去蒸馏 StairUp，教师看到的 `terrain_type` 是 `stair_up`，但它的网络是在 `flat` 上训练的，标签质量会下降。

### 4.4 教师环境为什么能对上

`task/teacher/tracking_env_cfg.py` 是 student 版本的副本，关键一行：

```python
# Teacher actor uses the privileged observation set.
actor_terms = critic_terms
```

也就是说：

```text
teacher 环境的 actor 组 = 特权 critic 组
teacher PPO 的 actor 输入 = 271 维
student 环境的 critic 组 = 同样的特权组
student 的 teacher 模型输入 = 271 维
```

两个文件 diff 下来，除了这行注释和 `speed_limit_penalty` 的 `max_speed`（1.0 vs 1.5），观测项和顺序完全一样。所以教师 checkpoint 的 `actor_state_dict` 可以 strict 加载到 student 的 `teacher` 模型里。

### 4.5 一个需要确认的噪声不一致

教师环境里：

```python
observations = {
    "actor": ObservationGroupCfg(
        terms=actor_terms,       # 特权项
        enable_corruption=True,  # 教师训练时也加噪声
    ),
    "critic": ObservationGroupCfg(
        terms=critic_terms,
        enable_corruption=False,
    ),
}
```

学生环境里教师模型读的是 `critic` 组，`enable_corruption=False`。

于是：

```text
教师 PPO 训练时：特权观测 + 噪声
学生蒸馏时：    特权观测（干净）
```

`base_ang_vel`、`base_lin_vel`、`height_scan`、`joint_vel` 等项本身带 `noise` 配置，教师训练时是加噪的；蒸馏时教师模型看到的是干净输入。严格说是 train/test 输入分布不一致。可能影响不大，但如果教师标签质量不稳定，这是第一个要检查的点。更干净的做法是把教师 actor 组的 `enable_corruption` 设为 `False`，或者单独定义一个 `teacher` 观测组并明确噪声策略。

---

## 5. Runner 构造链路：逐函数拆解

### 5.1 `MultiTeacherDistillationRunner.__init__`

`task/student/config/multi_teacher.py:18-55`

```python
def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
    self._terrain_key = str(train_cfg.get("terrain_key", ""))
    teachers = train_cfg.get("teachers") or {}
    teacher_checkpoint = train_cfg.get("teacher_checkpoint")
    if not teacher_checkpoint:
        teacher_checkpoint = teachers.get(self._terrain_key)
    self._teacher_checkpoint = teacher_checkpoint
```

第一步：选教师。

```text
train_cfg["teacher_checkpoint"]  # CLI 覆盖
    or teachers[terrain_key]      # teachers.py
    or None
```

第二步：去掉 MLPModel 不接受的字段。

```python
for key in ("student", "teacher"):
    if key in train_cfg:
        for opt in ("cnn_cfg", "distribution_cfg"):
            if train_cfg[key].get(opt) is None:
                train_cfg[key].pop(opt, None)
        if train_cfg[key].get("rnn_type") is None:
            for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
                train_cfg[key].pop(opt, None)
```

原因：

- `MLPModel.__init__` 的签名只有 `hidden_dims / activation / obs_normalization / distribution_cfg`；
- `cnn_cfg`、`rnn_type`、`rnn_hidden_dim`、`rnn_num_layers` 是 mjlab 给 `CNNModel` / `RNNModel` 用的字段，直接传给 `MLPModel` 会 `TypeError`。

注意 `distribution_cfg` 这里不是 None（配置里是 Gaussian），所以不会被删。真正把 `class_name` 从 `distribution_cfg` 里 pop 掉的是后面的 `MLPModel` 自己。

第三步：

```python
super().__init__(env, train_cfg, log_dir, device)
```

`super()` 是 `DistillationRunner`，它没有定义 `__init__`，所以实际进入 `OnPolicyRunner.__init__`。

第四步：加载教师。

```python
if self._teacher_checkpoint:
    print(...)
    self.load(
        self._teacher_checkpoint,
        load_cfg={"teacher": True},
        strict=True,
        map_location=device,
    )
else:
    print("[MultiTeacher] no teacher checkpoint configured ...")
```

这里只是打印警告，不抛异常；真正抛异常在 `DistillationRunner.learn` 里。

### 5.2 `OnPolicyRunner.__init__`

`rsl_rl/runners/on_policy_runner.py`

```python
def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
    self.env = env
    self.cfg = train_cfg
    self.device = device

    self._configure_multi_gpu()

    obs = self.env.get_observations()

    alg_class = resolve_callable(self.cfg["algorithm"]["class_name"])
    self.alg = alg_class.construct_algorithm(obs, self.env, self.cfg, self.device)

    self.logger = Logger(...)
    self.current_learning_iteration = 0
```

逐点解释：

1. `_configure_multi_gpu()`  
   单卡时给 `cfg["multi_gpu"] = None`；多卡时初始化 torch.distributed 并填 `global_rank/world_size`。

2. `env.get_observations()`  
   `RslRlVecEnvWrapper.get_observations()` 调用 `observation_manager.compute()`，返回一个 `TensorDict`，key 是观测组名：

   ```text
   actor, critic, rnd_state
   ```

3. `resolve_callable("Distillation")`  
   rsl_rl 的简单名解析：遍历 `rsl_rl` 顶层子模块，找到 `rsl_rl.algorithms` 里导出的 `Distillation` 类。

4. `alg_class.construct_algorithm(...)`  
   进入蒸馏算法的构造。

### 5.3 `Distillation.construct_algorithm`

`rsl_rl/algorithms/distillation.py`

```python
alg_class = resolve_callable(cfg["algorithm"].pop("class_name"))
student_class = resolve_callable(cfg["student"].pop("class_name"))
teacher_class = resolve_callable(cfg["teacher"].pop("class_name"))

default_sets = ["student", "teacher"]
cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

if cfg["algorithm"].get("rnd_cfg") is not None:
    raise ValueError(...)
cfg["algorithm"]["rnd_cfg"] = None

if cfg["algorithm"].get("symmetry_cfg") is not None:
    raise ValueError(...)
cfg["algorithm"]["symmetry_cfg"] = None

student = student_class(obs, cfg["obs_groups"], "student", env.num_actions, **cfg["student"])
teacher = teacher_class(obs, cfg["obs_groups"], "teacher", env.num_actions, **cfg["teacher"])

storage = RolloutStorage("distillation", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

alg = alg_class(
    student, teacher, storage,
    device=device,
    **cfg["algorithm"],
    multi_gpu_cfg=cfg["multi_gpu"],
)

alg.compile(cfg.get("torch_compile_mode"))
```

关键函数：

**`resolve_obs_groups`**

作用：

- 检查 `obs_groups` 里每个 set 对应的环境 group 是否真的存在；
- 检查有没有空列表；
- 如果缺少 `student` / `teacher` set，会尝试用同名环境组或 `policy` 组补；
- 返回解析后的映射。

这里传进去的是：

```python
{"student": ("actor",), "teacher": ("critic",)}
```

环境里确实有 `actor` 和 `critic`，所以解析结果不变。

**`MLPModel.__init__`**

```python
self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
```

- `student` 模型：`obs_groups["student"] = ["actor"]`，所以 `obs_dim = actor_dim = 244`。
- `teacher` 模型：`obs_groups["teacher"] = ["critic"]`，所以 `obs_dim = critic_dim = 271`。

然后：

```python
if obs_normalization:
    self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
```

学生和教师都开了 `obs_normalization=True`。教师加载 checkpoint 时会一起加载 `obs_normalizer._mean/_var/_std/count`，所以教师输入的归一化统计和它训练时一致。

然后：

```python
if distribution_cfg is not None:
    dist_class = resolve_callable(distribution_cfg.pop("class_name"))
    self.distribution = dist_class(output_dim, **distribution_cfg)
```

`GaussianDistribution` 的 `output_dim=12`，创建可学习的 `std_param`。  
教师配置里保留 Gaussian，主要是为了 strict 加载 PPO checkpoint 里的 `distribution.std_param` 键；教师推理时不会用它采样。

最后：

```python
self.mlp = MLP(self._get_latent_dim(), mlp_output_dim, hidden_dims, activation)
```

`512 -> 256 -> 128 -> 12`。

**`MLPModel.forward`**

```python
latent = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
latent = self.obs_normalizer(latent)
mlp_output = self.mlp(latent)

if self.distribution is not None:
    if stochastic_output:
        self.distribution.update(mlp_output)
        return self.distribution.sample()
    return self.distribution.deterministic_output(mlp_output)
```

- `stochastic_output=True`：更新 Gaussian 并采样，给学生 rollout 用。
- 默认 `stochastic_output=False`：`deterministic_output` 直接返回 MLP 输出的均值，给教师标签用。

**`RolloutStorage.__init__`**

`training_type="distillation"` 时会分配：

```text
observations      [T, N, ...]  # 所有观测组
actions           [T, N, 12]
rewards           [T, N, 1]    # 蒸馏不用，但仍记录
dones             [T, N, 1]
privileged_actions[T, N, 12]   # 教师标签
```

`T = num_steps_per_env = 24`，`N = num_envs`（脚本里 512）。

**`Distillation.__init__`**

- `self.student = student.to(device)`
- `self.teacher = teacher.to(device)`
- `_raw_student = student`、`_raw_teacher = teacher`
- `optimizer = Adam(self.student.parameters(), lr=1e-3)`  
  只优化学生，教师是冻结的。
- `storage`、`transition`、`last_hidden_states`
- `loss_fn = mse_loss` 或 `huber_loss`
- `num_updates = 0`

### 5.4 教师 checkpoint 加载链路

`__init__` 最后调用：

```python
self.load(self._teacher_checkpoint, load_cfg={"teacher": True}, strict=True, map_location=device)
```

调用顺序：

```text
MultiTeacherDistillationRunner.load
  -> OnPolicyRunner.load
       -> torch.load(path, weights_only=False, map_location=device)
       -> Distillation.load(loaded_dict, {"teacher": True}, strict=True)
            -> self._raw_teacher.load_state_dict(
                   loaded_dict.get("teacher_state_dict")
                   or loaded_dict["actor_state_dict"],
                   strict=True,
               )
            -> self.teacher_loaded = True
            -> return False   # 不加载 iteration
  -> 恢复 infos["env_state"]["common_step_counter"]
```

教师 checkpoint 是普通 PPO checkpoint：

```text
actor_state_dict
critic_state_dict
optimizer_state_dict
iter
infos.env_state.common_step_counter
```

`Distillation.load` 只取 `actor_state_dict`，因为教师模型对应的是 PPO 的 actor。  
`strict=True` 要求键名和形状完全一致：

```text
obs_normalizer._mean/_var/_std/count
mlp.0.weight/bias
mlp.2.weight/bias
mlp.4.weight/bias
mlp.6.weight/bias
distribution.std_param
```

当前 `logs/rsl_rl/teacher_model/flat/obs_critic_teacher_2100.pt` 的 actor 输入是 271，和当前 student critic 271 完全匹配，strict 加载可以成功。  
而 `obs_actor_LALRT_3649.pt` 是旧 5 类 one-hot 的 240/267，strict 加载会因形状不匹配失败。

### 5.5 `MultiTeacherDistillationRunner.save`

`task/student/config/multi_teacher.py:57-62`

```python
def save(self, path, infos=None):
    env_state = {
        "common_step_counter": self.env.unwrapped.common_step_counter,
    }
    infos = {**(infos or {}), "env_state": env_state}
    super().save(path, infos)
```

`super()` 是 `OnPolicyRunner.save`：

```python
saved_dict = self.alg.save()  # student + teacher + optimizer
saved_dict["iter"] = self.current_learning_iteration
saved_dict["infos"] = infos
torch.save(saved_dict, path)
self.logger.save_model(path, ...)
```

所以学生 checkpoint 里实际有：

```text
student_state_dict
teacher_state_dict      # 教师权重被一起存进去了
optimizer_state_dict
iter
infos.env_state.common_step_counter
```

这也是后面 resume 时教师会被覆盖的根源。

### 5.6 `MultiTeacherDistillationRunner.load`

`task/student/config/multi_teacher.py:64-83`

```python
def load(self, path, load_cfg=None, strict=True, map_location=None):
    if load_cfg == {"actor": True}:
        load_cfg = {"student": True}

    infos = super().load(path, load_cfg, strict, map_location)

    if infos and "env_state" in infos:
        env_state = infos["env_state"]
        if "common_step_counter" in env_state:
            self.env.unwrapped.common_step_counter = env_state["common_step_counter"]

    return infos
```

两个作用：

1. **play 适配**  
   `play.py` 按 PPO 习惯调用 `load_cfg={"actor": True}`。蒸馏 checkpoint 里没有 `actor_state_dict`，策略在 `student_state_dict` 里，所以这里映射成 `{"student": True}`。

2. **恢复环境步数**  
   学生 checkpoint 的 `infos.env_state.common_step_counter` 被恢复到环境，保证课程/退火接着上次训练。

---

## 6. 训练链路：逐函数拆解

### 6.1 `DistillationRunner.learn`

```python
def learn(self, num_learning_iterations, init_at_random_ep_len=False):
    if not self.alg.teacher_loaded:
        raise ValueError("Teacher model parameters not loaded. ...")
    super().learn(num_learning_iterations, init_at_random_ep_len)
```

所以：

- 注册表为空且没有 `--agent.teacher-checkpoint`，且没有 resume 学生 checkpoint → 这里直接 `ValueError`。
- 只要 `teacher_loaded=True`（教师 checkpoint 或学生 checkpoint 里的 teacher 被加载），就能继续。

### 6.2 `OnPolicyRunner.learn`

主循环：

```python
obs = env.get_observations().to(device)
self.alg.train_mode()  # student.train(); teacher.eval()

for it in range(start_it, total_it):
    with torch.inference_mode():
        for _ in range(num_steps_per_env):   # 24
            actions = self.alg.act(obs)
            obs, rewards, dones, extras = self.env.step(actions.to(env.device))
            check_nan(...)
            self.alg.process_env_step(obs, rewards, dones, extras)
            self.logger.process_env_step(...)
        self.alg.compute_returns(obs)        # 蒸馏里是空操作

    loss_dict = self.alg.update()

    self.logger.log(..., loss_dict=loss_dict, action_std=...)
    if it % save_interval == 0:
        self.save(...)
```

### 6.3 `Distillation.act`

```python
self.transition.actions = self.student(obs, stochastic_output=True).detach()
self.transition.privileged_actions = self.teacher(obs).detach()
self.transition.observations = obs
return self.transition.actions
```

这一步是蒸馏的核心：

```text
student(actor 组, 采样)  -> 真正执行的动作
teacher(critic 组, 确定性) -> 标签动作
transition.observations = obs_t
```

学生动作是采样的，所以 rollout 有探索；教师动作是均值，确定性标签。

### 6.4 `Distillation.process_env_step`

```python
self.student.update_normalization(obs)
self.transition.rewards = rewards
self.transition.dones = dones
self.storage.add_transition(self.transition)
self.transition.clear()
self.student.reset(dones)
self.teacher.reset(dones)
```

注意：这里传入的 `obs` 是 `env.step()` 返回的 **obs_{t+1}**，而 `transition.observations` 是 `act()` 时存下的 **obs_t**。所以：

- 学生 normalizer 用 obs_{t+1} 更新；
- loss 用 obs_t 和 teacher 在 obs_t 上的标签。

两者错开一步，这是 rsl_rl 5.4.2 这个版本的实现细节，影响通常很小。

`storage.add_transition` 会把：

```text
observations[step]      = obs_t
actions[step]           = student action
rewards[step]           = reward
dones[step]             = done
privileged_actions[step] = teacher action
```

存进 buffer。

### 6.5 `Distillation.update`

```python
mean_behavior_loss = 0
loss = 0
cnt = 0

for epoch in range(num_learning_epochs):          # 1
    self.student.reset(hidden_state=...)
    self.teacher.reset(hidden_state=...)
    self.student.detach_hidden_state()

    for batch in self.storage.generator():         # 24 个时间步
        actions = self.student(batch.observations) # 确定性均值
        behavior_loss = self.loss_fn(actions, batch.privileged_actions)

        loss = loss + behavior_loss
        mean_behavior_loss += behavior_loss.item()
        cnt += 1

        if cnt % self.gradient_length == 0:        # 每 15 步
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            if self.max_grad_norm:
                nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.student.detach_hidden_state()
            loss = 0

        self.student.reset(batch.dones.view(-1))
        self.teacher.reset(batch.dones.view(-1))
        self.student.detach_hidden_state(batch.dones.view(-1))

mean_behavior_loss /= cnt
self.storage.clear()
self.last_hidden_states = (...)
```

`storage.generator()` 每次 yield 一个时间步的所有 env：

```python
observations      [N, ...]
privileged_actions[N, 12]
dones             [N, 1]
```

所以一个“batch”不是随机 mini-batch，而是“同一时刻的所有并行环境”。  
`gradient_length=15` 表示累积 15 个时间步的 loss 再做一次 backward/step。

**这里有一个明显的配置问题：**

```text
num_learning_epochs = 1
num_steps_per_env = 24
gradient_length = 15
24 % 15 = 9
```

所以每个 iteration 只有前 15 个时间步会真正 backward；后 9 个时间步的 loss 被累加进 `mean_behavior_loss`，但永远不会 `backward()`。  
也就是 24 个时间步里有 9 个（37.5%）采了但没用于更新。当前 venv 的 `Distillation` 没有打印 warning，但行为就是这样。

建议把 `gradient_length` 改成 `24`、`12`、`8`、`6` 等 24 的因数，或者把 `num_steps_per_env` 改成 30/45 这类 15 的倍数。

### 6.6 学生动作的 std

`GaussianDistribution` 的 `std_param` 是可学习参数，但：

- rollout 时 `student(obs, stochastic_output=True)` 会采样；
- update 时 `student(batch.observations)` 是确定性均值；
- loss 只对均值做 MSE，不涉及 `std_param`。

所以 `std_param` 实际上没有梯度，会一直保持 `init_std=1.0`。  
也就是说学生 rollout 的动作噪声是 `N(0, 1.0)`（在归一化动作空间里），再乘 action scale 0.5。  
如果发现学生训练初期动作抖动很大，这个 1.0 是第一个要检查的地方；可以调小 `init_std`，或设 `learn_std=False` 明确固定。

### 6.7 `Distillation.save`

```python
saved_dict = {
    "student_state_dict": self._raw_student.state_dict(),
    "teacher_state_dict": self._raw_teacher.state_dict(),
    "optimizer_state_dict": self.optimizer.state_dict(),
}
```

学生 checkpoint 里带着教师权重，所以文件比普通 PPO checkpoint 大，也让 resume 可以完整恢复蒸馏状态。但这也导致下面 7.2 的覆盖问题。

### 6.8 `Logger.log`

记录：

```text
Loss/behavior
Policy/mean_std
Reward/...
Episode/...
```

`action_std` 来自 `self.alg.get_policy().output_std`，也就是 student 的 Gaussian std（1.0）。

---

## 7. Resume / Play 链路

### 7.1 学生 resume

`train.py`：

```python
if cfg.agent.resume:
    resume_path = get_checkpoint_path(log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint)
...
runner = runner_cls(env, agent_cfg, str(log_dir), device)
...
runner.load(str(resume_path))
```

`runner.load` 没有传 `load_cfg`，所以进入 `Distillation.load` 的自动判断：

```python
if load_cfg is None and any("actor_state_dict" in key for key in loaded_dict):
    load_cfg = {"teacher": True, "iteration": False}
elif load_cfg is None:
    load_cfg = {
        "student": True,
        "teacher": True,
        "optimizer": True,
        "iteration": True,
    }
```

学生 checkpoint 顶层是：

```text
student_state_dict
teacher_state_dict
optimizer_state_dict
iter
infos
```

没有 `actor_state_dict`，所以走第二个分支，全部加载：

- student 权重恢复；
- teacher 权重恢复；
- optimizer 恢复；
- `current_learning_iteration = loaded_dict["iter"]`；
- `MultiTeacher.load` 再恢复 `common_step_counter`。

### 7.2 一个关键问题：resume 会覆盖 `-t` 指定的教师

时序是这样的：

```text
MultiTeacherDistillationRunner.__init__
    -> self.load(teacher_ckpt, {"teacher": True})   # 先加载 -t 教师

train.py
    -> runner.load(student_ckpt)                    # load_cfg=None
        -> Distillation.load 自动选择
           {"student": True, "teacher": True, "optimizer": True, "iteration": True}
        -> teacher_state_dict 从学生 checkpoint 里加载
        -> 覆盖掉刚才 -t 加载的教师
```

所以：

```bash
./train/train_student.sh flat_student2 -e stair_up -t stair_up_teacher.pt -p flat_student_run
```

本意是“用 StairUp 教师继续训练 Flat 学生”，但实际教师会变成 Flat 学生 checkpoint 里保存的旧教师。  
如果目标是“共享学生、换地形教师”，当前 resume 链路是错的。

可行的修法：

- 在 `MultiTeacherDistillationRunner.load` 里判断：如果当前 checkpoint 是学生 checkpoint 且 `self._teacher_checkpoint` 已配置，则用  
  `load_cfg={"student": True, "optimizer": True, "iteration": True}`，不要加载 checkpoint 里的 teacher；
- 或者先 resume 学生，再显式加载一次教师；
- 或者学生 checkpoint 不保存 `teacher_state_dict`。

### 7.3 Play 链路

`play.py`：

```python
runner = runner_cls(env, asdict(agent_cfg), device=device)
runner.load(str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)
```

`MultiTeacher.load` 把 `{"actor": True}` 映射成 `{"student": True}`，所以：

```text
student_state_dict -> student 模型
```

`get_inference_policy`：

```python
self.alg.eval_mode()
return self.alg.get_policy().to(device)
```

`Distillation.get_policy()` 返回 `_raw_student`。  
`MLPModel.forward` 默认 `stochastic_output=False`，所以 play 时学生输出的是均值动作，不带噪声。

`CheckpointManager` 的热切换也是同一个 `load` 路径，所以可以在 viewer 里换 student checkpoint。

---

## 8. 实际训练/回放命令

```bash
# 1) 先训练教师（特权观测）
./train/train_teacher.sh flat_teacher -e flat

# 2) 训练学生，指定教师 checkpoint
./train/train_student.sh flat_student -e flat \
    -t logs/rsl_rl/go2_velocity/<teacher_run>/model_2100.pt

# 3) 继续训练学生
./train/train_student.sh flat_student2 -e flat \
    -p <student_run> -m 3000 \
    -t logs/rsl_rl/go2_velocity/<teacher_run>/model_2100.pt

# 4) 回放学生
./play/play_student.sh -e flat \
    -c logs/rsl_rl/go2_velocity/<student_run>/model_5000.pt
```

如果不想每次传 `-t`，就填 `task/student/config/teachers.py`：

```python
TEACHER_CHECKPOINTS = {
    "Flat": "logs/rsl_rl/go2_velocity/xxx/model_2100.pt",
    ...
}
```

---

## 9. 这套实现的设计边界与风险

### 9.1 不是“同时多教师”

当前是：

```text
一个 student task -> 一个 terrain_key -> 一个教师 checkpoint
```

不是：

```text
一个环境里多个地形 -> 多个教师同时在线 -> 按 env id 路由
```

`multi_teacher.py` 的 docstring 也写了：

> current per-terrain tasks only one teacher is active at a time; when the terrains are merged into one environment, this runner can be extended to route per-env teacher ids to simultaneous teacher models.

如果以后要合并地形，需要改 `Distillation.act`：

1. 维护 `teacher_id_per_env`（可以从 `env.scene.terrain.terrain_types` 或自定义 buffer 来）；
2. 为每个教师建一个 `MLPModel`；
3. 对每个教师选出对应 env 的子集，算教师动作，再 scatter 回 `privileged_actions`；
4. `update` 部分不用改，因为标签已经是 per-env 的。

### 9.2 `gradient_length=15` 与 `num_steps_per_env=24` 不整除

这是当前配置里最直接的效率问题。  
24 个时间步只用 15 个做梯度，剩下 9 个白采。建议改成 `gradient_length=24` 或 `12`。

### 9.3 教师 checkpoint 的 `common_step_counter` 会污染学生环境

`MultiTeacher.load` 无条件恢复 `infos.env_state.common_step_counter`。  
教师 PPO checkpoint 也带这个字段（`obs_critic_teacher_2100.pt` 里是 50424）。  
于是**全新学生训练**一开始就把环境步数设成教师的步数。

影响：

- `anneal_reward_weight` 用 `common_step_counter - _anneal_start_step` 算进度；`_anneal_start_step` 默认 0，所以进度直接变成 1.0。Plum/Stones 的 `goal_forward` 退火会从最终权重开始，而不是从 1.5 开始。
- `terrain_levels_reach_target` 里 `if env.common_step_counter == 0` 的初始冻结会被绕过（不过初始 reset 在 runner 构造前已经发生，影响相对小）。

建议只在对学生 checkpoint 做 resume 时恢复步数；加载教师时不要恢复。

### 9.4 教师观测的噪声开关不一致

教师 PPO 训练时 actor 组 `enable_corruption=True`，蒸馏时教师模型读学生 `critic` 组 `enable_corruption=False`。  
严格说是输入分布不一致。建议确认这是否是有意为之；如果教师标签要求确定性，应该让教师 actor 组也关闭噪声，或单独定义教师观测组。

### 9.5 教师 checkpoint 必须严格匹配

`strict=True` 下：

- 观测维度必须一致（当前 271）；
- 键名必须一致（`distribution.std_param` 等）；
- 隐藏层维度必须一致（512-256-128）；
- 不能是旧 5 类 one-hot 的 240/267 checkpoint。

换地形时也要保证教师训练时的 `terrain_type` one-hot 和当前任务一致。

### 9.6 注册表为空

`teachers.py` 目前全是空字符串。  
不传 `-t` 且不 resume 学生 checkpoint 时：

```text
__init__ 打印警告
learn() 抛 ValueError: Teacher model parameters not loaded
```

### 9.7 `rnd_state` 组是多余的

学生环境仍然定义并计算 `rnd_state`，`RolloutStorage` 也为它分配空间，但 `Distillation` 不使用。  
如果追求效率，可以在 student 环境里去掉这个组，或者至少知道它有额外开销。

### 9.8 `save` 不遵守 `upload_model`，也没有 ONNX 导出

`MultiTeacherDistillationRunner` 继承的是 `OnPolicyRunner`，不是 `MjlabOnPolicyRunner` / `VelocityOnPolicyRunner`：

- `save()` 总是调用 `logger.save_model()`，不检查 `cfg["upload_model"]`；
- 不会像 `VelocityOnPolicyRunner` 那样自动导出 ONNX；
- 不会做 `MjlabOnPolicyRunner.load` 里的旧 checkpoint key 迁移。

如果这些行为需要和 teacher/track 任务一致，可以让它继承 `MjlabOnPolicyRunner`，或者在自己的 `save/load` 里补上。

---

## 10. 一句话总结每条链路

```text
注册链路:
  entry point -> myrobot_mjlab.__init__ -> task.student.config
  -> register_mjlab_task(8 个地形, runner_cls=MultiTeacherDistillationRunner)

配置链路:
  terrain_key -> teachers.py / --agent.teacher-checkpoint
  -> MultiTeacherDistillationRunnerCfg
  -> asdict -> runner

环境链路:
  actor(244) = 可部署观测
  critic(271) = 特权观测
  teacher 环境 actor = critic(271)

模型链路:
  student: MLP(244 -> 512 -> 256 -> 128 -> 12) + Gaussian
  teacher: MLP(271 -> 512 -> 256 -> 128 -> 12) + Gaussian(仅用于 strict load)

训练链路:
  student 采样跑环境
  teacher 对同一 obs 输出确定性动作
  MSE(student(obs_t), teacher(obs_t))
  每 15 个时间步 backward 一次

保存链路:
  student_state_dict + teacher_state_dict + optimizer + iter + env_state

回放链路:
  load_cfg={"actor": True} -> {"student": True}
  -> _raw_student -> 确定性均值动作
```

---

## 11. 建议的修改顺序

如果下一步要把“共享学生 + 多地形教师”真正跑通，建议按这个顺序改：

1. **先修 resume 覆盖教师**  
   在 `MultiTeacherDistillationRunner.load` 里区分“加载教师”和“恢复学生”两种调用：恢复学生时只加载 `student/optimizer/iteration`，保留 `__init__` 里已经加载的教师。

2. **修教师步数污染**  
   只有学生 checkpoint 的 `load` 才恢复 `common_step_counter`；教师 checkpoint 的 `infos.env_state` 不应写进学生环境，否则 Plum/Stones 的奖励退火会从最终阶段开始。

3. **调 `gradient_length`**  
   改成 24、12、8、6 中任意一个，或者把 `num_steps_per_env` 调成 30/45，保证每个 rollout 的样本都能参与一次 optimizer step。

4. **确认教师噪声策略**  
   教师 actor 组现在是 `enable_corruption=True`，蒸馏时教师读的 `critic` 组是 `False`。如果教师标签要求干净、确定，建议统一；否则至少知道这里有一步输入分布差异。

5. **最后再考虑真正的多教师路由**  
   当前 `teachers.py` + `terrain_key` 只是“按任务选一个教师”。如果要把所有地形合进一个环境，需要改 `Distillation.act`：维护每个 env 的 teacher id，按 id 分组算教师动作，再 scatter 回 `privileged_actions`；`update` 的 MSE 部分不用动。

整体上，这套 student 代码的“蒸馏主链路”是完整可用的：观测组拆分、教师 checkpoint strict 加载、在线 DAgger 式 MSE、学生/教师权重一起保存、play 时 actor→student 映射都通了。真正需要先处理的是 **resume 与教师加载的相互覆盖**，以及 **教师步数被带到学生环境** 这两个工程问题。

---

## 附：关键代码位置速查

| 内容 | 位置 |
|---|---|
| student 任务注册 | `task/student/config/__init__.py:19-40` |
| 蒸馏算法/runner 配置 | `task/student/config/rl_cfg.py:12-99` |
| 教师注册表 | `task/student/config/teachers.py:7-16` |
| 选教师 / 加载教师 / save / load | `task/student/config/multi_teacher.py:18-83` |
| actor/critic 观测项 | `task/student/tracking_env_cfg.py:100-234` |
| 地形覆盖函数 | `task/student/config/env_cfg.py:69-75` |
| teacher actor = critic | `task/teacher/tracking_env_cfg.py:187` |
| Distillation 算法 | `.venv/lib/python3.13/site-packages/rsl_rl/algorithms/distillation.py` |
| DistillationRunner | `.venv/lib/python3.13/site-packages/rsl_rl/runners/distillation_runner.py` |
| OnPolicyRunner | `.venv/lib/python3.13/site-packages/rsl_rl/runners/on_policy_runner.py` |
| MLPModel | `.venv/lib/python3.13/site-packages/rsl_rl/models/mlp_model.py` |
| RolloutStorage | `.venv/lib/python3.13/site-packages/rsl_rl/storage/rollout_storage.py` |
| student 训练脚本 | `train/train_student.sh` |
| student 回放脚本 | `play/play_student.sh` |

# === Step 1: Robot Entity 配置 ===
# 文件: src/robot_cfg.py

from pathlib import Path

import mujoco

from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.actuator import BuiltinPositionActuatorCfg


GO2_XML = Path(
    "myrobot/go2/go2.xml"
)


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(GO2_XML))


# =========================
# 初始状态
# =========================

ROBOT_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.278),

    joint_pos={
        ".*thigh_joint": 0.9,
        ".*calf_joint": -1.8,
        ".*R_hip_joint": 0.1,
        ".*L_hip_joint": -0.1,
    },

    joint_vel={
        ".*": 0.0,
    },
)




# =========================
# PD Actuator
# =========================



MY_GO2ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*thigh_joint",
        ".*calf_joint",
        ".*R_hip_joint",
        ".*L_hip_joint",
    ),

    stiffness=25.0,
    damping=0.5,

    effort_limit=33.5,
)


# =========================
# Articulation
# =========================

MY_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        MY_GO2ACTUATOR_CFG,
    ),
)

GO2_ACTION_SCALE: dict[str, float] = {}
for a in MY_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    GO2_ACTION_SCALE[n] = 0.25 * e / s

# =========================
# Robot EntityCfg
# =========================

def get_go2_robot_cfg() -> EntityCfg:

    return EntityCfg(
        init_state=ROBOT_INIT_STATE,

        spec_fn=get_spec,

        articulation=MY_ARTICULATION,
    )


# =========================
# 测试
# =========================

if __name__ == "__main__":
    import mujoco.viewer as viewer

    robot = Entity(get_go2_robot_cfg())

    print("Robot loaded successfully!")

    model = robot.spec.compile()

    print("Number of joints:", model.njnt)
    print("Number of actuators:", model.nu)

    viewer.launch(model)
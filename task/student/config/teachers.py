"""Teacher checkpoint registry for multi-teacher distillation.

Fill in the PPO checkpoint path for each terrain.  The student task for a
terrain selects its teacher from this registry by ``terrain_key``.
"""

TEACHER_CHECKPOINTS: dict[str, str] = {
    "Flat": "logs/rsl_rl/teacher_model/flat/obs_critic_teacher_1300_v2.pt",
    "Stair": "",
    "StairUp": "",
    "StairDown": "logs/rsl_rl/teacher_model/stair/obs_critic_teacher_1950_stair_down_v2.pt",
    "Platform": "",
    "PlumBlossom": "",
    "Stones2Rows": "",
    "TireArray": "",
    "LowClearanceBar": "",
    # Generalist teacher trained on the flat + stair_down mixed terrain.
    # Fill this after training with:
    #   ./train/train_teacher.sh <run_name> -e flat_stair_down
    # Then switch the student FlatStairDown task to multi_teacher=False.
    "FlatStairDown": (
        "logs/rsl_rl/go2_velocity/"
        "2026-11-14_17-49-35_teacher_flat_stair_down_first/model_2200.pt"
    ),
    # Generalist teacher trained on flat + platform + stair_down.
    # Fill this after training with:
    #   ./train/train_teacher.sh <run_name> -e flat_platform_stair_down
    "FlatPlatformStairDown": "logs/rsl_rl/go2_velocity/2026-11-15_15-48-30_teacher_flat_stair_platform/model_5650.pt",
    # Generalist teacher trained on flat + stair_down + platform + plum.
    # Fill this after training with:
    #   ./train/train_teacher.sh <run_name> -e flat_platform_stair_down_plum
    "FlatPlatformStairDownPlum": "",
    # Composite half-competition map teacher.
    # Fill this after training with:
    #   ./train/train_teacher.sh <run_name> -e halfcompetitionmap
    "HalfCompetitionMap": "",
}

# Mixed flat + stair_up task.  Keys must match the sub-terrain keys in
# config/terrain/mixed_terrain.py.
TEACHER_CHECKPOINTS_FLAT_STAIR_UP: dict[str, str] = {
    "flat": TEACHER_CHECKPOINTS["Flat"],
    "stair_up": TEACHER_CHECKPOINTS["StairUp"],
}

# Mixed flat + stair_down task.
TEACHER_CHECKPOINTS_FLAT_STAIR_DOWN: dict[str, str] = {
    "flat": TEACHER_CHECKPOINTS["Flat"],
    "stair_down": TEACHER_CHECKPOINTS["StairDown"],
}

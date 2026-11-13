"""Teacher checkpoint registry for multi-teacher distillation.

Fill in the PPO checkpoint path for each terrain.  The student task for a
terrain selects its teacher from this registry by ``terrain_key``.
"""

TEACHER_CHECKPOINTS: dict[str, str] = {
    "Flat": "",
    "Stair": "",
    "StairUp": "",
    "StairDown": "",
    "Platform": "",
    "PlumBlossom": "",
    "Stones2Rows": "",
    "TireArray": "",
    "LowClearanceBar": "",
}

# Mixed flat + stair_up task.  Keys must match the sub-terrain keys in
# config/terrain/mixed_terrain.py.
TEACHER_CHECKPOINTS_FLAT_STAIR_UP: dict[str, str] = {
    "flat": TEACHER_CHECKPOINTS["Flat"],
    "stair_up": TEACHER_CHECKPOINTS["StairUp"],
}

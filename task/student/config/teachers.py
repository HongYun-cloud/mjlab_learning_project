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

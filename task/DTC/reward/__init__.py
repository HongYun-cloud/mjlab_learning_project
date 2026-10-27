"""DTC 奖励项 (reward terms).

分工 (与 observation 包对称): 本包只放被 env_cfg RewardTermCfg 引用的
奖励函数; ``task/DTC/tamols`` 只放 TAMOLS 求解器与桥接等计算内容.

首个奖励: Base Pose Tracking ``r_{B_n}`` (n=0/1/2), 见 :mod:`terms`.
"""

"""DTC 环境观测项 (actor / critic obs terms).

分工 (用户约定): 本包只放**环境观测项函数** (被 env_cfg 的
ObservationTermCfg 引用); ``task/DTC/tamols`` 只放 TAMOLS 求解器
(NLP/GN-SQP)、RL 桥接、落足点可视化与摆动腿数据层等 TAMOLS 计算
相关内容.

观测清单与数值约定见 :mod:`terms` 文档.
"""

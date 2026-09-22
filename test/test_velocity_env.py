from myrobot_mjlab.task.velocity.config.env_cfg import my_go2_flat_env_cfg

cfg = my_go2_flat_env_cfg()

print("CONFIG OK")
print("terrain_type:", cfg.scene.terrain.terrain_type)
print("terrain_generator:", cfg.scene.terrain.terrain_generator)

print("sensors:")
for sensor in cfg.scene.sensors or ():
    print("  ", sensor.name)

print("rewards:")
for name in cfg.rewards:
    print("  ", name)

print("terminations:")
for name in cfg.terminations:
    print("  ", name)
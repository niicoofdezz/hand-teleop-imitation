"""Test del config del robot: carga el YAML, construye el modelo, va a 'home'
y abre el visor. Uso: python scripts/test_robot.py [configs/robots/panda.yaml]"""
import sys
import time

import mujoco
import mujoco.viewer

from handteleop.robot_config import RobotConfig, build_model, reset_to_home, tcp_position

cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/robots/panda.yaml"
cfg = RobotConfig.from_yaml(cfg_path)
model = build_model(cfg)
data = mujoco.MjData(model)
reset_to_home(cfg, model, data)

print(f"Robot: {cfg.name} | {cfg.n_arm} articulaciones de brazo | timestep {model.opt.timestep} s")
print(f"TCP en home: {tcp_position(cfg, data).round(3)} m")

# Abrir y cerrar la pinza cada 2 s para comprobar el actuador desde el config
grip_id = model.actuator(cfg.gripper.actuator).id
with mujoco.viewer.launch_passive(model, data) as viewer:
    t0 = time.time()
    while viewer.is_running():
        is_open = int((time.time() - t0) / 2) % 2 == 0
        data.ctrl[grip_id] = cfg.gripper.ctrl_open if is_open else cfg.gripper.ctrl_closed
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)

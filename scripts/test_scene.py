"""Test de la escena de la tarea.
- Cada 3 s genera una disposición nueva (semillas de train 0, 1, 2...).
- Con --in-box coloca el disco dentro de la caja para comprobar is_success.
Uso: python scripts/test_scene.py [--in-box]"""
import argparse
import time

import mujoco
import mujoco.viewer

from handteleop.robot_config import RobotConfig
from handteleop.task_disc_to_box import TaskConfig, build_scene, is_success, reset

ap = argparse.ArgumentParser()
ap.add_argument("--in-box", action="store_true", help="coloca el disco dentro de la caja")
args = ap.parse_args()

robot = RobotConfig.from_yaml("configs/robots/panda.yaml")
task = TaskConfig.from_yaml("configs/tasks/disc_to_box.yaml")
model = build_scene(robot, task)
data = mujoco.MjData(model)


def new_episode(i: int) -> None:
    info = reset(robot, task, model, data, seed=task.seed("train", i))
    if args.in_box:  # suelta el disco 10 cm por encima del centro de la caja
        adr = model.jnt_qposadr[model.joint("disc_free").id]
        data.qpos[adr:adr + 3] = [*info["box_xy"], 0.10]
        mujoco.mj_forward(model, data)
    print(f"seed {info['seed']:>3} | disco {info['disc_xy'].round(3)} | caja {info['box_xy'].round(3)}")


episode = 0
new_episode(episode)
with mujoco.viewer.launch_passive(model, data) as viewer:
    t_ep = time.time()
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
        if time.time() - t_ep > 3:
            print(f"        -> éxito: {is_success(task, model, data)}")
            episode += 1
            new_episode(episode)
            t_ep = time.time()

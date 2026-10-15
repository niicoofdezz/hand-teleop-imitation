"""Muestra lo que verá la política: cámara frontal + cámara de muñeca.
Las imágenes son de 128x128; aquí se amplían x4 para verlas bien.
Teclas: n = siguiente disposición | g = abrir/cerrar pinza | q = salir"""
import cv2
import mujoco
import numpy as np

from handteleop.cameras import CameraRenderer
from handteleop.robot_config import RobotConfig
from handteleop.task_disc_to_box import TaskConfig, build_scene, camera_names, reset

robot = RobotConfig.from_yaml("configs/robots/panda.yaml")
task = TaskConfig.from_yaml("configs/tasks/disc_to_box.yaml")
model = build_scene(robot, task)
data = mujoco.MjData(model)
cams = CameraRenderer(model, camera_names(robot, task), task.image_size)
grip_id = model.actuator(robot.gripper.actuator).id

episode, gripper_open = 0, True
reset(robot, task, model, data, seed=task.seed("train", episode))
print(f"Cámaras: {cams.camera_names} | {task.image_size}x{task.image_size}")

while True:
    data.ctrl[grip_id] = robot.gripper.ctrl_open if gripper_open else robot.gripper.ctrl_closed
    for _ in range(16):  # ~30 fps de imagen: 16 pasos de 2 ms = 32 ms
        mujoco.mj_step(model, data)

    images = cams.render(data)
    row = np.concatenate([images[n] for n in cams.camera_names], axis=1)   # una al lado de otra
    row = cv2.resize(row, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    cv2.imshow("camaras (front | wrist)", cv2.cvtColor(row, cv2.COLOR_RGB2BGR))  # MuJoCo da RGB

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("n"):
        episode += 1
        info = reset(robot, task, model, data, seed=task.seed("train", episode))
        print(f"seed {info['seed']}")
    if key == ord("g"):
        gripper_open = not gripper_open

cams.close()
cv2.destroyAllWindows()

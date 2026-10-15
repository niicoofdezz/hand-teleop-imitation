"""El "entorno": junta robot + tarea + IK + cámaras en un único objeto.

Todo lo que mueva el robot (el experto, tu mano, el teclado y, al final, la
política) usa ESTE objeto de la misma forma:
    obs = env.reset(seed)
    obs = env.step(arm_targets, gripper_open)
Así la comparación entre fuentes de datos es justa: misma física, mismas
cámaras, misma frecuencia.
"""
import mujoco
import numpy as np

from handteleop.cameras import CameraRenderer
from handteleop.ik import ArmIK
from handteleop.robot_config import RobotConfig, robot_state, tcp_position
from handteleop.task_disc_to_box import TaskConfig, build_scene, camera_names, is_success, reset


class DiscToBoxEnv:
    def __init__(self, robot_cfg: str = "configs/robots/panda.yaml",
                 task_cfg: str = "configs/tasks/disc_to_box.yaml", render: bool = True):
        self.robot = RobotConfig.from_yaml(robot_cfg)
        self.task = TaskConfig.from_yaml(task_cfg)
        self.model = build_scene(self.robot, self.task)
        self.data = mujoco.MjData(self.model)
        self.ik = ArmIK(self.robot, self.model)

        self.dt = 1.0 / self.task.control_fps
        self.substeps = round(self.dt / self.model.opt.timestep)  # pasos de física por decisión
        self.arm_ids = [self.model.actuator(a).id for a in self.robot.arm_actuators]
        self.grip_id = self.model.actuator(self.robot.gripper.actuator).id
        # Sin render (render=False) va mucho más rápido: útil para medir el experto
        self.cameras = (CameraRenderer(self.model, camera_names(self.robot, self.task),
                                       self.task.image_size) if render else None)
        self.time = 0.0

    def reset(self, seed: int) -> dict:
        self.info = reset(self.robot, self.task, self.model, self.data, seed)
        self.ik.sync_from_scene(self.data)
        self.time = 0.0
        return self.observe()

    def observe(self) -> dict:
        """Solo lo que mediría un robot real: imágenes + estado propio."""
        obs = {"state": robot_state(self.robot, self.model, self.data)}
        if self.cameras is not None:
            obs["images"] = self.cameras.render(self.data)
        return obs

    def step(self, arm_targets: np.ndarray, gripper_open: bool) -> dict:
        self.data.ctrl[self.arm_ids] = arm_targets
        g = self.robot.gripper
        self.data.ctrl[self.grip_id] = g.ctrl_open if gripper_open else g.ctrl_closed
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        self.time += self.dt
        return self.observe()

    def ik_step(self, tcp_target: np.ndarray) -> np.ndarray:
        """Posición deseada del TCP -> ángulos objetivo del brazo."""
        return self.ik.solve(tcp_target, self.dt)

    def tcp(self) -> np.ndarray:
        return tcp_position(self.robot, self.data)

    def settle(self, seconds: float = 0.5) -> None:
        """Deja correr la física sin cambiar nada (para que el disco se asiente)."""
        for _ in range(round(seconds / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)

    def success(self) -> bool:
        return is_success(self.task, self.model, self.data)

    @property
    def timeout(self) -> bool:
        return self.time >= self.task.max_episode_time

    def close(self) -> None:
        if self.cameras is not None:
            self.cameras.close()

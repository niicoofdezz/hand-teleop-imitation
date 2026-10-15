"""Cinemática inversa (IK) del brazo con mink.

Qué resuelve: "quiero el TCP AQUÍ" -> "pon las articulaciones ASÍ".

Truco de diseño: la IK trabaja con un modelo que tiene SOLO el robot (sin
disco ni caja). Así no intenta "mover" el disco para acercarse al objetivo,
y no depende de la tarea. Copiamos los ángulos del brazo de la escena a la
IK por NOMBRE de articulación, así que funciona con cualquier robot del config.
"""
import mink
import mujoco
import numpy as np

from handteleop.robot_config import RobotConfig, build_model, reset_to_home


class ArmIK:
    def __init__(self, robot: RobotConfig, scene_model: mujoco.MjModel):
        self.robot = robot
        self.model = build_model(robot)          # modelo SOLO robot
        self.config = mink.Configuration(self.model)

        # Dónde está cada articulación del brazo en cada modelo (pueden no coincidir)
        self._ik_adr = [self.model.jnt_qposadr[self.model.joint(j).id] for j in robot.arm_joints]
        self._scene_adr = [scene_model.jnt_qposadr[scene_model.joint(j).id] for j in robot.arm_joints]

        # Tarea 1: llevar el TCP a la posición/orientación objetivo
        self.tcp_task = mink.FrameTask(frame_name=robot.end_effector.tcp_name, frame_type="site",
                                       position_cost=1.0, orientation_cost=0.5)
        # Tarea 2 (suave): no alejarse mucho de la postura home -> evita posturas raras
        self.posture_task = mink.PostureTask(self.model, cost=1e-2)
        # Límites: no salirse del rango de cada articulación
        self.limits = [mink.ConfigurationLimit(self.model)]

        data = mujoco.MjData(self.model)
        reset_to_home(robot, self.model, data)
        self.posture_task.set_target(data.qpos.copy())
        # Orientación de la pinza en home (mirando hacia abajo): la mantendremos fija
        self.config.update(data.qpos.copy())
        self.home_rotation = self.config.get_transform_frame_to_world(
            robot.end_effector.tcp_name, "site").rotation()

    def sync_from_scene(self, scene_data: mujoco.MjData) -> None:
        """Copia los ángulos actuales del brazo de la simulación a la IK."""
        q = self.config.q.copy()
        q[self._ik_adr] = scene_data.qpos[self._scene_adr]
        self.config.update(q)

    def solve(self, target_pos: np.ndarray, dt: float) -> np.ndarray:
        """Un paso de IK hacia target_pos (con la pinza mirando abajo).
        Devuelve los ángulos objetivo de las articulaciones del brazo."""
        target = mink.SE3.from_rotation_and_translation(self.home_rotation, np.asarray(target_pos))
        self.tcp_task.set_target(target)
        velocity = mink.solve_ik(self.config, [self.tcp_task, self.posture_task], dt,
                                 solver="daqp", damping=1e-3, limits=self.limits)
        self.config.integrate_inplace(velocity, dt)
        return self.config.q[self._ik_adr].copy()

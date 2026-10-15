"""Experto programado: resuelve la tarea sin humano, leyendo la posición
real del disco y la caja (él SÍ puede hacer trampa; la política no).

Sirve para dos cosas:
  1. Comprobar que la escena, la IK y la pinza funcionan antes de teleoperar.
  2. Generar demos "perfectas" para comparar con las humanas.

Funciona como una lista de pasos (máquina de estados): ir encima del disco,
bajar, cerrar, subir, ir encima de la caja, abrir. Cada paso es un punto
objetivo para el TCP y un estado de la pinza.
"""
import mujoco
import numpy as np

from handteleop.task_disc_to_box import TaskConfig

HOVER_Z = 0.15      # m, altura de paso por encima de los objetos
GRASP_Z = 0.02      # m, altura del TCP al agarrar (centro del disco = 0.015)
DROP_Z = 0.12       # m, altura a la que suelta el disco sobre la caja
MAX_SPEED = 0.25    # m/s, velocidad máxima del TCP
REACHED = 0.01      # m, distancia para dar un punto por alcanzado (1 cm)
GRIP_WAIT = 0.5     # s, tiempo que esperamos a que la pinza cierre/abra


class ScriptedExpert:
    def __init__(self, task: TaskConfig, model: mujoco.MjModel):
        self.task = task
        self.model = model

    def reset(self, data: mujoco.MjData, tcp_start: np.ndarray) -> None:
        disc = data.body("disc").xpos.copy()
        box = self.model.body_pos[self.model.body("box").id].copy()
        # (punto objetivo, pinza abierta?, espera en segundos al llegar)
        self.plan = [
            ([disc[0], disc[1], HOVER_Z], True, 0.0),     # 1. encima del disco
            ([disc[0], disc[1], GRASP_Z], True, 0.3),     # 2. bajar (y esperar a estar centrado)
            ([disc[0], disc[1], GRASP_Z], False, GRIP_WAIT),  # 3. cerrar pinza
            ([disc[0], disc[1], HOVER_Z], False, 0.0),    # 4. subir
            ([box[0], box[1], HOVER_Z], False, 0.0),      # 5. encima de la caja
            ([box[0], box[1], DROP_Z], False, 0.0),       # 6. bajar un poco
            ([box[0], box[1], DROP_Z], True, GRIP_WAIT),  # 7. abrir pinza
            ([box[0], box[1], HOVER_Z], True, 0.0),       # 8. subir
        ]
        self.step_idx = 0
        self.wait = 0.0
        self.command = np.array(tcp_start, float)  # punto que se va moviendo poco a poco

    @property
    def done(self) -> bool:
        return self.step_idx >= len(self.plan)

    def act(self, tcp_now: np.ndarray, dt: float) -> tuple[np.ndarray, bool]:
        """Devuelve (posición objetivo del TCP, pinza abierta?) para este instante."""
        if self.done:
            goal, grip_open, _ = self.plan[-1]
            return np.array(goal), grip_open
        goal, grip_open, wait = self.plan[self.step_idx]
        goal = np.array(goal)

        # Mover el punto de mando hacia el objetivo, sin pasar de MAX_SPEED
        delta = goal - self.command
        dist = np.linalg.norm(delta)
        if dist > MAX_SPEED * dt:
            delta *= MAX_SPEED * dt / dist
        self.command += delta

        # ¿Llegó el TCP REAL? Entonces esperar (si toca) y pasar al siguiente paso
        if np.linalg.norm(goal - tcp_now) < REACHED:
            self.wait += dt
            if self.wait >= wait:
                self.step_idx += 1
                self.wait = 0.0
        return self.command.copy(), grip_open

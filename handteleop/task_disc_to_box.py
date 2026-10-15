"""Escena y lógica de la tarea "disco a la caja".

Responsabilidades (y nada más):
  - build_scene: robot + suelo + luz + disco + caja -> MjModel
  - reset: postura home + disposición aleatoria a partir de una semilla
  - is_success: ¿está el disco dentro de la caja?
La tarea solo sabe de la tarea; lo del robot lo pide a RobotConfig.
"""
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import yaml

from handteleop.robot_config import REPO_ROOT, RobotConfig, robot_spec, validate_against_model

GEOM = mujoco.mjtGeom


@dataclass
class TaskConfig:
    name: str
    disc: dict
    box: dict
    randomization: dict
    seeds: dict
    success: dict

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TaskConfig":
        path = Path(path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        with open(path) as f:
            return cls(**yaml.safe_load(f))

    def seed(self, split: str, i: int) -> int:
        """Semilla del episodio i del split 'train' o 'test'."""
        return self.seeds[f"{split}_start"] + i


def _add_floor_and_light(spec: mujoco.MjSpec) -> None:
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.2, 0.3], width=300, height=300)
    mat = spec.add_material(name="floor_mat", texrepeat=[5, 5], texuniform=True)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    spec.worldbody.add_geom(name="floor", type=GEOM.mjGEOM_PLANE, size=[0, 0, 0.05],
                            material="floor_mat")
    spec.worldbody.add_light(pos=[0, 0, 1.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)


def _add_disc(spec: mujoco.MjSpec, d: dict) -> None:
    # Cuerpo libre (freejoint): 6 grados de libertad, la física decide dónde acaba
    body = spec.worldbody.add_body(name="disc", pos=[0.5, -0.15, d["half_height"]])
    body.add_freejoint(name="disc_free")
    body.add_geom(name="disc_geom", type=GEOM.mjGEOM_CYLINDER,
                  size=[d["radius"], d["half_height"], 0], mass=d["mass"],
                  friction=[d["friction"], 0.005, 0.0001], rgba=d["rgba"])


def _add_box(spec: mujoco.MjSpec, b: dict) -> None:
    # Cuerpo SIN joint = estático. Su posición se cambia en reset() vía model.body_pos
    body = spec.worldbody.add_body(name="box", pos=[0.5, 0.2, 0])
    s, h, t = b["inner_size"] / 2, b["wall_height"] / 2, b["wall_thickness"] / 2
    walls = {  # nombre: (posición del centro, medias dimensiones)
        "wall_px": ([s + t, 0, h], [t, s + 2 * t, h]),
        "wall_nx": ([-s - t, 0, h], [t, s + 2 * t, h]),
        "wall_py": ([0, s + t, h], [s, t, h]),
        "wall_ny": ([0, -s - t, h], [s, t, h]),
    }
    for name, (pos, size) in walls.items():
        body.add_geom(name=name, type=GEOM.mjGEOM_BOX, pos=pos, size=size, rgba=b["rgba"])


def build_scene(robot: RobotConfig, task: TaskConfig) -> mujoco.MjModel:
    spec = robot_spec(robot)
    _add_floor_and_light(spec)
    _add_disc(spec, task.disc)
    _add_box(spec, task.box)

    # El keyframe 'home' del robot no conoce el disco: le añadimos su qpos
    # (posición xyz + cuaternión wxyz) para que siga siendo válido.
    key = spec.key(robot.home_keyframe)
    key.qpos = list(key.qpos) + [0.5, -0.15, task.disc["half_height"], 1, 0, 0, 0]

    model = spec.compile()
    validate_against_model(robot, model)
    return model


def reset(robot: RobotConfig, task: TaskConfig, model: mujoco.MjModel,
          data: mujoco.MjData, seed: int) -> dict:
    """Postura home + disposición aleatoria. Misma semilla -> misma disposición."""
    rng = np.random.default_rng(seed)
    r = task.randomization
    disc_xy = np.array([rng.uniform(*r["disc_x"]), rng.uniform(*r["disc_y"])])
    box_xy = np.array([rng.uniform(*r["box_x"]), rng.uniform(*r["box_y"])])

    mujoco.mj_resetDataKeyframe(model, data, model.key(robot.home_keyframe).id)

    # Disco: escribir en su qpos (dirección del freejoint en el vector de estado)
    adr = model.jnt_qposadr[model.joint("disc_free").id]
    data.qpos[adr:adr + 3] = [*disc_xy, task.disc["half_height"]]
    data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]

    # Caja: es estática, así que se mueve cambiando el MODELO, no el estado
    model.body_pos[model.body("box").id][:2] = box_xy

    mujoco.mj_forward(model, data)
    return {"seed": seed, "disc_xy": disc_xy, "box_xy": box_xy}


def is_success(task: TaskConfig, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    disc = data.body("disc").xpos
    box = model.body_pos[model.body("box").id]
    half = task.box["inner_size"] / 2 - task.success["margin"]
    inside_xy = bool(np.all(np.abs(disc[:2] - box[:2]) < half))
    resting = disc[2] < task.box["wall_height"]
    return inside_xy and resting

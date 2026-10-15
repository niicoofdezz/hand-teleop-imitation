"""Configuración del robot: leer el YAML, validarlo y construir el modelo MuJoCo.

Idea clave: el resto del código NUNCA escribe "joint1" o "actuator8" a mano.
Pide los nombres a RobotConfig. Así, cambiar de robot = cambiar de YAML.
"""
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import yaml

# Raíz del repo: este archivo está en <repo>/handteleop/robot_config.py
REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class GripperConfig:
    actuator: str
    ctrl_open: float
    ctrl_closed: float
    state_joint: str


@dataclass
class EndEffectorConfig:
    body: str
    tcp_name: str
    tcp_offset: list[float]


@dataclass
class RobotConfig:
    name: str
    mjcf: Path
    arm_joints: list[str]
    arm_actuators: list[str]
    gripper: GripperConfig
    end_effector: EndEffectorConfig
    home_keyframe: str
    # Cámaras montadas EN el robot (p. ej. la de muñeca). Las fijas van en la tarea.
    cameras: dict = field(default_factory=dict)
    gravity_compensation: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RobotConfig":
        """Lee el YAML y lo convierte en objetos con tipos (no diccionarios sueltos)."""
        path = Path(path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        with open(path) as f:
            raw = yaml.safe_load(f)

        cfg = cls(
            name=raw["name"],
            mjcf=REPO_ROOT / raw["mjcf"],
            arm_joints=raw["arm_joints"],
            arm_actuators=raw["arm_actuators"],
            gripper=GripperConfig(**raw["gripper"]),
            end_effector=EndEffectorConfig(**raw["end_effector"]),
            home_keyframe=raw["home_keyframe"],
            cameras=raw.get("cameras", {}),
            gravity_compensation=raw.get("gravity_compensation", True),
        )
        cfg._check_basic()
        return cfg

    def _check_basic(self) -> None:
        """Errores que se pueden detectar sin abrir MuJoCo."""
        if not self.mjcf.exists():
            raise FileNotFoundError(f"[{self.name}] No existe el MJCF: {self.mjcf}")
        if len(self.arm_joints) != len(self.arm_actuators):
            raise ValueError(f"[{self.name}] arm_joints y arm_actuators deben tener la misma longitud")
        if len(self.end_effector.tcp_offset) != 3:
            raise ValueError(f"[{self.name}] tcp_offset debe tener 3 valores (x, y, z)")

    @property
    def n_arm(self) -> int:
        return len(self.arm_joints)


def robot_spec(cfg: RobotConfig) -> mujoco.MjSpec:
    """Carga el MJCF del robot y le añade el site del TCP (sin compilar).

    Devolvemos el *spec* (modelo editable) para que la tarea pueda añadir
    después suelo, objetos y cámaras antes de compilar.
    Usamos MjSpec para no tocar los archivos de Menagerie: el TCP es una
    decisión NUESTRA, así que vive en nuestro config.
    """
    spec = mujoco.MjSpec.from_file(str(cfg.mjcf))
    ee = cfg.end_effector
    body = spec.body(ee.body)
    if body is None:
        raise ValueError(f"[{cfg.name}] El cuerpo '{ee.body}' no existe en el MJCF")
    body.add_site(name=ee.tcp_name, pos=ee.tcp_offset, size=[0.01, 0.01, 0.01],
                  rgba=[1, 0, 0, 1])  # esfera roja: así se ve en el visor

    # Compensación de gravedad: el controlador "sujeta" el peso del brazo, como
    # hace el controlador del Panda real. Sin esto, el brazo se queda unos mm
    # por debajo de donde le mandas (los actuadores son muelles: necesitan error
    # para hacer fuerza).
    if cfg.gravity_compensation:
        for b in spec.bodies:
            b.gravcomp = 1.0

    # Cámaras pegadas a un cuerpo del robot: se mueven con él
    for name, cam in cfg.cameras.items():
        parent = spec.body(cam["body"])
        if parent is None:
            raise ValueError(f"[{cfg.name}] Cámara '{name}': no existe el cuerpo '{cam['body']}'")
        parent.add_camera(name=name, pos=cam["pos"], quat=euler_to_quat(cam["euler"]),
                          fovy=cam["fovy"])
    return spec


def euler_to_quat(euler_deg: list[float]) -> list[float]:
    """Ángulos en grados (giros sobre x, y, z) -> cuaternión (w, x, y, z) de MuJoCo."""
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.deg2rad(euler_deg), "xyz")
    return quat.tolist()


def build_model(cfg: RobotConfig) -> mujoco.MjModel:
    """Solo el robot (sin escena): útil para probar el config."""
    model = robot_spec(cfg).compile()
    validate_against_model(cfg, model)
    return model


def validate_against_model(cfg: RobotConfig, model: mujoco.MjModel) -> None:
    """Comprueba que todos los nombres del YAML existen en el modelo.

    Si te equivocas al escribir un nombre, mejor un error claro ahora que un
    comportamiento raro dentro de la IK dentro de dos semanas.
    """
    checks = (
        [(mujoco.mjtObj.mjOBJ_JOINT, n) for n in cfg.arm_joints]
        + [(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in cfg.arm_actuators]
        + [(mujoco.mjtObj.mjOBJ_ACTUATOR, cfg.gripper.actuator),
           (mujoco.mjtObj.mjOBJ_JOINT, cfg.gripper.state_joint),
           (mujoco.mjtObj.mjOBJ_KEY, cfg.home_keyframe)]
    )
    missing = [name for obj, name in checks if mujoco.mj_name2id(model, obj, name) == -1]
    if missing:
        raise ValueError(f"[{cfg.name}] Nombres que no existen en el modelo: {missing}")


def reset_to_home(cfg: RobotConfig, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Pone el robot en la postura 'home' (posición y consignas de los actuadores)."""
    mujoco.mj_resetDataKeyframe(model, data, model.key(cfg.home_keyframe).id)
    mujoco.mj_forward(model, data)  # recalcula posiciones de cuerpos y sites


def tcp_position(cfg: RobotConfig, data: mujoco.MjData) -> np.ndarray:
    """Posición del TCP en el mundo (m)."""
    return data.site(cfg.end_effector.tcp_name).xpos.copy()


def robot_state(cfg: RobotConfig, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Lo que el robot sabe de sí mismo (como un robot real): ángulos del brazo
    + apertura de la pinza normalizada (0 = cerrada, 1 = abierta)."""
    arm = [data.joint(j).qpos[0] for j in cfg.arm_joints]
    lo, hi = model.joint(cfg.gripper.state_joint).range
    grip = (data.joint(cfg.gripper.state_joint).qpos[0] - lo) / (hi - lo)
    return np.array([*arm, grip], dtype=np.float32)

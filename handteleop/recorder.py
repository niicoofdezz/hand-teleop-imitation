"""Grabar demos en formato LeRobot.

Cada demo (episodio) es una secuencia de frames. En cada frame guardamos:
  - observation.images.<cámara>: imagen 128x128 (se guarda como vídeo)
  - observation.state: 7 ángulos del brazo + apertura de la pinza (0..1)
  - action: lo que se mandó al robot = 7 ángulos objetivo + pinza (1 abierta, 0 cerrada)
La política aprenderá: (imágenes, estado) -> acción.

Además guardamos aparte (episodes_meta.json) datos que NO son para la
política sino para el estudio: semilla, fuente (script/mano/teclado), éxito
y duración de cada demo.
"""
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from handteleop.env import DiscToBoxEnv
from handteleop.robot_config import REPO_ROOT


def make_features(env: DiscToBoxEnv) -> dict:
    """Describe a LeRobot qué hay en cada frame (nombres, tamaños y tipos)."""
    r, s = env.robot, env.task.image_size
    names = [*r.arm_joints, "gripper"]
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
        "action": {"dtype": "float32", "shape": (len(names),), "names": names},
    }
    for cam in env.cameras.camera_names:
        features[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": (s, s, 3), "names": ["height", "width", "channels"]}
    return features


class DemoRecorder:
    def __init__(self, env: DiscToBoxEnv, name: str, source: str):
        """name: carpeta dentro de data/ | source: 'script', 'hand' o 'keyboard'."""
        self.env, self.source = env, source
        self.root = REPO_ROOT / "data" / name
        if self.root.exists():
            raise FileExistsError(f"{self.root} ya existe: usa otro nombre o bórrala")
        self.dataset = LeRobotDataset.create(
            repo_id=f"local/{name}", fps=env.task.control_fps, features=make_features(env),
            root=self.root, robot_type=env.robot.name, use_videos=True)
        self.meta: list[dict] = []
        self.n_frames = 0

    def add_frame(self, obs: dict, arm_targets: np.ndarray, gripper_open: bool) -> None:
        """Guarda la observación ANTES de actuar y la acción que se tomó."""
        frame = {
            "observation.state": obs["state"],
            "action": np.array([*arm_targets, float(gripper_open)], dtype=np.float32),
            "task": self.env.task.instruction,
        }
        for cam, img in obs["images"].items():
            frame[f"observation.images.{cam}"] = img
        self.dataset.add_frame(frame)
        self.n_frames += 1

    def end_episode(self, seed: int, success: bool, keep: bool = True) -> None:
        """Cierra la demo. keep=False la descarta (p. ej. si falló)."""
        if keep:
            self.dataset.save_episode()
            self.meta.append({"episode": len(self.meta), "seed": seed, "source": self.source,
                              "success": bool(success), "frames": self.n_frames,
                              "duration_s": round(self.n_frames / self.env.task.control_fps, 2)})
        else:
            self.dataset.clear_episode_buffer()
        self.n_frames = 0

    def close(self) -> None:
        self.dataset.finalize()
        with open(self.root / "episodes_meta.json", "w") as f:
            json.dump(self.meta, f, indent=2)

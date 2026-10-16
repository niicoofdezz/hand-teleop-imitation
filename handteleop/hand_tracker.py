"""Webcam + MediaPipe -> lo que necesita la teleoperación.

Devuelve en cada frame un HandState con solo 4 datos útiles:
  - detected: ¿hay mano?
  - wrist:    posición de la muñeca en la imagen (x, y), de 0 a 1
  - size:     tamaño aparente de la mano (más grande = más cerca de la cámara).
              Es nuestra forma de estimar la PROFUNDIDAD con una sola webcam.
  - pinch:    pellizco pulgar-índice normalizado (≈0.1 juntos, ≈1 abierto)
El resto del proyecto no sabe nada de OpenCV ni de MediaPipe: solo usa esto.
"""
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import yaml
from mediapipe.tasks.python import BaseOptions, vision

from handteleop.robot_config import REPO_ROOT

WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP = 0, 4, 8, 9


@dataclass
class HandState:
    detected: bool
    wrist: np.ndarray = field(default_factory=lambda: np.zeros(2))
    size: float = 0.0
    pinch: float = 0.0
    landmarks: list = field(default_factory=list)   # 21 puntos (x, y) en píxeles, para dibujar
    frame: np.ndarray | None = None                  # imagen BGR (espejo)
    t: float = 0.0                                   # instante de la captura (s)


def load_teleop_config(path: str | Path = "configs/teleop/hand.yaml") -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path) as f:
        return yaml.safe_load(f)


class HandTracker:
    def __init__(self, cfg: dict):
        cam = cfg["camera"]
        if cam.get("fix_exposure"):
            # Lo mismo que hacías a mano con v4l2-ctl tras cada reinicio.
            # Si el control no existe en tu cámara, no pasa nada.
            for ctrl in ("exposure_dynamic_framerate=0", "exposure_auto_priority=0"):
                subprocess.run(["v4l2-ctl", "-d", f"/dev/video{cam['index']}", "-c", ctrl],
                               capture_output=True)

        self.cap = cv2.VideoCapture(cam["index"], cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # antes que la resolución
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["height"])
        self.cap.set(cv2.CAP_PROP_FPS, cam["fps"])
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara {cam['index']}")

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(REPO_ROOT / cfg["model_path"])),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        )
        # Sin "with": el detector tiene que seguir vivo entre llamadas a read()
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.t_start = time.perf_counter()
        self.connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

    def read(self) -> HandState:
        ok, frame = self.cap.read()
        t = time.perf_counter() - self.t_start
        if not ok:
            return HandState(False, t=t)
        frame = cv2.flip(frame, 1)                       # espejo
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)     # MediaPipe quiere RGB
        result = self.landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(t * 1000))

        if not result.hand_landmarks:
            return HandState(False, frame=frame, t=t)

        lm = result.hand_landmarks[0]
        h, w = frame.shape[:2]

        def dist(a, b):
            return float(np.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y))

        size = dist(WRIST, MIDDLE_MCP)                   # largo de la palma
        return HandState(
            detected=True,
            wrist=np.array([lm[WRIST].x, lm[WRIST].y]),
            size=size,
            pinch=dist(THUMB_TIP, INDEX_TIP) / max(size, 1e-6),
            landmarks=[(int(p.x * w), int(p.y * h)) for p in lm],
            frame=frame,
            t=t,
        )

    def draw(self, state: HandState, color=(0, 255, 0)) -> None:
        """Dibuja el esqueleto de la mano sobre state.frame."""
        if not state.detected:
            return
        for c in self.connections:
            cv2.line(state.frame, state.landmarks[c.start], state.landmarks[c.end], color, 2)

    def close(self) -> None:
        self.cap.release()
        self.landmarker.close()


class PinchGripper:
    """Pellizco -> pinza abierta/cerrada, con histéresis (como un termostato)."""

    def __init__(self, close_below: float, open_above: float):
        assert close_below < open_above, "close_below debe ser menor que open_above"
        self.close_below, self.open_above = close_below, open_above
        self.is_open = True

    def __call__(self, pinch: float) -> bool:
        if self.is_open and pinch < self.close_below:
            self.is_open = False
        elif not self.is_open and pinch > self.open_above:
            self.is_open = True
        return self.is_open

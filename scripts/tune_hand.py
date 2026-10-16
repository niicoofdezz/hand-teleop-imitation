"""🧑 Herramienta para AJUSTAR la mano antes de teleoperar.

Qué ves:
  - Punto ROJO: la muñeca tal cual sale de MediaPipe (con temblor).
  - Punto VERDE: la muñeca filtrada (lo que usará el robot) + su estela.
  - Arriba: temblor medido (rojo vs verde), tamaño de la mano y pellizco.
  - Barra: estado de la pinza (ABIERTA/CERRADA) con la histéresis.
Deslizadores: min_cutoff y beta del filtro One Euro (pruébalos en directo).
Teclas: s = imprimir los valores para copiarlos en hand.yaml | q = salir
"""
from collections import deque

import cv2
import numpy as np

from handteleop.filters import OneEuroFilter
from handteleop.hand_tracker import HandTracker, PinchGripper, load_teleop_config

cfg = load_teleop_config()
tracker = HandTracker(cfg)
gripper = PinchGripper(**cfg["pinch"])
filt = OneEuroFilter(**cfg["filter"])

WIN = "ajuste de la mano"
cv2.namedWindow(WIN)
# Los deslizadores de OpenCV solo admiten enteros: guardamos x100 y x10
cv2.createTrackbar("min_cutoff x100", WIN, int(cfg["filter"]["min_cutoff"] * 100), 500, lambda v: None)
cv2.createTrackbar("beta x10", WIN, int(cfg["filter"]["beta"] * 10), 500, lambda v: None)

raw_hist, filt_hist = deque(maxlen=30), deque(maxlen=30)   # último segundo (30 frames)
trail = deque(maxlen=40)
font = cv2.FONT_HERSHEY_SIMPLEX

try:
    while True:
        filt.min_cutoff = max(cv2.getTrackbarPos("min_cutoff x100", WIN) / 100, 0.01)
        filt.beta = cv2.getTrackbarPos("beta x10", WIN) / 10

        s = tracker.read()
        if s.frame is None:
            continue
        frame = s.frame
        h, w = frame.shape[:2]

        if s.detected:
            tracker.draw(s, color=(200, 200, 200))
            raw = np.array([*s.wrist, s.size])
            f = filt(raw, s.t)
            raw_hist.append(raw[:2] * [w, h])
            filt_hist.append(f[:2] * [w, h])
            trail.append(tuple((f[:2] * [w, h]).astype(int)))

            cv2.circle(frame, tuple((raw[:2] * [w, h]).astype(int)), 6, (0, 0, 255), -1)
            for i in range(1, len(trail)):
                cv2.line(frame, trail[i - 1], trail[i], (0, 200, 0), 2)
            cv2.circle(frame, trail[-1], 6, (0, 255, 0), -1)

            # Temblor = cuánto se mueve el punto en el último segundo (px).
            # Úsalo con la mano QUIETA: el verde debe temblar mucho menos que el rojo.
            jr = np.std(np.array(raw_hist), axis=0).mean()
            jf = np.std(np.array(filt_hist), axis=0).mean()
            is_open = gripper(s.pinch)
            cv2.putText(frame, f"temblor  bruto {jr:4.1f}px  filtrado {jf:4.1f}px", (10, 25), font, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"tamano mano {f[2]:.3f}   pinch {s.pinch:.2f}", (10, 50), font, 0.6, (255, 255, 0), 2)
            color = (0, 200, 0) if is_open else (0, 0, 255)
            cv2.rectangle(frame, (10, h - 40), (200, h - 10), color, -1)
            cv2.putText(frame, "ABIERTA" if is_open else "CERRADA", (20, h - 18), font, 0.7, (255, 255, 255), 2)
        else:
            filt.reset()          # al volver la mano, que no "salte" desde la posición vieja
            trail.clear()
            cv2.putText(frame, "sin mano", (10, 25), font, 0.7, (0, 0, 255), 2)

        cv2.imshow(WIN, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            print(f"\nfilter:\n  min_cutoff: {filt.min_cutoff:.2f}\n  beta: {filt.beta:.1f}\n  d_cutoff: 1.0")
finally:
    tracker.close()
    cv2.destroyAllWindows()

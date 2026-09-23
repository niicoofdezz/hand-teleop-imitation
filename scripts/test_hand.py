"""Test de MediaPipe: detecta la mano, dibuja los 21 puntos y mide la latencia.
Pulsa q para salir."""
import time
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

CAM_INDEX = 0
MODEL_PATH = "assets/models/hand_landmarker.task"

# Índices de los puntos que usaremos para teleoperar
WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP = 0, 4, 8, 9

# --- Configurar el detector ---
options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    # VIDEO: aprovecha que los frames son consecutivos (hace tracking en vez de
    # buscar la mano desde cero en cada imagen) -> más rápido y más estable
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
)
CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS  # pares de puntos a unir

# --- Cámara (igual que en test_webcam.py) ---
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

latencies = []
t_start = time.perf_counter()

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # OpenCV da BGR, MediaPipe espera RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # En modo VIDEO cada frame necesita un timestamp en ms, siempre creciente
        timestamp_ms = int((time.perf_counter() - t_start) * 1000)

        t0 = time.perf_counter()
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        latencies.append((time.perf_counter() - t0) * 1000)

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]  # lista de 21 puntos con .x .y .z

            # x, y vienen NORMALIZADAS en [0, 1] -> convertir a píxeles para dibujar
            pts = [(int(p.x * w), int(p.y * h)) for p in lm]
            for c in CONNECTIONS:
                cv2.line(frame, pts[c.start], pts[c.end], (0, 255, 0), 2)
            for p in pts:
                cv2.circle(frame, p, 4, (0, 0, 255), -1)

            # Pellizco: distancia pulgar-índice, DIVIDIDA por el tamaño de la mano
            # (muñeca -> nudillo del corazón). Así no depende de lo cerca que
            # esté la mano de la cámara.
            def dist(a, b):
                return ((lm[a].x - lm[b].x) ** 2 + (lm[a].y - lm[b].y) ** 2) ** 0.5
            pinch = dist(THUMB_TIP, INDEX_TIP) / dist(WRIST, MIDDLE_MCP)

            wx, wy = lm[WRIST].x, lm[WRIST].y
            cv2.putText(frame, f"muneca x={wx:.2f} y={wy:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"pinch={pinch:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        avg = sum(latencies[-30:]) / len(latencies[-30:])
        cv2.putText(frame, f"MediaPipe: {avg:.1f} ms", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("hand", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
print(f"Latencia media MediaPipe: {sum(latencies) / len(latencies):.1f} ms")

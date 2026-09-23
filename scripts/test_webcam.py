"""Test de la webcam: muestra la imagen y los FPS reales. Pulsa q para salir."""
import time
import cv2

CAM_INDEX = 0  # /dev/video0

# CAP_V4L2: backend nativo de Linux para cámaras (más fiable que el automático)
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)

# El formato ANTES que la resolución: algunos drivers lo ignoran si va después.
# MJPG: la cámara envía imágenes comprimidas -> suele permitir 30 fps.
# Sin esto, muchas webcams usan YUYV y se quedan en 5-15 fps a 640x480.
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise RuntimeError(f"No se pudo abrir la cámara {CAM_INDEX}")

# Comprobar qué ha aceptado realmente la cámara.
# FOURCC es un número de 32 bits que en realidad son 4 letras ("MJPG", "YUYV"...)
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
fmt = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Formato: {fmt} | {w}x{h} | FPS pedidos por el driver: {cap.get(cv2.CAP_PROP_FPS)}")

frames, t0, fps = 0, time.perf_counter(), 0.0
while True:
    ok, frame = cap.read()          # frame = array numpy (alto, ancho, 3) en BGR
    if not ok:
        print("La cámara no devuelve imágenes")
        break

    # Calcula los FPS cada segundo
    frames += 1
    elapsed = time.perf_counter() - t0
    if elapsed >= 1.0:
        fps, frames, t0 = frames / elapsed, 0, time.perf_counter()

    frame = cv2.flip(frame, 1)      # efecto espejo: más natural para teleoperar
    cv2.putText(frame, f"{fps:.1f} fps | {fmt} {frame.shape[1]}x{frame.shape[0]}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):   # waitKey también refresca la ventana
        break

cap.release()
cv2.destroyAllWindows()
print(f"FPS final: {fps:.1f}")

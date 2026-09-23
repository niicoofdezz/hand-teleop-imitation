# Hand Teleoperation + Imitation Learning (MuJoCo)

Teleoperación de un robot simulado con la mano (webcam + MediaPipe) y
aprendizaje por imitación a partir de las demos grabadas.

En desarrollo.

## Notas de rendimiento
- Ejecutar con el portátil enchufado y en modo de energía "Rendimiento":
  la latencia de MediaPipe baja de ~33 ms a ~20 ms (i7-10870H).
- Webcam: MJPG 640x480 @ 30 fps. Si baja a ~7.5 fps con poca luz:
  `v4l2-ctl -d /dev/video0 -c exposure_dynamic_framerate=0`

"""Renderizar las cámaras de la simulación como imágenes (arrays numpy).

Es lo que "ve" la política: igual que un robot real solo tendría sus cámaras,
la política nunca recibe la posición del disco, solo estas imágenes.
"""
import mujoco
import numpy as np


class CameraRenderer:
    def __init__(self, model: mujoco.MjModel, camera_names: list[str], size: int):
        self.model = model
        self.camera_names = camera_names
        # Un renderizador "fuera de pantalla": dibuja en memoria, no en una ventana
        self.renderer = mujoco.Renderer(model, height=size, width=size)
        # Ocultar los sites (como la bolita roja del TCP): son ayudas para
        # nosotros, una cámara real no las vería
        self.options = mujoco.MjvOption()
        self.options.sitegroup[:] = 0

    def render(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        """Devuelve {nombre_cámara: imagen RGB (alto, ancho, 3) uint8}."""
        images = {}
        for name in self.camera_names:
            self.renderer.update_scene(data, camera=name, scene_option=self.options)
            images[name] = self.renderer.render().copy()  # copy: el buffer se reutiliza
        return images

    def close(self) -> None:
        self.renderer.close()

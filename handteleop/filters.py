"""Filtro One Euro (Casiez et al., 2012): suaviza señales con ruido.

El problema: los puntos de MediaPipe "tiemblan" unos píxeles aunque tengas
la mano quieta. Un filtro normal (media móvil) quita el temblor pero añade
retraso: el robot iría siempre por detrás de tu mano.

La idea del One Euro: ajustar cuánto suaviza según la VELOCIDAD.
  - Mano quieta o lenta -> suaviza mucho (adiós temblor).
  - Mano rápida         -> suaviza poco (casi sin retraso).
Es el filtro estándar en realidad virtual y en seguimiento de manos.
"""
import math

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    """Cuánto peso dar al dato nuevo (0..1) para una frecuencia de corte dada."""
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 5.0, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self.x_prev = None     # último valor filtrado
        self.dx_prev = None    # última velocidad filtrada
        self.t_prev = None

    def __call__(self, x, t: float) -> np.ndarray:
        """x: valor (número o vector) | t: instante en segundos. Devuelve x filtrado."""
        x = np.asarray(x, dtype=float)
        if self.x_prev is None:                 # primer dato: no hay nada que suavizar
            self.x_prev, self.dx_prev, self.t_prev = x, np.zeros_like(x), t
            return x
        dt = max(t - self.t_prev, 1e-6)
        self.t_prev = t

        # 1. Velocidad de la mano (también suavizada, si no sería puro ruido)
        dx = (x - self.x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        # 2. Cuanto más rápido te mueves, más alta la frecuencia de corte (= menos suavizado)
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat

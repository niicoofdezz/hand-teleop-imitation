"""Ejecuta el experto programado y mide su tasa de éxito.
  python scripts/run_expert.py                            # con visor
  python scripts/run_expert.py --episodes 100 --headless  # sin visor, solo la tasa de éxito
"""
import argparse
import time

import mujoco.viewer
import numpy as np

from handteleop.env import DiscToBoxEnv
from handteleop.expert import ScriptedExpert

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=10)
ap.add_argument("--headless", action="store_true", help="sin visor (más rápido)")
ap.add_argument("--split", default="train", choices=["train", "test"])
args = ap.parse_args()

env = DiscToBoxEnv(render=False)   # aquí no necesitamos las cámaras
expert = ScriptedExpert(env.task, env.model)


def run_episode(i: int, viewer=None) -> tuple[bool, float]:
    env.reset(env.task.seed(args.split, i))
    expert.reset(env.data, env.tcp())
    while not expert.done and not env.timeout:
        target, grip_open = expert.act(env.tcp(), env.dt)
        env.step(env.ik_step(target), grip_open)
        if viewer is not None:
            if not viewer.is_running():
                raise SystemExit
            viewer.sync()
            time.sleep(env.dt)   # velocidad real, para poder verlo
    env.settle()
    return env.success(), env.time


def report(i, ok, t):
    print(f"episodio {i:>3} (seed {env.task.seed(args.split, i)}) | {'ÉXITO' if ok else 'fallo'} | {t:.1f} s")


results = []
if args.headless:
    for i in range(args.episodes):
        ok, t = run_episode(i)
        results.append(ok)
        report(i, ok, t)
    print(f"\nTasa de éxito: {np.mean(results) * 100:.0f} % ({sum(results)}/{len(results)})")
else:
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for i in range(args.episodes):
            ok, t = run_episode(i, viewer)
            results.append(ok)
            report(i, ok, t)
        print(f"\nTasa de éxito: {np.mean(results) * 100:.0f} % ({sum(results)}/{len(results)})")
        # No cerramos la ventana desde el código: en algunos portátiles con NVIDIA
        # eso provoca un error de X11 (BadPixmap). Esperamos a que la cierres tú.
        print("Terminado. Cierra la ventana del visor para salir.")
        while viewer.is_running():
            time.sleep(0.1)

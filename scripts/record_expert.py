"""Graba demos del experto programado en formato LeRobot.
  python scripts/record_expert.py --episodes 150 --name script_v1
Solo se guardan las demos con éxito. Las semillas son las de TRAIN."""
import argparse

from handteleop.env import DiscToBoxEnv
from handteleop.expert import ScriptedExpert
from handteleop.recorder import DemoRecorder

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=150)
ap.add_argument("--name", default="script_v1", help="carpeta dentro de data/")
args = ap.parse_args()

env = DiscToBoxEnv()
expert = ScriptedExpert(env.task, env.model)
recorder = DemoRecorder(env, args.name, source="script")

kept = 0
for i in range(args.episodes):
    seed = env.task.seed("train", i)
    obs = env.reset(seed)
    expert.reset(env.data, env.tcp())
    while not expert.done and not env.timeout:
        target, grip_open = expert.act(env.tcp(), env.dt)
        arm_targets = env.ik_step(target)
        recorder.add_frame(obs, arm_targets, grip_open)   # observación + acción
        obs = env.step(arm_targets, grip_open)
    env.settle()
    ok = env.success()
    recorder.end_episode(seed, ok, keep=ok)   # solo se guardan las que salen bien
    kept += ok
    print(f"demo {i:>3} (seed {seed}) | {'guardada' if ok else 'DESCARTADA'}")

recorder.close()
env.close()
print(f"\n{kept}/{args.episodes} demos guardadas en {recorder.root}")

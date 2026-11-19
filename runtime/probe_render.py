#!/usr/bin/env python
"""Probe P1 (README section 8): GPU EGL rendering of LIBERO-Object in the container, prototype area "VLA".

In each of --procs spawn processes: LeRobot's LiberoEnv for one task (the evaluator's settings, eval_libero.py
_LiberoAdapter: obs pixels_agent_pos, both cameras at 360 x 360, control 20 Hz, hard reset, 10 settle steps), then
--resets times: reset to the init state (hard reset rebuilds the MuJoCo model and the renderer), --steps no-op
steps (LeRobot's dummy action [0, 0, 0, 0, 0, 0, -1], sim/integrations/libero.py:126-128) with the time of every
step. Recorded: reset and per-step seconds (mean, p50, p95, max), image shapes and statistics of both cameras,
check_success, the GL/EGL libraries mapped into the process (/proc/self/maps: libEGL_nvidia and libnvidia-* =
NVIDIA's EGL vendor; libEGL_mesa / libgallium = Mesa), the GL vendor/renderer strings when a context is current,
and two PNG frames per process. No policy, no network.

Usage (inside the container; README "Scripts"):
  python probe_render.py --out /out [--procs 1] [--task-id 0] [--init-state 0] [--steps 280] [--resets 2]
Writes /out/probe_render.json and /out/frames/p<k>_{agentview,wrist}_{reset,last}.png.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a8_common as C  # noqa: E402

GL_TOKENS = ("nvidia", "mesa", "osmesa", "gallium", "swrast", "llvmpipe", "libegl", "libgl", "libopengl", "glvnd")


def mapped_gl_libs() -> list[str]:
    maps = Path("/proc/self/maps").read_text().splitlines()
    libs = sorted({line.split()[-1] for line in maps if ".so" in line.split()[-1]})
    return [x for x in libs if any(t in x.rsplit("/", 1)[-1].lower() for t in GL_TOKENS)
            and not x.rsplit("/", 1)[-1].lower().startswith("libglib")]


def gl_strings() -> dict:
    out = {}
    try:
        from OpenGL import GL

        for name in ("GL_VENDOR", "GL_RENDERER", "GL_VERSION"):
            v = GL.glGetString(getattr(GL, name))
            out[name] = v.decode() if isinstance(v, bytes) else v
    except Exception as exc:                                                         # noqa: BLE001
        out["gl_error"] = f"{type(exc).__name__}: {exc}"[:200]
    try:
        from OpenGL import EGL

        dpy = EGL.eglGetCurrentDisplay()
        for name in ("EGL_VENDOR", "EGL_VERSION", "EGL_CLIENT_APIS"):
            v = EGL.eglQueryString(dpy, getattr(EGL, name))
            out[name] = v.decode() if isinstance(v, bytes) else v
    except Exception as exc:                                                         # noqa: BLE001
        out["egl_error"] = f"{type(exc).__name__}: {exc}"[:200]
    return out


def timing(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    pick = lambda q: s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]   # noqa: E731
    return {"n": len(s), "mean": round(sum(s) / len(s), 5), "p50": round(pick(0.5), 5), "p95": round(pick(0.95), 5),
            "max": round(s[-1], 5), "total": round(sum(s), 3)}


def image_stats(obs: dict) -> dict:
    out = {}
    for k, v in obs.items():
        if k.startswith("observation.images."):
            out[k] = {"shape": list(v.shape), "dtype": str(v.dtype), "mean": round(float(v.mean()), 3),
                      "std": round(float(v.std()), 3)}
    return out


def save_png(arr, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def probe_one(k: int, cfg: dict) -> dict:
    os.environ.update(cfg["env"])
    C.fix_environment()
    C.install_socket_guard()
    import numpy as np

    import eval_libero as E

    E.libero_setup()
    from libero.libero import benchmark
    from lerobot.sim.integrations.libero import get_libero_dummy_action

    res = {"proc": k, "pid": os.getpid()}
    t0 = time.perf_counter()
    suite = benchmark.get_benchmark_dict()[C.SUITE]()
    adapter = E._LiberoAdapter(suite, cfg["task_id"], cfg["settle_steps"], cfg["steps"])
    adapter.init_state = cfg["init_state"]
    adapter.seed = 0
    res["build_s"] = round(time.perf_counter() - t0, 3)
    res["task"] = adapter.env.task
    res["language"] = adapter.task_description
    action = np.asarray(get_libero_dummy_action(), dtype=np.float32)
    resets, steps_all = [], []
    for r in range(cfg["resets"]):
        t = time.perf_counter()
        obs = adapter.reset()
        rec = {"reset_s": round(time.perf_counter() - t, 3), "images_reset": image_stats(obs),
               "state_dim": int(obs["observation.state"].shape[-1])}
        if r == 0:
            res["gl_after_first_reset"] = gl_strings()
            for cam, name in (("observation.images.image", "agentview"), ("observation.images.image2", "wrist")):
                save_png(obs[cam], Path(cfg["out"]) / "frames" / f"p{k}_{name}_reset.png")
        times, success, terminated = [], False, False
        for _ in range(cfg["steps"]):
            t = time.perf_counter()
            obs, terminated, success = adapter.step(action)
            times.append(time.perf_counter() - t)
            if success or terminated:
                break
        rec.update(step_s=timing(times), n_steps=len(times), success=bool(success), terminated=bool(terminated),
                   images_last=image_stats(obs))
        if r == 0:
            for cam, name in (("observation.images.image", "agentview"), ("observation.images.image2", "wrist")):
                save_png(obs[cam], Path(cfg["out"]) / "frames" / f"p{k}_{name}_last.png")
        resets.append(rec)
        steps_all += times
    res["resets"] = resets
    res["step_s_all"] = timing(steps_all)
    libs = mapped_gl_libs()
    res["gl_libs_mapped"] = libs
    names = [x.rsplit("/", 1)[-1].lower() for x in libs]
    res["nvidia_egl_loaded"] = any(n.startswith("libegl_nvidia") for n in names)
    res["mesa_loaded"] = any("mesa" in n or "gallium" in n for n in names)
    res["peak_rss_mb"] = C.peak_rss_mb()
    res["blocked_connections"] = C.BLOCKED
    adapter.close()
    res["wall_s"] = round(time.perf_counter() - t0, 3)
    return res


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--procs", type=int, default=1)
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--init-state", type=int, default=0)
    p.add_argument("--steps", type=int, default=C.MAX_STEPS)
    p.add_argument("--resets", type=int, default=2)
    p.add_argument("--settle-steps", type=int, default=C.SETTLE_STEPS)
    args = p.parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    C.fix_environment()
    cfg = {"out": str(out), "task_id": args.task_id, "init_state": args.init_state, "steps": args.steps,
           "resets": args.resets, "settle_steps": args.settle_steps,
           "env": {k: os.environ[k] for k in os.environ if k.startswith(("HF_", "MUJOCO_", "PYOPENGL", "LIBERO_",
                                                                         "__EGL", "GALLIUM", "LIBGL"))}}
    t0 = time.time()
    if args.procs == 1:
        results = [probe_one(0, cfg)]
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(args.procs, mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(probe_one, range(args.procs), [cfg] * args.procs))
    doc = {"schema": "ptc_mm.libero_render_probe", "schema_version": "1.0", "probe": "P1",
           "settings": {k: v for k, v in cfg.items() if k != "env"} | {"procs": args.procs,
                                                                      "height": C.OBS_HEIGHT, "width": C.OBS_WIDTH},
           "env": {k: os.environ.get(k) for k in ("MUJOCO_GL", "PYOPENGL_PLATFORM", "MUJOCO_EGL_DEVICE_ID",
                                                   "NVIDIA_DRIVER_CAPABILITIES", "CUDA_VISIBLE_DEVICES",
                                                   "__EGL_VENDOR_LIBRARY_FILENAMES")},
           "egl_vendor_files": sorted(os.listdir("/usr/share/glvnd/egl_vendor.d"))
           if os.path.isdir("/usr/share/glvnd/egl_vendor.d") else None,
           "versions": C.versions(), "results": results,
           "nvidia_egl_loaded_all": all(r["nvidia_egl_loaded"] for r in results),
           "step_s_mean_all": round(sum(r["step_s_all"]["mean"] for r in results) / len(results), 5),
           "wall_s": round(time.time() - t0, 1), "script_sha256": C.sha256_file(Path(__file__).resolve()),
           "created_at": C.utc_now()}
    C.write_json(out / "probe_render.json", doc)
    print(json.dumps({k: doc[k] for k in ("nvidia_egl_loaded_all", "step_s_mean_all", "wall_s")}
                     | {"reset_s": [[x["reset_s"] for x in r["resets"]] for r in results],
                        "gl": results[0].get("gl_after_first_reset")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

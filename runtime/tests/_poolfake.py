"""Picklable stand-ins for eval_libero.run_pool's worker initializer and job function (spawn workers import this
module by name; multiprocessing passes the parent's sys.path to the child)."""

import os
import time
from pathlib import Path


def init_ok(cfg):
    return None


def init_crash(cfg):
    os._exit(3)


def init_hang(cfg):
    time.sleep(3600)                   # a worker stuck while starting (policy load, EGL)


def _hang(j):
    """Write this worker's pid (one line per attempt), then block as a hung EGL/MuJoCo call would."""
    with open(j["pid_file"], "a") as f:
        f.write(f"{os.getpid()}\n")
    time.sleep(3600)


def job(j):
    if j.get("crash"):
        os._exit(7)                    # a worker death (as a segfault in MuJoCo or EGL would be)
    if j.get("hang"):
        _hang(j)                       # hangs on every attempt
    if j.get("hang_once") and not Path(j["pid_file"]).exists():
        _hang(j)                       # hangs on the first attempt only
    if j.get("sleep"):
        time.sleep(j["sleep"])
    return {"job": j, "episode": {"outcome": "timeout"}, "error": None, "wall": 0.0, "pid": os.getpid()}

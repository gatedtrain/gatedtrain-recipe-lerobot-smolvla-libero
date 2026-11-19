#!/usr/bin/env python
"""Per-item LIBERO-Object evaluation of a SmolVLA cut, prototype area "VLA" (row 8 of the multimodal prototype scope; evaluator
`libero.lerobot_items@0.1`, path E2 of README section 4; prototype, unpromotable).

An item is one (task, init_state) pair of LIBERO-Object, id `libero_object/t<task_id>/s<init_state>` (MM-1
`item_meta {task_id, init_state}`); the gate items are init states 0-39 (400 items), the probe/pilot items init
states 40-49 (100 items) (the maintainers' area 8 (VLA) decisions). One episode per (item, eval seed),
reusing LeRobot 0.7.0's pieces (installed venv, file:line):

  env        lerobot.sim.integrations.libero.LiberoEnv (libero.py:143-436), one per task and worker, settings of
             LeRobot's LIBERO config (envs/configs.py:466-482): obs pixels_agent_pos 360 x 360, both cameras,
             control 20 Hz, relative control, hard reset, 10 no-op settle steps after reset (:384-388); the init
             state is chosen by `init_state_id` (:211, :381); max 280 steps for libero_object (:134-140, :224-229)
  observation  libero.canonical_observation (:49-58) then envs/utils.py prepare_canonical_observation (:134-144),
             task = the task's language (as LocalVectorEnv.call('task_description')), the env processors
             (make_env_pre_post_processors, envs/factory.py:38-62; identity for LIBERO, envs/configs.py:215-217)
  policy     SmolVLAPolicy.from_pretrained(<cut>/pretrained_model) and its saved processors through
             make_pre_post_processors with the device and rename overrides of lerobot-eval (lerobot_eval.py:870-881);
             select_action + transition_to_policy_action(policy_output_to_transition(...)) + postprocessor + env
             postprocessor, as rollout() (lerobot_eval.py:321-335)
  success    env.check_success() inside LiberoEnv.step (:410-421); the episode ends at the first success

Scoring (MM-4; README section 5): success 1; else 0 with outcome `timeout` (280 steps), `failed` (the simulator
reported done without success) or `nan_action` (a non-finite action; the episode stops before stepping). Any
exception from the simulator, the renderer or the policy step, a worker process that dies (message `worker_died:`),
and a job that exceeds the item wall-clock timeout while running alone (message `item_timeout:`; a job first
times out among others and is retried once alone) is an error `simulator_error`
(score null; errors are not scores, MM-4). Sub-metrics: steps_to_success (280 when none), n_policy_calls, steps,
wall_s, timeout, nan_action. (`timeout` is the episode's 280-step limit, a score of 0; `item_timeout` is the
evaluator's wall-clock guard, an error.)

Common random numbers: per (eval seed, item), torch is reseeded and the flow-matching noise comes from a CPU
generator seeded from (eval seed, item) (a8_common.NOISE_RULE); the env seed is derived from the same pair. So every
arm sees the same noise stream per item, whatever worker runs it.

Parallelism: --workers spawn processes (concurrent.futures, spawn context), each loading the policy on the device and
owning its LIBERO envs (one EGL context per open env, MUJOCO_EGL_DEVICE_ID; at most --envs-per-worker open, default
1, least recently used closed first; every reset is a hard reset, so this does not change an episode). The policy
step runs under torch.inference_mode, the policy in the dtypes LeRobot builds for the cut (as in training), and the
worker returns cached allocator blocks after each job (torch.cuda.empty_cache) and records its device memory in the
episode log. At most --workers jobs are in flight; when the pool breaks (a worker died), the jobs that were in
flight are rerun alone in new pools, and one that breaks a pool while alone is recorded as `simulator_error`
(worker_died). A job without a result --item-timeout-s (default 300) after a worker took it gets its pool killed
(SIGKILL; a ProcessPoolExecutor cannot replace one worker) and replaced: the job is retried once alone, the other
jobs in flight rerun with the same seeds, and a job that times out alone is `simulator_error` (item_timeout). So
the run always finishes and writes the arm. A pool that cannot start (broken, or no answer within 900 s) stops the
run. With --workers 0 (in process) there is no timeout.

Modes:
  python eval_libero.py --write-items
      writes items-gate/ (init states 0-39) and items-probe/ (40-49): items.jsonl + item_set_meta.json each, and
      ITEMS-MANIFEST.json (SHA-256 of every file)
  python eval_libero.py --checkpoint CUT --items gate|probe --out DIR --eval-seeds 11 [12] [--limit N]
                        [--per-task N] [--workers N] [--item-timeout-s S] [--envs-per-worker N]
                        [--device cuda] [--model-hash H] [--item-set-hash H]
      writes DIR/arm-success/{items.jsonl, eval_meta.json, env_version.json, episodes/<item>/r<k>.json},
      DIR/env_version.json, DIR/env-runtime.json, DIR/progress.jsonl, DIR/summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a8_common as C  # noqa: E402  (stdlib only)

SCRIPT = Path(__file__).resolve()
EVALUATOR_ID = "libero.lerobot_items@0.1"
ADAPTER = "lerobot_smolvla"
METRIC = "success"
SCORER_VERSION = "eval_libero@0.1"
ERROR_CLASSES = ("simulator_error",)
OUTCOMES = ("success", "timeout", "failed", "nan_action")
ITEM_SETS = {"gate": ("items-gate", C.GATE_STATES), "probe": ("items-probe", C.PROBE_STATES)}
ITEMS_MANIFEST = HERE / "ITEMS-MANIFEST.json"
CUT_MARKER = "cut.json"
DEFAULT_ITEM_TIMEOUT_S = 300.0      # an episode is <= 280 policy steps + 10 settle steps; 30-60 s with 8 workers
STARTUP_TIMEOUT_S = 900.0           # a new pool's first worker must load the policy and answer a no-op job


def item_id(task_id: int, init_state: int) -> str:
    return f"{C.SUITE}/t{task_id}/s{init_state}"


# ---------------------------------------------------------------------------------------------------------------
# Item sets
# ---------------------------------------------------------------------------------------------------------------

def libero_setup() -> None:
    """LIBERO reads its config from $LIBERO_CONFIG_PATH (libero/libero/__init__.py:5-8, :93-100) and its assets from
    the module global `_assets_path_cache` when set (:68-90); set before any env is built (README section 0.3)."""
    import libero.libero as ll

    ll._assets_path_cache = str(C.ASSETS)


def suite_tasks() -> list[dict]:
    """The libero_object tasks in benchmark order with their init-state files (the file LeRobot loads:
    get_libero_path('init_states') / problem_folder / init_states_file, libero.py:103-110)."""
    libero_setup()
    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()[C.SUITE]()
    root = Path(get_libero_path("init_states"))
    out = []
    for t in range(suite.n_tasks):
        task = suite.get_task(t)
        path = root / task.problem_folder / Path(task.init_states_file).name
        out.append({"task_id": t, "task": task.name, "language": task.language, "init_file": str(path),
                    "init_file_name": path.name, "init_file_sha256": C.sha256_file(path)})
    return out


def build_items(tasks: list[dict], states: list[int]) -> list[dict]:
    """Deterministic order: task id, then init state, both numeric."""
    rows = []
    for t in sorted(tasks, key=lambda r: r["task_id"]):
        for s in sorted(states):
            rows.append({"item": item_id(t["task_id"], s), "suite": C.SUITE, "task_id": t["task_id"],
                         "task": t["task"], "language": t["language"], "init_state": s,
                         "init_file": t["init_file_name"], "init_file_sha256": t["init_file_sha256"],
                         "row_index": s})
    return rows


def write_items(tasks: list[dict] | None = None, root: Path = HERE) -> dict:
    tasks = tasks if tasks is not None else suite_tasks()
    manifest = {"schema": "ptc_mm.libero_items_manifest", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
                "suite": C.SUITE, "n_tasks": len(tasks), "tasks": tasks, "sets": {}}
    for name, (dirname, states) in ITEM_SETS.items():
        d = root / dirname
        d.mkdir(exist_ok=True)
        rows = build_items(tasks, states)
        text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
        if (d / "items.jsonl").exists() and (d / "items.jsonl").read_text(encoding="utf-8") != text:
            raise SystemExit(f"{d / 'items.jsonl'} exists and differs; item sets are never regenerated in place")
        (d / "items.jsonl").write_text(text, encoding="utf-8")
        C.write_json(d / "item_set_meta.json", {
            "schema": "ptc_mm.libero_item_set", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
            "name": name, "suite": C.SUITE, "init_states": states, "n_items": len(rows),
            "item_id": "libero_object/t<task_id>/s<init_state>", "order": "task_id, then init_state (numeric)",
            "payload": ["suite", "task_id", "task", "language", "init_state", "init_file", "init_file_sha256",
                        "row_index"],
            "rule": "the maintainers' area 8 (VLA) decisions: init states 0-39 gate, 40-49 probe/pilot",
            "init_files": {t["init_file_name"]: t["init_file_sha256"] for t in tasks}})
        manifest["sets"][name] = {"dir": dirname, "n_items": len(rows),
                                  "files": {f: C.sha256_file(d / f) for f in ("items.jsonl", "item_set_meta.json")}}
    C.write_json(root / ITEMS_MANIFEST.name, manifest)
    return manifest


def read_items(items_dir: Path) -> list[dict]:
    rows = [json.loads(ln) for ln in (items_dir / "items.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    ids = [r["item"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{items_dir}: item ids are not unique")
    return rows


def select_items(rows: list[dict], limit: int | None, per_task: int | None) -> list[dict]:
    if per_task:
        seen: dict[int, int] = {}
        out = []
        for r in rows:
            if seen.get(r["task_id"], 0) < per_task:
                out.append(r)
                seen[r["task_id"]] = seen.get(r["task_id"], 0) + 1
        rows = out
    return rows[:limit] if limit else rows


# ---------------------------------------------------------------------------------------------------------------
# Episode (environment- and policy-agnostic; unit-tested with fakes)
# ---------------------------------------------------------------------------------------------------------------

def run_episode(env, act, max_steps: int, n_action_steps: int, keep_actions: bool = True) -> dict:
    """One episode. `env.reset()` returns the first (canonical) observation, `env.step(a)` returns (observation,
    terminated, is_success); `act(observation, step)` returns a 1-D float32 action. Mirrors lerobot_eval.rollout
    (:296-356) for one env with LocalVectorEnv's truncation at max_steps (local_vector_env.py:349-358)."""
    import numpy as np

    t0 = time.perf_counter()
    obs = env.reset()
    t_reset = time.perf_counter() - t0
    steps, outcome, success_step = 0, None, None
    actions = []
    t_policy = 0.0
    t_sim = 0.0
    while steps < max_steps:
        tp = time.perf_counter()
        a = np.asarray(act(obs, steps), dtype=np.float32).reshape(-1)
        t_policy += time.perf_counter() - tp
        if not np.all(np.isfinite(a)):
            outcome = "nan_action"
            break
        if keep_actions:
            actions.append(a.tolist())
        ts = time.perf_counter()
        obs, terminated, is_success = env.step(a)
        t_sim += time.perf_counter() - ts
        steps += 1
        if is_success:
            outcome, success_step = "success", steps
            break
        if terminated:
            outcome = "failed"
            break
    if outcome is None:
        outcome = "timeout"
    return {"outcome": outcome, "success": outcome == "success", "steps": steps, "success_step": success_step,
            "steps_to_success": success_step if success_step is not None else max_steps,
            "n_policy_calls": math.ceil(steps / n_action_steps) if steps else (1 if outcome == "nan_action" else 0),
            "actions": actions, "seconds": {"reset": round(t_reset, 3), "policy": round(t_policy, 3),
                                            "sim": round(t_sim, 3), "episode": round(time.perf_counter() - t0, 3)}}


def score_record(job: dict, ep: dict | None, error: dict | None, wall: float, eval_seed_index: int) -> dict:
    """MM-1 line (without output_ref) from an episode result or an error."""
    meta = {"task_id": job["task_id"], "init_state": job["init_state"]}
    if error is not None:
        return {"item": job["item"], "repeat": eval_seed_index, "seed": job["eval_seed"], "score": None,
                "error": error, "sub_metrics": {"wall_s": round(wall, 3)}, "item_meta": {**meta, "outcome": None}}
    sub = {"steps_to_success": ep["steps_to_success"], "n_policy_calls": ep["n_policy_calls"], "steps": ep["steps"],
           "wall_s": round(wall, 3), "timeout": int(ep["outcome"] == "timeout"),
           "nan_action": int(ep["outcome"] == "nan_action")}
    return {"item": job["item"], "repeat": eval_seed_index, "seed": job["eval_seed"],
            "score": 1 if ep["success"] else 0, "error": None, "sub_metrics": sub,
            "item_meta": {**meta, "outcome": ep["outcome"]}}


def action_summary(actions: list) -> dict:
    import numpy as np

    if not actions:
        return {"n": 0}
    a = np.asarray(actions, dtype=np.float64)
    return {"n": int(a.shape[0]), "dim": int(a.shape[1]), "min": a.min(0).round(6).tolist(),
            "max": a.max(0).round(6).tolist(), "mean": a.mean(0).round(6).tolist(),
            "abs_gt_1": int((np.abs(a) > 1).sum())}


# ---------------------------------------------------------------------------------------------------------------
# LeRobot-backed worker
# ---------------------------------------------------------------------------------------------------------------

_W: dict = {}


class _LiberoAdapter:
    """LeRobot's LiberoEnv for one task, observation canonicalized as LocalVectorEnv does."""

    def __init__(self, task_suite, task_id: int, settle_steps: int, max_steps: int):
        from lerobot.sim.integrations.libero import LiberoEnv, canonical_observation

        self.canonical = canonical_observation
        self.env = LiberoEnv(task_suite=task_suite, task_id=task_id, task_suite_name=C.SUITE,
                             episode_length=max_steps, camera_name=C.CAMERA_NAME, obs_type=C.OBS_TYPE,
                             observation_width=C.OBS_WIDTH, observation_height=C.OBS_HEIGHT,
                             init_states=C.INIT_STATES, episode_index=0, n_envs=1,
                             camera_name_mapping=dict(C.CAMERA_NAME_MAPPING), num_steps_wait=settle_steps,
                             control_freq=C.CONTROL_FREQ, control_mode=C.CONTROL_MODE, hard_reset=C.HARD_RESET)
        self.task_description = self.env.task_description
        self.init_state = 0
        self.seed = None

    def reset(self):
        self.env.init_state_id = self.init_state         # libero.py:211, :381 (stride 1 reset after reset)
        raw, _ = self.env.reset(seed=self.seed)
        return self.canonical(raw)

    def step(self, a):
        raw, _reward, terminated, _truncated, info = self.env.step(a)
        return self.canonical(raw), bool(terminated), bool(info.get("is_success", False))

    def close(self):
        self.env.close()


def _init_worker(cfg: dict) -> None:
    os.environ.update(cfg["env"])
    C.fix_environment()
    C.install_socket_guard()
    import torch

    torch.set_num_threads(cfg["torch_threads"])
    libero_setup()
    from lerobot.configs import PreTrainedConfig
    from lerobot.envs import make_env_pre_post_processors
    from lerobot.envs.configs import LiberoEnvConfig
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from libero.libero import benchmark

    pretrained = Path(cfg["pretrained"])
    pcfg = PreTrainedConfig.from_pretrained(pretrained)
    pcfg.pretrained_path = pretrained
    pcfg.device = cfg["device"]
    policy = SmolVLAPolicy.from_pretrained(pretrained, config=pcfg)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=pcfg, pretrained_path=pretrained,
        preprocessor_overrides={"device_processor": {"device": str(pcfg.device)},
                                "rename_observations_processor": {"rename_map": dict(C.RENAME)}})
    env_cfg = LiberoEnvConfig(task=C.SUITE, observation_height=C.OBS_HEIGHT, observation_width=C.OBS_WIDTH,
                              camera_name=C.CAMERA_NAME, control_mode=C.CONTROL_MODE, hard_reset=C.HARD_RESET)
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=pcfg)
    _W.update(cfg=cfg, policy=policy, pre=pre, post=post, env_pre=env_pre, env_post=env_post, envs={},
              suite=benchmark.get_benchmark_dict()[C.SUITE](), pid=os.getpid())


def _policy_act(adapter: _LiberoAdapter, gen, noise_shape):
    import torch
    from lerobot.envs.utils import prepare_canonical_observation
    from lerobot.processor import policy_output_to_transition, transition_to_policy_action
    from lerobot.utils.constants import ACTION

    policy, pre, post, env_pre, env_post = (_W[k] for k in ("policy", "pre", "post", "env_pre", "env_post"))

    def act(obs, step):
        # The whole step under inference_mode (it was the policy call only): the processors hold no trainable
        # parameters, so the values are unchanged; no autograd state is kept for the preprocessing tensors.
        with torch.inference_mode():
            batch = prepare_canonical_observation({k: v[None] for k, v in obs.items()})   # a batch of one env
            batch["task"] = [adapter.task_description]
            batch = env_pre(batch)
            batch = pre(batch)
            noise = torch.randn(noise_shape, generator=gen, dtype=torch.float32)
            action = transition_to_policy_action(policy_output_to_transition(policy.select_action(batch,
                                                                                                  noise=noise)))
            action = post(action)
            action = env_post({ACTION: action})[ACTION]
            return action.to("cpu").float().numpy()[0]

    return act


def _trim_envs(keep: int) -> None:
    """Close the least recently used task envs until at most `keep` stay open. Each open LiberoEnv holds a MuJoCo
    render context with its own EGL context on the device; every reset is a hard reset (model, sim and renderer
    rebuilt, seed and init state set by the job), so closing an env between episodes does not change an episode."""
    import gc

    envs = _W["envs"]
    closed = False
    while len(envs) > max(keep, 0):
        old = envs.pop(next(iter(envs)))
        try:
            old.close()
        except Exception:                                                             # noqa: BLE001
            pass
        closed = True
    if closed:
        gc.collect()                       # robosuite frees the GL context in MjRenderContext.__del__


def _device_memory(device: str) -> dict | None:
    """Device memory of this worker's torch allocator after the job (GB), and the device's free/total memory
    (all processes, EGL included). None on the CPU."""
    try:
        import torch

        if not str(device).startswith("cuda") or not torch.cuda.is_available():
            return None
        torch.cuda.empty_cache()           # return cached blocks, so EGL contexts of other workers can use them
        free, total = torch.cuda.mem_get_info()
        gb = 1e9
        return {"allocated_gb": round(torch.cuda.memory_allocated() / gb, 3),
                "reserved_gb": round(torch.cuda.memory_reserved() / gb, 3),
                "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / gb, 3),
                "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / gb, 3),
                "device_free_gb": round(free / gb, 3), "device_total_gb": round(total / gb, 3)}
    except Exception:                                                                 # noqa: BLE001
        return None


def run_job(job: dict) -> dict:
    """One (item, eval seed) in a worker. Returns {job, episode, error, wall, pid, device_memory}."""
    import traceback

    import torch

    cfg = _W["cfg"]
    t0 = time.perf_counter()
    ep, error = None, None
    try:
        envs = _W["envs"]
        adapter = envs.pop(job["task_id"], None)
        if adapter is None:
            _trim_envs(cfg.get("envs_per_worker", 1) - 1)
            adapter = _LiberoAdapter(_W["suite"], job["task_id"], cfg["settle_steps"], cfg["max_steps"])
        envs[job["task_id"]] = adapter                  # most recently used last
        adapter.init_state = job["init_state"]
        adapter.seed = job["env_seed"]
        torch.manual_seed(job["torch_seed"])
        gen = torch.Generator("cpu").manual_seed(job["noise_seed"])
        policy = _W["policy"]
        policy.reset()
        _W["pre"].reset()
        _W["post"].reset()
        shape = (1, policy.config.chunk_size, policy.config.max_action_dim)
        ep = run_episode(adapter, _policy_act(adapter, gen, shape), cfg["max_steps"], policy.config.n_action_steps)
    except Exception as exc:                                                         # noqa: BLE001
        error = {"class": "simulator_error", "message": f"{type(exc).__name__}: {exc}"[:300]}
        trace = traceback.format_exc()[-4000:]
        bad = _W["envs"].pop(job["task_id"], None)
        if bad is not None:
            try:
                bad.close()
            except Exception:                                                         # noqa: BLE001
                pass
        wall = time.perf_counter() - t0
        return {"job": job, "episode": None, "error": error, "trace": trace, "wall": wall, "pid": os.getpid(),
                "device_memory": _device_memory(cfg["device"])}
    wall = time.perf_counter() - t0
    return {"job": job, "episode": ep, "error": None, "wall": wall, "pid": os.getpid(),
            "device_memory": _device_memory(cfg["device"])}


# ---------------------------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------------------------

def check_cut(cut: Path) -> tuple[dict, dict]:
    """The cut's model files against cut.json, and its sampler.json against the deployment sampler rebuilt from its
    config.json (a8_common.deploy_sampler). Returns (cut.json, sampler)."""
    marker = json.loads((cut / CUT_MARKER).read_text(encoding="utf-8"))
    if marker.get("schema") != "ptc_mm.lerobot_cut":
        raise SystemExit(f"{cut / CUT_MARKER}: schema {marker.get('schema')!r} is not ptc_mm.lerobot_cut")
    for rel in marker["model_files"]:
        if C.sha256_file(cut / rel) != marker["files"][rel]:
            raise SystemExit(f"{cut / rel} differs from its SHA-256 in cut.json")
    config = json.loads((cut / C.PRETRAINED / "config.json").read_text(encoding="utf-8"))
    sampler = json.loads((cut / "sampler.json").read_text(encoding="utf-8"))
    if sampler != C.deploy_sampler(config):
        raise SystemExit("the cut's sampler.json is not this evaluator's deployment sampler for its config.json")
    norm = C.normalizer_record(cut / C.PRETRAINED)
    if norm["normalizer_sha256"] != marker["normalizer_sha256"]:
        raise SystemExit("the cut's normalizer files differ from cut.json normalizer_sha256 (MM-19)")
    return marker, sampler


def env_version_doc(tasks: list[dict], max_steps: int, settle_steps: int) -> dict:
    """MM-19 env_version (`{package, version, env_id, env_kwargs, assets_manifest_hash}` plus the init files)."""
    v = C.versions()
    assets = C.manifest_binding("libero-assets")
    return {"package": "hf-libero", "version": v["hf-libero"], "env_id": f"lerobot.sim.integrations.libero.LiberoEnv/"
            f"{C.SUITE}", "versions": {k: v[k] for k in ("hf-libero", "robosuite", "mujoco", "bddl", "lerobot")},
            "env_kwargs": {"suite": C.SUITE, "obs_type": C.OBS_TYPE, "observation_height": C.OBS_HEIGHT,
                           "observation_width": C.OBS_WIDTH, "camera_name": C.CAMERA_NAME,
                           "control_freq": C.CONTROL_FREQ, "control_mode": C.CONTROL_MODE,
                           "hard_reset": C.HARD_RESET, "init_states": C.INIT_STATES, "max_steps": max_steps,
                           "num_steps_wait": settle_steps},
            "nondefault": {k: v for k, v in (("max_steps", max_steps), ("num_steps_wait", settle_steps))
                           if v != {"max_steps": C.MAX_STEPS, "num_steps_wait": C.SETTLE_STEPS}[k]},
            "assets": {"repo": assets["repo"], "revision": assets["revision"]},
            "assets_manifest_hash": assets["files_sha256"],
            "init_files_sha256": {t["init_file_name"]: t["init_file_sha256"] for t in tasks},
            "success": "env.check_success() (the task's BDDL goal), any step; first success ends the episode"}


def make_jobs(items: list[dict], seeds: list[int]) -> list[dict]:
    jobs = []
    for r, seed in enumerate(seeds):
        for it in items:
            jobs.append({"item": it["item"], "task_id": it["task_id"], "init_state": it["init_state"],
                         "eval_seed": seed, "repeat": r,
                         "env_seed": C.derive_seed("libero-env", seed, it["item"]) % 2**31,
                         "torch_seed": C.derive_seed("libero-torch", seed, it["item"]),
                         "noise_seed": C.derive_seed("libero-noise", seed, it["item"])})
    return jobs


def _ping(_=None) -> int:
    return os.getpid()


_START_QUEUE = None


def _pool_init(init, cfg, start_queue) -> None:
    """Pool initializer: keep the start-mark queue, then the real initializer."""
    global _START_QUEUE
    _START_QUEUE = start_queue
    init(cfg)


def _run_marked(fn, key: int, job: dict):
    """Report (key, pid, start time) to the driver, then run the job: the item timeout counts from here, not from
    the submission (a fresh pool's workers may still be loading the policy), and the driver knows the pid."""
    if _START_QUEUE is not None:
        _START_QUEUE.put((key, os.getpid(), time.time()))
    return fn(job)


def _kill_pool(pool) -> None:
    """SIGKILL every worker process of a ProcessPoolExecutor (a hung worker may not react to anything else), then
    drop the pool. `_processes` ({pid: Process}) is private but present in CPython 3.8-3.13."""
    procs = list((getattr(pool, "_processes", None) or {}).values())
    for p in procs:
        try:
            p.kill()
        except Exception:                                                             # noqa: BLE001
            pass
    pool.shutdown(wait=False, cancel_futures=True)
    for p in procs:
        try:
            p.join(timeout=30)
        except Exception:                                                             # noqa: BLE001
            pass


def _error_result(job: dict, message: str, wall: float = 0.0, pid=None) -> dict:
    return {"job": job, "episode": None, "wall": wall, "pid": pid, "trace": None, "device_memory": None,
            "error": {"class": "simulator_error", "message": message}}


def run_pool(jobs: list[dict], worker_cfg: dict, workers: int, on_result, init=None, fn=None,
             item_timeout: float | None = DEFAULT_ITEM_TIMEOUT_S, startup_timeout: float = STARTUP_TIMEOUT_S) -> None:
    """Bounded in-flight scheduling over a spawn ProcessPoolExecutor; every job is passed to `on_result` exactly
    once.

    Start: each new pool first runs a no-op job; if that breaks the pool or does not return within
    `startup_timeout` s, the workers cannot start (policy load, EGL) and the run stops (an infrastructure failure,
    not an item's). A submitted job that no worker takes within `startup_timeout` s stops the run the same way.

    Worker death: when a pool breaks later, the jobs that were in flight become suspects and run alone, one at a
    time, in the next pools; a suspect that breaks a pool while alone is recorded as simulator_error (worker_died),
    the others run normally.

    Item timeout: a job without a result `item_timeout` s after a worker took it (None or 0: no limit) has its pool
    killed (SIGKILL to every worker; a ProcessPoolExecutor cannot replace a single worker, a dead worker breaks the
    whole pool) and a new pool started. The timed-out job is retried once alone; the other jobs that were in flight
    go back to the front of the queue (their results were lost with the pool; they rerun from the start with the
    same seeds). A job that times out while alone is recorded as simulator_error (item_timeout)."""
    import multiprocessing
    import queue as queue_mod
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    from concurrent.futures import TimeoutError as FutureTimeout
    from concurrent.futures.process import BrokenProcessPool

    init = init or _init_worker
    fn = fn or run_job
    limit = item_timeout if item_timeout and item_timeout > 0 else None
    poll = max(0.05, min(5.0, (limit or 50.0) / 10))
    queue = list(jobs)
    suspects: list[tuple[dict, str]] = []          # (job, why): "worker_died" or "item_timeout"
    ctx = multiprocessing.get_context("spawn")
    next_key = 0
    while queue or suspects:
        start_q = ctx.Queue()
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_pool_init,
                                   initargs=(init, worker_cfg, start_q))
        inflight: dict = {}                         # future -> job
        keys: dict = {}                             # future -> key
        submitted: dict = {}                        # key -> submission time
        started: dict = {}                          # key -> (pid, start time)
        isolated, why = False, None
        timed_out: list = []

        def submit(j):
            nonlocal next_key
            next_key += 1
            fut = pool.submit(_run_marked, fn, next_key, j)
            inflight[fut], keys[fut], submitted[next_key] = j, next_key, time.time()

        try:
            try:
                pool.submit(_ping).result(timeout=startup_timeout)
            except (BrokenProcessPool, FutureTimeout):
                _kill_pool(pool)
                raise SystemExit("the worker pool broke or hung while starting (policy load, EGL?); run with "
                                 "--workers 0 to see the exception") from None
            while queue or suspects or inflight:
                if isolated and inflight:           # a suspect is running alone: submit nothing
                    pass
                elif suspects:                      # isolation: one suspect alone
                    if not inflight:
                        j, why = suspects.pop(0)
                        submit(j)
                        isolated = True
                else:
                    isolated = False
                    while queue and len(inflight) < workers:
                        submit(queue.pop(0))
                done, _ = wait(list(inflight), timeout=poll, return_when=FIRST_COMPLETED)
                while True:
                    try:
                        k, pid, t = start_q.get_nowait()
                    except queue_mod.Empty:
                        break
                    started[k] = (pid, t)
                for fut in done:
                    j = inflight.pop(fut)
                    k = keys.pop(fut)
                    try:
                        res = fut.result()
                    except BrokenProcessPool:
                        inflight[fut], keys[fut] = j, k
                        raise
                    started.pop(k, None)
                    submitted.pop(k, None)
                    on_result(res)
                now = time.time()
                for fut, j in inflight.items():
                    k = keys[fut]
                    if k in started:
                        if limit is not None and now - started[k][1] > limit:
                            timed_out.append((fut, j, started[k][0], now - started[k][1]))
                    elif now - submitted[k] > startup_timeout:
                        _kill_pool(pool)
                        raise SystemExit(f"no worker took job {j.get('item')} within {startup_timeout:.0f} s "
                                         "(a worker hung while starting?)")
                if timed_out:
                    break
        except BrokenProcessPool:
            pool.shutdown(wait=False, cancel_futures=True)
            start_q.close()
            if isolated:
                for j in inflight.values():
                    on_result(_error_result(j, "worker_died: the worker process exited while running this job "
                                               "alone" + (" (after an item timeout)" if why == "item_timeout"
                                                          else "")))
            else:
                suspects.extend((j, "worker_died") for j in inflight.values())
            continue
        if timed_out:
            _kill_pool(pool)
            start_q.close()
            hung = {id(fut) for fut, *_ in timed_out}
            for fut, j, pid, elapsed in timed_out:
                print(f"item_timeout: {j.get('item')} r{j.get('repeat')} had no result after {elapsed:.0f} s "
                      f"(limit {limit:.0f} s, worker pid {pid}); pool killed, "
                      f"{'recorded as simulator_error' if isolated else 'retrying it alone'}", flush=True)
                if isolated:
                    note = (" (the first attempt also timed out with other jobs in flight)" if why == "item_timeout"
                            else " (it was running alone after a worker death)" if why == "worker_died" else "")
                    on_result(_error_result(j, f"item_timeout: no result within {limit:.0f} s while running "
                                               f"alone; the worker was killed{note}", wall=elapsed, pid=pid))
                else:
                    suspects.append((j, "item_timeout"))
            if not isolated:
                queue[:0] = [o for f, o in inflight.items() if id(f) not in hung]
            continue
        pool.shutdown(wait=True)
        start_q.close()


def stats(vals):
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if not n:
        return {"n": 0}
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else None
    return {"n": n, "mean": mean, "sd": sd}


def evaluate(args) -> int:
    t0 = time.time()
    cut = args.checkpoint.resolve()
    marker, sampler = check_cut(cut)
    items_dir = (HERE / ITEM_SETS[args.items][0]) if args.items in ITEM_SETS else Path(args.items)
    items = select_items(read_items(items_dir), args.limit, args.per_task)
    seeds = list(args.eval_seeds)
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    arm = out / f"arm-{METRIC}"
    (arm / "episodes").mkdir(parents=True, exist_ok=True)
    C.fix_environment()
    C.install_socket_guard()
    tasks = suite_tasks()
    by_task = {t["task_id"]: t for t in tasks}
    for it in items:
        if it["init_file_sha256"] != by_task[it["task_id"]]["init_file_sha256"]:
            raise SystemExit(f"{it['item']}: the init file differs from the item set's ({it['init_file']})")
    env_version = env_version_doc(tasks, args.max_steps, args.settle_steps)
    C.write_json(out / "env_version.json", env_version)
    C.write_json(arm / "env_version.json", env_version)
    worker_cfg = {"pretrained": str(cut / C.PRETRAINED), "device": args.device, "max_steps": args.max_steps,
                  "settle_steps": args.settle_steps, "torch_threads": args.torch_threads,
                  "envs_per_worker": args.envs_per_worker,
                  "env": {k: os.environ[k] for k in os.environ if k.startswith(("HF_", "MUJOCO_", "PYOPENGL",
                                                                                  "LIBERO_", "EGL", "__EGL",
                                                                                  "GALLIUM", "LIBGL"))}}
    jobs = make_jobs(items, seeds)
    rows: dict[tuple, dict] = {}
    device_mem: list = []

    def on_result(res):
        job = res["job"]
        rec = score_record(job, res["episode"], res["error"], res["wall"], job["repeat"])
        rel = f"episodes/{job['item']}/r{job['repeat']}.json"
        log = {"schema": "ptc_mm.libero_episode", "schema_version": "1.0", "item": job["item"],
               "repeat": job["repeat"], "eval_seed": job["eval_seed"], "task_id": job["task_id"],
               "init_state": job["init_state"], "env_seed": job["env_seed"], "torch_seed": job["torch_seed"],
               "noise_seed": job["noise_seed"], "error": res["error"], "trace": res.get("trace"),
               "worker_pid": res["pid"], "wall_s": round(res["wall"], 3),
               "device_memory": res.get("device_memory")}
        if res["episode"] is not None:
            ep = res["episode"]
            log.update({k: ep[k] for k in ("outcome", "success", "steps", "success_step", "steps_to_success",
                                           "n_policy_calls", "seconds")})
            log["actions_summary"] = action_summary(ep["actions"])
            log["actions"] = [[round(x, 6) for x in a] for a in ep["actions"]]
        path = arm / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        C.write_json(path, log)
        rec["output_ref"] = {"path": rel, "sha256": C.sha256_file(path)}
        rows[(job["repeat"], job["item"])] = rec
        device_mem.append(res.get("device_memory"))
        C.append_jsonl(out / "progress.jsonl", {**rec, "utc": C.utc_now()})
        print(f"{len(rows)}/{len(jobs)} {job['item']} r{job['repeat']}: "
              f"{rec['score'] if rec['error'] is None else rec['error']['class']} ({time.time() - t0:.0f} s)",
              flush=True)

    if args.workers == 0:                      # in-process (host tests): same worker code, no pool
        _init_worker(worker_cfg)
        for j in jobs:
            on_result(run_job(j))
    else:
        run_pool(jobs, worker_cfg, args.workers, on_result, item_timeout=args.item_timeout_s)
    order = {it["item"]: k for k, it in enumerate(items)}
    lines = [rows[k] for k in sorted(rows, key=lambda k: (k[0], order[k[1]]))]
    if len(lines) != len(jobs):
        raise SystemExit(f"{len(lines)} records for {len(jobs)} jobs")
    (arm / "items.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in lines), encoding="utf-8")

    code_sha = C.code_sha256(SCRIPT, C.COMMON_FILE)
    v = C.versions()
    meta = {
        "schema": "gatedtrain.item_eval", "schema_version": "1.0", "evaluator_id": EVALUATOR_ID, "adapter": ADAPTER,
        "metric": METRIC, "direction": "higher_is_better", "score_range": [0, 1],
        "checkpoint": {"model_hash": args.model_hash, "base_model_hash": None},
        "normalizer_sha256": marker["normalizer_sha256"], "env_version_sha256": C.doc_sha256(env_version),
        "item_set_hash": args.item_set_hash, "eval_seeds": seeds, "sampler": sampler,
        "sampler_hash": C.doc_sha256(sampler),
        "scorer": {"name": "libero_bddl_check_success",
                   "version": (f"{SCORER_VERSION}; success = LiberoEnv.step is_success (env.check_success(), BDDL "
                               f"goal) within {args.max_steps} steps after {args.settle_steps} settle steps; score 1 "
                               f"success, 0 timeout/failed/nan_action; simulator exceptions, worker death and the item "
                               f"wall-clock timeout ({args.item_timeout_s:g} s, retried once alone) are "
                               f"simulator_error; hf-libero {v['hf-libero']} robosuite {v['robosuite']} mujoco "
                               f"{v['mujoco']} lerobot {v['lerobot']}"),
                   "weights_sha256": None, "code_sha256": code_sha},
        "framework": {"name": "lerobot", "version": v["lerobot"]},
        "created_at": C.utc_now(),
    }
    C.write_json(arm / "eval_meta.json", meta)

    scored = [r for r in lines if r["error"] is None]
    per_task = {}
    for r in scored:
        per_task.setdefault(r["item_meta"]["task_id"], []).append(r["score"])
    by_item = {}
    for r in lines:
        by_item.setdefault(r["item"], []).append(r["score"])
    item_means = [None if any(x is None for x in xs) else sum(xs) / len(xs) for xs in by_item.values()]
    mean = stats(item_means).get("mean")
    errors = len(lines) - len(scored)
    outcomes = {o: sum(1 for r in scored if r["item_meta"]["outcome"] == o) for o in OUTCOMES}
    local_model_hash = marker["model_files_sha256"]
    summary = {
        "checkpoint": str(cut), "global_step": marker["global_step"], "profile": marker["profile"],
        "stage": marker["stage"], "items": args.items, "n_items": len(items), "eval_seeds": seeds,
        "episodes": len(lines), "errors": errors, "outcomes": outcomes,
        "success": stats([r["score"] for r in scored]),
        "per_task_success": {str(t): {"n": len(v), "mean": sum(v) / len(v)} for t, v in sorted(per_task.items())},
        "mm38": {"binary": True, "item_mean": mean, "n_items_scored": sum(1 for m in item_means if m is not None),
                 "degenerate": mean in (0, 1, 0.0, 1.0) if mean is not None else True, "zero_errors": errors == 0,
                 "rule": "binary metric: degenerate only at mean 0 or 1 (MM-38)"},
        "steps_to_success": stats([r["sub_metrics"]["steps_to_success"] for r in scored]),
        "wall_s_per_episode": stats([r["sub_metrics"]["wall_s"] for r in lines]),
        "seconds_wall": round(time.time() - t0, 1), "workers": args.workers, "device": args.device,
        "item_timeout_s": args.item_timeout_s if args.workers else None, "envs_per_worker": args.envs_per_worker,
        "error_reasons": {r: sum(1 for x in lines if x["error"] is not None
                                 and x["error"]["message"].startswith(r + ":"))
                          for r in ("item_timeout", "worker_died")},
        "device_memory_max": {k: max((m[k] for m in device_mem if m and m.get(k) is not None), default=None)
                              for k in ("peak_allocated_gb", "peak_reserved_gb")},
        "device_free_gb_min": min((m["device_free_gb"] for m in device_mem if m), default=None),
        "cut_model_files_sha256": local_model_hash, "normalizer_sha256": marker["normalizer_sha256"],
        "sampler_hash": meta["sampler_hash"], "env_version_sha256": meta["env_version_sha256"],
        "env_version_nondefault": env_version["nondefault"],
        "missing_hashes": [k for k, val in (("model_hash", args.model_hash), ("item_set_hash", args.item_set_hash))
                           if val is None],
        "blocked_connections": C.BLOCKED,
    }
    C.write_json(out / "summary.json", summary)
    C.write_json(out / "env-runtime.json", {
        "python": platform.python_version(), "executable": sys.executable,
        "image_cuda_version": os.environ.get("CUDA_VERSION"), "uv_lock_sha256": C.sha256_file(C.UV_LOCK),
        "script_sha256": C.sha256_file(SCRIPT), "common_sha256": C.sha256_file(C.COMMON_FILE),
        "code_sha256": code_sha, "versions": v, "checkpoint": str(cut),
        "items_dir": str(items_dir), "items_sha256": C.sha256_file(items_dir / "items.jsonl"),
        "env": {k: os.environ.get(k) for k in ("MUJOCO_GL", "PYOPENGL_PLATFORM", "MUJOCO_EGL_DEVICE_ID",
                                                "NVIDIA_DRIVER_CAPABILITIES", "CUDA_VISIBLE_DEVICES", "HF_HOME",
                                                "LIBERO_CONFIG_PATH")},
        "uid": os.getuid(), "gid": os.getgid(), "created_at": C.utc_now()})
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--write-items", action="store_true")
    p.add_argument("--checkpoint", type=Path, help="a cut directory (cut.json, sampler.json, pretrained_model/)")
    p.add_argument("--items", default="probe", help="gate | probe | an item set directory")
    p.add_argument("--out", type=Path)
    p.add_argument("--eval-seeds", type=int, nargs="+", default=[11])
    p.add_argument("--limit", type=int, help="first N items (after --per-task)")
    p.add_argument("--per-task", type=int, help="first N init states of each task (MM-38 smoke: 5)")
    p.add_argument("--workers", type=int, default=6, help="worker processes (0: in process, host tests)")
    p.add_argument("--item-timeout-s", type=float, default=DEFAULT_ITEM_TIMEOUT_S,
                   help="wall-clock limit per job in a worker (--workers >= 1; 0: none). On expiry the pool's "
                        "workers are killed, the job is retried once alone, then recorded as simulator_error "
                        "(item_timeout); the other jobs in flight rerun")
    p.add_argument("--envs-per-worker", type=int, default=1,
                   help="open LiberoEnv instances (one EGL context each) a worker keeps across tasks")
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model-hash")
    p.add_argument("--item-set-hash")
    p.add_argument("--max-steps", type=int, default=C.MAX_STEPS,
                   help="host tests only; a non-default value is recorded in env_version.json")
    p.add_argument("--settle-steps", type=int, default=C.SETTLE_STEPS,
                   help="host tests only; a non-default value is recorded in env_version.json")
    args = p.parse_args(argv)
    if not args.write_items:
        if args.checkpoint is None or args.out is None:
            p.error("evaluation needs --checkpoint and --out")
        if len(set(args.eval_seeds)) != len(args.eval_seeds):
            p.error("--eval-seeds must be distinct")
        for k in ("limit", "per_task", "max_steps"):
            if getattr(args, k) is not None and getattr(args, k) < 1:
                p.error(f"--{k.replace('_', '-')} must be positive")
        if args.workers < 0 or args.settle_steps < 0 or args.item_timeout_s < 0:
            p.error("--workers, --settle-steps and --item-timeout-s must be >= 0")
        if args.envs_per_worker < 1:
            p.error("--envs-per-worker must be positive")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.write_items:
        C.fix_environment()
        print(json.dumps(write_items(), indent=2, sort_keys=True))
        return 0
    return evaluate(args)


if __name__ == "__main__":
    sys.exit(main())

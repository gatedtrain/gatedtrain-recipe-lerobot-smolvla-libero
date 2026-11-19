#!/usr/bin/env python
"""CPU verification of the area "VLA" runtime (host, no GPU, no network): probe P0 of README.md.

Run with the area venv:  .venv/bin/python verify_cpu.py <step> [...]   (steps: config, env, dataset, policy, libero)

The script fixes its own environment before any library is imported: `CUDA_VISIBLE_DEVICES=""`,
`HF_HUB_OFFLINE=1`, `HF_HOME=hf-home/`, `HF_LEROBOT_HOME=data/`, `LIBERO_CONFIG_PATH=libero-config/`, token
variables removed (never read). A socket guard refuses every connection that is not to a local address, so a
step that tries to download fails loudly instead of fetching. Results go to `verify/<step>.json`.

  config   write libero-config/config.yaml (LIBERO otherwise calls input() at import)
  env      versions of the key packages -> env-freeze.txt header (the full freeze is appended by the caller)
  dataset  load lerobot/libero offline with video_backend=pyav, LIBERO-Object episodes 807-1260, decode frames
  policy   smolvla_base on CPU with the local SmolVLM2 backbone, LIBERO features, one forward/loss
  libero   LIBERO-Object benchmark, init states of task 0, one physics-only env reset/step (no renderer);
           with --render also an OffScreenRenderEnv (needs a MUJOCO_GL backend that works on the machine)
"""

from __future__ import annotations

import ipaddress
import json
import os
import resource
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "verify"
SMOLVLA = HERE / "models" / "smolvla_base"
SMOLVLM = HERE / "models" / "SmolVLM2-500M-Video-Instruct"
DATA_ROOT = HERE / "data" / "lerobot" / "libero"
ASSETS = HERE / "assets" / "libero-assets"
LIBERO_CONFIG = HERE / "libero-config"
OBJECT_EPISODES = list(range(807, 1261))
RENAME = {"observation.images.image": "observation.images.camera1",
          "observation.images.image2": "observation.images.camera2"}

for _name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH"):
    os.environ.pop(_name, None)                 # removed, never read
os.environ.update({
    "CUDA_VISIBLE_DEVICES": "",
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "HF_HOME": str(HERE / "hf-home"),
    "HF_LEROBOT_HOME": str(HERE / "data"),
    "LIBERO_CONFIG_PATH": str(LIBERO_CONFIG),
    "TOKENIZERS_PARALLELISM": "false",
    "WANDB_MODE": "disabled",
})

_real_connect = socket.socket.connect
BLOCKED: list[str] = []


def _guarded_connect(self, address):
    host = address[0] if isinstance(address, tuple) else address
    try:
        local = isinstance(address, (str, bytes)) or ipaddress.ip_address(host).is_loopback
    except ValueError:
        local = False
    if not local:
        BLOCKED.append(str(address))
        raise OSError(f"network blocked by verify_cpu.py: {address}")
    return _real_connect(self, address)


socket.socket.connect = _guarded_connect


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def save(step: str, result: dict) -> None:
    OUT.mkdir(exist_ok=True)
    result["blocked_connections"] = BLOCKED
    result["peak_rss_gb"] = round(peak_rss_gb(), 2)
    (OUT / f"{step}.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


def step_config() -> dict:
    import importlib.util

    spec = importlib.util.find_spec("libero")   # locate the package without importing libero.libero
    root = Path(next(iter(spec.submodule_search_locations))) / "libero"
    cfg = {
        "benchmark_root": str(root),
        "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"),
        "datasets": str(HERE / "libero-datasets-unused"),
        "assets": str(ASSETS),
    }
    LIBERO_CONFIG.mkdir(exist_ok=True)
    text = "".join(f"{k}: {v}\n" for k, v in cfg.items())
    (LIBERO_CONFIG / "config.yaml").write_text(text, encoding="utf-8")
    return {"config_yaml": str(LIBERO_CONFIG / "config.yaml"), "content": cfg}


def step_env() -> dict:
    from importlib.metadata import version

    import torch

    names = ["lerobot", "torch", "torchvision", "transformers", "accelerate", "datasets", "huggingface-hub", "av",
             "torchcodec", "numpy", "mujoco", "robosuite", "robomimic", "hf-libero", "bddl", "gymnasium",
             "hf-egl-probe", "egl-probe", "opencv-python", "opencv-python-headless", "trackio", "wandb"]
    vers = {n: version(n) for n in names}
    import lerobot

    return {"python": sys.version.split()[0], "versions": vers, "lerobot_module": lerobot.__file__,
            "torch_cuda_build": torch.version.cuda, "cuda_available_with_CUDA_VISIBLE_DEVICES_empty":
            torch.cuda.is_available()}


def _dataset(episodes=OBJECT_EPISODES, delta_timestamps=None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset("lerobot/libero", root=DATA_ROOT, episodes=episodes, video_backend="pyav",
                          delta_timestamps=delta_timestamps,
                          revision="a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4")


def step_dataset() -> dict:
    import torch

    t0 = time.time()
    ds = _dataset()
    t_load = time.time() - t0
    res = {"load_s": round(t_load, 2), "num_frames": ds.num_frames, "num_episodes": ds.num_episodes,
           "fps": ds.fps, "video_backend": ds._video_backend,
           "features": {k: {"dtype": v["dtype"], "shape": list(v["shape"])} for k, v in ds.features.items()},
           "items": []}
    for i in (0, 1, ds.num_frames // 2, ds.num_frames - 1):
        t0 = time.time()
        item = ds[i]
        dt = time.time() - t0
        row = {"i": i, "get_s": round(dt, 3)}
        for k, v in item.items():
            if isinstance(v, torch.Tensor):
                row[k] = {"shape": list(v.shape), "dtype": str(v.dtype)}
                if v.numel() == 1:
                    row[k]["value"] = v.item()
                elif k.startswith("observation.images"):
                    row[k].update(min=round(v.min().item(), 4), max=round(v.max().item(), 4),
                                  mean=round(v.float().mean().item(), 4))
            else:
                row[k] = v
        res["items"].append(row)
    eps = sorted({int(ds[i]["episode_index"]) for i in (0, ds.num_frames - 1)})
    res["first_last_episode"] = eps
    res["stats_keys"] = sorted(ds.meta.stats)
    res["stats_state_mean"] = [round(float(x), 4) for x in ds.meta.stats["observation.state"]["mean"]]
    return res


def step_policy() -> dict:
    import torch

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    torch.manual_seed(0)
    res: dict = {}
    cfg = PreTrainedConfig.from_pretrained(SMOLVLA)
    res["base_config"] = {"vlm_model_name": cfg.vlm_model_name, "load_vlm_weights": cfg.load_vlm_weights,
                          "input_features": sorted(cfg.input_features), "chunk_size": cfg.chunk_size,
                          "n_action_steps": cfg.n_action_steps, "train_expert_only": cfg.train_expert_only,
                          "freeze_vision_encoder": cfg.freeze_vision_encoder}
    cfg.pretrained_path = SMOLVLA
    cfg.vlm_model_name = str(SMOLVLM)           # local backbone instead of the repo id
    cfg.device = "cpu"
    cfg.input_features = {}                     # = --policy.input_features=null
    cfg.output_features = {}                    # = --policy.output_features=null

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata("lerobot/libero", root=DATA_ROOT,
                                  revision="a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4")
    t0 = time.time()
    policy = make_policy(cfg, ds_meta=meta, rename_map=RENAME)
    res["policy_load_s"] = round(time.time() - t0, 1)
    res["rss_after_load_gb"] = round(peak_rss_gb(), 2)
    res["policy_features"] = {
        "input": {k: list(v.shape) for k, v in policy.config.input_features.items()},
        "output": {k: list(v.shape) for k, v in policy.config.output_features.items()}}
    n_all = sum(p.numel() for p in policy.parameters())
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    res["params"] = {"total": n_all, "trainable": n_train,
                     "dtypes": sorted({str(p.dtype) for p in policy.parameters()})}

    from lerobot.processor.rename_processor import rename_stats

    stats = rename_stats(meta.stats, RENAME)
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=SMOLVLA, dataset_stats=stats,
        preprocessor_overrides={
            "device_processor": {"device": "cpu"},
            "normalizer_processor": {"features": {**policy.config.input_features, **policy.config.output_features},
                                     "norm_map": policy.config.normalization_mapping, "stats": stats},
            "rename_observations_processor": {"rename_map": RENAME},
            "tokenizer_processor": {"tokenizer_name": str(SMOLVLM)},
        },
        postprocessor_overrides={"unnormalizer_processor": {"features": policy.config.output_features,
                                                            "norm_map": policy.config.normalization_mapping,
                                                            "stats": stats}})
    delta = resolve_delta_timestamps(policy.config, meta, RENAME)
    res["delta_timestamps_keys"] = {k: len(v) for k, v in (delta or {}).items()}
    ds = _dataset(episodes=[807, 1000], delta_timestamps=delta)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    res["raw_batch"] = {k: list(v.shape) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    res["raw_batch_ids"] = {k: batch[k].tolist() for k in ("index", "episode_index", "frame_index")}
    res["task"] = batch["task"]
    batch = pre(batch)
    res["pre_batch"] = {k: list(v.shape) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    policy.train()
    t0 = time.time()
    loss, out = policy.forward(batch)
    res["forward_s"] = round(time.time() - t0, 2)
    t0 = time.time()
    loss.backward()
    res["backward_s"] = round(time.time() - t0, 2)
    res["loss"] = float(loss.detach())
    res["loss_finite"] = bool(torch.isfinite(loss))
    res["forward_output_keys"] = sorted(out) if isinstance(out, dict) else str(type(out))
    policy.eval()
    policy.reset()
    with torch.no_grad():
        t0 = time.time()
        one = {k: (v[:1] if isinstance(v, (torch.Tensor, list)) else v) for k, v in batch.items()}
        act = policy.select_action(one)
        res["select_action_s"] = round(time.time() - t0, 2)
    act = post(act)
    res["action_shape"] = list(act.shape)
    return res


def step_libero(render: bool) -> dict:
    import numpy as np

    import libero.libero as ll
    from libero.libero import benchmark

    ll._assets_path_cache = str(ASSETS)         # LIBERO would otherwise look in ~/.cache/libero/assets
    res = {"assets_path": ll.get_assets_path(), "config_file": ll.config_file}
    bench = benchmark.get_benchmark_dict()["libero_object"]()
    res["n_tasks"] = bench.n_tasks
    res["tasks"] = [bench.get_task(i).name for i in range(bench.n_tasks)]
    init = bench.get_task_init_states(0)        # LIBERO's own torch.load of its packaged init file
    res["init_states_task0"] = {"type": type(init).__name__, "shape": list(np.shape(init)),
                                "dtype": str(getattr(init, "dtype", ""))}
    task = bench.get_task(0)
    bddl = bench.get_task_bddl_file_path(0)
    res["task0"] = {"name": task.name, "language": task.language, "bddl": bddl}

    from libero.libero.envs.env_wrapper import ControlEnv

    t0 = time.time()
    env = ControlEnv(bddl_file_name=bddl, use_camera_obs=False, has_renderer=False,
                     has_offscreen_renderer=False, control_freq=20, hard_reset=True)
    env.seed(0)
    env.reset()
    obs = env.set_init_state(init[0])
    res["physics_env_build_reset_s"] = round(time.time() - t0, 2)
    t0 = time.time()
    for _ in range(10):
        obs, reward, done, info = env.step(np.zeros(7))
    res["physics_10_steps_s"] = round(time.time() - t0, 3)
    res["physics_step"] = {"obs_keys": sorted(obs)[:12], "n_obs_keys": len(obs), "reward": float(reward),
                           "done": bool(done), "check_success": bool(env.check_success()),
                           "eef_pos": [round(float(x), 4) for x in obs["robot0_eef_pos"]]}
    env.close()
    if render:
        from libero.libero.envs import OffScreenRenderEnv

        t0 = time.time()
        renv = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        renv.seed(0)
        renv.reset()
        robs = renv.set_init_state(init[0])
        robs, _, _, _ = renv.step(np.zeros(7))
        img = robs["agentview_image"]
        res["render"] = {"MUJOCO_GL": os.environ.get("MUJOCO_GL"), "build_reset_step_s": round(time.time() - t0, 2),
                         "agentview_shape": list(img.shape), "agentview_mean": round(float(img.mean()), 3),
                         "wrist_mean": round(float(robs["robot0_eye_in_hand_image"].mean()), 3)}
        # Which GL/EGL implementation the process actually mapped (evidence that no NVIDIA library was used).
        maps = Path("/proc/self/maps").read_text().splitlines()
        libs = sorted({line.split()[-1].rsplit("/", 1)[-1] for line in maps if line.endswith(".so") or ".so." in line})
        res["render"]["gl_libs_mapped"] = [x for x in libs if any(
            t in x.lower() for t in ("nvidia", "mesa", "osmesa", "gallium", "swrast", "llvmpipe", "libegl", "libgl"))]
        renv.close()
    return res


def main(argv: list[str]) -> int:
    step = argv[0]
    if step == "config":
        save("config", step_config())
    elif step == "env":
        save("env", step_env())
    elif step == "dataset":
        save("dataset", step_dataset())
    elif step == "policy":
        save("policy", step_policy())
    elif step == "libero":
        save("libero-render" if "--render" in argv else "libero", step_libero("--render" in argv))
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

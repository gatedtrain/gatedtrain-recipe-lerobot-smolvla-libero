#!/usr/bin/env python
"""Launcher around LeRobot 0.7.0 `lerobot-train` (`lerobot.scripts.lerobot_train:main`) for SmolVLA SFT on
LIBERO-Object, prototype area "VLA" (row 8 of the multimodal prototype scope; prototype, unpromotable).

No LeRobot source is edited (MM-27). The launcher builds the `lerobot-train` command line, sets `sys.argv` to it and
calls `lerobot_train.main()` in this process, after wrapping five module globals of `lerobot.scripts.lerobot_train`
(installed venv, file:line below):

  make_pre_post_processors   (imported at lerobot_train.py:84, called at :709) gets a `tokenizer_processor`
                             override naming the local SmolVLM2 directory: the saved `policy_preprocessor.json` of
                             smolvla_base names the Hub repo id, and `lerobot-train` overrides only device,
                             normalizer and rename (:671-701), so the tokenizer would otherwise be fetched (README
                             section 0.4, P0 step 3b).
  make_dataloaders           (:325, called at :734): records the sampler length (frames per epoch) for the
                             epoch column of samples.jsonl (EpisodeAwareSampler, datasets/sampler.py:142-159).
  _preprocess_dataset_batch  (:167-178, called at :861 for training batches, at :916 only when eval_steps > 0,
                             which stays 0): appends one line per step to samples.jsonl (step, epoch, per sample
                             `index`, `episode_index`, `frame_index`), flushed and fsynced before the original runs,
                             so before `update_policy` (:864) applies the step (MM-26, MM-27).
  update_policy              (:181-322, called at :864): per-step timing, loss, peak device memory
                             (update_policy resets the peak counters itself, :230-231) and RSS -> memory.jsonl.
  update_last_checkpoint     (imported at :60, called at :951 after `save_checkpoint` returned): after the
                             `last` link points at the new step directory (the step is then complete, README
                             section 2 "Completion marker"), the cut `cuts/checkpoint-<step>/` is built from hard
                             links to `checkpoints/<step>/pretrained_model/` and `training_state/`, plus
                             `sampler.json` (the deployment sampler, a8_common.deploy_sampler), `base_ref.json`,
                             and `cut.json` written last (schema ptc_mm.lerobot_cut), then renamed into place.

Outputs in --out (the run's only read-write mount):
  env-freeze.txt     the area venv's freeze (copy of env-freeze.txt; compared with the running distributions)
  env-runtime.json   interpreter, image CUDA version, lock and code SHA-256s, versions, freeze check
  run_args.json      every setting, the lerobot-train argv, sampler and its hash, bindings, base cut (stage 2)
  samples.jsonl      observed consumption log, one line per step (written before the step's update)
  memory.jsonl       per step (all of the first 20, then every --memory-every): seconds, loss, peak memory, RSS
  cuts.jsonl         one line per cut written
  attempts.jsonl     one line per start and finish of this command (a rerun after a preemption restarts)
  lerobot/           LeRobot's output_dir (checkpoints/<step>/..., `last`)
  cuts/checkpoint-<step>/   the cuts; `cut.json` written last

Profiles (README "Scripts"): probe (P2 memory probe, 200 steps, no checkpoint), base (stage 1 base ladder from
smolvla_base: 3,000 steps, a cut every 250), smoke (stage 2 continuation from a base cut, 250 steps), full (stage 2
continuation, length --steps, a cut every 1,000). Explicit flags override the profile.

Usage (inside the container; README "Scripts" has the docker commands):
  python train_lerobot_logged.py --profile base --out /out [--batch-size 32] [--full-ft] [--seed S]
  python train_lerobot_logged.py --profile smoke --policy-path /inputs/base-cut --out /out
  python train_lerobot_logged.py --profile full --steps 6000 --policy-path /inputs/base-cut --out /out
  python train_lerobot_logged.py ... --dry-run          # print the settings and the lerobot-train argv only
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a8_common as C  # noqa: E402  (stdlib only)

SCRIPT = Path(__file__).resolve()
LEROBOT_DIR = "lerobot"              # LeRobot's output_dir inside --out; must not exist before the run (train.py:334)
CUTS_DIR = "cuts"
CUT_MARKER = "cut.json"
CUT_SCHEMA = "ptc_mm.lerobot_cut"

COMMON = {
    "batch_size": 32,
    "learning_rate": None,           # None = the SmolVLA preset optimizer_lr 1e-4 (configuration_smolvla.py:73)
    "warmup_steps": None,            # None = preset 1,000, auto-scaled when steps < decay steps (optim/schedulers.py:146-154)
    "decay_steps": None,             # None = preset 30,000 (configuration_smolvla.py:80-82)
    "train_expert_only": True,       # smolvla_base default (config.json) and the paper's recipe (README section 1)
    "freeze_vision_encoder": True,
    "n_action_steps": 50,            # checkpoint default (README section 10: unless a probe shows a large loss vs 10)
    "num_workers": 8,                # DataLoader workers (AV1 decode with PyAV); LeRobot default 4
    "seed": 1000,                    # LeRobot default (configs/train.py `seed: 1000`)
    "log_freq": 50,
    "use_imagenet_stats": False,     # as the lerobot/smolvla_libero recipe (meta/lerobot_smolvla_libero/train_config.json)
    "mixed_precision": "no",         # LeRobot default (configs/accelerator.py AcceleratorConfig); SmolVLA keeps its VLM in bf16
}
PROFILES = {
    "probe": {"stage": "probe", "steps": 200, "save_freq": 0, "save_checkpoint": False},
    "base": {"stage": "base", "steps": 3000, "save_freq": 250, "save_checkpoint": True},
    "smoke": {"stage": "continuation", "steps": 250, "save_freq": 250, "save_checkpoint": True},
    "full": {"stage": "continuation", "steps": None, "save_freq": 1000, "save_checkpoint": True},
}


# ---------------------------------------------------------------------------------------------------------------
# Settings and the lerobot-train command line
# ---------------------------------------------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--profile", choices=sorted(PROFILES), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--policy-path", type=Path,
                   help="stage 1: models/smolvla_base (default); stage 2: a base cut directory (holding cut.json and "
                        "pretrained_model/)")
    p.add_argument("--steps", type=int)
    p.add_argument("--save-freq", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float, help="peak learning rate (policy.optimizer_lr; default the preset 1e-4)")
    p.add_argument("--warmup-steps", type=int)
    p.add_argument("--decay-steps", type=int)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--train-expert-only", dest="full_ft", action="store_false", default=None,
                   help="train the action expert only (default)")
    g.add_argument("--full-ft", dest="full_ft", action="store_true",
                   help="full fine-tune: train_expert_only=false, freeze_vision_encoder=false (smolvla_libero recipe)")
    p.add_argument("--n-action-steps", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--device", default="cuda", help="policy.device (cpu for the host tests only)")
    p.add_argument("--memory-every", type=int, default=10, help="memory.jsonl: every step up to 20, then every N")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def resolve_settings(args) -> dict:
    s = {**COMMON, **PROFILES[args.profile], "profile": args.profile}
    for key, val in (("steps", args.steps), ("save_freq", args.save_freq), ("batch_size", args.batch_size),
                     ("learning_rate", args.lr), ("warmup_steps", args.warmup_steps),
                     ("decay_steps", args.decay_steps), ("n_action_steps", args.n_action_steps),
                     ("num_workers", args.num_workers), ("seed", args.seed)):
        if val is not None:
            s[key] = val
    if args.full_ft is not None:
        s["train_expert_only"] = not args.full_ft
        s["freeze_vision_encoder"] = not args.full_ft
    if s["steps"] is None:
        raise SystemExit(f"--profile {args.profile} needs --steps (the continuation length is a parameter)")
    for key in ("steps", "batch_size", "n_action_steps", "seed"):
        if not (isinstance(s[key], int) and s[key] >= 1):
            raise SystemExit(f"{key} must be a positive integer, not {s[key]!r}")
    if s["save_freq"] < 0 or s["num_workers"] < 0:
        raise SystemExit("save_freq and num_workers must be >= 0")
    if not 1 <= s["n_action_steps"] <= 50:
        raise SystemExit("n_action_steps must lie in 1..chunk_size 50 (configuration_smolvla.py:115-119)")
    s["device"] = args.device
    s["full_ft"] = not s["train_expert_only"]
    # The save rule (common/train_utils.py:124-132): every save_freq steps and always after the last step.
    s["cut_steps"] = (sorted({k for k in range(1, s["steps"] + 1)
                              if (s["save_freq"] > 0 and k % s["save_freq"] == 0) or k == s["steps"]})
                      if s["save_checkpoint"] else [])
    return s


def resolve_policy_path(args, s: dict) -> tuple[Path, Path | None]:
    """(pretrained directory for --policy.path, base cut directory or None)."""
    path = (args.policy_path or C.SMOLVLA).resolve()
    if s["stage"] == "continuation":
        cut = path.parent if path.name == C.PRETRAINED else path
        if not (cut / CUT_MARKER).is_file():
            raise SystemExit(f"--profile {s['profile']} continues from a base cut: {cut} holds no {CUT_MARKER}")
        return cut / C.PRETRAINED, cut
    if (path / CUT_MARKER).is_file() or (path.parent / CUT_MARKER).is_file():
        raise SystemExit(f"--profile {s['profile']} starts from smolvla_base, not from a cut ({path})")
    return path, None


def build_lerobot_argv(s: dict, policy_dir: Path, output_dir: Path) -> list[str]:
    """The `lerobot-train` arguments (draccus syntax; parsed by configs/parser.py wrap(), :414-473)."""
    argv = [
        f"--policy.path={policy_dir}",
        # local backbone (config field; smolvlm_with_expert.py loads config/processor from it)
        f"--policy.vlm_model_name={C.SMOLVLM}",
        # P0 step 3b: the loaded policy's 500 tensors equal those of load_vlm_weights=true; the 2 GB SmolVLM2 weights
        # file is not read
        "--policy.load_vlm_weights=false",
        # refilled from the dataset with the rename map (policies/factory.py:338-341; docs/source/peft_training.mdx)
        "--policy.input_features=null",
        "--policy.output_features=null",
        f"--policy.device={s['device']}",
        "--policy.push_to_hub=false",
        f"--policy.n_action_steps={s['n_action_steps']}",
        f"--policy.train_expert_only={str(s['train_expert_only']).lower()}",
        f"--policy.freeze_vision_encoder={str(s['freeze_vision_encoder']).lower()}",
    ]
    if s["learning_rate"] is not None:
        argv.append(f"--policy.optimizer_lr={s['learning_rate']}")
    if s["warmup_steps"] is not None:
        argv.append(f"--policy.scheduler_warmup_steps={s['warmup_steps']}")
    if s["decay_steps"] is not None:
        argv.append(f"--policy.scheduler_decay_steps={s['decay_steps']}")
    argv += [
        f"--dataset.repo_id={C.DATASET_REPO}",
        f"--dataset.root={C.DATA_ROOT}",
        f"--dataset.revision={C.DATASET_REVISION}",
        "--dataset.episodes=[" + ",".join(str(e) for e in C.OBJECT_EPISODES) + "]",
        "--dataset.video_backend=pyav",
        f"--dataset.use_imagenet_stats={str(s['use_imagenet_stats']).lower()}",
        "--rename_map=" + json.dumps(C.RENAME, separators=(",", ":")),
        f"--output_dir={output_dir}",
        f"--job_name=a8_{s['profile']}",
        f"--seed={s['seed']}",
        f"--batch_size={s['batch_size']}",
        f"--steps={s['steps']}",
        f"--save_freq={s['save_freq']}",
        f"--save_checkpoint={str(s['save_checkpoint']).lower()}",
        f"--log_freq={s['log_freq']}",
        f"--num_workers={s['num_workers']}",
        # accelerator.mixed_precision stays at its default "no" (configs/accelerator.py:350); not passed on the CLI
        # because draccus reads the bare value `no` as the boolean False (AcceleratorConfig then refuses it).
        "--env_eval_freq=0",
        "--eval_steps=0",
        "--trackio.enable=false",
    ]
    return argv


# ---------------------------------------------------------------------------------------------------------------
# Pre-step (MM-33) and run record
# ---------------------------------------------------------------------------------------------------------------

def prepare_out(out: Path) -> int:
    """As areas 3 and 5: a non-empty --out is a rerun after a preemption; earlier contents move to
    `preempted-attempt-<k>/` (never deleted). Returns the attempt number."""
    out.mkdir(parents=True, exist_ok=True)
    existing = [p for p in out.iterdir() if not p.name.startswith("preempted-attempt-")]
    previous = sorted(p for p in out.iterdir() if p.name.startswith("preempted-attempt-"))
    attempt = len(previous) + 1
    if existing:
        attempt = len(previous) + 2
        dest = out / f"preempted-attempt-{attempt - 1}"
        dest.mkdir()
        for p in existing:
            p.rename(dest / p.name)
        if (dest / "attempts.jsonl").exists():
            shutil.copyfile(dest / "attempts.jsonl", out / "attempts.jsonl")
    C.append_jsonl(out / "attempts.jsonl", {"attempt": attempt, "started_at": C.utc_now(), "argv": sys.argv,
                                            "pid": os.getpid()})
    return attempt


def _norm(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def freeze_check() -> dict:
    """The area's env-freeze.txt (name==version lines) against the distributions of the running interpreter."""
    want = {}
    for line in C.ENV_FREEZE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "==" in line:
            n, v = line.split("==", 1)
            want[_norm(n)] = v.strip()
    have = {}
    for dist in importlib.metadata.distributions():
        n = dist.metadata["Name"]
        if n:
            have[_norm(n)] = dist.version
    def same(a, b):
        if a == b:
            return True
        try:                                       # PEP 440 equality (e.g. 0.1.1.post2209072238 == 0.1.1-2209072238)
            from packaging.version import Version

            return a is not None and b is not None and Version(a) == Version(b)
        except Exception:                          # noqa: BLE001
            return False

    diff = sorted(f"{n}: freeze {want.get(n)}, runtime {have.get(n)}" for n in set(want) | set(have)
                  if not same(want.get(n), have.get(n)))
    return {"matches": not diff, "differences": diff[:50], "n_freeze": len(want), "n_runtime": len(have)}


def write_runtime(out: Path, code: dict) -> dict:
    shutil.copyfile(C.ENV_FREEZE, out / "env-freeze.txt")
    runtime = {
        "python": platform.python_version(), "executable": sys.executable,
        "image_cuda_version": os.environ.get("CUDA_VERSION"),
        "uv_lock": str(C.UV_LOCK), "uv_lock_sha256": C.sha256_file(C.UV_LOCK),
        "env_freeze_sha256": C.sha256_file(out / "env-freeze.txt"), "env_freeze_check": freeze_check(),
        **code, "versions": C.versions(), "uid": os.getuid(), "gid": os.getgid(), "cpu_count": os.cpu_count(),
        "env": {k: os.environ.get(k) for k in ("HF_HOME", "HF_LEROBOT_HOME", "HF_HUB_OFFLINE", "LIBERO_CONFIG_PATH",
                                                "HOME", "XDG_CACHE_HOME", "NVIDIA_DRIVER_CAPABILITIES",
                                                "CUDA_VISIBLE_DEVICES", "MUJOCO_GL", "PYOPENGL_PLATFORM",
                                                "MUJOCO_EGL_DEVICE_ID", "TOKENIZERS_PARALLELISM")},
        "created_at": C.utc_now(),
    }
    C.write_json(out / "env-runtime.json", runtime)
    return runtime


def verify_cut(cut: Path, full: bool) -> dict:
    """A cut's cut.json file hashes (model files always; every file with `full`). Returns cut.json."""
    marker = json.loads((cut / CUT_MARKER).read_text(encoding="utf-8"))
    if marker.get("schema") != CUT_SCHEMA:
        raise SystemExit(f"{cut / CUT_MARKER}: schema {marker.get('schema')!r} is not {CUT_SCHEMA}")
    names = marker["files"] if full else marker["model_files"]
    for rel in names:
        if C.sha256_file(cut / rel) != marker["files"][rel]:
            raise SystemExit(f"{cut / rel} differs from its SHA-256 in {cut / CUT_MARKER}")
    on_disk = set(C.tree_files(cut)) - {CUT_MARKER}
    if on_disk != set(marker["files"]):
        raise SystemExit(f"{cut}: files on disk differ from {CUT_MARKER} ({sorted(on_disk ^ set(marker['files']))[:5]})")
    return marker


def base_binding() -> dict:
    base = C.manifest_binding("smolvla_base")
    vlm = C.manifest_binding("SmolVLM2-500M-Video-Instruct")
    data = C.manifest_binding("libero")
    return {"base": base, "vlm": vlm, "data": data}


# ---------------------------------------------------------------------------------------------------------------
# Hooks (wrappers around lerobot_train module globals; no LeRobot edit)
# ---------------------------------------------------------------------------------------------------------------

class RunState:
    def __init__(self, out: Path, s: dict, meta: dict):
        self.out = out
        self.s = s
        self.meta = meta                   # base_ref, sampler extras, code hashes, init_checkpoint
        self.step = 0                      # the step whose batch was last preprocessed (1-based)
        self.epoch = 0
        self.in_epoch = 0
        self.samples_per_epoch = None
        self.consumed = 0
        self.consumed_at = {0: 0}
        self.last_t = None
        self.peak = {"allocated_gb": 0.0, "reserved_gb": 0.0}
        self.cuts: list[int] = []


def make_preprocess_hook(state: RunState, orig):
    """Wraps lerobot_train._preprocess_dataset_batch (:167-178): one samples.jsonl line per training step, before
    the original runs (hence before update_policy, :864)."""

    def _preprocess_dataset_batch(batch, camera_keys, rename_map, preprocessor):
        cols = {}
        for k in ("index", "episode_index", "frame_index"):
            v = batch[k]
            cols[k] = [int(x) for x in (v.reshape(-1).tolist() if hasattr(v, "reshape") else v)]
        n = len(cols["index"])
        if not (len(cols["episode_index"]) == len(cols["frame_index"]) == n):
            raise RuntimeError("index, episode_index and frame_index differ in length")
        state.step += 1
        now = time.perf_counter()
        line = {"step": state.step, "epoch": state.epoch, "n": n,
                "samples": [{"index": i, "episode_index": e, "frame_index": f}
                            for i, e, f in zip(cols["index"], cols["episode_index"], cols["frame_index"])]}
        C.append_jsonl(state.out / "samples.jsonl", line)
        state.consumed += n
        state.consumed_at[state.step] = state.consumed
        if state.samples_per_epoch:
            state.in_epoch += n
            if state.in_epoch >= state.samples_per_epoch:   # drop_last=False: the epoch's last batch is short
                state.epoch += 1
                state.in_epoch -= state.samples_per_epoch
        state.step_started = now
        return orig(batch, camera_keys, rename_map, preprocessor)

    return _preprocess_dataset_batch


def make_dataloaders_hook(state: RunState, orig):
    def make_dataloaders(cfg, dataset, eval_dataset, step, parallel_dims):
        dl, edl = orig(cfg, dataset, eval_dataset, step, parallel_dims)
        sampler = getattr(dl, "sampler", None)
        state.samples_per_epoch = len(sampler) if sampler is not None else len(dataset)
        state.dataset_frames = int(dataset.num_frames)
        state.dataset_episodes = int(dataset.num_episodes)
        return dl, edl

    return make_dataloaders


def make_processors_hook(orig):
    def make_pre_post_processors(*args, **kwargs):
        pre = dict(kwargs.get("preprocessor_overrides") or {})
        pre["tokenizer_processor"] = {**pre.get("tokenizer_processor", {}), "tokenizer_name": str(C.SMOLVLM)}
        kwargs["preprocessor_overrides"] = pre
        return orig(*args, **kwargs)

    return make_pre_post_processors


def make_update_hook(state: RunState, orig):
    import torch

    def update_policy(train_metrics, policy, batch, optimizer, grad_clip_norm, accelerator, *a, **kw):
        t0 = time.perf_counter()
        result = orig(train_metrics, policy, batch, optimizer, grad_clip_norm, accelerator, *a, **kw)
        t1 = time.perf_counter()
        step = state.step
        rec = {"step": step, "update_s": round(t1 - t0, 4),
               "step_s": round(t1 - state.last_t, 4) if state.last_t is not None else None}
        state.last_t = t1
        try:
            m = train_metrics.metrics
            rec.update(loss=m["loss"].val, grad_norm=m["grad_norm"].val, lr=m["lr"].val)
        except (AttributeError, KeyError):
            pass
        if torch.cuda.is_available():
            alloc = torch.cuda.max_memory_allocated() / 2**30       # since the reset at lerobot_train.py:230-231
            reserved = torch.cuda.max_memory_reserved() / 2**30
            rec.update(peak_allocated_gb=round(alloc, 3), peak_reserved_gb=round(reserved, 3))
            state.peak["allocated_gb"] = max(state.peak["allocated_gb"], round(alloc, 3))
            state.peak["reserved_gb"] = max(state.peak["reserved_gb"], round(reserved, 3))
        rec.update(rss_now_mb=C.rss_now_mb(), peak_rss_mb=C.peak_rss_mb(), utc=C.utc_now())
        if step <= 20 or step % max(1, state.s["memory_every"]) == 0 or step == state.s["steps"]:
            C.append_jsonl(state.out / "memory.jsonl", rec)
        return result

    return update_policy


def _fsync_dir(d: Path) -> None:
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copyfile(src, dst)
        return "copy"


def write_cut(step_dir: Path, step: int, state: RunState) -> Path:
    """cuts/checkpoint-<step>/ from LeRobot's checkpoints/<step>/ (hard links; copies when linking fails), plus
    sampler.json, base_ref.json, and cut.json last; built in cuts/.checkpoint-<step>.partial/ and renamed."""
    cuts = state.out / CUTS_DIR
    cuts.mkdir(exist_ok=True)
    final = cuts / f"checkpoint-{step}"
    partial = cuts / f".checkpoint-{step}.partial"
    if final.exists():
        raise RuntimeError(f"{final} exists already")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir()
    how = set()
    for sub in (C.PRETRAINED, C.TRAINING_STATE):
        src = step_dir / sub
        if not src.is_dir():
            raise RuntimeError(f"{src} is missing")
        for rel in C.tree_files(src):
            how.add(_link_or_copy(src / rel, partial / sub / rel))
    config = json.loads((partial / C.PRETRAINED / "config.json").read_text(encoding="utf-8"))
    sampler = C.deploy_sampler(config)
    C.write_json(partial / "sampler.json", sampler)
    C.write_json(partial / "base_ref.json", state.meta["base_ref"])
    files = {}
    for rel in C.tree_files(partial):
        with open(partial / rel, "rb") as fh:
            os.fsync(fh.fileno())
        files[rel] = C.sha256_file(partial / rel)
    model_files = C.model_file_names(sorted(files))
    norm = C.normalizer_record(partial / C.PRETRAINED)
    training_step = json.loads((partial / C.TRAINING_STATE / "training_step.json").read_text(encoding="utf-8"))
    if int(training_step["step"]) != step:
        raise RuntimeError(f"{step_dir}: training_step.json says step {training_step['step']}, not {step}")
    marker = {
        "schema": CUT_SCHEMA, "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
        "global_step": step, "last_step_included": step, "samples_consumed": state.consumed_at.get(step),
        "batch_size": state.s["batch_size"], "files": files, "model_files": model_files,
        "resume_files": sorted(set(files) - set(model_files) - {"base_ref.json"}),
        "model_files_sha256": C.doc_sha256({k: files[k] for k in model_files}),
        "normalizer_sha256": norm["normalizer_sha256"], "normalizer": norm,
        "sampler_sha256": C.doc_sha256(sampler), "deployable": model_files,
        "seed": state.s["seed"], "profile": state.s["profile"], "stage": state.s["stage"],
        "init_checkpoint": state.meta["base_ref"].get("init_checkpoint"),
        "lerobot_step_dir": f"{LEROBOT_DIR}/checkpoints/{step_dir.name}", "file_origin": sorted(how),
        "script_sha256": state.meta["script_sha256"], "common_sha256": state.meta["common_sha256"],
        "consumption_log": "samples.jsonl", "created_at": C.utc_now(),
    }
    with open(partial / CUT_MARKER, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(marker, indent=2, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    _fsync_dir(partial)
    os.rename(partial, final)
    _fsync_dir(cuts)
    C.append_jsonl(state.out / "cuts.jsonl", {"step": step, "cut": f"{CUTS_DIR}/{final.name}",
                                               "samples_consumed": marker["samples_consumed"], "utc": C.utc_now()})
    state.cuts.append(step)
    return final


def make_last_checkpoint_hook(state: RunState, orig):
    def update_last_checkpoint(checkpoint_dir):
        orig(checkpoint_dir)
        write_cut(Path(checkpoint_dir), int(Path(checkpoint_dir).name), state)

    return update_last_checkpoint


def install_hooks(lt, state: RunState) -> None:
    lt._preprocess_dataset_batch = make_preprocess_hook(state, lt._preprocess_dataset_batch)
    lt.make_dataloaders = make_dataloaders_hook(state, lt.make_dataloaders)
    lt.make_pre_post_processors = make_processors_hook(lt.make_pre_post_processors)
    lt.update_policy = make_update_hook(state, lt.update_policy)
    lt.update_last_checkpoint = make_last_checkpoint_hook(state, lt.update_last_checkpoint)


# ---------------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    args = parse_args(argv)
    s = resolve_settings(args)
    s["memory_every"] = args.memory_every
    policy_dir, base_cut = resolve_policy_path(args, s)
    out = args.out.resolve()
    lerobot_argv = build_lerobot_argv(s, policy_dir, out / LEROBOT_DIR)
    if args.dry_run:
        print(json.dumps({"settings": s, "policy_dir": str(policy_dir), "base_cut": str(base_cut) if base_cut else None,
                          "lerobot_argv": lerobot_argv}, indent=2, sort_keys=True))
        return 0
    C.fix_environment()
    C.install_socket_guard()
    attempt = prepare_out(out)
    code = {"script_sha256": C.sha256_file(SCRIPT), "common_sha256": C.sha256_file(C.COMMON_FILE),
            "code_sha256": C.code_sha256(SCRIPT, C.COMMON_FILE)}
    runtime = write_runtime(out, code)
    t_start = time.time()

    b = base_binding()
    init_checkpoint = None
    if base_cut is not None:
        marker = verify_cut(base_cut, full=False)
        bref = json.loads((base_cut / "base_ref.json").read_text(encoding="utf-8"))
        if bref["base_model"]["revision"] != b["base"]["revision"] or \
                bref["base_model"]["files_sha256"] != b["base"]["files_sha256"]:
            raise SystemExit(f"the base cut {base_cut} names another smolvla_base than models/smolvla_base")
        init_checkpoint = {"cut_dir": str(base_cut), "cut_json_sha256": C.sha256_file(base_cut / CUT_MARKER),
                           "global_step": marker["global_step"], "model_files_sha256": marker["model_files_sha256"],
                           "normalizer_sha256": marker["normalizer_sha256"], "profile": marker["profile"],
                           "stage": marker["stage"], "seed": marker["seed"]}
    base_ref = {"base_model": C.binding_summary(b["base"]) | {"files": b["base"]["files"]},
                "vlm_backbone": C.binding_summary(b["vlm"]),
                "dataset": C.binding_summary(b["data"]) | {"episodes": [C.OBJECT_EPISODES[0], C.OBJECT_EPISODES[-1]]},
                "init_checkpoint": init_checkpoint}
    base_cfg = json.loads((policy_dir / "config.json").read_text(encoding="utf-8"))
    # the deployment sampler the cuts will carry (n_action_steps from the CLI, the rest from the checkpoint config)
    expected_sampler = C.deploy_sampler({**base_cfg, "n_action_steps": s["n_action_steps"]})
    state = RunState(out, s, {"base_ref": base_ref, **code})

    run_args = {
        "schema": "ptc_mm.lerobot_run_args", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
        "profile": s["profile"], "stage": s["stage"], "attempt": attempt, "settings": s, "seed": s["seed"],
        "seed_derivation": ("lerobot-train --seed: set_seed(seed) (lerobot_train.py:574-575) and the data order "
                            "EpisodeAwareSampler(shuffle=True, seed) = a pure function of (seed, epoch) "
                            "(lerobot_train.py:362-372, datasets/sampler.py:128-153)"),
        "framework": {"name": "lerobot", "version": importlib.metadata.version("lerobot"),
                      "entry": "lerobot.scripts.lerobot_train:main (in process, sys.argv set)"},
        "lerobot_argv": lerobot_argv, "lerobot_output_dir": f"{LEROBOT_DIR}",
        "policy_path": str(policy_dir), "base_checkpoint": str(policy_dir) if base_cut else None,
        "base_cut": init_checkpoint, "base_ref": {k: (C.binding_summary(v) if isinstance(v, dict) and "files" in v
                                                      else v) for k, v in base_ref.items()},
        "dataset": {"repo": C.DATASET_REPO, "revision": C.DATASET_REVISION, "root": str(C.DATA_ROOT),
                    "episodes": f"{C.OBJECT_EPISODES[0]}..{C.OBJECT_EPISODES[-1]} ({len(C.OBJECT_EPISODES)})",
                    "video_backend": "pyav", "rename_map": C.RENAME,
                    "normalizer_stats": "dataset.meta.stats of lerobot/libero (whole dataset, README section 3), "
                                        "renamed; a fresh --policy.path run replaces the checkpoint's stats with "
                                        "them (lerobot_train.py:666-692)"},
        "tokenizer_override": {"tokenizer_processor.tokenizer_name": str(C.SMOLVLM),
                               "why": "policy_preprocessor.json names the Hub repo id; offline training needs the "
                                      "local directory (README section 0.4)"},
        "sampler": expected_sampler, "sampler_sha256": C.doc_sha256(expected_sampler),
        "consumption_log": "samples.jsonl",
        "consumption_semantics": ("one line per training step, written and fsynced before the step's update: step "
                                  "(1-based, LeRobot's step counter after the update), epoch (0-based, counted from "
                                  "the samples consumed and the sampler length), n and per sample the anchor frame's "
                                  "index, episode_index, frame_index; each sample also loads the next chunk_size - 1 "
                                  "actions of the same episode (delta_timestamps). A cut at step k includes steps "
                                  "1..k (cut.json last_step_included, samples_consumed)."),
        "cut_steps": s["cut_steps"], "cuts_dir": CUTS_DIR,
        **code, "env_freeze_matches_runtime": runtime["env_freeze_check"]["matches"],
        "created_at": C.utc_now(),
    }
    C.write_json(out / "run_args.json", run_args)

    import torch  # noqa: F401  (after the environment is fixed)
    from lerobot.scripts import lerobot_train as lt

    install_hooks(lt, state)
    old_argv = sys.argv
    sys.argv = ["lerobot-train"] + lerobot_argv          # parser.wrap() and TrainPipelineConfig.validate read sys.argv
    try:
        lt.main()
    finally:
        sys.argv = old_argv
    if state.cuts != s["cut_steps"]:
        raise SystemExit(f"cuts written {state.cuts}, expected {s['cut_steps']}")
    if state.step != s["steps"]:
        raise SystemExit(f"{state.step} steps logged, expected {s['steps']}")
    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    C.append_jsonl(out / "attempts.jsonl", {
        "attempt": attempt, "finished_at": C.utc_now(), "steps": state.step, "samples_consumed": state.consumed,
        "epochs_started": state.epoch + (1 if state.in_epoch else 0), "samples_per_epoch": state.samples_per_epoch,
        "cuts": state.cuts, "peak_device_memory": state.peak, "peak_rss_mb": C.peak_rss_mb(), "gpu": gpu,
        "blocked_connections": C.BLOCKED, "wall_seconds": round(time.time() - t_start, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""LeRobot SmolVLA SFT adapter (area "VLA": LeRobot 0.7.0 `lerobot-train` of `lerobot/smolvla_base`, expert only, on the
LIBERO-Object part of `lerobot/libero`, evaluated in the LIBERO simulator; MULTIMODAL-SPEC.md section 6 row 8 and the row
8 deviations, the maintainers' decision record (not published) "Area 8 (VLA) decisions", "Area 8 probes P1/P2", "Area 8 base cut";
tests/README.md section 11.26, EXT-233 to EXT-247).

MM-12: a pure-Python class that imports no framework, torch, simulator or pickle loader. It reads JSON, JSONL, the text
`env-freeze.txt`, safetensors headers and (with pyarrow, which gatedtrain already uses for its own tables) four integer
columns of the dataset's parquet files, and hashes everything else.

Run layout, read from the recipe's runtime code `runtime/` (`train_lerobot_logged.py`
sha256 `93bf86f6...`, `a8_common.py` `ce29aa22...`, `eval_libero.py` `7846982d...`; README sections 2, 4 to 6 and 11; read
2026-11-15, re-pinned for ptc-hub 0.2.0 and again for gatedtrain 0.3.0) and verified against the GPU outputs `base-1/` (the stage-1 ladder: 3,000 steps, cuts every 250) and
`base-1-eval2-<step>/` (its probe evaluations):

- two stages, two launches. Stage 1 (`--profile base`) trains from `models/smolvla_base`; stage 2 (`--profile smoke|full
  --policy-path /inputs/base-cut`) is a new `lerobot-train` run (new optimizer and warm-up, step counter from 1) from a
  stage-1 cut mounted read-only, and records it: `run_args.json` `base_checkpoint` (`/inputs/base-cut/pretrained_model`,
  the `--policy.path`), `base_cut` (the cut's `cut.json` SHA-256, step, model files, normalizer, profile, stage, seed),
  and every stage-2 cut's `cut.json` and `base_ref.json` `init_checkpoint` = that record. The spec's base is the stage-1
  cut; ext-collect gates stage 2's last cut against it (EXT-236), and the baseline arm evaluates that cut;
- `run_args.json` (schema `ptc_mm.lerobot_run_args`): `profile`, `stage` (`base` | `continuation`), `attempt`,
  `settings` (with `cut_steps`), `seed`, `lerobot_argv` (the exact `lerobot-train` command line, rebuilt here from the
  settings), `policy_path`, `base_checkpoint`, `base_cut`, `base_ref` (smolvla_base, SmolVLM2 and dataset bindings from
  `MANIFEST.json`), `dataset`, `sampler` and `sampler_sha256`, `script_sha256`, `common_sha256`;
- `samples.jsonl` (MM-26, MM-27): one line per training step, written and fsynced before the step's update by the
  launcher's wrapper of `lerobot_train._preprocess_dataset_batch`: `{step, epoch, n, samples: [{index, episode_index,
  frame_index}]}`, the anchor frames of the batch (each sample also loads the next 49 actions of its episode, disclosed).
  LeRobot's `EpisodeAwareSampler` shuffles every frame of the selected episodes once per epoch and the DataLoader keeps
  the short last batch (`drop_last` false): N frames at batch B give ceil(N / B) steps per epoch;
- cuts `cuts/checkpoint-<step>/` (schema `ptc_mm.lerobot_cut`): hard links to LeRobot's `checkpoints/<step>/
  pretrained_model/` and `training_state/`, `sampler.json` (the deployment sampler), `base_ref.json`, then `cut.json`
  last (`files` with every SHA-256, `deployable` = `model_files`, `resume_files`, `model_files_sha256`,
  `normalizer_sha256`, `sampler_sha256`, `samples_consumed`, `last_step_included`), built in `.checkpoint-<step>.partial/`
  and renamed; `attempts.jsonl` (start and finish lines), `cuts.jsonl`, `env-runtime.json`, `env-freeze.txt`.

Identity (MM-17, MM-19): full weights. A cut's model hash is the manifest hash of its ten deployable files (registry
`components`); `normalizer_sha256` is the area's definition, the SHA-256 of the compact sorted JSON {normalizer: sha256,
unnormalizer: sha256} of the two processor state files the processor descriptions name. The training data of the spec is a
JSONL frame export of the LIBERO-Object frames (one shard per dataset data file, `frames_meta.json`), which the adapter
re-derives from the dataset's parquet files through the run's read-only mounts (EXT-237).
"""

from __future__ import annotations

import hashlib
import json
import math
import posixpath
import re
from pathlib import Path
from typing import Any

from gatedtrain.canonical import canonical_json, entries_hash, file_sha256, manifest_entries, manifest_hash, sha256_bytes, text_lines
from gatedtrain.errors import Problem
from gatedtrain.ext.registry import EVALUATOR_ROWS, TRAINER_ROWS
from gatedtrain.ext.sdk import (bound_mounts as _bound_mounts, compact_sha256, container_file, derive_seed, is_int as _int,
                        plain_file as _plain_file, read_json as _read_json, read_jsonl, read_run_args, resolve_option,
                        ro_table, runtime_script_problems, safetensors_problem, sha_or_none as _sha_or_none, within)

from .rows import PYTHON_DIR, RUNTIME as AREA8

NAME = "lerobot_smolvla"
ROW = TRAINER_ROWS[NAME]
EVALUATOR_ID = ROW["evaluators"][0]
# AREA8 (rows.RUNTIME): the runtime directory (this repository's runtime/ or GATEDTRAIN_RECIPE_LEROBOT_SMOLVLA_LIBERO_RUNTIME),
# mounted read-only at its own path; PYTHON_DIR: the directory of the uv-managed CPython its venv links to (None
# without a venv).
TRAIN_SCRIPT = "train_lerobot_logged.py"
EVAL_SCRIPT = "eval_libero.py"
SCRIPT_PATHS = {TRAIN_SCRIPT: f"{AREA8}/{TRAIN_SCRIPT}", EVAL_SCRIPT: f"{AREA8}/{EVAL_SCRIPT}"}
BASE_CUT_CONTAINER = "/inputs/base-cut"                      # stage 2's --policy-path (README section 11.3)
EVAL_CUT_CONTAINER = "/inputs/cut"                           # eval_libero.py --checkpoint
# The runtime image `gatedtrain-a8-runtime:1` (built as `ptc-a8-runtime:1` for the published run; pinned by ID) (decisions "Area 8 probes P1/P2", not published) and the area venv's lock (README section 0.2).
IMAGE = "sha256:136745e51640ee7bf92760d3dc3f8764187f3c409e4f2c928ce4c39707cd5242"
VENV_LOCK_SHA256 = "sha256:2a87da026100743dccb92e5c084e64941ff6346cedf2f40ae774e13dfac4b9a6"
FRAMEWORK_PACKAGE = "lerobot"
LEROBOT_VERSION = "0.7.0"
# Pinned downloads (MANIFEST.json; README section 0.1): repository, 40-hex revision.
SMOLVLA_REPO, SMOLVLA_REVISION = "lerobot/smolvla_base", "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6"
SMOLVLM_REPO, SMOLVLM_REVISION = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "4cde9d208e19519c36917aa170fffc7e381aa0ad"
DATASET_REPO, DATASET_REVISION = "lerobot/libero", "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
SMOLVLA_DIR = f"{AREA8}/models/smolvla_base"
VLM_DIR = f"{AREA8}/models/SmolVLM2-500M-Video-Instruct"
DATASET_ROOT = f"{AREA8}/data/lerobot/libero"
OBJECT_EPISODES = (807, 1260)                                 # README section 3: LIBERO-Object, 454 episodes
RENAME = {"observation.images.image": "observation.images.camera1",
          "observation.images.image2": "observation.images.camera2"}
OUT = "/out"
LEROBOT_DIR = "lerobot"
CUTS_DIR = "cuts"
CUT_MARKER = "cut.json"
CUT_SCHEMA = "ptc_mm.lerobot_cut"
RUN_ARGS_SCHEMA = "ptc_mm.lerobot_run_args"
CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")
PRETRAINED = "pretrained_model"
COMPONENTS = tuple(ROW["components"])                          # cut.json deployable = model_files (EXT-235)
RESUME_FILES = ("pretrained_model/train_config.json", "training_state/optimizer_param_groups.json",
                "training_state/optimizer_state.safetensors", "training_state/rng_state.safetensors",
                "training_state/scheduler_state.json", "training_state/training_step.json")
BASE_REF = "base_ref.json"
SAMPLER_FILE = "sampler.json"
CUT_FILES = tuple(sorted(COMPONENTS + RESUME_FILES + (BASE_REF,)))
WEIGHTS = "pretrained_model/model.safetensors"
# The processor state files (a8_common.normalizer_record: the `state_file` of the normalizer_processor step of
# policy_preprocessor.json and of the unnormalizer_processor step of policy_postprocessor.json).
NORMALIZER_ROLES = (("normalizer", "policy_preprocessor.json", "normalizer_processor"),
                    ("unnormalizer", "policy_postprocessor.json", "unnormalizer_processor"))
PINS = tuple(ROW["pins"])
FREEZE_FILE = "env-freeze.txt"
RUNTIME_FILE = "env-runtime.json"
SAMPLES = "samples.jsonl"
ATTEMPTS = "attempts.jsonl"
CUTS_LOG = "cuts.jsonl"
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
# Decided settings (decisions "Area 8 probes P1/P2", not published: expert only, batch 32, n_action_steps 50; the launcher's defaults
# for the rest, README section 11.1).
DECIDED = {"train_expert_only": True, "freeze_vision_encoder": True, "batch_size": 32, "n_action_steps": 50,
           "use_imagenet_stats": False, "mixed_precision": "no", "device": "cuda"}
STAGE_PROFILES = ("smoke", "full")                           # stage 2 (collected); stage 1 is `base`
# The frame export of the spec's training data (EXT-237).
EXPORT_META = "frames_meta.json"
EXPORT_SCHEMA = "ptc_mm.lerobot_frame_export"
EXPORT_SHARD_RE = re.compile(r"^chunk-(\d{3})_file-(\d{3})\.jsonl$")
EXPORT_COLUMNS = ("index", "episode_index", "frame_index", "task_index")
# eval_libero.py (EXT-241): outcomes, sub-metrics, episode settings, the default item timeout and the item set names.
OUTCOMES = ("success", "timeout", "failed", "nan_action")
SUB_METRICS = ("steps_to_success", "n_policy_calls", "steps", "wall_s", "timeout", "nan_action")
MAX_STEPS = 280
ITEM_TIMEOUT_S = 300.0
ITEM_SETS = {"gate": "items-gate", "probe": "items-probe"}
ITEM_RE = re.compile(r"^libero_object/t(\d+)/s(\d+)$")
EPISODE_SCHEMA = "ptc_mm.libero_episode"
ERROR_REASONS = ("item_timeout", "worker_died")
# argparse options of the two scripts (train_lerobot_logged.py parse_args :99-123, eval_libero.py parse_args :813-850):
# option -> 0 (a flag), 1 (one value) or "+" (one or more values). argparse accepts a unique prefix of an option.
TRAIN_OPTIONS = {"--profile": 1, "--out": 1, "--policy-path": 1, "--steps": 1, "--save-freq": 1, "--batch-size": 1,
                 "--lr": 1, "--warmup-steps": 1, "--decay-steps": 1, "--train-expert-only": 0, "--full-ft": 0,
                 "--n-action-steps": 1, "--num-workers": 1, "--seed": 1, "--device": 1, "--memory-every": 1,
                 "--dry-run": 0}
EVAL_OPTIONS = {"--write-items": 0, "--checkpoint": 1, "--items": 1, "--out": 1, "--eval-seeds": "+", "--limit": 1,
                "--per-task": 1, "--workers": 1, "--item-timeout-s": 1, "--envs-per-worker": 1, "--torch-threads": 1,
                "--device": 1, "--model-hash": 1, "--item-set-hash": 1, "--max-steps": 1, "--settle-steps": 1}
OPTIONS = {TRAIN_SCRIPT: TRAIN_OPTIONS, EVAL_SCRIPT: EVAL_OPTIONS}
# The mounts of every launch (README section 11.3): the interpreter tree and the runtime read-only at their own paths,
# the output at /out; stage 2 adds the base cut, an evaluation the evaluated cut.
MOUNTS = tuple(m for m in ((PYTHON_DIR, "ro"), (AREA8, "ro"), (OUT, "rw")) if m[0])
# Environment values judged at ext-launch (the record keeps names only). Rendering: robosuite takes its EGL device from
# MUJOCO_EGL_DEVICE_ID (README section 0.4) and the NVIDIA EGL library is injected only with `graphics` (P1).
ENV_VALUES = {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "MUJOCO_GL": "egl",
              "PYOPENGL_PLATFORM": "egl", "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics",
              "HF_LEROBOT_HOME": f"{AREA8}/data", "LIBERO_CONFIG_PATH": f"{AREA8}/libero-config",
              "WANDB_MODE": "disabled"}
EVAL_ENV_REQUIRED = ("MUJOCO_EGL_DEVICE_ID", "MUJOCO_GL", "NVIDIA_DRIVER_CAPABILITIES", "PYOPENGL_PLATFORM")
LIBRARY = ("LeRobot 0.7.0 lerobot-train through the area launcher train_lerobot_logged.py (cuts by its write_cut from "
           "LeRobot's checkpoints/<step>/, cut.json written last)")

CHECKPOINT_COMPLETENESS = {
    "library": LIBRARY,
    "rule": ("a cuts/checkpoint-<S> directory is a cut iff its cut.json (schema ptc_mm.lerobot_cut), written last after "
             "an atomic rename from cuts/.checkpoint-<S>.partial, parses with global_step = last_step_included = S and a "
             "positive integer samples_consumed, lists in `files` exactly the other 17 files of the directory (ten "
             "deployable: the weights, config, both processor descriptions and their normalizer state files, the three "
             "tokenizer files, sampler.json; six resume files: train_config.json and training_state/*; base_ref.json), "
             "each a plain file with its SHA-256, with deployable = model_files = the ten, model_files_sha256, "
             "normalizer_sha256 and sampler_sha256 recomputed, training_state/training_step.json at step S, and a whole "
             "model.safetensors; LeRobot's own lerobot/checkpoints/<step>/ directories are never cuts; the cut's last "
             "step included is S"),
    "complete_dir_pattern": f"{CUTS_DIR}/" + CHECKPOINT_DIR_RE.pattern.lstrip("^"),
    "sources": ["runtime/train_lerobot_logged.py:435-492 write_cut (cut.json last, os.rename)",
                "runtime/train_lerobot_logged.py:495-500 update_last_checkpoint hook (after LeRobot points `last`)",
                "runtime/a8_common.py model_file_names, normalizer_record, deploy_sampler",
                "the stage-1 ladder's cuts base-1/cuts/checkpoint-<250..3000> (GPU, 2026-11-15)"],
    "verified": True,
}
SEED_DERIVATION = ("lerobot-train --seed S: set_seed(S) (lerobot_train.py:574-575) and the data order "
                   "EpisodeAwareSampler(shuffle=True, seed=S), a pure function of (S, epoch) (lerobot_train.py:355-371, "
                   "datasets/sampler.py); the order used is the observed log (samples.jsonl), never a reconstruction. "
                   "The training seed is run_args.json `seed` (= settings.seed and the argv's --seed=S), which must equal "
                   "the launch record's training_seed and every cut's cut.json `seed`. Evaluation: per (eval seed, item) "
                   "torch, the flow-matching noise generator and the environment are seeded with "
                   "derive_seed('libero-torch'|'libero-noise'|'libero-env', seed, item) (first 8 bytes of SHA-256 of "
                   "the ':'-joined parts, >> 1), common to both arms; 05 section 14 H(run_seed, step, data_ref, "
                   "sample_index) is not claimed.")


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _norm_path(p: Any) -> Any:
    return posixpath.normpath(p) if isinstance(p, str) and p.startswith("/") else p


def frame_row_key(episode_index: int, frame_index: int) -> str:
    """MM-26 `data_ref.row` of a frame: SHA-256 of the canonical JSON {episode_index, frame_index}."""
    return sha256_bytes(canonical_json({"episode_index": episode_index, "frame_index": frame_index}))


def export_shard_name(source: str) -> str | None:
    """The export shard of a dataset data file: `data/chunk-CCC/file-FFF.parquet` -> `chunk-CCC_file-FFF.jsonl`."""
    m = re.fullmatch(r"data/chunk-(\d{3})/file-(\d{3})\.parquet", source) if isinstance(source, str) else None
    return f"chunk-{m.group(1)}_file-{m.group(2)}.jsonl" if m else None


def lerobot_steps_dir(step: int, steps: int) -> str:
    """LeRobot's checkpoint directory name: the step zero-padded to max(6, len(str(steps))) (train_utils.py:94-110)."""
    return f"{step:0{max(6, len(str(steps)))}d}"


def lerobot_argv(s: dict, policy_dir: str) -> list[str]:
    """train_lerobot_logged.py build_lerobot_argv (:168-215) with the area's constants and --out /out."""
    argv = [f"--policy.path={policy_dir}", f"--policy.vlm_model_name={VLM_DIR}", "--policy.load_vlm_weights=false",
            "--policy.input_features=null", "--policy.output_features=null", f"--policy.device={s.get('device')}",
            "--policy.push_to_hub=false", f"--policy.n_action_steps={s.get('n_action_steps')}",
            f"--policy.train_expert_only={str(s.get('train_expert_only')).lower()}",
            f"--policy.freeze_vision_encoder={str(s.get('freeze_vision_encoder')).lower()}"]
    if s.get("learning_rate") is not None:
        argv.append(f"--policy.optimizer_lr={s['learning_rate']}")
    if s.get("warmup_steps") is not None:
        argv.append(f"--policy.scheduler_warmup_steps={s['warmup_steps']}")
    if s.get("decay_steps") is not None:
        argv.append(f"--policy.scheduler_decay_steps={s['decay_steps']}")
    argv += [f"--dataset.repo_id={DATASET_REPO}", f"--dataset.root={DATASET_ROOT}",
             f"--dataset.revision={DATASET_REVISION}",
             "--dataset.episodes=[" + ",".join(str(e) for e in range(OBJECT_EPISODES[0], OBJECT_EPISODES[1] + 1)) + "]",
             "--dataset.video_backend=pyav", f"--dataset.use_imagenet_stats={str(s.get('use_imagenet_stats')).lower()}",
             "--rename_map=" + json.dumps(RENAME, separators=(",", ":")), f"--output_dir={OUT}/{LEROBOT_DIR}",
             f"--job_name=a8_{s.get('profile')}", f"--seed={s.get('seed')}", f"--batch_size={s.get('batch_size')}",
             f"--steps={s.get('steps')}", f"--save_freq={s.get('save_freq')}",
             f"--save_checkpoint={str(s.get('save_checkpoint')).lower()}", f"--log_freq={s.get('log_freq')}",
             f"--num_workers={s.get('num_workers')}", "--env_eval_freq=0", "--eval_steps=0", "--trackio.enable=false"]
    return argv


def cut_steps(s: dict) -> list[int] | None:
    """train_lerobot_logged.py resolve_settings: every save_freq steps and always the last step (LeRobot's save rule,
    common/train_utils.py:124-132)."""
    steps, freq = s.get("steps"), s.get("save_freq")
    if not (_int(steps) and steps >= 1 and _int(freq) and freq >= 0):
        return None
    return sorted({k for k in range(1, steps + 1) if (freq > 0 and k % freq == 0) or k == steps})


def _parse_argv(args: list, options: dict[str, Any]) -> tuple[list[tuple[str, list[str], str]], list[str]]:
    """([(option, values, token)], [tokens argparse would refuse]) as argparse reads them: an exact option or a unique
    prefix (allow_abbrev), `=value` or the next token(s); a token that is no option, a positional argument (neither
    script takes one) and a value option without its value are refused."""
    parsed, odd, i = [], [], 0
    while i < len(args):
        tok = args[i]
        opt, inline = resolve_option(tok, {o: bool(k) for o, k in options.items()})
        if opt is None:
            odd.append(tok if isinstance(tok, str) else repr(tok))
            i += 1
            continue
        kind = options[opt]
        values: list[str] = []
        if kind == 0:
            if inline is not None:
                odd.append(tok)
        elif inline is not None:
            values = [inline]
        else:
            j = i + 1
            while j < len(args) and isinstance(args[j], str) and not args[j].startswith("-") and (kind == "+" or not values):
                values.append(args[j])
                j += 1
            if not values:
                odd.append(tok)
            i = j - 1
        parsed.append((opt, values, tok))
        i += 1
    return parsed, odd


class LerobotSmolvlaAdapter:
    """The MM-12 interface for LeRobot SmolVLA SFT runs of the area runtime, two stages (EXT-233 to EXT-247)."""

    name = NAME
    trainer_id = ROW["uses"][0]
    evaluator_ids = tuple(ROW["evaluators"])
    framework_name = FRAMEWORK_PACKAGE
    checkpoint_completeness = CHECKPOINT_COMPLETENESS
    seed_derivation = SEED_DERIVATION
    binds_spec = bool(ROW["spec_binding"])
    arm_needs_items_dir = True                       # arm_problems() gets the registered items and the evaluated cut
    consumption_fix = "collect the run's own samples.jsonl as the launcher wrote it, with the spec's frame export (MM-26)"

    BASE_FIX = ("continue from the spec's stage-1 cut mounted read-only at /inputs/base-cut (train_lerobot_logged.py "
                "--policy-path /inputs/base-cut, recorded in run_args.json base_checkpoint and base_cut), and record "
                "that mount in the launch record")
    DATA_FIX = ("point the spec's training data at the frame export of the dataset the run read (frames_meta.json and "
                "one chunk-CCC_file-FFF.jsonl per data file of episodes 807-1260)")

    def metric(self, evaluator_id: str) -> dict[str, Any]:
        if evaluator_id not in self.evaluator_ids:
            raise KeyError(evaluator_id)
        row = EVALUATOR_ROWS[evaluator_id]
        return {"metric": row["metric"], "direction": row["direction"], "score_range": tuple(row["score_range"]),
                "error_classes": tuple(row["error_classes"]), "score_outcomes": tuple(row["score_outcomes"]),
                "scorer": dict(row["scorer"]), "binary": row["binary"], "tier": row["tier"]}

    # --- cuts (MM-17, MM-19, MM-20; EXT-234, EXT-235) ---

    @staticmethod
    def is_cut(d: Path) -> bool:
        return (Path(d) / CUT_MARKER).exists() or (Path(d) / CUT_MARKER).is_symlink()

    @staticmethod
    def _tree(d: Path) -> tuple[list[str], list[str]]:
        """(relative POSIX paths of the regular files below d, paths that are links or not plain files)."""
        files, bad = [], []
        for p in sorted(Path(d).rglob("*")):
            rel = p.relative_to(d).as_posix()
            if p.is_symlink():
                bad.append(rel)
            elif p.is_file():
                files.append(rel)
            elif not p.is_dir():
                bad.append(rel)
        return files, bad

    @staticmethod
    def _normalizer_files(pretrained: Path) -> tuple[dict[str, str] | None, str | None]:
        """({role: relative path in the cut}, None) from the processor descriptions' `state_file` entries."""
        out = {}
        for role, cfg_name, reg in NORMALIZER_ROLES:
            doc, why = _read_json(pretrained / cfg_name)
            if not isinstance(doc, dict) or not isinstance(doc.get("steps"), list):
                return None, f"{PRETRAINED}/{cfg_name}: {why or 'no steps list'}"
            names = [s.get("state_file") for s in doc["steps"]
                     if isinstance(s, dict) and s.get("registry_name") == reg and isinstance(s.get("state_file"), str)]
            if len(names) != 1 or "/" in names[0] or names[0] in (".", ".."):
                return None, f"{PRETRAINED}/{cfg_name} names {names!r}, not one {reg} state file"
            out[role] = f"{PRETRAINED}/{names[0]}"
        return out, None

    def normalizer_sha256(self, cut: Path) -> str:
        """MM-19 (README section 11.1): SHA-256 of the compact sorted JSON {normalizer: sha256, unnormalizer: sha256} of
        the two processor state files named in policy_preprocessor.json / policy_postprocessor.json. Raises OSError
        when they cannot be read or are not named."""
        names, why = self._normalizer_files(Path(cut) / PRETRAINED)
        if names is None:
            raise FileNotFoundError(why)                  # collect and ext-arm-fill catch OSError (EXT-154)
        return compact_sha256({role: file_sha256(Path(cut) / rel) for role, rel in names.items()})

    def _cut_files_problem(self, d: Path) -> str | None:
        """Review 9 M1 (EXT-154): a file that cannot be read makes the directory incomplete, never an exception."""
        try:
            return self._cut_files_problem_unguarded(Path(d))
        except (OSError, ValueError) as exc:
            name = Path(exc.filename).name if isinstance(exc, OSError) and exc.filename else Path(d).name
            return f"{name} cannot be read ({exc.__class__.__name__})"

    def _cut_files_problem_unguarded(self, d: Path) -> str | None:
        doc, why = _read_json(d / CUT_MARKER)
        if why:
            return f"{why} ({CUT_MARKER} is written last)"
        if not isinstance(doc, dict) or not isinstance(doc.get("files"), dict):
            return f"{CUT_MARKER} has no `files` object"
        if doc.get("schema") != CUT_SCHEMA:
            return f"{CUT_MARKER} schema is {doc.get('schema')!r}, not {CUT_SCHEMA}"
        files = doc["files"]
        if sorted(files) != list(CUT_FILES):
            return (f"{CUT_MARKER} lists {sorted(set(files) ^ set(CUT_FILES))[:4]} beyond or without the cut files "
                    f"(train_lerobot_logged.py write_cut)")
        present, odd = self._tree(d)
        if odd:
            return f"{odd[0]} is a symbolic link or not a plain file (MM-20)"
        if sorted(present) != sorted([*CUT_FILES, CUT_MARKER]):
            extra = sorted(set(present) - set(CUT_FILES) - {CUT_MARKER})
            return f"the directory holds {extra or sorted(set(CUT_FILES) - set(present))}, which {CUT_MARKER} does not list"
        for rel in CUT_FILES:
            if file_sha256(d / rel) != files[rel]:
                return f"{rel} is not the SHA-256 {CUT_MARKER} records"
        for key in ("deployable", "model_files"):
            if doc.get(key) != sorted(COMPONENTS):
                return f"{CUT_MARKER} {key} is not the ten deployable files {sorted(COMPONENTS)}"
        if doc.get("resume_files") != sorted(RESUME_FILES):
            return f"{CUT_MARKER} resume_files is not {sorted(RESUME_FILES)}"
        if doc.get("model_files_sha256") != compact_sha256({k: files[k] for k in sorted(COMPONENTS)}):
            return f"{CUT_MARKER} model_files_sha256 is not the hash of its model files (compact sorted JSON)"
        names, why = self._normalizer_files(d / PRETRAINED)
        if names is None:
            return why
        if sorted(names.values()) != sorted(c for c in COMPONENTS if c.endswith("_processor.safetensors")):
            return f"the processor descriptions name the state files {sorted(names.values())}, not the cut's"
        if doc.get("normalizer_sha256") != compact_sha256({role: files[rel] for role, rel in names.items()}):
            return f"{CUT_MARKER} normalizer_sha256 is not the hash of the two processor state files (MM-19)"
        sampler, why = _read_json(d / SAMPLER_FILE)
        if not isinstance(sampler, dict) or doc.get("sampler_sha256") != compact_sha256(sampler):
            return f"{CUT_MARKER} sampler_sha256 is not the hash of {SAMPLER_FILE} (compact sorted JSON)"
        step = doc.get("global_step")
        if not (_int(step) and step >= 1 and doc.get("last_step_included") == step):
            return f"{CUT_MARKER} global_step {step!r} is not a positive integer equal to last_step_included"
        if not (_int(doc.get("samples_consumed")) and doc["samples_consumed"] >= 1):
            return f"{CUT_MARKER} samples_consumed {doc.get('samples_consumed')!r} is not a positive integer"
        ts, why = _read_json(d / "training_state" / "training_step.json")
        if not (isinstance(ts, dict) and ts.get("step") == step):
            return f"training_state/training_step.json step is {(ts or {}).get('step') if isinstance(ts, dict) else why!r}, not {step}"
        return safetensors_problem(d / WEIGHTS)

    def model_dir_problem(self, d: Path) -> str | None:
        """None when `d` is a complete cut (EXT-234). The spec's base of this adapter is a stage-1 cut, never the plain
        smolvla_base directory (the gate's baseline is an under-trained SFT cut, MULTIMODAL-SPEC row 8)."""
        d = Path(d)
        if d.is_symlink() or not d.is_dir():
            return f"{d} is not a plain directory"
        if not self.is_cut(d):
            return (f"{d} holds no {CUT_MARKER}: the base of a lerobot_smolvla gate is a stage-1 cut "
                    "(cuts/checkpoint-<step>/), not the plain smolvla_base directory (EXT-236)")
        return self._cut_files_problem(d)

    def cut_problem(self, d: Path, step: int) -> str | None:
        d = Path(d)
        if d.is_symlink() or not d.is_dir():
            return f"{d} is not a plain directory"
        bad = self._cut_files_problem(d)
        if bad:
            return bad
        doc, _ = _read_json(d / CUT_MARKER)
        if doc.get("global_step") != step:
            return f"{CUT_MARKER} global_step is {doc.get('global_step')!r}, not {step}"
        return None

    def checkpoint_dirs(self, stage_out: Path) -> list[tuple[int, Path]]:
        """Every `cuts/checkpoint-<S>` directory, complete or not, in step order; `.checkpoint-<S>.partial` and
        LeRobot's `lerobot/checkpoints/<step>` are never one."""
        root = Path(stage_out) / CUTS_DIR
        if not root.is_dir():
            return []
        out = []
        for p in root.iterdir():
            m = CHECKPOINT_DIR_RE.match(p.name)
            if m and p.is_dir() and not p.is_symlink():
                out.append((int(m.group(1)), p))
        return sorted(out)

    def find_cuts(self, stage_out: Path) -> tuple[list[tuple[int, Path]], list[Path]]:
        complete, incomplete = [], []
        for step, d in self.checkpoint_dirs(stage_out):
            if self.cut_problem(d, step) is None:
                complete.append((step, d))
            else:
                incomplete.append(d)
        return complete, incomplete

    def identity(self, cut: Path, base: Any = None, *, entries: list[dict] | None = None) -> dict[str, Any]:
        """EXT-235 (MM-17): `model_hash` over the ten deployable files, `cut_hash` over the whole cut; full weights, so
        `base` is not used and `base_ref` is null. A directory without cut.json (refused as a base) hashes whole."""
        if entries is None:
            entries = manifest_entries(Path(cut))
        if not self.is_cut(cut):
            h = entries_hash(entries)
            return {"model_hash": h, "cut_hash": h, "components": [e["path"] for e in entries], "base_ref": None}
        comps = [e for e in entries if e["path"] in COMPONENTS]
        return {"model_hash": entries_hash(comps), "cut_hash": entries_hash(entries),
                "components": [e["path"] for e in comps], "base_ref": None}

    def extra_read_outputs(self, stage_out: Path) -> list[Path]:
        """EXT-64: the launcher's start/finish log and cut log are read too (EXT-239), so they must be this launch's."""
        return [Path(stage_out) / n for n in (ATTEMPTS, CUTS_LOG) if (Path(stage_out) / n).exists()]

    # --- the frame export (EXT-237) and consumption (MM-26, MM-27; EXT-238) ---

    @staticmethod
    def _export(data: Path) -> tuple[dict | None, dict[int, tuple[int, int, str]] | None, list[str]]:
        """(frames_meta.json, {index: (episode_index, frame_index, shard)}, problems) of the spec's frame export: every
        `*.jsonl` shard is named in the meta with its row count; rows {index, episode_index, frame_index, task_index}
        (integers), indices unique and increasing within a shard, episodes inside 807-1260, frames of an episode
        contiguous from 0."""
        data = Path(data)
        meta, why = _read_json(data / EXPORT_META)
        if not isinstance(meta, dict) or meta.get("schema") != EXPORT_SCHEMA or not isinstance(meta.get("shards"), dict):
            return None, None, [f"{data / EXPORT_META}: {why or 'not a ' + EXPORT_SCHEMA + ' document with shards'}"]
        names = sorted(p.name for p in data.glob("*.jsonl"))
        if names != sorted(meta["shards"]):
            return meta, None, [f"the export's shards {names[:3]}... are not those {EXPORT_META} lists"]
        frames: dict[int, tuple[int, int, str]] = {}
        by_episode: dict[int, list[int]] = {}
        out: list[str] = []
        for name in names:
            info = meta["shards"][name]
            if not EXPORT_SHARD_RE.fullmatch(name) or not isinstance(info, dict) \
                    or export_shard_name(info.get("source")) != name:
                out.append(f"export shard {name} does not name its data file (data/chunk-CCC/file-FFF.parquet)")
                continue
            rows, why = read_jsonl(data / name)
            if rows is None:
                out.append(f"export shard {why}")
                continue
            if info.get("rows") != len(rows):
                out.append(f"export shard {name} holds {len(rows)} rows, not the {info.get('rows')!r} {EXPORT_META} lists")
            last = None
            for r in rows:
                if sorted(r) != sorted(EXPORT_COLUMNS) or not all(_int(r[c]) for c in EXPORT_COLUMNS):
                    out.append(f"export shard {name}: a row is not {{{', '.join(EXPORT_COLUMNS)}}} integers")
                    break
                i, e, f = r["index"], r["episode_index"], r["frame_index"]
                if (last is not None and i <= last) or i in frames:
                    out.append(f"export shard {name}: index {i} repeats or is out of order")
                    break
                if not OBJECT_EPISODES[0] <= e <= OBJECT_EPISODES[1]:
                    out.append(f"export shard {name}: episode {e} is not a LIBERO-Object episode {OBJECT_EPISODES}")
                    break
                last = i
                frames[i] = (e, f, name)
                by_episode.setdefault(e, []).append(f)
            if len(out) >= 4:
                return meta, None, out
        if out:
            return meta, None, out
        if sorted(by_episode) != list(range(OBJECT_EPISODES[0], OBJECT_EPISODES[1] + 1)):
            return meta, None, [f"the export holds episodes {min(by_episode, default=None)}..{max(by_episode, default=None)} "
                                f"({len(by_episode)}), not every LIBERO-Object episode {OBJECT_EPISODES}"]
        for e, fs in by_episode.items():
            if sorted(fs) != list(range(len(fs))):
                return meta, None, [f"episode {e}'s frames in the export are not 0..{len(fs) - 1}"]
        if meta.get("n_frames") != len(frames):
            return meta, None, [f"{EXPORT_META} n_frames {meta.get('n_frames')!r} is not the {len(frames)} exported rows"]
        return meta, frames, []

    def _ledger_plan(self, stage_out: Path, data: Path, cuts: list) -> tuple[list[dict], dict, list[str]]:
        """(the samples.jsonl lines up to the last cut, the export's frames, problems). Every structural rule is
        re-derived from the export (N frames) and run_args.json (batch B, steps)."""
        args, why = read_run_args(stage_out)
        if args is None:
            return [], {}, [why]
        if not cuts:
            return [], {}, ["no complete cut"]
        _, frames, bad = self._export(data)
        if bad:
            return [], {}, bad
        s = args.get("settings") if isinstance(args.get("settings"), dict) else {}
        bs, steps = s.get("batch_size"), s.get("steps")
        if not (_int(bs) and bs > 0 and _int(steps) and steps > 0):
            return [], {}, [f"run_args.json settings.batch_size {bs!r} and steps {steps!r} are not positive integers"]
        n = len(frames)
        spe = math.ceil(n / bs)                          # steps per epoch (drop_last false)
        rest = n - (spe - 1) * bs                        # samples of an epoch's last step
        lines, why = read_jsonl(Path(stage_out) / SAMPLES)
        if lines is None:
            return [], {}, [why]
        out: list[str] = []

        def bad_line(msg: str) -> bool:
            out.append(msg)
            return len(out) >= 8

        seen: dict[int, set[int]] = {}
        consumed: dict[int, int] = {0: 0}
        for k, x in enumerate(lines, start=1):
            where = f"{SAMPLES} line {k}"
            if x.get("step") != k:
                return [], {}, [f"{where}: step {x.get('step')!r} is not {k} (one line per step, from 1)"]
            want_e = (k - 1) // spe
            want_n = rest if k % spe == 0 else bs
            smp = x.get("samples")
            if sorted(x) != ["epoch", "n", "samples", "step"] or not isinstance(smp, list):
                consumed[k] = consumed[k - 1] + (len(smp) if isinstance(smp, list) else 0)
                if bad_line(f"{where}: keys {sorted(x)} are not epoch, n, samples, step"):
                    return [], {}, out
                continue
            if x.get("epoch") != want_e:
                if bad_line(f"{where}: epoch {x.get('epoch')!r}, not {want_e} ({spe} steps per epoch: {n} frames at "
                            f"batch {bs})"):
                    return [], {}, out
            if x.get("n") != len(smp) or len(smp) != want_n:
                if bad_line(f"{where}: n {x.get('n')!r} and {len(smp)} samples, not {want_n} (batch {bs}; the last step of "
                            f"each epoch carries the rest, {rest})"):
                    return [], {}, out
            ep_seen = seen.setdefault(want_e, set())
            for sm in smp:
                ok = (isinstance(sm, dict) and sorted(sm) == ["episode_index", "frame_index", "index"]
                      and all(_int(sm[c]) for c in sm))
                hit = frames.get(sm["index"]) if ok else None
                if hit is None or hit[:2] != (sm["episode_index"], sm["frame_index"]):
                    if bad_line(f"{where}: sample {sm!r} is not a frame (index, episode_index, frame_index) of the "
                                "export (LIBERO-Object, the dataset revision the run read)"):
                        return [], {}, out
                    break
                if sm["index"] in ep_seen:
                    if bad_line(f"{where}: frame index {sm['index']} twice in epoch {want_e} (the sampler reads every "
                                "frame once per epoch)"):
                        return [], {}, out
                    break
                ep_seen.add(sm["index"])
            consumed[k] = consumed[k - 1] + len(smp)
        if out:
            return [], {}, out
        if len(lines) != steps:
            return [], {}, [f"{SAMPLES} logs {len(lines)} steps, not run_args.json settings.steps {steps} (the run "
                            "stops after its last step and saves it)"]
        # MM-27: each cut's samples_consumed is the log's count up to its step; no cut beyond the log.
        for step, d in cuts:
            doc, _ = _read_json(Path(d) / CUT_MARKER)
            doc = doc if isinstance(doc, dict) else {}
            if step > len(lines):
                out.append(f"{Path(d).name}: step {step} is beyond the {len(lines)} steps {SAMPLES} logs")
            elif doc.get("samples_consumed") != consumed[step]:
                out.append(f"{Path(d).name} cut.json samples_consumed is {doc.get('samples_consumed')!r}, but the steps "
                           f"up to {step} consumed {consumed[step]} samples")
        if out:
            return [], {}, out
        last = max(step for step, _ in cuts)
        return [x for x in lines if x["step"] <= last], frames, []

    def consumption_problems(self, stage_out: Path, data: Path, cuts: list) -> list[str]:
        """EXT-238: why the observed log cannot give the per-sample consumption up to the last cut."""
        return self._ledger_plan(stage_out, data, cuts)[2]

    def consumption(self, stage_out: Path, data: Path, cuts: list, *, run_id: str = "",
                    attempt_id: str = "0") -> dict[str, Any]:
        """Precision `row`, derivation `observed` (MM-26): one ledger record per sample of the steps up to the last cut;
        `data_ref.row` = SHA-256 of the canonical JSON {episode_index, frame_index} of the anchor frame, shard = the
        export shard of the data file holding the episode, `sample_id` = SHA-256 of the NUL-joined (run, attempt,
        epoch, step, position, index); `rewards`, `seed_inputs`, `sampler_config_hash`, `chunk` null (supervised; the
        49 later actions a sample also loads are disclosed, not recorded)."""
        from gatedtrain.ledger import WRITER_ID, check_precision

        plan, frames, bad = self._ledger_plan(stage_out, data, cuts)
        if bad:                                          # collect refuses these before writing
            raise ValueError("; ".join(bad))
        check_precision("row", "observed")
        snapshot = manifest_hash(data)
        rows = []
        for x in plan:
            for pos, sm in enumerate(x["samples"]):
                e, f, shard = frames[sm["index"]]
                key = "\x00".join(str(v) for v in (run_id, attempt_id, x["epoch"], x["step"], pos, sm["index"]))
                rows.append({
                    "run_id": run_id, "attempt_id": attempt_id, "step": x["step"],
                    "sample_id": "sha256:" + hashlib.sha256(key.encode("utf-8")).hexdigest(),
                    "data_ref": {"snapshot": snapshot, "shard": shard, "row": frame_row_key(e, f), "mixture": None,
                                 "component": None},
                    "parent_ref": None, "outcome": "consumed", "reason": None, "rule_version": None,
                    "gen_version": None, "reward_epoch": None, "rewards": None, "sampler_config_hash": None,
                    "seed_inputs": None, "chunk": None, "precision": "row", "derivation": "observed",
                    "writer_id": WRITER_ID,
                })
        last = max(step for step, _ in cuts)
        return {"precision": "row", "rows": rows, "derivation": "observed",
                "reason": (f"{SAMPLES} logs every sample of each step before its update (MM-26, MM-27): {len(rows)} "
                           f"records of steps 1 .. {last}, data_ref.row = sha256({{episode_index, frame_index}}) of the "
                           "anchor frame in the shard of its data file, derivation observed; each sample also loads the "
                           "next 49 actions of its episode (not recorded); the stage-1 run's consumption is not in "
                           "this ledger")}

    # --- pins and checks (MM-21, MM-29, MM-33; EXT-236, EXT-237, EXT-239) ---

    def framework_identity(self, stage_out: Path, launch: dict) -> dict[str, Any]:
        """`lerobot==<v>` in env-freeze.txt equals the launch record's framework, and the freeze pins torch,
        transformers, hf-libero, robosuite and mujoco (MM-33). Raises ValueError with the reason."""
        freeze = Path(stage_out) / FREEZE_FILE
        if not _plain_file(freeze):
            raise ValueError(f"{freeze} is missing")
        versions: dict[str, str] = {}
        for line in text_lines(freeze.read_text(encoding="utf-8")):
            line = line.strip()
            if "==" in line and not line.startswith("#"):
                name, _, version = line.partition("==")
                versions[name.strip().lower().replace("_", "-")] = version.strip()
        missing = [p for p in PINS if p not in versions]
        if missing:
            raise ValueError(f"{freeze} has no `<name>==<version>` line for {', '.join(missing)} (pins of {NAME})")
        version = versions[FRAMEWORK_PACKAGE]
        fw = launch.get("framework") if isinstance(launch.get("framework"), dict) else {}
        if fw.get("name") != FRAMEWORK_PACKAGE or fw.get("version") != version:
            raise ValueError(f"{freeze} pins {FRAMEWORK_PACKAGE}=={version}, but the launch record claims "
                             f"framework {fw.get('name')!r} {fw.get('version')!r}")
        return {"name": FRAMEWORK_PACKAGE, "version": version, "freeze_sha256": file_sha256(freeze)}

    def script_problems(self, stage_out: Path, launch: dict) -> list[Problem]:
        """EXT-83: env-runtime.json `script_sha256` equals the record's (the launcher hashes itself)."""
        return runtime_script_problems(stage_out, launch)

    spec_binding_argv = True        # launch.spec_binding_problems passes the argv

    def mount_binding_problems(self, mounts: Any, *, base: Path, data: Path, base_field: str, data_field: str,
                               flag: str = "--launch-record", argv: Any = None) -> list[Problem]:
        """`ext-launch --spec` (optional): a read-only mount whose host path is the spec's base cut, and the spec's frame
        export inside a read-only mount (EXT-236, EXT-237). Only a training launch is bound: an eval_libero.py launch
        mounts the cut it evaluates at /inputs/cut, which for the candidate arm is not the base cut. Without an argv
        (a direct call) the training rule applies."""
        
        if argv is not None and self._script(argv)[0] != TRAIN_SCRIPT:
            return []
        out = []
        if not _bound_mounts(mounts, base):
            out.append(Problem("EXT-BASE", base_field, f"{flag} mounts no read-only directory whose host path is the "
                               f"spec's base cut {base}", self.BASE_FIX))
        want = str(Path(data).resolve())
        if not any(within(want, host) for _, host in ro_table(mounts)):
            out.append(Problem("EXT-DATA", data_field, f"{flag} mounts no read-only directory that is or holds the "
                               f"spec's frame export {data}", self.DATA_FIX))
        return out

    def checks(self, *, last_cut: Path, base: Path, launch: dict, base_field: str, stage_out: Path | None = None,
               data: Path | None = None, data_field: str = "data", **_: Any) -> list[Problem]:
        """EXT-236 two stages and base binding, EXT-237 data binding, EXT-239 seed, profile, script, settings, pins and
        the finished run."""
        stage_out = Path(stage_out) if stage_out is not None else Path(last_cut).parent.parent
        args, why = read_run_args(stage_out)
        if args is None:
            return [Problem("EXT-BASE", base_field, f"{why}, so the run cannot be bound to the spec's base cut "
                            "(EXT-236)", self.BASE_FIX)]
        cuts = [d for _, d in self.find_cuts(stage_out)[0]] or [Path(last_cut)]
        out = self._base_problems(args, cuts, Path(base), launch, base_field)
        if data is not None:
            out += self._data_problems(args, Path(data), launch.get("mounts"), data_field)
        out += self._run_problems(args, cuts, Path(last_cut), stage_out, launch)
        return out

    def _base_problems(self, args: dict, cuts: list[Path], base: Path, launch: dict, field: str) -> list[Problem]:
        """EXT-236: stage 2 continued from the spec's stage-1 cut, through its read-only mount, and recorded it."""
        out: list[Problem] = []

        def bad(msg: str, fix: str = self.BASE_FIX) -> None:
            out.append(Problem("EXT-BASE", field, msg, fix))

        if args.get("stage") != "continuation":
            bad(f"run_args.json stage is {args.get('stage')!r}: ext-collect gates a stage-2 continuation against the "
                "stage-1 cut it loaded; a stage-1 run trains from smolvla_base and produces the baseline (EXT-236)",
                "collect the stage-2 run whose --policy-path is the spec's base cut")
        bdoc, why = _read_json(base / CUT_MARKER)
        bdoc = bdoc if isinstance(bdoc, dict) else {}
        if bdoc.get("stage") != "base" or bdoc.get("profile") != "base" or bdoc.get("init_checkpoint") is not None:
            bad(f"the spec's base cut {base} is stage {bdoc.get('stage')!r} profile {bdoc.get('profile')!r} with "
                f"init_checkpoint {bdoc.get('init_checkpoint')!r}, not a stage-1 cut trained from smolvla_base: chains "
                "of more than two stages are not supported (EXT-236)", "use a cut of the stage-1 (base) run")
        base_run = base.parent.parent
        rargs, why = read_run_args(base_run)
        if base.parent.name != CUTS_DIR or rargs is None:
            bad(f"the base cut's run {base_run}: {why or 'the cut is not in its cuts/ directory'}; the base cut must lie "
                "in its stage-1 run's output directory (EXT-236)", "point the spec's base at <stage-1 run>/cuts/checkpoint-<step>")
        elif (rargs.get("schema"), rargs.get("stage"), rargs.get("base_checkpoint"), rargs.get("base_cut")) != \
                (RUN_ARGS_SCHEMA, "base", None, None) or rargs.get("policy_path") != SMOLVLA_DIR:
            bad(f"the base cut's run {base_run} is not a stage-1 run from {SMOLVLA_DIR} (run_args.json stage "
                f"{rargs.get('stage')!r}, base_checkpoint {rargs.get('base_checkpoint')!r}) (EXT-236)",
                "use a cut of the stage-1 (base) run")
        elif rargs.get("seed") != bdoc.get("seed"):
            bad(f"the base cut's seed {bdoc.get('seed')!r} is not its run's {rargs.get('seed')!r}")
        # the mount and the recorded base cut
        mounts = _bound_mounts(launch.get("mounts"), base)
        conts = sorted({_norm_path(m.get("container")) for m in mounts})
        if conts != [BASE_CUT_CONTAINER]:
            bad(f"the launch record mounts the spec's base cut {base} at {conts}, not read-only at exactly "
                f"{BASE_CUT_CONTAINER} (EXT-236)")
        policy = f"{BASE_CUT_CONTAINER}/{PRETRAINED}"
        if args.get("base_checkpoint") != policy or args.get("policy_path") != policy:
            bad(f"run_args.json base_checkpoint {args.get('base_checkpoint')!r} and policy_path "
                f"{args.get('policy_path')!r} are not {policy}, the base cut's pretrained_model (EXT-236)")
        hit = container_file(f"{policy}/config.json", launch.get("mounts"))
        if hit is None or Path(hit[1]).resolve() != (base / PRETRAINED / "config.json").resolve():
            bad(f"{policy}/config.json is not, through the launch record's read-only mounts, the spec's base cut's "
                "(MM-21)")
        try:
            bsha = file_sha256(base / CUT_MARKER)
        except OSError:
            bsha = None
        want = {"cut_dir": BASE_CUT_CONTAINER, "cut_json_sha256": bsha, "global_step": bdoc.get("global_step"),
                "model_files_sha256": bdoc.get("model_files_sha256"), "normalizer_sha256": bdoc.get("normalizer_sha256"),
                "profile": bdoc.get("profile"), "stage": bdoc.get("stage"), "seed": bdoc.get("seed")}
        if args.get("base_cut") != want:
            bad("run_args.json base_cut (the cut the run loaded: cut.json SHA-256, step, model files, normalizer) is not "
                f"the spec's base cut {base}: the run started from other weights (EXT-236)")
        # every stage-2 cut names it, carries the same bindings and the same normalizer
        bref, _ = _read_json(base / BASE_REF)
        bref = bref if isinstance(bref, dict) else {}
        out += self._binding_problems(bref, f"the base cut's {BASE_REF}", field)
        for d in cuts:
            doc, _ = _read_json(Path(d) / CUT_MARKER)
            doc = doc if isinstance(doc, dict) else {}
            ref, _ = _read_json(Path(d) / BASE_REF)
            ref = ref if isinstance(ref, dict) else {}
            cfg, _ = _read_json(Path(d) / PRETRAINED / "config.json")
            cfg = cfg if isinstance(cfg, dict) else {}
            if doc.get("init_checkpoint") != args.get("base_cut") or ref.get("init_checkpoint") != args.get("base_cut"):
                bad(f"{Path(d).name}: cut.json and {BASE_REF} init_checkpoint are not run_args.json base_cut (EXT-236)")
            if {k: v for k, v in ref.items() if k != "init_checkpoint"} != \
                    {k: v for k, v in bref.items() if k != "init_checkpoint"}:
                bad(f"{Path(d).name}/{BASE_REF} names other smolvla_base, SmolVLM2 or dataset bindings than the base "
                    "cut's (MM-21)")
            if cfg.get("pretrained_path") != policy:
                bad(f"{Path(d).name}/{PRETRAINED}/config.json pretrained_path {cfg.get('pretrained_path')!r} is not "
                    f"{policy} (MM-21)")
            if doc.get("normalizer_sha256") != bdoc.get("normalizer_sha256"):
                out.append(Problem("EXT-DATA", field, f"{Path(d).name} normalizer_sha256 is not the base cut's: stage 2 "
                                   "replaced the normalizer statistics, so it trained on other dataset statistics than "
                                   "stage 1 (MM-19; README section 4)", "continue on the same dataset revision and "
                                   "episodes as stage 1"))
        rb = args.get("base_ref") if isinstance(args.get("base_ref"), dict) else {}
        summary = {"base_model": {k: v for k, v in (bref.get("base_model") or {}).items() if k != "files"},
                   "vlm_backbone": bref.get("vlm_backbone"), "dataset": bref.get("dataset"),
                   "init_checkpoint": args.get("base_cut")}
        if rb != summary:
            bad("run_args.json base_ref is not the base cut's bindings with this run's init_checkpoint (MM-21)")
        return out

    @staticmethod
    def _binding_problems(ref: dict, where: str, field: str) -> list[Problem]:
        """MM-18, MM-21: the pinned smolvla_base, SmolVLM2 and dataset revisions, and smolvla_base's files hash."""
        out = []
        fix = "train from the pinned downloads of the area runtime (MANIFEST.json)"
        bm, vlm, ds = (ref.get(k) if isinstance(ref.get(k), dict) else {} for k in ("base_model", "vlm_backbone",
                                                                                      "dataset"))
        for what, b, repo, rev in (("base_model", bm, SMOLVLA_REPO, SMOLVLA_REVISION),
                                   ("vlm_backbone", vlm, SMOLVLM_REPO, SMOLVLM_REVISION),
                                   ("dataset", ds, DATASET_REPO, DATASET_REVISION)):
            if (b.get("repo"), b.get("revision")) != (repo, rev) or not REVISION_RE.fullmatch(str(b.get("revision"))):
                out.append(Problem("EXT-BASE", field, f"{where} {what} is {b.get('repo')!r} @ {b.get('revision')!r}, not "
                                   f"{repo} @ {rev} (a 40-hex revision, MM-18)", fix))
        files = bm.get("files")
        if not (isinstance(files, dict) and files and bm.get("files_sha256") == compact_sha256(files)
                and bm.get("n_files") == len(files)):
            out.append(Problem("EXT-BASE", field, f"{where} base_model files_sha256 is not the hash of its files (compact "
                               "sorted JSON, a8_common.manifest_binding)", fix))
        if ds.get("episodes") != list(OBJECT_EPISODES):
            out.append(Problem("EXT-DATA", field, f"{where} dataset episodes {ds.get('episodes')!r} are not "
                               f"{list(OBJECT_EPISODES)} (LIBERO-Object)", fix))
        return out

    def _data_problems(self, args: dict, data: Path, mounts: Any, field: str) -> list[Problem]:
        """EXT-237: the spec's frame export is the dataset the run read: its binding is the run's (repository,
        revision, the 457-file hash, episodes 807-1260), and every shard's data file, reached through the launch
        record's read-only mounts at the run's dataset root, has the recorded SHA-256 and exactly the export's rows."""
        out: list[Problem] = []

        def bad(msg: str) -> None:
            out.append(Problem("EXT-DATA", field, msg, self.DATA_FIX))

        meta, frames, why = self._export(data)
        if meta is None or frames is None:
            for msg in why[:4]:
                bad(msg)
            return out
        ds = args.get("dataset") if isinstance(args.get("dataset"), dict) else {}
        rb = (args.get("base_ref") or {}).get("dataset") if isinstance(args.get("base_ref"), dict) else None
        rb = rb if isinstance(rb, dict) else {}
        want = {"repo": DATASET_REPO, "revision": DATASET_REVISION, "files_sha256": rb.get("files_sha256"),
                "episodes": list(OBJECT_EPISODES)}
        if meta.get("dataset") != want or not isinstance(rb.get("files_sha256"), str):
            bad(f"{EXPORT_META} dataset {meta.get('dataset')!r} is not the run's {want} (run_args.json base_ref.dataset)")
        if (ds.get("repo"), ds.get("revision"), ds.get("root"), ds.get("rename_map"), ds.get("video_backend")) != \
                (DATASET_REPO, DATASET_REVISION, DATASET_ROOT, RENAME, "pyav"):
            bad(f"run_args.json dataset is not {DATASET_REPO} @ {DATASET_REVISION} at {DATASET_ROOT} with the camera "
                "rename map and the pyav backend")
        n_ep = OBJECT_EPISODES[1] - OBJECT_EPISODES[0] + 1
        if ds.get("episodes") != f"{OBJECT_EPISODES[0]}..{OBJECT_EPISODES[1]} ({n_ep})":
            bad(f"run_args.json dataset episodes {ds.get('episodes')!r} are not LIBERO-Object's {OBJECT_EPISODES}")
        try:
            import pyarrow.parquet as pq
        except ImportError:                              # pragma: no cover (gatedtrain depends on pyarrow)
            bad("pyarrow is not importable, so the export cannot be checked against the dataset")
            return out
        by_shard: dict[str, list[tuple[int, ...]]] = {}
        for name in meta["shards"]:
            rows, _ = read_jsonl(data / name)
            by_shard[name] = sorted(tuple(r[c] for c in EXPORT_COLUMNS) for r in rows or [])
        for name, info in sorted(meta["shards"].items()):
            hit = container_file(f"{DATASET_ROOT}/{info['source']}", mounts)
            if hit is None:
                bad(f"{DATASET_ROOT}/{info['source']} (export shard {name}) does not resolve through the launch record's "
                    "read-only mounts: the run read no such data file")
                continue
            if _sha_or_none(Path(hit[1])) != info.get("source_sha256"):
                bad(f"{info['source']} is not the SHA-256 {EXPORT_META} records for shard {name}")
                continue
            try:
                table = pq.read_table(hit[1], columns=list(EXPORT_COLUMNS)).to_pydict()
            except Exception as exc:                                                    # noqa: BLE001
                bad(f"{info['source']} cannot be read as parquet ({exc.__class__.__name__})")
                continue
            got = sorted(v for v in zip(*(table[c] for c in EXPORT_COLUMNS))
                         if OBJECT_EPISODES[0] <= v[1] <= OBJECT_EPISODES[1])
            if got != by_shard.get(name):
                bad(f"export shard {name} is not the LIBERO-Object rows (index, episode_index, frame_index, task_index) "
                    f"of {info['source']}")
            if len(out) >= 6:
                break
        return out

    def _run_problems(self, args: dict, cuts: list[Path], last_cut: Path, stage_out: Path,
                      launch: dict) -> list[Problem]:
        """EXT-239: schemas, seed, profile, script, decided settings, the exact lerobot-train argv, pins, the finished
        run (its attempt's finish line) and every cut's own fields."""
        out: list[Problem] = []
        fix = "collect the run as train_lerobot_logged.py wrote it"
        s = args.get("settings") if isinstance(args.get("settings"), dict) else {}
        if args.get("schema") != RUN_ARGS_SCHEMA:
            out.append(Problem("EXT-STAGE", "--stage-out", f"run_args.json schema is {args.get('schema')!r}, not "
                               f"{RUN_ARGS_SCHEMA} (train_lerobot_logged.py)", fix))
        profile = args.get("profile")
        if profile != launch.get("profile") or s.get("profile") != profile or profile not in STAGE_PROFILES:
            out.append(Problem("EXT-PROFILE", "--launch-record.profile", f"run_args.json profile {profile!r} (settings "
                               f"{s.get('profile')!r}) is not the launch record's {launch.get('profile')!r} among "
                               f"{list(STAGE_PROFILES)}", "collect the run launched with this record's profile"))
        claimed, seed = launch.get("training_seed"), args.get("seed")
        if not (_int(seed) and seed == claimed and s.get("seed") == seed):
            out.append(Problem("EXT-LAUNCH", "--launch-record.training_seed", f"the launch record claims training_seed "
                               f"{claimed!r}, but run_args.json seed is {seed!r} (settings {s.get('seed')!r}) (MM-29)",
                               "launch with the seed the record names"))
        if args.get("script_sha256") != launch.get("script_sha256"):
            out.append(Problem("EXT-LAUNCH", "--launch-record.script_sha256", f"run_args.json script_sha256 "
                               f"{args.get('script_sha256')!r} is not the launch record's (EXT-83)",
                               "collect the run launched with this record's training script"))
        for key, want in DECIDED.items():
            if s.get(key) != want:
                out.append(Problem("EXT-STAGE", "--stage-out", f"run_args.json settings.{key} is {s.get(key)!r}, not "
                                   f"{want!r} (decisions \"Area 8 probes P1/P2\")", "train with the decided settings"))
        if s.get("stage") != args.get("stage") or s.get("full_ft") is not (not s.get("train_expert_only")):
            out.append(Problem("EXT-STAGE", "--stage-out", "run_args.json settings stage/full_ft disagree with its stage "
                               "and train_expert_only", fix))
        steps = cut_steps(s)
        if steps is None or s.get("cut_steps") != steps or args.get("cut_steps") != steps \
                or s.get("save_checkpoint") is not True:
            out.append(Problem("EXT-STAGE", "--stage-out", f"run_args.json cut_steps {args.get('cut_steps')!r} are not "
                               f"every save_freq steps and the last of settings.steps ({steps!r}) with checkpoints on",
                               fix))
        if args.get("lerobot_argv") != lerobot_argv(s, str(args.get("policy_path"))):
            out.append(Problem("EXT-STAGE", "--stage-out", "run_args.json lerobot_argv is not the lerobot-train command "
                               "line the launcher builds from its settings (train_lerobot_logged.py build_lerobot_argv; "
                               "dataset, episodes, backbone, output directory, flags)", fix))
        fw = args.get("framework") if isinstance(args.get("framework"), dict) else {}
        if (fw.get("name"), fw.get("version")) != (FRAMEWORK_PACKAGE, LEROBOT_VERSION):
            out.append(Problem("EXT-FREEZE", "--stage-out", f"run_args.json framework is {fw.get('name')!r} "
                               f"{fw.get('version')!r}, not {FRAMEWORK_PACKAGE} {LEROBOT_VERSION}", fix))
        rt, _ = _read_json(stage_out / RUNTIME_FILE)
        rt = rt if isinstance(rt, dict) else {}
        chk = rt.get("env_freeze_check") if isinstance(rt.get("env_freeze_check"), dict) else {}
        if rt.get("uv_lock_sha256") != VENV_LOCK_SHA256 or chk.get("matches") is not True \
                or args.get("env_freeze_matches_runtime") is not True \
                or rt.get("env_freeze_sha256") != _sha_or_none(stage_out / FREEZE_FILE):
            out.append(Problem("EXT-FREEZE", "--stage-out", f"{RUNTIME_FILE} does not show the pinned venv (uv.lock "
                               f"{VENV_LOCK_SHA256}) with env-freeze.txt equal to the running distributions (MM-32, "
                               "MM-33)", "run in the area venv built from its uv.lock"))
        common = args.get("common_sha256")
        if not (isinstance(common, str) and rt.get("common_sha256") == common):
            out.append(Problem("EXT-LAUNCH", "--stage-out", f"run_args.json common_sha256 {common!r} is not "
                               f"{RUNTIME_FILE}'s: a8_common.py changed during the run", fix))
        # the finished run: LeRobot saved its last step, and the attempt's finish line agrees
        m = CHECKPOINT_DIR_RE.match(Path(last_cut).name)
        last = int(m.group(1)) if m else None
        if not (_int(s.get("steps")) and last == s["steps"]):
            out.append(Problem("EXT-STAGE", "--stage-out", f"the last cut is step {last}, but settings.steps is "
                               f"{s.get('steps')!r}: the run did not save its last step",
                               "collect a run that finished and saved its last step"))
        got_steps = sorted(int(CHECKPOINT_DIR_RE.match(Path(d).name).group(1)) for d in cuts)
        if steps is not None and got_steps != steps:
            out.append(Problem("EXT-STAGE", "--stage-out", f"the complete cuts are steps {got_steps}, not cut_steps "
                               f"{steps}", "collect a run whose every cut was written"))
        lines, why = read_jsonl(stage_out / ATTEMPTS)
        fin = [x for x in (lines or []) if x.get("attempt") == args.get("attempt") and "finished_at" in x]
        if not fin:
            out.append(Problem("EXT-STAGE", "--stage-out", f"{ATTEMPTS} has no finish line of attempt "
                               f"{args.get('attempt')!r} ({why or 'the run did not finish'})",
                               "collect a run that finished"))
        else:
            f = fin[-1]
            if f.get("steps") != s.get("steps") or f.get("cuts") != steps or f.get("blocked_connections") != []:
                out.append(Problem("EXT-STAGE", "--stage-out", f"{ATTEMPTS} finish line: steps {f.get('steps')!r}, cuts "
                                   f"{f.get('cuts')!r}, blocked connections {f.get('blocked_connections')!r}: not the "
                                   "settings' steps and cut steps without a network attempt", fix))
        for d in cuts:
            doc, _ = _read_json(Path(d) / CUT_MARKER)
            doc = doc if isinstance(doc, dict) else {}
            step = doc.get("global_step")
            name = Path(d).name
            if doc.get("seed") != claimed:
                out.append(Problem("EXT-LAUNCH", "--launch-record.training_seed", f"{name}/cut.json seed is "
                                   f"{doc.get('seed')!r}, not the launch record's training_seed {claimed!r} (MM-29)",
                                   "collect the cuts of the run launched with this record"))
            if doc.get("script_sha256") != launch.get("script_sha256") or doc.get("common_sha256") != common:
                out.append(Problem("EXT-LAUNCH", "--launch-record.script_sha256", f"{name}/cut.json script_sha256 or "
                                   "common_sha256 is not the launch record's and the run's (EXT-83)",
                                   "collect the cuts of the run launched with this record's training script"))
            if (doc.get("profile"), doc.get("stage")) != (profile, args.get("stage")):
                out.append(Problem("EXT-PROFILE", "--launch-record.profile", f"{name}/cut.json profile/stage "
                                   f"{doc.get('profile')!r}/{doc.get('stage')!r} are not the run's",
                                   "collect the run launched with this record's profile"))
            if doc.get("batch_size") != s.get("batch_size") or doc.get("consumption_log") != SAMPLES or (
                    _int(step) and _int(s.get("steps"))
                    and doc.get("lerobot_step_dir") != f"{LEROBOT_DIR}/checkpoints/{lerobot_steps_dir(step, s['steps'])}"):
                out.append(Problem("EXT-STAGE", "--stage-out", f"{name}/cut.json batch_size, consumption_log or "
                                   "lerobot_step_dir is not the run's", fix))
            cfg, _ = _read_json(Path(d) / PRETRAINED / "config.json")
            cfg = cfg if isinstance(cfg, dict) else {}
            want_cfg = {"type": "smolvla", "vlm_model_name": VLM_DIR, "load_vlm_weights": False,
                        "train_expert_only": True, "freeze_vision_encoder": True,
                        "n_action_steps": DECIDED["n_action_steps"]}
            if {k: cfg.get(k) for k in want_cfg} != want_cfg:
                out.append(Problem("EXT-STAGE", "--stage-out", f"{name}/{PRETRAINED}/config.json is not a SmolVLA policy "
                                   f"with the local SmolVLM2 backbone, expert-only training and n_action_steps 50", fix))
        return out

    def sampler_problems(self, cuts: list[Path], sampler: Any) -> list[Problem]:
        """EXT-240: every cut's sampler.json is the registered sampler, its config-dependent keys are the cut's
        config.json (a8_common.deploy_sampler), and the run's run_args.json sampler is the same."""
        from gatedtrain.ext import arm

        out = []
        fix = "register eval_libero.py's sampler, the deployment sampler every cut carries (sampler.json)"
        for d in cuts:
            doc, _ = _read_json(Path(d) / SAMPLER_FILE)
            cfg, _ = _read_json(Path(d) / PRETRAINED / "config.json")
            if not isinstance(sampler, dict) or not arm.same_json(doc, sampler):
                out.append(Problem("EXT-SAMPLER", "registration.sampler", f"{Path(d).name}/{SAMPLER_FILE} is not the "
                                   "registered sampler: the cut deploys another sampler than the gate evaluates", fix))
                continue
            cfg = cfg if isinstance(cfg, dict) else {}
            derived = {"policy_type": cfg.get("type"), "n_action_steps": cfg.get("n_action_steps"),
                       "chunk_size": cfg.get("chunk_size"), "num_steps": cfg.get("num_steps"),
                       "max_action_dim": cfg.get("max_action_dim"),
                       "resize_imgs_with_padding": cfg.get("resize_imgs_with_padding"),
                       "tokenizer_max_length": cfg.get("tokenizer_max_length"), "rename_map": RENAME}
            if any(not arm.same_json(doc.get(k), v) for k, v in derived.items()):
                out.append(Problem("EXT-SAMPLER", "--stage-out", f"{Path(d).name}/{SAMPLER_FILE} is not "
                                   "a8_common.deploy_sampler of its config.json", "collect the cuts as the launcher "
                                   "wrote them"))
        if cuts:
            args, _ = read_run_args(Path(cuts[0]).parent.parent)
            if isinstance(args, dict) and isinstance(sampler, dict) and not (
                    arm.same_json(args.get("sampler"), sampler)
                    and args.get("sampler_sha256") == compact_sha256(args.get("sampler"))):
                out.append(Problem("EXT-SAMPLER", "--stage-out", "run_args.json sampler (and sampler_sha256) is not the "
                                   "registered sampler", fix))
        return out

    # --- arms (MM-1, MM-2, MM-4, MM-19; EXT-241) ---

    @staticmethod
    def arm_env_version_sha256(arm: Path) -> str:
        """MM-19: eval_libero.py's `env_version_sha256`: SHA-256 of the compact sorted JSON of the arm's
        env_version.json. Raises OSError or ValueError."""
        doc = json.loads((Path(arm) / "env_version.json").read_text(encoding="utf-8"))
        return compact_sha256(doc)

    def arm_problems(self, arm: Path, records: list[dict], item_rows: list[dict], items_dir: Path | None = None,
                     model_dir: Path | None = None) -> list[str]:
        """EXT-241: the evaluation beside the arm (eval_libero.py's run directory) evaluated the registered items
        (env-runtime.json items_sha256) and the cut the arm must name (summary.json: model files, normalizer, step,
        profile, stage; no non-default episode settings; the default item timeout); env_version.json in the arm and
        beside it have the arm's hash; every line agrees with its item (task, init state), its outcome and its episode log
        (seeds re-derived: common random numbers), and the error reasons agree with the summary."""
        arm = Path(arm)
        root = arm.parent
        out: list[str] = []
        rt, why = _read_json(root / RUNTIME_FILE)
        if not isinstance(rt, dict):
            return [f"{root / RUNTIME_FILE} (eval_libero.py writes it beside the arm) cannot be read: {why}"]
        if items_dir is not None:
            want = _sha_or_none(Path(items_dir) / "items.jsonl")
            if rt.get("items_sha256") != want:
                out.append(f"{RUNTIME_FILE} items_sha256 {rt.get('items_sha256')!r} is not the registered item set's "
                           f"items.jsonl {want}: the evaluation read {rt.get('items_dir')!r}")
        if rt.get("checkpoint") != EVAL_CUT_CONTAINER:
            out.append(f"{RUNTIME_FILE} checkpoint {rt.get('checkpoint')!r} is not {EVAL_CUT_CONTAINER}")
        code = {"eval_libero.py": rt.get("script_sha256"), "a8_common.py": rt.get("common_sha256")}
        if compact_sha256(code) != rt.get("code_sha256") or rt.get("code_sha256") != \
                EVALUATOR_ROWS[EVALUATOR_ID]["scorer"]["code_sha256"]:
            out.append(f"{RUNTIME_FILE} code_sha256 is not the registered scorer's composite of eval_libero.py and "
                       "a8_common.py")
        meta, _ = _read_json(arm / "eval_meta.json")
        meta = meta if isinstance(meta, dict) else {}
        if meta.get("framework") != {"name": FRAMEWORK_PACKAGE, "version": LEROBOT_VERSION}:
            out.append(f"eval_meta.json framework is {meta.get('framework')!r}, not {FRAMEWORK_PACKAGE} {LEROBOT_VERSION}")
        try:
            envs = {compact_sha256(json.loads((p / "env_version.json").read_text(encoding="utf-8")))
                    for p in (arm, root)}
        except (OSError, ValueError) as exc:
            envs = {f"unreadable ({exc.__class__.__name__})"}
        if envs != {meta.get("env_version_sha256")}:
            out.append("env_version.json in the arm and beside it do not both hash to eval_meta.json "
                       "env_version_sha256 (compact sorted JSON)")
        summ, why = _read_json(root / "summary.json")
        summ = summ if isinstance(summ, dict) else {}
        if model_dir is not None and self.is_cut(model_dir):
            cut, _ = _read_json(Path(model_dir) / CUT_MARKER)
            cut = cut if isinstance(cut, dict) else {}
            for key, want in (("cut_model_files_sha256", cut.get("model_files_sha256")),
                              ("normalizer_sha256", cut.get("normalizer_sha256")),
                              ("global_step", cut.get("global_step")), ("profile", cut.get("profile")),
                              ("stage", cut.get("stage"))):
                if summ.get(key) != want:
                    out.append(f"summary.json {key} {summ.get(key)!r} is not the evaluated cut's {want!r} "
                               f"({Path(model_dir).name}): the arm evaluated another cut")
                    break
        if summ.get("env_version_nondefault") != {} or summ.get("item_timeout_s") != ITEM_TIMEOUT_S \
                or summ.get("eval_seeds") != meta.get("eval_seeds"):
            out.append("summary.json shows non-default episode settings, another item timeout than "
                       f"{ITEM_TIMEOUT_S:g} s or other eval seeds than eval_meta.json")
        errs = [r for r in records if r.get("error") is not None]
        reasons = {k: sum(1 for r in errs if str(r["error"].get("message", "")).startswith(k + ":"))
                   for k in ERROR_REASONS}
        if summ.get("episodes") != len(records) or summ.get("errors") != len(errs) or summ.get("error_reasons") != reasons:
            out.append(f"summary.json episodes {summ.get('episodes')!r}, errors {summ.get('errors')!r} and error reasons "
                       f"{summ.get('error_reasons')!r} are not the arm's {len(records)}, {len(errs)}, {reasons}")
        nas = (meta.get("sampler") or {}).get("n_action_steps") if isinstance(meta.get("sampler"), dict) else None
        by_item = {r.get("item"): r for r in item_rows}
        for rec in records:
            msg = self._record_problem(arm, rec, by_item.get(rec.get("item")), nas)
            if msg:
                out.append(msg)
            if len(out) >= 8:
                break
        return out[:8]

    @staticmethod
    def _record_problem(arm: Path, rec: dict, row: dict | None, nas: Any) -> str | None:
        item, seed, err = rec.get("item"), rec.get("seed"), rec.get("error")
        if row is None:
            return None                                  # an unknown item: the gate reports it (MM-6)
        where = f"line {item!r} r{rec.get('repeat')}"
        im = rec.get("item_meta") if isinstance(rec.get("item_meta"), dict) else {}
        if (im.get("task_id"), im.get("init_state")) != (row.get("task_id"), row.get("init_state")):
            return f"{where}: item_meta task/init state {im.get('task_id')!r}/{im.get('init_state')!r} is not the item's"
        sub = rec.get("sub_metrics") if isinstance(rec.get("sub_metrics"), dict) else {}
        if err is not None:
            if im.get("outcome") is not None or not str(err.get("message", "")).strip() or set(sub) != {"wall_s"}:
                return f"{where}: an error line has an outcome, an empty message or score sub-metrics"
        else:
            o, score = im.get("outcome"), rec.get("score")
            if o not in OUTCOMES or score != (1 if o == "success" else 0):
                return f"{where}: outcome {o!r} and score {score!r} disagree (1 exactly for success; MM-4)"
            if sorted(sub) != sorted(SUB_METRICS):
                return f"{where}: sub_metrics are not {list(SUB_METRICS)}"
            steps = sub["steps"]
            ok = (_int(steps) and 0 <= steps <= MAX_STEPS and sub["timeout"] == int(o == "timeout")
                  and sub["nan_action"] == int(o == "nan_action")
                  and sub["steps_to_success"] == (steps if o == "success" else MAX_STEPS)
                  and (o != "timeout" or steps == MAX_STEPS)
                  and (not _int(nas) or sub["n_policy_calls"] == (math.ceil(steps / nas) if steps
                                                                  else int(o == "nan_action"))))
            if not ok:
                return (f"{where}: sub_metrics {sub} disagree with outcome {o!r} (steps <= {MAX_STEPS}, steps_to_success, "
                        "timeout, nan_action, n_policy_calls = ceil(steps / n_action_steps))")
        ref = rec.get("output_ref")
        if not isinstance(ref, dict):
            return f"{where}: no episode log (output_ref)"
        log, why = _read_json(arm / ref["path"])
        if not isinstance(log, dict):
            return f"{where}: episode log {ref['path']}: {why}"
        want = {"schema": EPISODE_SCHEMA, "item": item, "repeat": rec.get("repeat"), "eval_seed": seed,
                "task_id": row.get("task_id"), "init_state": row.get("init_state"),
                "env_seed": derive_seed("libero-env", seed, item) % 2**31,
                "torch_seed": derive_seed("libero-torch", seed, item),
                "noise_seed": derive_seed("libero-noise", seed, item), "error": err}
        if err is None:
            want.update(outcome=im.get("outcome"), steps=sub.get("steps"), steps_to_success=sub.get("steps_to_success"),
                        n_policy_calls=sub.get("n_policy_calls"))
        diff = sorted(k for k, v in want.items() if log.get(k) != v)
        if diff:
            return (f"{where}: episode log {ref['path']} differs in {diff} (the item, its seeds derive_seed('libero-*', "
                    "seed, item) or its result)")
        return None

    def report_notes(self, records_by_arm: dict[str, list[dict]]) -> list[str]:
        """Decision "Area 8 (VLA) decisions" (not published): per-task success of both arms (scored episodes), and
        the within-task clustering the per-item paired analysis does not model (08 section 7 P6b), as a disclosed gap."""
        parts = {}
        for arm, recs in records_by_arm.items():
            per: dict[int, list[float]] = {}
            for r in recs:
                m = ITEM_RE.fullmatch(str(r.get("item")))
                if m and r.get("score") is not None:
                    per.setdefault(int(m.group(1)), []).append(r["score"])
            parts[arm] = per
        tasks = sorted(set().union(*(set(p) for p in parts.values()))) if parts else []
        cells = []
        for t in tasks:
            vals = [(sum(parts[a][t]) / len(parts[a][t])) if parts.get(a, {}).get(t) else None
                    for a in ("baseline", "candidate")]
            cells.append(f"t{t} " + "/".join("n/a" if v is None else f"{v:.3f}" for v in vals))
        return [f"Per-task success (baseline/candidate, scored episodes): {'; '.join(cells) or 'none'}.",
                "Clustering: items of one task share the task and its objects; the gate's per-item paired analysis does "
                "not cluster by task (08 section 7 P6b, gap); read the per-task line before attributing a gain."]

    # --- launch rules (MM-31 to MM-36; EXT-243, EXT-244) ---

    @staticmethod
    def _script(argv: Any) -> tuple[str | None, int]:
        """(script file name, its argv index): argv[1], or argv[2] after the one allowed `-I` (EXT-199)."""
        from gatedtrain.ext.launch import script_index as _script_index
        from gatedtrain.ext.registry import interpreter_flags

        if not (isinstance(argv, list) and len(argv) > 1 and all(isinstance(a, str) for a in argv)):
            return None, 1
        si = _script_index(argv, interpreter_flags(NAME))
        return posixpath.basename(argv[si]), si

    def argv_problems(self, argv: Any, *, profile: str | None = None) -> list[str]:
        """EXT-243 (ext-launch, --recheck and ext-collect). argv[1] (after an optional `-I`) is train_lerobot_logged.py
        or eval_libero.py at its path in the read-only runtime mount; every token is an option of that script (a unique
        prefix counts, as argparse resolves it); `--out /out`. Training: `--profile` smoke or full under the launch
        profile of the same name (stage 2, with `--policy-path /inputs/base-cut`), or base under the launch profile
        smoke (stage 1, no `--policy-path`; never collected); `--seed` given; `--dry-run`, `--full-ft`, another device,
        batch size or n_action_steps than decided refused. Evaluation: `--checkpoint /inputs/cut`, `--items` gate or
        probe (gate under the full profile), no `--write-items`, `--max-steps`, `--settle-steps`, another item timeout,
        `--workers 0` or another device; `--limit` and `--per-task` refused under the full profile."""
        script, si = self._script(argv)
        if script is None:
            return []                                     # bind_argv reports the shape
        if script not in OPTIONS or argv[si] != SCRIPT_PATHS[script]:
            return [f"argv[{si}] {argv[si]!r} is not one of the runtime's scripts {sorted(SCRIPT_PATHS.values())} "
                    "(EXT-243)"]
        parsed, odd = _parse_argv(argv[si + 1:], OPTIONS[script])
        out = [f"{script} argument {tok!r} is not one of its options (argparse refuses it; a unique prefix counts)"
               for tok in odd]
        given: dict[str, list[str]] = {}
        for opt, values, _ in parsed:
            given[opt] = values                           # argparse keeps the last value
        one = {k: (v[0] if v else None) for k, v in given.items()}
        if one.get("--out") != OUT:
            out.append(f"{script} --out {one.get('--out')!r} is not {OUT} (the one read-write mount)")
        if one.get("--device", "cuda") != "cuda":
            out.append(f"{script} --device {one.get('--device')!r}: the GPU runs use cuda (cpu is for host tests)")
        if script == TRAIN_SCRIPT:
            allowed = {"smoke": ("smoke", "base"), "full": ("full",)}.get(profile, ("smoke", "full", "base"))
            prof = one.get("--profile")
            if prof not in allowed:
                out.append(f"{script} --profile {prof!r} is not {' or '.join(allowed)} under the launch profile "
                           f"{profile!r} (stage 2: smoke, full; stage 1: base under smoke; EXT-243)")
            if prof in STAGE_PROFILES and _norm_path(one.get("--policy-path")) != BASE_CUT_CONTAINER:
                out.append(f"{script} --profile {prof} continues from the base cut: --policy-path must be "
                           f"{BASE_CUT_CONTAINER}, not {one.get('--policy-path')!r}")
            if prof == "base" and "--policy-path" in given:
                out.append(f"{script} --profile base trains from smolvla_base: no --policy-path")
            if not re.fullmatch(r"[0-9]+", one.get("--seed") or ""):
                out.append(f"{script} --seed is {one.get('--seed')!r}: the training seed is given explicitly (MM-29; the "
                           "script's default 1000 is not taken implicitly)")
            for opt, why in (("--dry-run", "writes nothing"), ("--full-ft", "the decision is expert-only training")):
                if opt in given:
                    out.append(f"{script} {opt} is refused: {why}")
            for opt, key in (("--batch-size", "batch_size"), ("--n-action-steps", "n_action_steps")):
                if opt in given and one[opt] != str(DECIDED[key]):
                    out.append(f"{script} {opt} {one[opt]!r} is not the decided {DECIDED[key]} (decisions \"Area 8 "
                               "probes P1/P2\")")
        else:
            if one.get("--checkpoint") != EVAL_CUT_CONTAINER:
                out.append(f"{EVAL_SCRIPT} --checkpoint {one.get('--checkpoint')!r} is not {EVAL_CUT_CONTAINER} (the "
                           "cut's read-only mount)")
            items = one.get("--items", "probe")
            allowed_items = ("gate",) if profile == "full" else tuple(ITEM_SETS)
            if items not in allowed_items:
                out.append(f"{EVAL_SCRIPT} --items {items!r} is not {' or '.join(allowed_items)}: the item sets are "
                           "the registered directories inside the read-only runtime (gate for the gate arms)")
            for opt in ("--write-items", "--max-steps", "--settle-steps"):
                if opt in given:
                    out.append(f"{EVAL_SCRIPT} {opt} is refused: item writing and episode settings are host-test only "
                               "(a non-default episode changes env_version)")
            if "--item-timeout-s" in given:
                try:
                    ok = float(one["--item-timeout-s"]) == ITEM_TIMEOUT_S
                except (TypeError, ValueError):
                    ok = False
                if not ok:
                    out.append(f"{EVAL_SCRIPT} --item-timeout-s {one['--item-timeout-s']!r} is not "
                               f"{ITEM_TIMEOUT_S:g}: the registered scorer version names it")
            if one.get("--workers") is not None and one["--workers"] in ("0", "-0"):
                out.append(f"{EVAL_SCRIPT} --workers 0 runs in process without the item timeout (host tests only)")
            if profile == "full":
                for opt in ("--limit", "--per-task"):
                    if opt in given:
                        out.append(f"{EVAL_SCRIPT} {opt} under --profile full: a gate arm evaluates the whole "
                                   "registered item set")
        return out

    def launch_problems(self, doc: dict, *, kind: str) -> list[tuple[str, str]]:
        """EXT-244: [(field, message)] of this adapter's launch rules (request or record). Every launch runs the area
        image (`venv_in_container`, the pinned image ID) with the area venv lock (pinned SHA-256), LeRobot 0.7.0, on the
        GPU, with network none; the training `--seed` is the record's training_seed; the mounts are exactly the
        interpreter tree and the runtime read-only at their own paths and /out read-write, plus the base cut read-only
        at /inputs/base-cut for stage 2 or the evaluated cut at /inputs/cut for an evaluation; environment values
        (request only) are the area's (offline switches, EGL rendering, the dataset home and LIBERO configuration); an
        evaluation sets the four rendering names."""
        out: list[tuple[str, str]] = []
        script, si = self._script(doc.get("argv"))
        if script not in OPTIONS:
            return out                                    # bind_argv and argv_problems report the script
        rt = doc.get("runtime") if isinstance(doc.get("runtime"), dict) else {}
        if rt.get("kind") != "venv_in_container" or rt.get("image") != IMAGE:
            out.append(("runtime", f"{script} runs in the area image: runtime {{kind: venv_in_container, image: "
                        f"{IMAGE}}} (gatedtrain-a8-runtime:1), not {rt.get('kind')!r} {rt.get('image')!r}"))
        if kind == "request":
            lock = rt.get("venv_lock")
            got = _sha_or_none(Path(lock)) if isinstance(lock, str) and Path(lock).is_file() else None
        else:
            got = rt.get("venv_lock_sha256")
        if got != VENV_LOCK_SHA256:
            out.append(("runtime", f"the venv lock is {got!r}, not the area's uv.lock {VENV_LOCK_SHA256} (MM-32)"))
        fw = doc.get("framework") if isinstance(doc.get("framework"), dict) else {}
        if (fw.get("name"), fw.get("version")) != (FRAMEWORK_PACKAGE, LEROBOT_VERSION):
            out.append(("framework", f"framework {fw.get('name')!r} {fw.get('version')!r} is not "
                        f"{FRAMEWORK_PACKAGE} {LEROBOT_VERSION}"))
        if doc.get("network") != "none":
            out.append(("network", f"{script} runs with network none (every input is mounted), not "
                        f"{doc.get('network')!r}"))
        if doc.get("gpus") is not True:
            out.append(("gpus", f"{script} needs the GPU (gpus true)"))
        argv = doc["argv"]
        parsed, _ = _parse_argv(argv[si + 1:], OPTIONS[script])
        one = {o: (v[0] if v else None) for o, v, _ in parsed}
        want = list(MOUNTS)
        cut_mount = None
        if script == TRAIN_SCRIPT:
            seed = doc.get("training_seed")
            if not (_int(seed) and one.get("--seed") == str(seed)):
                out.append(("argv", f"{script} --seed {one.get('--seed')!r} is not the training_seed {seed!r} (MM-29)"))
            if "--policy-path" in one:
                cut_mount = BASE_CUT_CONTAINER
        else:
            cut_mount = EVAL_CUT_CONTAINER
        if cut_mount:
            want.append((cut_mount, "ro"))
        mounts = doc.get("mounts") if isinstance(doc.get("mounts"), list) else []
        got_m = sorted((_norm_path(m.get("container")), m.get("mode")) for m in mounts if isinstance(m, dict))
        if got_m != sorted(want) or len(got_m) != len(mounts):
            out.append(("mounts", f"{script} mounts {got_m}, not exactly {sorted(want)}"))
        for m in mounts:
            if isinstance(m, dict) and cut_mount and _norm_path(m.get("container")) == cut_mount:
                cdoc, why = _read_json(Path(str(m.get("host"))) / CUT_MARKER)
                if not (isinstance(cdoc, dict) and cdoc.get("schema") == CUT_SCHEMA):
                    out.append(("mounts", f"{cut_mount} mounts {m.get('host')!r}, which holds no {CUT_SCHEMA} cut.json "
                                f"({why or 'another schema'})"))
                elif cut_mount == BASE_CUT_CONTAINER and cdoc.get("stage") != "base":
                    out.append(("mounts", f"{BASE_CUT_CONTAINER} mounts a {cdoc.get('stage')!r} cut: stage 2 continues "
                                "from a stage-1 (base) cut (EXT-236)"))
        names = (sorted(doc.get("env")) if kind == "request" and isinstance(doc.get("env"), dict)
                 else doc.get("env_names") if isinstance(doc.get("env_names"), list) else [])
        if script == EVAL_SCRIPT:
            missing = [n for n in EVAL_ENV_REQUIRED if n not in names]
            if missing:
                out.append(("env", f"{EVAL_SCRIPT} renders with EGL on the GPU: set {', '.join(missing)} (README section "
                            "0.4, P1)"))
        if kind == "request" and isinstance(doc.get("env"), dict):
            for name, value in sorted(doc["env"].items()):   # values are never shown (MM-35)
                if name in ENV_VALUES and value != ENV_VALUES[name]:
                    out.append((f"env.{name}", f"{name} is not the value the area runs with (README section 11.3)"))
                if name == "MUJOCO_EGL_DEVICE_ID" and not re.fullmatch(r"[0-9]+", str(value)):
                    out.append((f"env.{name}", f"{name} is not a device number"))
        return out

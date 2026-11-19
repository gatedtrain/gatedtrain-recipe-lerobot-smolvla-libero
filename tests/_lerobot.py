"""Area "VLA" test helpers (verifier-owned; tests/README.md section 11.26, EXT-233 to EXT-247).

Written from the recipe's runtime (`runtime/`: README.md sections 2, 4 to 6 and 11;
`train_lerobot_logged.py` `build_lerobot_argv`, `write_cut`, `main`; `a8_common.py` `deploy_sampler`,
`normalizer_record`, `model_file_names`; `eval_libero.py` `evaluate`, `score_record`, `make_jobs`; read 2026-11-15) and its
GPU outputs, not from gatedtrain. Real pieces are copied into `fixtures/lerobot/`:
- `checkpoint-500/`: the small files of the gate's base cut `base-1/cuts/checkpoint-500/` (config.json, both
  processor descriptions and their normalizer state files, train_config.json, the tokenizer config and chat template,
  training_state/ JSON and rng state, sampler.json, base_ref.json). The 1.2 GB weights, the 413 MB optimizer state and
  the 3.5 MB tokenizer.json are replaced by stand-ins here. Its cut.json is not published: it records the hashes of
  files whose copies here had host paths replaced (below), so it would let anyone confirm a guessed path;
- `base-1/`: the stage-1 run's run_args.json, env-runtime.json, env-freeze.txt, attempts.jsonl, cuts.jsonl (copies)
  and the first 3 lines of its samples.jsonl;
  host paths in these copies (and in checkpoint-500's base_ref.json, config.json, train_config.json and eval2-500's
  env-runtime.json) were replaced by the neutral layout `/srv/gatedtrain/lerobot-smolvla-libero/runtime` (the runtime) and
  `/srv/python` (the interpreter directory); for the same reason, `env_freeze_sha256` (the hash of the original
  env-freeze.txt) is removed from the env-runtime.json copy (a scenario run records its own);
- `eval2-500/`: the probe evaluation of checkpoint-500 by eval_libero.py 59cd2182 (env-runtime.json, summary.json,
  env_version.json; the arm's eval_meta.json, env_version.json, items.jsonl and the episode log of libero_object/t0/s40);
- `items/items-gate.jsonl`, `items/items-probe.jsonl` (the registered item sets, 400 and 100 rows) and `area8-uv.lock`
  (a copy of `runtime/uv.lock`, the pinned hash; the published run used the same lock before the runtime project was
  renamed, RESULT.md).

The runtime path: the container sees the runtime at its own host path (identity mount), so the adapter's paths are
`rows.RUNTIME`'s. When that runtime has no venv (a fresh checkout), this module points
`GATEDTRAIN_RECIPE_LEROBOT_SMOLVLA_LIBERO_RUNTIME` at a stand-in runtime (a temporary `tests/.stand-in-runtime-*/` holding only `.venv/pyvenv.cfg`,
whose interpreter tree is the neutral `/srv/python/cpython-3.12.16-linux-x86_64-gnu`) before the recipe is imported; the
test CLI subprocesses inherit it. The fixtures were recorded with the original runtime path; the scenario rewrites that
path (`RECORDED_AREA8`, read from the fixture's run_args.json) to the runtime's (`localize`). Real-file tests (EXT-247)
read the real runtime (`REAL_RUNTIME`) and the GPU outputs (`GATEDTRAIN_RECIPE_LEROBOT_SMOLVLA_LIBERO_OUTPUTS`, the directory
holding base-1/ and base-1-eval2-500/; set the runtime override to the runtime those runs used as well) and skip where
absent.

`LerobotScenario(root)` builds a mirrored host (`GATEDTRAIN_EXT_MIRROR_ROOT` = `<root>/host` stands for the container's
`/`, EXT-112), so every container path is the real one:
- the interpreter tree's `bin/python3.12` (stand-in) and the runtime's `.venv/bin/python` linking to it; the runtime: the
  two scripts and a8_common.py (stand-ins, never executed), uv.lock (the real bytes), `items-gate/`
  (the first 4 init states of each task: 40 items) and `items-probe/`, a miniature `data/lerobot/libero/data/chunk-000/`
  (the 454 LIBERO-Object episodes 807-1260 with 2 frames each, 908 frames, in three parquet files that also hold one
  frame each of the neighbour episodes 806 and 1261), and `frames/`, the frame export of those files (the spec's data);
- `<output root>/lerobot-smolvla-libero/base-1/`: a stage-1 run (40 steps, cuts 20 and 40; checkpoint-20 is the spec's base);
  `full-1/`: the stage-2 run from it (seed 2000, 60 steps at batch 32: 29 steps per epoch, the 29th carrying 12; cuts 30
  and 60); `eval-base-1/` and `eval-cand-1/`: the two gate evaluations as eval_libero.py writes them (40 items x K 2);
- `<root>/`: spec, registration, trainer.yaml, the stage-2 launch record, a smoke record, the host work's GPU log.
"""

from __future__ import annotations

import atexit
import copy
import hashlib
import json
import math
import os
import random
import shutil
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import yaml

REPO = Path(__file__).resolve().parents[1]
RUNTIME_ENV = "GATEDTRAIN_RECIPE_LEROBOT_SMOLVLA_LIBERO_RUNTIME"
OUTPUTS_ENV = "GATEDTRAIN_RECIPE_LEROBOT_SMOLVLA_LIBERO_OUTPUTS"
NEUTRAL_TREE = "/srv/python/cpython-3.12.16-linux-x86_64-gnu"
# The real runtime (the override, else this checkout's runtime/): its committed scripts, lock, manifests and item sets
# are read by the real-file tests even when the scenario runs on the stand-in.
REAL_RUNTIME = Path(os.environ.get(RUNTIME_ENV) or REPO / "runtime")


def _stand_in_runtime() -> None:
    """Point the runtime override at a stand-in runtime when the configured one has no venv (module docstring)."""
    runtime = Path(os.environ.get(RUNTIME_ENV) or REPO / "runtime")
    if (runtime / ".venv" / "pyvenv.cfg").is_file():
        return
    if "gatedtrain_recipe_lerobot_smolvla_libero.rows" in sys.modules:
        raise RuntimeError(f"_lerobot must be imported before the recipe's rows (it sets {RUNTIME_ENV})")
    # Not under /tmp: the container's /tmp is a tmpfs, which no mount may nest in (EXT-88).
    tmp = Path(tempfile.mkdtemp(prefix=".stand-in-runtime-", dir=Path(__file__).resolve().parent))
    atexit.register(shutil.rmtree, tmp, True)
    (tmp / ".venv").mkdir()
    (tmp / ".venv" / "pyvenv.cfg").write_text(f"home = {NEUTRAL_TREE}/bin\nversion_info = 3.12.16\n", encoding="utf-8")
    os.environ[RUNTIME_ENV] = str(tmp)


_stand_in_runtime()
from gatedtrain_recipe_lerobot_smolvla_libero import rows as R  # noqa: E402  (after the runtime override)

from gatedtrain.testing.ext import ENV_HOST_WORK, ENV_INPUT_ROOTS, ENV_INTERPRETER_TREES, ENV_MIRROR_ROOT, ENV_OUTPUT_ROOT, gpu_step_script, host_config_sha256, launch_label, sha_of, write_canonical
EXT = Path(__file__).resolve().parent / "fixtures"     # the recipe's fixtures (gatedtrain: tests/fixtures/ext)
from gatedtrain.testing.support import canonical_bytes, file_sha256, manifest_entries, manifest_hash, run_cli
from gatedtrain.ext.arm import META_SCHEMA
from gatedtrain.ext.launch import RECORD_SCHEMA
from gatedtrain.ext.smoke import SMOKE_SCHEMA

LR = EXT / "lerobot"
REAL_CUT = LR / "checkpoint-500"
REAL_RUN = LR / "base-1"
REAL_EVAL = LR / "eval2-500"
ADAPTER = "lerobot_smolvla"
TRAINER_ID = "sft.lerobot@0.1"
EVALUATOR_ID = "libero.lerobot_items@0.1"
METRIC = "success"
AREA8 = R.RUNTIME                                     # the runtime, mounted at its own path
CPYTHON = R.HOST_ROWS["interpreter_trees"][0]         # the interpreter tree its venv links to
PYTHON_DIR = R.PYTHON_DIR                             # its parent, mounted at its own path
OUT_ROOT = "/srv/out"                                 # the scenario's output root (container view of the mirror)
OUT_NAME = R.OUT_NAME
# The real GPU outputs (base-1/, base-1-eval2-500/) for the real-file tests; absent unless OUTPUTS_ENV names them.
REAL_OUT = Path(os.environ.get(OUTPUTS_ENV) or "/nonexistent/lerobot-smolvla-libero-outputs")
# The runtime path the fixtures were recorded with (stage 1's policy path is <runtime>/models/smolvla_base).
RECORDED_AREA8 = json.loads((REAL_RUN / "run_args.json").read_text(encoding="utf-8"))["policy_path"].rsplit(
    "/models/", 1)[0]


def localize(obj: Any) -> Any:
    """`obj` (a fixture's JSON value or text) with the recorded runtime path replaced by this runtime's."""
    if isinstance(obj, str):
        return obj.replace(RECORDED_AREA8, AREA8)
    if isinstance(obj, list):
        return [localize(v) for v in obj]
    if isinstance(obj, dict):
        return {k: localize(v) for k, v in obj.items()}
    return obj


def read_fixture_json(path: Path) -> Any:
    return localize(json.loads(path.read_text(encoding="utf-8")))


def copy_fixture_text(src: Path, dst: Path) -> None:
    dst.write_text(localize(src.read_text(encoding="utf-8")), encoding="utf-8")
INTERPRETER = f"{AREA8}/.venv/bin/python"
TRAIN_PY = f"{AREA8}/train_lerobot_logged.py"
EVAL_PY = f"{AREA8}/eval_libero.py"
IMAGE = "sha256:136745e51640ee7bf92760d3dc3f8764187f3c409e4f2c928ce4c39707cd5242"      # gatedtrain-a8-runtime:1 (built as ptc-a8-runtime:1)
LOCK_SHA = "sha256:2a87da026100743dccb92e5c084e64941ff6346cedf2f40ae774e13dfac4b9a6"   # runtime/uv.lock
EVAL_LIBERO_SHA = "sha256:7846982d99e1974f92e5e561f11175d34fb48aaba7b522fc570dbb5856c1e7ec"
COMMON_SHA = "sha256:ce29aa22024a33e08ccbd31c7b2c8ad9a1b5c344e6d1bcbe3486da5d7481f8e6"
TRAIN_SHA = "sha256:93bf86f66a71dc943879c22564655eeda3b95ba843c17aabda7292820fa30894"
# The scripts that wrote the real outputs (before their re-pins for ptc-hub 0.2.0 and gatedtrain 0.3.0, RESULT.md).
PRE_RENAME_EVAL_LIBERO_SHA = "sha256:59cd2182939208d93ddb1bbb0ffff91746d3acd46546e0b65f84c8ba178bf0af"
PRE_RENAME_COMMON_SHA = "sha256:d01995ca744109caeee66fc4e40688d28270ddf8f9e2770ab20cf8d28d2ea23d"
BASE_CUT_CONTAINER = "/inputs/base-cut"
SMOLVLA_DIR = f"{AREA8}/models/smolvla_base"
VLM_DIR = f"{AREA8}/models/SmolVLM2-500M-Video-Instruct"
DATASET_ROOT = f"{AREA8}/data/lerobot/libero"
DATASET_REPO, DATASET_REVISION = "lerobot/libero", "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
EPISODES = (807, 1260)
RENAME = {"observation.images.image": "observation.images.camera1",
          "observation.images.image2": "observation.images.camera2"}
# The real arm's scorer (eval_libero.py 59cd2182, a8_common.py d01995ca) with the code hash of the scripts re-pinned
# for ptc-hub 0.2.0 and gatedtrain 0.3.0 (RESULT.md); every other scorer field is unchanged.
REAL_SCORER = json.loads((REAL_EVAL / "arm-success" / "eval_meta.json").read_text(encoding="utf-8"))["scorer"]
SCORER = {**REAL_SCORER, "code_sha256": "sha256:" + hashlib.sha256(json.dumps(
    {"eval_libero.py": EVAL_LIBERO_SHA, "a8_common.py": COMMON_SHA}, sort_keys=True, separators=(",", ":"),
    ensure_ascii=False).encode("utf-8")).hexdigest()}
SAMPLER = json.loads((REAL_CUT / "sampler.json").read_text(encoding="utf-8"))
ENV_VERSION = json.loads((REAL_EVAL / "env_version.json").read_text(encoding="utf-8"))
BASE_REF = read_fixture_json(REAL_CUT / "base_ref.json")
SEEDS = [11, 12]
SEED = 2000                                           # the stage-2 training seed of the scenario
BASE_SEED = 1000                                      # the stage-1 run's (the launcher's default)
NORMALIZER = "pretrained_model/policy_preprocessor_step_5_normalizer_processor.safetensors"
UNNORMALIZER = "pretrained_model/policy_postprocessor_step_0_unnormalizer_processor.safetensors"
DEPLOYABLE = ("pretrained_model/config.json", "pretrained_model/model.safetensors",
              "pretrained_model/policy_postprocessor.json", UNNORMALIZER, "pretrained_model/policy_preprocessor.json",
              NORMALIZER, "pretrained_model/tokenizer/chat_template.jinja", "pretrained_model/tokenizer/tokenizer.json",
              "pretrained_model/tokenizer/tokenizer_config.json", "sampler.json")
RESUME = ("pretrained_model/train_config.json", "training_state/optimizer_param_groups.json",
          "training_state/optimizer_state.safetensors", "training_state/rng_state.safetensors",
          "training_state/scheduler_state.json", "training_state/training_step.json")
CUT_FILES = tuple(sorted(DEPLOYABLE + RESUME + ("base_ref.json",)))
SCRIPT_TEXT = {"train_lerobot_logged.py": f"# fixture stand-in for {TRAIN_PY}; never executed\n",
               "eval_libero.py": f"# fixture stand-in for {EVAL_PY}; never executed\n",
               "a8_common.py": f"# fixture stand-in for {AREA8}/a8_common.py; never executed\n"}
LABEL = "mm-a8-vla-full"
AREA = 8
ENV = {"HF_DATASETS_OFFLINE": "1", "HF_HOME": "/tmp/hf-home", "HF_HUB_OFFLINE": "1", "HF_LEROBOT_HOME": f"{AREA8}/data",
       "HOME": "/tmp/home", "LIBERO_CONFIG_PATH": f"{AREA8}/libero-config", "MUJOCO_EGL_DEVICE_ID": "0",
       "MUJOCO_GL": "egl", "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics", "PYOPENGL_PLATFORM": "egl",
       "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "TOKENIZERS_PARALLELISM": "false",
       "TRANSFORMERS_OFFLINE": "1", "WANDB_MODE": "disabled", "XDG_CACHE_HOME": "/tmp/cache"}
ENV_NAMES = sorted(ENV)
# train_lerobot_logged.py COMMON and the stage profiles (README section 11.1); the scenario's lengths are small.
COMMON = {"batch_size": 32, "learning_rate": None, "warmup_steps": None, "decay_steps": None, "train_expert_only": True,
          "freeze_vision_encoder": True, "n_action_steps": 50, "num_workers": 8, "seed": 1000, "log_freq": 50,
          "use_imagenet_stats": False, "mixed_precision": "no"}
FRAMES_PER_EPISODE = 2
DATA_FILES = {211: (807, 958), 212: (959, 1110), 213: (1111, 1260)}   # data file -> its LIBERO-Object episodes
FIRST_INDEX = 153511                                                   # the dataset index of episode 807 frame 0


def sha256_text(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def renamed_copy(path: Path) -> None:
    """Rename the schema id of a copied pre-rename record in place (`ptc_skel.<x>` or `ptc_hub.<x>` ->
    `gatedtrain.<x>`), nothing else: for tests of the adapter logic on a real output written before the renames. Never
    applied to a fixture."""
    from gatedtrain.renamed import OLD_SCHEMA_PREFIXES
    doc = json.loads(path.read_text(encoding="utf-8"))
    schema = str(doc.get("schema", ""))
    old = next((p for p in OLD_SCHEMA_PREFIXES if schema.startswith(p)), None)
    if old is None:
        raise ValueError(f"{path}: schema {doc.get('schema')!r} is not a pre-rename id")
    doc["schema"] = "gatedtrain." + schema[len(old):]
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compact_hash(obj: Any) -> str:
    """The runtime's own hash encoding (a8_common.doc_sha256): compact sorted JSON, not RFC 8785."""
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def derive_seed(*parts) -> int:
    """a8_common.derive_seed: 63-bit seed from the SHA-256 of the parts joined by ':'."""
    return int.from_bytes(hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).digest()[:8], "big") >> 1


def write_json(path: Path, doc: Any) -> Path:
    """a8_common.write_json: indent 2, sorted keys, final newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, docs: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(d, sort_keys=True) + "\n" for d in docs), encoding="utf-8")
    return path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def safetensors_stub(names: tuple[str, ...], tag: str) -> bytes:
    """A whole safetensors file with the real `__metadata__` and the given tensor names (4 bytes each)."""
    header = {"__metadata__": {"format": "pt", "fixture": tag}}
    for k, n in enumerate(names):
        header[n] = {"dtype": "F32", "shape": [1], "data_offsets": [4 * k, 4 * k + 4]}
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw + bytes(4 * len(names))


# ---------------------------------------------------------------------------
# The miniature dataset and its frame export (EXT-237)
# ---------------------------------------------------------------------------

def dataset_frames() -> list[dict]:
    """Every LIBERO-Object frame of the miniature dataset, in index order: episode e, frame f, task index 20 + (e - 807)
    mod 10 (lerobot/libero's Object task indices are 20-29)."""
    rows, i = [], FIRST_INDEX
    for e in range(EPISODES[0], EPISODES[1] + 1):
        for f in range(FRAMES_PER_EPISODE):
            rows.append({"index": i, "episode_index": e, "frame_index": f, "task_index": 20 + (e - EPISODES[0]) % 10})
            i += 1
    return rows


def write_dataset(root: Path) -> Path:
    """`<root>/data/chunk-000/file-NNN.parquet` (the v3.0 layout): the Object frames of DATA_FILES plus one frame of the
    neighbour suites' episodes 806 (in file 211) and 1261 (in file 213), as the real files 211 and 308 share episodes."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    frames = dataset_frames()
    for f, (lo, hi) in DATA_FILES.items():
        rows = [r for r in frames if lo <= r["episode_index"] <= hi]
        if f == 211:
            rows = [{"index": FIRST_INDEX - 1, "episode_index": 806, "frame_index": 91, "task_index": 19}] + rows
        if f == 213:
            rows = rows + [{"index": rows[-1]["index"] + 1, "episode_index": 1261, "frame_index": 0, "task_index": 30}]
        table = pa.table({"observation.state": [[0.0] * 8 for _ in rows], "action": [[0.0] * 7 for _ in rows],
                          "timestamp": [r["frame_index"] / 10 for r in rows],
                          **{k: pa.array([r[k] for r in rows], type=pa.int64())
                             for k in ("frame_index", "episode_index", "index", "task_index")}})
        path = root / "data" / "chunk-000" / f"file-{f:03d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
    return root


def write_frame_export(dataset_root: Path, out: Path, *, files_sha256: str, sources: list[str] | None = None) -> Path:
    """The frame export as the proposed area step `export_frames.py` writes it (one `chunk-CCC_file-FFF.jsonl` per data
    file with the LIBERO-Object rows {episode_index, frame_index, index, task_index} in index order, and
    `frames_meta.json`), read back from the parquet files with pyarrow."""
    import pyarrow.parquet as pq

    out.mkdir(parents=True, exist_ok=True)
    if sources is None:
        sources = sorted(p.relative_to(dataset_root).as_posix() for p in (dataset_root / "data").rglob("*.parquet"))
    shards, n = {}, 0
    cols = ("index", "episode_index", "frame_index", "task_index")
    for src in sources:
        t = pq.read_table(dataset_root / src, columns=list(cols)).to_pydict()
        rows = sorted((dict(zip(cols, v)) for v in zip(*(t[c] for c in cols)) if EPISODES[0] <= v[1] <= EPISODES[1]),
                      key=lambda r: r["index"])
        if not rows:
            continue
        chunk, fname = src.split("/")[1], src.split("/")[2].replace(".parquet", ".jsonl")
        name = f"{chunk}_{fname}"
        write_jsonl(out / name, rows)
        shards[name] = {"source": src, "source_sha256": file_sha256(dataset_root / src), "rows": len(rows),
                        "first_index": rows[0]["index"], "last_index": rows[-1]["index"]}
        n += len(rows)
    write_json(out / "frames_meta.json", {
        "schema": "ptc_mm.lerobot_frame_export", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "files_sha256": files_sha256,
                    "episodes": list(EPISODES)},
        "n_frames": n, "columns": list(cols), "shards": shards})
    return out


def row_key(e: int, f: int) -> str:
    """MM-26 `data_ref.row`: SHA-256 of the RFC 8785 JSON {episode_index, frame_index} (sorted keys, no spaces)."""
    return sha256_text(f'{{"episode_index":{e},"frame_index":{f}}}'.encode("utf-8"))


# ---------------------------------------------------------------------------
# Runs and cuts (train_lerobot_logged.py)
# ---------------------------------------------------------------------------

def lerobot_argv(s: dict, policy_dir: str) -> list[str]:
    """train_lerobot_logged.py build_lerobot_argv with output_dir /out/lerobot (the runs' --out /out)."""
    argv = [f"--policy.path={policy_dir}", f"--policy.vlm_model_name={VLM_DIR}", "--policy.load_vlm_weights=false",
            "--policy.input_features=null", "--policy.output_features=null", f"--policy.device={s['device']}",
            "--policy.push_to_hub=false", f"--policy.n_action_steps={s['n_action_steps']}",
            f"--policy.train_expert_only={str(s['train_expert_only']).lower()}",
            f"--policy.freeze_vision_encoder={str(s['freeze_vision_encoder']).lower()}"]
    if s["learning_rate"] is not None:
        argv.append(f"--policy.optimizer_lr={s['learning_rate']}")
    if s["warmup_steps"] is not None:
        argv.append(f"--policy.scheduler_warmup_steps={s['warmup_steps']}")
    if s["decay_steps"] is not None:
        argv.append(f"--policy.scheduler_decay_steps={s['decay_steps']}")
    argv += [f"--dataset.repo_id={DATASET_REPO}", f"--dataset.root={DATASET_ROOT}",
             f"--dataset.revision={DATASET_REVISION}",
             "--dataset.episodes=[" + ",".join(str(e) for e in range(EPISODES[0], EPISODES[1] + 1)) + "]",
             "--dataset.video_backend=pyav", f"--dataset.use_imagenet_stats={str(s['use_imagenet_stats']).lower()}",
             "--rename_map=" + json.dumps(RENAME, separators=(",", ":")), "--output_dir=/out/lerobot",
             f"--job_name=a8_{s['profile']}", f"--seed={s['seed']}", f"--batch_size={s['batch_size']}",
             f"--steps={s['steps']}", f"--save_freq={s['save_freq']}",
             f"--save_checkpoint={str(s['save_checkpoint']).lower()}", f"--log_freq={s['log_freq']}",
             f"--num_workers={s['num_workers']}", "--env_eval_freq=0", "--eval_steps=0", "--trackio.enable=false"]
    return argv


def settings(*, profile: str, steps: int, save_freq: int, seed: int, **over) -> dict:
    stage = "base" if profile == "base" else "continuation"
    s = {**COMMON, "stage": stage, "steps": steps, "save_freq": save_freq, "save_checkpoint": True, "profile": profile,
         "seed": seed, **over}
    s["device"] = "cuda"
    s["full_ft"] = not s["train_expert_only"]
    s["cut_steps"] = sorted({k for k in range(1, steps + 1) if k % save_freq == 0 or k == steps})
    s["memory_every"] = 10
    return s


def simulate(frames: list[dict], s: dict) -> tuple[list[dict], dict[int, int]]:
    """(samples.jsonl lines, {step: samples consumed up to it}): every epoch a permutation of all frames (the
    EpisodeAwareSampler), batches of batch_size with the short last batch kept (drop_last false)."""
    bs, total = s["batch_size"], s["steps"]
    lines, consumed, n, step, epoch = [], {}, 0, 0, -1
    while step < total:
        epoch += 1
        order = list(frames)
        random.Random(s["seed"] * 1000 + epoch).shuffle(order)
        for b0 in range(0, len(order), bs):
            if step >= total:
                break
            step += 1
            batch = order[b0:b0 + bs]
            lines.append({"step": step, "epoch": epoch, "n": len(batch),
                          "samples": [{"index": r["index"], "episode_index": r["episode_index"],
                                       "frame_index": r["frame_index"]} for r in batch]})
            n += len(batch)
            consumed[step] = n
    return lines, consumed


def normalizer_hash(cut: Path) -> str:
    return compact_hash({"normalizer": file_sha256(cut / NORMALIZER), "unnormalizer": file_sha256(cut / UNNORMALIZER)})


def build_cut(run: Path, step: int, *, s: dict, samples_consumed: int, script_sha256: str, common_sha256: str,
              policy_path: str, init_checkpoint: dict | None, tag: str = "") -> Path:
    """`cuts/checkpoint-<step>/` as train_lerobot_logged.py write_cut writes it: LeRobot's pretrained_model/ and
    training_state/ (the real small files of checkpoint-500, stand-in weights, optimizer state and tokenizer.json),
    sampler.json, base_ref.json, then cut.json last."""
    d = run / "cuts" / f"checkpoint-{step}"
    pm, ts = d / "pretrained_model", d / "training_state"
    (pm / "tokenizer").mkdir(parents=True)
    ts.mkdir()
    for rel in ("policy_postprocessor.json", "policy_preprocessor.json", NORMALIZER.split("/", 1)[1],
                UNNORMALIZER.split("/", 1)[1], "tokenizer/chat_template.jinja", "tokenizer/tokenizer_config.json"):
        shutil.copyfile(REAL_CUT / "pretrained_model" / rel, pm / rel)
    copy_fixture_text(REAL_CUT / "pretrained_model" / "train_config.json", pm / "train_config.json")
    cfg = read_fixture_json(REAL_CUT / "pretrained_model" / "config.json")
    cfg["pretrained_path"] = policy_path
    (pm / "config.json").write_text(json.dumps(cfg, indent=4) + "\n", encoding="utf-8")
    (pm / "model.safetensors").write_bytes(safetensors_stub(("model.action_in_proj.bias", "model.action_in_proj.weight"),
                                                            f"{run.name} {step} {tag}"))
    (pm / "tokenizer" / "tokenizer.json").write_text(json.dumps({"fixture": "SmolVLM2 tokenizer stand-in"}) + "\n",
                                                      encoding="utf-8")
    for rel in ("optimizer_param_groups.json", "rng_state.safetensors", "scheduler_state.json"):
        shutil.copyfile(REAL_CUT / "training_state" / rel, ts / rel)
    (ts / "optimizer_state.safetensors").write_bytes(safetensors_stub(("state/345/step",), f"{run.name} {step}"))
    st = json.loads((REAL_CUT / "training_state" / "training_step.json").read_text(encoding="utf-8"))
    st["step"] = step
    (ts / "training_step.json").write_text(json.dumps(st, indent=4), encoding="utf-8")
    shutil.copyfile(REAL_CUT / "sampler.json", d / "sampler.json")
    write_json(d / "base_ref.json", dict(BASE_REF, init_checkpoint=init_checkpoint))
    seal_cut(d, s=s, step=step, samples_consumed=samples_consumed, script_sha256=script_sha256,
             common_sha256=common_sha256, init_checkpoint=init_checkpoint)
    return d


def seal_cut(d: Path, *, s: dict, step: int, samples_consumed: int, script_sha256: str, common_sha256: str,
             init_checkpoint: dict | None) -> None:
    files = {rel: file_sha256(d / rel) for rel in CUT_FILES}
    model = sorted(DEPLOYABLE)
    norm = {"normalizer": files[NORMALIZER], "unnormalizer": files[UNNORMALIZER]}
    sampler = json.loads((d / "sampler.json").read_text(encoding="utf-8"))
    pad = max(6, len(str(s["steps"])))
    (d / "cut.json").write_text(json.dumps({
        "schema": "ptc_mm.lerobot_cut", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
        "global_step": step, "last_step_included": step, "samples_consumed": samples_consumed,
        "batch_size": s["batch_size"], "files": files, "model_files": model,
        "resume_files": sorted(set(files) - set(model) - {"base_ref.json"}),
        "model_files_sha256": compact_hash({k: files[k] for k in model}), "normalizer_sha256": compact_hash(norm),
        "normalizer": {"files": {"normalizer": {"name": NORMALIZER.split("/")[1], "sha256": norm["normalizer"]},
                                 "unnormalizer": {"name": UNNORMALIZER.split("/")[1], "sha256": norm["unnormalizer"]}},
                       "normalizer_sha256": compact_hash(norm), "definition": "..."},
        "sampler_sha256": compact_hash(sampler), "deployable": model, "seed": s["seed"], "profile": s["profile"],
        "stage": s["stage"], "init_checkpoint": init_checkpoint, "lerobot_step_dir": f"lerobot/checkpoints/{step:0{pad}d}",
        "file_origin": ["hardlink"], "script_sha256": script_sha256, "common_sha256": common_sha256,
        "consumption_log": "samples.jsonl", "created_at": "2026-11-15T09:00:00Z"}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def reseal(d: Path) -> None:
    """Rewrite cut.json's derived hashes after a cut file was edited (the cut stays complete)."""
    doc = json.loads((d / "cut.json").read_text(encoding="utf-8"))
    files = {rel: file_sha256(d / rel) for rel in doc["files"]}
    doc["files"] = files
    doc["model_files_sha256"] = compact_hash({k: files[k] for k in doc["model_files"]})
    doc["normalizer_sha256"] = compact_hash({"normalizer": files[NORMALIZER], "unnormalizer": files[UNNORMALIZER]})
    doc["sampler_sha256"] = compact_hash(json.loads((d / "sampler.json").read_text(encoding="utf-8")))
    write_json(d / "cut.json", doc)


def cut_model_hash(cut: Path) -> str:
    """MM-17: T0-09 manifest hash of the cut's deployable files."""
    entries = [e for e in manifest_entries(cut) if e["path"] in DEPLOYABLE]
    return "sha256:" + hashlib.sha256(canonical_bytes(entries)).hexdigest()


def init_record(cut: Path) -> dict:
    """train_lerobot_logged.py main: the stage-2 record of the base cut it verified (`init_checkpoint`)."""
    doc = json.loads((cut / "cut.json").read_text(encoding="utf-8"))
    return {"cut_dir": BASE_CUT_CONTAINER, "cut_json_sha256": file_sha256(cut / "cut.json"),
            "global_step": doc["global_step"], "model_files_sha256": doc["model_files_sha256"],
            "normalizer_sha256": doc["normalizer_sha256"], "profile": doc["profile"], "stage": doc["stage"],
            "seed": doc["seed"]}


def run_args(*, s: dict, policy_path: str, base_cut: dict | None, script_sha256: str, common_sha256: str) -> dict:
    """train_lerobot_logged.py run_args.json (main :555-588)."""
    bref = {"base_model": {k: v for k, v in BASE_REF["base_model"].items() if k != "files"},
            "vlm_backbone": BASE_REF["vlm_backbone"], "dataset": BASE_REF["dataset"], "init_checkpoint": base_cut}
    return {
        "schema": "ptc_mm.lerobot_run_args", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
        "profile": s["profile"], "stage": s["stage"], "attempt": 1, "settings": s, "seed": s["seed"],
        "seed_derivation": "...", "framework": {"name": "lerobot", "version": "0.7.0", "entry": "..."},
        "lerobot_argv": lerobot_argv(s, policy_path), "lerobot_output_dir": "lerobot", "policy_path": policy_path,
        "base_checkpoint": policy_path if base_cut else None, "base_cut": base_cut, "base_ref": bref,
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "root": DATASET_ROOT,
                    "episodes": f"{EPISODES[0]}..{EPISODES[1]} ({EPISODES[1] - EPISODES[0] + 1})",
                    "video_backend": "pyav", "rename_map": RENAME, "normalizer_stats": "..."},
        "tokenizer_override": {}, "sampler": SAMPLER, "sampler_sha256": compact_hash(SAMPLER),
        "consumption_log": "samples.jsonl", "consumption_semantics": "...", "cut_steps": s["cut_steps"],
        "cuts_dir": "cuts", "script_sha256": script_sha256, "common_sha256": common_sha256,
        "code_sha256": compact_hash({"train_lerobot_logged.py": script_sha256, "a8_common.py": common_sha256}),
        "env_freeze_matches_runtime": True, "created_at": "2026-11-15T09:00:00Z"}


def build_run(out: Path, *, s: dict, script_sha256: str, common_sha256: str, base_cut: Path | None = None) -> Path:
    """A run directory as train_lerobot_logged.py writes it (stage 1 without `base_cut`, stage 2 from it)."""
    out.mkdir(parents=True)
    init = init_record(base_cut) if base_cut is not None else None
    policy = f"{BASE_CUT_CONTAINER}/pretrained_model" if base_cut is not None else SMOLVLA_DIR
    lines, consumed = simulate(dataset_frames(), s)
    copy_fixture_text(REAL_RUN / "env-freeze.txt", out / "env-freeze.txt")
    rt = read_fixture_json(REAL_RUN / "env-runtime.json")
    # The fixture records the published run's lock (before the runtime project's rename, RESULT.md); a run now records
    # the current one.
    rt.update(script_sha256=script_sha256, common_sha256=common_sha256, env_freeze_sha256=file_sha256(out / "env-freeze.txt"),
              uv_lock_sha256=LOCK_SHA,
              code_sha256=compact_hash({"train_lerobot_logged.py": script_sha256, "a8_common.py": common_sha256}))
    write_json(out / "env-runtime.json", rt)
    write_json(out / "run_args.json", run_args(s=s, policy_path=policy, base_cut=init, script_sha256=script_sha256,
                                               common_sha256=common_sha256))
    write_jsonl(out / "samples.jsonl", lines)
    (out / "lerobot" / "checkpoints").mkdir(parents=True)
    for step in s["cut_steps"]:
        build_cut(out, step, s=s, samples_consumed=consumed[step], script_sha256=script_sha256,
                  common_sha256=common_sha256, policy_path=policy, init_checkpoint=init)
        (out / "lerobot" / "checkpoints" / f"{step:06d}").mkdir()        # LeRobot's own directories: never cuts
    write_jsonl(out / "cuts.jsonl", [{"step": k, "cut": f"cuts/checkpoint-{k}", "samples_consumed": consumed[k],
                                      "utc": "2026-11-15T09:00:00Z"} for k in s["cut_steps"]])
    write_jsonl(out / "attempts.jsonl", [
        {"attempt": 1, "argv": ["..."], "pid": 1, "started_at": "2026-11-15T09:00:00Z"},
        {"attempt": 1, "finished_at": "2026-11-15T09:30:00Z", "steps": s["steps"], "samples_consumed": consumed[s["steps"]],
         "epochs_started": 3, "samples_per_epoch": len(dataset_frames()), "cuts": s["cut_steps"],
         "peak_device_memory": {"allocated_gb": 7.9, "reserved_gb": 8.0}, "peak_rss_mb": 4265.8,
         "gpu": "NVIDIA GeForce RTX 4090", "blocked_connections": [], "wall_seconds": 1800.0}])
    write_jsonl(out / "memory.jsonl", [{"step": 1, "loss": 2.4}])
    return out


# ---------------------------------------------------------------------------
# Evaluations (eval_libero.py evaluate)
# ---------------------------------------------------------------------------

def baseline_success(i: int, r: int) -> bool:
    return (i + r) % 10 < 3


def candidate_success(i: int, r: int) -> bool:
    return (i + r) % 10 < 8


def episode(item: dict, seed: int, repeat: int, success: bool, *, n_action_steps: int = 50) -> tuple[dict, dict]:
    """(items.jsonl line without output_ref, episode log) as eval_libero.py score_record and on_result write them."""
    steps = 100 + item["init_state"] * 7 + repeat if success else 280
    outcome = "success" if success else "timeout"
    sub = {"steps_to_success": steps if success else 280, "n_policy_calls": math.ceil(steps / n_action_steps),
           "steps": steps, "wall_s": 9.5, "timeout": int(outcome == "timeout"), "nan_action": 0}
    line = {"item": item["item"], "repeat": repeat, "seed": seed, "score": 1 if success else 0, "error": None,
            "sub_metrics": sub, "item_meta": {"task_id": item["task_id"], "init_state": item["init_state"],
                                              "outcome": outcome}}
    log = {"schema": "ptc_mm.libero_episode", "schema_version": "1.0", "item": item["item"], "repeat": repeat,
           "eval_seed": seed, "task_id": item["task_id"], "init_state": item["init_state"],
           "env_seed": derive_seed("libero-env", seed, item["item"]) % 2**31,
           "torch_seed": derive_seed("libero-torch", seed, item["item"]),
           "noise_seed": derive_seed("libero-noise", seed, item["item"]), "error": None, "trace": None,
           "worker_pid": 100, "wall_s": 9.5, "device_memory": None, "outcome": outcome, "success": success,
           "steps": steps, "success_step": steps if success else None, "steps_to_success": sub["steps_to_success"],
           "n_policy_calls": sub["n_policy_calls"], "seconds": {"reset": 1.0}, "actions_summary": {"n": steps},
           "actions": [[0.0] * 7 for _ in range(3)]}
    return line, log


def build_eval(out: Path, items: list[dict], success_fn: Callable[[int, int], bool], *, cut: Path, model: str | None,
               item_set: str | None, items_dir: Path, items_name: str = "gate") -> Path:
    """An evaluation run as eval_libero.py writes it into --out: arm-success/ (items.jsonl, eval_meta.json,
    env_version.json, episodes/<item>/r<k>.json), env_version.json, env-runtime.json, progress.jsonl, summary.json.
    `model`/`item_set` None: the hashes left null as the evaluator writes them without --model-hash/--item-set-hash."""
    arm = out / "arm-success"
    (arm / "episodes").mkdir(parents=True)
    marker = json.loads((cut / "cut.json").read_text(encoding="utf-8"))
    write_json(out / "env_version.json", ENV_VERSION)
    write_json(arm / "env_version.json", ENV_VERSION)
    lines = []
    for r, seed in enumerate(SEEDS):
        for i, it in enumerate(items):
            line, log = episode(it, seed, r, success_fn(i, r))
            rel = f"episodes/{it['item']}/r{r}.json"
            write_json(arm / rel, log)
            line["output_ref"] = {"path": rel, "sha256": file_sha256(arm / rel)}
            lines.append(line)
    write_jsonl(arm / "items.jsonl", lines)
    norm = normalizer_hash(cut)
    write_json(arm / "eval_meta.json", {
        "schema": META_SCHEMA, "schema_version": "1.0", "evaluator_id": EVALUATOR_ID, "adapter": ADAPTER,
        "metric": METRIC, "direction": "higher_is_better", "score_range": [0, 1],
        "checkpoint": {"model_hash": model, "base_model_hash": None}, "normalizer_sha256": norm,
        "env_version_sha256": compact_hash(ENV_VERSION), "item_set_hash": item_set, "eval_seeds": list(SEEDS),
        "sampler": SAMPLER, "sampler_hash": compact_hash(SAMPLER), "scorer": SCORER,
        "framework": {"name": "lerobot", "version": "0.7.0"}, "created_at": "2026-11-15T10:00:00Z"})
    write_jsonl(out / "progress.jsonl", lines)
    write_json(out / "summary.json", {
        "checkpoint": "/inputs/cut", "global_step": marker["global_step"], "profile": marker["profile"],
        "stage": marker["stage"], "items": items_name, "n_items": len(items), "eval_seeds": list(SEEDS),
        "episodes": len(lines), "errors": 0, "outcomes": {}, "success": {"mean": sum(l["score"] for l in lines) / len(lines)},
        "per_task_success": {}, "mm38": {}, "seconds_wall": 100.0, "workers": 8, "device": "cuda",
        "item_timeout_s": 300.0, "envs_per_worker": 1, "error_reasons": {"item_timeout": 0, "worker_died": 0},
        "cut_model_files_sha256": marker["model_files_sha256"], "normalizer_sha256": marker["normalizer_sha256"],
        "sampler_hash": compact_hash(SAMPLER), "env_version_sha256": compact_hash(ENV_VERSION),
        "env_version_nondefault": {}, "missing_hashes": [], "blocked_connections": []})
    rt = read_fixture_json(REAL_EVAL / "env-runtime.json")
    rt.update(checkpoint="/inputs/cut", items_dir=f"{AREA8}/items-{items_name}",
              items_sha256=file_sha256(items_dir / "items.jsonl"), script_sha256=EVAL_LIBERO_SHA,
              common_sha256=COMMON_SHA, code_sha256=SCORER["code_sha256"], uv_lock_sha256=LOCK_SHA)
    write_json(out / "env-runtime.json", rt)
    return out


def smoke_record(*, base_model: str, item_set: str, **overrides) -> dict[str, Any]:
    """A PASS smoke.json (MM-38) on the binary deciding metric `success`: 20 baseline scores, mean 0.3."""
    rec = {"schema": SMOKE_SCHEMA, "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
           "adapter": ADAPTER, "area": AREA, "base_model_hash": base_model, "item_set_hash": item_set,
           "peak_gpu_mem_gb": 8.0, "host_ram_peak_gb": 12.0, "swap_peak_gb": 0.0, "seconds_per_step": 0.5,
           "projected_gpu_h": 0.5, "gpu_log_total_h": 30.0,
           "baseline_scores": [1 if i % 10 < 3 else 0 for i in range(20)], "error_count": 0, "verdict": "PASS",
           "reasons": []}
    rec.update(overrides)
    return rec


class LerobotScenario:
    """A whole area-8 stage-2 run collected against its stage-1 base cut (module docstring)."""

    def __init__(self, root: Path, *, n_per_task: int = 4, filled: bool = True):
        self.root = root
        self.mirror = root / "host"
        self.cpython = self.mirror / CPYTHON.lstrip("/")
        self.runtime = self.mirror / AREA8.lstrip("/")
        self.output_root = self.mirror / OUT_ROOT.lstrip("/")
        self.host_work = root / "host-work"
        py = self.cpython / "bin" / "python3.12"
        py.parent.mkdir(parents=True)
        py.write_text("# fixture stand-in for the uv-managed CPython 3.12.16; never executed\n", encoding="utf-8")
        rt = self.runtime
        (rt / ".venv" / "bin").mkdir(parents=True)
        (rt / ".venv" / "bin" / "python").symlink_to(f"{CPYTHON}/bin/python3.12")
        for name, text in SCRIPT_TEXT.items():
            (rt / name).write_text(text, encoding="utf-8")
        self.script = rt / "train_lerobot_logged.py"
        shutil.copyfile(LR / "area8-uv.lock", rt / "uv.lock")
        self.lock = rt / "uv.lock"
        (rt / "libero-config").mkdir()
        (rt / "libero-config" / "config.yaml").write_text("# fixture stand-in\n", encoding="utf-8")
        (rt / "models" / "smolvla_base").mkdir(parents=True)
        for name, states in (("gate", range(0, n_per_task)), ("probe", range(40, 40 + n_per_task))):
            rows = [r for r in read_jsonl(LR / "items" / f"items-{name}.jsonl") if r["init_state"] in states]
            write_jsonl(rt / f"items-{name}" / "items.jsonl", rows)
            write_json(rt / f"items-{name}" / "item_set_meta.json", {"schema": "ptc_mm.libero_item_set", "name": name})
        self.items_dir = rt / "items-gate"
        self.probe_dir = rt / "items-probe"
        self.item_rows = read_jsonl(self.items_dir / "items.jsonl")
        self.items = [r["item"] for r in self.item_rows]
        self.dataset = write_dataset(rt / "data" / "lerobot" / "libero")
        self.data = write_frame_export(self.dataset, rt / "frames", files_sha256=BASE_REF["dataset"]["files_sha256"])
        (root / "trainer.yaml").write_text("# area 8 trainer config (fixture)\nbackend: lerobot_smolvla\nprofile: full\n",
                                           encoding="utf-8")
        gpu_step_script(self.host_work)
        (self.host_work / "gpu-log.md").write_bytes((EXT / "gpu-log.fixture.md").read_bytes())
        runs = self.output_root / OUT_NAME
        self.runs = runs
        script_sha, common_sha = file_sha256(self.script), file_sha256(rt / "a8_common.py")
        self.base_run = build_run(runs / "base-1", s=settings(profile="base", steps=40, save_freq=20, seed=BASE_SEED),
                                  script_sha256=TRAIN_SHA, common_sha256=common_sha)
        self.base = self.base_run / "cuts" / "checkpoint-20"
        self.settings = settings(profile="full", steps=60, save_freq=30, seed=SEED)
        self.stage_out = build_run(runs / "full-1", s=self.settings, script_sha256=script_sha, common_sha256=common_sha,
                                   base_cut=self.base)
        self.cuts = [self.stage_out / "cuts" / f"checkpoint-{k}" for k in self.settings["cut_steps"]]
        self.final_cut = self.cuts[-1]
        self.base_model = cut_model_hash(self.base)
        self.cand_model = cut_model_hash(self.final_cut)
        self.item_set_hash = manifest_hash(self.items_dir)
        self.render()
        self.base_eval = build_eval(runs / "eval-base-1", self.item_rows, baseline_success, cut=self.base,
                                    model=self.base_model if filled else None,
                                    item_set=self.item_set_hash if filled else None, items_dir=self.items_dir)
        self.cand_eval = build_eval(runs / "eval-cand-1", self.item_rows, candidate_success, cut=self.final_cut,
                                    model=self.cand_model if filled else None,
                                    item_set=self.item_set_hash if filled else None, items_dir=self.items_dir)
        if filled:
            for ev in (self.base_eval, self.cand_eval):        # ext-arm-fill writes the RFC 8785 sampler hash
                self.edit_json(ev / "arm-success" / "eval_meta.json", lambda d: d.update(sampler_hash=sha_of(SAMPLER)))
        self.baseline = self.base_eval / "arm-success"
        self.candidate = self.cand_eval / "arm-success"
        rec = self.launch_record()
        self.launch = write_canonical(root / "launch.json", rec)
        self.label = rec["label"]
        self.smoke = write_canonical(root / "smoke.json", smoke_record(base_model=self.base_model,
                                                                         item_set=self.item_set_hash))
        self.guard_log()

    # --- paths and environment ---

    def input_roots(self) -> list[Path]:
        return [self.mirror / PYTHON_DIR.lstrip("/"), self.runtime, self.runs]

    def env(self) -> dict[str, str]:
        return {ENV_HOST_WORK: str(self.host_work), ENV_OUTPUT_ROOT: str(self.output_root),
                ENV_INPUT_ROOTS: os.pathsep.join(str(p) for p in self.input_roots()),
                ENV_MIRROR_ROOT: str(self.mirror), ENV_INTERPRETER_TREES: str(self.cpython)}

    def config_sha(self) -> str:
        return host_config_sha256(self.host_work, self.output_root, self.input_roots(), mirror_root=self.mirror,
                                  interpreter=INTERPRETER, interpreter_trees=[self.cpython])

    def guard_log(self) -> None:
        lines = [f"2026-11-11T09:00:00Z START {self.label} attempt=1",
                 f"2026-11-11T10:00:00Z END {self.label} attempt=1 rc=0"]
        (self.host_work / "gpu-guard.log").write_text("".join(l + "\n" for l in lines), encoding="utf-8")

    @property
    def paths(self) -> dict[str, Path]:
        return {"spec": self.root / "spec.yaml", "registration": self.root / "registration.yaml"}

    def render(self, reg_mutate: Callable[[dict], None] | None = None,
               spec_mutate: Callable[[dict], None] | None = None) -> None:
        text = (LR / "registration.template.yaml").read_text(encoding="utf-8")
        text = text.replace("{item_set_hash}", self.item_set_hash).replace("{sampler}", json.dumps(SAMPLER))
        text = text.replace("{env_version}", compact_hash(ENV_VERSION))
        text = text.replace("{evaluator_hash}", sha_of({"evaluator": EVALUATOR_ID, "metric": METRIC,
                                                         "direction": "higher_is_better", "scorer": SCORER}))
        if reg_mutate is not None:
            reg = yaml.safe_load(text)
            reg_mutate(reg)
            text = yaml.safe_dump(reg, sort_keys=False)
        (self.root / "registration.yaml").write_text(text, encoding="utf-8")
        spec = (LR / "spec.template.yaml").read_text(encoding="utf-8")
        spec = spec.replace("{registration_sha256}", file_sha256(self.root / "registration.yaml")[len("sha256:"):])
        spec = spec.replace("{base}", str(self.base)).replace("{data}", str(self.data))
        spec = spec.replace("{items}", str(self.items_dir))
        if spec_mutate is not None:
            doc = yaml.safe_load(spec)
            spec_mutate(doc)
            spec = yaml.safe_dump(doc, sort_keys=False)
        (self.root / "spec.yaml").write_text(spec, encoding="utf-8")

    # --- launch ---

    def mounts(self, out: Path, *, base_cut: Path | None = None, cut: Path | None = None) -> list[dict]:
        ms = [{"host": str(self.mirror / PYTHON_DIR.lstrip("/")), "container": PYTHON_DIR, "mode": "ro"},
              {"host": str(self.runtime), "container": AREA8, "mode": "ro"}]
        if base_cut is not None:
            ms.append({"host": str(base_cut), "container": BASE_CUT_CONTAINER, "mode": "ro"})
        if cut is not None:
            ms.append({"host": str(cut), "container": "/inputs/cut", "mode": "ro"})
        return ms + [{"host": str(out), "container": "/out", "mode": "rw"}]

    def argv(self, profile: str = "full", seed: int = SEED, steps: int = 60, save_freq: int = 30) -> list[str]:
        argv = [INTERPRETER, "-I", TRAIN_PY, "--profile", profile]
        if profile != "base":
            argv += ["--policy-path", BASE_CUT_CONTAINER]
        argv += ["--out", "/out", "--seed", str(seed)]
        if profile == "full":
            argv += ["--steps", str(steps), "--save-freq", str(save_freq)]
        return argv

    def eval_argv(self, items: str = "gate", *extra: str) -> list[str]:
        return [INTERPRETER, "-I", EVAL_PY, "--checkpoint", "/inputs/cut", "--items", items, "--eval-seeds", "11", "12",
                "--workers", "8", "--out", "/out", *extra]

    def launch_record(self, **overrides) -> dict[str, Any]:
        """ext-launch's record of the stage-2 full run (README section 11.3 "Full")."""
        rec = {"schema": RECORD_SCHEMA, "schema_version": "2.0", "tags": ["prototype", "unpromotable"],
               "adapter": ADAPTER, "area": AREA, "profile": "full", "label": LABEL, "host_work": str(self.host_work),
               "runtime": {"kind": "venv_in_container", "image": IMAGE, "venv_lock_sha256": file_sha256(self.lock)},
               "framework": {"name": "lerobot", "version": "0.7.0", "commit": None},
               "training_seed": SEED, "gpus": True, "argv": self.argv(),
               "mounts": self.mounts(self.stage_out, base_cut=self.base), "tmpfs": ["/tmp"], "network": "none",
               "network_reason": None, "user": "1000:1000", "env_names": list(ENV_NAMES), "gpu_est_h": 0.4,
               "gpu_used_h": 30.0, "output_dir": str(self.stage_out), "output_root": str(self.output_root),
               "gpu_step_sha256": file_sha256(self.host_work / "bin" / "gpu-step.sh"), "allow_network": False,
               "interpreter_path": f"{CPYTHON}/bin/python3.12",
               "interpreter_sha256": file_sha256(self.cpython / "bin" / "python3.12"),
               "script_path": TRAIN_PY, "script_sha256": file_sha256(self.script),
               "host_config_sha256": self.config_sha()}
        rec.update(overrides)
        rec["label"] = launch_label(rec)
        return rec

    def rewrite_launch(self, **overrides) -> Path:
        rec = self.launch_record(**overrides)
        p = write_canonical(self.root / "launch.json", rec)
        self.label = rec["label"]
        self.guard_log()
        return p

    def new_out(self, name: str) -> Path:
        p = self.runs / name
        p.mkdir(parents=True)
        return p

    def request(self, out: Path, *, profile: str = "smoke", **overrides) -> dict[str, Any]:
        """An ext-launch request of the stage-2 trainer into `out` (smoke: `--profile smoke`, 250 steps)."""
        argv = self.argv(profile) if profile == "full" else [a for a in self.argv("smoke")]
        req = {"area": AREA, "label": f"mm-a8-vla-{profile}", "host_work": str(self.host_work),
               "runtime": {"kind": "venv_in_container", "image": IMAGE, "venv_lock": str(self.lock)},
               "framework": {"name": "lerobot", "version": "0.7.0", "commit": None}, "training_seed": SEED,
               "gpus": True, "network": "none", "network_reason": None, "user": "1000:1000", "env": dict(ENV),
               "mounts": self.mounts(out, base_cut=self.base), "tmpfs": ["/tmp"], "output_dir": str(out),
               "gpu_est_h": 0.1, "argv": argv}
        req.update(overrides)
        return req

    def eval_request(self, out: Path, *, cut: Path | None = None, items: str = "gate", extra: tuple = (),
                     **overrides) -> dict[str, Any]:
        """An evaluation request (eval_libero.py on a cut mounted read-only at /inputs/cut)."""
        req = self.request(out, label="mm-a8-vla-eval", mounts=self.mounts(out, cut=cut or self.base), gpu_est_h=0.6,
                           argv=self.eval_argv(items, *extra))
        req.update(overrides)
        return req

    def launch_request(self, req: dict, name: str, *, profile: str = "smoke", smoke: Path | None = None,
                       extra: list[str] = ()):
        path = self.root / f"{name}.request.yaml"
        path.write_text(yaml.safe_dump(req, sort_keys=False), encoding="utf-8")
        out = self.root / f"{name}.launch.json"
        args = ["ext-launch", "--adapter", ADAPTER, "--profile", profile, "--request", str(path), "--gpu-log",
                str(self.host_work / "gpu-log.md"), "--out-record", str(out)]
        if smoke is not None:
            args += ["--smoke-record", str(smoke)]
        return run_cli(args + list(extra), cwd=self.root, env=self.env()), out

    def recheck(self, record: Path, extra: list[str] = ()):
        return run_cli(["ext-launch", "--adapter", ADAPTER, "--recheck", str(record), *extra], cwd=self.root,
                       env=self.env())

    # --- edits and collect ---

    def edit_json(self, path: Path, fn: Callable[[dict], None]) -> None:
        doc = json.loads(path.read_text(encoding="utf-8"))
        fn(doc)
        write_json(path, doc)

    def samples(self) -> list[dict]:
        return read_jsonl(self.stage_out / "samples.jsonl")

    def edit_samples(self, fn: Callable[[list[dict]], None]) -> None:
        lines = self.samples()
        fn(lines)
        write_jsonl(self.stage_out / "samples.jsonl", lines)

    def collect(self, runs_root: Path, *, baseline: Path | None = None, candidate: Path | None = None,
                launch: Path | None = None, smoke: Path | None = None, timeout: int = 300):
        args = ["ext-collect", str(self.paths["spec"]), "--adapter", ADAPTER, "--stage-out", str(self.stage_out),
                "--launch-record", str(launch or self.launch), "--smoke-record", str(smoke or self.smoke),
                "--eval-baseline", str(baseline or self.baseline), "--eval-candidate", str(candidate or self.candidate),
                "--runs-root", str(runs_root)]
        return run_cli(args, cwd=self.root, timeout=timeout, env=self.env())

    def fill(self, arm: Path, checkpoint: Path, *, items: Path | None = None):
        args = ["ext-arm-fill", "--adapter", ADAPTER, "--checkpoint", str(checkpoint), "--items",
                str(items or self.items_dir)]
        return run_cli(args + [str(arm)], cwd=self.root, env=self.env())

    def checks(self, launch: dict | None = None, **kw):
        """The adapter's run checks (EXT-236, EXT-237, EXT-239) directly, with this scenario's record."""
        from gatedtrain.ext import registry

        ad = registry.adapter(ADAPTER)
        cuts = [d for _, d in ad.find_cuts(kw.pop("stage_out", self.stage_out))[0]]
        return ad.checks(last_cut=kw.pop("last_cut", cuts[-1] if cuts else self.final_cut),
                         base=kw.pop("base", self.base), launch=launch or json.loads(self.launch.read_text()),
                         base_field="artifacts.base.uri", stage_out=self.stage_out, data=kw.pop("data", self.data),
                         data_field="artifacts.frames.uri")


def copy_scenario_cut(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst, symlinks=True)
    return dst


def ast_constants(path: Path, names: tuple[str, ...]) -> dict[str, Any]:
    """Module-level constants of a runtime script read with `ast` (never imported)."""
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in names:
                try:
                    out[node.targets[0].id] = ast.literal_eval(node.value)
                except ValueError:
                    out[node.targets[0].id] = ast.unparse(node.value)
    return out


def argparse_options(path: Path, func: str = "parse_args") -> dict[str, Any]:
    """{option: nargs kind} of the `add_argument` calls inside `func` of a runtime script (`ast`, never imported):
    0 for store_true/store_false, "+" for nargs="+", 1 otherwise."""
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func:
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and getattr(call.func, "attr", None) == "add_argument":
                    opts = [a.value for a in call.args if isinstance(a, ast.Constant) and str(a.value).startswith("--")]
                    kw = {k.arg: k.value for k in call.keywords}
                    kind = 1
                    if isinstance(kw.get("action"), ast.Constant) and kw["action"].value in ("store_true", "store_false"):
                        kind = 0
                    if isinstance(kw.get("nargs"), ast.Constant) and kw["nargs"].value == "+":
                        kind = "+"
                    for o in opts:
                        out[o] = kind
    return out


def deepcopy(x):
    return copy.deepcopy(x)

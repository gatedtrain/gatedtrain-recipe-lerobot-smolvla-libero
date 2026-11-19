"""Shared constants and helpers of the area "VLA" scripts (LeRobot 0.7.0 SmolVLA SFT on LIBERO-Object; prototype,
unpromotable). Imported by train_lerobot_logged.py, eval_libero.py and probe_render.py; importing it loads no
LeRobot, torch or LIBERO module, so the scripts can fix their environment first.

Everything here is cited against the installed venv (`.venv/lib/python3.12/site-packages/lerobot`, LeRobot 0.7.0,
commit 410c095e) or the hf-libero 0.1.4 package; README.md sections 0-6 hold the research facts.

The deployment sampler (`deploy_sampler`) is the evaluator's and the cut writer's single definition: the cut writer
stores it as `sampler.json` in every cut, the evaluator rebuilds it from the cut's `config.json` and refuses a cut
whose `sampler.json` differs.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import ipaddress
import json
import math
import os
import resource
import socket
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SMOLVLA = HERE / "models" / "smolvla_base"
SMOLVLM = HERE / "models" / "SmolVLM2-500M-Video-Instruct"
DATA_HOME = HERE / "data"                                   # HF_LEROBOT_HOME: repo id lerobot/libero -> data/lerobot/libero
DATA_ROOT = DATA_HOME / "lerobot" / "libero"
ASSETS = HERE / "assets" / "libero-assets"
LIBERO_CONFIG = HERE / "libero-config"
MANIFEST = HERE / "MANIFEST.json"                           # fetch_models.py: every downloaded file's SHA-256
ENV_FREEZE = HERE / "env-freeze.txt"
UV_LOCK = HERE / "uv.lock"

DATASET_REPO = "lerobot/libero"
DATASET_REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
OBJECT_EPISODES = list(range(807, 1261))                    # README section 3: LIBERO-Object = episodes 807-1260
# Dataset camera keys -> smolvla_base camera names (docs/source/peft_training.mdx:20-23; the lerobot/smolvla_libero
# recipe, meta/lerobot_smolvla_libero/train_config.json `rename_map`).
RENAME = {"observation.images.image": "observation.images.camera1",
          "observation.images.image2": "observation.images.camera2"}

# LIBERO-Object episode settings (LeRobot's integration): sim/integrations/libero.py:134-140 (280 steps),
# :163-167 (num_steps_wait 10, control_freq 20, control_mode relative, hard_reset True); envs/configs.py:466-482
# (LiberoEnvConfig defaults: obs_type pixels_agent_pos, 360 x 360, both cameras, init_states True).
SUITE = "libero_object"
MAX_STEPS = 280
SETTLE_STEPS = 10
CONTROL_FREQ = 20
CONTROL_MODE = "relative"
OBS_TYPE = "pixels_agent_pos"
OBS_HEIGHT = 360
OBS_WIDTH = 360
CAMERA_NAME = "agentview_image,robot0_eye_in_hand_image"
CAMERA_NAME_MAPPING = {"agentview_image": "image", "robot0_eye_in_hand_image": "image2"}   # libero.py:192-196
HARD_RESET = True
INIT_STATES = True
N_INIT_STATES = 50
GATE_STATES = list(range(0, 40))                            # README section 5 / decision: 400 gate items
PROBE_STATES = list(range(40, 50))                          # 100 probe / pilot items

TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH")
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "HF_LEROBOT_HOME": str(DATA_HOME),
    "LIBERO_CONFIG_PATH": str(LIBERO_CONFIG),
    "TOKENIZERS_PARALLELISM": "false",
    "WANDB_MODE": "disabled",
}


def fix_environment() -> None:
    """Before any library import: token variables removed (never read), offline switches and the area paths set
    unless the launch already set them. HF_HOME must be writable (the `datasets` Arrow cache, README section 0.4);
    when it is not set, a per-user directory under the temporary directory is used."""
    for name in TOKEN_VARS:
        os.environ.pop(name, None)
    for key, value in OFFLINE_ENV.items():
        os.environ.setdefault(key, value)
    if "HF_HOME" not in os.environ:
        import tempfile

        os.environ["HF_HOME"] = str(Path(tempfile.gettempdir()) / f"gatedtrain-a8-hf-home-{os.getuid()}")


BLOCKED: list[str] = []
_real_connect = socket.socket.connect


def _guarded_connect(self, address):
    host = address[0] if isinstance(address, tuple) else address
    try:
        local = isinstance(address, (str, bytes)) or ipaddress.ip_address(host).is_loopback
    except ValueError:
        local = False
    if not local:
        BLOCKED.append(str(address))
        raise OSError(f"network blocked by the area-8 scripts: {address}")
    return _real_connect(self, address)


def install_socket_guard() -> None:
    """As verify_cpu.py (P0): refuse and record every non-loopback connection of this process (the runs also use
    `--network none`; this makes an attempted download visible in the run record)."""
    socket.socket.connect = _guarded_connect


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def canonical(doc) -> bytes:
    """Compact sorted JSON (the evaluators' encoding; gatedtrain converts a sampler hash in this encoding, EXT-126)."""
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def doc_sha256(doc) -> str:
    return sha256_bytes(canonical(doc))


def write_json(path: Path, doc) -> None:
    """Atomic: temporary file, fsync, rename."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(doc, indent=2, sort_keys=True, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def derive_seed(*parts) -> int:
    """63-bit seed from the SHA-256 of the parts joined by ':' (recorded derivation, MM-29; areas 3 and 5)."""
    return int.from_bytes(hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).digest()[:8], "big") >> 1


def peak_rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def rss_now_mb() -> float | None:
    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
    except (OSError, IndexError, ValueError):
        return None
    return round(pages * os.sysconf("SC_PAGE_SIZE") / 2**20, 1)


def finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def code_sha256(*paths: Path) -> str:
    """Composite code hash: SHA-256 of the canonical {file name: SHA-256} of the given files."""
    return doc_sha256({Path(p).name: sha256_file(Path(p)) for p in paths})


COMMON_FILE = Path(__file__).resolve()


def versions() -> dict:
    names = ("lerobot", "torch", "torchvision", "transformers", "accelerate", "datasets", "huggingface-hub", "av",
             "numpy", "mujoco", "robosuite", "robomimic", "hf-libero", "bddl", "gymnasium", "draccus")
    out = {}
    for n in names:
        try:
            out[n] = importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:
            out[n] = None
    return out


# ---------------------------------------------------------------------------------------------------------------
# Bindings (MANIFEST.json, fetch_models.py)
# ---------------------------------------------------------------------------------------------------------------

def manifest_binding(name: str) -> dict:
    """Repository, 40-hex revision and every file's SHA-256 of one pinned download (MM-18, MM-21)."""
    rec = json.loads(MANIFEST.read_text(encoding="utf-8"))["repos"][name]
    files = {k: v["sha256"] for k, v in rec["files"].items()}
    return {"repo": rec["repo"], "repo_type": rec["repo_type"], "revision": rec["revision"],
            "local_dir": str(HERE / rec["local_dir"]), "n_files": len(files), "files": files,
            "files_sha256": doc_sha256(files)}


def binding_summary(b: dict) -> dict:
    return {k: b[k] for k in ("repo", "repo_type", "revision", "local_dir", "n_files", "files_sha256")}


# ---------------------------------------------------------------------------------------------------------------
# Cut contents (README section 6.1)
# ---------------------------------------------------------------------------------------------------------------

PRETRAINED = "pretrained_model"                 # utils/constants.py PRETRAINED_MODEL_DIR
TRAINING_STATE = "training_state"               # utils/constants.py TRAINING_STATE_DIR
# Files of pretrained_model/ that are resume state, not model: train_config.json (common/train_utils.py:217-260
# docstring tree). Everything else in pretrained_model/ (weights, config, processor JSON and state files, and the
# tokenizer artifact `tokenizer/` that TokenizerProcessorStep.save_artifacts writes, processor/tokenizer_processor.py
# :298-305, processor/pipeline.py:572-586) is a model component.
RESUME_ONLY = ("train_config.json",)


def processor_state_files(pretrained: Path) -> dict:
    """{'normalizer': file name, 'unnormalizer': file name} from the processor JSON `state_file` entries
    (registry names normalizer_processor / unnormalizer_processor; the step index in the file name may vary)."""
    out = {}
    for cfg_name, reg, key in (("policy_preprocessor.json", "normalizer_processor", "normalizer"),
                               ("policy_postprocessor.json", "unnormalizer_processor", "unnormalizer")):
        doc = json.loads((pretrained / cfg_name).read_text(encoding="utf-8"))
        names = [s.get("state_file") for s in doc["steps"] if s.get("registry_name") == reg and s.get("state_file")]
        if len(names) != 1:
            raise RuntimeError(f"{pretrained / cfg_name}: expected one {reg} state file, found {names}")
        out[key] = names[0]
    return out


def normalizer_record(pretrained: Path) -> dict:
    """MM-19 `normalizer_sha256`: SHA-256 of the canonical JSON {normalizer: sha256, unnormalizer: sha256} of the
    two processor state files (README section 6.1). Returns {files: {role: {name, sha256}}, normalizer_sha256}."""
    names = processor_state_files(pretrained)
    files = {role: {"name": n, "sha256": sha256_file(pretrained / n)} for role, n in names.items()}
    return {"files": files, "normalizer_sha256": doc_sha256({r: f["sha256"] for r, f in files.items()}),
            "definition": "sha256 of canonical JSON {normalizer: sha256, unnormalizer: sha256} of the processor "
                          "state files named in policy_preprocessor.json / policy_postprocessor.json"}


def tree_files(root: Path) -> list[str]:
    """Relative POSIX paths of every regular file below root (sorted); a symbolic link is refused (MM-20)."""
    out = []
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise RuntimeError(f"{p} is a symbolic link; cuts must be plain (MM-20)")
        if p.is_file():
            out.append(p.relative_to(root).as_posix())
    return out


def model_file_names(cut_files: list[str]) -> list[str]:
    """Model components of a cut (relative to the cut): pretrained_model/* except the resume-only files, plus
    sampler.json."""
    out = [f for f in cut_files if f.startswith(PRETRAINED + "/")
           and f.split("/", 1)[1] not in RESUME_ONLY]
    if "sampler.json" in cut_files:
        out.append("sampler.json")
    return sorted(out)


# ---------------------------------------------------------------------------------------------------------------
# Deployment sampler (README section 6.1: stored in every cut as sampler.json)
# ---------------------------------------------------------------------------------------------------------------

NOISE_RULE = ("per (eval_seed, item): torch.Generator('cpu').manual_seed(derive_seed('libero-noise', eval_seed, item)); "
              "one standard-normal float32 draw of shape (1, chunk_size, max_action_dim) per environment step, passed "
              "as select_action(noise=...), which uses it when the action queue is empty "
              "(policies/smolvla/modeling_smolvla.py:256-285, :790-794); torch global RNG reseeded with "
              "derive_seed('libero-torch', eval_seed, item) before the episode")


def deploy_sampler(policy_config: dict) -> dict:
    """The deployment sampler of a SmolVLA cut on LIBERO-Object, from its config.json. Episode settings that belong
    to the environment (max steps, settle steps, control frequency) are in env_version.json, not here."""
    c = policy_config
    return {
        "policy_type": c["type"],
        "n_action_steps": c["n_action_steps"],
        "chunk_size": c["chunk_size"],
        "num_steps": c["num_steps"],                         # flow-matching denoising steps
        "max_action_dim": c["max_action_dim"],
        "resize_imgs_with_padding": list(c["resize_imgs_with_padding"]),
        "tokenizer_max_length": c["tokenizer_max_length"],
        "rename_map": dict(RENAME),
        "observation": {"obs_type": OBS_TYPE, "height": OBS_HEIGHT, "width": OBS_WIDTH, "camera_name": CAMERA_NAME,
                        "camera_name_mapping": dict(CAMERA_NAME_MAPPING),
                        "canonicalization": "lerobot.sim.integrations.libero.canonical_observation (images rotated "
                                            "180 degrees; state = eef pos, axis-angle, gripper qpos, 8-D)"},
        "control_mode": CONTROL_MODE,
        "init_states": INIT_STATES,
        "hard_reset": HARD_RESET,
        "noise": NOISE_RULE,
        "action_postprocess": "policy postprocessor (unnormalizer), float32, no clipping (as lerobot-eval rollout)",
    }

"""Shared setup of the area-8 host tests (CPU only, no network, no GPU).

Run each file with the area venv in isolated mode:
    env -u HF_TOKEN .venv/bin/python -I -B tests/<file>.py
`-I` does not put the script directory on sys.path, so this module adds the area directory. The environment is
fixed as verify_cpu.py (P0) did: CUDA hidden, offline, LIBERO config of the area, Mesa software EGL for rendering
(the host has no OSMesa; README section 0.4). Temporary files go below $GATEDTRAIN_A8_TEST_TMP (default: the system
temporary directory). The end-to-end CPU smoke (test_e2e_cpu.py) runs only with GATEDTRAIN_A8_SLOW=1.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

AREA = Path(__file__).resolve().parent.parent
if str(AREA) not in sys.path:
    sys.path.insert(0, str(AREA))

TMP_ROOT = Path(os.environ.get("GATEDTRAIN_A8_TEST_TMP", tempfile.gettempdir()))
TMP_ROOT.mkdir(parents=True, exist_ok=True)

for _name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH"):
    os.environ.pop(_name, None)

CPU_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HOME": str(TMP_ROOT / "a8-test-hf-home"),
    "HF_LEROBOT_HOME": str(AREA / "data"),
    "LIBERO_CONFIG_PATH": str(AREA / "libero-config"),
    "TOKENIZERS_PARALLELISM": "false",
    "WANDB_MODE": "disabled",
    # Mesa software EGL (verify/libero-render.json, P0)
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
    "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/50_mesa.json",
    "LIBGL_ALWAYS_SOFTWARE": "1",
    "GALLIUM_DRIVER": "llvmpipe",
    "MUJOCO_EGL_DEVICE_ID": "1",
}
os.environ.update(CPU_ENV)

SLOW = os.environ.get("GATEDTRAIN_A8_SLOW") == "1"


def tmpdir(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix, dir=TMP_ROOT))


def rmtree(p: Path) -> None:
    shutil.rmtree(p, ignore_errors=True)


def make_fake_step_dir(root: Path, step: int) -> Path:
    """A LeRobot checkpoint step directory (common/train_utils.py:217-260 layout) with small files: the real
    smolvla_base config and processor JSON and state files, a fake weights file, a tokenizer artifact."""
    import json

    base = AREA / "models" / "smolvla_base"
    d = root / f"{step:06d}"
    pm = d / "pretrained_model"
    ts = d / "training_state"
    (pm / "tokenizer").mkdir(parents=True)
    ts.mkdir(parents=True)
    cfg = json.loads((base / "config.json").read_text())
    cfg.update(vlm_model_name=str(AREA / "models" / "SmolVLM2-500M-Video-Instruct"), load_vlm_weights=False,
               n_action_steps=10)
    (pm / "config.json").write_text(json.dumps(cfg, indent=4))
    for name in ("policy_preprocessor.json", "policy_postprocessor.json",
                 "policy_preprocessor_step_5_normalizer_processor.safetensors",
                 "policy_postprocessor_step_0_unnormalizer_processor.safetensors"):
        shutil.copyfile(base / name, pm / name)
    (pm / "model.safetensors").write_bytes(b"fake weights " * 100)
    (pm / "train_config.json").write_text("{}")
    (pm / "tokenizer" / "tokenizer.json").write_text("{}")
    (ts / "training_step.json").write_text(json.dumps({"step": step}))
    (ts / "optimizer_state.safetensors").write_bytes(b"fake optimizer")
    return d

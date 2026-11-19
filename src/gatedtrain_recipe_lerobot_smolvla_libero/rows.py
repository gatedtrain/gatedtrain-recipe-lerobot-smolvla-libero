"""Registry rows of adapter `lerobot_smolvla`: LeRobot SmolVLA on LIBERO-Object (area 8).

The training row, the evaluator rows and the adapter class, loaded by `gatedtrain.ext.registry` through the
`gatedtrain.adapters` entry point (MULTIMODAL-SPEC section 6)."""

from __future__ import annotations

from typing import Any

from pathlib import Path

from gatedtrain.ext.rowkit import CONTAINER_PATHS, python_tree, runtime_dir, venv_runtime_rows

NAME = "lerobot_smolvla"
# The runtime directory: this repository's runtime/ (scripts, its .venv, models, dataset, assets, LIBERO config, items;
# mounted read-only at its own host path), or the override; earlier runs lie under the host's output root as
# `{output_root}/<OUT_NAME>`. PYTHON_DIR is the directory of the uv-managed CPython the venv links to (mounted read-only
# at its own path; None without a venv).
RUNTIME_ENV = "GATEDTRAIN_RECIPE_LEROBOT_SMOLVLA_LIBERO_RUNTIME"
OUT_NAME = "lerobot-smolvla-libero"
RUNTIME = runtime_dir(RUNTIME_ENV, __file__)
HOST_ROWS = venv_runtime_rows(RUNTIME, OUT_NAME)
_TREE = python_tree(RUNTIME)
PYTHON_DIR = str(Path(_TREE).parent) if _TREE else None

TRAINER_ROWS: dict[str, dict[str, Any]] = {}
EVALUATOR_ROWS: dict[str, dict[str, Any]] = {}

# Area "VLA" (MULTIMODAL-SPEC section 6 row 8 and the row 8 deviations; the maintainers' decision record (not
# published) "Area 8 (VLA) decisions", "Area 8 probes P1/P2", "Area 8 base cut"; the runtime runtime/ (README
# sections 2, 4 to 6 and 11; a8_common.py, train_lerobot_logged.py sha256 93bf86f6..., eval_libero.py sha256 7846982d...,
# read 2026-11-15, re-pinned for ptc-hub 0.2.0 and gatedtrain 0.3.0) and its GPU outputs out/area8-vla/base-1 and base-1-eval2-<step>; tests/README.md section 11.26,
# EXT-233 to EXT-247). LeRobot 0.7.0 `lerobot-train` through the area launcher (in process, five module globals wrapped,
# no LeRobot edit): SmolVLA expert-only SFT of lerobot/smolvla_base on the 454 LIBERO-Object demonstrations, batch 32,
# two stages: a stage-1 base ladder from smolvla_base, then a stage-2 run from the chosen stage-1 cut; ext-collect gates
# stage 2's last cut against that cut (the spec's base, EXT-236). Full weights (the trained expert inside the whole
# model file), cuts `cuts/checkpoint-<step>/` with `cut.json` last.
TRAINER_ROWS["lerobot_smolvla"] = {
    "uses": ("sft.lerobot@0.1",),
    # from env-freeze.txt (MM-33): the framework, its torch and transformers, the simulator stack (README section 6.1).
    "pins": ("lerobot", "torch", "transformers", "hf-libero", "robosuite", "mujoco"),
    "outputs": ("cuts/checkpoint-<step>/ full-weight cuts with cut.json written last (hard links to LeRobot's "
                "checkpoints/<step>/)", "run_args.json", "env-freeze.txt", "env-runtime.json", "samples.jsonl",
                "attempts.jsonl", "cuts.jsonl"),
    # EXT-235: train_lerobot_logged.py write_cut `model_files` (= cut.json `deployable`): pretrained_model/* except the
    # resume-only train_config.json (weights, config, the two processor descriptions and their normalizer state files,
    # the saved tokenizer artifact), plus the deployment sampler. training_state/, train_config.json and base_ref.json
    # are not components.
    "components": ("pretrained_model/config.json", "pretrained_model/model.safetensors",
                   "pretrained_model/policy_postprocessor.json",
                   "pretrained_model/policy_postprocessor_step_0_unnormalizer_processor.safetensors",
                   "pretrained_model/policy_preprocessor.json",
                   "pretrained_model/policy_preprocessor_step_5_normalizer_processor.safetensors",
                   "pretrained_model/tokenizer/chat_template.jinja", "pretrained_model/tokenizer/tokenizer.json",
                   "pretrained_model/tokenizer/tokenizer_config.json", "sampler.json"),
    # EXT-238: observed per-sample log (samples.jsonl, written before each update by the launcher's wrapper of
    # _preprocess_dataset_batch); a log that does not match the run or the dataset refuses, there is no fallback.
    "consumption": {"target": "row", "fallback": None},
    # estimate: stage 1 (3,000 steps, 0.7 h) and its 12 probe evaluations (about 0.07 h each), the stage-2 smoke and
    # its two 50-item evaluations, stage 2 (2,500 steps at 0.5 s), two gate arms of 800 episodes (about 2.6 s per
    # episode with 8 workers, base-1-eval2-500).
    "gpu_h_estimate": 3.5,
    "evaluators": ("libero.lerobot_items@0.1",),
    "framework": "lerobot",
    # EXT-244: the interpreter tree, the runtime directory (scripts, venv, models, dataset, assets, LIBERO config, items)
    # and the training runs' outputs (stage 2 mounts the base cut read-only at /inputs/base-cut, an evaluation a cut at
    # /inputs/cut).
    "input_roots": HOST_ROWS["input_roots"],
    "spec_binding": False,                           # the smoke launch precedes the registration (sigma from the smoke)
    # README section 6.1 item 3 and the GPU runs' flags (records/area8-vla): every name, none a credential. Values are
    # judged at ext-launch (`launch_problems`): MUJOCO_GL and PYOPENGL_PLATFORM `egl`, NVIDIA_DRIVER_CAPABILITIES
    # `compute,utility,graphics` (the NVIDIA EGL library is injected only with `graphics`), the area's dataset home and
    # LIBERO configuration.
    "env_names": ("HF_DATASETS_OFFLINE", "HF_HOME", "HF_HUB_OFFLINE", "HF_LEROBOT_HOME", "HOME", "LIBERO_CONFIG_PATH",
                  "MUJOCO_EGL_DEVICE_ID", "MUJOCO_GL", "NVIDIA_DRIVER_CAPABILITIES", "PYOPENGL_PLATFORM",
                  "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED", "PYTORCH_CUDA_ALLOC_CONF", "TOKENIZERS_PARALLELISM",
                  "TRANSFORMERS_OFFLINE", "WANDB_MODE", "XDG_CACHE_HOME"),
    "scan_skip": HOST_ROWS["scan_skip"],
    "interpreter_trees": HOST_ROWS["interpreter_trees"],
    "interpreter": HOST_ROWS["interpreter"],
    "container_paths": CONTAINER_PATHS,
    # HF_HOME must be writable (the `datasets` Arrow cache, README section 0.4): a tmpfs or the output.
    "env_paths": ("HF_HOME", "HOME", "XDG_CACHE_HOME"),
    # The GPU runs use `python -I <script>` (README section 11.3); the scripts put their own directory on sys.path.
    "interpreter_flags": {"train_lerobot_logged.py": "-I", "eval_libero.py": "-I"},
}

# Area "VLA" evaluator (decisions "Area 8 (VLA) decisions", not published: deciding binary `success`, T3 with both disclosures, K 2,
# eval seeds 11 and 12, 400 gate items; `steps_to_success` a sub-metric; per-task success reported). Scorer:
# eval_libero.py (path E2, LeRobot's LiberoEnv and the cut's processors and select_action, per-item common random
# numbers); code_sha256 is the composite it writes (SHA-256 of the compact sorted JSON {eval_libero.py: sha256,
# a8_common.py: sha256}; eval_libero.py 7846982d..., a8_common.py ce29aa22...; re-pinned for ptc-hub 0.2.0 and gatedtrain 0.3.0, RESULT.md), the version the string it writes with
# the default item timeout (base-1-eval2-<step>, 2026-11-15). An item is one (task, init state) of LIBERO-Object; its
# payload strings are the suite, task, language and init file; the init state and task id are integers in the rows. The
# exact-overlap key is the item id, which no training row carries: the demonstrations keep no initial object poses, so
# the meaningful check (demonstration initial states against the benchmark's) is unverified and disclosed.
EVAL_LIBERO_CODE_SHA256 = "sha256:335559ce7c551af24a41986972bb61192ace2d3b746dc4763301eeecce5e1dfd"
EVALUATOR_ROWS["libero.lerobot_items@0.1"] = {
    "metric": "success",
    "direction": "higher_is_better",
    "score_range": (0, 1),
    "binary": True,
    # 05 section 6.1: a simulator measurement (the task's BDDL goal), not the training signal (the flow-matching
    # imitation loss), so T3; the demonstrations were rendered in the same simulator (R15-1 analogue) and were probably
    # kept only when they passed the same success check (R16-1 analogue): both stated in the registration and report.
    "tier": "T3",
    "construct_aligned": {
        "reward": ("flow-matching imitation loss on LIBERO-Object demonstrations recorded in the LIBERO/robosuite "
                   "simulator that evaluates"),
        "statement": ("success is the BDDL goal check of the LIBERO simulator in which the training demonstrations "
                      "were recorded and rendered: a pass shows that the policy completes the ten LIBERO-Object tasks "
                      "from held-out benchmark initial states more often in that simulator, not a gain in real-robot "
                      "manipulation or under another simulator or renderer (05 section 6.1, R15-1 analogue)")},
    "shared_annotation": {
        "convention": ("LIBERO's BDDL goal predicates (env.check_success()) of the ten LIBERO-Object tasks; the "
                       "lerobot/libero demonstrations were probably kept only when they passed the same check (454 of "
                       "500 remain; unverified)"),
        "statement": ("The success labels are LIBERO's BDDL goal checks of the same ten tasks, the check that probably "
                      "filtered the training demonstrations (unverified); no training stage consumed the evaluation's "
                      "initial states as far as can be checked (the dataset keeps no initial object poses), so the "
                      "programmatic scorer stays T3 (05 section 6.1, R16-1). Part of a gain may be learning that "
                      "filter's convention: a pass shows more BDDL-goal completions from held-out initial states of "
                      "the ten tasks, not a general manipulation gain.")},
    "error_classes": ("simulator_error",),           # MM-4: a simulator, renderer or policy exception, a worker death,
                                                     # the item wall-clock timeout (eval_libero.py)
    "score_outcomes": ("timeout", "failed", "nan_action"),   # MM-4: the policy's faults, scored 0
    "payload_text_field": "item",
    "item_payload": ("suite", "task", "language", "init_file", "init_file_sha256"),
    "normalizer": True,                              # MM-19: the arm names the evaluated cut's normalizer
    "env_version": True,                             # MM-19: the arm names the registered env_version.json
    # EXT-240: the deployment sampler (a8_common.deploy_sampler, every cut's sampler.json), exactly these keys.
    "sampler_keys": ("action_postprocess", "chunk_size", "control_mode", "hard_reset", "init_states", "max_action_dim",
                     "n_action_steps", "noise", "num_steps", "observation", "policy_type", "rename_map",
                     "resize_imgs_with_padding", "tokenizer_max_length"),
    "framework": "lerobot",
    "samples": "400, 2",
    "scorer": {"name": "libero_bddl_check_success",
               "version": ("eval_libero@0.1; success = LiberoEnv.step is_success (env.check_success(), BDDL goal) "
                           "within 280 steps after 10 settle steps; score 1 success, 0 timeout/failed/nan_action; "
                           "simulator exceptions, worker death and the item wall-clock timeout (300 s, retried once "
                           "alone) are simulator_error; hf-libero 0.1.4 robosuite 1.4.0 mujoco 3.8.1 lerobot 0.7.0"),
               "weights_sha256": None, "code_sha256": EVAL_LIBERO_CODE_SHA256},
}

PLUGIN = {"adapters": {"lerobot_smolvla": "gatedtrain_recipe_lerobot_smolvla_libero.adapter:LerobotSmolvlaAdapter"}, "trainer_rows": TRAINER_ROWS,
          "evaluator_rows": EVALUATOR_ROWS}

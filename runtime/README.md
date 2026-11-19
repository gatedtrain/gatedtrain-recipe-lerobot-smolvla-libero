# Area "VLA": LeRobot SmolVLA SFT on LIBERO-Object (runtime set up, prototype, unpromotable)

Multimodal prototype, decision row 8 "VLA | LeRobot | SmolVLA SFT, LIBERO-Object" (the maintainers' decision record, not published:
scope table and "Area 8 (VLA) decisions"), gatedtrain's `MULTIMODAL-SPEC.md` training row 8 (`lerobot`,
`sft.lerobot@0.1`), evaluation row 8 (`libero.lerobot_items@0.1`), MM-19, MM-26, MM-38, MM-GAP-10.

Status (2026-11-16): the image was built and the GPU runs are done: the stage-1 ladder `base-1` and the stage-2 run
`full-1` passed the gate on 2026-11-16 (`RESULT.md`). This file was written on 2026-11-14, when the runtime had been
set up on the development machine (downloads of the chosen "necessary set", area venv, image Containerfile) and
verified on CPU (probe P0, §0), before any GPU job. §1-§6
keep the research facts (read from the Hugging Face API without a token, PyPI, arXiv, and the source clones); where
P0 settled a fact the text says so. "estimate" marks an unmeasured number, "unverified" a fact not yet checked in code
or by a probe.

## 0. Runtime as set up (2026-11-14)

### 0.1 Downloads (`fetch_models.py`, `MANIFEST.json`)

`fetch_models.py` (run as `env -u HF_TOKEN <any venv with huggingface_hub>/bin/python fetch_models.py
--accept-libero-assets-terms`; without the flag it refuses, because `lerobot/libero-assets` states no license: check
its terms first, see the repository README) resolves nothing at run time:
every repository is pinned to the full revision below, read from the Hub API (`/api/{models,datasets}/<repo>/revision/
<short>`, no token) before the download. `token=False` on every call, `HF_HOME=hf-home/` (fresh), Xet off; the file list
of each directory is compared with the Hub's list at the revision (minus excluded patterns) and every size checked.
huggingface_hub's `.cache/` metadata is moved to `hf-meta/<name>/`. `MANIFEST.json` (sha256 `78ac9441…bcd4`) holds
every file's SHA-256 and bytes; `fetch_models.py --verify` re-hashed all 1,067 files: ok. No symbolic links (MM-20).

| Directory | Repository @ full revision | Files | Bytes | Licence |
|---|---|---|---|---|
| `models/smolvla_base/` | `lerobot/smolvla_base` @ `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6` (whole repo; `model.safetensors` sha256 `7cd549ac2351fb06…`) | 10 | 914,741,027 | apache-2.0 |
| `models/SmolVLM2-500M-Video-Instruct/` | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` @ `4cde9d208e19519c36917aa170fffc7e381aa0ad`, without `onnx/*` (24 ONNX files, 5.88 GB, not fetched) | 14 | 2,034,857,723 | apache-2.0 |
| `data/lerobot/libero/` | `lerobot/libero` (dataset) @ `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` (whole repo) | 457 | 1,935,512,060 | apache-2.0 |
| `assets/libero-assets/` | `lerobot/libero-assets` (dataset) @ `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6` (whole repo) | 586 | 422,320,936 | none on the card (disclosed) |
| total | | 1,067 | 5,307,431,746 | |

Nothing else was downloaded (no ONNX, no LIBERO HDF5, no other dataset). `HF_LEROBOT_HOME=data/` makes
`data/lerobot/libero` the dataset root for repo id `lerobot/libero`.

### 0.2 Venv (`pyproject.toml`, `uv.lock`, `.venv/`)

- uv project, `package = false`, Python 3.12.16 (`<python-dir>/cpython-3.12.16-linux-x86_64-gnu`, a standalone CPython build),
  `exclude-newer = 2026-11-06T00:00:00Z`, Linux x86_64 only. Dependencies: `lerobot[smolvla,libero,training]==0.7.0`,
  `torch==2.15.0`, `torchvision==0.29.2` (PyPI, CUDA 13.2 build: `torch.version.cuda == "13.2"`, the stack of areas 4-5),
  `av>=15,<16` (PyAV, bundled FFmpeg with libdav1d). `[tool.uv.extra-build-dependencies]` adds `cmake>=3.29,<4` to the
  isolated builds of the sdists `hf-egl-probe` and `egl-probe` (built fine; the development machine has no cmake).
- `uv lock` resolved 180 packages; **`uv.lock` sha256 `034e5c3b716d245dabc904e1f5c534c804f82bd536792136796a6ab47d075788`**.
  `uv sync --frozen` into `.venv/` (7.5 GB). Commands: `env -u HF_TOKEN UV_PYTHON_DOWNLOADS=never uv lock|sync --frozen
  --python <interpreter above>`.
- Key versions (`env-freeze.txt`, full `uv pip freeze` below its header): lerobot 0.7.0, torch 2.15.0, torchvision
  0.29.2, transformers 5.22.1, accelerate 1.15.0, datasets 5.1.0, huggingface-hub 2.4.0, av 15.1.0, torchcodec 0.17.0,
  numpy 2.2.6, mujoco 3.8.1, robosuite 1.4.0, robomimic 0.2.0, hf-libero 0.1.4, bddl 1.0.1, gymnasium 1.4.0,
  trackio 0.45.0, wandb 0.31.0. Matches LeRobot's own lock except huggingface-hub (2.4.0 vs 1.33.0, within
  lerobot's `>=1.31,<3`).
- Two file-level overlaps in the resolved set (no resolver conflict, but two distributions write the same module):
  `opencv-python` 5.0.0.93 (from hf-libero/robomimic) and `opencv-python-headless` 4.13.0.92 (lerobot) both install
  `cv2`; the importable one is 5.0.0 (non-headless, needs libGL/libX11 at import, present in `gatedtrain-a5-runtime:1`).
  `egl-probe` and `hf-egl-probe` both install `egl_probe`. Neither broke P0.

### 0.3 LIBERO configuration (`libero-config/config.yaml`)

`verify_cpu.py config` writes `libero-config/config.yaml` (keys `benchmark_root`, `bddl_files`, `init_states` = the
venv's `libero/libero` package paths, `datasets` = an unused in-area path, `assets` = `assets/libero-assets`); with
`LIBERO_CONFIG_PATH=libero-config` LIBERO never calls `input()`. LIBERO ignores the `assets` key: `get_assets_path()`
returns the module global `libero.libero._assets_path_cache` if set, else the package's `assets/` (absent in the
wheel), else downloads to `$HOME/.cache/libero/assets` (`libero/libero/__init__.py:68-90`). The area code sets
`libero.libero._assets_path_cache` to `assets/libero-assets` before building an env (no LeRobot or LIBERO edit); the
alternative is a run `HOME` whose `.cache/libero/assets` is a read-only mount of that directory. The config paths
are absolute host paths, so the venv must be mounted at the same path in the container (as areas 2-5 do).

### 0.4 CPU verification P0 (`verify_cpu.py`, results in `verify/*.json` and logs)

`verify_cpu.py` sets `CUDA_VISIBLE_DEVICES=""`, `HF_HUB_OFFLINE=1` (plus `HF_DATASETS_OFFLINE`, `TRANSFORMERS_OFFLINE`),
`HF_HOME=hf-home/`, `HF_LEROBOT_HOME=data/`, `LIBERO_CONFIG_PATH=libero-config/`, removes token variables, and
replaces `socket.connect` with a guard that refuses every non-loopback connection and records it. Every step reported
`blocked_connections: []`: nothing tried to reach the network.

| Step | Result | Evidence |
|---|---|---|
| 1 env | `import lerobot` ok; versions above; `torch.cuda.is_available()` False with `CUDA_VISIBLE_DEVICES=""` | `verify/env.json`, `env-freeze.txt` |
| 2 dataset | `LeRobotDataset("lerobot/libero", root=data/lerobot/libero, episodes=807..1260, video_backend="pyav", revision=a1aaacb7…)` loads offline in 1.9 s: **454 episodes, 66,984 frames**, fps 10. Features: `observation.images.image`, `observation.images.image2` (video, 256×256×3, codec `av1`, decoded by PyAV's `libdav1d`), `observation.state` float32 [8], `action` float32 [7], `timestamp`, `frame_index`, `episode_index`, `index`, `task_index` (int64 [1]). Items 0, 1, 33,492, 66,983 decoded (images float32 [3,256,256] in [0,1], means 0.52-0.59) in 0.01-0.29 s; per-item `index` / `episode_index` / `frame_index` e.g. 153,511 / 807 / 0 (first Object frame), 220,494 / 1260 / 151 (last); `task` string per item | `verify/dataset.json` |
| 3 policy | `smolvla_base` from `models/smolvla_base` with `vlm_model_name` set to `models/SmolVLM2-500M-Video-Instruct` (config field, no LeRobot edit), `input_features`/`output_features` emptied (= `--policy.input_features=null --policy.output_features=null`), `make_policy(cfg, ds_meta, rename_map={image→camera1, image2→camera2})`: features become `camera1`, `camera2` [3,256,256], `observation.state` [8], `action` [7] (state/action are zero-padded to 32 inside the model, so no weight shape changes). 450,046,176 parameters, 99,880,992 trainable (expert only, base defaults), bf16 + fp32. Batch of 2 (episode 807 frames 0-1, `action` [2,50,7] via `delta_timestamps`): preprocessor (tokenizer 48 tokens) ok, **forward 4.9 s, loss 1.396 (finite), backward 29.1 s, `select_action` 3.7 s → action [1,7]**; peak RSS 3.9 GB, wall 42 s for the whole step | `verify/policy.json`, `verify/policy.log` (`/usr/bin/time -v`) |
| 3b backbone | `load_vlm_weights=false` gives the same 500 state-dict tensors with equal values as `true` after the policy load (dtypes not compared): the 2 GB SmolVLM2 `model.safetensors` is not needed at run time, only its config, processor and tokenizer files. `AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")` (the repo id in `policy_preprocessor.json`) **fails offline** (`OSError`, not in the hub cache): the run must override `tokenizer_processor.tokenizer_name` with the local directory (done in P0 through `make_pre_post_processors(..., preprocessor_overrides=...)`) | session check, script not kept |
| 4 LIBERO | `benchmark.get_benchmark_dict()["libero_object"]()`: 10 tasks in the order of §4; init states of task 0 (LIBERO's own `torch.load(..., weights_only=False)` of its packaged `.pruned_init`): ndarray float64 (50, 110). Physics-only `ControlEnv` (no renderer, no camera obs) for task 0: build + reset + `set_init_state(init[0])` 4.4 s, 10 zero-action steps 2.6 s, 38 observation keys, `check_success()` False | `verify/libero.json` |
| 4 render | Host has no `libOSMesa` (`ldconfig -p`), so `MUJOCO_GL=osmesa` fails (`'NoneType' object has no attribute 'glGetError'`, `verify/libero-render-osmesa.log`). CPU rendering does work through **Mesa's software EGL**: `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe MUJOCO_EGL_DEVICE_ID=1`: `OffScreenRenderEnv` 256×256, build + reset + 1 step 2.1 s, agentview mean 102.4; mapped GL libraries `libEGL_mesa`, `libgallium-25.2.8`, no NVIDIA library | `verify/libero-render.json`, `verify/libero-render-mesa-egl.log` |

Findings for the run configuration:
- robosuite's EGL context takes the device from `MUJOCO_EGL_DEVICE_ID`, else from `CUDA_VISIBLE_DEVICES`, and
  raises `ValueError: invalid literal for int()` when `CUDA_VISIBLE_DEVICES` is empty
  (`robosuite/renderers/context/egl_context.py:36-56`); runs must set `MUJOCO_EGL_DEVICE_ID` (add it to the env
  allow-list) or a non-empty `CUDA_VISIBLE_DEVICES`.
- `datasets` writes an Arrow cache of the parquet files to `$HF_HOME/datasets` (6.5 MB here) even offline: in the
  container `HF_HOME` (or `HF_DATASETS_CACHE`) must point to a writable per-run directory, not the read-only mount.
- `lerobot-train` builds the preprocessor from the saved `policy_preprocessor.json` and only overrides device,
  normalizer and rename (`lerobot_train.py:666-692`), so the tokenizer repo id stays. Offline training needs one of:
  a launcher that wraps `make_pre_post_processors` to add the `tokenizer_processor` override; a hub-cache layout of
  the SmolVLM2 tokenizer files under the run's `HF_HOME` (a derived copy, not a download); or a derived policy
  directory whose `policy_preprocessor.json` names the local path. Maintainers' choice; **unverified** with
  `lerobot-train` itself.
- robosuite prints "No private macro file found" warnings (harmless) and a `SyntaxWarning` at import.

### 0.5 Image (`container/Containerfile` of this repository)

`gatedtrain-a8-runtime:1` = `FROM gatedtrain-a5-runtime:1` plus `libegl1 libopengl0 libgles2 libglvnd0 libegl-mesa0 libosmesa6`
from the Ubuntu snapshot `20261029T000000Z` (area 5's), the glvnd vendor file
`/usr/share/glvnd/egl_vendor.d/10_nvidia.json` (`libEGL_nvidia.so.0`), `ENV NVIDIA_DRIVER_CAPABILITIES=
compute,utility,graphics`, and the package list in `/etc/ptc-a8-packages.txt` (a file name kept from the project's earlier name, so that the
RUN instruction stays byte-identical to the one that built the pinned image); user stays root as in the base. Build
(from `container/`, after building `gatedtrain-a5-runtime:1` from `container/Containerfile.a5`, a copy of the SD3.5 OCR
recipe's `container/Containerfile`):

    docker build --network host -t gatedtrain-a8-runtime:1 -f Containerfile .

Not built yet; GPU EGL inside it (whether the toolkit injects `libEGL_nvidia.so.0` with `graphics`) is **unverified**
until probe P1.

## Local material (this directory)

The source clones were kept outside this directory (`<data-root>/area8-ref/`,
never mounted into a run): `area8-ref/lerobot/` (LeRobot v0.7.0, commit
`410c095eacc0011d998a73383eb94abed2715af0`) and `area8-ref/hf-libero/` (huggingface/LIBERO commit
`8561c60eea2fb93096146f240194649df73d8b1e`, `version="0.1.4"` = PyPI `hf-libero` 0.1.4). Paths `ref/lerobot/...` and
`ref/hf-libero/...` cited below refer to those clones.

| Path | What | Source |
|---|---|---|
| `fetch_models.py`, `MANIFEST.json`, `hf-meta/`, `hf-home/` | download script, file manifest, moved download metadata, HF home (run-time caches) | §0.1 |
| `models/`, `data/`, `assets/` | pinned downloads | §0.1 |
| `pyproject.toml`, `uv.lock`, `.venv/`, `env-freeze.txt` | area venv | §0.2 |
| `libero-config/config.yaml` | LIBERO config | §0.3 |
| `verify_cpu.py`, `verify/` | CPU probe P0 and its results | §0.4 |
| `a8_common.py`, `train_lerobot_logged.py`, `eval_libero.py`, `probe_render.py`, `tests/` | area code: shared helpers and deployment sampler, training launcher, evaluator (path E2), render probe P1, host tests | §11 |
| `items-gate/`, `items-probe/`, `ITEMS-MANIFEST.json` | registered item sets (written by `eval_libero.py --write-items`) | §11 |
| `meta/smolvla_base/` | config, README, pre/postprocessor JSON of `lerobot/smolvla_base` @ `5e8d12a6` | HF, < 10 kB each |
| `meta/lerobot_smolvla_libero/` | config, `train_config.json`, processors of `lerobot/smolvla_libero` @ `a24f1f38` | HF |
| `meta/smolvla_libero/` | config, processors of `HuggingFaceVLA/smolvla_libero` @ `6721902b` | HF |
| `meta/smolvla_libero_ckpts/020000/` | `train_config.json`, config, step and scheduler state of `HuggingFaceVLA/smolvla_libero_ckpts` @ `23932fa7` | HF |
| `meta/lerobot_libero/` | README, `meta/info.json`, `meta/stats.json`, `meta/tasks.parquet`, `meta/episodes/chunk-000/file-000.parquet` of `lerobot/libero` @ `a1aaacb7` | HF, 87 kB largest |

Small data files (98 parquet files `data/chunk-000/file-211..308`, 5.1 MB, states and actions only, no video) were
read in a temporary directory to map LIBERO-Object episodes to tasks and are not kept here. The source clones (`ref/`)
are kept outside this directory (`<data-root>/area8-ref/`).

## 1. Model

| Fact | Value | Evidence |
|---|---|---|
| Base repo, revision | `lerobot/smolvla_base` @ `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6` (lastModified 2026-10-06) | HF API `api/models/lerobot/smolvla_base`, read 2026-11-14 |
| Licence, gating | Apache-2.0 (card `license: apache-2.0`), not gated | same |
| Files | `model.safetensors` 906,712,520 B (sha256 `7cd549ac…a01eb`), config 2.3 kB, two normalizer state files 640 B each (identical sha256 `490ab239…7c8`), notebook, 8 MB GIF | same, `?blobs=true` |
| Parameters | 450.05 M: 446.77 M BF16 + 3.27 M F32; VLM text layers 204.6 M, action expert 98.3 M, vision encoder 86.4 M, `lm_head` 47.3 M, connector 11.8 M | safetensors header (HTTP range read of the first bytes only) |
| VLM backbone | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` (card `base_model`; config `vlm_model_name`), first 16 LLM layers (`num_vlm_layers: 16`) | `meta/smolvla_base/config.json` |
| Backbone repo | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` @ `4cde9d208e19519c36917aa170fffc7e381aa0ad` (lastModified 2026-11-13), Apache-2.0, not gated; non-ONNX files about 2.04 GB (`model.safetensors` 2,029,990,624 B + tokenizer/processor files about 5 MB); ONNX files (about 8 GB) are not needed | HF API |
| Why the backbone repo is needed | `smolvla_base` config has `load_vlm_weights: true`; `SmolVLMWithExpertModel` then calls `AutoModelForImageTextToText.from_pretrained(model_id)` and always `AutoProcessor.from_pretrained(model_id)` | `ref/lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py:90-101`; the class default is `load_vlm_weights=False` (`configuration_smolvla.py:86`). P0 (§0.4 step 3b): with `load_vlm_weights=false` the loaded policy's 500 tensors equal those of `true`, so the SmolVLM2 weights file is not needed; config, processor and tokenizer files still are |
| Action chunking | `chunk_size 50`, `n_action_steps 50`, flow matching with `num_steps 10` | `meta/smolvla_base/config.json` |
| Trained parts by default | `freeze_vision_encoder: true`, `train_expert_only: true` (about 100 M trainable: expert plus projections) | same; the paper: "We train only the action expert module, keeping the VLM frozen" (arXiv 2506.01844v1 §4.3, read 2026-11-14) |
| Embodiment of the base | SO100: state 6, action 6, cameras `camera1..3` 256×256 | `meta/smolvla_base/config.json` |

**LIBERO checkpoints found** (HF search `smolvla libero`, 2026-11-14):

| Repo @ revision | What | Usable as our base? |
|---|---|---|
| `lerobot/smolvla_libero` @ `a24f1f380461b665158d52acd140152c79e8b36d` (2026-10-12), Apache-2.0, 906.7 MB, same 450 M architecture | SFT from `smolvla_base` on all of `lerobot/libero` (4 suites, 1,693 episodes), 25,000 steps × batch 32, **full fine-tune** (`freeze_vision_encoder false`, `train_expert_only false`), rename map `image→camera1`, `image2→camera2`, state 8, action 7, `n_action_steps 50` | `meta/lerobot_smolvla_libero/train_config.json`; LeRobot's own docs evaluate it (`docs/source/simulation.mdx:20-26`). No published LIBERO-Object number for it was found (**unverified**); likely near ceiling (see §4) |
| `HuggingFaceVLA/smolvla_libero` @ `6721902b` (2025-09-17), 1.22 GB | Different architecture: `vlm_model_name SmolVLM2-500M-Instruct`, `num_vlm_layers 0`, `expert_width_multiplier 0.5`, `n_action_steps 1` | `meta/smolvla_libero/config.json`; not a fine-tune of `smolvla_base` despite the card's `base_model` |
| `HuggingFaceVLA/smolvla_libero_ckpts` @ `23932fa7`, 40.5 GB | Steps 20k…100k of a 2.2B-VLM variant (`SmolVLM2-2.2B-Instruct`, `load_vlm_weights false`) on `physical-intelligence/libero`, 5.69 GB weights + 2.41 GB optimizer per step | `meta/smolvla_libero_ckpts/020000/pretrained_model/train_config.json`; not usable (other architecture, size) |

**Memory on 24 GB (estimates, to be replaced by the probe).** Inference: 450 M BF16 weights about 0.9 GB plus
activations, about 2 GB. Training, expert only, batch 32: frozen BF16 VLM plus 100 M trainable parameters with fp32
Adam (about 1.6 GB) plus activations, 6-12 GB. Full fine-tune (the `lerobot/smolvla_libero` recipe): 450 M × 16 B
about 7.2 GB of weights, gradients and Adam state plus activations through the vision tower at 512 px, 14-22 GB.
LeRobot states 20k steps at batch 64 take about 4 h on one A100 (`docs/source/smolvla.mdx:50`, :62-63), about
0.7 s per step; an RTX 4090 is assumed 1.0-1.5× slower (**estimate**).

## 2. Framework: LeRobot

| Fact | Value | Evidence |
|---|---|---|
| Version | 0.7.0 (PyPI upload 2026-11-05), Apache-2.0, `requires-python >=3.12` | PyPI JSON; `ref/lerobot/pyproject.toml:28`, :32 |
| Why 0.7.0 | 0.6.1 pins `torch<2.12` (PyPI JSON), so it cannot share the torch 2.15.0 + CUDA 13.2 stack that areas 4 and 5 run on this driver (610.57); 0.7.0 pins `torch>=2.7,<2.16`, `transformers>=5.22,<5.23` | PyPI JSON; area venvs `area4-whisper/.venv`, `area5-sd35/.venv` (`torch-2.15.0`, `nvidia_cuda_runtime-13.2.86`) |
| Extras | `smolvla` (transformers 5.22.x, num2words, accelerate), `libero` (`hf-libero>=0.1.4,<0.2.0`, dataset, scipy), `training` (dataset, accelerate, trackio) | `pyproject.toml:317`; PyPI `requires_dist` |
| Upstream lock (for reference) | torch 2.15.0, torchvision 0.29.2, transformers 5.22.1, accelerate 1.15.0, datasets 5.1.0, huggingface-hub 1.33.0, av 15.1.0, torchcodec 0.17.0 (+cpu), gymnasium 1.4.0, mujoco 3.8.1, robosuite 1.4.0, robomimic 0.2.0, hf-libero 0.1.4, bddl 1.0.1, numpy 2.2.6, trackio 0.45.0, wandb 0.31.0 | `ref/lerobot/uv.lock` |
| Date cut | transformers 5.22.1 was uploaded 2026-10-30, so the areas' `exclude-newer = 2026-10-29` cannot resolve 0.7.0; proposal `exclude-newer = 2026-11-06` | PyPI JSON |
| Train entry point | `lerobot-train` = `lerobot.scripts.lerobot_train:main`; `--policy.path=<smolvla_base dir>` for SFT; no `scripts/train.py` in 0.7.0 | `ref/lerobot/src/lerobot/scripts/lerobot_train.py`; `docs/source/libero.mdx:179-189`, `docs/source/peft_training.mdx:17-33` |
| Camera and feature remap from `smolvla_base` | `--policy.input_features=null --policy.output_features=null` plus `--rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'` | `docs/source/peft_training.mdx:20-23`; `meta/lerobot_smolvla_libero/train_config.json` (`rename_map`, `pretrained_path lerobot/smolvla_base`); empty features are refilled from the dataset by `make_policy` (`policies/factory.py`, `if not cfg.input_features`), verified in P0 with `make_policy` (§0.4 step 3); the CLI `null` path through `lerobot-train` is **unverified** |
| Data order | Map-style datasets always use `EpisodeAwareSampler(shuffle=True, seed=cfg.seed)`; the order is a pure function of `(seed, epoch)` | `lerobot_train.py:355-371`; `datasets/sampler.py:26-43` |
| Loop | `batch = next(dl_iter)` then `_preprocess_dataset_batch(...)` then `update_policy(...)`, once per step | `lerobot_train.py:852-874`; `_preprocess_dataset_batch` at :167-178 is a module-level function called at :861 (and at :916 only for the held-out loss eval, `eval_steps > 0`) |
| Checkpoint layout | `<output_dir>/checkpoints/<step zero-padded to max(6, len(str(steps)))>/pretrained_model/` (`config.json`, `model.safetensors`, `train_config.json`, `policy_preprocessor.json`, `policy_preprocessor_step_*.safetensors`, `policy_postprocessor.json`, `policy_postprocessor_step_*.safetensors`) and `training_state/` (`optimizer_state.safetensors`, `optimizer_param_groups.json`, `rng_state.safetensors`, `scheduler_state.json`, `training_step.json`); `checkpoints/last` symlink replaced after each save | `common/train_utils.py:94-147`, :217-260 (docstring tree); `lerobot_train.py:937-951` |
| Save rule | every `save_freq` steps and always after the last step | `train_utils.py:124-132` |
| Completion marker | none: a step directory is complete once `last` points at it (written after `save_checkpoint` returns) | `lerobot_train.py:937-951` |
| Resume | `--resume=true --config_path=<ckpt>/pretrained_model/train_config.json`; the sampler offset is derived from the saved step, sample-exact for the same batch size and world size; processor stats come from the checkpoint | `lerobot_train.py:373-404`, :686-692; `configs/train.py:140`, :242-281 |
| Normalizer (MM-19) | A fresh `policy.path` run replaces the base's normalizer stats with `dataset.meta.stats` (renamed); a resume keeps the checkpoint's | `lerobot_train.py:666-692`. The state files are `policy_preprocessor_step_5_normalizer_processor.safetensors` and `policy_postprocessor_step_0_unnormalizer_processor.safetensors` (names in the `smolvla_base` and `lerobot/smolvla_libero` file lists) |
| Logging and offline | Experiment tracking is Trackio, `enable: False` by default; a legacy `wandb` section is migrated (`configs/default.py:136-147`, `configs/train.py:65-84`); `hf-libero` still installs `wandb` as a dependency (PyPI JSON) but nothing calls it on our path (**unverified**). `--policy.push_to_hub=false`; `--env_eval_freq=0` (default 20,000, `configs/train.py:162`). Local datasets load from `--dataset.root` "while you're offline" (`datasets/lerobot_dataset.py:85`); P0 loaded the dataset, the policy and its processors with `HF_HUB_OFFLINE=1` and a socket guard (§0.4), but only with a local tokenizer override; a full offline `lerobot-train` run is **unverified** |
| Video decoding | `lerobot/libero` stores AV1 video; torchcodec 0.17.0+cpu needs system FFmpeg libraries, PyAV 15.1.0 wheels bundle FFmpeg 7 with `libdav1d` and `libaom` (wheel `av.libs/`, listed 2026-11-14). Use `--dataset.video_backend=pyav` so the image needs no FFmpeg; P0 decoded the AV1 frames with PyAV's `libdav1d` on the host (§0.4 step 2). The venv's torchcodec is the PyPI build 0.17.0, not LeRobot's `+cpu` index build (LeRobot's `[tool.uv.sources]` does not carry over), unused with `pyav` | `utils/import_utils.py:91-106`; PyAV wheel contents |

**Consumption hook for `row` precision (MM-26).** Every training batch carries `index`, `episode_index` and
`frame_index` (dataset features, `meta/lerobot_libero/meta/info.json`). A launcher `train_lerobot_logged.py` (area
code, no LeRobot edit) wraps the module global `lerobot.scripts.lerobot_train._preprocess_dataset_batch` and, before
calling the original, appends one line per step to `samples.jsonl` (`step`, and per sample `index`, `episode_index`,
`frame_index`), flushed and fsynced before `update_policy` runs (`lerobot_train.py:861`, :864). `eval_steps` stays 0
so the wrapper sees training batches only. Each sample also loads the next 49 actions of the chunk
(`chunk_size 50` through `delta_timestamps`); the row is the anchor frame, and the report discloses that a row's
action chunk reaches later frames of the same episode. Because the sampler is a pure function of `(seed, epoch)`,
the log can also be cross-checked by regenerating the permutation (area 1 style), but the log is the observed
record. In the v3.0 layout many episodes share one data file (`data/chunk-000/file-NNN.parquet`), so MM-26's
"shard = episode file" becomes "shard = the data file holding the episode" (`meta/episodes` columns
`data/chunk_index`, `data/file_index`); a spec wording change, not a design change.

## 3. Data: `lerobot/libero` (LIBERO-Object part)

| Fact | Value | Evidence |
|---|---|---|
| Repo, revision | `lerobot/libero` @ `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` (lastModified 2026-06-08), Apache-2.0 (card), not gated | HF API |
| Format | LeRobot v3.0, 1,693 episodes, 273,465 frames, 40 tasks, fps 10 (metadata), two AV1 videos 256×256 (`observation.images.image` agent view, `image2` wrist), state 8, action 7 | `meta/lerobot_libero/meta/info.json` |
| Size | 1.936 GB total: data 20 MB (377 parquet files), video 1.03 + 0.88 GB (37 files per camera) | HF API `?blobs=true` |
| LIBERO-Object episodes | task indices 20-29 ("pick up the X and place it in the basket"), episodes **807-1260 (454 episodes, 66,984 frames)**, data files 211-308, video files 18-27 (files 18 and 27 shared with neighbour suites) | `meta/tasks.parquet`; `meta/episodes` parquet; task_index read from data files 146, 211, 308 (boundaries) and all of 211-308 |
| Episodes per task | 45, 45, 45, 46, 44, 45, 47, 45, 42, 50 for task indices 20…29 (orange juice, ketchup, cream cheese, bbq sauce, alphabet soup, milk, salad dressing, butter, tomato sauce, chocolate pudding); mean length 132-159 frames, max 254 | same |
| Object-only download | 118 files, 591 MB (+ meta); the whole repo is 1.94 GB | HF API sizes |
| Alternatives | `HuggingFaceVLA/libero` (same episodes, PNG in parquet, 34.9 GB), `physical-intelligence/libero` (CC-BY-4.0, 34.9 GB, v2.1), `lerobot/libero_object_image` (9.25 GB, Apache-2.0), `IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot` (0.54 GB, no licence on the card) | HF API; `docs/source/libero.mdx:153-169` recommends `lerobot/libero` and pinning `--dataset.revision` |
| Normalization stats | `meta/stats.json` covers the whole dataset; `LeRobotDataset(..., episodes=807..1260)` does not recompute them: P0's `ds.meta.stats` state mean equals `meta/stats.json` (all four suites) to 4 decimals (`verify/dataset.json`); so the normalizer of an Object-only run uses whole-dataset stats unless the run recomputes them (whether `lerobot-train` does is **unverified**) (`datasets/factory.py:137-171` passes `episodes` to `LeRobotDataset`; the stats source is `dataset.meta.stats`, `lerobot_train.py:666`) | |

**Held-out unit.** Training uses demonstrations; evaluation uses LIBERO's fixed benchmark initial states (§4). The
split is by initial state (scenario), as 05 §3 (`design/05-evaluation-and-lineage.md:105`, "a (task, init_state)
pair (VLA)") and 05:326/348 ("held-out scenario distributions", "splits by scenario") ask. Whether any demonstration
started from one of the 50 benchmark states is **unverified**: the LeRobot dataset keeps no object poses. Check (CPU,
optional download): compare `data/demo_*/states[0]` of the original LIBERO HDF5 (`yifengzhu-hf/LIBERO-datasets`
@ `f13aa24a`, Apache-2.0, `libero_object/*_demo.hdf5`, 664-805 MB per task, 7.44 GB for all ten) with the 50
rows of the task's init file. The 454 of 500 demonstrations that remain (50 per task in the original, a count from
the LIBERO paper, **unverified** here) suggest the re-rendered dataset dropped failed replays, i.e. the
demonstrations were filtered with the same simulator success check the gate uses (**unverified**; disclosure).
Holding out whole tasks would give success near 0 (language and object generalization), a degenerate baseline.

## 4. Simulator evaluation

| Fact | Value | Evidence |
|---|---|---|
| Simulator stack | `hf-libero` 0.1.4 (MIT, PyPI), which pins `robosuite==1.4.0`, `robomimic==0.2.0`, `bddl==1.0.1`, `mujoco>=3.0,<3.9`, `hf-egl-probe>=1.0.1`, `wandb` | PyPI JSON |
| Builds from source | `hf-egl-probe` 1.0.2 and `egl-probe` 1.0.2 (robomimic) are sdists that run `cmake` (setup.py of the sdist); the development machine has gcc but no `cmake` (`which cmake` empty); the sdist vendors the EGL headers (glad). Done: `cmake>=3.29,<4` from PyPI in `[tool.uv.extra-build-dependencies]`; both built (§0.2) | sdist listing; host check |
| Assets | not in the wheel or the clone; `get_assets_path()` uses the package's `assets/` if present, else downloads `lerobot/libero-assets` into `$HOME/.cache/libero/assets` | `ref/hf-libero/libero/libero/__init__.py:68-90`; `utils/download_utils.py:109`, :127-139. `lerobot/libero-assets` @ `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`, 0.42 GB, no licence on the card (LeRobot's sim image pins this revision, `docker/Dockerfile.sim:55`) |
| Config file | LIBERO calls `input()` at import when `$LIBERO_CONFIG_PATH/config.yaml` (default `~/.libero`) is missing | `__init__.py:5-8`, :93-100; `docker/Dockerfile.sim:47-58` writes it at build time. Done: `libero-config/config.yaml` (§0.3), no prompt in P0 |
| Suite | `libero_object`: 10 tasks in benchmark order alphabet soup, cream cheese, salad dressing, bbq sauce, ketchup, tomato sauce, butter, milk, chocolate pudding, orange juice | `ref/hf-libero/libero/libero/benchmark/libero_suite_task_map.py:14-25` |
| Initial states | **50 per task**, `float64 (50, 110)` per `.pruned_init` file (identical to `.init`), all ten tasks | pickle opcodes read with `pickletools` (no unpickling) from `ref/hf-libero/libero/libero/init_files/libero_object/*.pruned_init`; P0 confirmed (50, 110) float64 for task 0 through LIBERO's own loader (§0.4 step 4) |
| Episode | 10 no-op settle steps after reset, control 20 Hz, at most **280 steps** for `libero_object` (longest demo 254), success = `env.check_success()` (the task's BDDL goal) at any step, which terminates the episode; no reward shaping is used | `ref/lerobot/src/lerobot/sim/integrations/libero.py:134-140`, :163-164, :375-400, :402-427 |
| Init-state mapping | each vector sub-env `i` starts at init state `i` and advances by `n_envs` per reset; the vector env never autoresets | `libero.py:209-211`, :380-382; `sim/local_vector_env.py:15`, :338-348 |
| Seeds | `lerobot-eval` seeds batch `b`, env `i` with `start_seed + b·n + i` (`--seed`), and calls `set_seed(cfg.seed)` once for torch | `scripts/lerobot_eval.py:580-593`, :833 |
| So | with `--eval.n_episodes=50 --env.init_states=true`, episode `e` of task `t` is init state `e` with env seed `seed + e`; `eval_info.json` keeps per-task success lists (`per_task[].metrics.successes`) | `lerobot_eval.py:938-939`, :1071-1078 |
| Rendering | MuJoCo with `MUJOCO_GL=egl` (`docs/source/libero.mdx:40-44`); LeRobot's sim image installs `libegl1 libgl1 libglx0 libgles2 libosmesa6 ...` and sets `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NVIDIA_DRIVER_CAPABILITIES=all` (`docker/Dockerfile.sim:29-37`). Default observation 360×360 for eval (`envs/configs.py:479-480`), policy resizes to 512 with padding | |
| Hard reset | default on (rebuilds model and renderer each episode); soft reset is faster but not bit-identical (`docs/source/libero.mdx:103-115`) | |

**Item and repeats.** Item `libero_object/t<task_id>/s<init_state>` (MM-1 `item_meta {task, task_id,
init_state}`), payload: suite, task name, language, init file sha256 and row index. K repeats use registered
`eval_seeds`; a repeat differs in the env seed and the flow-matching noise (torch), the initial state is the same.
Per-item common random numbers across arms hold only approximately with stock `lerobot-eval` (one torch seed per run;
batches run until their longest episode, so the noise stream diverges after the first batch). Two evaluator paths:
- **E1 (spec row 8 as written):** `lerobot-eval` per suite run (`--env.task=libero_object --eval.n_episodes=50
  --eval.batch_size=B --seed=S`), converted per item from `eval_info.json` by the mapping above. A simulator crash
  aborts the whole run (no per-item `error`).
- **E2 (proposed):** own `eval_libero.py` reusing LeRobot's `LiberoEnv`, processors and `rollout()`, one batch per
  (task, block of init states), torch reseeded per batch from (item block, repeat), per-item `try` so a simulator
  exception becomes `error: simulator_error` (MM-4), episode logs (`output_ref`), timing, and per-item steps to
  success. Maintainers' choice; E1 serves as a cross-check on the probe items.

**Published numbers.** SmolVLA (0.45 B) LIBERO Spatial 90, Object **96**, Goal 92, Long 71 (%), 10 trials per task,
trained 100k steps at batch 64 on `physical-intelligence/libero`, "only initialized from the VLM" (arXiv 2506.01844v1
Table 2 and §4.3, read 2026-11-14). `smolvla_base` zero-shot on LIBERO is not meaningful: SO100 state 6 and action 6
against LIBERO 8 and 7, other cameras (`meta/smolvla_base/config.json` vs `meta/lerobot_libero/meta/info.json`);
expected success 0 (MM-38 degenerate). A fully trained `lerobot/smolvla_libero` is likely near the 96 % ceiling on
Object (**estimate**), leaving little headroom.

**MM-38 consequence.** A binary metric is degenerate only at mean 0 or 1 (MULTIMODAL-SPEC MM-38). The base must be an
under-trained SFT checkpoint (spec evaluation row 8 already says so). Proposal: two phases, as area 2 (decision
"Area 2 decisions", base = cut 8 of the base run):
1. **Base phase:** SFT from `smolvla_base` on the 454 LIBERO-Object episodes, saving every 250-500 steps up to about
   3,000 steps; evaluate the cuts on the **probe items** (below) and pick, before any candidate result, the first cut
   whose success lies in about 0.2-0.6.
2. **Continuation:** a fresh run with `--policy.path=<base cut>/pretrained_model` (new optimizer and warm-up, same
   data and stats, so the normalizer is unchanged), length set from the probe's learning curve (about 5,000-10,000
   steps, **estimate**). The gate compares the continuation's last cut with the base cut.

## 5. Gate design (proposal)

- **Deciding metric:** `success` in {0, 1}, higher is better, range [0, 1], `binary: true`; sub-metrics
  `steps_to_success` (or 280), `n_policy_calls`, `wall_s`. Tier **T3**: a simulator metric (05 §6.1 table,
  `design/05-evaluation-and-lineage.md:252`, "simulator metrics") computed by the BDDL goal check, while the training
  signal is the flow-matching imitation loss, not a reward. Disclosures: (R15-1 analogue) the simulator that renders
  the training demonstrations is the evaluation simulator; (R16-1 analogue) the demonstrations were probably kept only
  if they passed the same success check (**unverified**, §3), so part of a gain may be learning that filter's
  convention.
- **Score vs error (MM-4):** timeout at 280 steps, wrong object, dropped object, NaN action = score 0; MuJoCo
  instability exception, EGL failure, worker death = `simulator_error`.
- **Items:** 10 tasks × 50 benchmark initial states = 500. To keep the base-cut choice and the sigma pilot off the
  gate items (selecting a base whose measured success is low by chance would bias toward PASS), use init states 40-49
  (100 items) as **probe items** and init states 0-39 (**400 gate items**).
- **K and seeds:** K 2, `eval_seeds [11, 12]` (recommended), or K 1.
- **Sigma:** `sigma_plan: fixed` = max(1.5 × ucb80(s_pilot), √2 × ucb80(s_base)) from a pilot of the base cut and the
  smoke cut on probe items (decisions for areas 5-7). For K 1 and independent binary outcomes at p0 0.4, p1 0.65:
  paired SD about 0.68; with a 20-item pilot sigma about 1.21, with 50 items about 1.13. Required items (one-sided
  α 0.025, power 0.8): planning effect 0.15 → 444-508; 0.20 → 250-286; 0.10 → about 1,000. With K 2 the per-item SD
  falls (by up to √2 for the noise part), so 400 items × K 2 fits a planning effect of about 0.15 (**estimate**).
  Achieved power at a true +0.25 is about 1.0 for N 300-500; at a true +0.10, N 500 gives 0.89 (normal
  approximation, this README's arithmetic).
- **Clustering:** items within a task share the task, so outcomes are correlated within 10 clusters; the gate's
  per-item paired analysis does not model that (08 §7 P6b asks for "intervals clustered by scenario",
  `design/08-families-execution-and-performance.md:431`). Report per-task success for both arms and flag it as a gap
  if the effect is carried by few tasks.
- **MM-38 smoke:** 20 items is the spec minimum; propose 50 probe items (5 per task) for base and smoke cut, binary
  rule (degenerate only at mean 0 or 1).

## 6. What gatedtrain needs (proposal for the adapter)

1. **Training adapter `lerobot_smolvla`, `uses: sft.lerobot@0.1`.** Cuts: a cut builder in the area's launcher
   copies each completed `checkpoints/<step>/` (complete when `last` points at it, or after the process exits) into
   `cuts/checkpoint-<step>/` with `cut.json` written last, as `nemo_rl_ext` does with NeMo RL's `step_N`.
   `model_hash` components: `pretrained_model/model.safetensors`, `config.json`, `policy_preprocessor.json`,
   `policy_postprocessor.json`, the normalizer and unnormalizer `.safetensors`, and a deployment sampler file
   `sampler.json` (`n_action_steps`, `num_steps`, `control_mode`, rename map, observation size, `init_states`,
   `hard_reset`). `train_config.json` and `training_state/` are resume state (in `cut_hash`, not `model_hash`).
   `normalizer_sha256` (MM-19) = the hash of the two normalizer state files. Pins: lerobot, torch, transformers,
   hf-libero, robosuite, mujoco versions (`env-freeze.txt`), `smolvla_base` and `lerobot/libero` revisions, SmolVLM2
   revision. Consumption `row`, observed from `samples.jsonl` (§2), shard = data file; no automatic fallback.
2. **Evaluator `libero.lerobot_items@0.1`:** metric `success`, `binary: true`, T3, `normalizer: true`,
   `env_version: true` (`env_version.json`: hf-libero 0.1.4, robosuite 1.4.0, mujoco version, suite, control mode,
   control_freq 20, max steps 280, `num_steps_wait` 10, assets manifest hash, init files sha256), error class
   `simulator_error`, score outcomes `timeout`, `failed`, `nan_action`. Item payload `(task, init_state)` per MM-6;
   exact-overlap check is trivially 0 (payload text never appears in the parquet); the meaningful contamination check
   is the init-state comparison of §3.
3. **Launch facts:** runtime `venv_in_container` (MM-32); `--network none`; uid 1000; `gpus: true`; env allow-list
   adds `HF_HUB_OFFLINE`, `HF_HOME`, `HF_LEROBOT_HOME`, `HOME`, `LIBERO_CONFIG_PATH`, `MUJOCO_GL`,
   `PYOPENGL_PLATFORM`, `MUJOCO_EGL_DEVICE_ID` (robosuite fails on an empty `CUDA_VISIBLE_DEVICES`, §0.4),
   `NVIDIA_DRIVER_CAPABILITIES` (must include `graphics` for the NVIDIA EGL library to be injected; the CUDA base
   image sets `compute,utility`, the area image sets `compute,utility,graphics`, §0.5; injection **unverified** until
   P1), `PYTHONUNBUFFERED`, `TOKENIZERS_PARALLELISM`, `XDG_CACHE_HOME`. `HF_HOME` (or `HF_DATASETS_CACHE`) must be
   writable (datasets' Arrow cache, §0.4). Assets: the area code sets `libero.libero._assets_path_cache` (§0.3), or
   `HOME` holds `.cache/libero/assets` as a read-only mount. Shared memory: LeRobot uses DataLoader workers (4) and async env
   workers; the launcher's default `--shm-size=8g` should do (**estimate**).
4. **Two phases:** as area 2, the base run is launched but not collected; the continuation records the base cut
   (`run_args.json` `base_checkpoint`, `base_cut`) and `ext-collect` gates it against that cut (EXT-101 style).

## 7. Runtime build (done 2026-11-14, see §0; original proposal notes kept below)

- **Venv on the host:** built as proposed (§0.2), 7.5 GB, lock sha256 `034e5c3b…5788`; mounted read-only in runs
  (areas 2-5).
- **Image:** `gatedtrain-a8-runtime:1` = `gatedtrain-a5-runtime:1` (which has `libgl1 libglib2.0-0t64 libx11-6 libxcb1 libxext6
  libsm6 libxrender1`, `area5-sd35-image/Containerfile:12-13`) plus `libegl1 libopengl0 libgles2 libglvnd0
  libegl-mesa0 libosmesa6` from the same Ubuntu snapshot `20261029T000000Z`, Containerfile written (§0.5), not built. The Docker daemon used had only `host` and `none` networks (`docker
  network ls`); area 5 built with `--network host` (`area5-sd35-image/Containerfile:5`). Under a `--network none`
  build the `.deb` files would be fetched outside the build first (`apt-get download`) and installed with `dpkg -i` from
  the build context.
- **Offline data:** `HF_LEROBOT_HOME=data/` (dataset root `data/lerobot/libero`), local directories
  `models/smolvla_base` (`--policy.path`) and `models/SmolVLM2-500M-Video-Instruct` (`--policy.vlm_model_name`). The
  tokenizer step does **not** follow `vlm_model_name`: it keeps the repo id saved in `policy_preprocessor.json` and
  fails offline; it needs the `tokenizer_processor` override (§0.4 findings). LIBERO `config.yaml` and assets on
  read-only mounts; `HF_HOME` writable.

## 8. Probe and run plan

| Step | What | Cost (estimate) |
|---|---|---|
| P0 CPU | **done 2026-11-14** (§0.4): venv build; imports; offline dataset load of episodes 807-1260 with PyAV; SmolVLA forward/loss on CPU; LIBERO-Object init states, physics reset and 10 steps; rendering on the CPU through Mesa software EGL (no OSMesa on the development machine) | 0 GPU-h |
| P1 GPU render | in the new image: EGL reset + 280 zero-action steps, 2 cameras 360 px; seconds per step; one `lerobot-eval` of `smolvla_base` would fail on features, so use a scripted zero policy | 0.05 |
| P2 GPU train | 200 steps batch 32, expert only, then batch 64 and full fine-tune; peak memory, s/step (pick first config with peak ≤ 22.5 GB) | 0.15 |
| P3 base ladder | base phase to about 3,000 steps (cuts every 250-500), each cut on 50 probe items | train 0.4-0.8 + eval 0.2 |
| Adapter | written against CPU fixtures | 0 |
| Smoke | `ext-launch --profile smoke` continuation 200-300 steps, MM-38 record (50 probe items, base and smoke cut) | 0.2 |
| Full | continuation 5,000-10,000 steps at about 0.5-1.0 s/step | 0.7-2.8 |
| Arms | 400 items × K 2 × 2 arms = 1,600 episodes; about 5-10 s per episode per worker with GPU EGL (280 steps at 10-25 ms plus 6 policy calls of about 0.1 s plus a hard reset of 1-3 s), 8-10 parallel workers | 0.4-1.0 |
| Total | | about 2.5-5 GPU-h (spec row 8: 2.5) |

Development machine: 32 cores, 31 GiB RAM (`free -g`); LIBERO workers about 0.5-1 GB each (**estimate**), training DataLoader 4
workers. Disk (data disk 348 GB free, `df -h`): dataset 1.94 GB, `smolvla_base` 0.91 GB, SmolVLM2 2.04 GB, assets
0.42 GB, venv 7.5 GB (measured), image +0.1 GB over `gatedtrain-a5-runtime:1` (7.64 GB), cuts about 1 GB weights + 0.4-3.6 GB
optimizer each × 10-20 cuts (kept: `cut_hash` covers `training_state/`), episode videos optional. Total under
60 GB.

## 9. Risks

1. **Rendering in the container.** EGL needs `graphics` driver capability and libglvnd EGL in the image; fallback
   OSMesa is CPU-bound (about 3-10× slower, **estimate**).
2. **Base ladder may jump.** Success can go from 0 to >0.8 within a few hundred steps on a single suite (unknown);
   mitigation: dense cuts (every 250 steps) and lower learning rate in the base phase.
3. **Clustered outcomes** (10 tasks) and binary scores make the per-item test optimistic; per-task report.
4. **Demonstration vs init-state overlap** unverified (§3).
5. **Build:** resolved (§0.2): the sdists built with PyPI cmake; robosuite 1.4.0 with mujoco 3.8.1 and numpy 2.2.6
   reset, stepped and rendered a LIBERO-Object task in P0. Open: `cv2` and `egl_probe` are each written by two
   distributions (§0.2).
6. **Offline loading:** dataset (with `revision`), policy and processor loaded offline in P0; the tokenizer needs a
   local override in `lerobot-train` (§0.4 findings, **unverified** with `lerobot-train`).
7. **Determinism:** MuJoCo is deterministic on one machine for a fixed init state and action sequence; GPU EGL
   rendering differences across driver versions could change images (only matters across reruns).
8. **Licences:** model, SmolVLM2, `lerobot/libero` Apache-2.0; hf-libero MIT; `lerobot/libero-assets` has no licence
   on its card (scanned objects, HOPE, TurboSquid meshes); disclosed, prototype only.

## 10. Open decisions (at the time of writing)

Open for the maintainers: evaluator path (E2 with E1 cross-check), expert-only vs full fine-tune
(memory probe), `n_action_steps` (checkpoint default 50 unless the probe shows a large loss vs 10), learning rate
and lengths of the two phases, planning effect.

## 11. Scripts (2026-11-14; CPU tests only, no GPU job had run)

| File | What |
|---|---|
| `a8_common.py` | Shared constants (paths, dataset revision, episodes 807-1260, rename map, LIBERO-Object episode settings), hashing and JSON helpers, the offline environment fix and socket guard of P0, MANIFEST bindings, cut contents (`model_file_names`, `normalizer_record`) and the **deployment sampler** `deploy_sampler(config.json)`, the single definition used by the cut writer and the evaluator. Imports no LeRobot, torch or LIBERO module. |
| `train_lerobot_logged.py` | Launcher: builds the `lerobot-train` argv, sets `sys.argv`, calls `lerobot.scripts.lerobot_train.main()` in process after wrapping five module globals (no LeRobot edit; file:line in the docstring): `make_pre_post_processors` (local SmolVLM2 tokenizer override), `make_dataloaders` (sampler length, for the epoch column), `_preprocess_dataset_batch` (consumption hook), `update_policy` (memory log), `update_last_checkpoint` (cut writer). |
| `eval_libero.py` | Evaluator `libero.lerobot_items@0.1` (path E2): `--write-items`, and per-item evaluation of a cut with LeRobot's `LiberoEnv`, processors and `select_action`; spawn worker processes, each with its policy on the GPU and its own EGL context(s); a per-item wall-clock timeout (`--item-timeout-s`, default 300) kills and replaces hung workers. |
| `probe_render.py` | Probe P1: LeRobot's `LiberoEnv` (the evaluator's settings), hard resets and 280 no-op steps, per-step timing, both cameras at 360 × 360, mapped GL/EGL libraries and GL/EGL vendor strings, PNG frames. |
| `tests/` | Host tests (CPU, Mesa EGL): `test_train_args.py`, `test_hooks.py`, `test_eval_logic.py`, and `test_e2e_cpu.py` (slow, `GATEDTRAIN_A8_SLOW=1`). Run each as `env -u HF_TOKEN .venv/bin/python -I -B tests/<file>.py` from this directory (`GATEDTRAIN_A8_TEST_TMP=<dir>` for temporary files). |

### 11.1 Design (choices made while writing)

- **Training run layout** (`--out`, the run's only read-write mount): `env-freeze.txt` (copy of this directory's
  freeze; `env-runtime.json` records whether it equals the running distributions), `env-runtime.json`,
  `run_args.json` (settings, the exact `lerobot_argv`, sampler and `sampler_sha256`, bindings of smolvla_base,
  SmolVLM2 and `lerobot/libero` from `MANIFEST.json`, `base_checkpoint` / `base_cut` for stage 2, script and
  common-module SHA-256s), `samples.jsonl`, `memory.jsonl`, `cuts.jsonl`, `attempts.jsonl` (start and finish, the
  finish line carries peak device memory, samples consumed, epochs, `blocked_connections`), `lerobot/` (LeRobot's
  `output_dir`; it must not exist before the run, `configs/train.py:334-338`) and `cuts/checkpoint-<step>/`. A
  non-empty `--out` moves earlier contents to `preempted-attempt-<k>/` (areas 3, 5).
- **Profiles:** `probe` (P2: 200 steps, `save_checkpoint=false`), `base` (stage 1 from `models/smolvla_base`: 3,000
  steps, a cut every 250, the last step always), `smoke` (stage 2 from a base cut: 250 steps, one cut), `full`
  (stage 2: `--steps` required, a cut every 1,000). Flags override: `--steps --save-freq --batch-size --lr
  --warmup-steps --decay-steps --full-ft|--train-expert-only --n-action-steps --num-workers --seed --device`.
  Defaults: batch 32, expert only (`train_expert_only`, `freeze_vision_encoder` true), `n_action_steps` 50, seed
  1000, 8 DataLoader workers, the SmolVLA optimizer and scheduler presets (AdamW lr 1e-4; cosine with warmup 1,000 /
  decay 30,000, which LeRobot scales down to the run length when `steps` < 30,000: base ladder warmup 100,
  `optim/schedulers.py:146-154`), `dataset.use_imagenet_stats=false` (the `lerobot/smolvla_libero` recipe; images are
  IDENTITY-normalized for SmolVLA anyway), `accelerator.mixed_precision` at its default `no` (not passed: draccus
  reads the bare value `no` as `False`). Stage 2 refuses a `--policy-path` without `cut.json`, verifies the base cut's
  model files and its smolvla_base binding, and records them (`run_args.json` `base_cut`, every cut's
  `init_checkpoint`).
- **lerobot-train argv** (parsed and validated by LeRobot's own parser in `test_train_args.py`):
  `--policy.path=<smolvla_base or cut>/pretrained_model`, `--policy.vlm_model_name=<models/SmolVLM2-...>`,
  `--policy.load_vlm_weights=false`, `--policy.input_features=null --policy.output_features=null`,
  `--policy.push_to_hub=false`, `--policy.n_action_steps`, `--policy.train_expert_only`,
  `--policy.freeze_vision_encoder`, `--dataset.repo_id=lerobot/libero --dataset.root=data/lerobot/libero
  --dataset.revision=a1aaacb7… --dataset.episodes=[807,…,1260] --dataset.video_backend=pyav`, `--rename_map` image →
  camera1, image2 → camera2, `--output_dir=/out/lerobot`, `--env_eval_freq=0 --eval_steps=0 --trackio.enable=false`.
- **Tokenizer:** the wrapper adds `tokenizer_processor.tokenizer_name=<local SmolVLM2 dir>` to the preprocessor
  overrides. LeRobot then saves the tokenizer as a checkpoint artifact (`pretrained_model/tokenizer/`,
  `policy_preprocessor.json` names `tokenizer`, `processor/tokenizer_processor.py:298-305`,
  `processor/pipeline.py:572-586`), so a cut is self-contained for the tokenizer; its `config.json` names the local
  SmolVLM2 directory as `vlm_model_name` (config and processor files are read from it at load), so cuts are usable
  only with this directory mounted at the same path.
- **Known limitation: evaluating a published cut from another location.** The published cuts' `config.json`
  (`pretrained_model/config.json`, `vlm_model_name`) holds the absolute path of the SmolVLM2 directory of the
  runtime that trained them. `eval_libero.py` loads the policy with that config as saved
  (`PreTrainedConfig.from_pretrained`, then `SmolVLAPolicy.from_pretrained`) and does not override the field, and
  LeRobot 0.7.0 reads the VLM's config and processor from it (`policies/smolvla/modeling_smolvla.py:550`,
  `policies/smolvla/smolvlm_with_expert.py:99,101`). From a runtime at any other path the load fails. Two ways round
  it until the scorer overrides the field (planned with the next registration, since `eval_libero.py` is the
  registered, hash-pinned scorer):
  1. Keep the cut unchanged and make the runtime's `models/SmolVLM2-500M-Video-Instruct` readable at the path the
     cut's `config.json` names (read it with `grep vlm_model_name <cut>/pretrained_model/config.json`), e.g. as an
     extra read-only mount at that path in the evaluation container. The cut's hashes, and so a re-judgement of the
     published gate, are unaffected.
  2. Evaluate a copy of the cut whose `pretrained_model/config.json` names this runtime's
     `models/SmolVLM2-500M-Video-Instruct` in `vlm_model_name` (only that value changed). `config.json` is one of the
     cut's model-hash components, so the copy has another `model_hash` than the published cut: fine for looking at a
     cut's behaviour, not for reproducing the registered comparison.
- **Consumption log** (`samples.jsonl`, MM-26/MM-27): one line per step, written and fsynced before the step's
  update: `step` (1-based), `epoch` (0-based, from the samples consumed and the sampler length; the epoch's last
  batch is short, `drop_last=False`), `n`, and per sample the anchor frame's `index`, `episode_index`,
  `frame_index`. Each sample also loads the next 49 actions of its episode (disclosure, §2). `cut.json` holds
  `last_step_included` and `samples_consumed` for the MM-27 count check.
- **Cuts** (`cuts/checkpoint-<step>/`, schema `ptc_mm.lerobot_cut`): hard links to LeRobot's
  `checkpoints/<step>/pretrained_model/` and `training_state/` (copies when linking fails), `sampler.json`,
  `base_ref.json` (smolvla_base repo, revision, every file's SHA-256; SmolVLM2 and dataset bindings; stage 2's
  `init_checkpoint`), then `cut.json` last (`files` with every SHA-256, `model_files` = `pretrained_model/*` except
  `train_config.json`, plus `sampler.json`; `resume_files` = `train_config.json`, `training_state/*`,
  `model_files_sha256`, `normalizer_sha256`, `sampler_sha256`, step, samples, seed, profile, stage, code hashes),
  built in `.checkpoint-<step>.partial/` and renamed; written right after LeRobot points `last` at the step.
  **`normalizer_sha256`** = SHA-256 of the canonical JSON `{normalizer: sha256, unnormalizer: sha256}` of the two
  processor state files named in `policy_preprocessor.json` / `policy_postprocessor.json` (the adapter must compute
  the same; README §6.1).
- **Deployment sampler** (`sampler.json`, `a8_common.deploy_sampler`): policy type, `n_action_steps`,
  `chunk_size`, flow-matching `num_steps`, `max_action_dim`, resize 512, tokenizer length, rename map, observation
  (pixels_agent_pos, 360 × 360, both cameras, camera mapping, canonicalization), control mode `relative`,
  `init_states`, `hard_reset`, the noise rule, the action postprocessing. Episode settings (280 steps, 10 settle
  steps, 20 Hz) are in `env_version.json`. **Observation size 360 × 360** is LeRobot's LIBERO default
  (`envs/configs.py:479-480`, the `lerobot-eval` cross-check E1 uses it); the training videos are 256 × 256 and the
  policy pads both to 512.
- **Evaluator:** items `libero_object/t<task>/s<init>` in numeric (task, init state) order; payload suite, task id
  and name, language, init file name and SHA-256, `row_index`. Per (item, eval seed): torch reseeded with
  `derive_seed('libero-torch', seed, item)`, flow-matching noise from a CPU generator seeded with
  `derive_seed('libero-noise', seed, item)` (one draw per step passed as `select_action(noise=...)`), env seed
  `derive_seed('libero-env', seed, item) mod 2^31`: common random numbers per item across arms, independent of the
  worker. The init state is set through `LiberoEnv.init_state_id` (`libero.py:211`, `:381`); a worker keeps at most
  `--envs-per-worker` (default 1) `LiberoEnv` instances open, least recently used closed first (each holds a MuJoCo
  render context with its own EGL context; every reset is a hard reset that rebuilds model, sim and renderer,
  robosuite `environments/base.py:247-252`, with the job's seed and init state, so closing an env between episodes
  does not change an episode). Outcomes: `success` (1), `timeout`, `failed` (done without success), `nan_action` (0,
  the action never reaches the simulator); an exception from the episode is `simulator_error` (the worker drops that
  env and rebuilds it); after a worker death the jobs that were in flight rerun alone in new pools, and one that
  kills its worker while alone is `simulator_error` (`worker_died`). **Item timeout** (`--item-timeout-s`, default
  300 s, counted from the moment a worker takes the job; 0 disables; none with `--workers 0`): a job without a result
  by then gets its pool killed (SIGKILL to every worker, since a `ProcessPoolExecutor` cannot replace one worker) and
  a new pool started; the job is retried once alone, the other jobs that were in flight rerun from the start with the
  same seeds (no result is lost or counted twice), and a job that times out again alone is `simulator_error` with the
  message `item_timeout: ...` (score null, MM-4; the episode's 280-step `timeout` outcome is a different thing, a
  score of 0). So the run always finishes and writes the arm. A pool whose workers cannot start (broken, or no answer
  to a no-op job within 900 s) stops the run. `summary.json` adds `item_timeout_s`, `envs_per_worker`,
  `error_reasons` {item_timeout, worker_died}, `device_memory_max` and `device_free_gb_min`; each episode log adds
  `device_memory` (torch allocated/reserved and their peaks, device free/total, after the job). Arm `arm-success/`:
  `items.jsonl` (MM-1: item, repeat, seed, score, error, `sub_metrics` {steps_to_success, n_policy_calls, steps,
  wall_s, timeout, nan_action}, `output_ref` = the episode log, `item_meta` {task_id, init_state, outcome}),
  `eval_meta.json` (exactly the MM-2 keys: `checkpoint.model_hash` and `item_set_hash` are null unless passed with
  `--model-hash` / `--item-set-hash`, to be filled by `gatedtrain ext-arm-fill`; `normalizer_sha256` from the cut;
  `env_version_sha256` = SHA-256 of the compact sorted JSON of `env_version.json`; `sampler_hash` in the same
  encoding, which ext-arm-fill converts), `env_version.json`, `episodes/<item>/r<k>.json` (seeds, outcome, success
  step, timings, action summary and every action). Run directory: `env_version.json`, `env-runtime.json`,
  `progress.jsonl` (one line per finished episode), `summary.json` (success, per-task success, outcomes, MM-38
  binary check, timings, the cut's `model_files_sha256`). `--max-steps` / `--settle-steps` exist for host tests; a
  non-default value is recorded in `env_version.json` (`nondefault`) and changes `env_version_sha256`.
- **Workers:** default 6 spawn processes on one GPU; 2 torch threads each. The pool keeps at most `--workers` jobs
  in flight. Device memory per worker (review after `base-1-eval-500`): weights 1.20 GB (the cut loads
  as 301.2 M bf16 + 148.8 M fp32 parameters, the dtypes LeRobot builds from the config in training too, checked on
  the CPU with `checkpoint-500`; a cast to bf16 would change the numerics, so it is not done), the CUDA context, and
  one EGL context per open `LiberoEnv` (P1 x8: about 0.67 GB per render process with one env and no torch). Before
  the fix a worker kept one env per task it had seen (up to 10), and the per-worker usage in `base-1-eval-500` grew
  to 2.1-4.9 GB (8 workers 24.0 GB, the whole card). Fixes: at most `--envs-per-worker` (1) open envs, the whole
  policy step (processors included) under `torch.inference_mode` (it was the policy call only; the processors hold
  no trainable parameters, the values are unchanged), `torch.cuda.empty_cache()` after each job. Expected about
  2-2.5 GB per worker (**estimate**, unverified on the GPU).

### 11.2 Host steps before the GPU work (CPU, no docker)

```
cd <runtime>
env -u HF_TOKEN .venv/bin/python -I -B eval_libero.py --write-items      # items-gate/ (400), items-probe/ (100)
```

Done 2026-11-14 (deterministic; a rerun refuses to change an existing file): `items-gate/items.jsonl` sha256
`13c8a438…d98e`, `items-probe/items.jsonl` sha256 `824c2f9c…8a9f`; `ITEMS-MANIFEST.json` lists every file's SHA-256
and the ten init files' SHA-256s. The gatedtrain `item_set_hash` is the manifest hash of the item directory, filled by
`ext-arm-fill`.

### 11.3 Docker commands (each is a GPU job)

Common flags (the runs never write into the read-only area mount; `HF_HOME`, `HOME` and caches go to the tmpfs):

```
A8=<runtime>                       # this runtime directory (absolute path)
OUTROOT=<output-root>/lerobot-smolvla-libero
PYDIR=<python-dir>                 # the directory holding the interpreter tree of .venv/pyvenv.cfg
IMG=sha256:136745e51640ee7bf92760d3dc3f8764187f3c409e4f2c928ce4c39707cd5242     # gatedtrain-a8-runtime:1
a8run() {   # a8run <host out dir> [extra docker args ...] -- <script> <args ...>
  local out=$1; shift; local extra=(); while [ "$1" != "--" ]; do extra+=("$1"); shift; done; shift
  docker run --rm --network none --user 1000:1000 --cap-drop ALL \
    --security-opt no-new-privileges --shm-size=8g --tmpfs /tmp:rw,exec,nosuid,nodev,size=8g --gpus all \
    -v $PYDIR:$PYDIR:ro \
    -v $A8:$A8:ro -v "$out":/out:rw "${extra[@]}" \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl \
    -e MUJOCO_EGL_DEVICE_ID=0 -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e HF_HOME=/tmp/hf-home -e HF_LEROBOT_HOME=$A8/data -e LIBERO_CONFIG_PATH=$A8/libero-config \
    -e HOME=/tmp/home -e XDG_CACHE_HOME=/tmp/cache -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
    -e TOKENIZERS_PARALLELISM=false -e WANDB_MODE=disabled -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $IMG $A8/.venv/bin/python -I "$@"
}
```

Each `<host out dir>` must be new, empty and owned by uid 1000 (`mkdir -p` as that user). Check the image id with
`docker image inspect --format '{{.Id}}' gatedtrain-a8-runtime:1` (the commands pass the id).

```
# P1 render probe: one process, then 8 in parallel (sizes the evaluator's --workers)
mkdir -p $OUTROOT/p1-render $OUTROOT/p1-render-x8
a8run $OUTROOT/p1-render -- $A8/probe_render.py --out /out --steps 280 --resets 2
a8run $OUTROOT/p1-render-x8 -- $A8/probe_render.py --out /out --steps 280 --resets 2 --procs 8

# P2 memory probe: 200 steps each, no checkpoint
mkdir -p $OUTROOT/p2-expert-b32 $OUTROOT/p2-expert-b64 $OUTROOT/p2-fullft-b32
a8run $OUTROOT/p2-expert-b32 -- $A8/train_lerobot_logged.py --profile probe --out /out --batch-size 32
a8run $OUTROOT/p2-expert-b64 -- $A8/train_lerobot_logged.py --profile probe --out /out --batch-size 64
a8run $OUTROOT/p2-fullft-b32 -- $A8/train_lerobot_logged.py --profile probe --out /out --batch-size 32 --full-ft

# P3 base ladder (stage 1): cuts every 250 steps to 3,000; batch and expert/full from P2
mkdir -p $OUTROOT/base-1
a8run $OUTROOT/base-1 -- $A8/train_lerobot_logged.py --profile base --out /out --batch-size 32
# each cut S on the probe items (100 items, seed 11; or --per-task 5 for 50), cut read-only at /inputs/cut
mkdir -p $OUTROOT/base-1-eval-S
a8run $OUTROOT/base-1-eval-S -v $OUTROOT/base-1/cuts/checkpoint-S:/inputs/cut:ro -- \
  $A8/eval_libero.py --checkpoint /inputs/cut --items probe --eval-seeds 11 --workers 6 --out /out

# Smoke (stage 2): 250 steps from the chosen base cut B, then the MM-38 arms (50 probe items) of B and of the smoke cut
mkdir -p $OUTROOT/smoke-1
a8run $OUTROOT/smoke-1 -v $OUTROOT/base-1/cuts/checkpoint-B:/inputs/base-cut:ro -- \
  $A8/train_lerobot_logged.py --profile smoke --policy-path /inputs/base-cut --out /out --batch-size 32
mkdir -p $OUTROOT/smoke-1-eval
a8run $OUTROOT/smoke-1-eval -v $OUTROOT/smoke-1/cuts/checkpoint-250:/inputs/cut:ro -- \
  $A8/eval_libero.py --checkpoint /inputs/cut --items probe --per-task 5 --eval-seeds 11 --workers 6 --out /out

# Full (stage 2): length N from the smoke and the ladder
mkdir -p $OUTROOT/full-1
a8run $OUTROOT/full-1 -v $OUTROOT/base-1/cuts/checkpoint-B:/inputs/base-cut:ro -- \
  $A8/train_lerobot_logged.py --profile full --steps N --policy-path /inputs/base-cut --out /out --batch-size 32

# Gate arms: base cut B and the full run's last cut, 400 gate items, K 2
mkdir -p $OUTROOT/gate-base
a8run $OUTROOT/gate-base -v $OUTROOT/base-1/cuts/checkpoint-B:/inputs/cut:ro -- \
  $A8/eval_libero.py --checkpoint /inputs/cut --items gate --eval-seeds 11 12 --workers 6 --out /out
# then: gatedtrain ext-arm-fill --adapter lerobot_smolvla --checkpoint <cut> --items $A8/items-gate <out>/arm-success
```

Read from P1 `probe_render.json`: `nvidia_egl_loaded_all` true (libEGL_nvidia mapped, `gl_after_first_reset`
GL_VENDOR NVIDIA), per-step and reset seconds, non-blank `frames/*.png`. From P2 `memory.jsonl`: `step_s`,
`update_s`, `peak_allocated_gb`, `peak_reserved_gb`, `rss_now_mb`, `loss`, and `attempts.jsonl`
`peak_device_memory`; pick the first configuration with `peak_reserved_gb` ≤ 22.5; `step_s` well above `update_s`
means data loading bounds the step (raise `--num-workers`). The gate's base is the first ladder cut whose probe
success lies in about 0.2-0.6 (the rule fixed before any candidate result).

### 11.4 Verified on the CPU / unverified until the GPU probes

Verified (2026-11-14, CPU, see the test files): the argv parses and validates with LeRobot's parser; hooks,
cut writer and the evaluator's cut check on a fake checkpoint; item sets; episode scoring with a fake env and policy;
the LIBERO-Object task list and init files; `probe_render.py` with Mesa software EGL (20 steps: 0.06 s per step at
360 × 360, two cameras; first hard reset 9.8 s, second 1.3 s; GL_VENDOR Mesa / llvmpipe); and the end-to-end CPU
smoke (`test_e2e_cpu.py`): see §11.5.

Unverified until P1/P2: NVIDIA EGL inside `gatedtrain-a8-runtime:1` (vendor file present, `libEGL_nvidia.so.0` injected by
the toolkit only with `graphics`); render and reset time on the GPU; several EGL contexts per GPU across worker
processes; device memory and step time of every training configuration; DataLoader throughput with 8 spawn workers
(PyAV AV1 decode); bf16/fp32 numerics of SmolVLA on CUDA; the evaluator's per-worker device memory after the
2026-11-15 fix (one open env per worker, `inference_mode` over the whole step, `empty_cache`; read
`summary.json` `device_memory_max` / `device_free_gb_min` and the per-process usage) and the cause of the
`libero_object/t2/s46` hang in `base-1-eval-500` (memory pressure on the full card or an EGL/MuJoCo hang; not
reproduced on the CPU); the item timeout against a real hung EGL/MuJoCo call on the GPU (SIGKILL of a process
blocked in the driver; a process in uninterruptible sleep would not die, and the interpreter could then wait for it
at exit after the arm is written); whether
`torch.backends.cudnn.benchmark` (set by LeRobot in training only) matters for the evaluation's determinism.

### 11.5 CPU test results (2026-11-14, CPU, Mesa EGL; rerun 2026-11-15 after the item timeout)

2026-11-15 (item timeout, env bound, memory): `test_eval_logic.py` 16 OK (new: a job that hangs on every attempt
with 2 workers and a 2.5 s limit is retried once alone in a new worker, then recorded as `simulator_error`
`item_timeout`, both hung workers are dead, the 5 other jobs score exactly once; a job that hangs only on the first
attempt scores on its retry; a worker that hangs while starting stops the run after the startup limit;
`--item-timeout-s` / `--envs-per-worker` parsing; `run_job` keeps at most `envs_per_worker` envs open, LRU, and
closes the others), `test_hooks.py` 7 OK, `test_train_args.py` 9 OK, `test_e2e_cpu.py` 1 OK with `GATEDTRAIN_A8_SLOW=1`
(146 s; the real policy step on the CPU under the widened `inference_mode`). The pool tests also caught an
isolation defect that the timeout poll would have exposed: a suspect must stay alone until it returns (fixed).

2026-11-14:

`test_train_args.py` 9 tests OK, `test_hooks.py` 7 OK, `test_eval_logic.py` 11 OK (including a spawn pool whose
worker dies on one job: that job is `simulator_error` `worker_died`, the others score; a pool that cannot start stops
the run), `test_e2e_cpu.py` 1 OK with `GATEDTRAIN_A8_SLOW=1` (389 s): the real `lerobot-train` through the launcher, 2 steps
at batch 2 from smolvla_base on the CPU (33 s per step, loss 1.96 then 1.35, peak RSS 3.2 GB, 66,984 samples per
epoch, `blocked_connections` []), so the CLI `null` features, the offline tokenizer override, the consumption hook,
the memory hook and the cut writer work with `lerobot-train` itself (README §0.4 "unverified" items); the cut's
`config.json` holds camera1/camera2/state 8/action 7 and the local `vlm_model_name`, its `policy_preprocessor.json`
names the saved `tokenizer/` artifact; then a stage-2 continuation of 1 step from that cut (base cut verified and
recorded, same `normalizer_sha256`); then `eval_libero.py` on 1 probe item with 3 steps on the CPU (reset 36 s with
Mesa, first policy call 24 s): MM-1/MM-2 arm written, outcome `timeout`, score 0. `env-runtime.json` compares the
freeze with the running distributions by PEP 440 version equality.

## Result (2026-11-16)

- **full-1** (stage 2, 2,500 steps from base-1 checkpoint-500, seed 2000, cuts every 500; 0.40 h on one GPU).
- **Gate PASS** (`gate/registration.yaml`, 400 gate items x seeds 11, 12): success 0.2175 -> 0.445 (delta +0.2275,
  95% cluster-bootstrap CI [0.186, 0.270], p <= 9.6e-05); all ten tasks improve. Run
  `run-20261116T024020Z-805b7cec`; consumption precision row (79,976 samples); impact maps
  cuts 500..2500.
- `gate/spec.yaml`'s base pin was corrected from the cut's model hash to its whole-directory manifest hash (what
  ext-collect checks); the cut itself is unchanged (maintainers' decision "Area 8 result", not published).

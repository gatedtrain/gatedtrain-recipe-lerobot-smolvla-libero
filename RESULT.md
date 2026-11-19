Prototype result of a single registered run; not a benchmark claim; not affiliated with or endorsed by the owners of the models, datasets, or frameworks used.

# Result (2026-11-16): gate PASS; carried forward to ptc-hub 0.2.0 and gatedtrain 0.3.0 as post-hoc amendments

ptc-hub, ptc-skel and ptc are earlier names of gatedtrain; see "Rename to gatedtrain 0.3.0".

LeRobot 0.7.0 SmolVLA expert-only SFT, stage 2: run full-1, 2,500 steps at batch 32 from the stage-1 cut base-1
checkpoint-500 (seed 2000, cuts every 500 steps, 0.40 GPU-h). The stage-1 ladder (3,000 steps from `smolvla_base`, cuts
every 250, each evaluated on the 100 probe items) chose checkpoint-500 as the gate's base by the rule fixed before any
stage-2 result: the first cut whose probe success lies in [0.2, 0.6] (0.26).

| | |
|---|---|
| Deciding metric | `success` (binary, LIBERO's BDDL goal check; T3 with the construct-alignment and shared-annotation disclosures) |
| Baseline (stage-1 checkpoint-500) | 0.2175 |
| Candidate (stage-2 checkpoint-2500) | 0.445 |
| Delta, 95% cluster-bootstrap interval | +0.2275 [0.186, 0.270], p <= 9.6e-05 |
| Items, samples per item | 400 (LIBERO-Object tasks x initial states 0-39), 2 (seeds 11, 12) |
| Achieved power | 0.9999 |
| Exact overlap with training | 0 (item ids; demonstration-vs-benchmark initial-state overlap unverified, disclosed) |
| Consumption precision | `row` (79,976 samples of steps 1-2500, observed) |

Every task improves (t3 least, 0.113 to 0.150); the paired analysis does not cluster by task (reported). The impact
query maps the cuts to steps 500 to 2500, none orphaned or superseded.

Re-collected on 2026-11-16 with this repository's adapter (installed as a plugin of ptc) on the original run's
outputs: every field of the gate report equals the re-collect with the adapter still inside ptc, except the
run-id-derived `lockfile_hash`; the ledger (79,976 rows) and the lineage are equal.

CPU probe P0 (`results/2026-11-16/verify/`, before any GPU job): the dataset loads offline (454 episodes, 66,984
frames), a SmolVLA forward and backward pass on the CPU gives a finite loss (1.396), LIBERO-Object builds, resets and
steps (10 tasks; 50 initial states for task 0), and offscreen rendering works with Mesa's software EGL (OSMesa was not
installed on that machine: `libero-render-osmesa.log`).

## Redacted copies

The files under results/2026-11-16/ are redacted copies (host paths replaced by `<data-root>/...`, and references to
the maintainers' internal decision log reworded): `gate/registration.yaml`, `gate/spec.yaml` and, under `verify/`,
`config.json`, `env.json`, `libero-render-mesa-egl.log`, `libero-render-osmesa.log`, `libero-render.json`,
`libero.json`, `libero.log` and `policy.log`. The gate-time originals are not published.

`spec.yaml` still pins the original `registration.yaml` hash (the gate ran with it), so `ptc-hub validate` reports
REG-HASH on the published copy; `spec-ptc-hub.yaml` pins `registration-ptc-hub.yaml`.

The other files under results/2026-11-16/ are unchanged. At this publication cleanup the runtime scripts whose hashes the
adapter pins (`eval_libero.py`, `a8_common.py`, `train_lerobot_logged.py`) and `runtime/uv.lock` were byte-identical to
the ones that produced the result; the rename below re-pinned the three scripts (`runtime/uv.lock` is unchanged). For publication, only documentation and comments, host paths in `runtime/env-freeze.txt` (its header comment) and
`runtime/ITEMS-MANIFEST.json` (the `init_file` paths, now `<runtime>/...`), and host paths in the test fixtures (then
`/srv/ptc/...`, `/srv/gatedtrain/...` since the rename to gatedtrain 0.3.0) were changed; the result was re-collected unchanged before this cleanup.

## Rename to ptc-hub 0.2.0

ptc-hub 0.2.0 renamed the persisted identifiers (schema ids, ledger writer id, analysis code version). The files under
results/2026-11-16/ and the byte-copied fixtures under tests/fixtures/lerobot/ (`eval2-500/`, `base-1/`,
`checkpoint-500/`) keep the names ptc-skel 0.1.0 wrote; they are read by ptc `78eb913` (this and the other framework
commits cited here, `e20736e` among them, are unpublished pre-release commits of the framework, earlier named
ptc-skel and ptc-hub; the published framework is `gatedtrain` `v0.3.0`). The gate definition is carried
forward by results/2026-11-16/gate/registration-ptc-hub.yaml (amends
sha256:2fe788f1a7197554623073129b40bd6060f8f9164d0311ed115d3632f47e71f6, post_hoc: true) and spec-ptc-hub.yaml.
Re-pinned for the rename (the gate-time versions of `eval_libero.py` and `train_lerobot_logged.py` are not published;
see the table): runtime/eval_libero.py -> 20f44f7ed81956bfbbeeeae180430110d44830411abf1a73962fbe8b8901f50b, runtime/a8_common.py
d01995ca744109caeee66fc4e40688d28270ddf8f9e2770ab20cf8d28d2ea23d -> f49150f8d826a091ff3f22f086a5097dafc4c86f1a2e8e81b7dd97651d2f221b,
runtime/train_lerobot_logged.py -> 93bf86f66a71dc943879c22564655eeda3b95ba843c17aabda7292820fa30894, the scorer composite (`rows.EVAL_LIBERO_CODE_SHA256`)
bafc1ce89922abcd1c1db057efba77cfa79ccfe7fbec5d2610edbb095e76b6c7 -> d2ab09c6b97c57eb8074f18bc63ef5b2c7fee18e0439900ce69aa3dbdcdab2ec,
the registered item set items-gate/ (manifest) ff603f4232be6c1d53cdfec494e3f73391ce4abdba1cdb7148fdcd7948587e05 ->
06003603876057c1bcfdb51dbabf147e76c7b6f1d1377bf53d4c31d5d7b35c8d (only the new pin is accepted).

The gate-time item-set hash `ff603f42...` and scorer composite `bafc1ce8...` cover gate-time files that held host
paths. Those files are not published, and the published copies have the paths replaced, so neither hash can be
recomputed from this repository. What can still be checked: the 400 item rows (`runtime/items-gate/items.jsonl`,
`13c8a438...`, unchanged since the gate) and the published per-item records of the base cut's probe evaluation with
their summary statistics (`tests/fixtures/lerobot/eval2-500/`). The gate's own per-item records are not part of this
repository.

| File | Reason | SHA-256 before | SHA-256 now |
|---|---|---|---|
| `runtime/eval_libero.py` (scorer, member of the composite `code_sha256`) | schema id `ptc_hub.item_eval`; internal decision-document name removed from the docstring and from the item-set `rule` text it writes | not published | `20f44f7ed81956bfbbeeeae180430110d44830411abf1a73962fbe8b8901f50b` |
| `runtime/a8_common.py` (member of the composite; its hash is recorded as `common_sha256` by every run) | `ptc-hub` in a docstring | `d01995ca744109caeee66fc4e40688d28270ddf8f9e2770ab20cf8d28d2ea23d` | `f49150f8d826a091ff3f22f086a5097dafc4c86f1a2e8e81b7dd97651d2f221b` |
| `runtime/train_lerobot_logged.py` (recorded as `script_sha256` by every training run) | internal decision-document name removed from the docstring | not published | `93bf86f66a71dc943879c22564655eeda3b95ba843c17aabda7292820fa30894` |
| `runtime/items-gate/item_set_meta.json` (registered item set; `item_set_hash` covers it) | internal decision-document name removed from `rule` | not published | `d752dd2db7f2ca3d6353846bbd2e7b6a076cde725eb558edd3312ef79822886e` |
| `runtime/items-probe/item_set_meta.json` | as above | not published | `e04dbaf003820d8356ca6c46dd65ab6d9438c2585d78dd4986f026351acf258a` |
| `runtime/ITEMS-MANIFEST.json` | the two new `item_set_meta.json` hashes | not published | `268f5041c2ae07d1434636487111af0bddb6c238728100621658038b61ad5720` |
| `runtime/export_frames.py` | `ptc-hub` in the docstring | `2bfcb2c5df8b377f32be05ba62d8e11b2c48e3bc5d26b0ce746940e34c8a1d96` | `6a2ed2234eb660ff113f99be7889c19003b55cbb0426003573abce4b514a76cd` |
| `runtime/pyproject.toml` | internal decision-document name removed from the header comment | not published | `8d58b7c7b1cdf543c4bbf9b93248e723725e993d22d952c67417e516218c391b` |

The historical runs (stage 1, stage 2, every evaluation and the gate) used the old versions. Scoring, the sampler,
`env_version.json` (it does not depend on the scripts' bytes; `d0cedb3c...` unchanged) and the 400 gate items
(`items.jsonl` `13c8a438...` unchanged) are the same. Left unchanged on purpose: the `row_key` text "... (ptc-skel)"
that `export_frames.py` writes into `frames/frames_meta.json`; that file is part of the training-data snapshot
(`data_ref.snapshot`, the frame export's manifest hash, in every ledger row of the published run), and the export
refuses to change an existing export.

## The amendment

`registration-ptc-hub.yaml` (sha256 `6bce6de0f60cbb1628dd5dcd38a4f83e6e5de33d5bcbc98c7512e33498422b29`) is a copy of
`registration.yaml` (sha256 `2fe788f1...`, the published copy of the gate-time file; it differs from the gate-time
file only in redacted comments) that differs in
`analysis_code_version: ptc-hub 0.2.0`, `amends`, `created_at` (2026-11-17), `post_hoc: true` (added), a header
comment, `evaluators[0].manifest_hash` `sha256:a6b1265cafb0fbb184b6ecfd8956093eaeb27de4cb2c01944e17a567e58cf4b6` ->
`sha256:432748cbda980ca8e1e3fe28844482eab9bfc91ae21abdc2c20351c57dec3dbe` (the evaluator identity with the re-pinned
scorer composite) and `eval_distribution.item_set_hash` `ff603f42...` -> `06003603...` (the reworded
`item_set_meta.json`). These are evaluator and item-set changes, not only a rename, so the amendment is post hoc.
`eval_distribution.env_version` is unchanged. `spec-ptc-hub.yaml` (sha256
`31eb1fc77527d6f7ad6025f95a37665591d483cacd87267a072fe3beb05fda86`) differs from `spec.yaml` in the registration URI
and the items URI's hash. With `<data-root>` replaced (and the items URI pointing at an item directory holding the
new `item_set_meta.json`), `ptc-hub validate spec-ptc-hub.yaml` exits 0.

The gate's verdict, PASS, remains reproducible with ptc `78eb913` and the original `registration.yaml` and
`spec.yaml` (checked 2026-11-17: a re-collect with ptc `78eb913` and this recipe at `47fb1fe` (before the rename; a
pre-release commit of this repository, not in its published history) on the
original records, arms and gate files gives PASS with every report field equal to the published report except
`run_id`, `created_at` and `lockfile_hash`); under ptc-hub 0.2.0 the amendment reports it INCONCLUSIVE (post_hoc).

## Re-judgement on a scratch replay (2026-11-17, ptc `e20736e`)

The run's records carry the old ids, which ptc-hub 0.2.0 refuses, so the gate was re-collected from scratch copies
(nothing under the real run, record or output directories was written):

1. Copies of `train-full-1.launch.json`, `train-smoke-1.smoke.json`, the evaluation directories `gate-base-1/` and
   `gate-cand-1/`, and `runtime/items-gate/` (the new `item_set_meta.json`, the same `items.jsonl`). In the copies
   only these bytes changed: the schema ids (`ptc_hub.launch_record`, `ptc_hub.smoke_record`, `ptc_hub.item_eval`);
   in both arms' `eval_meta.json` `scorer.code_sha256` `bafc1ce8...` -> `d2ab09c6...` and `item_set_hash`
   `ff603f42...` -> `06003603...`; in both evaluations' `env-runtime.json` the same scorer pin (`code_sha256`,
   `script_sha256` from the gate-time pin to `20f44f7e...`, `common_sha256` `d01995ca...` -> `f49150f8...`); in the smoke
   record `item_set_hash` `ff603f42...` -> `06003603...`.
2. The launch record's `label` is a checksum of the whole record and is logged in the host's guard log, so the
   renamed copy was moved onto a scratch host-work directory: a copy of `gpu-step.sh` with only its `WORK=` line
   changed, the run's two guard-log lines copied under the new label, and `host_work`, `gpu_step_sha256`,
   `host_config_sha256` and `label` recomputed with ptc's own functions (`host_config_sha256`, `launch_label`;
   label `mm-a8-vla-train-full-5daf4da43995` -> `mm-a8-vla-train-full-86b8871e9ed2`). The real guard log was not
   written.
3. `ptc-hub ext-collect spec-ptc-hub.yaml --adapter lerobot_smolvla` with the original stage output and base cut, the
   scratch copies, the original input roots and the runtime override at the original runtime directory.

Compared with the published report of run `run-20261116T024020Z-805b7cec`: `delta_hat` 0.2275, interval
[0.18625, 0.27], `p_value` 9.64e-05, `achieved_power` 0.99988, `n_items` 400, `baseline_score` 0.2175,
`candidate_score` 0.445, the sampler (`dcc283f1...`), model and checkpoint hashes and 61 report fields in all are
equal. The differences are exactly the expected ones: `status` PASS -> INCONCLUSIVE with `status_reasons`
[post_hoc], `post_hoc` false -> true, `evaluator_hash` and `evaluator_hash_registered` `a6b1265c...` -> `432748cb...`,
`item_set_hash` and `eval_distribution.item_set_hash` `ff603f42...` -> `06003603...`, `registration_hash`,
`registration_amends`, `analysis_code_version`, the report `schema` id, `lockfile_hash`, `run_id` and `created_at`.
The ledger (79,976 rows) is equal except `writer_id` (`ptc-hub.ledger@0.2.0`) and the run-id-derived `run_id` and
`sample_id`; the lineage tables are equal except `run_id`, and the lineage edges except the item-set and registration
hashes and the schema id.

## Rename to gatedtrain 0.3.0 (2026-11-18)

The project was renamed from ptc / ptc-hub to gatedtrain before its first publication: distribution and import
`gatedtrain` 0.3.0, persisted identifiers `gatedtrain.<x>` (schema ids), `gatedtrain.ledger@0.3.0` (ledger writer id)
and `gatedtrain 0.3.0` (analysis code version), environment variables `GATEDTRAIN_*`. This recipe followed: distribution
`gatedtrain-recipe-lerobot-smolvla-libero`, package `gatedtrain_recipe_lerobot_smolvla_libero`, entry point group
`gatedtrain.adapters`, runtime override `GATEDTRAIN_RECIPE_LEROBOT_SMOLVLA_LIBERO_RUNTIME` (and
`GATEDTRAIN_RECIPE_LEROBOT_SMOLVLA_LIBERO_OUTPUTS` for the tests), runtime test switches `GATEDTRAIN_A8_*`, the runtime
project `gatedtrain-recipe-lerobot-smolvla-libero-runtime` (`runtime/pyproject.toml`, now also `license =
"Apache-2.0"`), and the documented image tags `gatedtrain-a5-runtime:1` and `gatedtrain-a8-runtime:1`.

Kept on purpose (historical evidence or identifiers the published run's outputs carry):

- everything under results/2026-11-16/ that the run or the earlier amendment wrote (`registration.yaml`, `spec.yaml`,
  `registration-ptc-hub.yaml`, `spec-ptc-hub.yaml`, `trainer.yaml`, `verify/`), and the byte-copied fixtures under
  tests/fixtures/lerobot/ (`eval2-500/arm-success/eval_meta.json` keeps `ptc_skel.item_eval`);
- the recipe's own record schema ids `ptc_mm.*` (`ptc_mm.lerobot_cut`, `ptc_mm.lerobot_run_args`,
  `ptc_mm.lerobot_frame_export`, `ptc_mm.libero_episode`, `ptc_mm.libero_item_set`, `ptc_mm.libero_items_manifest`,
  `ptc_mm.vla_downloads`, ...): every cut, run record, episode log and the frame export of the published run carry
  them, and the adapter reads those outputs; renaming them would make it refuse the published run;
- the `row_key` text "... (ptc-skel)" in `runtime/frames/frames_meta.json` (part of the training-data snapshot, see
  above) and in `export_frames.py`, which writes it;
- the package-list file names `/etc/ptc-a5-packages.txt` and `/etc/ptc-a8-packages.txt` in the containers' RUN
  instructions: only comments, the documented tags and the `FROM` tag changed in `container/Containerfile` and
  `container/Containerfile.a5`; every RUN instruction is byte-identical to the one that built the published run's
  images, which the adapter pins by ID (`sha256:136745e5...`, built as `ptc-a8-runtime:1`).

Re-pinned for the rename (old -> new SHA-256):

| File | Reason | SHA-256 before | SHA-256 now |
|---|---|---|---|
| `runtime/eval_libero.py` (scorer, member of the composite `code_sha256`) | schema id `gatedtrain.item_eval` | `20f44f7ed81956bfbbeeeae180430110d44830411abf1a73962fbe8b8901f50b` | `7846982d99e1974f92e5e561f11175d34fb48aaba7b522fc570dbb5856c1e7ec` |
| `runtime/a8_common.py` (member of the composite) | the fallback `HF_HOME` directory name (`gatedtrain-a8-hf-home-<uid>`) and a docstring | `f49150f8d826a091ff3f22f086a5097dafc4c86f1a2e8e81b7dd97651d2f221b` | `ce29aa22024a33e08ccbd31c7b2c8ad9a1b5c344e6d1bcbe3486da5d7481f8e6` |
| scorer composite (`rows.EVAL_LIBERO_CODE_SHA256`) | the two files above | `d2ab09c6b97c57eb8074f18bc63ef5b2c7fee18e0439900ce69aa3dbdcdab2ec` | `335559ce7c551af24a41986972bb61192ace2d3b746dc4763301eeecce5e1dfd` |
| `runtime/uv.lock` (`adapter.VENV_LOCK_SHA256`; recorded as `uv_lock_sha256` by every run) | regenerated with `uv lock` for the runtime project's new name; every locked package and version is unchanged (only the project's own entry moved) | `034e5c3b716d245dabc904e1f5c534c804f82bd536792136796a6ab47d075788` | `2a87da026100743dccb92e5c084e64941ff6346cedf2f40ae774e13dfac4b9a6` |
| `runtime/pyproject.toml` | project name, `license`, header comment | `8d58b7c7b1cdf543c4bbf9b93248e723725e993d22d952c67417e516218c391b` | `b99ae76de06fad54b93a1736cbc5957fadd4ef77de4e088dca2946657d89903e` |
| `runtime/export_frames.py` (not pinned) | `gatedtrain` in the docstring | `6a2ed2234eb660ff113f99be7889c19003b55cbb0426003573abce4b514a76cd` | `55f7eeca908c249940e0d85e98fef37e7ab241317dc7365910d4bbded16cc6d1` |
| `runtime/fetch_models.py` (not pinned) | the `lerobot/libero-assets` download made opt-in (`--accept-libero-assets-terms`; README "Data and licenses") | `3c939353aa32163bf3841162a6129477984c3223e3c1d0bdde729ded7968a259` | `cba6e37d1085a005b4653525259480a10b8e4a868a59753410c3887d92b5c77d` |

`runtime/train_lerobot_logged.py` (`93bf86f6...`), the item sets (`items-gate/` manifest `06003603...`,
`ITEMS-MANIFEST.json`), `env_version.json` and the sampler are unchanged. `runtime/env-freeze.txt` keeps its header
(it names the lock the published run's venv was built from, `034e5c3b...`; the package list is the same). The fixture
`tests/fixtures/lerobot/area8-uv.lock` is now a copy of the new `runtime/uv.lock`; the neutral runtime path of the
redacted fixtures is `/srv/gatedtrain/lerobot-smolvla-libero/runtime`.

### The amendment

`results/2026-11-16/gate/registration-gatedtrain.yaml` (sha256
`c66bd893e58f6c9477200276124051f2109094617984c0ff3e555308cad6437a`) amends `registration-ptc-hub.yaml` (`amends:
sha256:6bce6de0...`): it differs in the header comment, `analysis_code_version: gatedtrain 0.3.0`, `amends`,
`created_at` (2026-11-18) and `evaluators[0].manifest_hash` `sha256:432748cb...` ->
`sha256:f070dea953c932fd680815364fc1bb5e73951286c4dcbf8108c322cdebd2e0f3` (the evaluator identity with the re-pinned
scorer composite). The scorer change is not a rename by the published name map alone (the `HF_HOME` directory name),
so no `rename_proof` is given and the amendment is not version-only (report `registration_amendment: null`). `post_hoc:
true` is kept from the prior amendment, so the verdict stays INCONCLUSIVE (post_hoc). `spec-gatedtrain.yaml` (sha256
`83a80e1dd33c3e9e18112134aaee84bbf682fddde2e07ab03f12224ebbe5b9f9`) differs from `spec-ptc-hub.yaml` only in the
registration URI. With `<data-root>` replaced and the items URI pointing at a copy of `runtime/items-gate/`,
`gatedtrain validate spec-gatedtrain.yaml` exits 0.

### Re-judgement on a scratch replay (2026-11-18)

As for the earlier rename, the gate was re-collected with this repository and gatedtrain 0.3.0 from scratch copies;
nothing under the real run, record or output directories was written.

1. Copies of `train-full-1.launch.json`, `train-smoke-1.smoke.json`, the evaluation directories `gate-base-1/` and
   `gate-cand-1/`, `runtime/items-gate/`, and of the stage-2 output directory `full-1/` (7.6 GB, hard links kept, file
   times preserved). In the copies only these bytes changed: the schema ids (`ptc_skel.launch_record`,
   `ptc_skel.smoke_record`, `ptc_skel.item_eval` -> `gatedtrain.*`); the venv lock `034e5c3b...` -> `2a87da02...` in
   the launch record's `runtime.venv_lock_sha256` and in the `uv_lock_sha256` of `full-1/env-runtime.json` and of both
   evaluations' `env-runtime.json`; in both arms' `eval_meta.json` `scorer.code_sha256` `bafc1ce8...` -> `335559ce...`
   and `item_set_hash` `ff603f42...` -> `06003603...`; in both evaluations' `env-runtime.json` the scorer pins
   (`code_sha256`, `script_sha256` from the gate-time pin to `7846982d...`, `common_sha256` `d01995ca...` -> `ce29aa22...`); in
   the smoke record `item_set_hash` `ff603f42...` -> `06003603...`; in the launch record the output directory, its
   read-write mount and `output_root`, now the scratch copy of `full-1/` and its root.
2. The launch record copy was moved onto a scratch host-work directory as before (a copy of `gpu-step.sh` with only
   its `WORK=` line changed, the run's two guard-log lines copied under the new label; `host_work`,
   `gpu_step_sha256`, `host_config_sha256` and `label` recomputed with gatedtrain's own functions; label
   `mm-a8-vla-train-full-5daf4da43995` -> `mm-a8-vla-train-full-ee88d1ed532a`). The real guard log was not written.
3. `gatedtrain ext-collect spec-gatedtrain.yaml --adapter lerobot_smolvla` with the scratch stage-2 output, the
   original base cut (read-only), the scratch copies and the runtime override at the original runtime directory.

Compared with the published report of run `run-20261116T024020Z-805b7cec`: `delta_hat` 0.2275, interval
[0.18625, 0.27], `p_value` 9.64e-05, `achieved_power` 0.99988, `n_items` 400, `baseline_score` 0.2175,
`candidate_score` 0.445, the sampler (`dcc283f1...`), model and checkpoint hashes and 61 report fields in all are
equal. The differences are exactly the expected ones: `status` PASS -> INCONCLUSIVE with `status_reasons`
[post_hoc], `post_hoc` false -> true, `evaluator_hash` and `evaluator_hash_registered` `a6b1265c...` -> `f070dea9...`,
`item_set_hash` and `eval_distribution.item_set_hash` `ff603f42...` -> `06003603...` (the 0.2.0 rewording),
`registration_hash`, `registration_amends`, `analysis_code_version`, the report `schema` id, `lockfile_hash`, `run_id`
and `created_at`. The ledger (79,976 rows) is equal except `writer_id` (`gatedtrain.ledger@0.3.0`) and the
run-id-derived `run_id` and `sample_id`; the lineage tables are equal except `run_id` and the checkpoint `path` (the
scratch copy of `full-1/`); the lineage edges are equal except the activity ids (run id), the item-set and
registration hashes and the schema id. Against the 2026-11-17 replay under ptc-hub 0.2.0 the edges differ only in the
run id, the registration hash and the schema id.

## Known limitation: published cuts name the original SmolVLM2 path (noted 2026-11-18)

Each published cut's `pretrained_model/config.json` holds, in `vlm_model_name`, the absolute path of the SmolVLM2
directory of the runtime that trained it, and `eval_libero.py` (the registered scorer, hash-pinned) does not override
that field, so a published cut cannot be evaluated from a runtime at another path as it stands. `runtime/README.md`
(section 11.1, "Known limitation") gives two ways round it: make the SmolVLM2 directory readable at the recorded path
(the cut and its hashes unchanged, so the published comparison can be reproduced), or evaluate a copy whose
`config.json` names the local directory (its `model_hash` then differs from the published cut's, since `config.json`
is a model-hash component). The scorer is to override `vlm_model_name` with the next registration; the published
result is unaffected.

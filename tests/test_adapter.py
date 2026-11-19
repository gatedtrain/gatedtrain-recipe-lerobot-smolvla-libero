"""External-run adapters, slice 9: area "VLA" (LeRobot 0.7.0 SmolVLA expert-only SFT on LIBERO-Object in two stages;
tests/README.md section 11.26, EXT-233 to EXT-247).

Sources: MULTIMODAL-SPEC.md MM-1 to MM-39 (section 6 row 8 and the row 8 deviations, MM-11, MM-17 to MM-21, MM-26, MM-27,
MM-29, MM-31 to MM-36, MM-38), design revision 16: 05 section 6.1 (R15-1, R16-1), sections 11, 11.2, 12.1, 12.2;
the maintainers' decision record (not published) "Area 8 (VLA) decisions", "Area 8 probes P1/P2", "Area 8 base cut" (2026-11-14/15); the
runtime `runtime/` (README sections 2, 4 to 6, 11; `train_lerobot_logged.py`, `a8_common.py`, `eval_libero.py`; read
2026-11-15) and its GPU outputs `base-1` and `base-1-eval2-<step>`.
Helpers: `tests/_lerobot.py`.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import _lerobot as L
from gatedtrain.testing.ext import sha_of, write_canonical
from gatedtrain.testing.support import EXIT_OK, EXIT_REFUSED, describe, file_sha256, manifest_hash, parse_errors, parse_run_dir

NOT_PASS = 1
REAL_AREA = L.REAL_RUNTIME
REAL_BASE_RUN = L.REAL_OUT / "base-1"
REAL_CUT = REAL_BASE_RUN / "cuts" / "checkpoint-500"
REAL_EVAL = L.REAL_OUT / "base-1-eval2-500"


def registry():
    from gatedtrain.ext import registry as reg
    return reg


def adapter():
    return registry().adapter(L.ADAPTER)


def msgs(problems) -> list[str]:
    return [f"{p.rule} {p.field}: {p.message}" for p in problems]


class Case(unittest.TestCase):
    """One scenario per test that changes it (`fresh`)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-lerobot-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.n = 0

    def fresh(self, **kw) -> L.LerobotScenario:
        type(self).n += 1
        return L.LerobotScenario(self.tmp / f"sc-{type(self).n}", **kw)

    def runs(self) -> Path:
        type(self).n += 1
        return self.tmp / f"runs-{type(self).n}"

    def refused(self, rule: str, field_prefix: str | None = None, *, sc: L.LerobotScenario, text: str | None = None,
                **kw):
        runs = self.runs()
        proc = sc.collect(runs, **kw)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertNotIn("Traceback", proc.stderr)
        errs = parse_errors(proc.stderr)
        hits = [e for e in errs if e["rule"] == rule and (field_prefix is None or e["field"].startswith(field_prefix))
                and (text is None or text in e["message"])]
        self.assertTrue(hits, f"expected error[{rule}] {field_prefix}... {text!r}; got {errs}\n{describe(proc)}")
        self.assertFalse(runs.exists() and any(runs.iterdir()), "a refusal writes no run directory")
        return hits

    def check_hits(self, problems, rule: str, text: str):
        hits = [p for p in problems if p.rule == rule and text in p.message]
        self.assertTrue(hits, f"expected {rule} ... {text!r}; got {msgs(problems)}")


# ---------------------------------------------------------------------------
# EXT-233: registry rows, docs, ids
# ---------------------------------------------------------------------------

class TestRegistryRows(unittest.TestCase):

    def test_trainer_row(self):
        row = registry().TRAINER_ROWS[L.ADAPTER]
        self.assertEqual(row["uses"], (L.TRAINER_ID,))
        self.assertEqual(row["pins"], ("lerobot", "torch", "transformers", "hf-libero", "robosuite", "mujoco"))
        self.assertEqual(row["consumption"], {"target": "row", "fallback": None})
        self.assertEqual(sorted(row["components"]), sorted(L.DEPLOYABLE))
        self.assertEqual(row["evaluators"], (L.EVALUATOR_ID,))
        self.assertEqual(row["framework"], "lerobot")
        self.assertEqual(row["interpreter"], L.INTERPRETER)
        self.assertEqual(row["interpreter_trees"], (L.CPYTHON,))
        self.assertEqual(row["input_roots"], (L.PYTHON_DIR, L.AREA8, "{output_root}/" + L.OUT_NAME))
        self.assertEqual(sorted(row["env_names"]), L.ENV_NAMES)
        self.assertEqual(row["env_paths"], ("HF_HOME", "HOME", "XDG_CACHE_HOME"))
        self.assertEqual(row["interpreter_flags"], {"train_lerobot_logged.py": "-I", "eval_libero.py": "-I"})
        self.assertFalse(row["spec_binding"])
        self.assertNotIn("HF_TOKEN", row["env_names"])
        self.assertIsNone(registry().sidecar(L.ADAPTER))
        self.assertEqual(registry().image_interpreters(L.ADAPTER), {})

    def test_evaluator_row(self):
        row = registry().EVALUATOR_ROWS[L.EVALUATOR_ID]
        self.assertEqual((row["metric"], row["direction"], tuple(row["score_range"]), row["binary"], row["tier"]),
                         ("success", "higher_is_better", (0, 1), True, "T3"))
        self.assertEqual(dict(row["scorer"]), L.SCORER)              # the real arm's scorer, re-pinned code hash
        self.assertEqual(row["error_classes"], ("simulator_error",))
        self.assertEqual(row["score_outcomes"], ("timeout", "failed", "nan_action"))
        self.assertTrue(row["normalizer"] and row["env_version"])
        self.assertEqual(tuple(sorted(row["sampler_keys"])), tuple(sorted(L.SAMPLER)))
        self.assertEqual(row["samples"], "400, 2")
        self.assertIn("reward", row["construct_aligned"])
        self.assertIn("convention", row["shared_annotation"])
        self.assertEqual(row["payload_text_field"], "item")

    def test_ids_and_identity_hash(self):
        self.assertEqual(registry().trainer_ids()[L.TRAINER_ID], L.ADAPTER)
        a = adapter()
        self.assertEqual((a.trainer_id, a.evaluator_ids, a.framework_name), (L.TRAINER_ID, (L.EVALUATOR_ID,), "lerobot"))
        self.assertEqual(registry().evaluator_identity_hash(L.EVALUATOR_ID),
                         sha_of({"evaluator": L.EVALUATOR_ID, "metric": "success", "direction": "higher_is_better",
                                 "scorer": L.SCORER}))
        from gatedtrain.items import EVALUATORS
        self.assertEqual(tuple(EVALUATORS[L.EVALUATOR_ID]["score_range"]), (0, 1))
        m = a.metric(L.EVALUATOR_ID)
        self.assertTrue(m["binary"])
        with self.assertRaises(KeyError):
            a.metric("t2v.wan_ocr_doctr_items@0.1")

    def test_real_item_sets_read(self):
        """The registered item sets (fixture copies of items-gate and items-probe) pass MM-6 with the row's payload."""
        from gatedtrain.ext.collect import read_item_set
        row = registry().EVALUATOR_ROWS[L.EVALUATOR_ID]
        for name, n in (("gate", 400), ("probe", 100)):
            d = Path(tempfile.mkdtemp(prefix="gatedtrain-lerobot-items-"))
            self.addCleanup(shutil.rmtree, d, True)
            shutil.copyfile(L.LR / "items" / f"items-{name}.jsonl", d / "items.jsonl")
            rows, bad = read_item_set(d, tuple(row["item_payload"]))
            self.assertEqual((len(rows), bad), (n, []))

    def test_module_imports_no_framework(self):
        import ast
        from gatedtrain_recipe_lerobot_smolvla_libero import adapter as M
        tree = ast.parse(Path(M.__file__).read_text(encoding="utf-8"))
        names = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        names |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module
                  and n.level == 0}
        self.assertFalse(names & {"torch", "lerobot", "libero", "robosuite", "mujoco", "numpy", "pickle", "transformers"})


# ---------------------------------------------------------------------------
# EXT-234: cuts; EXT-235: identity
# ---------------------------------------------------------------------------

class TestCuts(Case):

    def test_complete_cuts_found(self):
        sc = self.fresh()
        complete, incomplete = adapter().find_cuts(sc.stage_out)
        self.assertEqual(([s for s, _ in complete], incomplete), ([30, 60], []))
        self.assertTrue((sc.stage_out / "lerobot" / "checkpoints" / "000060").is_dir())   # LeRobot's: never a cut
        (sc.stage_out / "cuts" / ".checkpoint-90.partial").mkdir()
        self.assertEqual([s for s, _ in adapter().find_cuts(sc.stage_out)[0]], [30, 60])
        self.assertIsNone(adapter().model_dir_problem(sc.base))

    def test_incomplete_shapes(self):
        sc = self.fresh()
        cut = sc.cuts[0]

        def edit(fn):
            def go(d):
                p = d / "cut.json"
                doc = json.loads(p.read_text())
                fn(doc)
                p.write_text(json.dumps(doc))
            return go

        def write(rel, data):
            def go(d):
                (d / rel).write_bytes(data)
            return go

        def link(d):
            (d / "sampler.json").unlink()
            (d / "sampler.json").symlink_to(L.REAL_CUT / "sampler.json")

        shapes = {
            "no cut.json": lambda d: (d / "cut.json").unlink(),
            "no weights": lambda d: (d / "pretrained_model" / "model.safetensors").unlink(),
            "truncated weights": _truncate,
            "extra file": write("pretrained_model/extra.bin", b"x"),
            "symlink": link,
            "edited optimizer": write("training_state/optimizer_state.safetensors", b"edited"),
            "schema": edit(lambda c: c.update(schema="ptc_mm.whisper_sft_cut")),
            "deployable": edit(lambda c: c.update(deployable=c["deployable"][:-1])),
            "model_files_sha256": edit(lambda c: c.update(model_files_sha256="sha256:" + "0" * 64)),
            "normalizer_sha256": edit(lambda c: c.update(normalizer_sha256="sha256:" + "0" * 64)),
            "sampler_sha256": edit(lambda c: c.update(sampler_sha256="sha256:" + "0" * 64)),
            "last_step_included": edit(lambda c: c.update(last_step_included=29)),
            "samples_consumed": edit(lambda c: c.update(samples_consumed=0)),
            "resume_files": edit(lambda c: c.update(resume_files=[])),
        }
        for name, breaker in shapes.items():
            with self.subTest(shape=name):
                d = self.tmp / f"cut-{name.replace(' ', '-')}"
                shutil.copytree(cut, d, symlinks=True)
                breaker(d)
                self.assertIsNotNone(adapter().cut_problem(d, 30), name)
                self.assertIsNotNone(adapter().model_dir_problem(d), name)

    def test_training_step_and_global_step(self):
        sc = self.fresh()
        d = sc.cuts[0]
        self.assertIn("global_step", adapter().cut_problem(d, 31))
        st = d / "training_state" / "training_step.json"
        st.write_text(json.dumps({"step": 31}))
        L.reseal(d)
        self.assertIn("training_step.json", adapter().cut_problem(d, 30))

    def test_last_cut_incomplete_refused(self):
        sc = self.fresh()
        (sc.final_cut / "pretrained_model" / "model.safetensors").write_bytes(b"cut")
        self.refused("EXT-STAGE", "--stage-out", sc=sc, text="checkpoint-60 is incomplete")

    def test_plain_smolvla_base_refused_as_base(self):
        sc = self.fresh()
        plain = sc.runtime / "models" / "smolvla_base"
        (plain / "config.json").write_text("{}")
        self.assertIn("stage-1 cut", adapter().model_dir_problem(plain))
        sc.base = plain
        sc.render()
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc, text="stage-1 cut")


def _truncate(d: Path) -> None:
    p = d / "pretrained_model" / "model.safetensors"
    p.write_bytes(p.read_bytes()[:-2])
    L.reseal(d)


class TestIdentity(Case):

    def test_model_hash_components(self):
        sc = self.fresh()
        ident = adapter().identity(sc.final_cut)
        self.assertEqual(ident["model_hash"], L.cut_model_hash(sc.final_cut))
        self.assertEqual(sorted(ident["components"]), sorted(L.DEPLOYABLE))
        self.assertEqual(ident["cut_hash"], manifest_hash(sc.final_cut))
        self.assertIsNone(ident["base_ref"])
        self.assertEqual(adapter().identity(sc.final_cut, sc.base), ident)       # full weights: the base is not used

    def test_resume_only_edit_changes_cut_hash_only(self):
        sc = self.fresh()
        d = sc.final_cut
        before = adapter().identity(d)
        for rel in ("training_state/optimizer_state.safetensors", "pretrained_model/train_config.json", "base_ref.json"):
            p = d / rel
            p.write_bytes(p.read_bytes() + b" ")
        after = adapter().identity(d)
        self.assertEqual(after["model_hash"], before["model_hash"])
        self.assertNotEqual(after["cut_hash"], before["cut_hash"])

    def test_normalizer_is_a_component(self):
        """MM-19: another normalizer state changes the model hash and normalizer_sha256 (the area's definition)."""
        sc = self.fresh()
        d = sc.final_cut
        want = L.compact_hash({"normalizer": file_sha256(d / L.NORMALIZER), "unnormalizer": file_sha256(d / L.UNNORMALIZER)})
        self.assertEqual(adapter().normalizer_sha256(d), want)
        self.assertEqual(want, json.loads((d / "cut.json").read_text())["normalizer_sha256"])
        before = adapter().identity(d)["model_hash"]
        p = d / L.NORMALIZER
        p.write_bytes(p.read_bytes()[:-4] + b"\x00\x00\x80\x3f")
        self.assertNotEqual(adapter().identity(d)["model_hash"], before)
        self.assertNotEqual(adapter().normalizer_sha256(d), want)

    @unittest.skip("checkpoint-500/cut.real.json was removed from the publication: it records the hashes of "
                   "unpublished originals")
    def test_real_fixture_normalizer(self):
        """The fixture copies of checkpoint-500's state files give its cut.json normalizer_sha256."""
        real = json.loads((L.REAL_CUT / "cut.real.json").read_text())
        got = L.compact_hash({"normalizer": file_sha256(L.REAL_CUT / L.NORMALIZER),
                              "unnormalizer": file_sha256(L.REAL_CUT / L.UNNORMALIZER)})
        self.assertEqual(got, real["normalizer_sha256"])
        self.assertEqual(sorted(real["deployable"]), sorted(L.DEPLOYABLE))
        self.assertEqual(sorted(real["files"]), sorted(L.CUT_FILES))
        self.assertEqual(real["model_files"], real["deployable"])


# ---------------------------------------------------------------------------
# EXT-246: the whole run collected
# ---------------------------------------------------------------------------

class TestCollectPass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-lerobot-pass-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.sc = L.LerobotScenario(cls.tmp / "sc")
        cls.proc = cls.sc.collect(cls.tmp / "runs")
        cls.run_dir = parse_run_dir(cls.proc.stdout) if cls.proc.returncode == EXIT_OK else None

    def setUp(self):
        self.assertEqual(self.proc.returncode, EXIT_OK, describe(self.proc))

    def report(self) -> dict:
        return json.loads((self.run_dir / "report" / "gate_report.json").read_text(encoding="utf-8"))

    def ledger(self) -> list[dict]:
        import pyarrow.parquet as pq
        return pq.read_table(self.run_dir / "ledger" / "consumption.parquet").to_pylist()

    def test_pass_and_hashes(self):
        rep = self.report()
        self.assertEqual(rep["status"], "PASS")
        self.assertEqual((rep["adapter"], rep["independence_tier"], rep["direction"]), (L.ADAPTER, "T3", "higher_is_better"))
        self.assertEqual((rep["baseline_model_hash"], rep["candidate_model_hash"]), (self.sc.base_model, self.sc.cand_model))
        self.assertEqual(rep["contamination"], {"exact_overlap": 0, "allowed": False})
        self.assertEqual((rep["consumption_precision"], rep["n_items"], rep["samples_per_item"]), ("row", 40, 2))
        self.assertEqual((rep["gpu_seconds"], rep["wall_seconds"]), (3600, 3600))

    def test_disclosures_reported(self):
        rep = self.report()
        self.assertIn("simulator", rep["construct_alignment"]["reward"])
        self.assertIn("BDDL", rep["shared_annotation"]["convention"])
        text = (self.run_dir / "report" / "gate_report.txt").read_text(encoding="utf-8")
        self.assertIn("construct_alignment: ", text)
        self.assertIn("shared_annotation: ", text)

    def test_per_task_note(self):
        notes = self.report()["notes"]
        per = [n for n in notes if n.startswith("Per-task success")]
        self.assertEqual(len(per), 1, notes)
        self.assertIn("t0 0.625/1.000", per[0])
        self.assertTrue([n for n in notes if n.startswith("Clustering")])

    def test_ledger_rows(self):
        rows = self.ledger()
        self.assertEqual(len(rows), 60 * 32 - 2 * 20)                       # epochs of 908: 28 x 32 + 12, twice
        self.assertEqual({(r["precision"], r["derivation"], r["outcome"]) for r in rows}, {("row", "observed", "consumed")})
        self.assertEqual(len({r["sample_id"] for r in rows}), len(rows))
        snap = manifest_hash(self.sc.data)
        frames = {r["index"]: r for r in L.dataset_frames()}
        first = L.read_jsonl(self.sc.stage_out / "samples.jsonl")[0]["samples"][0]
        r0 = rows[0]
        self.assertEqual(r0["step"], 1)
        self.assertEqual(r0["data_ref"]["snapshot"], snap)
        self.assertEqual(r0["data_ref"]["row"], L.row_key(first["episode_index"], first["frame_index"]))
        e = frames[first["index"]]["episode_index"]
        want = next(f for f, (lo, hi) in L.DATA_FILES.items() if lo <= e <= hi)
        self.assertEqual(r0["data_ref"]["shard"], f"chunk-000_file-{want:03d}.jsonl")
        self.assertTrue(all(r["rewards"] is None and r["seed_inputs"] is None and r["chunk"] is None for r in rows))

    def test_lineage_uses_the_base_cut(self):
        edges = json.loads((self.run_dir / "lineage" / "edges.json").read_text(encoding="utf-8"))
        stage = next(a for a in edges["activities"] if a["kind"] == "training_stage")
        used = next(u for u in stage["used"] if u["role"] == "model")
        self.assertEqual(used["model_hash"], self.sc.base_model)
        self.assertEqual(used["hash"], manifest_hash(self.sc.base))
        self.assertEqual([g["last_step_included"] for g in stage["generated"]], [30, 60])
        self.assertFalse([a for a in edges["activities"] if a["kind"] == "lora_compose"])
        gate = next(a for a in edges["activities"] if a["kind"] == "gate")
        self.assertEqual(next(u for u in gate["used"] if u["role"] == "baseline")["hash"], self.sc.base_model)

    def test_impact_on_a_shard(self):
        from gatedtrain.impact import impact_query
        hits = impact_query(str(self.sc.data / "chunk-000_file-212.jsonl"), run_dir=self.run_dir)
        self.assertEqual(sorted(h["last_step_included"] for h in hits), [30, 60])
        self.assertEqual({(h["precision"], h["answer"]) for h in hits}, {("row", "consumed")})
        self.assertEqual(impact_query("sha256:" + "0" * 64, run_dir=self.run_dir), [])

    def test_episode_logs_copied(self):
        rec = L.read_jsonl(self.run_dir / "eval" / "candidate" / "items.jsonl")[0]
        self.assertTrue((self.run_dir / "eval" / "candidate" / rec["output_ref"]["path"]).is_file())


# ---------------------------------------------------------------------------
# EXT-236: two stages, base binding
# ---------------------------------------------------------------------------

class TestBaseBinding(Case):

    def test_stage1_run_as_candidate_refused(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda d: d.update(stage="base"))
        self.check_hits(sc.checks(), "EXT-BASE", "stage-2 continuation")

    def test_base_must_be_a_stage1_cut(self):
        sc = self.fresh()
        self.check_hits(sc.checks(base=sc.cuts[0]), "EXT-BASE", "not a stage-1 cut")
        sc.base = sc.cuts[0]
        sc.render()
        self.refused("EXT-BASE", "artifacts.base.uri", sc=sc, text="not a stage-1 cut")

    def test_base_mount_and_record(self):
        sc = self.fresh()
        rec = json.loads(sc.launch.read_text())
        rec["mounts"] = [dict(m, container="/inputs/base") if m["container"] == L.BASE_CUT_CONTAINER else m
                         for m in rec["mounts"]]
        self.check_hits(sc.checks(launch=rec), "EXT-BASE", "/inputs/base-cut")
        rec["mounts"] = [m for m in rec["mounts"] if m["container"] != "/inputs/base"]
        self.check_hits(sc.checks(launch=rec), "EXT-BASE", "through the launch record's read-only mounts")

    def test_recorded_base_cut_differs(self):
        for key, value in (("global_step", 40), ("cut_json_sha256", "sha256:" + "1" * 64), ("cut_dir", "/inputs/x")):
            with self.subTest(key=key):
                sc = self.fresh()
                sc.edit_json(sc.stage_out / "run_args.json", lambda d: d["base_cut"].update({key: value}))
                self.check_hits(sc.checks(), "EXT-BASE", "base_cut")

    def test_other_base_cut_of_the_ladder(self):
        """The spec names checkpoint-40 of the stage-1 run, the run continued from checkpoint-20."""
        sc = self.fresh()
        other = sc.base_run / "cuts" / "checkpoint-40"
        rec = json.loads(sc.launch.read_text())
        rec["mounts"] = [dict(m, host=str(other)) if m["container"] == L.BASE_CUT_CONTAINER else m for m in rec["mounts"]]
        self.check_hits(sc.checks(launch=rec, base=other), "EXT-BASE", "started from other weights")

    def test_cut_init_checkpoint_and_config(self):
        sc = self.fresh()
        d = sc.cuts[0]
        sc.edit_json(d / "base_ref.json", lambda r: r.update(init_checkpoint=None))
        cfg = json.loads((d / "pretrained_model" / "config.json").read_text())
        cfg["pretrained_path"] = L.SMOLVLA_DIR
        (d / "pretrained_model" / "config.json").write_text(json.dumps(cfg))
        L.reseal(d)
        probs = sc.checks()
        self.check_hits(probs, "EXT-BASE", "init_checkpoint are not run_args.json base_cut")
        self.check_hits(probs, "EXT-BASE", "pretrained_path")

    def test_bindings_pinned(self):
        sc = self.fresh()
        for d in sc.cuts:
            sc.edit_json(d / "base_ref.json", lambda r: r["vlm_backbone"].update(revision="main"))
            L.reseal(d)
        self.check_hits(sc.checks(), "EXT-BASE", "other smolvla_base, SmolVLM2 or dataset bindings")
        sc2 = self.fresh()
        sc2.edit_json(sc2.base / "base_ref.json", lambda r: r["base_model"].update(revision="main"))
        L.reseal(sc2.base)
        self.check_hits(sc2.checks(), "EXT-BASE", "40-hex revision")

    def test_normalizer_changed_in_stage2(self):
        sc = self.fresh()
        d = sc.cuts[0]
        p = d / L.NORMALIZER
        p.write_bytes(p.read_bytes()[:-4] + b"\x00\x00\x80\x3f")
        L.reseal(d)
        self.check_hits(sc.checks(), "EXT-DATA", "normalizer_sha256 is not the base cut's")

    def test_base_run_not_stage1(self):
        sc = self.fresh()
        sc.edit_json(sc.base_run / "run_args.json", lambda d: d.update(stage="continuation"))
        self.check_hits(sc.checks(), "EXT-BASE", "not a stage-1 run")
        sc.edit_json(sc.base_run / "run_args.json", lambda d: d.update(stage="base"))
        (sc.base_run / "run_args.json").unlink()
        self.check_hits(sc.checks(), "EXT-BASE", "stage-1 run's output directory")

    def test_run_args_base_ref(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda d: d["base_ref"].update(init_checkpoint=None))
        self.check_hits(sc.checks(), "EXT-BASE", "run_args.json base_ref")


# ---------------------------------------------------------------------------
# EXT-237: the frame export bound to the dataset the run read
# ---------------------------------------------------------------------------

class TestDataBinding(Case):

    def test_export_row_edited(self):
        sc = self.fresh()
        p = sc.data / "chunk-000_file-212.jsonl"
        rows = L.read_jsonl(p)
        rows[3]["task_index"] = 29 if rows[3]["task_index"] != 29 else 28      # the export stays well formed
        L.write_jsonl(p, rows)
        self.check_hits(sc.checks(), "EXT-DATA", "is not the LIBERO-Object rows")

    def test_parquet_changed(self):
        sc = self.fresh()
        p = sc.dataset / "data" / "chunk-000" / "file-213.parquet"
        p.write_bytes(p.read_bytes() + b"\x00")
        self.check_hits(sc.checks(), "EXT-DATA", "is not the SHA-256")

    def test_dataset_not_mounted(self):
        sc = self.fresh()
        rec = json.loads(sc.launch.read_text())
        rec["mounts"] = [m for m in rec["mounts"] if m["container"] != L.AREA8]
        self.check_hits(sc.checks(launch=rec), "EXT-DATA", "does not resolve through the launch record")

    def test_export_binding(self):
        sc = self.fresh()
        sc.edit_json(sc.data / "frames_meta.json", lambda d: d["dataset"].update(revision="main"))
        self.check_hits(sc.checks(), "EXT-DATA", "frames_meta.json dataset")

    def test_export_shapes(self):
        cases = {
            "missing shard": lambda sc: (sc.data / "chunk-000_file-211.jsonl").unlink(),
            "row count": lambda sc: sc.edit_json(sc.data / "frames_meta.json",
                                                  lambda d: d["shards"]["chunk-000_file-211.jsonl"].update(rows=1)),
            "foreign episode": lambda sc: L.write_jsonl(sc.data / "chunk-000_file-211.jsonl",
                                                        [{"index": 1, "episode_index": 806, "frame_index": 0,
                                                          "task_index": 19}] + L.read_jsonl(
                                                            sc.data / "chunk-000_file-211.jsonl")),
            "n_frames": lambda sc: sc.edit_json(sc.data / "frames_meta.json", lambda d: d.update(n_frames=1)),
        }
        for name, fn in cases.items():
            with self.subTest(case=name):
                sc = self.fresh()
                fn(sc)
                probs = sc.checks()
                self.assertTrue([p for p in probs if p.rule == "EXT-DATA"], msgs(probs))
                ad = adapter()
                self.assertTrue(ad.consumption_problems(sc.stage_out, sc.data, ad.find_cuts(sc.stage_out)[0]))

    def test_run_dataset_record(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json", lambda d: d["dataset"].update(episodes="0..1692 (1693)"))
        self.check_hits(sc.checks(), "EXT-DATA", "LIBERO-Object's")

    def test_refused_at_collect(self):
        sc = self.fresh()
        p = sc.dataset / "data" / "chunk-000" / "file-211.parquet"
        p.write_bytes(p.read_bytes() + b"\x00")
        self.refused("EXT-DATA", "artifacts.frames.uri", sc=sc, text="is not the SHA-256")

    def test_spec_binding_at_launch(self):
        """ext-launch --spec (optional): the base cut and the frame export inside read-only mounts."""
        sc = self.fresh()
        mounts = sc.mounts(sc.new_out("spec-1"), base_cut=sc.base)
        ad = adapter()
        self.assertEqual(ad.mount_binding_problems(mounts, base=sc.base, data=sc.data, base_field="b", data_field="d"), [])
        probs = ad.mount_binding_problems([m for m in mounts if m["container"] != L.BASE_CUT_CONTAINER], base=sc.base,
                                          data=sc.data, base_field="b", data_field="d")
        self.assertEqual([p.rule for p in probs], ["EXT-BASE"])
        no_base = [m for m in mounts if m["container"] != L.BASE_CUT_CONTAINER]
        probs = ad.mount_binding_problems(no_base, base=sc.base, data=sc.data, base_field="b", data_field="d",
                                          argv=sc.argv("smoke"))
        self.assertEqual([p.rule for p in probs], ["EXT-BASE"])
        # An evaluation launch mounts the cut it evaluates, not the base cut: not bound.
        self.assertEqual(ad.mount_binding_problems(no_base, base=sc.base, data=sc.data, base_field="b",
                                                   data_field="d", argv=sc.eval_argv("gate")), [])


# ---------------------------------------------------------------------------
# EXT-238: observed consumption
# ---------------------------------------------------------------------------

class TestConsumptionLog(Case):

    def problems(self, sc) -> list[str]:
        ad = adapter()
        return ad.consumption_problems(sc.stage_out, sc.data, ad.find_cuts(sc.stage_out)[0])

    def refused_log(self, fn, text: str):
        sc = self.fresh()
        sc.edit_samples(fn)
        probs = self.problems(sc)
        self.assertTrue([m for m in probs if text in m], probs)

    def test_scenario_passes(self):
        sc = self.fresh()
        self.assertEqual(self.problems(sc), [])
        lines = sc.samples()
        self.assertEqual([l["n"] for l in lines[27:30]], [32, 12, 32])
        self.assertEqual([l["epoch"] for l in lines[28:30]], [0, 1])

    def test_step_gap(self):
        self.refused_log(lambda ls: ls.pop(10), "is not 11")

    def test_epoch_field(self):
        self.refused_log(lambda ls: ls[29].update(epoch=0), "epoch 0, not 1")

    def test_batch_size(self):
        self.refused_log(lambda ls: ls[3].update(n=31, samples=ls[3]["samples"][:31]), "not 32")

    def test_frame_not_in_export(self):
        self.refused_log(lambda ls: ls[2]["samples"][0].update(index=10), "is not a frame")

    def test_frame_pair_mismatch(self):
        self.refused_log(lambda ls: ls[2]["samples"][0].update(frame_index=ls[2]["samples"][0]["frame_index"] + 1),
                         "is not a frame")

    def test_frame_twice_in_epoch(self):
        def fn(ls):
            ls[5]["samples"][0] = dict(ls[4]["samples"][0])
        self.refused_log(fn, "twice in epoch 0")

    def test_extra_key(self):
        self.refused_log(lambda ls: ls[0].update(attempt=1), "keys")

    def test_log_beyond_the_run(self):
        def fn(ls):
            used = {s["index"] for l in ls if l["epoch"] == 2 for s in l["samples"]}
            fresh = [f for f in L.dataset_frames() if f["index"] not in used][:32]
            ls.append({"step": 61, "epoch": 2, "n": 32, "samples": [{k: f[k] for k in ("index", "episode_index",
                                                                                     "frame_index")} for f in fresh]})
        self.refused_log(fn, "not run_args.json settings.steps 60")

    def test_cut_count(self):
        sc = self.fresh()
        sc.edit_json(sc.cuts[0] / "cut.json", lambda d: d.update(samples_consumed=959))
        self.assertTrue([m for m in self.problems(sc) if "samples_consumed is 959" in m], self.problems(sc))

    def test_refused_at_collect(self):
        sc = self.fresh()
        sc.edit_samples(lambda ls: ls.pop())
        self.refused("EXT-CONSUMPTION", "--stage-out", sc=sc, text="not run_args.json settings.steps")

    def test_consumption_raises_on_bad_log(self):
        sc = self.fresh()
        sc.edit_samples(lambda ls: ls.pop(3))
        ad = adapter()
        with self.assertRaises(ValueError):
            ad.consumption(sc.stage_out, sc.data, ad.find_cuts(sc.stage_out)[0])


# ---------------------------------------------------------------------------
# EXT-239: run checks; EXT-240: sampler
# ---------------------------------------------------------------------------

class TestRunChecks(Case):

    def test_scenario_has_none(self):
        sc = self.fresh()
        self.assertEqual(msgs(sc.checks()), [])

    def test_seed(self):
        sc = self.fresh()
        rec = json.loads(sc.launch.read_text())
        rec["training_seed"] = 7
        probs = sc.checks(launch=rec)
        self.check_hits(probs, "EXT-LAUNCH", "training_seed 7")
        self.assertTrue([p for p in probs if "cut.json seed" in p.message])

    def test_decided_settings(self):
        for key, value in (("batch_size", 64), ("train_expert_only", False), ("n_action_steps", 10)):
            with self.subTest(key=key):
                sc = self.fresh()
                sc.edit_json(sc.stage_out / "run_args.json", lambda d: d["settings"].update({key: value}))
                self.check_hits(sc.checks(), "EXT-STAGE", f"settings.{key}")

    def test_lerobot_argv(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "run_args.json",
                     lambda d: d["lerobot_argv"].__setitem__(1, "--policy.vlm_model_name=HuggingFaceTB/SmolVLM2"))
        self.check_hits(sc.checks(), "EXT-STAGE", "lerobot_argv")

    def test_profile(self):
        sc = self.fresh()
        rec = json.loads(sc.launch.read_text())
        rec["profile"] = "smoke"
        self.check_hits(sc.checks(launch=rec), "EXT-PROFILE", "launch record's 'smoke'")

    def test_script(self):
        sc = self.fresh()
        rec = json.loads(sc.launch.read_text())
        rec["script_sha256"] = "sha256:" + "2" * 64
        self.check_hits(sc.checks(launch=rec), "EXT-LAUNCH", "script_sha256")

    def test_finish_line(self):
        sc = self.fresh()
        lines = L.read_jsonl(sc.stage_out / "attempts.jsonl")
        L.write_jsonl(sc.stage_out / "attempts.jsonl", lines[:1])
        self.check_hits(sc.checks(), "EXT-STAGE", "no finish line")
        lines[1]["blocked_connections"] = ["('203.0.113.1', 443)"]
        L.write_jsonl(sc.stage_out / "attempts.jsonl", lines)
        self.check_hits(sc.checks(), "EXT-STAGE", "blocked connections")

    def test_runtime_pins(self):
        sc = self.fresh()
        sc.edit_json(sc.stage_out / "env-runtime.json", lambda d: d.update(uv_lock_sha256="sha256:" + "3" * 64))
        self.check_hits(sc.checks(), "EXT-FREEZE", "uv.lock")
        sc2 = self.fresh()
        sc2.edit_json(sc2.stage_out / "env-runtime.json", lambda d: d["env_freeze_check"].update(matches=False))
        self.check_hits(sc2.checks(), "EXT-FREEZE", "running distributions")

    def test_last_cut_not_steps(self):
        sc = self.fresh()
        shutil.rmtree(sc.final_cut)
        probs = sc.checks(last_cut=sc.cuts[0])
        self.check_hits(probs, "EXT-STAGE", "did not save its last step")

    def test_freeze_pin_at_collect(self):
        sc = self.fresh()
        text = (sc.stage_out / "env-freeze.txt").read_text().replace("lerobot==0.7.0", "lerobot==0.6.1")
        (sc.stage_out / "env-freeze.txt").write_text(text)
        sc.edit_json(sc.stage_out / "env-runtime.json",
                     lambda d: d.update(env_freeze_sha256=file_sha256(sc.stage_out / "env-freeze.txt")))
        self.refused("EXT-FREEZE", "--stage-out", sc=sc, text="lerobot==0.6.1")

    def test_cut_fields(self):
        sc = self.fresh()
        sc.edit_json(sc.cuts[0] / "cut.json", lambda d: d.update(lerobot_step_dir="lerobot/checkpoints/30"))
        self.check_hits(sc.checks(), "EXT-STAGE", "lerobot_step_dir")


class TestSampler(Case):

    def test_registered_sampler_differs(self):
        sc = self.fresh()
        reg = dict(L.SAMPLER, num_steps=5)
        probs = adapter().sampler_problems(sc.cuts, reg)
        self.assertEqual({p.field for p in probs}, {"registration.sampler", "--stage-out"})

    def test_cut_sampler_not_its_config(self):
        sc = self.fresh()
        d = sc.cuts[0]
        cfg = json.loads((d / "pretrained_model" / "config.json").read_text())
        cfg["num_steps"] = 5
        (d / "pretrained_model" / "config.json").write_text(json.dumps(cfg))
        L.reseal(d)
        probs = adapter().sampler_problems(sc.cuts, L.SAMPLER)
        self.assertTrue([p for p in probs if "deploy_sampler" in p.message], msgs(probs))

    def test_refused_at_collect(self):
        sc = self.fresh()
        sc.render(reg_mutate=lambda r: r["sampler"].update(n_action_steps=10))
        self.refused("EXT-SAMPLER", "registration.sampler", sc=sc)


# ---------------------------------------------------------------------------
# EXT-241: arms and ext-arm-fill
# ---------------------------------------------------------------------------

class TestArms(Case):

    def arm_problems(self, sc, arm: Path, cut: Path, items: Path | None = None) -> list[str]:
        from gatedtrain.ext.arm import read_lines
        row = registry().EVALUATOR_ROWS[L.EVALUATOR_ID]
        recs, bad = read_lines(arm, seeds=L.SEEDS, score_range=row["score_range"], error_classes=row["error_classes"])
        self.assertEqual(bad, [])
        return adapter().arm_problems(arm, recs, sc.item_rows, items_dir=items or sc.items_dir, model_dir=cut)

    def test_fill_then_collect(self):
        sc = self.fresh(filled=False)
        for arm, cut in ((sc.baseline, sc.base), (sc.candidate, sc.final_cut)):
            proc = sc.fill(arm, cut)
            self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
            meta = json.loads((arm / "eval_meta.json").read_text())
            self.assertEqual(meta["checkpoint"]["model_hash"], L.cut_model_hash(cut))
            self.assertEqual(meta["item_set_hash"], sc.item_set_hash)
        proc = sc.collect(self.runs())
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))

    def test_fill_checks_env_version_and_normalizer(self):
        sc = self.fresh(filled=False)
        sc.edit_json(sc.baseline / "env_version.json", lambda d: d["env_kwargs"].update(max_steps=3))
        proc = sc.fill(sc.baseline, sc.base)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertTrue([e for e in parse_errors(proc.stderr) if e["field"] == "ARM.env_version_sha256"])
        proc = sc.fill(sc.candidate, sc.base)                      # the candidate arm against another cut
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))     # same normalizer, other model hash: filled
        sc2 = self.fresh(filled=False)
        sc2.edit_json(sc2.candidate / "eval_meta.json", lambda d: d.update(normalizer_sha256="sha256:" + "4" * 64))
        proc = sc2.fill(sc2.candidate, sc2.final_cut)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertTrue([e for e in parse_errors(proc.stderr) if e["field"] == "ARM.normalizer_sha256"])

    def test_scenario_arms_pass(self):
        sc = self.fresh()
        self.assertEqual(self.arm_problems(sc, sc.baseline, sc.base), [])
        self.assertEqual(self.arm_problems(sc, sc.candidate, sc.final_cut), [])

    def test_other_cut_and_items(self):
        sc = self.fresh()
        probs = self.arm_problems(sc, sc.baseline, sc.final_cut)
        self.assertTrue([m for m in probs if "evaluated another cut" in m], probs)
        probs = self.arm_problems(sc, sc.baseline, sc.base, items=sc.probe_dir)
        self.assertTrue([m for m in probs if "items_sha256" in m], probs)

    def test_episode_log_seeds(self):
        sc = self.fresh()
        rec = L.read_jsonl(sc.candidate / "items.jsonl")[5]
        p = sc.candidate / rec["output_ref"]["path"]
        doc = json.loads(p.read_text())
        doc["noise_seed"] += 1
        L.write_json(p, doc)
        lines = L.read_jsonl(sc.candidate / "items.jsonl")
        lines[5]["output_ref"]["sha256"] = file_sha256(p)
        L.write_jsonl(sc.candidate / "items.jsonl", lines)
        probs = self.arm_problems(sc, sc.candidate, sc.final_cut)
        self.assertTrue([m for m in probs if "noise_seed" in m], probs)

    def test_outcome_and_score(self):
        cases = {"score": lambda l: l.update(score=1), "outcome": lambda l: l["item_meta"].update(outcome="failed"),
                 "steps_to_success": lambda l: l["sub_metrics"].update(steps_to_success=100),
                 "init_state": lambda l: l["item_meta"].update(init_state=5)}
        for name, fn in cases.items():
            with self.subTest(case=name):
                sc = self.fresh()
                lines = L.read_jsonl(sc.baseline / "items.jsonl")
                i = next(k for k, l in enumerate(lines) if l["score"] == 0)
                fn(lines[i])
                L.write_jsonl(sc.baseline / "items.jsonl", lines)
                self.assertTrue(self.arm_problems(sc, sc.baseline, sc.base))

    def test_error_lines_and_reasons(self):
        sc = self.fresh()
        lines = L.read_jsonl(sc.baseline / "items.jsonl")
        lines[0].update(score=None, error={"class": "simulator_error", "message": "item_timeout: no result within 300 s"},
                        sub_metrics={"wall_s": 300.0}, item_meta=dict(lines[0]["item_meta"], outcome=None))
        L.write_jsonl(sc.baseline / "items.jsonl", lines)
        p = sc.baseline / lines[0]["output_ref"]["path"]
        doc = json.loads(p.read_text())
        doc["error"] = lines[0]["error"]
        L.write_json(p, doc)
        lines[0]["output_ref"]["sha256"] = file_sha256(p)
        L.write_jsonl(sc.baseline / "items.jsonl", lines)
        probs = self.arm_problems(sc, sc.baseline, sc.base)
        self.assertTrue([m for m in probs if "error reasons" in m], probs)            # summary still says 0
        sc.edit_json(sc.base_eval / "summary.json",
                     lambda d: d.update(errors=1, error_reasons={"item_timeout": 1, "worker_died": 0}))
        self.assertEqual(self.arm_problems(sc, sc.baseline, sc.base), [])

    def test_nondefault_episode_settings(self):
        sc = self.fresh()
        sc.edit_json(sc.base_eval / "summary.json", lambda d: d.update(env_version_nondefault={"max_steps": 3}))
        probs = self.arm_problems(sc, sc.baseline, sc.base)
        self.assertTrue([m for m in probs if "non-default" in m], probs)

    def test_refused_at_collect(self):
        sc = self.fresh()
        sc.edit_json(sc.cand_eval / "summary.json", lambda d: d.update(global_step=30))
        self.refused("EXT-ARM", "--eval-candidate", sc=sc, text="evaluated another cut")

    def test_arm_env_version_must_be_registered(self):
        sc = self.fresh()
        sc.render(reg_mutate=lambda r: r["eval_distribution"].update(env_version="sha256:" + "5" * 64))
        self.refused("EXT-ARM", "--eval-baseline", sc=sc, text="env_version_sha256")


# ---------------------------------------------------------------------------
# EXT-242: registration
# ---------------------------------------------------------------------------

class TestRegistration(Case):

    def test_disclosures_required(self):
        for key in ("construct_alignment", "shared_annotation"):
            with self.subTest(key=key):
                sc = self.fresh()
                sc.render(reg_mutate=lambda r: r.pop(key))
                hits = self.refused("REG-FIELD", f"registration.{key}", sc=sc)
                self.assertTrue(hits)

    def test_sampler_keys(self):
        sc = self.fresh()
        sc.render(reg_mutate=lambda r: r["sampler"].pop("noise"))
        self.refused("REG-FIELD", "registration.sampler.noise", sc=sc)

    def test_env_version_required(self):
        sc = self.fresh()
        sc.render(reg_mutate=lambda r: r["eval_distribution"].pop("env_version"))
        self.refused("REG-FIELD", "registration.eval_distribution.env_version", sc=sc)


# ---------------------------------------------------------------------------
# EXT-243: argv rules
# ---------------------------------------------------------------------------

class TestArgv(unittest.TestCase):

    def problems(self, argv, profile="full"):
        return adapter().argv_problems(argv, profile=profile)

    def test_stage2_and_stage1(self):
        I, T = L.INTERPRETER, L.TRAIN_PY
        ok = [I, "-I", T, "--profile", "full", "--policy-path", "/inputs/base-cut", "--out", "/out", "--steps", "2500",
              "--save-freq", "500", "--seed", "2000"]
        self.assertEqual(self.problems(ok), [])
        self.assertEqual(self.problems([I, T] + ok[3:]), [])                   # without -I
        smoke = [I, "-I", T, "--profile", "smoke", "--policy-path=/inputs/base-cut", "--out", "/out", "--seed", "2000"]
        self.assertEqual(self.problems(smoke, "smoke"), [])
        base = [I, "-I", T, "--profile", "base", "--out", "/out", "--batch-size", "32", "--seed", "1000"]
        self.assertEqual(self.problems(base, "smoke"), [])
        self.assertEqual(self.problems([I, "-I", T, "--prof", "full", "--policy", "/inputs/base-cut", "--out", "/out",
                                        "--steps", "10", "--seed", "1"]), [])  # unique prefixes, as argparse

    def test_training_refusals(self):
        I, T = L.INTERPRETER, L.TRAIN_PY
        base = [I, "-I", T, "--profile", "full", "--policy-path", "/inputs/base-cut", "--out", "/out", "--steps", "60",
                "--seed", "2000"]
        cases = {
            "profile smoke under full": ([I, "-I", T, "--profile", "smoke", "--policy-path", "/inputs/base-cut",
                                          "--out", "/out", "--seed", "1"], "full", "--profile"),
            "probe": ([I, "-I", T, "--profile", "probe", "--out", "/out", "--seed", "1"], "smoke", "--profile"),
            "base under full": ([I, "-I", T, "--profile", "base", "--out", "/out", "--seed", "1"], "full", "--profile"),
            "policy path": (base[:6] + ["/inputs/cut"] + base[7:], "full", "--policy-path"),
            "no policy path": (base[:5] + base[7:], "full", "--policy-path"),
            "base with policy path": ([I, "-I", T, "--profile", "base", "--policy-path", "/inputs/base-cut", "--out",
                                       "/out", "--seed", "1"], "smoke", "no --policy-path"),
            "no seed": (base[:-2], "full", "--seed"),
            "out": (base[:8] + ["/tmp/out"] + base[9:], "full", "--out"),
            "dry run": (base + ["--dry-run"], "full", "--dry-run"),
            "full ft": (base + ["--full-ft"], "full", "--full-ft"),
            "batch": (base + ["--batch-size", "64"], "full", "--batch-size"),
            "n action steps": (base + ["--n-action-steps", "10"], "full", "--n-action-steps"),
            "device": (base + ["--device", "cpu"], "full", "--device"),
            "unknown": (base + ["--resume"], "full", "--resume"),
            "positional": (base + ["extra"], "full", "'extra'"),
            "value missing": (base + ["--lr"], "full", "--lr"),
            "copy elsewhere": ([I, "-I", "/inputs/code/train_lerobot_logged.py"] + base[3:], "full", "runtime's scripts"),
            "other flag": ([I, "-B", T] + base[3:], "full", "runtime's scripts"),
        }
        for name, (argv, profile, text) in cases.items():
            with self.subTest(case=name):
                probs = self.problems(argv, profile)
                self.assertTrue([m for m in probs if text in m], probs)

    def test_evaluation(self):
        I, E = L.INTERPRETER, L.EVAL_PY
        ok = [I, "-I", E, "--checkpoint", "/inputs/cut", "--items", "gate", "--eval-seeds", "11", "12", "--workers", "8",
              "--out", "/out"]
        self.assertEqual(self.problems(ok), [])
        probe = [I, "-I", E, "--checkpoint", "/inputs/cut", "--items", "probe", "--per-task", "5", "--eval-seeds", "11",
                 "--out", "/out", "--item-timeout-s", "300"]
        self.assertEqual(self.problems(probe, "smoke"), [])
        cases = {
            "probe under full": (ok[:6] + ["probe"] + ok[7:], "full", "--items"),
            "item dir": (ok[:6] + [f"{L.AREA8}/items-gate"] + ok[7:], "smoke", "--items"),
            "checkpoint": (ok[:4] + ["/inputs/base-cut"] + ok[5:], "full", "--checkpoint"),
            "limit under full": (ok + ["--limit", "5"], "full", "--limit"),
            "per task under full": (ok + ["--per-task", "5"], "full", "--per-task"),
            "write items": (ok + ["--write-items"], "smoke", "--write-items"),
            "max steps": (ok + ["--max-steps", "3"], "smoke", "--max-steps"),
            "settle": (ok + ["--settle-steps", "0"], "smoke", "--settle-steps"),
            "timeout": (ok + ["--item-timeout-s", "60"], "smoke", "--item-timeout-s"),
            "workers 0": (ok + ["--workers", "0"], "smoke", "--workers 0"),
            "ambiguous prefix": (ok + ["--item", "gate"], "smoke", "'--item'"),
        }
        for name, (argv, profile, text) in cases.items():
            with self.subTest(case=name):
                probs = self.problems(argv, profile)
                self.assertTrue([m for m in probs if text in m], probs)


# ---------------------------------------------------------------------------
# EXT-244: launch; EXT-245: smoke
# ---------------------------------------------------------------------------

class LaunchCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-lerobot-launch-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.sc = L.LerobotScenario(cls.tmp / "sc", n_per_task=2)
        cls.k = 0

    def out(self) -> Path:
        type(self).k += 1
        return self.sc.new_out(f"run-{type(self).k}")

    def launch(self, req: dict, **kw):
        type(self).k += 1
        return self.sc.launch_request(req, f"req-{type(self).k}", **kw)

    def refused(self, proc, out, rule, field, text: str | None = None):
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        errs = parse_errors(proc.stderr)
        self.assertTrue([e for e in errs if e["rule"] == rule and e["field"] == field
                         and (text is None or text in e["message"])], f"{rule} {field} {text!r}: {errs}")
        self.assertFalse(out.exists())


class TestLaunchRules(LaunchCase):

    def test_stage2_smoke_request(self):
        proc, out = self.launch(self.sc.request(self.out()))
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertIn("--runtime=nvidia", proc.stdout)
        self.assertIn(f"{L.IMAGE} {L.INTERPRETER} -I {L.TRAIN_PY} --profile smoke --policy-path /inputs/base-cut",
                      proc.stdout)
        self.assertIn(":/inputs/base-cut:ro", proc.stdout)
        for c in (L.PYTHON_DIR, L.AREA8):
            self.assertIn(f":{c}:ro", proc.stdout)
        self.assertIn("-e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics", proc.stdout)
        rec = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual((rec["adapter"], rec["runtime"]["venv_lock_sha256"], rec["script_path"]),
                         (L.ADAPTER, L.LOCK_SHA, L.TRAIN_PY))
        self.assertEqual(rec["interpreter_path"], f"{L.CPYTHON}/bin/python3.12")
        proc = self.sc.recheck(out)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))

    def test_full_request_needs_the_smoke(self):
        req = self.sc.request(self.out(), profile="full", gpu_est_h=0.4)
        proc, out = self.launch(req, profile="full")
        self.refused(proc, out, "EXT-SMOKE", "--smoke-record")
        proc, out = self.launch(self.sc.request(self.out(), profile="full"), profile="full", smoke=self.sc.smoke)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))

    def test_evaluation_requests(self):
        for cut, items, profile in ((self.sc.base, "probe", "smoke"), (self.sc.final_cut, "gate", "full")):
            with self.subTest(cut=cut.name):
                req = self.sc.eval_request(self.out(), cut=cut, items=items)
                proc, out = self.launch(req, profile=profile, smoke=self.sc.smoke if profile == "full" else None)
                self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
                self.assertIn(f"{L.INTERPRETER} -I {L.EVAL_PY} --checkpoint /inputs/cut", proc.stdout)
                self.assertEqual(json.loads(out.read_text())["script_path"], L.EVAL_PY)

    def test_spec_binds_training_not_evaluation(self):
        """`--spec` on the candidate arm (another cut at /inputs/cut) launches and rechecks; a stage-2 training
        request whose base mount is not the spec's base cut is refused."""
        spec = ["--spec", str(self.sc.paths["spec"])]
        req = self.sc.eval_request(self.out(), cut=self.sc.final_cut, items="gate")
        proc, out = self.launch(req, profile="full", smoke=self.sc.smoke, extra=spec)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        proc = self.sc.recheck(out, spec)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        o = self.out()
        req = self.sc.request(o, mounts=self.sc.mounts(o, base_cut=self.sc.final_cut))
        proc, out = self.launch(req, extra=spec)
        self.refused(proc, out, "EXT-BASE", "artifacts.base.uri")

    def test_mount_set(self):
        o = self.out()
        req = self.sc.request(o, mounts=self.sc.mounts(o))                       # stage 2 without the base cut
        proc, out = self.launch(req)
        self.refused(proc, out, "EXT-LAUNCH", "--request.mounts")
        o = self.out()
        req = self.sc.eval_request(o, mounts=self.sc.mounts(o))                  # --checkpoint without the cut
        proc, out = self.launch(req)
        self.refused(proc, out, "EXT-LAUNCH", "--request.mounts")
        o = self.out()
        req = self.sc.request(o, mounts=self.sc.mounts(o, base_cut=self.sc.cuts[0]))   # a stage-2 cut as the base
        proc, out = self.launch(req)
        self.refused(proc, out, "EXT-LAUNCH", "--request.mounts", "stage-1 (base) cut")

    def test_image_lock_network_gpus(self):
        bad_img = {"kind": "venv_in_container", "image": "sha256:" + "ab" * 32, "venv_lock": str(self.sc.lock)}
        proc, out = self.launch(self.sc.request(self.out(), runtime=bad_img))
        self.refused(proc, out, "EXT-LAUNCH", "--request.runtime", "gatedtrain-a8-runtime:1")
        lock = self.tmp / "other-uv.lock"
        lock.write_text("# another lock\n")
        proc, out = self.launch(self.sc.request(self.out(), runtime={"kind": "venv_in_container", "image": L.IMAGE,
                                                                     "venv_lock": str(lock)}))
        self.refused(proc, out, "EXT-LAUNCH", "--request.runtime", "uv.lock")
        proc, out = self.launch(self.sc.request(self.out(), gpus=False))
        self.refused(proc, out, "EXT-LAUNCH", "--request.gpus")

    def test_seed(self):
        proc, out = self.launch(self.sc.request(self.out(), training_seed=1))
        self.refused(proc, out, "EXT-LAUNCH", "--request.argv", "training_seed 1")

    def test_env_values_never_shown(self):
        env = dict(L.ENV, MUJOCO_GL="osmesa-SECRETVALUE")
        proc, out = self.launch(self.sc.request(self.out(), env=env))
        self.refused(proc, out, "EXT-LAUNCH", "--request.env.MUJOCO_GL")
        self.assertNotIn("SECRETVALUE", proc.stderr + proc.stdout)
        env = dict(L.ENV, LIBERO_CONFIG_PATH="/tmp/other")
        proc, out = self.launch(self.sc.request(self.out(), env=env))
        self.refused(proc, out, "EXT-LAUNCH", "--request.env.LIBERO_CONFIG_PATH")
        env = dict(L.ENV, HF_HOME=f"{L.AREA8}/hf-home")                          # an input, not a writable cache
        proc, out = self.launch(self.sc.request(self.out(), env=env))
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))

    def test_eval_needs_rendering_names(self):
        env = {k: v for k, v in L.ENV.items() if k != "MUJOCO_EGL_DEVICE_ID"}
        proc, out = self.launch(self.sc.eval_request(self.out(), env=env))
        self.refused(proc, out, "EXT-LAUNCH", "--request.env", "MUJOCO_EGL_DEVICE_ID")

    def test_other_env_names_refused(self):
        for name in ("HF_TOKEN", "CUDA_VISIBLE_DEVICES", "LD_PRELOAD"):
            with self.subTest(name=name):
                proc, out = self.launch(self.sc.request(self.out(), env=dict(L.ENV, **{name: "x"})))
                self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
                self.assertFalse(out.exists())

    def test_record_with_other_image_refused_at_collect(self):
        sc = L.LerobotScenario(self.tmp / "sc-image", n_per_task=2)
        sc.rewrite_launch(runtime={"kind": "venv_in_container", "image": "sha256:" + "cd" * 32,
                                   "venv_lock_sha256": file_sha256(sc.lock)})
        proc = sc.collect(self.tmp / "runs-image")
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertTrue([e for e in parse_errors(proc.stderr) if e["field"] == "--launch-record.runtime"])


class TestSmoke(LaunchCase):

    def full(self, smoke: dict):
        p = write_canonical(self.tmp / f"smoke-{type(self).k}.json", smoke)
        return self.launch(self.sc.request(self.out(), profile="full"), profile="full", smoke=p)

    def test_binary_rule(self):
        """MM-38 for a binary metric: degenerate only at mean 0 or 1; no mode or sd rule (70 % zeros pass)."""
        base = L.smoke_record(base_model=self.sc.base_model, item_set=self.sc.item_set_hash)
        proc, out = self.full(base)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        for scores in ([0] * 20, [1] * 20):
            with self.subTest(mean=scores[0]):
                proc, out = self.full(dict(base, baseline_scores=scores))
                self.refused(proc, out, "EXT-SMOKE", "--smoke-record", "degenerate_mean")

    def test_real_probe_scores_pass(self):
        """The base cut's real probe scores (base-1-eval2-500, mean 0.26) pass the binary rule."""
        lines = L.read_jsonl(L.REAL_EVAL / "arm-success" / "items.jsonl")
        scores = [l["score"] for l in lines]
        self.assertAlmostEqual(sum(scores) / len(scores), 0.26)
        proc, out = self.full(L.smoke_record(base_model=self.sc.base_model, item_set=self.sc.item_set_hash,
                                             baseline_scores=scores))
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))

    def test_degenerate_smoke_refused_at_collect(self):
        sc = L.LerobotScenario(self.tmp / "sc-smoke", n_per_task=2)
        p = write_canonical(self.tmp / "smoke-deg.json", L.smoke_record(base_model=sc.base_model,
                                                                         item_set=sc.item_set_hash,
                                                                         baseline_scores=[0] * 20))
        proc = sc.collect(self.tmp / "runs-smoke", smoke=p)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertTrue([e for e in parse_errors(proc.stderr) if e["rule"] == "EXT-SMOKE"])


# ---------------------------------------------------------------------------
# EXT-247: the real runtime and GPU outputs (skipped where absent; read only)
# ---------------------------------------------------------------------------

class TestRealFiles(unittest.TestCase):

    def need(self, *paths: Path):
        for p in paths:
            if not p.exists():
                self.skipTest(f"absent: {p}")

    def test_real_stage1_log_against_the_real_dataset(self):
        """The real base-1 samples.jsonl (3,000 steps, batch 32) passes against a frame export built from the real
        parquet files (98 data files, 66,984 frames): every sampled frame exists in episodes 807-1260, every epoch
        reads each frame once, the epoch's last step carries 8, and each cut's samples_consumed is the log's."""
        ds = REAL_AREA / "data" / "lerobot" / "libero"
        self.need(REAL_BASE_RUN / "samples.jsonl", ds / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
        import pyarrow.parquet as pq
        eps = pq.read_table(ds / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
                            columns=["episode_index", "data/chunk_index", "data/file_index"]).to_pylist()
        sources = sorted({f"data/chunk-{e['data/chunk_index']:03d}/file-{e['data/file_index']:03d}.parquet" for e in eps
                          if 807 <= e["episode_index"] <= 1260})
        tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-lerobot-real-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        args = json.loads((REAL_BASE_RUN / "run_args.json").read_text())
        export = L.write_frame_export(ds, tmp / "frames", files_sha256=args["base_ref"]["dataset"]["files_sha256"],
                                      sources=sources)
        meta = json.loads((export / "frames_meta.json").read_text())
        self.assertEqual((meta["n_frames"], len(meta["shards"])), (66984, 98))
        cuts = [(int(p.name.split("-")[1]), p) for p in (REAL_BASE_RUN / "cuts").glob("checkpoint-*")]
        ad = adapter()
        self.assertEqual(ad.consumption_problems(REAL_BASE_RUN, export, sorted(cuts)), [])
        mounts = [{"host": L.PYTHON_DIR, "container": L.PYTHON_DIR, "mode": "ro"},
                  {"host": str(REAL_AREA), "container": str(REAL_AREA), "mode": "ro"}]
        self.assertEqual(msgs(ad._data_problems(args, export, mounts, "d")), [])
        lines = L.read_jsonl(REAL_BASE_RUN / "samples.jsonl")
        self.assertEqual((len(lines), lines[2093]["n"], lines[2093]["epoch"], lines[2094]["epoch"]), (3000, 8, 0, 1))

    def test_real_cut_complete(self):
        """checkpoint-500 (the gate's base) is a complete cut; its components are cut.json's deployable list."""
        self.need(REAL_CUT / "cut.json")
        ad = adapter()
        self.assertIsNone(ad.cut_problem(REAL_CUT, 500))
        doc = json.loads((REAL_CUT / "cut.json").read_text())
        self.assertEqual(ad.normalizer_sha256(REAL_CUT), doc["normalizer_sha256"])
        self.assertEqual(sorted(doc["deployable"]), sorted(registry().TRAINER_ROWS[L.ADAPTER]["components"]))
        self.assertEqual(msgs(ad._binding_problems(json.loads((REAL_CUT / "base_ref.json").read_text()), "x", "f")), [])

    def test_real_run_args(self):
        """The real stage-1 run_args.json: its lerobot_argv is the launcher's command line rebuilt by the adapter and
        by the test helper, and its fixture copy is identical up to the runtime path (the fixtures were redacted to the
        neutral `L.RECORDED_AREA8`)."""
        self.need(REAL_BASE_RUN / "run_args.json")
        from gatedtrain_recipe_lerobot_smolvla_libero import adapter as M
        args = json.loads((REAL_BASE_RUN / "run_args.json").read_text())
        self.assertEqual(args["lerobot_argv"], M.lerobot_argv(args["settings"], args["policy_path"]))
        self.assertEqual(args["lerobot_argv"], L.lerobot_argv(args["settings"], args["policy_path"]))
        self.assertEqual(args["settings"]["cut_steps"], M.cut_steps(args["settings"]))
        real_root = args["policy_path"].rsplit("/models/", 1)[0]
        self.assertEqual((REAL_BASE_RUN / "run_args.json").read_text().replace(real_root, L.RECORDED_AREA8),
                         (L.REAL_RUN / "run_args.json").read_text())

    def test_real_arm(self):
        """base-1-eval2-500 (the base cut on the 100 probe items): its lines pass MM-1 and the adapter's arm rules
        with the probe items and checkpoint-500, except the scorer code hash: the arm was written by the scorer before
        its re-pins for ptc-hub 0.2.0 and gatedtrain 0.3.0 (RESULT.md); under the gate items they refuse; its scorer is otherwise the
        registry's."""
        self.need(REAL_EVAL / "arm-success" / "items.jsonl", REAL_CUT / "cut.json")
        from gatedtrain.ext.arm import read_lines
        row = registry().EVALUATOR_ROWS[L.EVALUATOR_ID]
        arm = REAL_EVAL / "arm-success"
        recs, bad = read_lines(arm, seeds=[11], score_range=row["score_range"], error_classes=row["error_classes"])
        self.assertEqual((len(recs), bad), (100, []))
        probe = REAL_AREA / "items-probe"
        rows = L.read_jsonl(probe / "items.jsonl")
        pre_rename = ("env-runtime.json code_sha256 is not the registered scorer's composite of eval_libero.py and "
                      "a8_common.py")
        self.assertEqual(adapter().arm_problems(arm, recs, rows, items_dir=probe, model_dir=REAL_CUT), [pre_rename])
        probs = adapter().arm_problems(arm, recs, rows, items_dir=REAL_AREA / "items-gate", model_dir=REAL_CUT)
        self.assertTrue([m for m in probs if "items_sha256" in m], probs)
        scorer = json.loads((arm / "eval_meta.json").read_text())["scorer"]
        self.assertEqual({**scorer, "code_sha256": row["scorer"]["code_sha256"]}, dict(row["scorer"]))
        self.assertEqual(scorer["code_sha256"], L.compact_hash({"eval_libero.py": L.PRE_RENAME_EVAL_LIBERO_SHA,
                                                                "a8_common.py": L.PRE_RENAME_COMMON_SHA}))
        self.assertEqual(adapter().arm_env_version_sha256(arm), L.compact_hash(L.ENV_VERSION))

    def test_real_fill_on_a_copy(self):
        """ext-arm-fill on a copy of the real arm with the real cut: fills the model and item set hashes (the copy's
        schema id renamed, `L.renamed_copy`: the arm predates the renames)."""
        self.need(REAL_EVAL / "arm-success" / "eval_meta.json", REAL_CUT / "cut.json")
        from gatedtrain.ext.fill import arm_fill
        tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-lerobot-fill-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        shutil.copytree(REAL_EVAL, tmp / "eval")
        L.renamed_copy(tmp / "eval" / "arm-success" / "eval_meta.json")
        res = arm_fill(adapter=L.ADAPTER, arm=tmp / "eval" / "arm-success", checkpoint=REAL_CUT,
                       items=REAL_AREA / "items-probe")
        self.assertEqual(sorted(res["filled"]), ["checkpoint.model_hash", "item_set_hash"])
        self.assertEqual(res["values"]["item_set_hash"], manifest_hash(REAL_AREA / "items-probe"))

    def test_runtime_pins(self):
        self.need(REAL_AREA / "eval_libero.py", REAL_AREA / "uv.lock")
        self.assertEqual(file_sha256(REAL_AREA / "uv.lock"), L.LOCK_SHA)
        self.assertEqual(file_sha256(L.LR / "area8-uv.lock"), L.LOCK_SHA)
        self.assertEqual((file_sha256(REAL_AREA / "eval_libero.py"), file_sha256(REAL_AREA / "a8_common.py")),
                         (L.EVAL_LIBERO_SHA, L.COMMON_SHA))
        self.assertEqual(L.compact_hash({"eval_libero.py": L.EVAL_LIBERO_SHA, "a8_common.py": L.COMMON_SHA}),
                         registry().EVALUATOR_ROWS[L.EVALUATOR_ID]["scorer"]["code_sha256"])
        self.assertEqual(file_sha256(REAL_AREA / "train_lerobot_logged.py"), L.TRAIN_SHA)
        man = json.loads((REAL_AREA / "ITEMS-MANIFEST.json").read_text())
        for name, d in (("gate", "items-gate"), ("probe", "items-probe")):
            self.assertEqual(man["sets"][name]["files"]["items.jsonl"], file_sha256(REAL_AREA / d / "items.jsonl"))
            self.assertEqual(file_sha256(L.LR / "items" / f"items-{name}.jsonl"),
                             file_sha256(REAL_AREA / d / "items.jsonl"))
        repos = json.loads((REAL_AREA / "MANIFEST.json").read_text())["repos"]
        from gatedtrain_recipe_lerobot_smolvla_libero import adapter as M
        self.assertEqual({k: (repos[k]["repo"], repos[k]["revision"]) for k in ("smolvla_base",
                                                                               "SmolVLM2-500M-Video-Instruct", "libero")},
                         {"smolvla_base": (M.SMOLVLA_REPO, M.SMOLVLA_REVISION),
                          "SmolVLM2-500M-Video-Instruct": (M.SMOLVLM_REPO, M.SMOLVLM_REVISION),
                          "libero": (M.DATASET_REPO, M.DATASET_REVISION)})
        freeze = (REAL_AREA / "env-freeze.txt").read_text()
        for pin in ("lerobot==0.7.0", "torch==2.15.0", "transformers==5.22.1", "hf-libero==0.1.4", "robosuite==1.4.0",
                    "mujoco==3.8.1"):
            self.assertIn(f"\n{pin}\n", freeze)

    def test_script_tables(self):
        """The scripts' argparse options, constants and evaluator facts (read with `ast`, never imported)."""
        self.need(REAL_AREA / "train_lerobot_logged.py", REAL_AREA / "eval_libero.py")
        from gatedtrain_recipe_lerobot_smolvla_libero import adapter as M
        self.assertEqual(L.argparse_options(REAL_AREA / "train_lerobot_logged.py"), M.TRAIN_OPTIONS)
        self.assertEqual(L.argparse_options(REAL_AREA / "eval_libero.py"), M.EVAL_OPTIONS)
        t = L.ast_constants(REAL_AREA / "train_lerobot_logged.py", ("CUT_SCHEMA", "CUTS_DIR", "CUT_MARKER", "LEROBOT_DIR",
                                                                     "PROFILES", "COMMON"))
        self.assertEqual((t["CUT_SCHEMA"], t["CUTS_DIR"], t["CUT_MARKER"], t["LEROBOT_DIR"]),
                         (M.CUT_SCHEMA, M.CUTS_DIR, M.CUT_MARKER, M.LEROBOT_DIR))
        self.assertEqual(sorted(t["PROFILES"]), ["base", "full", "probe", "smoke"])
        self.assertEqual({k: t["COMMON"][k] for k in ("batch_size", "train_expert_only", "freeze_vision_encoder",
                                                      "n_action_steps", "use_imagenet_stats", "mixed_precision")},
                         {k: v for k, v in M.DECIDED.items() if k != "device"})
        e = L.ast_constants(REAL_AREA / "eval_libero.py", ("EVALUATOR_ID", "ADAPTER", "METRIC", "ERROR_CLASSES",
                                                           "OUTCOMES", "DEFAULT_ITEM_TIMEOUT_S", "CUT_MARKER"))
        self.assertEqual((e["EVALUATOR_ID"], e["ADAPTER"], e["METRIC"]), (L.EVALUATOR_ID, L.ADAPTER, "success"))
        self.assertEqual(tuple(e["ERROR_CLASSES"]), registry().EVALUATOR_ROWS[L.EVALUATOR_ID]["error_classes"])
        self.assertEqual(tuple(e["OUTCOMES"]), M.OUTCOMES)
        self.assertEqual(e["DEFAULT_ITEM_TIMEOUT_S"], M.ITEM_TIMEOUT_S)
        c = L.ast_constants(REAL_AREA / "a8_common.py", ("MAX_STEPS", "RESUME_ONLY", "OBJECT_EPISODES", "RENAME",
                                                         "DATASET_REVISION"))
        self.assertEqual(c["MAX_STEPS"], M.MAX_STEPS)
        self.assertEqual(c["RESUME_ONLY"], ("train_config.json",))
        self.assertEqual((c["RENAME"], c["DATASET_REVISION"]), (M.RENAME, M.DATASET_REVISION))

    def test_credential_scan_of_the_runtime(self):
        """The read-only inputs pass the launcher's credential file-name scan with the row's skip directories."""
        self.need(REAL_AREA)
        from gatedtrain.ext.launch import credential_name_problem
        row = registry().TRAINER_ROWS[L.ADAPTER]
        self.assertIsNone(credential_name_problem(str(REAL_AREA), skip=tuple(row["scan_skip"]),
                                                  interpreter_trees=tuple(row["interpreter_trees"])))

    def test_fixture_copies(self):
        """The fixture files are byte copies of the real outputs they name (the cut's cut.json is not published: it
        records the hashes of unpublished originals)."""
        self.need(REAL_CUT / "sampler.json", REAL_EVAL / "summary.json")
        pairs = [(L.REAL_CUT / "sampler.json", REAL_CUT / "sampler.json"),
                 (L.REAL_CUT / L.NORMALIZER, REAL_CUT / L.NORMALIZER),
                 (L.REAL_EVAL / "summary.json", REAL_EVAL / "summary.json"),
                 (L.REAL_EVAL / "arm-success" / "eval_meta.json", REAL_EVAL / "arm-success" / "eval_meta.json")]
        for a, b in pairs:
            self.assertEqual(file_sha256(a), file_sha256(b), a.name)


if __name__ == "__main__":
    unittest.main()

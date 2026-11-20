"""The runtime scripts that run inside the GPU container (runtime/train_lerobot_logged.py, runtime/eval_libero.py,
runtime/a8_common.py), on CPU and without torch, LeRobot or LIBERO, in CI: the launcher's hooks with batches shaped
as LeRobot passes them, its cut writer, its settings and lerobot-train command line, a whole stage-2 launch through
main() against a stand-in `lerobot.scripts.lerobot_train`, and a whole evaluation through eval_libero.evaluate()
with a stand-in simulator; every output is checked with the adapter's collect rules.

Why: these scripts run only on a GPU host, so a fault in a part that needs no GPU (a batch column of another shape
than the hook expects, a cut file the adapter does not list, a log line the consumption check refuses, an argv the
adapter rebuilds differently) was found only after GPU time. runtime/tests/ holds deeper tests, but they need the
area venv and the downloaded models, so CI does not run them.

The scripts import only the standard library at module level. `torch`, `lerobot.scripts.lerobot_train` and LIBERO
are replaced inside the tests that call code needing them (sys.modules entries restored afterwards); numpy and
pyarrow come with gatedtrain. Nothing in runtime/ is changed: the scripts' SHA-256s are recorded in RESULT.md and
eval_libero.py with a8_common.py is the registered scorer's composite code hash.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import socket
import sys
import tempfile
import types
import unittest
import unittest.mock
from pathlib import Path

import _lerobot as L  # noqa: I001  (sets the runtime override before the recipe's rows are imported)

from gatedtrain.ext import arm as gt_arm
from gatedtrain.ext.sdk import compact_sha256

from gatedtrain_recipe_lerobot_smolvla_libero import adapter as A
from gatedtrain_recipe_lerobot_smolvla_libero import rows as R

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))
import a8_common as C  # noqa: E402
import eval_libero as E  # noqa: E402
import train_lerobot_logged as T  # noqa: E402

GATE = Path(__file__).resolve().parents[1] / "results" / "2026-11-16" / "gate"


def fake_torch() -> types.ModuleType:
    """What the launcher touches of torch on a CPU host: cuda.is_available() is False."""
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False, get_device_name=lambda *_: None)
    return torch


def lerobot_step_dir(root: Path, step: int, tag: str = "") -> Path:
    """LeRobot's checkpoints/<step>/ (common/train_utils.py:217-260): pretrained_model/ and training_state/ with the
    real small files of the published checkpoint-500 and stand-in weights, optimizer state and tokenizer.json."""
    d = root / f"{step:06d}"
    pm, ts = d / C.PRETRAINED, d / C.TRAINING_STATE
    (pm / "tokenizer").mkdir(parents=True)
    ts.mkdir(parents=True)
    src = L.REAL_CUT
    for rel in ("config.json", "policy_postprocessor.json", "policy_preprocessor.json",
                L.NORMALIZER.split("/", 1)[1], L.UNNORMALIZER.split("/", 1)[1], "tokenizer/chat_template.jinja",
                "tokenizer/tokenizer_config.json", "train_config.json"):
        shutil.copyfile(src / C.PRETRAINED / rel, pm / rel)
    (pm / "model.safetensors").write_bytes(L.safetensors_stub(("model.action_in_proj.bias",), f"{step} {tag}"))
    (pm / "tokenizer" / "tokenizer.json").write_text('{"fixture": "tokenizer stand-in"}\n', encoding="utf-8")
    for rel in ("optimizer_param_groups.json", "rng_state.safetensors", "scheduler_state.json"):
        shutil.copyfile(src / C.TRAINING_STATE / rel, ts / rel)
    (ts / "optimizer_state.safetensors").write_bytes(L.safetensors_stub(("state/0/step",), f"{step}"))
    (ts / "training_step.json").write_text(json.dumps({"step": step}), encoding="utf-8")
    return d


def base_ref(init_checkpoint=None) -> dict:
    """main()'s base_ref from the runtime's MANIFEST.json bindings."""
    b = T.base_binding()
    return {"base_model": C.binding_summary(b["base"]) | {"files": b["base"]["files"]},
            "vlm_backbone": C.binding_summary(b["vlm"]),
            "dataset": C.binding_summary(b["data"]) | {"episodes": [C.OBJECT_EPISODES[0], C.OBJECT_EPISODES[-1]]},
            "init_checkpoint": init_checkpoint}


def help_options(parse_args) -> set[str]:
    buf = io.StringIO()
    with contextlib.suppress(SystemExit), contextlib.redirect_stdout(buf):
        parse_args(["--help"])
    return set(re.findall(r"(--[a-z][a-z-]*)", buf.getvalue())) - {"--help"}


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="lerobot-runtime-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def isolate_process_state(self):
        """main() and evaluate() fix the environment and guard the socket; both are restored afterwards."""
        env = unittest.mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        connect = socket.socket.connect
        self.addCleanup(setattr, socket.socket, "connect", connect)
        blocked = list(C.BLOCKED)
        self.addCleanup(lambda: C.BLOCKED.__setitem__(slice(None), blocked))

    def state(self, **over) -> T.RunState:
        s = T.resolve_settings(T.parse_args(["--profile", "base", "--out", str(self.tmp / "out")]))
        s.update(memory_every=10, **over)
        out = self.tmp / "out"
        out.mkdir(exist_ok=True)
        return T.RunState(out, s, {"base_ref": base_ref(), "script_sha256": "sha256:" + "1" * 64,
                                   "common_sha256": C.sha256_file(C.COMMON_FILE)})


# ---------------------------------------------------------------------------
# The hooks, with batch columns shaped as LeRobot passes them
# ---------------------------------------------------------------------------

class TestPreprocessHook(Tmp):
    def batches(self):
        import numpy as np

        # index / episode_index / frame_index: (B,) tensors in LeRobot; numpy arrays have the same reshape/tolist;
        # also (B, 1) columns, plain lists and numpy integer scalars inside a list.
        yield {"index": np.array([10, 11]), "episode_index": np.array([807, 807]), "frame_index": np.array([0, 1])}
        yield {"index": np.array([[12], [13]]), "episode_index": np.array([[808], [808]]),
               "frame_index": np.array([[0], [1]])}
        yield {"index": [np.int64(14)], "episode_index": [809], "frame_index": [0]}
        yield {"index": [15, 16], "episode_index": [809, 810], "frame_index": [1, 0]}

    def test_one_line_per_step_before_the_original_runs(self):
        state = self.state()
        state.samples_per_epoch = 5                      # 5 frames: steps of 2, 2, 1 then the next epoch
        seen = []

        def orig(batch, camera_keys, rename_map, preprocessor):
            lines = (state.out / "samples.jsonl").read_text(encoding="utf-8").splitlines()
            seen.append((len(lines), camera_keys, rename_map, preprocessor))   # the line is on disk first
            return "processed"

        hook = T.make_preprocess_hook(state, orig)
        for b in self.batches():
            self.assertEqual(hook(b, ["cam"], {"a": "b"}, "pre"), "processed")
        self.assertEqual(seen, [(1, ["cam"], {"a": "b"}, "pre"), (2, ["cam"], {"a": "b"}, "pre"),
                                (3, ["cam"], {"a": "b"}, "pre"), (4, ["cam"], {"a": "b"}, "pre")])
        lines = [json.loads(x) for x in (state.out / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([(x["step"], x["epoch"], x["n"]) for x in lines], [(1, 0, 2), (2, 0, 2), (3, 0, 1), (4, 1, 2)])
        self.assertEqual(lines[1]["samples"], [{"index": 12, "episode_index": 808, "frame_index": 0},
                                               {"index": 13, "episode_index": 808, "frame_index": 1}])
        for x in lines:                                   # the adapter's line rule: these keys, plain integers
            self.assertEqual(sorted(x), ["epoch", "n", "samples", "step"])
            for sm in x["samples"]:
                self.assertEqual(sorted(sm), ["episode_index", "frame_index", "index"])
                self.assertTrue(all(type(v) is int for v in sm.values()))
        self.assertEqual(state.consumed_at, {0: 0, 1: 2, 2: 4, 3: 5, 4: 7})

    def test_columns_of_different_lengths_are_refused(self):
        hook = T.make_preprocess_hook(self.state(), lambda *a: None)
        with self.assertRaises(RuntimeError):
            hook({"index": [1, 2], "episode_index": [807], "frame_index": [0, 1]}, [], {}, None)

    def test_tokenizer_override_keeps_the_others(self):
        calls = []
        hook = T.make_processors_hook(lambda *a, **k: calls.append((a, k)) or "pp")
        self.assertEqual(hook("cfg", preprocessor_overrides={"device_processor": {"device": "cpu"},
                                                             "tokenizer_processor": {"max_length": 48}}), "pp")
        pre = calls[0][1]["preprocessor_overrides"]
        self.assertEqual(pre["device_processor"], {"device": "cpu"})
        self.assertEqual(pre["tokenizer_processor"], {"max_length": 48, "tokenizer_name": str(C.SMOLVLM)})
        hook("cfg")
        self.assertEqual(calls[1][1]["preprocessor_overrides"], {"tokenizer_processor": {"tokenizer_name": str(C.SMOLVLM)}})

    def test_update_hook_logs_memory_on_its_schedule(self):
        state = self.state(steps=45)
        metrics = types.SimpleNamespace(metrics={k: types.SimpleNamespace(val=v) for k, v in
                                                 (("loss", 0.5), ("grad_norm", 1.5), ("lr", 1e-4))})
        with unittest.mock.patch.dict(sys.modules, {"torch": fake_torch()}):
            hook = T.make_update_hook(state, lambda *a, **k: ("m", {}))
        for step in range(1, 46):
            state.step = step
            self.assertEqual(hook(metrics, None, {}, None, 1.0, None), ("m", {}))
        recs = [json.loads(x) for x in (state.out / "memory.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([r["step"] for r in recs], list(range(1, 21)) + [30, 40, 45])
        self.assertEqual((recs[0]["loss"], recs[0]["grad_norm"], recs[0]["lr"], recs[0]["step_s"]), (0.5, 1.5, 1e-4, None))


# ---------------------------------------------------------------------------
# Cuts (write_cut) as the adapter and eval_libero.py check them
# ---------------------------------------------------------------------------

class TestWriteCut(Tmp):
    def test_cut_is_what_the_adapter_and_the_evaluator_accept(self):
        state = self.state()
        state.consumed_at[250] = 8000
        step_dir = lerobot_step_dir(self.tmp / "out" / "lerobot" / "checkpoints", 250)
        cut = T.write_cut(step_dir, 250, state)
        self.assertEqual(cut, self.tmp / "out" / "cuts" / "checkpoint-250")
        ad = A.LerobotSmolvlaAdapter()
        self.assertIsNone(ad.cut_problem(cut, 250))
        self.assertEqual(ad.find_cuts(self.tmp / "out"), ([(250, cut)], []))
        doc = json.loads((cut / "cut.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["model_files"], sorted(R.TRAINER_ROWS["lerobot_smolvla"]["components"]))
        self.assertEqual(doc["normalizer_sha256"], ad.normalizer_sha256(cut))
        self.assertEqual(doc["samples_consumed"], 8000)
        self.assertEqual((cut / C.PRETRAINED / "model.safetensors").stat().st_ino,
                         (step_dir / C.PRETRAINED / "model.safetensors").stat().st_ino)      # hard links
        self.assertEqual(doc["file_origin"], ["hardlink"])
        marker, sampler = E.check_cut(cut)
        self.assertEqual(sampler, json.loads((L.REAL_CUT / "sampler.json").read_text(encoding="utf-8")))
        self.assertEqual(T.verify_cut(cut, full=True)["global_step"], 250)
        log = [json.loads(x) for x in (state.out / "cuts.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(log[0]["samples_consumed"], 8000)
        with self.assertRaises(RuntimeError):                 # never overwritten
            T.write_cut(step_dir, 250, state)

    def test_step_mismatch_and_links_are_refused(self):
        state = self.state()
        step_dir = lerobot_step_dir(self.tmp / "ck", 300)
        with self.assertRaises(RuntimeError):
            T.write_cut(step_dir, 301, state)             # training_step.json says 300
        os.symlink(step_dir / C.PRETRAINED / "config.json", step_dir / C.PRETRAINED / "link.json")
        with self.assertRaises(RuntimeError):
            T.write_cut(step_dir, 300, state)

    def test_deploy_sampler_is_the_registered_one(self):
        import yaml

        cfg = json.loads((L.REAL_CUT / C.PRETRAINED / "config.json").read_text(encoding="utf-8"))
        sampler = C.deploy_sampler(cfg)
        self.assertEqual(sampler, json.loads((L.REAL_CUT / "sampler.json").read_text(encoding="utf-8")))
        self.assertEqual(yaml.safe_load((GATE / "registration-gatedtrain.yaml").read_text(encoding="utf-8"))["sampler"],
                         sampler)
        self.assertEqual(sampler["n_action_steps"], T.COMMON["n_action_steps"])
        self.assertEqual(C.deploy_sampler({**cfg, "n_action_steps": 10})["n_action_steps"], 10)   # from the config

    def test_model_file_names_and_normalizer(self):
        files = list(L.CUT_FILES)
        self.assertEqual(C.model_file_names(files), sorted(A.COMPONENTS))
        self.assertEqual(sorted(set(files) - set(C.model_file_names(files)) - {"base_ref.json"}), sorted(A.RESUME_FILES))
        rec = C.normalizer_record(L.REAL_CUT / C.PRETRAINED)
        self.assertEqual({r: f["name"] for r, f in rec["files"].items()},
                         {"normalizer": L.NORMALIZER.split("/", 1)[1], "unnormalizer": L.UNNORMALIZER.split("/", 1)[1]})


# ---------------------------------------------------------------------------
# Settings and the lerobot-train command line
# ---------------------------------------------------------------------------

class TestSettings(unittest.TestCase):
    def settings(self, *argv):
        return T.resolve_settings(T.parse_args(["--out", "/x", *argv]))

    def test_profiles_and_the_save_rule(self):
        b = self.settings("--profile", "base")
        self.assertEqual((b["stage"], b["steps"], b["save_freq"], b["cut_steps"]),
                         ("base", 3000, 250, list(range(250, 3001, 250))))
        s = self.settings("--profile", "smoke")
        self.assertEqual((s["stage"], s["steps"], s["cut_steps"]), ("continuation", 250, [250]))
        f = self.settings("--profile", "full", "--steps", "2500")
        self.assertEqual(f["cut_steps"], [1000, 2000, 2500])               # every save_freq and the last step
        self.assertEqual(f["cut_steps"], A.cut_steps(f))
        self.assertEqual(self.settings("--profile", "probe", "--batch-size", "64")["cut_steps"], [])
        with self.assertRaises(SystemExit):
            self.settings("--profile", "full")
        for argv in (["--n-action-steps", "51"], ["--batch-size", "0"], ["--save-freq", "-1"]):
            with self.assertRaises(SystemExit):
                self.settings("--profile", "base", *argv)

    def test_decided_settings(self):
        s = self.settings("--profile", "full", "--steps", "6000")
        self.assertEqual({k: s[k] for k in A.DECIDED if k != "device"}, {k: v for k, v in A.DECIDED.items() if k != "device"})
        self.assertEqual(self.settings("--profile", "full", "--steps", "6000", "--device", "cuda")["device"],
                         A.DECIDED["device"])
        ft = self.settings("--profile", "base", "--full-ft")
        self.assertEqual((ft["train_expert_only"], ft["freeze_vision_encoder"], ft["full_ft"]), (False, False, True))

    def test_lerobot_argv_is_the_one_the_adapter_rebuilds(self):
        # The adapter rebuilds run_args.json lerobot_argv from the settings (/out view, its runtime path); the
        # launcher's must be the same line for the same settings.
        for argv in (["--profile", "base"], ["--profile", "probe"], ["--profile", "full", "--steps", "6000", "--lr", "5e-5",
                                             "--warmup-steps", "100", "--decay-steps", "6000", "--full-ft"]):
            s = self.settings(*argv)
            policy = "/inputs/base-cut/pretrained_model"
            got = T.build_lerobot_argv(s, Path(policy), Path("/out/lerobot"))
            want = [a.replace(str(A.AREA8), str(C.HERE)) for a in A.lerobot_argv(s, policy)]
            self.assertEqual(got, want, argv)

    def test_adapter_knows_every_option(self):
        self.assertEqual(help_options(T.parse_args), set(A.TRAIN_OPTIONS))
        self.assertEqual(help_options(E.parse_args), set(A.EVAL_OPTIONS))
        self.assertEqual(set(A.STAGE_PROFILES) | {"base", "probe"}, set(T.PROFILES))

    def test_policy_path_rules(self):
        with tempfile.TemporaryDirectory() as d:
            cut = Path(d) / "cut"
            (cut / C.PRETRAINED).mkdir(parents=True)
            args = T.parse_args(["--profile", "smoke", "--out", "/x", "--policy-path", str(cut)])
            with self.assertRaises(SystemExit):           # no cut.json yet
                T.resolve_policy_path(args, T.resolve_settings(args))
            (cut / "cut.json").write_text("{}", encoding="utf-8")
            for p in (cut, cut / C.PRETRAINED):
                args = T.parse_args(["--profile", "smoke", "--out", "/x", "--policy-path", str(p)])
                self.assertEqual(T.resolve_policy_path(args, T.resolve_settings(args)), (cut / C.PRETRAINED, cut))
            args = T.parse_args(["--profile", "base", "--out", "/x", "--policy-path", str(cut)])
            with self.assertRaises(SystemExit):           # stage 1 never starts from a cut
                T.resolve_policy_path(args, T.resolve_settings(args))


# ---------------------------------------------------------------------------
# A stage-2 launch through main(), against a stand-in lerobot_train
# ---------------------------------------------------------------------------

def fake_lerobot(frames: list[dict], step_root_files, skip_save: int | None = None) -> dict[str, types.ModuleType]:
    """`lerobot.scripts.lerobot_train` as far as the launcher uses it: main() reads sys.argv, builds the dataloaders,
    then per step preprocesses a batch (columns of shape (B,)), updates and, at the save rule's steps, writes
    checkpoints/<step>/ and calls update_last_checkpoint (lerobot_train.py:734-951)."""
    import numpy as np

    lt = types.ModuleType("lerobot.scripts.lerobot_train")
    lt.calls = {"processors": None, "last": []}

    def make_dataloaders(cfg, dataset, eval_dataset, step, parallel_dims):
        return types.SimpleNamespace(sampler=list(range(len(frames)))), None

    def make_pre_post_processors(*args, **kwargs):
        lt.calls["processors"] = kwargs.get("preprocessor_overrides")
        return None, None

    lt._preprocess_dataset_batch = lambda batch, camera_keys, rename_map, preprocessor: batch
    lt.update_policy = lambda train_metrics, policy, batch, optimizer, clip, acc, *a, **k: (train_metrics, {})
    lt.update_last_checkpoint = lambda d: lt.calls["last"].append(Path(d).name)
    lt.make_dataloaders = make_dataloaders
    lt.make_pre_post_processors = make_pre_post_processors

    def main():
        opts = dict(a[2:].split("=", 1) for a in sys.argv[1:])
        steps, freq, bs, seed = int(opts["steps"]), int(opts["save_freq"]), int(opts["batch_size"]), int(opts["seed"])
        out = Path(opts["output_dir"])
        assert not out.exists()                          # LeRobot refuses an existing output_dir (train.py:334)
        dataset = types.SimpleNamespace(num_frames=len(frames), num_episodes=len({f["episode_index"] for f in frames}))
        lt.make_dataloaders(None, dataset, None, 0, None)
        lt.make_pre_post_processors("policy cfg", preprocessor_overrides={"device_processor": {"device": "cpu"}})
        step, epoch = 0, 0
        while step < steps:
            order = np.random.default_rng(seed * 1000 + epoch).permutation(len(frames))
            for b0 in range(0, len(order), bs):
                if step >= steps:
                    break
                rows = [frames[int(i)] for i in order[b0:b0 + bs]]
                batch = {k: np.array([r[k] for r in rows], dtype=np.int64) for k in ("index", "episode_index",
                                                                                      "frame_index")}
                lt._preprocess_dataset_batch(batch, [], {}, None)
                step += 1
                lt.update_policy(types.SimpleNamespace(metrics={}), None, batch, None, 1.0, None)
                if ((freq > 0 and step % freq == 0) or step == steps) and step != skip_save:
                    d = step_root_files(out / "checkpoints", step)
                    lt.update_last_checkpoint(d)
            epoch += 1

    lt.main = main
    pkg, scripts = types.ModuleType("lerobot"), types.ModuleType("lerobot.scripts")
    pkg.scripts, scripts.lerobot_train = scripts, lt
    return {"lerobot": pkg, "lerobot.scripts": scripts, "lerobot.scripts.lerobot_train": lt, "torch": fake_torch()}


class TestMainStageTwo(Tmp):
    def test_launch_outputs_pass_the_adapters_collect_rules(self):
        self.isolate_process_state()
        frames = L.dataset_frames()                      # 908 LIBERO-Object frames of the miniature dataset
        # The stage-1 base cut this continuation starts from, written by the launcher's own write_cut.
        base_state = self.state(seed=1000)
        base_state.consumed_at[500] = 16000
        base_cut = T.write_cut(lerobot_step_dir(self.tmp / "base-run", 500), 500, base_state)
        out = self.tmp / "smoke-1"
        argv = ["--profile", "smoke", "--policy-path", str(base_cut), "--out", str(out), "--steps", "6",
                "--save-freq", "3", "--batch-size", "300", "--seed", "2000", "--device", "cpu"]
        mods = fake_lerobot(frames, lerobot_step_dir)
        version = T.importlib.metadata.version
        with unittest.mock.patch.dict(sys.modules, mods), \
                unittest.mock.patch.object(T.importlib.metadata, "version",
                                           lambda n: A.LEROBOT_VERSION if n == "lerobot" else version(n)), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(T.main(argv), 0)
        lt = mods["lerobot.scripts.lerobot_train"]
        self.assertEqual(lt.calls["last"], ["000003", "000006"])
        self.assertEqual(lt.calls["processors"]["tokenizer_processor"], {"tokenizer_name": str(C.SMOLVLM)})

        ad = A.LerobotSmolvlaAdapter()
        cuts, incomplete = ad.find_cuts(out)
        self.assertEqual(([s for s, _ in cuts], incomplete), ([3, 6], []))
        data = L.write_frame_export(L.write_dataset(self.tmp / "dataset"), self.tmp / "export", files_sha256="sha256:x")
        self.assertEqual(ad.consumption_problems(out, data, cuts), [])
        lines, _, bad = ad._ledger_plan(out, data, cuts)
        self.assertEqual(([x["epoch"] for x in lines], [x["n"] for x in lines], bad),
                         ([0, 0, 0, 0, 1, 1], [300, 300, 300, 8, 300, 300], []))   # 908 frames: the short last batch
        args = json.loads((out / "run_args.json").read_text(encoding="utf-8"))
        s = args["settings"]
        self.assertEqual(args["lerobot_argv"],
                         [a.replace(str(A.AREA8), str(C.HERE)).replace("/out/lerobot", str(out / "lerobot"))
                          for a in A.lerobot_argv(s, str(base_cut / C.PRETRAINED))])
        self.assertEqual(args["sampler"], json.loads((cuts[0][1] / "sampler.json").read_text(encoding="utf-8")))
        self.assertEqual(args["base_cut"]["cut_json_sha256"], C.sha256_file(base_cut / "cut.json"))
        for _, d in cuts:
            ref = json.loads((d / "base_ref.json").read_text(encoding="utf-8"))
            self.assertEqual(ref["init_checkpoint"], args["base_cut"])
            doc = json.loads((d / "cut.json").read_text(encoding="utf-8"))
            self.assertEqual((doc["seed"], doc["profile"], doc["stage"], doc["init_checkpoint"]),
                             (2000, "smoke", "continuation", args["base_cut"]))
            self.assertEqual(doc["script_sha256"], C.sha256_file(T.SCRIPT))
        rt = json.loads((out / "env-runtime.json").read_text(encoding="utf-8"))
        self.assertEqual(rt["code_sha256"], C.code_sha256(T.SCRIPT, C.COMMON_FILE))
        self.assertEqual(ad.script_problems(out, {"script_sha256": rt["script_sha256"]}), [])
        att = [json.loads(x) for x in (out / "attempts.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual((att[-1]["steps"], att[-1]["samples_consumed"], att[-1]["cuts"]), (6, 1508, [3, 6]))
        self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")                # set by C.fix_environment

    def launch(self, argv, mods):
        version = T.importlib.metadata.version
        with unittest.mock.patch.dict(sys.modules, mods), \
                unittest.mock.patch.object(T.importlib.metadata, "version",
                                           lambda n: A.LEROBOT_VERSION if n == "lerobot" else version(n)), \
                contextlib.redirect_stdout(io.StringIO()):
            return T.main(argv)

    def base_cut(self) -> Path:
        st = self.state(seed=1000)
        st.consumed_at[500] = 16000
        return T.write_cut(lerobot_step_dir(self.tmp / "base-run", 500), 500, st)

    def test_a_missing_cut_or_another_base_fails_the_launch(self):
        self.isolate_process_state()
        frames = L.dataset_frames()
        base_cut = self.base_cut()
        argv = ["--profile", "smoke", "--policy-path", str(base_cut), "--steps", "4", "--save-freq", "2",
                "--batch-size", "300", "--device", "cpu"]
        with self.assertRaises(SystemExit) as cm:         # LeRobot saved no checkpoint at step 4
            self.launch(argv + ["--out", str(self.tmp / "a")], fake_lerobot(frames, lerobot_step_dir, skip_save=4))
        self.assertIn("cuts written [2], expected [2, 4]", str(cm.exception))
        ref = json.loads((base_cut / "base_ref.json").read_text(encoding="utf-8"))
        ref["base_model"]["revision"] = "0" * 40
        C.write_json(base_cut / "base_ref.json", ref)       # not a model file: the cut still verifies
        with self.assertRaises(SystemExit) as cm:
            self.launch(argv + ["--out", str(self.tmp / "b")], fake_lerobot(frames, lerobot_step_dir))
        self.assertIn("names another smolvla_base", str(cm.exception))

    def test_dry_run_prints_the_argv_only(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(T.main(["--profile", "base", "--out", str(self.tmp / "o"), "--dry-run"]), 0)
        doc = json.loads(buf.getvalue())
        self.assertEqual(doc["lerobot_argv"][0], f"--policy.path={C.SMOLVLA}")
        self.assertFalse((self.tmp / "o").exists())


# ---------------------------------------------------------------------------
# eval_libero.py: items, episodes, records, and a whole evaluation with a stand-in simulator
# ---------------------------------------------------------------------------

class FakeEnv:
    """reset() -> observation; step(a) -> (observation, terminated, is_success) after `succeed_at` steps."""

    def __init__(self, succeed_at=None, terminate_at=None):
        self.succeed_at, self.terminate_at, self.n = succeed_at, terminate_at, 0

    def reset(self):
        self.n = 0
        return {"obs": 0}

    def step(self, a):
        self.n += 1
        return {"obs": self.n}, self.n == self.terminate_at, self.n == self.succeed_at


class TestEpisodes(unittest.TestCase):
    def test_outcomes(self):
        act = lambda obs, step: [0.1] * 7                                     # noqa: E731
        ep = E.run_episode(FakeEnv(succeed_at=7), act, 280, 50)
        self.assertEqual((ep["outcome"], ep["steps"], ep["steps_to_success"], ep["n_policy_calls"]), ("success", 7, 7, 1))
        ep = E.run_episode(FakeEnv(), act, 280, 50)
        self.assertEqual((ep["outcome"], ep["steps"], ep["steps_to_success"], ep["n_policy_calls"]),
                         ("timeout", 280, 280, 6))
        ep = E.run_episode(FakeEnv(terminate_at=3), act, 280, 50)
        self.assertEqual((ep["outcome"], ep["success"]), ("failed", False))
        ep = E.run_episode(FakeEnv(succeed_at=5, terminate_at=5), act, 280, 50)      # LIBERO ends a successful episode
        self.assertEqual((ep["outcome"], ep["success_step"]), ("success", 5))
        ep = E.run_episode(FakeEnv(), lambda o, s: [float("nan")] * 7, 280, 50)
        self.assertEqual((ep["outcome"], ep["steps"], ep["n_policy_calls"]), ("nan_action", 0, 1))
        ep = E.run_episode(FakeEnv(succeed_at=2), lambda o, s: [[1.0, 2.0]], 280, 50)        # (1, D) -> (D,)
        self.assertEqual(ep["actions"], [[1.0, 2.0], [1.0, 2.0]])

    def test_records_pass_the_adapters_line_rule(self):
        items = E.read_items(RUNTIME / "items-probe")
        job = E.make_jobs(items[:1], [11])[0]
        self.assertEqual((job["env_seed"], job["torch_seed"], job["noise_seed"]),
                         (C.derive_seed("libero-env", 11, job["item"]) % 2**31,
                          C.derive_seed("libero-torch", 11, job["item"]), C.derive_seed("libero-noise", 11, job["item"])))
        for ep in (E.run_episode(FakeEnv(succeed_at=60), lambda o, s: [0.0], 280, 50),
                   E.run_episode(FakeEnv(), lambda o, s: [0.0], 280, 50),
                   E.run_episode(FakeEnv(), lambda o, s: [float("inf")], 280, 50)):
            rec = E.score_record(job, ep, None, 1.234, 0)
            self.assertEqual(sorted(rec["sub_metrics"]), sorted(A.SUB_METRICS))
            self.assertEqual(rec["score"], int(ep["outcome"] == "success"))
        err = E.score_record(job, None, {"class": "simulator_error", "message": "x"}, 2.0, 0)
        self.assertEqual((err["score"], err["sub_metrics"], err["item_meta"]["outcome"]), (None, {"wall_s": 2.0}, None))
        self.assertEqual(E.action_summary([[0.5, -2.0], [1.5, 0.0]])["abs_gt_1"], 2)
        self.assertEqual(E.action_summary([]), {"n": 0})

    def test_item_sets_are_reproduced_from_the_manifest(self):
        manifest = json.loads((RUNTIME / "ITEMS-MANIFEST.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as d:
            doc = E.write_items(manifest["tasks"], Path(d))
            for name, (dirname, states) in E.ITEM_SETS.items():
                for f in ("items.jsonl", "item_set_meta.json"):
                    self.assertEqual((Path(d) / dirname / f).read_bytes(), (RUNTIME / dirname / f).read_bytes(),
                                     f"{dirname}/{f}")
            self.assertEqual(doc, manifest)
            (Path(d) / "items-gate" / "items.jsonl").write_text("changed\n", encoding="utf-8")
            with self.assertRaises(SystemExit):           # never regenerated in place
                E.write_items(manifest["tasks"], Path(d))
        gate = E.read_items(RUNTIME / "items-gate")
        self.assertEqual(len(gate), 400)
        self.assertEqual({r["init_state"] for r in gate}, set(C.GATE_STATES))
        self.assertTrue(all(A.ITEM_RE.fullmatch(r["item"]) for r in gate))
        sel = E.select_items(gate, 7, 2)
        self.assertEqual([(r["task_id"], r["init_state"]) for r in sel], [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1),
                                                                          (3, 0)])

    def test_parse_args_rules(self):
        for argv in (["--out", "x"], ["--checkpoint", "c", "--out", "x", "--eval-seeds", "11", "11"],
                     ["--checkpoint", "c", "--out", "x", "--limit", "0"], ["--checkpoint", "c", "--out", "x",
                                                                           "--envs-per-worker", "0"]):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                E.parse_args(argv)
        a = E.parse_args(["--checkpoint", "c", "--out", "x"])
        self.assertEqual((a.items, a.eval_seeds, a.item_timeout_s, a.max_steps, a.settle_steps),
                         ("probe", [11], A.ITEM_TIMEOUT_S, A.MAX_STEPS, C.SETTLE_STEPS))


class TestScorerPin(unittest.TestCase):
    def test_registered_scorer_is_this_code(self):
        self.assertEqual(C.code_sha256(E.SCRIPT, C.COMMON_FILE), R.EVAL_LIBERO_CODE_SHA256)


class TestEvaluate(Tmp):
    def test_evaluation_outputs_pass_the_arm_rules(self):
        self.isolate_process_state()
        state = self.state()
        state.consumed_at[500] = 16000
        cut = T.write_cut(lerobot_step_dir(self.tmp / "ck", 500), 500, state)
        manifest = json.loads((RUNTIME / "ITEMS-MANIFEST.json").read_text(encoding="utf-8"))

        def run_job(job):                                 # the worker, with a stand-in simulator and policy
            k = int(job["item"].rsplit("/s", 1)[1]) + job["task_id"] + job["repeat"]
            if k % 7 == 3:
                return E._error_result(job, "worker_died: stand-in", 1.0, 42)
            env = FakeEnv(succeed_at=40 + k if k % 3 == 0 else None, terminate_at=90 if k % 3 == 1 else None)
            act = (lambda o, s: [float("nan")] * 7) if k % 11 == 5 else (lambda o, s: [0.01 * k] * 7)
            ep = E.run_episode(env, act, 280, 50)
            return {"job": job, "episode": ep, "error": None, "wall": 0.5, "pid": 42, "device_memory": None}

        out = self.tmp / "eval"
        def run_pool(jobs, worker_cfg, workers, on_result, item_timeout=None, **_):
            self.assertEqual((workers, item_timeout, worker_cfg["pretrained"]), (6, A.ITEM_TIMEOUT_S,
                                                                                str(cut / C.PRETRAINED)))
            for j in jobs:                                # in process, in submission order
                on_result(run_job(j))

        args = E.parse_args(["--checkpoint", str(cut), "--out", str(out), "--items", "probe", "--per-task", "3",
                             "--eval-seeds", "11", "12", "--device", "cpu"])
        with unittest.mock.patch.object(E, "suite_tasks", lambda: manifest["tasks"]), \
                unittest.mock.patch.object(E, "run_pool", run_pool), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(E.evaluate(args), 0)
        arm = out / "arm-success"
        row = R.EVALUATOR_ROWS[E.EVALUATOR_ID]
        # As the run leaves it: the model hash is for gatedtrain to fill (ext-arm-fill), and no LeRobot is installed
        # here, so the framework version is null; filled as a launch would have them, the arm must pass MM-2.
        meta, bad = gt_arm.read_meta(arm)
        self.assertEqual((meta["checkpoint"], meta["framework"]),
                         ({"model_hash": None, "base_model_hash": None}, {"name": "lerobot", "version": None}))
        meta["checkpoint"]["model_hash"] = A.LerobotSmolvlaAdapter().identity(cut)["model_hash"]
        meta["framework"] = {"name": A.FRAMEWORK_PACKAGE, "version": A.LEROBOT_VERSION}
        C.write_json(arm / "eval_meta.json", meta)
        meta, bad = gt_arm.read_meta(arm)
        self.assertEqual(bad, [])
        records, bad = gt_arm.read_lines(arm, seeds=meta["eval_seeds"], score_range=row["score_range"],
                                         error_classes=row["error_classes"])
        self.assertEqual(bad, [])
        self.assertEqual(len(records), 2 * 30)
        self.assertEqual({r["item_meta"]["outcome"] for r in records if r["error"] is None},
                         {"success", "timeout", "failed", "nan_action"})
        self.assertTrue(any(r["error"] for r in records))
        self.assertEqual(meta["sampler_hash"], compact_sha256(meta["sampler"]))
        self.assertEqual((meta["evaluator_id"], meta["metric"], meta["direction"], meta["score_range"]),
                         (E.EVALUATOR_ID, row["metric"], row["direction"], list(row["score_range"])))
        code = C.code_sha256(E.SCRIPT, C.COMMON_FILE)              # its registration is TestScorerPin's
        self.assertEqual(meta["scorer"]["code_sha256"], code)
        # The adapter's arm rules (EXT-241) on this evaluation; only the container path differs from a real launch
        # (the cut is not mounted at /inputs/cut here). The scorer code is set to the registered pin, so that this
        # test checks the layout and TestScorerPin the code.
        rt = json.loads((out / "env-runtime.json").read_text(encoding="utf-8"))
        self.assertEqual(rt["code_sha256"], code)
        rt.update(checkpoint=A.EVAL_CUT_CONTAINER, code_sha256=row["scorer"]["code_sha256"],
                  script_sha256=L.EVAL_LIBERO_SHA, common_sha256=L.COMMON_SHA)
        C.write_json(out / "env-runtime.json", rt)
        items = E.read_items(RUNTIME / "items-probe")
        self.assertEqual(A.LerobotSmolvlaAdapter().arm_problems(arm, records, items, RUNTIME / "items-probe", cut), [])
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["error_reasons"]["worker_died"], sum(1 for r in records if r["error"]))
        recorded = json.loads((L.REAL_EVAL / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(set(summary) - {"blocked_connections"}, set(recorded) - {"blocked_connections"})


if __name__ == "__main__":
    unittest.main()

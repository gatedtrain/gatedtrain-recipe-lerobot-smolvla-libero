"""End-to-end CPU smoke (slow; runs only with GATEDTRAIN_A8_SLOW=1): the real launcher runs `lerobot-train` for 2 steps
at batch 2 from smolvla_base (one cut), then a stage-2 continuation of 1 step from that cut; the real evaluator runs
1 probe item for 3 steps with 2 settle steps on the CPU (Mesa EGL), in process (--workers 0). It checks the run
record, the consumption log, the cuts, and the MM-1/MM-2 arm. Takes several minutes and writes about 4 GB below
$GATEDTRAIN_A8_TEST_TMP (removed afterwards unless GATEDTRAIN_A8_KEEP=1)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))          # -I: the script directory is not on sys.path
import _support as S  # noqa: E402

import a8_common as C  # noqa: E402

PY = [sys.executable, "-I", "-B"]


def run(args, log: Path):
    env = {k: v for k, v in os.environ.items() if k not in C.TOKEN_VARS}
    with open(log, "w") as f:
        proc = subprocess.run(PY + args, cwd=S.AREA, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=3600)
    if proc.returncode != 0:
        raise AssertionError(f"{args[0]} exited {proc.returncode}; tail of {log}:\n" +
                             "\n".join(log.read_text().splitlines()[-40:]))


@unittest.skipUnless(S.SLOW, "set GATEDTRAIN_A8_SLOW=1 to run the end-to-end CPU smoke")
class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = S.tmpdir("a8-e2e-")

    @classmethod
    def tearDownClass(cls):
        if os.environ.get("GATEDTRAIN_A8_KEEP") != "1":
            S.rmtree(cls.root)

    def test_train_continue_eval(self):
        base_out = self.root / "base"
        run(["train_lerobot_logged.py", "--profile", "base", "--out", str(base_out), "--steps", "2",
             "--save-freq", "2", "--batch-size", "2", "--num-workers", "0", "--device", "cpu",
             "--n-action-steps", "10"], self.root / "base.log")
        ra = json.loads((base_out / "run_args.json").read_text())
        self.assertEqual((ra["stage"], ra["cut_steps"], ra["base_cut"]), ("base", [2], None))
        lines = [json.loads(x) for x in (base_out / "samples.jsonl").read_text().splitlines()]
        self.assertEqual([ln["step"] for ln in lines], [1, 2])
        self.assertTrue(all(ln["n"] == 2 and 807 <= s["episode_index"] <= 1260
                            for ln in lines for s in ln["samples"]))
        mem = [json.loads(x) for x in (base_out / "memory.jsonl").read_text().splitlines()]
        self.assertEqual([m["step"] for m in mem], [1, 2])
        self.assertTrue(all(isinstance(m.get("loss"), float) for m in mem))
        cut = base_out / "cuts" / "checkpoint-2"
        m = json.loads((cut / "cut.json").read_text())
        self.assertEqual((m["global_step"], m["samples_consumed"]), (2, 4))
        for rel in m["model_files"]:
            self.assertEqual(C.sha256_file(cut / rel), m["files"][rel])
        cfg = json.loads((cut / "pretrained_model" / "config.json").read_text())
        self.assertEqual((cfg["vlm_model_name"], cfg["load_vlm_weights"], cfg["n_action_steps"]),
                         (str(C.SMOLVLM), False, 10))
        self.assertEqual(sorted(cfg["input_features"]), ["observation.images.camera1", "observation.images.camera2",
                                                         "observation.state"])
        self.assertEqual(cfg["output_features"]["action"]["shape"], [7])
        pre = json.loads((cut / "pretrained_model" / "policy_preprocessor.json").read_text())
        tok = [s for s in pre["steps"] if s["registry_name"] == "tokenizer_processor"][0]
        self.assertEqual(tok["config"]["tokenizer_name"], "tokenizer")      # saved as a cut-relative artifact
        att = [json.loads(x) for x in (base_out / "attempts.jsonl").read_text().splitlines()]
        self.assertEqual(att[-1]["blocked_connections"], [])
        self.assertEqual(att[-1]["cuts"], [2])

        cont_out = self.root / "cont"
        run(["train_lerobot_logged.py", "--profile", "smoke", "--policy-path", str(cut), "--out", str(cont_out),
             "--steps", "1", "--batch-size", "2", "--num-workers", "0", "--device", "cpu", "--n-action-steps", "10"],
            self.root / "cont.log")
        ra2 = json.loads((cont_out / "run_args.json").read_text())
        self.assertEqual(ra2["base_cut"]["global_step"], 2)
        self.assertEqual(ra2["base_cut"]["model_files_sha256"], m["model_files_sha256"])
        m2 = json.loads((cont_out / "cuts" / "checkpoint-1" / "cut.json").read_text())
        self.assertEqual(m2["init_checkpoint"]["cut_json_sha256"], C.sha256_file(cut / "cut.json"))
        self.assertEqual(m2["normalizer_sha256"], m["normalizer_sha256"])    # same data, same stats

        ev_out = self.root / "eval"
        run(["eval_libero.py", "--checkpoint", str(cut), "--items", "probe", "--limit", "1", "--out", str(ev_out),
             "--eval-seeds", "11", "--workers", "0", "--device", "cpu", "--max-steps", "3", "--settle-steps", "2"],
            self.root / "eval.log")
        arm = ev_out / "arm-success"
        meta = json.loads((arm / "eval_meta.json").read_text())
        self.assertEqual((meta["evaluator_id"], meta["metric"], meta["score_range"], meta["eval_seeds"]),
                         ("libero.lerobot_items@0.1", "success", [0, 1], [11]))
        self.assertEqual(meta["normalizer_sha256"], m["normalizer_sha256"])
        self.assertEqual(meta["sampler_hash"], m["sampler_sha256"])
        rows = [json.loads(x) for x in (arm / "items.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual((r["item"], r["repeat"], r["seed"]), ("libero_object/t0/s40", 0, 11))
        self.assertIsNone(r["error"], r)
        self.assertIn(r["score"], (0, 1))
        self.assertEqual(r["sub_metrics"]["steps"], 3 if r["score"] == 0 else r["sub_metrics"]["steps"])
        ep = json.loads((arm / r["output_ref"]["path"]).read_text())
        self.assertEqual(C.sha256_file(arm / r["output_ref"]["path"]), r["output_ref"]["sha256"])
        self.assertEqual(len(ep["actions"]), ep["steps"])
        self.assertEqual(len(ep["actions"][0]), 7)
        summary = json.loads((ev_out / "summary.json").read_text())
        self.assertEqual(summary["env_version_nondefault"], {"max_steps": 3, "num_steps_wait": 2})
        self.assertEqual(summary["blocked_connections"], [])


if __name__ == "__main__":
    unittest.main()

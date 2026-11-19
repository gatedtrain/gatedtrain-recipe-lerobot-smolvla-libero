"""The launcher's wrappers (consumption log, tokenizer override, memory log) and the cut writer on a small fake
checkpoint step directory; the evaluator's cut check on the result."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))          # -I: the script directory is not on sys.path
import _support as S  # noqa: E402

import a8_common as C  # noqa: E402
import eval_libero as E  # noqa: E402
import train_lerobot_logged as T  # noqa: E402


def state_for(out: Path, steps=4, batch=3, profile="base"):
    args = T.parse_args(["--profile", profile, "--out", str(out), "--steps", str(steps), "--save-freq", "2",
                         "--batch-size", str(batch), "--n-action-steps", "10"])
    s = T.resolve_settings(args)
    s["memory_every"] = 10
    base_ref = {"base_model": {"repo": "lerobot/smolvla_base", "revision": "r" * 40, "files_sha256": "sha256:x"},
                "init_checkpoint": None}
    return T.RunState(out, s, {"base_ref": base_ref, "script_sha256": "sha256:s", "common_sha256": "sha256:c"})


class ConsumptionHook(unittest.TestCase):
    def setUp(self):
        self.out = S.tmpdir("a8-hooks-")

    def tearDown(self):
        S.rmtree(self.out)

    def test_lines_epochs_and_order(self):
        import torch

        st = state_for(self.out)
        st.samples_per_epoch = 5            # batches of 3: epoch 0 = steps 1-2 (3 + 2 short), epoch 1 = steps 3-4
        calls = []

        def orig(batch, camera_keys, rename_map, pre):
            calls.append(len((self.out / "samples.jsonl").read_text().splitlines()))   # line already on disk
            return {"done": True}

        hook = T.make_preprocess_hook(st, orig)
        for idx in ([10, 11, 12], [13, 14], [20, 21, 22], [23, 24]):
            n = len(idx)
            batch = {"index": torch.tensor(idx), "episode_index": torch.full((n,), 807),
                     "frame_index": torch.tensor([i - 10 for i in idx]),
                     "observation.images.image": torch.zeros(n, 3, 2, 2)}
            self.assertEqual(hook(batch, ["observation.images.image"], C.RENAME, None), {"done": True})
        lines = [json.loads(x) for x in (self.out / "samples.jsonl").read_text().splitlines()]
        self.assertEqual(calls, [1, 2, 3, 4])                 # written before the original (and the update) ran
        self.assertEqual([ln["step"] for ln in lines], [1, 2, 3, 4])
        self.assertEqual([ln["epoch"] for ln in lines], [0, 0, 1, 1])
        self.assertEqual([ln["n"] for ln in lines], [3, 2, 3, 2])
        self.assertEqual(lines[0]["samples"][1], {"index": 11, "episode_index": 807, "frame_index": 1})
        self.assertEqual(st.consumed_at, {0: 0, 1: 3, 2: 5, 3: 8, 4: 10})


class ProcessorAndUpdateHooks(unittest.TestCase):
    def test_tokenizer_override_added(self):
        seen = {}

        def orig(*a, **kw):
            seen.update(kw)
            return "pre", "post"

        hook = T.make_processors_hook(orig)
        hook(policy_cfg=None, pretrained_path="x",
             preprocessor_overrides={"device_processor": {"device": "cuda"}})
        self.assertEqual(seen["preprocessor_overrides"]["tokenizer_processor"], {"tokenizer_name": str(C.SMOLVLM)})
        self.assertEqual(seen["preprocessor_overrides"]["device_processor"], {"device": "cuda"})

    def test_memory_log(self):
        out = S.tmpdir("a8-mem-")
        try:
            st = state_for(out)

            class Meter:
                def __init__(self, v):
                    self.val = v

            class Tracker:
                metrics = {"loss": Meter(1.5), "grad_norm": Meter(2.0), "lr": Meter(1e-4)}

            hook = T.make_update_hook(st, lambda tm, *a, **k: (tm, None))
            for step in (1, 2):
                st.step = step
                hook(Tracker(), None, None, None, 10.0, None)
            recs = [json.loads(x) for x in (out / "memory.jsonl").read_text().splitlines()]
            self.assertEqual([r["step"] for r in recs], [1, 2])
            self.assertEqual((recs[0]["loss"], recs[0]["grad_norm"]), (1.5, 2.0))
            self.assertIsNone(recs[0]["step_s"])
            self.assertIsNotNone(recs[1]["step_s"])
        finally:
            S.rmtree(out)


class CutWriter(unittest.TestCase):
    def setUp(self):
        self.out = S.tmpdir("a8-cut-")
        self.ckpts = self.out / "lerobot" / "checkpoints"
        self.ckpts.mkdir(parents=True)

    def tearDown(self):
        S.rmtree(self.out)

    def write(self, step=4):
        st = state_for(self.out)
        st.consumed_at = {0: 0, 1: 3, 2: 6, 3: 9, 4: 12}
        step_dir = S.make_fake_step_dir(self.ckpts, step)
        hook = T.make_last_checkpoint_hook(st, lambda d: (d.parent / "last").symlink_to(d.name))
        hook(step_dir)
        return st, self.out / "cuts" / f"checkpoint-{step}"

    def test_cut_layout_and_marker(self):
        st, cut = self.write()
        self.assertEqual(st.cuts, [4])
        self.assertTrue((self.ckpts / "last").is_symlink())
        m = json.loads((cut / "cut.json").read_text())
        self.assertEqual((m["schema"], m["global_step"], m["last_step_included"], m["samples_consumed"]),
                         ("ptc_mm.lerobot_cut", 4, 4, 12))
        self.assertIn("pretrained_model/model.safetensors", m["model_files"])
        self.assertIn("pretrained_model/tokenizer/tokenizer.json", m["model_files"])
        self.assertIn("sampler.json", m["model_files"])
        self.assertNotIn("pretrained_model/train_config.json", m["model_files"])
        self.assertIn("training_state/optimizer_state.safetensors", m["resume_files"])
        self.assertIn("pretrained_model/train_config.json", m["resume_files"])
        for rel, sha in m["files"].items():
            self.assertEqual(C.sha256_file(cut / rel), sha)
        self.assertEqual(set(m["files"]), set(C.tree_files(cut)) - {"cut.json"})
        # hard links to LeRobot's step directory (same file system)
        self.assertEqual(os.stat(cut / "pretrained_model/model.safetensors").st_ino,
                         os.stat(self.ckpts / "000004/pretrained_model/model.safetensors").st_ino)
        # cut.json is the newest file of the cut (written last)
        newest = max((cut / rel).stat().st_mtime_ns for rel in m["files"])
        self.assertGreaterEqual((cut / "cut.json").stat().st_mtime_ns, newest)
        sam = json.loads((cut / "sampler.json").read_text())
        self.assertEqual(sam["n_action_steps"], 10)
        self.assertEqual(m["sampler_sha256"], C.doc_sha256(sam))
        norm = C.normalizer_record(cut / "pretrained_model")
        self.assertEqual(m["normalizer_sha256"], norm["normalizer_sha256"])
        self.assertEqual(norm["files"]["normalizer"]["name"],
                         "policy_preprocessor_step_5_normalizer_processor.safetensors")
        self.assertFalse(list((self.out / "cuts").glob(".*partial")))
        line = json.loads((self.out / "cuts.jsonl").read_text())
        self.assertEqual((line["step"], line["cut"]), (4, "cuts/checkpoint-4"))
        # the launcher's own check of a base cut (stage 2) and the evaluator's
        self.assertEqual(T.verify_cut(cut, full=True)["global_step"], 4)
        marker, sampler = E.check_cut(cut)
        self.assertEqual(sampler, sam)

    def test_evaluator_refuses_tampered_cut(self):
        _, cut = self.write()
        p = cut / "pretrained_model" / "model.safetensors"
        p.unlink()                                  # break the hard link, then change the content
        p.write_bytes(b"other weights")
        with self.assertRaises(SystemExit):
            E.check_cut(cut)

    def test_evaluator_refuses_other_sampler(self):
        _, cut = self.write()
        sam = json.loads((cut / "sampler.json").read_text())
        sam["n_action_steps"] = 50
        (cut / "sampler.json").unlink()
        (cut / "sampler.json").write_text(json.dumps(sam))
        with self.assertRaises(SystemExit):        # hash mismatch first; a re-hashed one would differ from config
            E.check_cut(cut)

    def test_step_mismatch_refused(self):
        st = state_for(self.out)
        step_dir = S.make_fake_step_dir(self.ckpts, 6)
        (step_dir / "training_state" / "training_step.json").write_text('{"step": 5}')
        with self.assertRaises(RuntimeError):
            T.write_cut(step_dir, 6, st)


if __name__ == "__main__":
    unittest.main()

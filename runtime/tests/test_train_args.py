"""Settings and the lerobot-train command line of train_lerobot_logged.py, parsed by LeRobot's own parser and
validated by TrainPipelineConfig.validate() (configs/train.py), on the CPU without loading a model."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))          # -I: the script directory is not on sys.path
import _support as S  # noqa: E402  (sets the environment and sys.path first)

import a8_common as C
import train_lerobot_logged as T


def settings(*argv):
    args = T.parse_args(["--out", "/nonexistent-out", *argv])
    return args, T.resolve_settings(args)


class Profiles(unittest.TestCase):
    def test_base_profile(self):
        args, s = settings("--profile", "base")
        self.assertEqual((s["steps"], s["save_freq"], s["stage"]), (3000, 250, "base"))
        self.assertEqual(s["cut_steps"], list(range(250, 3001, 250)))
        self.assertTrue(s["train_expert_only"] and s["freeze_vision_encoder"])
        policy_dir, cut = T.resolve_policy_path(args, s)
        self.assertEqual((policy_dir, cut), (C.SMOLVLA, None))

    def test_probe_profile_saves_nothing(self):
        _, s = settings("--profile", "probe", "--batch-size", "64", "--full-ft")
        self.assertEqual((s["steps"], s["save_checkpoint"], s["cut_steps"]), (200, False, []))
        self.assertFalse(s["train_expert_only"] or s["freeze_vision_encoder"])
        self.assertEqual(s["batch_size"], 64)

    def test_full_needs_steps_and_cut(self):
        with self.assertRaises(SystemExit):
            settings("--profile", "full")
        args, s = settings("--profile", "full", "--steps", "6000")
        self.assertEqual(s["cut_steps"], list(range(1000, 6001, 1000)))
        with self.assertRaises(SystemExit):           # a continuation must start from a cut
            T.resolve_policy_path(args, s)

    def test_last_step_always_cut(self):
        _, s = settings("--profile", "base", "--steps", "600", "--save-freq", "250")
        self.assertEqual(s["cut_steps"], [250, 500, 600])

    def test_continuation_from_cut_dir(self):
        d = S.tmpdir("a8-args-")
        try:
            (d / "cut.json").write_text("{}")
            (d / "pretrained_model").mkdir()
            args, s = settings("--profile", "smoke", "--policy-path", str(d))
            self.assertEqual(T.resolve_policy_path(args, s), (d.resolve() / "pretrained_model", d.resolve()))
            args, s = settings("--profile", "base", "--policy-path", str(d))
            with self.assertRaises(SystemExit):       # stage 1 never starts from a cut
                T.resolve_policy_path(args, s)
        finally:
            S.rmtree(d)

    def test_bad_values(self):
        for argv in (("--n-action-steps", "51"), ("--batch-size", "0"), ("--save-freq", "-1")):
            with self.assertRaises(SystemExit):
                settings("--profile", "base", *argv)


class LeRobotParse(unittest.TestCase):
    """The argv as `lerobot-train` parses it (configs/parser.py wrap(): path args filtered, draccus.parse,
    then TrainPipelineConfig.validate(), which reads --policy.path from sys.argv)."""

    def parse(self, s, policy_dir, out):
        import draccus
        import lerobot.scripts.lerobot_train  # noqa: F401  (registers the policy and env config types, as at run time)
        from lerobot.configs import parser
        from lerobot.configs.train import TrainPipelineConfig

        argv = T.build_lerobot_argv(s, policy_dir, out)
        old = sys.argv
        sys.argv = ["lerobot-train"] + argv
        try:
            cli = parser.filter_path_args(TrainPipelineConfig.__get_path_fields__(), argv)
            cfg = draccus.parse(config_class=TrainPipelineConfig, args=cli)
            cfg.validate()
        finally:
            sys.argv = old
        return cfg

    def test_parse_and_validate(self):
        out = S.tmpdir("a8-parse-") / "lerobot"           # must not exist (configs/train.py:334-338)
        _, s = settings("--profile", "base", "--device", "cpu", "--lr", "5e-5", "--n-action-steps", "10")
        cfg = self.parse(s, C.SMOLVLA, out)
        self.assertEqual(cfg.dataset.repo_id, "lerobot/libero")
        self.assertEqual(cfg.dataset.episodes, C.OBJECT_EPISODES)
        self.assertEqual(cfg.dataset.revision, C.DATASET_REVISION)
        self.assertEqual(cfg.dataset.video_backend, "pyav")
        self.assertFalse(cfg.dataset.use_imagenet_stats)
        self.assertEqual(cfg.rename_map, C.RENAME)
        self.assertEqual(Path(cfg.output_dir), out)
        self.assertEqual((cfg.steps, cfg.save_freq, cfg.batch_size, cfg.seed), (3000, 250, 32, 1000))
        self.assertEqual((cfg.env_eval_freq, cfg.eval_steps), (0, 0))
        self.assertFalse(cfg.trackio.enable)
        p = cfg.policy
        self.assertEqual(p.type, "smolvla")
        self.assertEqual(Path(p.pretrained_path), C.SMOLVLA)
        self.assertEqual(p.vlm_model_name, str(C.SMOLVLM))
        self.assertFalse(p.load_vlm_weights)
        self.assertFalse(p.push_to_hub)
        self.assertFalse(p.input_features)               # null -> refilled from the dataset by make_policy
        self.assertFalse(p.output_features)
        self.assertEqual((p.n_action_steps, p.device), (10, "cpu"))
        self.assertTrue(p.train_expert_only and p.freeze_vision_encoder)
        self.assertEqual(cfg.optimizer.lr, 5e-5)          # the preset follows policy.optimizer_lr
        self.assertEqual(cfg.scheduler.peak_lr, 5e-5)

    def test_full_ft_flags(self):
        out = S.tmpdir("a8-parse-") / "lerobot"
        _, s = settings("--profile", "probe", "--device", "cpu", "--full-ft")
        cfg = self.parse(s, C.SMOLVLA, out)
        self.assertFalse(cfg.policy.train_expert_only or cfg.policy.freeze_vision_encoder)
        self.assertFalse(cfg.save_checkpoint)


class DeploySampler(unittest.TestCase):
    def test_sampler_from_config(self):
        cfg = json.loads((C.SMOLVLA / "config.json").read_text())
        sam = C.deploy_sampler({**cfg, "n_action_steps": 10})
        self.assertEqual((sam["n_action_steps"], sam["chunk_size"], sam["num_steps"]), (10, 50, 10))
        self.assertEqual(sam["rename_map"], C.RENAME)
        self.assertEqual((sam["observation"]["height"], sam["control_mode"], sam["hard_reset"]), (360, "relative", True))
        self.assertEqual(C.doc_sha256(sam), C.doc_sha256(json.loads(json.dumps(sam))))


if __name__ == "__main__":
    unittest.main()

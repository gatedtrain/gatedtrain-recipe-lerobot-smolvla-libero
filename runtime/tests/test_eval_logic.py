"""eval_libero.py without a simulator or a model: item sets, episode loop and scoring with a fake env and policy,
job seeds (common random numbers), and the real LIBERO-Object task list (CPU, LIBERO import only)."""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))          # -I: the script directory is not on sys.path
import _support as S  # noqa: E402

import a8_common as C  # noqa: E402
import eval_libero as E  # noqa: E402

FAKE_TASKS = [{"task_id": t, "task": f"task_{t}", "language": f"pick up thing {t}", "init_file": f"/x/t{t}.pruned_init",
               "init_file_name": f"t{t}.pruned_init", "init_file_sha256": f"sha256:{t:064d}"} for t in range(10)]


class FakeEnv:
    """reset() -> observation; step(a) -> (observation, terminated, is_success)."""

    def __init__(self, success_at=None, done_at=None, raise_at=None):
        self.success_at, self.done_at, self.raise_at = success_at, done_at, raise_at
        self.t = 0
        self.actions = []

    def reset(self):
        self.t = 0
        return {"obs": 0}

    def step(self, a):
        self.t += 1
        self.actions.append(a)
        if self.raise_at == self.t:
            raise RuntimeError("mujoco instability")
        return {"obs": self.t}, self.done_at == self.t, self.success_at == self.t


def const_act(value=0.1):
    return lambda obs, step: [value] * 7


class Items(unittest.TestCase):
    def test_write_items(self):
        root = S.tmpdir("a8-items-")
        try:
            man = E.write_items(FAKE_TASKS, root)
            gate = E.read_items(root / "items-gate")
            probe = E.read_items(root / "items-probe")
            self.assertEqual((len(gate), len(probe)), (400, 100))
            self.assertEqual(gate[0]["item"], "libero_object/t0/s0")
            self.assertEqual(gate[39]["item"], "libero_object/t0/s39")
            self.assertEqual(gate[40]["item"], "libero_object/t1/s0")      # numeric order, task then state
            self.assertEqual(probe[0]["item"], "libero_object/t0/s40")
            self.assertEqual(probe[-1]["item"], "libero_object/t9/s49")
            self.assertFalse({r["item"] for r in gate} & {r["item"] for r in probe})
            r = probe[0]
            self.assertEqual((r["suite"], r["task_id"], r["init_state"], r["row_index"], r["init_file_sha256"]),
                             ("libero_object", 0, 40, 40, FAKE_TASKS[0]["init_file_sha256"]))
            self.assertEqual(man["sets"]["gate"]["n_items"], 400)
            first = (root / "items-gate" / "items.jsonl").read_bytes()
            E.write_items(FAKE_TASKS, root)                                   # deterministic: same bytes
            self.assertEqual(first, (root / "items-gate" / "items.jsonl").read_bytes())
            changed = [dict(t, language="other") for t in FAKE_TASKS]
            with self.assertRaises(SystemExit):                                # never regenerated in place
                E.write_items(changed, root)
            sel = E.select_items(probe, None, 5)
            self.assertEqual(len(sel), 50)
            self.assertEqual({r["task_id"] for r in sel}, set(range(10)))
            self.assertEqual([r["init_state"] for r in sel[:5]], [40, 41, 42, 43, 44])
            self.assertEqual(len(E.select_items(probe, 7, None)), 7)
        finally:
            S.rmtree(root)


class Episode(unittest.TestCase):
    def test_success(self):
        ep = E.run_episode(FakeEnv(success_at=60), const_act(), 280, 50)
        self.assertEqual((ep["outcome"], ep["steps"], ep["steps_to_success"], ep["n_policy_calls"]),
                         ("success", 60, 60, 2))
        self.assertEqual(len(ep["actions"]), 60)

    def test_timeout(self):
        env = FakeEnv()
        ep = E.run_episode(env, const_act(), 280, 50)
        self.assertEqual((ep["outcome"], ep["steps"], ep["steps_to_success"], ep["n_policy_calls"]),
                         ("timeout", 280, 280, 6))
        self.assertEqual(env.t, 280)

    def test_failed_and_nan(self):
        ep = E.run_episode(FakeEnv(done_at=5), const_act(), 280, 10)
        self.assertEqual((ep["outcome"], ep["steps"], ep["steps_to_success"]), ("failed", 5, 280))
        env = FakeEnv()
        ep = E.run_episode(env, lambda o, s: [math.nan if s == 3 else 0.0] * 7, 280, 10)
        self.assertEqual((ep["outcome"], ep["steps"]), ("nan_action", 3))
        self.assertEqual(env.t, 3)                           # the non-finite action never reached the simulator

    def test_exception_propagates(self):
        with self.assertRaises(RuntimeError):                # run_job turns it into simulator_error
            E.run_episode(FakeEnv(raise_at=2), const_act(), 280, 10)

    def test_score_records(self):
        job = {"item": "libero_object/t3/s41", "task_id": 3, "init_state": 41, "eval_seed": 12}
        ok = E.score_record(job, E.run_episode(FakeEnv(success_at=7), const_act(), 280, 50), None, 1.25, 1)
        self.assertEqual((ok["score"], ok["error"], ok["repeat"], ok["seed"]), (1, None, 1, 12))
        self.assertEqual(ok["sub_metrics"]["steps_to_success"], 7)
        self.assertEqual(ok["item_meta"], {"task_id": 3, "init_state": 41, "outcome": "success"})
        to = E.score_record(job, E.run_episode(FakeEnv(), const_act(), 280, 50), None, 1.0, 0)
        self.assertEqual((to["score"], to["sub_metrics"]["timeout"]), (0, 1))
        err = E.score_record(job, None, {"class": "simulator_error", "message": "x"}, 0.5, 0)
        self.assertIsNone(err["score"])
        self.assertEqual(err["error"]["class"], "simulator_error")
        for rec in (ok, to, err):                            # MM-1: sub_metrics are finite numbers
            self.assertTrue(all(C.finite(v) for v in rec["sub_metrics"].values()))
            self.assertEqual(set(rec), {"item", "repeat", "seed", "score", "error", "sub_metrics", "item_meta"})

    def test_action_summary(self):
        s = E.action_summary([[0.0] * 7, [2.0] * 7])
        self.assertEqual((s["n"], s["dim"], s["abs_gt_1"]), (2, 7, 7))


class WorkerEnvs(unittest.TestCase):
    """run_job keeps at most envs_per_worker LiberoEnv instances open (one EGL context each), LRU."""

    def test_env_cache_is_bounded(self):
        from types import SimpleNamespace

        closed = []

        class FakeAdapter(FakeEnv):
            def __init__(self, suite, task_id, settle_steps, max_steps):
                super().__init__(success_at=3)
                self.task_id = task_id

            def close(self):
                closed.append(self.task_id)

        policy = SimpleNamespace(reset=lambda: None,
                                 config=SimpleNamespace(chunk_size=50, max_action_dim=32, n_action_steps=10))
        proc = SimpleNamespace(reset=lambda: None)
        saved = (dict(E._W), E._LiberoAdapter, E._policy_act)
        try:
            E._LiberoAdapter = FakeAdapter
            E._policy_act = lambda adapter, gen, shape: const_act()
            for keep, tasks, want_open, want_closed in ((1, [0, 0, 1, 2, 2], [2], [0, 1]),
                                                       (2, [0, 1, 0, 2, 1], [2, 1], [1, 0])):
                closed.clear()
                E._W.clear()
                E._W.update(cfg={"envs_per_worker": keep, "settle_steps": 0, "max_steps": 10, "device": "cpu"},
                            envs={}, suite=None, policy=policy, pre=proc, post=proc)
                for t in tasks:
                    job = {"item": f"libero_object/t{t}/s40", "task_id": t, "init_state": 40, "env_seed": 1,
                           "torch_seed": 2, "noise_seed": 3}
                    res = E.run_job(job)
                    self.assertIsNone(res["error"])
                    self.assertEqual(res["episode"]["outcome"], "success")
                    self.assertIsNone(res["device_memory"])                 # CPU: nothing recorded
                    self.assertLessEqual(len(E._W["envs"]), keep)
                self.assertEqual(list(E._W["envs"]), want_open)
                self.assertEqual(closed, want_closed)
        finally:
            E._W.clear()
            E._W.update(saved[0])
            E._LiberoAdapter, E._policy_act = saved[1], saved[2]


class Jobs(unittest.TestCase):
    def test_seeds_common_random_numbers(self):
        items = [{"item": "libero_object/t0/s40", "task_id": 0, "init_state": 40},
                 {"item": "libero_object/t1/s40", "task_id": 1, "init_state": 40}]
        a = E.make_jobs(items, [11, 12])
        b = E.make_jobs(list(reversed(items)), [11, 12])      # another arm, another order: same seeds per item
        key = lambda j: (j["eval_seed"], j["item"])           # noqa: E731
        self.assertEqual(sorted(a, key=key), sorted(b, key=key))
        self.assertEqual(len(a), 4)
        self.assertEqual([j["repeat"] for j in a], [0, 0, 1, 1])
        self.assertNotEqual(a[0]["noise_seed"], a[2]["noise_seed"])   # repeats differ
        self.assertNotEqual(a[0]["noise_seed"], a[1]["noise_seed"])   # items differ
        self.assertTrue(all(0 <= j["env_seed"] < 2**31 for j in a))
        self.assertEqual(a[0]["noise_seed"], C.derive_seed("libero-noise", 11, "libero_object/t0/s40"))


class Pool(unittest.TestCase):
    """run_pool with spawn workers running stand-ins (tests/_poolfake.py)."""

    def test_worker_death_is_simulator_error(self):
        import _poolfake as F

        jobs = [{"item": f"i{k}", "crash": k == 2} for k in range(6)]
        got = []
        E.run_pool(jobs, {}, 2, got.append, init=F.init_ok, fn=F.job)
        by = {r["job"]["item"]: r for r in got}
        self.assertEqual(sorted(by), [f"i{k}" for k in range(6)])
        self.assertEqual(len(got), 6)                                   # every job recorded exactly once
        self.assertEqual(by["i2"]["error"]["class"], "simulator_error")
        self.assertIn("worker_died", by["i2"]["error"]["message"])
        self.assertTrue(all(by[f"i{k}"]["error"] is None for k in (0, 1, 3, 4, 5)))

    def test_start_up_failure_aborts(self):
        import _poolfake as F

        with self.assertRaises(SystemExit):
            E.run_pool([{"item": "i0"}, {"item": "i1"}], {}, 2, lambda r: None, init=F.init_crash, fn=F.job)

    @staticmethod
    def _dead(pid: int, wait_s: float = 10.0) -> bool:
        import os
        import time

        end = time.time() + wait_s
        while time.time() < end:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            time.sleep(0.1)
        return False

    def test_hung_item_is_retried_alone_then_error(self):
        import time

        import _poolfake as F

        root = S.tmpdir("a8-pool-hang-")
        try:
            pid_file = root / "i2.pids"
            jobs = [{"item": f"i{k}", "repeat": 0, "sleep": 1.0} for k in range(6)]
            jobs[2] = {"item": "i2", "repeat": 0, "hang": True, "pid_file": str(pid_file)}
            got = []
            t0 = time.time()
            E.run_pool(jobs, {}, 2, got.append, init=F.init_ok, fn=F.job, item_timeout=2.5)
            self.assertLess(time.time() - t0, 60)                       # the run finishes
            by = {r["job"]["item"]: r for r in got}
            self.assertEqual(len(got), 6)                               # every job recorded exactly once
            self.assertEqual(sorted(by), [f"i{k}" for k in range(6)])
            err = by["i2"]["error"]
            self.assertEqual(err["class"], "simulator_error")
            self.assertTrue(err["message"].startswith("item_timeout:"), err["message"])
            self.assertIn("first attempt also timed out", err["message"])
            self.assertIsNone(by["i2"]["episode"])
            self.assertGreaterEqual(by["i2"]["wall"], 2.5)
            self.assertTrue(all(by[f"i{k}"]["error"] is None for k in (0, 1, 3, 4, 5)))
            pids = [int(x) for x in pid_file.read_text().split()]
            self.assertEqual(len(pids), 2)                              # first attempt, then one retry alone
            self.assertNotEqual(pids[0], pids[1])                       # the retry ran in a new worker
            self.assertTrue(all(self._dead(p) for p in pids))           # both hung workers were killed
            rec = E.score_record(by["i2"]["job"] | {"task_id": 0, "init_state": 40, "eval_seed": 11},
                                 None, err, by["i2"]["wall"], 0)
            self.assertIsNone(rec["score"])                             # MM-4: an error, not a score
        finally:
            S.rmtree(root)

    def test_hung_once_item_recovers(self):
        import _poolfake as F

        root = S.tmpdir("a8-pool-hang1-")
        try:
            pid_file = root / "i1.pids"
            jobs = [{"item": f"i{k}", "repeat": 0, "sleep": 0.5} for k in range(5)]
            jobs[1] = {"item": "i1", "repeat": 0, "hang_once": True, "pid_file": str(pid_file)}
            got = []
            E.run_pool(jobs, {}, 3, got.append, init=F.init_ok, fn=F.job, item_timeout=2.0)
            by = {r["job"]["item"]: r for r in got}
            self.assertEqual((len(got), sorted(by)), (5, [f"i{k}" for k in range(5)]))
            self.assertTrue(all(r["error"] is None for r in got))       # the retry alone scored
            pids = [int(x) for x in pid_file.read_text().split()]
            self.assertEqual(len(pids), 1)
            self.assertTrue(self._dead(pids[0]))
        finally:
            S.rmtree(root)

    def test_start_up_hang_aborts(self):
        import time

        import _poolfake as F

        t0 = time.time()
        with self.assertRaises(SystemExit):
            E.run_pool([{"item": "i0"}], {}, 1, lambda r: None, init=F.init_hang, fn=F.job, startup_timeout=3)
        self.assertLess(time.time() - t0, 60)

    def test_item_timeout_flag(self):
        base = ["--checkpoint", "c", "--out", "o"]
        self.assertEqual(E.parse_args(base).item_timeout_s, 300.0)
        self.assertEqual(E.parse_args(base).envs_per_worker, 1)
        self.assertEqual(E.parse_args(base + ["--item-timeout-s", "0"]).item_timeout_s, 0.0)
        with self.assertRaises(SystemExit):
            E.parse_args(base + ["--item-timeout-s", "-1"])
        with self.assertRaises(SystemExit):
            E.parse_args(base + ["--envs-per-worker", "0"])


class RealSuite(unittest.TestCase):
    def test_libero_object_tasks(self):
        tasks = E.suite_tasks()
        self.assertEqual(len(tasks), 10)
        self.assertEqual(tasks[0]["task"], "pick_up_the_alphabet_soup_and_place_it_in_the_basket")
        self.assertEqual(tasks[9]["language"], "pick up the orange juice and place it in the basket")
        for t in tasks:
            self.assertTrue(t["init_file_name"].endswith(".pruned_init"))
            self.assertEqual(C.sha256_file(Path(t["init_file"])), t["init_file_sha256"])
        env_v = E.env_version_doc(tasks, C.MAX_STEPS, C.SETTLE_STEPS)
        self.assertEqual(env_v["nondefault"], {})
        self.assertEqual(env_v["env_kwargs"]["max_steps"], 280)
        self.assertEqual(len(env_v["init_files_sha256"]), 10)
        self.assertEqual(E.env_version_doc(tasks, 3, 2)["nondefault"], {"max_steps": 3, "num_steps_wait": 2})
        json.dumps(env_v)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python
"""Download the pinned model, dataset and simulator-asset repositories of area "VLA" (host step, network on, no GPU).

The "necessary set" chosen by the maintainers (decision "Area 8 (VLA) decisions"), every repository
at a full 40-hex revision resolved from the Hub API on 2026-11-14:

  models/smolvla_base/                    lerobot/smolvla_base                        (whole repository)
  models/SmolVLM2-500M-Video-Instruct/    HuggingFaceTB/SmolVLM2-500M-Video-Instruct  (without onnx/)
  data/lerobot/libero/                    lerobot/libero (dataset, whole repository; HF_LEROBOT_HOME=data/)
  assets/libero-assets/                   lerobot/libero-assets (dataset, whole repository)

Every directory is plain (no symbolic links, MM-20). huggingface_hub's own download metadata
(`<local_dir>/.cache/huggingface/`) is moved out to `hf-meta/<name>/`. No token is used or read: `token=False` on
every call, `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`, and `HF_HOME` points to the fresh `hf-home/` of this directory; a
token in the environment is removed from this process's environment before huggingface_hub is imported (its value
is never read). Xet transfer is disabled (plain HTTPS).

After each download the file list is compared with the repository's file list at the pinned revision (minus the
excluded patterns); `MANIFEST.json` records per repository: repo, revision, licence, excluded patterns, every file's
SHA-256 and bytes. `--verify` re-hashes offline and compares. Run with any venv that has huggingface_hub (for example
another recipe's venv, read-only), under `env -u HF_TOKEN`.

Licences: smolvla_base, SmolVLM2-500M-Video-Instruct and lerobot/libero declare apache-2.0; lerobot/libero-assets
declares no licence on its card and has no licence file (it holds scanned objects, HOPE and TurboSquid meshes, whose
own terms may restrict use), disclosed in the README. Its download is therefore opt-in: without
`--accept-libero-assets-terms` the script refuses to download anything and says why. Check the terms of those assets
yourself before passing the flag.

Usage (host):
  python fetch_models.py --accept-libero-assets-terms   # download what is missing, write MANIFEST.json
  python fetch_models.py --verify                       # re-hash every file (no network)
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "MANIFEST.json"

# name -> (repo_type, repo_id, revision, local dir relative to HERE, ignore patterns, licence)
SPECS = {
    "smolvla_base": ("model", "lerobot/smolvla_base", "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6",
                     "models/smolvla_base", [], "apache-2.0"),
    "SmolVLM2-500M-Video-Instruct": ("model", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
                                     "4cde9d208e19519c36917aa170fffc7e381aa0ad",
                                     "models/SmolVLM2-500M-Video-Instruct", ["onnx/*"], "apache-2.0"),
    "libero": ("dataset", "lerobot/libero", "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4",
               "data/lerobot/libero", [], "apache-2.0"),
    "libero-assets": ("dataset", "lerobot/libero-assets", "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6",
                      "assets/libero-assets", [], "none declared on the card (disclosed)"),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def plain_files(root: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise SystemExit(f"{p} is a symbolic link; the directory must be plain (MM-20)")
        if p.is_file():
            out[p.relative_to(root).as_posix()] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    return out


def _hub_env() -> None:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH"):
        os.environ.pop(name, None)          # removed, never read
    os.environ["HF_HOME"] = str(HERE / "hf-home")
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    (HERE / "hf-home").mkdir(parents=True, exist_ok=True)


def _move_meta(dest: Path, name: str) -> None:
    meta = dest / ".cache"
    if meta.exists():
        target = HERE / "hf-meta" / name
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(meta), str(target))


ASSETS = "libero-assets"
ASSETS_REFUSAL = (
    "refusing to download: the pinned set includes lerobot/libero-assets (the LIBERO simulator's meshes and textures, "
    "needed to evaluate), which states no license (no dataset card license, no license file) and contains scanned "
    "objects, HOPE and TurboSquid meshes that may carry their own terms. Check those terms for your use; if they allow "
    "it, rerun with --accept-libero-assets-terms (the README's \"Data and licenses\" section lists every download).")


def download() -> int:
    _hub_env()
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    record = {"schema": "ptc_mm.vla_downloads", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
              "repos": {}}
    for name, (rtype, repo, revision, rel, ignore, lic) in SPECS.items():
        info = (api.model_info if rtype == "model" else api.dataset_info)(
            repo, revision=revision, token=False, files_metadata=True)
        if info.sha != revision:
            raise SystemExit(f"{repo}: Hub resolved {revision} to {info.sha}")
        expected = {s.rfilename: s.size for s in info.siblings
                    if not any(fnmatch.fnmatch(s.rfilename, pat) for pat in ignore)}
        dest = HERE / rel
        snapshot_download(repo, repo_type=rtype, revision=revision, local_dir=dest, token=False,
                          ignore_patterns=ignore or None, max_workers=8)
        _move_meta(dest, name)
        got = plain_files(dest)
        if sorted(got) != sorted(expected):
            missing = sorted(set(expected) - set(got))[:5]
            extra = sorted(set(got) - set(expected))[:5]
            raise SystemExit(f"{dest}: file list differs from the Hub (missing {missing}, extra {extra})")
        wrong = [f for f, size in expected.items() if size is not None and got[f]["bytes"] != size]
        if wrong:
            raise SystemExit(f"{dest}: size differs from the Hub for {wrong[:5]}")
        record["repos"][name] = {"repo": repo, "repo_type": rtype, "revision": revision, "license": lic,
                                 "local_dir": rel, "excluded": ignore, "n_files": len(got),
                                 "bytes": sum(f["bytes"] for f in got.values()), "files": got}
        print(f"{name}: {len(got)} files, {record['repos'][name]['bytes'] / 1e6:.1f} MB", flush=True)
        MANIFEST.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def verify() -> int:
    bad = 0
    record = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for name, m in record["repos"].items():
        got = plain_files(HERE / m["local_dir"])
        ok = got == m["files"]
        bad += not ok
        print(f"{name}: {'ok' if ok else 'MISMATCH'} ({len(got)} files)", flush=True)
    return 1 if bad else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--verify", action="store_true")
    p.add_argument("--accept-libero-assets-terms", action="store_true",
                   help="download lerobot/libero-assets, which states no license; check its terms first")
    args = p.parse_args(argv)
    if args.verify:
        return verify()
    if ASSETS in SPECS and not args.accept_libero_assets_terms:
        print(ASSETS_REFUSAL, file=sys.stderr)
        return 2
    return download()


if __name__ == "__main__":
    sys.exit(main())

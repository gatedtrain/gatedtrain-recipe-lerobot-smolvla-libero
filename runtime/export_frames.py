#!/usr/bin/env python
"""Frame export of the LIBERO-Object part of lerobot/libero (area "VLA"; the spec's training data for gatedtrain's
`lerobot_smolvla` adapter, tests/README.md section 11.26 EXT-237). Draft for the area runtime (host step, CPU, no
download): reads four integer columns of the dataset's parquet files and writes

  <out>/frames_meta.json                 schema ptc_mm.lerobot_frame_export: the dataset binding (repository, revision,
                                         files_sha256 of MANIFEST.json, episodes [807, 1260]), n_frames, and per shard
                                         {source, source_sha256, rows, first_index, last_index}
  <out>/chunk-CCC_file-FFF.jsonl         one row per LIBERO-Object frame of data/chunk-CCC/file-FFF.parquet, in index
                                         order: {"episode_index", "frame_index", "index", "task_index"} (sorted keys)

Deterministic; refuses to change an existing export. Usage (needs pyarrow, e.g. the area venv or gatedtrain's):
  python -I -B export_frames.py --dataset <area8-vla>/data/lerobot/libero --manifest <area8-vla>/MANIFEST.json --out DIR
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

EPISODES = (807, 1260)
REPO, REVISION = "lerobot/libero", "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
COLUMNS = ("index", "episode_index", "frame_index", "task_index")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def compact_sha256(doc) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":"),
                                                 ensure_ascii=False).encode("utf-8")).hexdigest()


def build(dataset: Path, files_sha256: str) -> dict[str, str]:
    """{relative path: text} of the export."""
    import pyarrow.parquet as pq

    eps = pq.read_table(dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
                        columns=["episode_index", "data/chunk_index", "data/file_index"]).to_pylist()
    sources = sorted({(e["data/chunk_index"], e["data/file_index"]) for e in eps
                      if EPISODES[0] <= e["episode_index"] <= EPISODES[1]})
    shards, texts, n = {}, {}, 0
    for c, f in sources:
        src = f"data/chunk-{c:03d}/file-{f:03d}.parquet"
        t = pq.read_table(dataset / src, columns=list(COLUMNS)).to_pydict()
        rows = sorted((dict(zip(COLUMNS, v)) for v in zip(*(t[k] for k in COLUMNS))
                       if EPISODES[0] <= v[1] <= EPISODES[1]), key=lambda r: r["index"])
        name = f"chunk-{c:03d}_file-{f:03d}.jsonl"
        texts[name] = "".join(json.dumps({k: int(r[k]) for k in COLUMNS}, sort_keys=True) + "\n" for r in rows)
        shards[name] = {"source": src, "source_sha256": sha256_file(dataset / src), "rows": len(rows),
                        "first_index": int(rows[0]["index"]), "last_index": int(rows[-1]["index"])}
        n += len(rows)
    meta = {"schema": "ptc_mm.lerobot_frame_export", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
            "dataset": {"repo": REPO, "revision": REVISION, "files_sha256": files_sha256, "episodes": list(EPISODES)},
            "n_frames": n, "columns": list(COLUMNS),
            "row_key": "data_ref.row = sha256 of the RFC 8785 JSON {episode_index, frame_index} (ptc-skel)",
            "shards": shards}
    texts["frames_meta.json"] = json.dumps(meta, indent=2, sort_keys=True) + "\n"
    return texts


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    rec = json.loads(a.manifest.read_text(encoding="utf-8"))["repos"]["libero"]
    if (rec["repo"], rec["revision"]) != (REPO, REVISION):
        raise SystemExit("MANIFEST.json names another dataset revision")
    files_sha256 = compact_sha256({k: v["sha256"] for k, v in rec["files"].items()})
    texts = build(a.dataset, files_sha256)
    a.out.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        path = a.out / name
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise SystemExit(f"{path} exists and differs; the export is never rewritten in place")
    for name, text in texts.items():
        (a.out / name).write_text(text, encoding="utf-8")
    print(json.dumps({"n_frames": json.loads(texts["frames_meta.json"])["n_frames"], "shards": len(texts) - 1}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

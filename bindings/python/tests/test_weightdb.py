"""Run: python bindings/python/tests/test_weightdb.py <model_dir_or_safetensors> [<finetune>]"""
import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import weightdb  # noqa: E402


def read_safetensors(path: Path, name: str) -> np.ndarray:
    raw = path.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    info = json.loads(raw[8 : 8 + n])[name]
    b, e = info["data_offsets"]
    assert info["dtype"] == "F32"
    return np.frombuffer(raw[8 + n + b : 8 + n + e], dtype="<f4").reshape(info["shape"])


def main():
    model = Path(sys.argv[1])
    finetune = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    st_file = model / "model.safetensors" if model.is_dir() else model

    with tempfile.TemporaryDirectory() as tmp:
        db = weightdb.open(Path(tmp) / "repo", create=True)
        r = db.import_model(model, message="base")
        print("import:", r["commit"]["id"][:12], r["stats"])
        if finetune:
            r2 = db.import_model(finetune, message="finetune")
            print("import ft:", r2["stats"])
            assert r2["stats"]["delta_chunks"] > 0

        names = db.tensor_names()
        name = next(n for n in names if db.tensor_info(n)["dtype"] == "F32" and len(db.tensor_info(n)["shape"]) == 2)
        rev = "main~1" if finetune else "main"
        arr = db.read_tensor(name, rev)
        ref = read_safetensors(st_file, name)
        assert arr.shape == ref.shape and np.array_equal(arr, ref), "tensor mismatch"
        assert db.weight(name, 5, rev) == float(ref.reshape(-1)[5])
        assert np.array_equal(db[rev][name], ref)
        print(f"read_tensor ok: {name} {arr.shape} {arr.dtype}")

        info = db.tensor_info(name, rev)
        assert abs(info["stats"]["mean"] - float(ref.astype(np.float64).mean())) < 1e-6

        sd = db.state_dict(rev)
        assert len(sd) == len(names)
        print(f"state_dict ok: {len(sd)} tensors")

        rows = db.sql(
            "SELECT role, count(*) AS n, sum(nbytes) AS bytes FROM tensors "
            "WHERE commit_id = rev('main') GROUP BY role ORDER BY bytes DESC"
        )
        print("sql:", rows)
        assert sum(r["n"] for r in rows) == len(names)

        if finetune:
            d = db.diff("main~1", "main", numeric=True)
            print(f"diff: {len(d['changes'])} changed, {d['unchanged']} unchanged")
            print(db.sql_pretty(
                "SELECT t.name, tensor_cosine('main~1', 'main', t.name) AS cos "
                "FROM tensors t WHERE t.commit_id = rev('main') AND t.name LIKE 'h.11.mlp%' ORDER BY t.name"
            ))

        try:
            db.read_tensor("does.not.exist")
            raise AssertionError("expected error")
        except weightdb.WeightDBError as e:
            print("expected error:", e)

        print("stats:", {k: db.stats()[k] for k in ("chunks", "stored_bytes", "dedup_ratio")})
        print("PYTHON OK")


if __name__ == "__main__":
    main()

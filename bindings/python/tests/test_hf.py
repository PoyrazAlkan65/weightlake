"""Run: python bindings/python/tests/test_hf.py <hf_model_dir>   (needs torch + transformers)

<hf_model_dir> is a local transformers checkpoint with config.json and tokenizer
files, e.g. GPT-2.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import weightdb  # noqa: E402
import weightdb.hf as whf  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


@torch.no_grad()
def logits(model, ids):
    model.eval()
    return model(ids).logits


def main():
    src = Path(sys.argv[1])
    torch.manual_seed(0)
    ref = AutoModelForCausalLM.from_pretrained(src)
    tok = AutoTokenizer.from_pretrained(src)
    ids = tok("WeightDB stores weights as data, not files.", return_tensors="pt").input_ids
    ref_logits = logits(ref, ids)

    with tempfile.TemporaryDirectory() as tmp:
        db = weightdb.open(Path(tmp) / "repo", create=True)

        # 1) directory import -> transformers model, weights straight from the DB
        db.import_model(src, message="hub files")
        assert "config.json" in db.aux_files("main")
        m = whf.from_pretrained(db, "main")
        assert type(m).__name__ == type(ref).__name__
        assert torch.equal(logits(m, ids), ref_logits), "logits differ after import"
        tok2 = whf.load_tokenizer(db, "main")
        assert tok2(" hello").input_ids == tok(" hello").input_ids
        print("1 import -> from_pretrained: logits identical")

        # 2) model -> DB without files; unchanged weights dedupe against the hub import
        before = db.stats()["stored_bytes"]
        # (tokenizer omitted here: transformers re-serializes tokenizer.json, which is new bytes)
        r = whf.save_pretrained(db, ref, branch="hf", base="main", message="from model")
        grew = db.stats()["stored_bytes"] - before
        print(f"2 save_pretrained: {r['commit']['n_tensors']} tensors, stored +{grew} bytes")
        assert grew < 64 << 10, "identical weights should dedupe"
        m = whf.from_pretrained(db, "hf", model_cls=AutoModelForCausalLM)
        assert torch.equal(logits(m, ids), ref_logits)

        # 3) "fine-tune" a few layers -> stored as delta against the previous version
        ft = whf.from_pretrained(db, "hf")
        with torch.no_grad():
            for n, p in ft.named_parameters():
                if ".h.11.mlp." in n:
                    p.add_(torch.randn_like(p) * 1e-3)
        ft_logits = logits(ft, ids)
        r = whf.save_pretrained(db, ft, tokenizer=tok, branch="hf", message="ft step")
        print("3 fine-tune commit:", r["stats"])
        assert r["stats"]["delta_chunks"] > 0
        m = whf.from_pretrained(db, "hf")
        assert torch.equal(logits(m, ids), ft_logits)
        assert not torch.equal(ft_logits, ref_logits)
        d = db.diff("hf~1", "hf")
        changed = sorted(c["name"] for c in d["changes"])
        assert changed and all(".h.11.mlp." in n for n in changed), changed
        print(f"   diff hf~1..hf: {changed}")

        # 4) exported files load with plain transformers
        out = Path(tmp) / "export"
        db.export("hf", out)
        m = AutoModelForCausalLM.from_pretrained(out)
        assert torch.equal(logits(m, ids), ft_logits)
        print("4 export -> AutoModelForCausalLM.from_pretrained(dir): logits identical")

        # 5) commit_state_dict edge cases: numpy, bf16, non-contiguous, empty
        x = torch.randn(64, 48)
        sd = {
            "np.f32": np.arange(12, dtype=np.float32).reshape(3, 4),
            "np.be": np.arange(5, dtype=">i4"),
            "t.bf16": torch.randn(33, 7).to(torch.bfloat16),
            "t.T": x.t(),  # non-contiguous view
            "t.empty": torch.zeros(0, 5),
            "t.bool": torch.tensor([True, False, True]),
        }
        db.commit_state_dict(sd, aux={"note.txt": "hi"}, branch="misc", message="edge cases")
        assert np.array_equal(db.read_tensor("np.f32", "misc"), sd["np.f32"])
        assert np.array_equal(db.read_tensor("np.be", "misc"), np.arange(5))
        assert torch.equal(db.read_tensor("t.bf16", "misc", framework="torch"), sd["t.bf16"])
        assert torch.equal(db.read_tensor("t.T", "misc", framework="torch"), x.t())
        assert db.read_tensor("t.empty", "misc", framework="torch").shape == (0, 5)
        assert db.read_tensor("t.bool", "misc", framework="torch").tolist() == [True, False, True]
        assert db.read_aux("note.txt", "misc") == b"hi"
        for bad in ({"a": torch.zeros(2, dtype=torch.complex64)}, {}):
            try:
                db.commit_state_dict(bad, branch="misc")
                raise AssertionError("expected error")
            except (TypeError, weightdb.WeightDBError) as e:
                print("   expected error:", e)
        print("5 commit_state_dict edge cases ok")

        print("stats:", {k: db.stats()[k] for k in ("commits", "chunks", "stored_bytes", "dedup_ratio") if k in db.stats()})
        print("HF OK")


if __name__ == "__main__":
    main()

"""WeightDB — an embedded database for neural network weights.

    import weightdb
    db = weightdb.open("repo", create=True)
    db.import_model("models/gpt2", message="gpt2 base")
    w = db.read_tensor("h.0.attn.c_attn.weight")            # numpy array
    sd = db.state_dict("main", framework="torch")            # {name: torch.Tensor}
    db.commit_state_dict(model.state_dict(), message="step 1000")   # no files
    model = weightdb.hf.from_pretrained(db, "main")          # transformers model
    db.sql("SELECT name, l2 FROM tensors WHERE commit_id = rev('main') ORDER BY l2 DESC LIMIT 5")

Analogy: repo ~ database, tensor ~ collection (matrix), chunk ~ document,
weight ~ field, commit ~ model version.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Optional, Union

from . import _native

__all__ = ["Database", "open", "init", "WeightDBError", "__version__"]
__version__ = _native.__version__

WeightDBError = RuntimeError

# safetensors dtype -> numpy dtype string (little-endian). BF16 and FP8 have no
# native numpy dtype; they are returned as raw unsigned ints unless converted.
_NUMPY_DTYPES = {
    "F64": "<f8", "F32": "<f4", "F16": "<f2",
    "I64": "<i8", "I32": "<i4", "I16": "<i2", "I8": "i1",
    "U64": "<u8", "U32": "<u4", "U16": "<u2", "U8": "u1",
    "BOOL": "?", "BF16": "<u2", "F8_E4M3": "u1", "F8_E5M2": "u1",
}

_TORCH_DTYPES = {
    "F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16",
    "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8", "U8": "uint8",
    "BOOL": "bool", "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2",
    "U16": "uint16", "U32": "uint32", "U64": "uint64",
}


def _to_numpy(buf: bytearray, dtype: str, shape: List[int], bf16_as_float32: bool):
    import numpy as np

    arr = np.frombuffer(buf, dtype=np.dtype(_NUMPY_DTYPES[dtype])).reshape(shape)
    if dtype == "BF16" and bf16_as_float32:
        arr = (arr.astype(np.uint32) << 16).view(np.float32)
    return arr


def _to_torch(buf: bytearray, dtype: str, shape: List[int]):
    import torch

    tdtype = getattr(torch, _TORCH_DTYPES[dtype])
    if len(buf) == 0:
        return torch.empty(shape, dtype=tdtype)
    return torch.frombuffer(buf, dtype=torch.uint8).view(tdtype).reshape(shape)


_TORCH_TO_ST = {v: k for k, v in _TORCH_DTYPES.items()}

_NUMPY_TO_ST = {
    "float64": "F64", "float32": "F32", "float16": "F16",
    "int64": "I64", "int32": "I32", "int16": "I16", "int8": "I8",
    "uint64": "U64", "uint32": "U32", "uint16": "U16", "uint8": "U8", "bool": "BOOL",
}


def _as_buffer(value):
    """(owner, st_dtype, shape, address, nbytes) for a torch tensor or numpy array.

    `owner` is the contiguous CPU object that must stay alive while the
    address is in use.
    """
    mod = type(value).__module__
    if mod.startswith("torch"):
        t = value.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        if not t.is_contiguous():
            t = t.contiguous()
        name = str(t.dtype).removeprefix("torch.")
        if name not in _TORCH_TO_ST:
            raise TypeError(f"unsupported torch dtype {t.dtype}")
        return t, _TORCH_TO_ST[name], list(t.shape), t.data_ptr(), t.numel() * t.element_size()

    import numpy as np

    a = np.asarray(value)
    if a.dtype.byteorder == ">":
        a = a.astype(a.dtype.newbyteorder("<"))
    a = np.ascontiguousarray(a)
    if a.dtype.name not in _NUMPY_TO_ST:
        raise TypeError(f"unsupported numpy dtype {a.dtype}")
    return a, _NUMPY_TO_ST[a.dtype.name], list(a.shape), a.ctypes.data, a.nbytes


class Database:
    """Handle to a WeightDB repository (one process at a time)."""

    def __init__(self, path: str, create: bool = False):
        self._db = _native.Database(str(path), create)

    @property
    def path(self) -> str:
        return self._db.path()

    def __repr__(self) -> str:
        return f"weightdb.Database({self.path!r})"

    # ---- versions -------------------------------------------------------
    def import_model(
        self,
        path: str,
        *,
        branch: str = "main",
        message: Optional[str] = None,
        base: Optional[str] = None,
        delta: bool = True,
        compress: bool = False,
        chunk_mib: int = 4,
        include_aux: bool = True,
    ) -> Dict[str, Any]:
        """Import a .safetensors file or model directory as a new commit."""
        opts = {
            "branch": branch,
            "message": message or f"import {path}",
            "base": base,
            "delta": delta,
            "compress": compress,
            "chunk_bytes": max(1, int(chunk_mib)) << 20,
            "include_aux": include_aux,
        }
        return json.loads(self._db.import_model(str(path), json.dumps(opts)))

    def export(self, rev: str, out_dir: str) -> List[str]:
        """Recreate the original files of a version (byte-identical)."""
        return self._db.export(rev, str(out_dir))

    def commit_state_dict(
        self,
        state_dict: Dict[str, Any],
        *,
        aux: Optional[Dict[str, Union[bytes, str]]] = None,
        branch: str = "main",
        message: Optional[str] = None,
        base: Optional[str] = None,
        delta: bool = True,
        compress: bool = False,
        chunk_mib: int = 4,
    ) -> Dict[str, Any]:
        """Commit `{name: torch.Tensor | numpy array}` as a new version, without files.

        Data is read in place (zero-copy for contiguous CPU tensors; GPU or
        non-contiguous tensors are copied to CPU first). `aux` holds side files
        such as `{"config.json": b"..."}`. Returns the import report.
        """
        owners, specs = [], []
        for name, value in state_dict.items():
            owner, dtype, shape, addr, nbytes = _as_buffer(value)
            owners.append(owner)
            specs.append((str(name), dtype, shape, addr, nbytes))
        aux_list = [(n, d.encode() if isinstance(d, str) else bytes(d)) for n, d in (aux or {}).items()]
        opts = {
            "branch": branch,
            "message": message or "commit state_dict",
            "base": base,
            "delta": delta,
            "compress": compress,
            "chunk_bytes": max(1, int(chunk_mib)) << 20,
        }
        report = json.loads(self._db.import_tensors(specs, aux_list, json.dumps(opts)))
        del owners  # buffers may be released only after the native call returned
        return report

    def aux_files(self, rev: str = "main") -> List[str]:
        """Side files stored with a version (config.json, tokenizer files, ...)."""
        return self._db.aux_files(rev)

    def read_aux(self, name: str, rev: str = "main") -> bytes:
        return self._db.read_aux(rev, name)

    def resolve(self, rev: str) -> str:
        return self._db.resolve(rev)

    def log(self, rev: str = "main", limit: int = 100) -> List[Dict[str, Any]]:
        return json.loads(self._db.log(rev, limit))

    def branches(self) -> Dict[str, str]:
        return {b["name"]: b["commit_id"] for b in json.loads(self._db.branches())}

    def create_branch(self, name: str, rev: str = "main") -> str:
        return self._db.create_branch(name, rev)

    def delete_branch(self, name: str) -> None:
        self._db.delete_branch(name)

    # ---- tensors ("matrices") and weights ("fields") ------------------------
    def tensors(self, rev: str = "main") -> List[Dict[str, Any]]:
        return json.loads(self._db.tensors(rev))

    def tensor_names(self, rev: str = "main") -> List[str]:
        return [t["name"] for t in self.tensors(rev)]

    def tensor_info(self, name: str, rev: str = "main") -> Dict[str, Any]:
        """Metadata and statistics (from the chunk index, no weight reads)."""
        return json.loads(self._db.tensor_info(rev, name))

    def read_tensor(self, name: str, rev: str = "main", *, framework: str = "numpy", bf16_as_float32: bool = True):
        """Read a tensor as a numpy array (default) or torch tensor."""
        buf, dtype, shape = self._db.read_tensor(rev, name)
        if framework == "torch":
            return _to_torch(buf, dtype, shape)
        if framework == "numpy":
            return _to_numpy(buf, dtype, shape, bf16_as_float32)
        if framework == "raw":
            return bytes(buf), dtype, shape
        raise ValueError(f"unknown framework {framework!r}")

    def iter_tensors(self, rev: str = "main", **kw) -> Iterator:
        for name in self.tensor_names(rev):
            yield name, self.read_tensor(name, rev, **kw)

    def state_dict(self, rev: str = "main", *, framework: str = "numpy", **kw) -> Dict[str, Any]:
        """All tensors of a version: `model.load_state_dict(db.state_dict(rev, framework="torch"))`."""
        return dict(self.iter_tensors(rev, framework=framework, **kw))

    def weight(self, name: str, index: int, rev: str = "main") -> float:
        """One weight by flat index; decodes only the chunk that holds it."""
        return self._db.read_weight(rev, name, int(index))

    # ---- diff / SQL / maintenance ------------------------------------------
    def diff(self, a: str, b: str, numeric: bool = False) -> Dict[str, Any]:
        return json.loads(self._db.diff(a, b, numeric))

    def sql(self, query: str) -> List[Dict[str, Any]]:
        """Read-only SQL. Tables: commits, refs, tensors, chunks, tensor_chunks."""
        return json.loads(self._db.sql(query))

    def sql_df(self, query: str):
        """SQL result as a pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame(self.sql(query))

    def sql_pretty(self, query: str) -> str:
        return self._db.sql_pretty(query)

    def stats(self) -> Dict[str, Any]:
        return json.loads(self._db.stats())

    def gc(self, dry_run: bool = False) -> Dict[str, Any]:
        return json.loads(self._db.gc(dry_run))

    # ---- convenience --------------------------------------------------------
    def __getitem__(self, rev: str) -> "Version":
        self.resolve(rev)
        return Version(self, rev)


class Version:
    """`db["main"]["h.0.attn.c_attn.weight"]` -> numpy array."""

    def __init__(self, db: Database, rev: str):
        self.db, self.rev = db, rev

    def __getitem__(self, name: str):
        return self.db.read_tensor(name, self.rev)

    def __iter__(self):
        return iter(self.db.tensor_names(self.rev))

    def __len__(self) -> int:
        return len(self.db.tensors(self.rev))

    def keys(self) -> List[str]:
        return self.db.tensor_names(self.rev)

    def __repr__(self) -> str:
        return f"<weightdb.Version {self.rev} ({self.db.resolve(self.rev)[:12]})>"


def open(path: str, create: bool = False) -> Database:  # noqa: A001
    return Database(path, create=create)


def init(path: str) -> Database:
    db = Database.__new__(Database)
    db._db = _native.Database.init(str(path))
    return db


def __getattr__(name: str):
    # `weightdb.hf` is imported lazily so that transformers stays optional.
    if name == "hf":
        import importlib

        return importlib.import_module(".hf", __name__)
    raise AttributeError(f"module 'weightdb' has no attribute {name!r}")

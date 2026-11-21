"""Hugging Face `transformers` integration.

    import weightdb, weightdb.hf as whf
    from transformers import AutoModelForCausalLM, AutoTokenizer

    db = weightdb.open("repo", create=True)

    # model -> new version (weights go from RAM straight into the DB, no files)
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    whf.save_pretrained(db, model, tokenizer=AutoTokenizer.from_pretrained("gpt2"), message="gpt2 base")

    # ... fine-tune ...
    whf.save_pretrained(db, model, branch="ft", base="main", message="epoch 1")   # stored as delta

    # version -> model (weights come straight from the DB, no files)
    model = whf.from_pretrained(db, "ft")
    tok = whf.load_tokenizer(db, "ft")

Only the small side files (config.json, tokenizer files) are written to a
temporary directory, because transformers reads those through file paths.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

from . import Database
from . import open as _open

__all__ = ["save_pretrained", "from_pretrained", "load_tokenizer", "load_config"]

# Files that make up a tokenizer / processor (written by save_pretrained).
_CONFIG_FILES = {"config.json", "generation_config.json"}

DbLike = Union[Database, str]


def _db(db: DbLike) -> Database:
    return db if isinstance(db, Database) else _open(str(db))


def _materialize_aux(db: Database, rev: str, out_dir: Path, only=None) -> list:
    names = [n for n in db.aux_files(rev) if only is None or n in only]
    for n in names:
        (out_dir / n).write_bytes(db.read_aux(n, rev))
    return names


# ---------------------------------------------------------------- save


def save_pretrained(
    db: DbLike,
    model,
    *,
    tokenizer=None,
    message: Optional[str] = None,
    branch: str = "main",
    base: Optional[str] = None,
    delta: bool = True,
    compress: bool = False,
    chunk_mib: int = 4,
    extra_files: Optional[Dict[str, Union[bytes, str]]] = None,
) -> Dict[str, Any]:
    """Commit a `transformers` model (and optionally its tokenizer) as a new version.

    Weights are read from the model's memory without writing any files.
    Tied weights (e.g. `lm_head.weight` = `wte.weight`) cost nothing extra:
    identical chunks are stored once. Returns the import report.
    """
    db = _db(db)
    aux: Dict[str, bytes] = {}
    config = getattr(model, "config", None)
    if config is not None:
        aux["config.json"] = config.to_json_string(use_diff=True).encode()
    gen = getattr(model, "generation_config", None)
    if gen is not None and getattr(model, "can_generate", lambda: False)():
        aux["generation_config.json"] = gen.to_json_string(use_diff=True).encode()
    if tokenizer is not None:
        with tempfile.TemporaryDirectory() as tmp:
            tokenizer.save_pretrained(tmp)
            for p in Path(tmp).iterdir():
                if p.is_file():
                    aux[p.name] = p.read_bytes()
    for name, data in (extra_files or {}).items():
        aux[name] = data.encode() if isinstance(data, str) else bytes(data)

    name = getattr(config, "_name_or_path", "") or type(model).__name__
    return db.commit_state_dict(
        model.state_dict(),
        aux=aux,
        message=message or f"save {name}",
        branch=branch,
        base=base,
        delta=delta,
        compress=compress,
        chunk_mib=chunk_mib,
    )


# ---------------------------------------------------------------- load


def load_config(db: DbLike, rev: str = "main", **kwargs):
    """`AutoConfig` of a stored version."""
    from transformers import AutoConfig

    db = _db(db)
    with tempfile.TemporaryDirectory() as tmp:
        if "config.json" not in _materialize_aux(db, rev, Path(tmp), only=_CONFIG_FILES):
            raise FileNotFoundError(f"version {rev!r} has no config.json")
        return AutoConfig.from_pretrained(tmp, **kwargs)


def load_tokenizer(db: DbLike, rev: str = "main", **kwargs):
    """`AutoTokenizer` from the side files stored with a version."""
    from transformers import AutoTokenizer

    db = _db(db)
    with tempfile.TemporaryDirectory() as tmp:
        _materialize_aux(db, rev, Path(tmp))
        return AutoTokenizer.from_pretrained(tmp, **kwargs)


def _model_class(config, model_cls):
    import transformers

    if model_cls is not None:
        # Auto classes cannot take `None` as a path: resolve the concrete class.
        mapping = getattr(model_cls, "_model_mapping", None)
        return mapping[type(config)] if mapping is not None else model_cls
    for arch in getattr(config, "architectures", None) or []:
        cls = getattr(transformers, arch, None)
        if cls is not None:
            return cls
    return transformers.AutoModel


def from_pretrained(
    db: DbLike,
    rev: str = "main",
    *,
    model_cls=None,
    config=None,
    device: Optional[str] = None,
    **kwargs,
):
    """Build a `transformers` model from a stored version.

    The weights go from the database into the model without any files.
    `model_cls` defaults to `config.architectures[0]` (e.g. `GPT2LMHeadModel`),
    else `AutoModel`; Auto classes such as `AutoModelForCausalLM` work too.
    Extra kwargs (e.g. `dtype=torch.bfloat16`) go to `from_pretrained`.
    """
    db = _db(db)
    if config is None:
        config = load_config(db, rev)
    state_dict = db.state_dict(rev, framework="torch")
    cls = _model_class(config, model_cls)
    model = cls.from_pretrained(None, config=config, state_dict=state_dict, **kwargs)
    if device is not None:
        model = model.to(device)
    return model

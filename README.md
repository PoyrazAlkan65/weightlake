# WeightLake

**An embedded database for neural network weights.** WeightLake stores model weights as *data* instead of opaque files.
Models are versioned like Git (commits, branches, diffs), deduplicated and delta-compressed across versions, and queryable with SQL down to a single weight.
The core is written in Rust, with native APIs for **Rust, C/C++, Python and JavaScript**.

> 🇹🇷 Türkçe: [README.tr.md](README.tr.md)

```python
import weightlake, weightlake.hf as whf

db = weightlake.open("repo", create=True)
whf.save_pretrained(db, model, tokenizer=tok, message="base")              # model -> version, no files
# ... fine-tune ...
whf.save_pretrained(db, model, branch="ft", base="main", message="ep 1")   # only changed chunks stored, as XOR deltas
model = whf.from_pretrained(db, "ft")                                      # version -> model, no files

db.sql("""SELECT name, tensor_rel_l2('main', 'ft', name) AS drift
          FROM tensors WHERE commit_id = rev('ft') ORDER BY drift DESC LIMIT 5""")
```

---

## Table of contents

- [Why](#why)
- [Concepts](#concepts)
- [Features](#features)
- [Measured example (GPT-2)](#measured-example-gpt-2)
- [Getting started](#getting-started)
- [CLI](#cli)
- [Python](#python)
  - [Hugging Face transformers](#hugging-face-transformers)
- [Rust](#rust)
- [C and C++](#c-and-c)
- [JavaScript / TypeScript](#javascript--typescript)
- [SQL reference](#sql-reference)
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Building from source](#building-from-source)
- [Testing](#testing)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [License](#license)

---

## Why

Model checkpoints are usually handled as big binary files.
Ten fine-tunes of a 7B model are ten copies of ~14 GB, even when most layers did not change.
Answering "which layers moved the most?" means loading every checkpoint into a framework.

WeightLake treats a checkpoint as structured data:

- **Store each byte once.** Weights are split into content-addressed chunks. A layer that is unchanged in 50 versions is stored once.
- **Store changes as deltas.** A changed chunk is XOR'ed against the same chunk in the base version and compressed. Fine-tuned weights share most sign, exponent and high mantissa bits with their base, so the XOR is mostly zeros.
- **Query without loading.** Every chunk carries statistics (min, max, mean, std, L2, zeros, NaN/Inf). Questions like "largest-norm attention matrices" or "tensors with NaNs" are answered from the index, without reading weights.
- **Read exactly what you need.** You can read one tensor, or a single weight by index. Only the chunk that holds it is decoded.
- **Round-trip exactly.** Importing a `.safetensors` file and exporting it again gives byte-identical files.

## Concepts

If you know MongoDB or Git, this mapping covers most of it:

| MongoDB    | Git        | WeightLake   | What it is |
|------------|------------|------------|------------|
| Database   | Repository | **Repo**   | A directory holding one model family and all its versions |
| Collection | —          | **Tensor** | One weight matrix, e.g. `h.0.attn.c_attn.weight` |
| Document   | Blob       | **Chunk**  | A block of whole rows of a tensor (~4 MiB), addressed by its BLAKE3 hash |
| Field      | —          | **Weight** | A single value, readable with `weight(rev, tensor, index)` |
| Index      | —          | **Chunk statistics** | min/max/sum/sum²/zeros/NaN/Inf per chunk |
| —          | Commit     | **Commit** | A model version: manifest (tensor → chunk list) + parent + message |
| —          | Branch     | **Branch** | A named pointer to a commit (`main`, `ft`, …) |

A **revision** (`rev`) can be a branch name (`main`), a commit id prefix (`d1c2f4b6`), or an ancestor (`main~1`, `ft~3`).

Every tensor also gets a **role**, inferred from its name: `embedding`, `attention`, `mlp`, `norm`, `head` or `other`. It can be used in SQL and CLI filters.

## Features

| Area | What you get |
|---|---|
| Storage | Content-addressed chunks (BLAKE3), cross-version deduplication, memory-mapped immutable pack files, redb catalog |
| Compression | XOR delta vs. base version + byte-plane grouping + zstd; optional zstd for non-delta chunks |
| Versioning | Commits, branches, `rev~N`, log, structural and numeric diff, garbage collection |
| Query | Read-only SQL (Apache DataFusion) over commits, tensors and chunks, plus weight-level UDFs |
| Formats | safetensors import/export (single file or sharded HF directories, byte-identical); side files such as `config.json` and the tokenizer are kept |
| dtypes | F64, F32, F16, BF16, F8_E4M3, F8_E5M2, I64, I32, I16, I8, U64, U32, U16, U8, BOOL |
| APIs | Rust, C (stable ABI), C++17 header, Python (numpy / torch / pandas), Node.js (TypedArrays + TypeScript types), CLI |
| ML integration | Hugging Face `transformers`: `save_pretrained` / `from_pretrained` straight to and from the DB; `commit_state_dict` for any PyTorch/numpy state dict |

## Measured example (GPT-2)

These are real numbers from the CLI on GPT-2 small (124M parameters, F32, 525 MiB of weights) on a laptop SSD.
The second version is a synthetic fine-tune that changes 28 of 160 tensors (the last block and the final norm).

| Step | Result |
|---|---|
| `import gpt2` | 160 tensors, 525 MiB in **0.8 s** |
| `import gpt2-ft --base main` | 88% of chunks reused; 62 MiB changed → **41.5 MiB written** as XOR deltas, in **0.5 s** |
| Total | 2 versions = 1.02 GiB logical → **523 MiB stored** (2.00× dedup ratio) |
| `export ft` | 548 MB file, **byte-identical** to the original, in 0.9 s |
| `whf.save_pretrained` of an unchanged model on top of the hub import | **+1 080 bytes** (only `config.json` is new) |

The delta ratio depends heavily on the data. Synthetic random perturbations compress worse than real fine-tunes or training checkpoints, where consecutive versions differ by small updates.
Treat the numbers above as a smoke test, not a benchmark.

## Getting started

WeightLake is not published to crates.io, PyPI or npm yet. Build it from source (see [Building from source](#building-from-source)):

```sh
git clone https://github.com/PoyrazAlkan65/weightlake && cd weightlake
cargo build --release --workspace
scripts/build-bindings.sh          # puts the .pyd/.so, .node and C library next to their packages
```

Then:

```sh
./target/release/wlake -R repo init
./target/release/wlake -R repo import path/to/hf-model-dir -m "base"
./target/release/wlake -R repo ls main
```

To use the Python package from the checkout, put `bindings/python` on `PYTHONPATH`, or build a wheel with `maturin build -m bindings/python/Cargo.toml --release`.
For Node, `require('./bindings/node')`.

## CLI

```
wlake [-R <repo>] [--json] <command>
```

`-R` selects the repository directory (default: current directory). `--json` prints machine-readable output for every command.

| Command | Description |
|---|---|
| `init` | Create an empty repository |
| `import <path> [-b branch] [-m msg] [--base rev] [--no-delta] [--compress] [--chunk-mib N] [--no-aux]` | Import a `.safetensors` file or a model directory (all `*.safetensors` shards + side files ≤ 64 MiB) as a new commit |
| `export <rev> <out-dir>` | Write a version back as the original files (byte-identical) |
| `log [rev] [-n N]` | First-parent history |
| `branch` / `branch <name> [at]` / `branch -d <name>` | List, create or delete branches |
| `ls [rev] [--role R] [--grep S]` | List tensors with dtype, shape and size |
| `show <rev> <tensor>` | Tensor statistics from the chunk index (no weight reads) |
| `get <rev> <tensor> <index>...` | Read single weights by flat index |
| `diff <a> <b> [--numeric]` | Changed, added and removed tensors. `--numeric` adds relative L2 and cosine drift |
| `sql "<query>"` | Read-only SQL query, printed as a table |
| `stats` | Logical vs. stored size, dedup ratio, codec breakdown |
| `gc [--dry-run]` | Delete versions not reachable from any branch, compact pack files |

```sh
wlake -R repo init
wlake -R repo import ./gpt2 -m "gpt2 base"
wlake -R repo import ./gpt2-ft -b ft --base main -m "fine-tune"   # deltas against main
wlake -R repo log ft
wlake -R repo ls main --role attention
wlake -R repo show main h.0.attn.c_attn.weight
wlake -R repo get main h.0.attn.c_attn.weight 0 1 2
wlake -R repo diff main ft --numeric
wlake -R repo sql "SELECT role, count(*), sum(nbytes) FROM tensors WHERE commit_id = rev('main') GROUP BY role"
wlake -R repo stats
wlake -R repo branch -d ft && wlake -R repo gc
wlake -R repo export main ./out
```

## Python

```python
import weightlake

db = weightlake.open("repo", create=True)          # or weightlake.init("repo")

# versions
r = db.import_model("gpt2", message="base")       # file or directory; returns the import report
db.import_model("gpt2-ft", branch="ft", base="main", message="ft", compress=False, chunk_mib=4)
db.log("ft")                                      # list of commits
db.branches()                                     # {"main": "<id>", "ft": "<id>"}
db.create_branch("exp", "main~1"); db.delete_branch("exp")
db.export("ft", "out/")                           # byte-identical safetensors + side files

# tensors ("matrices") and weights ("fields")
db.tensor_names("main")
db.tensor_info("h.0.attn.c_attn.weight")          # dtype, shape, role, stats (from the index)
w  = db.read_tensor("h.0.attn.c_attn.weight")     # numpy; BF16 -> float32 by default
w  = db.read_tensor("h.0.attn.c_attn.weight", framework="torch")
w  = db["main"]["h.0.attn.c_attn.weight"]         # dict-style access
x  = db.weight("h.0.attn.c_attn.weight", 12345)   # one value; decodes one chunk only
sd = db.state_dict("main", framework="torch")     # model.load_state_dict(sd)

# commit tensors from memory (zero-copy for contiguous CPU tensors)
db.commit_state_dict(model.state_dict(), message="step 1000", aux={"config.json": cfg_json})
db.aux_files("main"); db.read_aux("config.json", "main")

# diff / SQL / maintenance
db.diff("main", "ft", numeric=True)
db.sql("SELECT name, l2 FROM tensors WHERE commit_id = rev('main')")   # list[dict]
db.sql_df("SELECT * FROM chunks")                                      # pandas.DataFrame
print(db.sql_pretty("SELECT codec, count(*) FROM chunks GROUP BY codec"))
db.stats(); db.gc(dry_run=True)
```

`commit_state_dict` accepts torch tensors and numpy arrays.
GPU or non-contiguous tensors are first copied to contiguous CPU memory, and big-endian numpy arrays are converted to little-endian.
The Rust core reads the buffers in place and releases the GIL while it hashes and compresses.

Errors are raised as `weightlake.WeightLakeError` (a `RuntimeError`).

### Hugging Face transformers

`weightlake.hf` moves weights between a `transformers` model in RAM and the database directly. No `.safetensors` file is written.
Only the small side files (`config.json`, tokenizer files) go through a temporary directory, because transformers reads those from a path.

```python
import torch, weightlake, weightlake.hf as whf
from transformers import AutoModelForCausalLM, AutoTokenizer

db = weightlake.open("repo", create=True)

model = AutoModelForCausalLM.from_pretrained("gpt2")
tok = AutoTokenizer.from_pretrained("gpt2")
whf.save_pretrained(db, model, tokenizer=tok, message="base")   # config, generation_config, tokenizer included

# ... training ...
whf.save_pretrained(db, model, branch="ft", base="main", message="epoch 1")

model = whf.from_pretrained(db, "ft")                            # class from config.architectures
model = whf.from_pretrained(db, "ft", model_cls=AutoModelForCausalLM, dtype=torch.bfloat16, device="cuda")
tok   = whf.load_tokenizer(db, "ft")
cfg   = whf.load_config(db, "ft")
```

Things to know:

- **Tied weights cost nothing extra.** For example, `lm_head.weight` and `wte.weight` are both stored, but their chunks are identical, so they are stored once.
- **Hub files and `state_dict()` can use different names.** Older hub files use names like `h.0.…`, while `model.state_dict()` produces `transformer.h.0.…`. Deltas only match tensors with the same name, but content dedup still works. For the best delta compression, commit every version of a model the same way.
- **Tokenizer re-serialization.** transformers writes `tokenizer.json` differently from the hub copy, so the first tokenizer save adds a few MiB. Later saves of the same tokenizer dedupe.
- **Loading needs about twice the model size in RAM for a moment.** The copy read from the database and the model's own parameters exist at the same time.

## Rust

```rust
use weightlake::{Database, ImportOptions, MemTensor, DType};

let db = Database::open_or_init("repo")?;
let r = db.import("gpt2", &ImportOptions { message: "base".into(), ..Default::default() })?;
println!("{} {:?}", r.commit.short(), r.stats);

for t in db.tensors("main")? {
    println!("{} {:?} {} {}", t.name, t.shape, t.dtype, t.role);
}
let t = db.read_tensor("main", "h.0.attn.c_attn.weight")?;   // t.info, t.data (LE bytes), t.to_f64()
let x = db.read_weight("main", "h.0.attn.c_attn.weight", 42)?;
let d = db.diff("main", "ft", true)?;
let rows = db.sql_json("SELECT name, l2 FROM tensors WHERE commit_id = rev('main') ORDER BY l2 DESC LIMIT 5")?;
let batches = db.sql("SELECT * FROM chunks")?;             // Arrow RecordBatches

// commit from memory
let data: Vec<u8> = vec![0; 16];
db.import_tensors(
    vec![MemTensor { name: "w".into(), dtype: DType::F32, shape: vec![2, 2], data: &data }],
    &[("config.json".into(), b"{}".to_vec())],
    &ImportOptions { branch: "scratch".into(), ..Default::default() },
)?;
```

`Database` is cheap to clone and safe to share across threads.

`ImportOptions` fields (all optional when deserialized from JSON):

| Field | Default | Meaning |
|---|---|---|
| `branch` | `"main"` | Branch to commit to |
| `message` | `"import"` | Commit message |
| `base` | `None` | Revision to delta against. `None` uses the branch head |
| `delta` | `true` | XOR-delta changed chunks against the base |
| `compress` | `false` | zstd-compress chunks that are not deltas |
| `chunk_bytes` | 4 MiB | Target chunk size (always whole rows) |
| `include_aux` | `true` | Keep non-weight files from a model directory |

## C and C++

Link against `weightlake.dll` / `libweightlake.so` / `libweightlake.dylib`, and include `bindings/c/include/weightlake.h` (C) or `weightlake.hpp` (C++17, header-only RAII wrapper).

```c
#include "weightlake.h"

wl_db* db;
if (wl_open_or_init("repo", &db) != 0) { fprintf(stderr, "%s\n", wl_last_error()); return 1; }

wl_tensor_info_t info;
wl_tensor_info(db, "main", "h.0.attn.c_attn.weight", &info);
float* buf = malloc(info.nbytes);
wl_read_tensor(db, "main", "h.0.attn.c_attn.weight", buf, info.nbytes);

char* json;
wl_sql_json(db, "SELECT name, l2 FROM tensors WHERE commit_id = rev('main')", &json);
puts(json);
wl_string_free(json);
wl_close(db);
```

Every function returns `0` on success and `-1` on error. `wl_last_error()` gives the message for the current thread.
Strings returned through `char**` must be freed with `wl_string_free`.
Available functions: `wl_open/init/open_or_init/close`, `wl_import`, `wl_export`, `wl_resolve`, `wl_log_json`, `wl_branches_json`, `wl_create_branch`, `wl_delete_branch`, `wl_tensors_json`, `wl_tensor_info(_json)`, `wl_read_tensor`, `wl_read_weight`, `wl_diff_json`, `wl_sql_json`, `wl_sql_pretty`, `wl_stats_json`, `wl_gc_json`, `wl_version`.

```cpp
#include "weightlake.hpp"

auto db = weightlake::Database::open("repo");
std::vector<float> w = db.read_tensor<float>("main", "h.0.attn.c_attn.weight");
std::cout << db.sql_pretty("SELECT role, count(*) FROM tensors GROUP BY role");
```

The C++ wrapper throws `weightlake::Error` on failure. A full example is in `bindings/c/examples/example.cpp`.

## JavaScript / TypeScript

```js
const weightlake = require('weightlake');            // or require('./bindings/node')

const db = weightlake.open('repo', { create: true });
db.importModel('gpt2', { message: 'base' });
db.importModel('gpt2-ft', { branch: 'ft', base: 'main', message: 'ft' });

const { data, shape, dtype } = db.readTensor('h.0.attn.c_attn.weight');   // Float32Array, etc.
const buf = db.readTensorBuffer('h.0.attn.c_attn.weight', 'ft');          // raw Buffer
const x = db.weight('h.0.attn.c_attn.weight', 42);
db.diff('main', 'ft', { numeric: true });
db.sql("SELECT name, l2 FROM tensors WHERE commit_id = rev('main')");
console.log(db.sqlPretty('SELECT codec, count(*) FROM chunks GROUP BY codec'));
```

Methods: `importModel`, `export`, `resolve`, `log`, `branches`, `createBranch`, `deleteBranch`, `tensors`, `tensorNames`, `tensorInfo`, `readTensor`, `readTensorBuffer`, `weight`, `diff`, `sql`, `sqlPretty`, `stats`, `gc`.
TypeScript types are in `bindings/node/index.d.ts`.
BF16 and F16 tensors are converted to `Float32Array` unless you pass `{ raw: true }`. FP8 tensors are returned as raw `Uint8Array`s.

## SQL reference

SQL is read-only: DDL and DML statements are rejected. The engine is [Apache DataFusion](https://datafusion.apache.org/), so standard SQL works: joins, aggregates, window functions, CTEs.

### Tables

| Table | Columns |
|---|---|
| `commits` | `commit_id`, `short_id`, `message`, `created`, `timestamp`, `parent`, `n_tensors`, `logical_bytes`, `branches` |
| `refs` | `name`, `commit_id` |
| `tensors` | `commit_id`, `name`, `dtype`, `shape`, `ndim`, `numel`, `nbytes`, `role`, `n_chunks`, `mean`, `std`, `l2`, `abs_mean`, `min`, `max`, `sparsity`, `nan_count`, `inf_count` |
| `chunks` | `hash`, `codec`, `raw_len`, `stored_len`, `pack`, `base`, `mean`, `std`, `l2`, `min`, `max`, `zeros` |
| `tensor_chunks` | `commit_id`, `tensor`, `idx`, `hash`, `nbytes` |

The statistics in `tensors` are aggregated from the chunk index and need no weight reads.

### Functions

| Function | Returns |
|---|---|
| `rev(text)` | Commit id for a branch, id prefix or `rev~N` |
| `weight(rev, tensor, index)` | One weight as `DOUBLE`. Decodes only the chunk that holds it |
| `tensor_cosine(a, b, tensor)` | Cosine similarity of a tensor between two versions |
| `tensor_rel_l2(a, b, tensor)` | ‖b − a‖ / ‖a‖ |
| `tensor_max_abs_diff(a, b, tensor)` | max \|b − a\| |

The last three read both tensors, so put them after a selective `WHERE` clause.

### Examples

```sql
-- largest-norm attention matrices
SELECT name, shape, l2 FROM tensors
WHERE commit_id = rev('main') AND role = 'attention' AND ndim = 2
ORDER BY l2 DESC LIMIT 5;

-- which layers did the fine-tune move, and how much
SELECT b.name, tensor_rel_l2('main', 'ft', b.name) AS drift
FROM tensors a JOIN tensors b ON a.name = b.name
WHERE a.commit_id = rev('main') AND b.commit_id = rev('ft') AND a.mean <> b.mean
ORDER BY drift DESC;

-- health check: any NaN / Inf in the latest version?
SELECT name, nan_count, inf_count FROM tensors
WHERE commit_id = rev('main') AND (nan_count > 0 OR inf_count > 0);

-- parameter count by role
SELECT role, sum(numel) AS params, sum(nbytes) AS bytes FROM tensors
WHERE commit_id = rev('main') GROUP BY role ORDER BY params DESC;

-- storage efficiency per codec
SELECT codec, count(*) n, sum(raw_len) raw, sum(stored_len) stored,
       round(1.0 * sum(stored_len) / sum(raw_len), 3) AS ratio
FROM chunks GROUP BY codec;

-- how many versions share each chunk
SELECT hash, count(DISTINCT commit_id) AS versions FROM tensor_chunks
GROUP BY hash ORDER BY versions DESC LIMIT 10;

-- spot-check single weights across versions
SELECT weight('main', 'h.0.attn.c_attn.weight', 0) AS base, weight('ft', 'h.0.attn.c_attn.weight', 0) AS ft;
```

## How it works

### On-disk layout

```
repo/
  catalog.redb          chunk index, manifests, commits, refs (redb, ACID)
  packs/000001.pack     immutable chunk data, 64-byte aligned, memory-mapped
  packs/000002.pack
```

Each import writes its new chunks into fresh pack files and then commits one catalog transaction.
Packs are never modified after that; only `gc` replaces them.
Readers can therefore memory-map packs safely while new data is being written, and a crashed import leaves no partial version.

### Chunking and hashing

A tensor is split into chunks of about `chunk_bytes` (default 4 MiB), always on **row boundaries**, so a chunk is a block of complete rows of the matrix.
Each chunk is hashed with **BLAKE3** over its raw bytes. That hash is its identity: the same bytes anywhere, in any tensor or version, are stored once.

While hashing, per-chunk statistics are computed in parallel (rayon): count, min, max, sum, sum of squares, zeros, NaN and Inf.
Tensor-level statistics (mean, std, L2, sparsity, …) are aggregated from these, so they cost no weight reads.

### Codecs

| Codec | When | Encoding |
|---|---|---|
| `raw` | Default for new chunks | Bytes as-is; reads are zero-copy slices of the mmap |
| `zstd` | `compress=true` | Byte-plane grouping (byte *k* of every element goes into plane *k*), then zstd. Float exponent bytes are very repetitive, so they compress well |
| `xor-zstd` | A changed chunk that has a base | XOR with the base chunk, byte-plane grouping, then zstd. Small updates leave the sign, exponent and high mantissa bits unchanged, so the XOR is mostly zero bytes |

A chunk's **base** is the chunk at the same index of the same tensor (same name, dtype and shape) in the base version.
The base version is `base=` if given, otherwise the head of the target branch.

### Manifests and commits

A **manifest** lists every tensor of a version (name, dtype, shape, role, ordered chunk refs), the layout of each original safetensors file (raw header, tensor order, data length) and the side files.
Because the original header is kept verbatim, export rebuilds files byte for byte.
Tensors committed from memory get a freshly built standard header with `__metadata__: {"format": "pt"}`.

A **commit** points to a manifest and its parent(s), with a message and a timestamp.

### Garbage collection

`gc` marks every commit reachable from a branch, along with their manifests, chunks, and the base chunks of any delta.
Everything else is deleted, and pack files are rewritten without the dead chunks.

### Concurrency

A repository is meant to be opened by **one process at a time**. Within that process, the `Database` handle is thread-safe.

## Repository layout

```
crates/wl-core      dtypes, ChunkHash, row-aligned chunking, statistics, manifest/commit types
crates/wl-store     pack files (mmap) + redb catalog, codecs (raw / zstd / xor-zstd), gc
crates/wl-version   log, structural and numeric diff, tensor statistics
crates/wl-io        safetensors import/export, in-memory import, side files
crates/wl-query     SQL engine (DataFusion table providers + UDFs)
crates/weightlake      public Rust API: every binding is built on this crate
crates/wl-cli       the `wlake` command
bindings/c           C ABI (cdylib/staticlib) + include/weightlake.h + include/weightlake.hpp
bindings/python      PyO3 module `weightlake._native` + `weightlake` package (numpy/torch/pandas, `weightlake.hf`)
bindings/node        napi-rs addon + `weightlake` npm package (TypedArrays, TypeScript types)
scripts/             build-bindings.sh, gen-libnode.sh
```

## Building from source

Requirements: Rust (stable, 2021 edition). For the Python binding, Python ≥ 3.9 (abi3, so one build works for every version). For the Node binding, Node.js ≥ 18.

```sh
cargo build --release --workspace     # thin LTO: a release build takes several minutes
scripts/build-bindings.sh             # builds and copies artifacts:
                                      #   bindings/python/weightlake/_native.{pyd,so}
                                      #   bindings/node/weightlake.node
                                      #   dist/c/{include,lib}
```

**Python wheel:** `pip install maturin && maturin build -m bindings/python/Cargo.toml --release`
Optional extras: `torch`, `pandas`. `transformers` is needed only for `weightlake.hf`.

**Windows notes.** The project was developed on Windows with the **GNU toolchain** (`stable-x86_64-pc-windows-gnu`) and MinGW-w64 (WinLibs) on `PATH`:

```sh
rustup override set stable-x86_64-pc-windows-gnu
scripts/gen-libnode.sh     # GNU only: builds an import library for N-API symbols from node.exe
```

`.cargo/config.toml` points `LIBNODE_PATH` at `bindings/node/libnode`. With a complete MSVC + Windows SDK installation, the default MSVC toolchain should also work, and `gen-libnode.sh` is not needed there.

Linux and macOS have not been tested yet, but the code has no platform-specific parts beyond library file extensions, which `build-bindings.sh` handles.

## Testing

```sh
cargo test --workspace                                         # Rust unit + integration tests

# Python (needs numpy; second argument optional)
python bindings/python/tests/test_weightlake.py <model_dir> [<finetuned_model_dir>]

# Hugging Face integration (needs torch + transformers)
python bindings/python/tests/test_hf.py <hf_model_dir>

# Node
node bindings/node/test.js <model_dir>

# C++
#   compile bindings/c/examples/example.cpp against dist/c and run it on a repo
```

The Python and HF tests check bit-exact tensor equality against the source files, and identical logits for models loaded from WeightLake.
They also check delta/dedup behaviour, diffs, export round-trips and dtype edge cases (BF16, big-endian, non-contiguous, empty, bool).

## Limitations

- **Single process per repository.** There is no server and no multi-writer locking yet.
- **safetensors only.** GGUF, PyTorch `.bin`/`.pt` pickles and ONNX are not supported. Convert to safetensors first.
- **Whole-tensor reads go to RAM.** `read_tensor`/`state_dict` materialize full tensors. Only `weight()` reads a single chunk.
- **Delta matching is by tensor name.** Renamed tensors still dedupe by content, but they are not delta-encoded.
- **Not yet published** to crates.io, PyPI or npm. The C and Node bindings do not expose `import_tensors` yet.

## Roadmap

1. ✅ Storage core: chunking, dedup, statistics, safetensors import/export
2. ✅ Versioning: commit / branch / log / diff, XOR-delta + zstd, gc
3. ✅ Query: SQL (DataFusion) + weight-level UDFs
4. ◐ APIs: Rust, C/C++, Python, JavaScript ✅ · Hugging Face transformers ✅ · GGUF ☐ · wheels / npm packages ☐
5. ☐ Server: gRPC + Arrow Flight, `push` / `pull` between repositories, remote SQL

## License

Dual-licensed under either of [MIT](LICENSE-MIT) or [Apache License 2.0](LICENSE-APACHE), at your option.

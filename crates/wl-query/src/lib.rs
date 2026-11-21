//! SQL over a WeightLake repo.
//!
//! Tables (metadata only — built from the catalog, no weight reads):
//! - `commits(commit_id, short_id, message, created, timestamp, parent, n_tensors, logical_bytes, branches)`
//! - `refs(name, commit_id)`
//! - `tensors(commit_id, name, dtype, shape, ndim, numel, nbytes, role, n_chunks,
//!            mean, std, l2, abs_mean, min, max, sparsity, nan_count, inf_count)`
//! - `chunks(hash, codec, raw_len, stored_len, pack, base, mean, std, l2, min, max, zeros)`
//! - `tensor_chunks(commit_id, tensor, idx, hash, nbytes)`
//!
//! Functions:
//! - `rev(text) -> text`                         resolve branch / id prefix / `main~1`
//! - `weight(rev, tensor, index) -> double`      one weight ("field") by flat index
//! - `tensor_cosine(rev_a, rev_b, tensor)`       cosine similarity between versions
//! - `tensor_rel_l2(rev_a, rev_b, tensor)`       ||b-a|| / ||a||
//! - `tensor_max_abs_diff(rev_a, rev_b, tensor)` max |b-a|

use anyhow::{Context, Result};
use datafusion::arrow::array::{
    Array, ArrayRef, Float64Array, Float64Builder, Int64Array, StringArray, StringBuilder, UInt32Array, UInt64Array,
};
use datafusion::arrow::compute::cast;
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::DataFusionError;
use datafusion::execution::context::SQLOptions;
use datafusion::logical_expr::{create_udf, ColumnarValue, ScalarUDF, Volatility};
use datafusion::prelude::SessionContext;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use wl_core::{ChunkHash, ChunkStats, Manifest, TensorEntry};
use wl_store::{ChunkRecord, Codec, Repo};

pub use datafusion::arrow;

pub struct QueryEngine {
    repo: Arc<Repo>,
    rt: tokio::runtime::Runtime,
}

/// Resolves revisions and caches manifests for UDF calls.
struct Resolver {
    repo: Arc<Repo>,
    manifests: Mutex<HashMap<String, Arc<Manifest>>>,
}

impl Resolver {
    fn manifest(&self, rev: &str) -> Result<Arc<Manifest>> {
        if let Some(m) = self.manifests.lock().unwrap().get(rev) {
            return Ok(m.clone());
        }
        let c = self.repo.commit(&self.repo.resolve(rev)?)?;
        let m = Arc::new(self.repo.manifest(&c.manifest_id)?);
        self.manifests.lock().unwrap().insert(rev.to_string(), m.clone());
        Ok(m)
    }

    fn tensor(&self, rev: &str, name: &str) -> Result<(Arc<Manifest>, TensorEntry)> {
        let m = self.manifest(rev)?;
        let t = m.tensor(name).with_context(|| format!("no tensor '{name}' in {rev}"))?.clone();
        Ok((m, t))
    }

    fn pair(&self, a: &str, b: &str, name: &str) -> Result<(Vec<f64>, Vec<f64>)> {
        let (_, ta) = self.tensor(a, name)?;
        let (_, tb) = self.tensor(b, name)?;
        anyhow::ensure!(ta.shape == tb.shape, "shape mismatch for {name}");
        let load = |t: &TensorEntry| -> Result<Vec<f64>> {
            wl_core::dtype::decode_all_f64(t.dtype, &wl_io::read_tensor(&self.repo, t)?)
                .with_context(|| format!("dtype {} not numeric", t.dtype))
        };
        Ok((load(&ta)?, load(&tb)?))
    }
}

fn df_err(e: anyhow::Error) -> DataFusionError {
    DataFusionError::Execution(format!("{e:#}"))
}

fn str_arg(a: &ArrayRef) -> datafusion::error::Result<StringArray> {
    let c = cast(a, &DataType::Utf8)?;
    Ok(c.as_any().downcast_ref::<StringArray>().unwrap().clone())
}

fn udf_f64_3(name: &str, f: impl Fn(&str, &str, &str) -> Result<f64> + Send + Sync + 'static) -> ScalarUDF {
    create_udf(
        name,
        vec![DataType::Utf8, DataType::Utf8, DataType::Utf8],
        DataType::Float64,
        Volatility::Immutable,
        Arc::new(move |args: &[ColumnarValue]| {
            let arrays = ColumnarValue::values_to_arrays(args)?;
            let (a, b, c) = (str_arg(&arrays[0])?, str_arg(&arrays[1])?, str_arg(&arrays[2])?);
            let mut out = Float64Builder::with_capacity(a.len());
            for i in 0..a.len() {
                if a.is_null(i) || b.is_null(i) || c.is_null(i) {
                    out.append_null();
                } else {
                    out.append_value(f(a.value(i), b.value(i), c.value(i)).map_err(df_err)?);
                }
            }
            Ok(ColumnarValue::Array(Arc::new(out.finish())))
        }),
    )
}

fn register_udfs(ctx: &SessionContext, repo: Arc<Repo>) {
    let r = Arc::new(Resolver { repo: repo.clone(), manifests: Mutex::new(HashMap::new()) });

    let rr = repo.clone();
    ctx.register_udf(create_udf(
        "rev",
        vec![DataType::Utf8],
        DataType::Utf8,
        Volatility::Stable,
        Arc::new(move |args: &[ColumnarValue]| {
            let arrays = ColumnarValue::values_to_arrays(args)?;
            let a = str_arg(&arrays[0])?;
            let mut out = StringBuilder::new();
            for i in 0..a.len() {
                if a.is_null(i) {
                    out.append_null();
                } else {
                    out.append_value(rr.resolve(a.value(i)).map_err(df_err)?);
                }
            }
            Ok(ColumnarValue::Array(Arc::new(out.finish())))
        }),
    ));

    let rw = r.clone();
    ctx.register_udf(create_udf(
        "weight",
        vec![DataType::Utf8, DataType::Utf8, DataType::Int64],
        DataType::Float64,
        Volatility::Immutable,
        Arc::new(move |args: &[ColumnarValue]| {
            let arrays = ColumnarValue::values_to_arrays(args)?;
            let (a, b) = (str_arg(&arrays[0])?, str_arg(&arrays[1])?);
            let idx = cast(&arrays[2], &DataType::Int64)?;
            let idx = idx.as_any().downcast_ref::<Int64Array>().unwrap();
            let mut out = Float64Builder::with_capacity(a.len());
            for i in 0..a.len() {
                if a.is_null(i) || b.is_null(i) || idx.is_null(i) || idx.value(i) < 0 {
                    out.append_null();
                    continue;
                }
                let (_, t) = rw.tensor(a.value(i), b.value(i)).map_err(df_err)?;
                out.append_value(wl_io::read_element(&rw.repo, &t, idx.value(i) as u64).map_err(df_err)?);
            }
            Ok(ColumnarValue::Array(Arc::new(out.finish())))
        }),
    ));

    let rc = r.clone();
    ctx.register_udf(udf_f64_3("tensor_cosine", move |a, b, n| {
        let (x, y) = rc.pair(a, b, n)?;
        let (mut dot, mut na, mut nb) = (0.0, 0.0, 0.0);
        for (u, v) in x.iter().zip(&y) {
            dot += u * v;
            na += u * u;
            nb += v * v;
        }
        Ok(dot / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE))
    }));
    let rl = r.clone();
    ctx.register_udf(udf_f64_3("tensor_rel_l2", move |a, b, n| {
        let (x, y) = rl.pair(a, b, n)?;
        let (mut nd, mut na) = (0.0, 0.0);
        for (u, v) in x.iter().zip(&y) {
            nd += (v - u) * (v - u);
            na += u * u;
        }
        Ok(nd.sqrt() / na.sqrt().max(f64::MIN_POSITIVE))
    }));
    let rm = r;
    ctx.register_udf(udf_f64_3("tensor_max_abs_diff", move |a, b, n| {
        let (x, y) = rm.pair(a, b, n)?;
        Ok(x.iter().zip(&y).map(|(u, v)| (v - u).abs()).fold(0.0, f64::max))
    }));
}

fn batch(fields: Vec<(&str, DataType, bool)>, cols: Vec<ArrayRef>) -> Result<RecordBatch> {
    let schema = Schema::new(fields.into_iter().map(|(n, t, null)| Field::new(n, t, null)).collect::<Vec<_>>());
    Ok(RecordBatch::try_new(Arc::new(schema), cols)?)
}

fn opt_f64(stats: &[Option<&ChunkStats>], f: impl Fn(&ChunkStats) -> f64) -> ArrayRef {
    Arc::new(Float64Array::from(stats.iter().map(|s| s.map(&f)).collect::<Vec<_>>()))
}

fn merge_stats(t: &TensorEntry, recs: &HashMap<ChunkHash, ChunkRecord>) -> Option<ChunkStats> {
    let mut acc: Option<ChunkStats> = None;
    for c in &t.chunks {
        let s = recs.get(&c.hash)?.stats.as_ref()?;
        match &mut acc {
            Some(a) => a.merge(s),
            None => acc = Some(s.clone()),
        }
    }
    acc
}

impl QueryEngine {
    pub fn new(repo: Arc<Repo>) -> Result<Self> {
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
        Ok(QueryEngine { repo, rt })
    }

    fn build_tables(&self, ctx: &SessionContext) -> Result<()> {
        let repo = &self.repo;
        let refs = repo.refs()?;
        let mut commits = repo.commits()?;
        commits.sort_by_key(|c| std::cmp::Reverse(c.timestamp));
        let recs: HashMap<ChunkHash, ChunkRecord> = repo.all_chunk_records()?.into_iter().collect();

        // refs
        ctx.register_batch(
            "refs",
            batch(
                vec![("name", DataType::Utf8, false), ("commit_id", DataType::Utf8, false)],
                vec![
                    Arc::new(StringArray::from(refs.iter().map(|r| r.0.clone()).collect::<Vec<_>>())),
                    Arc::new(StringArray::from(refs.iter().map(|r| r.1.clone()).collect::<Vec<_>>())),
                ],
            )?,
        )?;

        // commits
        let branches_of = |id: &str| -> Option<String> {
            let b: Vec<&str> = refs.iter().filter(|r| r.1 == id).map(|r| r.0.as_str()).collect();
            (!b.is_empty()).then(|| b.join(","))
        };
        ctx.register_batch(
            "commits",
            batch(
                vec![
                    ("commit_id", DataType::Utf8, false),
                    ("short_id", DataType::Utf8, false),
                    ("message", DataType::Utf8, false),
                    ("created", DataType::Utf8, false),
                    ("timestamp", DataType::UInt64, false),
                    ("parent", DataType::Utf8, true),
                    ("n_tensors", DataType::UInt64, false),
                    ("logical_bytes", DataType::UInt64, false),
                    ("branches", DataType::Utf8, true),
                ],
                vec![
                    Arc::new(StringArray::from(commits.iter().map(|c| c.id.clone()).collect::<Vec<_>>())),
                    Arc::new(StringArray::from(commits.iter().map(|c| c.short().to_string()).collect::<Vec<_>>())),
                    Arc::new(StringArray::from(commits.iter().map(|c| c.message.clone()).collect::<Vec<_>>())),
                    Arc::new(StringArray::from(
                        commits.iter().map(|c| wl_core::format_ts(c.timestamp)).collect::<Vec<_>>(),
                    )),
                    Arc::new(UInt64Array::from(commits.iter().map(|c| c.timestamp).collect::<Vec<_>>())),
                    Arc::new(StringArray::from(commits.iter().map(|c| c.parents.first().cloned()).collect::<Vec<_>>())),
                    Arc::new(UInt64Array::from(commits.iter().map(|c| c.n_tensors).collect::<Vec<_>>())),
                    Arc::new(UInt64Array::from(commits.iter().map(|c| c.logical_bytes).collect::<Vec<_>>())),
                    Arc::new(StringArray::from(commits.iter().map(|c| branches_of(&c.id)).collect::<Vec<_>>())),
                ],
            )?,
        )?;

        // tensors + tensor_chunks (one manifest load per distinct manifest)
        let mut manifests: HashMap<String, Manifest> = HashMap::new();
        for c in &commits {
            if !manifests.contains_key(&c.manifest_id) {
                manifests.insert(c.manifest_id.clone(), repo.manifest(&c.manifest_id)?);
            }
        }
        let mut t_commit = Vec::new();
        let mut t_entries: Vec<&TensorEntry> = Vec::new();
        let mut tc = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for c in &commits {
            for t in &manifests[&c.manifest_id].tensors {
                t_commit.push(c.id.clone());
                t_entries.push(t);
                for (i, ch) in t.chunks.iter().enumerate() {
                    tc.0.push(c.id.clone());
                    tc.1.push(t.name.clone());
                    tc.2.push(i as u32);
                    tc.3.push(ch.hash.to_hex());
                    tc.4.push(ch.len);
                }
            }
        }
        let stats: Vec<Option<ChunkStats>> = t_entries.iter().map(|t| merge_stats(t, &recs)).collect();
        let sref: Vec<Option<&ChunkStats>> = stats.iter().map(|s| s.as_ref()).collect();
        ctx.register_batch(
            "tensors",
            batch(
                vec![
                    ("commit_id", DataType::Utf8, false),
                    ("name", DataType::Utf8, false),
                    ("dtype", DataType::Utf8, false),
                    ("shape", DataType::Utf8, false),
                    ("ndim", DataType::UInt32, false),
                    ("numel", DataType::UInt64, false),
                    ("nbytes", DataType::UInt64, false),
                    ("role", DataType::Utf8, false),
                    ("n_chunks", DataType::UInt32, false),
                    ("mean", DataType::Float64, true),
                    ("std", DataType::Float64, true),
                    ("l2", DataType::Float64, true),
                    ("abs_mean", DataType::Float64, true),
                    ("min", DataType::Float64, true),
                    ("max", DataType::Float64, true),
                    ("sparsity", DataType::Float64, true),
                    ("nan_count", DataType::UInt64, true),
                    ("inf_count", DataType::UInt64, true),
                ],
                vec![
                    Arc::new(StringArray::from(t_commit)),
                    Arc::new(StringArray::from(t_entries.iter().map(|t| t.name.clone()).collect::<Vec<_>>())),
                    Arc::new(StringArray::from(t_entries.iter().map(|t| t.dtype.as_str()).collect::<Vec<_>>())),
                    Arc::new(StringArray::from(
                        t_entries
                            .iter()
                            .map(|t| format!("{:?}", t.shape))
                            .collect::<Vec<_>>(),
                    )),
                    Arc::new(UInt32Array::from(t_entries.iter().map(|t| t.shape.len() as u32).collect::<Vec<_>>())),
                    Arc::new(UInt64Array::from(t_entries.iter().map(|t| t.numel()).collect::<Vec<_>>())),
                    Arc::new(UInt64Array::from(t_entries.iter().map(|t| t.nbytes()).collect::<Vec<_>>())),
                    Arc::new(StringArray::from(t_entries.iter().map(|t| t.role.as_str()).collect::<Vec<_>>())),
                    Arc::new(UInt32Array::from(t_entries.iter().map(|t| t.chunks.len() as u32).collect::<Vec<_>>())),
                    opt_f64(&sref, |s| s.mean()),
                    opt_f64(&sref, |s| s.std()),
                    opt_f64(&sref, |s| s.l2()),
                    opt_f64(&sref, |s| s.abs_mean()),
                    opt_f64(&sref, |s| s.min),
                    opt_f64(&sref, |s| s.max),
                    opt_f64(&sref, |s| s.sparsity()),
                    Arc::new(UInt64Array::from(sref.iter().map(|s| s.map(|s| s.nan)).collect::<Vec<_>>())),
                    Arc::new(UInt64Array::from(sref.iter().map(|s| s.map(|s| s.inf)).collect::<Vec<_>>())),
                ],
            )?,
        )?;
        ctx.register_batch(
            "tensor_chunks",
            batch(
                vec![
                    ("commit_id", DataType::Utf8, false),
                    ("tensor", DataType::Utf8, false),
                    ("idx", DataType::UInt32, false),
                    ("hash", DataType::Utf8, false),
                    ("nbytes", DataType::UInt64, false),
                ],
                vec![
                    Arc::new(StringArray::from(tc.0)),
                    Arc::new(StringArray::from(tc.1)),
                    Arc::new(UInt32Array::from(tc.2)),
                    Arc::new(StringArray::from(tc.3)),
                    Arc::new(UInt64Array::from(tc.4)),
                ],
            )?,
        )?;

        // chunks
        let rows: Vec<(&ChunkHash, &ChunkRecord)> = recs.iter().collect();
        let cs: Vec<Option<&ChunkStats>> = rows.iter().map(|(_, r)| r.stats.as_ref()).collect();
        ctx.register_batch(
            "chunks",
            batch(
                vec![
                    ("hash", DataType::Utf8, false),
                    ("codec", DataType::Utf8, false),
                    ("raw_len", DataType::UInt64, false),
                    ("stored_len", DataType::UInt64, false),
                    ("pack", DataType::UInt32, false),
                    ("base", DataType::Utf8, true),
                    ("mean", DataType::Float64, true),
                    ("std", DataType::Float64, true),
                    ("l2", DataType::Float64, true),
                    ("min", DataType::Float64, true),
                    ("max", DataType::Float64, true),
                    ("zeros", DataType::UInt64, true),
                ],
                vec![
                    Arc::new(StringArray::from(rows.iter().map(|(h, _)| h.to_hex()).collect::<Vec<_>>())),
                    Arc::new(StringArray::from(rows.iter().map(|(_, r)| r.codec.name()).collect::<Vec<_>>())),
                    Arc::new(UInt64Array::from(rows.iter().map(|(_, r)| r.raw_len).collect::<Vec<_>>())),
                    Arc::new(UInt64Array::from(rows.iter().map(|(_, r)| r.loc.len).collect::<Vec<_>>())),
                    Arc::new(UInt32Array::from(rows.iter().map(|(_, r)| r.loc.pack).collect::<Vec<_>>())),
                    Arc::new(StringArray::from(
                        rows.iter()
                            .map(|(_, r)| match r.codec {
                                Codec::XorZstd { base, .. } => Some(base.to_hex()),
                                _ => None,
                            })
                            .collect::<Vec<_>>(),
                    )),
                    opt_f64(&cs, |s| s.mean()),
                    opt_f64(&cs, |s| s.std()),
                    opt_f64(&cs, |s| s.l2()),
                    opt_f64(&cs, |s| s.min),
                    opt_f64(&cs, |s| s.max),
                    Arc::new(UInt64Array::from(cs.iter().map(|s| s.map(|s| s.zeros)).collect::<Vec<_>>())),
                ],
            )?,
        )?;
        Ok(())
    }

    /// Run a read-only SQL query.
    pub fn sql(&self, query: &str) -> Result<Vec<RecordBatch>> {
        let ctx = SessionContext::new();
        self.build_tables(&ctx)?;
        register_udfs(&ctx, self.repo.clone());
        let opts = SQLOptions::new().with_allow_ddl(false).with_allow_dml(false).with_allow_statements(false);
        self.rt.block_on(async {
            let df = ctx.sql_with_options(query, opts).await?;
            df.collect().await
        })
        .map_err(|e| anyhow::anyhow!("{e}"))
    }
}

/// Render batches as an ASCII table.
pub fn pretty(batches: &[RecordBatch]) -> Result<String> {
    Ok(datafusion::arrow::util::pretty::pretty_format_batches(batches)?.to_string())
}

/// Render batches as a JSON array of row objects.
pub fn to_json(batches: &[RecordBatch]) -> Result<serde_json::Value> {
    use datafusion::arrow::util::display::{ArrayFormatter, FormatOptions};
    let mut rows = Vec::new();
    for b in batches {
        let schema = b.schema();
        for i in 0..b.num_rows() {
            let mut obj = serde_json::Map::new();
            for (ci, col) in b.columns().iter().enumerate() {
                let name = schema.field(ci).name().clone();
                let v = if col.is_null(i) {
                    serde_json::Value::Null
                } else {
                    json_value(col, i).unwrap_or_else(|| {
                        let f = ArrayFormatter::try_new(col.as_ref(), &FormatOptions::default()).ok();
                        serde_json::Value::String(f.map(|f| f.value(i).to_string()).unwrap_or_default())
                    })
                };
                obj.insert(name, v);
            }
            rows.push(serde_json::Value::Object(obj));
        }
    }
    Ok(serde_json::Value::Array(rows))
}

fn json_value(col: &ArrayRef, i: usize) -> Option<serde_json::Value> {
    use datafusion::arrow::array::*;
    use serde_json::{json, Value};
    let a = col.as_any();
    macro_rules! num {
        ($($t:ty),*) => {$(
            if let Some(x) = a.downcast_ref::<$t>() { return Some(json!(x.value(i))); }
        )*};
    }
    num!(Int8Array, Int16Array, Int32Array, Int64Array, UInt8Array, UInt16Array, UInt32Array, UInt64Array);
    if let Some(x) = a.downcast_ref::<Float64Array>() {
        let v = x.value(i);
        return Some(if v.is_finite() { json!(v) } else { Value::String(v.to_string()) });
    }
    if let Some(x) = a.downcast_ref::<Float32Array>() {
        let v = x.value(i) as f64;
        return Some(if v.is_finite() { json!(v) } else { Value::String(v.to_string()) });
    }
    if let Some(x) = a.downcast_ref::<BooleanArray>() {
        return Some(json!(x.value(i)));
    }
    if let Some(x) = a.downcast_ref::<StringArray>() {
        return Some(json!(x.value(i)));
    }
    if let Some(x) = a.downcast_ref::<LargeStringArray>() {
        return Some(json!(x.value(i)));
    }
    if let Some(x) = a.downcast_ref::<StringViewArray>() {
        return Some(json!(x.value(i)));
    }
    None
}

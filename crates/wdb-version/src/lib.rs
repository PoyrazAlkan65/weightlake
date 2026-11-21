//! Version history and diffs between model versions.

use anyhow::{Context, Result};
use std::collections::{BTreeSet, HashSet};
use wdb_core::{ChunkHash, ChunkStats, Commit, DType, TensorEntry};
use wdb_store::Repo;

/// First-parent history starting at `rev`.
pub fn log(repo: &Repo, rev: &str, limit: usize) -> Result<Vec<Commit>> {
    let mut out = Vec::new();
    let mut next = Some(repo.resolve(rev)?);
    while let Some(id) = next {
        if out.len() >= limit {
            break;
        }
        let c = repo.commit(&id)?;
        next = c.parents.first().cloned();
        out.push(c);
    }
    Ok(out)
}

/// Aggregate chunk statistics into tensor statistics (no weight reads).
pub fn tensor_stats(repo: &Repo, t: &TensorEntry) -> Result<Option<ChunkStats>> {
    let hashes: Vec<ChunkHash> = t.chunks.iter().map(|c| c.hash).collect();
    let mut acc: Option<ChunkStats> = None;
    for rec in repo.chunk_records(&hashes)? {
        let Some(s) = rec.stats else { return Ok(None) };
        match &mut acc {
            Some(a) => a.merge(&s),
            None => acc = Some(s),
        }
    }
    Ok(acc)
}

#[derive(Debug, Clone)]
pub struct NumericDiff {
    /// ||b - a|| / ||a||
    pub rel_l2: f64,
    pub cosine: f64,
    pub max_abs: f64,
}

#[derive(Debug, Clone)]
pub enum Change {
    Added(TensorEntry),
    Removed(TensorEntry),
    Modified {
        name: String,
        changed_chunks: usize,
        total_chunks: usize,
        changed_bytes: u64,
        shape_changed: bool,
        numeric: Option<NumericDiff>,
    },
}

#[derive(Debug, Default)]
pub struct DiffReport {
    pub changes: Vec<Change>,
    pub unchanged: usize,
}

fn to_f64(dtype: DType, data: &[u8]) -> Option<Vec<f64>> {
    if !dtype.is_float() {
        return None;
    }
    wdb_core::dtype::decode_all_f64(dtype, data)
}

fn numeric_diff(repo: &Repo, a: &TensorEntry, b: &TensorEntry) -> Result<Option<NumericDiff>> {
    if a.shape != b.shape || a.dtype != b.dtype {
        return Ok(None);
    }
    let (Some(x), Some(y)) = (
        to_f64(a.dtype, &wdb_io::read_tensor(repo, a)?),
        to_f64(b.dtype, &wdb_io::read_tensor(repo, b)?),
    ) else {
        return Ok(None);
    };
    let (mut dot, mut na, mut nb, mut nd, mut max_abs) = (0.0, 0.0, 0.0, 0.0, 0.0f64);
    for (u, v) in x.iter().zip(&y) {
        dot += u * v;
        na += u * u;
        nb += v * v;
        let d = v - u;
        nd += d * d;
        max_abs = max_abs.max(d.abs());
    }
    Ok(Some(NumericDiff {
        rel_l2: nd.sqrt() / na.sqrt().max(f64::MIN_POSITIVE),
        cosine: dot / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE),
        max_abs,
    }))
}

/// Compare two versions tensor by tensor. Chunk hashes make the structural
/// diff free; `numeric` additionally reads changed tensors to measure drift.
pub fn diff(repo: &Repo, rev_a: &str, rev_b: &str, numeric: bool) -> Result<DiffReport> {
    let ma = repo.manifest(&repo.commit(&repo.resolve(rev_a)?)?.manifest_id)?;
    let mb = repo.manifest(&repo.commit(&repo.resolve(rev_b)?)?.manifest_id)?;
    let names: BTreeSet<&str> = ma.tensors.iter().chain(&mb.tensors).map(|t| t.name.as_str()).collect();
    let mut report = DiffReport::default();
    for name in names {
        match (ma.tensor(name), mb.tensor(name)) {
            (None, Some(b)) => report.changes.push(Change::Added(b.clone())),
            (Some(a), None) => report.changes.push(Change::Removed(a.clone())),
            (Some(a), Some(b)) => {
                let ha: Vec<_> = a.chunks.iter().map(|c| c.hash).collect();
                let hb: Vec<_> = b.chunks.iter().map(|c| c.hash).collect();
                if ha == hb && a.shape == b.shape && a.dtype == b.dtype {
                    report.unchanged += 1;
                    continue;
                }
                let in_a: HashSet<_> = ha.iter().collect();
                let changed: Vec<_> = b.chunks.iter().filter(|c| !in_a.contains(&c.hash)).collect();
                let numeric = if numeric {
                    numeric_diff(repo, a, b).with_context(|| format!("diffing {name}"))?
                } else {
                    None
                };
                report.changes.push(Change::Modified {
                    name: name.to_string(),
                    changed_chunks: changed.len(),
                    total_chunks: b.chunks.len(),
                    changed_bytes: changed.iter().map(|c| c.len).sum(),
                    shape_changed: a.shape != b.shape || a.dtype != b.dtype,
                    numeric,
                });
            }
            (None, None) => unreachable!(),
        }
    }
    Ok(report)
}

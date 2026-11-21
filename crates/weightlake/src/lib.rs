//! # WeightLake — Rust API
//!
//! An embedded database for neural network weights. Every language binding
//! (C/C++, Python, JavaScript) is a thin layer over [`Database`].
//!
//! ```no_run
//! use weightlake::{Database, ImportOptions};
//! let db = Database::open("repo")?;
//! db.import("models/gpt2", &ImportOptions { message: "gpt2".into(), ..Default::default() })?;
//! for t in db.tensors("main")? {
//!     println!("{} {:?} {}", t.name, t.shape, t.dtype);
//! }
//! let w = db.read_tensor("main", "h.0.attn.c_attn.weight")?;   // raw little-endian bytes
//! let rows = db.sql_json("SELECT name, l2 FROM tensors WHERE commit_id = rev('main') ORDER BY l2 DESC LIMIT 5")?;
//! # Ok::<(), anyhow::Error>(())
//! ```

use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

pub use wl_core::{format_ts, ChunkStats, Commit, DType, Manifest, Role, TensorEntry};
pub use wl_io::{ImportOptions, ImportReport, MemTensor};
pub use wl_query::arrow::record_batch::RecordBatch;
pub use wl_store::{GcReport, StorageStats};

/// Aggregated statistics of one tensor (from the chunk index, no weight reads).
#[derive(Debug, Clone, Serialize)]
pub struct TensorStats {
    pub mean: f64,
    pub std: f64,
    pub l2: f64,
    pub abs_mean: f64,
    pub min: f64,
    pub max: f64,
    pub sparsity: f64,
    pub nan: u64,
    pub inf: u64,
}

impl From<&ChunkStats> for TensorStats {
    fn from(s: &ChunkStats) -> Self {
        TensorStats {
            mean: s.mean(),
            std: s.std(),
            l2: s.l2(),
            abs_mean: s.abs_mean(),
            min: s.min,
            max: s.max,
            sparsity: s.sparsity(),
            nan: s.nan,
            inf: s.inf,
        }
    }
}

/// Metadata of a tensor ("matrix").
#[derive(Debug, Clone, Serialize)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub role: String,
    pub numel: u64,
    pub nbytes: u64,
    pub n_chunks: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats: Option<TensorStats>,
}

/// A tensor with its raw little-endian data.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub info: TensorInfo,
    pub data: Vec<u8>,
}

impl Tensor {
    /// Decode all values as f64 (`None` for BOOL/FP8).
    pub fn to_f64(&self) -> Option<Vec<f64>> {
        wl_core::dtype::decode_all_f64(DType::parse(&self.info.dtype)?, &self.data)
    }
}

/// One row of a diff between two versions.
#[derive(Debug, Clone, Serialize)]
pub struct DiffEntry {
    /// "added" | "removed" | "modified"
    pub kind: String,
    pub name: String,
    pub changed_chunks: u64,
    pub total_chunks: u64,
    pub changed_bytes: u64,
    pub shape_changed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rel_l2: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cosine: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_abs: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Diff {
    pub changes: Vec<DiffEntry>,
    pub unchanged: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Branch {
    pub name: String,
    pub commit_id: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct RepoStats {
    #[serde(flatten)]
    pub storage: StorageStats,
    pub branches: u64,
    /// Sum of the weight bytes of every version.
    pub logical_bytes: u64,
    /// logical_bytes / stored_bytes
    pub dedup_ratio: f64,
}

struct Inner {
    repo: Arc<wl_store::Repo>,
    query: OnceLock<wl_query::QueryEngine>,
}

/// Handle to an open repository. Cheap to clone; safe to share across threads.
#[derive(Clone)]
pub struct Database {
    inner: Arc<Inner>,
}

impl Database {
    fn wrap(repo: wl_store::Repo) -> Self {
        Database { inner: Arc::new(Inner { repo: Arc::new(repo), query: OnceLock::new() }) }
    }

    /// Create a new, empty repository.
    pub fn init(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self::wrap(wl_store::Repo::init(path)?))
    }

    /// Open an existing repository.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self::wrap(wl_store::Repo::open(path)?))
    }

    /// Open, creating the repository if it does not exist.
    pub fn open_or_init(path: impl AsRef<Path>) -> Result<Self> {
        if path.as_ref().join("catalog.redb").exists() {
            Self::open(path)
        } else {
            Self::init(path)
        }
    }

    /// Low-level storage access.
    pub fn repo(&self) -> &wl_store::Repo {
        &self.inner.repo
    }

    pub fn path(&self) -> &Path {
        self.inner.repo.root()
    }

    // ------------------------------------------------------------ versions

    /// Import a `.safetensors` file or a model directory as a new commit.
    pub fn import(&self, path: impl AsRef<Path>, opts: &ImportOptions) -> Result<ImportReport> {
        wl_io::import(&self.inner.repo, path.as_ref(), opts)
    }

    /// Commit tensors held in memory (no files). `aux` = side files such as
    /// `config.json`, as `(name, bytes)`. Delta/dedup work as for [`Self::import`].
    pub fn import_tensors(
        &self,
        tensors: Vec<MemTensor<'_>>,
        aux: &[(String, Vec<u8>)],
        opts: &ImportOptions,
    ) -> Result<ImportReport> {
        wl_io::import_tensors(&self.inner.repo, tensors, aux, opts)
    }

    /// Names of the side files (config.json, tokenizer, ...) of a version.
    pub fn aux_files(&self, rev: &str) -> Result<Vec<String>> {
        Ok(self.manifest(rev)?.aux.into_iter().map(|a| a.name).collect())
    }

    /// Contents of one side file of a version.
    pub fn read_aux(&self, rev: &str, name: &str) -> Result<Vec<u8>> {
        wl_io::read_aux(&self.inner.repo, &self.manifest(rev)?, name)
    }

    /// Recreate the original files of a version in `out_dir`.
    pub fn export(&self, rev: &str, out_dir: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
        wl_io::export(&self.inner.repo, rev, out_dir.as_ref())
    }

    /// Full commit id for a branch name, id prefix or `rev~N`.
    pub fn resolve(&self, rev: &str) -> Result<String> {
        self.inner.repo.resolve(rev)
    }

    pub fn commit(&self, rev: &str) -> Result<Commit> {
        self.inner.repo.commit(&self.resolve(rev)?)
    }

    pub fn manifest(&self, rev: &str) -> Result<Manifest> {
        self.inner.repo.manifest(&self.commit(rev)?.manifest_id)
    }

    /// First-parent history.
    pub fn log(&self, rev: &str, limit: usize) -> Result<Vec<Commit>> {
        wl_version::log(&self.inner.repo, rev, limit)
    }

    pub fn branches(&self) -> Result<Vec<Branch>> {
        Ok(self.inner.repo.refs()?.into_iter().map(|(name, commit_id)| Branch { name, commit_id }).collect())
    }

    pub fn create_branch(&self, name: &str, rev: &str) -> Result<String> {
        let id = self.resolve(rev)?;
        self.inner.repo.set_ref(name, &id)?;
        Ok(id)
    }

    pub fn delete_branch(&self, name: &str) -> Result<()> {
        if !self.inner.repo.delete_ref(name)? {
            bail!("no branch '{name}'");
        }
        Ok(())
    }

    // ------------------------------------------------------------- tensors

    fn info(&self, t: &TensorEntry, with_stats: bool) -> Result<TensorInfo> {
        let stats = if with_stats {
            wl_version::tensor_stats(&self.inner.repo, t)?.as_ref().map(TensorStats::from)
        } else {
            None
        };
        Ok(TensorInfo {
            name: t.name.clone(),
            dtype: t.dtype.as_str().to_string(),
            shape: t.shape.clone(),
            role: t.role.as_str().to_string(),
            numel: t.numel(),
            nbytes: t.nbytes(),
            n_chunks: t.chunks.len() as u64,
            stats,
        })
    }

    /// All tensors of a version (metadata only).
    pub fn tensors(&self, rev: &str) -> Result<Vec<TensorInfo>> {
        let m = self.manifest(rev)?;
        m.tensors.iter().map(|t| self.info(t, false)).collect()
    }

    /// Metadata + statistics of one tensor.
    pub fn tensor_info(&self, rev: &str, name: &str) -> Result<TensorInfo> {
        let m = self.manifest(rev)?;
        let t = m.tensor(name).with_context(|| format!("no tensor '{name}' in {rev}"))?;
        self.info(t, true)
    }

    /// Read a whole tensor (raw little-endian bytes).
    pub fn read_tensor(&self, rev: &str, name: &str) -> Result<Tensor> {
        let m = self.manifest(rev)?;
        let t = m.tensor(name).with_context(|| format!("no tensor '{name}' in {rev}"))?;
        Ok(Tensor { info: self.info(t, false)?, data: wl_io::read_tensor(&self.inner.repo, t)? })
    }

    /// Read a whole tensor into `out` (must be exactly `nbytes` long).
    pub fn read_tensor_into(&self, rev: &str, name: &str, out: &mut [u8]) -> Result<()> {
        let m = self.manifest(rev)?;
        let t = m.tensor(name).with_context(|| format!("no tensor '{name}' in {rev}"))?;
        wl_io::read_tensor_into(&self.inner.repo, t, out)
    }

    /// Read one weight ("field") by flat index, decoding only its chunk.
    pub fn read_weight(&self, rev: &str, name: &str, index: u64) -> Result<f64> {
        let m = self.manifest(rev)?;
        let t = m.tensor(name).with_context(|| format!("no tensor '{name}' in {rev}"))?;
        wl_io::read_element(&self.inner.repo, t, index)
    }

    // ------------------------------------------------------ diff & queries

    pub fn diff(&self, a: &str, b: &str, numeric: bool) -> Result<Diff> {
        use wl_version::Change;
        let r = wl_version::diff(&self.inner.repo, a, b, numeric)?;
        let entry = |kind: &str, t: &TensorEntry| DiffEntry {
            kind: kind.into(),
            name: t.name.clone(),
            changed_chunks: t.chunks.len() as u64,
            total_chunks: t.chunks.len() as u64,
            changed_bytes: t.nbytes(),
            shape_changed: false,
            rel_l2: None,
            cosine: None,
            max_abs: None,
        };
        let changes = r
            .changes
            .into_iter()
            .map(|c| match c {
                Change::Added(t) => entry("added", &t),
                Change::Removed(t) => entry("removed", &t),
                Change::Modified { name, changed_chunks, total_chunks, changed_bytes, shape_changed, numeric } => {
                    DiffEntry {
                        kind: "modified".into(),
                        name,
                        changed_chunks: changed_chunks as u64,
                        total_chunks: total_chunks as u64,
                        changed_bytes,
                        shape_changed,
                        rel_l2: numeric.as_ref().map(|n| n.rel_l2),
                        cosine: numeric.as_ref().map(|n| n.cosine),
                        max_abs: numeric.as_ref().map(|n| n.max_abs),
                    }
                }
            })
            .collect();
        Ok(Diff { changes, unchanged: r.unchanged as u64 })
    }

    fn engine(&self) -> Result<&wl_query::QueryEngine> {
        if let Some(q) = self.inner.query.get() {
            return Ok(q);
        }
        let q = wl_query::QueryEngine::new(self.inner.repo.clone())?;
        Ok(self.inner.query.get_or_init(|| q))
    }

    /// Run a read-only SQL query; returns Arrow record batches.
    pub fn sql(&self, query: &str) -> Result<Vec<RecordBatch>> {
        self.engine()?.sql(query)
    }

    /// Run a SQL query; returns rows as a JSON array of objects.
    pub fn sql_json(&self, query: &str) -> Result<serde_json::Value> {
        wl_query::to_json(&self.sql(query)?)
    }

    /// Run a SQL query; returns an ASCII table.
    pub fn sql_pretty(&self, query: &str) -> Result<String> {
        wl_query::pretty(&self.sql(query)?)
    }

    // ---------------------------------------------------------- maintenance

    pub fn stats(&self) -> Result<RepoStats> {
        let storage = self.inner.repo.storage_stats()?;
        let logical_bytes: u64 = self.inner.repo.commits()?.iter().map(|c| c.logical_bytes).sum();
        Ok(RepoStats {
            branches: self.inner.repo.refs()?.len() as u64,
            logical_bytes,
            dedup_ratio: logical_bytes as f64 / storage.stored_bytes.max(1) as f64,
            storage,
        })
    }

    /// Remove versions unreachable from any branch and compact packs.
    pub fn gc(&self, dry_run: bool) -> Result<GcReport> {
        self.inner.repo.gc(dry_run)
    }
}

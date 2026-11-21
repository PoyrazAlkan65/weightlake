//! Import/export between safetensors files and a WeightDB repo.

pub mod safetensors;

use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use wdb_core::{
    split_ranges, AuxFile, ChunkHash, ChunkRef, ChunkStats, Commit, DType, FileEntry, Manifest, Role, TensorEntry,
    DEFAULT_CHUNK_BYTES,
};
use wdb_store::{ImportStats, PutOptions, Repo};

/// Small non-weight files that are kept with the model (HF directory layout).
const AUX_MAX_BYTES: u64 = 64 << 20;

/// Import options. Deserializable from JSON / dicts with all fields optional:
/// `{"branch": "main", "message": "...", "base": null, "delta": true,
///   "compress": false, "chunk_bytes": 4194304, "include_aux": true}`
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ImportOptions {
    pub branch: String,
    pub message: String,
    pub chunk_bytes: usize,
    pub include_aux: bool,
    /// Revision to delta-encode against. `None` = the branch head (if any).
    pub base: Option<String>,
    /// Delta-encode changed chunks against the base version.
    pub delta: bool,
    /// zstd-compress chunks that are not delta-encoded.
    pub compress: bool,
}

impl Default for ImportOptions {
    fn default() -> Self {
        ImportOptions {
            branch: wdb_store::DEFAULT_BRANCH.to_string(),
            message: "import".to_string(),
            chunk_bytes: DEFAULT_CHUNK_BYTES,
            include_aux: true,
            base: None,
            delta: true,
            compress: false,
        }
    }
}

#[derive(serde::Serialize)]
pub struct ImportReport {
    pub commit: Commit,
    pub stats: ImportStats,
    pub files: usize,
}

/// Collect the safetensors files (and aux files) to import from `path`,
/// which may be a single `.safetensors` file or a model directory.
fn collect_inputs(path: &Path, include_aux: bool) -> Result<(Vec<PathBuf>, Vec<PathBuf>)> {
    if path.is_file() {
        return Ok((vec![path.to_path_buf()], vec![]));
    }
    if !path.is_dir() {
        bail!("{} does not exist", path.display());
    }
    let mut weights = Vec::new();
    let mut aux = Vec::new();
    for e in std::fs::read_dir(path)? {
        let e = e?;
        let p = e.path();
        if !p.is_file() {
            continue;
        }
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        if name.ends_with(".safetensors") {
            weights.push(p);
        } else if include_aux && e.metadata()?.len() <= AUX_MAX_BYTES && !name.starts_with('.') {
            aux.push(p);
        }
    }
    weights.sort();
    aux.sort();
    if weights.is_empty() {
        bail!("no .safetensors files in {}", path.display());
    }
    Ok((weights, aux))
}

/// A tensor held in memory by the caller (e.g. a PyTorch parameter).
#[derive(Debug, Clone)]
pub struct MemTensor<'a> {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
    /// Little-endian, row-major element bytes.
    pub data: &'a [u8],
}

/// One weight file to commit: tensors in data order + the original header.
struct SrcFile<'a> {
    name: String,
    /// Raw safetensors header (byte-identical export); `None` = build on export.
    header: Option<String>,
    data_len: u64,
    tensors: Vec<MemTensor<'a>>,
}

/// Import a safetensors file or HF model directory as a new commit.
pub fn import(repo: &Repo, path: &Path, opts: &ImportOptions) -> Result<ImportReport> {
    let (weight_files, aux_files) = collect_inputs(path, opts.include_aux)?;

    let mut maps = Vec::with_capacity(weight_files.len());
    for wf in &weight_files {
        let file = File::open(wf).with_context(|| format!("opening {}", wf.display()))?;
        // SAFETY: the source file is only read; concurrent modification is a user error.
        maps.push(unsafe { Mmap::map(&file)? });
    }
    let mut files = Vec::with_capacity(maps.len());
    for (wf, map) in weight_files.iter().zip(&maps) {
        let st = safetensors::parse(map).with_context(|| format!("parsing {}", wf.display()))?;
        let data = &map[st.data_start..];
        files.push(SrcFile {
            name: wf.file_name().unwrap().to_string_lossy().to_string(),
            header: Some(st.header_raw.clone()),
            data_len: data.len() as u64,
            tensors: st
                .tensors
                .iter()
                .map(|t| MemTensor {
                    name: t.name.clone(),
                    dtype: t.dtype,
                    shape: t.shape.clone(),
                    data: &data[t.begin as usize..t.end as usize],
                })
                .collect(),
        });
    }
    let mut aux = Vec::with_capacity(aux_files.len());
    for af in &aux_files {
        aux.push((af.file_name().unwrap().to_string_lossy().to_string(), std::fs::read(af)?));
    }
    commit_sources(repo, files, &aux, opts)
}

/// Commit tensors that live in memory (no files involved). They are stored as
/// one logical `model.safetensors` file; `aux` holds small side files such as
/// `config.json` or tokenizer files, as `(file name, bytes)`.
pub fn import_tensors(
    repo: &Repo,
    tensors: Vec<MemTensor<'_>>,
    aux: &[(String, Vec<u8>)],
    opts: &ImportOptions,
) -> Result<ImportReport> {
    if tensors.is_empty() {
        bail!("no tensors to commit");
    }
    let mut seen = std::collections::HashSet::new();
    for t in &tensors {
        if !seen.insert(t.name.as_str()) {
            bail!("duplicate tensor name '{}'", t.name);
        }
        let want = t.shape.iter().product::<u64>() * t.dtype.size() as u64;
        if want != t.data.len() as u64 {
            bail!(
                "tensor '{}': shape {:?} x {} needs {want} bytes, got {}",
                t.name,
                t.shape,
                t.dtype.as_str(),
                t.data.len()
            );
        }
    }
    for (name, _) in aux {
        if name.is_empty() || name.contains(['/', '\\']) || name.starts_with('.') || name.ends_with(".safetensors") {
            bail!("invalid aux file name '{name}'");
        }
    }
    let data_len = tensors.iter().map(|t| t.data.len() as u64).sum();
    let file = SrcFile { name: "model.safetensors".into(), header: None, data_len, tensors };
    commit_sources(repo, vec![file], aux, opts)
}

fn commit_sources(
    repo: &Repo,
    files: Vec<SrcFile<'_>>,
    aux: &[(String, Vec<u8>)],
    opts: &ImportOptions,
) -> Result<ImportReport> {
    // Version to delta against: explicit base, else current branch head.
    let base_commit = match &opts.base {
        Some(rev) => Some(repo.resolve(rev)?),
        None => repo.ref_get(&opts.branch)?,
    };
    let base_manifest = match (&base_commit, opts.delta) {
        (Some(id), true) => Some(repo.manifest(&repo.commit(id)?.manifest_id)?),
        _ => None,
    };
    let base_chunk = |t: &MemTensor, idx: usize| -> Option<ChunkHash> {
        let bt = base_manifest.as_ref()?.tensor(&t.name)?;
        (bt.dtype == t.dtype && bt.shape == t.shape).then(|| bt.chunks.get(idx).map(|c| c.hash)).flatten()
    };

    let mut tx = repo.begin_import()?;
    let mut manifest = Manifest::default();
    let n_files = files.len();

    for f in &files {
        // Plan all chunks of the file, then hash + compute stats in parallel.
        let mut plan = Vec::new(); // (tensor idx, chunk idx, range within the tensor)
        for (ti, t) in f.tensors.iter().enumerate() {
            let ranges = split_ranges(t.data.len(), &t.shape, t.dtype.size(), opts.chunk_bytes);
            for (ci, r) in ranges.into_iter().enumerate() {
                plan.push((ti, ci, r));
            }
        }
        let hashed: Vec<(ChunkHash, Option<ChunkStats>)> = plan
            .par_iter()
            .map(|(ti, _, r)| {
                let t = &f.tensors[*ti];
                let bytes = &t.data[r.clone()];
                (ChunkHash::of(bytes), ChunkStats::compute(t.dtype, bytes))
            })
            .collect();

        let mut entries: Vec<TensorEntry> = f
            .tensors
            .iter()
            .map(|t| TensorEntry {
                name: t.name.clone(),
                dtype: t.dtype,
                shape: t.shape.clone(),
                role: Role::infer(&t.name),
                chunks: Vec::new(),
            })
            .collect();
        for ((ti, ci, r), (hash, stats)) in plan.into_iter().zip(hashed) {
            let t = &f.tensors[ti];
            let put = PutOptions { elem: t.dtype.size(), base: base_chunk(t, ci), compress: opts.compress };
            tx.put_chunk(&hash, &t.data[r.clone()], stats, &put)?;
            entries[ti].chunks.push(ChunkRef { hash, len: r.len() as u64 });
        }

        manifest.files.push(FileEntry {
            name: f.name.clone(),
            header: f.header.clone(),
            tensors: f.tensors.iter().map(|t| t.name.clone()).collect(),
            data_len: f.data_len,
        });
        for e in entries {
            if manifest.tensors.iter().any(|x| x.name == e.name) {
                bail!("tensor '{}' appears in more than one file", e.name);
            }
            manifest.tensors.push(e);
        }
    }

    for (name, bytes) in aux {
        let mut chunks = Vec::new();
        for r in split_ranges(bytes.len(), &[bytes.len() as u64], 1, opts.chunk_bytes) {
            let hash = ChunkHash::of(&bytes[r.clone()]);
            tx.put_chunk(&hash, &bytes[r.clone()], None, &PutOptions { elem: 1, base: None, compress: opts.compress })?;
            chunks.push(ChunkRef { hash, len: r.len() as u64 });
        }
        manifest.aux.push(AuxFile { name: name.clone(), chunks });
    }

    let stats = tx.stats.clone();
    let commit = tx.commit(manifest, &opts.branch, &opts.message, base_commit)?;
    Ok(ImportReport { commit, stats, files: n_files })
}

/// Read a side file (config.json, tokenizer, ...) of a version.
pub fn read_aux(repo: &Repo, manifest: &Manifest, name: &str) -> Result<Vec<u8>> {
    let a = manifest.aux.iter().find(|a| a.name == name).with_context(|| format!("no aux file '{name}'"))?;
    let mut out = Vec::with_capacity(a.chunks.iter().map(|c| c.len as usize).sum());
    for c in &a.chunks {
        out.extend_from_slice(&repo.read_chunk(&c.hash)?);
    }
    Ok(out)
}

/// Write the tensors of `entry` (in data order) to `w`, padding any gaps.
fn write_safetensors_file(repo: &Repo, manifest: &Manifest, entry: &FileEntry, w: &mut impl Write) -> Result<u64> {
    let header = match &entry.header {
        Some(h) => h.clone(),
        None => safetensors::build_header(entry, manifest)?,
    };
    let st_tensors = safetensors::header_tensors(&header)?;
    w.write_all(&(header.len() as u64).to_le_bytes())?;
    w.write_all(header.as_bytes())?;

    let zeros = [0u8; 4096];
    let pad = |w: &mut dyn Write, mut n: u64| -> Result<()> {
        while n > 0 {
            let k = n.min(zeros.len() as u64) as usize;
            w.write_all(&zeros[..k])?;
            n -= k as u64;
        }
        Ok(())
    };

    let mut pos = 0u64;
    for t in &st_tensors {
        let te = manifest.tensor(&t.name).with_context(|| format!("tensor {} missing from manifest", t.name))?;
        if t.begin < pos {
            bail!("overlapping tensors in {}", entry.name);
        }
        pad(w, t.begin - pos)?;
        let hashes: Vec<ChunkHash> = te.chunks.iter().map(|c| c.hash).collect();
        let mut written = 0u64;
        for rec in repo.chunk_records(&hashes)? {
            let view = repo.read_record(&rec)?;
            w.write_all(&view)?;
            written += view.len() as u64;
        }
        if written != t.end - t.begin {
            bail!("size mismatch for tensor {}", t.name);
        }
        pos = t.end;
    }
    pad(w, entry.data_len.saturating_sub(pos))?;
    Ok(8 + header.len() as u64 + entry.data_len.max(pos))
}

/// Export a commit to `out_dir`, recreating the original files.
pub fn export(repo: &Repo, rev: &str, out_dir: &Path) -> Result<Vec<PathBuf>> {
    let commit = repo.commit(&repo.resolve(rev)?)?;
    let manifest = repo.manifest(&commit.manifest_id)?;
    std::fs::create_dir_all(out_dir)?;
    let mut written = Vec::new();
    for f in &manifest.files {
        let path = out_dir.join(&f.name);
        let mut w = BufWriter::with_capacity(8 << 20, File::create(&path)?);
        write_safetensors_file(repo, &manifest, f, &mut w)?;
        w.into_inner().map_err(|e| e.into_error())?.sync_all()?;
        written.push(path);
    }
    for a in &manifest.aux {
        let path = out_dir.join(&a.name);
        let mut w = BufWriter::new(File::create(&path)?);
        for c in &a.chunks {
            w.write_all(&repo.read_chunk(&c.hash)?)?;
        }
        w.flush()?;
        written.push(path);
    }
    Ok(written)
}

/// Read a whole tensor into memory (concatenating its chunks).
pub fn read_tensor(repo: &Repo, tensor: &TensorEntry) -> Result<Vec<u8>> {
    let mut out = vec![0u8; tensor.nbytes() as usize];
    read_tensor_into(repo, tensor, &mut out)?;
    Ok(out)
}

/// Read a whole tensor into a caller-provided buffer of exactly `nbytes()`.
pub fn read_tensor_into(repo: &Repo, tensor: &TensorEntry, out: &mut [u8]) -> Result<()> {
    if out.len() as u64 != tensor.nbytes() {
        bail!("buffer is {} bytes, tensor {} needs {}", out.len(), tensor.name, tensor.nbytes());
    }
    let hashes: Vec<ChunkHash> = tensor.chunks.iter().map(|c| c.hash).collect();
    let recs = repo.chunk_records(&hashes)?;
    // Decode chunks in parallel straight into the output buffer.
    let mut slices: Vec<&mut [u8]> = Vec::with_capacity(recs.len());
    let mut rest = out;
    for r in &recs {
        let (head, tail) = rest.split_at_mut(r.raw_len as usize);
        slices.push(head);
        rest = tail;
    }
    slices.into_par_iter().zip(recs.par_iter()).try_for_each(|(dst, rec)| -> Result<()> {
        dst.copy_from_slice(&repo.read_record(rec)?);
        Ok(())
    })
}

/// Read a single weight ("field") by flat element index, decoding only the
/// chunk that contains it. Returns the value as f64.
pub fn read_element(repo: &Repo, tensor: &TensorEntry, index: u64) -> Result<f64> {
    if index >= tensor.numel() {
        bail!("index {index} out of range for {} ({} elements)", tensor.name, tensor.numel());
    }
    let elem = tensor.dtype.size() as u64;
    let mut byte = index * elem;
    for c in &tensor.chunks {
        if byte < c.len {
            let view = repo.read_chunk(&c.hash)?;
            let b = &view[byte as usize..(byte + elem) as usize];
            return wdb_core::dtype::decode_f64(tensor.dtype, b).with_context(|| format!("dtype {} not numeric", tensor.dtype));
        }
        byte -= c.len;
    }
    bail!("index beyond chunk data")
}

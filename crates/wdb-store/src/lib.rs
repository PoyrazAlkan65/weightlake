//! WeightDB storage engine.
//!
//! On-disk layout:
//! ```text
//! repo/
//!   catalog.redb       chunks / manifests / commits / refs
//!   packs/000001.pack  immutable, append-once chunk files (64-byte aligned, mmap-able)
//! ```
//! Each import writes new chunks into fresh pack files; once the import's
//! catalog transaction commits, those packs are never modified again (only
//! `gc` replaces them). That keeps readers (mmap) safe while new data is written.

pub mod codec;
mod gc;
mod pack;

pub use codec::Codec;
pub use gc::GcReport;

use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use pack::PackWriter;
use redb::{Database, ReadableTable, ReadableTableMetadata, TableDefinition, WriteTransaction};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use wdb_core::{ChunkHash, ChunkStats, Commit, Manifest};

pub(crate) const CHUNKS: TableDefinition<&[u8], &[u8]> = TableDefinition::new("chunks");
pub(crate) const MANIFESTS: TableDefinition<&str, &[u8]> = TableDefinition::new("manifests");
pub(crate) const COMMITS: TableDefinition<&str, &[u8]> = TableDefinition::new("commits");
pub(crate) const REFS: TableDefinition<&str, &str> = TableDefinition::new("refs");
const META: TableDefinition<&str, &str> = TableDefinition::new("meta");

pub const FORMAT_VERSION: &str = "2";
pub const DEFAULT_BRANCH: &str = "main";

/// Physical location of a chunk's (encoded) bytes inside a pack file.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Location {
    pub pack: u32,
    pub offset: u64,
    pub len: u64,
}

/// Catalog record for a stored chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkRecord {
    pub loc: Location,
    /// Decoded length in bytes.
    pub raw_len: u64,
    pub codec: Codec,
    pub stats: Option<ChunkStats>,
}

/// A chunk's decoded bytes: zero-copy (raw chunks, mmap) or owned (decoded).
pub enum ChunkView {
    Mapped { map: Arc<Mmap>, start: usize, len: usize },
    Owned(Vec<u8>),
}

impl Deref for ChunkView {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            ChunkView::Mapped { map, start, len } => &map[*start..*start + *len],
            ChunkView::Owned(v) => v,
        }
    }
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct StorageStats {
    pub chunks: u64,
    /// Sum of decoded chunk sizes (unique data).
    pub raw_bytes: u64,
    /// Bytes actually stored in packs (after compression/delta).
    pub stored_bytes: u64,
    pub pack_files: u64,
    pub pack_bytes: u64,
    pub commits: u64,
    pub raw_chunks: u64,
    pub zstd_chunks: u64,
    pub delta_chunks: u64,
}

pub struct Repo {
    root: PathBuf,
    db: Database,
    maps: Mutex<HashMap<u32, Arc<Mmap>>>,
}

impl Repo {
    pub fn init(root: impl AsRef<Path>) -> Result<Repo> {
        let root = root.as_ref().to_path_buf();
        if root.join("catalog.redb").exists() {
            bail!("a WeightDB repo already exists at {}", root.display());
        }
        std::fs::create_dir_all(root.join("packs"))?;
        let db = Database::create(root.join("catalog.redb"))?;
        let w = db.begin_write()?;
        {
            w.open_table(CHUNKS)?;
            w.open_table(MANIFESTS)?;
            w.open_table(COMMITS)?;
            w.open_table(REFS)?;
            let mut meta = w.open_table(META)?;
            meta.insert("format_version", FORMAT_VERSION)?;
        }
        w.commit()?;
        Ok(Repo { root, db, maps: Mutex::new(HashMap::new()) })
    }

    pub fn open(root: impl AsRef<Path>) -> Result<Repo> {
        let root = root.as_ref().to_path_buf();
        let cat = root.join("catalog.redb");
        if !cat.exists() {
            bail!("no WeightDB repo at {} (run `wdb init`)", root.display());
        }
        let db = Database::open(&cat).with_context(|| format!("opening {}", cat.display()))?;
        {
            let r = db.begin_read()?;
            let meta = r.open_table(META)?;
            let v = meta.get("format_version")?.map(|v| v.value().to_string()).unwrap_or_default();
            if v != FORMAT_VERSION {
                bail!("repo format version {v} is not supported (expected {FORMAT_VERSION})");
            }
        }
        Ok(Repo { root, db, maps: Mutex::new(HashMap::new()) })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub(crate) fn packs_dir(&self) -> PathBuf {
        self.root.join("packs")
    }

    pub(crate) fn db(&self) -> &Database {
        &self.db
    }

    pub(crate) fn drop_maps(&self) {
        self.maps.lock().unwrap().clear();
    }

    // ---------------------------------------------------------------- chunks

    pub fn chunk_record(&self, hash: &ChunkHash) -> Result<Option<ChunkRecord>> {
        let r = self.db.begin_read()?;
        let t = r.open_table(CHUNKS)?;
        match t.get(&hash.0[..])? {
            Some(v) => Ok(Some(postcard::from_bytes(v.value())?)),
            None => Ok(None),
        }
    }

    /// Records for many chunks in a single read transaction.
    pub fn chunk_records(&self, hashes: &[ChunkHash]) -> Result<Vec<ChunkRecord>> {
        let r = self.db.begin_read()?;
        let t = r.open_table(CHUNKS)?;
        hashes
            .iter()
            .map(|h| {
                let v = t.get(&h.0[..])?.with_context(|| format!("missing chunk {}", h.short()))?;
                Ok(postcard::from_bytes(v.value())?)
            })
            .collect()
    }

    /// Every chunk record in the repo.
    pub fn all_chunk_records(&self) -> Result<Vec<(ChunkHash, ChunkRecord)>> {
        let r = self.db.begin_read()?;
        let t = r.open_table(CHUNKS)?;
        let mut out = Vec::new();
        for item in t.iter()? {
            let (k, v) = item?;
            let mut h = [0u8; 32];
            h.copy_from_slice(k.value());
            out.push((ChunkHash(h), postcard::from_bytes(v.value())?));
        }
        Ok(out)
    }

    pub(crate) fn map_pack(&self, pack: u32) -> Result<Arc<Mmap>> {
        let mut maps = self.maps.lock().unwrap();
        if let Some(m) = maps.get(&pack) {
            return Ok(m.clone());
        }
        let path = pack::pack_path(&self.packs_dir(), pack);
        let file = std::fs::File::open(&path).with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: pack files are immutable once their import has committed.
        let map = Arc::new(unsafe { Mmap::map(&file)? });
        maps.insert(pack, map.clone());
        Ok(map)
    }

    /// Stored (possibly encoded) bytes at a location.
    pub fn read_location(&self, loc: &Location) -> Result<ChunkView> {
        let map = self.map_pack(loc.pack)?;
        let (start, len) = (loc.offset as usize, loc.len as usize);
        if start + len > map.len() {
            bail!("chunk location out of bounds in pack {}", loc.pack);
        }
        Ok(ChunkView::Mapped { map, start, len })
    }

    /// Decoded bytes of a chunk record.
    pub fn read_record(&self, rec: &ChunkRecord) -> Result<ChunkView> {
        let stored = self.read_location(&rec.loc)?;
        let raw_len = rec.raw_len as usize;
        match rec.codec {
            Codec::Raw => Ok(stored),
            Codec::Zstd { elem } => Ok(ChunkView::Owned(codec::decode_zstd(&stored, elem as usize, raw_len)?)),
            Codec::XorZstd { base, elem } => {
                let delta = codec::decode_zstd(&stored, elem as usize, raw_len)?;
                let base = self.read_chunk(&base)?;
                if base.len() != raw_len {
                    bail!("delta base length mismatch");
                }
                Ok(ChunkView::Owned(codec::xor(&base, &delta)))
            }
        }
    }

    pub fn read_chunk(&self, hash: &ChunkHash) -> Result<ChunkView> {
        let rec = self.chunk_record(hash)?.with_context(|| format!("missing chunk {}", hash.short()))?;
        self.read_record(&rec)
    }

    // ------------------------------------------------------- commits & refs

    pub fn commit(&self, id: &str) -> Result<Commit> {
        let r = self.db.begin_read()?;
        let t = r.open_table(COMMITS)?;
        let v = t.get(id)?.with_context(|| format!("no commit {id}"))?;
        Ok(serde_json::from_slice(v.value())?)
    }

    pub fn manifest(&self, manifest_id: &str) -> Result<Manifest> {
        let r = self.db.begin_read()?;
        let t = r.open_table(MANIFESTS)?;
        let v = t.get(manifest_id)?.with_context(|| format!("no manifest {manifest_id}"))?;
        Ok(serde_json::from_slice(v.value())?)
    }

    pub fn ref_get(&self, name: &str) -> Result<Option<String>> {
        let r = self.db.begin_read()?;
        let t = r.open_table(REFS)?;
        Ok(t.get(name)?.map(|v| v.value().to_string()))
    }

    pub fn refs(&self) -> Result<Vec<(String, String)>> {
        let r = self.db.begin_read()?;
        let t = r.open_table(REFS)?;
        let mut out = Vec::new();
        for item in t.iter()? {
            let (k, v) = item?;
            out.push((k.value().to_string(), v.value().to_string()));
        }
        Ok(out)
    }

    pub fn set_ref(&self, name: &str, commit_id: &str) -> Result<()> {
        if name.is_empty() || name.contains('~') {
            bail!("invalid branch name '{name}'");
        }
        self.commit(commit_id)?;
        let w = self.db.begin_write()?;
        w.open_table(REFS)?.insert(name, commit_id)?;
        w.commit()?;
        Ok(())
    }

    pub fn delete_ref(&self, name: &str) -> Result<bool> {
        let w = self.db.begin_write()?;
        let existed = w.open_table(REFS)?.remove(name)?.is_some();
        w.commit()?;
        Ok(existed)
    }

    /// Resolve a branch name, full commit id, unique id prefix, or `rev~N`.
    pub fn resolve(&self, rev: &str) -> Result<String> {
        if let Some((base, n)) = rev.rsplit_once('~') {
            let n: usize = if n.is_empty() { 1 } else { n.parse().context("bad ~N suffix")? };
            let mut id = self.resolve(base)?;
            for _ in 0..n {
                id = self.commit(&id)?.parents.first().cloned().with_context(|| format!("{rev}: history too short"))?;
            }
            return Ok(id);
        }
        if let Some(id) = self.ref_get(rev)? {
            return Ok(id);
        }
        if rev.len() < 4 {
            bail!("unknown revision '{rev}'");
        }
        let r = self.db.begin_read()?;
        let t = r.open_table(COMMITS)?;
        let mut matches = Vec::new();
        for item in t.range(rev..)? {
            let (k, _) = item?;
            let k = k.value();
            if !k.starts_with(rev) {
                break;
            }
            matches.push(k.to_string());
        }
        match matches.len() {
            1 => Ok(matches.pop().unwrap()),
            0 => bail!("unknown revision '{rev}'"),
            _ => bail!("ambiguous revision '{rev}' ({} commits match)", matches.len()),
        }
    }

    pub fn commits(&self) -> Result<Vec<Commit>> {
        let r = self.db.begin_read()?;
        let t = r.open_table(COMMITS)?;
        let mut out = Vec::new();
        for item in t.iter()? {
            let (_, v) = item?;
            out.push(serde_json::from_slice(v.value())?);
        }
        Ok(out)
    }

    pub fn storage_stats(&self) -> Result<StorageStats> {
        let mut s = StorageStats::default();
        for (_, rec) in self.all_chunk_records()? {
            s.chunks += 1;
            s.raw_bytes += rec.raw_len;
            s.stored_bytes += rec.loc.len;
            match rec.codec {
                Codec::Raw => s.raw_chunks += 1,
                Codec::Zstd { .. } => s.zstd_chunks += 1,
                Codec::XorZstd { .. } => s.delta_chunks += 1,
            }
        }
        s.commits = self.db.begin_read()?.open_table(COMMITS)?.len()?;
        for id in pack::list_packs(&self.packs_dir())? {
            s.pack_files += 1;
            s.pack_bytes += std::fs::metadata(pack::pack_path(&self.packs_dir(), id))?.len();
        }
        Ok(s)
    }

    // ---------------------------------------------------------------- writes

    /// Start an import. Chunks are written to new pack files and become
    /// visible atomically when [`ImportTxn::commit`] succeeds.
    pub fn begin_import(&self) -> Result<ImportTxn<'_>> {
        let txn = self.db.begin_write()?;
        let writer = PackWriter::new(self.packs_dir())?;
        Ok(ImportTxn { repo: self, txn, writer, stats: ImportStats::default() })
    }
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct ImportStats {
    pub new_chunks: u64,
    /// Decoded size of newly stored chunks.
    pub new_bytes: u64,
    /// Bytes written to packs for new chunks (after compression/delta).
    pub stored_bytes: u64,
    pub dup_chunks: u64,
    pub dup_bytes: u64,
    pub delta_chunks: u64,
    pub zstd_chunks: u64,
}

/// How to store a new chunk.
#[derive(Debug, Clone, Default)]
pub struct PutOptions {
    /// Element size in bytes (for byte grouping).
    pub elem: usize,
    /// Candidate delta base (same position in the base model's tensor).
    pub base: Option<ChunkHash>,
    /// Try plain zstd compression when no delta is used.
    pub compress: bool,
}

pub struct ImportTxn<'a> {
    repo: &'a Repo,
    txn: WriteTransaction,
    writer: PackWriter,
    pub stats: ImportStats,
}

/// Decoded bytes of `base` if it can serve as a delta base (stored raw or
/// zstd, in an already-committed pack).
fn delta_base_bytes(
    repo: &Repo,
    writer: &PackWriter,
    t: &impl ReadableTable<&'static [u8], &'static [u8]>,
    base: &ChunkHash,
) -> Result<Option<ChunkView>> {
    let Some(v) = t.get(&base.0[..])? else { return Ok(None) };
    let rec: ChunkRecord = postcard::from_bytes(v.value())?;
    if matches!(rec.codec, Codec::XorZstd { .. }) || writer.is_new(rec.loc.pack) {
        return Ok(None);
    }
    Ok(Some(repo.read_record(&rec)?))
}

impl<'a> ImportTxn<'a> {
    /// Store a chunk unless an identical one already exists. Returns `true`
    /// if new bytes were written.
    pub fn put_chunk(
        &mut self,
        hash: &ChunkHash,
        data: &[u8],
        stats: Option<ChunkStats>,
        opts: &PutOptions,
    ) -> Result<bool> {
        let mut t = self.txn.open_table(CHUNKS)?;
        if t.get(&hash.0[..])?.is_some() {
            self.stats.dup_chunks += 1;
            self.stats.dup_bytes += data.len() as u64;
            return Ok(false);
        }

        let elem = opts.elem.clamp(1, 8);
        let mut codec = Codec::Raw;
        let mut encoded: Option<Vec<u8>> = None;
        if let Some(base) = opts.base.filter(|b| b != hash) {
            if let Some(bv) = delta_base_bytes(self.repo, &self.writer, &t, &base)? {
                if bv.len() == data.len() {
                    let enc = codec::encode_zstd(&codec::xor(&bv, data), elem)?;
                    if enc.len() < data.len() * 9 / 10 {
                        codec = Codec::XorZstd { base, elem: elem as u8 };
                        encoded = Some(enc);
                    }
                }
            }
        }
        if encoded.is_none() && opts.compress {
            let enc = codec::encode_zstd(data, elem)?;
            if enc.len() < data.len() * 95 / 100 {
                codec = Codec::Zstd { elem: elem as u8 };
                encoded = Some(enc);
            }
        }
        let payload = encoded.as_deref().unwrap_or(data);
        let (pack, offset) = self.writer.append(payload)?;
        let rec = ChunkRecord {
            loc: Location { pack, offset, len: payload.len() as u64 },
            raw_len: data.len() as u64,
            codec,
            stats,
        };
        t.insert(&hash.0[..], postcard::to_allocvec(&rec)?.as_slice())?;
        self.stats.new_chunks += 1;
        self.stats.new_bytes += data.len() as u64;
        self.stats.stored_bytes += payload.len() as u64;
        match codec {
            Codec::XorZstd { .. } => self.stats.delta_chunks += 1,
            Codec::Zstd { .. } => self.stats.zstd_chunks += 1,
            Codec::Raw => {}
        }
        Ok(true)
    }

    /// Persist the manifest, create a commit on `branch` and make everything
    /// visible atomically. The parent is the branch head, or `fallback_parent`
    /// when the branch is new.
    pub fn commit(
        mut self,
        mut manifest: Manifest,
        branch: &str,
        message: &str,
        fallback_parent: Option<String>,
    ) -> Result<Commit> {
        if branch.is_empty() || branch.contains('~') {
            bail!("invalid branch name '{branch}'");
        }
        manifest.sort();
        let manifest_json = serde_json::to_vec(&manifest)?;
        let manifest_id = blake3::hash(&manifest_json).to_hex().to_string();

        let head = {
            let refs = self.txn.open_table(REFS)?;
            let p = refs.get(branch)?.map(|v| v.value().to_string());
            p
        };
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let parents: Vec<String> = head.or(fallback_parent).into_iter().collect();
        let id = {
            let mut h = blake3::Hasher::new();
            h.update(manifest_id.as_bytes());
            for p in &parents {
                h.update(p.as_bytes());
            }
            h.update(message.as_bytes());
            h.update(&timestamp.to_le_bytes());
            h.update(&std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().subsec_nanos().to_le_bytes());
            h.finalize().to_hex().to_string()
        };
        let commit = Commit {
            id: id.clone(),
            parents,
            message: message.to_string(),
            timestamp,
            manifest_id: manifest_id.clone(),
            n_tensors: manifest.tensors.len() as u64,
            logical_bytes: manifest.logical_bytes(),
        };

        // Make chunk bytes durable before the catalog points at them.
        self.writer.finish()?;
        {
            self.txn.open_table(MANIFESTS)?.insert(manifest_id.as_str(), manifest_json.as_slice())?;
            self.txn.open_table(COMMITS)?.insert(id.as_str(), serde_json::to_vec(&commit)?.as_slice())?;
            self.txn.open_table(REFS)?.insert(branch, id.as_str())?;
        }
        self.txn.commit()?;
        Ok(commit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wdb_core::{ChunkRef, DType, Role, TensorEntry};

    fn manifest_for(name: &str, hashes: &[(ChunkHash, u64)]) -> Manifest {
        Manifest {
            files: vec![],
            tensors: vec![TensorEntry {
                name: name.into(),
                dtype: DType::U8,
                shape: vec![hashes.iter().map(|h| h.1).sum()],
                role: Role::Other,
                chunks: hashes.iter().map(|(h, l)| ChunkRef { hash: *h, len: *l }).collect(),
            }],
            aux: vec![],
        }
    }

    fn raw() -> PutOptions {
        PutOptions { elem: 1, ..Default::default() }
    }

    #[test]
    fn dedup_commit_and_read_back() {
        let dir = tempfile::tempdir().unwrap();
        let repo = Repo::init(dir.path()).unwrap();
        let a = vec![1u8; 1000];
        let b = vec![2u8; 77];
        let (ha, hb) = (ChunkHash::of(&a), ChunkHash::of(&b));

        let mut tx = repo.begin_import().unwrap();
        assert!(tx.put_chunk(&ha, &a, None, &raw()).unwrap());
        assert!(tx.put_chunk(&hb, &b, None, &raw()).unwrap());
        assert!(!tx.put_chunk(&ha, &a, None, &raw()).unwrap());
        let c1 = tx.commit(manifest_for("t", &[(ha, 1000), (hb, 77)]), "main", "first", None).unwrap();

        let mut tx = repo.begin_import().unwrap();
        assert!(!tx.put_chunk(&ha, &a, None, &raw()).unwrap());
        let c2 = tx.commit(manifest_for("t", &[(ha, 1000)]), "main", "second", None).unwrap();
        assert_eq!(c2.parents, vec![c1.id.clone()]);

        drop(repo);
        let repo = Repo::open(dir.path()).unwrap();
        assert_eq!(&*repo.read_chunk(&hb).unwrap(), &b[..]);
        assert_eq!(&*repo.read_chunk(&ha).unwrap(), &a[..]);
        assert_eq!(repo.resolve("main").unwrap(), c2.id);
        assert_eq!(repo.resolve("main~1").unwrap(), c1.id);
        assert_eq!(repo.resolve(&c1.id[..10]).unwrap(), c1.id);
        let s = repo.storage_stats().unwrap();
        assert_eq!(s.chunks, 2);
        assert_eq!(s.raw_bytes, 1077);
    }

    #[test]
    fn delta_and_zstd_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let repo = Repo::init(dir.path()).unwrap();
        let base: Vec<u8> = (0..20_000u32).flat_map(|i| ((i as f32 * 0.37).sin()).to_le_bytes()).collect();
        let ft: Vec<u8> = (0..20_000u32).flat_map(|i| ((i as f32 * 0.37).sin() + 1e-4).to_le_bytes()).collect();
        let (hb, hf) = (ChunkHash::of(&base), ChunkHash::of(&ft));

        let mut tx = repo.begin_import().unwrap();
        tx.put_chunk(&hb, &base, None, &PutOptions { elem: 4, base: None, compress: true }).unwrap();
        tx.commit(manifest_for("t", &[(hb, base.len() as u64)]), "main", "base", None).unwrap();

        let mut tx = repo.begin_import().unwrap();
        tx.put_chunk(&hf, &ft, None, &PutOptions { elem: 4, base: Some(hb), compress: true }).unwrap();
        assert_eq!(tx.stats.delta_chunks, 1);
        assert!(tx.stats.stored_bytes < ft.len() as u64 / 2);
        tx.commit(manifest_for("t", &[(hf, ft.len() as u64)]), "ft", "ft", None).unwrap();

        assert_eq!(&*repo.read_chunk(&hf).unwrap(), &ft[..]);
        assert_eq!(&*repo.read_chunk(&hb).unwrap(), &base[..]);
    }
}

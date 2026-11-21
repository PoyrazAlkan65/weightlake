//! Garbage collection: drop commits unreachable from any ref, drop chunks no
//! live commit needs, and rewrite packs that contain garbage.
//!
//! Crash safety: live chunks are copied into new packs and fsync'd, then one
//! catalog transaction switches locations and deletes records. Old packs are
//! deleted only after that commit; leftovers (orphan packs) are removed by the
//! next gc.

use crate::pack::{list_packs, pack_path, PackWriter};
use crate::{ChunkRecord, Codec, Location, Repo, CHUNKS, COMMITS, MANIFESTS, REFS};
use anyhow::Result;
use redb::ReadableTable;
use serde::Serialize;
use std::collections::{BTreeSet, HashMap, HashSet};
use wl_core::{ChunkHash, Commit, Manifest};

#[derive(Debug, Default, Clone, Serialize)]
pub struct GcReport {
    pub dry_run: bool,
    pub live_commits: u64,
    pub removed_commits: u64,
    pub removed_manifests: u64,
    pub live_chunks: u64,
    pub removed_chunks: u64,
    pub removed_chunk_bytes: u64,
    pub rewritten_packs: u64,
    pub moved_bytes: u64,
    pub deleted_pack_files: u64,
    pub pack_bytes_before: u64,
    pub pack_bytes_after: u64,
    /// Pack files that could not be deleted (e.g. still memory-mapped elsewhere).
    pub undeletable_packs: Vec<String>,
}

impl Repo {
    pub fn gc(&self, dry_run: bool) -> Result<GcReport> {
        let mut rep = GcReport { dry_run, ..Default::default() };
        let packs_dir = self.packs_dir();
        for id in list_packs(&packs_dir)? {
            rep.pack_bytes_before += std::fs::metadata(pack_path(&packs_dir, id))?.len();
        }

        // ---- mark
        let commits: HashMap<String, Commit> = self.commits()?.into_iter().map(|c| (c.id.clone(), c)).collect();
        let mut live_commits: HashSet<String> = HashSet::new();
        let mut stack: Vec<String> = self.refs()?.into_iter().map(|(_, id)| id).collect();
        while let Some(id) = stack.pop() {
            if let Some(c) = commits.get(&id) {
                if live_commits.insert(id) {
                    stack.extend(c.parents.iter().cloned());
                }
            }
        }
        let live_manifests: BTreeSet<String> =
            live_commits.iter().map(|id| commits[id].manifest_id.clone()).collect();
        let all_manifests: BTreeSet<String> = commits.values().map(|c| c.manifest_id.clone()).collect();

        let records: HashMap<ChunkHash, ChunkRecord> = self.all_chunk_records()?.into_iter().collect();
        let mut live: HashSet<ChunkHash> = HashSet::new();
        for mid in &live_manifests {
            let m: Manifest = self.manifest(mid)?;
            for t in &m.tensors {
                live.extend(t.chunks.iter().map(|c| c.hash));
            }
            for a in &m.aux {
                live.extend(a.chunks.iter().map(|c| c.hash));
            }
        }
        // Delta bases must survive with their dependents.
        let bases: Vec<ChunkHash> = live
            .iter()
            .filter_map(|h| match records.get(h).map(|r| r.codec) {
                Some(Codec::XorZstd { base, .. }) => Some(base),
                _ => None,
            })
            .collect();
        live.extend(bases);

        let dead: Vec<ChunkHash> = records.keys().filter(|h| !live.contains(h)).copied().collect();
        let dead_commits: Vec<String> = commits.keys().filter(|id| !live_commits.contains(*id)).cloned().collect();
        let dead_manifests: Vec<String> = all_manifests.difference(&live_manifests).cloned().collect();

        // Packs holding any dead chunk get rewritten; packs referenced by no
        // record at all are orphans.
        let mut dirty_packs: BTreeSet<u32> = BTreeSet::new();
        for h in &dead {
            dirty_packs.insert(records[h].loc.pack);
            rep.removed_chunk_bytes += records[h].raw_len;
        }
        let referenced: BTreeSet<u32> = records.values().map(|r| r.loc.pack).collect();
        let orphan_packs: Vec<u32> = list_packs(&packs_dir)?.into_iter().filter(|p| !referenced.contains(p)).collect();

        rep.live_commits = live_commits.len() as u64;
        rep.removed_commits = dead_commits.len() as u64;
        rep.removed_manifests = dead_manifests.len() as u64;
        rep.live_chunks = live.len() as u64;
        rep.removed_chunks = dead.len() as u64;
        rep.rewritten_packs = dirty_packs.len() as u64;

        let to_move: Vec<(ChunkHash, ChunkRecord)> = records
            .iter()
            .filter(|(h, r)| live.contains(*h) && dirty_packs.contains(&r.loc.pack))
            .map(|(h, r)| (*h, r.clone()))
            .collect();
        rep.moved_bytes = to_move.iter().map(|(_, r)| r.loc.len).sum();

        if dry_run {
            rep.pack_bytes_after = rep.pack_bytes_before;
            return Ok(rep);
        }

        // ---- sweep
        let w = self.db().begin_write()?;
        let mut writer = PackWriter::new(packs_dir.clone())?;
        {
            let mut t = w.open_table(CHUNKS)?;
            for (h, rec) in &to_move {
                let bytes = self.read_location(&rec.loc)?;
                let (pack, offset) = writer.append(&bytes)?;
                let new = ChunkRecord { loc: Location { pack, offset, len: rec.loc.len }, ..rec.clone() };
                t.insert(&h.0[..], postcard::to_allocvec(&new)?.as_slice())?;
            }
            for h in &dead {
                t.remove(&h.0[..])?;
            }
            let mut ct = w.open_table(COMMITS)?;
            for id in &dead_commits {
                ct.remove(id.as_str())?;
            }
            let mut mt = w.open_table(MANIFESTS)?;
            for id in &dead_manifests {
                mt.remove(id.as_str())?;
            }
            // Refs are untouched; sanity check that none dangles.
            let rt = w.open_table(REFS)?;
            for item in rt.iter()? {
                let (_, v) = item?;
                anyhow::ensure!(live_commits.contains(v.value()), "ref points to unknown commit");
            }
        }
        writer.finish()?;
        w.commit()?;

        // Old packs are no longer referenced: unmap and delete them.
        self.drop_maps();
        for p in dirty_packs.iter().chain(&orphan_packs) {
            let path = pack_path(&packs_dir, *p);
            match std::fs::remove_file(&path) {
                Ok(()) => rep.deleted_pack_files += 1,
                Err(_) => rep.undeletable_packs.push(path.display().to_string()),
            }
        }
        for id in list_packs(&packs_dir)? {
            rep.pack_bytes_after += std::fs::metadata(pack_path(&packs_dir, id))?.len();
        }
        Ok(rep)
    }
}

#[cfg(test)]
mod tests {
    use crate::{PutOptions, Repo};
    use wl_core::{ChunkHash, ChunkRef, DType, Manifest, Role, TensorEntry};

    fn manifest(data: &[u8]) -> (ChunkHash, Manifest) {
        let h = ChunkHash::of(data);
        let m = Manifest {
            tensors: vec![TensorEntry {
                name: "t".into(),
                dtype: DType::U8,
                shape: vec![data.len() as u64],
                role: Role::Other,
                chunks: vec![ChunkRef { hash: h, len: data.len() as u64 }],
            }],
            ..Default::default()
        };
        (h, m)
    }

    #[test]
    fn gc_removes_unreachable_and_keeps_live() {
        let dir = tempfile::tempdir().unwrap();
        let repo = Repo::init(dir.path()).unwrap();
        let opts = PutOptions { elem: 1, ..Default::default() };
        let a = vec![7u8; 5000];
        let b = vec![9u8; 3000];
        let (ha, ma) = manifest(&a);
        let (hb, mb) = manifest(&b);

        let mut tx = repo.begin_import().unwrap();
        tx.put_chunk(&ha, &a, None, &opts).unwrap();
        tx.commit(ma, "main", "a", None).unwrap();
        let mut tx = repo.begin_import().unwrap();
        tx.put_chunk(&hb, &b, None, &opts).unwrap();
        tx.commit(mb, "tmp", "b", None).unwrap();

        let dry = repo.gc(true).unwrap();
        assert_eq!(dry.removed_chunks, 0);

        repo.delete_ref("tmp").unwrap();
        let r = repo.gc(false).unwrap();
        assert_eq!(r.removed_commits, 1);
        assert_eq!(r.removed_chunks, 1);
        assert_eq!(r.live_chunks, 1);
        assert!(r.undeletable_packs.is_empty(), "{:?}", r.undeletable_packs);
        assert!(repo.chunk_record(&hb).unwrap().is_none());
        assert_eq!(&*repo.read_chunk(&ha).unwrap(), &a[..]);
        assert!(r.pack_bytes_after < r.pack_bytes_before);
    }
}

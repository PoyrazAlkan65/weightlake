use anyhow::Result;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// Packs are rotated once they exceed this size.
pub const MAX_PACK_BYTES: u64 = 2 << 30;
/// Chunk start offsets are aligned so mmap'd data can be viewed as any dtype.
pub const ALIGN: u64 = 64;

pub fn pack_path(dir: &Path, id: u32) -> PathBuf {
    dir.join(format!("{id:06}.pack"))
}

/// All pack ids present on disk.
pub fn list_packs(dir: &Path) -> Result<Vec<u32>> {
    let mut ids = Vec::new();
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        if p.extension().is_some_and(|x| x == "pack") {
            if let Some(n) = p.file_stem().and_then(|s| s.to_str()).and_then(|s| s.parse::<u32>().ok()) {
                ids.push(n);
            }
        }
    }
    ids.sort();
    Ok(ids)
}

/// Writes chunks into fresh pack files (created lazily, never reopened).
pub struct PackWriter {
    dir: PathBuf,
    current: Option<(u32, BufWriter<File>, u64)>,
    finished: Vec<File>,
    created: Vec<u32>,
}

impl PackWriter {
    pub fn new(dir: PathBuf) -> Result<Self> {
        Ok(PackWriter { dir, current: None, finished: Vec::new(), created: Vec::new() })
    }

    /// Whether `id` was created by this writer (and is not yet durable/readable).
    pub fn is_new(&self, id: u32) -> bool {
        self.created.contains(&id)
    }

    fn open_new(&mut self) -> Result<()> {
        let id = list_packs(&self.dir)?.last().copied().unwrap_or(0).max(self.created.last().copied().unwrap_or(0)) + 1;
        let file = OpenOptions::new().write(true).create_new(true).open(pack_path(&self.dir, id))?;
        if let Some((_, w, _)) = self.current.take() {
            self.finished.push(w.into_inner().map_err(|e| e.into_error())?);
        }
        self.created.push(id);
        self.current = Some((id, BufWriter::with_capacity(8 << 20, file), 0));
        Ok(())
    }

    /// Append `data`; returns (pack id, offset).
    pub fn append(&mut self, data: &[u8]) -> Result<(u32, u64)> {
        let need_new = match &self.current {
            None => true,
            Some((_, _, pos)) => *pos > 0 && *pos + data.len() as u64 > MAX_PACK_BYTES,
        };
        if need_new {
            self.open_new()?;
        }
        let (id, w, pos) = self.current.as_mut().unwrap();
        let pad = (ALIGN - *pos % ALIGN) % ALIGN;
        if pad > 0 {
            w.write_all(&[0u8; ALIGN as usize][..pad as usize])?;
            *pos += pad;
        }
        let offset = *pos;
        w.write_all(data)?;
        *pos += data.len() as u64;
        Ok((*id, offset))
    }

    /// Flush and fsync all packs written by this writer.
    pub fn finish(&mut self) -> Result<()> {
        if let Some((_, w, _)) = self.current.take() {
            self.finished.push(w.into_inner().map_err(|e| e.into_error())?);
        }
        for f in self.finished.drain(..) {
            f.sync_all()?;
        }
        Ok(())
    }
}

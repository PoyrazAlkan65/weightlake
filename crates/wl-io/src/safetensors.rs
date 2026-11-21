//! Minimal safetensors header parser/writer. We keep the raw header so that
//! exports are byte-identical to the imported files.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use wl_core::{DType, FileEntry, Manifest};

#[derive(Debug, Clone)]
pub struct StTensor {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
    /// Offsets relative to the start of the data section.
    pub begin: u64,
    pub end: u64,
}

pub struct StFile {
    pub header_raw: String,
    pub data_start: usize,
    /// Sorted by data offset.
    pub tensors: Vec<StTensor>,
}

#[derive(Deserialize)]
struct Info {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

/// Parse tensors out of a header JSON string, sorted by data offset.
pub fn header_tensors(header: &str) -> Result<Vec<StTensor>> {
    let map: serde_json::Map<String, serde_json::Value> = serde_json::from_str(header).context("invalid header JSON")?;
    let mut out = Vec::with_capacity(map.len());
    for (name, v) in map {
        if name == "__metadata__" {
            continue;
        }
        let info: Info = serde_json::from_value(v).with_context(|| format!("bad entry for {name}"))?;
        let dtype = DType::parse(&info.dtype).with_context(|| format!("unsupported dtype {}", info.dtype))?;
        let [begin, end] = info.data_offsets;
        let expected = info.shape.iter().product::<u64>() * dtype.size() as u64;
        if end < begin || end - begin != expected {
            bail!("tensor {name}: offsets {begin}..{end} do not match shape {:?} {dtype}", info.shape);
        }
        out.push(StTensor { name, dtype, shape: info.shape, begin, end });
    }
    out.sort_by_key(|t| (t.begin, t.end));
    Ok(out)
}

pub fn parse(bytes: &[u8]) -> Result<StFile> {
    if bytes.len() < 8 {
        bail!("file too small to be safetensors");
    }
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    if n > 100 << 20 || 8 + n > bytes.len() {
        bail!("invalid safetensors header length {n}");
    }
    let header_raw = std::str::from_utf8(&bytes[8..8 + n]).context("header is not UTF-8")?.to_string();
    let tensors = header_tensors(&header_raw)?;
    let data_len = (bytes.len() - 8 - n) as u64;
    if let Some(t) = tensors.iter().find(|t| t.end > data_len) {
        bail!("tensor {} extends past end of file", t.name);
    }
    Ok(StFile { header_raw, data_start: 8 + n, tensors })
}

/// Build a fresh header for a file whose original header is not stored.
pub fn build_header(entry: &FileEntry, manifest: &Manifest) -> Result<String> {
    let mut map = serde_json::Map::new();
    let mut off = 0u64;
    for name in &entry.tensors {
        let t = manifest.tensor(name).with_context(|| format!("tensor {name} missing"))?;
        let end = off + t.nbytes();
        map.insert(
            name.clone(),
            serde_json::json!({ "dtype": t.dtype.as_str(), "shape": t.shape, "data_offsets": [off, end] }),
        );
        off = end;
    }
    // transformers refuses safetensors files without a known "format".
    map.insert("__metadata__".into(), serde_json::json!({ "format": "pt" }));
    let mut s = serde_json::to_string(&map)?;
    while s.len() % 8 != 0 {
        s.push(' ');
    }
    Ok(s)
}

/// Serialize named tensors into safetensors bytes (used by tests/tools).
pub fn serialize(tensors: &[(&str, DType, Vec<u64>, Vec<u8>)]) -> Vec<u8> {
    let mut map = serde_json::Map::new();
    let mut off = 0u64;
    for (name, dtype, shape, data) in tensors {
        let end = off + data.len() as u64;
        map.insert(
            name.to_string(),
            serde_json::json!({ "dtype": dtype.as_str(), "shape": shape, "data_offsets": [off, end] }),
        );
        off = end;
    }
    map.insert("__metadata__".into(), serde_json::json!({ "format": "pt" }));
    let mut header = serde_json::to_string(&map).unwrap();
    while header.len() % 8 != 0 {
        header.push(' ');
    }
    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for (_, _, _, data) in tensors {
        out.extend_from_slice(data);
    }
    out
}

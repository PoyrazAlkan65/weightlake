//! Chunk encodings.
//!
//! - `Raw`: bytes as-is (zero-copy mmap reads).
//! - `Zstd`: byte-grouped + zstd. Grouping puts byte k of every element into
//!   plane k; float exponent bytes are highly repetitive, so planes compress well.
//! - `XorZstd`: XOR against a base chunk (same tensor in the base model), then
//!   byte-grouped + zstd. Fine-tuned weights share sign/exponent/high mantissa
//!   bits with their base, so the XOR is mostly zero bytes.

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use wl_core::ChunkHash;

pub const ZSTD_LEVEL: i32 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Codec {
    Raw,
    Zstd { elem: u8 },
    XorZstd { base: ChunkHash, elem: u8 },
}

impl Codec {
    pub fn name(&self) -> &'static str {
        match self {
            Codec::Raw => "raw",
            Codec::Zstd { .. } => "zstd",
            Codec::XorZstd { .. } => "xor-zstd",
        }
    }
}

fn plane_count(len: usize, elem: usize) -> usize {
    if elem > 1 && len % elem == 0 {
        elem
    } else {
        1
    }
}

pub fn group(data: &[u8], elem: usize) -> Vec<u8> {
    let e = plane_count(data.len(), elem);
    if e == 1 {
        return data.to_vec();
    }
    let n = data.len() / e;
    let mut out = vec![0u8; data.len()];
    for (i, el) in data.chunks_exact(e).enumerate() {
        for (k, b) in el.iter().enumerate() {
            out[k * n + i] = *b;
        }
    }
    out
}

pub fn ungroup(data: &[u8], elem: usize) -> Vec<u8> {
    let e = plane_count(data.len(), elem);
    if e == 1 {
        return data.to_vec();
    }
    let n = data.len() / e;
    let mut out = vec![0u8; data.len()];
    for (i, el) in out.chunks_exact_mut(e).enumerate() {
        for (k, b) in el.iter_mut().enumerate() {
            *b = data[k * n + i];
        }
    }
    out
}

pub fn xor(a: &[u8], b: &[u8]) -> Vec<u8> {
    a.iter().zip(b).map(|(x, y)| x ^ y).collect()
}

pub fn encode_zstd(data: &[u8], elem: usize) -> Result<Vec<u8>> {
    Ok(zstd::bulk::compress(&group(data, elem), ZSTD_LEVEL)?)
}

pub fn decode_zstd(enc: &[u8], elem: usize, raw_len: usize) -> Result<Vec<u8>> {
    let g = zstd::bulk::decompress(enc, raw_len)?;
    ensure!(g.len() == raw_len, "decompressed size mismatch");
    Ok(ungroup(&g, elem))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_roundtrip() {
        let d: Vec<u8> = (0..400u32).map(|i| (i * 7) as u8).collect();
        for e in [1, 2, 4, 3, 8] {
            assert_eq!(ungroup(&group(&d, e), e), d);
        }
        let odd = vec![1u8, 2, 3, 4, 5];
        assert_eq!(ungroup(&group(&odd, 2), 2), odd);
    }

    #[test]
    fn zstd_roundtrip() {
        let d: Vec<u8> = (0..10_000u32).flat_map(|i| ((i as f32) * 1e-3).to_le_bytes()).collect();
        let enc = encode_zstd(&d, 4).unwrap();
        assert!(enc.len() < d.len());
        assert_eq!(decode_zstd(&enc, 4, d.len()).unwrap(), d);
    }
}

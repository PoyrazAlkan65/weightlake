//! WeightDB core types.
//!
//! Data model (MongoDB analogy):
//! - Repo      ~ database   : a model family
//! - Tensor    ~ collection : one weight matrix (`layers.0.attn.q_proj.weight`)
//! - Chunk     ~ document   : a row block of a tensor, content-addressed by BLAKE3
//! - Weight    ~ field      : the individual values inside a chunk
//! - Commit                 : a model version (manifest of tensors -> chunk hashes)

pub mod chunk;
pub mod dtype;
pub mod hash;
pub mod manifest;
pub mod stats;

pub use chunk::{split_ranges, DEFAULT_CHUNK_BYTES};
pub use dtype::DType;
pub use hash::ChunkHash;
pub use manifest::{AuxFile, ChunkRef, Commit, FileEntry, Manifest, Role, TensorEntry};
pub use stats::ChunkStats;

/// Format a unix timestamp as `YYYY-MM-DD HH:MM:SS` (UTC).
pub fn format_ts(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    // Civil-from-days (Howard Hinnant).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02} {:02}:{:02}:{:02}", rem / 3600, rem % 3600 / 60, rem % 60)
}

#[cfg(test)]
mod tests {
    #[test]
    fn ts() {
        assert_eq!(super::format_ts(0), "1970-01-01 00:00:00");
        assert_eq!(super::format_ts(1_795_218_068), "2026-11-20 23:41:08");
    }
}

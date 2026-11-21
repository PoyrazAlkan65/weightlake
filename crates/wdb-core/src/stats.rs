use crate::DType;
use serde::{Deserialize, Serialize};

/// Per-chunk statistics ("zone map"). Sums are stored instead of means so that
/// chunk stats merge exactly into tensor stats.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChunkStats {
    pub count: u64,
    pub nan: u64,
    pub inf: u64,
    pub zeros: u64,
    pub min: f64,
    pub max: f64,
    pub sum: f64,
    pub sum_sq: f64,
    pub abs_sum: f64,
}

impl Default for ChunkStats {
    fn default() -> Self {
        ChunkStats {
            count: 0,
            nan: 0,
            inf: 0,
            zeros: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            sum: 0.0,
            sum_sq: 0.0,
            abs_sum: 0.0,
        }
    }
}

macro_rules! scan {
    ($s:ident, $data:ident, $n:literal, |$c:ident| $conv:expr) => {
        for $c in $data.chunks_exact($n) {
            $s.push($conv);
        }
    };
}

impl ChunkStats {
    #[inline]
    fn push(&mut self, v: f64) {
        self.count += 1;
        if v.is_nan() {
            self.nan += 1;
            return;
        }
        if v.is_infinite() {
            self.inf += 1;
            return;
        }
        if v == 0.0 {
            self.zeros += 1;
        }
        if v < self.min {
            self.min = v;
        }
        if v > self.max {
            self.max = v;
        }
        self.sum += v;
        self.sum_sq += v * v;
        self.abs_sum += v.abs();
    }

    /// Compute statistics for raw little-endian data. Returns `None` for
    /// dtypes without a numeric interpretation here (BOOL, FP8).
    pub fn compute(dtype: DType, data: &[u8]) -> Option<ChunkStats> {
        let mut s = ChunkStats::default();
        match dtype {
            DType::F32 => scan!(s, data, 4, |c| f32::from_le_bytes(c.try_into().unwrap()) as f64),
            DType::F64 => scan!(s, data, 8, |c| f64::from_le_bytes(c.try_into().unwrap())),
            DType::F16 => scan!(s, data, 2, |c| half::f16::from_le_bytes([c[0], c[1]]).to_f64()),
            DType::BF16 => scan!(s, data, 2, |c| half::bf16::from_le_bytes([c[0], c[1]]).to_f64()),
            DType::U8 => scan!(s, data, 1, |c| c[0] as f64),
            DType::I8 => scan!(s, data, 1, |c| c[0] as i8 as f64),
            DType::I16 => scan!(s, data, 2, |c| i16::from_le_bytes([c[0], c[1]]) as f64),
            DType::U16 => scan!(s, data, 2, |c| u16::from_le_bytes([c[0], c[1]]) as f64),
            DType::I32 => scan!(s, data, 4, |c| i32::from_le_bytes(c.try_into().unwrap()) as f64),
            DType::U32 => scan!(s, data, 4, |c| u32::from_le_bytes(c.try_into().unwrap()) as f64),
            DType::I64 => scan!(s, data, 8, |c| i64::from_le_bytes(c.try_into().unwrap()) as f64),
            DType::U64 => scan!(s, data, 8, |c| u64::from_le_bytes(c.try_into().unwrap()) as f64),
            DType::BOOL | DType::F8_E4M3 | DType::F8_E5M2 => return None,
        }
        Some(s)
    }

    pub fn merge(&mut self, o: &ChunkStats) {
        self.count += o.count;
        self.nan += o.nan;
        self.inf += o.inf;
        self.zeros += o.zeros;
        self.min = self.min.min(o.min);
        self.max = self.max.max(o.max);
        self.sum += o.sum;
        self.sum_sq += o.sum_sq;
        self.abs_sum += o.abs_sum;
    }

    pub fn finite(&self) -> u64 {
        self.count - self.nan - self.inf
    }

    pub fn mean(&self) -> f64 {
        self.sum / self.finite().max(1) as f64
    }

    pub fn std(&self) -> f64 {
        let n = self.finite().max(1) as f64;
        let m = self.sum / n;
        (self.sum_sq / n - m * m).max(0.0).sqrt()
    }

    pub fn l2(&self) -> f64 {
        self.sum_sq.sqrt()
    }

    pub fn abs_mean(&self) -> f64 {
        self.abs_sum / self.finite().max(1) as f64
    }

    pub fn sparsity(&self) -> f64 {
        self.zeros as f64 / self.count.max(1) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_stats_and_merge() {
        let a: Vec<u8> = [1.0f32, -2.0, 0.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let b: Vec<u8> = [f32::NAN, 4.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut s = ChunkStats::compute(DType::F32, &a).unwrap();
        s.merge(&ChunkStats::compute(DType::F32, &b).unwrap());
        assert_eq!(s.count, 5);
        assert_eq!(s.nan, 1);
        assert_eq!(s.zeros, 1);
        assert_eq!(s.min, -2.0);
        assert_eq!(s.max, 4.0);
        assert_eq!(s.mean(), 0.75);
        assert_eq!(s.l2(), 21f64.sqrt());
    }

    #[test]
    fn bf16_stats() {
        let d: Vec<u8> = [half::bf16::from_f32(0.5), half::bf16::from_f32(-1.5)]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let s = ChunkStats::compute(DType::BF16, &d).unwrap();
        assert_eq!(s.sum, -1.0);
    }
}

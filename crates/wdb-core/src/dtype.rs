use serde::{Deserialize, Serialize};

/// Element type of a tensor. Variant names match the safetensors spelling.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    BOOL,
    U8,
    I8,
    F8_E4M3,
    F8_E5M2,
    I16,
    U16,
    F16,
    BF16,
    I32,
    U32,
    F32,
    I64,
    U64,
    F64,
}

impl DType {
    /// Size of one element in bytes.
    pub fn size(self) -> usize {
        use DType::*;
        match self {
            BOOL | U8 | I8 | F8_E4M3 | F8_E5M2 => 1,
            I16 | U16 | F16 | BF16 => 2,
            I32 | U32 | F32 => 4,
            I64 | U64 | F64 => 8,
        }
    }

    pub fn as_str(self) -> &'static str {
        use DType::*;
        match self {
            BOOL => "BOOL",
            U8 => "U8",
            I8 => "I8",
            F8_E4M3 => "F8_E4M3",
            F8_E5M2 => "F8_E5M2",
            I16 => "I16",
            U16 => "U16",
            F16 => "F16",
            BF16 => "BF16",
            I32 => "I32",
            U32 => "U32",
            F32 => "F32",
            I64 => "I64",
            U64 => "U64",
            F64 => "F64",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        use DType::*;
        Some(match s {
            "BOOL" => BOOL,
            "U8" => U8,
            "I8" => I8,
            "F8_E4M3" => F8_E4M3,
            "F8_E5M2" => F8_E5M2,
            "I16" => I16,
            "U16" => U16,
            "F16" => F16,
            "BF16" => BF16,
            "I32" => I32,
            "U32" => U32,
            "F32" => F32,
            "I64" => I64,
            "U64" => U64,
            "F64" => F64,
            _ => return None,
        })
    }

    pub fn is_float(self) -> bool {
        use DType::*;
        matches!(self, F8_E4M3 | F8_E5M2 | F16 | BF16 | F32 | F64)
    }
}

/// Decode one little-endian element as f64 (`None` for BOOL/FP8).
pub fn decode_f64(dtype: DType, b: &[u8]) -> Option<f64> {
    use DType::*;
    Some(match dtype {
        F32 => f32::from_le_bytes(b.try_into().ok()?) as f64,
        F64 => f64::from_le_bytes(b.try_into().ok()?),
        F16 => half::f16::from_le_bytes(b.try_into().ok()?).to_f64(),
        BF16 => half::bf16::from_le_bytes(b.try_into().ok()?).to_f64(),
        U8 => b[0] as f64,
        I8 => b[0] as i8 as f64,
        I16 => i16::from_le_bytes(b.try_into().ok()?) as f64,
        U16 => u16::from_le_bytes(b.try_into().ok()?) as f64,
        I32 => i32::from_le_bytes(b.try_into().ok()?) as f64,
        U32 => u32::from_le_bytes(b.try_into().ok()?) as f64,
        I64 => i64::from_le_bytes(b.try_into().ok()?) as f64,
        U64 => u64::from_le_bytes(b.try_into().ok()?) as f64,
        BOOL | F8_E4M3 | F8_E5M2 => return None,
    })
}

/// Decode a whole buffer as f64 values (`None` for BOOL/FP8).
pub fn decode_all_f64(dtype: DType, data: &[u8]) -> Option<Vec<f64>> {
    let n = dtype.size();
    data.chunks_exact(n).map(|c| decode_f64(dtype, c)).collect()
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

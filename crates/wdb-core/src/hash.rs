use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Content address of a chunk: BLAKE3 of its raw bytes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChunkHash(pub [u8; 32]);

impl ChunkHash {
    pub fn of(data: &[u8]) -> Self {
        ChunkHash(*blake3::hash(data).as_bytes())
    }

    pub fn to_hex(&self) -> String {
        let mut s = String::with_capacity(64);
        for b in self.0 {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }

    pub fn short(&self) -> String {
        self.to_hex()[..12].to_string()
    }

    pub fn from_hex(s: &str) -> anyhow::Result<Self> {
        anyhow::ensure!(s.len() == 64, "chunk hash must be 64 hex chars");
        let mut out = [0u8; 32];
        for (i, byte) in out.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&s[2 * i..2 * i + 2], 16)?;
        }
        Ok(ChunkHash(out))
    }
}

impl std::fmt::Debug for ChunkHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ChunkHash({})", self.short())
    }
}

impl Serialize for ChunkHash {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for ChunkHash {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        ChunkHash::from_hex(&s).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_roundtrip() {
        let h = ChunkHash::of(b"weights");
        assert_eq!(ChunkHash::from_hex(&h.to_hex()).unwrap(), h);
    }
}

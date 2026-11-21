use crate::{ChunkHash, DType};
use serde::{Deserialize, Serialize};

/// Coarse functional role of a tensor, inferred from its name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    Embedding,
    Attention,
    Mlp,
    Norm,
    Head,
    Other,
}

impl Role {
    pub fn infer(name: &str) -> Role {
        let n = name.to_lowercase();
        let has = |keys: &[&str]| keys.iter().any(|k| n.contains(k));
        if has(&["lm_head"]) || n == "output.weight" {
            Role::Head
        } else if has(&["embed", "wte", "wpe", "tok_embeddings"]) {
            Role::Embedding
        } else if has(&["norm", ".ln_", "ln_f", "layernorm"]) || n.starts_with("ln_") {
            Role::Norm
        } else if has(&["attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj"]) {
            Role::Attention
        } else if has(&["mlp", "ffn", "feed_forward", "gate_proj", "up_proj", "down_proj"]) {
            Role::Mlp
        } else {
            Role::Other
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Role::Embedding => "embedding",
            Role::Attention => "attention",
            Role::Mlp => "mlp",
            Role::Norm => "norm",
            Role::Head => "head",
            Role::Other => "other",
        }
    }
}

/// Reference from a tensor to one of its chunks ("document").
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkRef {
    pub hash: ChunkHash,
    pub len: u64,
}

/// A tensor ("collection"): metadata plus the ordered list of its chunks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorEntry {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub role: Role,
    pub chunks: Vec<ChunkRef>,
}

impl TensorEntry {
    pub fn nbytes(&self) -> u64 {
        self.chunks.iter().map(|c| c.len).sum()
    }

    pub fn numel(&self) -> u64 {
        self.shape.iter().product()
    }
}

/// Layout of one original safetensors file, so export is byte-identical.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    pub name: String,
    /// Raw JSON header exactly as found in the source file.
    pub header: Option<String>,
    /// Tensor names in data-offset order.
    pub tensors: Vec<String>,
    /// Length of the data section (after the header).
    pub data_len: u64,
}

/// Non-weight file kept alongside the model (config.json, tokenizer, ...).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuxFile {
    pub name: String,
    pub chunks: Vec<ChunkRef>,
}

/// Full content of a model version.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Manifest {
    pub files: Vec<FileEntry>,
    /// Sorted by name.
    pub tensors: Vec<TensorEntry>,
    #[serde(default)]
    pub aux: Vec<AuxFile>,
}

impl Manifest {
    pub fn tensor(&self, name: &str) -> Option<&TensorEntry> {
        self.tensors
            .binary_search_by(|t| t.name.as_str().cmp(name))
            .ok()
            .map(|i| &self.tensors[i])
    }

    pub fn logical_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.nbytes()).sum()
    }

    pub fn sort(&mut self) {
        self.tensors.sort_by(|a, b| a.name.cmp(&b.name));
    }
}

/// A model version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Commit {
    pub id: String,
    pub parents: Vec<String>,
    pub message: String,
    pub timestamp: u64,
    pub manifest_id: String,
    pub n_tensors: u64,
    pub logical_bytes: u64,
}

impl Commit {
    pub fn short(&self) -> &str {
        &self.id[..12.min(self.id.len())]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roles() {
        assert_eq!(Role::infer("model.layers.0.self_attn.q_proj.weight"), Role::Attention);
        assert_eq!(Role::infer("model.layers.0.post_attention_layernorm.weight"), Role::Norm);
        assert_eq!(Role::infer("model.layers.0.mlp.down_proj.weight"), Role::Mlp);
        assert_eq!(Role::infer("model.embed_tokens.weight"), Role::Embedding);
        assert_eq!(Role::infer("lm_head.weight"), Role::Head);
        assert_eq!(Role::infer("h.0.attn.c_attn.weight"), Role::Attention);
        assert_eq!(Role::infer("h.0.ln_1.weight"), Role::Norm);
    }
}

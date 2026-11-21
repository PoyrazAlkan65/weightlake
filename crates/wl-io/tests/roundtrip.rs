use half::bf16;
use wl_core::DType;
use wl_io::{export, import, import_tensors, safetensors::serialize, ImportOptions, MemTensor};
use wl_store::Repo;

fn bf16_bytes(n: usize, seed: f32) -> Vec<u8> {
    (0..n).flat_map(|i| bf16::from_f32(((i as f32) * 0.001 + seed).sin()).to_le_bytes()).collect()
}

fn model(seed_mlp: f32) -> Vec<u8> {
    serialize(&[
        ("model.embed_tokens.weight", DType::BF16, vec![512, 256], bf16_bytes(512 * 256, 0.1)),
        ("model.layers.0.self_attn.q_proj.weight", DType::BF16, vec![256, 256], bf16_bytes(256 * 256, 0.2)),
        ("model.layers.0.mlp.up_proj.weight", DType::BF16, vec![1024, 256], bf16_bytes(1024 * 256, seed_mlp)),
        ("model.norm.weight", DType::F32, vec![256], vec![0u8; 1024]),
    ])
}

#[test]
fn import_export_is_byte_identical_and_dedups() {
    let tmp = tempfile::tempdir().unwrap();
    let repo = Repo::init(tmp.path().join("repo")).unwrap();

    // Base model directory with an aux file.
    let base_dir = tmp.path().join("base");
    std::fs::create_dir_all(&base_dir).unwrap();
    let base = model(0.3);
    std::fs::write(base_dir.join("model.safetensors"), &base).unwrap();
    std::fs::write(base_dir.join("config.json"), br#"{"hidden_size":256}"#).unwrap();

    let opts = ImportOptions { chunk_bytes: 64 << 10, ..Default::default() };
    let r1 = import(&repo, &base_dir, &opts).unwrap();
    assert_eq!(r1.stats.dup_chunks, 0);

    // "Fine-tune": only the MLP tensor changes.
    let ft_path = tmp.path().join("ft.safetensors");
    std::fs::write(&ft_path, model(0.9)).unwrap();
    let r2 = import(&repo, &ft_path, &ImportOptions { branch: "ft".into(), ..opts.clone() }).unwrap();
    let mlp_bytes = 1024 * 256 * 2;
    assert_eq!(r2.stats.new_bytes, mlp_bytes as u64, "only the changed tensor should be stored");
    assert!(r2.stats.dup_bytes > 0);

    // Export both and compare bytes.
    let out = tmp.path().join("out_base");
    export(&repo, "main", &out).unwrap();
    assert_eq!(std::fs::read(out.join("model.safetensors")).unwrap(), base);
    assert_eq!(std::fs::read(out.join("config.json")).unwrap(), br#"{"hidden_size":256}"#);

    let out2 = tmp.path().join("out_ft");
    export(&repo, "ft", &out2).unwrap();
    assert_eq!(std::fs::read(out2.join("ft.safetensors")).unwrap(), model(0.9));

    // Tensor-level stats are available without reading weights.
    let m = repo.manifest(&repo.commit(&repo.resolve("main").unwrap()).unwrap().manifest_id).unwrap();
    let norm = m.tensor("model.norm.weight").unwrap();
    let rec = repo.chunk_record(&norm.chunks[0].hash).unwrap().unwrap();
    assert_eq!(rec.stats.unwrap().zeros, 256);
}

type Owned = (&'static str, DType, Vec<u64>, Vec<u8>);

fn to_mem(v: &[Owned]) -> Vec<MemTensor<'_>> {
    v.iter().map(|(n, d, s, b)| MemTensor { name: n.to_string(), dtype: *d, shape: s.clone(), data: b }).collect()
}

#[test]
fn import_tensors_from_memory_dedups_and_exports_loadable_file() {
    let tmp = tempfile::tempdir().unwrap();
    let repo = Repo::init(tmp.path().join("repo")).unwrap();
    let opts = ImportOptions { chunk_bytes: 64 << 10, ..Default::default() };

    let dir = tmp.path().join("base");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("model.safetensors"), model(0.3)).unwrap();
    import(&repo, &dir, &opts).unwrap();

    // Same weights, from memory: everything dedups.
    let emb = bf16_bytes(512 * 256, 0.1);
    let q = bf16_bytes(256 * 256, 0.2);
    let up = bf16_bytes(1024 * 256, 0.3);
    let norm = vec![0u8; 1024];
    let mem = |up: &[u8]| -> Vec<Owned> {
        vec![
            ("model.embed_tokens.weight", DType::BF16, vec![512, 256], emb.clone()),
            ("model.layers.0.self_attn.q_proj.weight", DType::BF16, vec![256, 256], q.clone()),
            ("model.layers.0.mlp.up_proj.weight", DType::BF16, vec![1024, 256], up.to_vec()),
            ("model.norm.weight", DType::F32, vec![256], norm.clone()),
        ]
    };
    let same = mem(&up);
    let aux = vec![("config.json".to_string(), b"{}".to_vec())];
    let r = import_tensors(&repo, to_mem(&same), &aux, &ImportOptions { branch: "mem".into(), ..opts.clone() }).unwrap();
    assert_eq!(r.stats.new_bytes, 2, "only config.json is new");

    // Changed MLP -> delta against the base.
    let changed = mem(&bf16_bytes(1024 * 256, 0.9));
    let r = import_tensors(&repo, to_mem(&changed), &[], &ImportOptions { branch: "mem".into(), ..opts.clone() }).unwrap();
    assert!(r.stats.delta_chunks > 0);

    // Export writes a fresh, standard safetensors file with format=pt.
    let out = tmp.path().join("out");
    export(&repo, "mem", &out).unwrap();
    let bytes = std::fs::read(out.join("model.safetensors")).unwrap();
    assert_eq!(bytes, serialize(&changed));

    // Validation.
    let bad = vec![MemTensor { name: "x".into(), dtype: DType::F32, shape: vec![3], data: &[0u8; 8] }];
    assert!(import_tensors(&repo, bad, &[], &opts).is_err());
    let dup = vec![
        MemTensor { name: "x".into(), dtype: DType::U8, shape: vec![1], data: &[0u8] },
        MemTensor { name: "x".into(), dtype: DType::U8, shape: vec![1], data: &[1u8] },
    ];
    assert!(import_tensors(&repo, dup, &[], &opts).is_err());
}

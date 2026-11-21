use half::bf16;
use weightlake::{Database, DType, ImportOptions};

fn bf16_bytes(n: usize, f: impl Fn(usize) -> f32) -> Vec<u8> {
    (0..n).flat_map(|i| bf16::from_f32(f(i)).to_le_bytes()).collect()
}

fn write_model(path: &std::path::Path, shift: f32) {
    let bytes = wl_io::safetensors::serialize(&[
        ("model.embed_tokens.weight", DType::BF16, vec![256, 128], bf16_bytes(256 * 128, |i| (i as f32 * 0.01).sin())),
        (
            "model.layers.0.mlp.up_proj.weight",
            DType::BF16,
            vec![512, 128],
            bf16_bytes(512 * 128, |i| (i as f32 * 0.003).cos() + shift),
        ),
        ("model.norm.weight", DType::F32, vec![128], (0..128).flat_map(|i| (i as f32).to_le_bytes()).collect()),
    ]);
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn end_to_end_api() {
    let tmp = tempfile::tempdir().unwrap();
    let db = Database::init(tmp.path().join("repo")).unwrap();
    write_model(&tmp.path().join("a.safetensors"), 0.0);
    write_model(&tmp.path().join("b.safetensors"), 0.01);

    let opts = ImportOptions { chunk_bytes: 16 << 10, ..Default::default() };
    db.import(tmp.path().join("a.safetensors"), &ImportOptions { message: "base".into(), ..opts.clone() }).unwrap();
    // Same branch -> delta against the previous head.
    let r = db.import(tmp.path().join("b.safetensors"), &ImportOptions { message: "tuned".into(), ..opts }).unwrap();
    assert!(r.stats.delta_chunks > 0, "{:?}", r.stats);
    assert!(r.stats.stored_bytes < r.stats.new_bytes);

    let ts = db.tensors("main").unwrap();
    assert_eq!(ts.len(), 3);
    let norm = db.read_tensor("main", "model.norm.weight").unwrap();
    assert_eq!(norm.to_f64().unwrap()[5], 5.0);
    assert_eq!(db.read_weight("main", "model.norm.weight", 7).unwrap(), 7.0);
    // Delta-decoded tensor equals the original file content.
    let up = db.read_tensor("main", "model.layers.0.mlp.up_proj.weight").unwrap();
    assert_eq!(up.data, bf16_bytes(512 * 128, |i| (i as f32 * 0.003).cos() + 0.01));

    let d = db.diff("main~1", "main", true).unwrap();
    assert_eq!(d.changes.len(), 1);
    assert!(d.changes[0].cosine.unwrap() > 0.99);

    // SQL
    let rows = db
        .sql_json("SELECT name, numel, role FROM tensors WHERE commit_id = rev('main') ORDER BY numel DESC")
        .unwrap();
    assert_eq!(rows[0]["name"], "model.layers.0.mlp.up_proj.weight");
    assert_eq!(rows[0]["numel"], 512 * 128);
    let rows = db.sql_json("SELECT weight('main', 'model.norm.weight', 3) AS w").unwrap();
    assert_eq!(rows[0]["w"], 3.0);
    let rows = db
        .sql_json("SELECT tensor_cosine('main~1', 'main', 'model.layers.0.mlp.up_proj.weight') AS c")
        .unwrap();
    assert!(rows[0]["c"].as_f64().unwrap() > 0.99);
    let rows = db.sql_json("SELECT codec, count(*) AS n FROM chunks GROUP BY codec ORDER BY codec").unwrap();
    assert!(rows.as_array().unwrap().iter().any(|r| r["codec"] == "xor-zstd"));
    assert!(db.sql("DROP TABLE tensors").is_err(), "DDL must be rejected");

    // Branch + gc
    db.create_branch("old", "main~1").unwrap();
    db.delete_branch("old").unwrap();
    let g = db.gc(false).unwrap();
    assert_eq!(g.removed_commits, 0, "main~1 is still reachable through main");

    let out = tmp.path().join("out");
    db.export("main", &out).unwrap();
    assert_eq!(std::fs::read(out.join("b.safetensors")).unwrap(), std::fs::read(tmp.path().join("b.safetensors")).unwrap());
}

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::time::Instant;
use weightdb::{format_ts, Database, ImportOptions};

#[derive(Parser)]
#[command(name = "wdb", version, about = "WeightDB — a database for neural network weights")]
struct Cli {
    /// Repository directory
    #[arg(short = 'R', long, global = true, default_value = ".")]
    repo: PathBuf,
    /// Print machine-readable JSON
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Create a new repository
    Init,
    /// Import a .safetensors file or a model directory as a new commit
    Import {
        path: PathBuf,
        #[arg(short, long, default_value = "main")]
        branch: String,
        #[arg(short, long)]
        message: Option<String>,
        /// Delta-encode against this revision (default: the branch head)
        #[arg(long)]
        base: Option<String>,
        /// Disable delta encoding
        #[arg(long)]
        no_delta: bool,
        /// zstd-compress chunks that are not delta-encoded
        #[arg(long)]
        compress: bool,
        /// Target chunk size in MiB
        #[arg(long, default_value_t = 4)]
        chunk_mib: usize,
        /// Do not store non-weight files (config.json, tokenizer, ...)
        #[arg(long)]
        no_aux: bool,
    },
    /// Export a version back to safetensors files
    Export { rev: String, out: PathBuf },
    /// Show version history
    Log {
        #[arg(default_value = "main")]
        rev: String,
        #[arg(short = 'n', long, default_value_t = 20)]
        limit: usize,
    },
    /// List branches, or create/delete one
    Branch {
        name: Option<String>,
        /// Revision the new branch points to
        #[arg(default_value = "main")]
        at: String,
        #[arg(short, long)]
        delete: bool,
    },
    /// List the tensors ("matrices") of a version
    Ls {
        #[arg(default_value = "main")]
        rev: String,
        /// Filter by role: embedding, attention, mlp, norm, head, other
        #[arg(long)]
        role: Option<String>,
        /// Substring filter on tensor names
        #[arg(long)]
        grep: Option<String>,
    },
    /// Show statistics of one tensor (from the chunk index, no weight reads)
    Show { rev: String, tensor: String },
    /// Read single weights ("fields") of a tensor by flat index
    Get {
        rev: String,
        tensor: String,
        #[arg(required = true)]
        index: Vec<u64>,
    },
    /// Compare two versions
    Diff {
        a: String,
        b: String,
        /// Read changed tensors and compute relative L2 / cosine drift
        #[arg(long)]
        numeric: bool,
    },
    /// Run a read-only SQL query (tables: commits, refs, tensors, chunks, tensor_chunks)
    Sql { query: String },
    /// Storage usage and deduplication ratio
    Stats,
    /// Remove versions unreachable from any branch and compact pack files
    Gc {
        #[arg(long)]
        dry_run: bool,
    },
}

fn human(n: u64) -> String {
    let units = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1024.0 && i < units.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{n} B")
    } else {
        format!("{v:.2} {}", units[i])
    }
}

fn shape(s: &[u64]) -> String {
    format!("[{}]", s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "))
}

fn print_json(v: &impl serde::Serialize) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(v)?);
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let json = cli.json;
    if let Cmd::Init = cli.cmd {
        Database::init(&cli.repo)?;
        println!("Initialized empty WeightDB repo in {}", cli.repo.display());
        return Ok(());
    }
    let db = Database::open(&cli.repo)?;
    match cli.cmd {
        Cmd::Init => unreachable!(),
        Cmd::Import { path, branch, message, base, no_delta, compress, chunk_mib, no_aux } => {
            let opts = ImportOptions {
                branch,
                message: message.unwrap_or_else(|| format!("import {}", path.display())),
                chunk_bytes: chunk_mib.max(1) << 20,
                include_aux: !no_aux,
                base,
                delta: !no_delta,
                compress,
            };
            let t0 = Instant::now();
            let r = db.import(&path, &opts)?;
            if json {
                return print_json(&r);
            }
            let s = &r.stats;
            let total = s.new_bytes + s.dup_bytes;
            println!("[{} {}] {}", opts.branch, r.commit.short(), r.commit.message);
            println!(
                "  {} files, {} tensors, {} in {:.1}s",
                r.files,
                r.commit.n_tensors,
                human(total),
                t0.elapsed().as_secs_f64()
            );
            println!(
                "  reused {} ({} chunks, {:.1}% dedup)",
                human(s.dup_bytes),
                s.dup_chunks,
                100.0 * s.dup_bytes as f64 / total.max(1) as f64
            );
            println!(
                "  new    {} ({} chunks: {} delta, {} zstd) -> {} written",
                human(s.new_bytes),
                s.new_chunks,
                s.delta_chunks,
                s.zstd_chunks,
                human(s.stored_bytes)
            );
        }
        Cmd::Export { rev, out } => {
            let t0 = Instant::now();
            let files = db.export(&rev, &out)?;
            for f in &files {
                println!("  {}", f.display());
            }
            println!("exported {} files in {:.1}s", files.len(), t0.elapsed().as_secs_f64());
        }
        Cmd::Log { rev, limit } => {
            let log = db.log(&rev, limit)?;
            if json {
                return print_json(&log);
            }
            for c in log {
                println!(
                    "{}  {}  {:>4} tensors  {:>10}  {}",
                    c.short(),
                    format_ts(c.timestamp),
                    c.n_tensors,
                    human(c.logical_bytes),
                    c.message
                );
            }
        }
        Cmd::Branch { name, at, delete } => match name {
            None => {
                let b = db.branches()?;
                if json {
                    return print_json(&b);
                }
                for b in b {
                    println!("{:<24} {}", b.name, &b.commit_id[..12]);
                }
            }
            Some(n) if delete => {
                db.delete_branch(&n)?;
                println!("deleted branch {n}");
            }
            Some(n) => {
                let id = db.create_branch(&n, &at)?;
                println!("branch {n} -> {}", &id[..12]);
            }
        },
        Cmd::Ls { rev, role, grep } => {
            let ts: Vec<_> = db
                .tensors(&rev)?
                .into_iter()
                .filter(|t| role.as_deref().is_none_or(|r| r == t.role))
                .filter(|t| grep.as_deref().is_none_or(|g| t.name.contains(g)))
                .collect();
            if json {
                return print_json(&ts);
            }
            for t in &ts {
                println!(
                    "{:<60} {:<5} {:<18} {:<9} {:>4} chunks {:>10}",
                    t.name,
                    t.dtype,
                    shape(&t.shape),
                    t.role,
                    t.n_chunks,
                    human(t.nbytes)
                );
            }
            println!("{} tensors, {}", ts.len(), human(ts.iter().map(|t| t.nbytes).sum()));
        }
        Cmd::Show { rev, tensor } => {
            let t = db.tensor_info(&rev, &tensor)?;
            if json {
                return print_json(&t);
            }
            println!("name      {}", t.name);
            println!("dtype     {}", t.dtype);
            println!("shape     {}  ({} elements)", shape(&t.shape), t.numel);
            println!("role      {}", t.role);
            println!("size      {} in {} chunks", human(t.nbytes), t.n_chunks);
            if let Some(s) = t.stats {
                println!("mean      {:.6e}", s.mean);
                println!("std       {:.6e}", s.std);
                println!("abs mean  {:.6e}", s.abs_mean);
                println!("l2 norm   {:.6e}", s.l2);
                println!("min/max   {:.6e} / {:.6e}", s.min, s.max);
                println!("zeros     {:.4}%", 100.0 * s.sparsity);
                println!("nan/inf   {} / {}", s.nan, s.inf);
            }
        }
        Cmd::Get { rev, tensor, index } => {
            for i in index {
                println!("{tensor}[{i}] = {}", db.read_weight(&rev, &tensor, i)?);
            }
        }
        Cmd::Diff { a, b, numeric } => {
            let r = db.diff(&a, &b, numeric)?;
            if json {
                return print_json(&r);
            }
            for c in &r.changes {
                let mark = match c.kind.as_str() {
                    "added" => '+',
                    "removed" => '-',
                    _ => '~',
                };
                let mut line = format!(
                    "{mark} {:<60} {}/{} chunks ({})",
                    c.name,
                    c.changed_chunks,
                    c.total_chunks,
                    human(c.changed_bytes)
                );
                if c.shape_changed {
                    line.push_str("  shape/dtype changed");
                }
                if let (Some(l), Some(cos), Some(m)) = (c.rel_l2, c.cosine, c.max_abs) {
                    line.push_str(&format!("  rel_l2={l:.3e} cos={cos:.6} max|d|={m:.3e}"));
                }
                println!("{line}");
            }
            println!("{} changed, {} unchanged", r.changes.len(), r.unchanged);
        }
        Cmd::Sql { query } => {
            if json {
                return print_json(&db.sql_json(&query)?);
            }
            println!("{}", db.sql_pretty(&query)?);
        }
        Cmd::Stats => {
            let s = db.stats()?;
            if json {
                return print_json(&s);
            }
            let st = &s.storage;
            println!("commits         {}", st.commits);
            println!("branches        {}", s.branches);
            println!(
                "chunks          {} (raw {}, zstd {}, delta {})",
                st.chunks, st.raw_chunks, st.zstd_chunks, st.delta_chunks
            );
            println!("logical size    {}  (sum of all versions' weights)", human(s.logical_bytes));
            println!("unique data     {}", human(st.raw_bytes));
            println!("stored size     {}", human(st.stored_bytes));
            println!("pack files      {} ({})", st.pack_files, human(st.pack_bytes));
            println!("dedup ratio     {:.2}x", s.dedup_ratio);
        }
        Cmd::Gc { dry_run } => {
            let r = db.gc(dry_run)?;
            if json {
                return print_json(&r);
            }
            println!("{}", if dry_run { "gc (dry run)" } else { "gc" });
            println!("  commits  live {}  removed {}", r.live_commits, r.removed_commits);
            println!(
                "  chunks   live {}  removed {} ({})",
                r.live_chunks,
                r.removed_chunks,
                human(r.removed_chunk_bytes)
            );
            println!("  packs    rewritten {}  moved {}", r.rewritten_packs, human(r.moved_bytes));
            println!("  disk     {} -> {}", human(r.pack_bytes_before), human(r.pack_bytes_after));
            for p in &r.undeletable_packs {
                println!("  warning: could not delete {p} (in use?) — it will be removed by the next gc");
            }
        }
    }
    Ok(())
}

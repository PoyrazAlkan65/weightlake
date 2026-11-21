//! Native layer of the `weightdb` Node.js package. Structured results cross as
//! JSON strings, tensor data as a Buffer; `index.js` builds the friendly API.

use napi::bindgen_prelude::Buffer;
use napi::{Error, Result, Status};
use napi_derive::napi;

trait ToJs<T> {
    fn js(self) -> Result<T>;
}

impl<T> ToJs<T> for anyhow::Result<T> {
    fn js(self) -> Result<T> {
        self.map_err(|e| Error::new(Status::GenericFailure, format!("{e:#}")))
    }
}

impl<T> ToJs<T> for std::result::Result<T, serde_json::Error> {
    fn js(self) -> Result<T> {
        self.map_err(|e| Error::new(Status::InvalidArg, e.to_string()))
    }
}

fn json(v: &impl serde::Serialize) -> Result<String> {
    serde_json::to_string(v).js()
}

#[napi]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[napi]
pub struct NativeDatabase {
    db: wdb::Database,
}

#[napi]
impl NativeDatabase {
    #[napi(factory)]
    pub fn open(path: String, create: Option<bool>) -> Result<Self> {
        let db = if create.unwrap_or(false) { wdb::Database::open_or_init(&path) } else { wdb::Database::open(&path) };
        Ok(NativeDatabase { db: db.js()? })
    }

    #[napi(factory)]
    pub fn init(path: String) -> Result<Self> {
        Ok(NativeDatabase { db: wdb::Database::init(&path).js()? })
    }

    #[napi]
    pub fn path(&self) -> String {
        self.db.path().display().to_string()
    }

    #[napi]
    pub fn import_model(&self, path: String, options_json: Option<String>) -> Result<String> {
        let opts: wdb::ImportOptions = match options_json {
            Some(s) => serde_json::from_str(&s).js()?,
            None => wdb::ImportOptions::default(),
        };
        json(&self.db.import(&path, &opts).js()?)
    }

    #[napi]
    pub fn export(&self, rev: String, out_dir: String) -> Result<Vec<String>> {
        Ok(self.db.export(&rev, &out_dir).js()?.into_iter().map(|p| p.display().to_string()).collect())
    }

    #[napi]
    pub fn resolve(&self, rev: String) -> Result<String> {
        self.db.resolve(&rev).js()
    }

    #[napi]
    pub fn log(&self, rev: String, limit: u32) -> Result<String> {
        json(&self.db.log(&rev, if limit == 0 { usize::MAX } else { limit as usize }).js()?)
    }

    #[napi]
    pub fn branches(&self) -> Result<String> {
        json(&self.db.branches().js()?)
    }

    #[napi]
    pub fn create_branch(&self, name: String, rev: String) -> Result<String> {
        self.db.create_branch(&name, &rev).js()
    }

    #[napi]
    pub fn delete_branch(&self, name: String) -> Result<()> {
        self.db.delete_branch(&name).js()
    }

    #[napi]
    pub fn tensors(&self, rev: String) -> Result<String> {
        json(&self.db.tensors(&rev).js()?)
    }

    #[napi]
    pub fn tensor_info(&self, rev: String, name: String) -> Result<String> {
        json(&self.db.tensor_info(&rev, &name).js()?)
    }

    /// Raw little-endian bytes of a tensor.
    #[napi]
    pub fn read_tensor(&self, rev: String, name: String) -> Result<Buffer> {
        Ok(self.db.read_tensor(&rev, &name).js()?.data.into())
    }

    #[napi]
    pub fn read_weight(&self, rev: String, name: String, index: f64) -> Result<f64> {
        if index < 0.0 || index.fract() != 0.0 {
            return Err(Error::new(Status::InvalidArg, "index must be a non-negative integer"));
        }
        self.db.read_weight(&rev, &name, index as u64).js()
    }

    #[napi]
    pub fn diff(&self, a: String, b: String, numeric: Option<bool>) -> Result<String> {
        json(&self.db.diff(&a, &b, numeric.unwrap_or(false)).js()?)
    }

    #[napi]
    pub fn sql(&self, query: String) -> Result<String> {
        json(&self.db.sql_json(&query).js()?)
    }

    #[napi]
    pub fn sql_pretty(&self, query: String) -> Result<String> {
        self.db.sql_pretty(&query).js()
    }

    #[napi]
    pub fn stats(&self) -> Result<String> {
        json(&self.db.stats().js()?)
    }

    #[napi]
    pub fn gc(&self, dry_run: Option<bool>) -> Result<String> {
        json(&self.db.gc(dry_run.unwrap_or(false)).js()?)
    }
}

//! Native layer of the `weightlake` Python package. Kept deliberately thin:
//! structured results cross as JSON strings, tensor data as a bytearray.
//! The friendly API (numpy / torch / dicts) lives in `weightlake/__init__.py`.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes};

trait ToPy<T> {
    fn py(self) -> PyResult<T>;
}

impl<T> ToPy<T> for anyhow::Result<T> {
    fn py(self) -> PyResult<T> {
        self.map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))
    }
}

impl<T> ToPy<T> for Result<T, serde_json::Error> {
    fn py(self) -> PyResult<T> {
        self.map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
}

fn json(v: &impl serde::Serialize) -> PyResult<String> {
    serde_json::to_string(v).py()
}

#[pyclass(frozen, module = "weightlake._native")]
struct Database {
    db: wl::Database,
}

#[pymethods]
impl Database {
    #[new]
    #[pyo3(signature = (path, create = false))]
    fn new(path: &str, create: bool) -> PyResult<Self> {
        let db = if create { wl::Database::open_or_init(path) } else { wl::Database::open(path) }.py()?;
        Ok(Database { db })
    }

    #[staticmethod]
    fn init(path: &str) -> PyResult<Self> {
        Ok(Database { db: wl::Database::init(path).py()? })
    }

    fn path(&self) -> String {
        self.db.path().display().to_string()
    }

    fn import_model(&self, path: &str, options_json: &str) -> PyResult<String> {
        let opts: wl::ImportOptions = serde_json::from_str(options_json).py()?;
        json(&self.db.import(path, &opts).py()?)
    }

    /// Commit in-memory tensors. Each tensor is `(name, dtype, shape, address, nbytes)`:
    /// the raw address of a contiguous little-endian buffer that the Python
    /// caller keeps alive and unmodified for the duration of the call
    /// (zero-copy; the GIL is released while hashing/compressing).
    fn import_tensors(
        &self,
        py: Python<'_>,
        tensors: Vec<(String, String, Vec<u64>, usize, usize)>,
        aux: Vec<(String, Vec<u8>)>,
        options_json: &str,
    ) -> PyResult<String> {
        let opts: wl::ImportOptions = serde_json::from_str(options_json).py()?;
        let mut mem = Vec::with_capacity(tensors.len());
        for (name, dtype, shape, addr, nbytes) in tensors {
            let dtype = wl::DType::parse(&dtype)
                .ok_or_else(|| PyRuntimeError::new_err(format!("tensor '{name}': unknown dtype {dtype}")))?;
            let data: &[u8] = if nbytes == 0 {
                &[]
            } else if addr == 0 {
                return Err(PyRuntimeError::new_err(format!("tensor '{name}': null data pointer")));
            } else {
                // SAFETY: see the contract above; the Python wrapper holds references
                // to every source tensor until this call returns.
                unsafe { std::slice::from_raw_parts(addr as *const u8, nbytes) }
            };
            mem.push(wl::MemTensor { name, dtype, shape, data });
        }
        let report = py.detach(|| self.db.import_tensors(mem, &aux, &opts)).py()?;
        json(&report)
    }

    fn aux_files(&self, rev: &str) -> PyResult<Vec<String>> {
        self.db.aux_files(rev).py()
    }

    fn read_aux<'py>(&self, py: Python<'py>, rev: &str, name: &str) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(py, &self.db.read_aux(rev, name).py()?))
    }

    fn export(&self, rev: &str, out_dir: &str) -> PyResult<Vec<String>> {
        Ok(self.db.export(rev, out_dir).py()?.into_iter().map(|p| p.display().to_string()).collect())
    }

    fn resolve(&self, rev: &str) -> PyResult<String> {
        self.db.resolve(rev).py()
    }

    fn log(&self, rev: &str, limit: usize) -> PyResult<String> {
        json(&self.db.log(rev, limit).py()?)
    }

    fn branches(&self) -> PyResult<String> {
        json(&self.db.branches().py()?)
    }

    fn create_branch(&self, name: &str, rev: &str) -> PyResult<String> {
        self.db.create_branch(name, rev).py()
    }

    fn delete_branch(&self, name: &str) -> PyResult<()> {
        self.db.delete_branch(name).py()
    }

    fn tensors(&self, rev: &str) -> PyResult<String> {
        json(&self.db.tensors(rev).py()?)
    }

    fn tensor_info(&self, rev: &str, name: &str) -> PyResult<String> {
        json(&self.db.tensor_info(rev, name).py()?)
    }

    /// Returns (bytearray, dtype, shape).
    fn read_tensor<'py>(
        &self,
        py: Python<'py>,
        rev: &str,
        name: &str,
    ) -> PyResult<(Bound<'py, PyByteArray>, String, Vec<u64>)> {
        let m = self.db.manifest(rev).py()?;
        let t = m.tensor(name).ok_or_else(|| PyRuntimeError::new_err(format!("no tensor '{name}' in {rev}")))?;
        let buf = PyByteArray::new_with(py, t.nbytes() as usize, |out| {
            self.db.read_tensor_into(rev, name, out).py()
        })?;
        Ok((buf, t.dtype.as_str().to_string(), t.shape.clone()))
    }

    fn read_weight(&self, rev: &str, name: &str, index: u64) -> PyResult<f64> {
        self.db.read_weight(rev, name, index).py()
    }

    fn diff(&self, a: &str, b: &str, numeric: bool) -> PyResult<String> {
        json(&self.db.diff(a, b, numeric).py()?)
    }

    fn sql(&self, query: &str) -> PyResult<String> {
        json(&self.db.sql_json(query).py()?)
    }

    fn sql_pretty(&self, query: &str) -> PyResult<String> {
        self.db.sql_pretty(query).py()
    }

    fn stats(&self) -> PyResult<String> {
        json(&self.db.stats().py()?)
    }

    fn gc(&self, dry_run: bool) -> PyResult<String> {
        json(&self.db.gc(dry_run).py()?)
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Database>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

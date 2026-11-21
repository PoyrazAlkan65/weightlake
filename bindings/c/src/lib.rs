//! C ABI for WeightLake. See `include/weightlake.h` for documentation.
//!
//! Conventions:
//! - Functions return 0 on success, -1 on error; `wl_last_error()` gives the
//!   message (thread-local).
//! - Strings returned through `char**` are heap-allocated and must be released
//!   with `wl_string_free`.
//! - Panics never cross the FFI boundary.

#![allow(clippy::missing_safety_doc)]

use std::cell::RefCell;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

use wl::{Database, ImportOptions};

pub struct WlDb {
    db: Database,
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_error(msg: String) {
    let c = CString::new(msg.replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = c);
}

/// Run `f`, converting errors and panics into -1 + last error.
fn guard(f: impl FnOnce() -> anyhow::Result<()>) -> c_int {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            set_error(format!("{e:#}"));
            -1
        }
        Err(p) => {
            let msg = p
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "panic".into());
            set_error(format!("internal panic: {msg}"));
            -1
        }
    }
}

unsafe fn cstr<'a>(p: *const c_char, what: &str) -> anyhow::Result<&'a str> {
    if p.is_null() {
        anyhow::bail!("{what} is NULL");
    }
    Ok(CStr::from_ptr(p).to_str()?)
}

unsafe fn db<'a>(p: *const WlDb) -> anyhow::Result<&'a Database> {
    if p.is_null() {
        anyhow::bail!("database handle is NULL");
    }
    Ok(&(*p).db)
}

unsafe fn out_str(out: *mut *mut c_char, s: String) -> anyhow::Result<()> {
    if out.is_null() {
        anyhow::bail!("output pointer is NULL");
    }
    *out = CString::new(s)?.into_raw();
    Ok(())
}

unsafe fn out_json(out: *mut *mut c_char, v: &impl serde::Serialize) -> anyhow::Result<()> {
    out_str(out, serde_json::to_string(v)?)
}

// ------------------------------------------------------------------ basics

#[no_mangle]
pub extern "C" fn wl_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

#[no_mangle]
pub extern "C" fn wl_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

#[no_mangle]
pub unsafe extern "C" fn wl_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

unsafe fn open_with(path: *const c_char, out: *mut *mut WlDb, f: fn(&str) -> anyhow::Result<Database>) -> c_int {
    guard(|| {
        if out.is_null() {
            anyhow::bail!("output pointer is NULL");
        }
        *out = ptr::null_mut();
        let db = f(cstr(path, "path")?)?;
        *out = Box::into_raw(Box::new(WlDb { db }));
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn wl_open(path: *const c_char, out: *mut *mut WlDb) -> c_int {
    open_with(path, out, |p| Database::open(p))
}

#[no_mangle]
pub unsafe extern "C" fn wl_init(path: *const c_char, out: *mut *mut WlDb) -> c_int {
    open_with(path, out, |p| Database::init(p))
}

#[no_mangle]
pub unsafe extern "C" fn wl_open_or_init(path: *const c_char, out: *mut *mut WlDb) -> c_int {
    open_with(path, out, |p| Database::open_or_init(p))
}

#[no_mangle]
pub unsafe extern "C" fn wl_close(db: *mut WlDb) {
    if !db.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(db))));
    }
}

// ---------------------------------------------------------------- versions

#[no_mangle]
pub unsafe extern "C" fn wl_import(
    h: *mut WlDb,
    path: *const c_char,
    options_json: *const c_char,
    out_report_json: *mut *mut c_char,
) -> c_int {
    guard(|| {
        let opts: ImportOptions = if options_json.is_null() {
            ImportOptions::default()
        } else {
            serde_json::from_str(cstr(options_json, "options_json")?)?
        };
        let r = db(h)?.import(cstr(path, "path")?, &opts)?;
        if !out_report_json.is_null() {
            out_json(out_report_json, &r)?;
        }
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn wl_export(h: *mut WlDb, rev: *const c_char, out_dir: *const c_char) -> c_int {
    guard(|| {
        db(h)?.export(cstr(rev, "rev")?, cstr(out_dir, "out_dir")?)?;
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn wl_resolve(h: *mut WlDb, rev: *const c_char, out_id: *mut *mut c_char) -> c_int {
    guard(|| out_str(out_id, db(h)?.resolve(cstr(rev, "rev")?)?))
}

#[no_mangle]
pub unsafe extern "C" fn wl_log_json(h: *mut WlDb, rev: *const c_char, limit: usize, out: *mut *mut c_char) -> c_int {
    guard(|| out_json(out, &db(h)?.log(cstr(rev, "rev")?, if limit == 0 { usize::MAX } else { limit })?))
}

#[no_mangle]
pub unsafe extern "C" fn wl_branches_json(h: *mut WlDb, out: *mut *mut c_char) -> c_int {
    guard(|| out_json(out, &db(h)?.branches()?))
}

#[no_mangle]
pub unsafe extern "C" fn wl_create_branch(h: *mut WlDb, name: *const c_char, rev: *const c_char) -> c_int {
    guard(|| {
        db(h)?.create_branch(cstr(name, "name")?, cstr(rev, "rev")?)?;
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn wl_delete_branch(h: *mut WlDb, name: *const c_char) -> c_int {
    guard(|| db(h)?.delete_branch(cstr(name, "name")?))
}

// ----------------------------------------------------------------- tensors

pub const WL_MAX_NDIM: usize = 8;

#[repr(C)]
pub struct WlTensorInfo {
    pub dtype: [c_char; 16],
    pub elem_size: u32,
    pub ndim: u32,
    pub shape: [u64; WL_MAX_NDIM],
    pub numel: u64,
    pub nbytes: u64,
    pub n_chunks: u64,
}

#[no_mangle]
pub unsafe extern "C" fn wl_tensors_json(h: *mut WlDb, rev: *const c_char, out: *mut *mut c_char) -> c_int {
    guard(|| out_json(out, &db(h)?.tensors(cstr(rev, "rev")?)?))
}

#[no_mangle]
pub unsafe extern "C" fn wl_tensor_info_json(
    h: *mut WlDb,
    rev: *const c_char,
    name: *const c_char,
    out: *mut *mut c_char,
) -> c_int {
    guard(|| out_json(out, &db(h)?.tensor_info(cstr(rev, "rev")?, cstr(name, "name")?)?))
}

#[no_mangle]
pub unsafe extern "C" fn wl_tensor_info(
    h: *mut WlDb,
    rev: *const c_char,
    name: *const c_char,
    out: *mut WlTensorInfo,
) -> c_int {
    guard(|| {
        if out.is_null() {
            anyhow::bail!("output pointer is NULL");
        }
        let m = db(h)?.manifest(cstr(rev, "rev")?)?;
        let name = cstr(name, "name")?;
        let t = m.tensor(name).ok_or_else(|| anyhow::anyhow!("no tensor '{name}'"))?;
        if t.shape.len() > WL_MAX_NDIM {
            anyhow::bail!("tensor has {} dims (max {WL_MAX_NDIM})", t.shape.len());
        }
        let mut info = WlTensorInfo {
            dtype: [0; 16],
            elem_size: t.dtype.size() as u32,
            ndim: t.shape.len() as u32,
            shape: [0; WL_MAX_NDIM],
            numel: t.numel(),
            nbytes: t.nbytes(),
            n_chunks: t.chunks.len() as u64,
        };
        for (i, b) in t.dtype.as_str().bytes().enumerate() {
            info.dtype[i] = b as c_char;
        }
        info.shape[..t.shape.len()].copy_from_slice(&t.shape);
        *out = info;
        Ok(())
    })
}

/// Copy a tensor's raw little-endian bytes into `buf` (`buf_len` must equal nbytes).
#[no_mangle]
pub unsafe extern "C" fn wl_read_tensor(
    h: *mut WlDb,
    rev: *const c_char,
    name: *const c_char,
    buf: *mut c_void,
    buf_len: usize,
) -> c_int {
    guard(|| {
        if buf.is_null() && buf_len > 0 {
            anyhow::bail!("buffer is NULL");
        }
        let out = if buf_len == 0 { &mut [][..] } else { std::slice::from_raw_parts_mut(buf as *mut u8, buf_len) };
        db(h)?.read_tensor_into(cstr(rev, "rev")?, cstr(name, "name")?, out)
    })
}

#[no_mangle]
pub unsafe extern "C" fn wl_read_weight(
    h: *mut WlDb,
    rev: *const c_char,
    name: *const c_char,
    index: u64,
    out: *mut f64,
) -> c_int {
    guard(|| {
        if out.is_null() {
            anyhow::bail!("output pointer is NULL");
        }
        *out = db(h)?.read_weight(cstr(rev, "rev")?, cstr(name, "name")?, index)?;
        Ok(())
    })
}

// --------------------------------------------------------- queries & admin

#[no_mangle]
pub unsafe extern "C" fn wl_diff_json(
    h: *mut WlDb,
    a: *const c_char,
    b: *const c_char,
    numeric: c_int,
    out: *mut *mut c_char,
) -> c_int {
    guard(|| out_json(out, &db(h)?.diff(cstr(a, "a")?, cstr(b, "b")?, numeric != 0)?))
}

#[no_mangle]
pub unsafe extern "C" fn wl_sql_json(h: *mut WlDb, query: *const c_char, out: *mut *mut c_char) -> c_int {
    guard(|| out_json(out, &db(h)?.sql_json(cstr(query, "query")?)?))
}

#[no_mangle]
pub unsafe extern "C" fn wl_sql_pretty(h: *mut WlDb, query: *const c_char, out: *mut *mut c_char) -> c_int {
    guard(|| out_str(out, db(h)?.sql_pretty(cstr(query, "query")?)?))
}

#[no_mangle]
pub unsafe extern "C" fn wl_stats_json(h: *mut WlDb, out: *mut *mut c_char) -> c_int {
    guard(|| out_json(out, &db(h)?.stats()?))
}

#[no_mangle]
pub unsafe extern "C" fn wl_gc_json(h: *mut WlDb, dry_run: c_int, out: *mut *mut c_char) -> c_int {
    guard(|| {
        let r = db(h)?.gc(dry_run != 0)?;
        if !out.is_null() {
            out_json(out, &r)?;
        }
        Ok(())
    })
}

/*
 * WeightDB — C API
 *
 * An embedded database for neural network weights.
 *
 *   wdb_db* db;
 *   if (wdb_open("repo", &db) != 0) { fprintf(stderr, "%s\n", wdb_last_error()); return 1; }
 *   char* rows;
 *   wdb_sql_json(db, "SELECT name, l2 FROM tensors WHERE commit_id = rev('main')", &rows);
 *   puts(rows);
 *   wdb_string_free(rows);
 *   wdb_close(db);
 *
 * Conventions
 *   - Every function returning int returns 0 on success and -1 on error.
 *     wdb_last_error() returns the message of the last error on this thread.
 *   - char* values returned through `char** out` are owned by the caller and
 *     must be released with wdb_string_free().
 *   - "rev" accepts a branch name, a commit id (or unique prefix of >= 4 chars)
 *     or "rev~N" (N-th first parent).
 *   - Tensor data is raw little-endian, row-major, in the tensor's dtype
 *     ("F32", "F16", "BF16", "F64", "I8", "U8", "I16", "U16", "I32", "U32",
 *      "I64", "U64", "BOOL", "F8_E4M3", "F8_E5M2").
 *   - A wdb_db handle may be shared between threads. Only one process may
 *     open a repository at a time.
 */
#ifndef WEIGHTDB_H
#define WEIGHTDB_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WDB_MAX_NDIM 8

typedef struct WdbDb wdb_db;

typedef struct wdb_tensor_info_t {
    char     dtype[16];     /* NUL-terminated, e.g. "BF16" */
    uint32_t elem_size;     /* bytes per element */
    uint32_t ndim;
    uint64_t shape[WDB_MAX_NDIM];
    uint64_t numel;
    uint64_t nbytes;
    uint64_t n_chunks;
} wdb_tensor_info_t;

/* ---- basics ---------------------------------------------------------- */
const char* wdb_version(void);
const char* wdb_last_error(void);
void        wdb_string_free(char* s);

int  wdb_open(const char* path, wdb_db** out);
int  wdb_init(const char* path, wdb_db** out);
int  wdb_open_or_init(const char* path, wdb_db** out);
void wdb_close(wdb_db* db);

/* ---- versions -------------------------------------------------------- */
/* options_json may be NULL; all fields optional:
 *   {"branch":"main","message":"...","base":null,"delta":true,
 *    "compress":false,"chunk_bytes":4194304,"include_aux":true}
 * out_report_json may be NULL. */
int wdb_import(wdb_db* db, const char* path, const char* options_json, char** out_report_json);
int wdb_export(wdb_db* db, const char* rev, const char* out_dir);
int wdb_resolve(wdb_db* db, const char* rev, char** out_commit_id);
int wdb_log_json(wdb_db* db, const char* rev, size_t limit /* 0 = all */, char** out);
int wdb_branches_json(wdb_db* db, char** out);
int wdb_create_branch(wdb_db* db, const char* name, const char* rev);
int wdb_delete_branch(wdb_db* db, const char* name);

/* ---- tensors ("matrices") and weights ("fields") ------------------------ */
int wdb_tensors_json(wdb_db* db, const char* rev, char** out);
int wdb_tensor_info_json(wdb_db* db, const char* rev, const char* name, char** out); /* incl. stats */
int wdb_tensor_info(wdb_db* db, const char* rev, const char* name, wdb_tensor_info_t* out);
/* buf_len must equal info.nbytes */
int wdb_read_tensor(wdb_db* db, const char* rev, const char* name, void* buf, size_t buf_len);
int wdb_read_weight(wdb_db* db, const char* rev, const char* name, uint64_t index, double* out);

/* ---- diff, SQL, maintenance ---------------------------------------------- */
int wdb_diff_json(wdb_db* db, const char* rev_a, const char* rev_b, int numeric, char** out);
/* Read-only SQL. Tables: commits, refs, tensors, chunks, tensor_chunks.
 * Functions: rev(), weight(), tensor_cosine(), tensor_rel_l2(), tensor_max_abs_diff(). */
int wdb_sql_json(wdb_db* db, const char* query, char** out);   /* JSON array of row objects */
int wdb_sql_pretty(wdb_db* db, const char* query, char** out); /* ASCII table */
int wdb_stats_json(wdb_db* db, char** out);
int wdb_gc_json(wdb_db* db, int dry_run, char** out /* may be NULL */);

#ifdef __cplusplus
}
#endif

#endif /* WEIGHTDB_H */

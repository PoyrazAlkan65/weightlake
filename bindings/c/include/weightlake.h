/*
 * WeightLake — C API
 *
 * An embedded database for neural network weights.
 *
 *   wl_db* db;
 *   if (wl_open("repo", &db) != 0) { fprintf(stderr, "%s\n", wl_last_error()); return 1; }
 *   char* rows;
 *   wl_sql_json(db, "SELECT name, l2 FROM tensors WHERE commit_id = rev('main')", &rows);
 *   puts(rows);
 *   wl_string_free(rows);
 *   wl_close(db);
 *
 * Conventions
 *   - Every function returning int returns 0 on success and -1 on error.
 *     wl_last_error() returns the message of the last error on this thread.
 *   - char* values returned through `char** out` are owned by the caller and
 *     must be released with wl_string_free().
 *   - "rev" accepts a branch name, a commit id (or unique prefix of >= 4 chars)
 *     or "rev~N" (N-th first parent).
 *   - Tensor data is raw little-endian, row-major, in the tensor's dtype
 *     ("F32", "F16", "BF16", "F64", "I8", "U8", "I16", "U16", "I32", "U32",
 *      "I64", "U64", "BOOL", "F8_E4M3", "F8_E5M2").
 *   - A wl_db handle may be shared between threads. Only one process may
 *     open a repository at a time.
 */
#ifndef WEIGHTLAKE_H
#define WEIGHTLAKE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WL_MAX_NDIM 8

typedef struct WlDb wl_db;

typedef struct wl_tensor_info_t {
    char     dtype[16];     /* NUL-terminated, e.g. "BF16" */
    uint32_t elem_size;     /* bytes per element */
    uint32_t ndim;
    uint64_t shape[WL_MAX_NDIM];
    uint64_t numel;
    uint64_t nbytes;
    uint64_t n_chunks;
} wl_tensor_info_t;

/* ---- basics ---------------------------------------------------------- */
const char* wl_version(void);
const char* wl_last_error(void);
void        wl_string_free(char* s);

int  wl_open(const char* path, wl_db** out);
int  wl_init(const char* path, wl_db** out);
int  wl_open_or_init(const char* path, wl_db** out);
void wl_close(wl_db* db);

/* ---- versions -------------------------------------------------------- */
/* options_json may be NULL; all fields optional:
 *   {"branch":"main","message":"...","base":null,"delta":true,
 *    "compress":false,"chunk_bytes":4194304,"include_aux":true}
 * out_report_json may be NULL. */
int wl_import(wl_db* db, const char* path, const char* options_json, char** out_report_json);
int wl_export(wl_db* db, const char* rev, const char* out_dir);
int wl_resolve(wl_db* db, const char* rev, char** out_commit_id);
int wl_log_json(wl_db* db, const char* rev, size_t limit /* 0 = all */, char** out);
int wl_branches_json(wl_db* db, char** out);
int wl_create_branch(wl_db* db, const char* name, const char* rev);
int wl_delete_branch(wl_db* db, const char* name);

/* ---- tensors ("matrices") and weights ("fields") ------------------------ */
int wl_tensors_json(wl_db* db, const char* rev, char** out);
int wl_tensor_info_json(wl_db* db, const char* rev, const char* name, char** out); /* incl. stats */
int wl_tensor_info(wl_db* db, const char* rev, const char* name, wl_tensor_info_t* out);
/* buf_len must equal info.nbytes */
int wl_read_tensor(wl_db* db, const char* rev, const char* name, void* buf, size_t buf_len);
int wl_read_weight(wl_db* db, const char* rev, const char* name, uint64_t index, double* out);

/* ---- diff, SQL, maintenance ---------------------------------------------- */
int wl_diff_json(wl_db* db, const char* rev_a, const char* rev_b, int numeric, char** out);
/* Read-only SQL. Tables: commits, refs, tensors, chunks, tensor_chunks.
 * Functions: rev(), weight(), tensor_cosine(), tensor_rel_l2(), tensor_max_abs_diff(). */
int wl_sql_json(wl_db* db, const char* query, char** out);   /* JSON array of row objects */
int wl_sql_pretty(wl_db* db, const char* query, char** out); /* ASCII table */
int wl_stats_json(wl_db* db, char** out);
int wl_gc_json(wl_db* db, int dry_run, char** out /* may be NULL */);

#ifdef __cplusplus
}
#endif

#endif /* WEIGHTLAKE_H */

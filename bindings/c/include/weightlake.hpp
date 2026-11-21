// WeightLake — header-only C++17 wrapper over weightlake.h
//
//   weightlake::Database db = weightlake::Database::open("repo");
//   auto info = db.tensor_info("main", "h.0.attn.c_attn.weight");
//   std::vector<float> w = db.read_tensor<float>("main", "h.0.attn.c_attn.weight");
//   std::string rows = db.sql_json("SELECT name, l2 FROM tensors WHERE commit_id = rev('main')");
//
// Errors are thrown as weightlake::Error.
#pragma once

#include "weightlake.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace weightlake {

class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

namespace detail {
inline void check(int rc) {
    if (rc != 0) throw Error(wl_last_error());
}
// Takes ownership of a char* returned by the C API.
inline std::string take(char* s) {
    std::string out = s ? std::string(s) : std::string();
    wl_string_free(s);
    return out;
}
}  // namespace detail

struct TensorInfo {
    std::string dtype;
    uint32_t elem_size = 0;
    std::vector<uint64_t> shape;
    uint64_t numel = 0;
    uint64_t nbytes = 0;
    uint64_t n_chunks = 0;
};

class Database {
public:
    static Database open(const std::string& path) { return Database(path, &wl_open); }
    static Database init(const std::string& path) { return Database(path, &wl_init); }
    static Database open_or_init(const std::string& path) { return Database(path, &wl_open_or_init); }

    Database(Database&& o) noexcept : db_(std::exchange(o.db_, nullptr)) {}
    Database& operator=(Database&& o) noexcept {
        if (this != &o) {
            close();
            db_ = std::exchange(o.db_, nullptr);
        }
        return *this;
    }
    Database(const Database&) = delete;
    Database& operator=(const Database&) = delete;
    ~Database() { close(); }

    void close() {
        if (db_) wl_close(db_);
        db_ = nullptr;
    }
    wl_db* handle() const { return db_; }

    // ---- versions
    /// options_json: {"branch":..,"message":..,"base":..,"delta":..,"compress":..}
    std::string import_model(const std::string& path, const std::string& options_json = "{}") {
        char* out = nullptr;
        detail::check(wl_import(db_, path.c_str(), options_json.c_str(), &out));
        return detail::take(out);
    }
    void export_to(const std::string& rev, const std::string& out_dir) {
        detail::check(wl_export(db_, rev.c_str(), out_dir.c_str()));
    }
    std::string resolve(const std::string& rev) {
        char* out = nullptr;
        detail::check(wl_resolve(db_, rev.c_str(), &out));
        return detail::take(out);
    }
    std::string log_json(const std::string& rev = "main", size_t limit = 0) {
        char* out = nullptr;
        detail::check(wl_log_json(db_, rev.c_str(), limit, &out));
        return detail::take(out);
    }
    std::string branches_json() {
        char* out = nullptr;
        detail::check(wl_branches_json(db_, &out));
        return detail::take(out);
    }
    void create_branch(const std::string& name, const std::string& rev) {
        detail::check(wl_create_branch(db_, name.c_str(), rev.c_str()));
    }
    void delete_branch(const std::string& name) { detail::check(wl_delete_branch(db_, name.c_str())); }

    // ---- tensors & weights
    std::string tensors_json(const std::string& rev = "main") {
        char* out = nullptr;
        detail::check(wl_tensors_json(db_, rev.c_str(), &out));
        return detail::take(out);
    }
    std::string tensor_info_json(const std::string& rev, const std::string& name) {
        char* out = nullptr;
        detail::check(wl_tensor_info_json(db_, rev.c_str(), name.c_str(), &out));
        return detail::take(out);
    }
    TensorInfo tensor_info(const std::string& rev, const std::string& name) {
        wl_tensor_info_t raw{};
        detail::check(wl_tensor_info(db_, rev.c_str(), name.c_str(), &raw));
        TensorInfo t;
        t.dtype = raw.dtype;
        t.elem_size = raw.elem_size;
        t.shape.assign(raw.shape, raw.shape + raw.ndim);
        t.numel = raw.numel;
        t.nbytes = raw.nbytes;
        t.n_chunks = raw.n_chunks;
        return t;
    }
    std::vector<uint8_t> read_tensor_bytes(const std::string& rev, const std::string& name) {
        TensorInfo t = tensor_info(rev, name);
        std::vector<uint8_t> buf(t.nbytes);
        detail::check(wl_read_tensor(db_, rev.c_str(), name.c_str(), buf.data(), buf.size()));
        return buf;
    }
    /// Read a tensor as T (sizeof(T) must equal the element size, e.g. float for F32,
    /// uint16_t for BF16/F16).
    template <class T>
    std::vector<T> read_tensor(const std::string& rev, const std::string& name) {
        TensorInfo t = tensor_info(rev, name);
        if (t.elem_size != sizeof(T))
            throw Error("element size mismatch: tensor is " + t.dtype + ", T has " + std::to_string(sizeof(T)) +
                        " bytes");
        std::vector<T> buf(t.numel);
        detail::check(wl_read_tensor(db_, rev.c_str(), name.c_str(), buf.data(), t.nbytes));
        return buf;
    }
    double read_weight(const std::string& rev, const std::string& name, uint64_t index) {
        double v = 0;
        detail::check(wl_read_weight(db_, rev.c_str(), name.c_str(), index, &v));
        return v;
    }

    // ---- diff, SQL, maintenance
    std::string diff_json(const std::string& a, const std::string& b, bool numeric = false) {
        char* out = nullptr;
        detail::check(wl_diff_json(db_, a.c_str(), b.c_str(), numeric ? 1 : 0, &out));
        return detail::take(out);
    }
    std::string sql_json(const std::string& query) {
        char* out = nullptr;
        detail::check(wl_sql_json(db_, query.c_str(), &out));
        return detail::take(out);
    }
    std::string sql_pretty(const std::string& query) {
        char* out = nullptr;
        detail::check(wl_sql_pretty(db_, query.c_str(), &out));
        return detail::take(out);
    }
    std::string stats_json() {
        char* out = nullptr;
        detail::check(wl_stats_json(db_, &out));
        return detail::take(out);
    }
    std::string gc_json(bool dry_run = false) {
        char* out = nullptr;
        detail::check(wl_gc_json(db_, dry_run ? 1 : 0, &out));
        return detail::take(out);
    }

private:
    using OpenFn = int (*)(const char*, wl_db**);
    Database(const std::string& path, OpenFn fn) { detail::check(fn(path.c_str(), &db_)); }
    wl_db* db_ = nullptr;
};

}  // namespace weightlake

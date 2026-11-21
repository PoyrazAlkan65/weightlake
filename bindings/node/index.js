'use strict';
/**
 * WeightLake — an embedded database for neural network weights (Node.js).
 *
 *   const weightlake = require('weightlake');
 *   const db = weightlake.open('repo', { create: true });
 *   db.importModel('models/gpt2', { message: 'gpt2 base' });
 *   const t = db.readTensor('h.0.attn.c_attn.weight');   // { data: Float32Array, shape, dtype }
 *   db.sql("SELECT name, l2 FROM tensors WHERE commit_id = rev('main') LIMIT 5");
 */
const path = require('path');

const native = require(process.env.WEIGHTLAKE_NATIVE || path.join(__dirname, 'weightlake.node'));

// dtype -> [TypedArray constructor, element size]
const TYPED = {
  F64: Float64Array, F32: Float32Array,
  I64: BigInt64Array, U64: BigUint64Array,
  I32: Int32Array, U32: Uint32Array,
  I16: Int16Array, U16: Uint16Array, F16: Uint16Array, BF16: Uint16Array,
  I8: Int8Array, U8: Uint8Array, BOOL: Uint8Array, F8_E4M3: Uint8Array, F8_E5M2: Uint8Array,
};

function halfToFloat(h) {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * 2 ** -14 * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * 2 ** (e - 15) * (1 + f / 1024);
}

function toTyped(buf, dtype, toFloat32) {
  // Copy into a fresh, aligned ArrayBuffer (Node Buffers may be pooled/unaligned).
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  if (dtype === 'BF16' && toFloat32) {
    const u16 = new Uint16Array(ab);
    const u32 = new Uint32Array(u16.length);
    for (let i = 0; i < u16.length; i++) u32[i] = u16[i] << 16;
    return new Float32Array(u32.buffer);
  }
  if (dtype === 'F16' && toFloat32) {
    const u16 = new Uint16Array(ab);
    const out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) out[i] = halfToFloat(u16[i]);
    return out;
  }
  const T = TYPED[dtype];
  if (!T) throw new Error(`unsupported dtype ${dtype}`);
  return new T(ab);
}

class Database {
  constructor(nativeDb) {
    this._db = nativeDb;
  }

  get path() { return this._db.path(); }

  // ---- versions
  importModel(modelPath, opts = {}) {
    const o = {
      branch: opts.branch ?? 'main',
      message: opts.message ?? `import ${modelPath}`,
      base: opts.base ?? null,
      delta: opts.delta ?? true,
      compress: opts.compress ?? false,
      chunk_bytes: (opts.chunkMiB ?? 4) * 1024 * 1024,
      include_aux: opts.includeAux ?? true,
    };
    return JSON.parse(this._db.importModel(String(modelPath), JSON.stringify(o)));
  }
  export(rev, outDir) { return this._db.export(rev, String(outDir)); }
  resolve(rev) { return this._db.resolve(rev); }
  log(rev = 'main', limit = 100) { return JSON.parse(this._db.log(rev, limit)); }
  branches() {
    return Object.fromEntries(JSON.parse(this._db.branches()).map((b) => [b.name, b.commit_id]));
  }
  createBranch(name, rev = 'main') { return this._db.createBranch(name, rev); }
  deleteBranch(name) { this._db.deleteBranch(name); }

  // ---- tensors ("matrices") and weights ("fields")
  tensors(rev = 'main') { return JSON.parse(this._db.tensors(rev)); }
  tensorNames(rev = 'main') { return this.tensors(rev).map((t) => t.name); }
  tensorInfo(name, rev = 'main') { return JSON.parse(this._db.tensorInfo(rev, name)); }

  /** @returns {{name, dtype, shape, data: TypedArray}} BF16/F16 -> Float32Array unless {raw: true} */
  readTensor(name, rev = 'main', { raw = false } = {}) {
    const info = this.tensorInfo(name, rev);
    const buf = this._db.readTensor(rev, name);
    return { name, dtype: info.dtype, shape: info.shape, data: toTyped(buf, info.dtype, !raw) };
  }
  readTensorBuffer(name, rev = 'main') { return this._db.readTensor(rev, name); }
  weight(name, index, rev = 'main') { return this._db.readWeight(rev, name, index); }

  // ---- diff / SQL / maintenance
  diff(a, b, { numeric = false } = {}) { return JSON.parse(this._db.diff(a, b, numeric)); }
  /** Read-only SQL; returns an array of row objects. */
  sql(query) { return JSON.parse(this._db.sql(query)); }
  sqlPretty(query) { return this._db.sqlPretty(query); }
  stats() { return JSON.parse(this._db.stats()); }
  gc({ dryRun = false } = {}) { return JSON.parse(this._db.gc(dryRun)); }
}

function open(repoPath, { create = false } = {}) {
  return new Database(native.NativeDatabase.open(String(repoPath), create));
}

function init(repoPath) {
  return new Database(native.NativeDatabase.init(String(repoPath)));
}

module.exports = { open, init, Database, version: native.version() };

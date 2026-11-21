// Run: node bindings/node/test.js <model_dir> [<finetune_dir>]
'use strict';
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const weightdb = require('./index.js');

function readSafetensorsF32(file, name) {
  const raw = fs.readFileSync(file);
  const n = Number(raw.readBigUInt64LE(0));
  const info = JSON.parse(raw.subarray(8, 8 + n).toString())[name];
  const [b, e] = info.data_offsets;
  const bytes = raw.subarray(8 + n + b, 8 + n + e);
  return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

const model = process.argv[2];
const finetune = process.argv[3];
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'wdb-node-'));
try {
  const db = weightdb.open(path.join(tmp, 'repo'), { create: true });
  console.log('weightdb', weightdb.version);
  const r = db.importModel(model, { message: 'base' });
  console.log('import:', r.commit.id.slice(0, 12), r.stats);
  if (finetune) {
    const r2 = db.importModel(finetune, { message: 'finetune' });
    console.log('import ft:', r2.stats);
    assert(r2.stats.delta_chunks > 0);
  }
  const rev = finetune ? 'main~1' : 'main';
  const name = db.tensors(rev).find((t) => t.dtype === 'F32' && t.shape.length === 2).name;
  const t = db.readTensor(name, rev);
  const ref = readSafetensorsF32(path.join(model, 'model.safetensors'), name);
  assert.strictEqual(t.data.length, ref.length);
  assert.deepStrictEqual(Buffer.from(t.data.buffer), Buffer.from(ref.buffer));
  assert.strictEqual(db.weight(name, 5, rev), ref[5]);
  console.log(`readTensor ok: ${name} [${t.shape}] ${t.data.constructor.name}`);

  const info = db.tensorInfo(name, rev);
  console.log('stats:', info.stats);

  const rows = db.sql(
    "SELECT role, count(*) AS n, sum(nbytes) AS bytes FROM tensors WHERE commit_id = rev('main') GROUP BY role ORDER BY bytes DESC"
  );
  console.table(rows);
  if (finetune) {
    const d = db.diff('main~1', 'main', { numeric: true });
    console.log(`diff: ${d.changes.length} changed, ${d.unchanged} unchanged`);
  }
  assert.throws(() => db.readTensor('does.not.exist'), /no tensor/);
  assert.throws(() => db.sql('DROP TABLE tensors'));
  console.log('NODE OK');
} finally {
  fs.rmSync(tmp, { recursive: true, force: true });
}

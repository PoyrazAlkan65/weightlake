# WeightLake

**Sinir ağı ağırlıkları için gömülü bir veritabanı.** WeightLake model ağırlıklarını kapalı dosyalar olarak değil, *veri* olarak saklar.
Modeller Git gibi sürümlenir (commit, branch, diff), sürümler arasında tekilleştirilir ve delta ile sıkıştırılır, ve tek bir ağırlığa kadar SQL ile sorgulanabilir.
Çekirdek Rust ile yazıldı. **Rust, C/C++, Python ve JavaScript** için native API'ler var.

> 🇬🇧 English: [README.md](README.md)

```python
import weightlake, weightlake.hf as whf

db = weightlake.open("repo", create=True)
whf.save_pretrained(db, model, tokenizer=tok, message="base")              # model -> sürüm, dosya yok
# ... fine-tune ...
whf.save_pretrained(db, model, branch="ft", base="main", message="ep 1")   # sadece değişen chunk'lar, XOR delta olarak
model = whf.from_pretrained(db, "ft")                                      # sürüm -> model, dosya yok

db.sql("""SELECT name, tensor_rel_l2('main', 'ft', name) AS drift
          FROM tensors WHERE commit_id = rev('ft') ORDER BY drift DESC LIMIT 5""")
```

---

## İçindekiler

- [Neden](#neden)
- [Kavramlar](#kavramlar)
- [Özellikler](#özellikler)
- [Ölçülmüş örnek (GPT-2)](#ölçülmüş-örnek-gpt-2)
- [Başlangıç](#başlangıç)
- [CLI](#cli)
- [Python](#python)
  - [Hugging Face transformers](#hugging-face-transformers)
- [Rust](#rust)
- [C ve C++](#c-ve-c)
- [JavaScript / TypeScript](#javascript--typescript)
- [SQL referansı](#sql-referansı)
- [Nasıl çalışır](#nasıl-çalışır)
- [Depo yapısı](#depo-yapısı)
- [Kaynaktan derleme](#kaynaktan-derleme)
- [Testler](#testler)
- [Sınırlamalar](#sınırlamalar)
- [Yol haritası](#yol-haritası)
- [Lisans](#lisans)

---

## Neden

Model checkpoint'leri genelde büyük ikili dosyalar olarak taşınır.
7B'lik bir modelin on fine-tune'u, katmanların çoğu değişmemiş olsa bile, ~14 GB'lık on kopya demektir.
"Hangi katmanlar en çok değişti?" sorusunu cevaplamak için her checkpoint'i bir framework'e yüklemek gerekir.

WeightLake checkpoint'i yapılandırılmış veri olarak ele alır:

- **Her bayt bir kez saklanır.** Ağırlıklar içerik-adresli chunk'lara bölünür. 50 sürümde değişmeyen bir katman tek kez saklanır.
- **Değişiklikler delta olarak saklanır.** Değişen bir chunk, base sürümdeki aynı chunk ile XOR'lanıp sıkıştırılır. Fine-tune edilmiş ağırlıklar işaret, üs ve yüksek mantis bitlerinin çoğunu base ile paylaşır, bu yüzden XOR sonucu çoğunlukla sıfırdır.
- **Yüklemeden sorgulanır.** Her chunk istatistik taşır (min, max, ortalama, std, L2, sıfır, NaN/Inf). "En büyük normlu attention matrisleri" ya da "NaN içeren tensörler" gibi sorular ağırlık okunmadan indeksten cevaplanır.
- **Sadece gereken okunur.** Tek bir tensör ya da indeksle tek bir ağırlık okunabilir. Yalnızca onu içeren chunk çözülür.
- **Bayt bayt geri dönüş.** Bir `.safetensors` dosyası import edilip export edildiğinde birebir aynı dosya çıkar.

## Kavramlar

MongoDB ya da Git biliyorsan şu eşleme işin çoğunu anlatır:

| MongoDB    | Git        | WeightLake   | Ne |
|------------|------------|------------|----|
| Database   | Repository | **Repo**   | Bir model ailesini ve tüm sürümlerini tutan dizin |
| Collection | —          | **Tensör** | Bir ağırlık matrisi, ör. `h.0.attn.c_attn.weight` |
| Document   | Blob       | **Chunk**  | Tensörün tam satırlarından oluşan blok (~4 MiB), BLAKE3 hash'i ile adreslenir |
| Field      | —          | **Ağırlık** | Tek bir değer, `weight(rev, tensor, index)` ile okunur |
| Index      | —          | **Chunk istatistikleri** | Chunk başına min/max/sum/sum²/sıfır/NaN/Inf |
| —          | Commit     | **Commit** | Bir model sürümü: manifest (tensör → chunk listesi) + parent + mesaj |
| —          | Branch     | **Branch** | Bir commit'e işaret eden isim (`main`, `ft`, …) |

Bir **revizyon** (`rev`) şunlardan biri olabilir: branch adı (`main`), commit id öneki (`d1c2f4b6`) ya da ata (`main~1`, `ft~3`).

Her tensöre adından çıkarılan bir **rol** atanır: `embedding`, `attention`, `mlp`, `norm`, `head` veya `other`. Rol SQL ve CLI filtrelerinde kullanılabilir.

## Özellikler

| Alan | İçerik |
|---|---|
| Depolama | İçerik-adresli chunk'lar (BLAKE3), sürümler arası tekilleştirme, mmap'lenen değişmez pack dosyaları, redb katalog |
| Sıkıştırma | Base sürüme göre XOR delta + bayt-düzlemi gruplama + zstd; delta olmayan chunk'lar için isteğe bağlı zstd |
| Sürümleme | Commit, branch, `rev~N`, log, yapısal ve sayısal diff, çöp toplama (gc) |
| Sorgu | Commit, tensör ve chunk'lar üzerinde salt-okunur SQL (Apache DataFusion) ve ağırlık seviyesinde UDF'ler |
| Formatlar | safetensors import/export (tek dosya veya parçalı HF dizini, bayt bayt aynı); `config.json` ve tokenizer gibi yan dosyalar korunur |
| dtype'lar | F64, F32, F16, BF16, F8_E4M3, F8_E5M2, I64, I32, I16, I8, U64, U32, U16, U8, BOOL |
| API'ler | Rust, C (kararlı ABI), C++17 header, Python (numpy / torch / pandas), Node.js (TypedArray + TypeScript tipleri), CLI |
| ML entegrasyonu | Hugging Face `transformers`: veritabanına doğrudan `save_pretrained` / `from_pretrained`; her PyTorch/numpy state dict için `commit_state_dict` |

## Ölçülmüş örnek (GPT-2)

Aşağıdakiler, bir dizüstü SSD'de GPT-2 small (124M parametre, F32, 525 MiB ağırlık) üzerinde CLI ile ölçülen gerçek değerlerdir.
İkinci sürüm, 160 tensörden 28'ini (son blok ve son norm) değiştiren sentetik bir fine-tune'dur.

| Adım | Sonuç |
|---|---|
| `import gpt2` | 160 tensör, 525 MiB, **0,8 sn** |
| `import gpt2-ft --base main` | Chunk'ların %88'i yeniden kullanıldı; değişen 62 MiB **41,5 MiB** XOR delta olarak yazıldı, **0,5 sn** |
| Toplam | 2 sürüm = 1,02 GiB mantıksal → **523 MiB saklanan** (2,00× tekilleştirme oranı) |
| `export ft` | 548 MB dosya, orijinalle **bayt bayt aynı**, 0,9 sn |
| Hub import'unun üzerine değişmemiş modelle `whf.save_pretrained` | **+1 080 bayt** (sadece `config.json` yeni) |

Delta oranı büyük ölçüde veriye bağlıdır. Sentetik rastgele değişiklikler, ardışık sürümlerin küçük güncellemelerle ayrıldığı gerçek fine-tune ya da eğitim checkpoint'lerine göre daha kötü sıkışır.
Bu sayıları kapsamlı bir benchmark olarak değil, hızlı bir doğrulama olarak gör.

## Başlangıç

WeightLake henüz crates.io, PyPI ya da npm'de yayınlanmadı. Kaynaktan derlenir (bkz. [Kaynaktan derleme](#kaynaktan-derleme)):

```sh
git clone https://github.com/PoyrazAlkan65/weightlake && cd weightlake
cargo build --release --workspace
scripts/build-bindings.sh          # .pyd/.so, .node ve C kütüphanesini paket dizinlerine koyar
```

Sonra:

```sh
./target/release/wlake -R repo init
./target/release/wlake -R repo import path/to/hf-model-dir -m "base"
./target/release/wlake -R repo ls main
```

Python paketini checkout'tan kullanmak için `bindings/python` dizinini `PYTHONPATH`'e ekle ya da `maturin build -m bindings/python/Cargo.toml --release` ile wheel üret.
Node için `require('./bindings/node')`.

## CLI

```
wlake [-R <repo>] [--json] <komut>
```

`-R` repo dizinini seçer (varsayılan: bulunduğun dizin). `--json` her komut için makine-okunur çıktı verir.

| Komut | Açıklama |
|---|---|
| `init` | Boş bir repo oluşturur |
| `import <yol> [-b branch] [-m mesaj] [--base rev] [--no-delta] [--compress] [--chunk-mib N] [--no-aux]` | Bir `.safetensors` dosyasını ya da model dizinini (tüm `*.safetensors` parçaları + ≤ 64 MiB yan dosyalar) yeni commit olarak alır |
| `export <rev> <çıktı-dizini>` | Bir sürümü orijinal dosyalar olarak yazar (bayt bayt aynı) |
| `log [rev] [-n N]` | İlk-parent geçmişi |
| `branch` / `branch <ad> [at]` / `branch -d <ad>` | Branch listeler, oluşturur veya siler |
| `ls [rev] [--role R] [--grep S]` | Tensörleri dtype, şekil ve boyutla listeler |
| `show <rev> <tensör>` | Chunk indeksinden tensör istatistikleri (ağırlık okumaz) |
| `get <rev> <tensör> <indeks>...` | Düz indeksle tek tek ağırlık okur |
| `diff <a> <b> [--numeric]` | Değişen, eklenen ve silinen tensörler. `--numeric` göreli L2 ve kosinüs sapmasını da ekler |
| `sql "<sorgu>"` | Salt-okunur SQL sorgusu, tablo olarak yazdırılır |
| `stats` | Mantıksal ve saklanan boyut, tekilleştirme oranı, codec dağılımı |
| `gc [--dry-run]` | Hiçbir branch'ten erişilemeyen sürümleri siler, pack dosyalarını sıkıştırır |

```sh
wlake -R repo init
wlake -R repo import ./gpt2 -m "gpt2 base"
wlake -R repo import ./gpt2-ft -b ft --base main -m "fine-tune"   # main'e göre delta
wlake -R repo log ft
wlake -R repo ls main --role attention
wlake -R repo show main h.0.attn.c_attn.weight
wlake -R repo get main h.0.attn.c_attn.weight 0 1 2
wlake -R repo diff main ft --numeric
wlake -R repo sql "SELECT role, count(*), sum(nbytes) FROM tensors WHERE commit_id = rev('main') GROUP BY role"
wlake -R repo stats
wlake -R repo branch -d ft && wlake -R repo gc
wlake -R repo export main ./out
```

## Python

```python
import weightlake

db = weightlake.open("repo", create=True)          # ya da weightlake.init("repo")

# sürümler
r = db.import_model("gpt2", message="base")       # dosya veya dizin; import raporunu döner
db.import_model("gpt2-ft", branch="ft", base="main", message="ft", compress=False, chunk_mib=4)
db.log("ft")                                      # commit listesi
db.branches()                                     # {"main": "<id>", "ft": "<id>"}
db.create_branch("exp", "main~1"); db.delete_branch("exp")
db.export("ft", "out/")                           # bayt bayt aynı safetensors + yan dosyalar

# tensörler ("matrisler") ve ağırlıklar ("field'lar")
db.tensor_names("main")
db.tensor_info("h.0.attn.c_attn.weight")          # dtype, şekil, rol, istatistik (indeksten)
w  = db.read_tensor("h.0.attn.c_attn.weight")     # numpy; BF16 varsayılan olarak float32'ye çevrilir
w  = db.read_tensor("h.0.attn.c_attn.weight", framework="torch")
w  = db["main"]["h.0.attn.c_attn.weight"]         # sözlük tarzı erişim
x  = db.weight("h.0.attn.c_attn.weight", 12345)   # tek değer; sadece bir chunk çözülür
sd = db.state_dict("main", framework="torch")     # model.load_state_dict(sd)

# bellekten commit (bitişik CPU tensörleri için sıfır kopya)
db.commit_state_dict(model.state_dict(), message="step 1000", aux={"config.json": cfg_json})
db.aux_files("main"); db.read_aux("config.json", "main")

# diff / SQL / bakım
db.diff("main", "ft", numeric=True)
db.sql("SELECT name, l2 FROM tensors WHERE commit_id = rev('main')")   # list[dict]
db.sql_df("SELECT * FROM chunks")                                      # pandas.DataFrame
print(db.sql_pretty("SELECT codec, count(*) FROM chunks GROUP BY codec"))
db.stats(); db.gc(dry_run=True)
```

`commit_state_dict` torch tensörlerini ve numpy dizilerini kabul eder.
GPU'daki ya da bitişik olmayan tensörler önce bitişik CPU belleğine kopyalanır, big-endian numpy dizileri little-endian'a çevrilir.
Rust çekirdeği buffer'ları yerinde okur, hash ve sıkıştırma sırasında GIL'i bırakır.

Hatalar `weightlake.WeightLakeError` (bir `RuntimeError`) olarak fırlatılır.

### Hugging Face transformers

`weightlake.hf` ağırlıkları RAM'deki bir `transformers` modeli ile veritabanı arasında doğrudan taşır. `.safetensors` dosyası yazılmaz.
Yalnızca küçük yan dosyalar (`config.json`, tokenizer dosyaları) geçici bir dizinden geçer, çünkü transformers bunları dosya yolundan okur.

```python
import torch, weightlake, weightlake.hf as whf
from transformers import AutoModelForCausalLM, AutoTokenizer

db = weightlake.open("repo", create=True)

model = AutoModelForCausalLM.from_pretrained("gpt2")
tok = AutoTokenizer.from_pretrained("gpt2")
whf.save_pretrained(db, model, tokenizer=tok, message="base")   # config, generation_config, tokenizer dahil

# ... eğitim ...
whf.save_pretrained(db, model, branch="ft", base="main", message="epoch 1")

model = whf.from_pretrained(db, "ft")                            # sınıf config.architectures'tan
model = whf.from_pretrained(db, "ft", model_cls=AutoModelForCausalLM, dtype=torch.bfloat16, device="cuda")
tok   = whf.load_tokenizer(db, "ft")
cfg   = whf.load_config(db, "ft")
```

Bilmen gerekenler:

- **Bağlı ağırlıklar ek yer kaplamaz.** Örneğin `lm_head.weight` ve `wte.weight` ikisi de saklanır, ama chunk'ları aynı olduğu için tek kez yazılır.
- **Hub dosyaları ile `state_dict()` farklı isimler kullanabilir.** Eski hub dosyalarında isimler `h.0.…` biçimindedir, `model.state_dict()` ise `transformer.h.0.…` üretir. Delta yalnızca aynı isimli tensörler arasında eşleşir, ama içerik tekilleştirmesi yine çalışır. En iyi delta sıkıştırması için bir modelin bütün sürümlerini aynı yolla commit et.
- **Tokenizer yeniden yazılır.** transformers `tokenizer.json`'ı hub kopyasından farklı biçimde yazar, bu yüzden ilk tokenizer kaydı birkaç MiB ekler. Aynı tokenizer'ın sonraki kayıtları tekilleşir.
- **Yükleme sırasında RAM kullanımı bir an için model boyutunun yaklaşık iki katına çıkar.** Veritabanından okunan kopya ile modelin kendi parametreleri aynı anda bellekte durur.

## Rust

```rust
use weightlake::{Database, ImportOptions, MemTensor, DType};

let db = Database::open_or_init("repo")?;
let r = db.import("gpt2", &ImportOptions { message: "base".into(), ..Default::default() })?;
println!("{} {:?}", r.commit.short(), r.stats);

for t in db.tensors("main")? {
    println!("{} {:?} {} {}", t.name, t.shape, t.dtype, t.role);
}
let t = db.read_tensor("main", "h.0.attn.c_attn.weight")?;   // t.info, t.data (LE baytlar), t.to_f64()
let x = db.read_weight("main", "h.0.attn.c_attn.weight", 42)?;
let d = db.diff("main", "ft", true)?;
let rows = db.sql_json("SELECT name, l2 FROM tensors WHERE commit_id = rev('main') ORDER BY l2 DESC LIMIT 5")?;
let batches = db.sql("SELECT * FROM chunks")?;             // Arrow RecordBatch'leri

// bellekten commit
let data: Vec<u8> = vec![0; 16];
db.import_tensors(
    vec![MemTensor { name: "w".into(), dtype: DType::F32, shape: vec![2, 2], data: &data }],
    &[("config.json".into(), b"{}".to_vec())],
    &ImportOptions { branch: "scratch".into(), ..Default::default() },
)?;
```

`Database` ucuz kopyalanır ve thread'ler arasında güvenle paylaşılabilir.

`ImportOptions` alanları (JSON'dan okunurken hepsi isteğe bağlı):

| Alan | Varsayılan | Anlamı |
|---|---|---|
| `branch` | `"main"` | Commit edilecek branch |
| `message` | `"import"` | Commit mesajı |
| `base` | `None` | Delta alınacak revizyon. `None` ise branch'in son commit'i kullanılır |
| `delta` | `true` | Değişen chunk'ları base'e göre XOR delta ile sakla |
| `compress` | `false` | Delta olmayan chunk'ları zstd ile sıkıştır |
| `chunk_bytes` | 4 MiB | Hedef chunk boyutu (her zaman tam satırlar) |
| `include_aux` | `true` | Model dizinindeki ağırlık dışı dosyaları sakla |

## C ve C++

`weightlake.dll` / `libweightlake.so` / `libweightlake.dylib` ile linkle ve `bindings/c/include/weightlake.h` (C) ya da `weightlake.hpp` (C++17, sadece header'dan oluşan RAII sarmalayıcı) dosyasını ekle.

```c
#include "weightlake.h"

wl_db* db;
if (wl_open_or_init("repo", &db) != 0) { fprintf(stderr, "%s\n", wl_last_error()); return 1; }

wl_tensor_info_t info;
wl_tensor_info(db, "main", "h.0.attn.c_attn.weight", &info);
float* buf = malloc(info.nbytes);
wl_read_tensor(db, "main", "h.0.attn.c_attn.weight", buf, info.nbytes);

char* json;
wl_sql_json(db, "SELECT name, l2 FROM tensors WHERE commit_id = rev('main')", &json);
puts(json);
wl_string_free(json);
wl_close(db);
```

Her fonksiyon başarıda `0`, hatada `-1` döner. `wl_last_error()` o thread'deki hata mesajını verir.
`char**` ile dönen string'ler `wl_string_free` ile serbest bırakılmalıdır.
Fonksiyonlar: `wl_open/init/open_or_init/close`, `wl_import`, `wl_export`, `wl_resolve`, `wl_log_json`, `wl_branches_json`, `wl_create_branch`, `wl_delete_branch`, `wl_tensors_json`, `wl_tensor_info(_json)`, `wl_read_tensor`, `wl_read_weight`, `wl_diff_json`, `wl_sql_json`, `wl_sql_pretty`, `wl_stats_json`, `wl_gc_json`, `wl_version`.

```cpp
#include "weightlake.hpp"

auto db = weightlake::Database::open("repo");
std::vector<float> w = db.read_tensor<float>("main", "h.0.attn.c_attn.weight");
std::cout << db.sql_pretty("SELECT role, count(*) FROM tensors GROUP BY role");
```

C++ sarmalayıcısı hata durumunda `weightlake::Error` fırlatır. Tam örnek: `bindings/c/examples/example.cpp`.

## JavaScript / TypeScript

```js
const weightlake = require('weightlake');            // ya da require('./bindings/node')

const db = weightlake.open('repo', { create: true });
db.importModel('gpt2', { message: 'base' });
db.importModel('gpt2-ft', { branch: 'ft', base: 'main', message: 'ft' });

const { data, shape, dtype } = db.readTensor('h.0.attn.c_attn.weight');   // Float32Array vb.
const buf = db.readTensorBuffer('h.0.attn.c_attn.weight', 'ft');          // ham Buffer
const x = db.weight('h.0.attn.c_attn.weight', 42);
db.diff('main', 'ft', { numeric: true });
db.sql("SELECT name, l2 FROM tensors WHERE commit_id = rev('main')");
console.log(db.sqlPretty('SELECT codec, count(*) FROM chunks GROUP BY codec'));
```

Metotlar: `importModel`, `export`, `resolve`, `log`, `branches`, `createBranch`, `deleteBranch`, `tensors`, `tensorNames`, `tensorInfo`, `readTensor`, `readTensorBuffer`, `weight`, `diff`, `sql`, `sqlPretty`, `stats`, `gc`.
TypeScript tipleri `bindings/node/index.d.ts` dosyasında.
BF16 ve F16 tensörler `{ raw: true }` verilmedikçe `Float32Array`'e çevrilir. FP8 tensörler ham `Uint8Array` olarak döner.

## SQL referansı

SQL salt-okunurdur: DDL ve DML komutları reddedilir. Motor [Apache DataFusion](https://datafusion.apache.org/) olduğu için standart SQL çalışır: join, aggregate, window fonksiyonları, CTE.

### Tablolar

| Tablo | Kolonlar |
|---|---|
| `commits` | `commit_id`, `short_id`, `message`, `created`, `timestamp`, `parent`, `n_tensors`, `logical_bytes`, `branches` |
| `refs` | `name`, `commit_id` |
| `tensors` | `commit_id`, `name`, `dtype`, `shape`, `ndim`, `numel`, `nbytes`, `role`, `n_chunks`, `mean`, `std`, `l2`, `abs_mean`, `min`, `max`, `sparsity`, `nan_count`, `inf_count` |
| `chunks` | `hash`, `codec`, `raw_len`, `stored_len`, `pack`, `base`, `mean`, `std`, `l2`, `min`, `max`, `zeros` |
| `tensor_chunks` | `commit_id`, `tensor`, `idx`, `hash`, `nbytes` |

`tensors` tablosundaki istatistikler chunk indeksinden toplanır, ağırlık okumaz.

### Fonksiyonlar

| Fonksiyon | Döndürdüğü |
|---|---|
| `rev(text)` | Branch, id öneki ya da `rev~N` için commit id |
| `weight(rev, tensor, index)` | Tek bir ağırlık (`DOUBLE`). Sadece onu içeren chunk çözülür |
| `tensor_cosine(a, b, tensor)` | Bir tensörün iki sürüm arasındaki kosinüs benzerliği |
| `tensor_rel_l2(a, b, tensor)` | ‖b − a‖ / ‖a‖ |
| `tensor_max_abs_diff(a, b, tensor)` | max \|b − a\| |

Son üçü iki tensörü de okur. Bu yüzden onları seçici bir `WHERE` koşulundan sonra kullan.

### Örnekler

```sql
-- en büyük normlu attention matrisleri
SELECT name, shape, l2 FROM tensors
WHERE commit_id = rev('main') AND role = 'attention' AND ndim = 2
ORDER BY l2 DESC LIMIT 5;

-- fine-tune hangi katmanları ne kadar değiştirdi
SELECT b.name, tensor_rel_l2('main', 'ft', b.name) AS drift
FROM tensors a JOIN tensors b ON a.name = b.name
WHERE a.commit_id = rev('main') AND b.commit_id = rev('ft') AND a.mean <> b.mean
ORDER BY drift DESC;

-- sağlık kontrolü: son sürümde NaN / Inf var mı?
SELECT name, nan_count, inf_count FROM tensors
WHERE commit_id = rev('main') AND (nan_count > 0 OR inf_count > 0);

-- role göre parametre sayısı
SELECT role, sum(numel) AS params, sum(nbytes) AS bytes FROM tensors
WHERE commit_id = rev('main') GROUP BY role ORDER BY params DESC;

-- codec başına depolama verimi
SELECT codec, count(*) n, sum(raw_len) raw, sum(stored_len) stored,
       round(1.0 * sum(stored_len) / sum(raw_len), 3) AS ratio
FROM chunks GROUP BY codec;

-- her chunk kaç sürümde paylaşılıyor
SELECT hash, count(DISTINCT commit_id) AS versions FROM tensor_chunks
GROUP BY hash ORDER BY versions DESC LIMIT 10;

-- tek ağırlıkları sürümler arasında karşılaştır
SELECT weight('main', 'h.0.attn.c_attn.weight', 0) AS base, weight('ft', 'h.0.attn.c_attn.weight', 0) AS ft;
```

## Nasıl çalışır

### Disk düzeni

```
repo/
  catalog.redb          chunk indeksi, manifest'ler, commit'ler, ref'ler (redb, ACID)
  packs/000001.pack     değişmez chunk verisi, 64 bayt hizalı, mmap'lenir
  packs/000002.pack
```

Her import yeni chunk'larını yeni pack dosyalarına yazar, ardından tek bir katalog transaction'ı commit edilir.
Pack'ler bundan sonra hiç değiştirilmez; onları sadece `gc` yeniden yazar.
Bu sayede yeni veri yazılırken okuyucular pack'leri güvenle mmap'leyebilir, yarıda kesilen bir import da yarım bir sürüm bırakmaz.

### Chunk'lama ve hash

Tensör yaklaşık `chunk_bytes` (varsayılan 4 MiB) boyutunda chunk'lara bölünür. Bölme her zaman **satır sınırlarında** yapılır, yani bir chunk matrisin tam satırlarından oluşan bir bloktur.
Her chunk'ın ham baytları **BLAKE3** ile hash'lenir. Bu hash chunk'ın kimliğidir: aynı baytlar hangi tensörde ya da sürümde olursa olsun bir kez saklanır.

Hash'leme sırasında chunk başına istatistikler paralel (rayon) hesaplanır: sayı, min, max, toplam, kareler toplamı, sıfır, NaN ve Inf.
Tensör seviyesindeki istatistikler (ortalama, std, L2, seyreklik, …) bunlardan toplanır, bu yüzden ağırlık okumaya gerek kalmaz.

### Codec'ler

| Codec | Ne zaman | Kodlama |
|---|---|---|
| `raw` | Yeni chunk'lar için varsayılan | Baytlar olduğu gibi; okuma mmap'ten sıfır kopyalı dilimdir |
| `zstd` | `compress=true` | Bayt-düzlemi gruplama (her elemanın *k*. baytı *k*. düzleme), sonra zstd. Float üs baytları çok tekrarlı olduğu için iyi sıkışır |
| `xor-zstd` | Base'i olan değişmiş chunk | Base chunk ile XOR, bayt-düzlemi gruplama, sonra zstd. Küçük güncellemeler işaret, üs ve yüksek mantis bitlerini değiştirmez, bu yüzden XOR çoğunlukla sıfır bayttır |

Bir chunk'ın **base**'i, base sürümde aynı tensörün (aynı ad, dtype ve şekil) aynı indeksteki chunk'ıdır.
Base sürüm, verilmişse `base=`, yoksa hedef branch'in son commit'idir.

### Manifest ve commit'ler

Bir **manifest**, sürümün bütün tensörlerini (ad, dtype, şekil, rol, sıralı chunk referansları), her orijinal safetensors dosyasının düzenini (ham header, tensör sırası, veri uzunluğu) ve yan dosyaları listeler.
Orijinal header aynen saklandığı için export dosyaları bayt bayt yeniden üretir.
Bellekten commit edilen tensörler için `__metadata__: {"format": "pt"}` içeren yeni, standart bir header üretilir.

Bir **commit** bir manifest'e ve parent(lar)ına işaret eder, mesaj ve zaman damgası taşır.

### Çöp toplama

`gc`, herhangi bir branch'ten erişilebilen bütün commit'leri, bunların manifest ve chunk'larını, ve delta'ların base chunk'larını işaretler.
Geri kalan her şey silinir ve pack dosyaları ölü chunk'lar olmadan yeniden yazılır.

### Eşzamanlılık

Bir repo **aynı anda tek bir process** tarafından açılmak üzere tasarlandı. O process içinde `Database` handle'ı thread-safe'tir.

## Depo yapısı

```
crates/wl-core      dtype'lar, ChunkHash, satır hizalı chunk'lama, istatistik, manifest/commit tipleri
crates/wl-store     pack dosyaları (mmap) + redb katalog, codec'ler (raw / zstd / xor-zstd), gc
crates/wl-version   log, yapısal ve sayısal diff, tensör istatistikleri
crates/wl-io        safetensors import/export, bellekten import, yan dosyalar
crates/wl-query     SQL motoru (DataFusion tablo sağlayıcıları + UDF'ler)
crates/weightlake      genel Rust API: bütün binding'ler bu crate üzerine kurulu
crates/wl-cli       `wlake` komutu
bindings/c           C ABI (cdylib/staticlib) + include/weightlake.h + include/weightlake.hpp
bindings/python      PyO3 modülü `weightlake._native` + `weightlake` paketi (numpy/torch/pandas, `weightlake.hf`)
bindings/node        napi-rs eklentisi + `weightlake` npm paketi (TypedArray, TypeScript tipleri)
scripts/             build-bindings.sh, gen-libnode.sh
```

## Kaynaktan derleme

Gereksinimler: Rust (stable, 2021 edition). Python binding'i için Python ≥ 3.9 (abi3 sayesinde tek derleme her sürümde çalışır). Node binding'i için Node.js ≥ 18.

```sh
cargo build --release --workspace     # thin LTO: release derleme birkaç dakika sürer
scripts/build-bindings.sh             # derler ve çıktıları kopyalar:
                                      #   bindings/python/weightlake/_native.{pyd,so}
                                      #   bindings/node/weightlake.node
                                      #   dist/c/{include,lib}
```

**Python wheel:** `pip install maturin && maturin build -m bindings/python/Cargo.toml --release`
İsteğe bağlı ekler: `torch`, `pandas`. `transformers` sadece `weightlake.hf` için gerekir.

**Windows notları.** Proje Windows'ta **GNU toolchain** (`stable-x86_64-pc-windows-gnu`) ve `PATH`'teki MinGW-w64 (WinLibs) ile geliştirildi:

```sh
rustup override set stable-x86_64-pc-windows-gnu
scripts/gen-libnode.sh     # sadece GNU: node.exe'den N-API sembolleri için import kütüphanesi üretir
```

`.cargo/config.toml`, `LIBNODE_PATH`'i `bindings/node/libnode` dizinine yönlendirir. Tam bir MSVC + Windows SDK kurulumunda varsayılan MSVC toolchain'i de çalışmalıdır, orada `gen-libnode.sh` gerekmez.

Linux ve macOS henüz test edilmedi. Ancak kodda kütüphane dosya uzantıları dışında platforma özgü bir şey yok, o farkı da `build-bindings.sh` hallediyor.

## Testler

```sh
cargo test --workspace                                         # Rust birim + entegrasyon testleri

# Python (numpy gerekir; ikinci argüman isteğe bağlı)
python bindings/python/tests/test_weightlake.py <model_dizini> [<finetune_model_dizini>]

# Hugging Face entegrasyonu (torch + transformers gerekir)
python bindings/python/tests/test_hf.py <hf_model_dizini>

# Node
node bindings/node/test.js <model_dizini>

# C++
#   bindings/c/examples/example.cpp dosyasını dist/c'ye karşı derleyip bir repo üzerinde çalıştır
```

Python ve HF testleri, tensörlerin kaynak dosyalarla bit bit aynı olduğunu ve WeightLake'den yüklenen modellerin aynı logitleri ürettiğini kontrol eder.
Ayrıca delta/tekilleştirme davranışını, diff'leri, export'un geri dönüşünü ve dtype uç durumlarını (BF16, big-endian, bitişik olmayan, boş, bool) da test ederler.

## Sınırlamalar

- **Repo başına tek process.** Henüz sunucu ve çoklu yazıcı kilitlemesi yok.
- **Sadece safetensors.** GGUF, PyTorch `.bin`/`.pt` pickle'ları ve ONNX desteklenmiyor. Önce safetensors'a çevir.
- **Tam tensör okumaları RAM'e gelir.** `read_tensor`/`state_dict` tensörleri bütünüyle belleğe alır. Sadece `weight()` tek bir chunk okur.
- **Delta eşleşmesi tensör adına göre yapılır.** Adı değişen tensörler içerikle yine tekilleşir, ama delta ile kodlanmaz.
- **Henüz yayınlanmadı.** crates.io, PyPI ve npm'de yok. C ve Node binding'lerinde `import_tensors` henüz yok.

## Yol haritası

1. ✅ Depolama çekirdeği: chunk'lama, tekilleştirme, istatistik, safetensors import/export
2. ✅ Sürümleme: commit / branch / log / diff, XOR-delta + zstd, gc
3. ✅ Sorgu: SQL (DataFusion) + ağırlık seviyesinde UDF'ler
4. ◐ API'ler: Rust, C/C++, Python, JavaScript ✅ · Hugging Face transformers ✅ · GGUF ☐ · wheel / npm paketleri ☐
5. ☐ Sunucu: gRPC + Arrow Flight, repolar arası `push` / `pull`, uzaktan SQL

## Lisans

[MIT](LICENSE-MIT) veya [Apache License 2.0](LICENSE-APACHE) lisanslarından biri, tercihe bağlı.

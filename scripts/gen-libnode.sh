#!/usr/bin/env bash
# Windows + GNU toolchain only: napi-rs links N-API symbols against "libnode".
# Official Node.js exports them from node.exe, so generate a MinGW import
# library (libnode.dll.a -> node.exe) from node.exe's export table.
# napi-build also checks that a file named libnode.dll exists in LIBNODE_PATH;
# a stub marker is created for that check — the linker uses libnode.dll.a.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=bindings/node/libnode
mkdir -p "$OUT"
NODE_EXE=$(cygpath -u "$(node -p process.execPath)")
TMP=$(mktemp -d)
( cd "$TMP" && gendef - "$NODE_EXE" > node.def )
# keep only N-API / uv symbols
{ echo "LIBRARY node.exe"; echo "EXPORTS"; grep -E '^(napi_|node_api_|uv_)' "$TMP/node.def"; } > "$OUT/node.def"
dlltool -d "$OUT/node.def" -l "$OUT/libnode.dll.a" -D node.exe
: > "$OUT/libnode.dll"
rm -rf "$TMP"
echo "generated $OUT/libnode.dll.a ($(grep -c . "$OUT/node.def") lines in def)"

#!/usr/bin/env bash
# Build release artifacts and place native modules next to their packages.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build --release --workspace
T=target/release

case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) DLL_PREFIX=""; DLL_EXT=dll ;;
  Darwin)               DLL_PREFIX=lib; DLL_EXT=dylib ;;
  *)                    DLL_PREFIX=lib; DLL_EXT=so ;;
esac

# Python: weightdb/_native.pyd (.so on Unix)
PY_EXT=$([ "$DLL_EXT" = dll ] && echo pyd || echo so)
cp "$T/${DLL_PREFIX}_native.$DLL_EXT" "bindings/python/weightdb/_native.$PY_EXT"

# Node: weightdb.node
cp "$T/${DLL_PREFIX}wdb_node.$DLL_EXT" bindings/node/weightdb.node

# C/C++: dist/c/{include,lib}
mkdir -p dist/c/include dist/c/lib
cp bindings/c/include/weightdb.h bindings/c/include/weightdb.hpp dist/c/include/
cp "$T/${DLL_PREFIX}weightdb.$DLL_EXT" dist/c/lib/
for f in "$T/libweightdb.dll.a" "$T/libweightdb.a" "$T/weightdb.lib" "$T/weightdb.dll.lib"; do
  [ -f "$f" ] && cp "$f" dist/c/lib/ || true
done
cp "$T/wdb"* dist/ 2>/dev/null || true
echo "artifacts:"; ls -la bindings/python/weightdb/_native.* bindings/node/weightdb.node dist/c/lib

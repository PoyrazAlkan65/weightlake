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

# Python: weightlake/_native.pyd (.so on Unix)
PY_EXT=$([ "$DLL_EXT" = dll ] && echo pyd || echo so)
cp "$T/${DLL_PREFIX}_native.$DLL_EXT" "bindings/python/weightlake/_native.$PY_EXT"

# Node: weightlake.node
cp "$T/${DLL_PREFIX}wl_node.$DLL_EXT" bindings/node/weightlake.node

# C/C++: dist/c/{include,lib}
mkdir -p dist/c/include dist/c/lib
cp bindings/c/include/weightlake.h bindings/c/include/weightlake.hpp dist/c/include/
cp "$T/${DLL_PREFIX}weightlake.$DLL_EXT" dist/c/lib/
for f in "$T/libweightlake.dll.a" "$T/libweightlake.a" "$T/weightlake.lib" "$T/weightlake.dll.lib"; do
  [ -f "$f" ] && cp "$f" dist/c/lib/ || true
done
cp "$T/wlake"* dist/ 2>/dev/null || true
echo "artifacts:"; ls -la bindings/python/weightlake/_native.* bindings/node/weightlake.node dist/c/lib

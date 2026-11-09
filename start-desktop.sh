#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(uname -s)" != Linux ]]; then
  echo '此入口用于 Linux；Windows 请使用 start-desktop.cmd。' >&2
  exit 1
fi

# Prefer an explicit interpreter or the optional project-local toolchain. Do not
# change the user's global Node installation or require a particular Conda env.
VERSION="$(cat "$ROOT/.node-version")"
case "$(uname -m)" in
  x86_64) ARCH=x64 ;;
  aarch64) ARCH=arm64 ;;
  *) ARCH="$(uname -m)" ;;
esac
NODE=''
if [[ -n "${TENSOR_TRAIL_NODE:-}" ]]; then
  NODE="$TENSOR_TRAIL_NODE"
else
  for candidate in "$ROOT/.tools/node-v$VERSION-linux-$ARCH/bin/node" "${CONDA_PREFIX:-/nonexistent}/bin/node" "$(command -v node || true)"; do
    if [[ -x "$candidate" ]] && "$candidate" -e 'const [a,b]=process.versions.node.split(".").map(Number);process.exit(a>22||(a===22&&b>=12)?0:1)' 2>/dev/null; then
      NODE="$candidate"
      break
    fi
  done
fi
if [[ ! -x "$NODE" ]] || ! "$NODE" -e 'const [a,b]=process.versions.node.split(".").map(Number);process.exit(a>22||(a===22&&b>=12)?0:1)' 2>/dev/null; then
  echo '需要 Node.js >=22.12。请激活包含 Node 的环境，或设置 TENSOR_TRAIL_NODE=/绝对路径/node。' >&2
  exit 1
fi
exec "$NODE" "$ROOT/scripts/linux-desktop.cjs" "$@"

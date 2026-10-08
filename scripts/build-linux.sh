#!/usr/bin/env bash
#
# Build Koharu's Linux bundles.
#
# The wasm step (`@koharu/bridge build`) does not cache correctly on NTFS
# mounts exposed through ntfs3: cargo recompiles the same units on every run and
# the build never converges. When the checkout lives on a filesystem like that,
# this script mirrors the sources onto a Linux-native filesystem, builds there,
# and copies the bundles back. On ext4/xfs/btrfs it builds in place.
#
# Usage:
#   scripts/build-linux.sh [--bundles <list>] [--staging-dir <path>] [--clean]
#
# Requires: bash, rsync, findmnt, cargo (with the tauri CLI), bun.

set -euo pipefail

SAFE_FILESYSTEMS=("ext4" "xfs" "btrfs")
DEFAULT_BUNDLES="deb"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

usage() {
    cat <<'EOF'
Build Koharu's Linux bundles, routing through a native filesystem when the
checkout is on ntfs3 and the wasm step would not cache.

Options:
  --bundles <list>       Comma-separated bundle types (default: deb).
                         Examples: deb | deb,rpm,appimage
  --staging-dir <path>   Native-FS build directory used when staging.
                         Overrides $KOHARU_LINUX_BUILD_DIR.
  --clean                Remove the staging directory and exit.
  -h, --help             Show this help.

Environment:
  KOHARU_LINUX_BUILD_DIR  Staging directory (default: ~/.cache/koharu/linux-build).
EOF
}

filesystem_of() {
    findmnt -no FSTYPE -T "$1" 2>/dev/null | head -n1
}

is_safe_filesystem() {
    local fs
    for fs in "${SAFE_FILESYSTEMS[@]}"; do
        [[ "$1" == "$fs" ]] && return 0
    done
    return 1
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

bundles=""
staging_arg=""
clean=0

while (($#)); do
    case "$1" in
        --bundles)
            [[ $# -ge 2 ]] || die "--bundles needs a value"
            bundles="$2"
            shift 2
            ;;
        --bundles=*)
            bundles="${1#*=}"
            shift
            ;;
        --staging-dir)
            [[ $# -ge 2 ]] || die "--staging-dir needs a value"
            staging_arg="$2"
            shift 2
            ;;
        --staging-dir=*)
            staging_arg="${1#*=}"
            shift
            ;;
        --clean)
            clean=1
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            die "unknown option: $1"
            ;;
    esac
done

bundles="${bundles:-$DEFAULT_BUNDLES}"
[[ -n "$bundles" ]] || die "--bundles must not be empty"

staging="${staging_arg:-${KOHARU_LINUX_BUILD_DIR:-$HOME/.cache/koharu/linux-build}}"

if ((clean)); then
    rm -rf -- "$staging"
    printf 'removed staging directory %s\n' "$staging"
    exit 0
fi

command -v findmnt >/dev/null || die "findmnt not found; install util-linux"
command -v cargo >/dev/null || die "cargo not found"
command -v bun >/dev/null || die "bun not found"

workspace_fs="$(filesystem_of "$ROOT")"
[[ -n "$workspace_fs" ]] || die "could not determine filesystem of $ROOT"

if is_safe_filesystem "$workspace_fs"; then
    printf 'checkout is on %s; building in place\n' "$workspace_fs"
    cd -- "$ROOT"
    cargo tauri build --bundles "$bundles"
    printf '\nbundles: %s\n' "$ROOT/target/release/bundle"
    exit 0
fi

printf 'checkout is on %s; staging onto a native filesystem\n' "$workspace_fs"

# A staging directory nested inside the checkout would be rsynced into itself.
case "$staging/" in
    "$ROOT"/*) die "staging directory $staging is inside the checkout $ROOT" ;;
esac

mkdir -p -- "$staging"
staging_fs="$(filesystem_of "$staging")"
is_safe_filesystem "$staging_fs" ||
    die "staging directory $staging is on $staging_fs; set --staging-dir or \$KOHARU_LINUX_BUILD_DIR to a ${SAFE_FILESYSTEMS[*]} path"

command -v rsync >/dev/null || die "rsync not found; install rsync"

printf 'syncing sources to %s\n' "$staging"
rsync -a --delete \
    --exclude 'target/' \
    --exclude '.next/' \
    --exclude 'out/' \
    "$ROOT/" "$staging/"

cd -- "$staging"
if [[ ! -d node_modules ]]; then
    printf 'installing workspace dependencies\n'
    bun install
fi

cargo tauri build --bundles "$bundles"

printf 'copying bundles back to %s\n' "$ROOT/target/release/bundle"
IFS=',' read -r -a bundle_types <<<"$bundles"
for bundle_type in "${bundle_types[@]}"; do
    source_dir="$staging/target/release/bundle/$bundle_type"
    if [[ ! -d "$source_dir" ]]; then
        printf 'warning: no %s bundle produced at %s\n' "$bundle_type" "$source_dir" >&2
        continue
    fi
    destination="$ROOT/target/release/bundle/$bundle_type"
    mkdir -p -- "$destination"
    cp -a -- "$source_dir/." "$destination/"
    printf '  %s -> %s\n' "$bundle_type" "$destination"
done

printf '\ndone. staging directory %s kept; pass --clean to remove it\n' "$staging"

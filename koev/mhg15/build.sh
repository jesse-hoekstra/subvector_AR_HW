#!/bin/sh
set -eu

cd "$(dirname "$0")"
CC="${CC:-cc}"      # respects $CC env var, defaults to cc (the system compiler)
case "$(uname)" in
    Darwin) EXT=dylib ;;
    *)      EXT=so ;;
esac

LIB="libmhg.$EXT"
STAMP="$LIB.mhg_core.sha256"
tmp_lib=""
tmp_stamp=""

cleanup() {
    if [ -n "$tmp_lib" ]; then
        rm -f -- "$tmp_lib"
    fi
    if [ -n "$tmp_stamp" ]; then
        rm -f -- "$tmp_stamp"
    fi
}
trap cleanup 0 1 2 15

sha256_file() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    elif command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v openssl >/dev/null 2>&1; then
        openssl dgst -sha256 "$1" | awk '{print $NF}'
    else
        echo "No SHA-256 utility found (tried shasum, sha256sum, openssl)." >&2
        return 1
    fi
}

source_hash_before="$(sha256_file mhg_core.c)"
tmp_lib="$(mktemp ".${LIB}.tmp.XXXXXX")"
tmp_stamp="$(mktemp ".${STAMP}.tmp.XXXXXX")"

$CC -O3 -fPIC -shared -Wall mhg_core.c -o "$tmp_lib"

# Refuse to stamp a binary if the source changed while the compiler was
# reading it.  Both temporary files live beside their destinations, so each
# final rename is atomic on the same filesystem.
source_hash_after="$(sha256_file mhg_core.c)"
if [ "$source_hash_before" != "$source_hash_after" ]; then
    echo "mhg_core.c changed during compilation; build discarded." >&2
    exit 1
fi
if [ "${#source_hash_after}" -ne 64 ]; then
    echo "Unexpected SHA-256 output for mhg_core.c; build discarded." >&2
    exit 1
fi
case "$source_hash_after" in
    *[!0-9a-fA-F]*)
        echo "Invalid SHA-256 output for mhg_core.c; build discarded." >&2
        exit 1
        ;;
esac
printf '%s\n' "$source_hash_after" > "$tmp_stamp"
chmod 0644 "$tmp_stamp"

# Install the library first.  If the second rename is interrupted, the old or
# missing stamp makes Python fail closed instead of trusting an uncertain
# source/binary pairing.
mv -f -- "$tmp_lib" "$LIB"
tmp_lib=""
mv -f -- "$tmp_stamp" "$STAMP"
tmp_stamp=""

echo "built $LIB (mhg_core.c sha256 $source_hash_after)"

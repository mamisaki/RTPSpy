#!/usr/bin/env bash
# Build librtp.so in the AFNI build container (afni/afni_make_build, Ubuntu
# 18.04), so that the library needs only GLIBC 2.27 or later. No AFNI
# source clone or build packages are needed on the host.
#
#   afni_module/build_librtp_container.sh [--runtime R] [--image REF] [--out FILE]
#
#   --runtime R   apptainer, singularity, docker, or podman (default: detected)
#   --image REF   Build image: a .sif file or a docker image reference
#                 (default: afni/afni_make_build:AFNI_26.2.05; with Apptainer
#                 it is saved as containers/afni_build_<tag>.sif)
#   --out FILE    Output library (default: containers/librtp.so, which
#                 rtpspy loads in place of the bundled rtpspy/librtp.so)
#
# libmri.a and AFNI's own f2c (libf2c.a) are built from the AFNI source in
# the image and linked statically with librtp.c; the library links
# dynamically only to libz, libexpat, libm, and libc.

set -euo pipefail

BUILD_IMAGE="afni/afni_make_build:AFNI_26.2.05"
RUNTIME=auto
IMAGE=""
RTPSPY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$RTPSPY_DIR/containers/librtp.so"

while [ $# -gt 0 ]; do
    case "$1" in
        --runtime) RUNTIME="$2"; shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        -h|--help) sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

[ "$(uname -s)" = Linux ] || { echo "ERROR: builds the Linux library only." >&2; exit 1; }

if [ "$RUNTIME" = auto ]; then
    RUNTIME=""
    for r in apptainer singularity docker podman; do
        if command -v $r >/dev/null 2>&1; then RUNTIME=$r; break; fi
    done
    [ -n "$RUNTIME" ] || { echo "ERROR: no container runtime found." >&2; exit 1; }
fi

mkdir -p "$RTPSPY_DIR/containers" "$(dirname "$OUT")"
OUT="$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")"

# Build image
case "$RUNTIME" in
    apptainer|singularity)
        if [ -z "$IMAGE" ]; then
            IMAGE="$RTPSPY_DIR/containers/afni_build_${BUILD_IMAGE##*:}.sif"
            if [ ! -s "$IMAGE" ]; then
                echo "Pulling $BUILD_IMAGE -> $IMAGE ..."
                "$RUNTIME" pull --disable-cache "$IMAGE.part" "docker://$BUILD_IMAGE" \
                    && mv "$IMAGE.part" "$IMAGE" \
                    || { rm -f "$IMAGE.part"; exit 1; }
            fi
        fi ;;
    docker|podman)
        IMAGE="${IMAGE:-$BUILD_IMAGE}"
        "$RUNTIME" image inspect "$IMAGE" >/dev/null 2>&1 || "$RUNTIME" pull "$IMAGE" ;;
    *) echo "ERROR: unknown runtime $RUNTIME" >&2; exit 1 ;;
esac

# Work directory (removed at the end)
B="$(mktemp -d "$RTPSPY_DIR/containers/librtp_build.XXXXXX")"
trap 'rm -rf "$B"' EXIT
cp "$RTPSPY_DIR/afni_module/librtp.c" "$RTPSPY_DIR/afni_module/librtp.h" \
   "$RTPSPY_DIR/afni_module/AFNI_version.h" "$B/"

cat > "$B/build.sh" <<'EOF'
#!/bin/bash
set -e
cd "$(dirname "$0")"
AFNI_SRC_DIR=$(dirname "$(find /opt/afni/src -maxdepth 2 -name Makefile.linux_ubuntu_16_64 | head -1)")
cp -r "$AFNI_SRC_DIR" afni_src
A=afni_src
echo "Building libmri.a and libf2c.a (a few minutes) ..."
(cd $A && make -f Makefile.linux_ubuntu_16_64 libmri.a libf2c.a > ../make.log 2>&1) \
    || { tail -30 make.log; exit 1; }
echo "Linking librtp.so ..."
gcc -O3 -m64 -fPIC -DREAD_WRITE_64 -DHAVE_ZLIB -shared -o librtp.so librtp.c \
    $A/libmri.a $A/libf2c.a \
    -I. -I$A -I$A/f2c -I$A/nifti/nifti2 -I$A/nifti/niftilib \
    -I$A/nifti/nifticdf -I$A/nifti/znzlib -I$A/3DEdge/src -I$A/rickr \
    -I$A/niml -lz -lexpat -lm -w
EOF
chmod +x "$B/build.sh"

echo "Building librtp.so in $IMAGE ($RUNTIME) ..."
case "$RUNTIME" in
    apptainer|singularity)
        "$RUNTIME" -s exec --cleanenv -B "$B" "$IMAGE" "$B/build.sh" ;;
    docker)
        docker run --rm -u "$(id -u):$(id -g)" -v "$B:$B" --entrypoint "$B/build.sh" "$IMAGE" ;;
    podman)
        podman run --rm --userns=keep-id -v "$B:$B" --entrypoint "$B/build.sh" "$IMAGE" ;;
esac

[ -s "$B/librtp.so" ] || { echo "ERROR: librtp.so was not built." >&2; exit 1; }
install -m 0755 "$B/librtp.so" "$OUT"
echo "Installed $OUT"
GLIBC_REQ="$(objdump -T "$OUT" 2>/dev/null | grep -oE 'GLIBC_[0-9.]+' | sort -Vu | tail -1 || true)"
[ -n "$GLIBC_REQ" ] && echo "Requires $GLIBC_REQ or later"

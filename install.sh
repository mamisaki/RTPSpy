#!/usr/bin/env bash
# RTPSpy installer
#
#   ./install.sh [options]
#
# 1. Installs uv (https://docs.astral.sh/uv/) if it is not found.
# 2. Makes a Python environment in .venv (uv downloads Python itself) and
#    installs RTPSpy and its Python dependencies, including PyTorch built
#    for the detected GPU driver (or CPU) and dcm2niix.
# 3. Pulls the container images of the AFNI command-line programs and
#    FastSurfer with Apptainer (saved as .sif files in containers/) or
#    Docker, and writes containers/containers.json.
#    The AFNI programs are called through wrapper scripts in
#    containers/afni_bin, which rtpspy puts on PATH by itself.
#
# Options:
#   --runtime R        Container runtime: auto (default), apptainer,
#                      singularity, docker, podman, or none.
#   --afni M           auto (default): use AFNI installed on this host if
#                      found, otherwise the container. container: always use
#                      the container. host: never pull the AFNI container.
#   --no-fastsurfer    Do not pull the FastSurfer container.
#   --no-psychopy      Do not make the PsychoPy environment (.venv-psychopy)
#                      for the neurofeedback presentation of the example
#                      applications (example/ROI-NF, example/LA-NF).
#   --torch-backend B  PyTorch build: auto (default; detected from the
#                      NVIDIA driver), cpu, cu126, cu128, cu130, ...
#   --python V         Python version (default 3.13; the newest one with
#                      wheels for all the dependencies).
#   --container-dir D  Where to save the .sif images (default: containers/
#                      in the RTPSpy directory).
#   --bind DIRS        Extra host directories (colon-separated) to make
#                      visible in the containers, e.g. /mnt:/data.
#   --build-librtp M   auto (default): build librtp.so in a container
#                      (afni_module/build_librtp_container.sh) if the bundled
#                      one does not load. yes: always build. no: never.
#   -h, --help         Show this help.
#
# Re-running the script updates the environment; images that already exist
# are not downloaded again.

set -euo pipefail

# --- Versions (tested) -------------------------------------------------------
AFNI_IMAGE="afni/afni_docker_universal:AFNI_AFNI_26.3.09"
FASTSURFER_VERSION="v2.6.0"
FASTSURFER_REPO="deepmi/fastsurfer"
PYTHON_VERSION="3.13"
PSYCHOPY_PYTHON="3.12"   # PsychoPy 2026.2 supports Python < 3.13

# --- Options -----------------------------------------------------------------
RUNTIME=auto
AFNI_MODE=auto
WITH_FASTSURFER=1
WITH_PSYCHOPY=1
BUILD_LIBRTP=auto
TORCH_BACKEND=auto
CONTAINER_DIR=""
EXTRA_BINDS=""

usage() { sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
        --runtime) RUNTIME="$2"; shift 2 ;;
        --afni) AFNI_MODE="$2"; shift 2 ;;
        --no-fastsurfer) WITH_FASTSURFER=0; shift ;;
        --no-psychopy) WITH_PSYCHOPY=0; shift ;;
        --torch-backend) TORCH_BACKEND="$2"; shift 2 ;;
        --python) PYTHON_VERSION="$2"; shift 2 ;;
        --container-dir) CONTAINER_DIR="$2"; shift 2 ;;
        --bind) EXTRA_BINDS="$2"; shift 2 ;;
        --build-librtp) BUILD_LIBRTP="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

RTPSPY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_DIR="${CONTAINER_DIR:-$RTPSPY_DIR/containers}"
CONF_DIR="$RTPSPY_DIR/containers"
VENV="$RTPSPY_DIR/.venv"
PY="$VENV/bin/python"
PSY_VENV="$RTPSPY_DIR/.venv-psychopy"
PSY_PY="$PSY_VENV/bin/python"
OS="$(uname -s)"

msg()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mWARNING: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

NOTES=()

# --- 1. uv -------------------------------------------------------------------
msg "uv"
if ! command -v uv >/dev/null 2>&1; then
    for d in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
        [ -x "$d/uv" ] && export PATH="$d:$PATH"
    done
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv ..."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        die "curl or wget is needed to install uv."
    fi
    export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || die "uv was not found after installation."
uv --version

# --- 2. Python environment ---------------------------------------------------
msg "Python environment ($VENV)"
# (Re)make the environment if it is missing or has another Python version
CUR_PY="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
if [ "$CUR_PY" != "$PYTHON_VERSION" ]; then
    [ -n "$CUR_PY" ] && echo "Remaking the environment: Python $CUR_PY -> $PYTHON_VERSION"
    uv python install "$PYTHON_VERSION"
    uv venv --clear --python "$PYTHON_VERSION" --seed "$VENV"
fi

if [ "$OS" = "Darwin" ] && [ "$TORCH_BACKEND" = "auto" ]; then
    TORCH_BACKEND=cpu   # PyPI torch for macOS (with MPS)
fi
echo "PyTorch backend: $TORCH_BACKEND"
# --upgrade: the newest versions of all the packages
uv pip install --python "$PY" --torch-backend "$TORCH_BACKEND" --upgrade \
    -e "$RTPSPY_DIR[gui,test]"

"$PY" - <<'EOF'
import torch, dcm2niix  # noqa: F401
print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
EOF

# --- 2b. PsychoPy environment for the example applications ------------------
# PsychoPy supports Python < 3.13, so it has its own environment. The
# presentation scripts (example/*/NF_psypy.py) do not import rtpspy; they
# use a copy of rtpspy/rtp_serve.py in each example directory.
if [ "$WITH_PSYCHOPY" = 1 ]; then
    msg "PsychoPy environment ($PSY_VENV)"
    PSY_CUR="$("$PSY_PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
    if [ "$PSY_CUR" != "$PSYCHOPY_PYTHON" ]; then
        uv python install "$PSYCHOPY_PYTHON"
        uv venv --clear --python "$PSYCHOPY_PYTHON" --seed "$PSY_VENV"
    fi
    uv pip install --python "$PSY_PY" --upgrade psychopy pandas
    "$PSY_PY" -c "import psychopy; print(f'psychopy {psychopy.__version__}')"
fi

# --- 3. Container runtime ----------------------------------------------------
msg "Container runtime"
if [ "$RUNTIME" = "auto" ]; then
    RUNTIME=none
    for r in apptainer singularity; do
        if command -v $r >/dev/null 2>&1; then RUNTIME=$r; break; fi
    done
    if [ "$RUNTIME" = none ]; then
        for r in docker podman; do
            if command -v $r >/dev/null 2>&1 && $r info >/dev/null 2>&1; then
                RUNTIME=$r; break
            fi
        done
    fi
fi
if [ "$RUNTIME" != none ]; then
    command -v "$RUNTIME" >/dev/null 2>&1 || die "$RUNTIME is not found."
fi
echo "Runtime: $RUNTIME"

IS_SIF=0
case "$RUNTIME" in apptainer|singularity) IS_SIF=1 ;; esac
mkdir -p "$CONF_DIR"
[ "$IS_SIF" = 1 ] && mkdir -p "$CONTAINER_DIR"

# Pull an image. Apptainer: save it as a .sif file (skipped if it exists).
# Prints the image reference to use (file path or docker image).
pull_image() {
    local ref="$1" sif="$2"
    if [ "$IS_SIF" = 1 ]; then
        if [ -s "$sif" ]; then
            echo "Exists: $sif" >&2
        else
            echo "Pulling $ref -> $sif (this takes a while) ..." >&2
            # The layer cache is not kept (it would double the disk use)
            "$RUNTIME" pull --disable-cache "$sif.part" "docker://$ref" >&2 \
                && mv "$sif.part" "$sif" \
                || { rm -f "$sif.part"; return 1; }
        fi
        echo "$sif"
    else
        echo "Pulling $ref ..." >&2
        "$RUNTIME" pull "$ref" >&2 || return 1
        echo "$ref"
    fi
}

# --- 4. AFNI -----------------------------------------------------------------
msg "AFNI"
AFNI_JSON='{"mode": "none"}'
# AFNI on the host (not the wrappers of a previous installation)
HOST_AFNI="$(PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -vxF "$CONF_DIR/afni_bin" | paste -sd:)" command -v 3dvolreg || true)"
case "$AFNI_MODE" in
    auto) if [ -n "$HOST_AFNI" ]; then AFNI_MODE=host; else AFNI_MODE=container; fi ;;
    host|container) ;;
    *) die "--afni must be auto, host, or container." ;;
esac
if [ "$AFNI_MODE" = container ] && [ "$RUNTIME" = none ]; then
    warn "No container runtime: AFNI cannot be run from a container."
    NOTES+=("Install Apptainer (Linux) or Docker and re-run ./install.sh, or install AFNI on the host.")
    AFNI_MODE=none
fi

if [ "$AFNI_MODE" = host ]; then
    echo "Using AFNI on this host: $(dirname "$HOST_AFNI")"
    AFNI_JSON='{"mode": "host"}'
elif [ "$AFNI_MODE" = container ]; then
    AFNI_REF="$(pull_image "$AFNI_IMAGE" "$CONTAINER_DIR/afni_${AFNI_IMAGE##*:}.sif")" \
        || die "Failed to pull $AFNI_IMAGE"

    # Wrapper scripts, one per program in the image's AFNI directory
    AFNI_BIN="$CONF_DIR/afni_bin"
    rm -rf "$AFNI_BIN"
    mkdir -p "$AFNI_BIN"
    cat > "$AFNI_BIN/.afni_wrapper" <<EOF
#!/bin/sh
# Run an AFNI program in the container (made by install.sh)
exec "$PY" -I "$RTPSPY_DIR/rtpspy/containers.py" exec afni "\$(basename "\$0")" "\$@"
EOF
    chmod +x "$AFNI_BIN/.afni_wrapper"

    # AFNI directory in the image (it may not be on the image PATH)
    in_afni() {
        if [ "$IS_SIF" = 1 ]; then
            "$RUNTIME" -s exec --cleanenv "$AFNI_REF" sh -c "$1"
        else
            "$RUNTIME" run --rm --entrypoint sh "$AFNI_REF" -c "$1"
        fi
    }
    AFNI_PATH="$(in_afni 'for d in "$(dirname "$(command -v 3dinfo 2>/dev/null || echo /x/x)")" /usr/local/abin /opt/afni/install /opt/abin /home/afni_user/abin; do if [ -x "$d/3dinfo" ]; then echo "$d"; break; fi; done')"
    [ -n "$AFNI_PATH" ] || die "AFNI programs were not found in $AFNI_IMAGE"
    echo "AFNI directory in the image: $AFNI_PATH"
    PROGS="$(in_afni "cd '$AFNI_PATH' && for f in *; do [ -f \"\$f\" ] && [ -x \"\$f\" ] && echo \"\$f\"; done; true")"
    n=0
    for p in $PROGS; do
        case "$p" in
            # GUI programs (not run in the container), libraries, and
            # generic image tools that would shadow the host's
            afni|suma|aiv|plugout_*|*.so|*.so.*|cjpeg|djpeg) continue ;;
        esac
        ln -s .afni_wrapper "$AFNI_BIN/$p"
        n=$((n + 1))
    done
    [ "$n" -gt 0 ] || die "No AFNI programs were found in the image."
    echo "$n AFNI programs wrapped in $AFNI_BIN"
    AFNI_JSON="{\"mode\": \"container\", \"image\": \"$AFNI_REF\", \"source\": \"$AFNI_IMAGE\", \"bin_dir\": \"$AFNI_BIN\", \"path\": \"$AFNI_PATH\"}"
    NOTES+=("The AFNI GUI is not installed by the container. Use AFNI on the host to view images.")
else
    AFNI_JSON='{"mode": "none"}'
fi

# --- 5. FastSurfer -----------------------------------------------------------
msg "FastSurfer"
FS_JSON='{"mode": "none"}'
if [ "$WITH_FASTSURFER" = 1 ] && [ "$RUNTIME" != none ]; then
    # Image variant for the NVIDIA driver
    FS_VARIANT=cpu
    if [ "$OS" = Linux ] && command -v nvidia-smi >/dev/null 2>&1; then
        DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1 || true)"
        if [ -n "$DRV" ]; then
            if [ "$DRV" -ge 595 ]; then FS_VARIANT=cu132
            elif [ "$DRV" -ge 525 ]; then FS_VARIANT=cu126
            fi
            echo "NVIDIA driver $DRV"
        fi
    fi
    FS_IMAGE="$FASTSURFER_REPO:$FS_VARIANT-$FASTSURFER_VERSION"
    FS_REF="$(pull_image "$FS_IMAGE" "$CONTAINER_DIR/fastsurfer_$FS_VARIANT-$FASTSURFER_VERSION.sif")" \
        || die "Failed to pull $FS_IMAGE"
    FS_JSON="{\"mode\": \"container\", \"image\": \"$FS_REF\", \"source\": \"$FS_IMAGE\", \"run_cmd\": \"/fastsurfer/run_fastsurfer.sh\"}"
    if [ "$IS_SIF" = 0 ] && [ "$FS_VARIANT" != cpu ]; then
        NOTES+=("Docker GPU access needs the NVIDIA Container Toolkit.")
    fi
elif [ "$WITH_FASTSURFER" = 1 ]; then
    warn "No container runtime: FastSurfer is not installed (masks are made without FastSeg)."
fi

# --- 6. Configuration --------------------------------------------------------
msg "Configuration"
BINDS_JSON="$("$PY" -I -c 'import json, os, sys
dirs = [d for d in sys.argv[1].split(":") if d]
dirs += [d for d in ("/mnt", "/media", "/data", "/scratch") if os.path.isdir(d)]
print(json.dumps(list(dict.fromkeys(dirs))))' "$EXTRA_BINDS")"
cat > "$CONF_DIR/containers.json" <<EOF
{
  "runtime": "$RUNTIME",
  "binds": $BINDS_JSON,
  "afni": $AFNI_JSON,
  "fastsurfer": $FS_JSON
}
EOF
"$PY" -I -m json.tool "$CONF_DIR/containers.json"

# Put the AFNI wrappers on PATH when the environment is activated
ACT="$VENV/bin/activate"
if ! grep -q "RTPSpy AFNI wrappers" "$ACT"; then
    cat >> "$ACT" <<EOF

# RTPSpy AFNI wrappers (install.sh)
if [ -d "$CONF_DIR/afni_bin" ]; then
    PATH="$CONF_DIR/afni_bin:\$PATH"
    export PATH
fi
EOF
fi

# --- 7. librtp --------------------------------------------------------------
msg "librtp"
# Can this Python load the library? (prints the error if not)
lib_loads() {
    "$PY" -I -c 'import ctypes, sys; ctypes.cdll.LoadLibrary(sys.argv[1])' "$1"
}
BUNDLED_LIB="$RTPSPY_DIR/rtpspy/librtp.so"
[ "$OS" = Darwin ] && BUNDLED_LIB="$RTPSPY_DIR/rtpspy/librtp.dylib"
BUILT_LIB="$CONF_DIR/librtp.so"
case "$BUILD_LIBRTP" in
    auto)
        if [ -s "$BUILT_LIB" ] && lib_loads "$BUILT_LIB" 2>/dev/null; then
            echo "Using the library built in the container: $BUILT_LIB"
            BUILD_LIBRTP=no
        elif [ -f "$BUNDLED_LIB" ] && lib_loads "$BUNDLED_LIB"; then
            echo "The bundled library loads: $BUNDLED_LIB"
            BUILD_LIBRTP=no
        else
            echo "The bundled library does not load; building it in a container."
            BUILD_LIBRTP=yes
        fi ;;
    yes|no) ;;
    *) die "--build-librtp must be auto, yes, or no." ;;
esac
if [ "$BUILD_LIBRTP" = yes ]; then
    if [ "$OS" != Linux ]; then
        NOTES+=("librtp.dylib must be compiled on macOS: see Compilelibrtp.md.")
    elif [ "$RUNTIME" = none ]; then
        NOTES+=("librtp.so could not be built without a container runtime: see Compilelibrtp.md.")
    else
        "$RTPSPY_DIR/afni_module/build_librtp_container.sh" --runtime "$RUNTIME" --out "$BUILT_LIB" \
            || die "Building librtp.so failed."
        lib_loads "$BUILT_LIB" || die "The built librtp.so does not load."
    fi
fi

# --- 8. Checks ---------------------------------------------------------------
msg "Checks"
if [ "$OS" = Linux ]; then
    # Qt 6 (PyQt6) needs libxcb-cursor0 to open windows on X11
    # (not 'ldconfig -p | grep -q': grep exits early and pipefail fails)
    if ! grep -q 'libxcb-cursor\.so\.0' <<< "$(ldconfig -p 2>/dev/null)"; then
        NOTES+=("For the GUI (Qt 6) on X11: sudo apt install libxcb-cursor0")
    fi
    if ! grep -qx dialout <<< "$(id -nG | tr ' ' '\n')"; then
        NOTES+=("For physio recording on a serial port: sudo usermod -a -G dialout \$USER (then log in again).")
    fi
fi
if [ "$AFNI_MODE" = container ]; then
    if (cd "$RTPSPY_DIR" && "$AFNI_BIN/3dinfo" -space tests/anat_mprage.nii.gz >/dev/null 2>&1); then
        echo "AFNI container: OK ($AFNI_IMAGE)"
    else
        warn "Running 3dinfo in the AFNI container failed."
    fi
fi
"$PY" -c "import rtpspy; print('import rtpspy: OK')"

msg "Done"
cat <<EOF
Activate the environment:   source $VENV/bin/activate
Test the installation:      $PY $RTPSPY_DIR/rtpspy_system_check.py
EOF
for n in "${NOTES[@]}"; do echo "NOTE: $n"; done

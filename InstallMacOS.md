# Installing RTPSpy on macOS

This page gives the full installation on a Mac with Apple silicon (M1 or later). It adds the macOS-specific steps to the general [Installation](./README.md#installation): the librtp library must be compiled on your Mac, and FastSurfer (for the anatomical segmentation) is either run from a container in Docker Desktop or installed locally.

The steps were tested on macOS 26 (Apple silicon) **without Docker**, with AFNI installed on the Mac. The real-time processing runs on the CPU (no GPU computation on macOS); it is fast enough.

## Contents

- [1. What you need](#1-what-you-need)
- [2. Install RTPSpy](#2-install-rtpspy)
- [3. Compile the librtp library](#3-compile-the-librtp-library)
- [4. FastSurfer without Docker](#4-fastsurfer-without-docker)
- [5. Check the installation](#5-check-the-installation)
- [Troubleshooting](#troubleshooting)

## 1. What you need

| | Notes |
|---|---|
| [Homebrew](https://brew.sh/) | For the packages needed to compile librtp. |
| Xcode Command Line Tools | The C compiler (`clang`). Install with `xcode-select --install` if `xcode-select -p` shows nothing. |
| AFNI **or** Docker Desktop | The AFNI command-line programs. Either install AFNI on the Mac ([AFNI macOS instructions](https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/background_install/install_instructs/steps_mac.html); the programs, e.g. `3dvolreg`, must be on your PATH), or install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and let the installer use the AFNI container. |

Install the Homebrew packages for librtp now:

```bash
brew install libpng jpeg expat freetype fontconfig openmotif libomp \
    libxt gsl glib pkg-config autoconf mesa mesa-glu libxpm netpbm libiconv
```

## 2. Install RTPSpy

```bash
git clone https://github.com/mamisaki/RTPSpy.git ~/RTPSpy
cd ~/RTPSpy
```

Then run the installer with the options for your setup.

**With AFNI installed on the Mac, without Docker** (tested):

```bash
./install.sh --runtime none --no-fastsurfer
```

- The installer uses the AFNI on your PATH (it shows `Using AFNI on this host: ...`).
- `--no-fastsurfer`: FastSurfer is installed locally in [step 4](#4-fastsurfer-without-docker).

**With Docker Desktop** (start Docker Desktop first):

```bash
./install.sh --runtime docker
```

This downloads the AFNI and FastSurfer container images; skip step 4.

The installer makes `~/RTPSpy/.venv` (RTPSpy, PyTorch for the CPU with Apple's MPS, and the other packages) and `~/RTPSpy/.venv-psychopy` (PsychoPy, for the example applications). At the end it shows the note "librtp.dylib must be compiled on macOS"; that is the next step.

> [!NOTE]
> If the downloads fail with `invalid peer certificate: UnknownIssuer` (this happens on networks that inspect secure connections), make uv use the certificates of macOS and run the installer again:
>
> ```bash
> UV_SYSTEM_CERTS=1 ./install.sh --runtime none --no-fastsurfer
> ```
>
> (Older versions of uv use `UV_NATIVE_TLS=1` instead.) Use the same variable for the `uv pip install` command in step 4.

## 3. Compile the librtp library

RTPSpy uses some AFNI C functions compiled into `librtp.dylib` (needed for motion correction and smoothing). It is not included for macOS. Compile it from the AFNI source code:

```bash
cd ~/RTPSpy/afni_module
git clone --depth 1 https://github.com/afni/afni.git
LOCAL_CC_PATH=/usr/bin/clang make -f Makefile.librtp.macos_13_ARM
```

The library is installed as `~/RTPSpy/rtpspy/librtp.dylib`.

> [!NOTE]
> `LOCAL_CC_PATH=/usr/bin/clang` uses Apple's compiler. The Makefile's default, Homebrew's `gcc-13` (or `gcc-14`), fails on a macOS newer than the one it was built for, with `fatal error: _stdio.h: No such file or directory`.

To update the library after updating RTPSpy, run the `make` command again (and `git -C afni pull` to update the AFNI source).

## 4. FastSurfer without Docker

RTPSpy makes the brain, white-matter, and ventricle masks with [FastSurfer](https://github.com/Deep-MI/FastSurfer). Without Docker, it uses a FastSurfer clone in `~/RTPSpy/rtpspy/FastSurfer` (this replaces [Using a local FastSurfer](./README.md#using-a-local-fastsurfer-instead-of-the-container) of the README on macOS).

### 4.1 Get FastSurfer

```bash
git clone --branch v2.6.0 https://github.com/Deep-MI/FastSurfer.git ~/RTPSpy/rtpspy/FastSurfer
```

### 4.2 Install its Python packages

Install FastSurfer's packages **together with RTPSpy**, so that the versions fit both:

```bash
cd ~/RTPSpy
.venv/bin/python -c "
import tomllib
p = tomllib.load(open('rtpspy/FastSurfer/pyproject.toml', 'rb'))['project']
o = p['optional-dependencies']
print('\n'.join(p['dependencies'] + o['qc'] + o['lit']))
" > fastsurfer_requirements.txt
uv pip install --python .venv/bin/python --torch-backend cpu \
    -e ".[gui,test]" -r fastsurfer_requirements.txt
uv pip check --python .venv/bin/python
rm fastsurfer_requirements.txt
```

The last command should print `All installed packages are compatible`. (If `uv` is not found, it is `~/.local/bin/uv`.)

> [!WARNING]
> Do not install FastSurfer's `requirements.txt` into the RTPSpy environment. It fixes exact versions (e.g., numpy 2.5, scipy 1.18) that break packages RTPSpy needs (ANTsPy requires numpy < 2.4 and scipy < 1.16). If you did, run the commands above to repair the environment.

### 4.3 Download the network weights

FastSurfer downloads its network weights (three files, about 22 MB each) the first time it runs. If your network blocks this (an `SSLError` in the log), download them yourself:

```bash
cd ~/RTPSpy/rtpspy/FastSurfer
mkdir -p checkpoints
for v in axial coronal sagittal; do
    curl -fL -o checkpoints/aparc_vinn_${v}_v2.0.0.pkl \
        https://zenodo.org/records/10390573/files/aparc_vinn_${v}_v2.0.0.pkl
done
```

## 5. Check the installation

```bash
~/RTPSpy/.venv/bin/python ~/RTPSpy/rtpspy_system_check.py
```

This makes the masks of the test data (FastSurfer on the CPU takes about 15 s per view) and runs the whole real-time pipeline, writing the volumes one by one at the TR. It takes a few minutes. If it ends with `RTPSpy system check finished.` and no `[ERROR]` lines, RTPSpy is ready.

## Troubleshooting

| Message | Cause and fix |
|---|---|
| `dlopen(.../librtp.dylib ...): ... (no such file)` | librtp is not compiled: see [step 3](#3-compile-the-librtp-library). |
| `_stdio.h: No such file or directory` when compiling librtp | Homebrew's gcc was used: give `LOCAL_CC_PATH=/usr/bin/clang` ([step 3](#3-compile-the-librtp-library)). |
| `invalid peer certificate: UnknownIssuer` (installer, `uv`) | Set `UV_SYSTEM_CERTS=1` (see the note in [step 2](#2-install-rtpspy)). |
| `WARNING: No container runtime: AFNI cannot be run from a container.` | AFNI on the Mac was not found: check that `command -v 3dvolreg` finds it (add the AFNI folder, e.g. `~/abin`, to your PATH in `~/.zshrc`), then run the installer again. |
| `No such file or directory: .../rtpspy/FastSurfer` | FastSurfer is not installed: see [step 4](#4-fastsurfer-without-docker). |
| `ERROR: Flag '--no_cc' unrecognized.` | An old FastSurfer clone (before v2.6.0) is in `rtpspy/FastSurfer`. Move it away and get v2.6.0 ([step 4.1](#41-get-fastsurfer)). |
| `not reachable (SSLError)` from FastSurfer | The weights could not be downloaded: see [step 4.3](#43-download-the-network-weights). |
| `... already holds 001.nii.gz, which is not ...` | A FastSurfer folder from an earlier failed run is left in the working folder (e.g., `tests/work/RTP/sub-unknown_acq-anatmprage_T1w/` for the system check). Delete it and run again. |
| `The package antspyx requires numpy<2.4.0` (`uv pip check`) | FastSurfer's `requirements.txt` was installed: repair the environment with [step 4.2](#42-install-its-python-packages). |

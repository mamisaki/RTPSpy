# Compile the librtp Library

RTPSpy uses some C functions of AFNI, compiled into a small library: `librtp.so` on Linux, `librtp.dylib` on macOS.

**On Linux, you usually do not need to do anything.** The Linux library is included in the package (`rtpspy/librtp.so`) and works on systems with GLIBC 2.27 or later (Ubuntu 18.04 or later). If it does not load, `install.sh` rebuilds it automatically in a container.

You need this page only if:

- you use **macOS** (the library must be compiled on your computer), or
- the library fails to load on Linux and the automatic rebuild did not run (e.g., no container runtime).

To check whether the library works, run the [system check](./README.md#check-the-installation).

## Linux: build in a container (recommended)

This needs Apptainer or Docker, but no AFNI source code or compilers on your computer.

```bash
~/RTPSpy/afni_module/build_librtp_container.sh
```

The library is built in the `afni/afni_make_build` image (Ubuntu 18.04) and saved as `~/RTPSpy/containers/librtp.so`. RTPSpy then loads it instead of the bundled `rtpspy/librtp.so`. (`./install.sh --build-librtp yes` does the same.)

To use a library at another location, set the `RTPSPY_LIBRTP` environment variable to its path.

## Build on your computer (macOS, or Linux without containers)

The build compiles part of the AFNI source code, so it needs the AFNI source and its build packages.

### 1. Install the build packages

Linux (Ubuntu):

```bash
sudo apt install build-essential zlib1g-dev libxt-dev libxext-dev libexpat1-dev \
    libmotif-dev libf2c2-dev libxi-dev libxpm-dev libxmu-dev libgsl-dev \
    libglu1-mesa-dev freeglut3-dev mesa-common-dev libnetpbm-dev libglib2.0-dev
```

macOS (with [Homebrew](https://brew.sh/)):

```bash
brew install libpng jpeg expat freetype fontconfig openmotif libomp \
    libxt gsl glib pkg-config gcc@13 autoconf mesa mesa-glu libxpm \
    netpbm libiconv
```

### 2. Get the AFNI source code

```bash
cd ~/RTPSpy/afni_module
git clone https://github.com/afni/afni.git
```

### 3. Compile

Use the Makefile for your system:

| System | Command |
|---|---|
| Linux (Ubuntu) | `make -f Makefile.librtp.linux_ubuntu` |
| macOS 13 or later, Apple silicon (M1 and later) | `make -f Makefile.librtp.macos_13_ARM` |
| Older macOS (10.15) | `make -f Makefile.librtp.macos_10.15` |

Run the command in `~/RTPSpy/afni_module`. The Apple silicon Makefile uses Homebrew's `gcc-13` (`/opt/homebrew/bin/gcc-13`; set `LOCAL_CC_PATH` to use another one). The library is installed into `~/RTPSpy/rtpspy/` (`librtp.so` or `librtp.dylib`).

> [!NOTE]
> On Linux, this replaces the bundled `rtpspy/librtp.so`, which is part of the git repository. To keep the bundled file, build in a container instead, or copy the new library elsewhere and set `RTPSPY_LIBRTP` to it.

### 4. Check

Run the [system check](./README.md#check-the-installation).

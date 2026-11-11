# Compile librtp library
RTPSpy uses some C library functions from AFNI, compiled into librtp.so (or librtp.dylib on macOS). The Linux library (librtp.so) is included in the package; it was built with `afni_module/build_librtp_container.sh` below and needs GLIBC 2.27 or later.

## Linux: build in a container (recommended)
No AFNI source or build packages are needed on the host; Apptainer or Docker is needed. `install.sh` does this automatically when the bundled library does not load.
```
~/RTPSpy/afni_module/build_librtp_container.sh
```
The library is built in the `afni/afni_make_build` image (Ubuntu 18.04), so it needs only GLIBC 2.27 or later. It is saved as `~/RTPSpy/containers/librtp.so`, which RTPSpy loads in place of the bundled `rtpspy/librtp.so` (the `RTPSPY_LIBRTP` environment variable can name another file).

## Build on the host
On macOS, or to build on the host, compile it from the source in the ~/RTPSpy/afni_module directory.

1. Clone AFNI source files
   ```
   cd ~/RTPSpy/afni_module
   git clone https://github.com/afni/afni.git

## Linux
```
sudo apt install libxi-dev libxext-dev libxpm-dev libxmu-dev libgsl-dev libglu1-mesa-dev freeglut3-dev mesa-common-dev libnetpbm-dev libglib2.0-dev
```

## Mac OS
```
brew install \
    libpng jpeg expat freetype fontconfig openmotif libomp \
    libxt gsl glib pkg-config gcc autoconf mesa mesa-glu libxpm \
	 netpbm libiconv
```

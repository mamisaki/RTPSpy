# Compile librtp library
RTPSpy uses some C library functions from AFNI, compiled into librtp.so (or librtp.dylib on OSX). The file included in the package usually works fine, but if you encounter an error related to this library, you may need to compile it from the source in the ~/RTPSpy/afni_module directory.

## Linux
1. Clone AFNI source files
   ```
   cd ~/RTPSpy/afni_module
   git clone https://github.com/afni/afni.git

## Mac OS
```
brew install \
    libpng jpeg expat freetype fontconfig openmotif libomp \
    libxt gsl glib pkg-config gcc autoconf mesa mesa-glu libxpm \
	 netpbm libiconv
```

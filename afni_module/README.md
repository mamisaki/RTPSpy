# Compile librtp.so module
RTPSpy uses some AFNI functions that are compiled in the librtp.so shared object. A prepared file, RTPSpy/rtpspy/librtp.so, usually works fine and you do not need to recompile it yourself.
However, if it does not work due to a library compatibility error or other reasons, you can compile it using the procedure below.

Install the packages needed to compile afni from source;
```
sudo apt install 
```

Clone the AFNI sources to afni_module and run make;
```
cd ~/RTPSpy/afni_module
git clone https://github.com/afni/afni.git
make -f Makefile.librtp.linux_ubuntu
```

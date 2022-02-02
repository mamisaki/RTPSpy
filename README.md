# RTPSpy: fMRI Real-Time Processing System in python

RTPSpy is a python library for real-time fMRI (rtfMRI) data processing systems.   
The package inculudes;
* A fast and comprehensive online fMRI processing pipeline comparable to offline processing.
* Utilities for fast and accurate anatomical image processing to identify a target region onsite.
* A simulation system of online fMRI processing to optimize a pipeline and target signal calculation.
* Interface to an external application for feedback presentation.
* A boilerplate graphical user interface (GUI) class integrating operations with RTPSpy.

Please see the article.   
https://biorxiv.org/cgi/content/short/2021.12.13.472468v1

Also, see the article below for the evaluation of the system.   
[Masaya Misaki and Jerzy Bodurka (2021) The impact of real-time fMRI denoising on online evaluation of brain activity and functional connectivity. J. Neural Eng. 18 046092](https://iopscience.iop.org/article/10.1088/1741-2552/ac0b33)

## Requirements and dependencies
### Supporting systems
RTPSpy has been developed on a Linux system (Ubuntu). It can also be run on Mac OSX and Windows with the Windows Subsystem for Linux (WSL), while GPU computation is not supported on OSX and WSL for now.

### External tools
RTPSpy is assumed to be run on a miniconda (https://docs.conda.io) or Anaconda (https://www.anaconda.com/) environment.
Install either one referring to these sites.   
A yaml file describing required python libraries that can be used to install the libraries in an anaconda environment is provided with the package for easy installation.  

AFNI (https://afni.nimh.nih.gov/) needs to be installed to run RTPSpy library methods. See https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/background_install/main_toc.html for the installation process for each system.

### For Linux, WSL ubuntu
Install git command.  
```
sudo apt install git
```

### For OSX  
Install the Homebrew following the instructions in https://brew.sh/  
Then, install tools and libraries by entering the command below.
```
brew install git gcc fftw openmotif
```

### GPU
RTPSpy can take advantage of GPU computation, while it is not mandatory for real-time image processing (computational speed is fast enough for real-time fMRI even without GPU). GPU is utilized in the online fMRI data processing and anatomical image processing with FastSurfer (https://deep-mi.org/research/fastsurfer/).  
To use GPU computation, you need to install a GPU driver compatible with the CUDA toolkit (https://developer.nvidia.com/cuda-toolkit). The toolkit will be installed with the yaml file.  

## Installation
Every command should be entered in the terminal (Terminal application on OSX and Ubuntu console on WSL).  
Instructions below are supposed to install the package in the home directory. If you want to install another place, replace the path '~/RTPSpy' appropriately.  

### Clone the packages  
```
cd ~
git clone https://github.com/mamisaki/RTPSpy.git
git clone https://github.com/Deep-MI/FastSurfer.git ~/RTPSpy/rtpspy/FastSurfer
```

### Install the package in a conda environment
```
cd ~/RTPSpy
conda env create -f RTPSpy.yml
conda activate RTPSpy
pip install -e ~/RTPSpy
```

### Test the system
```
~/RTPSpy/rtpspy_system_check.py
```

## Usage
Please refer to https://biorxiv.org/cgi/content/short/2021.12.13.472468v1 for the usage of the library.  
RTPSpy is not a complete application by itself but is supposed to be used as a part of a user's custom rtfMRI application.  
Example applications using the RTPSpy library are shown below.  

## Example GUI applications
We provide a boilerplate graphical user interface (GUI) application integrating operations with RTPSpy.  
We aslo present a sample external application for neurofeedback. This application uses PsychoPy (Peirce, 2008) for neurofeedback presentation and demonstrates how the RTPSpy communicates with an external application using the library module (RTP_SERV).  

The GUI application is presented as just one example of library usage. However, a user may develop a custom neurofeedback application with minimum modification on the example script.

To run the example application, a user needs to install the PsychoPy package in a conda environment, 'psychopy,' as indicated below.
```
conda create -n psychopy python=3.6
conda activate psychopy
pip install psychopy
```

On WSL, you also need to set up an X server application. The instruction of AFNI installation on WSL describes it, so I assume you have already done that.

### ROI-NF (example/ROI-NF)
A GUI application is built on the RTPSpy library. This application is provided as a boilerplate of the RTPSpy application to make a custom neurofeedback application with minimum scripting.  
See [example/ROI-NF](example/ROI-NF#readme)

### LA-NF (example/LA-NF)
The application implements the left amygdala neurofeedback session with happy autobiographical memory recall (Zotev et al., 2011;Young et al., 2017). This script is for a demonstration of a full-fledged GUI application using RTPSpy. Note that this application was not used in the previous studies and several parameters are different from the previous reports.  
See [example/LA-NF](example/LA-NF#readme)

Young, K.D., Siegle, G.J., Zotev, V., Phillips, R., Misaki, M., Yuan, H., Drevets, W.C., and Bodurka, J. (2017). Randomized Clinical Trial of Real-Time fMRI Amygdala Neurofeedback for Major Depressive Disorder: Effects on Symptoms and Autobiographical Memory Recall. Am J Psychiatry 174, 748-755.  
Zotev, V., Krueger, F., Phillips, R., Alvarez, R.P., Simmons, W.K., Bellgowan, P., Drevets, W.C., and Bodurka, J. (2011). Self-Regulation of Amygdala Activation Using Real-Time fMRI Neurofeedback. PLoS ONE 6, e24522.


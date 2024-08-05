# RTPSpy: fMRI Real-Time Processing System in Python

> [!NOTE]
> We are currently developing a new version of the library that will run in a container (i.e., Docker, Singularity, and Apptainer).  
> Some packages are under development and may not work outside of our environment.  
> If you need help setting them up in your environment, please contact me at (mamisaki@gmail.com). I may be able to assist you in implementing the system for your environment.

RTPSpy is a Python library for real-time fMRI (rtfMRI) data processing systems.
The package includes:
* A fast and comprehensive online fMRI processing pipeline comparable to offline processing, including RETROICOR [physiological noise correction](./PhysioRecording.md).  
* Utilities for fast and accurate anatomical image processing to identify a target region on-site.
* An interface to an external application for feedback presentation.

Please see the following article:
https://www.frontiersin.org/articles/10.3389/fnins.2022.834827/full

Also, refer to the article below for the evaluation of the system:  
[Masaya Misaki and Jerzy Bodurka (2021) The impact of real-time fMRI denoising on online evaluation of brain activity and functional connectivity. J. Neural Eng. 18 046092](https://iopscience.iop.org/article/10.1088/1741-2552/ac0b33)

## Requirements and Dependencies
### Supporting Systems
RTPSpy has been developed on a Ubuntu Linux system. It can also be run on macOS and Windows with the Windows Subsystem for Linux (WSL), although GPU computation is not currently supported on macOS and WSL.

### External Tools
RTPSpy is designed to run in a conda environment. Install Miniconda by following the instructions at  
https://docs.conda.io/en/latest/miniconda.html

Be sure to complete the setup so that you can call the conda command in a terminal.
The conda initialization script should be added to your ~/.bashrc file. You may need to restart your terminal application to activate the conda environment.
If you did not answer "yes" to the "Do you wish the installer to initialize ..." question at the end of the installation, you will need to run the following commands to activate the conda command:  
(If you are using Anaconda, replace miniconda3 in the command below with anaconda3. Replace YOUR_HOME_DIRECTORY with the full path of your home directory.)  
```
eval “$(YOUR_HOME_DIRECTORY/miniconda3/bin/conda shell.bash hook)” 
conda init
```

A YAML file describing the required Python libraries is provided with the package for easy installation in a conda environment.  

AFNI (https://afni.nimh.nih.gov/) must be installed to run RTPSpy library methods. See https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/background_install/main_toc.html for the installation process for each system.


### For Linux and WSL Ubuntu
Install git command.  
```
sudo apt install git
```

### For macOS  
Install Homebrew following the instructions at https://brew.sh/  
Then, install the required tools and libraries by entering the command below:
```
brew install git gcc fftw openmotif
```

### GPU
RTPSpy can take advantage of GPU computation; however, it is not mandatory for real-time image processing, as computational speed is sufficient for real-time fMRI even without a GPU. GPU acceleration is utilized in online fMRI data processing and anatomical image processing with FastSurfer (https://deep-mi.org/research/fastsurfer/).
To use GPU computation, you need to install a GPU driver compatible with the CUDA toolkit. The toolkit will be installed using the RTPSpy YAML file provided.  

## Installation
Every command should be entered in the terminal (Terminal application on macOS and Ubuntu console on WSL).
The instructions below are intended to install the package in the home directory. If you want to install it in another location, replace the path ~/RTPSpy appropriately.  

### Clone the package  
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

### (Linux, WSL) Add user to the dialout group to access the serial port
> [!TIP]
> Serial (USB) port access is required to record physiological signals using a USB GPIO device.
```
sudo usermod -a -G dialout $USER
```

You may need to reboot the system.  

### Test the system
```
conda activate RTPSpy
~/RTPSpy/rtpspy_system_check.py
```

## Usage
Please refer to [this article](https://www.frontiersin.org/articles/10.3389/fnins.2022.834827/full) for usage information about the library.
RTPSpy is not a complete application by itself but is intended to be used as part of a user's custom rtfMRI application.
Example applications using the RTPSpy library are shown below.

### Example GUI Applications
We provide a boilerplate graphical user interface (GUI) application that integrates operations with RTPSpy.  
We also present a sample external application for neurofeedback. This application uses PsychoPy (Peirce, 2008) for neurofeedback presentation and demonstrates how RTPSpy communicates with an external application using the library module (RTP_SERV).

The GUI application serves as just an example of library usage. However, users may develop a custom neurofeedback application with minimal modifications to the example script.

To run the example application, users need to install the PsychoPy package in a conda environment named psychopy, as indicated below:
```
conda create -n psychopy python=3.8
conda activate psychopy
pip install psychopy
```
On WSL, you also need to set up an X server application. The instructions for AFNI installation on WSL describe this setup, so I assume you have already completed it.

#### ROI-NF (example/ROI-NF)
This application extracts the signal from a region of interest (ROI) at each TR and sends it to an external application in real time. No feedback presentation script is provided, except for a script that receives the signal in real time from the system via a network socket. This serves as a boilerplate for the RTPSpy application to create a custom neurofeedback application with minimal scripting.
See [example/ROI-NF](example/ROI-NF#readme)

#### LA-NF (example/LA-NF)
This application implements a left amygdala neurofeedback session with happy autobiographical memory recall (Zotev et al., 2011; Young et al., 2017). This script serves as a demonstration of a full-fledged GUI application using RTPSpy. Note that this application was not used in the previous studies, and several parameters differ from those in earlier reports.  
See [example/LA-NF](example/LA-NF#readme)

Young, K.D., Siegle, G.J., Zotev, V., Phillips, R., Misaki, M., Yuan, H., Drevets, W.C., and Bodurka, J. (2017). Randomized Clinical Trial of Real-Time fMRI Amygdala Neurofeedback for Major Depressive Disorder: Effects on Symptoms and Autobiographical Memory Recall. Am J Psychiatry 174, 748-755.  
Zotev, V., Krueger, F., Phillips, R., Alvarez, R.P., Simmons, W.K., Bellgowan, P., Drevets, W.C., and Bodurka, J. (2011). Self-Regulation of Amygdala Activation Using Real-Time fMRI Neurofeedback. PLoS ONE 6, e24522.


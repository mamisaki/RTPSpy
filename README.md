# RTPSpy: fMRI Real-Time Processing System in Python

> [!NOTE]
> We are currently developing a new version of the library that will run in a container (i.e., Docker, Singularity (Apptainer)).  
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
`install.sh` sets up everything: the Python environment (with [uv](https://docs.astral.sh/uv/); conda is not needed), the AFNI command-line programs, FastSurfer, and dcm2niix.
AFNI and FastSurfer are run from container images, which needs [Apptainer](https://apptainer.org/docs/admin/main/installation.html) (recommended on Linux; `sudo apt install apptainer` or the .deb package from the Apptainer releases) or Docker (e.g., Docker Desktop on macOS).
Without a container runtime, AFNI installed on the host is used, and masks are made without FastSurfer.

The AFNI GUI is not run from the container.

### GPU
RTPSpy can take advantage of GPU computation; however, it is not mandatory for real-time image processing, as computational speed is sufficient for real-time fMRI even without a GPU. GPU acceleration is utilized in online fMRI data processing and anatomical image processing with FastSurfer (https://deep-mi.org/research/fastsurfer/).
To use GPU computation, install an NVIDIA GPU driver before running `install.sh`. The installer selects the PyTorch build and the FastSurfer image for the driver (`--torch-backend` overrides the PyTorch build, e.g., `cpu`, `cu126`).

## Installation
Every command should be entered in the terminal (Terminal application on macOS and Ubuntu console on WSL).

### Clone the package and run the installer
```
git clone https://github.com/mamisaki/RTPSpy.git ~/RTPSpy
cd ~/RTPSpy
./install.sh
```
The installer
1. installs uv if it is not found,
2. makes a Python 3.13 environment in `~/RTPSpy/.venv` (uv downloads Python) and installs RTPSpy, PyTorch, dcm2niix, and the other Python packages,
3. pulls the AFNI (`afni/afni_docker_universal`) and FastSurfer (`deepmi/fastsurfer`, v2.6.0) images; with Apptainer they are saved as `.sif` files in `~/RTPSpy/containers` (about 2 GB and 2-6 GB),
4. writes `containers/containers.json`, and wrapper scripts for the AFNI programs in `containers/afni_bin`,
5. builds librtp.so in a container if the bundled one does not load on this system (see below).

Re-running `./install.sh` updates the environment and does not download existing images again. Options (`./install.sh --help`):

| Option | |
|---|---|
| `--afni auto\|container\|host` | `auto` (default) uses AFNI on the host if found, otherwise the container. `container` always uses the container. |
| `--runtime apptainer\|docker\|...\|none` | Container runtime (default: detected). |
| `--no-fastsurfer` | Do not pull the FastSurfer image. |
| `--torch-backend B` | PyTorch build (`auto`, `cpu`, `cu126`, `cu130`, ...). |
| `--bind DIRS` | Extra host directories (colon-separated) visible in the containers. |
| `--container-dir D` | Where to save the `.sif` images. |
| `--build-librtp auto\|yes\|no` | Build librtp.so in a container: if the bundled one does not load (`auto`, default), always, or never. |

The containers see `$HOME`, `/tmp`, the current directory, `/mnt`, `/media`, `/data`, `/scratch`, and the top-level directory of each absolute path given to the program. Add others with `--bind` or the `RTPSPY_BIND` environment variable.

Activate the environment with
```
source ~/RTPSpy/.venv/bin/activate
```
The AFNI wrappers are put on PATH by the activation, and by `import rtpspy` in any case.

### (Linux, WSL) Qt library for the GUI
The GUI uses PyQt6 (Qt 6), which needs `libxcb-cursor0` on X11 (`install.sh` tells you if it is missing):
```
sudo apt install libxcb-cursor0
```

### (Linux, WSL) Add user to the dialout group to access the serial port
> [!TIP]
> Serial (USB) port access is required to record physiological signals using a USB GPIO device.  
> For details, see [Cardiogram, Respiration, and Scan Timing TTL Signal Recording](./PhysioRecording.md).   
```
sudo usermod -a -G dialout $USER
```

You may need to reboot the system.  

### Test the system
```
~/RTPSpy/.venv/bin/python ~/RTPSpy/rtpspy_system_check.py
```
> [!TIP]
> The bundled librtp.so is built in the AFNI build container (Ubuntu 18.04) and needs GLIBC 2.27 or later. If it does not load, `install.sh` rebuilds it in the container (`afni_module/build_librtp_container.sh`) and saves it as `containers/librtp.so`, which is then used in place of the bundled one (`./install.sh --build-librtp yes` always rebuilds it).  
> On macOS, compile the librtp library from the source; refer to [Compile librtp library](./Compilelibrtp.md).

### Using a local FastSurfer clone
Instead of the container, a FastSurfer clone in `rtpspy/FastSurfer` can be used (`RTPSPY_FASTSURFER=local`, or when the container is not set up):
```
git clone --branch v2.6.0 https://github.com/Deep-MI/FastSurfer.git ~/RTPSpy/rtpspy/FastSurfer
uv pip install --python ~/RTPSpy/.venv/bin/python -r ~/RTPSpy/rtpspy/FastSurfer/requirements.txt
```

### Running the tests
The unit tests in `tests/` use pytest and only the test data in `tests/` or synthetic data.
```
cd ~/RTPSpy
source .venv/bin/activate                                        # pytest is installed by install.sh
QT_QPA_PLATFORM=offscreen python -m pytest tests                 # all tests
python -m pytest tests -m "not gpu and not afni and not slow"    # quick CPU-only tests (as in CI)
```
Tests are marked `gpu` (needs CUDA), `afni` (compares with AFNI programs on PATH), and `slow` (long runs). The `gpu` and `afni` tests are skipped when CUDA or AFNI is not available.

## Usage
Please refer to [this article](https://www.frontiersin.org/articles/10.3389/fnins.2022.834827/full) for usage information about the library.
RTPSpy is not a complete application by itself but is intended to be used as part of a user's custom rtfMRI application.
Example applications using the RTPSpy library are shown below.

### Processing latency
RTPSpy records three times for each volume: the acquisition onset from the scanner TTL pulse (recorded by rt_physio), the time the image file was read (RtpWatch), and the time the feedback signal was sent to the external application (or saved to the signal file). From these, two latencies are derived:
- **TTL → image**: image reconstruction and transfer by the scanner (`acq_to_read`)
- **image → feedback**: RTPSpy processing (`read_to_fb`)

Check 'Latency' in the 'Use / Show' box to plot them in real time, with the total (**TTL → feedback**, `acq_to_fb`) in the bottom panel. The y range of each panel is fitted to the values. Volumes processed retrospectively (the wait period of RtpRegress) are shown as gray points in the image → feedback and total panels and excluded from their y range and median; the first real-time volume (its processing includes the initial fit) is also excluded from their y range. If no TTL pulse matches a volume (e.g., a manual start without the TTL), its onset is estimated from the scan onset and TR, and it is shown with a hollow marker.
The values of each volume are saved as `latency_{scan}.csv` in the `log` folder of the working directory at the end of a run (with `vol`, `onset`, `read`, `sent`, `acq_to_read`, `read_to_fb`, `acq_to_fb`, `onset_estimated`, `retrospective`).
The TTL of a volume is the pulse nearest to its expected onset, so both one pulse per volume and one pulse per slice work. The feedback time is recorded when an application sends an `NF {time},{volume_index},...;` message with `RtpApp.send_extApp`, so applications that override `do_proc` (e.g., example/ROI-NF) need no change.

### Compare real-time and offline processing
`rtpspy/rtp_offline_compare.py` quantifies how close the real-time (online, causal) processing is to standard offline processing of the same run, as evaluated in Misaki and Bodurka (2022), using your own data and parameters.
It runs the RTPSpy modules volume by volume (VOLREG → TSHIFT → SMOOTH → REGRESS, as in `RtpApp`; the RETROICOR regressors of each volume use only the physio samples recorded up to that volume), and processes the whole run offline with matched settings: `3dvolreg` (same reference volume and interpolation), `3dTshift` (same method, reference time, slice timing, and ignored initial volumes), `3dBlurInMask -preserve` (same FWHM and mask; the same algorithm as RtpSmooth), and a whole-run least-squares GLM (numpy) with the same regressors (Legendre polynomials, motion, GS/WM/Vent mean signals, and RICOR8 from the whole physio signals).
The AFNI programs must be on PATH.
```
python -m rtpspy.rtp_offline_compare --input func.nii.gz --masks_dir work/RTP \
    --roi work/RTP/ROI_alFunc.nii.gz --card ECG.1D --resp Resp.1D --physio_freq 40 \
    [--param rtp_params.json] --out compare_out
```
`--masks_dir` is the directory of the masks made by `RtpApp.make_masks` (`RTP_mask`, `GSR_mask`, `*_WM_alFunc`, `*_Vent_alFunc`); each mask can also be given with `--rtp_mask`, `--gs_mask`, `--wm_mask`, `--vent_mask`. `--param` is a JSON file of module parameters (`{"VOLREG": {"regmode": "cubic"}, "REGRESS": {...}, ...}`, the format of `rtp_sim.py`); the default is the setting of `rtpspy_system_check.py`.
The output directory has the online module outputs and QC (`online/RTP/`), the offline outputs and the AFNI commands (`offline/`), `metrics.csv`, `roi_series.csv`, and `report.html` (with `report.png`). The metrics are computed over the volumes output by the online pipeline, within the RTP mask: the correlation and maximum difference of each motion parameter, the voxelwise correlation and RMSE of each stage (median and 5th percentile), the ROI mean signal correlation, the noise variance removed by the regression (% of the input variance), and `REGRESS_causal`, the difference between the cumulative (online) GLM and the whole-run GLM applied to the same online input. The `*_realtime_vols` metrics exclude the volumes that REGRESS processes retrospectively at its first fit.

### Example GUI Applications
We provide a boilerplate graphical user interface (GUI) application that integrates operations with RTPSpy.  
We also present a sample external application for neurofeedback. This application uses PsychoPy (Peirce, 2008) for neurofeedback presentation and demonstrates how RTPSpy communicates with an external application using the library module (RTP_SERV).

The GUI application serves as just an example of library usage. However, users may develop a custom neurofeedback application with minimal modifications to the example script.

The feedback presentation scripts (`NF_psypy.py`) run with PsychoPy in a separate environment, `~/RTPSpy/.venv-psychopy` (Python 3.12, because PsychoPy supports Python < 3.13). `install.sh` makes it (skip it with `--no-psychopy`). The scripts do not import rtpspy; each example directory has a copy of `rtpspy/rtp_serve.py` for the communication with RTPSpy.

On WSL, you also need to set up an X server application. The instructions for AFNI installation on WSL describe this setup, so I assume you have already completed it.

#### ROI-NF (example/ROI-NF)
This application extracts the signal from a region of interest (ROI) at each TR and sends it to an external application in real time. No feedback presentation script is provided, except for a script that receives the signal in real time from the system via a network socket. This serves as a boilerplate for the RTPSpy application to create a custom neurofeedback application with minimal scripting.
See [example/ROI-NF](example/ROI-NF#readme)

#### LA-NF (example/LA-NF)
This application implements a left amygdala neurofeedback session with happy autobiographical memory recall (Zotev et al., 2011; Young et al., 2017). This script serves as a demonstration of a full-fledged GUI application using RTPSpy. Note that this application was not used in the previous studies, and several parameters differ from those in earlier reports.  
See [example/LA-NF](example/LA-NF#readme)

Young, K.D., Siegle, G.J., Zotev, V., Phillips, R., Misaki, M., Yuan, H., Drevets, W.C., and Bodurka, J. (2017). Randomized Clinical Trial of Real-Time fMRI Amygdala Neurofeedback for Major Depressive Disorder: Effects on Symptoms and Autobiographical Memory Recall. Am J Psychiatry 174, 748-755.  
Zotev, V., Krueger, F., Phillips, R., Alvarez, R.P., Simmons, W.K., Bellgowan, P., Drevets, W.C., and Bodurka, J. (2011). Self-Regulation of Amygdala Activation Using Real-Time fMRI Neurofeedback. PLoS ONE 6, e24522.


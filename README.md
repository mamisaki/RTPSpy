# RTPSpy: fMRI Real-Time Processing System in Python

RTPSpy is a Python library for building real-time fMRI (rtfMRI) applications, such as neurofeedback.
It provides:

- **Online fMRI processing** comparable to offline processing: motion correction, slice-timing correction, smoothing, and noise regression, including RETROICOR [physiological noise correction](./PhysioRecording.md).
- **Fast anatomical image processing** to define a target brain region on site (brain extraction, segmentation, and template-to-subject ROI transformation).
- **A link to an external application** (e.g., a PsychoPy script) that presents the feedback.

RTPSpy is a library, not a complete application. You build your own application on it; the [example applications](#example-applications) are a good starting point.

For a description of the system, see [Misaki and Bodurka (2022), Frontiers in Neuroscience](https://www.frontiersin.org/articles/10.3389/fnins.2022.834827/full).

For an evaluation of the online denoising, see [Misaki M. and Bodurka J. (2021) The impact of real-time fMRI denoising on online evaluation of brain activity and functional connectivity. J. Neural Eng. 18 046092](https://iopscience.iop.org/article/10.1088/1741-2552/ac0b33).

If you need help setting up RTPSpy in your environment, contact Masaya Misaki (<mamisaki@gmail.com>).

## Contents

- [What you need](#what-you-need)
- [Installation](#installation)
- [Check the installation](#check-the-installation)
- [Connecting to the scanner](#connecting-to-the-scanner)
- [How a real-time session works](#how-a-real-time-session-works)
- [Starting the scan: TTL trigger or the 'T' key](#starting-the-scan-ttl-trigger-or-the-t-key)
- [Testing without a scanner: the RT-MRI simulator](#testing-without-a-scanner-the-rt-mri-simulator)
- [Example applications](#example-applications)
- [Output files](#output-files)
- [Processing latency](#processing-latency)
- [Comparing real-time and offline processing](#comparing-real-time-and-offline-processing)
- [For developers](#for-developers)

## What you need

| | Required? | Notes |
|---|---|---|
| Linux (Ubuntu) | Yes, or one of the next two | RTPSpy is developed and used on Ubuntu. |
| macOS | | Works, without GPU computation. The librtp library must be [compiled](./Compilelibrtp.md). |
| Windows (WSL) | | Works in the Windows Subsystem for Linux, without GPU computation. An X server is needed for the GUI. |
| Apptainer or Docker | Recommended | Runs the AFNI programs and FastSurfer from container images. On Linux, Apptainer is recommended (`sudo apt install apptainer`, or the .deb package from the [Apptainer releases](https://apptainer.org/docs/admin/main/installation.html)); on macOS, Docker Desktop. Without a container runtime, AFNI installed on the computer is used, and masks are made without FastSurfer. |
| NVIDIA GPU | No | Speeds up the online processing and the anatomical segmentation (FastSurfer). Real-time processing is fast enough without a GPU. Install the NVIDIA driver before running the installer. |
| Physio and TTL devices | No | Needed only for RETROICOR and for starting the processing with the scanner trigger. Without them, start the scan with the 'T' key or the 'Manual start' button. See [Physiological signal and TTL recording](./PhysioRecording.md). |

You do not need conda or a separate AFNI installation: the installer sets up everything.

## Installation

Enter the commands in a terminal (on Windows, the Ubuntu console of WSL).

```bash
git clone https://github.com/mamisaki/RTPSpy.git ~/RTPSpy
cd ~/RTPSpy
./install.sh
```

The installer:

1. installs [uv](https://docs.astral.sh/uv/) (the Python package manager) if it is not found;
2. makes a Python 3.13 environment in `~/RTPSpy/.venv` and installs RTPSpy, PyTorch (built for your GPU driver, or for the CPU), dcm2niix, and the other Python packages;
3. downloads the AFNI and FastSurfer (v2.6.0) container images. With Apptainer, they are saved as `.sif` files in `~/RTPSpy/containers` (about 2 GB and 2-6 GB);
4. makes `~/RTPSpy/.venv-psychopy`, a separate Python 3.12 environment for the PsychoPy feedback scripts of the examples;
5. rebuilds the librtp library in a container if the bundled one does not work on your system.

Running `./install.sh` again updates the packages to their newest versions. Images that already exist are not downloaded again.

### Installer options

Run `./install.sh --help` for the full list.

| Option | Meaning |
|---|---|
| `--afni auto\|container\|host` | `auto` (default): use AFNI installed on this computer if found, otherwise the container. `container`: always the container. `host`: never download the AFNI container. |
| `--runtime auto\|apptainer\|singularity\|docker\|podman\|none` | Container runtime. The default is detected. |
| `--no-fastsurfer` | Do not download the FastSurfer image. |
| `--no-psychopy` | Do not make the PsychoPy environment. |
| `--torch-backend B` | PyTorch build: `auto` (default), `cpu`, `cu126`, `cu128`, `cu130`, ... |
| `--python V` | Python version (default 3.13). |
| `--container-dir D` | Where to save the `.sif` images. |
| `--bind DIRS` | Extra folders (colon-separated) to make visible in the containers. |
| `--build-librtp auto\|yes\|no` | Build librtp in a container: if the bundled one does not load (`auto`), always, or never. |

The containers can see your home folder, `/tmp`, the current folder, `/mnt`, `/media`, `/data`, `/scratch`, and the top-level folder of any absolute path given to a program. To add other folders, use `--bind` or the `RTPSPY_BIND` environment variable.

### After installing (Linux and WSL)

The GUI needs one system library on X11 (the installer tells you if it is missing):

```bash
sudo apt install libxcb-cursor0
```

To record physio and TTL signals with a USB device, your user must be allowed to use serial ports. Log out and in again (or reboot) after this command:

```bash
sudo usermod -a -G dialout $USER
```

### Using the environment

The example applications and tools have start scripts that use the environment directly. To use RTPSpy in your own scripts, activate the environment first:

```bash
source ~/RTPSpy/.venv/bin/activate
```

The AFNI programs (through wrapper scripts in `containers/afni_bin`) are put on the PATH by the activation, and also by `import rtpspy`.

## Check the installation

```bash
~/RTPSpy/.venv/bin/python ~/RTPSpy/rtpspy_system_check.py
```

This runs the whole real-time pipeline on the test data in `tests/`, writing the volumes one by one at the TR as a scanner would. If it ends without errors, RTPSpy is ready.

If the librtp library fails to load, see [Compile the librtp library](./Compilelibrtp.md).

## Connecting to the scanner

RTPSpy receives the images through a folder, the *watch directory*: the scanner writes each reconstructed volume there as a file, and RTPSpy processes each new file as it arrives.

- **Siemens scanners**: the BOLD AddIn writes each volume as a DICOM file into a folder that the RTPSpy computer shares over the network. See [Real-Time Image Export from a Siemens Scanner](./SiemensSetup.md). RTPSpy reads the DICOM files of the XA software (enhanced multi-frame) and of the older software (mosaic).
- **Other scanners**: DICOM files are read only for Siemens. Export the images as NIfTI or AFNI BRIK files into the watch directory, and set the 'File type' of the WATCH module accordingly.

## How a real-time session works

A typical neurofeedback session with an RTPSpy application goes as follows (the [ROI-NF example](example/ROI-NF#readme) walks through each step):

1. **Set the folders.** The *watch directory* is where the scanner exports the images in real time (see [Connecting to the scanner](#connecting-to-the-scanner)). The *working directory* is where RTPSpy saves its results.
2. **Scan the reference images.** Scan an anatomical image and a functional reference image, and convert them to NIfTI ('DICOM to NIfTI (dcm2niix)' button).
3. **Make the masks.** RTPSpy extracts the brain, segments white matter and ventricles, and brings the template ROI into the subject's functional space ('Create masks' button). Check the result in the built-in image viewer.
4. **Set up the processing.** Choose the processing steps and their parameters, then press 'RTP setup'.
5. **Get ready.** Press 'Ready'. RTPSpy now waits for the scan to start.
6. **Start the scan.** The processing starts with the scanner trigger, the 'T' key, or the 'Manual start' button (see the next section). Each new volume is processed as it arrives, and the feedback value is sent to the presentation application.
7. **End of the scan.** When no new image arrives, RTPSpy ends the run and saves the results.

## Starting the scan: TTL trigger or the 'T' key

RTPSpy needs to know when the scan starts. There are three ways:

- **Scanner TTL pulse.** If a TTL recording device is connected (see [Physiological signal and TTL recording](./PhysioRecording.md)), the processing starts with the first TTL pulse after 'Ready'.
- **The 'T' key.** Press `T` while the RTPSpy window is active. This also works with a trigger box that sends the 't' key, as long as the RTPSpy window has the keyboard focus.
- **The 'Manual start' button.**

The physio and TTL devices are optional. RTPSpy starts a helper program, rt_physio, which records the signals if a device is found. **If no device is found**:

- the physio signal window is not shown, and the 'Physio' check box is disabled;
- the 'Physio (RICOR)' option of REGRESS is disabled and set to 'None', since the RETROICOR regressors cannot be made without the signals. (It is set back when a device becomes available.)

Start the scan with the 'T' key or the 'Manual start' button in that case.

## Testing without a scanner: the RT-MRI simulator

The RT-MRI simulator replays a recorded scan session, so that you can test an application on any computer.

```bash
~/RTPSpy/boot_rtmri_simulator.sh
```

- **DICOM images.** Select a session folder of DICOM files and a series. The files are copied into the output folder (set it to RTPSpy's watch directory) at the times they were made on the scanner, plus a transfer delay (0.2 s by default). Start, pause, stop, or copy one file at a time; 'Auto-advance' continues with the next series.
- **TTL pulses.** A TTL pulse is sent to rt_physio at the acquisition time of each volume. rt_physio records these pulses only while it plays the simulated physio signals, so press 'Start Physio' first (see below).
- **The 'T' key.** With 'Press 'T' key', the simulator presses the `t` key at the scan start or at every volume, in the window whose title matches 'Key window' (default: the RTPSpy window). This tests a setup without a TTL device. It needs `xdotool` (`sudo apt install xdotool`).
- **Physio signals.** Feed rt_physio with cardiac and respiration signals from two text files (one value per line) or from a BIDS `*_physio.tsv[.gz]` file with its JSON sidecar. The signals are played in a loop, and the physio window appears.

Put your recorded sessions in `~/RTPSpy/simulation_data` (or choose another folder with 'Set'). The simulator reads the DICOM headers of a session once and caches them in the session folder (`.rtmri_series_cache.csv`).

## Example applications

The examples are GUI applications built on RTPSpy. They show how to write a custom neurofeedback application with little code. The feedback is presented by a separate PsychoPy script (`NF_psypy.py`), which RTPSpy starts and communicates with over a network socket.

- **[ROI-NF](example/ROI-NF#readme)**: extracts the mean signal of a region of interest (ROI) at each TR and sends it to the PsychoPy script, which shows the value. A template for your own application.
- **[LA-NF](example/LA-NF#readme)**: a complete left-amygdala neurofeedback session with happy autobiographical memory recall (Zotev et al., 2011; Young et al., 2017). Several parameters differ from those of the published studies.

The PsychoPy scripts run in `~/RTPSpy/.venv-psychopy` (Python 3.12, because PsychoPy supports Python < 3.13). They do not import rtpspy; each example folder has a copy of `rtpspy/rtp_serve.py` for the communication.

Young K.D., Siegle G.J., Zotev V., Phillips R., Misaki M., Yuan H., Drevets W.C., and Bodurka J. (2017). Randomized Clinical Trial of Real-Time fMRI Amygdala Neurofeedback for Major Depressive Disorder: Effects on Symptoms and Autobiographical Memory Recall. Am J Psychiatry 174, 748-755.
Zotev V., Krueger F., Phillips R., Alvarez R.P., Simmons W.K., Bellgowan P., Drevets W.C., and Bodurka J. (2011). Self-Regulation of Amygdala Activation Using Real-Time fMRI Neurofeedback. PLoS ONE 6, e24522.

## Output files

RTPSpy saves its results in the working directory. The file names follow the BIDS (derivatives) convention as far as the real-time setting allows. Tables are TSV files ('n/a' for missing values) with a JSON file describing the columns.

Each scan is named `<scan>` = `sub-<label>[_ses-<label>]_task-<label>_run-<index>`, taken from the DICOM header of its first volume:

| Entity | From |
|---|---|
| `sub` | Patient name (the ID of two letters and three digits, if it has one) |
| `ses` | Study date and time (`YYYYMMDDhhmmss`) |
| `task` | Series description (letters and digits only) |
| `run` | Series number |

For NIfTI input, the entities are read from the file name (a BIDS name, or a name like `sub-*_ses-*_ser-*_desc-*`). A name without entities gives `sub-unknown` and `task-<file name>`. If files of the same scan already exist (e.g., the same series replayed), the new files get `rep<k>` in `desc` instead of overwriting them.

| Folder | File | Content |
|---|---|---|
| `RTP/` | `<scan>_desc-<module>_bold.nii.gz` (+ `.json`) | Processed images of a module (`volreg`, `tshift`, `smooth`, `regress`) |
| `RTP/` | `<scan>_desc-regressqc_timeseries.tsv` | Online quality control of REGRESS (FD, DVARS, tSNR, censored volumes) |
| `RTP/` | `<scan>_desc-<module>proctime_timeseries.tsv` | Processing time of each volume |
| `log/` | `<scan>_desc-rtpspy_params.json` | Parameters of the modules and the application |
| `log/` | `<scan>_desc-roi_timeseries.tsv` | ROI signals |
| `log/` | `<scan>_desc-latency_timeseries.tsv` | Latency of each volume (see [Processing latency](#processing-latency)) |
| `log/` | `<scan>_physio.tsv.gz` (+ `.json`) | Cardiac and respiration signals and the TTL pulses (with a recording device) |

<details>
<summary>Mask and transform files made by 'Create masks'</summary>

The masks are saved in `RTP/`, named after the anatomy image (`sub-*[_ses-*]_T1w.nii.gz`, `_desc-brain_T1w`, `_label-WM_mask`, `_label-Vent_mask`, `_desc-aseg_dseg`) or the base function image (`<scan>_boldref.nii.gz` and its `.json`, `<scan>_desc-RTP_mask`, `_desc-GSR_mask`, `_label-WM_mask`, `_label-Vent_mask`, `_desc-aseg_dseg`, `_label-<ROI>_mask`).
The anatomy aligned to the function image is `_space-func_desc-brain_T1w` (`func` is the space of the base function image). The transforms are `_from-T1w_to-func_mode-image_xfm.aff12.1D` (AFNI) and `_from-<template>_to-func_mode-image_desc-{affine,warp,invwarp}_xfm` (ANTs). With distortion correction, images in the corrected space have `space-sdc`.
</details>

## Processing latency

RTPSpy records, for each volume, when it was acquired (from the scanner TTL pulse), when its image file was read, and when the feedback value was sent. From these it reports:

- **TTL → image**: image reconstruction and transfer by the scanner;
- **image → feedback**: processing by RTPSpy;
- **TTL → feedback**: the total delay.

Check 'Latency' in the 'Use / Show' box to plot them during the scan. The values are saved in `log/<scan>_desc-latency_timeseries.tsv` at the end of each run.

<details>
<summary>Details</summary>

- Volumes processed retrospectively (while REGRESS waits for enough volumes) are shown as gray points and excluded from the y range and the median. The first real-time volume (whose processing includes the initial fit) is also excluded from the y range.
- If no TTL pulse matches a volume (e.g., a start with the 'T' key or 'Manual start' without a TTL device), its onset is estimated from the scan start and the TR and shown with a hollow marker.
- The TTL of a volume is the pulse nearest to its expected onset, so both one pulse per volume and one pulse per slice work.
- The feedback time is recorded when the application sends an `NF {time},{volume_index},...;` message with `RtpApp.send_extApp`. Applications that override `do_proc` (e.g., ROI-NF) need no change.
- Columns of the file: `vol`, `onset`, `read`, `sent`, `acq_to_read`, `read_to_fb`, `acq_to_fb`, `onset_estimated`, `retrospective`.
</details>

## Comparing real-time and offline processing

`rtpspy/rtp_offline_compare.py` measures how close the real-time processing is to standard offline processing of the same run, with your own data and parameters (as evaluated in Misaki and Bodurka, 2022).

It runs the RTPSpy modules volume by volume, as in a real-time session (VOLREG → TSHIFT → SMOOTH → REGRESS; the RETROICOR regressors of each volume use only the physio signals recorded up to that volume). It then processes the whole run offline with matched settings (AFNI `3dvolreg`, `3dTshift`, `3dBlurInMask`, and a whole-run GLM with the same regressors), and compares the two.

```bash
python -m rtpspy.rtp_offline_compare --input func.nii.gz --masks_dir work/RTP \
    --roi work/RTP/sub-01_task-rest_run-01_label-ROI_mask.nii.gz \
    --card ECG.1D --resp Resp.1D --physio_freq 40 \
    [--param rtp_params.json] --out compare_out
```

- `--masks_dir`: the folder of the masks made by 'Create masks'. Each mask can also be given with `--rtp_mask`, `--gs_mask`, `--wm_mask`, `--vent_mask`.
- `--param`: a JSON file of module parameters, e.g., `{"VOLREG": {"regmode": "cubic"}, "REGRESS": {...}}`. The default is the setting of `rtpspy_system_check.py`.
- The output folder has `report.html` (with `report.png`), `metrics.csv`, `roi_series.csv`, the online outputs (`online/RTP/`), and the offline outputs with the AFNI commands (`offline/`).

<details>
<summary>The metrics</summary>

The metrics are computed over the volumes output by the online pipeline, within the RTP mask: the correlation and maximum difference of each motion parameter; the voxelwise correlation and RMSE of each stage (median and 5th percentile); the ROI mean signal correlation; the noise variance removed by the regression (% of the input variance); and `REGRESS_causal`, the difference between the cumulative (online) GLM and the whole-run GLM applied to the same online input. The `*_realtime_vols` metrics exclude the volumes that REGRESS processes retrospectively at its first fit.
</details>

## For developers

### Running the tests

The tests in `tests/` use pytest (installed by `install.sh`) and only the test data in `tests/` or synthetic data.

```bash
cd ~/RTPSpy
source .venv/bin/activate
QT_QPA_PLATFORM=offscreen python -m pytest tests                 # all tests
python -m pytest tests -m "not gpu and not afni and not slow"    # quick CPU-only tests
```

Tests marked `gpu` (needs CUDA) and `afni` (compares with AFNI programs) are skipped when CUDA or AFNI is not available; `slow` marks long runs.

### The librtp library

RTPSpy uses some AFNI C functions compiled into `librtp.so` (`librtp.dylib` on macOS). The bundled Linux library needs GLIBC 2.27 or later; if it does not load, `install.sh` rebuilds it in a container. See [Compile the librtp library](./Compilelibrtp.md).

### Using a local FastSurfer instead of the container

A FastSurfer clone in `rtpspy/FastSurfer` is used when `RTPSPY_FASTSURFER=local` is set, or when the container is not set up:

```bash
git clone --branch v2.6.0 https://github.com/Deep-MI/FastSurfer.git ~/RTPSpy/rtpspy/FastSurfer
uv pip install --python ~/RTPSpy/.venv/bin/python -r ~/RTPSpy/rtpspy/FastSurfer/requirements.txt
```

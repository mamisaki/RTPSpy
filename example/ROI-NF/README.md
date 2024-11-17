# RTPSpy ROI-NF application
This example demonstrates the use of the RTPSpy GUI for real-time processing. The application extracts the average signal from a region of interest (ROI) and transmits it in real time to an external application. The external application then displays the received value in a PsychoPy window.

## Operations
Here we use test data to demonstrate the ROI-NF operations with simulated real-time fMRI.

### 1. Start  
Run the following commands in a console.  
```
cd ~/RTPSpy/example/ROI-NF
./boot_ROINF_RTPSpy.sh
```

The ROI-NF App window opens.  
<img src="/example/ROI-NF/doc/initial.png" height=400>  

> [!NOTE]
> If this is not your first time using the application, you will be prompted with 'Load previous settings?' The application automatically saves your settings upon exit, allowing them to be reloaded during your next session. Select 'Yes' to load the previous settings.  

<img src="/example/ROI-NF/doc/load_settings.png" height=100>  

### 2. Set the watching and woking directories  
The first step is to set up the watching and working directories at the top of the window.  
The 'rtfMRI Watching Directory' is where fMRI images (e.g., DICOM files) are transferred in real time during the scan. It can also serve as the location where image files are copied to simulate real-time MRI.  
The Working Directory is used to store all processed images and related files.  
<img src="/example/ROI-NF/doc/SetWatch.png"  width=400>  

### 3. Scan anatomy and reference function images to make the mask images  
Scan anatomical and functional images to use as a reference and to create mask images for real-time processing.  
For the reference functional images, both a positional reference (e.g., sbref) and a parameter reference, which uses the same scanning parameters as the real-time fMRI scan (e.g., image space, TR, and multiband settings), need to be acquired.  

#### Convert DICOM files received in real time  
Let's convert the DICOM files of the anatomical and reference functional images.   
- Press the 'dcm2niix' button on the 'Preprocessing' tab in the 'App' panel.  
- Selct folder where the DICOM files for the current session stored.  

The 'dcm2niix' command will run for the selected folder, and the converted NIfTI files will be saved in the 'working' directory.

#### Set the 'Anatomy image' and 'Base function image'  
Once the NIfTI images are ready, use the 'Set' buttons to assign the 'Anatomy Image' and 'Base Function Image.  
<img src="/example/ROI-NF/doc/MaskCreation.png" height=400>  

#### Set the template files
Press the 'Template' tab and set the 'Template brain' and 'ROI on template' files. These files hsould be prepared before the scanning session.

> White matter and ventricle segmentation are performed in the native image space by FastSeg during the mask creation process. Therefore, 'White Matter on Template' and 'Ventricle on Template' do not need to be set. However, if FastSeg is unavailable, these images will be used to generate the masks.
<img src="/example/ROI-NF/doc/SetTemplate.png" height=400>  

#### Create masks
Go to the 'Preprocessing' tab and click the 'Create Masks' button.  
> If a GPU is not available, 'No FastSeg' will be selected. In this case, 3dSkullStrip and template white matter and ventricle masks will be used instead of FastSeg.  

The 'Mask Image Creation' dialog will then open.
<img src="/example/ROI-NF/doc/MaskCreationDlg.png" height=200>  

### Display the masks in AFNI  
Press the buttons in the 'Display the Masks in AFNI' box to verify image alignment.
The AFNI GUI will open, showing the created mask image overlaid on the reference image.

### 4. Set an external application  
> [!NOTE]  
> This step can be skipped if the setup was completed in a previous session and loaded via the 'Load Previous Settings?' prompt at startup.  

The external application that receives the ROI signals in real time can be configured in the 'Ext App' tab.  
<img src="/example/ROI-NF/doc/ExtApp.png" height=400>  

The 'App Command:' field displays the command line used to launch the external application.  
> Edit the options --screen, --size, and --pos to adjust the PsychoPy window's shape and position.

The example application, a PsychoPy script named NF_psypy.py, demonstrates how to receive ROI signals from RTPSpy in real time. It simply displays the received values on the screen.  

Press the 'Run App' button to open the application window.  
> - It may take a minute to load, depending on the PsychoPy installation environment.  
> - The initial window displays the address and port of the application server (RTP_SERV) for communication with RTPSpy.  
> - You can quit this sample application by pressing 'ESC' when the window has a focus.

### 5. RTP setup
1. Set the 'fMRI Parameter Reference' image on the 'Preprocessing' tab. The parameter reference must use the same scanning parameters as the real-time fMRI scan (e.g., image space, TR, and multiband settings).  

2. Press the 'RTP Setup' button to configure the real-time processing parameters. If the application window is not already open, it will launch automatically.  

3. Detailed settings for real-time processing parameter
> [!NOTE]  
> - This step can be skipped if the setup was completed in a previous session and loaded via the 'Load Previous Settings?' prompt at startup.  
> - Session-specific paramters are atomotically updated when the 'RTP setup' button is pressed.

You can adjust the real-time processing parameters in the 'RTP' panel.  
<img src="/example/ROI-NF/doc/REGRESS.png" height=400>  

Press a module button to set detailed parameters.
If the parameters are modified, you need to press the 'RTP Setup' button.

### 6. Ready to scan
Press the 'Ready' button.  
- The application window shows a 'Ready' message.  
- The watchdog thread will start monitoring the 'rtfMRI Watching Directory.'

### 7. Start the scan and real-time processing will
You are now ready to start the scan.

- If the TTL signal trigger is set, the RTP and PsychoPy application will start automatically with the trigger input.  
- Alternatively, you can start the process manually by pressing the 'Manual Start' button.

> Regression begins after receiving a certain number of images, as specified in the 'RTP' -> 'REGRESS' -> 'Wait REGRESS Until' field.

This boilerplate PsychoPy application simply displays the received values. To create a functional feedback application, edit the script at '[example/ROI-NF/NF_psypy.py](/example/ROI-NF/NF_psypy.py)'  

## Customize the application
The ROI-NF application class, [example/ROI-NF/roi_nf.py](/example/ROI-NF/roi_nf.py), extracts the mean signal from the ROI and sends the value to an external application.
The external PsychoPy application, [example/ROI-NF/NF_psypy.py](/example/ROI-NF/NF_psypy.py), displays the received value on the screen.
You can customize the application by editing these files.

### roi_nf.py
Neurofeedback signal extraction is performed in the 'do_proc' method of the ROINF class, defined in the roi_nf.py file.  
The figure below shows a code snippet for extracting the signal from a processed fMRI image (the output of RtpRegress). By modifying this section, you can define a custom neurofeedback signal extraction method.   
<img src="/example/ROI-NF/doc/roi_nf_custom.png" width=600>  

To customize the neurofeedback signal calculation, follow these steps:
1. Create a new application class inheriting from RtpApp (A).
2. Override the do_proc method (B) to define your custom processing.
3. Use the ROI_orig property (C), set during the mask creation process, to specify the ROI mask file.
4. Implement signal calculation logic, such as calculating the mean value within the ROI mask (D).
5. Format the signal using a specific format string (E).
6. Send the signal to an external application in real-time with the send_extApp method (F).

### NF_psypy.py
The neurofeedback application is defined in the NF_psypy.py file. This standalone PsychoPy application communicates with an RTPSpy application. The figure below shows a script snippet for handling this communication.  
<img src="/example/ROI-NF/doc/NF_psypy_custom.png" width=600>

The RTP_SERVE class, defined in the rtp_serve.py file within the RTPSpy package, facilitates signal and text message exchange with an RTPSpy application. Here’s how it works:
1. Instantiating the RTP_SERVE class starts a TCP/IP socket server in a separate thread (A), handling data exchange in the background.
2. The received neurofeedback data is stored in a Pandas DataFrame (https://pandas.pydata.org/) (B).
3. The example script displays the most recent value on the screen as text (C). You can modify this section to create a more advanced feedback presentation.

## Additional Example: LA-NF Application
For a more comprehensive application example, refer to the LA-NF application in the example directory: [example/LA-NF/roi_nf.py](/example/LA-NF/),

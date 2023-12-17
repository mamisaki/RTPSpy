# RTPSpy ROI-NF application
This is a boilerplate application with RTPSpy GUI. The application extracts the average signal in an ROI and sends the signal to an external application in real-time. The external application displays the received value on a psychopy window.
* On Mac OSX, the GUI layout is a bit broken due to an incompatibility of PyQt.  

## Operations
Here, we use test data to demonstrate the ROI-NF operations with simulating real-time fMRI.

### 1. Start
1. Open a console (Ubuntu console if you are working on the Windows WSL, or Terminal application on Max OSX).  
2. Activate the RTPSpy env and run the boot script.  
```
conda activate RTPSpy
cd ~/RTPSpy/example/ROI-NF
./run_RTPSpyApp.py
```
* On Mac OSX, if you run the 'run_RTPSpyApp.py' as a background job (with '&'), the process may be stacked at booting a PsychoPy application.


3. The ROI-NF App window opens.  
<img src="/example/ROI-NF/doc/initial.png" height=400>  

If this is not the first time, you will be asked, 'Load the previous settings?'  
The application automatically saves the previous settings at the exit and can be loaded the next time. Press 'Yes' if you want to retrieve the previous settings.  
<img src="/example/ROI-NF/doc/load_settings.png" height=100>  

### 2. Set the Watching and Woking directories.  
The first thing you should do is to set the Watching and Woking directories at the top of the window.  
The watching directory is the place where the fMRI image will be made in real-time during the scan. This place is also used to copy image files to simulate the real-time image reconstruction in a simulation.  
The working directory is the place where all processed images and files are saved.  

For the demonstration, let's make the directories, 'watch' and 'work' in the test folder and set them with the 'Set' button.  
<img src="/example/ROI-NF/doc/SetWatch.png"  width=400>  

### 3. Scan anatomy and reference function images to make the mask images.  
Start MRI scan for taking anatomical and reference functional images.  
For the demonstration, we will use sample images in the 'RTPSpy/test' directory.  

#### Set files for 'Anatomy image' and 'Base function image'.  
Once the images are ready, press the 'Mask creation tab' and set the 'Anatomy image' and 'Base function image'.  
For the demonstration, let's set the 'anat_mprage.nii.gz' and 'func_epi.nii.gz' files in the 'RTPSpy/test' direcory.  
<img src="/example/ROI-NF/doc/MaskCreation.png" height=400>  

#### Set template files
Press the 'Template tab' and set the 'Template brain', 'ROI on template', 'White matter on template', and 'Ventricle on template' files.  
For the demonstration, let's set 'MNI152_2009_template.nii.gz', 'MNI152_2009_template_LAmy.nii.gz', 'MNI152_2009_template_WM.nii.gz', and 'MNI152_2009_template_Vent.nii.gz' files in the 'RTPSpy/test' direcory.  
<img src="/example/ROI-NF/doc/SetTemplate.png" height=400>  

#### Create masks
Press the 'Mask creation' tab and press the 'Create masks ...' button.  
If GPU is not available, 'No FastSeg' is checked. Then 3dSkullStrip and template white-matter and ventricle masks will be used, instead of FastSeg.  
The 'Mask image creation' dialog will open.  
<img src="/example/ROI-NF/doc/MaskCreationDlg.png" height=200>  

### 4. Set simulation files (for the demonstration)
For the demonstration, we will run a real-time fMRI simulation. This will not be necessary for an actual scan session.  
Press the 'Simulation' tab.  
* Set the 'fMRI data', 'ECG data', and 'Respiration data' to 'func_epi.nii.gz', 'ECG.1D', and 'Resp.1D' in the 'RTPSpy/test' direcory.   
* Select the 'Simulation physio port'. This can be different depending on the PC.  
* Check the 'Enable simulation'.
<img src="/example/ROI-NF/doc/Simulation.png" height=400>  

### 5. Boot an external application
Press the 'Ext App'.  
<img src="/example/ROI-NF/doc/ExtApp.png" height=400>  

'App command:' field shows the command line to boot an external psychopy application.  
    * Edit the options '--screen', '--size', and '--pos' to set the psychopy window shape.  
Press the 'Run App' button, and the psychopy window opens.  
    * It may take a minute depending on the psychopy install environment.  
The initial window shows the address and port of the application server (RTP_SERV) to communicate with RTPSpy.  
    * You can quit the psychopy application by pressing 'ESC' when the window has a focus.  
<img src="/example/ROI-NF/doc/PsychoPyInit.png" height=400>  

### 6. Set up RTP
You can adjust the real-time processing parameters on the 'RTP' tab.  
1. Press the 'RTP' tab above the yellow 'ROI-NF' bar.
2. Press a module button to set detailed parameters.
    Here, let's press the 'REGRESS' button and set the 'RICOR regressor' to '8 RICOR (4 Resp and 4 Card)'.  
    * Mask files for the global signal, WM, and Vent will be automatically set at pressing the 'RTP setup' button if they are checked.
    <img src="/example/ROI-NF/doc/REGRESS.png" height=400>  
    
3. Return to the 'App' tab by pressing the 'App' above the yellow 'ROI-NF' bar.
Press the 'RTP setup' button.
The dialog 'TSHIFT: Select slice timing sample' will open. Let's select func_epi.nii.gz in the 'RTPSpy/test' directory.  
<img src="/example/ROI-NF/doc/TSHIFT_SelectSpliceTiming.png" height=300>  

### 7. Ready to scan start
Press the 'Ready' button.
The Physio and Motion windows will open. The psychopy window shows a 'Ready' message.
If the simulation is enabled, the 'Simulation' tab will be shown.  
<img src="/example/ROI-NF/doc/Ready.png" height=400>  

### 8. Start the scan and real-time processing
You are ready to start the scan.  
* If the TTL signal trigger is set (in the 'RTP' -> 'EXTSIG' tab), the RTP and the psychopy application will start with the trigger input.  
* You can also start the process manually by pressing the 'Manual start' button.  
Here, for the real-time simulation, let's press the 'Start scan simulation' button on the 'Simulation' tab.  
Then, the simulated ECG/respiration recordings and fMRI image creation will start.  
<img src="/example/ROI-NF/doc/RTPRunning.png" height=400>  

* The regression starts after receiving enough samples.   
* This boilerplate psychopy application just displays the received value. Edit the '[example/ROI-NF/NF_psypy.py](/example/ROI-NF/NF_psypy.py)' to make a proper feedback application.

## Customize the application
The ROI-NF application, '[example/ROI-NF/ROINF.py](/example/ROI-NF/roi_nf.py)', extracts the mean signal in the ROI to send the value to an external application.  
The external psychopy application, '[example/ROI-NF/NF_psypy.py](/example/ROI-NF/NF_psypy.py)', displays the received value on the screen.  
You can customize the application by editing these files.

### roi_nf.py
Neurofeedback signal extraction is performed in the 'do_proc' method in the ROINF class, defined in the 'roi_nf.py' file.  
The figure below shows the code snippet of the signal extraction from a processed fMRI image (the output of RtpRegress). By modifying this part, a user can define a custom neurofeedback signal extraction.  
<img src="/example/ROI-NF/doc/roi_nf_custom.png" width=600>  

To customize the neurofeedback signal calculation, you should make a new application class inheriting RtpApp (A) and override the 'do_proc' method (B). The example script calculates the mean value within the ROI mask (D). The ROI mask file is defined in the 'ROI_orig' property (C), which has been set in the mask creation process. The signal can be sent to an external application in real-time using the 'send_extApp' method (F) by putting it in a specific format string (E).

### NF_psypy.py
The neurofeedback application is defined in the 'NF_psypy.py' file. This is an independent PsychoPy application with the functionality of communicating to an RTPSpy application. The figure below shows the snippet of the script for the communication.  
<img src="/example/ROI-NF/doc/NF_psypy_custom.png" width=600>  

The RTP_SERVE class, defined in the 'rtp_serve.py' file in the RTPSpy package, allows exchanging signals (and text messages) with an RTPSpy application. Instantiating the class object starts a TCP/IP socket server running in another thread (A). This class does all the data exchange in the background. The RTP_SERVE object holds the received neurofeedback data in the pandas data frame format (https://pandas.pydata.org/) (B). The example script just shows the text of the latest received value on the screen (C). You can modify this part to make a decent feedback presentation.

* Refer also to the LA-NF application in the example directory, [example/LA-NF](/example/LA-NF#readme), for an example of full-fledges application.


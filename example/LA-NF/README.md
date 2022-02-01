# RTPSpy LA-NF application
This is an example application with RTPSpy GUI for the left amygdala neurofeedback seesion (Zotev et al., 2011;Young et al., 2017). The application extracts the average signal in the left amygdala (LA) region and sends the signal to an external application in real-time. This script is for a demonstration of a full-fledged GUI application using RTPSpy. Note that the application was not used in the previous studies (Zotev et al., 2011;Young et al., 2017), and several parameters are different from the previous reports.  
* On Mac OSX, the GUI layout is a bit broken due to an incompatibility of PyQt.  

    - Young et al. (2017). Randomized Clinical Trial of Real-Time fMRI Amygdala Neurofeedback for Major Depressive Disorder: Effects on Symptoms and Autobiographical Memory Recall. Am J Psychiatry 174, 748-755.  
    - Zotev et al. (2011). Self-Regulation of Amygdala Activation Using Real-Time fMRI Neurofeedback. PLoS ONE 6, e24522.  

## Operations
### 1. Start
1. Open a console (Ubuntu console if you are working on the Windows WSL, or Terminal application on Max OSX).  
2. Activate the RTPSpy env and run the boot script.  
```
conda activate RTPSpy
cd ~/RTPSpy/example/LA-NF
./run_RTPSpyApp.py
```
* On Mac OSX, if you run the 'run_RTPSpyApp.py' as a background job (with '&'), the process may be stacked at booting a PsychoPy application.

3. The dialogs, 'Select Watch directory' and 'Select working directory' are opened.  

<img src="/example/LA-NF/doc/select_watch_dir.png" height=200>  

Set the watching and warking directory. You can change these later. If you canceled the dialog, the previous settings are used. The application automatically saves the previous settings at the exit and loads the saved parameters at booting.  

The LA-NF App window opens.  
<img src="/example/LA-NF/doc/LA-NF_initial.png" height=400>  

* The previous settings are saved in the 'RTPSpy_params.pkl' file in the application directory. Remove this file if you don't want to load the previous settings, or press the 'Reset to default parameters' button to reset the parameters (wathcing and working directories have no default settings so they are not changed by the button).  

### 2. Scan anatomy and reference function images to make the mask images.  
Start MRI scan for taking anatomical and reference functional images.   

#### Set files for 'Anatomy image' and 'Base function image'.  
Once the images are ready, press the 'Go to mask creation' button and set the 'Anatomy image' and 'Base function image'.  
<img src="/example/LA-NF/doc/LA-NF_MaskCreation.png" height=400>  

#### Create masks
Press the 'Create masks ...' button.  
If GPU is not available, 'No FastSeg' is checked. Then, 3dSkullStrip and template white-matter and ventricle masks will be used, instead of FastSeg.  
The 'Mask image creation' dialog will open to show the progress.  
Once the process finishes, all the result images are automatically set as masks for the real-time processing.  
<img src="/example/LA-NF/doc/MaskCreationDlg.png" height=200>  

### 3. Set RTP parameters
You can adjust the real-time processing parameters on the 'RTP' tab above the yellow 'ROI-NF' bar.  
Press a module button to set detailed parameters.  
<img src="/example/LA-NF/doc/LA-NF_REGRESS.png" height=400>  

### 4. Run a task session
Return to the 'App' tab by pressing the 'App' above the yellow 'ROI-NF' bar.
Press the 'Go to task' button at the bottom of the 'Mask creation' tab or press the 'Task' tab to show the 'Task' operation interface.   
<img src="/example/LA-NF/doc/TaskOperation.png" height=400>  

Press one of the session setup buttons (e.g., resting state, baseline run without neurofeedback, neurofeedback training run 1, etc.) in the 'Session setup' area. This starts the neurofeedback presentation application if it is not running, and run the RTP setup. 

The neurofeedback presentation window opens with the task instruction screen.   
<img src="/example/LA-NF/doc/TaskInstruction.png" height=400>  

The size and the position of the application screen can be adjusted by command line variables in the 'App command:' field on the 'Ext App' tab.  
Add '--fullscr' option to the command line to make it full screen.  
<img src="/example/LA-NF/doc/LA-NF_ExtApp.png" height=400>  

The dialog 'TSHIFT: Select slice timing sample' will open if the slice timings are not set on the TSHIFT tab in the RTP tab. Select a function image file having the slice-timing information. This is necessary only for the first scan session.   
<img src="/example/LA-NF/doc/TSHIFT_SelectSpliceTiming.png" height=300>  

### 5. Ready to scan start
Press the 'Ready' button.
The psychopy application window shows a 'Ready' message.  
<img src="/example/LA-NF/doc/Ready.png" height=400>  

### 6. Start the scan and real-time processing
You are ready to start the scan.  
* If the TTL signal trigger is set (in the 'RTP' -> 'SCANONSET' tab), the RTP and the psychopy application will start with the trigger input.  
* You can also start the process manually by pressing the 'Manual start' button.  

The regression starts after receiving enough samples.   
The application shows the taks block indication and the neurofeedback signal in the happy block.  
<img src="/example/LA-NF/doc/LA-NF_happy.png" height=400>  

* Refer also to the ROI-NF application in the example directory, [example/ROI-NF](/example/ROI-NF), for a simple boilarplate application to develop a custom application.

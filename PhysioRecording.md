# Cardiogram and Respiration Signal sand Scan Timing TTL signal Recording

RTPSpy can apply the RETROICOR noise regression online.  
The 'RtpPhysio' class, defined in rtpspy/rtp_ttl_physio.py, records cardiogram and respiration signals in real-time and provides RETROICOR regressors via the RtpPhysio.get_retrots function.  

RtpPhysio currently supports two types of signal recorders:
* GE scanner devices whose signals are sent via serial port.
* Analog input from BIOPAC devices received via Numato USB GPIO device.

In addition, for simulation purposes, a dummy device that reads signals from existing files (text files) can also be used.

These devices are set by the 'device' property of the RtpPhysio class as device='Numato' for the Numato USB GPIO, device='GE' for the GE scanner device, and device='dummy' for the dummy device.

The TTL signal for the scan onset timing from a scanner is also 

## GE scanner devices whose signals are sent via serial port
Cardiogram and respiration signals from the GE scanner's pulse oximetry and respiration belt devices can be read in real-time via serial port signal.

## Analog input from BIOPAC devices received via Numato USB GPIO device
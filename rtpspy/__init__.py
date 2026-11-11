"""
RTPSpy: a library for building real-time fMRI processing applications

The processing modules run without a GUI (and without PyQt6). The GUI front
end is in rtpspy.ui; RtpGUI is imported from there on first access.
"""
from .rtp_common import RTP
from .rtp_volume import RtpVolume
from .rtp_watch import RtpWatch
from .rtp_tshift import RtpTshift
from .rtp_volreg import RtpVolreg
from .rtp_smooth import RtpSmooth
from .rtp_regress import RtpRegress
from .rtp_app import RtpApp
from .rtp_ttl_physio import RtpTTLPhysio
from .rtp_retrots import RtpRetroTS
from .rtp_imgproc import RtpImgProc
from .rpc_socket_server import RPCSocketServer, RPCSocketCom
from .dicom_converter import DicomConverter
from .dicom_reader import DicomReader
from .rt_physio import SharedMemoryRingBuffer
from .containers import setup_path as _setup_path

# AFNI programs in the container (install.sh) and ants_run.py on PATH
_setup_path()

# RtpGUI is not in __all__ so that 'from rtpspy import *' does not need Qt.
__all__ = ["RTP", "RtpVolume", "RtpWatch", "RtpTshift", "RtpVolreg",
           "RtpSmooth", "RtpRegress", "RtpApp", "RtpTTLPhysio", "RtpRetroTS",
           "RtpImgProc", "RPCSocketServer", "RPCSocketCom", "DicomConverter",
           "DicomReader", "SharedMemoryRingBuffer"]


def __getattr__(name):
    # The GUI (PyQt6) is imported only when it is used, so that the
    # processing modules run without Qt and without a display.
    if name == "RtpGUI":
        from .ui.rtp_gui import RtpGUI
        return RtpGUI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

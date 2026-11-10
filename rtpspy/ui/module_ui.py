# -*- coding: utf-8 -*-
"""
Base class of the GUI panels of the RTP modules

A panel class binds to a processing object (an RTP module or an RtpApp).
make_ui() makes the widgets and returns the rows that RtpGUI places in a
QFormLayout. The widgets are kept as ui_* attributes of the processing
object, as in earlier versions, so that an application class (e.g., one
overriding ui_set_param) and RtpGUI can access them.

The processing object calls the bound panel through RTP._ui_update (show a
parameter value in the widget), RTP._ui_value (read a GUI-only state), and
RTP._ui_call (run a GUI action). Without a bound panel these do nothing.

@author: mmisaki@laureateinstitute.org
"""

# %% import ===================================================================
import importlib

from PyQt5 import QtWidgets


# Text of a mask file field when the mask is made from the first received
# volume (mask_file 0; RtpSmooth and RtpRegress)
ZERO_OUT_MASK_TEXT = 'zero-out initial received volume'


def mask_text(mask_file):
    """Text of a mask file field."""
    if type(mask_file) is int and mask_file == 0:
        return ZERO_OUT_MASK_TEXT
    return str(mask_file)


# %% bind_ui ==================================================================
def bind_ui(obj):
    """Make the GUI panel of obj (its class attribute _ui_class,
    'module:class' in rtpspy.ui) and bind it to obj (obj._ui)."""
    mod_name, cls_name = obj._ui_class.split(":")
    mod = importlib.import_module(f"rtpspy.ui.{mod_name}")
    return getattr(mod, cls_name)(obj)


# %% RtpModuleUI ==============================================================
class RtpModuleUI:
    """GUI panel bound to a processing object (self.obj)."""

    # {parameter: (widget attribute of obj, setter[, value conversion])}
    # for the parameters shown in a single widget
    _widget_setters = {}

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, obj):
        self.obj = obj
        obj._ui = self

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def make_ui(self):
        """Make the widgets. Returns the list of rows for a QFormLayout."""
        raise NotImplementedError

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _widget(self, name):
        """Widget obj.<name>, or None if it is not made."""
        return getattr(self.obj, name, None)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def update_widget(self, attr, val, **kwargs):
        """Show the value of parameter attr in its widget."""
        o = self.obj
        if attr == "enabled":
            if hasattr(o, "ui_enabled_rdb"):
                o.ui_enabled_rdb.setChecked(val)

            if hasattr(o, "ui_objs"):
                for ui in o.ui_objs:
                    ui.setEnabled(val)

        elif attr == "save_proc":
            if hasattr(o, "ui_saveProc_chb"):
                o.ui_saveProc_chb.setChecked(val)

        elif attr in self._widget_setters:
            name, setter, *conv = self._widget_setters[attr]
            widget = self._widget(name)
            if widget is not None:
                if conv:
                    val = conv[0](val)
                getattr(widget, setter)(val)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def widget_value(self, key):
        """Return a state held only in the GUI (None if unknown)."""
        return None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def enabled_row(self):
        """'Enable' radio button row (obj.ui_enabled_rdb)."""
        o = self.obj
        o.ui_enabled_rdb = QtWidgets.QRadioButton("Enable")
        o.ui_enabled_rdb.setChecked(o.enabled)
        o.ui_enabled_rdb.toggled.connect(
            lambda checked: o.set_param(
                "enabled", checked, o.ui_enabled_rdb.setChecked))
        return (o.ui_enabled_rdb, None)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def save_proc_row(self, left=None):
        """'Save processed image' check box row (obj.ui_saveProc_chb) at the
        right, with the widget left at the left."""
        o = self.obj
        o.ui_saveProc_chb = QtWidgets.QCheckBox("Save processed image")
        o.ui_saveProc_chb.setChecked(o.save_proc)
        o.ui_saveProc_chb.stateChanged.connect(
            lambda state: setattr(o, "save_proc", state > 0))
        o.ui_objs.append(o.ui_saveProc_chb)

        chb_hLayout = QtWidgets.QHBoxLayout()
        chb_hLayout.addStretch()
        chb_hLayout.addWidget(o.ui_saveProc_chb)
        return (left, chb_hLayout)

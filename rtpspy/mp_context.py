# -*- coding: utf-8 -*-
"""
Multiprocessing context of RTPSpy.

The child processes of RTPSpy (physio recording and plot, DICOM conversion)
run bound methods of objects that hold open files, locks, sockets, and
loggers, and they use the state inherited from the parent process. They
need the 'fork' start method. Python 3.14 changed the default start method
on Linux from 'fork' to 'forkserver', which pickles the process target and
fails with these objects, so the 'fork' context is used explicitly on
Linux. Other platforms keep their default ('spawn' on macOS).

Create the processes and the objects shared with them (Lock, Queue, Value,
Array) from this context:
    from .mp_context import mp_ctx
    proc = mp_ctx.Process(target=...)
"""

import multiprocessing
import sys

if sys.platform.startswith('linux'):
    mp_ctx = multiprocessing.get_context('fork')
else:
    mp_ctx = multiprocessing.get_context()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Send GE DICOM image files to the RTPSpy computer as soon as they are
written.

Runs on the GE MR console (RHEL6, Python 2.6.6). Standard library only, so
the file can be copied to the console and run as is. Also runs on Python 3
for testing.

The script watches the GE image pool (p*/e*/s*/i*.MRDC.*) and sends each
new file to the receiver on the RTPSpy computer (rtpspy/rt_dicom_receiver.py,
started by RtpWatch with 'ge_port') over a single TCP connection.
It does not parse DICOM.

File detection uses inotify (through ctypes), so that the console is not
loaded by listing the large image pool:
  - New directories: the image pool, the latest p* directory and its latest
    e* directory are watched (IN_CREATE). A new p*/e* directory becomes the
    latest one. All p*/e*/s* directories are listed only at start, after an
    inotify queue overflow, and every 'full_rescan_interval' seconds (for a
    new series in an older exam).
  - Files: an active series directory is watched (IN_CLOSE_WRITE /
    IN_MOVED_TO), and the file is sent at once. If a file is closed again
    with a different size, it is sent again.
  - Missed files: every 'rescan_interval' seconds, only the modification
    time of each active series directory is checked. The directory is
    listed only if a file was created after the last inotify event. Files
    found this way are sent once their size is unchanged between two checks.
  - Late files: a series with no new file for 'active_timeout' seconds is
    no longer watched, but its modification time is still checked at each
    rescan for 'dormant_timeout' seconds. If files are added (e.g., the
    second acquisition of a 2D FLAIR, written 86 s after the first), the
    series is watched again and only the files not sent yet are sent.
If inotify is not available, the active series directories are polled, and
each one is listed only when its modification time has changed.

Connection to the receiver: the sender can run all the time, while the
receiver on the RTPSpy computer runs only when needed.
  - At each new series, the sender connects to the receiver again (the old
    connection may be stale after a receiver restart).
  - If the receiver is not available, the series waits: its files are not
    queued. While it is active, the connection is tried every
    'retry_interval' seconds. When it succeeds, all files of the series
    written so far are sent, then the new ones.
  - If the connection is lost during a series, the unsent files are kept
    and sent after the connection is made again.
  - Connection failures are logged once, not at every retry.

--dry_run: files are not sent. The number of files found and the detection
delay (time from the file modification to its detection) are logged every
10 s, to test the load and the speed on the console without the RTPSpy
computer.

Protocol (per file):
    sender  -> b'GEDC' + struct('>HI', len(name), len(data)) + name + data
    receiver -> b'K' (written) or b'E' (rejected)
'name' is the path relative to the image pool, e.g.
'p2948/e3139/s29216/i211267344.MRDC.1'.

Running all the time (--ensure_running, from cron):
    --ensure_running starts the sender detached from the session if it is
    not running, and exits. Run it at boot and every minute from cron.
  - The running sender holds a lock (flock) on its PID file
    (<log dir>/ge_dicom_sender.pid). The kernel releases the lock when the
    process ends, also on a crash, so a held lock means the sender runs.
  - If the running sender's arguments (/proc/<pid>/cmdline) differ from the
    requested ones, it is stopped and started with the new arguments.
  - Another lock serializes concurrent --ensure_running calls.
  - The log (--log_file) is rotated at 5 MB, keeping 3 old files. Output to
    stderr (e.g., a traceback) goes to <log dir>/ge_dicom_sender.stderr.

Usage (tcsh on the console; <host> is the IP address of the RTPSpy
computer):
    nohup python ge_dicom_sender.py --host <host> >& sender.log &
    python ge_dicom_sender.py --dry_run      (test without sending)
    python ge_dicom_sender.py --ensure_running \
        --log_file ~/rtpspy/log/ge_dicom_sender.log --host <host>

See GESetup.md in the RTPSpy package for the setup.

mmisaki@laureateinstitute.org
"""

# %% import ===================================================================
import os
import re
import sys
import time
import errno
import select
import socket
import struct
import signal
import fcntl
import logging
import logging.handlers
import subprocess
import ctypes
import ctypes.util
from optparse import OptionParser

MAGIC = 'GEDC'.encode('ascii')
DICOM_PAT = re.compile(r'^i\d+\.MRDC\.(\d+)$')

# inotify constants (linux/inotify.h)
IN_CLOSE_WRITE = 0x00000008
IN_MOVED_TO = 0x00000080
IN_CREATE = 0x00000100
IN_DELETE_SELF = 0x00000400
IN_Q_OVERFLOW = 0x00004000
IN_IGNORED = 0x00008000
IN_ISDIR = 0x40000000
EVENT_HEAD = struct.Struct('iIII')  # wd, mask, cookie, len


# %% class Inotify ============================================================
class Inotify(object):
    """Minimal inotify wrapper with ctypes."""

    def __init__(self):
        libc_name = ctypes.util.find_library('c') or 'libc.so.6'
        self._libc = ctypes.CDLL(libc_name, use_errno=True)
        self.fd = self._libc.inotify_init()
        if self.fd < 0:
            raise OSError(ctypes.get_errno(), 'inotify_init failed')
        self._wd_path = {}
        self._path_wd = {}

    def add_watch(self, path, mask):
        if path in self._path_wd:
            return self._path_wd[path]
        wd = self._libc.inotify_add_watch(
            self.fd, path.encode(sys.getfilesystemencoding()), mask)
        if wd < 0:
            raise OSError(ctypes.get_errno(), 'inotify_add_watch failed: '
                          + path)
        self._wd_path[wd] = path
        self._path_wd[path] = wd
        return wd

    def rm_watch(self, path):
        wd = self._path_wd.pop(path, None)
        if wd is not None:
            self._wd_path.pop(wd, None)
            self._libc.inotify_rm_watch(self.fd, wd)

    def read_events(self, timeout):
        """Return a list of (dir_path, mask, name) read within timeout."""
        try:
            ready = select.select([self.fd], [], [], timeout)[0]
        except select.error:
            e = sys.exc_info()[1]
            if e.args[0] == errno.EINTR:
                return []
            raise
        if not ready:
            return []

        buf = os.read(self.fd, 64 * 1024)
        events = []
        pos = 0
        while pos + EVENT_HEAD.size <= len(buf):
            wd, mask, cookie, nlen = EVENT_HEAD.unpack_from(buf, pos)
            pos += EVENT_HEAD.size
            name = buf[pos:pos + nlen].rstrip('\0'.encode('ascii'))
            pos += nlen
            name = name.decode(sys.getfilesystemencoding())
            if mask & IN_IGNORED:
                path = self._wd_path.pop(wd, None)
                if path is not None:
                    self._path_wd.pop(path, None)
                continue
            events.append((self._wd_path.get(wd), mask, name))
        return events

    def close(self):
        os.close(self.fd)


# %% class GeDicomSender ======================================================
class GeDicomSender(object):
    """Watch the GE image pool and send new DICOM files to RTPSpy."""

    def __init__(self, image_root, host, port, use_inotify=True,
                 poll_interval=0.05, rescan_interval=3.0,
                 full_rescan_interval=60.0, active_timeout=60.0,
                 dormant_timeout=1800.0,
                 connect_timeout=2.0, retry_interval=5.0, dry_run=False):
        """
        Parameters
        ----------
        image_root : str
            GE image pool root (p*/e*/s* directories are under this).
        host, port : str, int
            Address of rt_dicom_receiver.py.
        use_inotify : bool
            Use inotify to detect files. If False or inotify is not
            available, files are found by polling.
        poll_interval : float
            Interval (s) to check files in polling mode, and to check the
            size of files found by a directory listing.
        rescan_interval : float
            Interval (s) to check the active series directories for files
            missed by inotify.
        full_rescan_interval : float
            Interval (s) to list all p*/e*/s* directories.
        active_timeout : float
            A series directory with no new file for this period (s) is no
            longer watched.
        dormant_timeout : float
            A series directory that is no longer watched is still checked
            (its modification time, at each rescan) for this period (s).
            If files are added, e.g., the second acquisition of a FLAIR
            written more than active_timeout after the first, the
            directory is watched again and the new files are sent.
        connect_timeout : float
            Socket timeout (s) for connecting and sending.
        retry_interval : float
            Interval (s) to try to connect to the receiver while a series
            is waiting for it or files could not be sent.
        dry_run : bool
            Do not send files. Log the detection counts and delays.
        """
        self._logger = logging.getLogger('GeDicomSender')
        self.image_root = image_root
        self.host = host
        self.port = port
        self.poll_interval = poll_interval
        self.rescan_interval = rescan_interval
        self.full_rescan_interval = full_rescan_interval
        self.active_timeout = active_timeout
        self.dormant_timeout = dormant_timeout
        self.connect_timeout = connect_timeout
        self.retry_interval = retry_interval
        self.dry_run = dry_run

        self._sock = None
        self._next_connect = 0.0
        self._conn_failure_logged = False
        self._cancel = False
        # Active series whose files are not queued because the receiver was
        # not available
        self._waiting = set()

        self._inotify = None
        if use_inotify:
            try:
                self._inotify = Inotify()
            except (OSError, AttributeError):
                e = sys.exc_info()[1]
                self._logger.error('inotify is not available (%s). '
                                   'Use polling.' % e)

        # Series directories already present at start (not sent)
        self._known_series = set()
        # Active series directories {path: last_new_file_time}
        self._active = {}
        # Size of the files sent {series_dir: {name: size}}
        self._sent = {}
        # Files to send once the size is stable {path: size}
        self._pending = {}
        # Modification time of a series directory at its last listing
        self._listed = {}
        # Time of the last file event in a series directory
        self._last_event = {}
        # Dry run statistics {series_dir: [n_files, sum_delay, max_delay]}
        self._stats = {}
        self._next_report = 0.0
        # Files to send immediately (closed after writing)
        self._ready = []
        # Series directories no longer watched, still checked for new files
        # {path: (time of deactivation, directory mtime, {name: size} sent)}
        self._dormant = {}

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _listdir(self, path, prefix):
        try:
            return [d for d in os.listdir(path) if d.startswith(prefix)]
        except OSError:
            return []

    def _latest(self, path, prefix):
        """The subdirectory with the largest number (p* or e*), or None."""
        best, best_n = None, -1
        for d in self._listdir(path, prefix):
            ma = re.match(r'^%s(\d+)$' % prefix, d)
            if ma and int(ma.group(1)) > best_n:
                full = os.path.join(path, d)
                if os.path.isdir(full):
                    best, best_n = full, int(ma.group(1))
        return best

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _list_series_dirs(self):
        """Return the list of all p*/e*/s* directories."""
        series_dirs = []
        for p_d in self._listdir(self.image_root, 'p'):
            p_path = os.path.join(self.image_root, p_d)
            for e_d in self._listdir(p_path, 'e'):
                e_path = os.path.join(p_path, e_d)
                for s_d in self._listdir(e_path, 's'):
                    s_path = os.path.join(e_path, s_d)
                    if os.path.isdir(s_path):
                        series_dirs.append(s_path)
        return series_dirs

    def _latest_series_dirs(self):
        """Return the s* directories of the latest e* of the latest p*."""
        p_path = self._latest(self.image_root, 'p')
        if p_path is None:
            return []
        e_path = self._latest(p_path, 'e')
        if e_path is None:
            return []
        out = []
        for s_d in self._listdir(e_path, 's'):
            s_path = os.path.join(e_path, s_d)
            if os.path.isdir(s_path):
                out.append(s_path)
        return out

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _watch_latest(self):
        """Watch the image pool and the latest p* and e* directories for new
        directories. Older exams are covered by the full rescan."""
        mask = IN_CREATE | IN_MOVED_TO
        self._inotify.add_watch(self.image_root, mask)
        p_path = self._latest(self.image_root, 'p')
        if p_path is None:
            return
        self._inotify.add_watch(p_path, mask)
        e_path = self._latest(p_path, 'e')
        if e_path is not None:
            self._inotify.add_watch(e_path, mask)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def init_state(self):
        """Mark existing series directories and files as already handled.

        A series directory modified within the last minute is kept active, so
        files added after this start are still sent (e.g., the sender is
        restarted during a scan).
        """
        if self._inotify is not None:
            self._watch_latest()

        now = time.time()
        for s_path in self._list_series_dirs():
            self._known_series.add(s_path)
            try:
                mtime = os.stat(s_path).st_mtime
            except OSError:
                continue
            if now - mtime < 60:
                self._activate(s_path, mark_existing=True)
                self._logger.info('Series %s is active at start' % s_path)
        self._logger.info('%d series directories exist at start'
                          % len(self._known_series))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _activate(self, s_path, mark_existing=False, sent=None):
        """Start watching a series directory. sent: {name: size} of the
        files already sent (a series watched again)."""
        self._known_series.add(s_path)
        self._dormant.pop(s_path, None)
        self._sent[s_path] = sent is not None and sent or {}
        self._active[s_path] = time.time()
        self._last_event[s_path] = time.time()
        if self._inotify is not None:
            try:
                self._inotify.add_watch(s_path, IN_CLOSE_WRITE | IN_MOVED_TO)
            except OSError:
                e = sys.exc_info()[1]
                self._logger.error(str(e))

        if mark_existing:
            # Files written before this start are not sent
            self._list_series(s_path, mark_existing=True)
        if self.dry_run:
            if not mark_existing:
                self._list_series(s_path)
            return

        # Check the receiver at each new series: the old connection may be
        # stale after a receiver restart.
        self._close()
        self._next_connect = 0.0
        if self._connect():
            if not mark_existing:
                # Files written before the watch was added
                self._list_series(s_path)
        else:
            self._waiting.add(s_path)
            self._logger.warning(
                'Series %s is not sent now (receiver not available). '
                'Retry every %.0f s while the series is active.'
                % (s_path, self.retry_interval))

    def _retry_waiting(self):
        """Connect for the waiting series. When connected, queue all their
        files written so far."""
        if not self._waiting or not self._connect(log_failure=False):
            return
        for s_path in sorted(self._waiting):
            self._logger.info('Receiver available. Sending series %s from '
                              'its start' % s_path)
            self._list_series(s_path)
        self._waiting = set()

    def _list_series(self, s_path, mark_existing=False):
        """List a series directory and queue the files not sent yet."""
        try:
            self._listed[s_path] = os.stat(s_path).st_mtime
            names = os.listdir(s_path)
        except OSError:
            return
        sent = self._sent[s_path]
        for name in names:
            if name in sent or not DICOM_PAT.match(name):
                continue
            fpath = os.path.join(s_path, name)
            if mark_existing:
                try:
                    sent[name] = os.stat(fpath).st_size
                except OSError:
                    pass
            else:
                self._pending.setdefault(fpath, -1)

    def _touch_waiting(self, s_path):
        """Keep a waiting series active while files are written into it."""
        try:
            mtime = os.stat(s_path).st_mtime
        except OSError:
            return
        if mtime > self._listed.get(s_path, 0):
            self._listed[s_path] = mtime
            self._active[s_path] = time.time()

    def _dir_changed(self, s_path, since):
        """True if the directory was modified after 'since'."""
        try:
            return os.stat(s_path).st_mtime > since
        except OSError:
            return False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _deactivate(self, s_path):
        self._logger.info('Series %s is inactive' % s_path)
        del self._active[s_path]
        sent = self._sent.pop(s_path)
        if s_path not in self._waiting:
            # A waiting series is sent from its start when it is watched
            # again, so its sent record is not kept
            try:
                mtime = os.stat(s_path).st_mtime
                self._dormant[s_path] = (time.time(), mtime, sent)
            except OSError:
                pass
        self._waiting.discard(s_path)
        self._listed.pop(s_path, None)
        self._last_event.pop(s_path, None)
        if self._inotify is not None:
            self._inotify.rm_watch(s_path)
        for fpath in list(self._pending.keys()):
            if os.path.dirname(fpath) == s_path:
                del self._pending[fpath]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def rescan(self, full=False):
        """Find new series directories and files missed by inotify.

        full: list all p*/e*/s* directories. Otherwise only the latest exam
        is listed, and an active series directory is listed only if a file
        was created after the last inotify event (one stat per series).
        """
        now = time.time()
        if full:
            if self._inotify is not None:
                self._watch_latest()
            series_dirs = self._list_series_dirs()
        else:
            series_dirs = self._latest_series_dirs()
        for s_path in series_dirs:
            if s_path not in self._known_series:
                self._logger.info('New series %s%s'
                                  % (s_path, full and ' (full rescan)' or ''))
                self._activate(s_path)

        # Series no longer watched: watch again if files were added
        for s_path, (since, mtime, sent) in list(self._dormant.items()):
            if now - since > self.dormant_timeout:
                del self._dormant[s_path]
                continue
            try:
                changed = os.stat(s_path).st_mtime != mtime
            except OSError:
                del self._dormant[s_path]
                continue
            if changed:
                self._logger.info('Series %s has new files. Watching it '
                                  'again (%d files sent before)'
                                  % (s_path, len(sent)))
                self._activate(s_path, sent=sent)

        for s_path in list(self._active.keys()):
            if now - self._active[s_path] > self.active_timeout:
                self._deactivate(s_path)
                continue
            if s_path in self._waiting:
                self._touch_waiting(s_path)
                continue
            if self._inotify is None:
                continue
            # A file created more than 1 s after the last event may have
            # been missed (its close event would normally follow at once).
            since = max(self._listed.get(s_path, 0),
                        self._last_event.get(s_path, 0) + 1.0)
            if full or self._dir_changed(s_path, since):
                self._list_series(s_path)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def handle_events(self, timeout):
        """Read inotify events and queue the closed files."""
        for d_path, mask, name in self._inotify.read_events(timeout):
            if mask & IN_Q_OVERFLOW:
                self._logger.error('inotify queue overflow. Rescanning.')
                self._next_full_rescan = 0
                continue
            if d_path is None:
                continue
            path = os.path.join(d_path, name)
            if mask & IN_ISDIR:
                # A new p*, e*, or s* directory
                depth = len(os.path.relpath(path, self.image_root).split(
                    os.sep))
                self._new_dir(path, depth)
            elif mask & (IN_CLOSE_WRITE | IN_MOVED_TO):
                if d_path in self._waiting:
                    # Sent from the listing when the receiver is available
                    self._active[d_path] = time.time()
                elif d_path in self._active and DICOM_PAT.match(name):
                    self._ready.append(path)
                    self._pending.pop(path, None)
                    self._last_event[d_path] = time.time()

    def _new_dir(self, path, depth):
        """Watch a new directory (depth 1: p*, 2: e*, 3: s*)."""
        if depth < 3:
            # A new p* or e* directory becomes the latest one. Also handle
            # subdirectories created before the watch was added.
            try:
                self._inotify.add_watch(path, IN_CREATE | IN_MOVED_TO)
            except OSError:
                pass
            for sub in self._listdir(path, depth == 1 and 'e' or 's'):
                sub_path = os.path.join(path, sub)
                if os.path.isdir(sub_path):
                    self._new_dir(sub_path, depth + 1)
        elif depth == 3 and path not in self._known_series:
            self._logger.info('New series %s' % path)
            self._activate(path)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def poll_files(self):
        """Polling mode: list an active series directory only when its
        modification time has changed."""
        for s_path in list(self._active.keys()):
            if s_path in self._waiting:
                continue
            if self._dir_changed(s_path, self._listed.get(s_path, 0)):
                self._list_series(s_path)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def check_pending(self):
        """Queue pending files whose size is unchanged since the last check.
        """
        for fpath, prev_size in list(self._pending.items()):
            try:
                size = os.stat(fpath).st_size
            except OSError:
                del self._pending[fpath]
                continue
            if size > 0 and size == prev_size:
                del self._pending[fpath]
                self._ready.append(fpath)
            else:
                self._pending[fpath] = size

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def send_ready(self):
        """Send the queued files in the order of the number suffix."""
        def suffix(fpath):
            return int(DICOM_PAT.match(os.path.basename(fpath)).group(1))

        ready = sorted(set(self._ready), key=suffix)
        self._ready = []
        for i, fpath in enumerate(ready):
            s_path, name = os.path.split(fpath)
            sent = self._sent.get(s_path)
            if sent is None:
                continue
            try:
                size = os.stat(fpath).st_size
            except OSError:
                continue
            if sent.get(name) == size:
                # Already sent with the same size
                continue
            if self.dry_run:
                self._count(s_path, fpath)
                sent[name] = size
                self._active[s_path] = time.time()
                continue
            if not self.send_file(fpath):
                # Retry later
                self._ready.extend(ready[i:])
                return
            sent[name] = size
            self._active[s_path] = time.time()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _count(self, s_path, fpath):
        """Dry run: record the detection delay of a file."""
        try:
            delay = time.time() - os.stat(fpath).st_mtime
        except OSError:
            return
        st = self._stats.setdefault(s_path, [0, 0.0, 0.0])
        st[0] += 1
        st[1] += delay
        st[2] = max(st[2], delay)

    def report(self):
        """Dry run: log the files found since the last report."""
        for s_path, st in sorted(self._stats.items()):
            self._logger.info(
                'DRY RUN %s: %d files, detection delay mean %.1f ms, '
                'max %.1f ms' % (os.path.relpath(s_path, self.image_root),
                                 st[0], st[1] / st[0] * 1000, st[2] * 1000))
        self._stats = {}

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _connect(self, log_failure=True):
        """Connect to the receiver. A failure is logged once until the next
        success; the next try is after retry_interval."""
        if self._sock is not None:
            return True
        if time.time() < self._next_connect:
            return False
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.connect_timeout)
            sock.connect((self.host, self.port))
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._sock = sock
            if self._conn_failure_logged:
                self._logger.info('Connected to %s:%d'
                                  % (self.host, self.port))
            else:
                self._logger.debug('Connected to %s:%d'
                                   % (self.host, self.port))
            self._conn_failure_logged = False
            return True
        except (socket.error, socket.timeout):
            if log_failure and not self._conn_failure_logged:
                self._logger.warning('Receiver %s:%d is not available'
                                     % (self.host, self.port))
                self._conn_failure_logged = True
            self._next_connect = time.time() + self.retry_interval
            return False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except socket.error:
                pass
        self._sock = None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _recv_exact(self, n):
        buf = ''.encode('ascii')
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise socket.error('Connection closed by the receiver')
            buf += chunk
        return buf

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def send_file(self, fpath):
        """Send a file. Return True if the receiver acknowledged it."""
        try:
            fd = open(fpath, 'rb')
            try:
                data = fd.read()
            finally:
                fd.close()
        except IOError:
            self._logger.error('Failed to read %s' % fpath)
            return True  # Do not retry

        name = os.path.relpath(fpath, self.image_root)
        name = name.replace(os.sep, '/').encode('utf-8')

        if not self._connect():
            return False
        try:
            st = time.time()
            self._sock.sendall(MAGIC + struct.pack('>HI', len(name),
                                                   len(data)))
            self._sock.sendall(name)
            self._sock.sendall(data)
            ack = self._recv_exact(1)
        except (socket.error, socket.timeout):
            e = sys.exc_info()[1]
            self._logger.error('Failed to send %s: %s. Retry every %.0f s.'
                               % (fpath, e, self.retry_interval))
            self._conn_failure_logged = True
            self._close()
            self._next_connect = time.time() + self.retry_interval
            return False

        if ack != 'K'.encode('ascii'):
            # Rejected by the receiver. Do not retry.
            self._logger.error('Receiver rejected %s' % fpath)
            return True

        self._logger.debug('Sent %s (%d bytes, %.1f ms)'
                           % (name, len(data), (time.time() - st) * 1000))
        return True

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run(self):
        if self._inotify is not None:
            mode = 'inotify'
        else:
            mode = 'polling'
        self._logger.info('Start watching %s (%s), sending to %s:%d'
                          % (self.image_root, mode, self.host, self.port))
        if self.dry_run:
            self._logger.info('Dry run: files are not sent')
        # Wait for the image pool (e.g., not mounted yet after a reboot)
        if not os.path.isdir(self.image_root):
            self._logger.warning('%s does not exist. Waiting for it.'
                                 % self.image_root)
            while not os.path.isdir(self.image_root):
                if self._cancel:
                    return
                try:
                    time.sleep(1.0)
                except KeyboardInterrupt:
                    return
            self._logger.info('%s exists now' % self.image_root)
        st = time.time()
        self.init_state()
        self._logger.info('Initial listing took %.1f ms'
                          % ((time.time() - st) * 1000))
        if not self.dry_run:
            self._connect()
        self._next_rescan = 0.0
        self._next_full_rescan = time.time() + self.full_rescan_interval
        while not self._cancel:
            try:
                now = time.time()
                if now >= self._next_full_rescan:
                    st = time.time()
                    self.rescan(full=True)
                    self._logger.debug('Full rescan took %.1f ms'
                                       % ((time.time() - st) * 1000))
                    self._next_full_rescan = now + self.full_rescan_interval
                    self._next_rescan = now + self.rescan_interval
                elif now >= self._next_rescan:
                    self.rescan()
                    if self._inotify is not None:
                        self._next_rescan = now + self.rescan_interval
                    else:
                        self._next_rescan = now + 1.0
                if self.dry_run and now >= self._next_report:
                    self.report()
                    self._next_report = now + 10.0
                if not self.dry_run and self._waiting:
                    self._retry_waiting()

                if self._inotify is not None:
                    if self._pending:
                        timeout = self.poll_interval
                    else:
                        timeout = 0.5
                    self.handle_events(timeout)
                else:
                    self.poll_files()
                    time.sleep(self.poll_interval)

                self.check_pending()
                self.send_ready()

            except KeyboardInterrupt:
                break
            except Exception:
                self._logger.exception('Error in the main loop')
                time.sleep(1.0)

        self._close()
        if self._inotify is not None:
            self._inotify.close()
        self._logger.info('Exit')

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def stop(self, *args):
        self._cancel = True


# %% Running all the time =====================================================
LOG_MAX_BYTES = 5 * 2**20
LOG_BACKUPS = 3
PID_NAME = 'ge_dicom_sender.pid'
ENSURE_LOCK_NAME = 'ge_dicom_sender.ensure.lock'
STDERR_NAME = 'ge_dicom_sender.stderr'


def _read_cmdline(pid):
    """Argument vector of a process, or None."""
    try:
        fid = open('/proc/%d/cmdline' % pid, 'rb')
        try:
            data = fid.read()
        finally:
            fid.close()
    except (IOError, OSError):
        return None
    out = []
    for a in data.split(b'\0')[:-1]:
        if not isinstance(a, str):
            a = a.decode('utf-8', 'replace')
        out.append(a)
    return out


def _lock_holder(pid_file):
    """PID of the sender holding the lock on pid_file, 0 if the lock is
    held but the PID is not written yet, or None if no sender runs."""
    if not os.path.exists(pid_file):
        return None
    fid = open(pid_file, 'r')
    try:
        try:
            fcntl.flock(fid, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except IOError:
            txt = fid.read().strip()
            if txt.isdigit():
                return int(txt)
            return 0
        fcntl.flock(fid, fcntl.LOCK_UN)
        return None
    finally:
        fid.close()


def hold_pid_lock(log_dir):
    """Lock the PID file for the lifetime of this process and write the
    PID. Returns the open file, or None if another sender holds it."""
    logger = logging.getLogger('GeDicomSender')
    pid_file = os.path.join(log_dir, PID_NAME)
    fid = open(pid_file, 'a+')
    try:
        fcntl.flock(fid, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        fid.seek(0)
        logger.error('Another sender is running (pid %s, %s). Exit.'
                     % (fid.read().strip(), pid_file))
        fid.close()
        return None
    fid.seek(0)
    fid.truncate(0)
    fid.write('%d\n' % os.getpid())
    fid.flush()
    return fid


def ensure_running(log_dir, argv, start_timeout=10.0, stop_timeout=15.0):
    """Start the sender with argv (the options) if it is not running."""
    logger = logging.getLogger('GeDicomSender')
    want = [sys.executable, os.path.abspath(sys.argv[0])] + list(argv)
    pid_file = os.path.join(log_dir, PID_NAME)

    # Serialize concurrent calls (e.g., @reboot and the minute job)
    lock = open(os.path.join(log_dir, ENSURE_LOCK_NAME), 'a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        return 0

    pid = _lock_holder(pid_file)
    if pid == 0:
        return 0  # Starting now
    if pid is not None:
        cmd = _read_cmdline(pid)
        if cmd == want:
            return 0  # Running
        logger.info('The running sender (pid %d) has other arguments: %s. '
                    'Restart with: %s' % (pid, cmd, want))
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
        end_t = time.time() + stop_timeout
        while _lock_holder(pid_file) is not None:
            if time.time() > end_t:
                logger.error('The sender (pid %d) did not stop in %.0f s'
                             % (pid, stop_timeout))
                return 1
            time.sleep(0.2)

    # Start detached: new session, no terminal, no inherited files
    err_file = os.path.join(log_dir, STDERR_NAME)
    if os.path.exists(err_file) and os.path.getsize(err_file) > 2**20:
        os.rename(err_file, err_file + '.1')
    err = open(err_file, 'a')
    devnull = open(os.devnull, 'r')
    try:
        proc = subprocess.Popen(want, stdin=devnull, stdout=err,
                                stderr=err, close_fds=True,
                                preexec_fn=os.setsid, cwd='/')
    finally:
        err.close()
        devnull.close()

    end_t = time.time() + start_timeout
    while time.time() < end_t:
        if proc.poll() is not None:
            logger.error('The sender exited at start (code %s). See %s'
                         % (proc.returncode, err_file))
            return 1
        if _lock_holder(pid_file) == proc.pid:
            logger.info('Started the sender (pid %d): %s'
                        % (proc.pid, ' '.join(want)))
            return 0
        time.sleep(0.2)
    logger.error('The sender (pid %d) did not take the lock in %.0f s'
                 % (proc.pid, start_timeout))
    return 1


# %% __main__ =================================================================
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--host', default=None,
                      help='IP address of the RTPSpy computer (receiver); '
                      'required unless --dry_run')
    parser.add_option('--port', type='int', default=8322,
                      help='rt_dicom_receiver port')
    parser.add_option('--image_root',
                      default='/export/home1/sdc_image_pool/images',
                      help='GE image pool directory')
    parser.add_option('--polling', action='store_true', default=False,
                      help='Use polling instead of inotify')
    parser.add_option('--poll_interval', type='float', default=0.05,
                      help='File polling interval (s)')
    parser.add_option('--active_timeout', type='float', default=60.0,
                      help='Stop watching a series with no new file for this '
                      'period (s)')
    parser.add_option('--dormant_timeout', type='float', default=1800.0,
                      help='After that, check the series for new files for '
                      'this period (s) and watch it again if any')
    parser.add_option('--dry_run', action='store_true', default=False,
                      help='Do not send files. Log the detection delay.')
    parser.add_option('--log_file', default=None,
                      help='Log file (rotated at %d MB). Default: stderr'
                      % (LOG_MAX_BYTES // 2**20))
    parser.add_option('--ensure_running', action='store_true',
                      default=False,
                      help='Start the sender detached if it is not running, '
                      'and exit (for cron). Requires --log_file.')
    parser.add_option('--debug', action='store_true', default=False)
    opts, args = parser.parse_args()

    if opts.host is None and not opts.dry_run:
        parser.error('--host is required (the IP address of the RTPSpy '
                     'computer)')

    if opts.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    log_format = '%(asctime)s,[%(levelname)s],%(name)s,%(message)s'
    if opts.log_file:
        log_file = os.path.abspath(os.path.expanduser(opts.log_file))
        log_dir = os.path.dirname(log_file)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS)
        handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(level)
    else:
        log_dir = None
        logging.basicConfig(level=level, format=log_format)

    if opts.ensure_running:
        if log_dir is None:
            parser.error('--ensure_running requires --log_file')
        argv = [a for a in sys.argv[1:] if a != '--ensure_running']
        sys.exit(ensure_running(log_dir, argv))

    pid_lock = None
    if log_dir is not None:
        pid_lock = hold_pid_lock(log_dir)
        if pid_lock is None:
            sys.exit(1)

    sender = GeDicomSender(opts.image_root, opts.host, opts.port,
                           use_inotify=not opts.polling,
                           poll_interval=opts.poll_interval,
                           active_timeout=opts.active_timeout,
                           dormant_timeout=opts.dormant_timeout,
                           dry_run=opts.dry_run)
    signal.signal(signal.SIGTERM, sender.stop)
    sender.run()
    if pid_lock is not None:
        pid_lock.truncate(0)  # The lock is released at exit

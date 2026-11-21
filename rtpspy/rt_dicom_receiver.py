#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Receive DICOM files sent by ge_dicom_sender.py on the GE MR console.

A file 'p*/e*/s*/i*.MRDC.N' is written as
'{watch_dir}/p*/e*/s*/i*.MRDC.N.dcm'. It is first written to a hidden
temporary file and then renamed, so a partial file is never seen.

RtpWatch runs DicomReceiveServer in a thread when its 'ge_port' is set,
writing into its watch directory, where the watchdog observer finds the
files. This script can also run standalone to write the files into any
directory:
    python -m rtpspy.rt_dicom_receiver --watch_dir <dir> --port 8322

mmisaki@laureateinstitute.org
"""

# %% import ===================================================================
import argparse
from pathlib import Path
import logging
import os
import socketserver
import struct
from datetime import datetime

MAGIC = b'GEDC'
HEADER_FMT = '>HI'
HEADER_LEN = struct.calcsize(HEADER_FMT)
MAX_FILE_SIZE = 256 * 1024 * 1024


# %% class DicomReceiveHandler ================================================
class DicomReceiveHandler(socketserver.BaseRequestHandler):
    """Handle one sender connection. Receive files until it is closed."""

    def _recv_exact(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.request.recv(min(n - len(buf), 1024 * 1024))
            if not chunk:
                return None
            buf += chunk
        return bytes(buf)

    def _safe_path(self, name):
        """Return the output path, or None if the name is not acceptable."""
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts or len(rel.parts) == 0:
            return None
        out_f = self.server.watch_dir / rel.parent / (rel.name + '.dcm')
        return out_f

    def handle(self):
        logger = self.server.logger
        peer = self.client_address[0]
        if self.server.allow_hosts and peer not in self.server.allow_hosts:
            logger.error(f"Rejected a connection from {peer}")
            return

        logger.info(f"Connected from {peer}")
        while True:
            head = self._recv_exact(len(MAGIC) + HEADER_LEN)
            if head is None:
                break
            if head[:len(MAGIC)] != MAGIC:
                logger.error(f"Invalid header from {peer}. Closing.")
                break
            name_len, data_len = struct.unpack(HEADER_FMT,
                                               head[len(MAGIC):])
            if data_len > MAX_FILE_SIZE:
                logger.error(f"File too large ({data_len} bytes). Closing.")
                break

            name = self._recv_exact(name_len)
            data = self._recv_exact(data_len)
            if name is None or data is None:
                break
            name = name.decode('utf-8')

            out_f = self._safe_path(name)
            if out_f is None:
                logger.error(f"Invalid file name {name}")
                self.request.sendall(b'E')
                continue

            try:
                out_f.parent.mkdir(parents=True, exist_ok=True)
                # The temporary name must not contain '.dcm' because the
                # watcher matches its file pattern with re.search.
                tmp_f = out_f.parent / ('.' + Path(name).name + '.part')
                with open(tmp_f, 'wb') as fd:
                    fd.write(data)
                os.replace(tmp_f, out_f)
            except OSError as e:
                logger.error(f"Failed to write {out_f}: {e}")
                self.request.sendall(b'E')
                continue

            self.request.sendall(b'K')
            logger.debug(f"Received {name} ({data_len} bytes)")

            if self.server.callback is not None:
                try:
                    self.server.callback(out_f, data)
                except Exception:
                    logger.exception(f"Error in processing {name}")

        logger.info(f"Disconnected from {peer}")


# %% class DicomReceiveServer =================================================
class DicomReceiveServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, watch_dir, allow_hosts=None, callback=None):
        """
        Parameters
        ----------
        address : tuple
            (host, port) to listen.
        watch_dir : Path or str
            Directory to write the received files.
        allow_hosts : list of str, optional
            Accept connections only from these hosts.
        callback : function, optional
            Called as callback(file_path, data) after each file is written.
            Files from a connection are passed in the received order.
        """
        super().__init__(address, DicomReceiveHandler)
        self.watch_dir = Path(watch_dir)
        self.allow_hosts = allow_hosts
        self.callback = callback
        self.logger = logging.getLogger('RtDicomReceiver')


# %% __main__ =================================================================
if __name__ == '__main__':
    dstr = datetime.now().strftime("%Y%m%d")
    LOG_FILE = Path(f"log/RtDicomReceiver_{dstr}.log")

    parser = argparse.ArgumentParser(description="GE DICOM receiver")
    parser.add_argument('--watch_dir', required=True,
                        help='Directory to write the received files')
    parser.add_argument('--port', type=int, default=8322,
                        help='Listening port')
    parser.add_argument('--allow_hosts', nargs='*',
                        help='Accept connections only from these hosts')
    parser.add_argument('--log_file', default=LOG_FILE, help='Log file path')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    log_format = "%(asctime)s.%(msecs)03d,[%(levelname)s],%(name)s,%(message)s"
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format=log_format,
                            datefmt="%Y-%m-%dT%H:%M:%S")
    else:
        log_file = Path(args.log_file)
        if not log_file.parent.is_dir():
            os.makedirs(log_file.parent)
        logging.basicConfig(level=logging.INFO, filename=log_file,
                            filemode="a", format=log_format,
                            datefmt="%Y-%m-%dT%H:%M:%S")

    server = DicomReceiveServer(('', args.port), args.watch_dir,
                                allow_hosts=args.allow_hosts)
    server.logger.info(f"Receiving on port {args.port}, "
                       f"writing to {args.watch_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.logger.info("Closed RtDicomReceiver")

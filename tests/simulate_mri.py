#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate DICOM file creation and physio signasl feeding

@author: Masaya Misaki, mmisaki@laureateinstitute.org
"""


# %% import ===================================================================
from pathlib import Path
import argparse
import shutil
import time
import os
import sys
import re
from datetime import datetime
import pickle

import numpy as np
from dicom_reader import DicomReader
from tqdm import tqdm

from serial.tools.list_ports import comports
from simulate_physio_biopac import SendTTL

if '__file__' not in locals():
    __file__ = 'simulate_mri.py'


# %% RtMRISimulator ===========================================================
class RtMRISimulator():
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, physio_recording_rate_ms=2):
        # Setup
        self.dcm_reader = DicomReader()
        self.physio_recording_rate_ms = physio_recording_rate_ms
        self.set_physio_ports()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def read_test_data(self, test_data_dir, dicom_file_pat, overwrite=False):
        '''recusive call read_dir'''
        # ---------------------------------------------------------------------
        def read_dir(parent_dir, dicom_file_pat, level):
            dcm_fs = sorted([ff for ff in parent_dir.glob('**/*')
                             if ff.is_file() and
                             re.match(dicom_file_pat[level], ff.name)])
            if len(dcm_fs):
                ser = {}
                acq_times = {}
                content_times = {}
                file_times = {}
                for ii, ff in tqdm(enumerate(dcm_fs), total=len(dcm_fs),
                                   desc=f'read files in {parent_dir.name}'):
                    scan_info = self.dcm_reader.read_dicom_info(ff,
                                                                timeout=0.1)
                    if scan_info is None:
                        # Not a DICOM file
                        continue
                    ser[ii] = scan_info['SeriesNumber']
                    dtstr = scan_info['Scan date'] + ' ' + \
                        scan_info['ContentTime']
                    ct = datetime.strptime(dtstr, '%a %b %d %Y %H%M%S.%f')
                    content_times[ii] = ct.timestamp()
                    file_times[ii] = ff.stat().st_mtime
                    dtstr = scan_info['Scan date'] + ' ' + \
                        scan_info['AcquisitionTime']
                    at = datetime.strptime(dtstr, '%a %b %d %Y %H%M%S.%f')
                    acq_times[ii] = at

                if len(ser) == 0:
                    return

                # sort with content_time
                t_order = np.argsort(list(content_times.values())).ravel()
                sidx = np.array(list(content_times.keys()))[list(t_order)]
                dcm_fs = np.array(dcm_fs)[sidx]
                ctimes = np.array([content_times[idx] for idx in sidx])
                ftimes = np.array([file_times[idx] for idx in sidx])
                atimes = np.array([acq_times[idx] for idx in sidx])
                series = np.array([ser[idx] for idx in sidx])

                return dcm_fs, ctimes, ftimes, atimes, series

            sess_dirs = [dd for dd in parent_dir.glob('*') if dd.is_dir() and
                         re.match(dicom_file_pat[level], dd.name)]
            read_data = []
            ctimes = []
            ftimes = []
            atimes = []
            series = []
            level += 1
            for dd in sess_dirs:
                ret = read_dir(dd, dicom_file_pat, level)
                if ret is None:
                    continue
                data, ct, ft, at, ser = ret
                if len(data):
                    read_data.extend(data)
                    ctimes.extend(ct)
                    ftimes.extend(ft)
                    atimes.extend(at)
                    series.extend(ser)

            return read_data, ctimes, ftimes, atimes, series

        # ---------------------------------------------------------------------
        # Read test data
        data_info_f = test_data_dir.parent / (test_data_dir.name + '_info.pkl')
        if data_info_f.is_file() and not overwrite:
            with open(data_info_f, 'rb') as fd:
                saved_data = pickle.load(fd)
            src_data = saved_data['src_data']
            ctimes = saved_data['ctimes']
            ftimes = saved_data['ftimes']
            atimes = saved_data['atimes']
            series = saved_data['series']
        else:
            level = 0
            ret = read_dir(test_data_dir, dicom_file_pat, level)
            if ret is None:
                return

            src_data, ctimes, ftimes, atimes, series = ret

            # Sort by ctimes
            src_data = [f.relative_to(test_data_dir.parent) for f in src_data]
            src_data = np.array(src_data)
            ctimes = np.array(ctimes)
            sidx = np.argsort(ctimes)
            src_data = src_data[sidx]
            ctimes = ctimes[sidx]
            ftimes = ftimes[sidx]
            atimes = atimes[sidx]
            series = series[sidx]
            with open(data_info_f, 'wb') as fd:
                saved_data = {'src_data': src_data,
                              'ctimes': ctimes,
                              'ftimes': ftimes,
                              'atimes': atimes,
                              'series': series}
                pickle.dump(saved_data, fd)

        dctimes = ctimes - np.min(ctimes)
        dftimes = ftimes - np.min(ftimes)
        datimes = atimes - np.min(atimes)

        return src_data, dctimes, dftimes, datimes, series

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_physio_ports(self):
        """ Setup Physio recording
        """

        # --- Set devices -----------------------------------------------------
        # Select TTL port
        SUPPORT_DEVS = ['CDC RS-232 Emulation Demo',
                        'Numato Lab 8 Channel USB GPIO',
                        '.*']
        ser_ports = {}
        for pt in comports():
            if pt.description == 'n/a' and pt.device != '/dev/ttyS4':
                continue

            if np.any([re.match(devpat, pt.description) is not None
                       for devpat in SUPPORT_DEVS]):
                ser_ports[pt.device] = pt.description

        default_sport_ttl = '1'
        msg_txt = '\n' + '=' * 80 + '\n'
        msg_txt += "Select TTL port\n0)exit"
        for ii, (sport, lab) in enumerate(ser_ports.items()):
            msg_txt += f"\n{ii+1}){sport} ({lab})"

        msg_txt += f"\n{ii+2})None"
        none_i = ii+2
        msg_txt += f'\n[{default_sport_ttl}]: '
        while True:
            sport_ttl_i = input(msg_txt)
            if sport_ttl_i == '':
                sport_ttl_i = default_sport_ttl
            try:
                sport_ttl_i = int(sport_ttl_i)
                assert sport_ttl_i <= none_i
            except Exception:
                continue
            break
        if sport_ttl_i == '0':
            return -1
        elif sport_ttl_i == none_i:
            self.sport_ttl = None
        else:
            self.sport_ttl = list(ser_ports.keys())[sport_ttl_i-1]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def select_physio_data(self, physio_data_dir):
        default_phydata_i = '1'
        phys_fs = [
            f.name.replace('ECG_', '') for f in physio_data_dir.glob('ECG*.1D')
            if (physio_data_dir / f.name.replace('ECG', 'Resp')).is_file()]

        msg_txt = '\n' + '=' * 80 + '\n'
        msg_txt += "Select data\n0)exit"
        for ii, physf in enumerate(phys_fs):
            msg_txt += f"\n{ii+1}){physf}"
        msg_txt += f'\n[{default_phydata_i}]: '
        while True:
            phydata_i = input(msg_txt)
            if phydata_i == '':
                phydata_i = default_phydata_i
            try:
                phydata_i = int(phydata_i)
                assert phydata_i <= len(phys_fs)
            except Exception:
                continue
            break
        if phydata_i == '0':
            sys.exit()

        physf = phys_fs[phydata_i-1]
        ecg_f = physio_data_dir / ('ECG_' + physf)
        assert ecg_f.is_file()
        resp_f = physio_data_dir / ('Resp_' + physf)
        assert resp_f.is_file()

        return ecg_f, resp_f

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ask_selection(self, sel_list, default_sel='1', indent='',
                      full_automatic=False):
        # Select data
        max_len = np.max([len(sel) for sel in sel_list])
        msg_txt = indent + "Select data"
        if max_len > 10:
            msg_txt += '\n' + indent
        msg_txt += ' 0)exit'
        if max_len > 10:
            msg_txt += '\n'
        for ii, pat in enumerate(sel_list):
            msg_txt += f" {ii+1}){pat}"
            if max_len > 10:
                msg_txt += indent + '\n'
        msg_txt += f' [{default_sel}]: '

        if not full_automatic:
            while True:
                data_i = input(msg_txt)
                if data_i == '':
                    data_i = default_sel
                try:
                    data_i = int(data_i)
                    assert data_i <= len(sel_list)
                except Exception:
                    continue
                break
        else:
            print(f"{msg_txt} {default_sel}")
            data_i = int(default_sel)

        if data_i == 0:
            return None, None

        # Update the default to the next or exit
        if data_i < len(sel_list):
            default_sel = f"{int(data_i)+1}"
        else:
            default_sel = '0'

        sel_idx = data_i-1
        indent += '  '

        return sel_idx, default_sel

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run(self, test_data_dir, dicom_file_pat, dst_root, physio_data_dir,
            full_automatic=False, auto_dir_interval=3):

        self.sim_phys = None

        # Read test directory to list the sessions
        session_list = []
        for dd in test_data_dir.glob('*'):
            if not dd.is_dir() or dd.name == 'c' or \
                    re.search(dicom_file_pat[0], dd.name) is None:
                continue
            session_list.append(dd.name)

        if len(session_list) == 0:
            print(f"No data faund in {test_data_dir}")
            return

        src_data = {}
        dctimes = {}
        dftimes = {}
        datimes = {}
        series = {}

        default_sess = '1'
        CONT_SESS_LOOP = True
        while CONT_SESS_LOOP:
            # Select session
            sel_sess_idx, default_sess = \
                self.ask_selection(session_list, default_sess, indent='',
                                   full_automatic=full_automatic)

            if sel_sess_idx is None:
                CONT_SESS_LOOP = False
                continue

            sel_session = session_list[sel_sess_idx]

            # Read session data
            if sel_session not in src_data:
                read_data = self.read_test_data(test_data_dir / sel_session,
                                                dicom_file_pat[1:])
                if read_data is None:
                    print(f'No source data in {test_data_dir / sel_session}')
                    continue

                src_data[sel_session] = read_data[0]
                dctimes[sel_session] = read_data[1]
                dftimes[sel_session] = read_data[2]
                datimes[sel_session] = read_data[3]
                series[sel_session] = read_data[4]

            if len(src_data[sel_session]) == 0:
                print(f'No source data in {test_data_dir / sel_session}')
                continue

            # Clean destination
            dst_dir = dst_root / sel_session
            if dst_dir.is_dir():
                shutil.rmtree(dst_dir)

            # Find series divisions
            ser_div = np.argwhere(np.diff(series[sel_session]) > 0).ravel()
            td = [d.total_seconds() for d in np.diff(datimes[sel_session])]
            ser_dt = np.array(td)[ser_div]
            ser_dvi = ser_div[ser_dt > 3] + 1
            ser_dvi = np.concatenate([ser_dvi, [len(series[sel_session])]])
            if len(ser_dvi) == 0:
                ser_dvi = [len(src_data[sel_session])]

            series_list = []
            ser_idx = []
            sstart = 0
            for send in ser_dvi:
                sel_idx = np.arange(sstart, send, dtype=int)
                ser_idx.append(sel_idx)
                ser_nums = np.unique(series[sel_session][sel_idx])
                ser_label = f'Ser {ser_nums} (N={len(sel_idx)})'
                for sn in ser_nums:
                    sample = src_data[sel_session][
                        series[sel_session] == sn][0]
                    scan_info = self.dcm_reader.read_dicom_info(
                        test_data_dir / sample, timeout=0.1)
                    if scan_info is None:
                        continue
                    ser_label += f"; {scan_info['SeriesDescription']}"
                series_list.append(ser_label)
                sstart = send

            if full_automatic:
                global_st_time = time.time()
                tshift = dftimes[sel_session][0]

            default_ser = '1'
            CONT_SER_LOOP = True
            while CONT_SER_LOOP:
                # Select series
                sel_ser_idx, default_ser = \
                    self.ask_selection(series_list, default_ser, indent='  ',
                                       full_automatic=full_automatic)
                if sel_ser_idx is None:
                    CONT_SER_LOOP = False
                    continue

                sel_idx = ser_idx[sel_ser_idx]
                sel_data = src_data[sel_session][sel_idx]
                sel_dftimes = dctimes[sel_session][sel_idx]

                # Run simulation
                try:
                    # Select mode (auto/manual)
                    if not full_automatic:
                        while True:
                            mode_i = input(
                                '    ' +
                                "Select mode 0)Cancel 1)auto series 2)manual" +
                                " 3)Enter full automatic [1] : ")
                            try:
                                if mode_i == '':
                                    mode_i = '1'
                                mode_i = int(mode_i)
                                if mode_i == 0:
                                    break
                                elif mode_i == 3:
                                    full_automatic = True
                                    global_st_time = time.time()
                                    tshift = sel_dftimes[0]
                                    mode = 'auto'
                                    break
                                mode = ['auto', 'man'][mode_i-1]
                            except Exception:
                                continue
                            break
                    else:
                        mode = 'auto'
                        mode_i = 3

                    if mode_i == 0:
                        break

                    if not full_automatic:
                        global_st_time = None
                        tshift = None

                    # Cleare existing files
                    for ff in sel_data:
                        dst_f = dst_root / ff
                        if dst_f.is_file():
                            dst_f.unlink()

                    # Copying file
                    sim_ttl = SendTTL(self.sport_ttl)
                    self.copy_files(
                        test_data_dir, sel_data, sel_dftimes, dst_root, mode,
                        global_st_time, tshift, sim_ttl)

                except KeyboardInterrupt:
                    print("\nCancel simulation.\n")
                    full_automatic = False
                    default_ser = str(int(default_ser) - 1)

        print('\n' + 'x' * 80)
        print("End simulation")
        if self.sim_phys is not None:
            self.sim_phys.feed_Physio('stop')

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def copy_files(self, test_data_dir, src_data, dtime, dst_root, mode,
                   global_st_time, tshift, sim_ttl=None, auto_dir_interval=3):
        try:
            if not rtmri_dir.is_dir():
                os.makedirs(rtmri_dir)

            # Get image parameters
            scan_info = self.dcm_reader.read_dicom_info(
                test_data_dir / src_data[0], root_dir=test_data_dir)
            print(f"Series De: {scan_info['SeriesDescription']}")

            # Prepare the destination directory
            dcm_dir = scan_info['DICOM directory']
            dst_dir = dst_root / dcm_dir
            if not dst_dir.parent.is_dir():
                os.makedirs(dst_dir.parent)

            # if dst_dir.is_dir():
            #     shutil.rmtree(dst_dir)

            if mode == 'auto':
                if not dst_dir.is_dir():
                    os.makedirs(dst_dir)
                    print(f"Create {dst_dir}")

                if global_st_time is not None:
                    st = global_st_time
                    cp_dtime = dtime - tshift
                    print(f"Wait until {time.ctime(cp_dtime[0]+st)}" +
                          " (Ctrl+C to break waiting)")
                else:
                    time.sleep(auto_dir_interval)
                    st = time.time()
                    cp_dtime = dtime - np.min(dtime)

                cp_dtime = [dt.total_seconds() for dt in cp_dtime]

                ttl_t = 0
                TR = np.diff(cp_dtime).mean()
                for ii, src_f in enumerate(src_data):
                    dst_f = dst_dir / src_f.name
                    while time.time()-st < cp_dtime[ii]:
                        time.sleep(0.001)

                    if sim_ttl is not None:
                        if ii == 0:
                            ttl_t = time.time()
                            sim_ttl.send_singnal()
                        else:
                            if time.time() - ttl_t > 0.5:
                                ttl_t = time.time()
                                sim_ttl.send_singnal()
                    if ii < len(cp_dtime)-1:
                        while time.time()-st < cp_dtime[ii+1]:
                            time.sleep(0.001)
                    else:
                        while time.time()-st < cp_dtime[-1]+TR:
                            time.sleep(0.001)

                    shutil.copy(test_data_dir / src_f, dst_f)
                    print(f"{ii+1}/{len(src_data)} Create {dst_f}"
                          f" ({time.time()-st:.3f})")
                    sys.stdout.flush()

            elif mode == 'man':
                if not dst_dir.is_dir():
                    r = input("Enter to create destination dir (0 to cancel)")
                    if r == '0':
                        return
                    os.makedirs(dst_dir)
                    print(f"Create {dst_dir}")

                for src_f in src_data:
                    dst_f = dst_dir / src_f.name
                    r = input(f"Enter to copy file {dst_f.name}" +
                              " (0 to cancel)")
                    if r == '0':
                        return
                    if sim_ttl is not None:
                        sim_ttl.send_singnal()
                    shutil.copy(test_data_dir / src_f, dst_f)
                    print(f"Create {dst_f.name}")

        except Exception as e:
            print(e)


# %% __main__ =================================================================
if __name__ == '__main__':

    TEST_DATA_ROOT = Path().home() / 'RTPSpy_tests'
    RTMRI_DIR_SMB = Path('/tmp/RTExport')
    PHYSIO_DATA_ROOT = Path().home() / 'RTPSpy_tests' / 'Physio'

    # --- Parse arguments -----------------------------------------------------
    parser = argparse.ArgumentParser(
        description='Simulate MRI DICOM data creation')
    parser.add_argument(
        '--test_data_dir', default=TEST_DATA_ROOT,
        help=f'Test data ditectory. Default is {TEST_DATA_ROOT}')
    parser.add_argument(
        '--rtmri_dir', default=RTMRI_DIR_SMB,
        help='Copy destination directory to simulate ' +
        f'real-time MRI. Default is {RTMRI_DIR_SMB} ')
    parser.add_argument('--keep_dst', action='store_true',
                        help='Keep destination directory data')
    parser.add_argument('--auto_dir_interval', default=3,
                        help='Directory creation interval (secodns) in the' +
                        'autorun mode')
    parser.add_argument('--full_automatic', action='store_true',
                        help='Full automatic mode')
    parser.add_argument('--physio_data', help='physio data ditectory')

    args = parser.parse_args()
    test_data_dir = Path(args.test_data_dir)
    rtmri_dir = Path(args.rtmri_dir)
    keep_dst = args.keep_dst
    auto_dir_interval = args.auto_dir_interval
    full_automatic = args.full_automatic
    physio_data_dir = args.physio_data
    if physio_data_dir is not None:
        physio_data_dir = Path(physio_data_dir)

    dicom_file_pat = [r'.+',
                      r'.+']
    rt_sim = RtMRISimulator()

    rt_sim.run(test_data_dir, dicom_file_pat, rtmri_dir, physio_data_dir,
               full_automatic, auto_dir_interval)

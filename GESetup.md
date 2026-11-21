# Real-Time Image Export from a GE Scanner

This guide sets up a GE MRI scanner to send each fMRI image to the RTPSpy computer as soon as it is reconstructed.

**How it works.** A small program, `ge_dicom_sender.py`, runs on the scanner console. It watches the console's image folder (the *image pool*) and sends each new image file to the RTPSpy computer over the network. RTPSpy receives the files into its watch directory, puts the slices of each volume together, and processes the volumes as they are completed.

```text
Scanner console                                   RTPSpy computer (Linux)
image pool  ──▶  ge_dicom_sender.py  ── TCP 8322 ──▶  receiver (WATCH module)
(one file per slice)                                    │ writes the files into
                                                        ▼
                                             watch directory/p*/e*/s*/*.dcm
                                                        │
                                             slices → volumes → processing
```

GE writes one DICOM file per slice. RTPSpy waits until all the slices of a volume have arrived, then passes the volume on. No file share is used.

The sender has been used on a GE Discovery MR750 (software DV26, console OS based on RHEL 6, Python 2.6). The RTPSpy side was tested with recorded DV26 data (standard and multiband EPI) replayed at the original timing. Other GE systems and software versions may differ; test with a phantom first.

## Contents

1. [What you need](#1-what-you-need)
2. [Set up RTPSpy to receive the images](#2-set-up-rtpspy-to-receive-the-images)
3. [Install the sender on the console](#3-install-the-sender-on-the-console)
4. [Keep the sender running (cron)](#4-keep-the-sender-running-cron)
5. [Test with a phantom scan](#5-test-with-a-phantom-scan)
6. [How RTPSpy handles the GE images](#6-how-rtpspy-handles-the-ge-images)
7. [Troubleshooting](#7-troubleshooting)
8. [Updating or removing the sender](#8-updating-or-removing-the-sender)

## 1. What you need

- **A network connection from the console to the RTPSpy computer** (TCP port 8322 by default). The network settings of the console are usually made by GE service or your MR physicist.
- **A login to the console** with a terminal (e.g., ssh as the user `sdc`) to copy and run the sender. The sender needs no root rights and no installation: it uses only the standard library of the console's Python (2.6 or later).
- **Permission from your site** to run a program on the scanner console. The sender only reads the image pool, but check your site's rules and, if needed, with GE.
- **The IP addresses** of the RTPSpy computer and of the console. The RTPSpy computer's address should be fixed, because the sender is started with it.

In the examples below, replace:

| Placeholder | Meaning |
|---|---|
| `<RTPSpy PC IP>` | IP address of the RTPSpy computer |
| `<console IP>` | IP address of the scanner console |

## 2. Set up RTPSpy to receive the images

1. **Choose the watch directory**, a local folder on the RTPSpy computer, e.g., `~/rtfMRI/watch`. (A RAM disk can be used, as described in [the Siemens guide](SiemensSetup.md#2-make-the-folder-for-the-images).)

2. **Turn on the receiver.** In the RTPSpy application, open 'RTP' → 'WATCH' and set:

   | Setting | Value |
   |---|---|
   | File type | `Dicom` |
   | File name pattern | `.+\.dcm$` (the default) |
   | GE receiver port | `8322` |
   | GE console IP | `<console IP>` (files from other computers are refused; leave it empty to accept any) |

   The receiver runs as long as the application runs, also before 'Ready', so that the reference scans (anatomy, reference EPI) are received too. Set the **Watch directory** at the top of the window as usual; the receiver writes into it.

   To set them in your application's default parameters instead:

   ```python
   rtp_params = {'WATCH': {'ge_port': 8322, 'ge_allow_hosts': '<console IP>'}, ...}
   ```

3. **Open the port in the firewall**, if a firewall is enabled (`sudo ufw status` shows "active"):

   ```bash
   sudo ufw allow from <console IP> to any port 8322 proto tcp
   ```

The application log shows `GE receiver on port 8322 writes into ...` when the receiver starts.

## 3. Install the sender on the console

The sender is `rtpspy/ge_dicom_sender.py` in the RTPSpy package. It is a single file that runs as is on the console.

1. **Copy it to the console**, from the RTPSpy computer (here the console user is `sdc`):

   ```bash
   ssh sdc@<console IP> mkdir -p rtpspy/log
   scp ~/RTPSpy/rtpspy/ge_dicom_sender.py sdc@<console IP>:rtpspy/
   ```

   Old consoles may need `-o HostKeyAlgorithms=+ssh-rsa -o PubkeyAcceptedAlgorithms=+ssh-rsa` for `ssh` and `scp`.

2. **Test it without sending** (on the console). It reports how many new files it finds and how fast:

   ```bash
   cd ~/rtpspy
   python ge_dicom_sender.py --dry_run
   ```

   Run a short scan (or wait for one); the log shows the files found. Stop it with Ctrl-C.

3. **Run it once by hand** while RTPSpy is running with the receiver on:

   ```bash
   python ge_dicom_sender.py --host <RTPSpy PC IP>
   ```

   Run a short scan, check that its files appear in the watch directory on the RTPSpy computer (in `p*/e*/s*/` folders), and stop the sender with Ctrl-C.

**Options** (`python ge_dicom_sender.py --help`):

| Option | Default | Meaning |
|---|---|---|
| `--host` | (required) | IP address of the RTPSpy computer |
| `--port` | 8322 | Port of the receiver |
| `--image_root` | `/export/home1/sdc_image_pool/images` | The image pool of the console |
| `--log_file` | (stderr) | Log file, rotated at 5 MB |
| `--polling` | off | Poll the folders instead of using inotify (if inotify does not work) |
| `--dry_run` | off | Find the files but do not send them |
| `--ensure_running` | off | Start the sender in the background if it is not running, then exit (for cron; needs `--log_file`) |

The sender can run all the time. If RTPSpy is not running, the files of the current series wait and are sent when the receiver becomes available (it tries every 5 s). A series that the sender finds already finished when it starts is not sent.

## 4. Keep the sender running (cron)

GE consoles usually have no way for a normal user to start a program at boot except cron. The `--ensure_running` option starts the sender in the background when it is not running and otherwise does nothing, so it can be called every minute.

1. **Start it once with `--ensure_running`** (one line; use your log folder and the RTPSpy computer's address):

   ```bash
   /usr/bin/python ~/rtpspy/ge_dicom_sender.py --ensure_running --log_file /export/home/sdc/rtpspy/log/ge_dicom_sender.log --host <RTPSpy PC IP>
   cat ~/rtpspy/log/ge_dicom_sender.pid
   ```

   Running the same command again must not change the PID in the `.pid` file.

2. **Add two lines to your crontab.** Save the current crontab first, and do not change the existing lines (GE and site jobs):

   ```bash
   crontab -l > ~/rtpspy/crontab_backup.txt
   crontab -e
   ```

   Add at the end (each entry on one line, with exactly the same arguments as in step 1):

   ```text
   @reboot   /usr/bin/python /export/home/sdc/rtpspy/ge_dicom_sender.py --ensure_running --log_file /export/home/sdc/rtpspy/log/ge_dicom_sender.log --host <RTPSpy PC IP> > /dev/null 2>&1
   * * * * * /usr/bin/python /export/home/sdc/rtpspy/ge_dicom_sender.py --ensure_running --log_file /export/home/sdc/rtpspy/log/ge_dicom_sender.log --host <RTPSpy PC IP> > /dev/null 2>&1
   ```

   The sender starts at boot and is restarted within about a minute if it stops. If the arguments in the crontab differ from those of the running sender, it is restarted with the new ones.

3. **Check the restart**: stop the sender with ``kill `cat ~/rtpspy/log/ge_dicom_sender.pid` ``, wait about a minute, and check that the `.pid` file has a new PID and the log has "Started the sender".

After a GE software update or reinstallation, check with `crontab -l` that the two lines are still there.

## 5. Test with a phantom scan

1. Start the RTPSpy application with the receiver on (step 2), set the masks and the processing (see the [ROI-NF example](example/ROI-NF#readme)), and press 'Ready'.
2. Run an fMRI scan of a phantom.
3. Start the processing with the scanner TTL pulse, the 'T' key, or 'Manual start' (see the [README](README.md#starting-the-scan-ttl-trigger-or-the-t-key)).
4. Check that:
   - the files arrive in the watch directory (`p*/e*/s*/i*.MRDC.*.dcm`);
   - the RTPSpy log shows `GE fMRI series N (...): ... volumes, ... slices` and a processed volume at each TR;
   - with 'Latency' checked, the 'TTL → image' delay is stable.

**Without the scanner.** The [RT-MRI simulator](Simulator.md) sends the files of a recorded GE exam to the receiver in the same way as the sender, so the RTPSpy side can be tested on any computer.

**Reference images.** The anatomy and reference EPI scans are received into the same watch directory. Convert them with 'DICOM to NIfTI (dcm2niix)', selecting the exam folder (`p*/e*`) or a series folder (`p*/e*/s*`) in the watch directory.

## 6. How RTPSpy handles the GE images

- **Only fMRI series are processed**: series with more than one volume (`NumberOfTemporalPositions` > 1) and more than one slice. The other series (localizers, anatomy) are received but not processed.
- **One series per run**: after 'Ready', the first fMRI series is processed. Files of other series that arrive during the run (e.g., late files of an earlier series) are ignored.
- **Volumes in order**: a volume is made when all its slices have arrived, and the volumes are passed on in the order they were acquired. With multiband scans, the console sometimes completes a volume before the previous one; the later volume then waits for the earlier one.
- **Missing volumes**: if a volume is still incomplete after max(30 s, 10 TR) while later volumes are complete (e.g., a lost slice file), it is skipped with a warning in the log. The volumes after it are then numbered one less than their acquisition order.
- **Multi-echo scans**: only the first echo is used.
- **'Clean watch dir at ready'** (WATCH) moves the files left in the watch directory to `dicom/` in the working directory when you press 'Ready', keeping the `p*/e*/s*` folders.

## 7. Troubleshooting

| Problem | Check |
|---|---|
| No files arrive | Is RTPSpy running with 'GE receiver port' set? Does the RTPSpy log show "GE receiver on port ..."? Is the port open in the firewall (step 2.3)? Is the sender running on the console (`cat ~/rtpspy/log/ge_dicom_sender.pid`, `ps -u sdc -o pid,args`)? Its log (`~/rtpspy/log/ge_dicom_sender.log`) shows connection errors. |
| "Rejected a connection from ..." in the RTPSpy log | The console's address is not in 'GE console IP'. |
| "Failed to start the GE receiver" | Another program (or another RTPSpy application) uses the port. Use another port on both sides, or close the other program. |
| Files arrive, but no volume is processed | Was 'Ready' pressed before the scan? Is it an fMRI series (see section 6)? The log shows "is not fMRI" for series that are not processed. |
| Volumes come late | Multiband reconstruction can take longer than the TR on the console; the delay then grows during the scan. Check 'TTL → image' in the latency plot. |
| "GE volume N is not complete ... It is skipped" | A slice file was lost or arrived too late. Check the sender log and the network. |

## 8. Updating or removing the sender

**Updating.** `--ensure_running` does not restart a running sender whose arguments are unchanged, so stop it after replacing the file, when no scan is running:

```bash
scp ~/RTPSpy/rtpspy/ge_dicom_sender.py sdc@<console IP>:rtpspy/ge_dicom_sender.py.new
```

On the console:

```bash
cd ~/rtpspy
python -c "import py_compile; py_compile.compile('ge_dicom_sender.py.new', doraise=True)" && echo OK
cp -p ge_dicom_sender.py ge_dicom_sender.py.bak
mv ge_dicom_sender.py.new ge_dicom_sender.py
kill `cat ~/rtpspy/log/ge_dicom_sender.pid`
```

Cron starts the new sender within a minute; the log shows "Exit", then "Started the sender".

**Removing.** Delete the two lines with `crontab -e`, then stop the sender: ``kill `cat ~/rtpspy/log/ge_dicom_sender.pid` ``.

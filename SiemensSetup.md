# Real-Time Image Export from a Siemens Scanner

This guide sets up a Siemens MRI scanner to send each fMRI image to the RTPSpy computer as soon as it is reconstructed.

**How it works.** The RTPSpy computer shares a folder over the network (a Windows file share, made with Samba). On the scanner console, the **BOLD AddIn** is added to the fMRI scan and writes each reconstructed volume as a DICOM file into that shared folder. RTPSpy watches the folder and processes each file as it arrives.

```text
Scanner console                     RTPSpy computer (Linux)
BOLD AddIn  ── Windows share ──▶   shared folder  (e.g., /RTMRI)
(DICOM files)  (SMB)                └─ RTExport/  ◀── RTPSpy watch directory
```

The setup was made and tested with a Siemens Prisma running syngo MR XA30. The names of the console settings can differ in other software versions.

## Contents

1. [What you need](#1-what-you-need)
2. [Make the folder for the images](#2-make-the-folder-for-the-images)
3. [Share the folder with Samba](#3-share-the-folder-with-samba)
4. [Check the share from another computer](#4-check-the-share-from-another-computer)
5. [Set up the BOLD AddIn on the console](#5-set-up-the-bold-addin-on-the-console)
6. [Set up RTPSpy](#6-set-up-rtpspy)
7. [Test with a phantom scan](#7-test-with-a-phantom-scan)
8. [Troubleshooting](#8-troubleshooting)

## 1. What you need

- **A network connection between the console and the RTPSpy computer.** The console must reach the RTPSpy computer over the network (SMB, TCP port 445). A direct cable or a dedicated network for the real-time computer is common. The network settings of the console are usually made by Siemens service or your MR physicist.
- **The BOLD AddIn on the console.** If it is not available on your scanner, ask Siemens.
- **Administrator rights (sudo) on the RTPSpy computer**, to install and set up Samba.
- **The IP address of the RTPSpy computer** (`ip -4 addr` shows it). It should be fixed, because it is written into the console settings.

In the examples below, replace:

| Placeholder | Meaning |
|---|---|
| `<RTPSpy PC IP>` | IP address of the RTPSpy computer |
| `<user>` | Linux user that runs RTPSpy; the console logs in to the share as this user |
| `/RTMRI` | The shared folder. Any location can be used |

## 2. Make the folder for the images

Make the shared folder and, inside it, the folder that receives the images (here `/RTMRI` and `/RTMRI/RTExport`). The folder can be anywhere, for example in your home folder; use the same path in all the steps below.

```bash
sudo mkdir -p /RTMRI/RTExport
sudo chown -R <user>:<user> /RTMRI
sudo chmod -R 775 /RTMRI
```

**Optional: a RAM disk for the images.** Putting the receiving folder in memory (tmpfs) makes writing and reading the files fast, and keeps the disk from filling up. The files on a RAM disk are lost when the computer shuts down, so let RTPSpy move them to the working directory ('Clean watch dir at ready' in [step 6](#6-set-up-rtpspy)).

Add this line to `/etc/fstab` (choose a size that fits the memory of the computer, e.g., 16 GB):

```text
tmpfs   /RTMRI/RTExport   tmpfs   rw,size=16G   0   0
```

Then mount it and set the owner again (mounting replaces the folder's owner and permissions):

```bash
sudo mount /RTMRI/RTExport
sudo chown <user>:<user> /RTMRI/RTExport
sudo chmod 775 /RTMRI/RTExport
```

## 3. Share the folder with Samba

1. Install Samba:

   ```bash
   sudo apt install samba
   ```

2. Add this section at the end of `/etc/samba/smb.conf` (e.g., `sudo nano /etc/samba/smb.conf`):

   ```ini
   [RTMRI]
   path = /RTMRI
   browseable = yes
   read only = no
   guest ok = no
   ```

   `[RTMRI]` is the share name that the console uses. `path` is the folder of step 2.

3. Set a Samba password for `<user>`. The console logs in with this user name and password. (It is separate from the Linux login password.)

   ```bash
   sudo smbpasswd -a <user>
   ```

4. Restart Samba, and start it automatically at boot:

   ```bash
   sudo systemctl restart smbd
   sudo systemctl enable smbd
   ```

5. If a firewall is enabled (`sudo ufw status` shows "active"), allow Samba from the console:

   ```bash
   sudo ufw allow from <console IP> to any app Samba
   ```

## 4. Check the share from another computer

Before going to the scanner, check that the share accepts files. From another Linux computer on the network (or from the RTPSpy computer itself, with `localhost`):

```bash
smbclient //<RTPSpy PC IP>/RTMRI -U <user> -c 'cd RTExport; put /etc/hostname test.txt; ls'
```

`test.txt` should be listed, and appear in `/RTMRI/RTExport` on the RTPSpy computer. Delete it afterwards. (`smbclient` is in the `smbclient` package.)

From a Windows computer, open `\\<RTPSpy PC IP>\RTMRI` in the File Explorer and copy a file into `RTExport`.

## 5. Set up the BOLD AddIn on the console

On the scanner console:

1. Open the fMRI protocol and add the **BOLD AddIn** to the scan.
2. In the AddIn configuration, set the online export:

   | Setting | Value |
   |---|---|
   | Target Host | `\\<RTPSpy PC IP>\RTMRI` |
   | Target Directory | `RTExport` |
   | User Name | `<user>` |
   | Password | the Samba password of `<user>` (step 3) |

3. Save the protocol, so that the setting is used for every scan with it. Add the AddIn to every scan whose images RTPSpy should receive, including the reference scans if you want to receive them in real time.

During a scan, each reconstructed volume is written to `RTExport` as a DICOM file (`.dcm`). On XA software, each file is one volume of one echo (an "enhanced" multi-frame DICOM file); RTPSpy also reads the mosaic DICOM files of the older VE software.

## 6. Set up RTPSpy

In the RTPSpy application:

1. Set the **Watch directory** (at the top of the window) to the receiving folder, `/RTMRI/RTExport`.
2. Check the WATCH module settings ('RTP' tab → 'WATCH'):
   - **File type**: `Dicom`;
   - **File name pattern**: `.+\.dcm$` (the default);
   - **Poll the directory (for network drives)**: leave it unchecked. The shared folder is on the RTPSpy computer itself, so new files are noticed at once. (Polling is only for a folder on another computer, mounted over the network.)
   - **Clean watch dir at ready**: check it to move the files left in the folder to `dicom/` in the working directory when you press 'Ready'. Recommended, especially with a RAM disk.

RTPSpy skips the images that are not for the processing: derived images, the motion-corrected series (MoCoSeries), and phase images (series ending with `_Pha`).

## 7. Test with a phantom scan

1. Start the RTPSpy application, set the masks and the processing as for a real session (see the [ROI-NF example](example/ROI-NF#readme)), and press 'Ready'.
2. Start an fMRI scan of a phantom with the BOLD AddIn.
3. Start the processing with the scanner TTL pulse, the 'T' key, or 'Manual start' (see the [README](README.md#starting-the-scan-ttl-trigger-or-the-t-key)).
4. Check that:
   - a new file appears in `/RTMRI/RTExport` at each TR (`ls /RTMRI/RTExport`);
   - RTPSpy processes each volume (the log in the RTPSpy window);
   - with 'Latency' checked, the 'TTL → image' delay is stable (usually a few seconds or less; it includes the image reconstruction).

To practice without the scanner, the [RT-MRI simulator](README.md#testing-without-a-scanner-the-rt-mri-simulator) copies recorded DICOM files into the folder at the original timing.

## 8. Troubleshooting

| Problem | Check |
|---|---|
| The console cannot connect to the share | The network between the console and the RTPSpy computer (`ping`); the Target Host (backslashes, IP address, share name); the firewall (step 3.5); `sudo systemctl status smbd`. |
| The console connects, but cannot write | The Samba user name and password (step 3.3); the owner and permissions of the folder (step 2), also after mounting a RAM disk. |
| Files arrive, but RTPSpy does not process them | The Watch directory is the folder the files arrive in (`RTExport`, not `/RTMRI`); the File type and File name pattern of WATCH; that 'Ready' was pressed. |
| Files arrive late or in bursts | The network speed and load; a RAM disk for the folder (step 2). |
| The folder fills up | Check 'Clean watch dir at ready', or move the old files away. A RAM disk of the wrong size fills quickly with multi-echo or high-resolution data. |

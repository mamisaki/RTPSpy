#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Container images of the external tools: the AFNI command-line programs and
FastSurfer. install.sh pulls the images and writes the configuration file
containers/containers.json in the RTPSpy directory (or the file given by the
RTPSPY_CONTAINERS environment variable).

The AFNI programs are called through small wrapper scripts in
containers/afni_bin (one per program, made by install.sh), which run
    python -I containers.py exec afni <program> <args>
This file uses only the standard library and does not import the rtpspy
package, so a wrapper starts quickly.

Host directories visible in the container: $HOME, /tmp, the current
directory, the 'binds' list in containers.json, the RTPSPY_BIND environment
variable (colon-separated), and the top-level directory (e.g., /mnt) of each
absolute path found in the arguments.

The AFNI GUI (afni, plugout_drive) is not run in the container.

@author: mmisaki@laureateinstitute.org
"""

# %% import ===================================================================
import json
import os
import re
import sys
from pathlib import Path

PKG_DIR = Path(__file__).absolute().parent
DEFAULT_CONFIG = PKG_DIR.parent / 'containers' / 'containers.json'

# Top-level host directories that are never bound (system directories of the
# container)
_NO_BIND_TOP = {'', 'proc', 'sys', 'dev', 'usr', 'bin', 'sbin', 'lib',
                'lib32', 'lib64', 'libx32', 'etc', 'opt', 'var', 'run',
                'boot', 'root', 'srv', 'snap', 'lost+found', 'fastsurfer'}

# Host environment variables passed to the container (with --cleanenv)
_PASS_ENV = re.compile(r'^(AFNI_|OMP_NUM_THREADS$|TZ$|LANG$|LC_)')


# %% load_config ==============================================================
def config_file():
    return Path(os.environ.get('RTPSPY_CONTAINERS', DEFAULT_CONFIG))


def load_config():
    """Return the container configuration (dict), or None if not set up."""
    cfg_f = config_file()
    if not cfg_f.is_file():
        return None
    with open(cfg_f) as fd:
        return json.load(fd)


def tool_config(tool):
    """
    Return the configuration of a tool ('afni' or 'fastsurfer') if it is run
    in a container, otherwise None.
    """
    cfg = load_config()
    if cfg is None:
        return None
    tcfg = cfg.get(tool)
    if not tcfg or tcfg.get('mode') != 'container':
        return None
    return dict(tcfg, runtime=cfg['runtime'], binds=cfg.get('binds', []))


# %% bind_dirs ================================================================
def _top_dir(path):
    parts = Path(path).parts
    if len(parts) < 2 or parts[0] != '/':
        return None
    if parts[1] in _NO_BIND_TOP:
        return None
    return '/' + parts[1]


def bind_dirs(args=(), binds=(), cwd=None):
    """
    Host directories to bind into the container (see the module docstring).
    Each directory is bound to the same path in the container.
    """
    cands = [str(Path.home()), '/tmp']
    cwd = os.getcwd() if cwd is None else str(cwd)
    cands += [cwd, os.path.realpath(cwd)]
    cands += list(binds)
    cands += [b for b in os.environ.get('RTPSPY_BIND', '').split(':') if b]

    # Absolute paths in the arguments (also after '=' or ',' such as
    # -prefix=/data/x), and their symlink targets
    for arg in args:
        for m in re.finditer(r'(?:^|[=,:\s])(/[^=,:\s\'"]+)', str(arg)):
            pth = m.group(1)
            for pp in (pth, os.path.realpath(pth)):
                top = _top_dir(pp)
                if top is not None:
                    cands.append(top)

    out = []
    for dd in cands:
        dd = os.path.normpath(dd)
        if not os.path.isdir(dd) or dd in out:
            continue
        # Skip a directory inside another one already bound
        if any(dd.startswith(o.rstrip('/') + '/') for o in out):
            continue
        out = [o for o in out if not o.startswith(dd.rstrip('/') + '/')]
        out.append(dd)
    return out


# %% container_cmd ============================================================
def container_cmd(tool, argv, gpu=False, cwd=None, entrypoint=None):
    """
    Command line (list) to run argv in the container of tool.

    Parameters
    ----------
    tool : str
        'afni' or 'fastsurfer'.
    argv : list of str
        Program and its arguments, run in the container.
    gpu : bool, optional
        Make the NVIDIA GPU available in the container.
    cwd : str or Path, optional
        Working directory (the default is the current directory).
    entrypoint : str, optional
        Docker only: program to run instead of argv[0] (the image
        entrypoint is overridden in any case).
    """
    tcfg = tool_config(tool)
    if tcfg is None:
        raise RuntimeError(
            f"{tool} container is not set up (run install.sh; config:"
            f" {config_file()})")

    cwd = os.getcwd() if cwd is None else str(cwd)
    binds = bind_dirs(argv[1:], tcfg['binds'], cwd)
    env = {k: v for k, v in os.environ.items() if _PASS_ENV.match(k)}
    # Do not use the user's Python packages in ~/.local
    env['PYTHONNOUSERSITE'] = '1'

    runtime = tcfg['runtime']
    if runtime in ('apptainer', 'singularity'):
        cmd = [runtime, '-s', 'exec', '--cleanenv', '--pwd', cwd]
        if gpu:
            cmd.append('--nv')
        for bb in binds:
            cmd += ['-B', bb]
        for k, v in env.items():
            cmd += ['--env', f'{k}={v}']
        if tcfg.get('path'):
            # Directory of the programs in the image (not on the image PATH)
            cmd += ['--env', f"PREPEND_PATH={tcfg['path']}"]
        cmd += [tcfg['image']] + [str(a) for a in argv]

    elif runtime in ('docker', 'podman'):
        cmd = [runtime, 'run', '--rm', '-i', '-w', cwd,
               '-e', f'HOME={Path.home()}']
        if runtime == 'docker':
            cmd += ['-u', f'{os.getuid()}:{os.getgid()}']
        else:
            cmd += ['--userns=keep-id']
        if sys.stdin.isatty():
            cmd.append('-t')
        if gpu:
            cmd += ['--gpus', 'all']
        for bb in binds:
            cmd += ['-v', f'{bb}:{bb}']
        for k, v in env.items():
            cmd += ['-e', f'{k}={v}']
        if tcfg.get('path'):
            cmd += ['-e', f"PATH={tcfg['path']}:/usr/local/sbin:/usr/local/bin"
                    ":/usr/sbin:/usr/bin:/sbin:/bin"]
        cmd += ['--entrypoint', entrypoint or str(argv[0]), tcfg['image']]
        cmd += [str(a) for a in argv[1:]]

    else:
        raise RuntimeError(f"Unknown container runtime: {runtime}")

    return cmd


# %% setup_path ===============================================================
def setup_path():
    """
    Prepend to PATH the AFNI wrapper directory (when AFNI runs in the
    container) and the directory of this Python (for ants_run.py), so that
    the programs are found without activating the environment.
    """
    path = os.environ.get('PATH', '').split(os.pathsep)
    add = []

    bin_dir = str(Path(sys.executable).parent)
    if bin_dir not in path:
        add.append(bin_dir)

    tcfg = tool_config('afni')
    if tcfg is not None and tcfg.get('bin_dir'):
        afni_bin = tcfg['bin_dir']
        if afni_bin in path:
            path.remove(afni_bin)
        add.insert(0, afni_bin)

    if add:
        os.environ['PATH'] = os.pathsep.join(add + path)


# %% librtp_file ==============================================================
def librtp_file():
    """
    The librtp library to load: the RTPSPY_LIBRTP environment variable, or
    the library built by afni_module/build_librtp_container.sh
    (containers/librtp.so, made by install.sh when the bundled library does
    not load), or the bundled rtpspy/librtp.so (.dylib on macOS).
    """
    lib_name = 'librtp.dylib' if sys.platform == 'darwin' else 'librtp.so'
    if os.environ.get('RTPSPY_LIBRTP'):
        return os.environ['RTPSPY_LIBRTP']
    built = DEFAULT_CONFIG.parent / lib_name
    if built.is_file():
        return str(built)
    return str(PKG_DIR / lib_name)


# %% main =====================================================================
if __name__ == '__main__':
    # containers.py exec <tool> <program> [args...]
    if len(sys.argv) < 4 or sys.argv[1] != 'exec':
        sys.exit(f"Usage: {Path(__file__).name} exec <afni|fastsurfer>"
                 " <program> [args...]")
    tool, argv = sys.argv[2], sys.argv[3:]
    try:
        cmd = container_cmd(tool, argv)
    except RuntimeError as e:
        sys.exit(f"[rtpspy.containers] {e}")
    os.execvp(cmd[0], cmd)

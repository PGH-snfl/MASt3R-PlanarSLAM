"""Mount this laptop's maps folder on a GPU server through outbound SSH.

Uses the SSHFS documented passive mode: local sftp-server <-> SSH <-> sshfs.
No laptop SSH daemon, inbound port, Tailscale or server-side laptop key needed.
Leave this process running until SLAM has finished flushing all map writes.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from mast3r_slam.remote_archive import MARKER


# Only protocol bytes go to stdout once sshfs starts. Diagnostics go to stderr.
REMOTE_SETUP = r'''
import os, pathlib, shutil, sys
local, mount = sys.argv[1:]
mount = pathlib.Path(mount).expanduser().absolute()
if not shutil.which('sshfs'):
    sys.exit('Server needs sshfs: sudo apt install sshfs')
if not os.path.exists('/dev/fuse'):
    sys.exit('Server /dev/fuse is unavailable; enable FUSE before mounting')
if os.path.ismount(mount):
    sys.exit('Archive mount already exists. Stop its SLAM before unmounting/replacing it.')
mount.mkdir(parents=True, exist_ok=True)
if any(mount.iterdir()):
    sys.exit('Refusing to hide files: mount directory must be empty')
print('Mounting laptop archive at ' + str(mount), file=sys.stderr, flush=True)
os.execvp('sshfs', ['sshfs', '-f', '-o', 'passive,idmap=user,dir_cache=no,cache=no', ':' + local, str(mount)])
'''


def ssh_command(args, local_dir):
    # No shell on the laptop; quote the complete remote command for SSH's shell.
    remote = shlex.join(['python3', '-c', REMOTE_SETUP, str(local_dir), args.server_mount])
    command = ['ssh', '-T', '-p', str(args.ssh_port), '-o', 'ServerAliveInterval=15',
               '-o', 'ServerAliveCountMax=3']
    if args.identity:
        command += ['-i', os.path.expanduser(args.identity)]
    return command + ['--', args.server, remote]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', required=True, help='SSH user@public-IP or ~/.ssh/config alias')
    parser.add_argument('--ssh-port', type=int, default=22)
    parser.add_argument('--identity', help='Optional SSH private key path (never copied)')
    parser.add_argument('--local-dir', type=Path, default=Path(__file__).resolve().parent/'maps'/'server')
    parser.add_argument('--server-mount', default='~/laptop-maps')
    parser.add_argument('--sftp-server', default='/usr/lib/openssh/sftp-server')
    args = parser.parse_args()
    if not 1 <= args.ssh_port <= 65535 or args.server.startswith('-'):
        parser.error('Invalid SSH host or port')
    if not shutil.which(args.sftp_server):
        parser.error('Local sftp-server missing. On Ubuntu: sudo apt install openssh-sftp-server')
    local = args.local_dir.expanduser().resolve()
    local.mkdir(parents=True, exist_ok=True)
    marker = local / MARKER
    if not marker.exists():
        marker.write_text(json.dumps({'format': 'mast3r-laptop-archive-v1', 'directory': str(local)})+'\n')
    elif json.loads(marker.read_text()).get('format') != 'mast3r-laptop-archive-v1':
        parser.error(f'Unrecognized marker: {marker}')
    print(f'Laptop map destination: {local}', flush=True)
    print('Keep this terminal open. Stop SLAM and wait for "Saved map" before Ctrl+C here.', flush=True)
    # Wire the two child processes directly: no Python buffering of map files.
    sftp_read, ssh_write = os.pipe()
    ssh_read, sftp_write = os.pipe()
    children = []
    try:
        children.append(subprocess.Popen([args.sftp_server, '-d', str(local)],
                                         stdin=sftp_read, stdout=sftp_write))
        children.append(subprocess.Popen(ssh_command(args, local), stdin=ssh_read, stdout=ssh_write))
    except BaseException:
        for child in children:
            child.terminate()
            child.wait(timeout=5)
        raise
    finally:
        for fd in (sftp_read, ssh_write, ssh_read, sftp_write):
            os.close(fd)
    result = 0
    try:
        while all(child.poll() is None for child in children):
            time.sleep(.2)
        result = next((child.returncode for child in children if child.returncode), 0)
    except KeyboardInterrupt:
        print('\nClosing map connection. After SLAM stops, unmount on the server:')
        mount_arg = ('"$HOME"/' + shlex.quote(args.server_mount[2:])
                     if args.server_mount.startswith('~/') else shlex.quote(args.server_mount))
        print('fusermount3 -u ' + mount_arg)
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
    return result


if __name__ == '__main__':
    sys.exit(main())

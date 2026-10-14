"""Fail closed when a requested laptop archive is not an SSHFS mount."""
import json
import os
from pathlib import Path
import re

MARKER = '.mast3r-laptop-archive.json'


def sshfs_mounts(mountinfo):
    result = set()
    for line in mountinfo.splitlines():
        fields, separator, rest = line.partition(' - ')
        if separator and rest.split()[0] in ('fuse.sshfs', 'fuse.sshfs3'):
            path = fields.split()[4]
            result.add(re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), path))
    return result


def require_archive_mount(path, check_marker=True):
    path = Path(os.path.abspath(os.path.expanduser(str(path))))
    if str(path) not in sshfs_mounts(Path('/proc/self/mountinfo').read_text()):
        raise RuntimeError(f'Laptop archive is not mounted at {path}. Start mount_laptop_maps.py on the laptop. Refusing to save maps on the server.')
    if check_marker:
        marker = json.loads((path / MARKER).read_text())
        if marker.get('format') != 'mast3r-laptop-archive-v1':
            raise RuntimeError('Wrong laptop archive marker')
    return path

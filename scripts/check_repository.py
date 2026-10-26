#!/usr/bin/env python3
"""Check the Git index for generated artifacts, machine links and large files."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {'.local', 'external', 'lerobot', 'vendor', 'cache', 'outputs', 'runs',
             'reports', 'artifacts', 'inputs', 'weights', 'logs', '.venv', 'build', 'dist'}
BINARY_OUTPUTS = {'.pt', '.pth', '.ckpt', '.npy', '.npz', '.mp4', '.zip', '.tar', '.gz'}
FORBIDDEN_PREFIXES = ('portal/reports/', 'portal/data/', 'portal/artifacts/',
                      'portal/src/', 'portal/scripts/', 'portal/docs/',
                      'portal_host/bundles/')


def check(root=ROOT):
    records = subprocess.check_output(['git', '-C', str(root), 'ls-files', '--stage', '-z']).split(b'\0')
    errors, count = [], 0
    for record in records:
        if not record:
            continue
        metadata, raw_path = record.split(b'\t', 1)
        mode, _, stage = metadata.decode().split()
        relative = Path(raw_path.decode())
        count += 1
        if stage != '0':
            errors.append(f'{relative}: unresolved merge')
        if mode not in {'100644', '100755'}:
            errors.append(f'{relative}: tracked link/submodule or unsupported mode {mode}')
        if relative.parts[0] in FORBIDDEN or '__pycache__' in relative.parts:
            errors.append(f'{relative}: runtime/dependency path is tracked')
        if relative.as_posix().startswith(FORBIDDEN_PREFIXES):
            errors.append(f'{relative}: generated portal content or duplicate implementation')
        if relative.parts[0] == 'portal_host' and relative.suffix in {'.json', '.xml', '.png', '.jpg'} and relative.name != 'config.example.json':
            errors.append(f'{relative}: deployment/QA output is tracked')
        if relative.suffix.lower() in BINARY_OUTPUTS:
            errors.append(f'{relative}: generated binary must live outside Git')
        if relative.name in {'local.json', 'deployment.json'} or relative.name.endswith('.local.json'):
            errors.append(f'{relative}: machine configuration/receipt is tracked')
        path = root / relative
        if not path.is_file() or path.is_symlink():
            errors.append(f'{relative}: missing source file or symlink')
        elif path.stat().st_size > 2 * 1024 * 1024:
            errors.append(f'{relative}: exceeds the 2 MiB source-file review limit')
    return count, errors


if __name__ == '__main__':
    count, errors = check()
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'PASS: {count} tracked source/configuration/documentation files; no runtime artifacts or links')

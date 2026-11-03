"""Check staged or committed distribution paths without reading external data.

Run after staging the intended upload. Git's index defines the file list, so
untracked predictions and local environment files cannot enter this check.
"""
import csv
import json
from pathlib import Path, PurePosixPath
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {
    '.gitignore', 'README.md', 'README.tex', 'UPLOAD.tex', 'VERSION',
    'requirements.txt', 'requirements-chronos.txt', 'requirements-moirai.txt',
    'requirements-round1.txt',
}

def allowed(name):
    path = PurePosixPath(name)
    if name in ROOT_FILES:
        return True
    if len(path.parts) != 2 and path.parts[0] != 'results':
        return False
    if path.parts[0] in {'scripts', 'tests'}:
        return path.suffix in {'.py', '.sh'}
    if path.parts[0] == 'protocols':
        return path.suffix == '.json'
    if path.parts[0] == 'results':
        return path.suffix in {'.json', '.csv', '.pdf'} or path.name == 'README.tex'
    return False

def reject_nonfinite(value):
    raise ValueError(f'Nonstandard JSON numeric constant: {value}')

def main():
    raw = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT)
    names = raw.decode('utf-8').rstrip('\0').split('\0')
    errors = []
    size = 0
    for name in names:
        path = ROOT / name
        if not allowed(name):
            errors.append(f'Excluded distribution path: {name}')
            continue
        size += path.stat().st_size
        if path.suffix == '.json':
            json.loads(path.read_text(encoding='utf-8'), parse_constant=reject_nonfinite)
        if path.suffix == '.csv':
            with path.open(encoding='utf-8', newline='') as stream:
                for row in csv.DictReader(stream):
                    if row.get('wis_status') == 'withdrawn_full_predictions_required':
                        for key in ('wis_kwh', 'weighted_interval_score_kwh', 'mean_wis_kwh'):
                            if row.get(key):
                                errors.append(f'Withdrawn WIS is populated in {name}')
        if path.suffix == '.pdf' and not path.read_bytes().startswith(b'%PDF-'):
            errors.append(f'Invalid figure PDF: {name}')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(json.dumps({'passed': True, 'indexed_files': len(names),
                      'total_bytes': size, 'scope': 'programs, protocols, documentation, and aggregate results'}, indent=2))

if __name__ == '__main__':
    main()

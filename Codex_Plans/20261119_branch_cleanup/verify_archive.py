"""Verify archived bytes, source commits, tags and coverage against the inventory."""
import csv
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def main():
    index = ROOT / 'AI_docs/早期旧分支文档/文件清单.csv'
    with index.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        target = ROOT / row['归档路径']
        data = target.read_bytes()
        assert hashlib.sha256(data).hexdigest() == row['SHA256'], target
        assert git('hash-object', str(target)).decode().strip() == row['Git blob'], target
        assert git('rev-parse', row['原提交'] + ':' + row['原路径']).decode().strip() == row['Git blob'], target
        tag = 'archive/' + row['原分支'] + '-2026-11-19'
        assert git('rev-parse', tag + '^{commit}').decode().strip() == row['原提交'], tag
        git('ls-files', '--error-unmatch', row['归档路径'])
    for name in ['run.sh', 'run_trs.sh', 'run_rhl_norm.sh', 'run_origin.sh']:
        archived = ROOT / 'AI_docs/早期旧分支文档/main历史脚本' / name
        assert archived.read_bytes() == git('show', '47a512a:' + name), name
    print(f'PASS: {len(rows)} archive inventory rows, source commits/tags and tracked files; 4 original main runners.')


if __name__ == '__main__':
    main()

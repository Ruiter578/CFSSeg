"""Export branch-only documentation and reference scripts without merging code."""
import csv
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'AI_docs/早期旧分支文档'
BASE = '47a512a'
BRANCHES = {
    'deeplabv3plus-control': '7fa27e03b6174729ae959a1e6df5e9726eb32287',
    'adaptive-pseudo-label': '85d7ea1641563b2daba0f914d51d322048f23424',
    'rhl-se-boa-p0-p1': '7701b1223ca0ec67a81cc18b3dc7bbd4d8d70159',
}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def tree(ref):
    rows = {}
    for entry in git('ls-tree', '-r', '-z', ref).split(b'\0'):
        if not entry:
            continue
        meta, name = entry.split(b'\t', 1)
        mode, kind, blob = meta.decode().split()
        if kind == 'blob':
            rows[name.decode()] = blob
    return rows


def main():
    baseline = tree(BASE)
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = []
    for name, commit in BRANCHES.items():
        files = tree(commit)
        documents, references, results = [], [], []
        for source, blob in files.items():
            if baseline.get(source) == blob:
                continue
            path = Path(source)
            document = source.startswith(('AI_docs/', 'Codex_Plans/')) and path.suffix in {'.md', '.html', '.csv', '.tsv', '.json', '.svg'}
            document |= source in {'AGENTS.md', 'README.md'}
            reference = name != 'adaptive-pseudo-label' and (
                path.suffix == '.sh' and source not in baseline
                or source in {'tests/test_air_feature_sources.py', 'tests/test_summarize_v3plus_air_results.py', 'tools/summarize_v3plus_air_results.py'}
            )
            result = source.startswith(('artifacts/', 'logs/', 'checkpoints/')) and path.suffix in {'.json', '.csv', '.tsv'}
            if not (document or reference or result):
                continue
            category = '文档' if document else '结果' if result else '参考脚本'
            destination = OUT / name / category / source
            destination.parent.mkdir(parents=True, exist_ok=True)
            data = git('cat-file', 'blob', blob)
            destination.write_bytes(data)
            assert git('hash-object', str(destination)).decode().strip() == blob
            inventory.append([name, commit, source, str(destination.relative_to(ROOT)), blob, hashlib.sha256(data).hexdigest(), category])
            {'文档': documents, '结果': results, '参考脚本': references}[category].append(destination)
        rows = ['# ' + name + ' 历史资料', '',
                f'原分支提交：`{commit}`；恢复标签：`archive/{name}-2026-11-19`。', '',
                '这里保存与整合前main不同的文档版本、轻量结果及参考脚本；文件内容与原提交逐字节一致。历史命令、绝对路径和状态叙述按原样保留，当前使用方式见项目根README。完整源码可从归档标签恢复。', '']
        for title, paths in [('文档', documents), ('结果', results), ('参考脚本', references)]:
            if not paths:
                continue
            rows += ['## ' + title, '']
            for target in paths:
                relative = target.relative_to(OUT / name).as_posix()
                rows.append(f'- [{target.name}](<{relative}>)')
            rows.append('')
        (OUT / name / 'README.md').write_text('\n'.join(rows), encoding='utf-8')
    with (OUT / '文件清单.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['原分支', '原提交', '原路径', '归档路径', 'Git blob', 'SHA256', '类型'])
        writer.writerows(inventory)
    print(f'Archived {len(inventory)} files from {len(BRANCHES)} branches; all Git blob hashes verified.')


if __name__ == '__main__':
    main()

"""Independently compare exported metrics with Git blobs and report tables."""
import csv
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

EV = Path(__file__).resolve().parent
OUT = EV.parent
ROOT = OUT.parents[1]


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def same(a, b):
    return math.isclose(a, b, abs_tol=1e-11, rel_tol=1e-12)


def main():
    rows = json.loads((EV / 'raw_metrics.json').read_text())
    history = json.loads((EV / 'history_only_metrics.json').read_text())
    legacy = json.loads((EV / 'legacy_results.json').read_text())
    assert len(rows) == 191 and len(history) == 16 and len(legacy) == 110
    assert len({(r['path'], r['blob']) for r in rows + history}) == 207
    report = (OUT / '03_方法与实验完整回顾.md').read_text()
    current_table = report.split('<!-- TABLE:all_raw -->')[1].split('<!-- END:all_raw -->')[0]
    historical_table = report.split('<!-- TABLE:history -->')[1].split('<!-- END:history -->')[0]
    for r in rows + history:
        source = git('cat-file', 'blob', r['blob'])
        data = json.loads(source)
        assert data == r['source_data'], r['id']
        m = data.get('metrics', data)
        assert same(r['all_pct'], m['Mean IoU'] * 100), r['id']
        if r['id'].startswith('R'):
            assert hashlib.sha256(source).hexdigest() == r['result_sha256']
            assert m.get('Class IoU', {}) == r['per_class']
            keys = [r['old_group'], r['new_group']]
            table = current_table
        else:
            assert git('rev-parse', r['source_commit'] + ':' + r['path']).decode().strip() == r['blob']
            keys = ['0 to 15 mIoU', '16 to 20 mIoU']
            table = historical_table
        for field, key in zip(['old_pct', 'new_pct'], keys):
            if key in m:
                assert same(r[field], m[key] * 100), (r['id'], field)
            else:
                assert r[field] == '', (r['id'], field)
        line = next(line for line in table.splitlines() if '[' + r['id'] + ']' in line)
        for field in ['old_pct', 'new_pct', 'all_pct']:
            value = r[field]
            assert (f'{value:.6f}' if isinstance(value, (float, int)) else '—') in line, (r['id'], field)
    original_legacy = list(csv.DictReader((ROOT / 'AI_docs/阶段性总结/SegACIL实验结果与方法对应表.csv').open(encoding='utf-8-sig')))
    assert legacy == original_legacy
    per = list(csv.DictReader((EV / 'per_class_metrics.csv').open(encoding='utf-8-sig')))
    lookup = {r['id']: r for r in rows}
    assert len(per) == sum(len(r['per_class']) for r in rows) == 3983
    for p in per:
        assert same(float(p['iou_pct']), lookup[p['result_id']]['per_class'][p['class_id']] * 100)
    docrows = list(csv.DictReader((EV / 'document_inventory.csv').open(encoding='utf-8-sig')))
    assert len(docrows) == 112 and len({d['path'] for d in docrows}) == 97
    snapshot = json.loads((EV / 'snapshot.json').read_text())
    heads = {r['ref']: r['commit'] for r in snapshot['branches']}
    for d in docrows:
        for ref in d['refs'].split(';'):
            assert git('rev-parse', heads[ref] + ':' + d['path']).decode().strip() == d['blob']
    clean_decoder = [lookup[i]['all_pct'] for i in ['R183', 'R185', 'R187']]
    clean_aspp = [lookup[i]['all_pct'] for i in ['R175', 'R177', 'R179']]
    means = {'decoder_pct': sum(clean_decoder)/3, 'aspp_pct': sum(clean_aspp)/3}
    means['difference_pp'] = means['decoder_pct'] - means['aspp_pct']
    assert round(means['decoder_pct'], 4) == 59.6412
    assert round(means['aspp_pct'], 4) == 59.1228
    epsilon = {i: lookup[i]['all_pct'] - lookup['R183']['all_pct'] for i in ['R189', 'R191']}
    boa = lookup['R055']['all_pct'] - lookup['R053']['all_pct']
    assert round(boa, 6) == 0.078889
    assert all(v < 0 for v in epsilon.values())
    html = (OUT / '06_方法与实验交接.html').read_text()
    embedded = json.loads(re.search(r'<script id="records-data" type="application/json">(.*?)</script>', html, re.S).group(1))
    assert len(embedded) == 317
    for r in embedded:
        source = next(x for x in rows + history if x['id'] == r['id']) if r['grade'] == 'A' else next(x for x in legacy if x['证据编号'] == r['id'])
        assert r['source_data'] == (source['source_data'] if r['grade'] == 'A' else source)
    for asset in ['methods-map.svg', 'route-map.svg']:
        assert (OUT / 'assets' / asset).is_file()
    result = {
        'checked_date': '2026-10-25', 'status': 'PASS',
        'checks': {
            'raw_current_git_blobs_and_summary_cells': len(rows),
            'historical_git_blobs_and_summary_cells': len(history),
            'legacy_csv_rows': len(legacy),
            'per_class_values': len(per),
            'document_versions_and_commit_paths': len(docrows),
            'embedded_html_records': len(embedded)
        },
        'clean_means': means, 'epsilon_deltas_pp': epsilon, 'boa_delta_pp': boa,
        'scope': 'Git原始数值、导出文件与展示一致性；没有重跑训练或推理'
    }
    (EV / 'evidence_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

"""Read-only source audit; regenerate inventories inside this handoff directory."""
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
REFS = ["main", "feature/adaptive-pseudo-label", "origin/feature/adaptive-pseudo-label",
        "origin/feature/deeplabv3plus-control", "origin/feature/rhl-next", "origin/feature/rhl-se-boa-p0-p1"]

def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])

def write_csv(name, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

def classify_doc(path, content):
    if "Prompts" in path or "resume_epoch" in path or "已归档" in path:
        return "90_archive/process", "过程档案；按需追溯"
    if "3D部分" in path:
        return "90_archive/3d", "旁线保留；非2D必读"
    if "论文精读" in path:
        return "01_foundations/paper", "参考保留；多份精读宜合并索引"
    if "数学推导" in path:
        return "01_foundations/math", "原理参考；公式需与代码对照"
    if "代码理解" in path:
        return "01_foundations/code", "参考保留；历史实现描述需更新"
    if "RHL-SE" in path or "Snapshot" in path or "集成方案" in path:
        return "90_archive/ensemble", "集成路线暂停；保留负结果与代码入口"
    if "7-19_" in path and any(x in path for x in ["冻结", "W1", "matched-global", "全阶段", "W0"]):
        return "03_experiments/pseudo", "结论级必保留；优先阅读"
    if "clean_replay结果" in path or "clean验证" in path:
        return "03_experiments/protocol", "结论级必读；解释协议变更"
    if "buffer" in path and any(x in path for x in ["总", "最终", "停止"]):
        return "03_experiments/hyperparameters", "结果参考；归档调参线"
    if "阶段性总结" in path or "课题Home" in path:
        return "90_archive/snapshots", "历史快照；由本次交接替代当前入口"
    if "代码改动" in path:
        return "04_engineering/history", "按模块追溯；现行约定单独维护"
    if "idea验证" in path:
        return "03_experiments/legacy", "结果证据保留；标明协议与证据等级"
    return "02_methods/history", "设计档案；区分提案与已实现"

def main():
    trees = {}
    docs = {}
    metrics = {}
    summaries = []
    branches = []
    for ref in REFS:
        tree = {}
        for entry in git("ls-tree", "-r", "-z", ref).split(b"\0"):
            if not entry:
                continue
            meta, path = entry.split(b"\t", 1)
            tree[path.decode()] = meta.decode().split()[2]
        trees[ref] = tree
        counts = git("rev-list", "--left-right", "--count", "main..." + ref).decode().split()
        branches.append({"ref": ref, "commit": git("rev-parse", ref).decode().strip(),
                         "main_only": counts[0], "ref_only": counts[1],
                         "head_date_subject": git("log", "-1", "--format=%cI %s", ref).decode().strip(),
                         "ai_docs_md": sum(p.startswith("AI_docs/") and p.endswith(".md") for p in tree)})
        for path, blob in tree.items():
            if path.startswith("AI_docs/") and path.endswith(".md"):
                key = (path, blob)
                if key not in docs:
                    content = git("cat-file", "blob", blob).decode("utf-8", errors="replace")
                    dest, decision = classify_doc(path, content)
                    docs[key] = {"path": path, "blob": blob, "refs": [], "content": content,
                                 "lines": len(content.splitlines()), "proposed_directory": dest, "decision": decision}
                docs[key]["refs"].append(ref)
            if path.endswith((".tsv", ".csv")) and any(x in path for x in ["summar", "结果"]):
                summaries.append({"ref": ref, "path": path, "blob": blob})
            if path.endswith(".json") and ("test_results_" in path or "val_results_" in path or "full_test_audit_" in path):
                key = (path, blob)
                if key in metrics:
                    metrics[key]["refs"].append(ref)
                    continue
                data = json.loads(git("cat-file", "blob", blob))
                directory = str(Path(path).parent)
                manifest_path = directory + "/run_manifest.json"
                manifest = json.loads(git("cat-file", "blob", tree[manifest_path])) if manifest_path in tree else {}
                config_path = directory + "/run_config.json"
                config = json.loads(git("cat-file", "blob", tree[config_path])) if config_path in tree else {}
                m = data.get("metrics", data)
                if "Mean IoU" not in m:
                    continue
                row = {"path": path, "blob": blob, "refs": [ref], "run": path.split("/")[1],
                       "task": path.split("/")[3], "setting": path.split("/")[4], "step": Path(path).parent.name,
                       "split": "official-val-as-test" if "test_results_" in path or "full_test_audit_" in path else "validation",
                       "all_pct": m["Mean IoU"] * 100,
                       "old_pct": m.get("0 to 15 mIoU", ""), "new_pct": m.get("16 to 20 mIoU", ""),
                       "per_class": m.get("Class IoU", {}), "manifest": manifest_path if manifest else "",
                       "result_stamp": re.search(r"(20\d{6}(?:_\d{6})?)", Path(path).name).group(1),
                       "git_commit": manifest.get("git", {}).get("commit", manifest.get("git_commit", "")),
                       "git_dirty": manifest.get("git", {}).get("dirty", ""),
                       "base_hash": manifest.get("base_checkpoint_sha256", ""),
                       "validation_hash": manifest.get("validation_list_sha256", ""),
                       "source_data": data}
                old_key = next((k for k in m if k.startswith("0 to ") and k.endswith(" mIoU")), "")
                new_key = next((k for k in m if k.startswith("16 to ") and k.endswith(" mIoU")), "")
                row["old_group"] = old_key
                row["new_group"] = new_key
                row["old_pct"] = m.get(old_key, "")
                row["new_pct"] = m.get(new_key, "")
                row["result_sha256"] = hashlib.sha256(git("cat-file", "blob", blob)).hexdigest()
                for field in ["old_pct", "new_pct"]:
                    if isinstance(row[field], (float, int)):
                        row[field] *= 100
                for field in ["model", "buffer", "gamma", "random_seed", "rhl_seed", "rhl_norm", "rhl_init", "rhl_scale", "batch_size", "base_subpath", "analytic_tail_epsilon", "air_feature_source", "resolved_air_feature_source", "evaluation_mode"]:
                    row[field] = manifest.get(field, manifest.get("args", {}).get(field, config.get(field, "")))
                metrics[key] = row
    docrows = []
    extracts = []
    for idx, item in enumerate(sorted(docs.values(), key=lambda d:(d["path"], d["blob"]))):
        content = item.pop("content")
        item["refs"] = ";".join(item["refs"])
        item["id"] = "D%03d" % (idx + 1)
        docrows.append(item)
        extracts.append("\n## " + item["id"] + " " + item["path"] + "\nrefs: " + item["refs"] + "\n" +
                        "\n".join(line for line in content.splitlines() if line.startswith(("#", "|")) or any(k in line for k in ["未实现", "未完成", "未启动", "停止", "失败原因", "没有", "尚未"])))
    metricrows = []
    for idx, item in enumerate(sorted(metrics.values(), key=lambda d:d["path"])):
        item["refs"] = ";".join(item["refs"])
        item["id"] = "R%03d" % (idx + 1)
        metricrows.append(item)
    (OUT / "raw_metrics.json").write_text(json.dumps(metricrows, ensure_ascii=False, indent=2), encoding="utf-8")
    history = []
    known = {(r['blob'], r['path']) for r in metricrows}
    for entry in git('rev-list', '--objects', '--all').decode().splitlines():
        if ' ' not in entry:
            continue
        blob, path = entry.split(' ', 1)
        if not (path.endswith('.json') and any(x in path for x in ['test_results', 'val_results', 'full_test_audit'])) or (blob, path) in known:
            continue
        data = json.loads(git('cat-file', 'blob', blob))
        if 'Mean IoU' not in data:
            continue
        ref = ''
        for commit in git('log', '--all', '--format=%H', '--', path).decode().splitlines():
            entry_at_commit = git('ls-tree', commit, '--', path).decode()
            if blob in entry_at_commit:
                ref = commit
                break
        manifest_path = str(Path(path).parent.parent / 'step1' / 'run_manifest.json')
        manifest_exists = bool(ref and git('ls-tree', ref, '--', manifest_path).strip())
        manifest = json.loads(git('show', ref + ':' + manifest_path)) if manifest_exists else {}
        history.append({'id': 'H%03d' % (len(history)+1), 'path': path, 'blob': blob, 'source_commit': ref,
                        'run': path.split('/')[1], 'step': Path(path).parent.name,
                        'old_pct': data['0 to 15 mIoU']*100 if '0 to 15 mIoU' in data else '',
                        'new_pct': data['16 to 20 mIoU']*100 if '16 to 20 mIoU' in data else '',
                        'all_pct': data['Mean IoU']*100, 'git_dirty': manifest.get('git', {}).get('dirty'),
                        'decision': '已被正式clean replay替代；不参与方法选择', 'source_data': data})
    (OUT / 'history_only_metrics.json').write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding='utf-8')
    write_csv('history_only_metrics.csv', [{k:v for k,v in h.items() if k!='source_data'} for h in history])
    write_csv("metrics_inventory.csv", [{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v) for k,v in r.items() if k != "source_data"} for r in metricrows])
    write_csv("document_inventory.csv", docrows)
    write_csv("branch_inventory.csv", branches)
    write_csv("summary_inventory.csv", summaries)
    (OUT / "document_extracts.txt").write_text("\n".join(extracts), encoding="utf-8")
    snapshot = {"audit_date": "2026-10-24", "branches": branches, "unique_document_versions": len(docrows),
                "unique_document_paths": len({d['path'] for d in docrows}), "metric_files": len(metricrows),
                "local_model_weights_pth": len(list(ROOT.glob("checkpoints/**/*.pth"))),
                "worktrees": git("worktree", "list", "--porcelain").decode(),
                "inventory_generation_status": git("status", "--short").decode(),
                "initial_worktree_observation": "审计开始时两个worktree均无tracked修改或普通untracked；此处生成状态已包含本次报告文件。"}
    snapshot['history_only_metric_files'] = len(history)
    (OUT / "snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in snapshot.items() if k not in ["branches", "inventory_generation_status"]}, ensure_ascii=False, indent=2))
    for b in branches:
        print(b)

if __name__ == "__main__":
    main()

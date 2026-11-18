"""Build report tables and native SVGs from the frozen inventory and old reports."""
import csv
import hashlib
import html
import json
import re
from pathlib import Path
from urllib.parse import quote

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
EV = OUT / 'evidence'
ROWS = json.loads((EV / 'raw_metrics.json').read_text())
HEADS = {b['ref']: b['commit'] for b in json.loads((EV / 'snapshot.json').read_text())['branches']}
LEGACY = list(csv.DictReader((ROOT / 'AI_docs/阶段性总结/SegACIL实验结果与方法对应表.csv').open(encoding='utf-8-sig')))

def fmt(x):
    return f'{x:.6f}' if isinstance(x, (int, float)) else '—'

def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(x).replace('|', '\\|').replace('\n', ' ') for x in row) + ' |' for row in rows])

def source(r):
    p = ROOT / r['path']
    if matches_blob(p, r['blob']):
        return '['+r['id']+'](<'+'../../'+r['path']+'>)'
    ref = r['refs'].split(';')[0]
    return '['+r['id']+'](https://github.com/Ruiter578/CFSSeg/blob/'+HEADS[ref]+'/'+quote(r['path'])+')'

def matches_blob(path, blob):
    if not path.is_file():
        return False
    data = path.read_bytes()
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == blob

def raw_table(rows):
    return table(['run / 配置', 'step', 'old %', 'new %', 'all %', '原始证据'],
                 [[r['run'],r['step'],fmt(r['old_pct']),fmt(r['new_pct']),fmt(r['all_pct']),source(r)] for r in rows])

def final_rows(predicate):
    return [r for r in ROWS if r['step'] == 'step1' and predicate(r)]

def legacy_table(start, end):
    return table(['ID', '方法 / 实验', '协议', 'old %', 'new %', 'all %', '来源等级'],
                 [[r['证据编号'],r['实验名称'],r['协议']+' '+r['设置'],r['旧类mIoU(%)'],r['新类mIoU(%)'],r['全部mIoU(%)'],'B：历史CSV']
                  for r in LEGACY if start <= int(r['证据编号'][1:]) <= end])

def old_pseudo_tables(start, end):
    p = ROOT / 'AI_docs/idea构思与实验设计/自适应伪标签阈值/7月新篇/7-19_自适应伪标签阈值全阶段总结与小白版复盘.md'
    s = p.read_text()
    s = s[s.index(start):s.index(end)]
    out = ['来源：[7-19历史全阶段复盘](<../idea构思与实验设计/自适应伪标签阈值/7月新篇/7-19_自适应伪标签阈值全阶段总结与小白版复盘.md>)。以下仅提取已完成实验的表，0–1 mIoU已乘100；不沿用其过时的W0/W1待做段。\n']
    for line in s.splitlines():
        if line.startswith('#'):
            out.extend(['', '#### '+re.sub(r'^#+\s*(?:5\.\d+\s*)?', '', line), ''])
        elif line.startswith('|'):
            cells = line.split('|')
            for i,c in enumerate(cells):
                if re.fullmatch(r'\s*0\.\d+\s*',c):
                    cells[i]=' '+f'{float(c)*100:.6f}'+' '
            out.append('|'.join(cells))
        elif out and out[-1].startswith('|'):
            out.append('')
    return '\n'.join(out)

def svg_text(x,y,lines,size=18,color='#172B4D',weight=400):
    return '<text x="%s" y="%s" fill="%s" font-size="%s" font-weight="%s">%s</text>' % (
        x,y,color,size,weight,''.join('<tspan x="%s" dy="%s">%s</tspan>'%(x,0 if i==0 else size*1.65,html.escape(line)) for i,line in enumerate(lines)))

def diagrams():
    assets=OUT/'assets'; assets.mkdir(exist_ok=True)
    parts=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1260 900" role="img" aria-labelledby="title desc"><title id="title">SegACIL方法与结果地图</title><desc id="desc">历史基线、已完成方法、负结果、未实验候选与独立clean协议的分层总结。</desc><style>text{font-family:"Noto Sans CJK SC","Microsoft YaHei",sans-serif}</style><rect width="1260" height="900" fill="#F7FAFF"/>']
    parts.append(svg_text(36,53,['从冻结特征到闭式分割：研究走到哪里'],32,weight=700))
    parts.append(svg_text(36,88,['2026-10-24 · A=原始JSON可读，B=历史报告值；结果单位：mIoU %'],17))
    parts.append('<rect x="180" y="115" width="900" height="70" rx="12" fill="#2458C6"/>')
    parts.append(svg_text(220,158,['DeepLab特征  →  固定RHL  →  C-RLS  →  像素类别'],25,'white',600))
    panels=[
        ('复现与工程基座','#087F75',['V3 15-5：69.4606 / 69.5598  [A]','V3+ + aspp_up：70.3565  [B]','基线可保留；恢复权重后重放'],['架构升级 ≠ 新的解析方法']),
        ('RHL单模型改造','#8C5B00',['正交BOA：69.6387  [A]','比同组基线 +0.0789 pp','只完成单点，保留有限复核'],['反向成对：68.3424 / 68.1668']),
        ('旧集成路线','#A12E50',['P0 class-wise：69.5382  [B]','未超过强单模型69.5598','SE2.0 / Snapshot未完成'],['需求已取消 → 暂停新增实验']),
        ('归一化与像素采样','#A12E50',['L2：69.4505；LN：69.4150 [A]','L2√d：69.3036；无有效增益','V3+ cap：63.7635 / 65.3718 [B]'],['保留负结果；停止重复扫描']),
        ('伪标签机制止损','#A12E50',['fixed0.6：overlap 70.7731 [B]','类别阈值净增益仅 +0.0020 pp','W1均值 +0.0190 / +0.0339 pp'],['辅助伪标签保留；阈值/W1冻结']),
        ('尚未成立的方法','#8C5B00',['PGH / PowerNorm / CA-C-RLS','大多仍是设计或零散基础代码','soft target / 联邦：没有本仓结果'],['先提出可证伪假设，再写新代码'])
    ]
    for i,(title,color,body,foot) in enumerate(panels):
        x=36+(i%3)*410;y=225+(i//3)*250
        if i<3:parts.append(f'<path d="M630 185 V205 H{x+184} V225" stroke="#ACBEDC" fill="none" stroke-width="2"/>')
        parts.append(f'<rect x="{x}" y="{y}" width="390" height="225" rx="10" fill="white" stroke="#C9D6EA"/><rect x="{x}" y="{y}" width="6" height="225" fill="{color}"/>')
        parts.append(svg_text(x+20,y+38,[title],25,color,650))
        parts.append(svg_text(x+20,y+80,body,18))
        parts.append(svg_text(x+20,y+192,foot,17,color,550))
    parts.append('<rect x="36" y="752" width="1210" height="119" rx="10" fill="#E6EEFE"/>')
    parts.append(svg_text(58,785,['独立clean holdout：不能与上面的历史约70%横向比较'],23,'#2458C6',700))
    parts.append(svg_text(58,823,['E1.1 decoder均值59.6412 > aspp59.1228 [A]；E1.2 eps1e-3=59.7310，eps0/1e-4均未提升。','现有191份二维JSON + 110条历史汇总记录有重叠；当前checkout没有模型权重。'],18))
    parts.append('</svg>');(assets/'methods-map.svg').write_text(''.join(parts))
    p=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 690" role="img" aria-labelledby="title desc"><title id="title">研究路线选择图</title><desc id="desc">保留类增量分割基座，选择单模型机制或联邦持续分割；先验证问题，再扩实验。</desc><style>text{font-family:"Noto Sans CJK SC","Microsoft YaHei",sans-serif}</style><rect width="1200" height="690" fill="#F7FAFF"/>']
    p.append(svg_text(35,50,['保留分割基座，先证明新的问题'],34,weight=700))
    p.append('<rect x="170" y="88" width="860" height="93" rx="12" fill="#2458C6"/>')
    p.append(svg_text(205,126,['VOC类增量分割 + 冻结特征 + 可加和解析统计','共同前置：恢复权重与环境 / 固定clean协议 / 独立证据账本'],23,'white',550))
    p.append('<path d="M600 181 V211 H310 V241 M600 211 H890 V241" stroke="#8EA9D8" stroke-width="3" fill="none"/>')
    for x,title,body,foot in [(35,'路线A：原任务单模型',['标签可见性 / 固定预算语义特征','或统计压缩：只选一个主机制','BOA弱信号，仅作有限复核'],'先测瓶颈，再做3seed配对'),(625,'路线B：联邦持续分割',['数据留客户端；新增空间与时间异质性','AFL / AFCL / FISS已存在','候选：背景语义冲突或统计通信成本'],'不必换任务；必须有新增贡献')]:
        p.append(f'<rect x="{x}" y="241" width="540" height="240" rx="12" fill="white" stroke="#C9D6EA"/>')
        p.append(svg_text(x+25,285,[title],28,'#087F75',700));p.append(svg_text(x+25,332,body,21));p.append(svg_text(x+25,454,[foot],20,'#8C5B00',600))
    p.append('<path d="M305 481 V510 H600 M895 481 V510 H600 V539" stroke="#8EA9D8" stroke-width="3" fill="none"/>')
    p.append('<rect x="180" y="539" width="840" height="110" rx="12" fill="#E4F3F0"/>')
    p.append(svg_text(220,579,['1–2周：正确性 + 问题差距 + 新颖性检验','通过 → 多seed/长链/第二数据集；不通过 → 归档止损'],24,'#087F75',600));p.append('</svg>')
    (assets/'route-map.svg').write_text(''.join(p))

def main():
    base_runs={'1128','1128_trs','20260606','20260607','0615_step1_bs16_trs','20260621_baseline_bs16_16_trs','20260621_baseline_bs64_32','20260617_bs48_step1_8196'}
    tables={
        'baselines':raw_table([r for r in ROWS if r['run'] in base_runs and ((r['task']=='15-1' and r['step']=='step5') or (r['task']=='15-5' and r['step']=='step1'))]),
        'buffer':raw_table(final_rows(lambda r:'buffer_' in r['run'] or 'buffer_gamma' in r['run'])),
        'normalization':raw_table(final_rows(lambda r:'_rhl_' in r['run'] and any(k in r['run'] for k in ['rhl_l2','rhl_ln','rhl_none']))),
        'ensemble':legacy_table(28,46),
        'boa':raw_table(final_rows(lambda r:'p1_boa' in r['run'])),
        'v3plus':legacy_table(51,61),
        'pseudo_all':old_pseudo_tables('### 5.2','### 5.7'),
        'matched':old_pseudo_tables('### 5.7','## 6.'),
        'clean':raw_table(final_rows(lambda r:'20260718_clean_e1_1' in r['run'])),
        'epsilon':raw_table(final_rows(lambda r:'20260719_clean_e1_2' in r['run'] or r['run']=='20260718_clean_e1_1_replay_from_main_gpu2_decoder_b8224_g1_rhl2')),
        'all_raw':table(['ID / 原始JSON','run','任务 / setting / step','split','old %','new %','all %'],
                        [[source(r),r['run'],r['task']+' / '+r['setting']+' / '+r['step'],r['split'],fmt(r['old_pct']),fmt(r['new_pct']),fmt(r['all_pct'])] for r in ROWS]),
        'all_legacy':'### 历史阶段总表110条记录\n\n来源：[原CSV](<../阶段性总结/SegACIL实验结果与方法对应表.csv>)。这些行保留历史报告精度；即使部分数字与A类重合，也不增加独立样本数。\n\n'+legacy_table(1,110)
    }
    historical=json.loads((EV/'history_only_metrics.json').read_text())
    tables['history']=table(['ID / Git原始JSON','run','step','old %','new %','all %','dirty / 处置'],
        [['['+h['id']+'](https://github.com/Ruiter578/CFSSeg/blob/'+h['source_commit']+'/'+quote(h['path'])+')', h['run'],h['step'],fmt(h['old_pct']),fmt(h['new_pct']),fmt(h['all_pct']),str(h['git_dirty'])+' / 已替代'] for h in historical])
    path=OUT/'03_方法与实验完整回顾.md';s=path.read_text()
    for key,value in tables.items():
        marker='<!-- TABLE:'+key+' -->';end='<!-- END:'+key+' -->'
        s=re.sub(re.escape(marker)+r'(?:.*?'+re.escape(end)+')?',lambda _:marker+'\n\n'+value+'\n\n'+end,s,flags=re.S)
    path.write_text(s)
    per=[]
    for r in ROWS:
        for c,v in r['per_class'].items():
            per.append({'result_id':r['id'],'run':r['run'],'step':r['step'],'split':r['split'],'class_id':c,'iou_pct':v*100,'source':r['path']})
    with (EV/'per_class_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=per[0].keys());w.writeheader();w.writerows(per)
    docs=list(csv.DictReader((EV/'document_inventory.csv').open(encoding='utf-8-sig')))
    content=['# 文档逐篇目录（112个内容版本，97个路径）','',
             '路径相同但blob不同单列；相同blob在不同路径出现属于改名/副本候选。分类是建议，不代表已经移动。优先读本次02报告中的结论级清单。','']
    rows=[]
    for d in docs:
        href=('../../../'+d['path']) if matches_blob(ROOT/d['path'], d['blob']) and 'main' in d['refs'].split(';') else ('https://github.com/Ruiter578/CFSSeg/blob/'+HEADS[d['refs'].split(';')[0]]+'/'+quote(d['path']))
        rows.append([d['id'],'['+d['path']+'](<'+href+'>)',d['refs'],d['blob'][:10],d['proposed_directory'],d['decision']])
    content.append(table(['ID','原文','分支','blob','建议归属','阅读价值'],rows))
    (EV/'文档逐篇目录.md').write_text('\n'.join(content))
    diagrams()
    (EV/'legacy_results.json').write_text(json.dumps(LEGACY,ensure_ascii=False,indent=2))
    print('Built',len(tables),'tables;',len(per),'per-class rows;',len(docs),'document versions; 2 SVG maps')

if __name__=='__main__':
    main()

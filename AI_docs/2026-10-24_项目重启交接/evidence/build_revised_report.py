"""Build the reader-facing methods report from the existing result inventory."""
import html
import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
EV = OUT / 'evidence'
RAW = json.loads((EV / 'raw_metrics.json').read_text())
OLD = (OUT / '03_方法与实验完整回顾.md').read_text()
LEGACY = json.loads((EV / 'legacy_results.json').read_text())
R = {r['id']: r for r in RAW}
E = {r['证据编号']: r for r in LEGACY}
INDEX = {}
TABLES = []
STAR = '<sup>*</sup>'


def table(headers, rows):
    TABLES.append({'headers': headers, 'rows': rows})
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows])


def number(v):
    return f'{v:.6f}' if isinstance(v, (int, float)) else '—'


def record(key, label, group, protocol=None):
    if key.startswith('R'):
        r = R[key]
        values = [r['old_pct'], r['new_pct'], r['all_pct']]
        only_json, per = True, r['per_class']
        protocol = protocol or f"{r['task']} {r['setting']}"
    else:
        r = E[key]
        values = [float(r[k]) if r[k] else '' for k in ['旧类mIoU(%)', '新类mIoU(%)', '全部mIoU(%)']]
        protocol = protocol or r['协议'] + ' ' + r['设置']
        match = next((x for x in RAW if x['task'] == r['协议'] and x['setting'] == r['设置'] and x['step'] == r['最终步骤']
                      and abs(x['all_pct'] - values[2]) < .000051
                      and isinstance(x['old_pct'], (int, float)) and abs(x['old_pct'] - values[0]) < .000051), None)
        only_json, per = bool(match), match['per_class'] if match else {}
    step = r.get('step', r.get('最终步骤', 'step1'))
    if key not in INDEX:
        INDEX[key] = {'key': key, 'config': label, 'method': group, 'protocol': protocol, 'step': step,
                      'old': values[0], 'new': values[1], 'all': values[2], 'json_only': only_json, 'per_class': per}
    display = label + (STAR if only_json else '')
    formatted = [number(v) for v in values] if key.startswith('R') else [f'{v:.4f}' if isinstance(v, (int, float)) else '—' for v in values]
    return [display] + formatted


def raw_table(items, group, protocol='15-5 sequential'):
    return table(['配置', 'old %', 'new %', 'all %'], [record(k, label, group, protocol) for k, label in items])


def legacy_table(start, end, group):
    return table(['配置', 'old %', 'new %', 'all %'], [record(f'E{i:03d}', E[f'E{i:03d}']['实验名称'], group) for i in range(start, end + 1)])


def old_tables(marker):
    chunk = OLD.split('<!-- TABLE:' + marker + ' -->')[1].split('<!-- END:' + marker + ' -->')[0]
    found, current = [], []
    for line in chunk.splitlines() + ['']:
        if line.startswith('|'):
            current.append([c.strip() for c in line.strip('|').split('|')])
        elif current:
            found.append((current[0], current[2:])); current = []
    return found


PSEUDO = old_tables('pseudo_all')
MATCHED = old_tables('matched')


def pseudo_table(index, group, protocol, keep=None):
    _, rows = PSEUDO[index]
    rows = rows if keep is None else [rows[i] for i in keep]
    output = []
    for row in rows:
        _, config, all_, old, new, acceptance, _ = row
        key = f'P{index}-{config}'
        INDEX[key] = {'key': key, 'config': config, 'method': group, 'protocol': protocol, 'step': 'step1',
                      'old': float(old), 'new': float(new), 'all': float(all_), 'json_only': False, 'per_class': {}}
        output.append([config, old, new, all_, acceptance])
    return table(['配置', 'old %', 'new %', 'all %', '接收率'], output)


def summary_row(label, protocol, old, new, all_, paper=None, star=False):
    delta = f'{all_ - paper:+.4f}' if paper is not None else '—'
    return [label + (STAR if star else ''), protocol, f'{old:.4f}', f'{new:.4f}', f'**{all_:.4f}**', f'{paper:.4f}' if paper is not None else '—', delta]


def main():
    parts = ['# 基于闭式解的解析持续学习\n\n## 1. 二维实验总览\n\n'
             '结果为 PASCAL VOC 2012 的 mIoU（%），差值单位为百分点（pp）。上标<sup>*</sup>表示当前跨服务器同步记录中只保留了该实验的JSON结果及配套配置。\n\n'
             '### 1.1 原论文结果\n']
    paper = [('15-5', 'sequential', 78.1, 42.0, 70.0), ('15-1', 'sequential', 78.1, 42.0, 70.0),
             ('15-1', 'disjoint', 77.66, 40.33, 68.77), ('10-1', 'disjoint', 70.85, 42.13, 57.17),
             ('15-1', 'overlap', 79.16, 38.0, 69.36), ('10-1', 'overlap', 75.02, 41.20, 58.91)]
    parts.append(table(['任务', '设置', 'old %', 'new %', 'all %'], [[t,s,f'{o:.2f}',f'{n:.2f}',f'**{a:.2f}**'] for t,s,o,n,a in paper]))
    parts.append('配置：`15-5`先学15个前景类，再一次加入5类；`15-1`分5次各加入1类；`10-1`分10次各加入1类。背景类始终单独计入。模型为DeepLabV3–ResNet101。原论文表III、IV的结果见[CFSSeg原文](https://arxiv.org/html/2412.10834v2#S5)。\n\n'
                 '### 1.2 各已实验方法的最佳结果\n\n“最佳”取该方法已完成配置中的最高all；多seed均值在对应方法小节列出。')
    summary = []
    for label, rid, protocol in [('CFSSeg基础复现','R004','15-5 sequential'), ('CFSSeg长链复现','R016','15-1 sequential'),
            ('Buffer/γ搜索','R139','15-5 sequential'), ('RHL-Norm：特征归一化','R023','15-5 sequential'),
            ('BOA-RHL：分块正交随机映射','R055','15-5 sequential')]:
        r = R[rid]; summary.append(summary_row(label, protocol, r['old_pct'], r['new_pct'], r['all_pct'], 70.0, True))
    summary.extend([
        summary_row('RHL-SE：随机头集成','15-5 sequential',77.9811,42.5228,69.5386,70),
        summary_row('DLV3+：分割架构与特征源升级','15-5 sequential',77.9276,46.1287,70.3565,70),
        summary_row('Class-cap：类别像素限额采样','15-5 sequential',71.0924,47.0660,65.3718,70),
        summary_row('FPL：固定阈值伪标签','15-1 overlap',79.6171,40.9558,70.4120,69.36),
        summary_row('FPL：固定阈值伪标签','15-5 overlap',79.6761,42.2836,70.7731),
        summary_row('FPL：固定阈值伪标签','15-5 disjoint',77.8582,42.6021,69.4639),
        summary_row('BGQ：批内全局分位阈值','15-5 overlap',79.6345,42.2601,70.7358),
        summary_row('BCQ：批内类别分位阈值','15-5 overlap',79.6758,42.2691,70.7694),
        summary_row('APT：离线自适应类别阈值','15-5 overlap',79.7185,42.2949,70.8082),
        summary_row('APT：离线自适应类别阈值','15-5 disjoint',77.8822,42.6233,69.4872),
        summary_row('MG：总体分位点匹配的全局阈值','15-5 overlap',79.7124,42.2965,70.8038),
        summary_row('MG：总体分位点匹配的全局阈值','15-5 disjoint',77.8822,42.6237,69.4873),
        summary_row('RW-C-RLS：可靠性加权','15-5 overlap',79.6944,42.3291,70.7979),
        summary_row('RW-C-RLS：可靠性加权','15-5 disjoint',77.8475,43.0866,69.5711),
        summary_row('CV：独立验证集特征源选择','15-5 sequential / clean',R['R183']['old_pct'],R['R183']['new_pct'],R['R183']['all_pct'],star=True),
        summary_row('Tail-ε：新增类别初始化调整','15-5 sequential / clean',R['R191']['old_pct'],R['R191']['new_pct'],R['R191']['all_pct'],star=True)
    ])
    parts.append(table(['方法', '协议', 'old %', 'new %', '最佳 all %', '论文 all %', 'Δall pp'],summary))
    parts.append('配置：每行在该方法内部选择最优配置；RW-C-RLS在overlap中最佳为confidence权重，在disjoint中为confidence×margin。Tail-ε行取新尝试的ε=0、1e-4中较好的1e-4；原ε=1e-3对照为59.7310%。`clean`为独立训练留出集。原论文没有15-5 overlap/disjoint或clean结果，对应论文差值记“—”。\n\n'
                 '### 1.3 总结\n\n'
                 '**架构升级是目前幅度最明确的改进。** DeepLabV3+结合上采样ASPP特征达到70.3565%，相对论文表列70.0%高0.3565 pp；新类达到46.1287%，比论文42.0%高4.1287 pp。V3+仅使用decoder时只有68.9663%，说明收益依赖特征源选择。\n\n'
                 '**随机特征侧的收益仍然较小。** Buffer搜索的最佳单次为70.0157%，但8224维的三seed均值为69.9014%；BOA正交映射比同组默认映射提升0.0789 pp。固定归一化没有带来有效增益，随机头集成的最佳结果69.5386%仍低于69.5598%的强单模型。\n\n'
                 '**伪标签的主要收益来自补回更多旧类像素。** 固定0.6阈值在15-5 overlap/disjoint上相对不使用伪标签分别提升0.4640/0.5201 pp；15-1 overlap最终达到70.4120%，比论文同协议69.36%高1.0520 pp。类别自适应阈值虽达到70.8082%，但与总体分位点匹配的全局阈值比较，六组平均只高0.0020 pp；类别阈值本身贡献很小。\n\n'
                 '**可靠性加权没有形成稳定净收益，clean实验则确定了后续比较起点。** W1两种权重平均提升仅0.0190/0.0339 pp，并表现出旧类与新类之间的权衡。独立验证集上，decoder三seed均值比aspp高0.5184 pp；将新增列初始化ε从1e-3改为0或1e-4均未提升。后续宜保留V3+特征适配和固定伪标签，优先研究语义特征或解析目标的改造。\n\n'
                 '![已实验方法及结果总结](assets/methods-map-revised.svg)\n\n'
                 '## 2. 实验协议与指标\n\n'
                 '`sequential`保留已见旧类与当前新类标注；`disjoint`将当前步的旧类标为背景，且图像按任务分开；`overlap`允许图像跨任务出现，背景还可能包含未来类别。本文的overlap对应原论文的overlapped。')
    parts.append(table(['评价集合', '典型all', '用途'],[
        ['VOC官方验证集（代码中命名为test）','约69–71%','复现、历史方法与参数比较'],
        ['按当前任务过滤的旧验证集','约44–45%','观察到类别覆盖不全，已由clean方案替代'],
        ['独立clean holdout','约59%','从训练池留出10%，用于选择特征源与超参数'],
        ['官方验证集全类复评','约69–70%','检查旧验证集的排序与类别覆盖']]))
    parts.append('配置：历史官方验证集与clean holdout使用不同的图片和step0模型，分别比较。15-5最终`old`为0–15共16类（含背景），`new`为16–20共5类，`all=(16×old+5×new)/21`。15-1中间步的new为截至该步累计学到的新类。\n\n'
                 '原论文sequential表列78.1/42.0/70.0；前两项按16∶5加权约为69.50，与表列all存在约0.50 pp差异。本文保留论文原值70.0作对照，各方法的实际改进幅度主要看同组实验。\n\n'
                 '## 3. 原方法复现与batch调整\n\n'
                 'CFSSeg先在step0训练DeepLab特征提取器，增量阶段冻结特征，通过固定随机隐层RHL将像素特征映射到高维，再用C-RLS递归岭回归更新分类头。15-5只进行一次增量更新，15-1进行五次；复现实验检验基础流程，batch调整则观察数据分批、随机增强和数值计算对结果的影响。第3–8节均为sequential；除基础复现中的15-1外，其余为15-5。\n\n'
                 '### 3.1 基础复现')
    base = [('R010','15-1_local'),('R016','15-1_trs'),('R019','15-5_bs64'),('R021','15-5_bs32'),
            ('R003','15-5_bs16_trs_eval1'),('R004','15-5_bs16_trs_eval2'),('R061','15-5_base16_bs16_trs'),('R063','15-5_base64_bs32')]
    parts.append(table(['配置','old %','new %','all %'],[record(k,n,'基础复现') for k,n in base]))
    parts.append('配置：`local/trs`区分运行服务器；`bsN`为增量阶段batch size；`base16/base64`表示step0的batch size；`eval1/eval2`表示同一实验目录的两次评估。15-1取最终step5，15-5取最终step1；这组默认buffer为8196、γ=1。\n\n'
                 '**结论：** 15-5基础复现落在69.46–69.60%，15-1两条完整链为68.9464%和69.5079%。改变batch会带来小幅波动，但没有改变基础方法的性能区间。\n\n'
                 '### 3.2 batch与局部buffer调整\n\n'
                 '这组实验沿用原方法，改变增量batch和随机隐层输出维度，检查训练设置敏感性。buffer越大表示高维随机特征越多，但并不保证结果单调提高。')
    tuning=[('R037','bs48_legacy'),('R045','bs32_b8192'),('R047','bs32_b8196'),('R049','bs48_legacy_repeat'),('R051','bs48_b8196'),
        ('R065','trs_b8204_bs32_repeat1'),('R067','local_b8216_bs32'),('R069','trs_b8216_bs32_repeat1'),('R071','trs_b8200_bs32_repeat1'),
        ('R073','trs_b8200_bs32_repeat2'),('R075','base64_bs32_b8200'),('R077','trs_b8208_bs32'),('R079','trs_b8196_bs32'),
        ('R081','trs_b8192_bs32'),('R083','trs_b8204_bs32_repeat2'),('R085','trs_b8212_bs32'),('R087','trs_b8216_bs32_repeat2')]
    parts.append(raw_table(tuning,'batch与局部buffer'))
    parts.append('配置：`bN`为buffer维度，`bsN`为增量batch，`base64`为step0 batch64，`repeatN`为相同名义配置的重复运行。`legacy`保留当时旧buffer设置，不能等同于显式指定b8196。TRS这组局部搜索复用base16模型；γ固定为1。\n\n'
                 '**结论：** bs48没有优于bs32，buffer的微调呈非单调波动。局部调整中b8208达到69.7343%，随后用统一队列进一步比较buffer和γ。\n\n'
                 '## 4. Buffer与γ超参数搜索\n\n'
                 'Buffer控制RHL输出维度，γ是岭回归正则强度。较大的buffer增加非线性表示容量，也增加矩阵计算成本；γ限制分类头权重幅值。先在相同设置下扫描维度和γ，再用多个独立RHL随机种子判断结果是否稳定。\n\n'
                 '### 4.1 单seed维度与正则搜索')
    phasea=[]
    for r in RAW:
        if r['step']=='step1' and 'phaseA_' in r['run']:
            phasea.append((r['id'],r['run'].split('phaseA_')[1].removesuffix('_trs')))
    parts.append(raw_table(phasea,'Buffer与γ搜索'))
    parts.append('配置：`b8196`表示8196维随机特征；`g0p1/g1/g10`表示γ=0.1/1/10；`rhlm1`表示rhl_seed=-1，即沿用全局随机状态。所有行增量bs32、norm=none，复用同一base16模型。\n\n'
                 '**结论：** γ从0.1变到10几乎不改变结果；buffer局部变化的影响更大，但没有单调趋势。b8208是这一轮最优点，all=69.7344%。\n\n'
                 '### 4.2 多seed与右侧维度扩展')
    mult=[]
    for r in RAW:
        if r['step']=='step1' and any(v in r['run'] for v in ['phaseB_','right_seed_search','final_confirmation']):
            m=re.search(r'b(\d+)_g(\w+?)_rhl(\d+)',r['run'])
            label=f'b{m[1]}_g{m[2]}_rhl{m[3]}' + ('_repeat' if 'final_confirmation' in r['run'] and m[1]=='8224' else '')
            mult.append((r['id'],label))
    parts.append(raw_table(mult,'Buffer多seed'))
    parts.append('配置：`rhl1…rhl5`只改变RHL映射的随机种子；`repeat`为重复确认。其余均为bs32、γ=1、norm=none和同一base16模型。b8228–b8240是向更大维度的扩展。')
    means=[]
    for b,ids in [(8208,['R115','R117','R119','R135','R137']),(8224,['R121','R123','R125','R139','R141']),
                  (8224,['R123','R139','R141']),(8232,['R129','R145','R147'])]:
        means.append([f'b{b} / rhl'+('1–5' if len(ids)==5 else '2、4、5')+STAR]+[number(sum(R[i][field] for i in ids)/len(ids)) for field in ['old_pct','new_pct','all_pct']])
    parts.append(table(['配置与种子范围','平均old %','平均new %','平均all %'],means))
    parts.append('配置：前两行使用rhl_seed=1、2、3、4、5；后两行使用rhl_seed=2、4、5，按相同种子配对比较。\n\n'
                 '**结论：** 单次最高为b8224、rhl4的70.0157%。8224对8232的三seed均值高0.0789 pp，对8208的五seed均值高约0.0612 pp。8224适合作为固定起点，继续逐个维度搜索的收益有限。\n\n'
                 '## 5. RHL-Norm：随机特征归一化\n\n'
                 '归一化作用于RHL与ReLU之后、解析分类头之前。L2将每个像素特征缩放到单位长度；L2√d再乘随机特征维度d的平方根；LayerNorm同时消除通道均值并调整方差。目的是减少像素间幅值差异，使岭回归处理更一致的特征尺度。')
    norm=[('R035','rhl_none_g1_bs64'),('R025','rhl_l2_g1_bs64'),('R033','rhl_ln_g1_bs64'),('R027','rhl_l2sqrt_g0p1_bs64'),
          ('R031','rhl_l2sqrt_g1_bs64'),('R029','rhl_l2sqrt_g10_bs64'),('R023','rhl_l2sqrt_g1_retry_bs32')]
    parts.append(raw_table(norm,'RHL-Norm'))
    parts.append('配置：`rhl`指随机隐层输出；`none/l2/ln/l2sqrt`依次为不归一化、L2、无可学习仿射参数的LayerNorm、L2√d；`g0p1/g1/g10`表示γ=0.1/1/10；`bs64/bs32`表示增量batch；`retry`是重跑标记，不是新算法，原`rhl_l2sqrt_g1_retry`实际使用bs32。\n\n'
                 '**结论：** 相同bs64下，none的69.4606%最高，L2基本持平，LayerNorm和L2√d下降；γ搜索没有改善L2√d。bs32重跑的69.5146%也低于同batch默认映射的69.5598%。完全压平幅值可能削弱特征中的有效强度信息，后续若继续归一化，应考虑部分保留幅值，而不是重复强归一化扫描。\n\n'
                 '## 6. RHL-SE：随机隐层子空间集成\n\n'
                 'RHL-SE固定同一个特征提取器，训练不同随机映射对应的解析分类头，再融合各头输出。等权融合直接平均；全局加权对模型分配固定权重；类别或old/new分组加权按类别调整模型贡献；confidence gate根据当前像素的置信度选择或加权成员。它依靠不同随机头的错误互补获得提升。')
    parts.append(legacy_table(28,46,'RHL-SE'))
    parts.append('配置：`seed1/2/3`为独立RHL种子，成员训练时全局seed1、batch64、buffer8196、γ=1。`K2`融合seed2/3，`K3`融合seed1/2/3；`logit`融合激活前分数，`prob`融合类别概率分数。K3加权logit使用0.2/0.4/0.4；加权概率A使用0.1/0.45/0.45，B使用0.2/0.4/0.4，均按seed1/2/3顺序。`old/new`为旧类偏seed1、新类偏seed2/3的两组权重；手工`class-wise`按逐类测试诊断设权重，“验证集驱动”则按验证表现选择权重后固定评估。置信度门控以seed1为主模型、门限0.10、融合权重2/4/4；CPU表示在CPU上复评。表中batch1/16/32均为评估batch；两条同名batch1记录是重复评估。\n\n'
                 '**结论：** 最佳融合结果为69.5386%，比最佳成员seed3的69.4989%高0.0397 pp，但低于69.5598%的强单模型。成员之间高度相似，互补空间小；增加融合规则没有产生足够收益，暂不继续扩展。\n\n'
                 '## 7. BOA-RHL：分块正交与反向成对随机映射\n\n'
                 'BOA-RHL（Block-Orthogonal Antithetic RHL）改变随机投影矩阵的构造：分块正交使同一块内投影方向尽量不重复；反向成对用W与−W同时投影，使ReLU保留输入在正负方向的响应。在固定总维度下，成对结构也会减少不同投影轴的数量。')
    parts.append(raw_table([('R053','gaussian_legacy_seed1_bs32'),('R055','orthogonal_legacy_seed1_bs32'),
                            ('R057','orthogonal_antithetic_legacy_seed1_bs32'),('R059','orthogonal_antithetic_kaiming_seed1_bs32')],'BOA-RHL'))
    parts.append('配置：`gaussian`是兼容名称，legacy路径实际使用PyTorch Linear默认初始化；`orthogonal`表示分块正交；`antithetic`表示W/−W成对。`legacy/kaiming`分别使用原始尺度和较大的Kaiming尺度；`seed1`为RHL seed1，`bs32`为增量batch32。固定buffer8196、γ=1、norm=none。\n\n'
                 '**结论：** 正交映射达到69.6387%，相对同组默认映射提升0.0789 pp，新类提升0.2408 pp。反向成对两组分别降到68.3424%和68.1668%，主要退化在新类；放大尺度未改善。保留正交方向的小规模多seed复核，反向成对不再优先。\n\n'
                 '## 8. DeepLabV3+升级与类别像素采样\n\n'
                 '### 8.1 架构与解析特征源\n\n'
                 'DeepLabV3+在ASPP高层语义特征之外融合低层空间信息，并通过decoder恢复细节。解析分类头可以读取decoder输出、ASPP输出，或上采样后的ASPP特征。这里改变的是分割头结构与特征接口，ResNet101骨干保持一致。')
    parts.append(table(['配置','old %','new %','all %'],[record(f'E{i:03d}', E[f'E{i:03d}']['实验名称'], 'DeepLabV3+') for i in [51,52,53,54,55,56,57,60]]))
    parts.append('配置：`decoder`为V3+融合后的解码特征；`decoder_stride8`将decoder特征降采样到步长8；`aspp`为原ASPP空间分辨率；`aspp_up`将ASPP特征上采样后送入RHL。buffer8200/8216两行使用decoder，其余特征源对照使用8196；默认step0 batch32、step1 batch16，“step1 batch32”改为32。“主线黄金重放”使用auto→aspp_up，重跑最佳配置。')
    parts.append(raw_table([], 'unused') if False else '**结论：** aspp_up最高，为70.3565%；比同架构decoder的68.9663%高1.3902 pp。aspp为69.9464%，decoder_stride8为68.5985%。更高层语义特征与合适空间分辨率的结合，比直接使用融合后的decoder更适合这一组解析头。\n\n'
                 '### 8.2 Class-cap：按类别限制像素数\n\n'
                 'Class-cap在解析拟合前对每类像素设置数量上限，减少多数类占比和高维矩阵计算量。它相当于改变用于拟合的像素分布，可能提高稀少类的相对影响，也会丢弃旧类样本。')
    parts.append(table(['配置','old %','new %','all %'],[record('E051','aspp_up_no_cap','Class-cap'),record('E058','aspp_up_cap4096','Class-cap'),record('E059','aspp_up_cap8192','Class-cap')]))
    parts.append('配置：`no_cap`使用不限制类别像素数的V3+ aspp_up基线；`cap4096/cap8192`表示拟合采样时每类最多保留4096/8192个像素；这不是buffer维度。\n\n'
                 '**结论：** cap8192的新类提高到47.0660%，但旧类降至71.0924%，all只有65.3718%；cap4096进一步降至63.7635%。旧类样本损失超过新类收益，不继续作为涨点方法。\n\n'
                 '## 9. 伪标签：从补回旧类到阈值与权重\n\n'
                 '### 9.1 为什么需要伪标签\n\n'
                 '在overlap/disjoint中，当前任务只标注新类，图片里的旧类可能被写成背景。例如前一步学过“汽车”，当前步学习“羊”，汽车像素在当前mask中也可能是背景。直接拿这张mask训练，会要求模型把汽车也预测成背景。伪标签让上一阶段模型充当teacher：当它认为某个背景像素是旧类且分数足够高，就把该像素的训练目标恢复成相应旧类；显式新类标注保持原样。\n\n'
                 '具体处理顺序是：①取当前标签为背景、teacher最高分预测为旧类的像素作为候选；②读取其最高类别分数p；③按阈值决定是否接收；④接收的像素改用teacher旧类标签，未接收的仍为背景。例如p=0.72、阈值0.6时接收，阈值0.8时不接收。**接收率是接收像素数/候选像素数，不是正确率。**\n\n'
                 '本组teacher使用逐类别sigmoid分数，最高分可用于排序，但所有类别分数之和不要求等于1。下面依次比较“是否补标签”“如何选阈值”“接收后如何分配权重”。\n\n'
                 '### 9.2 FPL：固定阈值\n\n'
                 'FPL（Fixed-threshold Pseudo-Labeling）对所有旧类采用同一个分数门槛。低阈值补回更多像素，也可能接收更多错误标签；高阈值更保守，却可能丢掉困难但有用的旧类。本组属于原有伪标签机制的开关与阈值调参。')
    parts.append(pseudo_table(0,'FPL','15-5 overlap',range(5)))
    parts.append('配置：本表为15-5 overlap、seed1、step1；`off`不使用伪标签，`fixed 0.6/0.7/0.8/0.9`分别要求p≥0.6/0.7/0.8/0.9。seed控制整次训练随机性，和前面只控制随机映射的rhl_seed不同。\n\n'
                 '**结论：** fixed0.6最好，all从70.3091%升至70.7731%，提升0.4640 pp；阈值继续升高时，接收率和结果一起下降。本组更需要旧类覆盖，而非更激进地筛除低分像素。\n\n'
                 '### 9.3 BGQ / BCQ：批内分位数阈值\n\n'
                 '分位数阈值不是固定分数，而是按候选分数的排序位置决定门槛。q=0.1表示取第10百分位，通常保留较高的约90%；q=0.7通常只保留约30%。BGQ（Batch Global Quantile）把一个batch的候选像素合在一起求门槛；BCQ（Batch Class Quantile）按teacher预测类别分组，各类分别求门槛，避免高分大类主导统一门槛。')
    parts.append(pseudo_table(0,'BGQ / BCQ','15-5 overlap',range(5,14)))
    parts.append('配置：15-5 overlap、seed1、step1；`batch_global/batch_class`分别为BGQ/BCQ；`q0.1/0.3/0.5/0.7`表示分位点；`min_conf0.6`将最终阈值设为max(分位阈值,0.6)。例如`batch_class q0.3 + min_conf0.6`先在每类中取30%分位点，再将低于0.6的门槛抬到0.6。\n\n'
                 '**结论：** BGQ最佳70.7358%，BCQ最佳70.7694%，均未超过fixed0.6的70.7731%。降低q、提高覆盖明显有利；按类别分组只缩小了与固定阈值的差距。\n\n'
                 '### 9.4 APT：离线自适应类别阈值\n\n'
                 'APT（Adaptive Pseudo-label Thresholding）先用teacher扫描训练数据，收集各旧类候选分数分布，再为每类计算分位阈值并保存到文件。训练期间直接读取这些门槛，因此称为`artifact_class`；artifact只是“离线保存的阈值文件”，不是另一种模型。与BCQ相比，门槛来自整个校准训练集，不随当前batch跳动。')
    parts.append(pseudo_table(1,'APT','15-5 overlap'))
    parts.append('配置：15-5 overlap、seed1；`artifact qX`表示按类使用离线X分位阈值；未写min_conf时为0。q越小，阈值越低、接收越多；q=0近似全接收。`min_conf0.6`为绝对分数下限，不是再保留60%的像素。\n\n'
                 '**结论：** q=0.01最好，为70.8082%，比fixed0.6高0.0351 pp；q=0.005几乎相同。将最低分数设为0.6后，结果回到fixed0.6附近，增益与额外接收的低分候选密切相关。\n\n'
                 '同样的方法用于disjoint：')
    parts.append(pseudo_table(2,'APT / FPL','15-5 disjoint'))
    parts.append('配置：15-5 disjoint、seed1、step1；`off`关闭伪标签，`fixed 0.6`使用统一分数门槛0.6，`artifact qX`按类使用离线X分位阈值。未写min_conf时为0，`min_conf0.6`将最低分数门槛设为0.6。teacher与校准数据均取disjoint对应配置。\n\n'
                 '**结论：** fixed0.6比off提高0.5201 pp；APT最佳为q=0的69.4872%，比fixed0.6再高0.0233 pp。近乎全接收优于强筛选，但伪标签提高旧类的同时会轻微降低新类。\n\n'
                 '### 9.5 固定阈值与APT的多seed比较\n\n'
                 '在相同seed下比较两种阈值，观察单次的微小增益是否重复出现；两边使用同一个teacher和基础模型。')
    for idx, setting, q in [(3,'overlap','0.01'),(4,'disjoint','0')]:
        headers, rows=PSEUDO[idx]
        parts.append(table(['seed','fixed all %','APT all %','Δall pp','fixed old %','APT old %','fixed new %','APT new %'],rows))
        parts.append(f'配置：15-5 {setting}；fixed为统一阈值0.6，APT为artifact_class、q={q}、min_conf=0；seed=1/2/3对应三组完整训练随机种子。')
        for row in rows:
            seed,fa,aa,_,fo,ao,fn,an=row
            for policy,old,new,all_ in [('fixed0.6',fo,fn,fa),(f'artifact_q{q}',ao,an,aa)]:
                key=f'M-{setting}-{seed}-{policy}'
                INDEX[key]={'key':key,'config':f'{policy}_seed{seed}','method':'伪标签多seed','protocol':f'15-5 {setting}','step':'step1','old':float(old),'new':float(new),'all':float(all_),'json_only':False,'per_class':{}}
    parts.append('**结论：** 六组均比fixed0.6略高，平均约+0.0362 pp；改善可以重复，但幅度很小。下一组实验进一步区分收益究竟来自“类别自适应”，还是“把总体阈值降低”。\n\n'
                 '### 9.6 MG：匹配总体分位点的全局阈值\n\n'
                 'MG（Matched Global threshold）保留APT校准时的总体分位门槛，但所有旧类共用它，不再分配类别专属阈值。这样，两种方法都接收约99%–100%的候选，差别主要落在“统一阈值还是分类阈值”。如果APT仍明显更好，才能说明按类别分配门槛有独立作用。')
    thresholds={'overlap':['0.447265625','0.419921875','0.443359375'],'disjoint':['0.029296875','0.048828125','0.048828125']}
    rows=[]
    for row in MATCHED[0][1]:
        setting,seed,art,glob,delta,dold,dnew=row
        rows.append([f'{setting}_seed{seed}',thresholds[setting][int(seed)-1],art,glob,delta,dold,dnew])
    for i in range(65,71):
        setting = E[f'E{i:03d}']['设置']
        seed = (i-65)%3+1
        record(f'E{i:03d}', f'matched_global_t{thresholds[setting][seed-1]}_seed{seed}', 'MG')
    parts.append(table(['配置','MG统一阈值','APT all %','MG all %','Δall pp','Δold pp','Δnew pp'],rows))
    parts.append('配置：均为15-5 step1；overlap的APT使用q=0.01，disjoint使用q=0；MG统一阈值逐seed取对应校准文件中的global_threshold。三个Δ均为APT−MG。MG匹配总体分位门槛，不要求两边接收每一个完全相同的像素。\n\n'
                 '**结论：** overlap平均只高0.004062 pp，disjoint平均低0.000051 pp，六组平均+0.002005 pp。APT相对fixed0.6的大部分改善可以用更低的总体阈值解释，类别阈值分配的净收益很小，不再继续这一阈值搜索路线。\n\n'
                 '### 9.7 W0：置信度能否判断伪标签可靠性\n\n'
                 'W0按teacher分数将候选像素分成十档，再用完整标注统计每档的正确率。confidence是最高类别分数p₁；margin是最高分与第二高分之差p₁−p₂，表示第一名领先第二名多少。两个分数都高，通常意味着teacher更加确定。')
    parts.append(table(['设置','接收像素正确率','隐藏旧类召回率','真背景误标旧类（像素）','旧类间错分（像素）'],[
        ['overlap','89.4548%','90.2821%','3,703,358','542,322'],['disjoint','77.1731%','83.3488%','7,657,741','1,629,323']]))
    parts.append('配置：15-5、seed1、MG阈值。接收像素正确率=接收伪标签中预测正确的比例；隐藏旧类召回率=被背景标注遮住的真实旧类中，被正确恢复的比例；错误像素数来自同次统计。\n\n'
                 '**结论：** confidence与confidence×margin的十档正确率呈单调趋势，桶级Spearman相关系数为1.0。分数有可靠性排序能力，主要错误是真背景被改成旧类；这为连续加权提供了动机。\n\n'
                 '### 9.8 RW-C-RLS：接收以后按可靠性加权\n\n'
                 'W1保持MG已接收的像素及其hard标签不变，只调整每个伪标签对解析拟合的影响。confidence模式用w=p₁；confidence×margin模式用w=p₁(p₁−p₂)。例如p₁=0.8、p₂=0.5，两种权重分别为0.8和0.24。显式GT像素保留权重1，ignore像素权重0，未改成伪标签的背景仍按原背景目标处理。\n\n'
                 '加权岭回归同时将像素特征z和目标y乘√w：\n\n'
                 '\\[\n\\min_W \\sum_i w_i\\|z_iW-y_i\\|_2^2+\\gamma\\|W\\|_F^2,\\qquad z_i^{\\prime}=\\sqrt{w_i}z_i,\\quad y_i^{\\prime}=\\sqrt{w_i}y_i.\n\\]\n\n'
                 '这样低置信度像素影响较小，高置信度像素影响较大；它与前面的hard threshold不同，后者决定像素是否换标签，这里决定换标签后贡献多大。')
    w1=[['overlap','MG / weight=1',79.7124,42.2965,70.8038,0],['overlap','confidence',79.6944,42.3291,70.7979,-.0059],
        ['overlap','confidence×margin',79.6653,42.3804,70.7879,-.0159],['disjoint','MG / weight=1',77.8822,42.6237,69.4873,0],
        ['disjoint','confidence',77.8750,42.8312,69.5312,.0439],['disjoint','confidence×margin',77.8475,43.0866,69.5711,.0838]]
    parts.append(table(['配置','old %','new %','all %','较MG Δall pp'],[[s+' / '+c,f'{o:.4f}',f'{n:.4f}',f'{a:.4f}',f'{d:+.4f}'] for s,c,o,n,a,d in w1]))
    for s,c,o,n,a,d in w1:
        key='W1-'+s+'-'+c; INDEX[key]={'key':key,'config':c+'_seed1','method':'RW-C-RLS','protocol':'15-5 '+s,'step':'step1','old':o,'new':n,'all':a,'json_only':False,'per_class':{}}
    parts.append('配置：15-5、seed1、bs32；overlap使用MG阈值0.447265625，disjoint使用0.029296875。`weight=1`为不加权对照；confidence与confidence×margin仅改变已接收伪标签的像素权重。\n\n'
                 '**结论：** 两种加权跨设置平均仅提升0.0190/0.0339 pp。overlap略降；disjoint改善主要来自新类，同时旧类下降。margin模式在disjoint使新类提高0.4629 pp、旧类下降0.0347 pp，说明更强降权放大了旧新类权衡。整体收益不足，暂停后续W2/W3。\n\n'
                 '### 9.9 15-1长链：固定伪标签是否持续有效\n\n'
                 '将off、fixed0.6和fixed0.7沿五个增量step连续运行，观察伪标签作用能否保持到最后一类，而不只改善一次增量。')
    parts.append(table(['配置','step1 all %','step2 all %','step3 all %','step4 all %','step5 all %'],PSEUDO[5][1]))
    parts.append('配置：15-1 overlap、seed1；off关闭伪标签，fixed0.6/0.7使用固定分数阈值。step1–5每步增加1类，all分别覆盖当时已见的17、18、19、20、21类。')
    parts.append(table(['配置','step','old %','累计new %'],PSEUDO[6][1]))
    parts.append('配置：与上表为同三条链；old始终为0–15，累计new分别为16、16–17、16–18、16–19、16–20。')
    for config,step,old,new in PSEUDO[6][1]:
        all_=next(r[int(step)] for r in PSEUDO[5][1] if r[0]==config)
        key=f'chain-{config}-{step}'; INDEX[key]={'key':key,'config':config+'_seed1','method':'固定伪标签长链','protocol':'15-1 overlap','step':'step'+step,'old':float(old),'new':float(new),'all':float(all_),'json_only':False,'per_class':{}}
    parts.append('**结论：** fixed0.6最终70.4120%，比off高0.6417 pp，比论文同协议高1.0520 pp；fixed0.7最终70.1606%。固定伪标签的增益能保持到长链末端，0.6仍优于更保守的0.7。\n\n'
                 '## 10. Clean validation：独立验证集与后续参数实验\n\n'
                 '**Clean validation是把“用于选参数的图片”从所有训练阶段完整隔离。** 从训练池划出10%（1058张）作为holdout，step0训练、step1重对齐和后续增量更新都排除这些图片；step0也重新训练。这样，验证结果反映模型在未参与拟合的图片上的表现。\n\n'
                 '做这组实验有两个目的：一是修正旧的任务过滤验证集类别覆盖不全的问题；二是在固定评价集合上选择AIR特征源、buffer和新增类别初始化。本节均使用DeepLabV3：aspp为ASPP输出，decoder为ASPP之后再经过卷积、BN和ReLU的特征，没有V3+的低层特征融合。\n\n'
                 '### 10.1 旧验证集与全类评估的排序\n\n'
                 '同一模型分别在旧任务过滤验证集和官方验证集全类评估，检查特征源排序是否受评价集合影响。')
    audit=[]
    for full, val in [('R149','R150'),('R152','R153'),('R155','R156'),('R158','R159'),('R161','R162'),('R164','R165'),('R167','R168'),('R170','R171')]:
        r=R[full]; source='aspp' if '_aspp_' in r['run'] else 'decoder'; label=f"{source}_b{r['buffer']}_g1_rhl{r['rhl_seed']}"
        audit.append([label+STAR,number(R[val]['all_pct']),number(r['old_pct']),number(r['new_pct']),number(r['all_pct'])])
        record(full,label,'特征源全类评估'); record(val,label,'旧任务过滤验证集', '15-5 sequential / task-filtered val')
    parts.append(table(['配置','旧val all %','全类old %','全类new %','全类all %'],audit))
    parts.append('配置：均为V3、bs32、γ=1；aspp/decoder为解析头输入特征源，b为buffer维度，rhl为随机映射种子。旧val按当前任务筛图，右侧三项使用官方验证集全类评估。\n\n'
                 '**结论：** 旧val上aspp看似更好，全类评估却由decoder领先。评价集合的类别覆盖会改变排序，需要用统一的clean holdout重新比较。\n\n'
                 '### 10.2 E1.1：独立验证集上的特征源与buffer\n\n'
                 '复用同一clean step0，在独立holdout上重新比较V3的aspp与decoder，并在aspp中比较8208/8224/8240维度。')
    clean=[]
    for rid in ['R173','R175','R177','R179','R181','R183','R185','R187']:
        r=R[rid]; source='aspp' if '_aspp_' in r['run'] else 'decoder';clean.append((rid,f"{source}_b{r['buffer']}_g1_rhl{r['rhl_seed']}"))
    parts.append(raw_table(clean,'Clean特征源选择','15-5 sequential / clean'))
    parts.append('配置：`aspp/decoder`为特征源；`b8208/8224/8240`为RHL维度；`g1`为γ=1；`rhl2/4/5`为独立RHL种子。step0 batch16、增量batch32、norm=none，全部使用同一个clean holdout。')
    parts.append(table(['配置','平均old %','平均new %','平均all %'],[[name+STAR]+[number(sum(R[i][f] for i in ids)/3) for f in ['old_pct','new_pct','all_pct']]
        for name,ids in [('aspp_b8224 / rhl2、4、5',['R175','R177','R179']),('decoder_b8224 / rhl2、4、5',['R183','R185','R187'])]]))
    parts.append('配置：两行均为b8224、γ=1，按rhl_seed=2、4、5的三次结果取均值。\n\n'
                 '**结论：** decoder均值59.6412%，比aspp的59.1228%高0.5184 pp。aspp新类略好，但旧类损失更大；V3后续使用decoder。aspp内部8224优于8208和8240。\n\n'
                 '### 10.3 E1.2：新增类别权重的初始化幅度\n\n'
                 '解析分类头加入新类别时，会增加新的输出权重列。ε控制这些列初始随机值的幅度；ε=0从零开始，ε=1e-4或1e-3保留不同强度的随机初值。它影响新增列初始化，γ则控制岭回归正则，两者作用不同。')
    parts.append(raw_table([('R183','decoder_b8224_g1_rhl2_eps1e-3'),('R189','decoder_b8224_g1_rhl2_eps0'),('R191','decoder_b8224_g1_rhl2_eps1e-4')],
                           'Tail-ε','15-5 sequential / clean'))
    parts.append('配置：decoder、buffer8224、γ=1、rhl_seed=2、增量bs32和同一clean step0固定，仅改变`eps`；`eps1e-3`为0.001，`eps1e-4`为0.0001，`eps0`为零初始化。\n\n'
                 '**结论：** ε=0与1e-4的all分别比1e-3低0.0192/0.0167 pp，没有提升。保留1e-3作为当前性能基线，不继续围绕这一极小差距扩大搜索。\n\n'
                 '## 11. 尚未完成的方法：保留与尝试优先级\n\n'
                 '### 优先级1：保留PGH-RHL，作为主要新方法候选\n\n'
                 '**PGH-RHL（Prototype-Guided Hybrid RHL，原型引导混合随机隐层）**在随机特征之外增加“这个像素与各类别原型有多像”的响应，让解析头同时看到随机展开和语义锚点。方案包括cosine或RBF相似度、单原型或多原型；目前停留在设计。建议先用step0可见类别建立固定原型，并与同维度随机特征比较。原型随step改变会改变特征坐标，需要先解决历史统计如何沿用，再扩展为动态原型。\n\n'
                 '### 优先级2：保留BOA 1.5，做一次有限复核\n\n'
                 '**BOA 1.5**不是另一套结构，而是对现有正交初始化补齐多seed、投影尺度与容量对照。基础初始化和runner已有，后续矩阵尚未完成。先固定buffer和特征能量比较默认/正交映射的三个配对seed；若仍只有约0.08 pp增益，就作为辅助组件保留，不扩大反向成对和大buffer搜索。\n\n'
                 '### 优先级3：保留PowerNorm与CA-C-RLS作为独立储备\n\n'
                 '**PowerNorm-RHL（幅值保留归一化）**在none与L2√d之间连续调整：将特征乘以`(√d/‖h‖)^β`，β=0保留原特征，β=1为完全归一化。它要检验“保留一部分幅值”是否比完全压平更合适；需要同时匹配特征尺度与γ。完整实现和新实验尚未完成，优先级低于语义特征改造。\n\n'
                 '**CA-C-RLS（Class-Aware C-RLS，类别感知递归岭回归）**按训练像素的类别频次分配权重，例如少数类使用较大权重，再同时对特征和目标乘√w。它与W1的区别是：CA依据类别稀缺程度，W1依据teacher对单个像素的可靠性。原方案是每像素一个标量权重，可使用共享加权Gram；只有要求事后重新调整历史类别权重时，才需要额外的类别统计。训练计数与权重流程尚未完成，可先做简单old/new组权重对照。\n\n'
                 '### 暂停：SE 2.0与Snapshot\n\n'
                 '**RHL-SE 2.0**进一步按像素可靠性选择成员；**Snapshot Analytic Ensemble（快照解析集成）**使用不同step0训练时刻的模型产生多个成员。两者都增加集成复杂度，而现有随机头集成收益有限；在不再需要集成学习主题的前提下，暂不继续。\n\n'
                 'clean协议下的新γ搜索、batch稳定性和step0学习率/loss调整属于后续基线维护；在主方法确定后按需补充，不单独作为创新方向。')
    text='\n\n'.join(parts)+'\n'
    (OUT/'03_方法与实验完整回顾_修订版.md').write_text(text)
    (EV/'revised_results.json').write_text(json.dumps(list(INDEX.values()),ensure_ascii=False,indent=2))
    (EV/'revised_tables.json').write_text(json.dumps(TABLES,ensure_ascii=False,indent=2))
    diagram()
    print('Revised report:',len(TABLES),'tables;',len(INDEX),'searchable result rows')


def diagram():
    def txt(x,y,content,size=20,color='#172b4d',weight=400):
        content=html.escape(content).replace('%*', '%<tspan baseline-shift="super" font-size="60%">*</tspan>')
        return f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" font-weight="{weight}">{content}</text>'
    p=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 1660" role="img" aria-labelledby="title desc"><title id="title">基于闭式解的解析持续学习：方法与结果</title><desc id="desc">按相同任务协议比较各方法最优结果与CFSSeg论文结果。</desc><style>text{font-family:"Noto Sans CJK SC","Microsoft YaHei",sans-serif}</style><rect width="1440" height="1660" fill="#f7faff"/>']
    p.append(txt(36,55,'基于闭式解的解析持续学习',36,weight=700))
    p.append(txt(36,91,'冻结表示 → RHL随机特征 → C-RLS解析分类头',22,color='#4f617c'))
    p.append('<rect x="36" y="118" width="1368" height="85" rx="10" fill="#172b4d"/>')
    p.append(txt(60,152,'CFSSeg论文：sequential 15-5 / 15-1 = 70.00%；overlap 15-1 = 69.36%',23,'white',600))
    p.append(txt(60,184,'下列Δ均与同协议论文all比较；15-5 overlap/disjoint及clean在论文中未报告。',18,'#dce8ff'))
    cards=[
        ('CFSSeg','原方法复现','69.6015%*','较论文 −0.3985 pp','15-5 sequential','复现性能接近，batch改变有小幅波动','#2458c6'),
        ('Buffer / γ','维度与正则搜索','70.0157%*','较论文 +0.0157 pp','15-5 sequential','单次接近论文，更多收益来自参数选择','#8c5b00'),
        ('RHL-Norm','随机特征归一化','69.5146%*','较论文 −0.4854 pp','15-5 sequential','强归一化没有带来有效提升','#a12e50'),
        ('BOA-RHL','分块正交随机映射','69.6387%*','较论文 −0.3613 pp','15-5 sequential','正交比同组基线略好，反向成对下降','#8c5b00'),
        ('RHL-SE','随机隐层集成','69.5386%','较论文 −0.4614 pp','15-5 sequential','没有超过强单模型，暂停扩展','#a12e50'),
        ('DLV3+','架构与特征源升级','70.3565%','较论文 +0.3565 pp','15-5 sequential','上采样语义特征改善新类识别','#087f75'),
        ('Class-cap','类别像素限额采样','65.3718%','较论文 −4.6282 pp','15-5 sequential','新类提高，但旧类损失过大','#a12e50'),
        ('FPL','固定阈值伪标签','70.7731%','15-5 overlap：论文未报告','15-1：70.4120%，较论文+1.0520 pp','补回旧类标签，长链增益持续','#087f75'),
        ('APT','自适应类别阈值','70.8082%','论文同协议未报告','15-5 overlap','收益主要来自覆盖率，类别阈值贡献小','#8c5b00'),
        ('BGQ','批内全局分位阈值','70.7358%','论文同协议未报告','15-5 overlap','低分位更好，仍不及固定阈值','#8c5b00'),
        ('BCQ','批内类别分位阈值','70.7694%','论文同协议未报告','15-5 overlap','按类分组略改善，未超过固定阈值','#8c5b00'),
        ('MG','匹配总体分位点的全局阈值','70.8038%','论文同协议未报告','15-5 overlap','总体低阈值可解释APT的大部分收益','#8c5b00'),
    ]
    for i,(abbr,cn,value,delta,protocol,conclusion,color) in enumerate(cards):
        x=36+i%3*464;y=229+i//3*257
        p.append(f'<rect x="{x}" y="{y}" width="440" height="235" rx="9" fill="white" stroke="#c9d6ea"/><rect x="{x}" y="{y}" width="5" height="235" fill="{color}"/>')
        p.extend([txt(x+21,y+33,abbr,25,color,700),txt(x+21,y+65,cn,20),txt(x+21,y+112,value,32,color,700),txt(x+21,y+147,delta,20,color,550),txt(x+21,y+175,protocol,16,'#4f617c'),txt(x+21,y+210,conclusion,18)])
    p.append('<rect x="36" y="1283" width="1368" height="170" rx="9" fill="#eaf1ff"/>')
    p.extend([txt(57,1318,'RW-C-RLS  可靠性加权：overlap 70.7979%；disjoint 69.5711%',23,'#2458c6',650),
              txt(57,1350,'论文无对应15-5结果；相对本组不加权对照，平均收益小，旧新类权衡明显。',19),
              txt(57,1392,'CV / Tail-ε  独立验证与初始化：59.7310%* / 59.7143%*',23,'#2458c6',650),
              txt(57,1427,'论文无同评价集结果；decoder优于aspp，减小初始化幅度未提升。',19)])
    p.append('<rect x="36" y="1475" width="1368" height="118" rx="9" fill="#e6f3ef"/>')
    p.extend([txt(57,1512,'后续保留：PGH-RHL 原型引导混合特征 → BOA 1.5 正交复核',24,'#087f75',650),
              txt(57,1555,'次级储备：PowerNorm 幅值保留归一化；CA-C-RLS 类别感知加权',23,'#087f75')])
    p.append(txt(36,1630,'* 当前同步记录仅保留JSON结果。图中为最佳单次配置；多seed比较见对应方法小节。',17,'#4f617c'))
    p.append('</svg>')
    (OUT/'assets/methods-map-revised.svg').write_text(''.join(p))


if __name__=='__main__':
    main()

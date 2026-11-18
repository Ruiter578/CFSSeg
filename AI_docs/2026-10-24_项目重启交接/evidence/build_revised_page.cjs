/* Build the standalone reader edition from the revised Markdown and result index. */
const fs = require('node:fs');
const path = require('node:path');
const deps = process.env.REPORT_NODE_MODULES || '/tmp/segacil-handoff-tools/node_modules';
const MarkdownIt = require(path.join(deps, 'markdown-it'));
const katex = require(path.join(deps, 'katex'));
const OUT = path.resolve(__dirname, '..');
const read = name => fs.readFileSync(path.join(OUT, name), 'utf8');
const esc = value => String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const safeJSON = value => JSON.stringify(value).replace(/</g, '\\u003c');
const source = read('03_方法与实验完整回顾_修订版.md');
const records = JSON.parse(read('evidence/revised_results.json'));
const tables = JSON.parse(read('evidence/revised_tables.json'));
const css = read('evidence/build_pages.cjs').match(/const CSS = `([\s\S]*?)`;/)[1];
const mathRoot = path.join(deps, 'katex/dist');
const mathCSS = fs.readFileSync(path.join(mathRoot, 'katex.min.css'), 'utf8').replace(/url\(([^)]+)\)/g, (_, raw) => {
  const file = raw.replace(/["']/g, '');
  return `url(data:font/${path.extname(file).slice(1)};base64,${fs.readFileSync(path.join(mathRoot, file)).toString('base64')})`;
});
const md = new MarkdownIt({html:true, linkify:false, typographer:false});
let heading = 0;
const toc = [];
md.renderer.rules.heading_open = (tokens, i) => {
  const id = `report-${++heading}`;
  if (tokens[i].tag === 'h2') toc.push({id, title:tokens[i + 1].content});
  return `<${tokens[i].tag} id="${id}">`;
};
md.renderer.rules.table_open = () => '<div class="table-wrap" tabindex="0" role="region" aria-label="结果表格，可横向滚动"><table>';
md.renderer.rules.table_close = () => '</table></div>\n';
md.renderer.rules.th_open = () => '<th scope="col">';
md.renderer.rules.td_open = (tokens, i) => {
  const text = tokens[i + 1].content.replace(/\*\*/g, '');
  return /^[-+−]?[\d,.]+\s*%?$/.test(text) ? '<td class="number">' : '<td>';
};
md.renderer.rules.image = () => `<figure class="map" tabindex="0" aria-label="方法与结果总结图，可横向滚动">${read('assets/methods-map-revised.svg')}</figure>`;
const formulas = [];
let body = md.render(source.replace(/^# .*\n/, '').replace(/\\\[([\s\S]*?)\\\]/g, (_, math) => {
  const i = formulas.push(katex.renderToString(math.trim(), {displayMode:true, throwOnError:true, output:'htmlAndMathml'})) - 1;
  return `\n\n<div class="math-display">FORMULA_${i}</div>\n\n`;
}));
formulas.forEach((formula, i) => {body = body.replace(`FORMULA_${i}`, formula);});

// Reuse each table's own configuration explanation in the result inspector.
const notes = new Map();
for (const match of source.matchAll(/((?:^\|.*\n)+)\s*配置：([^\n]+)/gm)) {
  for (const line of match[1].trim().split('\n').slice(2)) {
    const config = line.split('|')[1].replace(/<sup>\*<\/sup>/g, '').replace(/`/g, '').trim();
    if (!notes.has(config)) notes.set(config, []);
    notes.get(config).push(match[2]);
  }
}
const methodNotes = {
  'MG':'matched_global使用离线校准文件中的总体分位阈值；t之后是实际分数门槛，seed为完整训练随机种子。15-5 overlap使用q=0.01对应门槛，disjoint使用q=0对应门槛。所有旧类共用同一阈值。',
  '伪标签多seed':'fixed0.6为固定门槛0.6；artifact_q为离线按类别计算的分位阈值。overlap使用q=0.01，disjoint使用q=0，min_conf均为0；seed表示整次训练随机种子。',
  '固定伪标签长链':'15-1 overlap、seed1；off关闭伪标签，fixed0.6/0.7使用固定分数门槛。step1至step5每次增加1类，old为0–15，new为截至当前步累计新增的类别。',
  'RW-C-RLS':'15-5、seed1、增量batch32；MG / weight=1为不加权对照。confidence权重为最高分p₁，confidence×margin为p₁(p₁−p₂)。overlap阈值0.447265625，disjoint阈值0.029296875；显式GT权重始终为1。'
};
for (const r of records) {
  const candidates = notes.get(r.config) || [];
  r.explanation = candidates.find(note=>note.includes(r.protocol)) || candidates[0] || methodNotes[r.method] || '';
  if (!r.explanation) throw new Error(`Missing configuration explanation: ${r.key} / ${r.config}`);
}

function plainTable(headers, rows, label) {
  return `<div class="table-wrap" tabindex="0" role="region" aria-label="${esc(label)}"><table><thead><tr>${headers.map(h=>`<th scope="col">${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map((v,i)=>`<td${i>1?' class="number"':''}>${v}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}
function slide(label, title, content) {
  return `<section class="slide" data-slide-label="${esc(label)}"><h2 tabindex="-1">${title}</h2>${content}</section>`;
}
const best = tables[1].rows;
const overview = plainTable(['方法','协议','最佳 all %','论文 all %','Δall pp'], [
  [best[0][0],best[0][1],best[0][4].replace(/\*\*/g,''),'70.0000',best[0][6]],
  [best[2][0],best[2][1],best[2][4].replace(/\*\*/g,''),'70.0000',best[2][6]],
  [best[6][0],best[6][1],best[6][4].replace(/\*\*/g,''),'70.0000',best[6][6]],
  [best[8][0],best[8][1],best[8][4].replace(/\*\*/g,''),'69.3600',best[8][6]],
  [best[13][0],best[13][1],best[13][4].replace(/\*\*/g,''),'—','—']
], '代表性最佳结果与原论文比较');
const slides = [
  slide('结果总览','架构与特征源升级的收益最明确',`${overview}
    <p class="caption">配置：各方法取已完成配置中的最佳all；完整配置见下方正文。论文没有15-5 overlap结果。<sup>*</sup>当前同步记录仅保留JSON。</p>
    <div class="grid two"><div class="cell teal"><h3>值得保留</h3><p>V3+配合上采样ASPP特征；overlap/disjoint中的固定阈值伪标签；8224维作为已选参数起点。</p></div><div class="cell amber"><h3>贡献有限</h3><p>强归一化、随机头集成未超过强基线；类别自适应阈值相对匹配的全局阈值，六组平均仅提高0.0020 pp。</p></div></div>`),
  slide('方法原理','保留旧知识的统计量，再解一次分类头',`
    <p class="intro">先把每个像素变成固定特征，再记录“特征之间如何共同出现”和“特征对应什么类别”。新增数据到来时，将这些统计相加。</p>
    <div class="steps"><div class="step-box"><strong>冻结表示</strong><p>step0训练分割网络；后续像素始终使用同一套特征坐标。</p></div><span class="step-arrow" aria-hidden="true">→</span><div class="step-box"><strong>RHL 随机展开</strong><p>固定随机映射与ReLU把特征展开；buffer是展开后的维度。</p></div><span class="step-arrow" aria-hidden="true">→</span><div class="step-box"><strong>C-RLS 闭式更新</strong><p>累积特征与标签统计，递归求解带正则的线性分类头。</p></div></div>
    <div class="stat-equations"><span>G ← G + ZᵀZ</span><span>B ← B + ZᵀY</span><span>W = (G + γI)⁻¹B</span></div>
    <p class="caption">Z为像素的随机特征，Y为类别目标；G记录特征相关性，B记录特征与类别的关系，γ控制正则强度。上式为零初始化的岭回归形式，递归更新避免每一步重新拟合全部旧数据。</p>
    <div class="grid two"><div class="cell"><h3>改进特征</h3><p>V3+、RHL-Norm、BOA改变Z的质量、尺度或方向；PGH计划增加类别原型响应。</p></div><div class="cell"><h3>改进学习目标</h3><p>伪标签修正Y中被背景遮住的旧类；可靠性加权调整每个像素对统计量的贡献。</p></div></div>`),
  slide('伪标签','背景标注里的汽车，怎样重新成为汽车',`
    <p class="intro">当前任务只标“羊”，已经学过的“汽车”也可能被标为背景。上一阶段模型提供汽车预测；阈值决定是否用这个预测替换背景标签。</p>
    <div class="pseudo-demo"><div class="pixel-panel"><div class="seg-picture" aria-label="图中蓝色汽车和绿色新类别羊">${['bg','car','car','bg','bg','bg','car','car','car','car','bg','bg','bg','car','car','bg','animal','animal','bg','bg','bg','animal','animal','animal'].map(c=>`<span class="pixel ${c}"></span>`).join('')}</div><p><span class="dot"></span>旧类汽车 <span class="dot teal"></span>新类羊</p><p>当前mask：汽车 → 背景<br>teacher：汽车，分数 <output id="score-label">0.72</output></p></div><div class="pseudo-controls"><label for="teacher-score">汽车预测分数 p <output id="score-output">0.72</output></label><input id="teacher-score" type="range" min="0" max="1" step="0.01" value="0.72"><label for="pseudo-threshold">固定阈值 τ <output id="threshold-output">0.60</output></label><input id="pseudo-threshold" type="range" min="0" max="1" step="0.01" value="0.60"><div class="decision" id="pseudo-decision" role="status">p ≥ τ，接收伪标签：背景 → 汽车</div><p class="caption">接收率是候选像素中被接收的比例；它不等于伪标签正确率。</p></div></div>
    <div class="grid"><div class="cell"><h3>FPL 固定阈值</h3><p>所有旧类使用相同分数门槛。15-5 overlap中0.6比不补标签提高0.4640 pp。</p></div><div class="cell amber"><h3>APT 类别阈值</h3><p>按每类分数分布离线设置门槛；和接收数量接近的全局阈值相比，收益很小。</p></div><div class="cell amber"><h3>RW-C-RLS 加权</h3><p>保留已接收标签，降低不可靠像素的贡献；disjoint略增，overlap略降。</p></div></div>`),
  slide('独立验证','Clean validation：留出图片，贯穿所有训练阶段',`
    <p class="intro">从训练池留出10%（1058张），step0、重对齐和后续增量都不使用它们。用这些未参与拟合的图片选择特征源和超参数。</p>
    <div class="holdout-visual"><div><strong>90% 训练图片</strong><span>step0训练 → 重对齐 → 增量学习</span></div><div><strong>10% 独立验证图片</strong><span>只用于验证与参数选择</span></div></div>
    <div class="grid two"><div class="cell teal"><h3>V3特征源比较</h3><p>buffer8224，RHL种子2、4、5的平均all：</p><div class="comparison-pair"><span>aspp <b>59.1228%</b></span><span>decoder <b>59.6412%</b></span></div><p>decoder平均高0.5184 pp，主要保住了旧类。</p></div><div class="cell amber"><h3>新增类别初始化</h3><p>decoder、buffer8224、RHL seed2：</p><p>ε=1e-3：<b>59.7310%</b><sup>*</sup><br>ε=0：比基线低0.0192 pp<br>ε=1e-4：比基线低0.0167 pp</p><p>减小初始幅度没有提升。</p></div></div>
    <p class="caption">配置：均为15-5 sequential、增量batch32、γ=1；ε为新增输出列的随机初始化幅度。clean使用独立holdout；论文没有这一评价集的结果。</p>`),
  slide('后续方法','优先研究语义特征，保留有限的解析头对照',`
    <ol class="research-priority"><li><h3>PGH-RHL：原型引导混合特征</h3><p>将“像素与类别原型的相似度”接到随机特征旁边。先固定step0原型，与相同总维度的随机特征比较；目前为设计方案。</p></li><li><h3>BOA 1.5：补齐正交映射复核</h3><p>已有初始化实现，尚缺多seed和尺度匹配对照。若配对实验仍只有约0.08 pp增益，则保留为辅助组件。</p></li><li><h3>PowerNorm / CA-C-RLS：次级储备</h3><p>前者部分保留特征幅值，后者按类别频次给像素加权。分别检验尺度问题和类别不平衡，先不叠加多个改动。</p></li></ol>
    <p class="pause-note"><strong>暂停：</strong>RHL-SE 2.0、Snapshot集成；继续扩大伪标签阈值或W1权重的小网格搜索。</p>`)
];

const overrideCSS = `
.brand{letter-spacing:0}.edition{letter-spacing:0;margin-top:7px}.hero h1{max-width:1000px;font-size:clamp(32px,3.6vw,51px);letter-spacing:0}.hero p{max-width:850px}.hero-links{display:flex;gap:25px;font-size:15px}.deck-bar{align-items:center}.slide{min-height:520px}.slide h2:focus{outline:none}.slide p{max-width:880px}.slide table{min-width:650px}.slide td:first-child{min-width:155px}.deck-progress{display:flex;gap:7px;flex-wrap:wrap;margin-top:10px}.deck-progress button{font-size:13px;padding:5px 10px}.deck-progress button[aria-current=true]{background:var(--blue);color:white;border-color:var(--blue)}.deck-label{font-size:14px}.stat-equations{display:flex;flex-wrap:wrap;justify-content:space-around;gap:18px;background:#eaf1ff;padding:23px 15px;font-size:24px;font-family:Cambria,"Noto Serif",serif}.step-box p{font-size:15px;margin-bottom:0}.pseudo-demo{display:grid;grid-template-columns:1fr 1.35fr;gap:32px;padding:10px 0 22px}.pixel-panel{text-align:center;border:1px solid var(--line);border-radius:8px;padding:14px}.pixel-panel p{font-size:15px}.pixel.bg{background:#e4eaf4}.pseudo-controls{padding:12px 0}.pseudo-controls label{display:flex;justify-content:space-between;font-size:16px}.pseudo-controls input{display:block;width:100%;margin:12px 0 24px;accent-color:var(--blue)}.decision{padding:14px 17px;border-left:4px solid var(--teal);background:#e6f3ef;font-weight:650}.decision.rejected{border-color:var(--amber);background:#fff8e9}.holdout-visual{display:grid;grid-template-columns:2.1fr 1fr;gap:14px;margin:28px 0}.holdout-visual>div{padding:24px;background:#edf2fb;border-top:4px solid var(--blue)}.holdout-visual>div+div{background:#e6f3ef;border-color:var(--teal)}.holdout-visual strong,.holdout-visual span{display:block}.holdout-visual span{font-size:14px;margin-top:7px}.comparison-pair{display:flex;gap:22px;flex-wrap:wrap}.comparison-pair b{display:block;font-size:25px;color:var(--blue)}.research-priority{padding-left:28px;max-width:900px}.research-priority li{padding:0 0 10px 8px}.research-priority li::marker{font-size:23px;font-weight:700;color:var(--blue)}.research-priority h3{font-size:21px;margin:15px 0 5px}.research-priority p{margin:5px 0 12px;font-size:16px}.pause-note{border-top:1px solid var(--line);padding-top:18px;font-size:15px;color:var(--muted)}.report-section>h2{font-size:30px}.report table{font-variant-numeric:tabular-nums}.report .table-wrap:nth-of-type(2) th{background:#172b4d;color:white}.report tbody tr:hover{background:#edf3ff}.result-controls{margin-bottom:0}.results-table{min-width:850px}.results-table td:nth-child(1){min-width:230px;width:32%}.results-table td:nth-child(2){width:15%;min-width:110px}.results-table td:nth-child(3){min-width:135px}.results-table td:nth-child(n+4){min-width:80px}.details-panel h4{margin:24px 0 5px}.details-panel .table-wrap{max-width:700px}.details-head h3{overflow-wrap:anywhere}.details-metrics{display:flex;gap:30px;flex-wrap:wrap;border-bottom:1px solid var(--line);padding-bottom:15px;margin:15px 0}.details-metrics strong{display:block;color:var(--blue);font-size:27px}.details-panel p{max-width:850px;font-size:15px}.details-panel code{background:#edf2fa;padding:1px 4px;border-radius:3px}.sidebar .toc{font-size:13px}.muted{color:var(--muted)}.page-status{font-size:14px;min-width:82px;text-align:center}sup{font-size:.7em;line-height:0}.report>h2:first-child{border:0;margin-top:0;padding-top:0}.noscript-note{padding:15px;background:#edf2fb}
@media(max-width:760px){.slide{min-height:0}.hero h1{font-size:34px}.hero-links{font-size:14px}.pseudo-demo{grid-template-columns:1fr;gap:15px}.holdout-visual{grid-template-columns:1fr}.deck-bar{gap:10px}.deck-progress{gap:5px}.deck-progress button{padding:4px 7px;font-size:12px}.stat-equations{font-size:20px}.research-priority{padding-left:24px}.research-priority h3{font-size:19px}.comparison-pair{gap:22px}.report-section{margin-top:40px}}
@media print{.deck-progress,.hero-links,.pseudo-controls input{display:none}.slide{min-height:0}.pseudo-demo{grid-template-columns:1fr 1fr}.report-section{border:0}.report tbody tr:hover{background:inherit}}
`;
const script = `
(() => {
  const records = JSON.parse(document.getElementById('result-data').textContent);
  const q = selector => document.querySelector(selector);
  const escape = value => String(value ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  const fmt = value => typeof value === 'number' ? value.toFixed(4) : '—';
  const slides = [...document.querySelectorAll('.slide')];
  const tabs = [...document.querySelectorAll('[data-slide-target]')];
  let currentSlide = 0;
  function setSlide(i, focus=false) {
    currentSlide = Math.max(0,Math.min(slides.length-1,i));
    slides.forEach((slide,j)=>slide.setAttribute('aria-hidden',String(j!==currentSlide)));
    tabs.forEach((tab,j)=>tab.setAttribute('aria-current',String(j===currentSlide)));
    q('#page-count').textContent = (currentSlide+1)+' / '+slides.length;
    q('#slide-label').textContent = slides[currentSlide].dataset.slideLabel;
    q('#slide-prev').disabled = currentSlide===0;
    q('#slide-next').disabled = currentSlide===slides.length-1;
    if (focus) slides[currentSlide].querySelector('h2').focus({preventScroll:true});
  }
  q('#slide-prev').addEventListener('click',()=>setSlide(currentSlide-1));
  q('#slide-next').addEventListener('click',()=>setSlide(currentSlide+1));
  tabs.forEach((tab,i)=>tab.addEventListener('click',()=>setSlide(i)));
  q('#deck').addEventListener('keydown',event=>{
    if (/INPUT|SELECT|TEXTAREA/.test(event.target.tagName)) return;
    if(event.key==='ArrowRight'||event.key==='ArrowLeft') {event.preventDefault();setSlide(currentSlide+(event.key==='ArrowRight'?1:-1),true);}
  });
  setSlide(0);
  document.documentElement.classList.add('js');
  function updatePseudo() {
    const score=Number(q('#teacher-score').value), threshold=Number(q('#pseudo-threshold').value);
    q('#score-label').textContent=q('#score-output').textContent=score.toFixed(2);
    q('#threshold-output').textContent=threshold.toFixed(2);
    const accepted=score>=threshold;
    q('#pseudo-decision').classList.toggle('rejected',!accepted);
    q('#pseudo-decision').textContent=accepted?'p ≥ τ，接收伪标签：背景 → 汽车':'p < τ，不接收伪标签：仍用背景目标';
  }
  ['#teacher-score','#pseudo-threshold'].forEach(selector=>q(selector).addEventListener('input',updatePseudo));
  const classes=['背景','飞机','自行车','鸟','船','瓶子','公交车','汽车','猫','椅子','牛','餐桌','狗','马','摩托车','人','盆栽','羊','沙发','火车','电视'];
  const methodSelect=q('#method-filter'), protocolSelect=q('#protocol-filter');
  [...new Set(records.map(r=>r.method))].forEach(method=>methodSelect.add(new Option(method,method)));
  [...new Set(records.map(r=>r.protocol))].sort().forEach(protocol=>protocolSelect.add(new Option(protocol,protocol)));
  let matches=records, page=0, selected=null;const pageSize=14;
  function renderResults() {
    const totalPages=Math.max(1,Math.ceil(matches.length/pageSize));
    page=Math.min(page,totalPages-1);
    q('#result-count').textContent='共 '+matches.length+' 条结果';
    q('#result-page').textContent=(page+1)+' / '+totalPages;
    q('#result-prev').disabled=page===0;q('#result-next').disabled=page===totalPages-1;
    q('#result-body').innerHTML=matches.length?matches.slice(page*pageSize,(page+1)*pageSize).map(r=>'<tr><td>'+escape(r.config)+(r.json_only?'<sup>*</sup>':'')+'</td><td>'+escape(r.method)+'</td><td>'+escape(r.protocol)+'<br><span class="small">'+escape(r.step)+'</span></td><td class="number">'+fmt(r.old)+'</td><td class="number">'+fmt(r.new)+'</td><td class="number"><strong>'+fmt(r.all)+'</strong></td><td><button type="button" data-result="'+escape(r.key)+'" aria-expanded="'+String(selected===r.key)+'" aria-controls="result-detail" aria-label="展开 '+escape(r.config)+'">展开</button></td></tr>').join(''):'<tr><td colspan="7">没有匹配结果，可缩短关键词或清除筛选。</td></tr>';
  }
  function filterResults() {
    const terms=q('#result-search').value.trim().toLowerCase().split(/\\s+/).filter(Boolean);
    matches=records.filter(r=>(!methodSelect.value||r.method===methodSelect.value)&&(!protocolSelect.value||r.protocol===protocolSelect.value)&&terms.every(term=>[r.config,r.method,r.protocol,r.step].join(' ').toLowerCase().includes(term)));
    page=0;selected=null;q('#result-detail').hidden=true;renderResults();
  }
  q('#result-search').addEventListener('input',filterResults);
  methodSelect.addEventListener('change',filterResults);protocolSelect.addEventListener('change',filterResults);
  q('#result-clear').addEventListener('click',()=>{q('#result-search').value='';methodSelect.value='';protocolSelect.value='';filterResults();q('#result-search').focus();});
  q('#result-prev').addEventListener('click',()=>{page--;renderResults();});
  q('#result-next').addEventListener('click',()=>{page++;renderResults();});
  q('#result-body').addEventListener('click',event=>{
    const button=event.target.closest('[data-result]');if(!button)return;
    const r=records.find(row=>row.key===button.dataset.result);selected=r.key;
    q('#detail-title').textContent=r.config;
    q('#detail-context').textContent=r.method+' · '+r.protocol+' · '+r.step;
    q('#detail-metrics').innerHTML=['old','new','all'].map(k=>'<span>'+k+' mIoU<strong>'+fmt(r[k])+'%</strong></span>').join('');
    q('#detail-explanation').innerHTML=escape(r.explanation).replace(/\x60([^\x60]+)\x60/g,'<code>$1</code>');
    const per=Object.entries(r.per_class||{});
    q('#detail-class-section').hidden=!per.length;
    q('#detail-class-body').innerHTML=per.map(([c,value])=>'<tr><td>'+escape(classes[Number(c)]||c)+'</td><td class="number">'+fmt(value*100)+'</td></tr>').join('');
    q('#result-detail').hidden=false;renderResults();q('#detail-title').focus({preventScroll:true});
    q('#result-detail').scrollIntoView({block:'nearest',behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});
  });
  q('#detail-close').addEventListener('click',()=>{
    const old=selected;selected=null;q('#result-detail').hidden=true;renderResults();
    const button=[...document.querySelectorAll('[data-result]')].find(button=>button.dataset.result===old);if(button)button.focus();
  });
  renderResults();
})();
`;

const html = `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>基于闭式解的解析持续学习</title><meta name="description" content="CFSSeg二维类增量分割的方法、实验结果与后续研究方向。"><style>${css}\n${overrideCSS}\n${mathCSS}</style></head><body>
<a href="#main" class="skip">跳到正文</a><aside class="sidebar"><div class="brand">SegACIL / CFSSeg</div><div class="edition">类增量语义分割</div><nav class="nav-links" aria-label="主导航"><a href="#deck">五页讲解</a><a href="#results">实验结果检索</a><a href="#report">完整方法与结果</a></nav><details open><summary>正文目录</summary><nav class="toc" aria-label="正文目录">${toc.map(t=>`<a href="#${t.id}">${esc(t.title)}</a>`).join('')}</nav></details></aside>
<main id="main"><header class="hero"><h1>基于闭式解的解析持续学习</h1><p>冻结图像表示，以随机特征和递归岭回归学习新增类别。围绕特征、标签与解析目标，比较已有方法的实际收益。</p><div class="hero-links"><a href="#report">阅读完整结果</a><a href="#results">查找实验配置</a></div></header>
<section class="deck" id="deck" aria-label="五页方法讲解"><div class="deck-bar js-only js-flex"><div><span class="deck-label" id="slide-label">结果总览</span><nav class="deck-progress" aria-label="讲解页面">${slides.map((_,i)=>`<button type="button" data-slide-target="${i}" aria-label="第${i+1}页：${['结果总览','方法原理','伪标签','独立验证','后续方法'][i]}">${i+1} ${['总览','原理','伪标签','验证','后续'][i]}</button>`).join('')}</nav></div><div class="deck-controls"><button id="slide-prev" type="button" aria-label="上一讲解页">上一页</button><span class="page-count" id="page-count" aria-live="polite">1 / 5</span><button id="slide-next" type="button" class="primary" aria-label="下一讲解页">下一页</button></div></div>${slides.join('')}</section>
<section id="results" class="explorer report-section"><div class="section-top"><h2>实验结果检索</h2><a href="#report">查看方法解释与配置释义</a></div><p class="small"><sup>*</sup>当前跨服务器同步记录中只保留了该实验的JSON结果及配套配置。所有数值为mIoU（%）。</p><noscript><p class="noscript-note">下方完整报告包含全部结果表格；启用JavaScript后可搜索配置。</p></noscript><div class="js-only"><div class="filters"><label for="result-search">搜索配置或方法<input id="result-search" type="search" placeholder="如：b8224、fixed、decoder" autocomplete="off"></label><label for="method-filter">方法<select id="method-filter"><option value="">全部方法</option></select></label><label for="protocol-filter">实验协议<select id="protocol-filter"><option value="">全部协议</option></select></label></div><div class="result-controls"><span class="count" id="result-count" role="status"></span><button id="result-clear" type="button">清除筛选</button><button id="result-prev" type="button" aria-label="上一页结果">上一页</button><span id="result-page" class="page-status"></span><button id="result-next" type="button" aria-label="下一页结果">下一页</button></div><div class="table-wrap" tabindex="0" role="region" aria-label="实验结果，可横向滚动"><table class="results-table"><thead><tr><th scope="col">配置</th><th scope="col">方法</th><th scope="col">协议</th><th scope="col" class="number">old %</th><th scope="col" class="number">new %</th><th scope="col" class="number">all %</th><th scope="col">详解</th></tr></thead><tbody id="result-body"></tbody></table></div><section id="result-detail" class="details-panel" hidden aria-labelledby="detail-title"><div class="details-head"><h3 id="detail-title" tabindex="-1"></h3><button id="detail-close" type="button">收起</button></div><p id="detail-context" class="muted"></p><div class="details-metrics" id="detail-metrics"></div><h4>配置含义</h4><p id="detail-explanation"></p><div id="detail-class-section"><h4>逐类别IoU</h4><div class="table-wrap" tabindex="0" role="region" aria-label="逐类别IoU"><table><thead><tr><th scope="col">类别</th><th scope="col" class="number">IoU %</th></tr></thead><tbody id="detail-class-body"></tbody></table></div></div></section></div></section>
<article class="report report-section" id="report" aria-label="完整方法与实验回顾">${body}</article><footer class="footer"><a href="#main">返回顶部</a></footer></main><script id="result-data" type="application/json">${safeJSON(records)}</script><script>${script}</script></body></html>`;
for (const name of ['06_方法与实验交接_修订版.html','06_方法与实验交接.html']) fs.writeFileSync(path.join(OUT,name),html);
console.log(`Revised HTML: ${slides.length} slides; ${records.length} result rows; ${tables.length} report tables; ${html.length} characters.`);

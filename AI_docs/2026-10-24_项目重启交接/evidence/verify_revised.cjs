/* Validate revised report numbers, rendering, and reader interactions. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const {pathToFileURL} = require('node:url');
const deps = process.env.REPORT_NODE_MODULES || '/tmp/segacil-handoff-tools/node_modules';
const {chromium} = require(path.join(deps,'playwright'));
const OUT=path.resolve(__dirname,'..'), ROOT=path.resolve(OUT,'../..');
const read=name=>fs.readFileSync(path.join(OUT,name),'utf8');
const load=name=>JSON.parse(read('evidence/'+name+'.json'));
const report=read('03_方法与实验完整回顾_修订版.md');
const original=read('03_方法与实验完整回顾.md');
const raw=load('raw_metrics'), legacy=load('legacy_results'), records=load('revised_results'), tables=load('revised_tables');
const checks=[];const pass=(name,detail)=>checks.push({name,detail});
const close=(a,b,tolerance=0.000051)=>assert.ok(Math.abs(a-b)<tolerance,`${a} differs from ${b}`);
const clean=s=>String(s).replace(/<sup>\*<\/sup>/g,'*').replace(/\*\*/g,'').replace(/`/g,'');
const parseTables=source=>[...source.matchAll(/(?:^\|.*\n)+/gm)].map(m=>{const rows=m[0].trim().split('\n').map(line=>line.split('|').slice(1,-1).map(s=>s.trim()));return{headers:rows[0],rows:rows.slice(2)};});
const sourceTriples=[];
for(const t of parseTables(original)){
  const indexes=['old','new','all'].map(key=>t.headers.findIndex(h=>new RegExp('^'+key+' (?:%|mIoU)$').test(h)));
  if(indexes.every(i=>i>=0))for(const row of t.rows){const v=indexes.map(i=>Number(row[i]));if(v.every(Number.isFinite))sourceTriples.push(v);}
}
for(const r of raw)sourceTriples.push([r.old_pct,r.new_pct,r.all_pct]);
for(const r of legacy)sourceTriples.push(['旧类mIoU(%)','新类mIoU(%)','全部mIoU(%)'].map(k=>Number(r[k])));
const originalTables=parseTables(original);
const chainAll=originalTables.find(t=>t.headers[0]==='策略'&&t.headers.some(h=>h.startsWith('step1')));
const chainGroups=originalTables.find(t=>t.headers.includes('当前累计新类 mIoU'));
for(const [config,step,old,new_] of chainGroups.rows){
  const row=chainAll.rows.find(r=>r[0]===config);
  sourceTriples.push([Number(old),Number(new_),Number(row[Number(step)])]);
}

(async()=>{
  const parsed=parseTables(report);assert.deepEqual(parsed,tables);assert.equal(tables.length,28);
  assert.equal([...report.matchAll(/^配置：/gm)].length,28);
  assert.equal([...report.matchAll(/^## /gm)].length,11);
  const banned=/Material Passport|来源等级|原始证据|以下直接来自原始|已确认：|证据分级|给你和师兄|缺少权重|旧conda|重跑GPU|第一轮讨论产物|联邦/;
  assert.ok(!banned.test(report));
  for(const t of tables){assert.ok(!t.headers.some(h=>/^(ID|原始证据|来源等级)$/.test(h)));for(const row of t.rows)assert.ok(!/\b202\d{5}|^\d{4}_/.test(row[0]),row[0]);}
  let numericRows=0;
  for(const t of tables.slice(1)){
    const indices=['old','new','all'].map(key=>t.headers.findIndex(h=>new RegExp('^(?:最佳 )?'+key+' %$').test(h)));
    if(!indices.every(i=>i>=0))continue;
    for(const row of t.rows){const v=indices.map(i=>Number(clean(row[i])));assert.ok(sourceTriples.some(s=>s.every((n,i)=>typeof n==='number'&&Math.abs(n-v[i])<0.000051)),JSON.stringify(row));numericRows++;}
  }
  let blobs=0,classes=0;
  for(const r of records){
    assert.ok(!/\b202\d{5}/.test(r.config),r.config);
    if(/^R\d{3}$/.test(r.key)){
      const src=raw.find(x=>x.id===r.key);
      const data=JSON.parse(execFileSync('git',['cat-file','blob',src.blob],{cwd:ROOT,encoding:'utf8'}));
      const m=data.metrics||data;
      close(r.all,m['Mean IoU']*100,1e-10);close(r.old,m[src.old_group]*100,1e-10);close(r.new,m[src.new_group]*100,1e-10);
      assert.equal(r.json_only,true);assert.deepEqual(r.per_class,m['Class IoU']);blobs++;classes+=Object.keys(r.per_class).length;
    }else if(/^E\d{3}$/.test(r.key)){
      const src=legacy.find(x=>x['证据编号']===r.key);
      for(const [key,column] of [['old','旧类mIoU(%)'],['new','新类mIoU(%)'],['all','全部mIoU(%)']])close(r[key],Number(src[column]),1e-10);
    }else{
      assert.ok(sourceTriples.some(s=>s.every((n,i)=>typeof n==='number'&&Math.abs(n-[r.old,r.new,r.all][i])<0.000051)),r.key);
    }
  }
  const mean=(ids,field)=>ids.reduce((s,id)=>s+raw.find(r=>r.id===id)[field],0)/ids.length;
  const means=[['R115','R117','R119','R135','R137'],['R121','R123','R125','R139','R141'],['R123','R139','R141'],['R129','R145','R147'],['R175','R177','R179'],['R183','R185','R187']];
  for(const ids of means){const values=['old_pct','new_pct','all_pct'].map(f=>mean(ids,f).toFixed(6));assert.ok(tables.some(t=>t.rows.some(row=>values.every(v=>row.includes(v)))),JSON.stringify(values));}
  for(const row of tables[1].rows){const protocol=row[1], delta=row[6];if(delta==='—')continue;const paper=protocol==='15-1 overlap'?69.36:70;assert.equal(Number(row[5]),paper);close(Number(clean(row[4]))-paper,Number(delta),0.000051);}
  pass('数值与结构',`${numericRows}组old/new/all，${blobs}个Git结果blob，${classes}项逐类别值，6组均值，28张表的配置释义`);
  assert.equal(read('06_方法与实验交接.html'),read('06_方法与实验交接_修订版.html'));
  const browser=await chromium.launch({headless:true});
  const errors=[],network=[];const context=await browser.newContext({viewport:{width:1440,height:1000},reducedMotion:'reduce'});
  const p=await context.newPage();p.on('pageerror',e=>errors.push(e.message));p.on('request',r=>{if(/^https?:/.test(r.url()))network.push(r.url());});
  const url=pathToFileURL(path.join(OUT,'06_方法与实验交接_修订版.html')).href;
  const shots='/tmp/segacil-revised-preview';fs.mkdirSync(shots,{recursive:true});
  await p.goto(url);await p.evaluate(()=>document.fonts.ready);
  assert.equal(await p.title(),'基于闭式解的解析持续学习');
  assert.equal(await p.locator('.slide').count(),5);assert.equal(await p.locator('.slide[aria-hidden=false]').count(),1);
  await p.screenshot({path:path.join(shots,'desktop.png')});
  for(let i=1;i<5;i++)await p.locator('#slide-next').click();
  assert.equal(await p.locator('#page-count').innerText(),'5 / 5');assert.equal(await p.locator('#slide-next').isDisabled(),true);
  assert.ok(!banned.test(await p.locator('.slide').nth(4).innerText()));
  await p.locator('#slide-prev').click();await p.keyboard.press('ArrowLeft');assert.equal(await p.locator('#page-count').innerText(),'3 / 5');
  const range=async(selector,value)=>p.locator(selector).evaluate((el,v)=>{el.value=v;el.dispatchEvent(new Event('input',{bubbles:true}));},value);
  await range('#pseudo-threshold','0.8');assert.match(await p.locator('#pseudo-decision').innerText(),/不接收/);
  await range('#teacher-score','0.9');assert.match(await p.locator('#pseudo-decision').innerText(),/背景 → 汽车/);
  await range('#teacher-score','0.72');await range('#pseudo-threshold','0.6');
  await p.locator('#deck').screenshot({path:path.join(shots,'pseudo-slide.png')});
  pass('五页讲解','按钮、页面选择、键盘、边界和伪标签交互通过');
  const rendered=await p.locator('.report table').evaluateAll(tables=>tables.map(t=>({headers:[...t.querySelectorAll('thead th')].map(c=>c.innerText),rows:[...t.querySelectorAll('tbody tr')].map(row=>[...row.cells].map(c=>c.innerText))})));
  assert.deepEqual(rendered,tables.map(t=>({headers:t.headers.map(clean),rows:t.rows.map(row=>row.map(clean))})));
  assert.equal(await p.locator('.math-display .katex').count(),1);
  assert.equal(await p.locator('.katex-error').count(),0);
  const text=await p.locator('body').innerText();assert.ok(!banned.test(text));assert.ok(!/\b202\d{5}/.test(text));
  assert.equal(await p.locator('[download],#download-results,#evidence-filter').count(),0);
  const embedded=JSON.parse(await p.locator('#result-data').textContent());assert.equal(embedded.length,records.length);
  for(const row of embedded)assert.ok(row.explanation.length>20,row.config);
  assert.match(embedded.find(r=>r.key==='P2-off').explanation,/15-5 disjoint/);
  assert.match(embedded.find(r=>r.key==='P0-off').explanation,/15-5 overlap/);
  assert.equal((await p.locator('a[href^="#"]').evaluateAll(links=>links.filter(a=>!document.getElementById(a.getAttribute('href').slice(1))).map(a=>a.outerHTML))).length,0);
  pass('正文与Markdown一致','28张表全部单元格一致；公式、配置释义与目录正常；无日期前缀、来源分级和下载功能');
  await p.locator('#result-search').fill('decoder_b8224_g1_rhl2_eps0');
  assert.equal(await p.locator('#result-body tr').count(),1);assert.match(await p.locator('#result-body').innerText(),/59\.7119/);
  await p.locator('[data-result="R189"]').click();assert.equal(await p.locator('#result-detail').isVisible(),true);
  assert.equal(await p.locator('#detail-class-body tr').count(),21);assert.match(await p.locator('#detail-explanation').innerText(),/零初始化/);
  await p.locator('#result-detail').screenshot({path:path.join(shots,'result-detail.png')});
  await p.locator('#detail-close').click();await p.locator('#result-clear').click();
  await p.locator('#method-filter').selectOption('MG');assert.match(await p.locator('#result-count').innerText(),/6 条/);
  await p.locator('#protocol-filter').selectOption('15-5 overlap');assert.match(await p.locator('#result-count').innerText(),/3 条/);
  await p.locator('#result-clear').click();await p.locator('#result-next').click();assert.match(await p.locator('#result-page').innerText(),/^2 \/ /);
  await p.locator('#result-search').fill('no_such_config');assert.match(await p.locator('#result-body').innerText(),/没有匹配/);
  await p.locator('#result-clear').click();await p.locator('#result-search').fill('matched_global');await p.locator('[data-result="E065"]').click();
  assert.equal(await p.locator('#detail-class-section').isVisible(),false);
  pass('实验检索',`${records.length}条结果；配置搜索、两种筛选、分页、空结果、配置释义、逐类别详情与收起通过`);
  for(const width of [1440,390]){
    await p.setViewportSize({width,height:width===390?844:1000});await p.goto(url);await p.evaluate(()=>document.fonts.ready);
    assert.ok(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
    if(width===390){await p.screenshot({path:path.join(shots,'mobile.png')});await p.locator('[data-slide-target="2"]').click();assert.equal(await p.locator('#page-count').innerText(),'3 / 5');}
  }
  await p.emulateMedia({media:'print'});assert.equal(await p.locator('.slide:visible').count(),5);await p.emulateMedia({media:'screen'});
  const nojs=await browser.newContext({javaScriptEnabled:false});const nj=await nojs.newPage();await nj.goto(url);
  assert.equal(await nj.locator('.slide:visible').count(),5);assert.equal(await nj.locator('.report table').count(),28);
  pass('阅读适配','桌面1440px、移动端390px无整页溢出；无JavaScript及打印可读全部五页和正文');
  await p.setViewportSize({width:1440,height:1660});await p.goto(pathToFileURL(path.join(OUT,'assets/methods-map-revised.svg')).href);await p.evaluate(()=>document.fonts.ready);
  const overflow=await p.locator('text').evaluateAll(texts=>texts.map(t=>{const b=t.getBBox();return{text:t.textContent,x:b.x,right:b.x+b.width,y:b.y,bottom:b.y+b.height};}).filter(b=>b.x<0||b.right>1440||b.y<0||b.bottom>1660));
  assert.deepEqual(overflow,[]);await p.locator('svg').screenshot({path:path.join(shots,'methods-map.png')});
  assert.deepEqual(errors,[]);assert.deepEqual(network,[]);await browser.close();
  pass('图与离线性','总结图文字在画布内；0个页面错误，0个自动HTTP请求');
  const result={status:'PASS',checks,screenshots:shots};fs.writeFileSync(path.join(__dirname,'revised_validation.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
})().catch(error=>{console.error(error);process.exit(1);});

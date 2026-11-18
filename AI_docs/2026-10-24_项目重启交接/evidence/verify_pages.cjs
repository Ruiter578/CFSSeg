/* Artifact acceptance checks, not ML training tests. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {pathToFileURL} = require('node:url');
const deps = process.env.REPORT_NODE_MODULES || '/tmp/segacil-handoff-tools/node_modules';
const {chromium} = require(path.join(deps,'playwright'));
const MarkdownIt = require(path.join(deps,'markdown-it'));
const OUT = path.resolve(__dirname,'..');
const shots = '/tmp/segacil-handoff-preview';
fs.mkdirSync(shots,{recursive:true});
const checks = [];
function pass(name, details) {checks.push({name,status:'PASS',details});}
const pages = ['06_方法与实验交接.html','07_联邦持续学习与研究路线.html'];
const read = n=>fs.readFileSync(path.join(OUT,n),'utf8');

(async()=>{
 const browser = await chromium.launch({headless:true});
 const errors=[], network=[];
 const context=await browser.newContext({viewport:{width:1440,height:1000},reducedMotion:'reduce'});
 context.on('page',p=>{p.on('pageerror',error=>errors.push(error.message));p.on('request',r=>{if(/^https?:/.test(r.url()))network.push(r.url());});});
 const p=await context.newPage();
 for(const [index,name] of pages.entries()){
   await p.goto(pathToFileURL(path.join(OUT,name)).href);
   await p.evaluate(()=>document.fonts.ready);
   assert.equal(await p.locator('.slide').count(),5);
   assert.equal(await p.locator('.slide[aria-hidden=false]').count(),1);
   await p.screenshot({path:path.join(shots,index+'-desktop.png')});
   for(let i=1;i<5;i++) await p.locator('#slide-next').click();
   assert.equal(await p.locator('#slide-next').isDisabled(),true);
   assert.equal(await p.locator('#slide-count').innerText(),'5 / 5');
   await p.locator('#slide-prev').click();
   await p.keyboard.press('ArrowLeft');
   assert.equal(await p.locator('#slide-count').innerText(),'3 / 5');
   pass(name+'：翻页、边界与键盘', '5页；前后按钮与左右键');
   const badAnchors=await p.locator('a[href^="#"]').evaluateAll(links=>links.map(a=>a.getAttribute('href')).filter(h=>!document.getElementById(h.slice(1))));
   assert.deepEqual(badAnchors,[]);
   pass(name+'：目录锚点', '全部可达');
   const overflow=await p.evaluate(()=>({body:document.documentElement.scrollWidth,viewport:innerWidth}));
   assert.ok(overflow.body<=overflow.viewport+1,JSON.stringify(overflow));
   await p.emulateMedia({media:'print'});
   assert.equal(await p.locator('.slide').evaluateAll(es=>es.filter(e=>getComputedStyle(e).display!=='none').length),5);
   await p.emulateMedia({media:'screen'});
   pass(name+'：打印展开','全部讲解页可见');
   if(index===0){
     const embedded=await p.locator('#records-data').textContent();
     assert.equal(JSON.parse(embedded).length,317);
     await p.locator('#result-search').fill('R189');
     assert.equal(await p.locator('#result-body tr').count(),1);
     assert.match(await p.locator('#result-body').innerText(),/59\.711857/);
     await p.locator('[data-record="R189"]').click();
     assert.equal(await p.locator('#record-details').isVisible(),true);
     assert.equal(await p.locator('#record-details tbody tr').count(),21);
     await p.locator('#close-details').click();
     await p.locator('#result-search').fill('');
     for(const [value,expected] of [['current',191],['history',16],['legacy',110]]){
       await p.locator('#evidence-filter').selectOption(value);
       assert.match(await p.locator('#result-count').innerText(),new RegExp('^'+expected+'条'));
     }
     await p.locator('#evidence-filter').selectOption('all');
     await p.locator('#split-filter').selectOption('invalid');
     assert.ok(!/^0条/.test(await p.locator('#result-count').innerText()));
     await p.locator('#split-filter').selectOption('clean');
     const cleanCount=await p.locator('#result-count').innerText();
     assert.ok(!/^0条/.test(cleanCount));
     await p.locator('#split-filter').selectOption('all');
     await p.locator('#results-next').click();
     assert.equal(await p.locator('#results-page').innerText(),'2 / 27');
     await p.locator('#result-search').fill('NO_SUCH_EXPERIMENT');
     assert.match(await p.locator('#result-body').innerText(),/没有匹配/);
     await p.locator('#result-search').fill('R191');
     const downloadPromise=p.waitForEvent('download');
     await p.locator('#download-results').click();
     const download=await downloadPromise;
     const csv=fs.readFileSync(await download.path(),'utf8');
     assert.match(csv,/R191/);assert.match(csv,/59\.714327400697/);
     pass('实验检索与下载','317条内嵌记录；来源筛选191/16/110；搜索、空结果、分页、21类详情、CSV下载通过；'+cleanCount);
   }else{
     await p.locator('#slide-prev').click();
     const setRange=async(id,value)=>p.locator(id).evaluate((el,v)=>{el.value=v;el.dispatchEvent(new Event('input',{bubbles:true}));},value);
     await setRange('#client-size','20');
     assert.equal(await p.locator('#pooled-output').innerText(),(10/91).toFixed(6));
     assert.equal(await p.locator('#average-output').innerText(),(10/30*10/11).toFixed(6));
     await p.locator('#overview').screenshot({path:path.join(shots,'route-math.png')});
     await p.locator('#slide-prev').click();
     await setRange('#task-time','2');
     assert.equal(await p.locator('[data-client-labels] .label-tag').count(),8);
     await p.locator('#overview').screenshot({path:path.join(shots,'route-clients.png')});
     assert.equal(await p.locator('.katex-error').count(),0);
     assert.equal(await p.locator('.math-display .katex').count(),3);
     pass('联邦概念交互与数学渲染','时间滑块、可见标签、两个闭式数值、3个KaTeX公式通过');
   }
   await p.setViewportSize({width:390,height:844});
   await p.goto(pathToFileURL(path.join(OUT,name)).href);
   await p.evaluate(()=>document.fonts.ready);
   const mobile=await p.evaluate(()=>({body:document.documentElement.scrollWidth,viewport:innerWidth}));
   assert.ok(mobile.body<=mobile.viewport+1,JSON.stringify(mobile));
   await p.screenshot({path:path.join(shots,index+'-mobile.png')});
   await p.locator('#slide-next').click();
   assert.equal(await p.locator('#slide-count').innerText(),'2 / 5');
   pass(name+'：移动端','390×844，无整页横向溢出；图与表格在容器内横向滚动');
   await p.setViewportSize({width:1440,height:1000});
 }
 const nojs=await browser.newContext({javaScriptEnabled:false,viewport:{width:1440,height:1000}});
 const nj=await nojs.newPage();
 for(const name of pages){await nj.goto(pathToFileURL(path.join(OUT,name)).href);assert.equal(await nj.locator('.slide:visible').count(),5);assert.ok((await nj.locator('.report').innerText()).length>6000);}
 pass('无JavaScript阅读','两个页面均显示全部讲解页及完整正文');
 for(const name of ['methods-map','route-map']){
   await p.goto(pathToFileURL(path.join(OUT,'assets',name+'.svg')).href);
   await p.locator('svg').screenshot({path:path.join(shots,name+'.png')});
   const boxes=await p.locator('text').evaluateAll(ts=>ts.map(t=>({text:t.textContent,x:t.getBBox().x,right:t.getBBox().x+t.getBBox().width,y:t.getBBox().y,bottom:t.getBBox().y+t.getBBox().height})));
   const width=name==='methods-map'?1260:1200,height=name==='methods-map'?900:690;
   assert.ok(boxes.every(b=>b.x>=0&&b.right<=width&&b.y>=0&&b.bottom<=height),JSON.stringify(boxes.filter(b=>b.right>width||b.bottom>height)));
   pass(name+'：SVG文字边界','全部文字在画布内；截图供人工复核');
 }
 assert.deepEqual(errors,[]);assert.deepEqual(network,[]);
 pass('运行时错误与离线性','0个pageerror；0个HTTP(S)自动请求');
 const md=new MarkdownIt({html:true});const broken=[];
 const docs=fs.readdirSync(OUT).filter(n=>n.endsWith('.md'));
 for(const name of [...docs,'evidence/文档逐篇目录.md']){
   const base=path.dirname(path.join(OUT,name));
   const walk=tokens=>{for(const token of tokens){for(const attr of ['href','src']){const href=token.attrGet(attr);if(!href||/^(?:https?:|#|mailto:|data:)/.test(href))continue;const target=decodeURIComponent(href.split('#')[0]);if(!fs.existsSync(path.resolve(base,target)))broken.push({document:name,href});}if(token.children)walk(token.children);}};
   walk(md.parse(read(name),{}));
 }
 assert.deepEqual(broken,[]);
 pass('Markdown相对链接','主体、入口、设计与逐篇目录的本地链接全部存在');
 await browser.close();
 const result={checked_date:'2026-10-25',browser:'Chromium via Playwright',checks,errors,network_requests:network,screenshots:shots};
 fs.writeFileSync(path.join(__dirname,'page_validation.json'),JSON.stringify(result,null,2));
 console.log(JSON.stringify(result,null,2));
})().catch(error=>{console.error(error);process.exit(1);});

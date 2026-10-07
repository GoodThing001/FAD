const fs = require('fs');
const path = require('path');
const {chromium} = require('C:/Users/mmc/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root=process.cwd();
const doc=fs.readFileSync(path.join(root,'runs/search_seven_validation_20261005/report_template.md'),'utf8');
const codes=[...doc.matchAll(/```mermaid\r?\n([\s\S]*?)```/g)].map(m=>m[1]);
const names=['18_迭代局部搜索','19_模拟退火','20_无交叉AdaLead'];
const out=path.join(root,'docs/汇报/9月/figures/GBSA总汇报_20261005');
(async()=>{
 const browser=await chromium.launch({headless:true,channel:'msedge'});
 const page=await browser.newPage({viewport:{width:1400,height:1100},deviceScaleFactor:1.5});
 await page.setContent('<html><body></body></html>');
 await page.addScriptTag({path:path.join(root,'runs/doc_diagram_preview_20261005/mermaid.min.js')});
 await page.evaluate(()=>mermaid.initialize({startOnLoad:false,securityLevel:'strict',fontFamily:'Microsoft YaHei, Segoe UI, sans-serif',theme:'default',flowchart:{htmlLabels:true,useMaxWidth:false,curve:'linear',nodeSpacing:32,rankSpacing:35}}));
 for(let i=0;i<codes.length;i++){
  const svg=await page.evaluate(async({code,id})=>(await mermaid.render(id,code)).svg,{code:codes[i],id:'seven_fig_'+i});
  fs.writeFileSync(path.join(out,names[i]+'.svg'),svg,'utf8');
  await page.setContent('<html><head><style>body{margin:20px;background:white;}svg{display:block;max-width:none!important;}</style></head><body>'+svg+'</body></html>');
  const bounds=await page.locator('svg').boundingBox();
  await page.setViewportSize({width:Math.ceil(bounds.width)+40,height:Math.ceil(bounds.height)+40});
  await page.locator('svg').screenshot({path:path.join(out,names[i]+'.png')});
  console.log(names[i],Math.round(bounds.width),Math.round(bounds.height));
 }
 await browser.close();
})().catch(e=>{console.error(e.message);process.exit(1);});

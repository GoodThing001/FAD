/* Render editable Mermaid sources for the report, using the available runtime. */
const fs = require('fs');
const path = require('path');
const {chromium} = require('C:/Users/mmc/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root = process.cwd();
const out = path.join(root, 'docs/汇报/figures/阶段总汇报_20261006');
(async () => {
  const browser = await chromium.launch({headless: true, channel: 'msedge'});
  try {
    const page = await browser.newPage({viewport: {width: 1400, height: 1100}, deviceScaleFactor: 1.5});
    await page.setContent('<html><body></body></html>');
    await page.addScriptTag({path: path.join(root, 'runs/doc_diagram_preview_20261005/mermaid.min.js')});
    await page.evaluate(() => mermaid.initialize({startOnLoad: false, securityLevel: 'strict',
      fontFamily: 'Microsoft YaHei, Segoe UI, sans-serif', theme: 'default',
      flowchart: {htmlLabels: false, useMaxWidth: false, curve: 'linear', nodeSpacing: 30, rankSpacing: 35}}));
    for (const file of fs.readdirSync(out).filter(n => n.endsWith('.mmd')).sort()) {
      const name = path.basename(file, '.mmd');
      const code = fs.readFileSync(path.join(out, file), 'utf8');
      const svg = await page.evaluate(async ({code, id}) => {
        const raw = (await mermaid.render(id, code)).svg;
        const holder = document.createElement('div');
        holder.innerHTML = raw;
        return new XMLSerializer().serializeToString(holder.firstElementChild);
      },
        {code, id: 'phase_' + name.slice(0, 3)});
      fs.writeFileSync(path.join(out, name + '.svg'), svg, 'utf8');
      await page.setContent('<html><head><style>body{margin:20px;background:white}svg{display:block;max-width:none!important}</style></head><body>' + svg + '</body></html>');
      const bounds = await page.locator('svg').boundingBox();
      await page.setViewportSize({width: Math.ceil(bounds.width) + 40, height: Math.ceil(bounds.height) + 40});
      await page.locator('svg').screenshot({path: path.join(out, name + '.png')});
      console.log(name, Math.round(bounds.width), Math.round(bounds.height));
    }
  } finally { await browser.close(); }
})().catch(e => {console.error(e.message); process.exitCode = 1;});

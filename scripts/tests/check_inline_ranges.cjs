const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const cases=[
 ['/Robot-Navigation-Survey/',['103–104','k≈1.5–2.0','k≈1.0–1.5','k≈0.5–1.0']],
 ['/en/Robot-Navigation-Survey/',['103–104','k≈1.5–2.0','k≈1.0–1.5','k≈0.5–1.0']],
 ['/LLM-Training-Survey/',['4N–8N']],
 ['/Spatial-Intelligence-Survey/',['7×–500×']],
 ['/AI-Agent-Survey/',[]],['/en/AI-Agent-Survey/',[]]
];
(async()=>{const browser=await chromium.launch({headless:true,channel:'chrome'});try{
 const page=await browser.newPage();
 await page.route('**/*',r=>r.request().resourceType()==='image'||/google-analytics|googletagmanager|vercount|api.github.com|buttons.github.io|use.typekit.net/.test(r.request().url())?r.abort():r.continue());
 for(const [route,expected] of cases){
  await page.goto((process.argv[2]||'http://127.0.0.1:4175')+route,{waitUntil:'domcontentloaded'});
  await page.waitForFunction(()=>window.MathJax&&MathJax.startup&&MathJax.startup.promise);await page.evaluate(()=>MathJax.startup.promise);
  for(const width of [1440,390,320]){
   await page.setViewportSize({width,height:844});
   const ranges=await page.locator('.entry mjx-container').evaluateAll(nodes=>nodes.map(n=>({text:n.querySelector('mjx-assistive-mml math')?.textContent.replace(/\s/g,''),display:n.getAttribute('display'),overflow:getComputedStyle(n).overflowX})).filter(x=>x.text?.includes('–')));
   for(const text of expected){const matches=ranges.filter(x=>x.text===text);assert.equal(matches.length,1,route+' '+width+': missing unified range '+text);assert.notEqual(matches[0].display,'true','Range must stay inline');}
   if(!expected.length){const price=page.locator('.entry span').filter({hasText:'USD 3.80–4.20'});assert.equal(await price.count(),1);assert.equal(await price.locator('mjx-container').count(),0);const lines=await price.evaluate(n=>{const r=document.createRange();r.selectNodeContents(n);return new Set([...r.getClientRects()].map(x=>x.top)).size;});assert.equal(lines,1);}
   console.log(route+' '+width+'px: '+(expected.length||1)+' complete inline ranges');
  }
 }
}finally{await browser.close();}})().catch(e=>{console.error(e);process.exitCode=1;});

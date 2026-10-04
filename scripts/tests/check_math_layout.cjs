// Browser regression for MathJax rendering and containment in either language.
// Requires Playwright and Chrome. Usage: node check_math_layout.cjs [URL] [--all|/route/ ...]
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

function mathRoutes(directory, relative = '') {
  return fs.readdirSync(path.join(directory, relative), { withFileTypes: true }).flatMap(item => {
    const name = path.join(relative, item.name);
    if (item.isDirectory()) return mathRoutes(directory, name);
    if (item.name !== 'index.html') return [];
    const html = fs.readFileSync(path.join(directory, name), 'utf8');
    return html.includes('mathjax@3') && html.includes('class="entry"')
      ? [`/${path.dirname(name).split(path.sep).join('/')}/`] : [];
  }).sort();
}

(async () => {
  const base = process.argv[2] || 'http://127.0.0.1:4173';
  const requested = process.argv.slice(3);
  const routes = requested.includes('--all')
    ? mathRoutes(path.resolve(__dirname, '../../_site'))
    : requested.length ? requested : ['/en/Embodied-Agent-Papers/'];
  assert(routes.length > 0, 'No math-enabled articles found');
  const browser = await chromium.launch({ headless: true, channel: 'chrome' });
  try {
    const page = await browser.newPage();
    await page.route('**/*', route => {
      const request = route.request();
      return request.resourceType() === 'image' || /google-analytics|googletagmanager|busuanzi|api.github.com|buttons.github.io|use.typekit.net|events.vercount/.test(request.url())
        ? route.abort() : route.continue();
    });
    const failures = [];
    let total = 0;
    for (const route of routes) {
      await page.goto(`${base}${route}`, { waitUntil: 'domcontentloaded' });
      await page.waitForFunction(() => window.MathJax && MathJax.startup && MathJax.startup.promise);
      await page.evaluate(() => MathJax.startup.promise);
      await page.evaluate(() => document.fonts.ready);
      for (const width of [1440, 390]) {
        await page.setViewportSize({ width, height: 1000 });
        const metrics = await page.evaluate(() => {
          const formulas = [...document.querySelectorAll('.entry mjx-container')];
          const clipped = formulas.filter(container => {
            if (getComputedStyle(container).overflowY !== 'hidden') return false;
            const bounds = container.getBoundingClientRect();
            // Assistive MathML is intentionally clipped for screen readers.
            // Measure only the visible CommonHTML output.
            return [...container.querySelectorAll('mjx-math, mjx-math *')].some(child => {
              const ink = child.getBoundingClientRect();
              return ink.height > 0 && (ink.top < bounds.top - 1 || ink.bottom > bounds.bottom + 1);
            });
          }).map(container => formulas.indexOf(container));
          const uncontained = formulas.filter(container => {
            const math = container.querySelector('mjx-math');
            if (!math) return false;
            const bounds = math.getBoundingClientRect();
            if (bounds.right <= innerWidth + 1 && bounds.left >= -1) return false;
            for (let node = container; node && node !== document.body; node = node.parentElement) {
              if (['auto', 'scroll'].includes(getComputedStyle(node).overflowX)) return false;
            }
            return true;
          }).map(container => formulas.indexOf(container));
          const raw = [];
          const walker = document.createTreeWalker(document.querySelector('.entry'), NodeFilter.SHOW_TEXT);
          while (walker.nextNode()) {
            const node = walker.currentNode;
            if (node.parentElement.closest('mjx-container,pre,code,script,style,svg,textarea')) continue;
            if (/\\(?:\(|\[|frac\b|mathcal\b|mathbf\b|begin\{)|\$\$/.test(node.textContent)) raw.push(node.textContent.slice(0,200));
          }
          return {
            formulas: formulas.length,
            errors: document.querySelectorAll('.entry mjx-merror,.entry [data-mjx-error]').length,
            clipped,
            uncontained,
            raw,
            scrollableFormulas: formulas.filter(n => n.scrollWidth > n.clientWidth + 1).length,
          };
        });
        if (width === 1440) total += metrics.formulas;
        if (metrics.errors || metrics.clipped.length || metrics.uncontained.length || metrics.raw.length) {
          failures.push({route,width,...metrics});
        }
        if (route === '/en/Embodied-Agent-Papers/') {
          assert(metrics.formulas > 100, 'Paper collection did not finish typesetting');
          if (width === 390) assert(metrics.scrollableFormulas > 0, 'Long formulas must remain locally scrollable');
        }
        console.log(`${route} ${width}px: ${metrics.formulas} formulas; errors=${metrics.errors}, clipped=${metrics.clipped.length}, uncontained=${metrics.uncontained.length}, raw=${metrics.raw.length}`);
      }
    }
    assert.deepEqual(failures, [], 'Math rendering/layout regressions');
    console.log(`Passed: ${routes.length} pages, ${total} formulas at desktop and mobile widths`);
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

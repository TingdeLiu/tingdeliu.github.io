// Browser regression: glyphs in an English math scroller must remain visible.
// Requires Playwright and Chrome. Pass the built site's URL as the first argument.
const assert = require('node:assert/strict');
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true, channel: 'chrome' });
  try {
    const page = await browser.newPage();
    await page.route(/google-analytics|googletagmanager|busuanzi|api.github.com|buttons.github.io|use.typekit.net|events.vercount/, route => route.abort());
    const base = process.argv[2] || 'http://127.0.0.1:4173';
    for (const width of [1440, 390]) {
      await page.setViewportSize({ width, height: 1000 });
      await page.goto(`${base}/en/Embodied-Agent-Papers/`, { waitUntil: 'networkidle' });
      await page.waitForFunction(() => window.MathJax && MathJax.startup && MathJax.startup.promise);
      await page.evaluate(() => MathJax.startup.promise);
      await page.evaluate(() => document.fonts.ready);
      const metrics = await page.evaluate(() => {
        const formulas = [...document.querySelectorAll('.entry mjx-container')];
        const clipped = formulas.filter(container => {
          if (getComputedStyle(container).overflowY !== 'hidden') return false;
          const bounds = container.getBoundingClientRect();
          return [...container.querySelectorAll('*')].some(child => {
            const ink = child.getBoundingClientRect();
            return ink.height > 0 && (ink.top < bounds.top - 1 || ink.bottom > bounds.bottom + 1);
          });
        }).map(container => container.textContent);
        return {
          formulas: formulas.length,
          errors: document.querySelectorAll('mjx-merror').length,
          clipped,
          pageOverflow: document.documentElement.scrollWidth > innerWidth + 1,
          scrollableFormulas: formulas.filter(n => n.scrollWidth > n.clientWidth + 1).length,
        };
      });
      assert(metrics.formulas > 100, 'Paper collection did not finish typesetting');
      assert.equal(metrics.errors, 0, 'MathJax errors');
      assert.deepEqual(metrics.clipped, [], 'Formula glyphs extend outside a clipping container');
      assert.equal(metrics.pageOverflow, false, 'Long formulas overflow the page');
      if (width === 390) assert(metrics.scrollableFormulas > 0, 'Long formulas must remain locally scrollable');
      console.log(`${width}px: ${metrics.formulas} formulas, no clipped glyphs or page overflow`);
    }
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
